/**
 * `pgai dblab deploy` and `pgai dblab instances list|watch|delete`
 * (postgres-ai/platform-all#876), on the surface `pgai mon deploy` shares
 * (./deploy-surface, postgresai#412). See ./deploy.
 */

import type { Command } from "commander";
import { createInterface } from "readline";
import { lookup } from "dns/promises";
import { writeEvent } from "./json-stderr";
import { HttpStatusError } from "./util";
import {
  DBLAB_STEPS,
  StepView,
  terminalOutcome,
  withEnvPassword,
  DBLAB_STEPS as STEP_LABELS,
  deployDblab,
  destroyDblab,
  estimateMonthlyCents,
  formatCents,
  getDeployOptions,
  listCloudInstances,
  resolveSshKeys,
  watchDblabDeploy,
  DBLAB_PROVIDERS,
  DBLAB_CLOUDS,
  cloudName,
  hasPassword,
  probeSource,
  sourceLine,
  sourceName,
  type ApiParams,
  type DblabProvider,
  type DblabCloud,
  type ProbeReply,
  type CloudInstance,
  type WatchOutcome,
} from "./deploy";
import {
  DEPLOY_EXIT,
  billingPage,
  databaseUrlArg,
  dblabStatus,
  estimateLine,
  jsonOutput,
  nameFromDatabase,
  noCardNext,
  notAdminNext,
  pickInstance,
  sharedDeployOptions,
  waitMinutes,
  type DeployStatus,
} from "./deploy-surface";
import { detectCloudProvider } from "./connect";

export interface DeployCliDeps {
  /** The api key and base url, or throws when no api key is configured. */
  resolveApi(debug: boolean): ApiParams;
  withOrgOptions(cmd: Command): Command;
  printResult(result: unknown, json?: boolean): void;
  isTty?: () => boolean;
  /** The Console's base URL, for the billing page (as `pgai mon deploy` names it). */
  uiBaseUrl?: () => string;
}

// The step view and its spinner go to stderr: stdout is the command's result.
const isTtyDefault = (): boolean =>
  !!process.stderr.isTTY && !process.env.CI && process.env.TERM !== "dumb";

/** Asks on stderr: stdout stays for results (--json, redirects). */
export async function confirm(
  prompt: string,
  io: { input?: NodeJS.ReadableStream; output?: NodeJS.WritableStream } = {},
): Promise<boolean> {
  const rl = createInterface({ input: io.input ?? process.stdin, output: io.output ?? process.stderr });
  try {
    const answer: string = await new Promise((resolve) => rl.question(prompt, resolve));
    return /^y(es)?$/i.test(answer.trim());
  } finally {
    rl.close();
  }
}

// Tests shorten the poll; users never need to.
const pollMs = (): number | undefined => {
  const v = Number.parseInt(process.env.PGAI_DEPLOY_POLL_MS ?? "", 10);
  return Number.isFinite(v) && v > 0 ? v : undefined;
};

/** A result of deploy / watch / delete, in pgai mon deploy's shape. */
interface DblabResult {
  status: DeployStatus;
  id?: number;
  name?: string;
  error?: string;
  next: string;
  [key: string]: unknown;
}

/** Prints a result (JSON on stdout with --json; else `next` on stderr) and sets the exit code. */
function emit(deps: DeployCliDeps, result: DblabResult, json: boolean | undefined): void {
  if (jsonOutput(json)) deps.printResult(result, true);
  else if (result.status === "failed" || result.status === "action_required") console.error(result.next);
  process.exitCode = DEPLOY_EXIT[result.status];
}

/**
 * A private RDS: its name resolves to a private (RFC 1918) address. A timeout alone is not proof (a
 * public RDS behind a closed security group times out too): that stays a failure, with a hint.
 */
async function looksPrivate(dbUrl: string): Promise<boolean> {
  let host = "";
  try {
    host = new URL(dbUrl).hostname;
  } catch {
    return false;
  }
  try {
    const { address } = await lookup(host, { family: 4 });
    return /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(address);
  } catch {
    return false;
  }
}

const failed = (deps: DeployCliDeps, err: unknown, json: boolean | undefined): void =>
  emit(deps, { status: "failed", next: err instanceof Error ? err.message : String(err) }, json);

/**
 * Runs `watch` with Ctrl-C detaching (the deploy keeps going on the platform)
 * instead of killing the process mid-redraw.
 */
async function withDetach<T>(view: StepView, resumeHint: string, watch: () => Promise<T>): Promise<T | "detached"> {
  let detached = false;
  const onSigint = (): void => {
    detached = true;
    view.stop();
    console.error(`\nStopped watching. The deploy continues on PostgresAI.\nResume: ${resumeHint}`);
    process.exit(130);
  };
  process.once("SIGINT", onSigint);
  try {
    const r = await watch();
    return detached ? "detached" : r;
  } finally {
    // Also when the watch throws: a running spinner would keep the process alive.
    view.stop();
    process.removeListener("SIGINT", onSigint);
  }
}

function sshHint(inst: CloudInstance): string {
  if (!inst.server_ip) return "";
  return [
    `Connect to a clone through SSH (clones listen on the server's localhost), e.g. port 6000:`,
    `  ssh -N -L 6000:127.0.0.1:6000 ${inst.ssh_user ?? "root"}@${inst.server_ip}`,
    `  psql "host=127.0.0.1 port=6000 user=<user> dbname=<db>"`,
  ].join("\n");
}

const STATE_LABELS: Record<string, string> = {
  launching: "Deploying",
  installing: "Deploying",
  retrieving: "Copying data",
  ready: "Ready",
  failed: "Deploy failed",
  destroying: "Deleting",
  destroyed: "Deleted",
  destroy_failed: "Delete failed",
};

const stepLabel = (key: string | null | undefined): string | null => STEP_LABELS.find((s) => s.key === key)?.label ?? null;

/** The step view; with JSON output, each step line is a JSON event on stderr (as pgai mon deploy's). */
const viewFor = (title: string, steps: typeof DBLAB_STEPS, tty: boolean, json: boolean | undefined): StepView =>
  jsonOutput(json)
    ? new StepView(title, steps, false, (s) => s.split("\n").filter(Boolean).forEach((message) => writeEvent({ event: "step", message })))
    : new StepView(title, steps, tty);

/** How a deploy that has ended (or is still going) reads, as a result. */
function dblabResult(outcome: WatchOutcome, id: number): DblabResult {
  const inst = outcome.instance;
  // A failed or deleted deploy's server is gone; its IP may already be someone else's (as in list).
  const ipShown = inst?.server_ip && inst.deploy_status !== "destroyed" && inst.deploy_status !== "failed";
  const base = {
    id, name: inst?.project_name, ...(ipShown ? { server_ip: inst!.server_ip } : {}),
    ...(inst?.cloud ? { cloud: inst.cloud, location: inst.location } : {}),
  };
  switch (outcome.kind) {
    case "ready":
      return { ...base, status: "ready", next: `PGAI_CLONE_DB_PASSWORD=<password> pgai dblab clone create --project ${inst!.project_name} --db-user <user>` };
    case "timeout":
      return { ...base, status: "in_progress", next: `pgai dblab instances watch ${id}` };
    case "deleted":
      return { ...base, status: "deleted", next: "none" };
    case "gone":
      return { id, status: "deleted", next: `DBLab ${id} is no longer in this organization.` };
    case "failed": {
      const removed =
        inst!.deploy_status === "destroying" ? "The server is being removed; you are not charged for it."
        : inst!.deploy_status === "destroyed" ? "The server was removed; you were not charged for it."
        : inst!.deploy_status === "destroy_failed" ? `Removing the server failed, so it may still be running: run pgai dblab instances delete ${id} again.`
        : "";
      const why = `Deploy failed${stepLabel(inst!.deploy_step) ? ` at "${stepLabel(inst!.deploy_step)}"` : ""}${inst!.deploy_error ? `: ${inst!.deploy_error}` : "."}`;
      return { ...base, status: "failed", ...(inst!.deploy_error ? { error: inst!.deploy_error } : {}), next: [why, removed].filter(Boolean).join("\n") };
    }
  }
}

function reportEnd(deps: DeployCliDeps, outcome: WatchOutcome, id: number, json: boolean | undefined): void {
  const result = dblabResult(outcome, id);
  if (!jsonOutput(json) && result.status === "ready") {
    const inst = outcome.instance!;
    console.log(`\nDBLab "${inst.project_name}" (id ${id}) is ready.`);
    console.log(`Create a clone: ${result.next}`);
    const hint = sshHint(inst);
    if (hint) console.log(hint);
  }
  if (!jsonOutput(json) && result.status === "in_progress") console.error(`\nStill deploying. Follow it: ${result.next}`);
  else if (!jsonOutput(json) && outcome.kind === "deleted") console.error(`\nDBLab "${outcome.instance.project_name}" (id ${id}) was deleted.`);
  else if (!jsonOutput(json) && outcome.kind === "gone") console.error(`\n${result.next}`);
  else emit(deps, result, json);
  process.exitCode = DEPLOY_EXIT[result.status];
}

async function followDblab(
  api: ApiParams,
  id: number,
  name: string,
  tty: boolean,
  json: boolean | undefined,
  minutes: number,
  deps: DeployCliDeps,
  where?: { cloud?: string | null; location?: string | null },
): Promise<void> {
  // Where it runs, as the platform chose it (e.g. the source's region): "on AWS, eu-central-1".
  const on = where?.location ? ` on ${cloudName(where.cloud || "hetzner")}, ${where.location}` : "";
  const view = viewFor(`Deploying DBLab "${name}" (id ${id})${on}`, DBLAB_STEPS, tty, json);
  view.start();
  const outcome = await withDetach(view, `pgai dblab instances watch ${id}`, () =>
    watchDblabDeploy(api, id, view, { pollMs: pollMs(), maxMs: minutes * 60_000 }));
  if (outcome === "detached") return;
  reportEnd(deps, outcome, id, json);
}

/** A DBLab of the org by id or name. */
async function dblabInstance(api: ApiParams, ref: string): Promise<CloudInstance> {
  return pickInstance(await listCloudInstances(api), ref, (r) => String(r.id), (r) => r.project_name, "pgai dblab instances list",
    (r) => dblabStatus(r.deploy_status) !== "deleted");
}

/** One line for a failed deploy: its headline and, when a database error says why, that cause. */
function failureSummary(error: string): string {
  const first = error.split(/\r?\n/)[0];
  const cause = first.match(/\bERROR:\s+(.*)$/)?.[1];
  return cause ? `${first.split(": ")[0]}: ${cause}` : first;
}

export function registerDblabDeployCommands(dblab: Command, deps: DeployCliDeps): void {
  const tty = deps.isTty ?? isTtyDefault;

  sharedDeployOptions(deps.withOrgOptions(dblab.command("deploy")), 120,
    "instance name: lowercase letters, digits, hyphens (default: from the database's name)",
    "Hetzner location: fsn1, nbg1 or hel1 (default: any with capacity); with --cloud aws or gcp, a region "
      + "(default: the source's region, else us-east-1 / us-central1)")
    .description("deploy DBLab + Joe in PostgresAI cloud, then follow it until it is ready")
    .option("--size <size>", "server size (S, M, L)", "S")
    .option("--disk <gib>", "disk size in GiB", "50")
    .option("--ssh-key <name|id...>", "org SSH key(s) for reaching the server and its clones (required)")
    .option("--provider <provider>", "where the database runs: generic (direct PostgreSQL URL; alias: self-managed), rds or supabase (default: detected)")
    .option("--cloud <cloud>", "where the DBLab server runs: hetzner, aws or gcp (default: hetzner); --location is then its region")
    .addHelpText("after", [
      "",
      "The password: in the URL, or set PGPASSWORD.",
      "A Supabase project without its password: deploy it from the Console (Connect Supabase).",
      "Exit codes: 0 ready or in progress, 1 failed, 3 action required (see \"next\").",
    ].join("\n"))
    .action(async (urlArg: string | undefined, opts: {
      dbUrl?: string; name?: string; size: string; disk: string; sshKey?: string[]; location?: string;
      provider?: string; cloud?: string; wait?: string | boolean; yes?: boolean; debug?: boolean; json?: boolean;
    }, cmd: Command) => {
      // The URL is the argument or --db-url: without either, Commander's own error for a missing argument.
      if (!urlArg?.trim() && !opts.dbUrl?.trim()) return cmd.error("error: missing required argument 'database-url'");
      try {
        const url = databaseUrlArg(urlArg, opts.dbUrl, "pgai dblab deploy");
        const provider = opts.provider?.trim().toLowerCase() as DblabProvider | undefined;
        if (provider && !DBLAB_PROVIDERS.includes(provider)) {
          throw new Error(`--provider must be one of: ${DBLAB_PROVIDERS.join(", ")}`);
        }
        const cloud = opts.cloud?.trim().toLowerCase() as DblabCloud | undefined;
        if (cloud !== undefined && !DBLAB_CLOUDS.includes(cloud)) {
          throw new Error(`--cloud must be one of: ${DBLAB_CLOUDS.join(", ")}`);
        }
        const name = opts.name?.trim() || nameFromDatabase(url);
        const minutes = waitMinutes(opts.wait, 120);
        const api = deps.resolveApi(!!opts.debug);
        const diskGib = Number.parseInt(opts.disk, 10);
        if (!Number.isFinite(diskGib) || String(diskGib) !== opts.disk.trim()) {
          throw new Error("--disk must be a whole number of GiB");
        }
        const options = await getDeployOptions(api, cloud);
        // A cloud this environment does not offer, before anything is asked.
        const offered = cloud ? options.clouds?.find((c) => c.code === cloud) : undefined;
        if (cloud && offered && !offered.available) throw new Error(`${cloudName(cloud)} is not available.`);
        if (cloud && opts.location && options.locations?.length && !options.locations.includes(opts.location)) {
          throw new Error(`Unknown location "${opts.location}" for ${cloudName(cloud)}. Choose one of: ${options.locations.join(", ")}.`);
        }
        const notAdmin = (): void => emit(deps, { status: "action_required", name, next: notAdminNext }, opts.json);
        if (options.is_admin === false) return notAdmin();
        // Supabase without its password: the Console connects the account and makes a read-only user.
        const dbUrl = withEnvPassword(url);
        const isSupabase = provider ? provider === "supabase" : detectCloudProvider(url) === "supabase";
        if (isSupabase && !hasPassword(dbUrl)) {
          const console_ = `${(deps.uiBaseUrl?.() ?? "https://console.postgres.ai").replace(/\/+$/, "")}/${options.org_alias ?? "<org>"}`;
          return emit(deps, {
            status: "action_required",
            name,
            next: "This Supabase URL has no password. Add it to the URL or set PGPASSWORD, or finish in the Console, "
              + `which needs no password: ${console_}/instances/deploy?source=supabase`,
          }, opts.json);
        }
        const cents = estimateMonthlyCents(options, opts.size, diskGib);
        const on = cloud ? ` on ${cloudName(cloud)}` : "";
        const price = cents === null ? null : `${formatCents(cents)}/month (size ${opts.size.toUpperCase()}${on}, ${diskGib} GiB), prorated.`;
        // The source as the deploy will see it, before anything is asked or paid for.
        let probe: ProbeReply | null = null;
        try {
          probe = await probeSource(api, dbUrl, provider);
        } catch (err) {
          if (err instanceof HttpStatusError && err.status === 403) return notAdmin();
          // Not testable now (e.g. too many tests): the deploy checks the source itself.
        }
        // A private RDS (platform-all#925): reached over PrivateLink, which the Console sets up.
        if (probe && !probe.reachable && (probe.provider === "rds" || probe.provider === "aurora")
            && await looksPrivate(dbUrl)) {
          const console_ = `${(deps.uiBaseUrl?.() ?? "https://console.postgres.ai").replace(/\/+$/, "")}/${options.org_alias ?? "<org>"}`;
          return emit(deps, {
            status: "action_required",
            name,
            next: "This RDS is not reachable from the internet. If it is private, deploy it over PrivateLink in the Console: "
              + `${console_}/instances/deploy?source=rds&network=private`,
          }, opts.json);
        }
        if (probe && !probe.reachable) {
          // The advice on its own line, as the platform's refusals print it.
          const viaPrivateLink = (probe.provider === "rds" || probe.provider === "aurora")
            && /timeout|timed out/i.test(probe.error ?? "")
            ? `\nA private RDS deploys over PrivateLink in the Console: ${(deps.uiBaseUrl?.() ?? "https://console.postgres.ai").replace(/\/+$/, "")}/${options.org_alias ?? "<org>"}/instances/deploy?source=rds&network=private`
            : "";
          throw new Error(`Cannot connect to the source database: ${probe.error ?? "unknown error"}${probe.hint ? `\nHint: ${probe.hint}` : ""}${viaPrivateLink}`);
        }
        // Always shown before asking, on stderr: stdout is the result. After the source checks: a private
        // RDS goes to the PrivateLink wizard, where it runs on AWS, not at this price.
        if (price) {
          if (jsonOutput(opts.json)) writeEvent({ event: "billing", message: estimateLine(price) });
          else console.error(estimateLine(price));
        }
        const noCard = (): void =>
          emit(deps, { status: "action_required", name, next: noCardNext(billingPage(deps.uiBaseUrl?.(), options.org_alias)) }, opts.json);
        if (options.has_payment_method === false) return noCard();
        if (probe) {
          if (jsonOutput(opts.json)) writeEvent({ event: "source", message: sourceLine(probe) });
          else console.error(sourceLine(probe));
        }
        const keys = opts.sshKey?.length ? opts.sshKey : [];
        if (!keys.length) {
          throw new Error(
            "--ssh-key is required: it is how you reach the server and its clones." +
              (options.ssh_keys.length
                ? ` Org keys: ${options.ssh_keys.map((k) => k.name ?? k.id).join(", ")}.`
                : " Add one in the Console (Organization > SSH keys) first."),
          );
        }
        const sshKeyIds = resolveSshKeys(options, keys);
        if (!opts.yes) {
          if (!process.stdin.isTTY || jsonOutput(opts.json)) {
            return emit(deps, { status: "action_required", name, next: `Re-run with --yes to accept ${price ?? "the monthly cost"}` }, opts.json);
          }
          if (!(await confirm("Deploy? [y/N] "))) {
            console.error("Cancelled.");
            return;
          }
        }
        let reply;
        try {
          reply = await deployDblab(api, {
            name, dbUrl, size: opts.size, diskGib, sshKeyIds, location: opts.location, provider, cloud,
          });
        } catch (err) {
          // The platform's own check (PT402), when the options could not tell.
          if (err instanceof HttpStatusError && err.status === 402) return noCard();
          if (err instanceof HttpStatusError && err.status === 403) return notAdmin();
          throw err;
        }
        if (minutes === 0) {
          const started: DblabResult = {
            status: "in_progress", id: reply.id, name: reply.project_name,
            ...(reply.cloud ? { cloud: reply.cloud, location: reply.location ?? null } : {}),
            next: `pgai dblab instances watch ${reply.id}`,
          };
          if (!jsonOutput(opts.json)) console.error(`Deploy started: id ${reply.id}. Follow it: ${started.next}`);
          emit(deps, started, opts.json);
          return;
        }
        await followDblab(api, reply.id, reply.project_name, tty(), opts.json, minutes, deps, reply);
      } catch (err) {
        failed(deps, err, opts.json);
      }
    });

  const instances = dblab.command("instances").description("DBLab instances in this organization");

  deps.withOrgOptions(instances.command("list"))
    .description("list DBLab instances with their deploy status")
    .option("--debug", "print HTTP requests (secrets masked)")
    .option("--json", "JSON output")
    .action(async (opts: { debug?: boolean; json?: boolean }) => {
      try {
        const rows = await listCloudInstances(deps.resolveApi(!!opts.debug));
        if (jsonOutput(opts.json)) {
          // The same status words as pgai mon instances list (null: not deployed by PostgresAI).
          deps.printResult(rows.map((r) => ({ ...r, status: r.is_cloud ? dblabStatus(r.deploy_status) : null })), true);
          return;
        }
        if (!rows.length) {
          console.log("No DBLab instances. Deploy one: pgai dblab deploy --help");
          return;
        }
        console.log(`${"ID".padEnd(6)} ${"NAME".padEnd(28)} ${"STATE".padEnd(14)} SIZE / SERVER / SOURCE`);
        for (const r of rows) {
          const failedDeploy = r.deploy_status === "destroyed" && !!r.deploy_error;
          const status = r.is_cloud
            ? failedDeploy ? "Deploy failed" : STATE_LABELS[r.deploy_status ?? ""] ?? r.deploy_status
            : r.is_job_backed ? "Connected" : "Self-managed";
          // A failed or deleted deploy's server is gone; its IP may already be someone else's.
          const ip = r.server_ip && r.deploy_status !== "destroyed" && r.deploy_status !== "failed" ? ` ${r.server_ip}` : "";
          const source = sourceName(r.source_provider, r.source_name);
          const where = r.cloud && r.cloud !== "hetzner" ? ` ${cloudName(r.cloud)} ${r.location ?? ""}` : "";
          const extra = r.is_cloud ? ` ${r.size ?? ""} ${r.disk_gib ?? ""}GiB${where}${ip}${source ? ` from ${source}` : ""}` : "";
          console.log(`${String(r.id).padEnd(6)} ${r.project_name.padEnd(28)} ${String(status).padEnd(14)}${extra}`);
          if (r.deploy_error && r.deploy_status !== "ready") {
            const summary = failureSummary(r.deploy_error);
            console.log(`       ${summary.length > 160 ? `${summary.slice(0, 157)}...` : summary} (details: pgai dblab instances watch ${r.id})`);
          }
        }
      } catch (err) {
        failed(deps, err, opts.json);
      }
    });

  deps.withOrgOptions(instances.command("watch <id-or-name>"))
    .alias("status")
    .description("follow a deploy until it is ready or has failed")
    .option("--wait <minutes>", "how long to follow it (0 = show it now)", "120")
    .option("--no-wait", "show it now (the same as --wait 0)")
    .option("--debug", "print HTTP requests (secrets masked)")
    .option("--json", "JSON output: one result on stdout, progress on stderr")
    .action(async (ref: string, opts: { wait?: string | boolean; debug?: boolean; json?: boolean }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const minutes = waitMinutes(opts.wait, 120);
        const inst = await dblabInstance(api, ref);
        if (!inst.is_cloud) {
          const said = `DBLab ${inst.id} is not deployed in PostgresAI cloud: there is no deploy to follow.`;
          if (!jsonOutput(opts.json)) console.log(said);
          return emit(deps, { status: "ready", id: inst.id, name: inst.project_name, next: said }, opts.json);
        }
        const ended = terminalOutcome(inst);
        if (ended && ended.kind !== "gone") return reportEnd(deps, ended, inst.id, opts.json);
        if (minutes === 0) return reportEnd(deps, { kind: "timeout", instance: inst }, inst.id, opts.json);
        await followDblab(api, inst.id, inst.project_name, tty(), opts.json, minutes, deps, inst);
      } catch (err) {
        failed(deps, err, opts.json);
      }
    });

  deps.withOrgOptions(instances.command("delete <id-or-name>"))
    .description("delete a DBLab; one deployed in PostgresAI cloud also has its server destroyed")
    .option("-y, --yes", "do not ask for confirmation")
    .option("--debug", "print HTTP requests (secrets masked)")
    .option("--json", "JSON output")
    .action(async (ref: string, opts: { yes?: boolean; debug?: boolean; json?: boolean }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const inst = await dblabInstance(api, ref);
        const label = `DBLab "${inst.project_name}" (id ${inst.id})`;
        if (!opts.yes) {
          if (!process.stdin.isTTY || jsonOutput(opts.json)) {
            return emit(deps, { status: "action_required", id: inst.id, name: inst.project_name, next: `pgai dblab instances delete ${inst.id} --yes` }, opts.json);
          }
          if (!(await confirm(`Delete ${label} and its server? Its clones and its copy of the data are deleted, and billing for it stops. [y/N] `))) {
            console.error("Cancelled.");
            return;
          }
        }
        await destroyDblab(api, inst.id);
        if (!jsonOutput(opts.json)) console.log(`Deleting ${label}. Follow it: pgai dblab instances list`);
        emit(deps, { status: "deleting", id: inst.id, name: inst.project_name, next: "pgai dblab instances list" }, opts.json);
      } catch (err) {
        failed(deps, err, opts.json);
      }
    });
}
