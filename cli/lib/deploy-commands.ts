/**
 * `pgai dblab deploy`, `pgai dblab instances list|watch|delete` and
 * `pgai mon deploy` (postgres-ai/platform-all#876). See ./deploy.
 */

import type { Command } from "commander";
import { createInterface } from "readline";
import { HttpStatusError } from "./util";
import {
  DBLAB_STEPS,
  MONITORING_STEPS,
  StepView,
  createMonitoring,
  launchRefusal,
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
  watchMonitoringDeploy,
  type ApiParams,
  type CloudInstance,
  type WatchOutcome,
} from "./deploy";

export interface DeployCliDeps {
  /** The api key and base url, or throws when no api key is configured. */
  resolveApi(debug: boolean): ApiParams;
  withOrgOptions(cmd: Command): Command;
  printResult(result: unknown, json?: boolean): void;
  isTty?: () => boolean;
  /** The Console's base URL, for the billing page (as `pgai connect` names it). */
  uiBaseUrl?: () => string;
}

const isTtyDefault = (): boolean =>
  !!process.stdout.isTTY && !process.env.CI && process.env.TERM !== "dumb";

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

const fail = (err: unknown): void => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
};

/** No card on file: what `pgai connect` does -- name the billing page, create nothing, exit 3. */
function needsCard(deps: DeployCliDeps, orgAlias: string | null | undefined, json: boolean | undefined): void {
  const ui = (deps.uiBaseUrl?.() ?? "https://console.postgres.ai").replace(/\/+$/, "");
  const page = orgAlias ? `${ui}/${orgAlias}/billing` : `${ui} (your organization > Billing)`;
  const next = `Add a payment method at ${page}, then re-run.`;
  if (json) deps.printResult({ status: "action_required", next }, true);
  else console.error(next);
  process.exitCode = 3;
}

/**
 * Runs `watch` with Ctrl-C detaching (the deploy keeps going on the platform)
 * instead of killing the process mid-redraw.
 */
async function withDetach<T>(view: StepView, resumeHint: string, watch: () => Promise<T>): Promise<T | "detached"> {
  let detached = false;
  const onSigint = (): void => {
    detached = true;
    view.stop();
    console.log(`\nStopped watching. The deploy continues on PostgresAI.\nResume: ${resumeHint}`);
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

/** An instance id as typed: digits only (parseInt would read "12x" as 12 and "abc" as all instances). */
function parseInstanceId(id: string): number {
  if (!/^\d+$/.test(id.trim())) throw new Error(`"${id}" is not a DBLab instance id; see pgai dblab instances list.`);
  return Number(id.trim());
}

function sshHint(inst: CloudInstance): string {
  if (!inst.server_ip) return "";
  return [
    `Connect to a clone through SSH (clones listen on the server's localhost), e.g. port 6000:`,
    `  ssh -N -L 6000:127.0.0.1:6000 root@${inst.server_ip}`,
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

/** A step view that writes nothing: under --json, stdout carries only the final object. */
const viewFor = (title: string, steps: typeof DBLAB_STEPS, tty: boolean, json: boolean | undefined): StepView =>
  json ? new StepView(title, steps, false, () => {}) : new StepView(title, steps, tty);

function reportEnd(outcome: Exclude<WatchOutcome, { kind: "gone" }>, id: number, json: boolean | undefined): void {
  const inst = outcome.instance;
  const name = `DBLab "${inst.project_name}" (id ${id})`;
  if (outcome.kind === "ready") {
    if (!json) {
      console.log(`\n${name} is ready.`);
      console.log(`Create a clone: PGAI_CLONE_DB_PASSWORD=<password> pgai dblab clone create --project ${inst.project_name} --db-user <user>`);
      const hint = sshHint(inst);
      if (hint) console.log(hint);
    }
    return;
  }
  if (outcome.kind === "deleted") {
    console.error(`\n${name} was deleted.`);
  } else {
    const removed =
      inst.deploy_status === "destroying" ? "The server is being removed; you are not charged for it."
      : inst.deploy_status === "destroyed" ? "The server was removed; you were not charged for it."
      : inst.deploy_status === "destroy_failed" ? "Removing the server failed, so it may still be running: run pgai dblab instances delete " + id + " again."
      : "";
    console.error(
      `\nDeploy failed${stepLabel(inst.deploy_step) ? ` at "${stepLabel(inst.deploy_step)}"` : ""}${inst.deploy_error ? `: ${inst.deploy_error}` : "."}` +
        (removed ? `\n${removed}` : ""),
    );
  }
  process.exitCode = 1;
}

async function followDblab(
  api: ApiParams,
  id: number,
  name: string,
  tty: boolean,
  json: boolean | undefined,
  deps: DeployCliDeps,
): Promise<void> {
  const view = viewFor(`Deploying DBLab "${name}" (id ${id})`, DBLAB_STEPS, tty, json);
  view.start();
  const outcome = await withDetach(view, `pgai dblab instances watch ${id}`, () => watchDblabDeploy(api, id, view, { pollMs: pollMs() }));
  if (outcome === "detached") return;
  if (json) {
    deps.printResult(outcome, true);
  }
  if (outcome.kind !== "gone") {
    reportEnd(outcome, id, json);
    return;
  }
  console.error(`\nDBLab ${id} is no longer in this organization.`);
  process.exitCode = 1;
}

export function registerDblabDeployCommands(dblab: Command, deps: DeployCliDeps): void {
  const tty = deps.isTty ?? isTtyDefault;

  deps.withOrgOptions(dblab.command("deploy"))
    .description("deploy DBLab + Joe in PostgresAI cloud, then follow it until it is ready")
    .requiredOption("--name <name>", "instance name (lowercase letters, digits, hyphens)")
    .requiredOption("--db-url <url>", "source database: postgresql://user@host:5432/dbname (password: set PGPASSWORD, or put it in the URL)")
    .option("--size <size>", "server size (S, M, L)", "S")
    .option("--disk <gib>", "disk size in GiB", "50")
    .option("--ssh-key <name|id...>", "org SSH key(s) for reaching the server and its clones (required)")
    .option("--location <location>", "Hetzner location (default: any with capacity)")
    .option("--no-wait", "return once the deploy has started")
    .option("-y, --yes", "skip the cost confirmation")
    .option("--debug", "enable debug output")
    .option("--json", "output JSON")
    .action(async (opts: {
      name: string; dbUrl: string; size: string; disk: string; sshKey?: string[]; location?: string;
      wait: boolean; yes?: boolean; debug?: boolean; json?: boolean;
    }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const diskGib = Number.parseInt(opts.disk, 10);
        if (!Number.isFinite(diskGib) || String(diskGib) !== opts.disk.trim()) {
          throw new Error("--disk must be a whole number of GiB");
        }
        const options = await getDeployOptions(api);
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
        const cents = estimateMonthlyCents(options, opts.size, diskGib);
        if (cents !== null) {
          // Always shown before asking; on stderr under --json.
          (opts.json ? console.error : console.log)(
            `Estimated cost: ${formatCents(cents)}/month (size ${opts.size.toUpperCase()}, ${diskGib} GiB), prorated.`,
          );
        }
        if (options.has_payment_method === false) {
          needsCard(deps, options.org_alias, opts.json);
          return;
        }
        if (!opts.yes) {
          if (!process.stdin.isTTY) throw new Error("Pass --yes to confirm the monthly cost when not running in a terminal.");
          if (!(await confirm("Deploy? [y/N] "))) {
            console.error("Cancelled.");
            return;
          }
        }
        let reply;
        try {
          reply = await deployDblab(api, {
            name: opts.name, dbUrl: withEnvPassword(opts.dbUrl), size: opts.size, diskGib, sshKeyIds, location: opts.location,
          });
        } catch (err) {
          // The platform's own check (PT402), when the options could not tell.
          if (err instanceof HttpStatusError && err.status === 402) {
            needsCard(deps, options.org_alias, opts.json);
            return;
          }
          throw err;
        }
        if (!opts.wait) {
          if (opts.json) deps.printResult(reply, true);
          else console.log(`Deploy started: id ${reply.id}. Follow it: pgai dblab instances watch ${reply.id}`);
          return;
        }
        await followDblab(api, reply.id, reply.project_name, tty(), opts.json, deps);
      } catch (err) {
        fail(err);
      }
    });

  const instances = dblab.command("instances").description("DBLab instances in this organization");

  deps.withOrgOptions(instances.command("list"))
    .description("list DBLab instances with their deploy status")
    .option("--debug", "enable debug output")
    .option("--json", "output JSON")
    .action(async (opts: { debug?: boolean; json?: boolean }) => {
      try {
        const rows = await listCloudInstances(deps.resolveApi(!!opts.debug));
        if (opts.json) {
          deps.printResult(rows, true);
          return;
        }
        if (!rows.length) {
          console.log("No DBLab instances. Deploy one: pgai dblab deploy --help");
          return;
        }
        console.log(`${"ID".padEnd(6)} ${"NAME".padEnd(28)} ${"STATE".padEnd(14)} SIZE / SERVER`);
        for (const r of rows) {
          const failed = r.deploy_status === "destroyed" && !!r.deploy_error;
          const status = r.is_cloud
            ? failed ? "Deploy failed" : STATE_LABELS[r.deploy_status ?? ""] ?? r.deploy_status
            : r.is_job_backed ? "Connected" : "Self-managed";
          // A failed or deleted deploy's server is gone; its IP may already be someone else's.
          const ip = r.server_ip && r.deploy_status !== "destroyed" && r.deploy_status !== "failed" ? ` ${r.server_ip}` : "";
          const extra = r.is_cloud ? ` ${r.size ?? ""} ${r.disk_gib ?? ""}GiB${ip}` : "";
          console.log(`${String(r.id).padEnd(6)} ${r.project_name.padEnd(28)} ${String(status).padEnd(14)}${extra}`);
          if (r.deploy_error && r.deploy_status !== "ready") {
            const first = r.deploy_error.split("\n")[0];
            console.log(`       ${first.length > 160 ? `${first.slice(0, 157)}...` : first} (details: pgai dblab instances watch ${r.id})`);
          }
        }
      } catch (err) {
        fail(err);
      }
    });

  deps.withOrgOptions(instances.command("watch <id>"))
    .description("follow a deploy until it is ready or has failed")
    .option("--debug", "enable debug output")
    .option("--json", "output JSON")
    .action(async (id: string, opts: { debug?: boolean; json?: boolean }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const n = parseInstanceId(id);
        const inst = (await listCloudInstances(api, n))[0];
        if (!inst) throw new Error(`No DBLab instance ${id} in this organization.`);
        if (!inst.is_cloud) {
          const said = `DBLab ${n} is not deployed in PostgresAI cloud: there is no deploy to follow.`;
          if (opts.json) {
            deps.printResult({ kind: "not_cloud", instance: inst }, true);
            console.error(said);
          } else {
            console.log(said);
          }
          return;
        }
        const ended = terminalOutcome(inst);
        if (ended && ended.kind !== "gone") {
          if (opts.json) deps.printResult(ended, true);
          reportEnd(ended, n, opts.json);
          if (ended.kind === "failed" && !opts.json) console.error("Fix the cause and deploy again.");
          return;
        }
        await followDblab(api, n, inst.project_name, tty(), opts.json, deps);
      } catch (err) {
        fail(err);
      }
    });

  deps.withOrgOptions(instances.command("delete <id>"))
    .description("delete a DBLab; one deployed in PostgresAI cloud also has its server destroyed")
    .option("-y, --yes", "skip the confirmation")
    .option("--debug", "enable debug output")
    .option("--json", "output JSON")
    .action(async (id: string, opts: { yes?: boolean; debug?: boolean; json?: boolean }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const n = parseInstanceId(id);
        const inst = (await listCloudInstances(api, n))[0];
        const label = inst ? `DBLab "${inst.project_name}" (id ${n})` : `DBLab ${n}`;
        if (!opts.yes) {
          if (!process.stdin.isTTY) throw new Error("Pass --yes to confirm the deletion when not running in a terminal.");
          if (!(await confirm(`Delete ${label} and its server? Its clones and its copy of the data are deleted, and billing for it stops. [y/N] `))) {
            console.error("Cancelled.");
            return;
          }
        }
        const result = await destroyDblab(api, n);
        if (opts.json) deps.printResult(result, true);
        else console.log(`Deleting ${label}. Follow it: pgai dblab instances list`);
      } catch (err) {
        fail(err);
      }
    });
}

export function registerMonDeployCommand(mon: Command, deps: DeployCliDeps & { orgId(apiKey: string): number | undefined }): void {
  const tty = deps.isTty ?? isTtyDefault;

  deps.withOrgOptions(mon.command("deploy"))
    .description("deploy managed monitoring in PostgresAI cloud (same as the Console), then follow it")
    .requiredOption("--db-url <url>", "database to monitor: postgresql://user@host:5432/dbname (password: set PGPASSWORD, or put it in the URL)")
    .option("--name <name>", "project name")
    .option("--plan <plan>", "monitoring plan", "scale")
    .option("--location <location>", "Hetzner location", "fsn1")
    .option("--ssh-key <id...>", "org SSH key id(s) for reaching the server")
    .option("--vcpus <n>", "the database's vCPU count (for AAS thresholds)")
    .option("--no-wait", "return once the deploy has started")
    .option("--debug", "enable debug output")
    .option("--json", "output JSON")
    .action(async (opts: {
      dbUrl: string; name?: string; plan: string; location: string; sshKey?: string[]; vcpus?: string;
      wait: boolean; debug?: boolean; json?: boolean;
    }) => {
      try {
        const api = deps.resolveApi(!!opts.debug);
        const orgId = deps.orgId(api.apiKey);
        const reply = await createMonitoring(api, {
          orgId,
          dbUrl: withEnvPassword(opts.dbUrl),
          plan: opts.plan,
          name: opts.name,
          location: opts.location,
          sshKeyIds: opts.sshKey ?? [],
          vcpus: opts.vcpus ? Number.parseInt(opts.vcpus, 10) : undefined,
        });
        const id = (reply as { id?: string }).id;
        if (!id) throw new Error(`Unexpected reply from the platform: ${JSON.stringify(reply)}`);
        const refused = launchRefusal(reply);
        if (refused) throw new Error(`The deploy did not start: ${refused}`);
        if (!opts.wait) {
          if (opts.json) deps.printResult(reply, true);
          else console.log(`Monitoring deploy started: ${id}.`);
          return;
        }
        const view = viewFor(`Deploying monitoring${opts.name ? ` "${opts.name}"` : ""} (${id})`, MONITORING_STEPS, tty(), opts.json);
        view.start();
        const outcome = await withDetach(view, "check the Console's Monitoring page", () => watchMonitoringDeploy(api, id, view, { pollMs: pollMs() }));
        if (outcome === "detached") return;
        if (opts.json) deps.printResult(outcome.status, true);
        if (outcome.kind === "ready") {
          if (!opts.json) console.log(`\nMonitoring is ready: ${outcome.status.grafana_url ?? "(no Grafana URL yet)"}`);
          return;
        }
        console.error(`\nDeploy failed${outcome.status.error ? `: ${outcome.status.error}` : "."}`);
        process.exitCode = 1;
      } catch (err) {
        fail(err);
      }
    });
}
