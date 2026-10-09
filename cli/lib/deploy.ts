/**
 * Deploy to PostgresAI cloud (postgres-ai/platform-all#876).
 *
 * `pgai dblab deploy` calls the SAME platform rpc the Console calls
 * (`v1.dblab_instance_deploy`), then follows the platform's own status until the instance is ready. Nothing
 * here talks to the provisioning service: the platform launches the server
 * with its own credentials, and the box reports progress back to the platform.
 */

import {
  HttpRequestTimeoutError,
  HttpStatusError,
  describeFetchError,
  formatHttpError,
  isFetchTimeout,
  maskSecret,
  normalizeBaseUrl,
  redactSecretsForLog,
  redactTextSecrets,
  requestTimeoutSignal,
} from "./util";
import { buildAuthHeaders } from "./org-scope";

export interface ApiParams {
  apiKey: string;
  apiBaseUrl: string;
  debug?: boolean;
}

/** POST {base}/rpc/<fn> with the CLI's access-token; throws HttpStatusError on non-2xx. */
/** A URL's password replaced: everything between the user and the LAST '@' (a password may hold '/' or '@'). */
export function maskUrlPassword(url: string): string {
  const scheme = url.indexOf("://");
  const at = url.lastIndexOf("@");
  if (scheme < 0 || at <= scheme) return url;
  const userinfo = url.slice(scheme + 3, at);
  const colon = userinfo.indexOf(":");
  return colon < 0 ? url : `${url.slice(0, scheme + 3)}${userinfo.slice(0, colon)}:[REDACTED]${url.slice(at)}`;
}

function maskDbUrlInBody(payload: string): string {
  try {
    const body = JSON.parse(payload) as Record<string, unknown>;
    if (typeof body.db_url !== "string") return payload;
    return JSON.stringify({ ...body, db_url: maskUrlPassword(body.db_url) });
  } catch {
    return payload;
  }
}

export async function callDeployRpc<T>(
  params: ApiParams,
  fn: string,
  body: Record<string, unknown>,
  operation: string,
): Promise<T> {
  if (!params.apiKey) throw new Error("API key is required");
  const base = normalizeBaseUrl(params.apiBaseUrl);
  const url = `${base}/rpc/${fn}`;
  const headers = buildAuthHeaders(params.apiKey);
  const payload = JSON.stringify(body);

  if (params.debug) {
    console.error(`Debug: POST ${url}`);
    console.error(`Debug: headers ${JSON.stringify({ ...headers, "access-token": maskSecret(params.apiKey) })}`);
    // The source URL carries the password: masked by structure (it may hold '/' or '@'), then the
    // usual scrubs.
    console.error(`Debug: body ${redactTextSecrets(redactSecretsForLog(maskDbUrlInBody(payload)))}`);
  }

  const timeout = requestTimeoutSignal(120_000); // a deploy call checks the source DB first
  let response: Response;
  try {
    response = await fetch(url, { method: "POST", headers, body: payload, signal: timeout.signal });
  } catch (err) {
    if (isFetchTimeout(err)) throw new HttpRequestTimeoutError(operation, timeout.timeoutMs);
    throw new Error(describeFetchError(operation, base, err));
  }
  const text = await response.text();
  if (params.debug) console.error(`Debug: ${response.status} ${redactTextSecrets(redactSecretsForLog(text))}`);
  if (!response.ok) {
    throw new HttpStatusError(refusalText(operation, response.status, text, response.statusText, !!params.debug), response.status);
  }
  return (text ? JSON.parse(text) : null) as T;
}

/** The platform's own words for a refusal (details + hint); the HTTP status only with --debug. */
export function refusalText(operation: string, status: number, text: string, statusText: string, debug: boolean): string {
  let body: Record<string, unknown> | null = null;
  try {
    const v = JSON.parse(text) as unknown;
    if (v && typeof v === "object" && !Array.isArray(v)) body = v as Record<string, unknown>;
  } catch {
    body = null;
  }
  const said = [body?.details, body?.message].find((x) => typeof x === "string" && x.trim()) as string | undefined;
  if (!said) return formatHttpError(operation, status, text, statusText);
  const hint = typeof body?.hint === "string" && body.hint.trim() ? `\nHint: ${body.hint}` : "";
  return `${debug ? `${operation}: HTTP ${status}\n` : ""}${said}${hint}`;
}

/**
 * The URL with the password from PGPASSWORD when it carries none, so the password can stay out of the
 * command line and shell history. A URL that has a password is used as is.
 */
export function withEnvPassword(dbUrl: string, env: Record<string, string | undefined> = process.env): string {
  const pw = env.PGPASSWORD;
  if (!pw) return dbUrl;
  let u: URL;
  try {
    u = new URL(dbUrl);
  } catch {
    return dbUrl;
  }
  if (u.password || !u.username) return dbUrl;
  u.password = encodeURIComponent(pw);
  return u.toString();
}

// ---------------------------------------------------------------------------
// DBLab
// ---------------------------------------------------------------------------

export interface DeployOptions {
  create_options: string[];
  sizes: { code: string; vcpus: number; ram_gib: number; monthly_price_cents: number; available: boolean }[];
  disk_monthly_price_cents_per_gib: number;
  min_disk_gib: number;
  max_disk_gib: number;
  max_instances: number;
  instances_used: number;
  locations: string[];
  ssh_keys: { id: string; name: string | null }[];
  is_admin: boolean;
  /** null when the platform could not read Stripe: the deploy's own PT402 decides then. */
  has_payment_method?: boolean | null;
  org_alias?: string | null;
}

export interface CloudInstance {
  id: number;
  project_id: number;
  project_name: string;
  created_at: string;
  is_cloud: boolean;
  deploy_status: string | null;
  deploy_step: string | null;
  deploy_error: string | null;
  deploy_updated_at: string | null;
  size: string | null;
  disk_gib: number | null;
  location: string | null;
  server_ip: string | null;
  is_job_backed: boolean;
  joe_instance_id: number | null;
}

export interface DeployReply {
  id: number;
  project_id: number;
  project_name: string;
  deploy_status: string;
  monthly_price_cents: number;
}

export const getDeployOptions = (p: ApiParams): Promise<DeployOptions> =>
  callDeployRpc<DeployOptions>(p, "dblab_cloud_deploy_options", {}, "Read deploy options");

export const listCloudInstances = async (p: ApiParams, instanceId?: number): Promise<CloudInstance[]> =>
  (await callDeployRpc<CloudInstance[] | null>(
    p,
    "dblab_cloud_instances",
    instanceId === undefined ? {} : { p_instance_id: instanceId },
    "List DBLab instances",
  )) ?? [];

export const deployDblab = (
  p: ApiParams,
  args: { name: string; dbUrl: string; size: string; diskGib: number; sshKeyIds: string[]; location?: string },
): Promise<DeployReply> =>
  callDeployRpc<DeployReply>(
    p,
    "dblab_instance_deploy",
    {
      name: args.name,
      db_url: args.dbUrl,
      size: args.size,
      disk_gib: args.diskGib,
      ssh_key_ids: args.sshKeyIds,
      ...(args.location ? { location: args.location } : {}),
    },
    "Deploy DBLab",
  );

export const destroyDblab = (p: ApiParams, instanceId: number): Promise<unknown> =>
  callDeployRpc(p, "dblab_instance_destroy", { instance_id: instanceId }, "Delete DBLab");

/** Monthly estimate in cents for a size and disk, from the catalogue. */
export function estimateMonthlyCents(opts: DeployOptions, size: string, diskGib: number): number | null {
  const s = opts.sizes.find((x) => x.code === size.toUpperCase());
  if (!s) return null;
  return s.monthly_price_cents + diskGib * (opts.disk_monthly_price_cents_per_gib ?? 0);
}

export const formatCents = (cents: number): string => `$${(cents / 100).toFixed(2)}`;

/** Resolves --ssh-key values (names or ids) against the org's keys. */
export function resolveSshKeys(opts: DeployOptions, wanted: string[]): string[] {
  const out: string[] = [];
  for (const w of wanted) {
    const key = opts.ssh_keys.find((k) => k.id === w) ?? opts.ssh_keys.filter((k) => k.name === w);
    if (Array.isArray(key)) {
      if (key.length === 1) out.push(key[0].id);
      else if (key.length > 1) throw new Error(`More than one SSH key is named "${w}"; pass its id instead.`);
      else throw new Error(`No SSH key "${w}" in this organization. Known: ${opts.ssh_keys.map((k) => k.name ?? k.id).join(", ") || "none"}.`);
    } else {
      out.push(key.id);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Progress view
// ---------------------------------------------------------------------------

export interface StepDef {
  key: string;
  label: string;
}

/** pgai mon deploy's steps, in the order its progress events come (lib/connect). */
export const MONITORING_STEPS: StepDef[] = [
  { key: "preparing", label: "Prepare the database" },
  { key: "provisioning", label: "Create the monitoring box" },
  { key: "checkup", label: "Express checkup" },
  { key: "box", label: "Install monitoring (about 5 min)" },
  { key: "ready", label: "Ready" },
];

/** DBLab deploy steps, in the order the box reports them. */
export const DBLAB_STEPS: StepDef[] = [
  { key: "create_server", label: "Create server" },
  { key: "install_engine", label: "Install DBLab Engine" },
  { key: "install_joe", label: "Install Joe" },
  { key: "connect_agents", label: "Connect DBLab and Joe to PostgresAI" },
  { key: "retrieve_data", label: "Copy data from the source" },
  { key: "ready", label: "Ready" },
];

type StepState = "pending" | "running" | "done" | "failed";

const FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";

export const formatDuration = (ms: number): string => {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
};

/**
 * A step list that redraws in place on a TTY (spinner -> ✓/✗ with each step's
 * duration, the total underneath) and prints one plain line per change
 * otherwise. Pure rendering: `render()` returns the frame, so it is testable.
 */
export class StepView {
  private states = new Map<string, StepState>();
  private started = new Map<string, number>();
  private finished = new Map<string, number>();
  private frame = 0;
  private drawnLines = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private note = "";
  private stopped = false;
  private readonly t0: number;

  constructor(
    private readonly title: string,
    private readonly steps: StepDef[],
    private readonly tty: boolean,
    // stderr: stdout carries only the command's result (postgresai#412).
    private readonly write: (s: string) => void = (s) => process.stderr.write(s),
    private readonly now: () => number = Date.now,
  ) {
    this.t0 = now();
    for (const s of steps) this.states.set(s.key, "pending");
  }

  start(): void {
    if (this.tty) {
      this.write("\x1b[?25l"); // hide the cursor while animating
      this.draw();
      this.timer = setInterval(() => {
        this.frame++;
        this.draw();
      }, 100);
    } else {
      this.write(`${this.title}\n`);
    }
  }

  /** Marks `key` running and everything before it done. Unknown keys are ignored. */
  advance(key: string): void {
    const idx = this.steps.findIndex((s) => s.key === key);
    if (idx < 0) return;
    for (let i = 0; i < this.steps.length; i++) {
      const k = this.steps[i].key;
      if (i < idx && this.states.get(k) !== "done") this.set(k, "done");
      if (i === idx && this.states.get(k) === "pending") this.set(k, "running");
    }
  }

  /** Marks every step done. */
  complete(): void {
    for (const s of this.steps) if (this.states.get(s.key) !== "done") this.set(s.key, "done");
  }

  /** Marks the running step (or `key`) failed. */
  fail(key?: string): void {
    const k = key && this.states.has(key) ? key : this.steps.find((s) => this.states.get(s.key) === "running")?.key;
    if (!k) return;
    // The failing step may come after the one last seen running: everything before it is done.
    this.advance(k);
    this.set(k, "failed");
  }

  setNote(note: string): void {
    this.note = note;
  }

  /** Text above the steps (the express checkup's findings): the animation is redrawn below it. */
  log(text: string): void {
    const body = text.endsWith("\n") ? text : `${text}\n`;
    if (!this.tty || this.stopped) {
      this.write(body);
      return;
    }
    if (this.drawnLines > 0) this.write(`\x1b[${this.drawnLines}A\r\x1b[0J`);
    this.drawnLines = 0;
    this.write(body);
    this.draw();
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.tty) {
      this.draw();
      this.write("\x1b[?25h");
    }
  }

  private set(key: string, st: StepState): void {
    const prev = this.states.get(key);
    if (prev === st) return;
    this.states.set(key, st);
    const t = this.now();
    if (st === "running") this.started.set(key, t);
    if (st === "done" || st === "failed") {
      if (!this.started.has(key)) this.started.set(key, t);
      this.finished.set(key, t);
    }
    if (!this.tty) {
      const label = this.steps.find((s) => s.key === key)?.label ?? key;
      const dur = st === "done" || st === "failed" ? ` (${formatDuration(t - (this.started.get(key) ?? t))})` : "";
      const word = st === "running" ? "started" : st === "done" ? "done" : st === "failed" ? "FAILED" : st;
      this.write(`${label}: ${word}${dur}\n`);
    }
  }

  /** The current frame, one string per line. */
  render(): string[] {
    const lines = [`${BOLD}${this.title}${RESET}`];
    for (const s of this.steps) {
      const st = this.states.get(s.key);
      const t = this.now();
      const dur =
        st === "running"
          ? formatDuration(t - (this.started.get(s.key) ?? t))
          : st === "done" || st === "failed"
            ? formatDuration((this.finished.get(s.key) ?? t) - (this.started.get(s.key) ?? t))
            : "";
      const icon =
        st === "done" ? `${GREEN}✓${RESET}`
        : st === "failed" ? `${RED}✗${RESET}`
        : st === "running" ? FRAMES[this.frame % FRAMES.length]
        : `${DIM}·${RESET}`;
      const label = st === "pending" ? `${DIM}${s.label}${RESET}` : s.label;
      lines.push(`  ${icon} ${label}${dur ? ` ${DIM}${dur}${RESET}` : ""}`);
    }
    lines.push(`  ${DIM}elapsed ${formatDuration(this.now() - this.t0)}${this.note ? ` · ${this.note}` : ""}${RESET}`);
    return lines;
  }

  private draw(): void {
    const lines = this.render();
    let out = "";
    if (this.drawnLines > 0) out += `\x1b[${this.drawnLines}A`;
    for (const l of lines) out += `\r\x1b[2K${l}\n`;
    this.drawnLines = lines.length;
    this.write(out);
  }
}

// ---------------------------------------------------------------------------
// Following a deploy
// ---------------------------------------------------------------------------

export type WatchOutcome =
  | { kind: "ready"; instance: CloudInstance }
  | { kind: "failed"; instance: CloudInstance }
  | { kind: "deleted"; instance: CloudInstance }
  | { kind: "gone"; instance: CloudInstance | null }
  // --wait ran out first: the deploy goes on (pgai dblab instances watch <id>).
  | { kind: "timeout"; instance: CloudInstance };

const TERMINAL_FAILED = new Set(["failed", "destroying", "destroyed", "destroy_failed"]);

/** A deploy that has ended: ready, failed, or deleted by someone (no error recorded); null while it runs. */
export function terminalOutcome(inst: CloudInstance): WatchOutcome | null {
  if (inst.deploy_status === "ready") return { kind: "ready", instance: inst };
  if (!inst.deploy_status || !TERMINAL_FAILED.has(inst.deploy_status)) return null;
  if (!inst.deploy_error && (inst.deploy_status === "destroying" || inst.deploy_status === "destroyed")) {
    return { kind: "deleted", instance: inst };
  }
  return { kind: "failed", instance: inst };
}

/** Polls the instance until it is ready or has failed, driving `view`. */
export async function watchDblabDeploy(
  p: ApiParams,
  id: number,
  view: StepView,
  opts: { pollMs?: number; sleep?: (ms: number) => Promise<void>; maxErrors?: number; maxMs?: number; now?: () => number } = {},
): Promise<WatchOutcome> {
  const pollMs = opts.pollMs ?? 5000;
  const now = opts.now ?? Date.now;
  const deadline = opts.maxMs === undefined ? Infinity : now() + opts.maxMs;
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const maxErrors = opts.maxErrors ?? 6;
  let errors = 0;
  let lastStep: string | null = null;
  for (;;) {
    let inst: CloudInstance | undefined;
    try {
      inst = (await listCloudInstances(p, id))[0];
      errors = 0;
    } catch (err) {
      errors++;
      if (errors >= maxErrors) throw err;
      view.setNote(`retrying (${err instanceof Error ? err.message : String(err)})`);
      await sleep(pollMs);
      continue;
    }
    if (!inst) return { kind: "gone", instance: null };
    view.setNote(inst.server_ip ? `server ${inst.server_ip}` : "");
    const end = terminalOutcome(inst);
    if (end?.kind === "ready") view.complete();
    if (end?.kind === "failed") view.fail(inst.deploy_step ?? lastStep ?? undefined);
    if (end) return end;
    if (inst.deploy_step) {
      lastStep = inst.deploy_step;
      view.advance(inst.deploy_step);
    }
    if (now() >= deadline) return { kind: "timeout", instance: inst };
    await sleep(pollMs);
  }
}
