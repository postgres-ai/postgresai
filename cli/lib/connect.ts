import { Client } from "pg";
import { parse as parseConnString, parseIntoClientConfig } from "pg-connection-string";
import { generateAllReports } from "./checkup";
import { createCheckupReport, uploadCheckupReportJson } from "./checkup-upload";
import { findService } from "./clickhouse";
import {
  applyInitPlan, buildInitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER,
  maskConnectionString, redactPasswordsInSql, resolveAdminConnection, resolveMonitoringPassword, verifyInitSetup,
} from "./init";
import { assertEncodedUserinfo, estimateLine, monitoringStatus, noCardNext, notAdminNext, type DeployStatus } from "./deploy-surface";
import { collectorConnection, splitChannelBinding, verifyCollectorTls } from "./instances";
import { callRpc, type ProjectListItem } from "./joe";
import { listOrgs, type OrgScope } from "./org-scope";
import { HttpStatusError, requestTimeoutSignal } from "./util";

// `pgai mon deploy <database-url>`: put a database under PostgresAI's care
// (postgres-ai/internal#354). Each step is skipped when already done, so a
// re-run is safe; every outcome carries the exact next action.

export type Provider = "clickhouse" | "rds" | "supabase" | "self-managed";
/** One vocabulary with pgai dblab deploy (lib/deploy-surface). */
export type Status = DeployStatus;

/**
 * The express checkup run while the box starts: what it found, or why it could not run.
 * Every check is in one of findings (warning, ok), info (an inventory, no verdict) or failed.
 * report_id: saved as that report (pgai reports files <id>); upload_error: why it was not.
 */
export type CheckupResult = {
  checks: number;
  findings: { check_id: string; title: string; status: string; message: string }[];
  info: string[];
  failed?: string[];
  report_id?: number;
  upload_error?: string;
} | { error: string };

/** A step of mon deploy, once each: a line for a person, a JSON event for an agent. */
export interface ProgressEvent {
  event: "billing" | "preparing" | "provisioning" | "checkup" | "box";
  /** Seconds since mon deploy started. */
  elapsed_s: number;
  message: string;
  /** event box: the box's state (launch_requested, registered, active, ...). */
  state?: string;
  /** event box: the instance, for following it later (pgai mon instances watch <id>). */
  id?: string;
  checkup?: CheckupResult;
}

export interface ConnectResult {
  status: Status;
  provider: Provider;
  name: string;
  id?: string;
  /** The project's Health Matrix in the console. */
  health_url?: string;
  dashboard_url?: string | null;
  host_metrics?: boolean;
  first_checkup_eta?: string;
  checkup?: CheckupResult;
  /** A new box: what it costs ("$512.00/month per database cluster (scale plan)", "free (1 of 2 free slots)", "included: same database cluster as ..."). */
  price?: string;
  requires_payment_method?: boolean;
  coupon?: { code: string; valid: boolean; description?: string; error?: string };
  next: string;
  sql?: string;
}

/** v1.cloud_monitoring_quote: what the next box costs the org. Amounts in cents. */
export interface Quote {
  plan: string;
  org_alias: string;
  billed: boolean;
  free_slots: { remaining: number; total: number; until?: string };
  subscription: boolean;
  quantity: number;
  price: { amount: number; currency: string; interval: string };
  has_payment_method: boolean;
  requires_payment_method: boolean;
  promo?: { code: string; valid: boolean; error?: string; discount_description?: string; duration?: string; duration_in_months?: number };
  amount_after_promo?: number;
  /** The org's billing page on this platform's console (a preview's, on a preview). */
  billing_url?: string;
  /** A database the org already monitors in this cluster: this one is included (billed: false). */
  same_cluster?: string;
}

/** A row of v1.cloud_monitoring_list. */
export interface Database {
  id: string;
  name: string;
  /** lower(host):port of the server, when known. */
  cluster?: string | null;
  /** The percent-encoded database path, when known. */
  database?: string | null;
  provider: string;
  status: string | null;
  /** The project's Health Matrix in the console (added by platformDeps, not the platform). */
  health_url?: string;
  dashboard_url: string | null;
  host_metrics: boolean;
  /** When the box registered with the platform (it then sets up the monitoring). */
  registered_at?: string | null;
  /** The platform keeps postgres_ai_mon's password for this database's server (another database there connects without it). */
  monitoring_password_stored?: boolean;
  /** Why billing failed: a declined first charge (billing starts when the box is active) removes the box. */
  billing_error?: string | null;
}

/**
 * The monitoring URL (with a note for the user, if any), or what to do first.
 * `generated`: this run created the role with a password generated here, which
 * nobody has once the run ends.
 */
export type Prepared = { monitoringUrl: string; note?: string; generated?: true; storedPassword?: true } | { next: string; sql?: string; resettable?: true } | { checked: true };

export interface ConnectDeps {
  verifyTls?(url: string): Promise<void>;
  list(): Promise<Database[]>;
  /** A row with its console links, for the result shown: the Health Matrix, and Grafana through the PostgresAI sign-in. */
  links?(row: Database): Promise<Database>;
  create(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; status: string; error?: string }>;
  /** `check`: only whether the URL can work, nothing changed ({ checked: true } or what to do first). */
  prepare(url: string, provider: Provider, opts?: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[] }): Promise<Prepared>;
  /** Drops the role a `generated` prepare created; false when it could not. */
  unprepare(url: string): Promise<boolean>;
  localStackRunning(): boolean;
  clickhouseOrg(host: string, keyId: string, keySecret: string): Promise<{ orgId: string; state: string }>;
  /** The express checkup over the monitoring role's URL, saved as a report of `project`. */
  checkup(url: string, project: string): Promise<CheckupResult>;
  selfHosted(monitoringUrl: string, env: Record<string, string>): Promise<void>;
  handoffUrl(provider: "rds" | "supabase"): Promise<string>;
  /** What the next box costs, with the coupon checked. */
  /** `cluster`: lower(host):port of the URL, which the price is per. */
  quote(coupon?: string, cluster?: string): Promise<Quote>;
  /** The console page where the org adds a payment method, when the quote names none. */
  billingUrl(orgAlias: string): string;
  /** Asks a person; false when nobody can be asked. */
  confirm(question: string): Promise<boolean>;
  /** Asks a person for the database server's vCPUs; undefined when skipped or nobody can be asked. */
  askVcpus?(question: string): Promise<number | undefined>;
  /** The platform's per-server lock for --reset-password: one run at a time may reset postgres_ai_mon. */
  resetLock(server: string): Promise<{ lock_id: string }>;
  resetUnlock(lockId: string): Promise<unknown>;
  sleep(ms: number): Promise<void>;
  now(): number;
  progress(event: ProgressEvent): void;
  /** Ctrl-C was pressed: stop before the box is requested; once it is, detach. */
  aborted?(): boolean;
}

export interface ConnectOptions {
  provider?: string;
  clickhouseKey?: string;
  selfHosted?: boolean;
  /** An existing postgres_ai_mon gets a new password (its old one is not needed). */
  resetPassword?: boolean;
  waitMs: number;
  /** The URL comes from an agent (the MCP tool): nothing of this process's environment or files goes to the host it names. */
  agent?: boolean;
  /** A billed box is accepted without asking. */
  yes?: boolean;
  /** A Stripe promotion code for the org's subscription. */
  coupon?: string;
  /** --name: the instance's name (default: host[:port]/database). */
  name?: string;
  /** --location: the Hetzner location (default: any with capacity). */
  location?: string;
  /** --vcpus: the database server's vCPUs; AAS is collected only with them. */
  vcpus?: number;
}

/** Asked at a terminal when --vcpus is missing: AAS (database load) is collected only with it. */
export const VCPUS_QUESTION = "How many vCPUs does the database server have? If you skip this (Enter), you get no AAS (database load) data for this database. vCPUs: ";
/** Said when a box is requested without vCPUs. */
export const NO_VCPUS_NOTE = "Note: no vCPUs given, so you get no AAS (database load) data for this database (pass --vcpus <n> when deploying).";

const money = (cents: number, currency: string) =>
  currency.toLowerCase() === "usd" ? `$${(cents / 100).toFixed(2)}` : `${(cents / 100).toFixed(2)} ${currency.toUpperCase()}`;

/** The quote as one line: free, included (a cluster already billed), the price per cluster, or the price after the coupon. */
export function priceText(q: Quote): string {
  if (!q.billed && q.same_cluster) return `included: same database cluster as ${q.same_cluster}, no extra charge`;
  if (!q.billed) return q.free_slots.total ? `free (${q.free_slots.remaining} of ${q.free_slots.total} free slots)` : "free";
  const { amount, currency, interval } = q.price;
  const base = `${money(amount, currency)}/${interval} per database cluster (${q.plan} plan)${q.subscription ? `, cluster ${q.quantity + 1} on the subscription` : ""}`;
  if (q.promo?.valid && q.amount_after_promo !== undefined && q.amount_after_promo !== null) {
    const after = `${money(q.amount_after_promo, currency)}`;
    const how = `with ${q.promo.code} (${q.promo.discount_description})`;
    if (q.promo.duration === "forever") return `${after}/${interval} ${how}, instead of ${base}`;
    if (q.promo.duration === "repeating") return `${after}/${interval} for ${q.promo.duration_in_months} months ${how}, then ${base}`;
    return `${after} the first ${interval} ${how}, then ${base}`;
  }
  return base;
}

// The reporter's first run is 30 minutes after the stack starts
// (REPORTER_INITIAL_DELAY_SECONDS in config/scripts/postgres-reports.sh).
const FIRST_CHECKUP_DELAY_MS = 30 * 60_000;
const POLL_MS = 15_000;
export const PROVIDERS: Provider[] = ["clickhouse", "rds", "supabase", "self-managed"];

/** `new URL`, or undefined for text that is not a URL (URL.canParse needs Node 18.17). */
export function parseUrl(url: string): URL | undefined {
  try {
    return new URL(url);
  } catch {
    return undefined;
  }
}

export function detectCloudProvider(url: string): Provider {
  const host = new URL(url).hostname.toLowerCase().replace(/\.$/, "");
  if (host.endsWith(".clickhouse.cloud")) return "clickhouse";
  if (host.endsWith(".rds.amazonaws.com")) return "rds";
  if (host.endsWith(".supabase.co") || host.endsWith(".pooler.supabase.com")) return "supabase";
  return "self-managed";
}

/**
 * The instance name the platform derives from the monitoring URL
 * (monitoring_instance_create, #712). That URL always names the database, so
 * for a URL without one this is the database pg connects to (PGDATABASE, else the user).
 */
export function databaseName(url: string): string {
  const u = new URL(url);
  const port = u.port && u.port !== "5432" ? `:${u.port}` : "";
  let db = decodeURIComponent(u.pathname.replace(/^\//, ""));
  if (!db) {
    try {
      db = new Client(resolveAdminConnection({ conn: url }).clientConfig).database ?? "";
    } catch {
      // Not a URL pg can use: the connection reports that.
    }
  }
  return `${u.hostname}${port}${db ? `/${db}` : ""}`;
}

/** `<key-id>:<key-secret>`, from --clickhouse-key or CLICKHOUSE_KEY_ID + CLICKHOUSE_KEY_SECRET. */
export function parseClickhouseKey(value: string | undefined, env: Record<string, string | undefined>) {
  if (!value) {
    return env.CLICKHOUSE_KEY_ID && env.CLICKHOUSE_KEY_SECRET
      ? { keyId: env.CLICKHOUSE_KEY_ID, keySecret: env.CLICKHOUSE_KEY_SECRET }
      : undefined;
  }
  const i = value.indexOf(":");
  if (i <= 0 || i === value.length - 1) throw new Error("--clickhouse-key must be <key-id>:<key-secret>");
  return { keyId: value.slice(0, i), keySecret: value.slice(i + 1) };
}

/** A delete in flight (not one that failed to launch, which can be retried). */
export const disconnecting = (status: string | null) => /delet/.test(status ?? "") && !/fail/.test(status ?? "");

/** The platform's instance state in the words mon deploy, mon instances status and mon instances list all use. */
export function stateOf(raw: string | null): Status {
  return monitoringStatus(raw);
}

/** A rejected ClickHouse Cloud key, or one that cannot see the service: the user's to fix (exit 3). */
export class ClickhouseKeyError extends Error {}

/** Ctrl-C before the box was requested: the run stopped, its lock released and its generated role dropped. */
export class CancelledError extends Error {
  constructor() {
    super("Cancelled: nothing was created.");
  }
}

/** The server of a cluster or legacy database name, ignoring case and one trailing dot. */
const serverOf = (name: string) => name.split("/")[0].toLowerCase().replace(/\.(?=$|:)/, "");

/** lower(host):port of a URL (5432 when it names none): the cluster the price is per, as the platform keys it. */
export function clusterOf(url: string): string | undefined {
  const u = parseUrl(url);
  return u?.hostname ? `${serverOf(u.hostname)}:${u.port || "5432"}` : undefined;
}

/** An error's text; an AggregateError (node tried each address of a host name) has none of its own: its errors'. */
export const errorText = (err: unknown): string => {
  if (err instanceof AggregateError && !err.message) return err.errors.map(errorText).join("; ");
  // A platform refusal: its own words (details, hint), not the "<call>: HTTP 400 - Bad Request" line above them.
  if (err instanceof HttpStatusError) {
    const [first, ...rest] = err.message.split("\n");
    if (rest.length && /: HTTP \d{3}\b/.test(first)) return rest.join("\n");
  }
  return err instanceof Error ? err.message : String(err);
};

const urlPassword = (u: URL) => {
  for (const key of [...u.searchParams.keys()]) if (key !== "password") u.searchParams.delete(key);
  return parseConnString(u.toString()).password || "";
};

// What of the given URL's query string goes to the box: `options`, certificate
// paths and the rest describe this machine's session, not the box's.
const URL_PARAMS_KEPT = ["sslmode", "channel_binding", "application_name"];
// Kept for the logins made from this machine: the files are here.
const URL_PARAMS_TLS = ["sslrootcert", "sslcert", "sslkey", "uselibpqcompat"];

/**
 * Refuses query parameters that pg obeys over the URL itself. `host` and `port`
 * would prepare one server while the name and the box's URL say another. An
 * agent's URL may carry only what the box gets (and its password): `sslcert`
 * or `sslkey` would present this machine's files to a host the agent chose.
 */
export function checkUrlParams(url: string, agent?: boolean) {
  const u = new URL(url);
  const keys = [...new Set(u.searchParams.keys())];
  if (agent) {
    for (const key of ["password", "user", "host", "port", "dbname"]) {
      if (u.searchParams.getAll(key).length > 1) throw new Error(`database_url query parameter ${key} must appear only once`);
    }
    if (u.password && u.searchParams.has("password")) throw new Error("database_url must give the password only once, in the authority or in ?password=");
    const refused = keys.filter((k) => k !== "password" && !URL_PARAMS_KEPT.includes(k));
    if (refused.length) throw new Error(`database_url may carry only these query parameters: ${URL_PARAMS_KEPT.join(", ")} (got: ${refused.join(", ")})`);
    u.searchParams.delete("sslmode");
    u.searchParams.delete("channel_binding");
    const config = parseIntoClientConfig(u.toString());
    if (!/^postgres(ql)?:$/.test(u.protocol) || typeof config.password !== "string" || !config.password) throw new Error("database_url must be postgresql://user:password@host:5432/dbname, with the password in it");
    return { ...config, password: config.password };
  }
  const moved = keys.filter((k) => k === "host" || k === "port");
  if (moved.length) throw new Error(`The URL's query string sets ${moved.join(" and ")}: put the host and the port in the URL itself (postgresql://user:password@host:5432/dbname), so that the server prepared is the server monitored`);
}

/**
 * What a box URL without a password lacks before the platform fills in the one it keeps
 * (cloud_monitoring_connect): sslmode=require or verify-*, each kept parameter once, no empty
 * part. Undefined when it lacks nothing.
 */
function storedPasswordNeeds(monitoringUrl: string): string | undefined {
  const query = new URL(monitoringUrl).search.replace(/^\?/, "");
  const parts = query ? query.split("&") : [];
  const keys = parts.map((p) => p.split("=")[0]);
  const twice = URL_PARAMS_KEPT.filter((k) => keys.indexOf(k) !== keys.lastIndexOf(k));
  const tls = !twice.includes("sslmode") && parts.some((p) => /^sslmode=(require|verify-ca|verify-full)$/.test(p));
  const needs = [
    ...(tls ? [] : [`sslmode=require (or verify-full)${twice.includes("sslmode") ? ", once" : ""}`]),
    ...twice.filter((k) => k !== "sslmode").map((k) => `${k} once`),
    ...(parts.some((p) => !p) ? ["no stray &"] : []),
    // A kept key spelled with %-escapes: the platform reads the query string as written.
    ...(keys.some((k) => k && !URL_PARAMS_KEPT.includes(k)) ? ["the parameter names without %-escapes"] : []),
  ];
  return needs.length ? needs.join(" and ") : undefined;
}

/** A word for the user's shell: quoted unless plain (a name made in the console is free text). */
const shellWord = (s: string) => (/^[\w./:@%+=,-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);

/** `pgai mon instances delete` for each database named. */
const disconnectEach = (names: string[]) => names.map((n) => `pgai mon instances delete ${shellWord(n)} --yes`).join(" and ");

/** postgres_ai_mon's URL for the prepared database, with the query parameters in `kept`. */
function roleUrlFor(url: string, db: string, password: string, kept: string[]): string {
  const u = new URL(url);
  u.username = DEFAULT_MONITORING_USER;
  u.password = encodeURIComponent(password);
  u.pathname = `/${encodeURIComponent(db)}`;
  for (const name of [...u.searchParams.keys()]) if (!kept.includes(name)) u.searchParams.delete(name);
  // `mon targets add` refuses a raw '@' after the host (a delete above already encoded it).
  u.search = u.search.replace(/@/g, "%40");
  return u.toString();
}

/** The URL a box uses: only postgres_ai_mon's credentials, and the prepared database by name. */
const monitoringUrlFor = (url: string, db: string, password: string) => roleUrlFor(url, db, password, URL_PARAMS_KEPT.filter((p) => p !== "channel_binding"));

/** The same login from this machine: with the given URL's TLS files (a private CA, say). */
const loginUrlFor = (url: string, db: string, password: string) => roleUrlFor(url, db, password, [...URL_PARAMS_KEPT, ...URL_PARAMS_TLS]);

/** What to tell the user when the URL verifies the server with a CA file the box will not have. */
export function caNote(url: string): string | undefined {
  const q = new URL(url).searchParams;
  if (!q.get("sslrootcert") || !/^verify-/.test(q.get("sslmode") ?? "")) return undefined;
  return `the monitoring box has no copy of the CA in sslrootcert: with sslmode=${q.get("sslmode")} it connects only to a server certificate signed by a public CA (else use sslmode=require)`;
}

/** A platform row as a result with its next action; `fresh` (provisioned just now) adds the first-checkup ETA. */
export function connectStatus(row: Database, provider = row.provider as Provider, fresh = false): ConnectResult {
  const base = { provider, name: row.name, id: row.id, ...(row.health_url ? { health_url: row.health_url } : {}), dashboard_url: row.dashboard_url, host_metrics: row.host_metrics };
  const status = stateOf(row.status);
  // Removed because its first charge failed: the user's to fix, whichever command reads it (deploy, watch, list).
  if (row.billing_error && (status === "deleting" || status === "deleted")) {
    return { ...base, status: "action_required",
      next: `The first charge failed (${row.billing_error}): the box is being removed and nothing is billed. Update the payment method, then re-run pgai mon deploy.` };
  }
  if (status === "ready") {
    return {
      ...base, status,
      ...(fresh ? { first_checkup_eta: new Date(Date.now() + FIRST_CHECKUP_DELAY_MS).toISOString() } : {}),
      next: row.health_url || row.dashboard_url ? `Open ${row.health_url || row.dashboard_url}` : `pgai mon instances watch ${row.id}`,
    };
  }
  if (status === "deleting" || status === "deleted" || status === "inactive") return { ...base, status, next: "none" };
  if (status === "failed") return { ...base, status, next: `pgai mon instances delete ${row.id} --yes, then pgai mon deploy again` };
  return { ...base, status, next: `pgai mon instances watch ${row.id}` };
}

/** The monitoring URL with the admin URL's TLS files: for a login from this machine, never for the box. */
function withLocalTls(monitoringUrl: string, url: string): string {
  const u = new URL(monitoringUrl);
  for (const [k, v] of new URL(url).searchParams) if (URL_PARAMS_TLS.includes(k)) u.searchParams.set(k, v);
  return u.toString();
}

/** What renders progress: JSON events, plain lines, or a step view at a terminal. */
export interface StepViewLike {
  start(): void;
  advance(key: string): void;
  log(text: string): void;
  setNote(note: string): void;
  complete(): void;
  fail(key?: string): void;
  stop(): void;
}

/**
 * mon deploy's progress: JSON events (--json), a line a step (piped), or at a
 * terminal a step view that starts only after the price, since a prompt may
 * follow the price; with no price (a re-run, --self-hosted) no view at all.
 */
export function stepProgress(o: { json: boolean; tty: boolean; makeView: () => StepViewLike; writeLine: (s: string) => void;
  writeEvent: (e: ProgressEvent) => void; onId?: (id: string) => void }) {
  let view: StepViewLike | undefined;
  let priced = false;
  const progress = (e: ProgressEvent): void => {
    if (e.id) o.onId?.(e.id);
    if (o.json) return o.writeEvent(e);
    if (!o.tty) return o.writeLine(progressText(e));
    if (e.event === "billing") {
      priced = true;
      return o.writeLine(e.message);
    }
    if (!view && !priced) return o.writeLine(e.message);
    if (!view) {
      view = o.makeView();
      view.start();
    }
    view.advance(e.event);
    if ((e.event === "checkup" && !/^Running /.test(e.message)) || /^(Note|Warning):/.test(e.message)) view.log(e.message);
    if (e.event === "box" && e.state) view.setNote(`box: ${boxWords(e.state)}`);
  };
  return { progress, view: () => view };
}

/** A result for a person at a terminal: a few sentences, links whole (JSON keeps every field). */
export function resultLines(r: ConnectResult): string[] {
  const who = r.name || "this database";
  // What the next action carries after its first part ("; --name ignored ...", a prepare note).
  const extra = (r.next ?? "").split("; ").slice(1).join("; ");
  const billing = (r as { billing?: string }).billing;
  switch (r.status) {
    case "ready":
      return [
        `Monitoring for ${who} is ready.`,
        ...(r.health_url ? [`  Health Matrix: ${r.health_url}`] : []),
        ...(r.dashboard_url ? [`  Grafana:       ${r.dashboard_url}`] : []),
        ...(r.price ? [`  Price:         ${r.price}`] : []),
        ...(!r.health_url && !r.dashboard_url && r.next && r.next !== "none" ? [`Next: ${r.next}`] : []),
        ...(extra && (r.health_url || r.dashboard_url) ? [`Note: ${extra}`] : []),
      ];
    case "in_progress":
      return [`Monitoring for ${who} is still being set up. Follow it: ${r.next}`];
    case "deleting":
    case "deleted":
      return [`Deleting the monitoring of ${who}.${billing ? ` Billing: ${billing}.` : ""}${r.next && r.next !== "none" ? ` ${r.next}` : ""}`];
    default:
      return [r.next];
  }
}

/** The express checkup for a person: a line per warning, then the other checks by id, and where it was saved. */
export function checkupLines(c: CheckupResult): string[] {
  if ("error" in c) return [`Express checkup could not run: ${c.error}`];
  const warnings = c.findings.filter((f) => f.status === "warning");
  const ok = c.findings.filter((f) => f.status === "ok");
  const failed = c.failed ?? [];
  const counts = [`${warnings.length} warning${warnings.length === 1 ? "" : "s"}`, `${ok.length} ok`, `${c.info.length} info`, ...(failed.length ? [`${failed.length} could not run`] : [])];
  return [
    `Express checkup while the box starts (${c.checks} checks: ${counts.join(", ")}):`,
    ...warnings.map((f) => `  ${f.check_id} ${f.title}: ${f.message}`),
    ...(ok.length ? [`  ok: ${ok.map((f) => f.check_id).join(" ")}`] : []),
    ...(c.info.length ? [`  info: ${c.info.join(" ")}`] : []),
    ...(failed.length ? [`  could not run: ${failed.join(" ")}`] : []),
    c.report_id ? `Saved as report ${c.report_id}: pgai reports files ${c.report_id}` : `Not saved to PostgresAI: ${c.upload_error}`,
    "The full checkup (query analysis and trends) follows on the box.",
  ];
}

/** A progress event as a person reads it: the time since the start ends its first line (before a colon). */
export function progressText(e: ProgressEvent): string {
  const s = Math.round(e.elapsed_s);
  const elapsed = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  const [first, ...rest] = e.message.split("\n");
  const colon = first.endsWith(":") ? ":" : "";
  return [`${first.slice(0, first.length - colon.length)} (+${elapsed})${colon}`, ...rest].join("\n");
}

/** The box's state as deploy shows it: registered between the launch and active. */
const boxState = (row: Database) => (row.status === "launch_requested" && row.registered_at ? "registered" : row.status ?? "starting");

/** The box's state in words (the raw state stays in the JSON event's `state`). */
export function boxWords(state: string | null | undefined): string {
  if (!state || state === "starting" || state === "launch_requested") return "starting";
  if (state === "registered") return "installing monitoring";
  return monitoringStatus(state).replace("_", " ");
}

type Billing = Pick<ConnectResult, "price" | "requires_payment_method" | "coupon">;

/**
 * A new box's billing: its fields for the result, the billing page for a
 * missing payment method, and `stop` when something comes first (a coupon
 * that does not apply, no payment method, a billed box not accepted).
 */
async function billingFor(q: Quote, name: string, opts: ConnectOptions, deps: ConnectDeps, show: (price: string) => void): Promise<{ billing: Billing; billingPage: Billing & { next: string }; billingUrl: string; accepted: boolean; stop?: Billing & { next: string } }> {
  const billing: Billing = { price: priceText(q), requires_payment_method: q.requires_payment_method };
  if (opts.coupon) {
    billing.coupon = q.promo?.valid
      ? { code: q.promo.code, valid: true, description: q.promo.discount_description }
      : { code: opts.coupon, valid: false, error: q.promo?.error ?? "not valid" };
  }
  const billingUrl = q.billing_url ?? deps.billingUrl(q.org_alias);
  const billingPage = { ...billing, requires_payment_method: true, next: noCardNext(billingUrl) };
  const stop = (next: string) => ({ billing, billingPage, billingUrl, accepted: false, stop: { ...billing, next } });
  if (billing.coupon && !billing.coupon.valid) {
    return stop(`Promo code ${opts.coupon}: ${billing.coupon.error}. Nothing was changed: re-run with a valid code, or without ${opts.agent ? "coupon" : "--coupon"}`);
  }
  if (q.requires_payment_method) return { billing, billingPage, billingUrl, accepted: false, stop: billingPage };
  // The price on its own line first: a prompt may wrap or be cut where it is shown.
  show(billing.price!);
  if (q.billed && !opts.yes && !(await deps.confirm(`Provision ${name}? (y/N): `))) {
    return stop(opts.agent ? `Call deploy_monitoring again with yes: true to accept ${billing.price}` : `Re-run with --yes to accept ${billing.price}`);
  }
  return { billing, billingPage, billingUrl, accepted: q.billed };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
  const started = deps.now();
  const progress = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}) =>
    deps.progress({ event, elapsed_s: Math.round((deps.now() - started) / 1000), message, ...more });
  const provider = (opts.provider ?? detectCloudProvider(url)) as Provider;
  if (!PROVIDERS.includes(provider)) throw new Error(`--provider must be one of: ${PROVIDERS.join(", ")}`);
  // Before any name is taken from the URL (the MCP path does not go through the CLI's argument check).
  assertEncodedUserinfo(url);
  checkUrlParams(url, opts.agent);
  const name = databaseName(url);
  try {
    const collector = await collectorConnection(url, deps.verifyTls);
    const required = splitChannelBinding(url).value === "require";
    url = collector.url;
    if (collector.note) progress("preparing", `${required ? "Warning" : "Note"}: ${collector.note}`);
    if (required && caNote(url)) progress("preparing", `Note: ${caNote(url)}`);
  } catch (err) {
    return { status: "action_required", provider, name, next: err instanceof Error ? err.message : String(err) };
  }
  const cluster = clusterOf(url)!;
  let vcpus = opts.vcpus;
  // The exported key pair is read for ClickHouse only, and never for an agent's URL; elsewhere only the flag is an error.
  const key = parseClickhouseKey(opts.clickhouseKey, provider === "clickhouse" && !opts.agent ? process.env : {});
  if (key && provider !== "clickhouse") throw new Error("--clickhouse-key applies to ClickHouse Managed Postgres only");

  if (!opts.selfHosted && (provider === "rds" || provider === "supabase")) {
    return { status: "action_required", provider, name, next: `Finish in the console: ${await deps.handoffUrl(provider)}` };
  }

  if (opts.selfHosted && opts.resetPassword) {
    return { status: "action_required", provider, name, next: "--reset-password works with PostgresAI Cloud only (it checks what else monitors this server): deploy without --self-hosted, or set PGAI_MON_PASSWORD" };
  }

  if (opts.selfHosted && deps.localStackRunning()) {
    return { status: "action_required", provider, name, next: "A monitoring stack already runs on this machine: add the database with PGAI_DB_URL='<postgres_ai_mon URL>' pgai mon targets add" };
  }

  const findDatabase = (list: Database[]) => {
    const live = list.filter((d) => !disconnecting(d.status));
    return live.find((d) => d.cluster && serverOf(d.cluster) === cluster && d.database != null && decodeURIComponent(d.database) === name.split("/").slice(1).join("/"))
      ?? live.find((d) => (d.cluster == null || d.database == null) && d.name === name);
  };
  const rows = opts.selfHosted ? [] : await deps.list();
  let row = findDatabase(rows);
  const fresh = !row;
  let note = "";
  let checkup: CheckupResult | undefined;
  let billing: Billing = {};
  let billingPage: Billing & { next: string } | undefined;
  let accepted = false;
  let billingUrl: string | undefined;
  // Drops the role this run created with a generated password, once its box is gone.
  let undoRole: (() => Promise<unknown>) | undefined;
  // The key is checked first, on a re-run too: a rejected key or a stopped service changes nothing.
  const findOrg = async () => {
    try {
      return await deps.clickhouseOrg(new URL(url).hostname, key!.keyId, key!.keySecret);
    } catch (err) {
      if (err instanceof ClickhouseKeyError) return { error: err.message };
      throw err;
    }
  };
  // A database already monitored keeps its name: --name names a new instance only.
  if (row && opts.name && row.name !== opts.name) note = `; --name ignored: this database is already monitored as ${row.name}`;
  if (row && opts.resetPassword) {
    return { status: "action_required", provider, name, id: row.id, next: `${name} is already monitored, and its monitoring uses the current password: pgai mon instances delete ${name} --yes first` };
  }
  if (row && key) {
    const found = await findOrg();
    if ("error" in found) return { status: "action_required", provider, name, id: row.id, next: `${found.error} Nothing was changed: re-run with the right key, or without one` };
  }
  if (!row) {
    if (!/^[A-Za-z0-9._-]*$/.test(new URL(url).pathname.replace(/^\//, "")) && rows.some((d) =>
      !disconnecting(d.status) && d.cluster && serverOf(d.cluster) === cluster && d.database == null && /^Monitoring [0-9a-f]{6}$/.test(d.name))) {
      return { status: "action_required", provider, name, next: "An earlier box on this server may already monitor this database (its name is not known): see pgai mon instances list, and delete that box first (pgai mon instances delete <id>) if it is this database" };
    }
    let ch: { orgId: string } | undefined;
    if (key) {
      const found = await findOrg();
      if ("error" in found) return { status: "action_required", provider, name, next: `${found.error} Then re-run` };
      if (found.state !== "running") {
        return { status: "action_required", provider, name, next: `Start the service in the ClickHouse Cloud console (it is ${found.state}), then re-run` };
      }
      ch = found;
    }
    // A refused launch starts no box, so a role with a generated password is
    // dropped again: nobody has that password, and the re-run would stop at it.
    const undo = async (p: Prepared) => { if ("generated" in p && p.generated) await deps.unprepare(url).catch(() => false); };
    let lockId: string | undefined;
    // postgres_ai_mon is one role for the whole server: a new password cuts off what uses the old one.
    // Without a cluster, the server is not known.
    const unread = (d: Database) => !d.cluster;
    const onServer = (d: Database) => !!d.cluster && serverOf(d.cluster) === cluster;
    const othersOnServer = (list: Database[]) => list.filter((d) => !disconnecting(d.status) && (unread(d) || onServer(d)));
    // The platform keeps postgres_ai_mon's password for a server it monitors a database on, for this org,
    // and fills it in while a box there is not being destroyed (a delete that failed counts as one).
    const storedPassword = rows.some((d) => d.monitoring_password_stored && !/delet/.test(d.status ?? "") && onServer(d)) || undefined;
    const cutOff = (others: Database[]): ConnectResult => {
      const on = others.filter((d) => !unread(d)).map((d) => d.name).join(", ");
      const maybe = others.filter(unread).map((d) => d.name).join(", ");
      const what = [on && `would cut off the monitoring of ${on} on this server`, maybe && `may cut off the monitoring of ${maybe} (the platform could not read its URL, so its server is not known)`].filter(Boolean).join(", and ");
      const instead = [
        ...(storedPassword ? ["deploy without --reset-password (PostgresAI keeps its password for this server, and sends it only over TLS)"] : []),
        "set PGAI_MON_PASSWORD to its password instead",
        `${disconnectEach(others.map((d) => d.name))} first`,
      ];
      return { status: "action_required", provider, name, next: `A new password for ${DEFAULT_MONITORING_USER} ${what}: ${instead.join(", or ")}` };
    };
    const others = othersOnServer(rows).filter((d) => !unread(d)).map((d) => d.name);
    if (opts.resetPassword && othersOnServer(rows).length) return cutOff(othersOnServer(rows));
    // What the box costs, before anything is touched: a billed box is accepted
    // (--yes, or at the prompt), and needs the org's payment method.
    if (opts.coupon !== undefined && !opts.coupon.trim()) {
      return { status: "action_required", provider, name, next: `${opts.agent ? "coupon" : "--coupon"} is empty: pass a promotion code, or leave ${opts.agent ? "coupon" : "--coupon"} out` };
    }
    const prepOpts = { resetPassword: opts.resetPassword, ...(storedPassword ? { storedPassword, others } : {}) };
    if (!opts.selfHosted) {
      // A URL that cannot work is told so before any price is asked (nothing is changed here).
      const probe = await deps.prepare(url, provider, { ...prepOpts, check: true }).catch((err) => {
        // A server that cannot be reached: say which ("timeout expired" alone does not).
        if (err instanceof HttpStatusError || /Could not connect to/.test(errorText(err))) throw err;
        throw new Error(`Could not connect to ${cluster}: ${errorText(err)}`);
      });
      if ("next" in probe) {
        const { resettable, ...refusal } = probe;
        if (resettable && !opts.resetPassword && !opts.agent && !opts.yes && await deps.confirm("A monitoring role from an earlier connection exists, and its password isn't stored. Reset it now? Anything else using this role will need the new password. [y/N] ")) {
          return connect(url, { ...opts, resetPassword: true }, deps);
        }
        return { status: "action_required", provider, name, ...refusal };
      }
      let quote: Quote;
      try {
        quote = await deps.quote(opts.coupon, clusterOf(url));
      } catch (err) {
        // Not an org admin: the user's to act on, as with pgai dblab deploy (postgresai#412).
        if (err instanceof HttpStatusError && err.status === 403) return { status: "action_required", provider, name, next: notAdminNext };
        throw err;
      }
      const b = await billingFor(quote, name, opts, deps, (price) => progress("billing", estimateLine(price)));
      if (b.stop) return { status: "action_required", provider, name, ...b.stop };
      if (deps.aborted?.()) throw new CancelledError();
      ({ billing, billingPage, billingUrl, accepted } = b);
      // ClickHouse's vCPUs come from ClickHouse Cloud; elsewhere only the user knows them.
      if (vcpus === undefined && provider !== "clickhouse") {
        vcpus = !opts.yes && !opts.agent && deps.askVcpus ? await deps.askVcpus(VCPUS_QUESTION) : undefined;
        if (deps.aborted?.()) throw new CancelledError();
        if (vcpus === undefined) progress("preparing", opts.agent ? NO_VCPUS_NOTE.replace("--vcpus <n>", "vcpus") : NO_VCPUS_NOTE);
      }
    }
    if (opts.resetPassword) {
      // Another run at once for a database on this server would reset the
      // password too: the platform lets one run at a time hold the server.
      // Taken after the price is accepted: a stop above leaves the server free,
      // and the 15-minute lease does not run while a person decides.
      let lock: { lock_id?: unknown };
      try {
        lock = await deps.resetLock(cluster);
      } catch (err) {
        if (err instanceof HttpStatusError && err.status === 409) {
          return { status: "action_required", provider, name, ...billing, next: `Another pgai mon deploy --reset-password for ${cluster} is running: wait for it to finish, then re-run (a run that stopped frees the server 15 minutes after it started)` };
        }
        if (err instanceof HttpStatusError && err.status === 404) {
          return { status: "action_required", provider, name, ...billing, next: `This platform cannot lock the server for --reset-password yet: set PGAI_MON_PASSWORD to ${DEFAULT_MONITORING_USER}'s password instead` };
        }
        throw err;
      }
      // Without a lock id nothing holds the server: no reset.
      if (typeof lock.lock_id !== "string" || !lock.lock_id) throw new Error("cloud_monitoring_reset_lock returned no lock_id: nothing was changed");
      lockId = lock.lock_id;
    }
    let prepared: Prepared;
    let created: Awaited<ReturnType<ConnectDeps["create"]>> | "payment" | "declined" | "price" | "stored";
    // No answer to the box request: it may still be created with this URL.
    let unanswered = false;
    try {
      // What another run connected on this server before this one got the lock.
      if (lockId) {
        const now = await deps.list();
        const same = findDatabase(now);
        if (same) return { status: "action_required", provider, name, id: same.id, ...billing, next: `${name} is already monitored, and its monitoring uses the current password: pgai mon instances delete ${name} --yes first` };
        const others = othersOnServer(now);
        if (others.length) return { ...cutOff(others), ...billing };
      }
      if (deps.aborted?.()) throw new CancelledError();
      progress("preparing", `Preparing ${maskConnectionString(url)}`);
      prepared = await deps.prepare(url, provider, prepOpts);
      if (deps.aborted?.()) {
        await undo(prepared);
        throw new CancelledError();
      }
      if ("next" in prepared) {
        const { resettable, ...refusal } = prepared;
        return { status: "action_required", provider, name, ...billing, ...refusal };
      }
      if (!("monitoringUrl" in prepared)) throw new Error("prepare returned no monitoring URL");
      if (prepared.note) note = `; ${prepared.note}`;
      if (opts.selfHosted) {
        await deps.selfHosted(prepared.monitoringUrl, key && ch
          ? { CLICKHOUSE_ORG_ID: ch.orgId, CLICKHOUSE_KEY_ID: key.keyId, CLICKHOUSE_KEY_SECRET: key.keySecret }
          : {});
        return { status: "ready", provider, name, dashboard_url: "http://localhost:3000", host_metrics: !!ch, next: `pgai mon health${note}` };
      }
      progress("provisioning", `Provisioning monitoring for ${name}`);
      created = await deps.create({
        db_url: prepared.monitoringUrl,
        ...(provider === "clickhouse" ? { provider } : {}),
        ...(key && ch ? { clickhouse_org_id: ch.orgId, clickhouse_key_id: key.keyId, clickhouse_key_secret: key.keySecret } : {}),
        ...(opts.coupon ? { promo_code: opts.coupon } : {}),
        // The price shown was accepted: without it the platform creates no billed box.
        ...(accepted ? { accept_price: true } : {}),
        // pgai mon deploy --name / --location (postgresai#412).
        ...(opts.name ? { project_name: opts.name } : {}),
        ...(opts.location ? { server_location: opts.location } : {}),
        // The database server's vCPUs: without them AAS is never collected.
        ...(vcpus ? { vcpus } : {}),
      }).catch(async (err) => {
        // A 4xx is a refusal; after anything else (5xx, no answer) a box may be starting with this URL.
        if (err instanceof HttpStatusError && err.status >= 400 && err.status < 500) await undo(prepared);
        else unanswered = true;
        // 402: the payment method went away since the quote; nothing was created.
        if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
        // 412: billed although quoted free (the last free slot went meanwhile); nothing was created.
        if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
        // 409 "none is stored" for a URL without a password: the platform no longer keeps the server's
        // (its last box went meanwhile). Its other 409s (a name taken, say) say what they are.
        if (err instanceof HttpStatusError && err.status === 409 && "storedPassword" in prepared && prepared.storedPassword && /none is stored/.test(err.message)) return "stored" as const;
        throw err;
      });
      if (created === "payment") return { status: "action_required", provider, name, ...billingPage! };
      // A card on file that Stripe declined is not a missing one.
      if (created === "declined") {
        return { status: "action_required", provider, name, ...billingPage!, next: billingPage!.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
      }
      if (created === "price") return { status: "action_required", provider, name, ...billing, next: "The price changed since it was shown: re-run pgai mon deploy to see it" };
      if (created === "stored") {
        // No box of this org on the server keeps it now; what else logs in as the role is not known here.
        const gone = `PostgresAI no longer keeps the password of ${DEFAULT_MONITORING_USER} for this server`;
        return { status: "action_required", provider, name, ...billing, next: opts.agent
          ? `${gone}: pass its URL as database_url, or run pgai mon deploy <admin-url> --reset-password in a terminal`
          : `${gone}: re-run with --reset-password (anything else that logs in as ${DEFAULT_MONITORING_USER} then needs the new one), or set PGAI_MON_PASSWORD to its password` };
      }
      // Before the lock goes: the next run may be creating the role this drops.
      if (created.status === "failed") {
        await undo(prepared);
        return { status: "failed", provider, name, id: created.id, next: `${created.error} Re-run pgai mon deploy later.` };
      }
    } finally {
      // The box has its URL (or none was requested): the next run may go. A box
      // that may still be starting keeps the server until the lease ends: a run
      // that did not see it would reset the password its URL carries.
      if (lockId && !unanswered) await deps.resetUnlock(lockId).catch(() => {});
    }
    row = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: !!ch };
    // Ctrl-C while the box was requested: it exists now, so the run detaches (it keeps going on PostgresAI).
    if (deps.aborted?.()) return { ...connectStatus(row, provider, true), ...billing };
    const made = prepared;
    undoRole = () => undo(made);
    if ((prepared as { storedPassword?: true }).storedPassword) {
      // The password stays with the platform, and the checkup never runs as the admin: its SQL resolves
      // names with the database's search_path, where the database's owner can put functions. (The
      // prepare step's plan runs as the admin with that search_path too: postgres-ai/internal#384.)
      progress("checkup", `Express checkup skipped: it runs only as ${DEFAULT_MONITORING_USER}, whose password PostgresAI keeps for this server. The full checkup follows on the box.`, { id: created.id });
    } else {
      // First value while the box starts (minutes): the express checkup, as the monitoring role.
      progress("checkup", "Running the express checkup while the box starts", { id: created.id });
      checkup = await deps.checkup(withLocalTls((prepared as { monitoringUrl: string }).monitoringUrl, url), created.name).catch((err) => ({ error: errorText(err) }));
      progress("checkup", checkupLines(checkup).join("\n"), { checkup });
    }
  }

  // The box's state, each time it changes while deploy waits for it (not on a re-run of a connected database).
  let shown: string | undefined;
  for (const deadline = deps.now() + opts.waitMs; ;) {
    if (row.billing_error && disconnecting(row.status)) {
      if (undoRole) await undoRole();
      return { status: "action_required", provider, name, id: row.id, ...billing, ...(checkup ? { checkup } : {}),
        next: `The first charge failed (${row.billing_error}): the box is being removed and nothing is billed. Update the payment method${billingUrl ? ` at ${billingUrl}` : ""}, then re-run` };
    }
    const result = connectStatus(row, provider, fresh);
    const state = boxState(row);
    if (state !== shown && (shown !== undefined || result.status === "in_progress")) progress("box", `Monitoring box: ${boxWords(state)}`, { state, id: row.id });
    shown = state;
    if (result.status !== "in_progress" || deps.now() >= deadline) {
      const final = deps.links ? connectStatus(await deps.links(row), provider, fresh) : result;
      if (final.status === "ready" && provider === "clickhouse" && !row.host_metrics) {
        final.next += "; for CPU, memory and disk, delete the instance (pgai mon instances delete) and deploy again with --clickhouse-key <key-id>:<key-secret>";
      }
      const { next, ...rest } = final;
      return { ...rest, ...billing, ...(checkup ? { checkup } : {}), next: next + note };
    }
    await deps.sleep(POLL_MS);
    // The box is already requested: a failed poll keeps the last known state.
    const listed = await deps.list().catch(() => undefined);
    const now = listed?.find((d) => d.id === row!.id);
    if (listed && !now && undoRole) {
      await undoRole();
      return { status: "failed", provider, name, id: row.id, ...billing, next: "The monitoring box was removed before it became active: see pgai mon instances list, then re-run pgai mon deploy" };
    }
    row = now ?? row;
  }
}

type PgClientClass = Parameters<typeof connectWithSslFallback>[0];
type PgClient = Awaited<ReturnType<typeof connectWithSslFallback>>["client"];

export interface PrepareOptions {
  /** The URL comes from an agent (the MCP tool): PGAI_MON_PASSWORD is not read and a TLS failure is not retried in plaintext. */
  agent?: boolean;
  /** An existing postgres_ai_mon gets a new password (PGAI_MON_PASSWORD, else a generated one). */
  resetPassword?: boolean;
  /** Only whether the URL can work: every refusal, nothing created or changed. */
  check?: boolean;
  /** The platform keeps postgres_ai_mon's password for this server: an existing role needs no PGAI_MON_PASSWORD, and the box's URL carries none. */
  storedPassword?: boolean;
  /** The org's other databases on this server (named where they must be disconnected first). */
  others?: string[];
  /** The pg client class (tests). */
  Client?: PgClientClass;
}

/** A session over `url`; `refusedTls`: the server refused TLS, and the session fell back to plaintext. */
async function openConnection(url: string, opts: PrepareOptions) {
  const config = checkUrlParams(url, opts.agent);
  const conn = resolveAdminConnection({ conn: url });
  if (config) {
    const { connectionString, ...explicit } = conn.clientConfig;
    conn.clientConfig = { ...config, ...explicit };
  }
  const fallback = !opts.agent && !!conn.sslFallbackEnabled;
  const { client, usedSsl } = await connectWithSslFallback(opts.Client ?? Client, { ...conn, sslFallbackEnabled: fallback });
  await client.query("set search_path = pg_catalog, pg_temp");
  return { client, usedSsl, refusedTls: fallback && !usedSsl };
}

/** Drops the monitoring role and what it was granted in this database; false when the server refuses. */
async function dropMonitoringRole(client: PgClient): Promise<boolean> {
  // Alone first: a role with no grants yet needs no DROP OWNED, which a CREATEROLE admin may not run.
  for (const sql of [`drop role ${DEFAULT_MONITORING_USER}`, `drop owned by ${DEFAULT_MONITORING_USER}; drop role ${DEFAULT_MONITORING_USER}`]) {
    if (await client.query(sql).then(() => true, () => false)) return true;
  }
  return false;
}

/** Undoes a `generated` prepareDatabase over the same admin URL: drops the role that run created. */
export async function unprepareDatabase(url: string, opts: PrepareOptions = {}): Promise<boolean> {
  const { client } = await openConnection(url, opts);
  try {
    return await dropMonitoringRole(client);
  } finally {
    await client.end();
  }
}

/** Creates (or checks) the monitoring role over the given URL. The admin URL is used for this run only. */
export async function prepareDatabase(url: string, provider: Provider, opts: PrepareOptions = {}): Promise<Prepared> {
  const pgProvider = provider === "rds" ? "self-managed" : provider;
  const open = (u: string) => openConnection(u, opts);
  // Whether `u` logs in; false only when the server rejects the password.
  const logsIn = async (u: string): Promise<boolean> => {
    try {
      await (await open(u)).client.end();
      return true;
    } catch (err) {
      const { code, routine } = err as { code?: string; routine?: string };
      if (code === "28P01") return false;
      // No CONNECT on the database yet: CheckMyDatabase runs after the password was accepted.
      if (code === "42501" && routine === "CheckMyDatabase") return true;
      throw err;
    }
  };
  const { client, usedSsl, refusedTls } = await open(url).catch((err) => {
    // pg's words for "the server asked for a password and there is none".
    if (/client password must be a string/.test(String(err?.message))) throw new Error("The URL has no password (and PGPASSWORD is not set): pgai mon deploy postgresql://user:password@host:5432/dbname");
    throw err;
  });
  try {
    // Admin: a superuser, or CREATEROLE that can run the rest of the plan too: grant pg_monitor and
    // pg_read_all_stats (PG 16+ needs ADMIN OPTION for that), create the postgres_ai schema, and
    // find pg_stat_statements installed (creating it takes a superuser).
    const me = (await client.query(
      `select session_user as name, current_database() as db,
         rolsuper or (rolcreaterole
           and (current_setting('server_version_num')::int < 160000
             or (pg_has_role(current_user, 'pg_monitor', 'USAGE WITH ADMIN OPTION')
               and pg_has_role(current_user, 'pg_read_all_stats', 'USAGE WITH ADMIN OPTION')))
           and has_database_privilege(current_user, current_database(), 'CREATE')
           and exists (select 1 from pg_extension where extname = 'pg_stat_statements')) as admin,
         exists (select 1 from pg_roles where rolname = '${DEFAULT_MONITORING_USER}') as mon_exists,
         current_setting('scram_iterations', true) as iterations, current_setting('ssl') as ssl
       from pg_roles where rolname = current_user`,
    )).rows[0];
    const notes = [caNote(url)].filter((n): n is string => !!n);
    const noted = () => (notes.length ? { note: notes.join("; ") } : {});
    // A server that does not check passwords for this client (trust) accepts a random one too.
    const acceptsAnyPassword = async () =>
      logsIn(loginUrlFor(url, me.db, (await resolveMonitoringPassword({ monitoringUser: DEFAULT_MONITORING_USER })).password)).catch(() => false);
    const unchecked = (what: string) => `this server accepts any password from this host, so ${what} was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password`;
    if (me.name === DEFAULT_MONITORING_USER) {
      const v = await verifyInitSetup({ client, database: me.db, monitoringUser: me.name, includeOptionalPermissions: false, provider: pgProvider });
      if (v.ok) {
        // A URL without a password (PGPASSWORD, say): the box gets the one this client logged in with.
        const used = (client as { password?: unknown }).password;
        const inUrl = urlPassword(new URL(url));
        const password = inUrl || (typeof used === "string" ? used : "");
        const putInUrl = `Put the password of ${DEFAULT_MONITORING_USER} in the URL: the monitoring box logs in with it`;
        if (!password) return { next: putInUrl };
        if (await acceptsAnyPassword()) {
          // What pg read from PGPASSWORD may be for another role or server: unchecked, it is not sent on.
          if (!inUrl) return { next: `This server accepts any password from this host, so the one from the environment was not checked and is not used. ${putInUrl}` };
          notes.push(unchecked("the password in the URL"));
        }
        return { monitoringUrl: monitoringUrlFor(url, me.db, password), ...noted() };
      }
    } else if (me.admin && pgProvider !== "supabase") {
      // The role is cluster-wide: another database here may use its password,
      // so it is never changed, and only a password that logs in is used.
      const exists = `${DEFAULT_MONITORING_USER} already exists on this server`;
      const setPassword = `Set PGAI_MON_PASSWORD to the password of ${DEFAULT_MONITORING_USER}, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as ${DEFAULT_MONITORING_USER} then needs the new password)`;
      // The platform fills in the password it keeps for the box (it never sends it here), over TLS only:
      // what the URL lacks for that, and whether the server takes TLS at all (it refused it, or has ssl off).
      const needs = storedPasswordNeeds(monitoringUrlFor(url, me.db, ""));
      const noTls = refusedTls || (!usedSsl && me.ssl === "off");
      const tlsOnly = `PostgresAI sends the password it keeps for ${DEFAULT_MONITORING_USER} only over TLS`;
      const others = opts.others?.length ? opts.others : undefined;
      // Nobody has the password (the first deploy generated it): only a new one, once nothing uses the old one.
      const disconnectOthers = others ? disconnectEach(others) : "pgai mon instances delete the server's other databases (pgai mon instances list)";
      const reconnectOthers = `(and deploy ${others ? others.join(", ") : "them"} again with it)`;
      const noTlsWayOut = opts.agent
        ? `Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password. If nobody has it, in a terminal: ${disconnectOthers}, then pgai mon deploy <admin-url> --reset-password with PGAI_MON_PASSWORD set to a new one ${reconnectOthers}`
        : `Set PGAI_MON_PASSWORD to its password, or turn on TLS on the server (ssl = on) and put sslmode=require in the URL. If nobody has the password: ${disconnectOthers}, then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one ${reconnectOthers}`;
      const reuse = me.mon_exists && !opts.resetPassword && !!opts.storedPassword && !(opts.agent ? "" : process.env.PGAI_MON_PASSWORD?.trim());
      if (reuse && needs) {
        if (noTls) return { next: `${exists}, but this server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}` };
        return { next: `${exists}. ${opts.agent ? "database_url" : "The URL"} needs ${needs}: the password PostgresAI keeps for this server is sent only over TLS, to a URL with each parameter once` };
      }
      if (reuse) {
        if (opts.check) return { checked: true };
        // The role is there: the plan grants this database and keeps its password (the one generated here is never set).
        const { password: unused } = await resolveMonitoringPassword({ monitoringUser: DEFAULT_MONITORING_USER });
        const plan = await buildInitPlan({ database: me.db, monitoringPassword: unused, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
        await applyInitPlan({ client, plan });
        return { monitoringUrl: monitoringUrlFor(url, me.db, ""), storedPassword: true as const, ...noted() };
      }
      if (me.mon_exists && opts.agent) return { next: `${exists}. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password` };
      // A PGAI_MON_PASSWORD that does not log in, where the platform keeps the password: without it, that one
      // is used (over TLS only).
      const unset = !opts.storedPassword ? setPassword
        : noTls ? `This server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}`
        : !needs ? `Unset PGAI_MON_PASSWORD: PostgresAI keeps the password of ${DEFAULT_MONITORING_USER} for this server`
        : `Unset PGAI_MON_PASSWORD, and the URL needs ${needs}: PostgresAI keeps the password of ${DEFAULT_MONITORING_USER} for this server, and sends it only over TLS, to a URL with each parameter once`;
      const { password, generated } = await resolveMonitoringPassword({ passwordEnv: opts.agent ? undefined : process.env.PGAI_MON_PASSWORD, monitoringUser: DEFAULT_MONITORING_USER });
      const reset = !!opts.resetPassword && me.mon_exists;
      if (me.mon_exists && !reset) {
        if (!process.env.PGAI_MON_PASSWORD?.trim()) return { next: `${exists}. ${setPassword}`, resettable: true };
        let accepted: boolean;
        try {
          accepted = await logsIn(loginUrlFor(url, me.db, password));
        } catch (err) {
          // pg_hba for this client, say: neither accepted nor rejected. With the platform's password, no login is needed.
          const orUnset = opts.storedPassword && !noTls ? ", or unset PGAI_MON_PASSWORD: PostgresAI keeps its password for this server" : "";
          return { next: `${exists}, and PGAI_MON_PASSWORD could not be checked from this host (${err instanceof Error ? err.message : String(err)}). Run pgai mon deploy from a host that ${DEFAULT_MONITORING_USER} may connect from${orUnset}` };
        }
        if (!accepted) return { next: `${exists} and PGAI_MON_PASSWORD is not its password. ${unset}` };
        if (await acceptsAnyPassword()) notes.push(unchecked("PGAI_MON_PASSWORD"));
      }
      if (opts.check) return { checked: true };
      const monitoringUrl = monitoringUrlFor(url, me.db, password);
      const loginUrl = loginUrlFor(url, me.db, password);
      const plan = await buildInitPlan({ database: me.db, monitoringPassword: password, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: !reset });
      try {
        await applyInitPlan({ client, plan });
      } catch (err) {
        // A later step failed after the role was created with a generated password (it logs in,
        // so the role is this run's): nobody has that password, so the role is dropped again.
        if (generated && !me.mon_exists && (await logsIn(loginUrl).catch(() => false))) await dropMonitoringRole(client);
        throw err;
      }
      // Another session may have created the role meanwhile, with its own password.
      // Other login errors (pg_hba for this client, say) do not tell, so they pass.
      if (!(await logsIn(loginUrl).catch(() => true))) return { next: `${DEFAULT_MONITORING_USER} was created by someone else meanwhile. ${setPassword}` };
      return { monitoringUrl, ...noted(), ...(generated && !me.mon_exists ? { generated: true as const } : {}) };
    }
    const plan = await buildInitPlan({ database: me.db, monitoringPassword: "<password>", includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
    return {
      sql: plan.steps.map((s) => `-- ${s.name}${s.optional ? " (optional: an error in this part can be ignored)" : ""}\n${redactPasswordsInSql(s.sql)}`).join("\n\n"),
      next: `Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai mon deploy again with the ${DEFAULT_MONITORING_USER} URL`,
    };
  } finally {
    await client.end();
  }
}

/**
 * The express checkup (pgai checkup's checks) over `url`: the warnings, then what passed; the plain
 * inventories by id. `save` stores the reports and returns the report id; its error is in the result.
 */
export async function expressCheckup(url: string, opts: PrepareOptions & { save?: (reports: Record<string, unknown>) => Promise<number> } = {}): Promise<CheckupResult> {
  const { client } = await openConnection(url, opts);
  let reports: Awaited<ReturnType<typeof generateAllReports>>;
  const failed: string[] = [];
  try {
    reports = await generateAllReports(client as Parameters<typeof generateAllReports>[0], "node-01", undefined, (f) => failed.push(f.checkId));
  } finally {
    await client.end();
  }
  const all = Object.values(reports).map((r) => ({ check_id: r.checkId, title: r.checkTitle, status: r.summary!.status, message: r.summary!.message }));
  const findings = all.filter((f) => f.status !== "info").sort((a, b) => Number(b.status === "warning") - Number(a.status === "warning"));
  const result = { checks: all.length + failed.length, findings, info: all.filter((f) => f.status === "info").map((f) => f.check_id), ...(failed.length ? { failed } : {}) };
  if (!opts.save) return result;
  try {
    return { ...result, report_id: await opts.save(reports) };
  } catch (err) {
    return { ...result, upload_error: errorText(err) };
  }
}

/**
 * Saves checkup reports as one report of `project`, the way the box's reporter does: created
 * pending, a file per check, then completed (failed when a file did not upload).
 */
export async function saveCheckupReport(rpc: <T>(fn: string, body: Record<string, unknown>) => Promise<T>, accessToken: string, project: string, reports: Record<string, unknown>): Promise<number> {
  const { reportId } = await createCheckupReport(rpc, { apiKey: accessToken, project });
  const setStatus = (status: string) => rpc("checkup_report_status_update", { access_token: accessToken, report_id: reportId, status });
  try {
    for (const [checkId, report] of Object.entries(reports)) {
      await uploadCheckupReportJson(rpc, {
        apiKey: accessToken, reportId, filename: `${checkId}.json`, checkId,
        jsonText: JSON.stringify(report, null, 2),
      });
    }
  } catch (err) {
    await setStatus("failed").catch(() => {});
    throw err;
  }
  await setStatus("completed");
  return reportId;
}

/** The ClickHouse Cloud organization that runs the service at `host`, found with the key itself. */
export async function clickhouseOrgFor(host: string, keyId: string, keySecret: string) {
  const apiUrl = process.env.CLICKHOUSE_API_URL || "https://api.clickhouse.cloud";
  const response = await fetch(`${new URL(apiUrl).origin}/v1/organizations`, {
    headers: { Authorization: `Basic ${Buffer.from(`${keyId}:${keySecret}`).toString("base64")}` },
    signal: requestTimeoutSignal().signal,
    redirect: "error",
  });
  if (response.status === 401) throw new ClickhouseKeyError("ClickHouse Cloud rejected the API key (401). Check the key id and secret.");
  if (!response.ok) throw new Error(`ClickHouse Cloud API request failed (${response.status}).`);
  const orgs = ((await response.json()) as { result?: { id: string }[] }).result;
  if (!Array.isArray(orgs)) throw new Error("ClickHouse Cloud API returned no organization list.");
  // An organization the key cannot read (403) says more than "no such service" in the next one.
  let notFound: unknown = new ClickhouseKeyError("The API key belongs to no ClickHouse Cloud organization.");
  let failed: unknown;
  for (const org of orgs) {
    try {
      const service = await findService({ apiUrl, orgId: org.id, keyId, keySecret, hostname: host });
      return { orgId: org.id, state: service.state };
    } catch (err) {
      if (/^No ClickHouse Managed Postgres service/.test(errorText(err))) notFound = new ClickhouseKeyError(errorText(err));
      // A key without the reader role (401/403 on the services) is the user's to fix, like a rejected one.
      else failed ??= /\((401|403)\)/.test(errorText(err)) ? new ClickhouseKeyError(errorText(err)) : err;
    }
  }
  throw failed ?? notFound;
}

/** What a delete did to the billing (monitoring_instance_delete's reply), for a person; undefined when nothing was released. */
export function disconnectBilling(reply: unknown): string | undefined {
  const r = (reply ?? {}) as { billing?: { subscription?: string; quantity?: number; refunded?: number; currency?: string; credited?: boolean }; billing_warning?: string };
  // The last cluster: the unused time is refunded to the card; when that refund fails it stays as account credit.
  if (r.billing?.subscription === "canceled" && r.billing.credited && r.billing_warning) return `subscription canceled: no further charges; the refund failed (${r.billing_warning})`;
  if (r.billing_warning) return `not released (${r.billing_warning}): contact support`;
  if (r.billing?.subscription === "canceled" && typeof r.billing.refunded === "number" && r.billing.refunded > 0) {
    return `subscription canceled: no further charges; refunded ${money(r.billing.refunded, r.billing.currency ?? "usd")} to your card`;
  }
  if (r.billing?.subscription === "canceled") return "subscription canceled: no further charges; the unused part of this period is credited (prorated)";
  if (r.billing?.subscription === "active" && typeof r.billing.quantity === "number") {
    return `${r.billing.quantity} ${r.billing.quantity === 1 ? "database cluster" : "database clusters"} left on the subscription`;
  }
  return undefined;
}

/** The project's Health Matrix: <console>/<org alias>/projects/<project alias>/health. */
export function healthMatrixUrl(uiBaseUrl: string, orgAlias: string, projectAlias: string): string {
  return `${uiBaseUrl.replace(/\/+$/, "")}/${encodeURIComponent(orgAlias)}/projects/${encodeURIComponent(projectAlias)}/health`;
}

/** The box's Grafana through its PostgresAI sign-in (generic OAuth), landing where the address points. */
export function grafanaOAuthUrl(dashboardUrl: string | null): string | null {
  let u: URL;
  try {
    u = new URL(dashboardUrl ?? "");
  } catch {
    return dashboardUrl;
  }
  if (u.pathname === "/login/generic_oauth") return dashboardUrl;
  return `${u.origin}/login/generic_oauth?redirectTo=${encodeURIComponent(`${u.pathname}${u.search}`)}`;
}

/** The platform side of mon deploy, for the CLI and the MCP server alike. */
export function platformDeps(p: { apiKey: string; apiBaseUrl: string; uiBaseUrl: string; orgScope?: OrgScope; debug?: boolean; agent?: boolean }) {
  const rpc = <T>(fn: string, body: Record<string, unknown> = {}) =>
    callRpc<T>({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn, body, operation: fn.replace(/_/g, " "), debug: p.debug, orgScope: p.orgScope });
  // The console links of a row: the org alias once, the projects again only while one is not found.
  let orgAlias: Promise<string | undefined> | undefined;
  let projects: Promise<ProjectListItem[]> | undefined;
  const healthUrl = async (row: Database): Promise<string | undefined> => {
    orgAlias ??= p.orgScope?.alias ? Promise.resolve(p.orgScope.alias) : listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl })
      .then((orgs) => (orgs.length === 1 ? orgs[0] : orgs.find((o) => o.org_id === p.orgScope?.id))?.alias);
    projects ??= rpc<ProjectListItem[]>("projects_list");
    const [org, list] = await Promise.all([orgAlias, projects]);
    const project = list.find((x) => x.monitoring_instance_ids?.includes(row.id)) ?? list.find((x) => x.name === row.name);
    if (!project) projects = undefined;
    return org && project?.alias ? healthMatrixUrl(p.uiBaseUrl, org, project.alias) : undefined;
  };
  const withLinks = async (row: Database): Promise<Database> => {
    // A link that cannot be looked up is left out: it never fails the list.
    const health_url = await healthUrl(row).catch(() => {
      orgAlias = projects = undefined;
      return undefined;
    });
    // Ordered for a person reading it: the Health Matrix, then Grafana.
    const { dashboard_url, host_metrics, ...rest } = row;
    return { ...rest, ...(health_url ? { health_url } : {}), dashboard_url: grafanaOAuthUrl(dashboard_url), host_metrics } as Database;
  };
  return {
    verifyTls: verifyCollectorTls,
    list: () => rpc<Database[]>("cloud_monitoring_list"),
    links: withLinks,
    create: (body: Record<string, string | boolean | number>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_connect", body),
    disconnect: (id: string) => rpc("cloud_monitoring_disconnect", { instance_id: id }),
    resetLock: (server: string) => rpc<{ lock_id: string }>("cloud_monitoring_reset_lock", { server }),
    resetUnlock: (lockId: string) => rpc("cloud_monitoring_reset_unlock", { lock_id: lockId }),
    prepare: (url: string, provider: Provider, o: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[] } = {}) => prepareDatabase(url, provider, { ...o, agent: p.agent }),
    unprepare: (url: string) => unprepareDatabase(url, { agent: p.agent }),
    clickhouseOrg: clickhouseOrgFor,
    checkup: (url: string, project: string) =>
      expressCheckup(url, { agent: p.agent, save: (reports) => saveCheckupReport(rpc, p.apiKey, project, reports) }),
    quote: (coupon?: string, cluster?: string) => rpc<Quote>("cloud_monitoring_quote", { ...(coupon ? { promo_code: coupon } : {}), ...(cluster ? { db_server: cluster } : {}) }),
    billingUrl: (orgAlias: string) => `${p.uiBaseUrl}/${orgAlias}/billing`,
    handoffUrl: async (provider: "rds" | "supabase") => {
      const orgs = await listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl });
      const org = orgs.length === 1 ? orgs[0] : orgs.find((o) => o.alias === p.orgScope?.alias || o.org_id === p.orgScope?.id);
      return `${p.uiBaseUrl}/${org?.alias ?? "<org>"}/monitoring/scale/create/${provider}`;
    },
    sleep: (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
    now: () => Date.now(),
  };
}
