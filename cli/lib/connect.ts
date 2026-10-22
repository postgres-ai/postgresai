import { Client } from "pg";
import { parse as parseConnString, parseIntoClientConfig } from "pg-connection-string";
import { findService } from "./clickhouse";
import {
  applyInitPlan, buildInitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER,
  maskConnectionString, quoteIdent, quoteLiteral, redactPasswordsInSql, resolveAdminConnection, resolveMonitoringPassword, verifyInitSetup,
} from "./init";
import { assertEncodedUserinfo, estimateLine, monitoringStatus, noCardNext, notAdminNext, type DeployStatus } from "./deploy-surface";
import { collectorConnection, splitChannelBinding, verifyCollectorTls } from "./instances";
import { callRpc, type ProjectListItem } from "./joe";
import { listOrgs, type OrgScope } from "./org-scope";
import { SUPABASE_API_BASE, extractProjectRefFromUrl } from "./supabase";
import { HttpStatusError, requestTimeoutSignal, stripControlsToSpace } from "./util";

// `pgai mon deploy <database-url>`: put a database under PostgresAI's care
// (postgres-ai/internal#354). Each step is skipped when already done, so a
// re-run is safe; every outcome carries the exact next action.

export type Provider = "clickhouse" | "rds" | "supabase" | "self-managed";
/** One vocabulary with pgai dblab deploy (lib/deploy-surface). */
export type Status = DeployStatus;

/** A step of mon deploy, once each: a line for a person, a JSON event for an agent. */
export interface ProgressEvent {
  event: "billing" | "preparing" | "provisioning" | "box";
  /** Where the box runs (the "Monitoring box: <cloud>, <location>" line, platform-all#972). */
  cloud?: string;
  location?: string;
  location_reason?: string;
  /** Seconds since mon deploy started. */
  elapsed_s: number;
  message: string;
  /** event box: the box's state (launch_requested, registered, active, ...). */
  state?: string;
  /** event box: the instance, for following it later (pgai mon instances watch <id>). */
  id?: string;
  /** Supabase: the page where the org approves PostgresAI's access (deploy waits for it). */
  auth_url?: string;
  /** Supabase without a URL: the project picked, as the step view names it. */
  project?: string;
}

export interface ConnectResult {
  status: Status;
  provider: Provider;
  name: string;
  id?: string;
  /** Where a new box runs: aws | hetzner, its region, and why (platform-all#972). */
  cloud?: string;
  location?: string;
  location_reason?: string;
  /** Why there, in words ("same region as your database"). */
  location_reason_text?: string;
  /** Not deployed for want of --yes: where else the box can run, each available cloud's locations. */
  choices?: Record<string, string[]>;
  /** The project's Health Matrix in the console. */
  health_url?: string;
  dashboard_url?: string | null;
  host_metrics?: boolean;
  first_checkup_eta?: string;
  /** A new box: what it costs ("$512.00/month per database cluster (scale plan)", "free (1 of 2 free slots)", "included: same database cluster as ..."). */
  price?: string;
  requires_payment_method?: boolean;
  coupon?: { code: string; valid: boolean; description?: string; error?: string };
  next: string;
  sql?: string;
  /** --provider supabase without a URL: the Supabase organization connected to the org. */
  supabase_org?: string;
  /** --provider supabase without a URL: the projects to pick from (postgresai#438). */
  projects?: SupabaseProject[];
}

/** A project of the Supabase organization connected to the org (v1.cloud_monitoring_supabase_projects). */
export interface SupabaseProject {
  ref: string;
  name: string;
  region: string | null;
  /** active, inactive (paused), unhealthy, coming_up, unknown, ... */
  status: string;
  postgres_version: string | null;
  /** Active and not monitored yet. */
  selectable: boolean;
  /** Why it can't be picked: monitored, paused or not_ready:<status> (null when selectable). */
  reason: string | null;
  /** The monitoring box it already has. */
  monitored: { id: string; name: string | null; status: string | null } | null;
}

/** v1.cloud_monitoring_supabase_projects: the org's Supabase projects, or the consent page when it is not connected. */
export type SupabaseProjects =
  | { connected: true; supabase_org_name: string; synced: boolean; projects: SupabaseProject[] }
  | { connected: false; auth_url: string; session_id: string; expires_at?: string };

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
  /** The database server's vCPUs; AAS is collected only with them (null: not set). */
  vcpus?: number | null;
  /** Where a box in PostgresAI cloud runs: aws | hetzner and its region (null: self-hosted). */
  cloud?: string | null;
  location?: string | null;
}

/**
 * The monitoring URL (with a note for the user, if any), or what to do first.
 * `generated`: this run created the role with a password generated here, which
 * nobody has once the run ends.
 */
export type Prepared = { monitoringUrl: string; note?: string; generated?: true; storedPassword?: true } | { next: string; sql?: string; resettable?: true } | { checked: true };

/** Gets a session's TLS warning (connectWithSslFallback). */
type Warn = (message: string) => void;

export interface ConnectDeps {
  verifyTls?(url: string): Promise<void>;
  list(): Promise<Database[]>;
  /** A row with its console links, for the result shown: the Health Matrix, and Grafana through the PostgresAI sign-in. */
  links?(row: Database): Promise<Database>;
  create(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; status: string; error?: string }>;
  /** `check`: only whether the URL can work, nothing changed ({ checked: true } or what to do first). */
  prepare(url: string, provider: Provider, opts?: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[]; otherRefs?: string[]; monitoringUser?: string; warn?: Warn }): Promise<Prepared>;
  /** Drops the role a `generated` prepare created; false when it could not. */
  unprepare(url: string, monitoringUser?: string, opts?: { warn?: Warn }): Promise<boolean>;
  localStackRunning(): boolean;
  clickhouseOrg(host: string, keyId: string, keySecret: string): Promise<{ orgId: string; state: string }>;
  selfHosted(monitoringUrl: string, env: Record<string, string>): Promise<void>;
  handoffUrl(provider: "rds"): Promise<string>;
  /**
   * Supabase (v1.cloud_monitoring_supabase_*): the org's connection, or the consent page that makes one;
   * whether that consent came back; the box for a project of the connected org.
   */
  supabaseStart(dbUrl: string): Promise<{ connected: true; project_ref: string } | { auth_url: string; session_id: string; expires_at?: string }>;
  supabaseStatus(sessionId: string): Promise<{ status: "pending" | "connected" | "error"; error?: string }>;
  supabaseConnect(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; status: string; error?: string }>;
  /** --provider supabase without a URL: the org's projects (synced first), or the consent page (postgresai#438). */
  supabaseProjects?(): Promise<SupabaseProjects>;
  /** Asks a person at a terminal to pick a project (its ref); undefined when skipped or nobody can be asked. */
  chooseSupabaseProject?(projects: SupabaseProject[], org: string): Promise<string | undefined>;
  /** Opens a page in this machine's browser (a person at a terminal only). */
  openUrl?(url: string): void;
  /**
   * v1.monitoring_deploy_options (platform-all#972): where the box would run for this database
   * and --cloud/--location, decided by the platform (the Console asks the same).
   */
  deployOptions?(args: { db_host?: string; supabase_project_ids?: string[]; cloud?: string; location?: string; check_stock?: boolean }): Promise<DeployOptions>;
  /** What the next box costs, with the coupon checked. */
  /** `cluster`: lower(host):port of the URL, which the price is per. */
  quote(coupon?: string, cluster?: string): Promise<Quote>;
  /** The console page where the org adds a payment method, when the quote names none. */
  billingUrl(orgAlias: string): string;
  /** Asks a person; false when nobody can be asked. */
  confirm(question: string): Promise<boolean>;
  /**
   * A person at a terminal, without --yes, --cloud or --location: deploy where the platform put the box
   * (true), not at all (false), or at the cloud and location they picked from `options`. Absent when
   * nobody can be asked: the result then carries the choices.
   */
  choosePlace?(options: DeployOptions, name: string): Promise<boolean | { cloud: string; location: string }>;
  /** Asks a person for the database server's vCPUs; undefined when skipped or nobody can be asked. */
  askVcpus?(question: string): Promise<number | undefined>;
  /** Sets an instance's vCPUs (v1.cloud_monitoring_set_vcpus, #904). */
  setVcpus?(instanceId: string, vcpus: number): Promise<unknown>;
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
  /** --monitoring-user: the monitoring role (default postgres_ai_mon). */
  monitoringUser?: string;
  waitMs: number;
  /** The URL comes from an agent (the MCP tool): nothing of this process's environment or files goes to the host it names. */
  agent?: boolean;
  /** A billed box is accepted without asking. */
  yes?: boolean;
  /** A Stripe promotion code for the org's subscription. */
  coupon?: string;
  /** --name: the instance's name (default: host[:port]/database). */
  name?: string;
  /** --cloud: aws or hetzner (default: the platform's choice for the database, platform-all#972). */
  cloud?: string;
  /** --location: a region of the box's cloud (default: the platform's choice). */
  location?: string;
  /** --vcpus: the database server's vCPUs; AAS is collected only with them. */
  vcpus?: number;
  /** --supabase-project: the Supabase project (ref or name) to monitor, instead of its URL (postgresai#438). */
  supabaseProject?: string;
}

/**
 * An error's HTTP status, read by shape: the bundled CLI holds two copies of
 * HttpStatusError, so `instanceof` can miss the one a call threw.
 */
export function httpStatus(err: unknown): number | undefined {
  const status = err instanceof Error ? (err as { status?: unknown }).status : undefined;
  return typeof status === "number" ? status : undefined;
}

/** Asked at a terminal when --vcpus is missing: AAS (database load) is collected only with it. */
export const VCPUS_QUESTION = "How many vCPUs does the database server have? If you skip this (Enter), you get no AAS (database load) data for this database. vCPUs: ";
/** Said when a box is requested without vCPUs. */
export const NO_VCPUS_NOTE = "Note: no vCPUs given, so you get no AAS (database load) data for this database. Add them later: pgai mon instances set <id-or-name> --vcpus <n>.";

const money = (cents: number, currency: string) =>
  currency.toLowerCase() === "usd" ? `$${(cents / 100).toFixed(2)}` : `${(cents / 100).toFixed(2)} ${currency.toUpperCase()}`;

/** v1.monitoring_deploy_options: the clouds, their locations, and the box the deploy would get. */
export interface DeployOptions {
  clouds: { code: string; label: string; offered?: boolean; available: boolean }[];
  locations: Record<string, { code: string; label: string; world_part: string }[]>;
  choice: { cloud: string; cloud_label: string; location: string; location_label: string | null; reason: string; reason_text: string } | null;
  error: { code: string; message: string; detail: string | null; hint: string | null } | null;
}

/** Where a new box runs, as the result and the progress event carry it. */
type Place = { cloud: string; location: string; location_reason: string; location_reason_text: string };

/** placeBox's answer: the place, and the platform's options it came from (the choices). */
type Placed = { place: Place; options: DeployOptions };

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

// The earliest first checkup: the reporter's first run starts 3 minutes after
// the stack starts (REPORTER_INITIAL_DELAY_SECONDS in
// config/scripts/postgres-reports.sh; a test keeps the two equal), then waits
// until every target has data, so a slow first collection makes it later.
export const FIRST_CHECKUP_DELAY_MS = 3 * 60_000;
const POLL_MS = 15_000;
export const PROVIDERS: Provider[] = ["clickhouse", "rds", "supabase", "self-managed"];

/**
 * --provider as given, trimmed and case-insensitive (as pgai dblab deploy reads it).
 * Generic names the direct PostgreSQL setup path, not who operates the database:
 * it keeps the established value, self-managed, in results and preparation.
 */
export const providerArg = (provider: string | undefined): string | undefined => {
  const p = provider?.trim().toLowerCase();
  return p === "generic" ? "self-managed" : p;
};

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

/** What `pgai mon instances delete` takes for `d`: its name, or its id when another live box has that name. */
const deleteRef = (d: Database, list: Database[]) =>
  (list.some((o) => o.id !== d.id && !disconnecting(o.status) && o.name === d.name) ? d.id : d.name);

/** `pgai mon instances delete` for each database named. */
const disconnectEach = (names: string[]) => names.map((n) => `pgai mon instances delete ${shellWord(n)} --yes`).join(" and ");

/**
 * The monitoring role's login through `url`: a URL user `<session user>.<suffix>` (the
 * Supabase pooler's project ref, which the pooler routes on) keeps its suffix.
 */
function loginName(url: string, sessionUser: string, role: string): string {
  const user = decodeURIComponent(new URL(url).username);
  return user.startsWith(`${sessionUser}.`) ? role + user.slice(sessionUser.length) : role;
}

/** The monitoring role's URL (`user` logs in as it) for the prepared database, with the query parameters in `kept`. */
function roleUrlFor(url: string, user: string, db: string, password: string, kept: string[]): string {
  const u = new URL(url);
  u.username = encodeURIComponent(user);
  u.password = encodeURIComponent(password);
  u.pathname = `/${encodeURIComponent(db)}`;
  for (const name of [...u.searchParams.keys()]) if (!kept.includes(name)) u.searchParams.delete(name);
  // `mon targets add` refuses a raw '@' after the host (a delete above already encoded it).
  u.search = u.search.replace(/@/g, "%40");
  return u.toString();
}

/** The URL a box uses: only postgres_ai_mon's credentials, and the prepared database by name. */
const monitoringUrlFor = (url: string, user: string, db: string, password: string) => roleUrlFor(url, user, db, password, URL_PARAMS_KEPT.filter((p) => p !== "channel_binding"));

/** The same login from this machine: with the given URL's TLS files (a private CA, say). */
const loginUrlFor = (url: string, user: string, db: string, password: string) => roleUrlFor(url, user, db, password, [...URL_PARAMS_KEPT, ...URL_PARAMS_TLS]);

/** What to tell the user when the URL verifies the server with a CA file the box will not have. */
export function caNote(url: string): string | undefined {
  const q = new URL(url).searchParams;
  if (!q.get("sslrootcert") || !/^verify-/.test(q.get("sslmode") ?? "")) return undefined;
  return `the monitoring box has no copy of the CA in sslrootcert: with sslmode=${q.get("sslmode")} it connects only to a server certificate signed by a public CA (else use sslmode=require)`;
}

/** Why the box is removed for its first charge, and the fix: a promo code Stripe refused needs another code, not another card. */
function chargeFailed(error: string, rerun: string, billingUrl?: string, couponFlag = "--coupon"): string {
  const fix = /promotion code|coupon/i.test(error)
    ? `The promo code cannot be used by this organization: ${rerun} without ${couponFlag}, or with a code this organization can use`
    : `Update the payment method${billingUrl ? ` at ${billingUrl}` : ""}, then ${rerun}`;
  return `The first charge failed (${error}): the box is being removed and nothing is billed. ${fix}`;
}

/** A platform row as a result with its next action; `fresh` (provisioned just now) adds the first-checkup ETA. */
export function connectStatus(row: Database, provider = row.provider as Provider, fresh = false): ConnectResult {
  const base = { provider, name: row.name, id: row.id, ...(row.health_url ? { health_url: row.health_url } : {}), dashboard_url: row.dashboard_url, host_metrics: row.host_metrics };
  const status = stateOf(row.status);
  // Removed because its first charge failed: the user's to fix, whichever command reads it (deploy, watch, list).
  if (row.billing_error && (status === "deleting" || status === "deleted")) {
    return { ...base, status: "action_required",
      next: `${chargeFailed(row.billing_error, "re-run pgai mon deploy")}.` };
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
// ESC or C1 sequences: CSI (colors, cursor, clear), OSC (titles, links) to its end, and two-byte ones (a reset).
const ESCAPES = /(?:\x1b\[|\x9b)[0-?]*[ -\/]*[@-~]|(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?|\x1b[ -~]?/g;

/** Text from the platform, for a terminal: escape sequences dropped, other control characters a space; line breaks kept. */
export const terminalText = (text: string): string => text.replace(ESCAPES, "").split("\n").map(stripControlsToSpace).join("\n");

export function stepProgress(o: { json: boolean; tty: boolean; makeView: () => StepViewLike; writeLine: (s: string) => void;
  writeEvent: (e: ProgressEvent) => void; onId?: (id: string) => void }) {
  let view: StepViewLike | undefined;
  let priced = false;
  const progress = (e: ProgressEvent): void => {
    if (e.id) o.onId?.(e.id);
    if (o.json) return o.writeEvent(e);
    e = { ...e, message: terminalText(e.message), ...(e.state ? { state: terminalText(e.state) } : {}) };
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
    if (/^(Note|Warning):/.test(e.message)) view.log(e.message);
    if (e.event === "box" && e.state) view.setNote(`box: ${boxWords(e.state)}`);
  };
  return { progress, view: () => view };
}

/** A result for a person at a terminal: a few sentences, links whole (JSON keeps every field). */
export function resultLines(r: ConnectResult): string[] {
  return personLines(r).map(terminalText);
}

/** A box's cloud as users read it (the platform's labels). */
const cloudWord = (cloud: string): string => ({ aws: "AWS", hetzner: "Hetzner" } as Record<string, string>)[cloud] ?? cloud;

function personLines(r: ConnectResult): string[] {
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
        ...(r.cloud ? [`  Runs on:       ${cloudWord(r.cloud)}, ${r.location}`] : []),
        ...(!r.health_url && !r.dashboard_url && r.next && r.next !== "none" ? [`Next: ${r.next}`] : []),
        ...(extra && (r.health_url || r.dashboard_url) ? [`Note: ${extra}`] : []),
      ];
    case "in_progress":
      return [`Monitoring for ${who} is still being set up. Follow it: ${r.next}`];
    case "deleting":
    case "deleted":
      return [`Deleting the monitoring of ${who}.${billing ? ` Billing: ${billing}.` : ""}${r.next && r.next !== "none" ? ` ${r.next}` : ""}`];
    default:
      // Supabase without a URL: the projects one per line under the reason (JSON has them in `projects`).
      if (r.projects?.length) return [r.next, ...supabaseProjectRows(r.projects)];
      // Not deployed for want of --yes: where else it can run, a line per cloud (JSON has them in `choices`).
      return r.choices ? [r.next, ...Object.entries(r.choices).map(([cloud, codes]) => `  --cloud ${cloud} --location ${codes.join("|")}`)] : [r.next];
  }
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
async function billingFor(q: Quote, name: string, opts: ConnectOptions, deps: ConnectDeps, show: (price: string) => void,
  ask?: (price: string) => Promise<Decision>): Promise<{ billing: Billing; billingPage: Billing & { next: string }; billingUrl: string; accepted: boolean; stop?: Partial<ConnectResult> & { next: string }; placed?: Placed }> {
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
  // The platform placed the box: one question for any box, free or billed (A17).
  if (ask) {
    const d = await ask(billing.price!);
    if (!d.go) return { billing, billingPage, billingUrl, accepted: false, stop: { ...billing, ...d.stop } };
    return { billing, billingPage, billingUrl, accepted: q.billed, placed: d.place };
  }
  // A platform from before --cloud: a billed box only, as before.
  if (q.billed && !opts.yes && !(await deps.confirm(`Provision ${terminalText(name)}? (y/N): `))) {
    return stop(opts.agent ? `Call deploy_monitoring again with yes: true to accept ${billing.price}` : `Re-run with --yes to accept ${billing.price}`);
  }
  return { billing, billingPage, billingUrl, accepted: q.billed };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
  const started = deps.now();
  let step: ProgressEvent["event"] = "preparing";
  const progress = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}) => {
    if (event !== "billing") step = event;
    deps.progress({ event, elapsed_s: Math.round((deps.now() - started) / 1000), message, ...more });
  };
  // Each session warns about its TLS: said once a run, as progress (a raw stderr line breaks the step view).
  const warned = new Set<string>();
  const warn = (message: string) => {
    if (warned.has(message)) return;
    warned.add(message);
    progress(step, message);
  };
  // --supabase-project names a Supabase project: no URL is needed (postgresai#438).
  const project = opts.supabaseProject?.trim();
  const provider = (providerArg(opts.provider) ?? (!url && project ? "supabase" : detectCloudProvider(url))) as Provider;
  if (!PROVIDERS.includes(provider)) throw new Error(`--provider must be one of: generic, ${PROVIDERS.join(", ")}`);
  if (project !== undefined && provider !== "supabase") {
    return { status: "action_required", provider, name: url ? databaseName(url) : "", next: "--supabase-project applies to Supabase only: leave it out, or use --provider supabase" };
  }
  if (!url) {
    if (provider !== "supabase") return { status: "action_required", provider, name: "", next: "Pass the database URL: pgai mon deploy postgresql://user:password@host:5432/dbname (Supabase: --provider supabase)" };
    if (opts.selfHosted) return { status: "action_required", provider, name: "", next: "--self-hosted needs the database URL: pass the project's connection string" };
    return deploySupabaseProject(undefined, opts, deps, progress);
  }
  // Before any name is taken from the URL (the MCP path does not go through the CLI's argument check).
  assertEncodedUserinfo(url);
  // Supabase is never logged in to from here: its URL only names the project.
  if (!opts.selfHosted && provider === "supabase") return deploySupabase(url, opts, deps, progress);
  checkUrlParams(url, opts.agent);
  const name = databaseName(url);
  // The platform keeps postgres_ai_mon's password only; --reset-password locks the server for any role.
  const asked = opts.monitoringUser?.trim();
  if (opts.monitoringUser !== undefined && !asked) {
    const flag = opts.agent ? "monitoring_user" : "--monitoring-user";
    return { status: "action_required", provider, name, next: `${flag} is empty: pass a role name, or leave it out for ${DEFAULT_MONITORING_USER}` };
  }
  const monitoringUser = asked && asked !== DEFAULT_MONITORING_USER ? asked : undefined;
  const role = monitoringUser ?? DEFAULT_MONITORING_USER;
  // The setup SQL puts the role in quoted literals, dollar-quoted DO bodies and format() strings.
  if (monitoringUser && /['$%\\\u0000-\u001f\u007f]/.test(monitoringUser)) {
    return { status: "action_required", provider, name, next: `${opts.agent ? "monitoring_user" : "--monitoring-user"} must not contain ', $, %, backslash or control characters` };
  }
  if (monitoringUser && Buffer.byteLength(monitoringUser) > 63) {
    return { status: "action_required", provider, name, next: `${opts.agent ? "monitoring_user" : "--monitoring-user"} must be at most 63 bytes (Postgres truncates longer role names)` };
  }
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

  if (!opts.selfHosted && provider === "rds") {
    const ignored = monitoringUser ? ` (${opts.agent ? "monitoring_user" : "--monitoring-user"} is not used there)` : "";
    return { status: "action_required", provider, name, next: `Finish in the console: ${await deps.handoffUrl(provider)}${ignored}` };
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
  let billing: Billing = {};
  let billingPage: Billing & { next: string } | undefined;
  let accepted = false;
  let billingUrl: string | undefined;
  let place: Place | undefined;
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
  // Its vCPUs (#904): missing ones are set (--vcpus, or asked at a terminal); a set
  // value is never changed here. ClickHouse's come from ClickHouse Cloud.
  if (row && !disconnecting(row.status) && provider !== "clickhouse") {
    const has = (row.vcpus ?? 0) > 0;
    let want = opts.vcpus;
    if (want === undefined && !has && !opts.yes && !opts.agent && deps.askVcpus) want = await deps.askVcpus(VCPUS_QUESTION);
    if (want !== undefined && !has && deps.setVcpus) {
      try {
        await deps.setVcpus(row.id, want);
        note += `; vCPUs set to ${want}: AAS (database load) is collected from now on`;
      } catch (err) {
        if (httpStatus(err) !== 403) throw err;
        note += `; vCPUs not set: only an organization admin can (pgai mon instances set ${row.id} --vcpus ${want})`;
      }
    } else if (want !== undefined && has && row.vcpus !== want) {
      note += `; --vcpus ignored: ${row.name} already has ${row.vcpus} vCPUs (to change them: pgai mon instances set ${row.id} --vcpus ${want})`;
    } else if (!has) {
      note += `; no vCPUs, so no AAS (database load) data for this database (add them: pgai mon instances set ${row.id} --vcpus <n>)`;
    }
  }
  if (row && opts.resetPassword) {
    return { status: "action_required", provider, name, id: row.id, next: `${name} is already monitored, and its monitoring uses the current password: ${disconnectEach([deleteRef(row, rows)])} first` };
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
    const undo = async (p: Prepared) => { if ("generated" in p && p.generated) await deps.unprepare(url, monitoringUser, { warn }).catch(() => false); };
    let lockId: string | undefined;
    // The role is one for the whole server: a new password cuts off what uses the old one.
    // Without a cluster, the server is not known.
    const unread = (d: Database) => !d.cluster;
    const onServer = (d: Database) => !!d.cluster && serverOf(d.cluster) === cluster;
    // What may log in as the role: not a box being destroyed or gone, a deleted self-hosted one
    // (deactivated: inactive), or a launch that failed before its box registered (it never connected).
    // Only the raw status failed is a launch outcome: error is a running box whose health check failed.
    const usesRole = (d: Database) =>
      !["deleting", "deleted", "inactive"].includes(stateOf(d.status)) && !(d.status === "failed" && !d.registered_at);
    const othersOnServer = (list: Database[]) => list.filter((d) => usesRole(d) && (unread(d) || onServer(d)));
    // The platform keeps postgres_ai_mon's password for a server it monitors a database on, for this org,
    // and fills it in while a box there is not being destroyed (a delete that failed counts as one).
    const storedPassword = !monitoringUser && rows.some((d) => d.monitoring_password_stored && !/delet/.test(d.status ?? "") && onServer(d)) || undefined;
    const cutOff = (others: Database[], list: Database[]): ConnectResult => {
      const label = (d: Database) => (deleteRef(d, list) === d.id ? `${d.name} (id ${d.id})` : d.name);
      const on = others.filter((d) => !unread(d)).map(label).join(", ");
      const maybe = others.filter(unread).map(label).join(", ");
      const what = [on && `would cut off the monitoring of ${on} on this server`, maybe && `may cut off the monitoring of ${maybe} (the platform could not read its URL, so its server is not known)`].filter(Boolean).join(", and ");
      const instead = [
        ...(storedPassword ? ["deploy without --reset-password (PostgresAI keeps its password for this server, and sends it only over TLS)"] : []),
        "set PGAI_MON_PASSWORD to its password instead",
        `${disconnectEach(others.map((d) => deleteRef(d, list)))} first`,
      ];
      return { status: "action_required", provider, name, next: `A new password for ${role} ${what}: ${instead.join(", or ")}` };
    };
    const onThisServer = othersOnServer(rows).filter((d) => !unread(d));
    const others = onThisServer.map((d) => d.name);
    const otherRefs = onThisServer.map((d) => deleteRef(d, rows));
    if (opts.resetPassword && othersOnServer(rows).length) return cutOff(othersOnServer(rows), rows);
    // What the box costs, before anything is touched: a billed box is accepted
    // (--yes, or at the prompt), and needs the org's payment method.
    if (opts.coupon !== undefined && !opts.coupon.trim()) {
      return { status: "action_required", provider, name, next: `${opts.agent ? "coupon" : "--coupon"} is empty: pass a promotion code, or leave ${opts.agent ? "coupon" : "--coupon"} out` };
    }
    const prepOpts = { resetPassword: opts.resetPassword, warn, ...(storedPassword ? { storedPassword, others, otherRefs } : {}), ...(monitoringUser ? { monitoringUser } : {}) };
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
      // Where the box runs, before the price and before the role is created: a
      // cloud or location the platform refuses changes nothing (platform-all#972).
      const where = { db_host: new URL(url).hostname };
      const placed = await placeBox(deps, opts, where, progress);
      place = placed?.place;
      let quote: Quote;
      try {
        quote = await deps.quote(opts.coupon, clusterOf(url));
      } catch (err) {
        // Not an org admin: the user's to act on, as with pgai dblab deploy (postgresai#412).
        if (err instanceof HttpStatusError && err.status === 403) return { status: "action_required", provider, name, next: notAdminNext };
        throw err;
      }
      const b = await billingFor(quote, name, opts, deps, (price) => progress("billing", estimateLine(price)),
        placed && ((price) => decide(placed, name, price, opts, deps, where, progress)));
      // Where the box would run goes with every result from here on.
      if (b.stop) return { status: "action_required", provider, name, ...place, ...b.stop };
      if (b.placed) place = b.placed.place;
      if (deps.aborted?.()) throw new CancelledError();
      ({ billing, billingPage, billingUrl, accepted } = b);
      // ClickHouse's vCPUs come from ClickHouse Cloud; elsewhere only the user knows them.
      if (vcpus === undefined && provider !== "clickhouse") {
        vcpus = !opts.yes && !opts.agent && deps.askVcpus ? await deps.askVcpus(VCPUS_QUESTION) : undefined;
        if (deps.aborted?.()) throw new CancelledError();
        if (vcpus === undefined) progress("preparing", NO_VCPUS_NOTE);
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
          return { status: "action_required", provider, name, ...billing, next: `This platform cannot lock the server for --reset-password yet: set PGAI_MON_PASSWORD to ${role}'s password instead` };
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
        if (same) return { status: "action_required", provider, name, id: same.id, ...billing, next: `${name} is already monitored, and its monitoring uses the current password: ${disconnectEach([deleteRef(same, now)])} first` };
        const others = othersOnServer(now);
        if (others.length) return { ...cutOff(others, now), ...billing };
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
        // pgai mon deploy --name (postgresai#412); where the box runs, as the platform answered above.
        ...(opts.name ? { project_name: opts.name } : {}),
        ...(place ? { cloud: place.cloud, server_location: place.location }
          : opts.location ? { server_location: opts.location } : {}),
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
  }

  return follow(row, { provider, name, fresh, billing, billingUrl, undoRole, note, place }, opts, deps, progress);
}

/**
 * Where a new box runs: the platform's answer for this database and --cloud/--location
 * (v1.monitoring_deploy_options, platform-all#972), shown as "Monitoring box: AWS, us-east-2
 * (same region as your database)". A refusal is thrown (exit 1) before anything is created.
 * A platform from before --cloud has no answer: the box goes where it puts it.
 */
async function placeBox(deps: ConnectDeps, opts: ConnectOptions, where: { db_host?: string; supabase_project_ids?: string[] },
  progress: Progress): Promise<Placed | undefined> {
  if (!deps.deployOptions) return undefined;
  const cloud = opts.cloud?.trim().toLowerCase() || undefined;
  const location = opts.location?.trim() || undefined;
  const flag = (f: string) => (opts.agent ? f : `--${f}`);
  let options: DeployOptions;
  try {
    options = await deps.deployOptions({ ...where, ...(cloud ? { cloud } : {}), ...(location ? { location } : {}), check_stock: true });
  } catch (err) {
    if (httpStatus(err) !== 404) throw err;
    if (cloud) throw new Error(`This PostgresAI platform does not take ${flag("cloud")} yet: re-run without it. Nothing was created.`);
    // A database URL still sends --location as server_location; a Supabase box would drop it.
    if (location && where.supabase_project_ids) {
      throw new Error(`This PostgresAI platform does not take ${flag("location")} for Supabase yet: re-run without it. Nothing was created.`);
    }
    return undefined;
  }
  if (!options.choice) {
    const said = options.error?.detail ?? options.error?.message ?? "The platform did not say where the monitoring box can run.";
    // A location of another cloud: name the flag that picks that cloud.
    const elsewhere = location
      ? options.clouds.find((c) => c.available && c.code !== cloud && options.locations[c.code]?.some((l) => l.code === location))
      : undefined;
    const available = options.clouds.filter((c) => c.available).map((c) => c.code);
    const how = elsewhere ? ` ${cloud ? "Use" : "Add"} ${flag("cloud")} ${elsewhere.code}.`
      : /^(Monitoring on .* is not (available here|offered yet)|Unknown cloud)/.test(said) && available.length
        ? ` Re-run with ${flag("cloud")} ${available.length === 1 ? available[0] : `set to one of: ${available.join(", ")}`}.` : "";
    throw new Error(`${said}${how} Nothing was created.`);
  }
  const c = options.choice;
  progress("preparing", `Monitoring box: ${c.cloud_label}, ${c.location} (${c.reason_text})`, { cloud: c.cloud, location: c.location, location_reason: c.reason });
  return { place: { cloud: c.cloud, location: c.location, location_reason: c.reason, location_reason_text: c.reason_text }, options };
}

/**
 * The terminal's question once the place and price are shown (A17): deploy here (y), not at all
 * (n, Enter), or choose another cloud and location from the platform's options (c). `ask` reads an
 * answer, `say` prints a line; three wrong answers are a no.
 */
export async function pickPlace(options: DeployOptions, name: string, ask: (question: string) => Promise<string>,
  say: (line: string) => void): Promise<boolean | { cloud: string; location: string }> {
  const number = async (what: string, n: number): Promise<number | undefined> => {
    for (let tries = 0; tries < 3; tries++) {
      const answer = (await ask(`${what} (1-${n}, Enter to cancel): `)).trim();
      if (!answer) return undefined;
      const i = Number(answer);
      if (Number.isInteger(i) && i >= 1 && i <= n) return i - 1;
      say(`Type a number from 1 to ${n}.`);
    }
    return undefined;
  };
  for (let tries = 0; tries < 3; tries++) {
    const answer = (await ask(`Deploy ${terminalText(name)} here? [y]es / [c]hoose another location / [N]o: `)).trim().toLowerCase();
    if (answer === "y" || answer === "yes") return true;
    if (!answer || answer === "n" || answer === "no") return false;
    if (answer !== "c" && answer !== "choose") { say("Type y, c or n."); continue; }
    const clouds = options.clouds.filter((c) => c.available && options.locations[c.code]?.length);
    say("Clouds:");
    clouds.forEach((c, i) => say(`  ${i + 1}. ${c.label}`));
    const ci = await number("Cloud", clouds.length);
    if (ci === undefined) return false;
    const cloud = clouds[ci]!;
    const locations = options.locations[cloud.code]!;
    say(`Locations on ${cloud.label}:`);
    let part = "";
    locations.forEach((l, i) => {
      if (l.world_part !== part) say(`  ${(part = l.world_part)}`);
      say(`    ${i + 1}. ${l.code}${l.label ? `  ${terminalText(l.label)}` : ""}`);
    });
    const li = await number("Location", locations.length);
    if (li === undefined) return false;
    return { cloud: cloud.code, location: locations[li]!.code };
  }
  return false;
}

/** What deploy does once the box's place and price are shown: go (maybe elsewhere), or stop with what to do. */
type Decision = { go: true; place?: Placed } | { go: false; stop: Partial<ConnectResult> & { next: string } };

/**
 * Whether the box is made, and where (A17, platform-all#972), once its place and price are shown:
 * --yes deploys there; a person at a terminal is asked, and without --cloud/--location may pick
 * another place, which the platform places again; anyone else (piped, --json, an agent) gets the
 * place, the price and the choices back, and nothing is created.
 */
async function decide(placed: Placed, name: string, price: string, opts: ConnectOptions, deps: ConnectDeps,
  where: { db_host?: string; supabase_project_ids?: string[] }, progress: Progress): Promise<Decision> {
  if (opts.yes) return { go: true };
  const { place, options } = placed;
  const label = options.choice?.cloud_label ?? cloudWord(place.cloud);
  const explicit = !!(opts.cloud?.trim() || opts.location?.trim());
  const nothing = "Nothing was created";
  if (deps.choosePlace) {
    if (explicit) {
      return await deps.confirm(`Deploy ${terminalText(name)} on ${label}, ${place.location}? [y/N] `)
        ? { go: true } : { go: false, stop: { next: `${nothing}: re-run with --yes to deploy on ${label}, ${place.location}` } };
    }
    const answer = await deps.choosePlace(options, name);
    if (answer === true) return { go: true };
    if (answer === false) return { go: false, stop: { next: `${nothing}: re-run to deploy, with --cloud and --location to choose where` } };
    // Still the summary before the deploy: plain lines, so no step view starts before a later question (vCPUs).
    const summary: Progress = (_event, message, more) => progress("billing", message, more);
    const again = await placeBox(deps, { ...opts, cloud: answer.cloud, location: answer.location }, where, summary);
    if (!again) throw new Error(`The platform did not place the box on ${answer.cloud}, ${answer.location}. ${nothing}.`);
    summary("billing", `Next time, to skip this question: --cloud ${again.place.cloud} --location ${again.place.location} --yes`);
    return { go: true, place: again };
  }
  // The clouds the picker offers: available, with a location.
  const choices = Object.fromEntries(options.clouds.filter((c) => c.available && options.locations[c.code]?.length)
    .map((c) => [c.code, options.locations[c.code]!.map((l) => l.code)]));
  const here = `${label}, ${place.location} (${place.location_reason_text})`;
  const next = opts.agent
    ? `Show the user where the monitoring box would run (${here}) and the price (${price}), then call deploy_monitoring again with yes: true, or with cloud and location from choices and yes: true. ${nothing}.`
    : `The monitoring box would run on ${here} for ${price}: re-run with --yes to deploy there, or with --cloud and --location (choices below) and --yes. ${nothing}.`;
  return { go: false, stop: { location_reason_text: place.location_reason_text, choices, next } };
}

const isSupabaseUrl = (link: string): boolean => {
  try {
    const u = new URL(link);
    return u.origin === SUPABASE_API_BASE && !u.username && !u.password;
  } catch {
    return false;
  }
};

// How long deploy waits for the consent in the browser, and how often it asks.
const OAUTH_WAIT_MS = 10 * 60_000;
const OAUTH_POLL_MS = 2_000;

/**
 * A Supabase project: the platform creates postgres_ai_mon through the Supabase Management API,
 * so the database is never logged in to from here. The org connects Supabase once, in the browser.
 */
async function deploySupabase(url: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const provider = "supabase";
  const ref = extractProjectRefFromUrl(url);
  if (!ref) return { status: "action_required", provider, name: databaseName(url), next: "Not a Supabase project URL: use the project's connection string (db.<project-ref>.supabase.co, or the pooler with user postgres.<project-ref>), or pass --provider supabase without a URL to pick the project" };
  // One name for the direct and the pooler URL of a project.
  const db = decodeURIComponent(new URL(url).pathname.replace(/^\//, "")) || "postgres";
  const name = databaseName(supabaseProjectUrl(ref, db));
  const stop = (next: string, more: Partial<ConnectResult> = {}): ConnectResult => ({ status: "action_required", provider, name, ...more, next });
  // The box prepares and monitors the project's postgres database (prepare-db --supabase).
  if (db !== "postgres") return stop("PostgresAI monitors a Supabase project through its postgres database: use the URL of the postgres database");
  const refused = supabaseOptionsRefused(opts);
  if (refused) return stop(refused);
  // Both the URL and --supabase-project: they must name the same project, which the list decides.
  if (opts.supabaseProject?.trim()) return deploySupabaseProject(ref, opts, deps, progress);
  const row = await boxOnProject(ref, deps);
  if (row) {
    const note = opts.name && row.name !== opts.name ? `; --name ignored: this database is already monitored as ${row.name}` : "";
    return follow(row, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
  }
  let started: Awaited<ReturnType<ConnectDeps["supabaseStart"]>>;
  try {
    started = await deps.supabaseStart(supabaseProjectUrl(ref));
  } catch (err) {
    if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
    // Another Supabase organization is connected to this org: the platform says which.
    if (err instanceof HttpStatusError && err.status === 409) return stop(errorText(err));
    throw err;
  }
  if ("auth_url" in started) {
    const waited = await supabaseConsent(started, opts, deps, progress, stop);
    if (waited) return waited;
  }
  return supabaseBox(ref, name, opts, deps, progress, stop);
}

/**
 * The box that already monitors a Supabase project's postgres database, if any: also one the
 * platform does not link to the project (made through the generic path). A row of another
 * database there (deployed as self-managed) is not this box.
 */
async function boxOnProject(ref: string, deps: ConnectDeps): Promise<Database | undefined> {
  const cluster = clusterOf(supabaseProjectUrl(ref))!;
  return (await deps.list()).find((d) => !disconnecting(d.status) && d.cluster && serverOf(d.cluster) === cluster && (!d.database || d.database === "postgres"));
}

/** The URL that names a Supabase project's database: what the platform finds it by, and the instance name's source. */
function supabaseProjectUrl(ref: string, db = "postgres"): string {
  return `postgresql://db.${ref}.supabase.co:5432/${encodeURIComponent(db)}`;
}

/** Options that do not apply to a Supabase project, or undefined. */
function supabaseOptionsRefused(opts: ConnectOptions): string | undefined {
  if (opts.resetPassword) return "--reset-password does not apply to Supabase: PostgresAI creates postgres_ai_mon through the Supabase Management API";
  // The platform creates postgres_ai_mon there, and keeps its password: a custom role would not be used.
  if (opts.monitoringUser !== undefined && opts.monitoringUser.trim() !== DEFAULT_MONITORING_USER) {
    return `${opts.agent ? "monitoring_user" : "--monitoring-user"} does not apply to Supabase: PostgresAI creates ${DEFAULT_MONITORING_USER} through the Supabase Management API`;
  }
  return undefined;
}

/**
 * The org approves PostgresAI's access to Supabase in the browser: a person at a terminal waits for it;
 * an agent, or --wait 0, stops with the link. undefined once connected, else the result to return.
 */
async function supabaseConsent(started: { auth_url: string; session_id: string; expires_at?: string }, opts: ConnectOptions, deps: ConnectDeps,
  progress: Progress, stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult): Promise<ConnectResult | undefined> {
  // The link goes to the browser opener and the terminal: Supabase's own https page only.
  if (!isSupabaseUrl(started.auth_url)) return { ...stop(""), status: "failed", next: `The platform sent a Supabase consent link that is not ${SUPABASE_API_BASE}: not opened. Re-run pgai mon deploy later.` };
  const approve = `Open ${started.auth_url} in a browser and approve PostgresAI's access to Supabase`;
  if (opts.agent || opts.waitMs <= 0) return stop(`${approve}, then ${opts.agent ? "call deploy_monitoring again" : "re-run"}`);
  progress("preparing", `Connect Supabase to PostgresAI: approve access in the browser (open ${started.auth_url} if it did not open). Waiting...`, { auth_url: started.auth_url });
  deps.openUrl?.(started.auth_url);
  // A declined consent is never reported: the session stays pending until it expires.
  const expires = started.expires_at ? Date.parse(started.expires_at) : NaN;
  for (const deadline = deps.now() + Math.min(opts.waitMs, OAUTH_WAIT_MS); ;) {
    if (deps.aborted?.()) throw new CancelledError();
    const s = await deps.supabaseStatus(started.session_id);
    if (s.status === "connected") return undefined;
    if (s.status === "error") return stop(`Supabase did not connect: ${(s.error ?? "no reason given").replace(/\.$/, "")}. Re-run to try again`);
    const now = deps.now();
    if (now >= expires) return stop("The Supabase consent link expired before it was approved: re-run pgai mon deploy for a new one");
    if (now >= deadline) return stop(`${approve}, then re-run`);
    await deps.sleep(OAUTH_POLL_MS);
  }
}

/** A Supabase project status in words. */
export function supabaseStatusWords(status: string): string {
  if (status === "inactive" || status === "paused") return "paused";
  if (status === "coming_up") return "starting";
  if (status === "unknown") return "in an unknown state";
  return status.replace(/_/g, " ");
}

/** Why a project can't be picked, in words ("" when it can). */
function supabaseWhy(p: SupabaseProject): string {
  if (p.reason === "monitored") return `already monitored${p.monitored?.name ? ` as ${p.monitored.name}` : ""}`;
  if (p.reason === "paused") return "paused";
  if (p.reason?.startsWith("not_ready:")) return `not ready (${supabaseStatusWords(p.reason.slice("not_ready:".length))})`;
  return "";
}

/**
 * The projects as aligned rows for a person: name, region, project ID, and why one can't be
 * picked. `numbered`: the pickable ones get 1, 2, ... (the picker); the others "-".
 */
export function supabaseProjectRows(projects: SupabaseProject[], numbered = false): string[] {
  const nameW = Math.max(...projects.map((p) => p.name.length));
  const regionW = Math.max(...projects.map((p) => (p.region ?? "").length));
  const numW = String(projects.filter((p) => p.selectable).length).length;
  let n = 0;
  return projects.map((p) => {
    const why = supabaseWhy(p);
    const mark = numbered ? `${(p.selectable ? `${++n}.` : "-").padStart(numW + 1)} ` : "";
    return `  ${mark}${p.name.padEnd(nameW)}  ${(p.region ?? "").padEnd(regionW)}  ${p.ref}${why ? `  ${why}` : ""}`.trimEnd();
  });
}

/** The project --supabase-project names: by ref, else by its exact name (any case); a name several projects share is refused. */
export function matchSupabaseProject(projects: SupabaseProject[], wanted: string): { project: SupabaseProject } | { none: true } | { ambiguous: SupabaseProject[] } {
  const value = wanted.trim();
  const byRef = projects.find((p) => p.ref === value);
  if (byRef) return { project: byRef };
  const byName = projects.filter((p) => p.name.toLowerCase() === value.toLowerCase());
  if (byName.length === 1) return { project: byName[0]! };
  if (byName.length > 1) return { ambiguous: byName };
  return { none: true };
}

/**
 * --provider supabase without a URL (postgresai#438): connect Supabase if needed, list the org's
 * projects, pick one (--supabase-project, or a person at a terminal), then the box, as for a URL.
 * `urlRef`: a URL was given too, and the project picked must be that one.
 */
async function deploySupabaseProject(urlRef: string | undefined, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const provider = "supabase";
  let name = urlRef ? databaseName(supabaseProjectUrl(urlRef)) : "";
  const stop = (next: string, more: Partial<ConnectResult> = {}): ConnectResult => ({ status: "action_required", provider, name, ...more, next });
  const refused = supabaseOptionsRefused(opts);
  if (refused) return stop(refused);
  const urlHint = "pass the project's URL instead (pgai mon deploy 'postgresql://postgres@db.<project-ref>.supabase.co:5432/postgres')";
  if (!deps.supabaseProjects) return stop(`Picking a Supabase project needs a newer pgai: ${urlHint}`);
  const listed = async (): Promise<SupabaseProjects | ConnectResult> => {
    try {
      return await deps.supabaseProjects!();
    } catch (err) {
      if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
      // An API from before postgresai#438 has no such rpc.
      if (err instanceof HttpStatusError && err.status === 404) return stop(`This PostgresAI API cannot list Supabase projects yet: ${urlHint}`);
      throw err;
    }
  };
  let list = await listed();
  if ("status" in list) return list;
  if (!list.connected) {
    const waited = await supabaseConsent(list, opts, deps, progress, stop);
    if (waited) return waited;
    list = await listed();
    if ("status" in list) return list;
    // Approved, yet not connected: the grant went missing between the two calls.
    if (!list.connected) return stop("Supabase did not stay connected after the approval: re-run pgai mon deploy --provider supabase");
  }
  const org = list.supabase_org_name;
  const projects = list.projects;
  const shown = { supabase_org: org, projects };
  // Supabase did not answer the sync: the stored list, which may be out of date.
  if (!list.synced && !projects.length) return stop("Supabase did not answer the project list: re-run in a minute");
  const stale = list.synced ? "" : " (Supabase did not answer just now: the list may be out of date)";
  const inOrg = `organization "${org}"`;
  const useFlag = "--supabase-project <name or project ID>";
  if (!projects.length) return stop(`There are no projects in the Supabase ${inOrg}: create one there, then re-run`, shown);
  const wanted = opts.supabaseProject?.trim();
  let picked: SupabaseProject | undefined;
  if (wanted) {
    const m = matchSupabaseProject(projects, wanted);
    if ("none" in m) return stop(`No Supabase project "${wanted}" in ${inOrg}: re-run with ${useFlag} of one of its projects`, shown);
    if ("ambiguous" in m) return stop(`Several projects in ${inOrg} are called "${wanted}": re-run with the project ID instead`, { supabase_org: org, projects: m.ambiguous });
    picked = m.project;
  } else {
    // Nothing to pick: say why before any prompt.
    if (!projects.some((p) => p.selectable)) {
      return stop(projects.some((p) => p.monitored)
        ? `Every active project in ${inOrg} is already monitored: see pgai mon instances list (a paused project can be monitored once it is resumed in Supabase)`
        : `No active project in ${inOrg}: resume one in Supabase, then re-run`, shown);
    }
    const ask = !opts.agent && !opts.yes ? deps.chooseSupabaseProject : undefined;
    if (!ask) return stop(`Choose the Supabase project to monitor, then re-run with ${useFlag}${stale}`, shown);
    const ref = await ask(projects, org);
    if (deps.aborted?.()) throw new CancelledError();
    if (!ref) return stop(`No project chosen: re-run and pick one, or pass ${useFlag}`, shown);
    picked = projects.find((p) => p.ref === ref);
  }
  if (!picked) return stop(`No project chosen: re-run and pick one, or pass ${useFlag}`, shown);
  if (urlRef && picked.ref !== urlRef) {
    const urlProject = projects.find((p) => p.ref === urlRef);
    return stop(`--supabase-project is ${picked.name}, but the database URL is for ${urlProject ? urlProject.name : `project ${urlRef}`}: pass only one of them`);
  }
  name = databaseName(supabaseProjectUrl(picked.ref));
  // Already monitored: that box, as on a re-run with the URL; no second box.
  if (picked.monitored) {
    const box = (await deps.list()).find((d) => d.id === picked!.monitored!.id);
    if (box && !disconnecting(box.status)) {
      const note = opts.name && box.name !== opts.name ? `; --name ignored: this database is already monitored as ${box.name}` : "";
      return follow(box, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
    }
    return stop(`${picked.name} is already monitored${picked.monitored.name ? ` as ${picked.monitored.name}` : ""}: see pgai mon instances list`);
  }
  if (picked.reason === "paused") return stop(`${picked.name} is paused: resume it in Supabase, then re-run`);
  if (!picked.selectable) return stop(`${picked.name} is not ready in Supabase (${supabaseStatusWords(picked.status)}): re-run when it is active`);
  // A box on the project's database that the platform does not link to the project: that one, not a second.
  const existing = await boxOnProject(picked.ref, deps);
  if (existing) {
    const note = opts.name && existing.name !== opts.name ? `; --name ignored: this database is already monitored as ${existing.name}` : "";
    return follow(existing, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
  }
  return supabaseBox(picked.ref, name, opts, deps, progress, stop, `${picked.name} (Supabase)`);
}

/** The price, then the box for a project of the connected Supabase organization, then following it. */
async function supabaseBox(ref: string, name: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress,
  stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult, label?: string): Promise<ConnectResult> {
  const provider = "supabase";
  // Where the box runs: by default the project's own region on AWS (platform-all#972).
  const where = { supabase_project_ids: [ref] };
  const placed = await placeBox(deps, opts, where, progress);
  let place = placed?.place;
  let quote: Quote;
  try {
    // The org's price, not a cluster's: the Supabase provision bills every box.
    quote = await deps.quote(opts.coupon);
  } catch (err) {
    if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
    throw err;
  }
  const b = await billingFor(quote, label ?? name, opts, deps, (price) => progress("billing", estimateLine(price), label ? { project: label } : {}),
    placed && ((price) => decide(placed, label ?? name, price, opts, deps, where, progress)));
  if (b.stop) return { status: "action_required", provider, name, ...place, ...b.stop };
  if (b.placed) place = b.placed.place;
  if (deps.aborted?.()) throw new CancelledError();
  progress("provisioning", `Provisioning monitoring for ${label ?? name}`, label ? { project: label } : {});
  // vCPUs: the platform reads the project's compute size; --vcpus overrides it.
  const created = await deps.supabaseConnect({
    project_ref: ref,
    ...(opts.name ? { name: opts.name } : {}),
    ...(b.accepted ? { accept_price: true } : {}),
    ...(opts.coupon ? { promo_code: opts.coupon } : {}),
    ...(opts.vcpus ? { vcpus: opts.vcpus } : {}),
    ...(place ? { cloud: place.cloud, location: place.location } : {}),
  }).catch((err) => {
    if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
    if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
    // An org billed by a partner (or a box that exists already): the platform says what to do.
    if (err instanceof HttpStatusError && err.status === 409) return { refused: errorText(err) };
    throw err;
  });
  if (typeof created === "object" && "refused" in created) return stop(created.refused);
  if (created === "payment") return { status: "action_required", provider, name, ...b.billingPage };
  if (created === "declined") return { status: "action_required", provider, name, ...b.billingPage, next: b.billingPage.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
  if (created === "price") return stop("The price changed since it was shown: re-run pgai mon deploy to see it", b.billing);
  if (created.status === "failed") return { status: "failed", provider, name, id: created.id, ...b.billing, next: `${created.error} Re-run pgai mon deploy later.` };
  const box: Database = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: false };
  // Ctrl-C while the box was requested: it exists now, so the run detaches (it keeps going on PostgresAI).
  if (deps.aborted?.()) return { ...connectStatus(box, provider, true), ...b.billing };
  return follow(box, { provider, name, fresh: true, billing: b.billing, billingUrl: b.billingUrl, note: "", place }, opts, deps, progress);
}

type Progress = (event: ProgressEvent["event"], message: string, more?: Partial<ProgressEvent>) => void;

/** What deploy knows of a box when it starts following it. */
interface Followed {
  provider: Provider;
  name: string;
  /** Provisioned by this run: the result gets the first checkup's ETA. */
  fresh: boolean;
  billing: Billing;
  billingUrl?: string;
  /** Drops the role this run created with a generated password, once its box is gone. */
  undoRole?: () => Promise<unknown>;
  /** Appended to the next action ("; --name ignored ..."). */
  note: string;
  /** Where the box runs (placeBox), for the result. */
  place?: Place;
}

/** How the next action starts when deploy's wait ended before the box was ready (the step view is not failed then). */
export const STILL_IN_PROGRESS = "Still in progress";

/** Waits for the box until it is no longer starting, or opts.waitMs passed: its state as the result. */
async function follow(row: Database, f: Followed, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const { provider, name, fresh, billingUrl, undoRole, note } = f;
  // The result carries where a new box runs, next to its billing.
  const billing = { ...f.billing, ...f.place };
  // The box's state, each time it changes while deploy waits for it (not on a re-run of a connected database).
  let shown: string | undefined;
  for (const deadline = deps.now() + opts.waitMs; ;) {
    if (row.billing_error && disconnecting(row.status)) {
      if (undoRole) await undoRole();
      return { status: "action_required", provider, name, id: row.id, ...billing,
        next: chargeFailed(row.billing_error, "re-run", billingUrl, opts.agent ? "coupon" : "--coupon") };
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
      // The wait ran out with the box still starting: not done (exit 3); it goes on on the platform.
      if (final.status === "in_progress" && opts.waitMs > 0) {
        return { ...rest, status: "action_required", ...billing,
          next: `${STILL_IN_PROGRESS} (box: ${boxWords(boxState(row))}) when the wait ended: resume with ${next}${note}` };
      }
      return { ...rest, ...billing, next: next + note };
    }
    await deps.sleep(POLL_MS);
    // The box is already requested: a failed poll keeps the last known state.
    const listed = await deps.list().catch(() => undefined);
    const now = listed?.find((d) => d.id === row.id);
    // A box this run created that is gone (its role, if this run made one, is dropped).
    if (listed && !now && fresh) {
      if (undoRole) await undoRole();
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
  /** What pgai mon instances delete takes for each of `others` (an id where a name is shared). */
  otherRefs?: string[];
  /** The monitoring role (default postgres_ai_mon). */
  monitoringUser?: string;
  /** The pg client class (tests). */
  Client?: PgClientClass;
  /** Gets the TLS warnings (default: stderr). */
  warn?: Warn;
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
  const { client, usedSsl } = await connectWithSslFallback(opts.Client ?? Client, { ...conn, sslFallbackEnabled: fallback }, false, opts.warn);
  await client.query("set search_path = pg_catalog, pg_temp");
  return { client, usedSsl, refusedTls: fallback && !usedSsl };
}

/** Drops the monitoring role and what it was granted in this database; false when the server refuses. */
async function dropMonitoringRole(client: PgClient, role: string): Promise<boolean> {
  const q = quoteIdent(role);
  // Alone first: a role with no grants yet needs no DROP OWNED, which a CREATEROLE admin may not run.
  for (const sql of [`drop role ${q}`, `drop owned by ${q}; drop role ${q}`]) {
    if (await client.query(sql).then(() => true, () => false)) return true;
  }
  return false;
}

/** What the plan grants (pg_monitor and pg_read_all_stats), and what pg_monitor itself is a member of. */
const MONITORING_GROUPS = ["pg_monitor", "pg_read_all_stats", "pg_read_all_settings", "pg_stat_scan_tables"];

/** The functions the plan grants EXECUTE on (sql/04, sql/05, sql/06), as schema.name(argument types). */
const MONITORING_FUNCTIONS = ["pg_catalog.pg_stat_file(text)", "pg_catalog.pg_stat_file(text, boolean)", "pg_catalog.pg_ls_dir(text)",
  "pg_catalog.pg_ls_dir(text, boolean, boolean)", "rds_tools.pg_ls_multixactdir()", "postgres_ai.table_describe(text)"];

/** Undoes a `generated` prepareDatabase over the same admin URL: drops the role that run created. */
export async function unprepareDatabase(url: string, opts: PrepareOptions = {}): Promise<boolean> {
  const { client } = await openConnection(url, opts);
  try {
    return await dropMonitoringRole(client, opts.monitoringUser || DEFAULT_MONITORING_USER);
  } finally {
    await client.end();
  }
}

/** Creates (or checks) the monitoring role over the given URL. The admin URL is used for this run only. */
export async function prepareDatabase(url: string, provider: Provider, opts: PrepareOptions = {}): Promise<Prepared> {
  const pgProvider = provider === "rds" ? "self-managed" : provider;
  const role = opts.monitoringUser || DEFAULT_MONITORING_USER;
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
         exists (select 1 from pg_roles where rolname = ${quoteLiteral(role)}) as mon_exists,
         exists (select 1 from pg_roles where rolname = ${quoteLiteral(role)} and (rolsuper or rolcreatedb or rolcreaterole or rolreplication or rolbypassrls)) as mon_privileged,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles m join pg_roles g
            on g.oid <> m.oid and pg_has_role(m.oid, g.oid, 'MEMBER')
          where m.rolname = ${quoteLiteral(role)} and g.rolname not in (${MONITORING_GROUPS.map(quoteLiteral).join(", ")})) as mon_groups,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles m
          join pg_auth_members a on a.member = m.oid join pg_roles g on g.oid = a.roleid
          where m.rolname = ${quoteLiteral(role)} and a.admin_option) as mon_admin_groups,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles g
          where g.rolname not in (${[...MONITORING_GROUPS, role].map(quoteLiteral).join(", ")})
            and (pg_has_role('pg_monitor', g.oid, 'MEMBER') or pg_has_role('pg_read_all_stats', g.oid, 'MEMBER'))) as mon_plan_groups,
         exists (select 1 from pg_shdepend d join pg_roles m on m.oid = d.refobjid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'o' and m.rolname = ${quoteLiteral(role)}) as mon_owns,
         (select string_agg(distinct db.datname::text, ', ') from pg_shdepend d join pg_roles m on m.oid = d.refobjid join pg_database db on db.oid = d.dbid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'a' and m.rolname = ${quoteLiteral(role)}
            and db.datname <> current_database()) as mon_other_dbs,
         (select string_agg(distinct pg_describe_object(d.classid, d.objid, d.objsubid), ', ') from pg_shdepend d join pg_roles m on m.oid = d.refobjid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'a' and m.rolname = ${quoteLiteral(role)}
            and d.dbid in (0, (select oid from pg_database where datname = current_database()))
            and not exists (select 1 from (
                select relacl as acl, 'SELECT' as ok from pg_class where d.classid = 'pg_class'::regclass and d.objsubid = 0 and oid = d.objid
                  and ((relnamespace not in ('pg_catalog'::regnamespace, 'pg_toast'::regnamespace) and relkind <> 't') or oid = 'pg_catalog.pg_index'::regclass)
                union all select attacl, 'SELECT' from pg_attribute where d.classid = 'pg_class'::regclass and attrelid = d.objid and attnum = d.objsubid
                  and (select relnamespace not in ('pg_catalog'::regnamespace, 'pg_toast'::regnamespace) and relkind <> 't' from pg_class where oid = d.objid)
                union all select nspacl, 'USAGE' from pg_namespace where d.classid = 'pg_namespace'::regclass and oid = d.objid and nspname !~ '^pg_toast'
                union all select proacl, 'EXECUTE' from pg_proc where d.classid = 'pg_proc'::regclass and oid = d.objid
                  and pronamespace::regnamespace::text || '.' || proname || '(' || array_to_string(proargtypes::oid[]::regtype[], ', ') || ')'
                    in (${MONITORING_FUNCTIONS.map(quoteLiteral).join(", ")})
                union all select datacl, 'CONNECT' from pg_database where d.classid = 'pg_database'::regclass and oid = d.objid) o
              where not exists (select 1 from aclexplode(o.acl) e where e.grantee = d.refobjid and (e.privilege_type <> o.ok or e.is_grantable)))) as mon_grants,
         current_setting('scram_iterations', true) as iterations, current_setting('ssl') as ssl
       from pg_roles where rolname = current_user`,
    )).rows[0];
    const user = loginName(url, me.name, role);
    // A custom role is the user's own: one that can do more than monitor is refused (its search_path
    // would be changed, and its credentials sent to the platform). Other memberships may carry
    // superuser or file access; owned objects may be the application's tables. Grants beyond the
    // plan's may write data, read password verifiers (pg_authid, or its TOAST) or run file functions.
    // Memberships count through any chain, also through pg_monitor and pg_read_all_stats, which the plan
    // grants: a role it has not made their member yet would inherit what they are members of. A grant in another database
    // cannot be judged from here (its catalog is not readable), so any is refused.
    const custom = role !== DEFAULT_MONITORING_USER;
    const unsafe = me.mon_privileged ? "is a superuser, or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS: a monitoring role must not"
      : me.mon_groups ? `is a member of ${me.mon_groups}: a monitoring role may be a member of ${MONITORING_GROUPS.slice(0, -1).join(", ")} and ${MONITORING_GROUPS.at(-1)} only`
      : me.mon_admin_groups ? `is a member of ${me.mon_admin_groups} WITH ADMIN OPTION: a monitoring role must not have ADMIN OPTION`
      : me.mon_plan_groups ? `would become a member of ${me.mon_plan_groups} through pg_monitor or pg_read_all_stats: a monitoring role may be a member of ${MONITORING_GROUPS.slice(0, -1).join(", ")} and ${MONITORING_GROUPS.at(-1)} only`
      : me.mon_owns ? "owns objects on this server: a monitoring role must own nothing"
      : me.mon_other_dbs ? `has grants in database ${me.mon_other_dbs}: a monitoring role may have grants in this database only`
      : me.mon_grants ? `has grants on ${me.mon_grants}: a monitoring role may have only CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog and pg_toast, and the grants pgai makes`
      : "";
    if (custom && unsafe) return { next: `${role} ${unsafe}. Use another ${opts.agent ? "monitoring_user" : "--monitoring-user"}` };
    const notes = [caNote(url)].filter((n): n is string => !!n);
    const noted = () => (notes.length ? { note: notes.join("; ") } : {});
    // A server that does not check passwords for this client (trust) accepts a random one too.
    const acceptsAnyPassword = async () =>
      logsIn(loginUrlFor(url, user, me.db, (await resolveMonitoringPassword({ monitoringUser: role })).password)).catch(() => false);
    const unchecked = (what: string) => `this server accepts any password from this host, so ${what} was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password`;
    if (me.name === role) {
      const v = await verifyInitSetup({ client, database: me.db, monitoringUser: me.name, includeOptionalPermissions: false, provider: pgProvider });
      if (v.ok) {
        // A URL without a password (PGPASSWORD, say): the box gets the one this client logged in with.
        const used = (client as { password?: unknown }).password;
        const inUrl = urlPassword(new URL(url));
        const password = inUrl || (typeof used === "string" ? used : "");
        const putInUrl = `Put the password of ${role} in the URL: the monitoring box logs in with it`;
        if (!password) return { next: putInUrl };
        if (await acceptsAnyPassword()) {
          // What pg read from PGPASSWORD may be for another role or server: unchecked, it is not sent on.
          if (!inUrl) return { next: `This server accepts any password from this host, so the one from the environment was not checked and is not used. ${putInUrl}` };
          notes.push(unchecked("the password in the URL"));
        }
        return { monitoringUrl: monitoringUrlFor(url, user, me.db, password), ...noted() };
      }
    } else if (me.admin && pgProvider !== "supabase") {
      // The role is cluster-wide: another database here may use its password,
      // so it is never changed, and only a password that logs in is used.
      const exists = `${role} already exists on this server`;
      const setPassword = custom
        ? `Set PGAI_MON_PASSWORD to the password of ${role}, or give it a new one: PGAI_MON_PASSWORD=<new password> pgai mon deploy <admin-url> --monitoring-user ${role} --reset-password (anything else that logs in as ${role} then needs the new password)`
        : `Set PGAI_MON_PASSWORD to the password of ${role}, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as ${role} then needs the new password)`;
      // The platform fills in the password it keeps for the box (it never sends it here), over TLS only:
      // what the URL lacks for that, and whether the server takes TLS at all (it refused it, or has ssl off).
      const needs = storedPasswordNeeds(monitoringUrlFor(url, user, me.db, ""));
      const noTls = refusedTls || (!usedSsl && me.ssl === "off");
      const tlsOnly = `PostgresAI sends the password it keeps for ${role} only over TLS`;
      const others = opts.others?.length ? opts.others : undefined;
      // Nobody has the password (the first deploy generated it): only a new one, once nothing uses the old one.
      const disconnectOthers = others ? disconnectEach(opts.otherRefs ?? others) : "pgai mon instances delete the server's other databases (pgai mon instances list)";
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
        const { password: unused } = await resolveMonitoringPassword({ monitoringUser: role });
        const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: unused, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
        await applyInitPlan({ client, plan });
        return { monitoringUrl: monitoringUrlFor(url, user, me.db, ""), storedPassword: true as const, ...noted() };
      }
      if (me.mon_exists && opts.agent) return { next: `${exists}. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password` };
      // A PGAI_MON_PASSWORD that does not log in, where the platform keeps the password: without it, that one
      // is used (over TLS only).
      const unset = !opts.storedPassword ? setPassword
        : noTls ? `This server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}`
        : !needs ? `Unset PGAI_MON_PASSWORD: PostgresAI keeps the password of ${role} for this server`
        : `Unset PGAI_MON_PASSWORD, and the URL needs ${needs}: PostgresAI keeps the password of ${role} for this server, and sends it only over TLS, to a URL with each parameter once`;
      const reset = !!opts.resetPassword && me.mon_exists;
      // PostgresAI keeps no password for a custom role: one it set is the user's (a generated one would be lost).
      if (custom && (!me.mon_exists || reset) && !(opts.agent ? "" : process.env.PGAI_MON_PASSWORD?.trim())) {
        return { next: opts.agent
          ? `${role} is a custom role, and PostgresAI keeps no password for it: pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set`
          : `Set PGAI_MON_PASSWORD to a password for ${role}: PostgresAI keeps no password for a custom role, so a generated one would be lost` };
      }
      const { password, generated } = await resolveMonitoringPassword({ passwordEnv: opts.agent ? undefined : process.env.PGAI_MON_PASSWORD, monitoringUser: role });
      if (me.mon_exists && !reset) {
        if (!process.env.PGAI_MON_PASSWORD?.trim()) return { next: `${exists}. ${setPassword}`, ...(custom ? {} : { resettable: true as const }) };
        let accepted: boolean;
        try {
          accepted = await logsIn(loginUrlFor(url, user, me.db, password));
        } catch (err) {
          // pg_hba for this client, say: neither accepted nor rejected. With the platform's password, no login is needed.
          const orUnset = opts.storedPassword && !noTls ? ", or unset PGAI_MON_PASSWORD: PostgresAI keeps its password for this server" : "";
          return { next: `${exists}, and PGAI_MON_PASSWORD could not be checked from this host (${err instanceof Error ? err.message : String(err)}). Run pgai mon deploy from a host that ${role} may connect from${orUnset}` };
        }
        if (!accepted) return { next: `${exists} and PGAI_MON_PASSWORD is not its password. ${unset}` };
        if (await acceptsAnyPassword()) notes.push(unchecked("PGAI_MON_PASSWORD"));
      }
      if (opts.check) return { checked: true };
      const monitoringUrl = monitoringUrlFor(url, user, me.db, password);
      const loginUrl = loginUrlFor(url, user, me.db, password);
      const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: password, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: !reset });
      try {
        await applyInitPlan({ client, plan });
      } catch (err) {
        // A later step failed after the role was created with a generated password (it logs in,
        // so the role is this run's): nobody has that password, so the role is dropped again.
        if (generated && !me.mon_exists && (await logsIn(loginUrl).catch(() => false))) await dropMonitoringRole(client, role);
        throw err;
      }
      // Another session may have created the role meanwhile, with its own password.
      // Other login errors (pg_hba for this client, say) do not tell, so they pass.
      if (!(await logsIn(loginUrl).catch(() => true))) return { next: `${role} was created by someone else meanwhile. ${setPassword}` };
      return { monitoringUrl, ...noted(), ...(generated && !me.mon_exists ? { generated: true as const } : {}) };
    }
    const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: "<password>", includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
    return {
      sql: plan.steps.map((s) => `-- ${s.name}${s.optional ? " (optional: an error in this part can be ignored)" : ""}\n${redactPasswordsInSql(s.sql)}`).join("\n\n"),
      next: `Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai mon deploy again with the ${role} URL`,
    };
  } finally {
    await client.end();
  }
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
    setVcpus: (id: string, vcpus: number) => rpc<{ vcpus: number; previous_vcpus: number | null; changed: boolean; aas_status: string }>("cloud_monitoring_set_vcpus", { instance_id: id, vcpus }),
    resetLock: (server: string) => rpc<{ lock_id: string }>("cloud_monitoring_reset_lock", { server }),
    resetUnlock: (lockId: string) => rpc("cloud_monitoring_reset_unlock", { lock_id: lockId }),
    prepare: (url: string, provider: Provider, o: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[]; otherRefs?: string[]; monitoringUser?: string; warn?: Warn } = {}) => prepareDatabase(url, provider, { ...o, agent: p.agent }),
    unprepare: (url: string, monitoringUser?: string, o: { warn?: Warn } = {}) => unprepareDatabase(url, { ...o, agent: p.agent, monitoringUser }),
    clickhouseOrg: clickhouseOrgFor,
    quote: (coupon?: string, cluster?: string) => rpc<Quote>("cloud_monitoring_quote", { ...(coupon ? { promo_code: coupon } : {}), ...(cluster ? { db_server: cluster } : {}) }),
    billingUrl: (orgAlias: string) => `${p.uiBaseUrl}/${orgAlias}/billing`,
    supabaseStart: (dbUrl: string) => rpc<{ connected: true; project_ref: string } | { auth_url: string; session_id: string; expires_at?: string }>("cloud_monitoring_supabase_start", { db_url: dbUrl }),
    supabaseStatus: (sessionId: string) => rpc<{ status: "pending" | "connected" | "error"; error?: string }>("cloud_monitoring_supabase_status", { session_id: sessionId }),
    supabaseConnect: (body: Record<string, string | boolean | number>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_supabase_connect", body),
    supabaseProjects: () => rpc<SupabaseProjects>("cloud_monitoring_supabase_projects", {}),
    deployOptions: (args: Parameters<NonNullable<ConnectDeps["deployOptions"]>>[0]) => rpc<DeployOptions>("monitoring_deploy_options", args),
    handoffUrl: async (provider: "rds") => {
      const orgs = await listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl });
      const org = orgs.length === 1 ? orgs[0] : orgs.find((o) => o.alias === p.orgScope?.alias || o.org_id === p.orgScope?.id);
      return `${p.uiBaseUrl}/${org?.alias ?? "<org>"}/monitoring/scale/create/${provider}`;
    },
    sleep: (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
    now: () => Date.now(),
  };
}
