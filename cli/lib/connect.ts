import { Client } from "pg";
import { generateAllReports } from "./checkup";
import { createCheckupReport, uploadCheckupReportJson } from "./checkup-upload";
import { findService } from "./clickhouse";
import {
  applyInitPlan, buildInitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER,
  maskConnectionString, redactPasswordsInSql, resolveAdminConnection, resolveMonitoringPassword, verifyInitSetup,
} from "./init";
import { callRpc } from "./joe";
import { listOrgs, type OrgScope } from "./org-scope";
import { HttpStatusError, requestTimeoutSignal } from "./util";

// `pgai connect <database-url>`: put a database under PostgresAI's care
// (postgres-ai/internal#354). Each step is skipped when already done, so a
// re-run is safe; every outcome carries the exact next action.

export type Provider = "clickhouse" | "rds" | "supabase" | "self-managed";
export type Status = "connected" | "provisioning" | "disconnecting" | "disconnected" | "action_required" | "failed";

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

/** A step of connect, once each: a line for a person, a JSON event for an agent. */
export interface ProgressEvent {
  event: "billing" | "preparing" | "provisioning" | "checkup" | "box";
  /** Seconds since connect started. */
  elapsed_s: number;
  message: string;
  /** event box: the box's state (launch_requested, registered, active, ...). */
  state?: string;
  checkup?: CheckupResult;
}

export interface ConnectResult {
  status: Status;
  provider: Provider;
  name: string;
  id?: string;
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
  provider: string;
  status: string | null;
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
export type Prepared = { monitoringUrl: string; note?: string; generated?: true; storedPassword?: true } | { next: string; sql?: string } | { checked: true };

export interface ConnectDeps {
  list(): Promise<Database[]>;
  create(body: Record<string, string | boolean>): Promise<{ id: string; name: string; status: string; error?: string }>;
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
  sleep(ms: number): Promise<void>;
  now(): number;
  progress(event: ProgressEvent): void;
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
}

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

/** A disconnect in flight (not one that failed to launch, which can be retried). */
export const disconnecting = (status: string | null) => /delet/.test(status ?? "") && !/fail/.test(status ?? "");

/** The platform's instance state in the words connect, status and databases all use. */
export function stateOf(raw: string | null): Status {
  if (raw === "active") return "connected";
  if (raw === "deleted") return "disconnected";
  if (disconnecting(raw)) return "disconnecting";
  if (/fail|error/.test(raw ?? "")) return "failed";
  return "provisioning";
}

/** A rejected ClickHouse Cloud key, or one that cannot see the service: the user's to fix (exit 3). */
export class ClickhouseKeyError extends Error {}

/** host[:port] of a name (host[:port]/db), lowercased: the platform keys a server on lower(host). */
const serverOf = (name: string) => name.split("/")[0].toLowerCase();

/** lower(host):port of a URL (5432 when it names none): the cluster the price is per, as the platform keys it. */
export function clusterOf(url: string): string | undefined {
  const u = parseUrl(url);
  return u?.hostname ? `${u.hostname.toLowerCase()}:${u.port || "5432"}` : undefined;
}

/** An error's text; an AggregateError (node tried each address of a host name) has none of its own: its errors'. */
export const errorText = (err: unknown): string =>
  err instanceof AggregateError && !err.message ? err.errors.map(errorText).join("; ") : err instanceof Error ? err.message : String(err);

const urlPassword = (u: URL) => decodeURIComponent(u.password) || u.searchParams.get("password") || "";

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
export function checkUrlParams(url: string, agent?: boolean): void {
  const keys = [...new Set(new URL(url).searchParams.keys())];
  if (agent) {
    const refused = keys.filter((k) => k !== "password" && !URL_PARAMS_KEPT.includes(k));
    if (refused.length) throw new Error(`database_url may carry only these query parameters: ${URL_PARAMS_KEPT.join(", ")} (got: ${refused.join(", ")})`);
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

/** `pgai disconnect` for each database named. */
const disconnectEach = (names: string[]) => names.map((n) => `pgai disconnect ${shellWord(n)} --yes`).join(" and ");

/** postgres_ai_mon's URL for the prepared database, with the query parameters in `kept`. */
function roleUrlFor(url: string, db: string, password: string, kept: string[]): string {
  const u = new URL(url);
  u.username = DEFAULT_MONITORING_USER;
  u.password = encodeURIComponent(password);
  u.pathname = `/${encodeURIComponent(db)}`;
  for (const name of [...u.searchParams.keys()]) if (!kept.includes(name)) u.searchParams.delete(name);
  return u.toString();
}

/** The URL a box uses: only postgres_ai_mon's credentials, and the prepared database by name. */
const monitoringUrlFor = (url: string, db: string, password: string) => roleUrlFor(url, db, password, URL_PARAMS_KEPT);

/** The same login from this machine: with the given URL's TLS files (a private CA, say). */
const loginUrlFor = (url: string, db: string, password: string) => roleUrlFor(url, db, password, [...URL_PARAMS_KEPT, ...URL_PARAMS_TLS]);

/** What to tell the user when the URL verifies the server with a CA file the box will not have. */
function caNote(url: string): string | undefined {
  const q = new URL(url).searchParams;
  if (!q.get("sslrootcert") || !/^verify-/.test(q.get("sslmode") ?? "")) return undefined;
  return `the monitoring box has no copy of the CA in sslrootcert: with sslmode=${q.get("sslmode")} it connects only to a server certificate signed by a public CA (else connect with sslmode=require)`;
}

/** A platform row as a result with its next action; `fresh` (provisioned just now) adds the first-checkup ETA. */
export function connectStatus(row: Database, provider = row.provider as Provider, fresh = false): ConnectResult {
  const base = { provider, name: row.name, id: row.id, dashboard_url: row.dashboard_url, host_metrics: row.host_metrics };
  const status = stateOf(row.status);
  if (status === "connected") {
    return {
      ...base, status,
      ...(fresh ? { first_checkup_eta: new Date(Date.now() + FIRST_CHECKUP_DELAY_MS).toISOString() } : {}),
      next: row.dashboard_url ? `Open ${row.dashboard_url}` : `pgai status ${row.name}`,
    };
  }
  if (status === "disconnecting" || status === "disconnected") return { ...base, status, next: "none" };
  if (status === "failed") return { ...base, status, next: `pgai disconnect ${row.name} --yes, then pgai connect again` };
  return { ...base, status, next: `pgai status ${row.name}` };
}

/** The monitoring URL with the admin URL's TLS files: for a login from this machine, never for the box. */
function withLocalTls(monitoringUrl: string, url: string): string {
  const u = new URL(monitoringUrl);
  for (const [k, v] of new URL(url).searchParams) if (URL_PARAMS_TLS.includes(k)) u.searchParams.set(k, v);
  return u.toString();
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

/** The box's state as connect shows it: registered between the launch and active. */
const boxState = (row: Database) => (row.status === "launch_requested" && row.registered_at ? "registered" : row.status ?? "starting");

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
  const billingPage = { ...billing, requires_payment_method: true, next: `Add a payment method at ${billingUrl}, then re-run` };
  const stop = (next: string) => ({ billing, billingPage, billingUrl, accepted: false, stop: { ...billing, next } });
  if (billing.coupon && !billing.coupon.valid) {
    return stop(`Promo code ${opts.coupon}: ${billing.coupon.error}. Nothing was changed: re-run with a valid code, or without ${opts.agent ? "coupon" : "--coupon"}`);
  }
  if (q.requires_payment_method) return { billing, billingPage, billingUrl, accepted: false, stop: billingPage };
  // The price on its own line first: a prompt may wrap or be cut where it is shown.
  show(billing.price!);
  if (q.billed && !opts.yes && !(await deps.confirm(`Provision ${name}? (y/N): `))) {
    return stop(opts.agent ? `Call connect_database again with yes: true to accept ${billing.price}` : `Re-run with --yes to accept ${billing.price}`);
  }
  return { billing, billingPage, billingUrl, accepted: q.billed };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
  const started = deps.now();
  const progress = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}) =>
    deps.progress({ event, elapsed_s: Math.round((deps.now() - started) / 1000), message, ...more });
  const provider = (opts.provider ?? detectCloudProvider(url)) as Provider;
  if (!PROVIDERS.includes(provider)) throw new Error(`--provider must be one of: ${PROVIDERS.join(", ")}`);
  checkUrlParams(url, opts.agent);
  const name = databaseName(url);
  // The exported key pair is read for ClickHouse only, and never for an agent's URL; elsewhere only the flag is an error.
  const key = parseClickhouseKey(opts.clickhouseKey, provider === "clickhouse" && !opts.agent ? process.env : {});
  if (key && provider !== "clickhouse") throw new Error("--clickhouse-key applies to ClickHouse Managed Postgres only");

  if (!opts.selfHosted && (provider === "rds" || provider === "supabase")) {
    return { status: "action_required", provider, name, next: `Finish in the console: ${await deps.handoffUrl(provider)}` };
  }

  if (opts.selfHosted && opts.resetPassword) {
    return { status: "action_required", provider, name, next: "--reset-password works with PostgresAI Cloud only (it checks what else monitors this server): connect without --self-hosted, or set PGAI_MON_PASSWORD" };
  }

  if (opts.selfHosted && deps.localStackRunning()) {
    return { status: "action_required", provider, name, next: "A monitoring stack already runs on this machine: add the database with pgai mon targets add '<postgres_ai_mon URL>'" };
  }

  const rows = opts.selfHosted ? [] : await deps.list();
  let row = rows.find((d) => d.name === name && !disconnecting(d.status));
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
  if (row && opts.resetPassword) {
    return { status: "action_required", provider, name, id: row.id, next: `${name} is already connected, and its monitoring uses the current password: pgai disconnect ${name} --yes first` };
  }
  if (row && key) {
    const found = await findOrg();
    if ("error" in found) return { status: "action_required", provider, name, id: row.id, next: `${found.error} Nothing was changed: re-run with the right key, or without one` };
  }
  if (!row) {
    let ch: { orgId: string } | undefined;
    if (key) {
      const found = await findOrg();
      if ("error" in found) return { status: "action_required", provider, name, next: `${found.error} Then re-run` };
      if (found.state !== "running") {
        return { status: "action_required", provider, name, next: `Start the service in the ClickHouse Cloud console (it is ${found.state}), then re-run` };
      }
      ch = found;
    }
    // The platform keeps postgres_ai_mon's password for a server it monitors a database on, for this org,
    // and fills it in while a box there is not being destroyed (a delete that failed counts as one).
    const storedPassword = rows.some((d) => d.monitoring_password_stored && !/delet/.test(d.status ?? "") && serverOf(d.name) === serverOf(name)) || undefined;
    // postgres_ai_mon is one role for the whole server: a new password cuts off what uses the old one
    // (a box whose delete failed may still run).
    const others = rows.filter((d) => d.name !== name && !disconnecting(d.status) && serverOf(d.name) === serverOf(name)).map((d) => d.name);
    if (opts.resetPassword && others.length) {
      const instead = [
        ...(storedPassword ? ["connect without --reset-password (PostgresAI keeps its password for this server, and sends it only over TLS)"] : []),
        "set PGAI_MON_PASSWORD to its password instead",
        `${disconnectEach(others)} first`,
      ];
      return { status: "action_required", provider, name, next: `A new password for ${DEFAULT_MONITORING_USER} would cut off the monitoring of ${others.join(", ")} on this server: ${instead.join(", or ")}` };
    }
    // What the box costs, before anything is touched: a billed box is accepted
    // (--yes, or at the prompt), and needs the org's payment method.
    if (opts.coupon !== undefined && !opts.coupon.trim()) {
      return { status: "action_required", provider, name, next: `${opts.agent ? "coupon" : "--coupon"} is empty: pass a promotion code, or leave ${opts.agent ? "coupon" : "--coupon"} out` };
    }
    const prepOpts = { resetPassword: opts.resetPassword, ...(storedPassword ? { storedPassword, others } : {}) };
    if (!opts.selfHosted) {
      // A URL that cannot work is told so before any price is asked (nothing is changed here).
      const probe = await deps.prepare(url, provider, { ...prepOpts, check: true });
      if ("next" in probe) return { status: "action_required", provider, name, ...probe };
      const b = await billingFor(await deps.quote(opts.coupon, clusterOf(url)), name, opts, deps, (price) => progress("billing", `Billing: ${price}`));
      if (b.stop) return { status: "action_required", provider, name, ...b.stop };
      ({ billing, billingPage, billingUrl, accepted } = b);
    }
    progress("preparing", `Preparing ${maskConnectionString(url)}`);
    const prepared = await deps.prepare(url, provider, prepOpts);
    if ("next" in prepared) return { status: "action_required", provider, name, ...billing, ...prepared };
    if (!("monitoringUrl" in prepared)) throw new Error("prepare returned no monitoring URL");
    if (prepared.note) note = `; ${prepared.note}`;
    if (opts.selfHosted) {
      await deps.selfHosted(prepared.monitoringUrl, key && ch
        ? { CLICKHOUSE_ORG_ID: ch.orgId, CLICKHOUSE_KEY_ID: key.keyId, CLICKHOUSE_KEY_SECRET: key.keySecret }
        : {});
      return { status: "connected", provider, name, dashboard_url: "http://localhost:3000", host_metrics: !!ch, next: `pgai mon health${note}` };
    }
    progress("provisioning", `Provisioning monitoring for ${name}`);
    // A refused launch starts no box, so a role with a generated password is
    // dropped again: nobody has that password, and the re-run would stop at it.
    const undo = async () => { if (prepared.generated) await deps.unprepare(url).catch(() => false); };
    const created = await deps.create({
      db_url: prepared.monitoringUrl,
      ...(provider === "clickhouse" ? { provider } : {}),
      ...(key && ch ? { clickhouse_org_id: ch.orgId, clickhouse_key_id: key.keyId, clickhouse_key_secret: key.keySecret } : {}),
      ...(opts.coupon ? { promo_code: opts.coupon } : {}),
      // The price shown was accepted: without it the platform creates no billed box.
      ...(accepted ? { accept_price: true } : {}),
    }).catch(async (err) => {
      // A 4xx is a refusal; after anything else (5xx, no answer) a box may be starting with this URL.
      if (err instanceof HttpStatusError && err.status >= 400 && err.status < 500) await undo();
      // 402: the payment method went away since the quote; nothing was created.
      if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
      // 412: billed although quoted free (the last free slot went meanwhile); nothing was created.
      if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
      // 409 "none is stored" for a URL without a password: the platform no longer keeps the server's
      // (its last box went meanwhile). Its other 409s (a name taken, say) say what they are.
      if (err instanceof HttpStatusError && err.status === 409 && prepared.storedPassword && /none is stored/.test(err.message)) return "stored" as const;
      throw err;
    });
    if (created === "payment") return { status: "action_required", provider, name, ...billingPage! };
    // A card on file that Stripe declined is not a missing one.
    if (created === "declined") {
      return { status: "action_required", provider, name, ...billingPage!, next: billingPage!.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
    }
    if (created === "price") return { status: "action_required", provider, name, ...billing, next: "The price changed since it was shown: re-run pgai connect to see it" };
    if (created === "stored") {
      // No box of this org on the server keeps it now; what else logs in as the role is not known here.
      const gone = `PostgresAI no longer keeps the password of ${DEFAULT_MONITORING_USER} for this server`;
      return { status: "action_required", provider, name, ...billing, next: opts.agent
        ? `${gone}: pass its URL as database_url, or run pgai connect <admin-url> --reset-password in a terminal`
        : `${gone}: re-run with --reset-password (anything else that logs in as ${DEFAULT_MONITORING_USER} then needs the new one), or set PGAI_MON_PASSWORD to its password` };
    }
    if (created.status === "failed") {
      await undo();
      return { status: "failed", provider, name, id: created.id, next: `${created.error} Re-run pgai connect later.` };
    }
    row = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: !!ch };
    undoRole = undo;
    if (prepared.storedPassword) {
      // The password stays with the platform, and the checkup never runs as the admin: its SQL resolves
      // names with the database's search_path, where the database's owner can put functions. (The
      // prepare step's plan runs as the admin with that search_path too: postgres-ai/internal#384.)
      progress("checkup", `Express checkup skipped: it runs only as ${DEFAULT_MONITORING_USER}, whose password PostgresAI keeps for this server. The full checkup follows on the box.`);
    } else {
      // First value while the box starts (minutes): the express checkup, as the monitoring role.
      checkup = await deps.checkup(withLocalTls(prepared.monitoringUrl, url), created.name).catch((err) => ({ error: errorText(err) }));
      progress("checkup", checkupLines(checkup).join("\n"), { checkup });
    }
  }

  // The box's state, each time it changes while connect waits for it (not on a re-run of a connected database).
  let shown: string | undefined;
  for (const deadline = deps.now() + opts.waitMs; ;) {
    if (undoRole && row.billing_error && disconnecting(row.status)) {
      await undoRole();
      return { status: "action_required", provider, name, id: row.id, ...billing, ...(checkup ? { checkup } : {}),
        next: `The first charge failed (${row.billing_error}): the box was removed and nothing is billed. Update the payment method at ${billingUrl}, then re-run` };
    }
    const result = connectStatus(row, provider, fresh);
    const state = boxState(row);
    if (state !== shown && (shown !== undefined || result.status === "provisioning")) progress("box", `Monitoring box: ${state}`, { state });
    shown = state;
    if (result.status !== "provisioning" || deps.now() >= deadline) {
      if (result.status === "connected" && provider === "clickhouse" && !row.host_metrics) {
        result.next += "; for CPU, memory and disk, disconnect and reconnect with --clickhouse-key <key-id>:<key-secret>";
      }
      const { next, ...rest } = result;
      return { ...rest, ...billing, ...(checkup ? { checkup } : {}), next: next + note };
    }
    await deps.sleep(POLL_MS);
    // The box is already requested: a failed poll keeps the last known state.
    const listed = await deps.list().catch(() => undefined);
    const now = listed?.find((d) => d.id === row!.id);
    if (listed && !now && undoRole) {
      await undoRole();
      return { status: "failed", provider, name, id: row.id, ...billing, next: "The monitoring box was removed before it became active: see pgai databases, then re-run pgai connect" };
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
  const conn = resolveAdminConnection({ conn: url });
  const fallback = !opts.agent && !!conn.sslFallbackEnabled;
  const { client, usedSsl } = await connectWithSslFallback(opts.Client ?? Client, { ...conn, sslFallbackEnabled: fallback });
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
    if (/client password must be a string/.test(String(err?.message))) throw new Error("The URL has no password (and PGPASSWORD is not set): pgai connect postgresql://user:password@host:5432/dbname");
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
    const unchecked = (what: string) => `this server accepts any password from this host, so ${what} was not checked; if no data arrives, disconnect and connect again with the right password`;
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
      const setPassword = `Set PGAI_MON_PASSWORD to the password of ${DEFAULT_MONITORING_USER}, or give it a new one: pgai connect <admin-url> --reset-password (anything else that logs in as ${DEFAULT_MONITORING_USER} then needs the new password)`;
      // The platform fills in the password it keeps for the box (it never sends it here), over TLS only:
      // what the URL lacks for that, and whether the server takes TLS at all (it refused it, or has ssl off).
      const needs = storedPasswordNeeds(monitoringUrlFor(url, me.db, ""));
      const noTls = refusedTls || (!usedSsl && me.ssl === "off");
      const tlsOnly = `PostgresAI sends the password it keeps for ${DEFAULT_MONITORING_USER} only over TLS`;
      const others = opts.others?.length ? opts.others : undefined;
      // Nobody has the password (the first connect generated it): only a new one, once nothing uses the old one.
      const disconnectOthers = others ? disconnectEach(others) : "pgai disconnect the server's other databases (pgai databases)";
      const reconnectOthers = `(and connect ${others ? others.join(", ") : "them"} again with it)`;
      const noTlsWayOut = opts.agent
        ? `Pass its URL as database_url, or run pgai connect in a terminal with PGAI_MON_PASSWORD set to its password. If nobody has it, in a terminal: ${disconnectOthers}, then pgai connect <admin-url> --reset-password with PGAI_MON_PASSWORD set to a new one ${reconnectOthers}`
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
      if (me.mon_exists && opts.agent) return { next: `${exists}. Pass its URL as database_url, or run pgai connect in a terminal with PGAI_MON_PASSWORD set to its password` };
      // A PGAI_MON_PASSWORD that does not log in, where the platform keeps the password: without it, that one
      // is used (over TLS only).
      const unset = !opts.storedPassword ? setPassword
        : noTls ? `This server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}`
        : !needs ? `Unset PGAI_MON_PASSWORD: PostgresAI keeps the password of ${DEFAULT_MONITORING_USER} for this server`
        : `Unset PGAI_MON_PASSWORD, and the URL needs ${needs}: PostgresAI keeps the password of ${DEFAULT_MONITORING_USER} for this server, and sends it only over TLS, to a URL with each parameter once`;
      const { password, generated } = await resolveMonitoringPassword({ passwordEnv: opts.agent ? undefined : process.env.PGAI_MON_PASSWORD, monitoringUser: DEFAULT_MONITORING_USER });
      const reset = !!opts.resetPassword && me.mon_exists;
      if (me.mon_exists && !reset) {
        if (!process.env.PGAI_MON_PASSWORD?.trim()) return { next: `${exists}. ${setPassword}` };
        let accepted: boolean;
        try {
          accepted = await logsIn(loginUrlFor(url, me.db, password));
        } catch (err) {
          // pg_hba for this client, say: neither accepted nor rejected. With the platform's password, no login is needed.
          const orUnset = opts.storedPassword && !noTls ? ", or unset PGAI_MON_PASSWORD: PostgresAI keeps its password for this server" : "";
          return { next: `${exists}, and PGAI_MON_PASSWORD could not be checked from this host (${err instanceof Error ? err.message : String(err)}). Run pgai connect from a host that ${DEFAULT_MONITORING_USER} may connect from${orUnset}` };
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
      next: `Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai connect again with the ${DEFAULT_MONITORING_USER} URL`,
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

/** What a disconnect did to the billing (monitoring_instance_delete's reply), for a person; undefined when nothing was released. */
export function disconnectBilling(reply: unknown): string | undefined {
  const r = (reply ?? {}) as { billing?: { subscription?: string; quantity?: number }; billing_warning?: string };
  if (r.billing_warning) return `not released (${r.billing_warning}): contact support`;
  if (r.billing?.subscription === "canceled") return "subscription canceled: no further charges; the unused part of this period is credited (prorated)";
  if (r.billing?.subscription === "active" && typeof r.billing.quantity === "number") {
    return `${r.billing.quantity} ${r.billing.quantity === 1 ? "database cluster" : "database clusters"} left on the subscription`;
  }
  return undefined;
}

/** The platform side of connect, for the CLI and the MCP server alike. */
export function platformDeps(p: { apiKey: string; apiBaseUrl: string; uiBaseUrl: string; orgScope?: OrgScope; debug?: boolean; agent?: boolean }) {
  const rpc = <T>(fn: string, body: Record<string, unknown> = {}) =>
    callRpc<T>({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn, body, operation: fn.replace(/_/g, " "), debug: p.debug, orgScope: p.orgScope });
  return {
    list: () => rpc<Database[]>("cloud_monitoring_list"),
    create: (body: Record<string, string | boolean>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_connect", body),
    disconnect: (id: string) => rpc("cloud_monitoring_disconnect", { instance_id: id }),
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
