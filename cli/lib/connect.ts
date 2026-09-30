import { Client } from "pg";
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
export type Status = "connected" | "provisioning" | "disconnecting" | "action_required" | "failed";

export interface ConnectResult {
  status: Status;
  provider: Provider;
  name: string;
  id?: string;
  dashboard_url?: string | null;
  host_metrics?: boolean;
  first_checkup_eta?: string;
  next: string;
  sql?: string;
}

/** A row of v1.cloud_monitoring_list. */
export interface Database {
  id: string;
  name: string;
  provider: string;
  status: string | null;
  dashboard_url: string | null;
  host_metrics: boolean;
}

/**
 * The monitoring URL (with a note for the user, if any), or what to do first.
 * `generated`: this run created the role with a password generated here, which
 * nobody has once the run ends.
 */
export type Prepared = { monitoringUrl: string; note?: string; generated?: true } | { next: string; sql?: string };

export interface ConnectDeps {
  list(): Promise<Database[]>;
  create(body: Record<string, string>): Promise<{ id: string; name: string; status: string; error?: string }>;
  prepare(url: string, provider: Provider): Promise<Prepared>;
  /** Drops the role a `generated` prepare created; false when it could not. */
  unprepare(url: string): Promise<boolean>;
  localStackRunning(): boolean;
  clickhouseOrg(host: string, keyId: string, keySecret: string): Promise<{ orgId: string; state: string }>;
  selfHosted(monitoringUrl: string, env: Record<string, string>): Promise<void>;
  handoffUrl(provider: "rds" | "supabase"): Promise<string>;
  sleep(ms: number): Promise<void>;
  progress(line: string): void;
}

export interface ConnectOptions {
  provider?: string;
  clickhouseKey?: string;
  selfHosted?: boolean;
  waitMs: number;
  /** The URL comes from an agent (the MCP tool): nothing of this process's environment or files goes to the host it names. */
  agent?: boolean;
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
  if (row.status === "active") {
    return {
      ...base, status: "connected",
      ...(fresh ? { first_checkup_eta: new Date(Date.now() + FIRST_CHECKUP_DELAY_MS).toISOString() } : {}),
      next: row.dashboard_url ? `Open ${row.dashboard_url}` : `pgai status ${row.name}`,
    };
  }
  if (disconnecting(row.status)) return { ...base, status: "disconnecting", next: "none" };
  if (/fail|error/.test(row.status ?? "")) {
    return { ...base, status: "failed", next: `pgai disconnect ${row.name} --yes, then pgai connect again` };
  }
  return { ...base, status: "provisioning", next: `pgai status ${row.name}` };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
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

  if (opts.selfHosted && deps.localStackRunning()) {
    return { status: "action_required", provider, name, next: "A monitoring stack already runs on this machine: add the database with pgai mon targets add '<postgres_ai_mon URL>'" };
  }

  let row = opts.selfHosted ? undefined : (await deps.list()).find((d) => d.name === name && !disconnecting(d.status));
  const fresh = !row;
  let note = "";
  if (!row) {
    // The key is checked before the database is touched: a rejected key or a stopped service changes nothing.
    let ch: { orgId: string } | undefined;
    if (key) {
      const found = await deps.clickhouseOrg(new URL(url).hostname, key.keyId, key.keySecret);
      if (found.state !== "running") {
        return { status: "action_required", provider, name, next: `Start the service in the ClickHouse Cloud console (it is ${found.state}), then re-run` };
      }
      ch = found;
    }
    deps.progress(`Preparing ${maskConnectionString(url)}`);
    const prepared = await deps.prepare(url, provider);
    if ("next" in prepared) return { status: "action_required", provider, name, ...prepared };
    if (prepared.note) note = `; ${prepared.note}`;
    if (opts.selfHosted) {
      await deps.selfHosted(prepared.monitoringUrl, key && ch
        ? { CLICKHOUSE_ORG_ID: ch.orgId, CLICKHOUSE_KEY_ID: key.keyId, CLICKHOUSE_KEY_SECRET: key.keySecret }
        : {});
      return { status: "connected", provider, name, dashboard_url: "http://localhost:3000", host_metrics: !!ch, next: `pgai mon health${note}` };
    }
    deps.progress(`Provisioning monitoring for ${name}`);
    // A refused launch starts no box, so a role with a generated password is
    // dropped again: nobody has that password, and the re-run would stop at it.
    const undo = async () => { if (prepared.generated) await deps.unprepare(url).catch(() => false); };
    const created = await deps.create({
      db_url: prepared.monitoringUrl,
      ...(provider === "clickhouse" ? { provider } : {}),
      ...(key && ch ? { clickhouse_org_id: ch.orgId, clickhouse_key_id: key.keyId, clickhouse_key_secret: key.keySecret } : {}),
    }).catch(async (err) => {
      // A 4xx is a refusal; after anything else (5xx, no answer) a box may be starting with this URL.
      if (err instanceof HttpStatusError && err.status >= 400 && err.status < 500) await undo();
      throw err;
    });
    if (created.status === "failed") {
      await undo();
      return { status: "failed", provider, name, id: created.id, next: `${created.error} Re-run pgai connect later.` };
    }
    row = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: !!ch };
  }

  for (const deadline = Date.now() + opts.waitMs; ;) {
    const result = connectStatus(row, provider, fresh);
    if (result.status !== "provisioning" || Date.now() >= deadline) {
      if (result.status === "connected" && provider === "clickhouse" && !row.host_metrics) {
        result.next += "; for CPU, memory and disk, disconnect and reconnect with --clickhouse-key <key-id>:<key-secret>";
      }
      result.next += note;
      return result;
    }
    deps.progress(`Waiting for the monitoring box (${row.status ?? "starting"})`);
    await deps.sleep(POLL_MS);
    // The box is already requested: a failed poll keeps the last known state.
    row = (await deps.list().catch(() => [] as Database[])).find((d) => d.id === row!.id) ?? row;
  }
}

type PgClientClass = Parameters<typeof connectWithSslFallback>[0];
type PgClient = Awaited<ReturnType<typeof connectWithSslFallback>>["client"];

export interface PrepareOptions {
  /** The URL comes from an agent (the MCP tool): PGAI_MON_PASSWORD is not read and a TLS failure is not retried in plaintext. */
  agent?: boolean;
  /** The pg client class (tests). */
  Client?: PgClientClass;
}

function openConnection(url: string, opts: PrepareOptions) {
  const conn = resolveAdminConnection({ conn: url });
  return connectWithSslFallback(opts.Client ?? Client, opts.agent ? { ...conn, sslFallbackEnabled: false } : conn);
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
  const { client } = await open(url).catch((err) => {
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
         current_setting('scram_iterations', true) as iterations
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
      if (me.mon_exists && opts.agent) return { next: `${exists}. Pass its URL as database_url, or run pgai connect in a terminal with PGAI_MON_PASSWORD set to its password` };
      const setPassword = `Set PGAI_MON_PASSWORD to the password of ${DEFAULT_MONITORING_USER}, or change it explicitly with: pgai prepare-db <admin-url> --reset-password --password <new-password> (then update every monitoring box that uses it)`;
      const { password, generated } = await resolveMonitoringPassword({ passwordEnv: opts.agent ? undefined : process.env.PGAI_MON_PASSWORD, monitoringUser: DEFAULT_MONITORING_USER });
      if (me.mon_exists) {
        if (!process.env.PGAI_MON_PASSWORD?.trim()) return { next: `${exists}. ${setPassword}` };
        let accepted: boolean;
        try {
          accepted = await logsIn(loginUrlFor(url, me.db, password));
        } catch (err) {
          // pg_hba for this client, say: neither accepted nor rejected.
          return { next: `${exists}, and PGAI_MON_PASSWORD could not be checked from this host (${err instanceof Error ? err.message : String(err)}). Run pgai connect from a host that ${DEFAULT_MONITORING_USER} may connect from` };
        }
        if (!accepted) return { next: `${exists} and PGAI_MON_PASSWORD is not its password. ${setPassword}` };
        if (await acceptsAnyPassword()) notes.push(unchecked("PGAI_MON_PASSWORD"));
      }
      const monitoringUrl = monitoringUrlFor(url, me.db, password);
      const loginUrl = loginUrlFor(url, me.db, password);
      const plan = await buildInitPlan({ database: me.db, monitoringPassword: password, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
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

/** The ClickHouse Cloud organization that runs the service at `host`, found with the key itself. */
export async function clickhouseOrgFor(host: string, keyId: string, keySecret: string) {
  const apiUrl = process.env.CLICKHOUSE_API_URL || "https://api.clickhouse.cloud";
  const response = await fetch(`${new URL(apiUrl).origin}/v1/organizations`, {
    headers: { Authorization: `Basic ${Buffer.from(`${keyId}:${keySecret}`).toString("base64")}` },
    signal: requestTimeoutSignal().signal,
    redirect: "error",
  });
  if (response.status === 401) throw new Error("ClickHouse Cloud rejected the API key (401). Check the key id and secret.");
  if (!response.ok) throw new Error(`ClickHouse Cloud API request failed (${response.status}).`);
  const orgs = ((await response.json()) as { result?: { id: string }[] }).result;
  if (!Array.isArray(orgs)) throw new Error("ClickHouse Cloud API returned no organization list.");
  // An organization the key cannot read (403) says more than "no such service" in the next one.
  let notFound: unknown = new Error("The API key belongs to no ClickHouse Cloud organization.");
  let failed: unknown;
  for (const org of orgs) {
    try {
      const service = await findService({ apiUrl, orgId: org.id, keyId, keySecret, hostname: host });
      return { orgId: org.id, state: service.state };
    } catch (err) {
      if (/^No ClickHouse Managed Postgres service/.test(err instanceof Error ? err.message : "")) notFound = err;
      else failed ??= err;
    }
  }
  throw failed ?? notFound;
}

/** The platform side of connect, for the CLI and the MCP server alike. */
export function platformDeps(p: { apiKey: string; apiBaseUrl: string; uiBaseUrl: string; orgScope?: OrgScope; debug?: boolean; agent?: boolean }) {
  const rpc = <T>(fn: string, body: Record<string, unknown> = {}) =>
    callRpc<T>({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn, body, operation: fn.replace(/_/g, " "), debug: p.debug, orgScope: p.orgScope });
  return {
    list: () => rpc<Database[]>("cloud_monitoring_list"),
    create: (body: Record<string, string>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_connect", body),
    disconnect: (id: string) => rpc("cloud_monitoring_disconnect", { instance_id: id }),
    prepare: (url: string, provider: Provider) => prepareDatabase(url, provider, { agent: p.agent }),
    unprepare: (url: string) => unprepareDatabase(url, { agent: p.agent }),
    clickhouseOrg: clickhouseOrgFor,
    handoffUrl: async (provider: "rds" | "supabase") => {
      const orgs = await listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl });
      const org = orgs.length === 1 ? orgs[0] : orgs.find((o) => o.alias === p.orgScope?.alias || o.org_id === p.orgScope?.id);
      return `${p.uiBaseUrl}/${org?.alias ?? "<org>"}/monitoring/scale/create/${provider}`;
    },
    sleep: (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
  };
}
