import { Client } from "pg";
import { findService } from "./clickhouse";
import {
  applyInitPlan, buildInitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER,
  maskConnectionString, redactPasswordsInSql, resolveAdminConnection, resolveMonitoringPassword, verifyInitSetup,
} from "./init";
import { callRpc } from "./joe";
import { listOrgs, type OrgScope } from "./org-scope";
import { requestTimeoutSignal } from "./util";

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
  mode: string;
  status: string | null;
  dashboard_url: string | null;
  host_metrics: boolean;
}

export interface ConnectDeps {
  list(): Promise<Database[]>;
  create(body: Record<string, string>): Promise<{ id: string; name: string; status: string; error?: string }>;
  prepare(url: string, provider: Provider): Promise<{ monitoringUrl: string } | { next: string; sql?: string }>;
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
}

// The reporter's first run is 30 minutes after the stack starts
// (REPORTER_INITIAL_DELAY_SECONDS in docker-compose.yml).
const FIRST_CHECKUP_DELAY_MS = 30 * 60_000;
const POLL_MS = 15_000;
const PROVIDERS: Provider[] = ["clickhouse", "rds", "supabase", "self-managed"];

export function detectCloudProvider(url: string): Provider {
  const host = new URL(url).hostname.toLowerCase().replace(/\.$/, "");
  if (host.endsWith(".clickhouse.cloud")) return "clickhouse";
  if (host.endsWith(".rds.amazonaws.com")) return "rds";
  if (host.endsWith(".supabase.co") || host.endsWith(".pooler.supabase.com")) return "supabase";
  return "self-managed";
}

/** The instance name the platform derives from the URL (monitoring_instance_create, #712). */
export function databaseName(url: string): string {
  const u = new URL(url);
  const port = u.port && u.port !== "5432" ? `:${u.port}` : "";
  const db = decodeURIComponent(u.pathname.replace(/^\//, ""));
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
  if (/delet/.test(row.status ?? "")) return { ...base, status: "disconnecting", next: "none" };
  if (/fail|error/.test(row.status ?? "")) {
    return { ...base, status: "failed", next: `pgai disconnect ${row.name} --yes, then pgai connect again` };
  }
  return { ...base, status: "provisioning", next: `pgai status ${row.name}` };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
  const provider = (opts.provider ?? detectCloudProvider(url)) as Provider;
  if (!PROVIDERS.includes(provider)) throw new Error(`--provider must be one of: ${PROVIDERS.join(", ")}`);
  const name = databaseName(url);
  const key = parseClickhouseKey(opts.clickhouseKey, process.env);
  if (key && provider !== "clickhouse") throw new Error("--clickhouse-key applies to ClickHouse Managed Postgres only");

  if (!opts.selfHosted && (provider === "rds" || provider === "supabase")) {
    return { status: "action_required", provider, name, next: `Finish in the console: ${await deps.handoffUrl(provider)}` };
  }

  if (opts.selfHosted && deps.localStackRunning()) {
    return { status: "action_required", provider, name, next: "A monitoring stack already runs on this machine: add the database with pgai mon targets add '<postgres_ai_mon URL>'" };
  }

  let row = opts.selfHosted ? undefined : (await deps.list()).find((d) => d.name === name && !/delet/.test(d.status ?? ""));
  const fresh = !row;
  if (!row) {
    deps.progress(`Preparing ${maskConnectionString(url)}`);
    const prepared = await deps.prepare(url, provider);
    if ("next" in prepared) return { status: "action_required", provider, name, ...prepared };
    let ch: { orgId: string } | undefined;
    if (key) {
      const found = await deps.clickhouseOrg(new URL(url).hostname, key.keyId, key.keySecret);
      if (found.state !== "running") {
        return { status: "action_required", provider, name, next: `Start the service in the ClickHouse Cloud console (it is ${found.state}), then re-run` };
      }
      ch = found;
    }
    if (opts.selfHosted) {
      await deps.selfHosted(prepared.monitoringUrl, key && ch
        ? { CLICKHOUSE_ORG_ID: ch.orgId, CLICKHOUSE_KEY_ID: key.keyId, CLICKHOUSE_KEY_SECRET: key.keySecret }
        : {});
      return { status: "connected", provider, name, dashboard_url: "http://localhost:3000", host_metrics: !!ch, next: "pgai mon health" };
    }
    deps.progress(`Provisioning monitoring for ${name}`);
    const created = await deps.create({
      db_url: prepared.monitoringUrl,
      ...(provider === "clickhouse" ? { provider } : {}),
      ...(key && ch ? { clickhouse_org_id: ch.orgId, clickhouse_key_id: key.keyId, clickhouse_key_secret: key.keySecret } : {}),
    });
    if (created.status === "failed") return { status: "failed", provider, name, id: created.id, next: `${created.error} Re-run pgai connect later.` };
    row = { id: created.id, name: created.name, provider, mode: "cloud", status: created.status, dashboard_url: null, host_metrics: !!ch };
  }

  for (const deadline = Date.now() + opts.waitMs; ;) {
    const result = connectStatus(row, provider, fresh);
    if (result.status !== "provisioning" || Date.now() >= deadline) {
      if (result.status === "connected" && provider === "clickhouse" && !row.host_metrics) {
        result.next += "; for CPU, memory and disk, disconnect and reconnect with --clickhouse-key <key-id>:<key-secret>";
      }
      return result;
    }
    deps.progress(`Waiting for the monitoring box (${row.status ?? "starting"})`);
    await deps.sleep(POLL_MS);
    row = (await deps.list()).find((d) => d.id === row!.id) ?? row;
  }
}

/** Creates (or checks) the monitoring role over the given URL. The admin URL is used for this run only. */
export async function prepareDatabase(url: string, provider: Provider): Promise<{ monitoringUrl: string } | { next: string; sql?: string }> {
  const pgProvider = provider === "rds" ? "self-managed" : provider;
  const { client } = await connectWithSslFallback(Client, resolveAdminConnection({ conn: url }));
  try {
    const me = (await client.query(
      `select current_user as name, current_database() as db, rolsuper or rolcreaterole as admin,
         exists (select 1 from pg_roles where rolname = '${DEFAULT_MONITORING_USER}') as mon_exists
       from pg_roles where rolname = current_user`,
    )).rows[0];
    if (me.name === DEFAULT_MONITORING_USER) {
      const v = await verifyInitSetup({ client, database: me.db, monitoringUser: me.name, includeOptionalPermissions: false, provider: pgProvider });
      if (v.ok) return { monitoringUrl: url };
    } else if (me.admin && pgProvider !== "supabase") {
      // The role is cluster-wide: another database here may use its password.
      if (me.mon_exists && !process.env.PGAI_MON_PASSWORD) {
        return { next: `${DEFAULT_MONITORING_USER} already exists on this server; re-run with PGAI_MON_PASSWORD=<its password>, or connect with its URL` };
      }
      const { password } = await resolveMonitoringPassword({ passwordEnv: process.env.PGAI_MON_PASSWORD, monitoringUser: DEFAULT_MONITORING_USER });
      await applyInitPlan({ client, plan: await buildInitPlan({ database: me.db, monitoringPassword: password, includeOptionalPermissions: true, provider: pgProvider }) });
      const u = new URL(url);
      u.username = DEFAULT_MONITORING_USER;
      u.password = password;
      u.pathname = `/${encodeURIComponent(me.db)}`;
      u.searchParams.delete("user");
      u.searchParams.delete("password");
      return { monitoringUrl: u.toString() };
    }
    const plan = await buildInitPlan({ database: me.db, monitoringPassword: "<password>", includeOptionalPermissions: true, provider: pgProvider });
    return {
      sql: plan.steps.map((s) => `-- ${s.name}\n${redactPasswordsInSql(s.sql)}`).join("\n\n"),
      next: `Run the SQL above as an admin (or re-run with an admin URL), then pgai connect again with the ${DEFAULT_MONITORING_USER} URL`,
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
  let lastError: unknown = new Error("The API key belongs to no ClickHouse Cloud organization.");
  for (const org of ((await response.json()) as { result: { id: string }[] }).result) {
    try {
      const service = await findService({ apiUrl, orgId: org.id, keyId, keySecret, hostname: host });
      return { orgId: org.id, state: service.state };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError;
}

/** The platform side of connect, for the CLI and the MCP server alike. */
export function platformDeps(p: { apiKey: string; apiBaseUrl: string; uiBaseUrl: string; orgScope?: OrgScope; debug?: boolean }) {
  const rpc = <T>(fn: string, body: Record<string, unknown> = {}) =>
    callRpc<T>({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn, body, operation: fn.replace(/_/g, " "), debug: p.debug, orgScope: p.orgScope });
  return {
    list: () => rpc<Database[]>("cloud_monitoring_list"),
    create: (body: Record<string, string>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_connect", body),
    disconnect: (id: string) => rpc("cloud_monitoring_disconnect", { instance_id: id }),
    prepare: prepareDatabase,
    clickhouseOrg: clickhouseOrgFor,
    handoffUrl: async (provider: "rds" | "supabase") => {
      const orgs = await listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl });
      const org = orgs.length === 1 ? orgs[0] : orgs.find((o) => o.alias === p.orgScope?.alias || o.org_id === p.orgScope?.id);
      return `${p.uiBaseUrl}/${org?.alias ?? "<org>"}/monitoring/scale/create/${provider}`;
    },
    sleep: (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
  };
}
