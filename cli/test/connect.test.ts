import { describe, expect, test } from "bun:test";
import { handleToolCall } from "../lib/mcp-server";
import { clickhouseOrgFor, connect, connectStatus, databaseName, detectCloudProvider, parseClickhouseKey, type ConnectDeps, type Database } from "../lib/connect";

// `pgai connect` (postgres-ai/internal#354): the step machine, with every
// outside effect faked and recorded. Whole results are compared, so a change
// to what a user or an agent sees shows up here.

const CH = "postgresql://postgres:adminpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const CH_NAME = "abc123.us-east-1.aws.pg.clickhouse.cloud/postgres";
const MON = "postgresql://postgres_ai_mon:genpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const ORG = "11111111-2222-3333-4444-555555555555";
const KEY = "AbCdEf0123456789XyZa:Sec4b1dTestSecret0123456789";

function fake(over: Partial<ConnectDeps> & { rows?: (Database | undefined)[] } = {}) {
  const calls: string[] = [];
  const rows = over.rows ?? [];
  let listed = 0;
  const deps: ConnectDeps = {
    list: async () => { calls.push("list"); const r = rows[Math.min(listed++, rows.length - 1)]; return r ? [r] : []; },
    create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); return { id: "i-1", name: CH_NAME, status: "launch_requested" }; },
    prepare: async (url, provider) => { calls.push(`prepare ${provider}`); return { monitoringUrl: MON }; },
    localStackRunning: () => false,
    clickhouseOrg: async (host, keyId) => { calls.push(`clickhouseOrg ${host} ${keyId}`); return { orgId: ORG, state: "running" }; },
    selfHosted: async (url, env) => { calls.push(`selfHosted ${url} ${JSON.stringify(env)}`); },
    handoffUrl: async (provider) => `https://console.postgres.ai/acme/monitoring/scale/create/${provider}`,
    sleep: async () => { calls.push("sleep"); },
    progress: () => {},
    ...over,
  };
  return { deps, calls };
}

const row = (status: string | null, extra: Partial<Database> = {}): Database => ({
  id: "i-1", name: CH_NAME, provider: "clickhouse", mode: "cloud", status,
  dashboard_url: status === "active" ? "https://abc.pgai.watch" : null, host_metrics: true, ...extra,
});

describe("provider and name", () => {
  test("provider from the host", () => {
    expect([
      CH,
      "postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app",
      "postgresql://u:p@db.xyz.supabase.co:5432/postgres",
      "postgresql://u:p@aws-0-eu-central-1.pooler.supabase.com:6543/postgres",
      "postgresql://u:p@10.0.0.5:5432/app",
    ].map(detectCloudProvider)).toEqual(["clickhouse", "rds", "supabase", "supabase", "self-managed"]);
  });

  test("name is what the platform derives: host, port unless 5432, database", () => {
    expect(databaseName(CH)).toBe(CH_NAME);
    expect(databaseName("postgresql://u:p@db2.example.com:6432/app")).toBe("db2.example.com:6432/app");
  });

  test("ClickHouse key from the flag or the environment", () => {
    expect(parseClickhouseKey(KEY, {})).toEqual({ keyId: "AbCdEf0123456789XyZa", keySecret: "Sec4b1dTestSecret0123456789" });
    expect(parseClickhouseKey(undefined, { CLICKHOUSE_KEY_ID: "a", CLICKHOUSE_KEY_SECRET: "b" })).toEqual({ keyId: "a", keySecret: "b" });
    expect(parseClickhouseKey(undefined, {})).toBeUndefined();
    expect(() => parseClickhouseKey("no-colon", {})).toThrow("--clickhouse-key must be <key-id>:<key-secret>");
  });
});

describe("connect", () => {
  test("ClickHouse with a key: prepare, find the org, provision, wait, dashboard", async () => {
    const { deps, calls } = fake({ rows: [undefined, row("launch_requested"), row("active")] });
    const result = await connect(CH, { clickhouseKey: KEY, waitMs: 60_000 }, deps);
    expect(calls).toEqual([
      "list",
      "prepare clickhouse",
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      `create ${JSON.stringify({ db_url: MON, provider: "clickhouse", clickhouse_org_id: ORG, clickhouse_key_id: "AbCdEf0123456789XyZa", clickhouse_key_secret: "Sec4b1dTestSecret0123456789" })}`,
      "sleep", "list", "sleep", "list",
    ]);
    const { first_checkup_eta, ...rest } = result;
    expect(rest).toEqual({
      status: "connected", provider: "clickhouse", name: CH_NAME, id: "i-1",
      dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch",
    });
    expect(Date.parse(first_checkup_eta!) - Date.now()).toBeGreaterThan(29 * 60_000);
    expect(JSON.stringify(result)).not.toContain("adminpw");
    expect(JSON.stringify(result)).not.toContain("Sec4b1d");
  });

  test("already connected: nothing is prepared or provisioned again", async () => {
    const { deps, calls } = fake({ rows: [row("active")] });
    const result = await connect(CH, { waitMs: 60_000 }, deps);
    expect(calls).toEqual(["list"]);
    expect(result).toEqual({
      status: "connected", provider: "clickhouse", name: CH_NAME, id: "i-1",
      dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch",
    });
  });

  test("a row still being deleted (a disconnect in flight) is not reused", async () => {
    const { deps, calls } = fake({ rows: [row("deleting_launched"), row("launch_requested")] });
    await connect(CH, { waitMs: 0 }, deps);
    expect(calls.slice(0, 2)).toEqual(["list", "prepare clickhouse"]);
  });

  test("pgai status shows a disconnect in flight as disconnecting, not provisioning", () => {
    expect(connectStatus(row("deleting_launched"))).toEqual({
      status: "disconnecting", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: true, next: "none",
    });
  });

  test("RDS and Supabase hand off to the console flow, touching nothing", async () => {
    const { deps, calls } = fake();
    expect(await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app", { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "rds", name: "db.abc.us-east-1.rds.amazonaws.com/app",
      next: "Finish in the console: https://console.postgres.ai/acme/monitoring/scale/create/rds",
    });
    expect(calls).toEqual([]);
  });

  test("a URL that cannot create the role: the SQL and the next step, nothing provisioned", async () => {
    const { deps, calls } = fake({ prepare: async () => ({ sql: "-- 01.role\ncreate role ...", next: "Run the SQL" }) });
    expect(await connect(CH, { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, sql: "-- 01.role\ncreate role ...", next: "Run the SQL",
    });
    expect(calls).toEqual(["list"]);
  });

  test("a stopped ClickHouse service: start it first, nothing provisioned", async () => {
    const { deps, calls } = fake({ clickhouseOrg: async () => ({ orgId: ORG, state: "stopped" }) });
    const result = await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps);
    expect(result).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "Start the service in the ClickHouse Cloud console (it is stopped), then re-run",
    });
    expect(calls.some((c) => c.startsWith("create"))).toBe(false);
  });

  test("the platform could not launch the box", async () => {
    const { deps } = fake({ create: async () => ({ id: "i-9", name: CH_NAME, status: "failed", error: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later." }) });
    expect(await connect(CH, { waitMs: 60_000 }, deps)).toEqual({
      status: "failed", provider: "clickhouse", name: CH_NAME, id: "i-9",
      next: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later. Re-run pgai connect later.",
    });
  });

  test("the box reports a failed deploy", async () => {
    const { deps } = fake({ rows: [row("failed")] });
    expect((await connect(CH, { waitMs: 60_000 }, deps)).status).toBe("failed");
  });

  test("--wait 0: provisioning, with how to check", async () => {
    const { deps, calls } = fake();
    const result = await connect(CH, { waitMs: 0 }, deps);
    expect(result).toEqual({
      status: "provisioning", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null,
      host_metrics: false, next: `pgai status ${CH_NAME}`,
    });
    expect(calls).not.toContain("sleep");
  });

  test("ClickHouse without a key: connected, and told how to get host metrics", async () => {
    const { deps } = fake({ rows: [row("active", { host_metrics: false })] });
    expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe(
      "Open https://abc.pgai.watch; for CPU, memory and disk, disconnect and reconnect with --clickhouse-key <key-id>:<key-secret>");
  });

  test("--self-hosted: the local stack gets the monitoring URL and the key as env", async () => {
    const { deps, calls } = fake();
    const result = await connect(CH, { clickhouseKey: KEY, selfHosted: true, waitMs: 0 }, deps);
    expect(calls).toEqual([
      "prepare clickhouse",
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      `selfHosted ${MON} ${JSON.stringify({ CLICKHOUSE_ORG_ID: ORG, CLICKHOUSE_KEY_ID: "AbCdEf0123456789XyZa", CLICKHOUSE_KEY_SECRET: "Sec4b1dTestSecret0123456789" })}`,
    ]);
    expect(result).toEqual({ status: "connected", provider: "clickhouse", name: CH_NAME, dashboard_url: "http://localhost:3000", host_metrics: true, next: "pgai mon health" });
  });

  test("--self-hosted with a stack already running here: add the database to it, nothing prepared", async () => {
    const { deps, calls } = fake({ localStackRunning: () => true });
    expect(await connect(CH, { selfHosted: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "A monitoring stack already runs on this machine: add the database with pgai mon targets add '<postgres_ai_mon URL>'",
    });
    expect(calls).toEqual([]);
  });

  test("a ClickHouse key on another provider, or an unknown provider, is an error", async () => {
    const { deps } = fake();
    await expect(connect("postgresql://u:p@10.0.0.5:5432/app", { clickhouseKey: KEY, waitMs: 0 }, deps)).rejects.toThrow("--clickhouse-key applies to ClickHouse Managed Postgres only");
    await expect(connect(CH, { provider: "oracle", waitMs: 0 }, deps)).rejects.toThrow("--provider must be one of");
  });
});

describe("clickhouseOrgFor (a fake ClickHouse Cloud API)", () => {
  const OTHER = "99999999-8888-7777-6666-555555555555";
  const SERVICE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
  async function withApi(orgsStatus: number, fn: () => Promise<void>) {
    const seen: string[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(req) {
        const path = new URL(req.url).pathname;
        seen.push(`${req.headers.get("authorization")?.slice(0, 6)} ${path}`);
        const json = (result: unknown) => Response.json({ result });
        if (path === "/v1/organizations") return orgsStatus === 200 ? json([{ id: OTHER }, { id: ORG }]) : new Response("", { status: orgsStatus });
        if (path === `/v1/organizations/${OTHER}/postgres`) return json([]);
        if (path === `/v1/organizations/${ORG}/postgres`) return json([{ id: SERVICE, name: "svc", state: "running" }]);
        if (path === `/v1/organizations/${ORG}/postgres/${SERVICE}`) return json({ id: SERVICE, name: "svc", state: "running", hostname: "abc123.us-east-1.aws.pg.clickhouse.cloud" });
        return new Response("", { status: 404 });
      },
    });
    process.env.CLICKHOUSE_API_URL = `http://127.0.0.1:${server.port}`;
    try {
      await fn();
    } finally {
      delete process.env.CLICKHOUSE_API_URL;
      server.stop(true);
    }
    return seen;
  }

  test("the key's organization that runs the host, with Basic auth", async () => {
    const seen = await withApi(200, async () => {
      expect(await clickhouseOrgFor("abc123.us-east-1.aws.pg.clickhouse.cloud", "kid", "secret")).toEqual({ orgId: ORG, state: "running" });
    });
    expect(seen[0]).toBe("Basic  /v1/organizations");
  });

  test("a rejected key and an unknown host are clear errors", async () => {
    await withApi(401, async () => {
      await expect(clickhouseOrgFor("x.pg.clickhouse.cloud", "kid", "bad")).rejects.toThrow("ClickHouse Cloud rejected the API key (401)");
    });
    await withApi(200, async () => {
      await expect(clickhouseOrgFor("nope.pg.clickhouse.cloud", "kid", "secret")).rejects.toThrow("has hostname nope.pg.clickhouse.cloud");
    });
  });
});

describe("MCP connect_database", () => {
  test("same JSON as the CLI; RDS hands off to the console with the org's alias", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([row("active")]);
        if (path.endsWith("/rpc/orgs_list")) return Response.json([{ org_id: 7, alias: "acme", name: "Acme", is_active: true }]);
        return new Response("not found", { status: 404 });
      },
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    const call = async (args: Record<string, unknown>) =>
      JSON.parse((await handleToolCall({ params: { name: "connect_database", arguments: args } }, opts)).content[0].text);
    try {
      expect(await call({ database_url: CH })).toEqual({
        status: "connected", provider: "clickhouse", name: CH_NAME, id: "i-1",
        dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch",
      });
      expect((await call({ database_url: "postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app" })).next)
        .toBe("Finish in the console: https://console.example/acme/monitoring/scale/create/rds");
    } finally {
      server.stop(true);
    }
  });
});

test("debug logs never carry the database URL or the ClickHouse key secret", () => {
  const { redactSecretsForLog } = require("../lib/util");
  const out = redactSecretsForLog(JSON.stringify({ db_url: MON, clickhouse_key_id: "kid", clickhouse_key_secret: "Sec4b1d" }));
  expect(out).not.toContain("genpw");
  expect(out).not.toContain("Sec4b1d");
  expect(out).toContain("kid");
});
