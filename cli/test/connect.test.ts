import { describe, expect, test } from "bun:test";
import { handleToolCall } from "../lib/mcp-server";
import { HttpStatusError } from "../lib/util";
import { resolveAdminConnection } from "../lib/init";
import { disconnectBilling, errorText, priceText, checkupLines, checkUrlParams, ClickhouseKeyError, clickhouseOrgFor, connect, connectStatus, stateOf, databaseName, detectCloudProvider, parseClickhouseKey, prepareDatabase, progressText, saveCheckupReport, type ConnectDeps, type Database, type PrepareOptions, type ProgressEvent } from "../lib/connect";

// `pgai connect` (postgres-ai/internal#354): the step machine, with every
// outside effect faked and recorded. Whole results are compared, so a change
// to what a user or an agent sees shows up here.

const CH = "postgresql://postgres:adminpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const CH_NAME = "abc123.us-east-1.aws.pg.clickhouse.cloud/postgres";
const MON = "postgresql://postgres_ai_mon:genpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const ORG = "11111111-2222-3333-4444-555555555555";
const KEY = "AbCdEf0123456789XyZa:Sec4b1dTestSecret0123456789";
// What the express checkup found, as connect returns it.
const CHECKUP = {
  checks: 4,
  findings: [
    { check_id: "H002", title: "Unused indexes", status: "warning", message: "3 unused indexes (1.20 MiB)" },
    { check_id: "A002", title: "Postgres major version", status: "ok", message: "PostgreSQL 17" },
  ],
  info: ["A003", "A004"],
  report_id: 7,
};

const FREE_QUOTE = {
  plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 1 }, subscription: false, quantity: 0,
  price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false,
};
const FREE = { price: "free (1 of 1 free slots)", requires_payment_method: false };

function fake(over: Partial<ConnectDeps> & { rows?: (Database | undefined)[] } = {}) {
  const calls: string[] = [];
  const rows = over.rows ?? [];
  let listed = 0;
  const deps: ConnectDeps = {
    list: async () => { calls.push("list"); const r = rows[Math.min(listed++, rows.length - 1)]; return r ? [r] : []; },
    create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); return { id: "i-1", name: CH_NAME, status: "launch_requested" }; },
    // The check before the price (nothing changed) is recorded apart from the prepare.
    prepare: async (url, provider, o) => { calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return o?.check ? { checked: true } : { monitoringUrl: MON }; },
    unprepare: async () => { calls.push("unprepare"); return true; },
    localStackRunning: () => false,
    clickhouseOrg: async (host, keyId) => { calls.push(`clickhouseOrg ${host} ${keyId}`); return { orgId: ORG, state: "running" }; },
    checkup: async (url, project) => { calls.push(`checkup ${url} as ${project}`); return CHECKUP; },
    selfHosted: async (url, env) => { calls.push(`selfHosted ${url} ${JSON.stringify(env)}`); },
    handoffUrl: async (provider) => `https://console.postgres.ai/acme/monitoring/scale/create/${provider}`,
    // The billing step is its own describe below; here a box is on a free slot.
    quote: async () => FREE_QUOTE,
    billingUrl: (alias) => `https://console.postgres.ai/${alias}/billing`,
    confirm: async () => false,
    sleep: async () => { calls.push("sleep"); },
    now: () => Date.now(),
    progress: () => {},
    ...over,
  };
  return { deps, calls };
}

const row = (status: string | null, extra: Partial<Database> = {}): Database => ({
  id: "i-1", name: CH_NAME, provider: "clickhouse", status,
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

  test("a URL without a database: the name has the one pg connects to, as the monitoring URL will", () => {
    expect(databaseName("postgresql://postgres:p@db2.example.com:6432")).toBe("db2.example.com:6432/postgres");
    expect(databaseName("postgresql://db2.example.com/?user=app_admin&password=p&sslmode=require")).toBe("db2.example.com/app_admin");
    process.env.PGDATABASE = "orders";
    try {
      expect(databaseName("postgresql://postgres:p@db2.example.com")).toBe("db2.example.com/orders");
    } finally {
      delete process.env.PGDATABASE;
    }
  });

  test("ClickHouse key from the flag or the environment", () => {
    expect(parseClickhouseKey(KEY, {})).toEqual({ keyId: "AbCdEf0123456789XyZa", keySecret: "Sec4b1dTestSecret0123456789" });
    expect(parseClickhouseKey(undefined, { CLICKHOUSE_KEY_ID: "a", CLICKHOUSE_KEY_SECRET: "b" })).toEqual({ keyId: "a", keySecret: "b" });
    expect(parseClickhouseKey(undefined, {})).toBeUndefined();
    expect(() => parseClickhouseKey("no-colon", {})).toThrow("--clickhouse-key must be <key-id>:<key-secret>");
  });
});

describe("connect", () => {
  test("ClickHouse with a key: find the org, prepare, provision, wait, dashboard", async () => {
    const { deps, calls } = fake({ rows: [undefined, row("launch_requested"), row("active")] });
    const result = await connect(CH, { clickhouseKey: KEY, waitMs: 60_000 }, deps);
    expect(calls).toEqual([
      "list",
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      "check clickhouse",
      "prepare clickhouse",
      `create ${JSON.stringify({ db_url: MON, provider: "clickhouse", clickhouse_org_id: ORG, clickhouse_key_id: "AbCdEf0123456789XyZa", clickhouse_key_secret: "Sec4b1dTestSecret0123456789" })}`,
      // First value while the box starts: the express checkup, over the monitoring role.
      `checkup ${MON} as ${CH_NAME}`,
      "sleep", "list", "sleep", "list",
    ]);
    const { first_checkup_eta, ...rest } = result;
    expect(rest).toEqual({
      status: "connected", provider: "clickhouse", name: CH_NAME, id: "i-1",
      dashboard_url: "https://abc.pgai.watch", host_metrics: true, checkup: CHECKUP, ...FREE, next: "Open https://abc.pgai.watch",
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
    expect(calls.slice(0, 3)).toEqual(["list", "check clickhouse", "prepare clickhouse"]);
  });

  test("a disconnect that failed to launch is failed (and can be retried), not disconnecting", () => {
    expect(connectStatus(row("deleting_failed_to_launch")).status).toBe("failed");
  });

  test("pgai status shows a disconnect in flight as disconnecting, not provisioning", () => {
    expect(connectStatus(row("deleting_launched"))).toEqual({
      status: "disconnecting", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: true, next: "none",
    });
  });

  test("one state vocabulary for connect, status and databases: the platform's states mapped", () => {
    const raw = [null, "launch_requested", "registered", "active", "failed_to_launch", "failed", "deleting_launched", "deleting_failed_to_launch", "deleted"];
    expect(Object.fromEntries(raw.map((r) => [String(r), stateOf(r)]))).toEqual({
      null: "provisioning", launch_requested: "provisioning", registered: "provisioning", active: "connected",
      failed_to_launch: "failed", failed: "failed", deleting_launched: "disconnecting", deleting_failed_to_launch: "failed", deleted: "disconnected",
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

  test("a stopped ClickHouse service: start it first, the database is not touched", async () => {
    const { deps, calls } = fake({ clickhouseOrg: async () => ({ orgId: ORG, state: "stopped" }) });
    const result = await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps);
    expect(result).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "Start the service in the ClickHouse Cloud console (it is stopped), then re-run",
    });
    expect(calls).toEqual(["list"]);
  });

  const REJECTED = "ClickHouse Cloud rejected the API key (401). Check the key id and secret.";

  test("a rejected ClickHouse key: action required (exit 3) before the database is touched", async () => {
    const { deps, calls } = fake({ clickhouseOrg: async () => { throw new ClickhouseKeyError(REJECTED); } });
    expect(await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, next: `${REJECTED} Then re-run`,
    });
    expect(calls).toEqual(["list"]);
  });

  test("a re-run with a key checks the key: a rejected one is action required, not connected", async () => {
    const { deps, calls } = fake({ rows: [row("active")], clickhouseOrg: async (host, keyId) => { calls.push(`clickhouseOrg ${host} ${keyId}`); throw new ClickhouseKeyError(REJECTED); } });
    expect(await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, id: "i-1",
      next: `${REJECTED} Nothing was changed: re-run with the right key, or without one`,
    });
    expect(calls).toEqual(["list", "clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa"]);
  });

  test("a re-run with a good key: connected, nothing prepared again", async () => {
    const { deps, calls } = fake({ rows: [row("active")] });
    expect((await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps)).status).toBe("connected");
    expect(calls).toEqual(["list", "clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa"]);
  });

  test("the ClickHouse API failing (not the key) is a failure, exit 1", async () => {
    const { deps } = fake({ rows: [row("active")], clickhouseOrg: async () => { throw new Error("ClickHouse Cloud API request failed (503)."); } });
    await expect(connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps)).rejects.toThrow("(503)");
  });

  const REFUSED = { id: "i-9", name: CH_NAME, status: "failed", error: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later." };

  test("the platform could not launch the box", async () => {
    const { deps, calls } = fake({ create: async () => REFUSED });
    expect(await connect(CH, { waitMs: 60_000 }, deps)).toEqual({
      status: "failed", provider: "clickhouse", name: CH_NAME, id: "i-9",
      next: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later. Re-run pgai connect later.",
    });
    // The role was there before, or has the user's PGAI_MON_PASSWORD: it stays.
    expect(calls).toEqual(["list", "check clickhouse", "prepare clickhouse"]);
  });

  // The role this run created with a generated password: nobody has the
  // password, so the re-run would stop at "postgres_ai_mon already exists".
  describe("a launch that fails after the role was created with a generated password", () => {
    const generated: Partial<ConnectDeps> = { prepare: async () => ({ monitoringUrl: MON, generated: true }) };

    test("refused by the platform (in the reply, or a 4xx): the role is dropped again", async () => {
      const inReply = fake({ ...generated, create: async () => REFUSED });
      expect((await connect(CH, { waitMs: 0 }, inReply.deps)).status).toBe("failed");
      expect(inReply.calls).toEqual(["list", "unprepare"]);

      const http = fake({ ...generated, create: async () => { throw new HttpStatusError("Failed to cloud monitoring connect: HTTP 403", 403); } });
      await expect(connect(CH, { waitMs: 0 }, http.deps)).rejects.toThrow("HTTP 403");
      expect(http.calls).toEqual(["list", "unprepare"]);
    });

    test("a drop that fails does not hide the refusal", async () => {
      const { deps } = fake({ ...generated, create: async () => REFUSED, unprepare: async () => { throw new Error("connection refused"); } });
      expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe(`${REFUSED.error} Re-run pgai connect later.`);
    });

    test("a 5xx or no answer: a box may be starting with this URL, so the role stays", async () => {
      for (const err of [new HttpStatusError("HTTP 502", 502), new Error("timed out")]) {
        const { deps, calls } = fake({ ...generated, create: async () => { throw err; } });
        await expect(connect(CH, { waitMs: 0 }, deps)).rejects.toThrow(err.message);
        expect(calls).toEqual(["list"]);
      }
    });

    test("a launch that starts: the role stays", async () => {
      const { deps, calls } = fake({ ...generated });
      expect((await connect(CH, { waitMs: 0 }, deps)).status).toBe("provisioning");
      expect(calls).not.toContain("unprepare");
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
      host_metrics: false, checkup: CHECKUP, ...FREE, next: `pgai status ${CH_NAME}`,
    });
    expect(calls).not.toContain("sleep");
  });

  test("progress: each step and each change of the box's state once, with the time since the start", async () => {
    const events: ProgressEvent[] = [];
    let t = 0;
    const registered = row("launch_requested", { registered_at: "2026-10-02T00:17:39Z" });
    const { deps } = fake({
      // The launch, then 13 polls of launch_requested, registered, then active.
      rows: [undefined, ...Array(13).fill(row("launch_requested")), registered, registered, row("active")],
      now: () => t,
      checkup: async () => { t += 14_000; return CHECKUP; },
      sleep: async (ms) => { t += ms; },
      progress: (e) => events.push(e),
    });
    expect((await connect(CH, { waitMs: 20 * 60_000 }, deps)).status).toBe("connected");
    expect(events.map(progressText)).toEqual([
      "Billing: free (1 of 1 free slots) (+0s)",
      "Preparing postgresql://postgres:*****@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require (+0s)",
      `Provisioning monitoring for ${CH_NAME} (+0s)`,
      [
        "Express checkup while the box starts (4 checks: 1 warning, 1 ok, 2 info) (+14s):",
        "  H002 Unused indexes: 3 unused indexes (1.20 MiB)",
        "  ok: A002",
        "  info: A003 A004",
        "Saved as report 7: pgai reports files 7",
        "The full checkup (query analysis and trends) follows on the box.",
      ].join("\n"),
      "Monitoring box: launch_requested (+14s)",
      "Monitoring box: registered (+3m44s)",
      "Monitoring box: active (+4m14s)",
    ]);
    // For an agent the same steps, as events.
    expect(events.map(({ message, checkup, ...e }) => e)).toEqual([
      { event: "billing", elapsed_s: 0 },
      { event: "preparing", elapsed_s: 0 },
      { event: "provisioning", elapsed_s: 0 },
      { event: "checkup", elapsed_s: 14 },
      { event: "box", elapsed_s: 14, state: "launch_requested" },
      { event: "box", elapsed_s: 224, state: "registered" },
      { event: "box", elapsed_s: 254, state: "active" },
    ]);
    expect(events[3].checkup).toEqual(CHECKUP);
  });

  test("a re-run of a connected database shows no progress", async () => {
    const events: ProgressEvent[] = [];
    const { deps } = fake({ rows: [row("active")], progress: (e) => events.push(e) });
    expect((await connect(CH, { waitMs: 60_000 }, deps)).status).toBe("connected");
    expect(events).toEqual([]);
  });

  test("the express summary adds up: warnings, ok, info and what could not run are all counted and named", () => {
    expect(checkupLines({ ...CHECKUP, checks: 6, failed: ["F004", "I001"] })).toEqual([
      "Express checkup while the box starts (6 checks: 1 warning, 1 ok, 2 info, 2 could not run):",
      "  H002 Unused indexes: 3 unused indexes (1.20 MiB)",
      "  ok: A002",
      "  info: A003 A004",
      "  could not run: F004 I001",
      "Saved as report 7: pgai reports files 7",
      "The full checkup (query analysis and trends) follows on the box.",
    ]);
  });

  test("an express checkup that could not be saved says why; the summary is still shown", () => {
    const { report_id, ...unsaved } = CHECKUP;
    expect(checkupLines({ ...unsaved, upload_error: "Rate limit exceeded: only 1 report upload(s) allowed per 10 minutes." }).slice(-2)).toEqual([
      "Not saved to PostgresAI: Rate limit exceeded: only 1 report upload(s) allowed per 10 minutes.",
      "The full checkup (query analysis and trends) follows on the box.",
    ]);
  });

  test("a failed express checkup does not stop connect: its error is in the result", async () => {
    const { deps } = fake({ checkup: async () => { throw new Error("permission denied for view pg_stat_statements"); } });
    expect((await connect(CH, { waitMs: 0 }, deps)).checkup).toEqual({ error: "permission denied for view pg_stat_statements" });
  });

  test("the express checkup logs in from this machine: the URL's TLS files are kept for it, not sent to the box", async () => {
    const tls = "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=verify-full&sslrootcert=%2Ftmp%2Fca.pem";
    const box = "postgresql://postgres_ai_mon:genpw@db.example.com:5432/app?sslmode=verify-full";
    const { deps, calls } = fake({ prepare: async () => ({ monitoringUrl: box }) });
    await connect(tls, { waitMs: 0 }, deps);
    expect(calls.filter((c) => /^(create|checkup)/.test(c))).toEqual([
      `create ${JSON.stringify({ db_url: box })}`,
      `checkup ${box}&sslrootcert=%2Ftmp%2Fca.pem as ${CH_NAME}`,
    ]);
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
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      "prepare clickhouse",
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

  test("an exported ClickHouse key pair is for ClickHouse only: other providers connect without it", async () => {
    process.env.CLICKHOUSE_KEY_ID = "kid";
    process.env.CLICKHOUSE_KEY_SECRET = "Sec4b1d";
    try {
      const { deps, calls } = fake();
      expect((await connect("postgresql://u:p@10.0.0.5:5432/app", { waitMs: 0 }, deps)).status).toBe("provisioning");
      expect(calls).toEqual(["list", "check self-managed", "prepare self-managed", `create ${JSON.stringify({ db_url: MON })}`, `checkup ${MON} as ${CH_NAME}`]);
      expect((await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app", { waitMs: 0 }, deps)).next)
        .toBe("Finish in the console: https://console.postgres.ai/acme/monitoring/scale/create/rds");
      await connect(CH, { waitMs: 0 }, deps);
      expect(calls.at(-5)).toBe("clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud kid");
    } finally {
      delete process.env.CLICKHOUSE_KEY_ID;
      delete process.env.CLICKHOUSE_KEY_SECRET;
    }
  });

  test("a poll that fails while waiting keeps the requested box: provisioning, then connected", async () => {
    let polls = 0;
    const { deps } = fake({
      list: async () => {
        if (polls++ === 1) throw new Error("HTTP 502");
        return polls === 1 ? [] : [row("active")];
      },
    });
    expect((await connect(CH, { waitMs: 60_000 }, deps)).status).toBe("connected");
    expect(polls).toBe(3);
  });

  test("every poll failing until the deadline: provisioning, from the state the launch returned", async () => {
    let polls = 0;
    const { deps } = fake({
      list: async () => {
        if (polls++ > 0) throw new Error("HTTP 502");
        return [];
      },
      sleep: (ms) => new Promise((r) => setTimeout(r, ms / 1000)),
    });
    expect(await connect(CH, { waitMs: 100 }, deps)).toEqual({
      status: "provisioning", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: false, checkup: CHECKUP, ...FREE, next: `pgai status ${CH_NAME}`,
    });
    expect(polls).toBeGreaterThan(2);
  });

  test("host or port in the query string is refused: the server prepared must be the server named", async () => {
    const { deps, calls } = fake();
    await expect(connect("postgresql://postgres:pw@db.example.invalid:5432/postgres?host=127.0.0.1&port=32785", { waitMs: 0 }, deps))
      .rejects.toThrow("The URL's query string sets host and port: put the host and the port in the URL itself (postgresql://user:password@host:5432/dbname), so that the server prepared is the server monitored");
    await expect(connect("postgresql://postgres:pw@db.example.invalid/postgres?host=%2Fvar%2Frun%2Fpostgresql", { selfHosted: true, waitMs: 0 }, deps)).rejects.toThrow("sets host:");
    expect(calls).toEqual([]);
    // Certificate files and the rest are the user's own to give.
    expect(() => checkUrlParams("postgresql://postgres:pw@db.example.com/postgres?sslmode=verify-full&sslrootcert=/tmp/ca.pem&sslcert=/tmp/c.pem&sslkey=/tmp/k.pem&options=-c%20role%3Dx&connect_timeout=5")).not.toThrow();
  });

  test("an agent's URL: only the query parameters the box gets, and no ClickHouse key from the environment", async () => {
    const { deps, calls } = fake();
    for (const q of ["sslcert=/home/u/.postgresql/postgresql.crt&sslkey=/home/u/.postgresql/postgresql.key", "sslrootcert=/etc/passwd", "host=127.0.0.1", "options=-c%20role%3Dx", "user=postgres"]) {
      await expect(connect(`postgresql://postgres:pw@db.example.invalid:5432/postgres?sslmode=require&${q}`, { waitMs: 0, agent: true }, deps))
        .rejects.toThrow(`database_url may carry only these query parameters: sslmode, channel_binding, application_name (got: ${q.split("&").map((kv) => kv.split("=")[0]).join(", ")})`);
    }
    expect(calls).toEqual([]);
    expect(() => checkUrlParams("postgresql://postgres@db.example.com/postgres?password=pw&sslmode=require&channel_binding=require&application_name=x", true)).not.toThrow();

    process.env.CLICKHOUSE_KEY_ID = "kid";
    process.env.CLICKHOUSE_KEY_SECRET = "Sec4b1d";
    try {
      expect((await connect(CH, { waitMs: 0, agent: true }, deps)).host_metrics).toBe(false);
      expect(calls).toEqual(["list", "check clickhouse", "prepare clickhouse", `create ${JSON.stringify({ db_url: MON, provider: "clickhouse" })}`, `checkup ${MON} as ${CH_NAME}`]);
      // The key the agent passes is used.
      await connect(CH, { clickhouseKey: KEY, waitMs: 0, agent: true }, deps);
      expect(calls).toContain("clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa");
    } finally {
      delete process.env.CLICKHOUSE_KEY_ID;
      delete process.env.CLICKHOUSE_KEY_SECRET;
    }
  });

  test("--reset-password: refused while another database on the same server is monitored with postgres_ai_mon", async () => {
    const other = row("active", { id: "i-7", name: "abc123.us-east-1.aws.pg.clickhouse.cloud/orders" });
    const { deps, calls } = fake({ rows: [other] });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "A new password for postgres_ai_mon would cut off the monitoring of abc123.us-east-1.aws.pg.clickhouse.cloud/orders on this server: set PGAI_MON_PASSWORD to its password instead",
    });
    expect(calls).toEqual(["list"]);
  });

  test("--reset-password with --self-hosted: refused, the org's cloud monitoring of this server is not checked there", async () => {
    const { deps, calls } = fake();
    expect(await connect(CH, { resetPassword: true, selfHosted: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "--reset-password works with PostgresAI Cloud only (it checks what else monitors this server): connect without --self-hosted, or set PGAI_MON_PASSWORD",
    });
    expect(calls).toEqual([]);
  });

  test("--reset-password for a database already connected: nothing reset, disconnect first", async () => {
    const { deps, calls } = fake({ rows: [row("active")] });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, id: "i-1",
      next: `${CH_NAME} is already connected, and its monitoring uses the current password: pgai disconnect ${CH_NAME} --yes first`,
    });
    expect(calls).toEqual(["list"]);
  });

  test("--reset-password reaches the prepare step when nothing else on the server is monitored", async () => {
    let seen: unknown;
    const { deps } = fake({ rows: [row("deleting_launched")], prepare: async (url, provider, opts) => { seen = opts; return { monitoringUrl: MON }; } });
    await connect(CH, { resetPassword: true, waitMs: 0 }, deps);
    expect(seen).toEqual({ resetPassword: true });
  });

  test("a note from the prepare step ends the next action", async () => {
    const { deps } = fake({ prepare: async () => ({ monitoringUrl: MON, note: "the password was not checked" }) });
    expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe(`pgai status ${CH_NAME}; the password was not checked`);
    expect((await connect(CH, { selfHosted: true, waitMs: 0 }, deps)).next).toBe("pgai mon health; the password was not checked");
  });
});

// The prepare step with a faked pg client: what the server answers to each
// login is scripted, and every statement run over the admin connection is recorded.
describe("connect on the paid path: the price before the box", () => {
  const PAID = {
    plan: "scale", org_alias: "acme", billed: true, free_slots: { remaining: 0, total: 0 },
    subscription: false, quantity: 0, price: { amount: 51200, currency: "usd", interval: "month" },
    has_payment_method: true, requires_payment_method: false,
  };
  const quoted = (q: Record<string, unknown>, calls: string[]) => async (coupon?: string) => {
    calls.push(`quote${coupon ? ` ${coupon}` : ""}`);
    return { ...PAID, ...q } as never;
  };
  const SH = "postgresql://postgres:adminpw@db.example.com:5432/app";
  const SH_NAME = "db.example.com/app";
  const make = (q: Record<string, unknown> = {}, over: Partial<ConnectDeps> = {}) => {
    const f = fake({ rows: [undefined, row("active", { name: SH_NAME, provider: "self-managed", host_metrics: false })], ...over });
    f.deps.quote = quoted(q, f.calls);
    f.deps.billingUrl = (alias: string) => `https://console.example/${alias}/billing`;
    return f;
  };

  test("a billed box without --yes, where nobody can be asked: the price and how to accept, nothing prepared", async () => {
    const { deps, calls } = make();
    deps.confirm = async () => false;
    expect(await connect(SH, { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      next: "Re-run with --yes to accept $512.00/month per database cluster (scale plan)",
    });
    expect(calls).toEqual(["list", "check self-managed", "quote"]);
  });

  test("a billed box, confirmed at the prompt (which names the price): provisioned, the price in the result", async () => {
    const { deps, calls } = make();
    const asked: string[] = [];
    deps.confirm = async (q) => { asked.push(q); return true; };
    const result = await connect(SH, { waitMs: 60_000 }, deps);
    expect(asked).toEqual([`Provision ${SH_NAME}? (y/N): `]);
    expect(calls.slice(0, 4)).toEqual(["list", "check self-managed", "quote", "prepare self-managed"]);
    // The price was accepted: the platform creates a billed box only with accept_price.
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON, accept_price: true })}`);
    expect(result).toMatchObject({ status: "connected", price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false });
  });

  test("--yes: no prompt; a box added to the subscription says which box it is", async () => {
    const { deps, calls } = make({ subscription: true, quantity: 1 });
    deps.confirm = async () => { throw new Error("asked"); };
    const result = await connect(SH, { waitMs: 60_000, yes: true }, deps);
    expect(calls.slice(0, 4)).toEqual(["list", "check self-managed", "quote", "prepare self-managed"]);
    expect(result.price).toBe("$512.00/month per database cluster (scale plan), cluster 2 on the subscription");
  });

  test("declined at the prompt: nothing prepared", async () => {
    const { deps, calls } = make();
    deps.confirm = async () => false;
    expect((await connect(SH, { waitMs: 0 }, deps)).status).toBe("action_required");
    expect(calls).toEqual(["list", "check self-managed", "quote"]);
  });

  test("no payment method: stop before the database is touched, with the billing page (exit 3)", async () => {
    const { deps, calls } = make({ has_payment_method: false, requires_payment_method: true });
    expect(await connect(SH, { waitMs: 0, yes: true }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: true,
      next: "Add a payment method at https://console.example/acme/billing, then re-run",
    });
    expect(calls).toEqual(["list", "check self-managed", "quote"]);
  });

  test("a free slot: no prompt, the result says free (N of M free slots)", async () => {
    const { deps, calls } = make({ billed: false, free_slots: { remaining: 1, total: 2 } });
    deps.confirm = async () => { throw new Error("asked"); };
    const result = await connect(SH, { waitMs: 60_000 }, deps);
    expect(calls.slice(0, 4)).toEqual(["list", "check self-managed", "quote", "prepare self-managed"]);
    expect(result).toMatchObject({ status: "connected", price: "free (1 of 2 free slots)", requires_payment_method: false });
  });

  test("--coupon: the discounted price is shown, and the code goes with the box", async () => {
    const promo = { code: "LAUNCH100", valid: true, discount_description: "100% off (first billing period)", percent_off: 100, duration: "once", promotion_code_id: "promo_1" };
    const { deps, calls } = make({ promo, amount_after_promo: 0 });
    const result = await connect(SH, { waitMs: 60_000, yes: true, coupon: "LAUNCH100" }, deps);
    expect(calls[2]).toBe("quote LAUNCH100");
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON, promo_code: "LAUNCH100", accept_price: true })}`);
    expect(result).toMatchObject({
      status: "connected",
      price: "$0.00 the first month with LAUNCH100 (100% off (first billing period)), then $512.00/month per database cluster (scale plan)",
      coupon: { code: "LAUNCH100", valid: true, description: "100% off (first billing period)" },
    });
  });

  test("an invalid or expired coupon: a clear error (exit 3), nothing prepared or provisioned", async () => {
    const { deps, calls } = make({ promo: { code: "OLD", valid: false, error: "Promo code is expired" } });
    expect(await connect(SH, { waitMs: 0, yes: true, coupon: "OLD" }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      coupon: { code: "OLD", valid: false, error: "Promo code is expired" },
      next: "Promo code OLD: Promo code is expired. Nothing was changed: re-run with a valid code, or without --coupon",
    });
    expect(calls).toEqual(["list", "check self-managed", "quote OLD"]);
  });

  test("the platform refusing for payment (402, a card removed meanwhile): the billing page, not a failure", async () => {
    const { deps } = make({}, { create: async () => { throw new HttpStatusError("Payment Required", 402); } });
    expect(await connect(SH, { waitMs: 0, yes: true }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: true,
      next: "Add a payment method at https://console.example/acme/billing, then re-run",
    });
  });

  test("a 402 although the org has a card (it was declined): says so, with the billing page", async () => {
    const { deps } = make({}, { create: async () => { throw new HttpStatusError("Your card was declined.", 402); } });
    expect((await connect(SH, { waitMs: 0, yes: true }, deps)).next)
      .toBe("The payment method on file was declined: update it at https://console.example/acme/billing, then re-run");
  });

  test("a free slot sends no accept_price; the platform billing it after all (412, the slot went meanwhile) is: re-run to see the price", async () => {
    const { deps, calls } = make({ billed: false, free_slots: { remaining: 1, total: 1 } }, { create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); throw new HttpStatusError("Precondition Failed", 412); } });
    expect(await connect(SH, { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "free (1 of 1 free slots)", requires_payment_method: false,
      next: "The price changed since it was shown: re-run pgai connect to see it",
    });
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON })}`);
  });

  test("--coupon '' (an empty variable): refused, not dropped; nothing prepared", async () => {
    const { deps, calls } = make();
    expect(await connect(SH, { waitMs: 0, yes: true, coupon: " " }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      next: "--coupon is empty: pass a promotion code, or leave --coupon out",
    });
    expect(calls).toEqual(["list"]);
  });

  // Billing starts when the box is active: a declined first charge then removes the box while connect waits.
  test("the first charge declined at activation: why, where to fix it, and the generated role dropped", async () => {
    const f = make({}, { rows: [undefined, row("registered", { name: SH_NAME, provider: "self-managed", host_metrics: false }), row("deleting_launched", { name: SH_NAME, provider: "self-managed", host_metrics: false, billing_error: "Payment Required: Stripe payment required for POST /subscriptions: Your card was declined." })] });
    f.deps.prepare = async (_u, provider, o) => { f.calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }; };
    expect(await connect(SH, { waitMs: 60_000, yes: true }, f.deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME, id: "i-1",
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false, checkup: CHECKUP,
      next: "The first charge failed (Payment Required: Stripe payment required for POST /subscriptions: Your card was declined.): the box was removed and nothing is billed. Update the payment method at https://console.example/acme/billing, then re-run",
    });
    expect(f.calls.filter((c) => c === "unprepare")).toEqual(["unprepare"]);
  });

  test("the box removed while connect waits, with no reason given: failed, not waiting out the deadline", async () => {
    const f = make({}, { rows: [undefined, row("registered", { name: SH_NAME, provider: "self-managed", host_metrics: false }), undefined] });
    let t = 0;
    f.deps.now = () => (t += 15_000);
    const r = await connect(SH, { waitMs: 60 * 60_000, yes: true }, f.deps);
    expect(r.status).toBe("failed");
    expect(r.next).toBe("The monitoring box was removed before it became active: see pgai databases, then re-run pgai connect");
    expect(f.calls.filter((c) => c === "sleep").length).toBeLessThan(5);
  });

  // Pricing is per Postgres cluster observed: the quote is asked for this URL's cluster (host:port).
  test("the quote is asked for the URL's cluster; another database in a billed cluster is included, never prompted", async () => {
    const seen: (string | undefined)[] = [];
    const { deps, calls } = make({ billed: false, same_cluster: "db.example.com/other", subscription: true, quantity: 1 });
    const inner = deps.quote;
    deps.quote = async (coupon, cluster) => { seen.push(cluster); return inner(coupon, cluster); };
    deps.confirm = async () => { throw new Error("asked"); };
    const r = await connect("postgresql://postgres:adminpw@DB.Example.com/app", { waitMs: 0 }, deps);
    expect(seen).toEqual(["db.example.com:5432"]);
    expect(r.price).toBe("included: same database cluster as db.example.com/other, no extra charge");
    expect(calls).toContain(`create ${JSON.stringify({ db_url: MON })}`);
  });

  test("the price is per database cluster, and names the cluster's place on the subscription", () => {
    expect(priceText({ ...PAID } as never)).toBe("$512.00/month per database cluster (scale plan)");
    expect(priceText({ ...PAID, subscription: true, quantity: 2 } as never)).toBe("$512.00/month per database cluster (scale plan), cluster 3 on the subscription");
  });

  test("a repeating or permanent discount is not described as the first month only", async () => {
    const promo = (duration: string, extra: Record<string, unknown> = {}) => ({ code: "C", valid: true, discount_description: "x", duration, ...extra });
    const q = (p: Record<string, unknown>) => priceText({ ...PAID, promo: p, amount_after_promo: 25600 } as never);
    expect(q(promo("repeating", { duration_in_months: 3 }))).toBe("$256.00/month for 3 months with C (x), then $512.00/month per database cluster (scale plan)");
    expect(q(promo("forever"))).toBe("$256.00/month with C (x), instead of $512.00/month per database cluster (scale plan)");
  });

  test("a URL that cannot work (not an admin): the SQL before any price is asked, and nothing quoted", async () => {
    const { deps, calls } = make({}, {});
    deps.prepare = async (_u, provider, o) => { calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return { next: "Run the SQL as an admin", sql: "create role ..." }; };
    deps.confirm = async () => { throw new Error("asked"); };
    expect(await connect(SH, { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      next: "Run the SQL as an admin", sql: "create role ...",
    });
    expect(calls).toEqual(["list", "check self-managed"]);
  });

  test("the price is shown on its own line before the prompt asks", async () => {
    const { deps } = make();
    const seen: string[] = [];
    deps.progress = (e) => seen.push(progressText(e));
    deps.confirm = async (q) => { seen.push(`ask ${q}`); return false; };
    await connect(SH, { waitMs: 0 }, deps);
    expect(seen).toEqual(["Billing: $512.00/month per database cluster (scale plan) (+0s)", `ask Provision ${SH_NAME}? (y/N): `]);
  });

  test("the billing page is the platform's own (a preview's console, not production's)", async () => {
    const { deps } = make({ has_payment_method: false, requires_payment_method: true, billing_url: "https://console-pr.pgai.green/acme/billing" });
    expect((await connect(SH, { waitMs: 0, yes: true }, deps)).next).toBe("Add a payment method at https://console-pr.pgai.green/acme/billing, then re-run");
  });

  test("an agent: the price, and to call again with yes; a coupon is checked first", async () => {
    const { deps, calls } = make();
    expect(await connect(SH, { waitMs: 0, agent: true }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      next: "Call connect_database again with yes: true to accept $512.00/month per database cluster (scale plan)",
    });
    const again = make({ promo: { code: "NOPE", valid: false, error: "Promo code not found or expired" } });
    expect((await connect(SH, { waitMs: 0, agent: true, yes: true, coupon: "NOPE" }, again.deps)).next)
      .toBe("Promo code NOPE: Promo code not found or expired. Nothing was changed: re-run with a valid code, or without coupon");
    expect([...calls, ...again.calls].filter((c) => c.startsWith("quote"))).toEqual(["quote", "quote NOPE"]);
  });

  test("a re-run of a connected database asks no price", async () => {
    const f = fake({ rows: [row("active")] });
    f.deps.quote = quoted({}, f.calls);
    await connect(CH, { waitMs: 0 }, f.deps);
    expect(f.calls).toEqual(["list"]);
  });
});

describe("prepareDatabase (a fake pg client)", () => {
  const ADMIN = "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=require&options=-c%20role%3Dx&sslrootcert=%2Ftmp%2Fca.pem&application_name=pgai";
  const SET_PASSWORD = "Set PGAI_MON_PASSWORD to the password of postgres_ai_mon, or give it a new one: pgai connect <admin-url> --reset-password (anything else that logs in as postgres_ai_mon then needs the new password)";
  const pgError = (code: string, message: string) => Object.assign(new Error(message), { code });

  /** `logins` answers each postgres_ai_mon login in turn (an error to throw, or "ok"); the last one repeats. */
  function server(me: { name?: string; admin?: boolean; mon_exists: boolean; fails?: RegExp }, logins: (Error | "ok")[] = ["ok"]) {
    const ran: string[] = [];
    const sqls: string[] = [];
    const monLogins: string[] = [];
    const monUrls: string[] = [];
    class FakeClient {
      password: string;
      private user: string;
      constructor(private config: { connectionString: string; ssl?: unknown }) {
        const u = new URL(config.connectionString);
        this.user = u.username;
        // As pg does: PGPASSWORD for a URL without a password.
        this.password = decodeURIComponent(u.password) || process.env.PGPASSWORD || "";
        if (this.user === "postgres_ai_mon") monUrls.push(config.connectionString);
      }
      async connect() {
        if (this.config.ssl && me.name === "no-tls") throw new Error("The server does not support SSL connections");
        if (this.user !== "postgres_ai_mon") return;
        const answer = logins[Math.min(monLogins.length, logins.length - 1)];
        monLogins.push(this.password);
        if (answer !== "ok") throw answer;
      }
      async query(sql: string) {
        if (/session_user as name/.test(sql)) return { rows: [{ name: "postgres", db: "app", admin: true, iterations: "4096", ...me }] };
        if (this.user !== "postgres_ai_mon" && !/statement_timeout/.test(sql)) ran.push(sql.trim().split("\n")[0]);
        if (this.user !== "postgres_ai_mon" && !/session_user as name|statement_timeout/.test(sql)) sqls.push(sql);
        if (me.fails?.test(sql)) throw pgError("42501", "permission denied");
        // The monitoring role's own session: every check of verifyInitSetup passes.
        if (me.name === "postgres_ai_mon") return { rowCount: 1, rows: [{ ok: true, rolconfig: ["search_path=postgres_ai, public, pg_catalog"] }] };
        return { rows: [] };
      }
      async end() {}
    }
    const prepare = (opts: PrepareOptions = {}, url = ADMIN) => prepareDatabase(url, "self-managed", { ...opts, Client: FakeClient as unknown as PrepareOptions["Client"] });
    return { prepare, ran, sqls, monLogins, monUrls };
  }
  const withMonPassword = async <T>(value: string | undefined, fn: () => Promise<T>) => {
    if (value !== undefined) process.env.PGAI_MON_PASSWORD = value;
    try {
      return await fn();
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
  };

  test("an existing role and a rejected PGAI_MON_PASSWORD: refused before anything runs", async () => {
    const s = server({ mon_exists: true }, [pgError("28P01", "password authentication failed")]);
    expect(await withMonPassword("not-the-password", () => s.prepare())).toEqual({
      next: `postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. ${SET_PASSWORD}`,
    });
    expect(s.monLogins).toEqual(["not-the-password"]);
    expect(s.ran).toEqual([]);
  });

  test("an existing role and --reset-password: a new password is set, and the box's URL carries it", async () => {
    const s = server({ mon_exists: true });
    const result = await withMonPassword(undefined, () => s.prepare({ resetPassword: true }));
    const password = new URL((result as { monitoringUrl: string }).monitoringUrl).password;
    expect(password.length).toBeGreaterThan(20);
    expect(result).toEqual({ monitoringUrl: `postgresql://postgres_ai_mon:${password}@db.example.com:5432/app?sslmode=require&application_name=pgai` });
    expect(s.sqls.some((q) => /alter user "?postgres_ai_mon"? with password 'SCRAM-SHA-256\$/.test(q))).toBe(true);
    // The new password logs in before the box gets it.
    expect(s.monLogins.at(-1)).toBe(password);
  });

  test("--reset-password is not for an agent's URL", async () => {
    const s = server({ mon_exists: true });
    expect((await s.prepare({ agent: true, resetPassword: true }) as { next: string }).next).toStartWith("postgres_ai_mon already exists on this server. Pass its URL as database_url");
    expect(s.sqls).toEqual([]);
  });

  test("an existing role and no PGAI_MON_PASSWORD: refused, no login tried", async () => {
    const s = server({ mon_exists: true });
    expect(await withMonPassword(undefined, () => s.prepare())).toEqual({ next: `postgres_ai_mon already exists on this server. ${SET_PASSWORD}` });
    expect(s.monLogins).toEqual([]);
    expect(s.ran).toEqual([]);
  });

  test("a role someone else created while the plan ran: its password is not ours, refused after the plan", async () => {
    const s = server({ mon_exists: false }, [pgError("28P01", "password authentication failed")]);
    expect(await withMonPassword("ours", () => s.prepare())).toEqual({ next: `postgres_ai_mon was created by someone else meanwhile. ${SET_PASSWORD}` });
    expect(s.monLogins).toEqual(["ours"]);
    expect(s.ran.filter((q) => q === "begin;").length).toBeGreaterThan(0);
  });

  test("an existing role whose login this host may not even try (pg_hba): says so, nothing runs", async () => {
    const s = server({ mon_exists: true }, [pgError("28000", "no pg_hba.conf entry for host")]);
    expect(await withMonPassword("its-password", () => s.prepare())).toEqual({
      next: "postgres_ai_mon already exists on this server, and PGAI_MON_PASSWORD could not be checked from this host (no pg_hba.conf entry for host). Run pgai connect from a host that postgres_ai_mon may connect from",
    });
    expect(s.ran).toEqual([]);
  });

  test("a new role whose login this host may not try: prepared, the URL carries the password set", async () => {
    const s = server({ mon_exists: false }, [pgError("28000", "no pg_hba.conf entry for host")]);
    expect(await withMonPassword("ours", () => s.prepare())).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:ours@db.example.com:5432/app?sslmode=require&application_name=pgai",
    });
  });

  test("a server that accepts any password from this host: prepared, with a note that the password was not checked", async () => {
    const s = server({ mon_exists: true });
    expect(await withMonPassword("maybe", () => s.prepare())).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:maybe@db.example.com:5432/app?sslmode=require&application_name=pgai",
      note: "this server accepts any password from this host, so PGAI_MON_PASSWORD was not checked; if no data arrives, disconnect and connect again with the right password",
    });
    expect(s.monLogins.length).toBe(3);
    expect(s.monLogins[1]).not.toBe("maybe");
  });

  test("an existing role and the right password on a server that checks it: prepared, no note", async () => {
    const s = server({ mon_exists: true }, ["ok", pgError("28P01", "password authentication failed"), "ok"]);
    expect(await withMonPassword("right", () => s.prepare())).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:right@db.example.com:5432/app?sslmode=require&application_name=pgai",
    });
  });

  test("a new role with a generated password is marked, so a refused launch can drop it; with PGAI_MON_PASSWORD it is not", async () => {
    const result = await withMonPassword(undefined, () => server({ mon_exists: false }).prepare());
    expect(result).toMatchObject({ generated: true });
    expect(await withMonPassword("ours", () => server({ mon_exists: false }).prepare())).not.toHaveProperty("generated");
  });

  test("a plan that fails after the role was created with a generated password: the role is dropped again", async () => {
    const s = server({ mon_exists: false, fails: /create extension/ });
    await expect(withMonPassword(undefined, () => s.prepare())).rejects.toThrow('Failed at step "02.extensions": permission denied');
    expect(s.ran.at(-1)).toBe("drop role postgres_ai_mon");
  });

  test("a plan that fails: a role with the user's PGAI_MON_PASSWORD, or one that does not log in with ours, is left alone", async () => {
    const mine = server({ mon_exists: false, fails: /create extension/ });
    await expect(withMonPassword("ours", () => mine.prepare())).rejects.toThrow('Failed at step "02.extensions"');
    const notOurs = server({ mon_exists: false, fails: /create extension/ }, [pgError("28P01", "password authentication failed")]);
    await expect(withMonPassword(undefined, () => notOurs.prepare())).rejects.toThrow('Failed at step "02.extensions"');
    for (const s of [mine, notOurs]) expect(s.ran.some((q) => q.startsWith("drop"))).toBe(false);
  });

  test("the MCP tool's prepare: PGAI_MON_PASSWORD is not read, and TLS is not given up", async () => {
    const existing = server({ mon_exists: true });
    expect(await withMonPassword("its-password", () => existing.prepare({ agent: true }))).toEqual({
      next: "postgres_ai_mon already exists on this server. Pass its URL as database_url, or run pgai connect in a terminal with PGAI_MON_PASSWORD set to its password",
    });
    expect(existing.monLogins).toEqual([]);

    const created = server({ mon_exists: false });
    const result = await withMonPassword("from-the-environment", () => created.prepare({ agent: true }));
    expect(JSON.stringify(result)).not.toContain("from-the-environment");
    expect(result).toHaveProperty("monitoringUrl");

    // A server without TLS and a URL that does not ask for a mode: the CLI retries in plaintext, the tool does not.
    const noTls = server({ name: "no-tls", mon_exists: false });
    const url = "postgresql://postgres:adminpw@db.example.com:5432/app";
    await expect(noTls.prepare({ agent: true }, url)).rejects.toThrow("The server does not support SSL connections");
    expect(await noTls.prepare({}, url)).toHaveProperty("monitoringUrl");
  });

  // The first login in these is the given URL's own session; the second is the random-password probe.
  const MON_URL = "postgresql://postgres_ai_mon:its-password@db.example.com:5432/app?sslmode=require";
  const REJECTED = pgError("28P01", "password authentication failed");

  test("a postgres_ai_mon URL on a server that checks passwords: its URL, no note", async () => {
    const s = server({ name: "postgres_ai_mon", mon_exists: true }, ["ok", REJECTED]);
    expect(await s.prepare({}, MON_URL)).toEqual({ monitoringUrl: MON_URL });
    expect(s.monLogins.length).toBe(2);
    expect(s.monLogins[1]).not.toBe("its-password");
  });

  test("a postgres_ai_mon URL on a server that accepts any password: a note that the URL's password was not checked", async () => {
    const s = server({ name: "postgres_ai_mon", mon_exists: true });
    expect(await s.prepare({}, MON_URL)).toEqual({
      monitoringUrl: MON_URL,
      note: "this server accepts any password from this host, so the password in the URL was not checked; if no data arrives, disconnect and connect again with the right password",
    });
  });

  test("a postgres_ai_mon URL without a password: PGPASSWORD is used only when the server checked it", async () => {
    const bare = "postgresql://postgres_ai_mon@db.example.com:5432/app?sslmode=require";
    process.env.PGPASSWORD = "unrelated-secret";
    try {
      const trusting = await server({ name: "postgres_ai_mon", mon_exists: true }).prepare({}, bare);
      expect(trusting).toEqual({ next: "This server accepts any password from this host, so the one from the environment was not checked and is not used. Put the password of postgres_ai_mon in the URL: the monitoring box logs in with it" });
      expect(JSON.stringify(trusting)).not.toContain("unrelated-secret");
      expect(await server({ name: "postgres_ai_mon", mon_exists: true }, ["ok", REJECTED]).prepare({}, bare)).toEqual({
        monitoringUrl: "postgresql://postgres_ai_mon:unrelated-secret@db.example.com:5432/app?sslmode=require",
      });
    } finally {
      delete process.env.PGPASSWORD;
    }
  });

  test("a private CA (sslrootcert): the logins from this machine use it, the box's URL does not carry it, and next says so", async () => {
    const url = "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=verify-full&sslrootcert=%2Ftmp%2Fca.pem&sslcert=%2Ftmp%2Fc.pem&sslkey=%2Ftmp%2Fk.pem&options=-c%20role%3Dx&user=postgres";
    const CA_NOTE = "the monitoring box has no copy of the CA in sslrootcert: with sslmode=verify-full it connects only to a server certificate signed by a public CA (else connect with sslmode=require)";
    const s = server({ mon_exists: true }, ["ok", REJECTED, "ok"]);
    expect(await withMonPassword("right", () => s.prepare({}, url))).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:right@db.example.com:5432/app?sslmode=verify-full",
      note: CA_NOTE,
    });
    // Pre-check, probe, post-plan check: the TLS files, and nothing else of the admin's session.
    expect(s.monUrls.length).toBe(3);
    for (const u of s.monUrls) expect([...new URL(u).searchParams.keys()].sort()).toEqual(["sslcert", "sslkey", "sslrootcert"]);

    // The monitoring role's own URL: the same note; with a trusting server, both.
    const own = await server({ name: "postgres_ai_mon", mon_exists: true }).prepare({}, "postgresql://postgres_ai_mon:pw@db.example.com/app?sslmode=verify-ca&sslrootcert=%2Ftmp%2Fca.pem");
    expect(own).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:pw@db.example.com/app?sslmode=verify-ca",
      note: `${CA_NOTE.replace("verify-full", "verify-ca")}; this server accepts any password from this host, so the password in the URL was not checked; if no data arrives, disconnect and connect again with the right password`,
    });
    // sslmode=require does not verify at the box: nothing to say.
    expect(await withMonPassword("ours", () => server({ mon_exists: false }).prepare())).not.toHaveProperty("note");
  });

  test("a role that may not grant pg_monitor: the SQL to run, nothing created", async () => {
    const s = server({ admin: false, mon_exists: false });
    const result = await s.prepare();
    expect((result as { next: string }).next).toBe("Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai connect again with the postgres_ai_mon URL");
    expect((result as { sql: string }).sql).toContain("-- 01.role");
    expect(s.ran).toEqual([]);
  });
});

describe("clickhouseOrgFor (a fake ClickHouse Cloud API)", () => {
  const OTHER = "99999999-8888-7777-6666-555555555555";
  const SERVICE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
  async function withApi(orgsStatus: number, fn: () => Promise<void>, otherStatus = 200, orgsBody?: unknown) {
    const seen: string[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(req) {
        const path = new URL(req.url).pathname;
        seen.push(`${req.headers.get("authorization")?.slice(0, 6)} ${path}`);
        const json = (result: unknown) => Response.json({ result });
        if (path === "/v1/organizations" && orgsBody) return Response.json(orgsBody);
        if (path === "/v1/organizations") return orgsStatus === 200 ? json([{ id: OTHER }, { id: ORG }]) : new Response("", { status: orgsStatus });
        if (path === `/v1/organizations/${OTHER}/postgres`) return otherStatus === 200 ? json([]) : new Response("", { status: otherStatus });
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
      // The user's to fix (exit 3), unlike an API outage.
      await expect(clickhouseOrgFor("x.pg.clickhouse.cloud", "kid", "bad")).rejects.toBeInstanceOf(ClickhouseKeyError);
    });
    await withApi(200, async () => {
      await expect(clickhouseOrgFor("nope.pg.clickhouse.cloud", "kid", "secret")).rejects.toThrow("has hostname nope.pg.clickhouse.cloud");
      await expect(clickhouseOrgFor("nope.pg.clickhouse.cloud", "kid", "secret")).rejects.toBeInstanceOf(ClickhouseKeyError);
    });
  });

  test("a key that cannot read the services (403) is the user's to fix too, not an outage", async () => {
    await withApi(200, async () => {
      await expect(clickhouseOrgFor("nope.pg.clickhouse.cloud", "kid", "secret")).rejects.toBeInstanceOf(ClickhouseKeyError);
    }, 403);
  });

  test("an organization the key cannot read is reported, not the next one's missing service", async () => {
    await withApi(200, async () => {
      await expect(clickhouseOrgFor("nope.pg.clickhouse.cloud", "kid", "secret")).rejects.toThrow(`cannot read Postgres services in organization ${OTHER} (403)`);
    }, 403);
  });

  test("a response without an organization list is a clear error", async () => {
    await withApi(200, async () => {
      await expect(clickhouseOrgFor("x.pg.clickhouse.cloud", "kid", "secret")).rejects.toThrow("ClickHouse Cloud API returned no organization list.");
    }, 200, {});
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

  test("a box still provisioning is answered at once (no wait); a failed one is an error result", async () => {
    let status = "launch_requested";
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch: () => Response.json([row(status)]),
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    const call = () => handleToolCall({ params: { name: "connect_database", arguments: { database_url: CH } } }, opts);
    try {
      const started = Date.now();
      const provisioning = await call();
      expect(Date.now() - started).toBeLessThan(3000);
      expect(provisioning.isError).toBe(false);
      expect(JSON.parse(provisioning.content[0].text)).toEqual({
        status: "provisioning", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: true, next: `pgai status ${CH_NAME}`,
      });
      status = "failed";
      const failed = await call();
      expect(failed.isError).toBe(true);
      expect(JSON.parse(failed.content[0].text).status).toBe("failed");
    } finally {
      server.stop(true);
    }
  });

  test("a database that cannot be reached: the agent is told so, and no price is quoted first", async () => {
    const quotes: unknown[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
        if (path.endsWith("/rpc/cloud_monitoring_quote")) { quotes.push(await req.json()); return Response.json({}); }
        return new Response("not found", { status: 404 });
      },
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    try {
      const result = await handleToolCall({ params: { name: "connect_database", arguments: { database_url: "postgresql://postgres:pw@db.example.invalid:5432/app", yes: true } } }, opts);
      expect(result.isError).toBe(true);
      expect(quotes).toEqual([]);
    } finally {
      server.stop(true);
    }
  });

  test("query parameters pg would obey (host, port, certificate files) are refused before anything is contacted", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    const result = await handleToolCall({ params: { name: "connect_database", arguments: { database_url: "postgresql://postgres:pw@db.example.invalid:5432/postgres?host=127.0.0.1&port=32785&sslkey=/home/u/key.pem" } } }, opts);
    expect(result).toEqual({ content: [{ type: "text", text: "database_url may carry only these query parameters: sslmode, channel_binding, application_name (got: host, port, sslkey)" }], isError: true });
  });

  test("the URL must carry its password: nothing of this machine (PGPASSWORD) is sent to a host an agent chose", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    for (const database_url of ["postgresql://postgres@db.example.com:5432/app", "host=db dbname=app", undefined]) {
      const result = await handleToolCall({ params: { name: "connect_database", arguments: { database_url } } }, opts);
      expect(result).toEqual({ content: [{ type: "text", text: "database_url must be postgresql://user:password@host:5432/dbname, with the password in it" }], isError: true });
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

test("sslmode in the URL wins over PGSSLMODE, as in libpq; PGSSLMODE applies to a URL without one", () => {
  process.env.PGSSLMODE = "disable";
  try {
    const strict = resolveAdminConnection({ conn: "postgresql://u:p@h:5432/d?sslmode=verify-full" });
    expect(strict.clientConfig.ssl).toBe(true);
    expect(strict.sslFallbackEnabled).toBe(false);
    expect(resolveAdminConnection({ conn: "postgresql://u:p@h:5432/d" }).clientConfig.ssl).toBe(false);
  } finally {
    delete process.env.PGSSLMODE;
  }
});

describe("saving the express checkup", () => {
  const reports = { A002: { checkId: "A002" }, H002: { checkId: "H002" } };
  const fakeRpc = (answer: (fn: string, body: Record<string, unknown>) => unknown) => {
    const calls: string[] = [];
    const rpc = async <T>(fn: string, body: Record<string, unknown>): Promise<T> => {
      calls.push(`${fn}${body.status ? ` ${body.status}` : ""}${body.filename ? ` ${body.filename}` : ""}`);
      const a = answer(fn, body);
      if (a instanceof Error) throw a;
      return a as T;
    };
    return { rpc, calls };
  };
  const ok = (fn: string) => (fn === "checkup_report_create" ? { report_id: 9 } : fn === "checkup_report_file_post" ? { report_chunck_id: 1 } : { status: "completed" });

  test("created pending, a file per check, then completed", async () => {
    const { rpc, calls } = fakeRpc(ok);
    expect(await saveCheckupReport(rpc, "tok", "db/app", reports)).toBe(9);
    expect(calls).toEqual(["checkup_report_create", "checkup_report_file_post A002.json", "checkup_report_file_post H002.json", "checkup_report_status_update completed"]);
  });

  test("a report not marked completed is not called saved", async () => {
    const { rpc } = fakeRpc((fn) => (fn === "checkup_report_status_update" ? new Error("HTTP 500") : ok(fn)));
    await expect(saveCheckupReport(rpc, "tok", "db/app", reports)).rejects.toThrow("HTTP 500");
  });

  test("an answer without an id is a failed upload, and the report is marked failed", async () => {
    const { rpc, calls } = fakeRpc((fn) => (fn === "checkup_report_file_post" ? { message: "Upload rejected" } : ok(fn)));
    await expect(saveCheckupReport(rpc, "tok", "db/app", reports)).rejects.toThrow("Upload rejected");
    expect(calls.at(-1)).toBe("checkup_report_status_update failed");
    await expect(saveCheckupReport(fakeRpc(() => ({})).rpc, "tok", "db/app", reports)).rejects.toThrow("checkup_report_create");
  });
});

describe("disconnect: what happened to the billing", () => {
  test("the last box: the subscription is canceled; another box left: how many remain; a failure is said; nothing released: nothing", () => {
    expect(disconnectBilling({ billing: { subscription: "canceled", quantity: 0 } })).toBe("subscription canceled: no further charges; the unused part of this period is credited (prorated)");
    expect(disconnectBilling({ billing: { subscription: "canceled" } })).toBe("subscription canceled: no further charges; the unused part of this period is credited (prorated)");
    expect(disconnectBilling({ billing: { subscription: "active", quantity: 2 } })).toBe("2 database clusters left on the subscription");
    expect(disconnectBilling({ billing: { subscription: "active", quantity: 1 } })).toBe("1 database cluster left on the subscription");
    expect(disconnectBilling({ billing_warning: "Failed to cancel org subscription: stripe down" })).toBe("not released (Failed to cancel org subscription: stripe down): contact support");
    // The last cluster: the unused time goes back to the card (or stays as credit when the refund failed).
    expect(disconnectBilling({ billing: { subscription: "canceled", quantity: 0, refunded: 49907, currency: "usd" } })).toBe("subscription canceled: no further charges; refunded $499.07 to your card");
    expect(disconnectBilling({ billing: { subscription: "canceled", quantity: 0, credited: true }, billing_warning: "The refund failed: card expired. The unused time stays as account credit." }))
      .toBe("subscription canceled: no further charges; the refund failed (The refund failed: card expired. The unused time stays as account credit.)");
    expect(disconnectBilling({})).toBeUndefined();
    expect(disconnectBilling(null)).toBeUndefined();
  });
});

describe("errorText", () => {
  // node's connect tries every address of a host name (IPv6 and IPv4) and,
  // when all refuse, throws an AggregateError whose own message is empty.
  test("an AggregateError with no message says what each attempt got", () => {
    const err = Object.assign(new AggregateError([
      Object.assign(new Error("connect ECONNREFUSED 2001:db8::1:25499"), { code: "ECONNREFUSED" }),
      Object.assign(new Error("connect ECONNREFUSED 192.0.2.1:25499"), { code: "ECONNREFUSED" }),
    ], ""), { code: "ECONNREFUSED" });
    expect(errorText(err)).toBe("connect ECONNREFUSED 2001:db8::1:25499; connect ECONNREFUSED 192.0.2.1:25499");
  });

  test("an error with a message, and a non-error, as before", () => {
    expect(errorText(new Error("boom"))).toBe("boom");
    expect(errorText("plain")).toBe("plain");
  });
});
