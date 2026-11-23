// First-install bugs from https://gitlab.com/postgres-ai/postgresai/-/work_items/529.
// CLI runs use the offline sandbox; PGAI_TEST_PG_FAKE gives prepare-db a fake database.
import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";
import { createCliSandbox } from "./cli-sandbox";
import * as checkup from "../lib/checkup";
import * as summary from "../lib/checkup-summary";
import * as init from "../lib/init";
import { GLOBAL_TOKEN_PREFIX } from "../lib/config";
import { getAllCheckupCodes } from "../lib/checkup-dictionary";

const GLOBAL_TOKEN = `${GLOBAL_TOKEN_PREFIX}${"a".repeat(43)}`;
const DB_URL = "postgresql://admin@db.example.test:5432/appdb";

let sandbox: ReturnType<typeof createCliSandbox> | null = null;
afterEach(() => { sandbox?.cleanup(); sandbox = null; });
function runCli(args: string[], env: NodeJS.ProcessEnv = {}) {
  sandbox ??= createCliSandbox();
  return sandbox.run(args, { PGPASSWORD: "", ...env });
}

/** prepare-db against the fake database; returns the run and the statements it sent. */
function prepareDb(args: string[], roleExists: boolean, env: NodeJS.ProcessEnv = {}) {
  sandbox ??= createCliSandbox();
  const log = path.join(sandbox.root, "statements.sql");
  const r = runCli(["prepare-db", DB_URL, ...args], {
    PGAI_TEST_PG_FAKE: "1",
    PGAI_TEST_PG_ROLE_EXISTS: roleExists ? "1" : "0",
    PGAI_TEST_PG_LOG: log,
    PGAI_MON_PASSWORD: "",
    ...env,
  });
  const sql = fs.existsSync(log) ? fs.readFileSync(log, "utf8") : "";
  return { ...r, sql };
}

const ALTER_PASSWORD = /alter user "postgres_ai_mon" with password/i;

describe("bug 1: checkup --no-upload needs no organization", () => {
  test("a stored global token does not stop an offline run", () => {
    const r = runCli(["checkup", DB_URL, "--no-upload"], { PGAI_API_KEY: GLOBAL_TOKEN });
    expect(r.stderr).not.toContain("needs an organization");
    // It gets as far as the database (the offline fixture has none).
    expect(r.stderr).toContain("Database unavailable");
  });

  test("an uploading run with a global token still needs an organization", () => {
    const r = runCli(["checkup", DB_URL], { PGAI_API_KEY: GLOBAL_TOKEN });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("needs an organization");
  });
});

describe("bug 2: re-running prepare-db keeps the monitoring password", () => {
  test("existing role, no password given: the password is not changed", () => {
    const r = prepareDb([], true);
    expect(r.status).toBe(0);
    expect(r.sql).toContain("Role creation"); // the role step still runs
    expect(r.sql).not.toMatch(ALTER_PASSWORD);
    expect(r.stdout + r.stderr).toMatch(/password is kept as is/i);
  });

  test("existing role, --json: success, no generated password, passwordChanged false", () => {
    const r = prepareDb(["--json"], true);
    expect(r.status).toBe(0);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(true);
    expect(out.generatedPassword).toBeUndefined();
    expect(out.passwordChanged).toBe(false);
    expect(r.sql).not.toMatch(ALTER_PASSWORD);
  });

  test("--reset-password with an explicit password changes it", () => {
    const r = prepareDb(["--reset-password", "--password", "synthetic-pw-1"], true);
    expect(r.status).toBe(0);
    expect(r.sql).toMatch(ALTER_PASSWORD);
  });

  test("an explicit password (PGAI_MON_PASSWORD) still sets it on an existing role", () => {
    const r = prepareDb([], true, { PGAI_MON_PASSWORD: "synthetic-pw-2" });
    expect(r.status).toBe(0);
    expect(r.sql).toMatch(ALTER_PASSWORD);
  });
});

describe("--reset-password without a password on an existing role generates a new one", () => {
  test("--json, non-TTY: exit 1, JSON error, the password is not changed", () => {
    const r = prepareDb(["--reset-password", "--json"], true);
    expect(r.status).toBe(1);
    expect(r.stdout).not.toContain("generatedPassword");
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(false);
    expect(out.error.message).toMatch(/--print-password/);
    expect(r.sql).not.toMatch(ALTER_PASSWORD);
  });

  test("--json --print-password: one ALTER, the new password returned, passwordChanged true", () => {
    const r = prepareDb(["--reset-password", "--json", "--print-password"], true);
    expect(r.status).toBe(0);
    expect(r.sql.match(new RegExp(ALTER_PASSWORD.source, "gi"))?.length).toBe(1);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(true);
    expect(out.action).toBe("reset-password");
    expect(typeof out.generatedPassword).toBe("string");
    expect(out.passwordChanged).toBe(true);
  });
});

describe("bug 3: no generated password in non-TTY output without --print-password", () => {
  test("new role, --json, non-TTY: no password on stdout, JSON error, nothing applied", () => {
    const r = prepareDb(["--json"], false);
    expect(r.stdout).not.toContain("generatedPassword");
    expect(r.status).toBe(1);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(false);
    expect(out.error.message).toMatch(/--print-password/);
    expect(r.sql).not.toContain("create user \"postgres_ai_mon\"");
  });

  test("new role, --json --print-password: the password is returned", () => {
    const r = prepareDb(["--json", "--print-password"], false);
    expect(r.status).toBe(0);
    const out = JSON.parse(r.stdout);
    expect(typeof out.generatedPassword).toBe("string");
    expect(out.passwordChanged).toBe(true);
  });
});

describe("bug 4: prepare-db <url> --print-sql needs no monitoring password", () => {
  test("non-TTY, no password: prints the plan with the password redacted", () => {
    const r = prepareDb(["--print-sql"], false);
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("--- SQL plan ---");
    expect(r.stdout).not.toContain("SCRAM-SHA-256$");
    expect(r.stderr).not.toContain("auto-generated");
  });
});

/** A server that asks for SCRAM and then waits: what a real Postgres does until authentication_timeout. */
function scramServer(): Promise<{ port: number; closed: Promise<void>; stop: () => void }> {
  return new Promise((resolve) => {
    let onClosed!: () => void;
    const closed = new Promise<void>((r) => { onClosed = r; });
    const sockets: net.Socket[] = [];
    const server = net.createServer((socket) => {
      sockets.push(socket);
      let step = 0;
      socket.on("data", () => {
        if (step === 0) {
          // Startup message -> AuthenticationSASL (SCRAM-SHA-256).
          const mech = Buffer.from("SCRAM-SHA-256\0\0");
          const msg = Buffer.alloc(9 + mech.length);
          msg.write("R", 0); msg.writeInt32BE(8 + mech.length, 1); msg.writeInt32BE(10, 5); mech.copy(msg, 9);
          socket.write(msg);
        } else if (step === 1) {
          // SASLInitialResponse -> AuthenticationSASLContinue.
          const data = Buffer.from("r=abcdefghijklmnopqrstuvwxyz0123456789,s=c2FsdA==,i=4096");
          const msg = Buffer.alloc(9 + data.length);
          msg.write("R", 0); msg.writeInt32BE(8 + data.length, 1); msg.writeInt32BE(11, 5); data.copy(msg, 9);
          socket.write(msg);
        }
        step++;
      });
      socket.on("close", () => onClosed());
      socket.on("error", () => {});
    });
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as net.AddressInfo).port;
      resolve({ port, closed, stop: () => { for (const s of sockets) s.destroy(); server.close(); } });
    });
  });
}

describe("bug 5: no admin password fails fast with a clear message", () => {
  test("the error names the fix and the connection is closed at once", async () => {
    const { Client } = await import("pg");
    const srv = await scramServer();
    const saved = { PGPASSWORD: process.env.PGPASSWORD, PGPASSFILE: process.env.PGPASSFILE };
    delete process.env.PGPASSWORD;
    process.env.PGPASSFILE = "/nonexistent/pgpass-529";
    try {
      const adminConn = init.resolveAdminConnection({ conn: `postgresql://admin@127.0.0.1:${srv.port}/appdb?sslmode=disable` });
      const err = await init.connectWithSslFallback(Client as any, adminConn, false, () => {}).then(
        () => null,
        (e: Error) => e,
      );
      expect(err).not.toBeNull();
      expect(err!.message).not.toContain("SASL");
      expect(err!.message).toMatch(/PGPASSWORD/);
      const closedInTime = await Promise.race([
        srv.closed.then(() => true),
        new Promise<boolean>((r) => setTimeout(() => r(false), 2000)),
      ]);
      expect(closedInTime).toBe(true);
    } finally {
      if (saved.PGPASSWORD === undefined) delete process.env.PGPASSWORD; else process.env.PGPASSWORD = saved.PGPASSWORD;
      if (saved.PGPASSFILE === undefined) delete process.env.PGPASSFILE; else process.env.PGPASSFILE = saved.PGPASSFILE;
      srv.stop();
    }
  });
});

describe("bug 6: checkup --json failure and exit codes", () => {
  test("a failed run prints a JSON error object on stdout and exits 1", () => {
    const r = runCli(["checkup", DB_URL, "--no-upload", "--json"]);
    expect(r.status).toBe(1);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(false);
    expect(out.error.message).toContain("Database unavailable");
  });

  test("checkup --help documents the exit codes", () => {
    const r = runCli(["checkup", "--help"]);
    expect(r.stdout).toMatch(/Exit codes:/);
    expect(r.stdout).toMatch(/0\s+.*completed/);
    expect(r.stdout).toMatch(/1\s+.*error/i);
  });
});

/** Like runCli, but async: the test process can serve a fake API meanwhile. */
async function runCliAsync(args: string[], env: NodeJS.ProcessEnv = {}) {
  sandbox ??= createCliSandbox();
  const childEnv: NodeJS.ProcessEnv = { ...process.env };
  for (const name of Object.keys(childEnv)) if (name.startsWith("PGAI_")) delete childEnv[name];
  const proc = Bun.spawn([
    process.execPath, "--preload", path.join(import.meta.dir, "cli-offline-preload.ts"), sandbox.cliPath, ...args,
  ], {
    cwd: sandbox.projectDir,
    env: {
      ...childEnv, PGPASSWORD: "", ...env,
      HOME: sandbox.home, XDG_CONFIG_HOME: sandbox.configHome,
      PATH: `${sandbox.binDir}${path.delimiter}${process.env.PATH || ""}`,
    },
    stdout: "pipe", stderr: "pipe",
  });
  const killer = setTimeout(() => proc.kill(), 10000);
  try {
    const [status, stdout, stderr] = await Promise.all([
      proc.exited, new Response(proc.stdout).text(), new Response(proc.stderr).text(),
    ]);
    return { status, stdout, stderr };
  } finally {
    clearTimeout(killer);
  }
}

/** A fake PostgresAI API: the key check (GET) answers getStatus, every RPC (POST) fails with 400. */
async function withFakeApi<T>(getStatus: number, run: (origin: string) => Promise<T>): Promise<T> {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(req) {
    if (req.method === "GET") return Response.json([], { status: getStatus });
    return Response.json({ message: "synthetic rpc failure" }, { status: 400 });
  } });
  try { return await run(server.url.origin); } finally { server.stop(true); }
}

function jsonError(r: { status: number | null; stdout: string }): { message: string } {
  expect(r.status).toBe(1);
  const out = JSON.parse(r.stdout);
  expect(out.success).toBe(false);
  expect(typeof out.error.message).toBe("string");
  return out.error;
}

describe("checkup --json: every failure path prints the JSON error object", () => {
  const FAKE_DB = { PGAI_TEST_PG_FAKE: "1" };

  test("output directory that cannot be created", () => {
    sandbox ??= createCliSandbox();
    // A path under a dangling symlink: mkdir fails with ENOENT, also for root.
    fs.symlinkSync(path.join(sandbox.root, "missing-target"), path.join(sandbox.root, "dangling"));
    const out = path.join(sandbox.root, "dangling", "sub");
    const r = runCli(["checkup", DB_URL, "--no-upload", "--json", "--output", out]);
    expect(jsonError(r).message).toContain("Failed to create output directory");
  });

  test("--upload with no API key", () => {
    const r = runCli(["checkup", DB_URL, "--upload", "--json"]);
    expect(jsonError(r).message).toContain("API key is required for upload");
  });

  test("the API rejects the key (HTTP 401)", async () => {
    await withFakeApi(401, async (origin) => {
      const r = await runCliAsync(["checkup", DB_URL, "--json"], {
        PGAI_API_KEY: "synthetic-api-key", PGAI_API_BASE_URL: origin, PGAI_TEST_API_URL: origin,
      });
      expect(jsonError(r).message).toContain("API key was rejected");
    });
  });

  test("the database user lacks required permissions", () => {
    const r = runCli(["checkup", DB_URL, "--no-upload", "--json"], { ...FAKE_DB, PGAI_TEST_PG_PERMS_MISSING: "1" });
    expect(jsonError(r).message).toContain("missing required permissions");
  });

  const notExpress = getAllCheckupCodes().find((c) => /^[A-Z]\d{3}$/.test(c) && !(c in checkup.REPORT_GENERATORS));
  test.skipIf(!notExpress)("a dictionary check that express mode does not run", () => {
    const r = runCli(["checkup", notExpress!, DB_URL, "--no-upload", "--json"], FAKE_DB);
    expect(jsonError(r).message).toContain("not yet available in express mode");
  });

  test("an unknown check ID", () => {
    const r = runCli(["checkup", "Z999", DB_URL, "--no-upload", "--json"], FAKE_DB);
    expect(jsonError(r).message).toBe("Unknown check ID: Z999");
  });

  test("the upload fails (RPC error)", async () => {
    await withFakeApi(200, async (origin) => {
      const r = await runCliAsync(["checkup", "A002", DB_URL, "--json"], {
        ...FAKE_DB, PGAI_API_KEY: "synthetic-api-key", PGAI_API_BASE_URL: origin, PGAI_TEST_API_URL: origin,
      });
      const err = jsonError(r);
      expect(err.message).toContain("HTTP 400");
      expect(err.message).toContain("synthetic rpc failure");
    });
  });
});

describe("bug 11: checkup --help titles", () => {
  test("F009 and I001 have their real titles", () => {
    expect(checkup.CHECK_INFO.F009).toBe("Xmin horizon and blockers");
    expect(checkup.CHECK_INFO.I001).toBe("I/O statistics (pg_stat_io)");
    for (const [id, title] of Object.entries(checkup.CHECK_INFO)) expect(title).not.toBe(id);
  });
});

describe("bug 7: F003 wording", () => {
  const table = (extra: Record<string, unknown> = {}) => ({
    schema_name: "public", table_name: "orders", n_live_tup: 1_000_000, n_dead_tup: 500_000, dead_pct: 33,
    last_autovacuum: null, last_autovacuum_epoch: 0, last_vacuum: null, last_vacuum_epoch: 0,
    autovacuum_count: 0, vacuum_count: 0, autovacuum_disabled: false, table_size_bytes: 1, table_size_pretty: "1 B",
    exceeds_dead_tuple_thresholds: true, autovacuum_disabled_flagged: false,
    over_insert_trigger: true, n_ins_since_vacuum: 2_000_000, insert_trigger_point: 1000, ...extra,
  });
  const keepup = (extra: Record<string, unknown> = {}) => checkup.judgeKeepingUp(
    { queue_length: 1, analyze_queue_length: 0, insert_queue_length: 0 } as any,
    { active_workers: 0, max_workers: 3, free_slots: 3, anti_wraparound_workers: 0, ...extra } as any,
    [], [], 0,
  );

  test("a table past its insert trigger is not called append-only", () => {
    const { conclusions } = checkup.buildKeepupConclusions([table() as any], keepup());
    expect(conclusions.join("\n")).toContain('"public"."orders" is past its autovacuum insert trigger');
    expect(conclusions.join("\n")).not.toMatch(/append-only/i);
  });

  test("with no busy worker the queue is not 'being worked'", () => {
    expect(keepup().judgment).not.toContain("being worked");
    expect(keepup().judgment).toContain("; no autovacuum worker is processing it right now");
    expect(keepup({ active_workers: 2, free_slots: 1 }).judgment).toContain("being worked");
  });

  const f003Report = (k: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
    results: { node1: { data: { db1: {
      dead_tuples_tables: [], total_count: 1, flagged_count: 0,
      autovacuum_disabled_count: 0, autovacuum_disabled_flagged_count: 0,
      autovacuum_keepup: { queue_length: 1, saturated: false, chronic_under_provisioning: false,
        anti_wraparound_present: false, blocked_workers: [], ...k },
      ...extra,
    } } } },
  });

  test("summary: no busy worker, or autovacuum off on a table, is not 'being worked'", () => {
    const idle = summary.generateCheckSummary("F003", f003Report({ active_workers: 0 })).message;
    expect(idle).not.toContain("being worked");
    expect(idle).toContain("1 table past vacuum trigger, no autovacuum worker busy");
    const disabled = f003Report({ active_workers: 2 }, { flagged_count: 1, autovacuum_disabled_count: 1, autovacuum_disabled_flagged_count: 1 });
    expect(summary.generateCheckSummary("F003", disabled).message).not.toContain("being worked");
    expect(summary.generateCheckSummary("F003", f003Report({ active_workers: 2 })).message).toContain("being worked");
  });
});

describe("bug 9: H002 notes recent usage stats", () => {
  const now = Math.floor(Date.now() / 1000);
  const h002 = (stats: Record<string, unknown>) => ({
    results: { node1: { data: { db1: { unused_indexes: [], total_count: 1, total_size_bytes: 8192, stats_reset: stats } } } },
  });

  test("stats from a server started minutes ago get a note", () => {
    const r = summary.generateCheckSummary("H002", h002({ stats_reset_epoch: null, postmaster_startup_epoch: now - 600 }));
    expect(r.message).toMatch(/Found 1 unused index/);
    expect(r.message).toMatch(/usage stats.*10 min/i);
  });

  test("stats reset two hours ago get a note", () => {
    const r = summary.generateCheckSummary("H002", h002({ stats_reset_epoch: now - 7200, postmaster_startup_epoch: now - 90 * 86400 }));
    expect(r.message).toMatch(/usage stats.*2 h.*reset/i);
  });

  test("old stats get no note", () => {
    const r = summary.generateCheckSummary("H002", h002({ stats_reset_epoch: now - 60 * 86400, postmaster_startup_epoch: now - 90 * 86400 }));
    expect(r.message).toBe("Found 1 unused index (8 KiB)");
  });
});

describe("bug 10 / N6: D004 summary counts the settings in the report", () => {
  test("five settings under data.settings are five", () => {
    const settings = Object.fromEntries(
      ["max", "track", "track_utility", "save", "track_planning"].map((n) => [`pg_stat_statements.${n}`, { value: "x" }]),
    );
    const report = { results: { node1: { data: {
      settings,
      pg_stat_statements_status: { extension_available: true },
      pg_stat_kcache_status: { extension_available: false },
    } } } };
    expect(summary.generateCheckSummary("D004", report).message).toBe("5 settings collected");
  });
});

// Review round (#529).
describe("B1: a new role whose later step fails is dropped again", () => {
  const FAIL_02 = "Extensions required";

  test("--json --print-password, 02 fails: the new role is dropped; the re-run generates and returns a password", () => {
    const first = prepareDb(["--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: FAIL_02 });
    expect(first.status).toBe(1);
    const err = JSON.parse(first.stdout);
    expect(err.success).toBe(false);
    expect(first.sql).toMatch(/drop role "postgres_ai_mon"/);
    // Dropped: nobody needs the password.
    expect(err.generatedPassword).toBeUndefined();
    sandbox?.cleanup(); sandbox = null;
    const again = prepareDb(["--json", "--print-password"], false);
    expect(again.status).toBe(0);
    const out = JSON.parse(again.stdout);
    expect(typeof out.generatedPassword).toBe("string");
    expect(out.passwordChanged).toBe(true);
  });

  test("the drop fails too: the JSON error carries the password and passwordChanged", () => {
    const r = prepareDb(["--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: `${FAIL_02}|drop role` });
    expect(r.status).toBe(1);
    const err = JSON.parse(r.stdout);
    expect(err.success).toBe(false);
    expect(typeof err.generatedPassword).toBe("string");
    expect(err.passwordChanged).toBe(true);
  });

  test("an existing role is never dropped when a later step fails", () => {
    const r = prepareDb(["--json"], true, { PGAI_TEST_PG_FAIL: FAIL_02 });
    expect(r.status).toBe(1);
    expect(r.sql).not.toMatch(/drop role/);
  });
});

// --supabase has its own copy of the password and role-drop handling. The preload
// answers the Management API query endpoint from the same fake database.
describe("prepare-db --supabase: password and role-drop handling", () => {
  const SUPABASE = ["--supabase", "--supabase-project-ref", "abcdefghij0123456789"];
  const prepareSupabase = (args: string[], roleExists: boolean, env: NodeJS.ProcessEnv = {}) =>
    prepareDb([...SUPABASE, ...args], roleExists, { SUPABASE_ACCESS_TOKEN: "synthetic-token", ...env });

  test("existing role, no password, --json: the password is not changed", () => {
    const r = prepareSupabase(["--json"], true);
    expect(r.status).toBe(0);
    expect(r.sql).toContain("Role creation");
    expect(r.sql).not.toMatch(ALTER_PASSWORD);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(true);
    expect(out.mode).toBe("supabase");
    expect(out.generatedPassword).toBeUndefined();
    expect(out.passwordChanged).toBe(false);
  });

  test("new role, --json, non-TTY: no password on stdout, JSON error, nothing applied", () => {
    const r = prepareSupabase(["--json"], false);
    expect(r.stdout).not.toContain("generatedPassword");
    expect(r.status).toBe(1);
    const out = JSON.parse(r.stdout);
    expect(out.success).toBe(false);
    expect(out.error.message).toMatch(/--print-password/);
    expect(r.sql).not.toContain('create user "postgres_ai_mon"');
  });

  test("new role, a later step fails: the new role is dropped, no password in the error", () => {
    const r = prepareSupabase(["--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: "Extensions required" });
    expect(r.status).toBe(1);
    expect(r.sql).toMatch(/drop role "postgres_ai_mon"/);
    const err = JSON.parse(r.stdout);
    expect(err.success).toBe(false);
    expect(err.mode).toBe("supabase");
    expect(err.generatedPassword).toBeUndefined();
  });

  test("the drop fails too: the JSON error carries the password and passwordChanged", () => {
    const r = prepareSupabase(["--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: "Extensions required|drop role" });
    expect(r.status).toBe(1);
    const err = JSON.parse(r.stdout);
    expect(err.success).toBe(false);
    expect(typeof err.generatedPassword).toBe("string");
    expect(err.passwordChanged).toBe(true);
  });

  test("an existing role is never dropped when a later step fails", () => {
    const r = prepareSupabase(["--json"], true, { PGAI_TEST_PG_FAIL: "Extensions required" });
    expect(r.status).toBe(1);
    expect(r.sql).not.toMatch(/drop role/);
  });
});

describe("B2: the role step checks what prepare-db saw", () => {
  test("existing role kept: the role step fails if the role is gone, and never creates it", () => {
    const r = prepareDb([], true);
    expect(r.status).toBe(0);
    expect(r.sql).toContain("was dropped during prepare-db");
    expect(r.sql).not.toContain('create user "postgres_ai_mon"');
  });

  test("new role with a generated password: the role step fails if another session created it", () => {
    const r = prepareDb(["--json", "--print-password"], false);
    expect(r.status).toBe(0);
    expect(r.sql).toContain("was created by another session during prepare-db");
    expect(r.sql).not.toMatch(ALTER_PASSWORD);
  });

  test("pgai mon deploy's plan (keepExistingPassword only) is unchanged: create if missing", async () => {
    const plan = await init.buildInitPlan({
      database: "appdb", monitoringPassword: "synthetic-pw", includeOptionalPermissions: false, keepExistingPassword: true,
    });
    const role = plan.steps.find((s) => s.name === "01.role")!.sql;
    expect(role).toContain('create user "postgres_ai_mon"');
    expect(role).not.toContain("during prepare-db");
  });
});

describe("N1: the kept-password note does not promise a password", () => {
  test("it says the password is kept as is", () => {
    const r = prepareDb([], true);
    expect(r.stdout + r.stderr).toMatch(/kept as is/);
  });
});

describe("N2: F003 wording with unknown workers or autovacuum off", () => {
  const agg = { queue_length: 1, analyze_queue_length: 0, insert_queue_length: 0 } as any;

  test("failed worker snapshot (max_workers null): neither 'being worked' nor 'no worker'", () => {
    const j = checkup.judgeKeepingUp(agg, { active_workers: 0, max_workers: null, free_slots: null, anti_wraparound_workers: 0 }, [], [], 0).judgment;
    expect(j).not.toContain("being worked");
    expect(j).not.toContain("no autovacuum worker");
  });

  test("a queued table with autovacuum disabled: not 'being worked'", () => {
    const j = checkup.judgeKeepingUp(agg, { active_workers: 2, max_workers: 3, free_slots: 1, anti_wraparound_workers: 0 }, [], [], 0, 1).judgment;
    expect(j).not.toContain("being worked");
    expect(j).toContain("autovacuum disabled");
  });

  test("summary with max_workers null: neither phrase", () => {
    const report = { results: { node1: { data: { db1: {
      dead_tuples_tables: [], total_count: 1, flagged_count: 0, autovacuum_disabled_count: 0, autovacuum_disabled_flagged_count: 0,
      autovacuum_keepup: { queue_length: 1, saturated: false, chronic_under_provisioning: false, anti_wraparound_present: false,
        blocked_workers: [], active_workers: 0, max_workers: null },
    } } } } };
    const m = summary.generateCheckSummary("F003", report).message;
    expect(m).not.toContain("being worked");
    expect(m).not.toContain("no autovacuum worker");
  });
});

describe("N5: closing the socket never replaces the connect error", () => {
  test("an end() that throws leaves the original error", async () => {
    class Broken {
      constructor(_c: unknown) {}
      async connect() { throw new Error("original connect failure"); }
      end(): never { throw new Error("end exploded"); }
      async query() { return { rows: [] }; }
    }
    const adminConn = init.resolveAdminConnection({ conn: "postgresql://admin@127.0.0.1:1/appdb?sslmode=disable" });
    const err = await init.connectWithSslFallback(Broken as any, adminConn, false, () => {}).then(() => null, (e: Error) => e);
    expect(err?.message).toBe("original connect failure");
  });
});

describe("B1': no role step, no role of this run", () => {
  const FAIL_03 = "Required permissions for postgres_ai monitoring user";

  test("--provider supabase, 03 fails: no drop role, no generated password, passwordChanged not true", () => {
    const r = prepareDb(["--provider", "supabase", "--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: FAIL_03 });
    expect(r.status).toBe(1);
    expect(r.sql).not.toMatch(/drop role/);
    const err = JSON.parse(r.stdout);
    expect(err.success).toBe(false);
    expect(err.generatedPassword).toBeUndefined();
    expect(err.passwordChanged).not.toBe(true);
  });

  test("--provider supabase success: no generated password, passwordChanged false", () => {
    const r = prepareDb(["--provider", "supabase", "--json", "--print-password"], false);
    expect(r.status).toBe(0);
    const out = JSON.parse(r.stdout);
    expect(out.generatedPassword).toBeUndefined();
    expect(out.passwordChanged).toBe(false);
  });

  test("a failing role step itself drops nothing", () => {
    const r = prepareDb(["--json", "--print-password"], false, { PGAI_TEST_PG_FAIL: "Role creation" });
    expect(r.status).toBe(1);
    expect(r.sql).not.toMatch(/drop role/);
    expect(JSON.parse(r.stdout).generatedPassword).toBeUndefined();
  });
});
