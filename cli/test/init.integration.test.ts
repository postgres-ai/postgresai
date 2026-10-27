/**
 * Integration tests for prepare-db command
 * Requires PostgreSQL binaries (initdb, postgres) to be available
 * These tests spin up a temporary PostgreSQL instance for realistic testing
 */
import { describe, test, expect, afterAll } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as net from "net";
import { Client } from "pg";
import { prepareDatabase } from "../lib/connect";
import { buildInitPlan } from "../lib/init";
import { applyInitPlanViaSupabase, type SupabaseClient } from "../lib/supabase";

const TEST_TIMEOUT = 30000; // 30 seconds

function sqlLiteral(value: string): string {
  return `'${String(value).replace(/'/g, "''")}'`;
}

function findOnPath(cmd: string): string | null {
  const result = Bun.spawnSync(["sh", "-c", `command -v ${cmd}`]);
  if (result.exitCode === 0) {
    return new TextDecoder().decode(result.stdout).trim();
  }
  return null;
}

function findPgBin(cmd: string): string | null {
  const p = findOnPath(cmd);
  if (p) return p;

  // Debian/Ubuntu (GitLab CI node:*-bullseye images): binaries usually live here.
  const probe = Bun.spawnSync([
    "sh",
    "-c",
    `ls -1 /usr/lib/postgresql/*/bin/${cmd} 2>/dev/null | head -n 1 || true`,
  ]);
  const out = new TextDecoder().decode(probe.stdout).trim();
  if (out) return out;

  return null;
}

/** First directory with initdb and postgres of at least `minMajor`: PATH, then Debian/Homebrew layouts. */
function findPgBinDirAtLeast(minMajor: number): string | null {
  const onPath = findPgBin("initdb");
  const probe = Bun.spawnSync([
    "sh",
    "-c",
    "ls -1d /usr/lib/postgresql/*/bin /opt/homebrew/opt/postgresql@*/bin 2>/dev/null || true",
  ]);
  const dirs = [
    ...(onPath ? [path.dirname(onPath)] : []),
    ...new TextDecoder().decode(probe.stdout).split("\n").filter(Boolean),
  ];
  for (const dir of dirs) {
    const initdb = path.join(dir, "initdb");
    if (!fs.existsSync(initdb) || !fs.existsSync(path.join(dir, "postgres"))) continue;
    const version = new TextDecoder().decode(Bun.spawnSync([initdb, "--version"]).stdout);
    const major = Number(version.match(/\(PostgreSQL\) (\d+)/)?.[1] ?? 0);
    if (major >= minMajor) return dir;
  }
  return null;
}

function havePostgresBinaries(): boolean {
  return !!(findPgBin("initdb") && findPgBin("postgres"));
}

function isRunningAsRoot(): boolean {
  return process.getuid?.() === 0;
}

async function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address() as net.AddressInfo;
      srv.close((err) => {
        if (err) return reject(err);
        resolve(addr.port);
      });
    });
    srv.on("error", reject);
  });
}

async function waitFor<T>(
  fn: () => Promise<T>,
  { timeoutMs = 10000, intervalMs = 100 } = {}
): Promise<T> {
  const start = Date.now();
  while (true) {
    try {
      return await fn();
    } catch (e) {
      if (Date.now() - start > timeoutMs) throw e;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  }
}

interface TempPostgres {
  port: number;
  socketDir: string;
  adminUri: string;
  postgresPassword: string;
  cleanup: () => Promise<void>;
}

async function createTempPostgres(binDir?: string): Promise<TempPostgres> {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "postgresai-init-"));
  const dataDir = path.join(tmpRoot, "data");
  const socketDir = path.join(tmpRoot, "sock");
  fs.mkdirSync(socketDir, { recursive: true });

  const initdb = binDir ? path.join(binDir, "initdb") : findPgBin("initdb");
  const postgresBin = binDir ? path.join(binDir, "postgres") : findPgBin("postgres");
  if (!initdb || !postgresBin) {
    throw new Error("PostgreSQL binaries not found (need initdb and postgres)");
  }

  const init = Bun.spawnSync([initdb, "-D", dataDir, "-U", "postgres", "-A", "trust"]);
  if (init.exitCode !== 0) {
    throw new Error(new TextDecoder().decode(init.stderr) || new TextDecoder().decode(init.stdout));
  }

  // Configure: local socket trust, TCP scram.
  const hbaPath = path.join(dataDir, "pg_hba.conf");
  fs.writeFileSync(
    hbaPath,
    "\n# Added by postgresai init integration tests\nlocal all all trust\nhost all all 127.0.0.1/32 scram-sha-256\nhost all all ::1/128 scram-sha-256\n",
    "utf8"
  );

  const port = await getFreePort();

  const postgresProc = Bun.spawn(
    [postgresBin, "-D", dataDir, "-k", socketDir, "-h", "127.0.0.1", "-p", String(port)],
    { stdio: ["ignore", "pipe", "pipe"] }
  );

  const cleanup = async () => {
    postgresProc.kill("SIGTERM");
    try {
      // 30s timeout to handle slower CI environments gracefully
      await waitFor(
        async () => {
          if (postgresProc.exitCode === null) throw new Error("still running");
        },
        { timeoutMs: 30000, intervalMs: 100 }
      );
    } catch {
      postgresProc.kill("SIGKILL");
    }
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  };

  const connectLocal = async (database = "postgres"): Promise<Client> => {
    const c = new Client({ host: socketDir, port, user: "postgres", database });
    await c.connect();
    return c;
  };

  // Wait for Postgres to start (30s timeout for slower CI environments)
  await waitFor(async () => {
    const c = await connectLocal();
    await c.end();
  }, { timeoutMs: 30000, intervalMs: 100 });

  const postgresPassword = "postgrespw";
  {
    const c = await connectLocal();
    await c.query(`alter user postgres password ${sqlLiteral(postgresPassword)};`);
    await c.query("create database testdb");
    await c.end();
  }

  const adminUri = `postgresql://postgres:${postgresPassword}@127.0.0.1:${port}/testdb`;
  return { port, socketDir, adminUri, postgresPassword, cleanup };
}

function runCliInit(
  args: string[],
  env: Record<string, string> = {}
): { status: number | null; stdout: string; stderr: string } {
  return runCli("prepare-db", args, env);
}

function runCli(
  command: string,
  args: string[],
  env: Record<string, string> = {}
): { status: number | null; stdout: string; stderr: string } {
  const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const result = Bun.spawnSync(["bun", cliPath, command, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

// Skip all tests if PostgreSQL binaries are not available or running as root
// (initdb cannot be run as root)
const skipTests = !havePostgresBinaries() || isRunningAsRoot();

// CI sets PGAI_TEST_REQUIRE_PG16=1: fail loudly instead of skipping the suite or
// the SCRAM iteration test when the required binaries are missing.
if (process.env.PGAI_TEST_REQUIRE_PG16 === "1" && (skipTests || !findPgBinDirAtLeast(16))) {
  throw new Error("PGAI_TEST_REQUIRE_PG16=1 but the integration suite cannot run PostgreSQL >= 16 here");
}

describe.skipIf(skipTests)("integration: prepare-db", () => {
  let pg: TempPostgres;

  // Use a shared postgres instance for all tests in this describe block
  // Each test will reset state as needed

  test("mon deploy setup ignores public.format and installs pg_stat_statements in public", async () => {
    pg = await createTempPostgres();
    try {
      const c = new Client({ connectionString: pg.adminUri });
      await c.connect();
      try {
        await c.query("alter database testdb set search_path = public, pg_catalog");
        await c.query(`create function public.format(text, text) returns text language plpgsql as $$
begin
  raise exception 'public.format called';
end;
$$`);
      } finally {
        await c.end();
      }
      const prepared = await prepareDatabase(`${pg.adminUri}?sslmode=disable`, "self-managed", { agent: true });
      expect(prepared).toHaveProperty("monitoringUrl");
      const check = new Client({ connectionString: pg.adminUri });
      await check.connect();
      try {
        expect((await check.query(`select n.nspname from pg_catalog.pg_extension e
join pg_catalog.pg_namespace n on n.oid = e.extnamespace
where e.extname = 'pg_stat_statements'`)).rows).toEqual([{ nspname: "public" }]);
      } finally {
        await check.end();
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: 60000 });

  // internal#384: a database owner plants exact-match overloads in public. With
  // the server's default search_path they win over the pg_catalog built-ins and
  // run as the admin. adminUri has no sslmode, so the CLI takes the plaintext
  // retry after the server refuses TLS.
  test("prepare-db, checkup and unprepare-db never call overloads planted in public", async () => {
    pg = await createTempPostgres();
    const configHome = fs.mkdtempSync(path.join(os.tmpdir(), "postgresai-cfg-"));
    try {
      const c = new Client({ connectionString: pg.adminUri });
      await c.connect();
      try {
        for (const fn of ["format(text, text)", "quote_ident(name)"]) {
          await c.query(`create function public.${fn} returns text language plpgsql as $$
begin
  raise exception 'planted public overload called';
end;
$$`);
        }
      } finally {
        await c.end();
      }

      const env = { XDG_CONFIG_HOME: configHome };
      const runs: Array<[string, string[]]> = [
        ["prepare-db", [pg.adminUri, "--password", "monpw", "--skip-optional-permissions"]],
        ["checkup", [pg.adminUri, "--no-upload", "--json"]],
        ["unprepare-db", [pg.adminUri, "--force"]],
      ];
      const results = runs.map(([command, args]) => {
        const r = runCli(command, args, env);
        return { command, status: r.status, planted: `${r.stdout}${r.stderr}`.includes("planted public overload called") };
      });
      expect(results).toEqual(runs.map(([command]) => ({ command, status: 0, planted: false })));
    } finally {
      fs.rmSync(configHome, { recursive: true, force: true });
      await pg.cleanup();
    }
  }, { timeout: 120000 });

  test("supports URI / conninfo / psql-like connection styles", async () => {
    pg = await createTempPostgres();

    try {
      // 1) positional URI
      {
        const r = runCliInit([pg.adminUri, "--password", "monpw", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
      }

      // 2) conninfo
      {
        const conninfo = `dbname=testdb host=127.0.0.1 port=${pg.port} user=postgres password=${pg.postgresPassword}`;
        const r = runCliInit([conninfo, "--password", "monpw2", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
      }

      // 3) psql-like options (+ PGPASSWORD)
      {
        const r = runCliInit(
          [
            "-h", "127.0.0.1",
            "-p", String(pg.port),
            "-U", "postgres",
            "-d", "testdb",
            "--password", "monpw3",
            "--skip-optional-permissions",
          ],
          { PGPASSWORD: pg.postgresPassword }
        );
        expect(r.status).toBe(0);
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("init and uninit safely quote unusual monitoring role names", async () => {
    pg = await createTempPostgres();
    const admin = new Client({ connectionString: pg.adminUri });
    // Quote, backslash and dollar-quote delimiters; then multi-byte Unicode (Cyrillic, CJK, combining mark).
    const roles: Array<[string, string]> = [
      [String.raw`pgai'audit\\$$$pgai_do_0$`, "do $pgai_do_1$"],
      ["pgai_мониторинг_監視_e\u0301", "do $pgai_do_0$"],
    ];
    try {
      await admin.connect();
      await admin.query("set standard_conforming_strings = off");
      const { buildInitPlan, applyInitPlan, buildUninitPlan, applyUninitPlan } = await import("../lib/init");
      for (const [role, doTag] of roles) {
        const plan = await buildInitPlan({
          database: "testdb", monitoringUser: role,
          monitoringPassword: "temporary-test-only", includeOptionalPermissions: false,
        });
        expect(plan.steps[0].sql).toContain(doTag);
        await applyInitPlan({ client: admin, plan });
        const created = await admin.query("select count(*)::int as n from pg_catalog.pg_roles where rolname = $1", [role]);
        expect(created.rows[0].n).toBe(1);
        const uninit = await buildUninitPlan({ database: "testdb", monitoringUser: role });
        const cleanup = await applyUninitPlan({ client: admin, plan: uninit });
        expect(cleanup.errors).toEqual([]);
        const remaining = await admin.query("select count(*)::int as n from pg_catalog.pg_roles where rolname = $1", [role]);
        expect(remaining.rows[0].n).toBe(0);
      }
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("prepare-db refuses a precreated monitoring schema and foreign-owned helper", async () => {
    pg = await createTempPostgres();
    const admin = new Client({ connectionString: pg.adminUri });
    try {
      await admin.connect();
      await admin.query("create role pgai_schema_squatter");
      await admin.query("grant create on database testdb to pgai_schema_squatter");
      await admin.query("set role pgai_schema_squatter");
      await admin.query("create schema postgres_ai");
      await admin.query("create function postgres_ai.table_describe(text) returns text language sql as 'select current_user::text'");
      await admin.query("reset role");

      const { buildInitPlan, applyInitPlan } = await import("../lib/init");
      const plan = await buildInitPlan({
        database: "testdb", monitoringPassword: "temporary-test-only", includeOptionalPermissions: false,
      });
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/schema must be owned by/);
      const noMonitorGrant = await admin.query("select not pg_has_role('postgres_ai_mon', 'pg_monitor', 'member') as safe");
      expect(noMonitorGrant.rows[0].safe).toBe(true);

      await admin.query("alter schema postgres_ai owner to postgres");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/objects owned by another role/);
      await admin.query("alter function postgres_ai.table_describe(text) owner to postgres");

      await admin.query("grant create on schema postgres_ai to pgai_schema_squatter");
      await admin.query("set role pgai_schema_squatter");
      await admin.query("create view postgres_ai.foreign_view as select 1 as n");
      await admin.query("reset role");
      await admin.query("revoke create on schema postgres_ai from pgai_schema_squatter");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/objects owned by another role/);
      await admin.query("drop view postgres_ai.foreign_view");

      await admin.query("grant create on schema postgres_ai to pgai_schema_squatter");
      await admin.query("set role pgai_schema_squatter");
      await admin.query("create operator postgres_ai.## (leftarg = integer, rightarg = integer, procedure = pg_catalog.int4pl)");
      await admin.query("reset role");
      await admin.query("revoke create on schema postgres_ai from pgai_schema_squatter");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/objects owned by another role/);
      await admin.query("drop operator postgres_ai.## (integer, integer)");

      // Predefined role owners do not get pg_shdepend ownership rows. A
      // preexisting function owned by pg_database_owner must still be refused.
      await admin.query("create function postgres_ai.pinned_owner_probe() returns integer language sql as 'select 1'");
      await admin.query("alter function postgres_ai.pinned_owner_probe() owner to pg_database_owner");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/objects owned by another role/);
      const noPinnedGrant = await admin.query("select not pg_has_role('postgres_ai_mon', 'pg_monitor', 'member') as safe");
      expect(noPinnedGrant.rows[0].safe).toBe(true);
      await admin.query("drop function postgres_ai.pinned_owner_probe()");

      // Extension members may be owned by bootstrap superuser oid 10 and
      // therefore have no pg_shdepend owner row. The schema itself must still
      // reject an extension in the monitoring role's search path.
      await admin.query("create extension pg_trgm with schema postgres_ai");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/schema contains an extension/);
      await admin.query("drop extension pg_trgm");

      await admin.query("grant create on schema postgres_ai to pgai_schema_squatter");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/grants CREATE to another role/);
      await admin.query("revoke create on schema postgres_ai from pgai_schema_squatter");
      await admin.query("grant create on schema postgres_ai to public");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/grants CREATE to another role/);
      await admin.query("revoke create on schema postgres_ai from public");
      await applyInitPlan({ client: admin, plan });
      const owner = await admin.query("select pg_get_userbyid(proowner) as owner from pg_proc where oid = 'postgres_ai.table_describe(text)'::regprocedure");
      expect(owner.rows[0].owner).toBe("postgres");

      await admin.query("drop schema postgres_ai cascade");
      await admin.query("create schema postgres_ai authorization pg_database_owner");
      await expect(applyInitPlan({ client: admin, plan })).rejects.toThrow(/schema must be owned by/);
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("role SQL authenticates with SCRAM even when password_encryption is md5", async () => {
    pg = await createTempPostgres();
    const admin = new Client({ connectionString: pg.adminUri });
    try {
      await admin.connect();
      await admin.query("set password_encryption = 'md5'");
      const { buildInitPlan, applyInitPlan, redactPasswordsInSql } = await import("../lib/init");
      // Exercise both first creation and updating an existing role.
      for (const password of ["test-only-initial", String.raw`test-only-pa'ss\word$$!`]) {
        const plan = await buildInitPlan({ database: "testdb", monitoringPassword: password, includeOptionalPermissions: false });
        plan.steps = plan.steps.filter(step => step.name === "01.role");
        const sql = plan.steps[0].sql;
        expect(sql.includes(password)).toBe(false);
        const verifiers = [...sql.matchAll(/password '(SCRAM-SHA-256\$[^']+)'/g)].map(m => m[1]);
        expect(verifiers.length).toBe(2);
        expect(verifiers[0] === verifiers[1]).toBe(true);
        expect(redactPasswordsInSql(sql).includes(verifiers[0])).toBe(false);
        await applyInitPlan({ client: admin, plan });
        const stored = await admin.query("select rolpassword from pg_authid where rolname = 'postgres_ai_mon'");
        expect(stored.rows[0].rolpassword === verifiers[0]).toBe(true);
        const config = { host: "127.0.0.1", port: pg.port, user: "postgres_ai_mon", database: "testdb", password };
        const mon = new Client(config);
        try {
          await mon.connect();
          expect((await mon.query("select current_user as u")).rows[0].u).toBe("postgres_ai_mon");
        } finally { await mon.end(); }
        // A negative control proves HBA does not accidentally allow trust.
        const wrong = new Client({ ...config, password: "test-only-wrong" });
        try { await expect(wrong.connect()).rejects.toThrow(/password authentication failed/); }
        finally { await wrong.end(); }
      }
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: 60000 });

  test("creation and reset honor the server SCRAM iteration count", async () => {
    // scram_iterations exists since PostgreSQL 16. CI sets PGAI_TEST_REQUIRE_PG16=1 so
    // this test fails instead of passing vacuously when only older binaries exist.
    const binDir = findPgBinDirAtLeast(16);
    if (!binDir) {
      if (process.env.PGAI_TEST_REQUIRE_PG16 === "1") {
        throw new Error("PGAI_TEST_REQUIRE_PG16=1 but no PostgreSQL >= 16 initdb/postgres found");
      }
      console.warn("skipped: SCRAM iteration test needs PostgreSQL >= 16 binaries");
      return;
    }
    pg = await createTempPostgres(binDir);
    const admin = new Client({ connectionString: pg.adminUri });
    try {
      await admin.connect();
      const setting = await admin.query("select current_setting('scram_iterations', true) as iterations");
      expect(setting.rows[0].iterations).not.toBeNull();
      // Set the default for the new connections opened by prepare-db.
      await admin.query("alter database testdb set scram_iterations = 10000");
      for (const reset of [false, true]) {
        const password = reset ? "test-only-reset" : "test-only-create";
        const result = runCliInit([
          pg.adminUri, "--password", password, "--skip-optional-permissions",
          ...(reset ? ["--reset-password"] : []),
        ]);
        expect(result.status).toBe(0);
        const stored = await admin.query("select rolpassword from pg_authid where rolname = 'postgres_ai_mon'");
        // Assert a boolean to avoid exposing the verifier in failure output.
        expect(stored.rows[0].rolpassword.startsWith("SCRAM-SHA-256$10000:")).toBe(true);
        const mon = new Client({
          host: "127.0.0.1", port: pg.port, database: "testdb",
          user: "postgres_ai_mon", password,
        });
        try {
          await mon.connect();
          expect((await mon.query("select current_user as u")).rows[0].u).toBe("postgres_ai_mon");
        } finally {
          await mon.end();
        }
      }
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("requires explicit monitoring password in non-interactive mode", async () => {
    pg = await createTempPostgres();

    try {
      // Should fail without --print-password in non-interactive mode
      {
        const r = runCliInit([pg.adminUri, "--skip-optional-permissions"]);
        expect(r.status).not.toBe(0);
        expect(r.stderr).toMatch(/not printed in non-interactive mode/i);
        expect(r.stderr).toMatch(/--print-password/);
      }

      // With explicit opt-in, it should succeed
      {
        const r = runCliInit([pg.adminUri, "--print-password", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
        expect(r.stderr).toMatch(/Generated monitoring password for postgres_ai_mon/i);
        expect(r.stderr).toMatch(/PGAI_MON_PASSWORD=/);
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test(
    "fixes slightly-off permissions idempotently",
    async () => {
      pg = await createTempPostgres();

      try {
        // Create monitoring role with wrong password, no grants.
        {
          const c = new Client({ connectionString: pg.adminUri });
          await c.connect();
          await c.query(
            "do $$ begin if not exists (select 1 from pg_roles where rolname='postgres_ai_mon') then create role postgres_ai_mon login password 'wrong'; end if; end $$;"
          );
          await c.end();
        }

        // Run init (should grant everything).
        {
          const r = runCliInit([pg.adminUri, "--password", "correctpw", "--skip-optional-permissions"]);
          expect(r.status).toBe(0);
        }

        // Verify privileges.
        {
          const c = new Client({ connectionString: pg.adminUri });
          await c.connect();
          const dbOk = await c.query(
            "select has_database_privilege('postgres_ai_mon', current_database(), 'CONNECT') as ok"
          );
          expect(dbOk.rows[0].ok).toBe(true);
          const roleOk = await c.query("select pg_has_role('postgres_ai_mon', 'pg_monitor', 'member') as ok");
          expect(roleOk.rows[0].ok).toBe(true);
          const idxOk = await c.query(
            "select has_table_privilege('postgres_ai_mon', 'pg_catalog.pg_index', 'SELECT') as ok"
          );
          expect(idxOk.rows[0].ok).toBe(true);
          const viewOk = await c.query(
            "select has_table_privilege('postgres_ai_mon', 'postgres_ai.pg_statistic', 'SELECT') as ok"
          );
          expect(viewOk.rows[0].ok).toBe(true);
          const sp = await c.query("select rolconfig from pg_roles where rolname='postgres_ai_mon'");
          expect(Array.isArray(sp.rows[0].rolconfig)).toBe(true);
          expect(sp.rows[0].rolconfig.some((v: string) => String(v).includes("search_path="))).toBe(true);
          await c.end();
        }

        // Run init again (idempotent).
        {
          const r = runCliInit([pg.adminUri, "--password", "correctpw", "--skip-optional-permissions"]);
          expect(r.status).toBe(0);
        }
      } finally {
        await pg.cleanup();
      }
    },
    { timeout: TEST_TIMEOUT }
  );

  test("reports nicely when lacking permissions", async () => {
    pg = await createTempPostgres();

    try {
      // Create limited user that can connect but cannot create roles / grant.
      const limitedPw = "limitedpw";
      {
        const c = new Client({ connectionString: pg.adminUri });
        await c.connect();
        await c.query(`do $$ begin
          if not exists (select 1 from pg_roles where rolname='limited') then
            begin
              create role limited login password ${sqlLiteral(limitedPw)};
            exception when duplicate_object then
              null;
            end;
          end if;
        end $$;`);
          await c.query("grant connect on database testdb to limited");
          await c.end();
        }

      const limitedUri = `postgresql://limited:${limitedPw}@127.0.0.1:${pg.port}/testdb`;
      const r = runCliInit([limitedUri, "--password", "monpw", "--skip-optional-permissions"]);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toMatch(/Error: prepare-db:/);
      expect(r.stderr).toMatch(/Failed at step "/);
      expect(r.stderr).toMatch(/Fix: connect as a superuser/i);
    } finally {
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test(
    "--verify returns 0 when ok and non-zero when missing",
    async () => {
      pg = await createTempPostgres();

      try {
        // Prepare: run init
        {
          const r = runCliInit([pg.adminUri, "--password", "monpw", "--skip-optional-permissions"]);
          expect(r.status).toBe(0);
        }

        // Verify should pass
        {
          const r = runCliInit([pg.adminUri, "--verify", "--skip-optional-permissions"]);
          expect(r.status).toBe(0);
          expect(r.stdout).toMatch(/prepare-db verify: OK/i);
        }

        // Break a required privilege and ensure verify fails
        {
          const c = new Client({ connectionString: pg.adminUri });
          await c.connect();
          await c.query("revoke select on pg_catalog.pg_index from public");
          await c.query("revoke select on pg_catalog.pg_index from postgres_ai_mon");
          await c.end();
        }
        {
          const r = runCliInit([pg.adminUri, "--verify", "--skip-optional-permissions"]);
          expect(r.status).not.toBe(0);
          expect(r.stderr).toMatch(/prepare-db verify failed/i);
          expect(r.stderr).toMatch(/pg_catalog\.pg_index/i);
        }
      } finally {
        await pg.cleanup();
      }
    },
    { timeout: TEST_TIMEOUT }
  );

  // Grants prepare-db (any version) may give the monitoring role; unprepare must remove them.
  const MON_GRANTS_SQL = `select
    has_schema_privilege('postgres_ai_mon', 'postgres_ai', 'USAGE') as schema_usage,
    pg_has_role('postgres_ai_mon', 'pg_read_all_stats', 'member') as read_all_stats,
    has_function_privilege('postgres_ai_mon', 'pg_catalog.pg_stat_file(text)', 'EXECUTE') as stat_file,
    has_function_privilege('postgres_ai_mon', 'pg_catalog.pg_stat_file(text, boolean)', 'EXECUTE') as stat_file_missing_ok,
    has_function_privilege('postgres_ai_mon', 'pg_catalog.pg_ls_dir(text)', 'EXECUTE') as ls_dir,
    has_function_privilege('postgres_ai_mon', 'pg_catalog.pg_ls_dir(text, boolean, boolean)', 'EXECUTE') as ls_dir_flags`;

  function runCliUnprepare(args: string[]): { status: number | null; output: string } {
    const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const r = Bun.spawnSync(["bun", cliPath, "unprepare-db", ...args, "--force"]);
    return { status: r.exitCode, output: new TextDecoder().decode(r.stdout) + new TextDecoder().decode(r.stderr) };
  }

  test("unprepare-db --keep-role preserves customer schema objects and revokes every prepare grant", async () => {
    pg = await createTempPostgres();
    const admin = new Client({ connectionString: pg.adminUri });
    try {
      await admin.connect();
      await admin.query("create schema postgres_ai");
      await admin.query("create table postgres_ai.sentinel (id integer primary key)");
      await admin.query("insert into postgres_ai.sentinel values (73)");

      const prepared = runCliInit([pg.adminUri, "--password", "monpw"]);
      expect(prepared.status).toBe(0);
      // Older prepare-db versions granted pg_read_all_stats directly.
      await admin.query("grant pg_read_all_stats to postgres_ai_mon");
      expect(Object.values((await admin.query(MON_GRANTS_SQL)).rows[0]).every(Boolean)).toBe(true);

      for (let attempt = 0; attempt < 2; attempt++) {
        const cleaned = runCliUnprepare([pg.adminUri, "--keep-role"]);
        expect(cleaned.output).not.toMatch(/error/i);
        expect(cleaned.status).toBe(0);
      }

      expect((await admin.query("select id from postgres_ai.sentinel")).rows).toEqual([{ id: 73 }]);
      expect((await admin.query("select 1 from pg_catalog.pg_roles where rolname = 'postgres_ai_mon'")).rowCount).toBe(1);
      expect(Object.values((await admin.query(MON_GRANTS_SQL)).rows[0]).some(Boolean)).toBe(false);
      expect((await admin.query("select to_regclass('postgres_ai.pg_statistic') as view")).rows[0].view).toBeNull();
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: 60000 });

  test("unprepare-db runs as a non-superuser admin with ADMIN OPTION on pg_monitor only", async () => {
    // PG 16+ needs ADMIN OPTION to revoke a role and grant option to revoke EXECUTE.
    const binDir = findPgBinDirAtLeast(16);
    if (!binDir) {
      if (process.env.PGAI_TEST_REQUIRE_PG16 === "1") {
        throw new Error("PGAI_TEST_REQUIRE_PG16=1 but no PostgreSQL >= 16 initdb/postgres found");
      }
      console.warn("skipped: non-superuser unprepare test needs PostgreSQL >= 16 binaries");
      return;
    }
    pg = await createTempPostgres(binDir);
    const admin = new Client({ connectionString: pg.adminUri });
    const admUri = pg.adminUri.replace("postgres:postgrespw@", "adm:adm-pw@");
    try {
      await admin.connect();
      await admin.query("create role adm login createrole password 'adm-pw'");
      await admin.query("grant pg_monitor to adm with admin option");
      await admin.query("alter database testdb owner to adm");
      await admin.query("create extension pg_stat_statements");

      const prepared = runCliInit([admUri, "--password", "monpw", "--skip-optional-permissions"]);
      expect(prepared.status).toBe(0);
      // Superuser grants that this admin cannot revoke (for example, an older install).
      await admin.query("grant pg_read_all_stats to postgres_ai_mon");
      await admin.query("grant execute on function pg_catalog.pg_ls_dir(text) to postgres_ai_mon");

      // The other revokes still run. The user sees each grant that is left, and the exit code is 1.
      // The server must send the WARNINGs even when the admin's setting hides them.
      await admin.query("alter role adm set client_min_messages = error");
      const kept = runCliUnprepare([admUri, "--keep-role"]);
      expect(kept.output).toMatch(/grant left.*revoke pg_read_all_stats from postgres_ai_mon/);
      expect(kept.output).toMatch(/grant left.*revoke execute on function pg_ls_dir\(text\) from postgres_ai_mon/);
      expect(kept.status).toBe(1);
      const state = await admin.query(`select
        has_schema_privilege('postgres_ai_mon', 'postgres_ai', 'USAGE') as schema_usage,
        pg_has_role('postgres_ai_mon', 'pg_monitor', 'member') as monitor,
        pg_has_role('postgres_ai_mon', 'pg_read_all_stats', 'member') as read_all_stats,
        has_function_privilege('postgres_ai_mon', 'pg_catalog.pg_ls_dir(text)', 'EXECUTE') as ls_dir,
        to_regclass('postgres_ai.pg_statistic') is not null as view`);
      expect(state.rows[0]).toEqual({ schema_usage: false, monitor: false, read_all_stats: true, ls_dir: true, view: false });

      // PG 16+ keeps one membership row per grantor. The superuser must revoke
      // the pg_monitor membership that adm granted, plus the grants adm left.
      expect(runCliInit([admUri, "--password", "monpw", "--skip-optional-permissions"]).status).toBe(0);
      const bySuperuser = runCliUnprepare([pg.adminUri, "--keep-role"]);
      expect(bySuperuser.output).not.toMatch(/grant left/);
      expect(bySuperuser.status).toBe(0);
      expect(Object.values((await admin.query(MON_GRANTS_SQL)).rows[0]).some(Boolean)).toBe(false);
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: 60000 });

  test("unprepare-db never runs an operator or function planted in public", async () => {
    pg = await createTempPostgres();
    const admin = new Client({ connectionString: pg.adminUri });
    try {
      await admin.connect();
      expect(runCliInit([pg.adminUri, "--password", "monpw"]).status).toBe(0);
      // A role with CREATE on public can plant exact-type matches that win over pg_catalog.
      await admin.query(`
        create table public.planted_calls (name text);
        create function public.oid_eq_regclass(oid, regclass) returns boolean language sql as
          $$ insert into public.planted_calls values ('='); select $1 operator(pg_catalog.=) $2::oid $$;
        create operator public.= (leftarg = oid, rightarg = regclass, function = public.oid_eq_regclass);
        create function public.format(text, text) returns text language sql as
          $$ insert into public.planted_calls values ('format'); select pg_catalog.format($1, $2) $$;`);

      const cleaned = runCliUnprepare([pg.adminUri]);
      expect(cleaned.status).toBe(0);
      expect((await admin.query("select name from public.planted_calls")).rows).toEqual([]);
      expect((await admin.query("select 1 from pg_catalog.pg_roles where rolname = 'postgres_ai_mon'")).rowCount).toBe(0);
    } finally {
      await admin.end();
      await pg.cleanup();
    }
  }, { timeout: 60000 });

  // 15s timeout for PostgreSQL startup + two CLI init commands in slow CI
  test("--reset-password updates the monitoring role login password", async () => {
    pg = await createTempPostgres();

    try {
      // Initial setup with password pw1
      {
        const r = runCliInit([pg.adminUri, "--password", "pw1", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
      }

      // Reset to pw2
      {
        const r = runCliInit([pg.adminUri, "--reset-password", "--password", "pw2", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
        expect(r.stdout).toMatch(/password reset/i);
      }

      // Connect as monitoring user with new password should work
      {
        const c = new Client({
          connectionString: `postgresql://postgres_ai_mon:pw2@127.0.0.1:${pg.port}/testdb`,
        });
        await c.connect();
        const ok = await c.query("select 1 as ok");
        expect(ok.rows[0].ok).toBe(1);
        await c.end();
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("prepare-db removes legacy explain_generic idempotently", async () => {
    pg = await createTempPostgres();
    const c = new Client({ connectionString: pg.adminUri });

    try {
      await c.connect();
      // Harmless stubs model the 0.15.0 layout: both helpers SECURITY DEFINER,
      // table_describe with search_path pg_catalog, public.
      await c.query(`
        create schema postgres_ai;
        create function postgres_ai.explain_generic(text, text, text)
        returns text language sql security definer as $$ select 'legacy'::text $$;
        create function postgres_ai.table_describe(in table_name text, out result text)
        language sql security definer set search_path = pg_catalog, public as $$ select 'legacy'::text $$;
      `);
      const signature = "postgres_ai.explain_generic(text,text,text)";
      const legacy = await c.query("select to_regprocedure($1) as helper", [signature]);
      expect(legacy.rows[0].helper).not.toBeNull();

      for (let run = 0; run < 2; run++) {
        const r = runCliInit([pg.adminUri, "--password", "monpw", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
        const result = await c.query("select to_regprocedure($1) as helper", [signature]);
        expect(result.rows[0].helper).toBeNull();
        const describe = await c.query(
          "select prosecdef, proconfig from pg_proc where oid = 'postgres_ai.table_describe(text)'::regprocedure"
        );
        expect(describe.rows).toEqual([{ prosecdef: false, proconfig: ["search_path=pg_catalog, pg_temp"] }]);
      }

      const verified = runCliInit([pg.adminUri, "--verify", "--skip-optional-permissions"]);
      expect(verified.status).toBe(0);
      expect(verified.stdout).toMatch(/prepare-db verify: OK/i);
    } finally {
      await c.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  test("prepare-db refuses a foreign schema while verify reports legacy helpers", async () => {
    pg = await createTempPostgres();
    const c = new Client({ connectionString: pg.adminUri });

    try {
      await c.connect();
      // Another admin created the schema and the legacy helpers; this admin is a
      // CREATEROLE non-superuser that owns none of them, so it cannot drop or replace them.
      await c.query(`
        create extension if not exists pg_stat_statements;
        create role legacy_owner nologin;
        create role second_admin login createrole password 'adminpw';
        grant pg_monitor, pg_read_all_stats to second_admin with admin option;
        grant create on database testdb to second_admin;
        create schema postgres_ai authorization legacy_owner;
        grant usage, create on schema postgres_ai to second_admin with grant option;
        create function postgres_ai.explain_generic(text, text, text)
        returns text language sql security definer as $$ select 'legacy'::text $$;
        alter function postgres_ai.explain_generic(text, text, text) owner to legacy_owner;
        create function postgres_ai.table_describe(in table_name text, out result text)
        language sql security definer set search_path = pg_catalog, public as $$ select 'legacy'::text $$;
        alter function postgres_ai.table_describe(text) owner to legacy_owner;
      `);
      const adminUri = pg.adminUri.replace("postgres:postgrespw@", "second_admin:adminpw@");

      // The schema guard rejects the foreign owner before granting the monitoring
      // role access. Verify still reports the installed legacy helpers separately.
      const stillInstalled = /explain_generic\(text, text, text\) is still installed/;
      const describeInstalled = /table_describe\(text\) is still installed/;
      const warning = (f: string) =>
        `legacy SECURITY DEFINER helper ${f} is still installed; its owner or a superuser must drop it, then re-run prepare-db`;
      const failed = runCliInit([adminUri, "--password", "monpw", "--skip-optional-permissions"]);
      expect(failed.status).toBe(1);
      expect(failed.stderr).toMatch(/Failed at step "03\.permissions"/);
      expect(failed.stderr).toMatch(/schema must be owned by the prepare-db administrator/);

      const failedJson = runCliInit([adminUri, "--password", "monpw", "--skip-optional-permissions", "--json"]);
      expect(failedJson.status).toBe(1);
      const failure = JSON.parse(failedJson.stdout);
      expect(failure.error.step).toBe("03.permissions");
      expect(failure.warnings).toEqual([
        warning("postgres_ai.explain_generic(text, text, text)"),
        warning("postgres_ai.table_describe(text)"),
      ]);

      // The owner drops only explain_generic. The legacy definer table_describe stays,
      // so --verify must still fail and name it.
      await c.query(`
        set role legacy_owner;
        drop function postgres_ai.explain_generic(text, text, text);
        reset role;
      `);
      const describeLeft = runCliInit([adminUri, "--verify", "--skip-optional-permissions"]);
      expect(describeLeft.status).toBe(1);
      expect(describeLeft.stderr).toMatch(describeInstalled);

      // With only explain_generic left to another owner, prepare-db refuses.
      await c.query(`
        drop function postgres_ai.table_describe(text);
        create function postgres_ai.explain_generic(text, text, text)
        returns text language sql security definer as $$ select 'legacy'::text $$;
        alter function postgres_ai.explain_generic(text, text, text) owner to legacy_owner;
      `);
      // A missing table_describe is listed as missing, not raised as 42883.
      const describeGone = runCliInit([adminUri, "--verify", "--skip-optional-permissions", "--json"]);
      expect(describeGone.status).toBe(1);
      expect(JSON.parse(describeGone.stdout).missingRequired).toEqual(expect.arrayContaining([
        "function postgres_ai.table_describe(text) exists",
        warning("postgres_ai.explain_generic(text, text, text)"),
      ]));
      expect(describeGone.stdout + describeGone.stderr).not.toMatch(describeInstalled);
      const stillBlocked = runCliInit([adminUri, "--password", "monpw", "--skip-optional-permissions"]);
      expect(stillBlocked.status).toBe(1);
      expect(stillBlocked.stderr).toMatch(/schema must be owned by the prepare-db administrator/);

      const verified = runCliInit([adminUri, "--verify", "--skip-optional-permissions"]);
      expect(verified.status).toBe(1);
      expect(verified.stderr).toMatch(/verify failed: missing required items/i);
      expect(verified.stderr).toMatch(stillInstalled);

      const verifiedJson = runCliInit([adminUri, "--verify", "--skip-optional-permissions", "--json"]);
      expect(verifiedJson.status).toBe(1);
      const verification = JSON.parse(verifiedJson.stdout);
      expect(verification).toMatchObject({ success: false, verified: false });
      expect(verification.missingRequired.some((m: string) => stillInstalled.test(m))).toBe(true);
      const helpers = await c.query(
        "select to_regprocedure('postgres_ai.explain_generic(text,text,text)') as legacy, to_regprocedure('postgres_ai.table_describe(text)') as describe"
      );
      expect(helpers.rows[0].legacy).not.toBeNull();
      expect(helpers.rows[0].describe).toBeNull();

      // A privileged DBA inspects and removes the legacy helper, transfers the
      // dedicated schema to the actual prepare-db administrator, then retries.
      await c.query("drop function postgres_ai.explain_generic(text, text, text)");
      await c.query("alter schema postgres_ai owner to second_admin");
      const recovered = runCliInit([adminUri, "--password", "monpw", "--skip-optional-permissions"]);
      expect(recovered.status).toBe(0);
      const recoveredVerify = runCliInit([adminUri, "--verify", "--skip-optional-permissions"]);
      expect(recoveredVerify.status).toBe(0);
    } finally {
      await c.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  // Each database gets its own role search_path with its pg_stat_statements schema.
  // The role-wide value (setdatabase 0) is the base path, for databases not prepared yet.
  test("role search_path per database: public, extensions and \"My Ext\", both prepare orders", async () => {
    pg = await createTempPostgres();
    const third = 'My "DB"';
    const uriFor = (db: string, user = `postgres:${pg.postgresPassword}`) =>
      `postgresql://${user}@127.0.0.1:${pg.port}/${encodeURIComponent(db)}`;
    const admin = async (db: string, ...sqls: string[]) => {
      const c = new Client({ connectionString: uriFor(db) });
      await c.connect();
      try {
        let rows: any[] = [];
        for (const sql of sqls) rows = (await c.query(sql)).rows;
        return rows;
      } finally {
        await c.end();
      }
    };
    const base = "search_path=pg_catalog, postgres_ai, public, pg_temp";
    const ext = "search_path=pg_catalog, postgres_ai, extensions, public, pg_temp";
    const myExt = 'search_path=pg_catalog, postgres_ai, "My Ext", public, pg_temp';
    const settings = () => admin("testdb", `select coalesce(d.datname, '') as db, s.setconfig
      from pg_db_role_setting s left join pg_database d on d.oid = s.setdatabase
      where s.setrole = (select oid from pg_roles where rolname = 'postgres_ai_mon') order by coalesce(d.datname::text, '') collate "C"`);
    const prepare = (db: string) => expect({ db, ...runCliInit([uriFor(db), "--password", "monpw", "--skip-optional-permissions"]) })
      .toMatchObject({ db, status: 0 });
    const verify = (db: string, status: number, user?: string) => {
      const r = runCliInit([uriFor(db, user), "--verify", "--skip-optional-permissions"]);
      expect({ db, user, status: r.status, out: r.stdout + r.stderr }).toMatchObject({ db, user, status });
      return r.stdout + r.stderr;
    };
    try {
      await admin("testdb", "create extension pg_stat_statements", "create database extdb", 'create database "My ""DB"""');
      await admin("extdb", "create schema extensions", "create extension pg_stat_statements with schema extensions");
      await admin(third, 'create schema "My Ext"', 'create extension pg_stat_statements with schema "My Ext"');

      prepare("testdb");
      prepare("extdb");
      expect(await settings()).toEqual([
        { db: "", setconfig: [base] },
        { db: "extdb", setconfig: [ext] },
        { db: "testdb", setconfig: [base] },
      ]);
      verify("testdb", 0);
      verify("extdb", 0);

      await admin("testdb", "alter role postgres_ai_mon reset all",
        "alter role postgres_ai_mon in database testdb reset all", "alter role postgres_ai_mon in database extdb reset all");
      prepare("extdb");
      prepare("testdb");
      prepare(third);
      expect(await settings()).toEqual([
        { db: "", setconfig: [base] },
        { db: third, setconfig: [myExt] },
        { db: "extdb", setconfig: [ext] },
        { db: "testdb", setconfig: [base] },
      ]);
      for (const db of ["testdb", "extdb", third]) {
        verify(db, 0);
        verify(db, 0, "postgres_ai_mon:monpw");
      }

      // A row for this database without search_path: the role-wide value applies.
      await admin("testdb", "alter role postgres_ai_mon in database testdb reset search_path",
        "alter role postgres_ai_mon in database testdb set work_mem = '8MB'");
      verify("testdb", 0);
      // A role-wide value from an older prepare-db: verify fails, prepare-db sets the base path again.
      await admin("testdb", "alter role postgres_ai_mon set search_path = pg_catalog, postgres_ai, extensions, public, pg_temp");
      expect(verify("testdb", 1)).toContain("role search_path is pg_catalog, postgres_ai, public, pg_temp");
      prepare("testdb");
      expect((await settings()).filter((r) => r.db === "" || r.db === "testdb")).toEqual([
        { db: "", setconfig: [base] },
        { db: "testdb", setconfig: ["work_mem=8MB", base] },
      ]);
      verify("testdb", 0);

      // pg_stat_statements moved after prepare-db: verify fails until prepare-db runs again.
      await admin("extdb", "alter extension pg_stat_statements set schema public");
      expect(verify("extdb", 1)).toContain("role search_path is pg_catalog, postgres_ai, public, pg_temp");
      prepare("extdb");
      verify("extdb", 0);
    } finally {
      await pg.cleanup();
    }
  }, { timeout: 120000 });

  // CVE-2018-1058 class: the search_path block runs as the admin. An exact-type operator
  // or function in public must not run. Run 2 reads the role-wide row (setdatabase = 0).
  // Run 3 is the printed SQL in psql autocommit (--print-sql, mon deploy with no admin URL).
  test("search_path block ignores public.=(oid, integer) and public.format(text, name, text), CLI and psql", async () => {
    pg = await createTempPostgres();
    const c = new Client({ connectionString: pg.adminUri });
    await c.connect();
    try {
      await c.query(`create table public.hijack (what text);
        create function public.oid_eq_int(oid, integer) returns boolean language plpgsql as
          $f$ begin insert into public.hijack values ('='); return $1 = $2::oid; end $f$;
        create operator public.= (leftarg = oid, rightarg = integer, function = public.oid_eq_int);
        create function public.format(text, name, text) returns text language plpgsql as
          $f$ begin insert into public.hijack values ('format'); return pg_catalog.format($1, $2, $3); end $f$;`);
      for (let run = 1; run <= 2; run++) {
        expect({ run, status: runCliInit([pg.adminUri, "--password", "monpw", "--skip-optional-permissions"]).status })
          .toEqual({ run, status: 0 });
      }
      const plan = await buildInitPlan({ database: "testdb", monitoringPassword: "monpw", includeOptionalPermissions: false });
      const step = plan.steps.find((s) => s.name === "03.permissions")!;
      const psql = Bun.spawnSync([findPgBin("psql")!, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-d", pg.adminUri], {
        stdin: new TextEncoder().encode(step.sql),
      });
      expect(psql.exitCode).toBe(0);
      expect((await c.query("select what from public.hijack")).rows).toEqual([]);
    } finally {
      await c.end();
      await pg.cleanup();
    }
  }, { timeout: TEST_TIMEOUT });

  // Security regression for #387: explain_generic was a SECURITY DEFINER helper that
  // ran caller-supplied SQL through EXPLAIN. The planner folds IMMUTABLE/STABLE functions
  // at plan time in the definer's (superuser) context, making it an RCE/data-exfil vector.
  // It has been removed; prepare-db must NOT create it.
  test("explain_generic is not created by prepare-db (RCE helper removed, #387)", async () => {
    pg = await createTempPostgres();

    try {
      {
        const r = runCliInit([pg.adminUri, "--password", "pw1", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
      }

      const c = new Client({ connectionString: pg.adminUri });
      await c.connect();

      try {
        const res = await c.query(
          "select count(*)::int as n from pg_proc where proname = 'explain_generic' and pronamespace = (select oid from pg_namespace where nspname = 'postgres_ai')"
        );
        expect(res.rows[0].n).toBe(0);
      } finally {
        await c.end();
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: 60000 });
  test("Supabase mode: 03.permissions applies as a role with ADMIN OPTION on pg_monitor only", async () => {
    // The Supabase Management API runs queries as `postgres`: not a superuser, CREATEROLE,
    // pg_monitor WITH ADMIN OPTION, no ADMIN OPTION on pg_read_all_stats. PG 15 lets
    // CREATEROLE grant any role, so only PostgreSQL >= 16 reproduces it.
    const binDir = findPgBinDirAtLeast(16);
    if (!binDir) {
      if (process.env.PGAI_TEST_REQUIRE_PG16 === "1") {
        throw new Error("PGAI_TEST_REQUIRE_PG16=1 but no PostgreSQL >= 16 initdb/postgres found");
      }
      console.warn("skipped: Supabase grant test needs PostgreSQL >= 16 binaries");
      return;
    }
    pg = await createTempPostgres(binDir);
    const admin = new Client({ connectionString: pg.adminUri });
    const supa = new Client({ connectionString: pg.adminUri.replace("postgres:postgrespw@", "supa_postgres:supa-pw@") });
    try {
      await admin.connect();
      await admin.query("create role supa_postgres login createrole password 'supa-pw'");
      await admin.query("grant pg_monitor to supa_postgres with admin option");
      await admin.query("grant create on database testdb to supa_postgres");
      await admin.query("create extension pg_stat_statements");
      await supa.connect();
      const plan = await buildInitPlan({ database: "testdb", monitoringUser: "postgres_ai_mon", monitoringPassword: "mon-pw", includeOptionalPermissions: false });
      const client = { query: (sql: string) => supa.query(sql) } as unknown as SupabaseClient;
      const { applied } = await applyInitPlanViaSupabase({ client, plan });
      expect(applied).toContain("03.permissions");
      const r = await admin.query("select pg_has_role('postgres_ai_mon', 'pg_monitor', 'usage') as monitor, pg_has_role('postgres_ai_mon', 'pg_read_all_stats', 'usage') as stats");
      expect(r.rows[0]).toEqual({ monitor: true, stats: true });
    } finally {
      await supa.end().catch(() => {});
      await admin.end().catch(() => {});
      await pg.cleanup();
    }
  }, { timeout: 60000 });
});
