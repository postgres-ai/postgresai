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

async function createTempPostgres(): Promise<TempPostgres> {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "postgresai-init-"));
  const dataDir = path.join(tmpRoot, "data");
  const socketDir = path.join(tmpRoot, "sock");
  fs.mkdirSync(socketDir, { recursive: true });

  const initdb = findPgBin("initdb");
  const postgresBin = findPgBin("postgres");
  if (!initdb || !postgresBin) {
    throw new Error("PostgreSQL binaries not found (need initdb and postgres)");
  }

  const init = Bun.spawnSync([initdb, "-D", dataDir, "-U", "postgres", "-A", "trust"]);
  if (init.exitCode !== 0) {
    throw new Error(new TextDecoder().decode(init.stderr) || new TextDecoder().decode(init.stdout));
  }

  // Configure: local socket trust, TCP scram.
  const hbaPath = path.join(dataDir, "pg_hba.conf");
  fs.appendFileSync(
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
  const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const result = Bun.spawnSync(["bun", cliPath, "prepare-db", ...args], {
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

describe.skipIf(skipTests)("integration: prepare-db", () => {
  let pg: TempPostgres;

  // Use a shared postgres instance for all tests in this describe block
  // Each test will reset state as needed

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

  // 60s timeout for PostgreSQL startup + multiple SQL queries in slow CI
  test("explain_generic validates input and prevents SQL injection", async () => {
    pg = await createTempPostgres();

    try {
      // Run init first
      {
        const r = runCliInit([pg.adminUri, "--password", "pw1", "--skip-optional-permissions"]);
        expect(r.status).toBe(0);
      }

      const c = new Client({ connectionString: pg.adminUri });
      await c.connect();

      try {
        // Check PostgreSQL version - generic_plan requires 16+
        const versionRes = await c.query("show server_version_num");
        const version = parseInt(versionRes.rows[0].server_version_num, 10);

        if (version < 160000) {
          // Skip this test on older PostgreSQL versions
          console.log("Skipping explain_generic tests: requires PostgreSQL 16+");
          return;
        }

        // Test 1: Empty query should be rejected
        await expect(
          c.query("select postgres_ai.explain_generic('')")
        ).rejects.toThrow(/query cannot be empty/);

        // Test 2: Null query should be rejected
        await expect(
          c.query("select postgres_ai.explain_generic(null)")
        ).rejects.toThrow(/query cannot be empty/);

        // Test 3: Multiple statements (semicolon in middle) should be rejected
        await expect(
          c.query("select postgres_ai.explain_generic('select 1; select 2')")
        ).rejects.toThrow(/semicolon|multiple statements/i);

        // Test 4: Trailing semicolon should be stripped and work
        {
          const res = await c.query("select postgres_ai.explain_generic('select 1;') as result");
          expect(res.rows[0].result).toBeTruthy();
          expect(res.rows[0].result).toMatch(/Result/i);
        }

        // Test 5: Valid query should work
        {
          const res = await c.query("select postgres_ai.explain_generic('select $1::int', 'text') as result");
          expect(res.rows[0].result).toBeTruthy();
        }

        // Test 6: JSON format should work
        {
          const res = await c.query("select postgres_ai.explain_generic('select 1', 'json') as result");
          const plan = JSON.parse(res.rows[0].result);
          expect(Array.isArray(plan)).toBe(true);
          expect(plan[0].Plan).toBeTruthy();
        }

        // Test 7: Whitespace-only query should be rejected
        await expect(
          c.query("select postgres_ai.explain_generic('   ')")
        ).rejects.toThrow(/query cannot be empty/);

        // Test 8: Semicolon in string literal is rejected (documented limitation)
        // Note: This is a known limitation - the simple heuristic cannot parse SQL strings
        await expect(
          c.query("select postgres_ai.explain_generic('select ''hello;world''')")
        ).rejects.toThrow(/semicolon/i);

        // Test 9: SQL comments should work (no semicolons)
        {
          const res = await c.query("select postgres_ai.explain_generic('select 1 -- comment') as result");
          expect(res.rows[0].result).toBeTruthy();
        }

        // Test 10: Escaped quotes should work (no semicolons)
        {
          const res = await c.query("select postgres_ai.explain_generic('select ''test''''s value''') as result");
          expect(res.rows[0].result).toBeTruthy();
        }

        // Test 11: Case-insensitive format parameter
        {
          const res = await c.query("select postgres_ai.explain_generic('select 1', 'JSON') as result");
          const plan = JSON.parse(res.rows[0].result);
          expect(Array.isArray(plan)).toBe(true);
        }

      } finally {
        await c.end();
      }
    } finally {
      await pg.cleanup();
    }
  }, { timeout: 60000 });
});
