/**
 * Integration tests for checkup command (express mode)
 * Validates that CLI-generated reports match JSON schemas used by the Python reporter.
 * This ensures compatibility between "express" and "full" (monitoring) modes.
 */
import { describe, test, expect, afterAll, beforeAll } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as net from "net";
import { Client } from "pg";
import { resolve } from "path";
import { readFileSync } from "fs";
import Ajv2020 from "ajv/dist/2020";

import * as checkup from "../lib/checkup";
import { checkCurrentUserPermissions, formatPermissionCheckMessages } from "../lib/init";

const ajv = new Ajv2020({ allErrors: true, strict: false });
const schemasDir = resolve(import.meta.dir, "../../reporter/schemas");

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
  cleanup: () => Promise<void>;
  connect: (database?: string) => Promise<Client>;
}

async function createTempPostgres(): Promise<TempPostgres> {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "postgresai-checkup-"));
  const dataDir = path.join(tmpRoot, "data");
  const socketDir = path.join(tmpRoot, "sock");
  fs.mkdirSync(socketDir, { recursive: true });

  const initdb = findPgBin("initdb");
  const postgresBin = findPgBin("postgres");
  if (!initdb || !postgresBin) {
    throw new Error("PostgreSQL binaries not found");
  }

  const init = Bun.spawnSync([initdb, "-D", dataDir, "-U", "postgres", "-A", "trust"]);
  if (init.exitCode !== 0) {
    throw new Error(new TextDecoder().decode(init.stderr) || new TextDecoder().decode(init.stdout));
  }

  const hbaPath = path.join(dataDir, "pg_hba.conf");
  fs.appendFileSync(hbaPath, "\nlocal all all trust\n", "utf8");

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

  const connect = async (database = "postgres"): Promise<Client> => {
    const c = new Client({ host: socketDir, port, user: "postgres", database });
    await c.connect();
    return c;
  };

  // Wait for Postgres to start (30s timeout for slower CI environments)
  await waitFor(async () => {
    const c = await connect();
    await c.end();
  }, { timeoutMs: 30000, intervalMs: 100 });

  return { port, socketDir, cleanup, connect };
}

function validateAgainstSchema(report: any, checkId: string): void {
  const schemaPath = resolve(schemasDir, `${checkId}.schema.json`);
  if (!fs.existsSync(schemaPath)) {
    throw new Error(`Schema not found: ${schemaPath}`);
  }
  const schema = JSON.parse(readFileSync(schemaPath, "utf8"));
  const validate = ajv.compile(schema);
  const valid = validate(report);
  if (!valid) {
    const errors = validate.errors?.map(e => `${e.instancePath}: ${e.message}`).join(", ");
    throw new Error(`${checkId} schema validation failed: ${errors}`);
  }
}

// Skip tests if PostgreSQL binaries are not available
const skipReason = !havePostgresBinaries()
  ? "PostgreSQL binaries not available"
  : isRunningAsRoot()
  ? "Cannot run as root (PostgreSQL refuses)"
  : null;

// In CI, warn if integration tests are being skipped (helps catch configuration issues)
const isCI = process.env.CI === "true" || process.env.GITLAB_CI === "true";
if (skipReason && isCI) {
  console.warn(`[CI WARNING] Integration tests skipped: ${skipReason}`);
  console.warn("This may indicate a CI configuration issue - PostgreSQL binaries should be available.");
}

describe.skipIf(!!skipReason)("checkup integration: express mode schema compatibility", () => {
  let pg: TempPostgres;
  let client: Client;

  // 60s timeout for hooks - PostgreSQL startup can take 30s+ in slow CI
  beforeAll(async () => {
    // Create empty config directory for tests
    const emptyConfigDir = "/tmp/postgresai-test-empty-config/postgresai";
    fs.mkdirSync(emptyConfigDir, { recursive: true });
    fs.writeFileSync(path.join(emptyConfigDir, "config.json"), "{}");

    pg = await createTempPostgres();
    client = await pg.connect();
  }, { timeout: 60000 });

  afterAll(async () => {
    if (client) await client.end();
    if (pg) await pg.cleanup();
  }, { timeout: 60000 });

  // Test all checks supported by express mode
  const expressChecks = Object.keys(checkup.CHECK_INFO);

  test("vanilla database preflight is non-fatal and full CLI output marks F004/F005 degraded", async () => {
    const permissions = await checkCurrentUserPermissions(client);
    expect(permissions.ok).toBe(true);
    expect(permissions.missingOptional.some(
      (row) => row.permission_name === "postgres_ai schema exists"
    )).toBe(true);

    const connString = `postgresql://postgres@localhost:${pg.port}/postgres?host=${encodeURIComponent(pg.socketDir)}`;
    const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync(
      [bunBin, cliPath, "checkup", connString, "--no-upload", "--json"],
      { env: { ...process.env, XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" } }
    );

    const stderr = new TextDecoder().decode(result.stderr);
    if (result.exitCode !== 0) {
      throw new Error(`CLI exited ${result.exitCode}: ${stderr}`);
    }
    const reports = JSON.parse(new TextDecoder().decode(result.stdout));
    expect(Object.keys(reports)).toHaveLength(expressChecks.length);
    expect(stderr).toContain("optional: postgres_ai schema not found");
    for (const checkId of ["F004", "F005"]) {
      const dbEntry = reports[checkId].results["node-01"].data.postgres;
      expect(dbEntry.status.ok).toBe(false);
      expect(dbEntry.status.reason).toBe("missing_schema");
      expect(dbEntry.status.error).toMatch(/schema "postgres_ai" does not exist/i);
    }
  }, { timeout: 60000 });

  // Regression for issue #229: on a database where prepare-db has never been
  // run (no postgres_ai schema), has_schema_privilege() used to RAISE
  // `schema "postgres_ai" does not exist`, aborting the whole pre-flight query
  // and killing checkup with a bare error before any check ran. These must run
  // BEFORE the CLI contract tests below, which prepare the shared temp
  // instance and thereby create the postgres_ai schema.
  test("pre-flight permission check does not throw on a DB without postgres_ai schema", async () => {
    // The temp instance has not run prepare-db yet, so postgres_ai must not exist.
    const schemaRes = await client.query(
      "select 1 from pg_namespace where nspname = 'postgres_ai'"
    );
    expect(schemaRes.rowCount).toBe(0);

    // Must not throw.
    const permCheck = await checkCurrentUserPermissions(client);

    // Required permissions are fine (superuser), so checkup may proceed.
    expect(permCheck.ok).toBe(true);

    // The missing schema degrades to an actionable optional warning.
    const schemaExists = permCheck.rows.find(
      (r) => r.permission_name === "postgres_ai schema exists"
    );
    expect(schemaExists?.granted).toBe(false);
    expect(schemaExists?.fix_command).toContain("postgresai prepare-db");

    // The dependent checks are skipped (null), not reported as missing.
    for (const name of [
      "usage on postgres_ai schema",
      "postgres_ai.pg_statistic view exists",
      "select on postgres_ai.pg_statistic",
    ]) {
      const row = permCheck.rows.find((r) => r.permission_name === name);
      expect(row?.granted).toBeNull();
    }

    const messages = formatPermissionCheckMessages(permCheck);
    expect(messages.failed).toBe(false);
    expect(messages.warnings.some((w) => w.includes("prepare-db"))).toBe(true);
  });

  test("CLI checkup completes on a DB without postgres_ai schema (no prepare-db)", async () => {
    const connString = `postgresql://postgres@/postgres?host=${pg.socketDir}&port=${pg.port}`;
    const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";

    const result = Bun.spawnSync(
      [bunBin, cliPath, "checkup", connString, "--check-id", "H002", "--no-upload"],
      {
        env: {
          ...process.env,
          XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config",
        },
      }
    );

    const stdout = new TextDecoder().decode(result.stdout);
    const stderr = new TextDecoder().decode(result.stderr);

    // Used to die with: Error: schema "postgres_ai" does not exist
    expect(stderr).not.toContain('schema "postgres_ai" does not exist');
    expect(result.exitCode).toBe(0);

    // Degrades to a warning pointing the user at prepare-db.
    expect(stderr).toContain("postgres_ai schema not found");
    expect(stderr).toContain("prepare-db");

    // The check actually ran and produced a report.
    expect(stdout).toContain("H002");
  }, { timeout: 60000 });

  for (const checkId of expressChecks) {
    test(`${checkId} report validates against shared schema`, async () => {
      const generator = checkup.REPORT_GENERATORS[checkId];
      expect(generator).toBeDefined();

      const report = await generator(client, "test-node");

      // Validate basic report structure (matching schema requirements)
      expect(report).toHaveProperty("checkId", checkId);
      expect(report).toHaveProperty("checkTitle");
      expect(report).toHaveProperty("timestamptz");
      expect(report).toHaveProperty("nodes");
      expect(report).toHaveProperty("results");
      expect(report.results).toHaveProperty("test-node");

      // Validate against JSON schema (same schema used by Python reporter)
      validateAgainstSchema(report, checkId);
    });
  }

  test("generateAllReports produces valid reports for all checks", async () => {
    const reports = await checkup.generateAllReports(client, "test-node");

    expect(Object.keys(reports).length).toBe(expressChecks.length);

    for (const [checkId, report] of Object.entries(reports)) {
      validateAgainstSchema(report, checkId);
    }
  });

  test("report structure matches Python reporter format", async () => {
    // Generate A003 (settings) report and verify structure matches what Python produces
    const report = await checkup.generateA003(client, "test-node");

    // Check required fields match Python reporter output structure (per schema)
    expect(report).toHaveProperty("checkId", "A003");
    expect(report).toHaveProperty("checkTitle", "Postgres settings");
    expect(report).toHaveProperty("timestamptz");
    expect(report).toHaveProperty("nodes");
    expect(report.nodes).toHaveProperty("primary");
    expect(report.nodes).toHaveProperty("standbys");
    expect(report).toHaveProperty("results");

    // Results should have node-specific data
    const nodeResult = report.results["test-node"];
    expect(nodeResult).toHaveProperty("data");

    // A003 should have settings as keyed object
    expect(typeof nodeResult.data).toBe("object");

    // Check postgres_version if present
    if (nodeResult.postgres_version) {
      expect(nodeResult.postgres_version).toHaveProperty("version");
      expect(nodeResult.postgres_version).toHaveProperty("server_version_num");
      expect(nodeResult.postgres_version).toHaveProperty("server_major_ver");
      expect(nodeResult.postgres_version).toHaveProperty("server_minor_ver");
    }
  });

  test("H001 (invalid indexes) has correct data structure", async () => {
    const report = await checkup.generateH001(client, "test-node");
    validateAgainstSchema(report, "H001");

    const nodeResult = report.results["test-node"];
    expect(nodeResult).toHaveProperty("data");
    // data should be an object with indexes (may be empty on fresh DB)
    expect(typeof nodeResult.data).toBe("object");
  });

  test("H001 returns index_definition with CREATE INDEX statement", async () => {
    // Create a table and an index, then mark the index as invalid
    await client.query(`
      CREATE TABLE IF NOT EXISTS test_invalid_idx_table (id serial PRIMARY KEY, value text);
      CREATE INDEX IF NOT EXISTS test_invalid_idx ON test_invalid_idx_table(value);
    `);

    // Mark the index as invalid (simulating a failed CONCURRENTLY build)
    await client.query(`
      UPDATE pg_index SET indisvalid = false
      WHERE indexrelid = 'test_invalid_idx'::regclass;
    `);

    try {
      const report = await checkup.generateH001(client, "test-node");
      validateAgainstSchema(report, "H001");

      const nodeResult = report.results["test-node"];
      const dbName = Object.keys(nodeResult.data)[0];
      expect(dbName).toBeTruthy();

      const dbData = nodeResult.data[dbName] as any;
      expect(dbData.invalid_indexes).toBeDefined();
      expect(dbData.invalid_indexes.length).toBeGreaterThan(0);

      // Find our test index
      const testIndex = dbData.invalid_indexes.find(
        (idx: any) => idx.index_name === "test_invalid_idx"
      );
      expect(testIndex).toBeDefined();

      // Verify index_definition contains the actual CREATE INDEX statement
      expect(testIndex.index_definition).toMatch(/^CREATE INDEX/);
      expect(testIndex.index_definition).toContain("test_invalid_idx");
      expect(testIndex.index_definition).toContain("test_invalid_idx_table");
    } finally {
      // Cleanup: restore the index and drop test objects
      await client.query(`
        UPDATE pg_index SET indisvalid = true
        WHERE indexrelid = 'test_invalid_idx'::regclass;
        DROP INDEX IF EXISTS test_invalid_idx;
        DROP TABLE IF EXISTS test_invalid_idx_table;
      `);
    }
  });

  test("H002 (unused indexes) has correct data structure", async () => {
    const report = await checkup.generateH002(client, "test-node");
    validateAgainstSchema(report, "H002");

    const nodeResult = report.results["test-node"];
    expect(nodeResult).toHaveProperty("data");
    expect(typeof nodeResult.data).toBe("object");
  });

  test("H004 (redundant indexes) has correct data structure", async () => {
    const report = await checkup.generateH004(client, "test-node");
    validateAgainstSchema(report, "H004");

    const nodeResult = report.results["test-node"];
    expect(nodeResult).toHaveProperty("data");
    expect(typeof nodeResult.data).toBe("object");
  });

  test("F003 flags a table with dead tuples and per-table disabled autovacuum", async () => {
    // Reproduce the footgun the check exists for: a table with autovacuum
    // disabled via reloptions accumulating dead tuples from UPDATE/DELETE.
    await client.query(`
      CREATE TABLE f003_dead_tuples_test (id int PRIMARY KEY, payload text);
      ALTER TABLE f003_dead_tuples_test SET (autovacuum_enabled = false);
      INSERT INTO f003_dead_tuples_test SELECT g, repeat('x', 50) FROM generate_series(1, 20000) g;
      UPDATE f003_dead_tuples_test SET payload = payload || 'y';
    `);

    try {
      // Cumulative stats are flushed asynchronously; poll until the dead
      // tuples from the UPDATE become visible in pg_stat_user_tables.
      await waitFor(async () => {
        const r = await client.query(
          "select n_dead_tup from pg_stat_user_tables where relname = 'f003_dead_tuples_test'"
        );
        if (!r.rows.length || parseInt(r.rows[0].n_dead_tup, 10) < 20000) {
          throw new Error("dead tuple stats not flushed yet");
        }
      }, { timeoutMs: 15000, intervalMs: 250 });

      const report = await checkup.REPORT_GENERATORS.F003(client, "test-node");
      validateAgainstSchema(report, "F003");

      const nodeResult = report.results["test-node"];
      const dbName = Object.keys(nodeResult.data)[0];
      const dbData = nodeResult.data[dbName] as any;

      const table = dbData.dead_tuples_tables.find(
        (t: any) => t.table_name === "f003_dead_tuples_test"
      );
      expect(table).toBeDefined();
      expect(table.autovacuum_disabled).toBe(true);
      expect(table.n_dead_tup).toBeGreaterThanOrEqual(20000);
      expect(table.dead_pct).toBeGreaterThanOrEqual(checkup.F003_DEAD_PCT_MIN);
      // 20k dead tuples is below F003_DEAD_TUPLES_MIN (100k), so the
      // dead-tuple thresholds must NOT fire, but the disabled-autovacuum
      // flag must (>= 10k tuples with autovacuum off).
      expect(table.exceeds_dead_tuple_thresholds).toBe(false);
      expect(table.autovacuum_disabled_flagged).toBe(true);
      expect(dbData.autovacuum_disabled_count).toBeGreaterThanOrEqual(1);
      expect(
        dbData.conclusions.some((c: string) => c.includes("f003_dead_tuples_test"))
      ).toBe(true);
      expect(
        dbData.recommendations.some((r: string) =>
          r.includes('alter table "public"."f003_dead_tuples_test" reset (autovacuum_enabled);')
        )
      ).toBe(true);
    } finally {
      await client.query("DROP TABLE IF EXISTS f003_dead_tuples_test;");
    }
  });

  test("F001 resolves effective values, throughput budget, and flags a per-table cost_delay=0", async () => {
    // A per-table autovacuum_vacuum_cost_delay=0 override (write-storm footgun)
    // must be surfaced by the reloptions overview + the per-table rule, and the
    // report must carry the effective-value resolution and throughput budget.
    await client.query(`
      CREATE TABLE f001_cost_delay_test (id int PRIMARY KEY, payload text);
      ALTER TABLE f001_cost_delay_test SET (autovacuum_vacuum_cost_delay = 0);
    `);

    try {
      const report = await checkup.REPORT_GENERATORS.F001(client, "test-node");
      validateAgainstSchema(report, "F001");

      const node = report.results["test-node"] as any;

      // Effective-value resolution + inheritance chains are present.
      expect(node.effective_values).toBeDefined();
      expect(node.effective_values.cost_limit).toHaveProperty("inheritance_chain");
      expect(node.effective_values.cost_delay_ms).toHaveProperty("effective");

      // Throughput budget is computed from the effective cost model.
      expect(node.throughput_budget).toHaveProperty("tokens_per_sec");
      expect(node.throughput_budget).toHaveProperty("dirty_write_mbps");

      // The per-table cost_delay=0 override is surfaced (rule + overview).
      const relopts = node.settings_analysis.reloptions_overview;
      expect(
        relopts.cost_delay_zero_tables.some((t: string) => t.includes("f001_cost_delay_test"))
      ).toBe(true);
      const firedIds = node.settings_analysis.rules_fired.map((r: any) => r.id);
      expect(firedIds).toContain("per_table_cost_delay_zero");

      // Never-recommend list holds on real output: no recommendation sets cost_delay=0.
      for (const rec of node.recommendations as string[]) {
        expect(rec).not.toMatch(/cost_delay\s*=\s*0(?![.\d])/i);
      }
    } finally {
      await client.query("DROP TABLE IF EXISTS f001_cost_delay_test;");
    }
  });

  test("F001 surfaces a toast-level cost_delay=0 override (stored on the toast relation)", async () => {
    // A toast.* option is stored on the toast relation (relkind 't') as a plain
    // autovacuum_* option, NOT as a 'toast.'-prefixed option on the parent. The
    // metric must join heap->toast and attribute it back to the owning table.
    await client.query(`
      CREATE TABLE f001_toast_test (id int PRIMARY KEY, blob text)
        WITH (toast.autovacuum_vacuum_cost_delay = 0);
    `);

    try {
      // Confirm the option really lives on the toast relation, not the parent.
      const parentOpts = await client.query(
        "select reloptions from pg_class where relname = 'f001_toast_test'"
      );
      expect(parentOpts.rows[0].reloptions).toBeNull();

      const report = await checkup.REPORT_GENERATORS.F001(client, "test-node");
      validateAgainstSchema(report, "F001");

      const node = report.results["test-node"] as any;
      const relopts = node.settings_analysis.reloptions_overview;
      expect(
        relopts.cost_delay_zero_tables.some((t: string) => t.includes("f001_toast_test"))
      ).toBe(true);
      // The zero lives only on the toast relation, so it is classified toast-only
      // and the rule names it as such.
      expect(
        relopts.cost_delay_zero_toast_only_tables.some((t: string) => t.includes("f001_toast_test"))
      ).toBe(true);
      const costDelayRule = node.settings_analysis.rules_fired.find(
        (r: any) => r.id === "per_table_cost_delay_zero"
      );
      expect(costDelayRule).toBeDefined();
      expect(costDelayRule.conclusion).toContain("(toast-level)");
    } finally {
      await client.query("DROP TABLE IF EXISTS f001_toast_test;");
    }
  });

  test("F009 identifies a read-committed idle-in-transaction xid holder", async () => {
    const holder = await pg.connect();
    try {
      await holder.query("BEGIN");
      await holder.query("SELECT txid_current()");
      // READ COMMITTED releases its snapshot between statements, leaving only
      // backend_xid to pin the horizon. Advance the current XID so its age and
      // holder attribution are deterministic.
      await client.query("SELECT txid_current()");

      await waitFor(async () => {
        const state = await client.query(
          "select state, backend_xid, backend_xmin from pg_stat_activity where pid = $1",
          [(holder as any).processID],
        );
        if (
          state.rows[0]?.state !== "idle in transaction" ||
          !state.rows[0]?.backend_xid ||
          state.rows[0]?.backend_xmin
        ) {
          throw new Error("xid-only holder is not visible yet");
        }
      });

      const report = await checkup.REPORT_GENERATORS.F009(client, "test-node");
      validateAgainstSchema(report, "F009");
      const data = report.results["test-node"].data as any;
      const activity = data.components.pg_stat_activity;
      expect(activity.count).toBeGreaterThanOrEqual(1);
      expect(activity.top_blocker.pid).toBe((holder as any).processID);
      expect(activity.top_blocker.state).toBe("idle in transaction");
      expect(data.dominant_holder.source).toBe("activity");
      expect(data.recommendations.join(" ")).toContain("pg_terminate_backend");
      expect(data.recommendations.join(" ")).toContain("idle_in_transaction_session_timeout");
    } finally {
      await holder.query("ROLLBACK").catch(() => undefined);
      await holder.end().catch(() => undefined);
    }
  });

  // ---------------------------------------------------------------------------
  // CLI JSON contract (ABI) tests
  //
  // These exercise the *real* CLI path — `checkup --no-upload --json` — that
  // host applications embedding checkup depend on, and assert the stable
  // contract: single JSON object keyed by check ID on stdout, schema-valid
  // reports, envelope invariants (contract_version + generation_mode +
  // summary), stderr free of report JSON, and exit-code semantics.
  //
  // The CLI runs a permissions preflight, so we first provision the database
  // with `prepare-db` (creating the monitoring role) and then run checkup as
  // that role, passing its password via PGPASSWORD (never argv) — the exact
  // embedding flow documented for host applications.
  // ---------------------------------------------------------------------------
  const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin =
    typeof process.execPath === "string" && process.execPath.length > 0
      ? process.execPath
      : "bun";
  const MON_USER = "postgres_ai_mon";
  const MON_PASSWORD = "checkup_contract_test_pw";
  let dbPrepared = false;

  function runCli(
    args: string[],
    extraEnv: Record<string, string> = {}
  ): { code: number | null; stdout: string; stderr: string } {
    const result = Bun.spawnSync([bunBin, cliPath, ...args], {
      // Explicit separate pipes: the contract requires stdout to carry ONLY the
      // JSON payload and stderr the diagnostics. Under `bun test` the default
      // stdio must not be left to merge the two streams.
      stdin: "ignore",
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config",
        ...extraEnv,
      },
    });
    return {
      code: result.exitCode,
      stdout: new TextDecoder().decode(result.stdout),
      stderr: new TextDecoder().decode(result.stderr),
    };
  }

  // TCP connection string (the temp Postgres also listens on 127.0.0.1).
  const adminConn = () => `postgresql://postgres@127.0.0.1:${pg.port}/postgres`;
  const monConn = () => `postgresql://${MON_USER}@127.0.0.1:${pg.port}/postgres`;

  // Provision the monitoring role once, via the real `prepare-db` command.
  function ensurePrepared(): void {
    if (dbPrepared) return;
    const { code, stderr } = runCli([
      "prepare-db",
      adminConn(),
      "--monitoring-user",
      MON_USER,
      "--password",
      MON_PASSWORD,
    ]);
    if (code !== 0) {
      throw new Error(`prepare-db failed (exit ${code}): ${stderr}`);
    }
    dbPrepared = true;
  }

  test("checkup --no-upload --json: stdout is a single JSON object keyed by check ID, schema-valid, with envelope invariants", () => {
    ensurePrepared();
    const { code, stdout, stderr } = runCli(
      ["checkup", monConn(), "--no-upload", "--json"],
      { PGPASSWORD: MON_PASSWORD }
    );

    // Exit code 0 on success
    expect(code).toBe(0);

    // stdout is exactly one JSON object (not NDJSON, not multiple documents)
    let payload: Record<string, any>;
    expect(() => {
      payload = JSON.parse(stdout);
    }).not.toThrow();
    payload = JSON.parse(stdout);
    expect(typeof payload).toBe("object");
    expect(Array.isArray(payload)).toBe(false);

    // Keyed by check ID; every value is that check's report and validates
    // against its shared schema, with the required envelope invariants.
    const keys = Object.keys(payload);
    expect(keys.length).toBeGreaterThan(0);
    for (const checkId of keys) {
      expect(checkId).toMatch(/^[A-Z]\d{3}$/);
      const report = payload[checkId];
      expect(report.checkId).toBe(checkId);

      // Envelope invariants that make up the versioned contract.
      expect(typeof report.contract_version).toBe("string");
      expect(report.contract_version).toMatch(/^\d+\.\d+\.\d+$/);
      expect(report.contract_version).toBe(checkup.CONTRACT_VERSION);
      expect(report.generation_mode).toBe("express");

      // Folded-in severity summary (D4): status + message.
      expect(report.summary).toBeDefined();
      expect(["ok", "warning", "info"]).toContain(report.summary.status);
      expect(typeof report.summary.message).toBe("string");

      validateAgainstSchema(report, checkId);
    }

    // stderr must not carry report JSON — machine consumers read only stdout.
    // Diagnostics (plain-text warnings) on stderr are fine; report objects are
    // not. The report envelope's distinctive keys must never appear there.
    expect(stderr).not.toContain('"checkId"');
    expect(stderr).not.toContain('"contract_version"');
    expect(stderr).not.toContain('"results"');
  });

  test("checkup --check-id --no-upload --json: single-check payload carries contract_version + summary", () => {
    ensurePrepared();
    const { code, stdout } = runCli(
      ["checkup", monConn(), "--check-id", "H002", "--no-upload", "--json"],
      { PGPASSWORD: MON_PASSWORD }
    );

    expect(code).toBe(0);
    const payload = JSON.parse(stdout);
    expect(Object.keys(payload)).toEqual(["H002"]);
    expect(payload.H002.contract_version).toBe(checkup.CONTRACT_VERSION);
    expect(payload.H002.summary).toBeDefined();
    validateAgainstSchema(payload.H002, "H002");
  });

  test("checkup with an unknown check ID exits non-zero and writes no JSON to stdout", () => {
    ensurePrepared();
    const { code, stdout } = runCli(
      ["checkup", monConn(), "--check-id", "Z999", "--no-upload", "--json"],
      { PGPASSWORD: MON_PASSWORD }
    );

    expect(code).not.toBe(0);
    expect(stdout.trim()).toBe("");
  });
});
