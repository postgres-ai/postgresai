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

  test("CLI --markdown flag works without API key", async () => {
    // Test that --markdown works even without an API key
    const connString = `postgresql://postgres@${pg.socketDir}:${pg.port}/postgres`;
    const cliPath = path.resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";

    const result = Bun.spawnSync(
      [bunBin, cliPath, "checkup", connString, "--check-id", "H002", "--markdown", "--no-upload"],
      {
        env: {
          ...process.env,
          XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config",
        },
      }
    );

    const stderr = new TextDecoder().decode(result.stderr);

    // Should not complain about missing API key
    expect(stderr).not.toMatch(/API key is required/i);
  });
});
