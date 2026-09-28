import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { randomBytes } from "crypto";
import { Client } from "pg";

const adminUrl = process.env.PGAI_TEST_CLICKHOUSE_LIKE_URL;

function runCli(args: string[]) {
  const result = Bun.spawnSync([process.execPath, "./bin/postgres-ai.ts", ...args], {
    cwd: `${import.meta.dir}/..`,
    env: { ...process.env, PGSSLMODE: "disable" },
    timeout: 120000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}

describe.skipIf(!adminUrl)("ClickHouse-like Postgres", () => {
  let prepared: ReturnType<typeof runCli>;
  let monUrl: string;

  beforeAll(async () => {
    const admin = new Client({ connectionString: adminUrl, connectionTimeoutMillis: 10000 });
    try {
      await admin.connect();
      const preload = await admin.query("show shared_preload_libraries");
      expect(preload.rows[0].shared_preload_libraries.split(",").map((value: string) => value.trim())).toEqual(["pg_stat_statements"]);
      const unavailable = await admin.query("select name from pg_available_extensions where name in ('pg_stat_kcache', 'pg_wait_sampling')");
      expect(unavailable.rows).toEqual([]);
      const version = await admin.query("select version()");
      console.log(version.rows[0].version);
    } finally {
      await admin.end();
    }
    const password = randomBytes(24).toString("hex");
    const url = new URL(adminUrl!);
    url.username = "postgres_ai_mon";
    url.password = password;
    monUrl = url.toString();
    prepared = runCli(["prepare-db", adminUrl!, "--provider", "clickhouse", "--password", password]);
    expect(prepared.exitCode, prepared.stderr).toBe(0);
  });

  afterAll(() => {
    const result = runCli(["unprepare-db", adminUrl!, "--provider", "clickhouse", "--force"]);
    expect(result.exitCode, result.stderr).toBe(0);
  });

  test("recognizes clickhouse without an unknown-provider warning", () => {
    expect(prepared.stderr).not.toContain('Unknown provider "clickhouse"');
  });

  test("announces the scope of the admin connection", () => {
    expect(prepared.stdout.split("\n")).toContain("-- scope: role postgres_ai_mon gets pg_monitor, pg_read_all_stats; this admin connection is used for this run only and is not stored");
  });

  test("verifies the prepared database", () => {
    const result = runCli(["prepare-db", adminUrl!, "--provider", "clickhouse", "--verify"]);
    expect(result.exitCode, result.stderr).toBe(0);
  });

  test("grants monitoring access without superuser", async () => {
    const mon = new Client({ connectionString: monUrl, connectionTimeoutMillis: 10000 });
    try {
      await mon.connect();
      const roles = await mon.query(`select
        pg_has_role('postgres_ai_mon', 'pg_monitor', 'member') as monitor,
        pg_has_role('postgres_ai_mon', 'pg_read_all_stats', 'member') as stats,
        rolsuper from pg_roles where rolname = current_user`);
      expect(roles.rows).toEqual([{ monitor: true, stats: true, rolsuper: false }]);
      const statements = await mon.query("select count(*) from pg_stat_statements");
      expect(Number(statements.rows[0].count)).toBeGreaterThanOrEqual(0);
    } finally {
      await mon.end();
    }
  });

  test("D004 reports unavailable kcache without an error", async () => {
    const result = runCli(["checkup", monUrl, "--check-id", "D004", "--no-upload", "--json"]);
    expect(result.exitCode, result.stderr).toBe(0);
    expect(() => JSON.parse(result.stdout)).not.toThrow();
    const report = JSON.parse(result.stdout).D004;
    expect(report).toBeDefined();
    expect(report).not.toHaveProperty("error");
    const node = report.results["node-01"];
    expect(node).not.toHaveProperty("error");
    expect(node.data).not.toHaveProperty("error");
    expect(node.data.pg_stat_statements_status).not.toHaveProperty("error");
    const normalized = { pg_stat_kcache_status: node.data.pg_stat_kcache_status };
    expect(JSON.stringify(normalized, null, 2) + "\n").toBe(await Bun.file(`${import.meta.dir}/fixtures/clickhouse-d004.golden.json`).text());
  }, 120000);

  test("full express checkup succeeds without kcache or wait_sampling errors", () => {
    const result = runCli(["checkup", monUrl, "--no-upload", "--json"]);
    expect(result.exitCode, result.stderr).toBe(0);
    expect(() => JSON.parse(result.stdout)).not.toThrow();
    const reports = JSON.parse(result.stdout);
    expect(Object.keys(reports).length).toBeGreaterThan(1);
    expect(reports).toHaveProperty("D004");
    const errors: string[] = [];
    JSON.stringify(reports, (key, value) => {
      if (/error/i.test(key)) errors.push(JSON.stringify(value));
      return value;
    });
    expect(errors.filter(error => /kcache|wait_sampling/i.test(error))).toEqual([]);
    expect(result.stderr).not.toMatch(/(?:error[^\n]*(?:kcache|wait_sampling)|(?:kcache|wait_sampling)[^\n]*error)/i);
  }, 120000);
});
