import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { randomBytes } from "crypto";
import { Client } from "pg";
import { load } from "js-yaml";

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

  test("full collector presets match expected failures", async () => {
    const mon = new Client({ connectionString: monUrl, connectionTimeoutMillis: 10000 });
    const failures: { metric: string; error: string }[] = [];
    try {
      await mon.connect();
      const version = await mon.query("show server_version_num");
      const major = Math.floor(Number(version.rows[0].server_version_num) / 10000);
      for (const sink of ["prometheus", "postgres"]) {
        const config = load(await Bun.file(`${import.meta.dir}/../../config/pgwatch-${sink}/metrics.yml`).text()) as {
          metrics: Record<string, { node_status?: string; sqls: Record<string, string> }>;
          presets: { full: { metrics: Record<string, number> } };
        };
        const names = Object.keys(config.presets.full.metrics).sort();
        expect(names.length).toBeGreaterThan(0);
        let collected = 0;
        for (const name of names) {
          const metric = config.metrics[name];
          expect(metric).toBeDefined();
          if (metric.node_status === "standby") continue;
          const key = Object.keys(metric.sqls)
            .filter(key => Number.isFinite(Number(key)) && Number(key) <= major)
            .sort((a, b) => Number(b) - Number(a))[0];
          expect(key, `${sink}/${name}: no SQL for PostgreSQL ${major}`).toBeDefined();
          const sql = metric.sqls[key];
          expect(sql.trim().length).toBeGreaterThan(0);
          await mon.query("begin");
          try {
            await mon.query("set local statement_timeout = '30s'");
            await mon.query(sql);
          } catch (error) {
            failures.push({ metric: `${sink}/${name}`, error: error instanceof Error ? error.message : String(error) });
          } finally {
            await mon.query("rollback");
          }
          collected++;
        }
        console.log(`PostgreSQL ${major}: ${sink} full preset, ${collected} collected, ${names.length - collected} standby-only skipped`);
      }
    } finally {
      await mon.end();
    }
    expect(failures).toEqual(await Bun.file(`${import.meta.dir}/fixtures/clickhouse-like-collector-failures.golden.json`).json());
  }, 1800000);
});
