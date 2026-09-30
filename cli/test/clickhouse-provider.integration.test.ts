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

const scope = "-- scope: role postgres_ai_mon gets: create/update role | create extension pg_stat_statements | connect; pg_monitor, pg_read_all_stats; select on pg_catalog.pg_index; schema postgres_ai (view pg_statistic); usage on schema public; alter user set search_path | execute on postgres_ai.table_describe (SECURITY INVOKER, catalog only) | execute on rds_tools.pg_ls_multixactdir (RDS only) | execute on pg_catalog.pg_ls_dir, pg_catalog.pg_stat_file; this admin connection is used for this run only and is not stored";

// The suite below skips without a database. In CI that must never happen silently:
// the cli:clickhouse-like:tests job sets PGAI_TEST_CLICKHOUSE_LIKE_URL, so a missing
// variable there is a wiring bug, not a reason for a green run with zero tests.
test.if(!!process.env.CI)("PGAI_TEST_CLICKHOUSE_LIKE_URL is set in CI", () => {
  expect(adminUrl).toBeTruthy();
});

describe.skipIf(!adminUrl)("ClickHouse-like Postgres", () => {
  let prepared: ReturnType<typeof runCli>;
  let preparedJson: ReturnType<typeof runCli>;
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
    preparedJson = runCli(["prepare-db", adminUrl!, "--provider", "clickhouse", "--password", password, "--json"]);
  });

  afterAll(() => {
    const result = runCli(["unprepare-db", adminUrl!, "--provider", "clickhouse", "--force"]);
    expect(result.exitCode, result.stderr).toBe(0);
  });

  test("recognizes clickhouse without an unknown-provider warning", () => {
    expect(prepared.stderr).not.toContain('Unknown provider "clickhouse"');
  });

  test("keeps JSON stdout separate from the scope announcement", () => {
    expect(preparedJson.exitCode, preparedJson.stderr).toBe(0);
    expect(() => JSON.parse(preparedJson.stdout)).not.toThrow();
    expect(preparedJson.stdout).not.toContain("-- scope:");
    expect(preparedJson.stderr.split("\n")).toContain(scope);
  });

  test("announces the scope of the admin connection", () => {
    expect(prepared.stdout.split("\n")).toContain(scope);
    expect(prepared.stdout.indexOf(scope)).toBeLessThan(prepared.stdout.indexOf("Connecting to:"));
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
      // pg_read_all_stats: reading pg_stat_statements as the monitoring role must not throw
      await mon.query("select count(*) from pg_stat_statements");
    } finally {
      await mon.end();
    }
  });

  // table_describe runs as the caller and reads only the catalog, so the monitoring
  // role needs no USAGE on application schemas, and objects it can create in public
  // must not shadow the built-ins the function calls.
  test("table_describe is SECURITY INVOKER and cannot be hijacked through public", async () => {
    const admin = new Client({ connectionString: adminUrl, connectionTimeoutMillis: 10000 });
    const mon = new Client({ connectionString: monUrl, connectionTimeoutMillis: 10000 });
    try {
      await admin.connect();
      await admin.query("create table public.td_probe (id int primary key, note text default 'x')");
      await admin.query("create schema td_app");
      await admin.query("create table td_app.orders (id int references public.td_probe (id))");
      await admin.query("grant create on schema public to postgres_ai_mon");
      await mon.connect();
      const fn = await mon.query("select prosecdef, proconfig from pg_proc where oid = 'postgres_ai.table_describe(text)'::regprocedure");
      expect(fn.rows).toEqual([{ prosecdef: false, proconfig: ["search_path=pg_catalog, pg_temp"] }]);
      const usage = await mon.query("select has_schema_privilege('td_app', 'usage') as usage");
      expect(usage.rows[0].usage).toBe(false);
      // An exact-type overload in public outranks pg_catalog's polymorphic array_append.
      await mon.query(`create function public.array_append(text[], text) returns text[]
        language plpgsql as $$ begin raise exception 'hijacked as %', current_user; end $$`);
      await mon.query(`create function public.format(text, text, text) returns text
        language plpgsql as $$ begin raise exception 'hijacked as %', current_user; end $$`);
      const probe = await mon.query("select postgres_ai.table_describe('td_probe') as r");
      expect(probe.rows[0].r).toContain("Table: public.td_probe");
      expect(probe.rows[0].r).toContain("td_app.orders");
      const orders = await mon.query("select postgres_ai.table_describe('td_app.orders') as r");
      expect(orders.rows[0].r).toContain("Table: td_app.orders");
      const catalog = await mon.query("select postgres_ai.table_describe('pg_class') as r");
      expect(catalog.rows[0].r).toContain("Table: pg_catalog.pg_class");
      await expect(mon.query("select postgres_ai.table_describe('td_missing')")).rejects.toThrow('relation "td_missing" does not exist');
    } finally {
      await mon.end().catch(() => {});
      for (const sql of [
        "drop function if exists public.array_append(text[], text)",
        "drop function if exists public.format(text, text, text)",
        "revoke create on schema public from postgres_ai_mon",
        "drop schema if exists td_app cascade",
        "drop table if exists public.td_probe",
      ]) await admin.query(sql).catch(() => {});
      await admin.end();
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
