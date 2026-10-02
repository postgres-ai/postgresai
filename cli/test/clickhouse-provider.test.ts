import { describe, expect, test } from "bun:test";
import * as init from "../lib/init";
import { buildClientConfig } from "../lib/instances";

const host = "quickstart-pg-c1406b50.pg7dd324nz0a1qm1fqskxbjn7m.c0.us-east-1.aws.pg.clickhouse.cloud";
const scope = "-- scope: role postgres_ai_mon gets: create/update role | create extension pg_stat_statements | connect; pg_monitor, pg_read_all_stats; select on pg_catalog.pg_index; schema postgres_ai (view pg_statistic); usage on schema public; alter user set search_path | execute on postgres_ai.table_describe (SECURITY INVOKER, catalog only) | execute on rds_tools.pg_ls_multixactdir (RDS only) | execute on pg_catalog.pg_ls_dir, pg_catalog.pg_stat_file; this admin connection is used for this run only and is not stored";
const scopeWithoutOptional = "-- scope: role postgres_ai_mon gets: create/update role | create extension pg_stat_statements | connect; pg_monitor, pg_read_all_stats; select on pg_catalog.pg_index; schema postgres_ai (view pg_statistic); usage on schema public; alter user set search_path | execute on postgres_ai.table_describe (SECURITY INVOKER, catalog only); this admin connection is used for this run only and is not stored";

function runCli(args: string[]) {
  const result = Bun.spawnSync([process.execPath, "./bin/postgres-ai.ts", ...args], {
    cwd: `${import.meta.dir}/..`,
    env: { ...process.env, PGHOST: "", PGDATABASE: "" },
    timeout: 15000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}

function printSql(args: string[]) {
  return runCli(["prepare-db", "--print-sql", "--password", "x", "-d", "postgres", ...args]);
}

describe("ClickHouse provider", () => {
  test("is a known provider", () => {
    expect<readonly string[]>(init.KNOWN_PROVIDERS).toContain("clickhouse");
  });

  test("validates without a warning", () => {
    expect(init.validateProvider("clickhouse")).toBeNull();
  });

  test.each([
    [`postgres://postgres:pass@${host}:5432/postgres?channel_binding=require`, "clickhouse"],
    ["postgresql://u:p@h.pg.clickhouse.cloud/db", "clickhouse"],
    ["postgres://u@H.PG.CLICKHOUSE.CLOUD/db", "clickhouse"],
    [`host=${host} port=5432 dbname=postgres user=postgres`, "clickhouse"],
    ["dbname=postgres host='H.PG.CLICKHOUSE.CLOUD' user=postgres", "clickhouse"],
    ["postgres://u@h.pg.clickhouse.cloud./db", "clickhouse"],
    ["h.pg.clickhouse.cloud", "clickhouse"],
    ["h.pg.clickhouse.cloud.", "clickhouse"],
    ["localhost", null],
    ["postgres://u@pg.clickhouse.cloud.evil.com/db", null],
    ["postgres://u@xpg.clickhouse.cloud/db", null],
    ["postgres://u@clickhouse.cloud/db", null],
    ["postgres://u@db.example.supabase.co/db", null],
    ["host=pg.clickhouse.cloud.evil.com dbname=postgres", null],
    ["host=xpg.clickhouse.cloud dbname=postgres", null],
    ["postgres://h.pg.clickhouse.cloud@localhost/db", null],
    ["postgres://localhost/db?application_name=h.pg.clickhouse.cloud", null],
    ["", null],
    ["not a connection string", null],
    ["postgres://[malformed", null],
  ] as const)("detects provider for %j", (connectionString, expected) => {
    expect(init.detectProvider(connectionString)).toBe(expected);
  });

  test("uses the self-managed superuser plan", async () => {
    const options = { database: "postgres", monitoringPassword: "CLICKHOUSE_MONITORING_PASSWORD", includeOptionalPermissions: true };
    const plan = await init.buildInitPlan({ ...options, provider: "clickhouse" });
    const selfManaged = await init.buildInitPlan({ ...options, provider: "self-managed" });
    expect(plan.steps.map(step => step.name)).toEqual(["01.role", "02.extensions", "03.permissions", "06.helpers", "04.optional_rds", "05.optional_self_managed"]);
    // Salted SCRAM verifiers (!336) differ per plan, so compare with passwords redacted.
    const redacted = (steps: typeof plan.steps) => steps.map((step) => ({ ...step, sql: init.redactPasswordsInSql(step.sql) }));
    expect(redacted(plan.steps)).toEqual(redacted(selfManaged.steps));
  });

  test("prints the exact scope before SQL", () => {
    const result = printSql(["--provider", "clickhouse"]);
    expect(result.exitCode).toBe(0);
    expect(result.stdout.split("\n")).toContain(scope);
    expect(result.stdout.indexOf(scope)).toBeLessThan(result.stdout.indexOf("-- 01.role"));
  });

  test("scope line is derived from the plan steps", async () => {
    const options = { database: "postgres", monitoringPassword: "x", provider: "clickhouse" };
    const full = await init.buildInitPlan({ ...options, includeOptionalPermissions: true });
    const noOptional = await init.buildInitPlan({ ...options, includeOptionalPermissions: false });
    expect(init.describeInitScope(full)).toBe(scope);
    expect(init.describeInitScope(noOptional)).toBe(scopeWithoutOptional);
    expect(init.describeInitScope({ ...full, steps: full.steps.filter(step => step.name === "01.role") })).toBe(
      "-- scope: role postgres_ai_mon gets: create/update role; this admin connection is used for this run only and is not stored",
    );
    // Every step in the plan has a description; an unknown step would leak its raw name.
    for (const step of full.steps) expect(init.describeInitScope(full)).not.toContain(`| ${step.name}`);
  });

  test("scope line shrinks with --skip-optional-permissions", () => {
    const result = printSql(["--provider", "clickhouse", "--skip-optional-permissions"]);
    expect(result.exitCode).toBe(0);
    expect(result.stdout.split("\n")).toContain(scopeWithoutOptional);
    expect(result.stdout).not.toContain("pg_ls_dir");
  });

  test("scope line is not printed for self-managed", () => {
    const result = printSql([]);
    expect(result.exitCode).toBe(0);
    expect(result.stdout).not.toContain("-- scope:");
  });

  // A connection string makes --print-sql connect first, and the host does not resolve here.
  // The detection note must be on screen BEFORE the admin connection is attempted.
  test.each([
    ["prepare-db", ["prepare-db", "--print-sql", "--password", "x", `postgres://u:p@${host}:5432/postgres`]],
    ["unprepare-db", ["unprepare-db", "--print-sql", `postgres://u:p@${host}:5432/postgres`]],
  ])("%s detects the provider from the host before connecting", (command, args) => {
    const plain = runCli(args);
    expect(plain.stderr).toContain("ENOTFOUND");
    expect(plain.stdout.split("\n")).toContain("Provider: clickhouse (detected from host)");

    const json = runCli([...args, "--json"]);
    expect(json.stdout).not.toContain("Provider: clickhouse");
    expect(json.stdout).not.toContain("-- scope:");
    expect(() => JSON.parse(json.stdout)).not.toThrow();
    expect(json.stderr.split("\n")).toContain("Provider: clickhouse (detected from host)");
  });

  test("detects the provider from --host", () => {
    const result = runCli(["prepare-db", "--print-sql", "--password", "x", "-h", host, "-U", "postgres", "-d", "postgres"]);
    expect(result.stdout.split("\n")).toContain("Provider: clickhouse (detected from host)");
  });

  test("--reset-password prints no scope line (it grants nothing)", () => {
    const result = runCli(["prepare-db", "--print-sql", "--reset-password", "--password", "x", `postgres://u:p@${host}:5432/postgres`]);
    expect(result.stdout).not.toContain("-- scope:");
    expect(result.stderr).not.toContain("-- scope:");
  });

  test.each([
    [undefined, `postgres://postgres:pass@${host}:5432/postgres`, { provider: "clickhouse", detected: true }],
    ["self-managed", `postgres://postgres:pass@${host}:5432/postgres`, { provider: "self-managed", detected: false }],
    [undefined, "postgresql://u@db.example.com/db", { provider: "self-managed", detected: false }],
    [undefined, undefined, { provider: "self-managed", detected: false }],
  ] as const)("resolveProvider(%j, %j)", (explicit, conn, expected) => {
    expect(init.resolveProvider(explicit, conn)).toEqual(expected);
  });
});

describe("ClickHouse URI channel binding", () => {
  test.each(["require", "disable", undefined])("resolveAdminConnection: %s", mode => {
    const uri = `postgres://u:p@h.pg.clickhouse.cloud:5432/postgres${mode ? `?channel_binding=${mode}` : ""}`;
    const { clientConfig, sslFallbackEnabled } = init.resolveAdminConnection({ conn: uri });
    if (mode === "require") expect(clientConfig).toHaveProperty("enableChannelBinding", true);
    else expect(clientConfig).not.toHaveProperty("enableChannelBinding");
    expect(clientConfig.connectionString).toBe("postgres://u:p@h.pg.clickhouse.cloud:5432/postgres");
    // require never retries over plaintext; the others keep sslmode=prefer behaviour
    expect(sslFallbackEnabled).toBe(mode !== "require");
  });

  test.each([
    "host=h.pg.clickhouse.cloud user=u dbname=postgres channel_binding=require",
    "host=h.pg.clickhouse.cloud user=u dbname=postgres channel_binding=require sslmode=require",
  ])("resolveAdminConnection conninfo: %s", conninfo => {
    const { clientConfig, sslFallbackEnabled } = init.resolveAdminConnection({ conn: conninfo });
    expect(clientConfig).toHaveProperty("enableChannelBinding", true);
    expect(clientConfig).not.toHaveProperty("channel_binding");
    expect(sslFallbackEnabled).toBe(false);
  });

  test("conninfo without channel_binding stays unchanged", () => {
    const { clientConfig } = init.resolveAdminConnection({ conn: "host=h.pg.clickhouse.cloud user=u dbname=postgres" });
    expect(clientConfig).not.toHaveProperty("enableChannelBinding");
  });

  test.each([
    "postgres://u:p@h.pg.clickhouse.cloud:5432/postgres?channel_binding=require&sslmode=disable",
    "host=h.pg.clickhouse.cloud user=u dbname=postgres channel_binding=require sslmode=disable",
  ])("channel_binding=require refuses sslmode=disable: %s", conn => {
    expect(() => init.resolveAdminConnection({ conn })).toThrow("channel_binding=require needs TLS");
  });

  test("buildClientConfig: channel_binding=require refuses sslmode=disable", () => {
    expect(() => buildClientConfig("postgres://u:p@h.pg.clickhouse.cloud:5432/postgres?channel_binding=require&sslmode=disable")).toThrow("channel_binding=require needs TLS");
  });

  test.each(["require", "disable", undefined])("buildClientConfig: %s", mode => {
    const uri = `postgres://u:p@h.pg.clickhouse.cloud:5432/postgres${mode ? `?channel_binding=${mode}` : ""}`;
    const config = buildClientConfig(uri, { connectionTimeoutMillis: 1234 });
    if (mode === "require") expect(config).toHaveProperty("enableChannelBinding", true);
    else expect(config).not.toHaveProperty("enableChannelBinding");
    expect(config).toMatchObject({ host: "h.pg.clickhouse.cloud", port: 5432, user: "u", password: "p", database: "postgres", connectionTimeoutMillis: 1234 });
    expect(config).not.toHaveProperty("connectionString");
    expect(config).not.toHaveProperty("channel_binding");
  });
});
