import { describe, expect, test } from "bun:test";
import * as init from "../lib/init";
import { buildClientConfig } from "../lib/instances";

const host = "quickstart-pg-c1406b50.pg7dd324nz0a1qm1fqskxbjn7m.c0.us-east-1.aws.pg.clickhouse.cloud";
const scope = "-- scope: role postgres_ai_mon gets pg_monitor, pg_read_all_stats; this admin connection is used for this run only and is not stored";

function printSql(args: string[]) {
  const result = Bun.spawnSync([process.execPath, "./bin/postgres-ai.ts", "prepare-db", "--print-sql", "--password", "x", "-d", "postgres", ...args], {
    cwd: `${import.meta.dir}/..`,
    timeout: 15000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
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
    expect(plan.steps).toEqual(selfManaged.steps);
  });

  test("prints the exact scope before SQL", () => {
    const result = printSql(["--provider", "clickhouse"]);
    expect(result.exitCode).toBe(0);
    expect(result.stdout.split("\n")).toContain(scope);
    expect(result.stdout.indexOf(scope)).toBeLessThan(result.stdout.indexOf("-- 01.role"));
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
    const { clientConfig } = init.resolveAdminConnection({ conn: uri });
    if (mode === "require") expect(clientConfig).toHaveProperty("enableChannelBinding", true);
    else expect(clientConfig).not.toHaveProperty("enableChannelBinding");
    expect(clientConfig.connectionString).toBe("postgres://u:p@h.pg.clickhouse.cloud:5432/postgres");
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
