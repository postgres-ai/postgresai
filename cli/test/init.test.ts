import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { resolve } from "path";
import * as fs from "fs";
import * as os from "os";

// Import from source directly since we're using Bun
import * as init from "../lib/init";
const DEFAULT_MONITORING_USER = init.DEFAULT_MONITORING_USER;

function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

function runPgai(args: string[], env: Record<string, string> = {}) {
  // For testing, run the CLI directly since pgai is just a thin wrapper
  // In production, pgai wrapper will properly resolve and spawn the postgresai CLI
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

describe("init module", () => {
  test("maskConnectionString hides password when present", () => {
    const masked = init.maskConnectionString("postgresql://user:secret@localhost:5432/mydb");
    expect(masked).toMatch(/postgresql:\/\/user:\*{5}@localhost:5432\/mydb/);
    expect(masked).not.toMatch(/secret/);
  });

  test("parseLibpqConninfo parses basic host/dbname/user/port/password", () => {
    const cfg = init.parseLibpqConninfo("dbname=mydb host=localhost user=alice port=5432 password=secret");
    expect(cfg.database).toBe("mydb");
    expect(cfg.host).toBe("localhost");
    expect(cfg.user).toBe("alice");
    expect(cfg.port).toBe(5432);
    expect(cfg.password).toBe("secret");
  });

  test("parseLibpqConninfo supports quoted values", () => {
    const cfg = init.parseLibpqConninfo("dbname='my db' host='local host'");
    expect(cfg.database).toBe("my db");
    expect(cfg.host).toBe("local host");
  });

  test("buildInitPlan includes a race-safe role DO block", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    expect(plan.database).toBe("mydb");
    const roleStep = plan.steps.find((s: { name: string }) => s.name === "01.role");
    expect(roleStep).toBeTruthy();
    expect(roleStep!.sql).toMatch(/do\s+\$\$/i);
    expect(roleStep!.sql).toMatch(/create\s+user/i);
    expect(roleStep!.sql).toMatch(/alter\s+user/i);
    expect(plan.steps.some((s: { optional?: boolean }) => s.optional)).toBe(false);
  });

  test("buildInitPlan handles special characters in monitoring user and database identifiers", async () => {
    const monitoringUser = 'user "with" quotes ✓';
    const database = 'db name "with" quotes ✓';
    const plan = await init.buildInitPlan({
      database,
      monitoringUser,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const roleStep = plan.steps.find((s: { name: string }) => s.name === "01.role");
    expect(roleStep).toBeTruthy();
    expect(roleStep!.sql).toMatch(/create\s+user\s+"user ""with"" quotes ✓"/i);
    expect(roleStep!.sql).toMatch(/alter\s+user\s+"user ""with"" quotes ✓"/i);

    const permStep = plan.steps.find((s: { name: string }) => s.name === "03.permissions");
    expect(permStep).toBeTruthy();
    expect(permStep!.sql).toMatch(/grant connect on database "db name ""with"" quotes ✓" to "user ""with"" quotes ✓"/i);
  });

  test("buildInitPlan keeps backslashes in passwords (no unintended escaping)", async () => {
    const pw = String.raw`pw\with\backslash`;
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: pw,
      includeOptionalPermissions: false,
    });
    const roleStep = plan.steps.find((s: { name: string }) => s.name === "01.role");
    expect(roleStep).toBeTruthy();
    expect(roleStep!.sql).toContain(`password '${pw}'`);
  });

  test("buildInitPlan rejects identifiers with null bytes", async () => {
    await expect(
      init.buildInitPlan({
        database: "mydb",
        monitoringUser: "bad\0user",
        monitoringPassword: "pw",
        includeOptionalPermissions: false,
      })
    ).rejects.toThrow(/Identifier cannot contain null bytes/);
  });

  test("buildInitPlan rejects literals with null bytes", async () => {
    await expect(
      init.buildInitPlan({
        database: "mydb",
        monitoringUser: DEFAULT_MONITORING_USER,
        monitoringPassword: "pw\0bad",
        includeOptionalPermissions: false,
      })
    ).rejects.toThrow(/Literal cannot contain null bytes/);
  });

  test("buildInitPlan inlines password safely for CREATE/ALTER ROLE grammar", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pa'ss",
      includeOptionalPermissions: false,
    });
    const step = plan.steps.find((s: { name: string }) => s.name === "01.role");
    expect(step).toBeTruthy();
    expect(step!.sql).toMatch(/password 'pa''ss'/);
    expect(step!.params).toBeUndefined();
  });

  test("buildInitPlan includes optional steps when enabled", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: true,
    });
    expect(plan.steps.some((s: { optional?: boolean }) => s.optional)).toBe(true);
  });

  test("buildInitPlan skips role creation for supabase provider", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
      provider: "supabase",
    });
    expect(plan.steps.some((s) => s.name === "01.role")).toBe(false);
    expect(plan.steps.some((s) => s.name === "03.permissions")).toBe(true);
  });

  test("buildInitPlan removes ALTER USER for supabase provider", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
      provider: "supabase",
    });
    const permStep = plan.steps.find((s) => s.name === "03.permissions");
    expect(permStep).toBeDefined();
    expect(permStep!.sql.toLowerCase()).not.toMatch(/alter user/);
  });

  test("buildInitPlan includes role creation for unknown provider", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
      provider: "some-custom-provider",
    });
    expect(plan.steps.some((s) => s.name === "01.role")).toBe(true);
  });

  test("resolveAdminConnection accepts positional URI", () => {
    const r = init.resolveAdminConnection({ conn: "postgresql://u:p@h:5432/d" });
    expect(r.clientConfig.connectionString).toBeTruthy();
    expect(r.display).not.toMatch(/:p@/);
  });

  test("resolveAdminConnection accepts positional conninfo", () => {
    const r = init.resolveAdminConnection({ conn: "dbname=mydb host=localhost user=alice" });
    expect(r.clientConfig.database).toBe("mydb");
    expect(r.clientConfig.host).toBe("localhost");
    expect(r.clientConfig.user).toBe("alice");
  });

  test("resolveAdminConnection rejects invalid psql-like port", () => {
    expect(() => init.resolveAdminConnection({ host: "localhost", port: "abc", username: "u", dbname: "d" }))
      .toThrow(/Invalid port value/);
  });

  test("resolveAdminConnection rejects when only PGPASSWORD is provided (no connection details)", () => {
    expect(() => init.resolveAdminConnection({ envPassword: "pw" })).toThrow(/Connection is required/);
  });

  test("resolveAdminConnection rejects when connection is missing", () => {
    expect(() => init.resolveAdminConnection({})).toThrow(/Connection is required/);
  });

  test("resolveMonitoringPassword auto-generates a strong, URL-safe password by default", async () => {
    const r = await init.resolveMonitoringPassword({ monitoringUser: DEFAULT_MONITORING_USER });
    expect(r.generated).toBe(true);
    expect(typeof r.password).toBe("string");
    expect(r.password.length).toBeGreaterThanOrEqual(30);
    expect(r.password).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  test("applyInitPlan preserves Postgres error fields on step failures", async () => {
    const plan = {
      monitoringUser: DEFAULT_MONITORING_USER,
      database: "mydb",
      steps: [{ name: "01.role", sql: "select 1" }],
    };

    const pgErr = Object.assign(new Error("permission denied to create role"), {
      code: "42501",
      detail: "some detail",
      hint: "some hint",
      schema: "pg_catalog",
      table: "pg_roles",
      constraint: "some_constraint",
      routine: "aclcheck_error",
    });

    const calls: string[] = [];
    const client = {
      query: async (sql: string) => {
        calls.push(sql);
        if (sql === "begin;") return { rowCount: 1 };
        if (sql === "rollback;") return { rowCount: 1 };
        if (sql === "select 1") throw pgErr;
        throw new Error(`unexpected sql: ${sql}`);
      },
    };

    try {
      await init.applyInitPlan({ client: client as any, plan: plan as any });
      expect(true).toBe(false); // Should not reach here
    } catch (e: any) {
      expect(e).toBeInstanceOf(Error);
      expect(e.message).toMatch(/Failed at step "01\.role":/);
      expect(e.code).toBe("42501");
      expect(e.detail).toBe("some detail");
      expect(e.hint).toBe("some hint");
      expect(e.schema).toBe("pg_catalog");
      expect(e.table).toBe("pg_roles");
      expect(e.constraint).toBe("some_constraint");
      expect(e.routine).toBe("aclcheck_error");
    }

    expect(calls).toEqual(["begin;", "select 1", "rollback;"]);
  });

  test("verifyInitSetup runs inside a repeatable read snapshot and rolls back", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, extensions, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege") && String(sql).includes("pg_catalog.pg_index")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass('postgres_ai.pg_statistic')")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege") && String(sql).includes("postgres_ai.pg_statistic")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // Query for pg_stat_statements extension schema location
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "pg_catalog" }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);

    expect(calls.length).toBeGreaterThan(2);
    expect(calls[0].toLowerCase()).toMatch(/^begin isolation level repeatable read/);
    expect(calls[calls.length - 1].toLowerCase()).toBe("rollback;");
  });

  test("verifyInitSetup skips search_path check for supabase provider", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        // Return empty rolconfig - would fail without provider=supabase
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: null }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // Query for pg_stat_statements extension schema location (Supabase uses 'extensions' schema)
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "extensions" }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    // With provider=supabase, should pass even without search_path
    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
      provider: "supabase",
    });
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);
    // Should not have queried for rolconfig since we skip search_path check
    expect(calls.some((c) => c.includes("select rolconfig"))).toBe(false);
  });

  test("verifyInitSetup checks extensions schema when pg_stat_statements is there", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, extensions, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // pg_stat_statements is in 'extensions' schema
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "extensions" }] };
        }
        // Check for USAGE on extensions schema
        if (String(sql).includes("has_schema_privilege") && params?.[1] === "extensions") {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);
    // Should have queried for pg_stat_statements schema location
    expect(calls.some((c) => c.includes("pg_extension e") && c.includes("pg_stat_statements"))).toBe(true);
  });

  test("verifyInitSetup reports missing extensions schema access", async () => {
    const client = {
      query: async (sql: string, params?: any) => {
        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // pg_stat_statements is in 'extensions' schema
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "extensions" }] };
        }
        // No USAGE on extensions schema
        if (String(sql).includes("has_schema_privilege") && params?.[1] === "extensions") {
          return { rowCount: 1, rows: [{ ok: false }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    expect(r.ok).toBe(false);
    // Should report missing USAGE on extensions schema
    expect(r.missingRequired.some((m) => m.includes("extensions") && m.includes("pg_stat_statements"))).toBe(true);
    // Should also report missing extensions in search_path
    expect(r.missingRequired.some((m) => m.includes("search_path") && m.includes("extensions"))).toBe(true);
  });

  test("buildInitPlan includes dynamic search_path with extension schema detection", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const permStep = plan.steps.find((s) => s.name === "03.permissions");
    expect(permStep).toBeTruthy();
    // Should use dynamic DO block to set search_path based on detected extension schema
    expect(permStep!.sql).toMatch(/alter\s+user.*set\s+search_path\s*=/i);
    // Should detect pg_stat_statements extension schema dynamically
    expect(permStep!.sql).toMatch(/quote_ident\(ext_schema\)/i);
  });

  test("buildInitPlan includes dynamic extension schema grant", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const permStep = plan.steps.find((s) => s.name === "03.permissions");
    expect(permStep).toBeTruthy();
    // Should include DO block that grants USAGE on extension schema
    expect(permStep!.sql).toMatch(/do\s+\$\$/i);
    expect(permStep!.sql).toMatch(/pg_stat_statements/);
    expect(permStep!.sql).toMatch(/grant usage on schema/i);
  });

  test("verifyInitSetup handles pg_stat_statements not installed", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, extensions, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // pg_stat_statements is NOT installed - empty result
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 0, rows: [] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    // Should pass without errors - missing extension shouldn't cause failure
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);
    // Should have queried for pg_stat_statements schema location
    expect(calls.some((c) => c.includes("pg_extension e") && c.includes("pg_stat_statements"))).toBe(true);
  });

  test("verifyInitSetup skips extension schema check when in pg_catalog", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // pg_stat_statements is in pg_catalog (standard location)
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "pg_catalog" }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    // Should pass - pg_catalog doesn't need extra USAGE grant
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);
    // Should NOT have queried for has_schema_privilege on pg_catalog specifically
    // (the code skips the check for pg_catalog and public schemas)
    const pgCatalogPrivCheck = calls.filter(
      (c) => c.includes("has_schema_privilege") && c.includes("pg_catalog")
    );
    // Should only have the standard public schema check, not a pg_catalog check for extension
    expect(pgCatalogPrivCheck.length).toBe(0);
  });

  test("verifyInitSetup skips extension schema check when in public", async () => {
    const calls: string[] = [];
    const client = {
      query: async (sql: string, params?: any) => {
        calls.push(String(sql));

        if (String(sql).toLowerCase().startsWith("begin isolation level repeatable read")) {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).toLowerCase() === "rollback;") {
          return { rowCount: 1, rows: [] };
        }
        if (String(sql).includes("select rolconfig")) {
          return { rowCount: 1, rows: [{ rolconfig: ['search_path=postgres_ai, "$user", public, pg_catalog'] }] };
        }
        if (String(sql).includes("from pg_catalog.pg_roles")) {
          return { rowCount: 1, rows: [{ rolname: DEFAULT_MONITORING_USER }] };
        }
        if (String(sql).includes("has_database_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("pg_has_role")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_table_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("to_regclass")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        if (String(sql).includes("has_function_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }
        // pg_stat_statements is in public schema
        if (String(sql).includes("pg_extension e") && String(sql).includes("pg_stat_statements")) {
          return { rowCount: 1, rows: [{ schema: "public" }] };
        }
        if (String(sql).includes("has_schema_privilege")) {
          return { rowCount: 1, rows: [{ ok: true }] };
        }

        throw new Error(`unexpected sql: ${sql} params=${JSON.stringify(params)}`);
      },
    };

    const r = await init.verifyInitSetup({
      client: client as any,
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      includeOptionalPermissions: false,
    });
    // Should pass - public doesn't need extra USAGE grant for extension
    expect(r.ok).toBe(true);
    expect(r.missingRequired.length).toBe(0);
  });

  test("buildInitPlan preserves comments when filtering ALTER USER", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
      provider: "supabase",
    });
    const permStep = plan.steps.find((s) => s.name === "03.permissions");
    expect(permStep).toBeDefined();
    // Should have removed ALTER USER but kept comments
    expect(permStep!.sql.toLowerCase()).not.toMatch(/^\s*alter\s+user/m);
    // Should still have comment lines
    expect(permStep!.sql).toMatch(/^--/m);
  });

  test("validateProvider returns null for known providers", () => {
    expect(init.validateProvider(undefined)).toBe(null);
    expect(init.validateProvider("self-managed")).toBe(null);
    expect(init.validateProvider("supabase")).toBe(null);
  });

  test("validateProvider returns warning for unknown providers", () => {
    const warning = init.validateProvider("unknown-provider");
    expect(warning).not.toBe(null);
    expect(warning).toMatch(/Unknown provider/);
    expect(warning).toMatch(/unknown-provider/);
  });

  test("redactPasswordsInSql redacts password literals with embedded quotes", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pa'ss",
      includeOptionalPermissions: false,
    });
    const step = plan.steps.find((s: { name: string }) => s.name === "01.role");
    expect(step).toBeTruthy();
    const redacted = init.redactPasswordsInSql(step!.sql);
    expect(redacted).toMatch(/password '<redacted>'/i);
  });

  // Tests for buildUninitPlan
  test("buildUninitPlan generates correct steps with dropRole=true", async () => {
    const plan = await init.buildUninitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      dropRole: true,
    });

    expect(plan.database).toBe("mydb");
    expect(plan.monitoringUser).toBe(DEFAULT_MONITORING_USER);
    expect(plan.dropRole).toBe(true);
    expect(plan.steps.length).toBe(3);
    expect(plan.steps.map((s) => s.name)).toEqual([
      "01.drop_helpers",
      "02.revoke_permissions",
      "03.drop_role",
    ]);
  });

  test("buildUninitPlan skips role drop when dropRole=false", async () => {
    const plan = await init.buildUninitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      dropRole: false,
    });

    expect(plan.dropRole).toBe(false);
    expect(plan.steps.length).toBe(2);
    expect(plan.steps.map((s) => s.name)).toEqual([
      "01.drop_helpers",
      "02.revoke_permissions",
    ]);
  });

  test("buildUninitPlan skips role drop for supabase provider", async () => {
    const plan = await init.buildUninitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      dropRole: true,
      provider: "supabase",
    });

    // Even with dropRole=true, supabase provider skips role operations
    expect(plan.steps.length).toBe(2);
    expect(plan.steps.some((s) => s.name === "03.drop_role")).toBe(false);
  });

  test("buildUninitPlan handles special characters in identifiers", async () => {
    const monitoringUser = 'user "with" quotes';
    const database = 'db "name"';
    const plan = await init.buildUninitPlan({
      database,
      monitoringUser,
      dropRole: true,
    });

    // Check that identifiers are properly quoted in SQL
    const dropHelpersStep = plan.steps.find((s) => s.name === "01.drop_helpers");
    expect(dropHelpersStep).toBeTruthy();

    const revokeStep = plan.steps.find((s) => s.name === "02.revoke_permissions");
    expect(revokeStep).toBeTruthy();
    expect(revokeStep!.sql).toContain('"user ""with"" quotes"');
    expect(revokeStep!.sql).toContain('"db ""name"""');

    const dropRoleStep = plan.steps.find((s) => s.name === "03.drop_role");
    expect(dropRoleStep).toBeTruthy();
    // Uses ROLE_LITERAL (single-quoted) for format('%I', ...) in dynamic SQL
    expect(dropRoleStep!.sql).toContain("'user \"with\" quotes'");
  });

  test("buildUninitPlan rejects identifiers with null bytes", async () => {
    await expect(
      init.buildUninitPlan({
        database: "mydb",
        monitoringUser: "bad\0user",
        dropRole: true,
      })
    ).rejects.toThrow(/Identifier cannot contain null bytes/);
  });

  test("applyUninitPlan continues on errors and reports them", async () => {
    const plan = {
      monitoringUser: DEFAULT_MONITORING_USER,
      database: "mydb",
      dropRole: true,
      steps: [
        { name: "01.drop_helpers", sql: "drop function if exists postgres_ai.test()" },
        { name: "02.revoke_permissions", sql: "select 1/0" }, // Will fail
        { name: "03.drop_role", sql: "select 1" },
      ],
    };

    const calls: string[] = [];
    const client = {
      query: async (sql: string) => {
        calls.push(sql);
        if (sql === "begin;") return { rowCount: 1 };
        if (sql === "commit;") return { rowCount: 1 };
        if (sql === "rollback;") return { rowCount: 1 };
        if (sql.includes("1/0")) throw new Error("division by zero");
        return { rowCount: 1 };
      },
    };

    const result = await init.applyUninitPlan({ client: client as any, plan: plan as any });

    // Should have applied steps 1 and 3, with step 2 in errors
    expect(result.applied).toContain("01.drop_helpers");
    expect(result.applied).toContain("03.drop_role");
    expect(result.applied).not.toContain("02.revoke_permissions");
    expect(result.errors.length).toBe(1);
    expect(result.errors[0]).toMatch(/02\.revoke_permissions.*division by zero/);
  });

  test("buildInitPlan includes 02.extensions step with pg_stat_statements", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const extStep = plan.steps.find((s) => s.name === "02.extensions");
    expect(extStep).toBeTruthy();
    // Should create pg_stat_statements with IF NOT EXISTS
    expect(extStep!.sql).toMatch(/create extension if not exists pg_stat_statements/i);
  });

  test("buildInitPlan creates extensions before permissions", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const stepNames = plan.steps.map((s) => s.name);
    const extIndex = stepNames.indexOf("02.extensions");
    const permIndex = stepNames.indexOf("03.permissions");
    expect(extIndex).toBeGreaterThanOrEqual(0);
    expect(permIndex).toBeGreaterThanOrEqual(0);
    // Extensions should come before permissions
    expect(extIndex).toBeLessThan(permIndex);
  });

  test("buildInitPlan uses IF NOT EXISTS for postgres_ai schema (idempotent)", async () => {
    const plan = await init.buildInitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
    });

    const permStep = plan.steps.find((s) => s.name === "03.permissions");
    expect(permStep).toBeTruthy();
    // Should use IF NOT EXISTS for idempotent behavior
    expect(permStep!.sql).toMatch(/create schema if not exists postgres_ai/i);
  });

  test("buildUninitPlan does NOT drop pg_stat_statements extension", async () => {
    const plan = await init.buildUninitPlan({
      database: "mydb",
      monitoringUser: DEFAULT_MONITORING_USER,
      dropRole: true,
    });

    // Check all steps - none should drop pg_stat_statements
    for (const step of plan.steps) {
      expect(step.sql.toLowerCase()).not.toMatch(/drop extension.*pg_stat_statements/);
    }
  });
});

describe("CLI commands", () => {
  test("cli: prepare-db with missing connection prints help/options", () => {
    const r = runCli(["prepare-db"]);
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/--print-sql/);
    expect(r.stderr).toMatch(/--monitoring-user/);
  });

  test("cli: prepare-db --print-sql works without connection (offline mode)", () => {
    const r = runCli(["prepare-db", "--print-sql", "-d", "mydb", "--password", "monpw"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/SQL plan \(offline; not connected\)/);
    expect(r.stdout).toMatch(new RegExp(`grant connect on database "mydb" to "${DEFAULT_MONITORING_USER}"`, "i"));
  });

  test("cli: prepare-db --print-sql with --provider supabase skips role step", () => {
    const r = runCli(["prepare-db", "--print-sql", "-d", "mydb", "--password", "monpw", "--provider", "supabase"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/provider: supabase/);
    // Should not have 01.role step
    expect(r.stdout).not.toMatch(/-- 01\.role/);
    // Should have 02.extensions and 03.permissions steps
    expect(r.stdout).toMatch(/-- 02\.extensions/);
    expect(r.stdout).toMatch(/-- 03\.permissions/);
  });

  test("cli: prepare-db warns about unknown provider", () => {
    const r = runCli(["prepare-db", "--print-sql", "-d", "mydb", "--password", "monpw", "--provider", "unknown-cloud"]);
    expect(r.status).toBe(0);
    // Should warn about unknown provider
    expect(r.stderr).toMatch(/Unknown provider.*unknown-cloud/);
  });

  test("cli: prepare-db --reset-password with supabase provider would have no role step", async () => {
    // When using supabase provider, the role creation step is skipped.
    // This means --reset-password (which only runs 01.role) would have no steps.
    // The CLI should error in this case. We test the underlying plan logic here.
    const plan = await (await import("../lib/init")).buildInitPlan({
      database: "mydb",
      monitoringUser: "mon",
      monitoringPassword: "pw",
      includeOptionalPermissions: false,
      provider: "supabase",
    });
    // Simulate what --reset-password does: filter to only 01.role step
    const resetPasswordSteps = plan.steps.filter((s) => s.name === "01.role");
    // For supabase, this should be empty (role creation is skipped)
    expect(resetPasswordSteps.length).toBe(0);
  });

  test("pgai wrapper forwards to postgresai CLI", () => {
    const r = runPgai(["--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/postgresai|PostgresAI/i);
  });

  test("cli: prepare-db command exists and shows help", () => {
    const r = runCli(["prepare-db", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/monitoring user/i);
    expect(r.stdout).toMatch(/--print-sql/);
  });

  test("cli: mon local-install command exists and shows help", () => {
    const r = runCli(["mon", "local-install", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--demo/);
    expect(r.stdout).toMatch(/--api-key/);
  });

  test("cli: mon local-install --api-key and --db-url skip interactive prompts", () => {
    // This test verifies that when --api-key and --db-url are provided,
    // the CLI uses them directly without prompting for input.
    // The command will fail later (no Docker, invalid DB), but we check
    // that the options were parsed and used correctly.
    const r = runCli([
      "mon", "local-install",
      "--api-key", "test-api-key-12345",
      "--db-url", "postgresql://user:pass@localhost:5432/testdb"
    ]);

    // Should show that API key was provided via CLI option (not prompting)
    expect(r.stdout).toMatch(/Using API key provided via --api-key parameter/);
    // Should show that DB URL was provided via CLI option (not prompting)
    expect(r.stdout).toMatch(/Using database URL provided via --db-url parameter/);
  });

  test("cli: auth login --help shows --set-key option", () => {
    const r = runCli(["auth", "login", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--set-key/);
  });

  test("cli: mon local-install reads global --api-key option", () => {
    // The fix ensures --api-key works when passed as a global option (before subcommand)
    // Commander.js routes global options to program.opts(), not subcommand opts
    const r = runCli([
      "--api-key", "global-api-key-test",
      "mon", "local-install",
      "--db-url", "postgresql://user:pass@localhost:5432/testdb"
    ]);

    // Should detect the API key from global options
    expect(r.stdout).toMatch(/Using API key provided via --api-key parameter/);
  });

  test("cli: mon local-install works with --api-key after subcommand", () => {
    // Test that --api-key works when passed after the subcommand
    // Note: Commander.js routes --api-key to global opts, the fix reads from both
    const r = runCli([
      "mon", "local-install",
      "--api-key", "test-key-after-subcommand",
      "--db-url", "postgresql://user:pass@localhost:5432/testdb"
    ]);

    // Should detect the API key regardless of position
    expect(r.stdout).toMatch(/Using API key provided via --api-key parameter/);
    // Verify the key was saved
    expect(r.stdout).toMatch(/API key saved/);
  });

  test("cli: mon local-install with --yes and no --api-key skips API setup", () => {
    // When --yes is provided without --api-key, the CLI should skip
    // the interactive prompt and proceed without API key
    const r = runCli([
      "mon", "local-install",
      "--db-url", "postgresql://user:pass@localhost:5432/testdb",
      "--yes"
    ]);

    // Should indicate auto-yes mode without API key
    expect(r.stdout).toMatch(/Auto-yes mode: no API key provided/);
    expect(r.stderr).toMatch(/Reports will be generated locally only/);
  });

  test("cli: mon local-install --demo configures demo monitoring target", () => {
    // --demo should copy instances.demo.yml to instances.yml and print confirmation.
    // The command will fail later (no Docker), but we verify the demo target step succeeded.
    const r = runCli(["mon", "local-install", "--demo"]);
    expect(r.stdout).toMatch(/Demo mode enabled/);
    expect(r.stdout).toMatch(/Demo monitoring target configured/);
  });

  test("cli: mon local-install --demo with global --api-key shows error", () => {
    // When --demo is used with global --api-key, it should still be detected and error
    const r = runCli([
      "--api-key", "global-api-key-test",
      "mon", "local-install",
      "--demo"
    ]);

    // Should reject demo mode with API key (from global option)
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/Cannot use --api-key with --demo mode/);
  });

  // Tests for unprepare-db command
  test("cli: unprepare-db with missing connection prints help/options", () => {
    const r = runCli(["unprepare-db"]);
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/--print-sql/);
    expect(r.stderr).toMatch(/--monitoring-user/);
  });

  test("cli: unprepare-db --print-sql works without connection (offline mode)", () => {
    const r = runCli(["unprepare-db", "--print-sql", "-d", "mydb"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/SQL plan \(offline; not connected\)/);
    expect(r.stdout).toMatch(/drop schema if exists postgres_ai/i);
  });

  test("cli: unprepare-db --print-sql with --keep-role skips role drop", () => {
    const r = runCli(["unprepare-db", "--print-sql", "-d", "mydb", "--keep-role"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/drop role: false/);
    // Should not have 03.drop_role step
    expect(r.stdout).not.toMatch(/-- 03\.drop_role/);
    // Should have 01 and 02 steps
    expect(r.stdout).toMatch(/-- 01\.drop_helpers/);
    expect(r.stdout).toMatch(/-- 02\.revoke_permissions/);
  });

  test("cli: unprepare-db --print-sql with --provider supabase skips role step", () => {
    const r = runCli(["unprepare-db", "--print-sql", "-d", "mydb", "--provider", "supabase"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/provider: supabase/);
    // Should not have 03.drop_role step
    expect(r.stdout).not.toMatch(/-- 03\.drop_role/);
  });

  test("cli: unprepare-db command exists and shows help", () => {
    const r = runCli(["unprepare-db", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--keep-role/);
    expect(r.stdout).toMatch(/--print-sql/);
    expect(r.stdout).toMatch(/--force/);
  });
});

// Check if Docker is available for imageTag tests
function isDockerAvailable(): boolean {
  try {
    const result = Bun.spawnSync(["docker", "info"], { timeout: 5000 });
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

const dockerAvailable = isDockerAvailable();

describe.skipIf(!dockerAvailable)("imageTag priority behavior", () => {
  // Tests for the imageTag priority: --tag flag > PGAI_TAG env var > pkg.version
  // This verifies the fix that prevents stale .env PGAI_TAG from being used
  // These tests require Docker and spawn subprocesses so need longer timeout

  let tempDir: string;

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-test-"));
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("stale .env PGAI_TAG is NOT used - CLI version takes precedence", () => {
    // Create a stale .env with an old tag value
    const testDir = resolve(tempDir, "stale-tag-test");
    fs.mkdirSync(testDir, { recursive: true });
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=beta\n");
    // Create minimal docker-compose.yml so resolvePaths() finds it
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    // Run from the test directory (so resolvePaths finds docker-compose.yml)
    // Note: Command may hang on Docker check in CI without Docker, so we use a timeout
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"], {
      env: { ...process.env, PGAI_TAG: undefined },
      cwd: testDir,
      timeout: 30000, // Kill subprocess after 30s if it hangs on Docker
    });

    // Read the .env that was written
    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // The .env should NOT contain the stale "beta" tag - it should use pkg.version
    expect(envContent).not.toMatch(/PGAI_TAG=beta/);
    // It should contain the CLI version (0.0.0-dev.0 in dev)
    expect(envContent).toMatch(/PGAI_TAG=\d+\.\d+\.\d+|PGAI_TAG=0\.0\.0-dev/);
  }, 60000);

  test("--tag flag takes priority over pkg.version", () => {
    const testDir = resolve(tempDir, "tag-flag-test");
    fs.mkdirSync(testDir, { recursive: true });
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "local-install", "--tag", "v1.2.3-custom", "--db-url", "postgresql://u:p@h:5432/d", "--yes"], {
      env: { ...process.env, PGAI_TAG: undefined },
      cwd: testDir,
      timeout: 30000,
    });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(envContent).toMatch(/PGAI_TAG=v1\.2\.3-custom/);

    // Verify stdout confirms the tag being used
    const stdout = new TextDecoder().decode(result.stdout);
    expect(stdout).toMatch(/Using image tag: v1\.2\.3-custom/);
  }, 60000);

  test("PGAI_TAG env var is intentionally ignored (Bun auto-loads .env)", () => {
    // Note: We do NOT use process.env.PGAI_TAG because Bun auto-loads .env files,
    // which would cause stale .env values to pollute the environment.
    // Users should use --tag flag to override, not env vars.
    const testDir = resolve(tempDir, "env-var-ignored-test");
    fs.mkdirSync(testDir, { recursive: true });
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"], {
      env: { ...process.env, PGAI_TAG: "v2.0.0-from-env" },
      cwd: testDir,
      timeout: 30000,
    });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    // PGAI_TAG env var should be IGNORED - uses pkg.version instead
    expect(envContent).not.toMatch(/PGAI_TAG=v2\.0\.0-from-env/);
    expect(envContent).toMatch(/PGAI_TAG=\d+\.\d+\.\d+|PGAI_TAG=0\.0\.0-dev/);
  }, 60000);

  test("existing registry and password are preserved while tag is updated", () => {
    const testDir = resolve(tempDir, "preserve-test");
    fs.mkdirSync(testDir, { recursive: true });
    // Create .env with stale tag but valid registry and password
    fs.writeFileSync(resolve(testDir, ".env"),
      "PGAI_TAG=stale-tag\nPGAI_REGISTRY=my.registry.com\nGF_SECURITY_ADMIN_PASSWORD=secret123\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"], {
      env: { ...process.env, PGAI_TAG: undefined },
      cwd: testDir,
      timeout: 30000,
    });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // Tag should be updated (not stale-tag)
    expect(envContent).not.toMatch(/PGAI_TAG=stale-tag/);

    // But registry and password should be preserved
    expect(envContent).toMatch(/PGAI_REGISTRY=my\.registry\.com/);
    expect(envContent).toMatch(/GF_SECURITY_ADMIN_PASSWORD=secret123/);
  }, 60000);
});
