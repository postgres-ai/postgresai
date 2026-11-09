import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";
import Ajv2020 from "ajv/dist/2020";
import * as checkup from "../lib/checkup";
import { D001_MAX_STORED_THRESHOLDS, d001Config, d001StoredMs, d001Thresholds } from "../lib/checkup";
import { createMockClient } from "./test-utils";

const schema = JSON.parse(readFileSync(resolve(import.meta.dir, "../../reporter/schemas/D001.schema.json"), "utf8"));

const SETTINGS = [
  { name: "log_min_duration_statement", setting: "-1", unit: "ms", category: "c", context: "superuser", vartype: "integer",
    source: "default", boot_val: "-1", pending_restart: false },
  { name: "auto_explain.log_min_duration", setting: "10000", unit: "ms", category: "c", context: "superuser", vartype: "integer",
    source: "configuration file", boot_val: "-1", pending_restart: false },
];

/** The base mock, plus the D001 queries by a fragment of their text; a fragment in `broken` throws. */
function d001Client(answers: Record<string, unknown[]> = {}, broken: string[] = []) {
  const base = createMockClient({ settingsRows: [
    { tag_setting_name: "log_min_duration_statement", tag_setting_value: "-1", tag_unit: "ms", tag_category: "c",
      tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
    { tag_setting_name: "work_mem", tag_setting_value: "4MB", tag_unit: "", tag_category: "c",
      tag_vartype: "string", is_default: 1, setting_normalized: null, unit_normalized: null },
  ] } as any) as any;
  const rows: Record<string, unknown[]> = {
    "current_user as role": [{ role: "mon", database: "app" }],
    "from pg_settings\nwhere name ~": SETTINGS,
    "from pg_db_role_setting": [
      { role: "app", database: null, config: ["statement_timeout=8s", "search_path=x"] },
      { role: null, database: null, config: ["log_statement=ddl"] },
    ],
    "r.rolcanlogin": [{ role: "app", superuser: false, member_of: ["m"], sessions: "2",
                        oldest_session_start: "2026-11-01T00:00:00Z", applications: ["api"] }],
    pg_has_role: [{ visible: true }],
    "from pg_extension": [{ schema: "extensions" }],
    "with s as": [{ role: "app", groups: "40", calls: "900", longest_ms: "12500.04", stats_since: "2026-10-01",
                    groups_at_or_over: { "1000": 3, "10000": 1 }, calls_of_groups_at_or_over: { "1000": 70, "10000": 2 } }],
    pg_stat_statements_info: [{ dealloc: "0", stats_reset: "2026-10-01" }],
    "from pg_stat_database as d": [{ stats_reset: "2026-10-30T00:00:00Z", read_at: "2026-11-06T00:00:00Z",
                                     temp_files: "7", temp_bytes: "700", autovacuum_runs: "3" }],
    ...answers,
  };
  return {
    ...base,
    query: async (sql: string, params?: unknown[]) => {
      for (const key of broken) if (sql.includes(key)) throw new Error(`failed at ${key}`);
      for (const [key, value] of Object.entries(rows)) if (sql.includes(key)) return { rows: value };
      return base.query(sql, params);
    },
  };
}

async function reportOf(client: unknown) {
  const report = await checkup.REPORT_GENERATORS.D001(client as any, "n1");
  const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);
  expect(validate(report), JSON.stringify(validate.errors)).toBe(true);
  return (report as any).results.n1;
}

describe("D001 facts (the same as the reporter's D001 file)", () => {
  test("a role row keeps logging settings and the statement limit only", () => {
    expect(d001Config(["search_path=a", "log_min_duration_statement=1s", "statement_timeout=3s",
      "auto_explain.log_min_duration=1s", "work_mem=1GB"])).toEqual({
      log_min_duration_statement: "1s", statement_timeout: "3s", "auto_explain.log_min_duration": "1s",
    });
    expect(d001Config(null)).toEqual({});
  });

  test("the node's own positive thresholds join 1 s and 10 s", () => {
    expect(d001Thresholds({
      log_min_duration_statement: { setting: "250", unit: "ms" },
      "auto_explain.log_min_duration": { setting: "-1", unit: "ms" },
    })).toEqual([250, 1000, 10000]);
    expect(d001Thresholds({})).toEqual([1000, 10000]);
  });

  test("a value stored for a role is a threshold too", () => {
    expect(d001Thresholds({ log_min_duration_statement: { setting: "-1", unit: "ms" } }, {
      values: [
        { setting: "log_min_duration_statement", value: "5s" },
        { setting: "auto_explain.log_min_duration", value: "250" },
        { setting: "log_min_duration_sample", value: "1min" },
        { setting: "log_min_duration_statement", value: "-1" },
        { setting: "log_min_duration_statement", value: "0" },
        { setting: "statement_timeout", value: "3s" },
      ],
    })).toEqual([250, 1000, 5000, 10000, 60000]);
  });

  test("a stored value reads as Postgres reads it", () => {
    for (const [text, ms] of [["1.5s", 1500], ["1h", 3_600_000], ["1d", 86_400_000], ["1.005s", 1005], ["1.1h", 3_960_000], ["1500us", 2]] as const) {
      expect(d001StoredMs(text)).toBe(ms);
    }
    for (const text of ["5S", "garbage", "", "-1", "25d", "9".repeat(400)]) {
      expect(d001StoredMs(text)).toBeNull();
    }
  });

  test("a stored value reaches the statement read", async () => {
    const node = await reportOf(d001Client({ "from pg_db_role_setting": [
      { role: "app", database: null, config: ["log_min_duration_statement=5s"] },
    ] }));
    expect(node.logging.statements.thresholds_ms).toContain(5000);
  });

  test("stored thresholds are capped", () => {
    const values = Array.from({ length: 99 }, (_, n) => ({ setting: "log_min_duration_statement", value: `${n + 1}ms` }));
    expect(d001Thresholds({}, { values }).length).toBe(D001_MAX_STORED_THRESHOLDS + 2);
  });

  test("the file has the settings with their context and every fact, and passes the schema", async () => {
    const node = await reportOf(d001Client());
    expect(node.data.log_min_duration_statement.context).toBe("superuser");
    expect(node.logging.setting_sources["auto_explain.log_min_duration"]).toEqual(
      { source: "configuration file", is_default: false, pending_restart: false });
    expect(node.logging.setting_sources.log_min_duration_statement.is_default).toBe(true);
    expect(node.logging.overrides.rows_read).toBe(2);
    expect(node.logging.overrides.values).toEqual([
      { scope: "role", role: "app", database: null, setting: "statement_timeout", value: "8s" },
      { scope: "all", role: null, database: null, setting: "log_statement", value: "ddl" },
    ]);
    expect(node.logging.login_roles[0].sessions).toBe(2);
    const role = node.logging.statements.by_role[0];
    expect(role.longest_call_ms).toBe(12500);
    expect(role.groups_at_or_over_ms).toEqual({ "1000": 3, "10000": 1 });
    expect(role.calls_of_groups_at_or_over_ms["1000"]).toBe(70);
    expect(node.logging.temp_files).toEqual(
      { start: "2026-10-30T00:00:00Z", end: "2026-11-06T00:00:00Z", days: 7, total: 7 });
  });

  test("no reset time (PG15+ before the first reset): the totals stand, with no window", async () => {
    const node = await reportOf(d001Client({ "from pg_stat_database as d": [
      { stats_reset: null, read_at: "2026-11-06T00:00:00Z", temp_files: "7", temp_bytes: "700", autovacuum_runs: "3" }] }));
    expect(node.logging.temp_files).toEqual({ start: null, end: "2026-11-06T00:00:00Z", days: null, total: 7 });
    expect(node.logging.autovacuum_runs.total).toBe(3);
    expect(node.logging.temp_files_error).toBeUndefined();
  });

  test("a reader that cannot see other sessions writes unknown, not zero", async () => {
    const node = await reportOf(d001Client({ pg_has_role: [{ visible: false }] }));
    expect(node.logging.login_roles[0].sessions).toBeNull();
    expect(node.logging.sessions_error).toContain("pg_read_all_stats");
  });

  test("a failed session check keeps the facts and says unknown", async () => {
    const node = await reportOf(d001Client({}, ["pg_has_role"]));
    expect(node.logging.login_roles[0].sessions).toBeNull();
    expect(node.logging.sessions_error).toContain("failed at");
    expect(node.logging.statements.by_role.length).toBe(1);
  });

  test("a settings read that fails leaves no sources and keeps the metric's rows", async () => {
    const node = await reportOf(d001Client({}, ["from pg_settings\nwhere name ~"]));
    expect(node.logging.settings_error).toContain("failed at");
    expect(node.logging.setting_sources).toBeUndefined();
    expect(node.data.log_min_duration_statement.context).toBe("");
    expect(node.data.work_mem).toBeUndefined();
  });

  test("no pg_stat_statements leaves the calls out with the reason, never zero", async () => {
    const node = await reportOf(d001Client({ "from pg_extension": [] }));
    expect(node.logging.statements).toBeUndefined();
    expect(node.logging.statements_error).toContain("pg_stat_statements");
  });
});
