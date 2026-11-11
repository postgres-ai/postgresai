import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";
import Ajv2020 from "ajv/dist/2020";
import * as checkup from "../lib/checkup";
import { A005_MAX_DATABASES, d004PerRound, setA005OpenDatabase } from "../lib/checkup";
import { createMockClient } from "./test-utils";

const schema = JSON.parse(readFileSync(resolve(import.meta.dir, "../../reporter/schemas/D004.schema.json"), "utf8"));

const SETTINGS = [
  { name: "pg_stat_statements.max", setting: "5000", unit: "", category: "c", context: "postmaster", vartype: "integer",
    source: "default", boot_val: "5000", pending_restart: false },
  { name: "shared_preload_libraries", setting: "pg_stat_statements", unit: "", category: "c", context: "postmaster",
    vartype: "string", source: "configuration file", boot_val: "", pending_restart: false },
];

/** The base mock, plus the D004 queries by a fragment of their text; a fragment in `broken` throws. */
function d004Client(answers: Record<string, unknown[]> = {}, broken: string[] = []) {
  const base = createMockClient({} as any) as any;
  const rows: Record<string, unknown[]> = {
    "current_user as role": [{ role: "mon", database: "app" }],
    "from pg_settings\nwhere name ~": SETTINGS,
    "from pg_available_extensions": [{ name: "pg_stat_statements", default_version: "1.11", installed_version: "1.10" }],
    "from pg_db_role_setting": [{ role: "app", database: null, config: ["pg_stat_statements.track=all", "work_mem=1GB"] }],
    "where d.datallowconn": [{ datname: "app", is_current: true, can_connect: true },
                             { datname: "other", is_current: false, can_connect: true }],
    "from pg_extension as e": [{ name: "pg_stat_statements", version: "1.10", schema: "extensions" }],
    called_once: [{ database: "app", entries: "4800", calls: "90000", called_once: "3000", temp_table_text: "2500",
                    texts_visible: true }],
    pg_stat_statements_info: [{ dealloc: "2577", stats_reset: new Date("2026-07-31T21:31:00Z"), read_at: new Date("2026-09-23T08:42:00Z") }],
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
  const report = await checkup.REPORT_GENERATORS.D004(client as any, "n1");
  const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);
  expect(validate(report), JSON.stringify(validate.errors)).toBe(true);
  return (report as any).results.n1;
}

describe("D004 facts (the same as the reporter's D004 file)", () => {
  test("a round removes 5% but at least 10 and at most all", () => {
    expect([5000, 4785, 100, 7, 0].map(d004PerRound)).toEqual([250, 239, 10, 7, 0]);
  });

  test("states per database, entries against the maximum and deallocation rounds", async () => {
    const previous = setA005OpenDatabase(async () => {
      const other = d004Client({ "from pg_extension as e": [] });
      return { ...other, end: async () => {} } as any;
    });
    try {
      const obs = (await reportOf(d004Client())).observability;
      expect(obs.available).toEqual({ pg_stat_statements: { default_version: "1.11" } });
      expect(obs.databases.app.installed.pg_stat_statements).toEqual({ version: "1.10", schema: "extensions" });
      expect(obs.databases.other.installed).toEqual({});
      expect(obs.settings["pg_stat_statements.max"]).toMatchObject({ context: "postmaster", is_default: true });
      expect(obs.overrides.values).toEqual([
        { scope: "role", role: "app", database: null, setting: "pg_stat_statements.track", value: "all" }]);
      expect(obs.statements).toMatchObject({ entries: 4800, dealloc: 2577, per_round_at_max: 250, database: "app",
        days_since_reset: 53.47, stats_reset: "2026-07-31T21:31:00.000Z" });
      expect(obs.statements.by_database[0]).toEqual(
        { database: "app", entries: 4800, calls: 90000, called_once: 3000, temp_table_text: 2500 });
      expect(obs.monitored_calls_window_error).toContain("no monitoring");
    } finally {
      setA005OpenDatabase(previous);
    }
  });

  test("past the database cap the rest are not read", async () => {
    const many = [{ datname: "app", is_current: true, can_connect: true },
      ...Array.from({ length: A005_MAX_DATABASES + 2 }, (_, i) => ({ datname: `db${i}`, is_current: false, can_connect: true }))];
    const previous = setA005OpenDatabase(async () => ({ ...d004Client({ "from pg_extension as e": [] }), end: async () => {} }) as any);
    try {
      const obs = (await reportOf(d004Client({ "where d.datallowconn": many }))).observability;
      const capped = Object.values(obs.databases as Record<string, { error?: string }>).filter((d) => d.error?.includes("more than"));
      expect(capped.length).toBe(2);
      expect(obs.statements.entries).toBe(4800);
    } finally {
      setA005OpenDatabase(previous);
    }
  });

  test("the view in another database is read there and its connection closed", async () => {
    let ended = 0;
    const previous = setA005OpenDatabase(async () => ({
      ...d004Client({ called_once: [{ database: "other", entries: "7", calls: "1", called_once: "1", temp_table_text: "0", texts_visible: true }] }),
      end: async () => { ended += 1; },
    }) as any);
    try {
      const obs = (await reportOf(d004Client({ "from pg_extension as e": [] }))).observability;
      expect(obs.statements).toMatchObject({ database: "other", entries: 7 });
      expect(ended).toBe(1);
    } finally {
      setA005OpenDatabase(previous);
    }
  });

  test("with the view in both, the connected database wins and the other connection is closed", async () => {
    let ended = 0;
    const previous = setA005OpenDatabase(async () => ({
      ...d004Client({ called_once: [{ database: "other", entries: "7", calls: "1", called_once: "1", temp_table_text: "0", texts_visible: true }] }),
      end: async () => { ended += 1; },
    }) as any);
    try {
      const obs = (await reportOf(d004Client({ "where d.datallowconn": [
        { datname: "other", is_current: false, can_connect: true }, { datname: "app", is_current: true, can_connect: true }] }))).observability;
      expect(obs.statements).toMatchObject({ database: "app", entries: 4800 });
      expect(ended).toBe(1);
    } finally {
      setA005OpenDatabase(previous);
    }
  });

  test("a database it cannot open is recorded; the rest stands", async () => {
    const obs = (await reportOf(d004Client())).observability; // a mock cannot reconnect
    expect(obs.databases.other.error).toContain("cannot open another connection");
    expect(obs.statements.entries).toBe(4800);
  });

  test("no pg_stat_statements anywhere is said, never zero entries", async () => {
    const obs = (await reportOf(d004Client({ "from pg_extension as e": [], "where d.datallowconn": [
      { datname: "app", is_current: true, can_connect: true, roles_with_create: 0 }] }))).observability;
    expect(obs.statements).toBeUndefined();
    expect(obs.statements_error).toContain("not created");
  });

  test("hidden texts leave the temporary-table count out", async () => {
    const obs = (await reportOf(d004Client({ called_once: [{ database: "app", entries: "10", calls: "1",
      called_once: "1", temp_table_text: "0", texts_visible: false }] }))).observability;
    expect(obs.statements.by_database[0].temp_table_text).toBeUndefined();
    expect(obs.statements.temp_table_text_error).toContain("pg_read_all_stats");
  });

  test("a failed database list keeps the facts read before it", async () => {
    const obs = (await reportOf(d004Client({}, ["where d.datallowconn"]))).observability;
    expect(obs.settings["pg_stat_statements.max"]).toBeDefined();
    expect(obs.available.pg_stat_statements).toBeDefined();
    expect(obs.databases).toBeUndefined();
    expect(obs.statements).toBeUndefined();
    expect(obs.statements_error).toBeUndefined();
    expect(obs.databases_error).toContain("failed at");
  });

  test("a failed fact is left out with its reason", async () => {
    const obs = (await reportOf(d004Client({}, ["from pg_available_extensions", "pg_stat_statements_info"]))).observability;
    expect(obs.available).toBeUndefined();
    expect(obs.available_error).toContain("failed at");
    expect(obs.statements.dealloc).toBeUndefined();
    expect(obs.statements.dealloc_error).toContain("failed at");
  });
});
