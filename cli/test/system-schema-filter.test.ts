/**
 * System-schema exclusion for the per-index checks (#345).
 *
 * A customer checkup listed `pg_catalog.pg_class_tblspc_relfilenode_index`
 * as an unused index to drop. Catalog indexes cannot be dropped, so they must
 * never reach an index-health report. Three layers are covered here:
 *
 *   1. the SQL in config/pgwatch-{prometheus,postgres}/metrics.yml, for every
 *      PG version key of every affected metric;
 *   2. the copy embedded into the CLI at build time;
 *   3. the defensive filter in the H001/H002/H004 getters, which protects
 *      reports produced by a CLI built against an older metrics.yml.
 */
import { describe, test, expect } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";

import * as checkup from "../lib/checkup";
import { getMetricSql, isSystemSchema, METRIC_NAMES } from "../lib/metrics-loader";
import { createMockClient } from "./test-utils";

const configDir = resolve(import.meta.dir, "../../config");

type MetricsYml = {
  metrics: Record<string, { sqls: Record<string, string> }>;
};

const loadMetrics = (relPath: string) =>
  Bun.YAML.parse(readFileSync(resolve(configDir, relPath), "utf8")) as MetricsYml;

const promMetrics = loadMetrics("pgwatch-prometheus/metrics.yml");
const pgMetrics = loadMetrics("pgwatch-postgres/metrics.yml");

/**
 * The per-index metrics that must never surface a system-schema row, with the
 * number of row-emitting CTEs each one needs the exclusion on
 * (index_definitions needs two — `indexes` and `redundant_index_pairs`).
 *
 * The exact count detects a DELETED insertion point, which `.toMatch` would
 * not. It says nothing about placement: a predicate moved into a non-emitting
 * `fk_indexes` CTE would keep the count. Placement is proven at runtime by the
 * substitution control test in checkup.integration.test.ts.
 */
const INDEX_METRICS: Array<[string, MetricsYml, number]> = [
  ["pg_invalid_indexes", promMetrics, 1],
  ["redundant_indexes", promMetrics, 1],
  ["unused_indexes", promMetrics, 1],
  ["rarely_used_indexes", promMetrics, 1],
  ["index_definitions", pgMetrics, 2],
];

const countOccurrences = (sql: string, re: RegExp) =>
  (sql.match(new RegExp(re.source, "g")) ?? []).length;

/**
 * The exclusion is expressed against whichever alias holds the schema name
 * (`n.nspname`, `pn.nspname` or `tnsp.nspname`), so match on the predicate
 * rather than on a fixed alias.
 */
const NOT_IN_SYSTEM_SCHEMAS =
  /\bnot in \('pg_catalog', 'information_schema', 'pg_toast'\)/;
/** Temp schemas are matched with one POSIX regex, not two LIKE predicates. */
const NOT_MATCH_TEMP_SCHEMAS = /!~ '\^pg_\(toast_\)\?temp_'/;
/**
 * Guards the regex form against a revert to `not like 'pg\_temp\_%'`. The
 * reason is readability, not correctness: one predicate covers both temp
 * variants with no LIKE escaping to reason about. PostgreSQL keeps `\_`
 * intact under either standard_conforming_strings setting, so the LIKE form
 * was NOT GUC-dependent — commit ada60bb claims otherwise and is wrong.
 */
const NO_LIKE_ESCAPE = /not like 'pg\\_/;

describe("#345 - isSystemSchema", () => {
  for (const schema of ["pg_catalog", "information_schema", "pg_toast"]) {
    test(`treats ${schema} as a system schema`, () => {
      expect(isSystemSchema(schema)).toBe(true);
    });
  }

  test("treats per-backend temp schemas as system schemas", () => {
    expect(isSystemSchema("pg_temp_1")).toBe(true);
    expect(isSystemSchema("pg_temp_374")).toBe(true);
    expect(isSystemSchema("pg_toast_temp_1")).toBe(true);
    expect(isSystemSchema("pg_toast_temp_374")).toBe(true);
  });

  test("leaves user schemas alone", () => {
    for (const schema of ["public", "app", "postgres_ai", "reporting", "pgcatalog"]) {
      expect(isSystemSchema(schema)).toBe(false);
    }
  });

  test("does not match schemas that merely start with pg_", () => {
    // Only the reserved pg_temp_N / pg_toast_temp_N schemas are excluded;
    // an extension schema such as pg_partman must still be checked.
    expect(isSystemSchema("pg_temp")).toBe(false);
    expect(isSystemSchema("pg_partman")).toBe(false);
    expect(isSystemSchema("pg_stat_statements")).toBe(false);
  });

  test("does not treat the $other$ aggregate sentinel as a system schema", () => {
    expect(isSystemSchema("$other$")).toBe(false);
  });

  test("handles null/undefined/empty input", () => {
    expect(isSystemSchema(null)).toBe(false);
    expect(isSystemSchema(undefined)).toBe(false);
    expect(isSystemSchema("")).toBe(false);
  });
});

describe("#345 - metrics.yml excludes system schemas", () => {
  for (const [metricName, doc, expectedOccurrences] of INDEX_METRICS) {
    const metric = doc.metrics[metricName];

    test(`${metricName} exists in metrics.yml`, () => {
      expect(metric).toBeDefined();
      expect(Object.keys(metric.sqls).length).toBeGreaterThan(0);
    });

    for (const versionKey of Object.keys(metric?.sqls ?? {})) {
      test(`${metricName} (sqls: ${versionKey}) filters out system schemas`, () => {
        const sql = metric.sqls[versionKey];
        expect(sql).toMatch(NOT_IN_SYSTEM_SCHEMAS);
        expect(sql).toMatch(NOT_MATCH_TEMP_SCHEMAS);
        expect(sql).not.toMatch(NO_LIKE_ESCAPE);
        // Exact count: .toMatch is satisfied by a single occurrence, which
        // would silently pass if one of the two insertion points were lost.
        expect(countOccurrences(sql, NOT_IN_SYSTEM_SCHEMAS)).toBe(expectedOccurrences);
        expect(countOccurrences(sql, NOT_MATCH_TEMP_SCHEMAS)).toBe(expectedOccurrences);
      });
    }
  }
});

describe("#345 - embedded metrics keep the exclusion", () => {
  // The CLI runs the copy generated by scripts/embed-metrics.ts, not the YAML.
  for (const checkId of ["H001", "H002", "H004"] as const) {
    test(`${checkId} (${METRIC_NAMES[checkId]}) embedded SQL filters out system schemas`, () => {
      const sql = getMetricSql(METRIC_NAMES[checkId], 16);
      expect(sql).toMatch(NOT_IN_SYSTEM_SCHEMAS);
      expect(sql).toMatch(NOT_MATCH_TEMP_SCHEMAS);
      expect(sql).not.toMatch(NO_LIKE_ESCAPE);
    });
  }
});

describe("#345 - getters drop system-schema rows", () => {
  const invalidRow = (schema: string, indexName: string) => ({
    schema_name: schema,
    table_name: "some_table",
    index_name: indexName,
    relation_name: `${schema}.some_table`,
    index_definition: `CREATE INDEX ${indexName} ON ${schema}.some_table USING btree (a)`,
    index_size_bytes: "16384",
    is_pk: false,
    is_unique: false,
    constraint_name: null,
    table_row_estimate: "1000",
    has_valid_duplicate: false,
    valid_index_name: null,
    valid_index_definition: null,
    supports_fk: 0,
  });

  const unusedRow = (schema: string, indexName: string) => ({
    schema_name: schema,
    table_name: "some_table",
    index_name: indexName,
    index_definition: `CREATE INDEX ${indexName} ON ${schema}.some_table USING btree (a)`,
    reason: "Never Used Indexes",
    index_size_bytes: "4194304",
    idx_scan: "0",
    idx_is_btree: true,
    supports_fk: false,
  });

  const redundantRow = (schema: string, indexName: string) => ({
    schema_name: schema,
    table_name: "some_table",
    index_name: indexName,
    relation_name: `${schema}.some_table`,
    access_method: "btree",
    reason: "some_table_a_b_idx",
    index_size_bytes: "4194304",
    table_size_bytes: "16777216",
    index_usage: "0",
    supports_fk: false,
    index_definition: `CREATE INDEX ${indexName} ON ${schema}.some_table USING btree (a)`,
    redundant_to_json: "[]",
  });

  test("getInvalidIndexes keeps public and drops pg_catalog (H001)", async () => {
    const client = createMockClient({
      invalidIndexesRows: [
        invalidRow("pg_catalog", "pg_class_tblspc_relfilenode_index"),
        invalidRow("public", "orders_created_idx"),
      ],
    });

    const indexes = await checkup.getInvalidIndexes(client as any);
    expect(indexes.map((i) => i.schema_name)).toEqual(["public"]);
    expect(indexes.map((i) => i.index_name)).toEqual(["orders_created_idx"]);
  });

  test("getUnusedIndexes keeps public and drops pg_catalog (H002)", async () => {
    const client = createMockClient({
      unusedIndexesRows: [
        unusedRow("pg_catalog", "pg_class_tblspc_relfilenode_index"),
        unusedRow("public", "products_old_idx"),
      ],
    });

    const indexes = await checkup.getUnusedIndexes(client as any);
    expect(indexes.map((i) => i.schema_name)).toEqual(["public"]);
    expect(indexes.map((i) => i.index_name)).toEqual(["products_old_idx"]);
  });

  test("getRedundantIndexes keeps public and drops pg_catalog (H004)", async () => {
    const client = createMockClient({
      redundantIndexesRows: [
        redundantRow("pg_catalog", "pg_depend_reference_index"),
        redundantRow("public", "orders_a_idx"),
      ],
    });

    const indexes = await checkup.getRedundantIndexes(client as any);
    expect(indexes.map((i) => i.schema_name)).toEqual(["public"]);
    expect(indexes.map((i) => i.index_name)).toEqual(["orders_a_idx"]);
  });

  test("every system schema variant is dropped by all three getters", async () => {
    const systemSchemas = [
      "pg_catalog",
      "information_schema",
      "pg_toast",
      "pg_temp_3",
      "pg_toast_temp_3",
    ];

    const client = createMockClient({
      invalidIndexesRows: systemSchemas.map((s) => invalidRow(s, `${s}_idx`)),
      unusedIndexesRows: systemSchemas.map((s) => unusedRow(s, `${s}_idx`)),
      redundantIndexesRows: systemSchemas.map((s) => redundantRow(s, `${s}_idx`)),
    });

    expect(await checkup.getInvalidIndexes(client as any)).toEqual([]);
    expect(await checkup.getUnusedIndexes(client as any)).toEqual([]);
    expect(await checkup.getRedundantIndexes(client as any)).toEqual([]);
  });

  test("the $other$ aggregate row survives the filter", async () => {
    // Only that the filter does not eat the sentinel. Whether the aggregate row
    // should count toward total_count at all is pre-existing behaviour, out of
    // scope for #345, and deliberately not asserted as correct here.
    const client = createMockClient({
      unusedIndexesRows: [unusedRow("$other$", "$other$")],
    });

    const indexes = await checkup.getUnusedIndexes(client as any);
    expect(indexes.length).toBe(1);
    expect(indexes[0].schema_name).toBe("$other$");
  });
});

describe("#345 - generated reports carry no system-schema rows", () => {
  test("generateH002 counts only the user-schema index", async () => {
    const client = createMockClient({
      unusedIndexesRows: [
        {
          schema_name: "pg_catalog",
          table_name: "pg_class",
          index_name: "pg_class_tblspc_relfilenode_index",
          index_definition:
            "CREATE INDEX pg_class_tblspc_relfilenode_index ON pg_catalog.pg_class USING btree (reltablespace, relfilenode)",
          reason: "Never Used Indexes",
          index_size_bytes: "1000000",
          idx_scan: "0",
          idx_is_btree: true,
          supports_fk: false,
        },
        {
          schema_name: "public",
          table_name: "logs",
          index_name: "logs_created_idx",
          index_definition: "CREATE INDEX logs_created_idx ON public.logs USING btree (created_at)",
          reason: "Never Used Indexes",
          index_size_bytes: "8388608",
          idx_scan: "0",
          idx_is_btree: true,
          supports_fk: false,
        },
      ],
    });

    const report = await checkup.generateH002(client as any, "test-node");
    const dbData = (report.results["test-node"].data as any).testdb;

    expect(dbData.total_count).toBe(1);
    expect(dbData.total_size_bytes).toBe(8388608);
    expect(JSON.stringify(report)).not.toContain("pg_class_tblspc_relfilenode_index");
  });
});
