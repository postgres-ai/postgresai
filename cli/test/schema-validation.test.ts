/**
 * JSON Schema validation tests for express checkup reports.
 * Validates that generated reports match schemas in reporter/schemas/.
 */
import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { readFileSync } from "fs";
import Ajv2020 from "ajv/dist/2020";

import * as checkup from "../lib/checkup";
import { createMockClient } from "./test-utils";

const ajv = new Ajv2020({ allErrors: true, strict: false });
const schemasDir = resolve(import.meta.dir, "../../reporter/schemas");

function validateAgainstSchema(report: any, checkId: string): void {
  const schemaPath = resolve(schemasDir, `${checkId}.schema.json`);
  const schema = JSON.parse(readFileSync(schemaPath, "utf8"));
  const validate = ajv.compile(schema);
  const valid = validate(report);
  if (!valid) {
    const errors = validate.errors?.map(e => `${e.instancePath}: ${e.message}`).join(", ");
    throw new Error(`${checkId} schema validation failed: ${errors}`);
  }
}

// Test data for index reports
const indexTestData = {
  H001: {
    emptyRows: { invalidIndexesRows: [] },
    dataRows: {
      invalidIndexesRows: [
        { schema_name: "public", table_name: "users", index_name: "users_email_idx", relation_name: "users", index_size_bytes: "1048576", index_definition: "CREATE INDEX users_email_idx ON public.users USING btree (email)", supports_fk: false },
      ],
    },
  },
  H002: {
    emptyRows: { unusedIndexesRows: [] },
    dataRows: {
      unusedIndexesRows: [
        { schema_name: "public", table_name: "logs", index_name: "logs_created_idx", index_definition: "CREATE INDEX logs_created_idx ON public.logs USING btree (created_at)", reason: "Never Used Indexes", idx_scan: "0", index_size_bytes: "8388608", idx_is_btree: true, supports_fk: false },
      ],
    },
  },
  H004: {
    emptyRows: { redundantIndexesRows: [] },
    dataRows: {
      redundantIndexesRows: [
        { schema_name: "public", table_name: "orders", index_name: "orders_user_id_idx", relation_name: "orders", access_method: "btree", reason: "public.orders_user_id_created_idx", index_size_bytes: "2097152", table_size_bytes: "16777216", index_usage: "0", supports_fk: false, index_definition: "CREATE INDEX orders_user_id_idx ON public.orders USING btree (user_id)", redundant_to_json: JSON.stringify([{ index_name: "public.orders_user_id_created_idx", index_definition: "CREATE INDEX ...", index_size_bytes: 1048576 }]) },
      ],
    },
  },
};

describe("Schema validation", () => {
  // Index health checks (H001, H002, H004) - test empty and with data
  for (const [checkId, testData] of Object.entries(indexTestData)) {
    const generator = checkup.REPORT_GENERATORS[checkId];

    test(`${checkId} validates with empty data`, async () => {
      const mockClient = createMockClient(testData.emptyRows);
      const report = await generator(mockClient as any, "node-01");
      validateAgainstSchema(report, checkId);
    });

    test(`${checkId} validates with sample data`, async () => {
      const mockClient = createMockClient(testData.dataRows);
      const report = await generator(mockClient as any, "node-01");
      validateAgainstSchema(report, checkId);
    });
  }

  // F003 (Autovacuum: dead tuples) - test empty and with data
  test("F003 validates with empty data", async () => {
    const mockClient = createMockClient({ deadTuplesRows: [] });
    const report = await checkup.REPORT_GENERATORS.F003(mockClient as any, "node-01");
    validateAgainstSchema(report, "F003");
  });

  test("F003 validates with sample data", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        {
          tag_schemaname: "public",
          tag_relname: "events",
          n_live_tup: "6361538",
          n_dead_tup: "8270000",
          dead_pct: 56.52,
          last_autovacuum: "0",
          last_vacuum: "1704067200",
          autovacuum_count: "0",
          vacuum_count: "1",
          autovacuum_disabled: 1,
          table_size_b: "2147483648",
        },
      ],
    });
    const report = await checkup.REPORT_GENERATORS.F003(mockClient as any, "node-01");
    validateAgainstSchema(report, "F003");
  });

  test("F009 validates with healthy snapshot data", async () => {
    const mockClient = createMockClient();
    const report = await checkup.REPORT_GENERATORS.F009(mockClient as any, "node-01");
    validateAgainstSchema(report, "F009");
  });

  test("F009 validates with pg_flight_recorder history", async () => {
    const mockClient = createMockClient({
      xminHorizonRows: [{
        is_in_recovery: false,
        skip_reason: null,
        snapshot_xmin: "1000",
        autovacuum_freeze_max_age: "200000000",
        has_full_visibility: true,
        query_preview_enabled: true,
        pg_flight_recorder_detected: true,
        data_horizon_age_tx: "0",
        catalog_horizon_age_tx: "0",
        components: {
          pg_stat_activity: { age_tx: 0, count: 0, top_blocker: null },
          pg_replication_slots: { age_tx: 0, count: 0, top_blocker: null },
          pg_replication_slots_catalog: { age_tx: 0, count: 0, top_blocker: null },
          pg_stat_replication: { age_tx: 0, count: 0, top_blocker: null },
          pg_prepared_xacts: { age_tx: 0, count: 0, top_blocker: null },
        },
      }],
    });
    const report = await checkup.REPORT_GENERATORS.F009(mockClient as any, "node-01");
    validateAgainstSchema(report, "F009");
  });

  for (const checkId of ["F004", "F005"]) {
    test(`${checkId} distinguishes a healthy empty result from missing schema`, async () => {
      const healthyClient = createMockClient();
      const healthy = await checkup.REPORT_GENERATORS[checkId](healthyClient as any, "node-01");
      const healthyDb = healthy.results["node-01"].data.testdb;
      expect(healthyDb.status).toEqual({ ok: true, reason: null, error: null });
      validateAgainstSchema(healthy, checkId);

      const missingSchemaClient = createMockClient({
        bloatCapabilityRows: [
          { schema_exists: false, schema_usage: false, view_exists: false, view_select: false },
        ],
      });
      const degraded = await checkup.REPORT_GENERATORS[checkId](missingSchemaClient as any, "node-01");
      const degradedDb = degraded.results["node-01"].data.testdb;
      expect(degradedDb.status).toEqual({
        ok: false,
        reason: "missing_schema",
        error: 'schema "postgres_ai" does not exist',
      });
      validateAgainstSchema(degraded, checkId);
    });

    test(`${checkId} exposes missing-grant degradation`, async () => {
      const client = createMockClient({
        bloatCapabilityRows: [
          { schema_exists: true, schema_usage: true, view_exists: true, view_select: false },
        ],
      });
      const report = await checkup.REPORT_GENERATORS[checkId](client as any, "node-01");
      const dbEntry = report.results["node-01"].data.testdb;
      expect(dbEntry.status).toEqual({
        ok: false,
        reason: "missing_grant",
        error: "permission denied for relation postgres_ai.pg_statistic",
      });
      validateAgainstSchema(report, checkId);
    });
  }

  // Settings reports (D004, F001, G001) - single test each
  for (const checkId of ["D004", "F001", "G001"]) {
    test(`${checkId} validates against schema`, async () => {
      const mockClient = createMockClient();
      const report = await checkup.REPORT_GENERATORS[checkId](mockClient as any, "node-01");
      validateAgainstSchema(report, checkId);
    });
  }

  test("I001 validates with available pg_stat_io data", () => {
    const report = {
      version: null,
      build_ts: null,
      generation_mode: null,
      checkId: "I001",
      checkTitle: "I/O statistics (pg_stat_io)",
      timestamptz: new Date("2026-01-01T00:00:00.000Z").toISOString(),
      nodes: { primary: "node-01", standbys: [] },
      results: {
        "node-01": {
          data: {
            available: true,
            by_backend_type: [{
              backend_type: "total",
              reads: 10,
              read_bytes_mb: 64,
              read_time_ms: 20,
              writes: 5,
              write_bytes_mb: 32,
              write_time_ms: 10,
              writebacks: 4,
              writeback_bytes_mb: 16,
              writeback_time_ms: 8,
              fsyncs: 2,
              fsync_time_ms: 6,
              extends: 3,
              extend_bytes_mb: 24,
              hits: 90,
              evictions: 7,
              reuses: 11,
            }],
            analysis: {
              total_read_mb: 64,
              total_write_mb: 32,
              total_io_time_ms: 30,
              read_hit_ratio_pct: 90,
              avg_read_time_ms: 2,
              avg_write_time_ms: 2,
            },
            stats_reset_s: 7200,
          },
        },
      },
    };

    validateAgainstSchema(report, "I001");
  });

  test("I001 validates with unavailable pg_stat_io data", () => {
    const report = {
      version: null,
      build_ts: null,
      generation_mode: null,
      checkId: "I001",
      checkTitle: "I/O statistics (pg_stat_io)",
      timestamptz: new Date("2026-01-01T00:00:00.000Z").toISOString(),
      nodes: { primary: "node-01", standbys: [] },
      results: {
        "node-01": {
          data: {
            available: false,
            min_version_required: "16",
            by_backend_type: [],
            analysis: {
              total_read_mb: 0,
              total_write_mb: 0,
              total_io_time_ms: 0,
              read_hit_ratio_pct: 0,
              avg_read_time_ms: null,
              avg_write_time_ms: null,
            },
            stats_reset_s: null,
          },
        },
      },
    };

    validateAgainstSchema(report, "I001");
  });
});
