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
  test("F002 validates with sample data", async () => {
    const mockClient = createMockClient({
      wraparoundDatabaseRows: [{ tag_datname: "testdb", age_datfrozenxid: "250000000", age_datminmxid: "1000" }],
      wraparoundTableRows: [{
        tag_schema_name: "public", tag_table_name: "events", tag_ranked_by: "xid",
        xid_age: "250000000", multixact_age: "1000", effective_freeze_max_age: "200000000",
        effective_multixact_freeze_max_age: "400000000", table_size_bytes: "1048576",
      }],
    });
    const report = await checkup.REPORT_GENERATORS.F002(mockClient as any, "node-01");
    validateAgainstSchema(report, "F002");
  });

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

  // WI #271: additive keeping-up fields (trigger math + queue/worker snapshot).
  test("F003 validates with keeping-up trigger + saturation data", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        {
          tag_schemaname: "public",
          tag_relname: "orders",
          n_live_tup: "10000000",
          n_dead_tup: "5000000",
          dead_pct: 33.33,
          n_mod_since_analyze: "0",
          n_ins_since_vacuum: "0",
          last_autovacuum: "0",
          last_vacuum: "0",
          last_autoanalyze: "0",
          autovacuum_count: "0",
          vacuum_count: "0",
          autovacuum_disabled: 0,
          toast_autovacuum_disabled: 0,
          reltuples: "10000000",
          relpages: "500000",
          eff_vacuum_threshold: "50",
          eff_vacuum_scale_factor: "0.2",
          vacuum_settings_from_reloptions: 0,
          eff_analyze_threshold: "50",
          eff_analyze_scale_factor: "0.1",
          eff_insert_threshold: "1000",
          eff_insert_scale_factor: "0.2",
          insert_settings_from_reloptions: 0,
          vacuum_trigger_point: "2000050",
          analyze_trigger_point: "1000050",
          insert_trigger_point: "2000050",
          over_trigger_ratio: "2.5",
          over_vacuum_trigger: 1,
          over_analyze_trigger: 0,
          over_insert_trigger: 0,
          table_size_b: "4294967296",
          relations_total: "120000",
          candidates_considered: "8000",
          queue_length: "3",
          analyze_queue_length: "0",
          insert_queue_length: "0",
          total_dead_tuples_all: "5000000",
        },
      ],
      autovacuumWorkerRows: [{ max_workers: "2", active_workers: "2", anti_wraparound_workers: "1" }],
      autovacuumBlockedRows: [
        { tag_worker_pid: "4242", tag_blocker_pid: "9001", tag_blocker_queryid: "12345", wait_seconds: "63.4" },
      ],
      vacuumProgressRows: [
        {
          tag_schema_name: "public",
          tag_table_name: "orders",
          tag_vacuum_mode: "aggressive_autovacuum",
          tag_phase: "3",
          heap_blks_total: "1000",
          heap_blks_scanned: "400",
          heap_blks_vacuumed: "200",
          index_vacuum_count: "1",
          is_anti_wraparound: "1",
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

  // F001 with a rich analysis payload (fired rules, throughput, largest tables)
  test("F001 validates with autovacuum-linter findings", async () => {
    const mockClient = createMockClient({
      settingsRows: [
        { tag_setting_name: "autovacuum", tag_setting_value: "off", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "bool", is_default: 0, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "autovacuum_vacuum_cost_delay", tag_setting_value: "20", tag_unit: "ms", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 0, setting_normalized: 0.02, unit_normalized: "seconds" },
        { tag_setting_name: "autovacuum_vacuum_cost_limit", tag_setting_value: "-1", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "vacuum_cost_limit", tag_setting_value: "200", tag_unit: "", tag_category: "Resource Usage", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "autovacuum_work_mem", tag_setting_value: "-1", tag_unit: "kB", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "maintenance_work_mem", tag_setting_value: "65536", tag_unit: "kB", tag_category: "Resource Usage", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "autovacuum_vacuum_scale_factor", tag_setting_value: "0.2", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "real", is_default: 1, setting_normalized: null, unit_normalized: null },
      ],
      autovacuumReloptsRows: [
        { tag_schemaname: "public", tag_relname: "events", tag_relkind: "r", tag_category: "largest", relpages: "9999999", total_relation_size_b: String(2 * 1024 * 1024 * 1024 * 1024), has_av_override: 0, reloptions: "", relations_total: "12000", tables_with_av_overrides: "1" },
        { tag_schemaname: "public", tag_relname: "hot", tag_relkind: "r", tag_category: "override", relpages: "5", total_relation_size_b: "8192", has_av_override: 1, reloptions: "autovacuum_vacuum_cost_delay=0", relations_total: "12000", tables_with_av_overrides: "1" },
      ],
    });
    const report = await checkup.REPORT_GENERATORS.F001(mockClient as any, "node-01");
    validateAgainstSchema(report, "F001");
  });

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
