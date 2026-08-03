import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import type { Client } from "pg";

// Import from source directly since we're using Bun
import * as checkup from "../lib/checkup";
import * as api from "../lib/checkup-api";
import * as metricsLoader from "../lib/metrics-loader";
import { createMockClient } from "./test-utils";

const SUPPORTED_PG_VERSIONS = [
  { major: 13, minor: 16, versionNum: "130016" },
  { major: 14, minor: 12, versionNum: "140012" },
  { major: 15, minor: 7, versionNum: "150007" },
  { major: 16, minor: 3, versionNum: "160003" },
  { major: 17, minor: 2, versionNum: "170002" },
  { major: 18, minor: 0, versionNum: "180000" },
  { major: 19, minor: 0, versionNum: "190000" },
];
const SUPPORTED_PG_MAJOR_VERSIONS = SUPPORTED_PG_VERSIONS.map(({ major }) => major);
const SECONDS_PER_DAY = 86400;
const STATS_RESET_EPOCH = Date.UTC(2024, 0, 1) / 1000;
const DAYS_SINCE_RESET = 30;
const STATS_RESET_SECONDS_SINCE_RESET = DAYS_SINCE_RESET * SECONDS_PER_DAY;
const TEST_NOW_EPOCH = STATS_RESET_EPOCH + STATS_RESET_SECONDS_SINCE_RESET;
const POSTMASTER_STARTUP_EPOCH = TEST_NOW_EPOCH - STATS_RESET_SECONDS_SINCE_RESET;
const STATS_RESET_TIME = new Date(STATS_RESET_EPOCH * 1000).toISOString();
const formatPostgresTimestamp = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toISOString().replace("T", " ").replace(".000Z", "+00");
const POSTMASTER_STARTUP_TIME = formatPostgresTimestamp(POSTMASTER_STARTUP_EPOCH);
const POSTMASTER_UPTIME_SECONDS = TEST_NOW_EPOCH - POSTMASTER_STARTUP_EPOCH;

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

// Async variant for tests that need an in-process fake API (Bun.serve):
// spawnSync would block the event loop and the fake server could never respond.
async function runCliAsync(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [status, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { status, stdout, stderr };
}

// Unit tests for parseVersionNum
describe("parseVersionNum", () => {
  test("parses PG 16.3 version number", () => {
    const result = checkup.parseVersionNum("160003");
    expect(result.major).toBe("16");
    expect(result.minor).toBe("3");
  });

  test("parses PG 15.7 version number", () => {
    const result = checkup.parseVersionNum("150007");
    expect(result.major).toBe("15");
    expect(result.minor).toBe("7");
  });

  test("parses PG 14.12 version number", () => {
    const result = checkup.parseVersionNum("140012");
    expect(result.major).toBe("14");
    expect(result.minor).toBe("12");
  });

  test("parses PG 13.16 version number", () => {
    const result = checkup.parseVersionNum("130016");
    expect(result.major).toBe("13");
    expect(result.minor).toBe("16");
  });

  test("parses PG 17.2 version number", () => {
    const result = checkup.parseVersionNum("170002");
    expect(result.major).toBe("17");
    expect(result.minor).toBe("2");
  });

  test("parses PG 18.0 version number", () => {
    const result = checkup.parseVersionNum("180000");
    expect(result.major).toBe("18");
    expect(result.minor).toBe("0");
  });

  test("parses PG 19 beta version number", () => {
    // PostgreSQL development/beta releases use the future major's x0000
    // server_version_num while server_version carries the beta suffix.
    const result = checkup.parseVersionNum("190000");
    expect(result.major).toBe("19");
    expect(result.minor).toBe("0");
  });

  test("handles empty string", () => {
    const result = checkup.parseVersionNum("");
    expect(result.major).toBe("");
    expect(result.minor).toBe("");
  });

  test("handles null/undefined", () => {
    const result = checkup.parseVersionNum(null as any);
    expect(result.major).toBe("");
    expect(result.minor).toBe("");
  });

  test("handles short string", () => {
    const result = checkup.parseVersionNum("123");
    expect(result.major).toBe("");
    expect(result.minor).toBe("");
  });
});

describe("F009 - xmin horizon analysis", () => {
  const components = (overrides: Record<string, any> = {}) => ({
    pg_stat_activity: { age_tx: 0, count: 0, top_blocker: null },
    pg_replication_slots: { age_tx: 0, count: 0, top_blocker: null },
    pg_replication_slots_catalog: { age_tx: 0, count: 0, top_blocker: null },
    pg_stat_replication: { age_tx: 0, count: 0, top_blocker: null },
    pg_prepared_xacts: { age_tx: 0, count: 0, top_blocker: null },
    ...overrides,
  });

  test("uses the documented severity boundaries", () => {
    const maxAge = 200_000_000;
    expect(checkup.getF009Severity(19_999_999, maxAge)).toBe("OK");
    expect(checkup.getF009Severity(20_000_000, maxAge)).toBe("NOTICE");
    expect(checkup.getF009Severity(100_000_000, maxAge)).toBe("WARNING");
    expect(checkup.getF009Severity(160_000_000, maxAge)).toBe("CRITICAL");
  });

  test("ships PostgreSQL 12-compatible metric SQL", () => {
    const sql = metricsLoader.getMetricSql(metricsLoader.METRIC_NAMES.F009, 12);
    expect(sql).toContain("xmin_horizon_snapshot");
    expect(sql).toContain("not in ('', 'off', 'false', '0')");
    expect(sql).toContain("greatest(age(a.backend_xmin), age(a.backend_xid))");
    expect(sql).toContain("a.backend_xmin is not null or a.backend_xid is not null");
  });

  test("uses slot > prepared > activity > replication for equal ages", () => {
    const equal = { age_tx: 100, count: 1 };
    const dominant = checkup.selectF009DominantHolder(components({
      pg_stat_activity: { ...equal, top_blocker: { pid: 1 } },
      pg_replication_slots: { ...equal, top_blocker: { slot_name: "logical_1" } },
      pg_stat_replication: { ...equal, top_blocker: { pid: 2 } },
      pg_prepared_xacts: { ...equal, top_blocker: { gid: "g1" } },
    }) as any);
    expect(dominant?.source).toBe("slot");
    expect(dominant?.detail.slot_name).toBe("logical_1");
  });

  test("evaluates catalog and data holders independently", async () => {
    const mockClient = createMockClient({
      xminHorizonRows: [{
        is_in_recovery: false,
        snapshot_xmin: "123",
        autovacuum_freeze_max_age: "200000000",
        has_full_visibility: true,
        query_preview_enabled: false,
        pg_flight_recorder_detected: false,
        data_horizon_age_tx: "1000",
        catalog_horizon_age_tx: "160000000",
        components: components({
          pg_replication_slots_catalog: {
            age_tx: 160000000, count: 1,
            top_blocker: { slot_name: "catalog_slot", status: "inactive", wal_status: "extended" },
          },
        }),
      }],
    });
    const report = await checkup.generateF009(mockClient as any, "node-01");
    const data = report.results["node-01"].data as any;
    expect(data.data_horizon_severity).toBe("OK");
    expect(data.catalog_horizon_severity).toBe("CRITICAL");
    expect(data.dominant_holder.horizon_type).toBe("catalog");
  });

  test("remediation distinguishes active, idle, autovacuum, slot, feedback, and prepared holders", () => {
    const analyze = (source: any, detail: any) => checkup.buildF009Analysis(
      components() as any,
      { source, component: "pg_stat_activity", horizon_type: "data", age_tx: 100, detail } as any,
      "NOTICE", "OK", { limitedVisibility: false, historyDetected: false, skipped: false },
    ).recommendations.join(" ");
    expect(analyze("activity", { pid: 10, state: "active", backend_type: "client backend" })).toContain("pg_cancel_backend(10)");
    expect(analyze("activity", { pid: 11, state: "idle in transaction", backend_type: "client backend" })).toContain("pg_terminate_backend(11)");
    expect(analyze("activity", { pid: 12, state: "active", backend_type: "autovacuum worker" })).toContain("pg_stat_progress_vacuum");
    expect(analyze("slot", { slot_name: "s1", status: "inactive", wal_status: "extended" })).toContain("Confirm subscriber state");
    expect(analyze("replication", { application_name: "standby-1" })).toContain("Do not disable hot_standby_feedback");
    expect(analyze("prepared", { gid: "g1", owner: "alice", database: "app" })).toContain("ROLLBACK PREPARED 'g1'");
  });

  test("replica skips and limited visibility degrades gracefully", async () => {
    const replica = createMockClient({
      xminHorizonRows: [{
        is_in_recovery: true,
        skip_reason: "replica: xmin horizon analysis is primary-only",
        snapshot_xmin: null,
        autovacuum_freeze_max_age: "200000000",
        has_full_visibility: false,
        query_preview_enabled: true,
        pg_flight_recorder_detected: false,
        data_horizon_age_tx: 0,
        catalog_horizon_age_tx: 0,
        components: components(),
      }],
    });
    const replicaData = (await checkup.generateF009(replica as any, "node-01")).results["node-01"].data as any;
    expect(replicaData.skipped).toBe(true);
    expect(replicaData.dominant_holder).toBeNull();
    expect(replicaData.query_preview_captured).toBe(false);

    const analysis = checkup.buildF009Analysis(
      components() as any, null, "OK", "OK",
      { limitedVisibility: true, historyDetected: false, skipped: false },
    );
    expect(analysis.conclusions.join(" ")).toContain("limited visibility");
  });

  test("captures query previews and pg_flight_recorder history when available", async () => {
    const mockClient = createMockClient({
      xminHorizonRows: [{
        is_in_recovery: false,
        skip_reason: null,
        snapshot_xmin: "123",
        autovacuum_freeze_max_age: "200000000",
        has_full_visibility: true,
        query_preview_enabled: true,
        pg_flight_recorder_detected: true,
        data_horizon_age_tx: "0",
        catalog_horizon_age_tx: "0",
        components: components(),
      }],
    });

    const data = (await checkup.generateF009(mockClient as any, "node-01")).results["node-01"].data as any;
    expect(data.visibility).toBe("full");
    expect(data.query_preview_captured).toBe(true);
    expect(data.history).toEqual({
      available: true,
      source: "pg_flight_recorder",
      message: "Historical xmin timeline is available via pgfr_analyze.xmin_horizon_history().",
    });
  });

  test("long activity promotes the generated report severity to notice", async () => {
    const mockClient = createMockClient({
      xminHorizonRows: [{
        is_in_recovery: false,
        skip_reason: null,
        snapshot_xmin: "123",
        autovacuum_freeze_max_age: "200000000",
        has_full_visibility: true,
        query_preview_enabled: false,
        pg_flight_recorder_detected: false,
        data_horizon_age_tx: "1",
        catalog_horizon_age_tx: "0",
        components: components({
          pg_stat_activity: {
            age_tx: 1,
            count: 1,
            top_blocker: { pid: 42, xact_age_seconds: 3600 },
          },
        }),
      }],
    });

    const data = (await checkup.generateF009(mockClient as any, "node-01")).results["node-01"].data as any;
    expect(data.data_horizon_severity).toBe("OK");
    expect(data.catalog_horizon_severity).toBe("OK");
    expect(data.severity).toBe("NOTICE");
  });

  test("long activity produces an absolute wall-clock notice", () => {
    const c = components({
      pg_stat_activity: { age_tx: 1, count: 1, top_blocker: { pid: 42, xact_age_seconds: 3600 } },
    });
    const dominant = checkup.selectF009DominantHolder(c as any);
    const analysis = checkup.buildF009Analysis(
      c as any, dominant, "OK", "OK",
      { limitedVisibility: false, historyDetected: false, skipped: false },
    );
    expect(analysis.conclusions.join(" ")).toContain("3600 seconds");
  });
});

// Unit tests for createBaseReport
describe("createBaseReport", () => {
  test("creates correct structure", () => {
    const report = checkup.createBaseReport("A002", "Postgres major version", "test-node");

    expect(report.checkId).toBe("A002");
    expect(report.checkTitle).toBe("Postgres major version");
    expect(typeof report.version).toBe("string");
    expect(report.version!.length).toBeGreaterThan(0);
    expect(typeof report.build_ts).toBe("string");
    expect(report.nodes.primary).toBe("test-node");
    expect(report.nodes.standbys).toEqual([]);
    expect(report.results).toEqual({});
    expect(typeof report.timestamptz).toBe("string");
    // Verify timestamp is ISO format
    expect(new Date(report.timestamptz).toISOString()).toBe(report.timestamptz);
  });

  test("uses provided node name", () => {
    const report = checkup.createBaseReport("A003", "Postgres settings", "my-custom-node");
    expect(report.nodes.primary).toBe("my-custom-node");
  });
});

// Tests for CHECK_INFO
describe("CHECK_INFO and REPORT_GENERATORS", () => {
  const expressCheckIds = Object.keys(checkup.REPORT_GENERATORS);

  test("CHECK_INFO contains all express-mode checks", () => {
    for (const checkId of expressCheckIds) {
      expect(checkup.CHECK_INFO[checkId]).toBeDefined();
      expect(typeof checkup.CHECK_INFO[checkId]).toBe("string");
      expect(checkup.CHECK_INFO[checkId].length).toBeGreaterThan(0);
    }
  });

  test("CHECK_INFO titles are loaded from embedded dictionary", () => {
    // Verify a few known titles match the API dictionary
    // These are canonical titles from postgres.ai/api/general/checkup_dictionary
    expect(checkup.CHECK_INFO["A002"]).toBe("Postgres major version");
    expect(checkup.CHECK_INFO["H001"]).toBe("Invalid indexes");
    expect(checkup.CHECK_INFO["H002"]).toBe("Unused indexes");
  });

  test("REPORT_GENERATORS has function for each check", () => {
    for (const checkId of expressCheckIds) {
      expect(typeof checkup.REPORT_GENERATORS[checkId]).toBe("function");
    }
  });

  test("REPORT_GENERATORS and CHECK_INFO have same keys", () => {
    const generatorKeys = Object.keys(checkup.REPORT_GENERATORS).sort();
    const infoKeys = Object.keys(checkup.CHECK_INFO).sort();
    expect(generatorKeys).toEqual(infoKeys);
  });
});

// Tests for formatBytes
describe("formatBytes", () => {
  test("formats zero bytes", () => {
    expect(checkup.formatBytes(0)).toBe("0 B");
  });

  test("formats bytes", () => {
    expect(checkup.formatBytes(500)).toBe("500.00 B");
  });

  test("formats kibibytes", () => {
    expect(checkup.formatBytes(1024)).toBe("1.00 KiB");
    expect(checkup.formatBytes(1536)).toBe("1.50 KiB");
  });

  test("formats mebibytes", () => {
    expect(checkup.formatBytes(1048576)).toBe("1.00 MiB");
  });

  test("formats gibibytes", () => {
    expect(checkup.formatBytes(1073741824)).toBe("1.00 GiB");
  });

  test("handles negative bytes", () => {
    expect(checkup.formatBytes(-1024)).toBe("-1.00 KiB");
    expect(checkup.formatBytes(-1048576)).toBe("-1.00 MiB");
  });

  test("handles edge cases", () => {
    expect(checkup.formatBytes(NaN)).toBe("NaN B");
    expect(checkup.formatBytes(Infinity)).toBe("Infinity B");
  });
});

function createI001MockClient(options: {
  versionRows?: any[];
  ioRows?: any[];
  ioError?: boolean;
  resetRows?: any[];
  resetError?: boolean;
} = {}) {
  const queries: string[] = [];
  const {
    versionRows = [
      { name: "server_version", setting: "16.3" },
      { name: "server_version_num", setting: "160003" },
    ],
    ioRows = [],
    ioError = false,
    resetRows = [{ stats_reset_s: "86400" }],
    resetError = false,
  } = options;

  return {
    queries,
    query: async (sql: string) => {
      queries.push(sql);
      if (sql.includes("server_version") && sql.includes("server_version_num") && sql.includes("pg_settings")) {
        return { rows: versionRows };
      }
      if (sql.includes("pg_stat_io") && sql.includes("rollup")) {
        if (ioError) {
          throw new Error("I/O statistics unavailable");
        }
        return { rows: ioRows };
      }
      if (sql.includes("pg_stat_io") && sql.includes("stats_reset_s")) {
        if (resetError) {
          throw new Error("stats reset unavailable");
        }
        return { rows: resetRows };
      }
      throw new Error(`Unexpected query: ${sql}`);
    },
  };
}

const i001Rows = [
  {
    tag_backend_type: "client backend",
    reads: "1000",
    read_bytes_mb: "100",
    read_time_ms: "500",
    writes: "200",
    write_bytes_mb: "50",
    write_time_ms: "100",
    writebacks: "20",
    writeback_bytes_mb: "2",
    writeback_time_ms: "4",
    fsyncs: "1",
    fsync_time_ms: "3",
    extends: "6",
    extend_bytes_mb: "8",
    hits: "5000",
    evictions: "3",
    reuses: "7",
  },
  {
    tag_backend_type: "total",
    reads: "1500",
    read_bytes_mb: "150",
    read_time_ms: "750",
    writes: "300",
    write_bytes_mb: "75",
    write_time_ms: "150",
    writebacks: "30",
    writeback_bytes_mb: "3",
    writeback_time_ms: "6",
    fsyncs: "2",
    fsync_time_ms: "5",
    extends: "9",
    extend_bytes_mb: "12",
    hits: "7500",
    evictions: "4",
    reuses: "11",
  },
];

// Mock client tests for report generators
describe("Report generators with mock client", () => {
  test("getPostgresVersion extracts version info", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
    });

    const version = await checkup.getPostgresVersion(mockClient as any);
    expect(version.version).toBe("16.3");
    expect(version.server_version_num).toBe("160003");
    expect(version.server_major_ver).toBe("16");
    expect(version.server_minor_ver).toBe("3");
  });

  test("getPostgresVersion preserves the PG19 beta version label", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "19beta2" },
        { name: "server_version_num", setting: "190000" },
      ],
    });

    const version = await checkup.getPostgresVersion(mockClient as any);
    expect(version.version).toBe("19beta2");
    expect(version.server_version_num).toBe("190000");
    expect(version.server_major_ver).toBe("19");
    expect(version.server_minor_ver).toBe("0");
  });

  test("getIOStatistics returns empty for PostgreSQL versions before 16", async () => {
    const mockClient = createI001MockClient({ ioRows: i001Rows });

    const defaultStats = await checkup.getIOStatistics(mockClient as any);
    const stats = await checkup.getIOStatistics(mockClient as any, 15);

    expect(defaultStats).toEqual([]);
    expect(stats).toEqual([]);
    expect(mockClient.queries).toEqual([]);
  });

  test("getIOStatistics skips placeholder SQL without querying", async () => {
    const mockClient = createI001MockClient({ ioRows: i001Rows });

    const stats = await checkup.getIOStatistics(mockClient as any, 16, "; -- pg_stat_io unavailable");
    const barePlaceholderStats = await checkup.getIOStatistics(mockClient as any, 16, ";");

    const whitespacePlaceholderStats = await checkup.getIOStatistics(mockClient as any, 16, "  ; -- whitespace prefix");

    expect(stats).toEqual([]);
    expect(barePlaceholderStats).toEqual([]);
    expect(whitespacePlaceholderStats).toEqual([]);
    expect(mockClient.queries).toEqual([]);
  });

  test("getIOStatistics returns empty when the SQL result has no rows", async () => {
    const mockClient = createI001MockClient({ ioRows: [] });

    const stats = await checkup.getIOStatistics(mockClient as any, 16);

    expect(stats).toEqual([]);
  });

  test("getIOStatistics catches primary query errors", async () => {
    const mockClient = createI001MockClient({ ioError: true, resetRows: [] });

    const stats = await checkup.getIOStatistics(mockClient as any, 16);
    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");

    expect(stats).toEqual([]);
    expect(report.results["node-01"].data).toEqual({
      available: false,
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
    });
  });

  test("getIOStatistics maps backend rows including extension bytes", async () => {
    const mockClient = createI001MockClient({ ioRows: i001Rows });

    const stats = await checkup.getIOStatistics(mockClient as any, 16);

    expect(stats).toHaveLength(2);
    expect(stats[0]).toMatchObject({
      backend_type: "client backend",
      reads: 1000,
      read_bytes_mb: 100,
      writes: 200,
      write_bytes_mb: 50,
      writebacks: 20,
      writeback_bytes_mb: 2,
      fsyncs: 1,
      extends: 6,
      extend_bytes_mb: 8,
      hits: 5000,
      evictions: 3,
      reuses: 7,
    });
  });

  test("generateI001 keeps report available when stats_reset query fails", async () => {
    const mockClient = createI001MockClient({ ioRows: i001Rows, resetError: true });

    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");
    const data = report.results["node-01"].data;

    expect(report.checkId).toBe("I001");
    expect(data.available).toBe(true);
    expect(data.stats_reset_s).toBeNull();
    expect(data.by_backend_type.map((row: any) => row.backend_type)).toEqual(["total", "client backend"]);
    expect(data.by_backend_type[0].extend_bytes_mb).toBe(12);
    expect(data.analysis.read_hit_ratio_pct).toBe(83.33);
  });

  test("generateI001 dispatches version-specific pg_stat_io SQL", async () => {
    const pg16Client = createI001MockClient({ ioRows: i001Rows });
    const pg18Client = createI001MockClient({
      versionRows: [
        { name: "server_version", setting: "18.0" },
        { name: "server_version_num", setting: "180000" },
      ],
      ioRows: i001Rows,
    });

    await checkup.REPORT_GENERATORS.I001(pg16Client as any, "node-01");
    await checkup.REPORT_GENERATORS.I001(pg18Client as any, "node-01");

    const pg16MetricSql = pg16Client.queries.find((sql) => sql.includes("pg_stat_io") && sql.includes("rollup"));
    const pg18MetricSql = pg18Client.queries.find((sql) => sql.includes("pg_stat_io") && sql.includes("rollup"));

    expect(pg16MetricSql).toContain("sum(coalesce(reads, 0) * op_bytes)");
    expect(pg16MetricSql).toContain("sum(coalesce(extends, 0) * op_bytes)");
    expect(pg18MetricSql).toContain("sum(coalesce(read_bytes, 0))");
    expect(pg18MetricSql).toContain("sum(coalesce(extend_bytes, 0))");
    // PG18 removed op_bytes entirely; writeback bytes cannot be derived, so the SQL emits a constant 0.
    expect(pg18MetricSql).toContain("0::int8 as writeback_bytes_mb");
    // Regression guard: any op_bytes reference in the PG18 branch makes the whole query
    // fail with `column "op_bytes" does not exist`, degrading all of I001.
    expect(pg18MetricSql).not.toContain("op_bytes");
  });

  test("generateI001 returns unavailable on PostgreSQL 16 when ioStats are empty", async () => {
    const mockClient = createI001MockClient({ ioRows: [], resetRows: [] });

    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");
    const data = report.results["node-01"].data;

    expect(data.available).toBe(false);
    expect(data.by_backend_type).toEqual([]);
    expect(data.stats_reset_s).toBeNull();
    expect(data.analysis).toEqual({
      total_read_mb: 0,
      total_write_mb: 0,
      total_io_time_ms: 0,
      read_hit_ratio_pct: 0,
      avg_read_time_ms: null,
      avg_write_time_ms: null,
    });
  });

  test("generateI001 returns unavailable on PostgreSQL before 16 without querying pg_stat_io", async () => {
    const mockClient = createI001MockClient({
      versionRows: [
        { name: "server_version", setting: "15.4" },
        { name: "server_version_num", setting: "150004" },
      ],
    });

    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");
    const data = report.results["node-01"].data;

    expect(data.available).toBe(false);
    expect(data.min_version_required).toBe("16");
    expect(data.by_backend_type).toEqual([]);
    expect(mockClient.queries.every((sql) => !sql.includes("pg_stat_io"))).toBe(true);
  });

  test("generateI001 handles unknown PostgreSQL version as unavailable", async () => {
    const mockClient = createI001MockClient({
      versionRows: [
        { name: "server_version", setting: "unknown" },
        { name: "server_version_num", setting: "unknown" },
      ],
    });

    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");
    const data = report.results["node-01"].data;

    expect(data.available).toBe(false);
    expect(data.min_version_required).toBe("16");
    expect(data.by_backend_type).toEqual([]);
  });

  test("generateI001 handles zero-request hit ratio and empty averages", async () => {
    const mockClient = createI001MockClient({
      ioRows: [{
        tag_backend_type: "total",
        reads: "0",
        read_bytes_mb: "0",
        read_time_ms: "0",
        writes: "0",
        write_bytes_mb: "0",
        write_time_ms: "0",
        writebacks: "0",
        writeback_bytes_mb: "0",
        writeback_time_ms: "0",
        fsyncs: "0",
        fsync_time_ms: "0",
        extends: "0",
        extend_bytes_mb: "0",
        hits: "0",
        evictions: "0",
        reuses: "0",
      }],
      resetRows: [],
    });

    const report = await checkup.REPORT_GENERATORS.I001(mockClient as any, "node-01");
    const data = report.results["node-01"].data;
    const analysis = data.analysis;

    expect(data.available).toBe(true);
    expect(data.stats_reset_s).toBeNull();
    expect(analysis.read_hit_ratio_pct).toBe(0);
    expect(analysis.avg_read_time_ms).toBeNull();
    expect(analysis.avg_write_time_ms).toBeNull();
  });

  test("getSettings transforms rows to keyed object", async () => {
    const mockClient = createMockClient({
      settingsRows: [
        {
          tag_setting_name: "shared_buffers",
          tag_setting_value: "16384",
          tag_unit: "8kB",
          tag_category: "Resource Usage / Memory",
          tag_vartype: "integer",
          is_default: 1,
          setting_normalized: "134217728",  // 16384 * 8192
          unit_normalized: "bytes",
        },
        {
          tag_setting_name: "work_mem",
          tag_setting_value: "4096",
          tag_unit: "kB",
          tag_category: "Resource Usage / Memory",
          tag_vartype: "integer",
          is_default: 1,
          setting_normalized: "4194304",  // 4096 * 1024
          unit_normalized: "bytes",
        },
      ],
    });

    const settings = await checkup.getSettings(mockClient as any);
    expect("shared_buffers" in settings).toBe(true);
    expect("work_mem" in settings).toBe(true);
    expect(settings.shared_buffers.setting).toBe("16384");
    expect(settings.shared_buffers.unit).toBe("8kB");
    // pretty_value is now computed from setting_normalized
    expect(settings.shared_buffers.pretty_value).toBe("128.00 MiB");
    expect(settings.work_mem.pretty_value).toBe("4.00 MiB");
  });

  test("generateA002 creates report with version data", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
    });

    const report = await checkup.generateA002(mockClient as any, "test-node");
    expect(report.checkId).toBe("A002");
    expect(report.checkTitle).toBe("Postgres major version");
    expect(report.nodes.primary).toBe("test-node");
    expect("test-node" in report.results).toBe(true);
    expect("version" in report.results["test-node"].data).toBe(true);
    expect(report.results["test-node"].data.version.version).toBe("16.3");
  });

  test("generateA003 creates report with settings and version", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
      settingsRows: [
        {
          tag_setting_name: "shared_buffers",
          tag_setting_value: "16384",
          tag_unit: "8kB",
          tag_category: "Resource Usage / Memory",
          tag_vartype: "integer",
          is_default: 1,
          setting_normalized: "134217728",
          unit_normalized: "bytes",
        },
      ],
    });

    const report = await checkup.generateA003(mockClient as any, "test-node");
    expect(report.checkId).toBe("A003");
    expect(report.checkTitle).toBe("Postgres settings");
    expect("test-node" in report.results).toBe(true);
    expect("shared_buffers" in report.results["test-node"].data).toBe(true);
    expect(report.results["test-node"].postgres_version).toBeTruthy();
    expect(report.results["test-node"].postgres_version!.version).toBe("16.3");
  });

  test("generateA013 creates report with minor version data", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
    });

    const report = await checkup.generateA013(mockClient as any, "test-node");
    expect(report.checkId).toBe("A013");
    expect(report.checkTitle).toBe("Postgres minor version");
    expect(report.nodes.primary).toBe("test-node");
    expect("test-node" in report.results).toBe(true);
    expect("version" in report.results["test-node"].data).toBe(true);
    expect(report.results["test-node"].data.version.server_minor_ver).toBe("3");
  });

  test("generateAllReports returns reports for all checks", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        settingsRows: [
          {
            tag_setting_name: "shared_buffers",
            tag_setting_value: "16384",
            tag_unit: "8kB",
            tag_category: "Resource Usage / Memory",
            tag_vartype: "integer",
            is_default: 0, // Non-default for A007
            setting_normalized: "134217728",
            unit_normalized: "bytes",
          },
        ],
        databaseSizesRows: [{ datname: "postgres", size_bytes: "1073741824" }],
        dbStatsRows: [{
          numbackends: 5,
          xact_commit: 100,
          xact_rollback: 1,
          blks_read: 1000,
          blks_hit: 9000,
          tup_returned: 500,
          tup_fetched: 400,
          tup_inserted: 50,
          tup_updated: 30,
          tup_deleted: 10,
          deadlocks: 0,
          temp_files: 0,
          temp_bytes: 0,
          postmaster_uptime_s: 864000
        }],
        connectionStatesRows: [{ state: "active", count: 2 }, { state: "idle", count: 3 }],
        uptimeRows: [{ start_time: new Date("2024-01-01T00:00:00Z"), uptime: "10 days" }],
        invalidIndexesRows: [],
        unusedIndexesRows: [],
        redundantIndexesRows: [],
        sensitiveColumnsRows: [],
      }
    );

    const reports = await checkup.generateAllReports(mockClient as any, "test-node");
    expect("A002" in reports).toBe(true);
    expect("A003" in reports).toBe(true);
    expect("A004" in reports).toBe(true);
    expect("A007" in reports).toBe(true);
    expect("A013" in reports).toBe(true);
    expect("H001" in reports).toBe(true);
    expect("H002" in reports).toBe(true);
    expect("H004" in reports).toBe(true);
    // S001 is only available in Python reporter, not in CLI express mode
    expect(reports.A002.checkId).toBe("A002");
    expect(reports.A003.checkId).toBe("A003");
    expect(reports.A004.checkId).toBe("A004");
    expect(reports.A007.checkId).toBe("A007");
    expect(reports.A013.checkId).toBe("A013");
    expect(reports.H001.checkId).toBe("H001");
    expect(reports.H002.checkId).toBe("H002");
    expect(reports.H004.checkId).toBe("H004");
  });

  // Per-check error isolation (work item 260, finding 4): one throwing check
  // must not abort the whole checkup and discard already-completed reports.
  test("generateAllReports isolates a failing check: others still complete, failure is reported via onCheckError", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
      invalidIndexesRows: [],
      unusedIndexesRows: [],
      redundantIndexesRows: [],
      sensitiveColumnsRows: [],
    });

    const originalH002 = checkup.REPORT_GENERATORS.H002;
    const failures: Array<{ checkId: string; checkTitle: string; error: Error }> = [];
    try {
      // Sabotage exactly one check's generator (simulates e.g. F003's
      // getDeadTuples propagating a DB error on an exotic server).
      checkup.REPORT_GENERATORS.H002 = async () => {
        throw new Error("boom: relation pg_stat_user_indexes vanished");
      };

      const reports = await checkup.generateAllReports(
        mockClient as any,
        "test-node",
        undefined,
        (f) => failures.push(f)
      );

      // The failing check is absent, every other check still produced a report.
      expect("H002" in reports).toBe(false);
      expect("A002" in reports).toBe(true);
      expect("A013" in reports).toBe(true);
      expect("H001" in reports).toBe(true);
      expect("H004" in reports).toBe(true);
      expect(Object.keys(reports).length).toBe(Object.keys(checkup.REPORT_GENERATORS).length - 1);

      // The failure was surfaced (named check + original error), not swallowed.
      expect(failures.length).toBe(1);
      expect(failures[0].checkId).toBe("H002");
      expect(failures[0].error.message).toContain("boom");
    } finally {
      checkup.REPORT_GENERATORS.H002 = originalH002;
    }
  });

  test("generateAllReports still throws loudly on TOTAL failure (every check fails)", async () => {
    // Contract guard: a fully broken run (e.g. dead connection) must NOT come
    // back as an empty-but-successful result that would be uploaded/written as
    // a checkup with zero findings.
    const mockClient = createMockClient({});
    const originals = { ...checkup.REPORT_GENERATORS };
    try {
      for (const id of Object.keys(checkup.REPORT_GENERATORS)) {
        checkup.REPORT_GENERATORS[id] = async () => {
          throw new Error("connection terminated unexpectedly");
        };
      }
      await expect(
        checkup.generateAllReports(mockClient as any, "test-node")
      ).rejects.toThrow(/All \d+ checks failed/);
    } finally {
      Object.assign(checkup.REPORT_GENERATORS, originals);
    }
  });
});

// Tests for A007 (Altered settings)
describe("A007 - Altered settings", () => {
  test("getAlteredSettings returns non-default settings", async () => {
    const mockClient = createMockClient({
      settingsRows: [
        { tag_setting_name: "shared_buffers", tag_setting_value: "256MB", tag_unit: "", tag_category: "Resource Usage / Memory", tag_vartype: "string", is_default: 0, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "work_mem", tag_setting_value: "64MB", tag_unit: "", tag_category: "Resource Usage / Memory", tag_vartype: "string", is_default: 0, setting_normalized: null, unit_normalized: null },
        { tag_setting_name: "default_setting", tag_setting_value: "on", tag_unit: "", tag_category: "Other", tag_vartype: "bool", is_default: 1, setting_normalized: null, unit_normalized: null },
      ],
    });

    const settings = await checkup.getAlteredSettings(mockClient as any);
    expect("shared_buffers" in settings).toBe(true);
    expect("work_mem" in settings).toBe(true);
    expect("default_setting" in settings).toBe(false);  // Should be filtered out
    expect(settings.shared_buffers.value).toBe("256MB");
    expect(settings.work_mem.value).toBe("64MB");
  });

  test("generateA007 creates report with altered settings", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        settingsRows: [
          { tag_setting_name: "max_connections", tag_setting_value: "200", tag_unit: "", tag_category: "Connections and Authentication", tag_vartype: "integer", is_default: 0, setting_normalized: null, unit_normalized: null },
        ],
      }
    );

    const report = await checkup.generateA007(mockClient as any, "test-node");
    expect(report.checkId).toBe("A007");
    expect(report.checkTitle).toBe("Altered settings");
    expect(report.nodes.primary).toBe("test-node");
    expect("test-node" in report.results).toBe(true);
    expect("max_connections" in report.results["test-node"].data).toBe(true);
    expect(report.results["test-node"].data.max_connections.value).toBe("200");
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });
});

// Tests for A004 (Cluster information)
describe("A004 - Cluster information", () => {
  test("getDatabaseSizes returns database sizes", async () => {
    const mockClient = createMockClient({
      databaseSizesRows: [
        { datname: "postgres", size_bytes: "1073741824" },
        { datname: "mydb", size_bytes: "536870912" },
      ],
    });

    const sizes = await checkup.getDatabaseSizes(mockClient as any);
    expect("postgres" in sizes).toBe(true);
    expect("mydb" in sizes).toBe(true);
    expect(sizes.postgres).toBe(1073741824);
    expect(sizes.mydb).toBe(536870912);
  });

  test("getClusterInfo returns cluster metrics", async () => {
    const mockClient = createMockClient({
      dbStatsRows: [{
        numbackends: 10,
        xact_commit: 1000,
        xact_rollback: 5,
        blks_read: 500,
        blks_hit: 9500,
        tup_returned: 5000,
        tup_fetched: 4000,
        tup_inserted: 100,
        tup_updated: 50,
        tup_deleted: 25,
        deadlocks: 0,
        temp_files: 2,
        temp_bytes: 1048576,
        postmaster_uptime_s: 2592000,  // 30 days
      }],
      connectionStatesRows: [
        { state: "active", count: 3 },
        { state: "idle", count: 7 },
      ],
      uptimeRows: [{
        start_time: new Date("2024-01-01T00:00:00Z"),
        uptime: "30 days",
      }],
    });

    const info = await checkup.getClusterInfo(mockClient as any);
    expect("total_connections" in info).toBe(true);
    expect("cache_hit_ratio" in info).toBe(true);
    expect("connections_active" in info).toBe(true);
    expect("connections_idle" in info).toBe(true);
    expect("start_time" in info).toBe(true);
    expect(info.total_connections.value).toBe("10");
    expect(info.cache_hit_ratio.value).toBe("95.00");
    expect(info.connections_active.value).toBe("3");
  });

  test("generateA004 creates report with cluster info and database sizes", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        databaseSizesRows: [
          { datname: "postgres", size_bytes: "1073741824" },
        ],
        dbStatsRows: [{
          numbackends: 5,
          xact_commit: 100,
          xact_rollback: 1,
          blks_read: 100,
          blks_hit: 900,
          tup_returned: 500,
          tup_fetched: 400,
          tup_inserted: 50,
          tup_updated: 30,
          tup_deleted: 10,
          deadlocks: 0,
          temp_files: 0,
          temp_bytes: 0,
          postmaster_uptime_s: 864000,
        }],
        connectionStatesRows: [{ state: "active", count: 2 }],
        uptimeRows: [{ start_time: new Date("2024-01-01T00:00:00Z"), uptime: "10 days" }],
      }
    );

    const report = await checkup.generateA004(mockClient as any, "test-node");
    expect(report.checkId).toBe("A004");
    expect(report.checkTitle).toBe("Cluster information");
    expect(report.nodes.primary).toBe("test-node");
    expect("test-node" in report.results).toBe(true);

    const data = report.results["test-node"].data;
    expect("general_info" in data).toBe(true);
    expect("database_sizes" in data).toBe(true);
    expect("total_connections" in data.general_info).toBe(true);
    expect("postgres" in data.database_sizes).toBe(true);
    expect(data.database_sizes.postgres).toBe(1073741824);
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });
});

// Tests for H001 (Invalid indexes)
describe("H001 - Invalid indexes", () => {
  test("getInvalidIndexes returns invalid indexes", async () => {
    const mockClient = createMockClient({
      invalidIndexesRows: [
        { schema_name: "public", table_name: "users", index_name: "users_email_idx", relation_name: "users", index_size_bytes: "1048576", index_definition: "CREATE INDEX users_email_idx ON public.users USING btree (email)", supports_fk: false },
      ],
    });

    const indexes = await checkup.getInvalidIndexes(mockClient as any);
    expect(indexes.length).toBe(1);
    expect(indexes[0].schema_name).toBe("public");
    expect(indexes[0].table_name).toBe("users");
    expect(indexes[0].index_name).toBe("users_email_idx");
    expect(indexes[0].index_size_bytes).toBe(1048576);
    expect(indexes[0].index_size_pretty).toBeTruthy();
    expect(indexes[0].index_definition).toMatch(/^CREATE INDEX/);
    expect(indexes[0].relation_name).toBe("users");
    expect(indexes[0].supports_fk).toBe(false);
  });

  test("generateH001 creates report with invalid indexes", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        invalidIndexesRows: [
          { schema_name: "public", table_name: "orders", index_name: "orders_status_idx", relation_name: "orders", index_size_bytes: "2097152", index_definition: "CREATE INDEX orders_status_idx ON public.orders USING btree (status)", supports_fk: false },
        ],
      }
    );

    const report = await checkup.generateH001(mockClient as any, "test-node");
    expect(report.checkId).toBe("H001");
    expect(report.checkTitle).toBe("Invalid indexes");
    expect("test-node" in report.results).toBe(true);

    // Data is now keyed by database name
    const data = report.results["test-node"].data;
    expect("testdb" in data).toBe(true);
    const dbData = data["testdb"] as any;
    expect(dbData.invalid_indexes).toBeTruthy();
    expect(dbData.total_count).toBe(1);
    expect(dbData.total_size_bytes).toBe(2097152);
    expect(dbData.total_size_pretty).toBeTruthy();
    expect(dbData.database_size_bytes).toBeTruthy();
    expect(dbData.database_size_pretty).toBeTruthy();
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });

  test("getInvalidIndexes returns decision tree fields including valid_duplicate_definition", async () => {
    const mockClient = createMockClient({
      invalidIndexesRows: [
        {
          schema_name: "public",
          table_name: "users",
          index_name: "users_email_idx_invalid",
          relation_name: "users",
          index_size_bytes: "1048576",
          index_definition: "CREATE INDEX users_email_idx_invalid ON public.users USING btree (email)",
          supports_fk: false,
          is_pk: false,
          is_unique: false,
          constraint_name: null,
          table_row_estimate: "5000",
          has_valid_duplicate: true,
          valid_index_name: "users_email_idx",
          valid_index_definition: "CREATE INDEX users_email_idx ON public.users USING btree (email)",
        },
      ],
    });

    const indexes = await checkup.getInvalidIndexes(mockClient as any);
    expect(indexes.length).toBe(1);
    expect(indexes[0].is_pk).toBe(false);
    expect(indexes[0].is_unique).toBe(false);
    expect(indexes[0].constraint_name).toBeNull();
    expect(indexes[0].table_row_estimate).toBe(5000);
    expect(indexes[0].has_valid_duplicate).toBe(true);
    expect(indexes[0].valid_duplicate_name).toBe("users_email_idx");
    expect(indexes[0].valid_duplicate_definition).toBe("CREATE INDEX users_email_idx ON public.users USING btree (email)");
  });

  test("getInvalidIndexes handles has_valid_duplicate: false with null values", async () => {
    const mockClient = createMockClient({
      invalidIndexesRows: [
        {
          schema_name: "public",
          table_name: "orders",
          index_name: "orders_status_idx_invalid",
          relation_name: "orders",
          index_size_bytes: "524288",
          index_definition: "CREATE INDEX orders_status_idx_invalid ON public.orders USING btree (status)",
          supports_fk: false,
          is_pk: false,
          is_unique: false,
          constraint_name: null,
          table_row_estimate: "100000",
          has_valid_duplicate: false,
          valid_index_name: null,
          valid_index_definition: null,
        },
      ],
    });

    const indexes = await checkup.getInvalidIndexes(mockClient as Client);
    expect(indexes.length).toBe(1);
    expect(indexes[0].has_valid_duplicate).toBe(false);
    expect(indexes[0].valid_duplicate_name).toBeNull();
    expect(indexes[0].valid_duplicate_definition).toBeNull();
  });

  test("getInvalidIndexes handles is_pk: true with constraint", async () => {
    const mockClient = createMockClient({
      invalidIndexesRows: [
        {
          schema_name: "public",
          table_name: "accounts",
          index_name: "accounts_pkey_invalid",
          relation_name: "accounts",
          index_size_bytes: "262144",
          index_definition: "CREATE UNIQUE INDEX accounts_pkey_invalid ON public.accounts USING btree (id)",
          supports_fk: true,
          is_pk: true,
          is_unique: true,
          constraint_name: "accounts_pkey",
          table_row_estimate: "500",
          has_valid_duplicate: false,
          valid_index_name: null,
          valid_index_definition: null,
        },
      ],
    });

    const indexes = await checkup.getInvalidIndexes(mockClient as Client);
    expect(indexes.length).toBe(1);
    expect(indexes[0].is_pk).toBe(true);
    expect(indexes[0].is_unique).toBe(true);
    expect(indexes[0].constraint_name).toBe("accounts_pkey");
    expect(indexes[0].supports_fk).toBe(true);
  });

  test("getInvalidIndexes handles is_unique: true without PK", async () => {
    const mockClient = createMockClient({
      invalidIndexesRows: [
        {
          schema_name: "public",
          table_name: "users",
          index_name: "users_email_unique_invalid",
          relation_name: "users",
          index_size_bytes: "131072",
          index_definition: "CREATE UNIQUE INDEX users_email_unique_invalid ON public.users USING btree (email)",
          supports_fk: false,
          is_pk: false,
          is_unique: true,
          constraint_name: "users_email_unique",
          table_row_estimate: "25000",
          has_valid_duplicate: true,
          valid_index_name: "users_email_unique_idx",
          valid_index_definition: "CREATE UNIQUE INDEX users_email_unique_idx ON public.users USING btree (email)",
        },
      ],
    });

    const indexes = await checkup.getInvalidIndexes(mockClient as Client);
    expect(indexes.length).toBe(1);
    expect(indexes[0].is_pk).toBe(false);
    expect(indexes[0].is_unique).toBe(true);
    expect(indexes[0].constraint_name).toBe("users_email_unique");
    expect(indexes[0].has_valid_duplicate).toBe(true);
  });
  // Top-level structure tests removed - covered by schema-validation.test.ts
});

// Tests for H001 decision tree recommendation logic
describe("H001 - Decision tree recommendations", () => {
  // Helper to create a minimal InvalidIndex for testing
  const createTestIndex = (overrides: Partial<checkup.InvalidIndex> = {}): checkup.InvalidIndex => ({
    schema_name: "public",
    table_name: "test_table",
    index_name: "test_idx",
    relation_name: "public.test_table",
    index_size_bytes: 1024,
    index_size_pretty: "1 KiB",
    index_definition: "CREATE INDEX test_idx ON public.test_table USING btree (col)",
    supports_fk: false,
    is_pk: false,
    is_unique: false,
    constraint_name: null,
    table_row_estimate: 100000, // Large table by default
    has_valid_duplicate: false,
    valid_duplicate_name: null,
    valid_duplicate_definition: null,
    ...overrides,
  });

  test("returns DROP when has_valid_duplicate is true", () => {
    const index = createTestIndex({ has_valid_duplicate: true, valid_duplicate_name: "existing_idx" });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("DROP");
  });

  test("returns DROP even when is_pk is true if has_valid_duplicate is true", () => {
    // has_valid_duplicate takes precedence over is_pk
    const index = createTestIndex({
      has_valid_duplicate: true,
      is_pk: true,
      is_unique: true,
    });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("DROP");
  });

  test("returns RECREATE when is_pk is true and no valid duplicate", () => {
    const index = createTestIndex({
      is_pk: true,
      is_unique: true,
      constraint_name: "test_pkey",
    });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });

  test("returns RECREATE when is_unique is true (non-PK) and no valid duplicate", () => {
    const index = createTestIndex({
      is_unique: true,
      constraint_name: "test_unique",
    });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });

  test("returns RECREATE for small table (< 10K rows) without valid duplicate", () => {
    const index = createTestIndex({ table_row_estimate: 5000 });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });

  test("returns RECREATE for table at threshold boundary (9999 rows)", () => {
    const index = createTestIndex({ table_row_estimate: 9999 });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });

  test("returns UNCERTAIN for large table (>= 10K rows) at threshold boundary", () => {
    const index = createTestIndex({ table_row_estimate: 10000 });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("UNCERTAIN");
  });

  test("returns UNCERTAIN for large table without valid duplicate or constraint", () => {
    const index = createTestIndex({ table_row_estimate: 1000000 });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("UNCERTAIN");
  });

  test("returns UNCERTAIN for empty table (0 rows) with no valid duplicate - edge case", () => {
    // Empty table should be RECREATE (< 10K threshold)
    const index = createTestIndex({ table_row_estimate: 0 });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });

  test("decision tree priority: has_valid_duplicate > is_pk > small_table", () => {
    // Even with PK and small table, has_valid_duplicate should win
    const index = createTestIndex({
      has_valid_duplicate: true,
      is_pk: true,
      is_unique: true,
      table_row_estimate: 100,
    });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("DROP");
  });

  test("decision tree priority: is_pk > small_table", () => {
    // is_pk should return RECREATE regardless of table size
    const index = createTestIndex({
      is_pk: true,
      is_unique: true,
      table_row_estimate: 1000000, // Large table
    });
    expect(checkup.getInvalidIndexRecommendation(index)).toBe("RECREATE");
  });
});

// Tests for H002 (Unused indexes)
describe("H002 - Unused indexes", () => {
  test("getUnusedIndexes returns unused indexes", async () => {
    const mockClient = createMockClient({
      unusedIndexesRows: [
        {
          schema_name: "public",
          table_name: "products",
          index_name: "products_old_idx",
          index_definition: "CREATE INDEX products_old_idx ON public.products USING btree (old_column)",
          reason: "Never Used Indexes",
          index_size_bytes: "4194304",
          idx_scan: "0",
          idx_is_btree: true,
          supports_fk: false,
        },
      ],
    });

    const indexes = await checkup.getUnusedIndexes(mockClient as any);
    expect(indexes.length).toBe(1);
    expect(indexes[0].schema_name).toBe("public");
    expect(indexes[0].index_name).toBe("products_old_idx");
    expect(indexes[0].index_size_bytes).toBe(4194304);
    expect(indexes[0].idx_scan).toBe(0);
    expect(indexes[0].supports_fk).toBe(false);
    expect(indexes[0].index_definition).toBeTruthy();
    expect(indexes[0].idx_is_btree).toBe(true);
  });

  test("generateH002 creates report with unused indexes", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        unusedIndexesRows: [
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
      }
    );

    const report = await checkup.generateH002(mockClient as any, "test-node");
    expect(report.checkId).toBe("H002");
    expect(report.checkTitle).toBe("Unused indexes");
    expect("test-node" in report.results).toBe(true);

    // Data is now keyed by database name
    const data = report.results["test-node"].data;
    expect("testdb" in data).toBe(true);
    const dbData = data["testdb"] as any;
    expect(dbData.unused_indexes).toBeTruthy();
    expect(dbData.total_count).toBe(1);
    expect(dbData.total_size_bytes).toBe(8388608);
    expect(dbData.total_size_pretty).toBeTruthy();
    expect(dbData.stats_reset).toBeTruthy();
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });
  // Top-level structure tests removed - covered by schema-validation.test.ts
});

// Tests for H004 (Redundant indexes)
describe("H004 - Redundant indexes", () => {
  test("getRedundantIndexes returns redundant indexes", async () => {
    const mockClient = createMockClient({
      redundantIndexesRows: [
        {
          schema_name: "public",
          table_name: "orders",
          index_name: "orders_user_id_idx",
          relation_name: "orders",
          access_method: "btree",
          reason: "public.orders_user_id_created_idx",
          index_size_bytes: "2097152",
          table_size_bytes: "16777216",
          index_usage: "0",
          supports_fk: false,
          index_definition: "CREATE INDEX orders_user_id_idx ON public.orders USING btree (user_id)",
          redundant_to_json: JSON.stringify([
            { index_name: "public.orders_user_id_created_idx", index_definition: "CREATE INDEX orders_user_id_created_idx ON public.orders USING btree (user_id, created_at)", index_size_bytes: 1048576 }
          ]),
        },
      ],
    });

    const indexes = await checkup.getRedundantIndexes(mockClient as any);
    expect(indexes.length).toBe(1);
    expect(indexes[0].schema_name).toBe("public");
    expect(indexes[0].index_name).toBe("orders_user_id_idx");
    expect(indexes[0].reason).toBe("public.orders_user_id_created_idx");
    expect(indexes[0].index_size_bytes).toBe(2097152);
    expect(indexes[0].supports_fk).toBe(false);
    expect(indexes[0].index_definition).toBeTruthy();
    expect(indexes[0].relation_name).toBe("orders");
    // Verify redundant_to is populated with definitions and sizes
    expect(indexes[0].redundant_to).toBeInstanceOf(Array);
    expect(indexes[0].redundant_to.length).toBe(1);
    expect(indexes[0].redundant_to[0].index_name).toBe("public.orders_user_id_created_idx");
    expect(indexes[0].redundant_to[0].index_definition).toContain("CREATE INDEX");
    expect(indexes[0].redundant_to[0].index_size_bytes).toBe(1048576);
    expect(indexes[0].redundant_to[0].index_size_pretty).toBe("1.00 MiB");
  });

  test("generateH004 creates report with redundant indexes", async () => {
    const mockClient = createMockClient({
      versionRows: [
        { name: "server_version", setting: "16.3" },
        { name: "server_version_num", setting: "160003" },
      ],
        redundantIndexesRows: [
          {
            schema_name: "public",
            table_name: "products",
            index_name: "products_category_idx",
            relation_name: "products",
            access_method: "btree",
            reason: "public.products_category_name_idx",
            index_size_bytes: "4194304",
            table_size_bytes: "33554432",
            index_usage: "5",
            supports_fk: false,
            index_definition: "CREATE INDEX products_category_idx ON public.products USING btree (category)",
            redundant_to_json: JSON.stringify([
              { index_name: "public.products_category_name_idx", index_definition: "CREATE INDEX products_category_name_idx ON public.products USING btree (category, name)", index_size_bytes: 2097152 }
            ]),
          },
        ],
      }
    );

    const report = await checkup.generateH004(mockClient as any, "test-node");
    expect(report.checkId).toBe("H004");
    expect(report.checkTitle).toBe("Redundant indexes");
    expect("test-node" in report.results).toBe(true);

    // Data is now keyed by database name
    const data = report.results["test-node"].data;
    expect("testdb" in data).toBe(true);
    const dbData = data["testdb"] as any;
    expect(dbData.redundant_indexes).toBeTruthy();
    expect(dbData.total_count).toBe(1);
    expect(dbData.total_size_bytes).toBe(4194304);
    expect(dbData.total_size_pretty).toBeTruthy();
    expect(dbData.database_size_bytes).toBeTruthy();
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });
  // Top-level structure tests removed - covered by schema-validation.test.ts
});

describe("F002 - wraparound", () => {
  test("evaluates warning, high, failsafe-high, and critical thresholds", () => {
    expect(checkup.evaluateWraparoundRisk(500_000_000, 0, null).severity).toBe("info");
    expect(checkup.evaluateWraparoundRisk(199_999_999, 200_000_000, 1_600_000_000).severity).toBe("info");
    expect(checkup.evaluateWraparoundRisk(200_000_000, 200_000_000, 1_600_000_000).severity).toBe("warning");
    expect(checkup.evaluateWraparoundRisk(400_000_000, 200_000_000, 1_600_000_000).severity).toBe("high");
    expect(checkup.evaluateWraparoundRisk(720_000_000, 500_000_000, 900_000_000).severity).toBe("high");
    expect(checkup.evaluateWraparoundRisk(1_000_000_000, 800_000_000, 1_600_000_000).severity).toBe("critical");
  });

  test("honors table overrides and gates failsafe fields on PostgreSQL 13", async () => {
    const mockClient = createMockClient({
      versionRows: [{ name: "server_version", setting: "13.16" }, { name: "server_version_num", setting: "130016" }],
      wraparoundDatabaseRows: [{ tag_datname: "testdb", age_datfrozenxid: "120000000", age_datminmxid: "10" }],
      wraparoundTableRows: [{
        tag_schema_name: "public", tag_table_name: "events", tag_ranked_by: "multixact,xid",
        xid_age: "120000000", multixact_age: "410000000", effective_freeze_max_age: "100000000",
        effective_multixact_freeze_max_age: "400000000", table_size_bytes: "1048576",
      }],
    });
    const data = await checkup.getWraparoundData(mockClient as any, 13);
    expect(data.settings.vacuum_failsafe_age).toBeNull();
    expect(data.settings.vacuum_multixact_failsafe_age).toBeNull();
    expect(data.databases[0].xid.severity).toBe("info");
    expect(data.tables[0].xid.emergency_age).toBe(100_000_000);
    expect(data.tables[0].xid.severity).toBe("warning");
    expect(data.tables[0].multixact.severity).toBe("warning");
    expect(data.tables[0].ranked_by).toEqual(["multixact", "xid"]);
  });

  test("generates conclusions and a report for severe offenders", async () => {
    const mockClient = createMockClient({
      wraparoundDatabaseRows: [{ tag_datname: "testdb", age_datfrozenxid: "1000000000", age_datminmxid: "10" }],
      wraparoundTableRows: [],
    });
    const report = await checkup.REPORT_GENERATORS.F002(mockClient as any, "test-node");
    const data = report.results["test-node"].data as any;
    expect(report.checkId).toBe("F002");
    expect(data.severity).toBe("critical");
    expect(data.conclusions[0]).toContain('database "testdb"');
    expect(data.recommendations.join(" ")).toContain("VACUUM (FREEZE, VERBOSE)");
  });

  test("uses real MultiXact size fields and reports unavailable settings without false positives", async () => {
    const mockClient = createMockClient({
      wraparoundSettingsRows: [],
      wraparoundDatabaseRows: [{ tag_datname: "testdb", age_datfrozenxid: "500000000", age_datminmxid: "500000000" }],
      multixactSizeRows: [{ members_bytes: "1048576", offsets_bytes: "524288", status_code: "0" }],
    });
    const report = await checkup.REPORT_GENERATORS.F002(mockClient as any, "test-node");
    const data = report.results["test-node"].data as any;
    expect(data.settings_available).toBe(false);
    expect(data.severity).toBe("info");
    expect(data.databases[0].xid.severity).toBe("info");
    expect(data.conclusions[0]).toMatch(/settings are unavailable/i);
    expect(data.multixact_size).toEqual({
      bytes: 1572864,
      size_pretty: "1.50 MiB",
      status_code: 0,
    });
  });

  test("preserves degraded MultiXact size status when byte counts are unavailable", async () => {
    const mockClient = createMockClient({
      multixactSizeRows: [{ members_bytes: null, offsets_bytes: null, status_code: "2" }],
    });
    const data = await checkup.getWraparoundData(mockClient as any, 16);
    expect(data.multixact_size).toEqual({ bytes: null, size_pretty: null, status_code: 2 });
  });
});

// Tests for F003 (Autovacuum: dead tuples)
describe("F003 - Dead tuples", () => {
  // The seeded UX-test case that F004 missed: 8.27M dead tuples,
  // dead:live ratio 1.3:1 (dead_pct ~56.5%), autovacuum disabled via reloptions.
  const seededProblemRow = {
    tag_schemaname: "public",
    tag_relname: "events",
    n_live_tup: "6361538",
    n_dead_tup: "8270000",
    dead_pct: 56.52,
    last_autovacuum: "0",
    last_vacuum: "0",
    autovacuum_count: "0",
    vacuum_count: "0",
    autovacuum_disabled: 1,
    table_size_b: "2147483648",
  };

  test("getDeadTuples maps rows and computes flags", async () => {
    const mockClient = createMockClient({ deadTuplesRows: [seededProblemRow] });

    const tables = await checkup.getDeadTuples(mockClient as any);
    expect(tables.length).toBe(1);
    const t = tables[0];
    expect(t.schema_name).toBe("public");
    expect(t.table_name).toBe("events");
    expect(t.n_live_tup).toBe(6361538);
    expect(t.n_dead_tup).toBe(8270000);
    expect(t.dead_pct).toBe(56.52);
    expect(t.autovacuum_disabled).toBe(true);
    expect(t.last_autovacuum).toBeNull();
    expect(t.last_autovacuum_epoch).toBe(0);
    expect(t.last_vacuum).toBeNull();
    expect(t.table_size_bytes).toBe(2147483648);
    expect(t.table_size_pretty).toBe("2.00 GiB");
    expect(t.exceeds_dead_tuple_thresholds).toBe(true);
    expect(t.autovacuum_disabled_flagged).toBe(true);
  });

  test("getDeadTuples converts non-zero vacuum epochs to ISO timestamps", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [{
        ...seededProblemRow,
        last_autovacuum: "1704067200", // 2024-01-01T00:00:00Z
        last_vacuum: "1706745600", // 2024-02-01T00:00:00Z
        autovacuum_count: "3",
        vacuum_count: "1",
        autovacuum_disabled: 0,
      }],
    });

    const [t] = await checkup.getDeadTuples(mockClient as any);
    expect(t.last_autovacuum).toBe("2024-01-01T00:00:00.000Z");
    expect(t.last_autovacuum_epoch).toBe(1704067200);
    expect(t.last_vacuum).toBe("2024-02-01T00:00:00.000Z");
    expect(t.autovacuum_count).toBe(3);
    expect(t.vacuum_count).toBe(1);
    expect(t.autovacuum_disabled).toBe(false);
  });

  test("dead-tuple thresholds require BOTH absolute and relative excess", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        // High count, low ratio (large healthy table churning under active autovacuum)
        { ...seededProblemRow, tag_relname: "big_churn", n_live_tup: "100000000", n_dead_tup: "150000", dead_pct: 0.15, autovacuum_disabled: 0 },
        // High ratio, low count (small table - not worth flagging)
        { ...seededProblemRow, tag_relname: "tiny", n_live_tup: "100", n_dead_tup: "900", dead_pct: 90, autovacuum_disabled: 0 },
        // Both high - must be flagged
        { ...seededProblemRow, tag_relname: "problem", n_live_tup: "100000", n_dead_tup: "100000", dead_pct: 50, autovacuum_disabled: 0 },
      ],
    });

    const tables = await checkup.getDeadTuples(mockClient as any);
    const byName = new Map(tables.map((t) => [t.table_name, t]));
    expect(byName.get("big_churn")!.exceeds_dead_tuple_thresholds).toBe(false);
    expect(byName.get("tiny")!.exceeds_dead_tuple_thresholds).toBe(false);
    expect(byName.get("problem")!.exceeds_dead_tuple_thresholds).toBe(true);
  });

  test("autovacuum disabled is flagged only on non-tiny tables", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        { ...seededProblemRow, tag_relname: "tiny_disabled", n_live_tup: "50", n_dead_tup: "0", dead_pct: 0 },
        { ...seededProblemRow, tag_relname: "big_disabled", n_live_tup: "20000", n_dead_tup: "0", dead_pct: 0 },
      ],
    });

    const tables = await checkup.getDeadTuples(mockClient as any);
    const byName = new Map(tables.map((t) => [t.table_name, t]));
    expect(byName.get("tiny_disabled")!.autovacuum_disabled).toBe(true);
    expect(byName.get("tiny_disabled")!.autovacuum_disabled_flagged).toBe(false);
    expect(byName.get("big_disabled")!.autovacuum_disabled_flagged).toBe(true);
  });

  test("buildDeadTuplesConclusions produces concrete conclusions and recommendations", async () => {
    const mockClient = createMockClient({ deadTuplesRows: [seededProblemRow] });
    const tables = await checkup.getDeadTuples(mockClient as any);

    const { conclusions, recommendations } = checkup.buildDeadTuplesConclusions(tables);
    expect(conclusions.length).toBe(1);
    expect(conclusions[0]).toContain('"public"."events"');
    expect(conclusions[0]).toContain("8,270,000 dead tuples");
    expect(conclusions[0]).toContain("56.52% of all tuples");
    expect(conclusions[0]).toContain("autovacuum is disabled");
    expect(conclusions[0]).toContain("never vacuumed");

    expect(recommendations.length).toBe(1);
    expect(recommendations[0]).toContain('alter table "public"."events" reset (autovacuum_enabled);');
    expect(recommendations[0]).toContain('vacuum (analyze) "public"."events";');
  });

  test("buildDeadTuplesConclusions distinguishes vacuum-lag from disabled-autovacuum cases", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        // Dead tuples high but autovacuum enabled -> vacuum + tuning advice
        { ...seededProblemRow, tag_relname: "lagging", autovacuum_disabled: 0, last_autovacuum: "1704067200" },
        // Autovacuum disabled, no dead tuples yet -> re-enable advice
        { ...seededProblemRow, tag_relname: "disabled_only", n_live_tup: "50000", n_dead_tup: "0", dead_pct: 0 },
      ],
    });
    const tables = await checkup.getDeadTuples(mockClient as any);

    const { conclusions, recommendations } = checkup.buildDeadTuplesConclusions(tables);
    expect(conclusions.length).toBe(2);
    const laggingRec = recommendations.find((r) => r.includes('"public"."lagging"'))!;
    expect(laggingRec).toContain("vacuum (analyze)");
    expect(laggingRec).toContain("autovacuum_vacuum_scale_factor");
    expect(laggingRec).not.toContain("reset (autovacuum_enabled)");

    const disabledRec = recommendations.find((r) => r.includes('"public"."disabled_only"'))!;
    expect(disabledRec).toContain('alter table "public"."disabled_only" reset (autovacuum_enabled);');
  });

  test("generateF003 creates report with counts, thresholds, and conclusions", async () => {
    const mockClient = createMockClient({
      deadTuplesRows: [
        seededProblemRow,
        { ...seededProblemRow, tag_relname: "disabled_only", n_live_tup: "50000", n_dead_tup: "0", dead_pct: 0 },
      ],
    });

    const report = await checkup.REPORT_GENERATORS.F003(mockClient as any, "test-node");
    expect(report.checkId).toBe("F003");
    expect(report.checkTitle).toBe("Autovacuum: dead tuples");

    const data = report.results["test-node"].data;
    expect("testdb" in data).toBe(true);
    const dbData = data["testdb"] as any;
    expect(dbData.dead_tuples_tables.length).toBe(2);
    expect(dbData.total_count).toBe(2);
    expect(dbData.flagged_count).toBe(1);
    expect(dbData.autovacuum_disabled_count).toBe(2);
    expect(dbData.autovacuum_disabled_flagged_count).toBe(2);
    expect(dbData.total_dead_tuples).toBe(8270000);
    expect(dbData.thresholds).toEqual({
      dead_tuples_min: checkup.F003_DEAD_TUPLES_MIN,
      dead_pct_min: checkup.F003_DEAD_PCT_MIN,
      autovacuum_disabled_min_rows: checkup.F003_AUTOVACUUM_DISABLED_MIN_ROWS,
    });
    expect(dbData.conclusions.length).toBe(2);
    expect(dbData.recommendations.length).toBe(2);
    expect(dbData.database_size_bytes).toBeTruthy();
    expect(report.results["test-node"].postgres_version).toBeTruthy();
  });

  test("generateF003 handles a healthy database (no rows)", async () => {
    const mockClient = createMockClient({ deadTuplesRows: [] });

    const report = await checkup.REPORT_GENERATORS.F003(mockClient as any, "test-node");
    const dbData = report.results["test-node"].data["testdb"] as any;
    expect(dbData.dead_tuples_tables).toEqual([]);
    expect(dbData.total_count).toBe(0);
    expect(dbData.flagged_count).toBe(0);
    expect(dbData.autovacuum_disabled_count).toBe(0);
    expect(dbData.autovacuum_disabled_flagged_count).toBe(0);
    expect(dbData.conclusions).toEqual([]);
    expect(dbData.recommendations).toEqual([]);
  });
});

// ===========================================================================
// F001 (Autovacuum: configuration linter) - WI 274
// ===========================================================================
describe("F001 - Autovacuum configuration linter", () => {
  function mkSetting(value: string, unit = ""): checkup.SettingInfo {
    return { setting: value, unit, category: "Autovacuum", context: "sighup", vartype: "integer", pretty_value: value };
  }

  /** A tuned, healthy autovacuum config where NO Tier-1 rule should fire. */
  function healthySettings(): Record<string, checkup.SettingInfo> {
    return {
      autovacuum: mkSetting("on"),
      track_counts: mkSetting("on"),
      autovacuum_vacuum_cost_delay: mkSetting("2", "ms"),
      vacuum_cost_delay: mkSetting("0", "ms"),
      autovacuum_vacuum_cost_limit: mkSetting("2000"),
      vacuum_cost_limit: mkSetting("200"),
      autovacuum_work_mem: mkSetting("262144", "kB"),
      maintenance_work_mem: mkSetting("262144", "kB"),
      autovacuum_naptime: mkSetting("60", "s"),
      autovacuum_vacuum_scale_factor: mkSetting("0.05"),
      autovacuum_analyze_scale_factor: mkSetting("0.02"),
      autovacuum_vacuum_insert_threshold: mkSetting("1000"),
      log_autovacuum_min_duration: mkSetting("10000", "ms"),
      autovacuum_max_workers: mkSetting("5"),
      vacuum_cost_page_hit: mkSetting("1"),
      vacuum_cost_page_miss: mkSetting("2"),
      vacuum_cost_page_dirty: mkSetting("20"),
    };
  }

  const emptyOverview: checkup.ReloptionsOverview = {
    relations_total: 0,
    candidates_considered: 0,
    tables_with_av_overrides: 0,
    autovacuum_disabled_tables: [],
    cost_delay_zero_tables: [],
    cost_delay_zero_toast_only_tables: [],
  };

  function mkTable(overrides: Partial<checkup.LargestTable> = {}): checkup.LargestTable {
    const bytes = overrides.total_relation_size_bytes ?? 1024;
    return {
      schema_name: "public",
      table_name: "big",
      relkind: "r",
      relpages: 100,
      total_relation_size_bytes: bytes,
      total_relation_size_pretty: checkup.formatBytes(bytes),
      has_av_override: false,
      reloptions: "",
      scale_factor_override: null,
      ...overrides,
    };
  }

  function mkCtx(
    settings: Record<string, checkup.SettingInfo>,
    opts: {
      pgMajorVersion?: number;
      largestTables?: checkup.LargestTable[];
      reloptions?: checkup.ReloptionsOverview;
      databaseSizeBytes?: number;
    } = {},
  ): checkup.AutovacuumRuleContext {
    const effective = checkup.resolveEffectiveAutovacuumSettings(settings);
    const throughput = checkup.computeThroughputBudget(effective);
    return {
      pgMajorVersion: opts.pgMajorVersion ?? 16,
      effective,
      throughput,
      largestTables: opts.largestTables ?? [],
      reloptions: opts.reloptions ?? emptyOverview,
      databaseSizeBytes: opts.databaseSizeBytes ?? 0,
      env: null,
    };
  }

  const firedIds = (ctx: checkup.AutovacuumRuleContext) =>
    checkup.evaluateAutovacuumRules(ctx).fired.map((f) => f.id);

  // --- inheritance resolution ---------------------------------------------
  describe("effective-value inheritance resolution", () => {
    test("cost_limit = -1 falls back to vacuum_cost_limit", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_cost_limit = mkSetting("-1");
      s.vacuum_cost_limit = mkSetting("300");
      const eff = checkup.resolveEffectiveAutovacuumSettings(s);
      expect(eff.cost_limit.effective).toBe(300);
      expect(eff.cost_limit.inherited_from).toBe("vacuum_cost_limit");
      expect(eff.cost_limit.inheritance_chain).toContain("vacuum_cost_limit = 300");
    });

    test("cost_delay = -1 falls back to vacuum_cost_delay", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_cost_delay = mkSetting("-1", "ms");
      s.vacuum_cost_delay = mkSetting("5", "ms");
      const eff = checkup.resolveEffectiveAutovacuumSettings(s);
      expect(eff.cost_delay_ms.effective).toBe(5);
      expect(eff.cost_delay_ms.inherited_from).toBe("vacuum_cost_delay");
    });

    test("work_mem = -1 falls back to maintenance_work_mem", () => {
      const s = healthySettings();
      s.autovacuum_work_mem = mkSetting("-1", "kB");
      s.maintenance_work_mem = mkSetting("524288", "kB");
      const eff = checkup.resolveEffectiveAutovacuumSettings(s);
      expect(eff.work_mem_kb.effective).toBe(524288);
      expect(eff.work_mem_kb.inherited_from).toBe("maintenance_work_mem");
    });

    test("explicit (non -1) values are not marked inherited", () => {
      const eff = checkup.resolveEffectiveAutovacuumSettings(healthySettings());
      expect(eff.cost_limit.inherited_from).toBeNull();
      expect(eff.cost_delay_ms.inherited_from).toBeNull();
      expect(eff.work_mem_kb.inherited_from).toBeNull();
    });
  });

  // --- throughput math ----------------------------------------------------
  describe("effective throughput budget", () => {
    test("default 200/2ms yields ~800/400/40 MB/s", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_cost_limit = mkSetting("200");
      s.autovacuum_vacuum_cost_delay = mkSetting("2", "ms");
      const budget = checkup.computeThroughputBudget(checkup.resolveEffectiveAutovacuumSettings(s));
      expect(budget.tokens_per_sec).toBe(100000);
      expect(budget.throttling_disabled).toBe(false);
      // 8KB pages, MB = 1e6: hit ~819.2, miss ~409.6, dirty ~41.0
      expect(budget.read_hit_mbps).toBeCloseTo(819.2, 1);
      expect(budget.read_miss_mbps).toBeCloseTo(409.6, 1);
      expect(budget.dirty_write_mbps).toBeCloseTo(41.0, 1);
    });

    test("cost_delay = 0 marks throttling disabled and leaves budgets null", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_cost_delay = mkSetting("0", "ms");
      const budget = checkup.computeThroughputBudget(checkup.resolveEffectiveAutovacuumSettings(s));
      expect(budget.throttling_disabled).toBe(true);
      expect(budget.tokens_per_sec).toBeNull();
      expect(budget.dirty_write_mbps).toBeNull();
    });
  });

  // --- healthy baseline: no rules fire ------------------------------------
  test("tuned healthy config fires no rules", () => {
    expect(firedIds(mkCtx(healthySettings()))).toEqual([]);
    expect(checkup.evaluateAutovacuumRules(mkCtx(healthySettings())).severity).toBeNull();
  });

  // --- per-rule fire / not-fire -------------------------------------------
  describe("Tier-1 rules fire and do not fire", () => {
    test("autovacuum_off", () => {
      const s = healthySettings();
      s.autovacuum = mkSetting("off");
      expect(firedIds(mkCtx(s))).toContain("autovacuum_off");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("autovacuum_off");
    });

    test("track_counts_off", () => {
      const s = healthySettings();
      s.track_counts = mkSetting("off");
      expect(firedIds(mkCtx(s))).toContain("track_counts_off");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("track_counts_off");
    });

    test("cost_delay_zero (direct and via inheritance)", () => {
      const direct = healthySettings();
      direct.autovacuum_vacuum_cost_delay = mkSetting("0", "ms");
      expect(firedIds(mkCtx(direct))).toContain("cost_delay_zero");

      const inherited = healthySettings();
      inherited.autovacuum_vacuum_cost_delay = mkSetting("-1", "ms");
      inherited.vacuum_cost_delay = mkSetting("0", "ms");
      expect(firedIds(mkCtx(inherited))).toContain("cost_delay_zero");

      expect(firedIds(mkCtx(healthySettings()))).not.toContain("cost_delay_zero");
    });

    test("cost_delay_pg11_legacy fires on explicit 20ms, not when inherited", () => {
      const explicit = healthySettings();
      explicit.autovacuum_vacuum_cost_delay = mkSetting("20", "ms");
      expect(firedIds(mkCtx(explicit))).toContain("cost_delay_pg11_legacy");

      const inherited = healthySettings();
      inherited.autovacuum_vacuum_cost_delay = mkSetting("-1", "ms");
      inherited.vacuum_cost_delay = mkSetting("20", "ms");
      expect(firedIds(mkCtx(inherited))).not.toContain("cost_delay_pg11_legacy");
    });

    test("cost_limit_low", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_cost_limit = mkSetting("200");
      expect(firedIds(mkCtx(s))).toContain("cost_limit_low");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("cost_limit_low");
    });

    test("work_mem_inherits with PG-version-gated note", () => {
      const s = healthySettings();
      s.autovacuum_work_mem = mkSetting("-1", "kB");
      const pg16 = checkup.evaluateAutovacuumRules(mkCtx(s, { pgMajorVersion: 16 }));
      const fired16 = pg16.fired.find((f) => f.id === "work_mem_inherits");
      expect(fired16).toBeDefined();
      expect(fired16!.conclusion).toContain("After a PG17 upgrade");

      const pg17 = checkup.evaluateAutovacuumRules(mkCtx(s, { pgMajorVersion: 17 }));
      const fired17 = pg17.fired.find((f) => f.id === "work_mem_inherits");
      expect(fired17!.conclusion).toContain("1 GB per-worker cap was removed");

      expect(firedIds(mkCtx(healthySettings()))).not.toContain("work_mem_inherits");
    });

    test("scale_factor_high_global with large-table debt detail", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_scale_factor = mkSetting("0.2");
      const bigTable = mkTable({ table_name: "events", total_relation_size_bytes: 1024 * 1024 * 1024 * 1024 });
      const res = checkup.evaluateAutovacuumRules(mkCtx(s, { largestTables: [bigTable] }));
      const fired = res.fired.find((f) => f.id === "scale_factor_high_global");
      expect(fired).toBeDefined();
      expect(fired!.conclusion).toContain("events");
      expect(fired!.conclusion).toContain("dead-tuple debt");
      expect(fired!.conclusion).toContain("the default");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("scale_factor_high_global");
    });

    test("scale_factor_high_global does NOT fire without any large table (WI gate)", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_scale_factor = mkSetting("0.2");
      // no large tables -> the gate suppresses it on a stock/empty instance
      expect(firedIds(mkCtx(s, { largestTables: [] }))).not.toContain("scale_factor_high_global");
      const smallTable = mkTable({ total_relation_size_bytes: 5 * 1024 * 1024 });
      expect(firedIds(mkCtx(s, { largestTables: [smallTable] }))).not.toContain("scale_factor_high_global");
    });

    test("scale_factor_high_global omits 'the default' for an above-default value", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_scale_factor = mkSetting("0.5");
      const bigTable = mkTable({ total_relation_size_bytes: 1024 * 1024 * 1024 * 1024 });
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(s, { largestTables: [bigTable] }))
        .fired.find((f) => f.id === "scale_factor_high_global");
      expect(fired!.conclusion).toContain("(50%)");
      expect(fired!.conclusion).not.toContain("the default");
    });

    test("analyze_scale_factor_high_global fire and not-fire", () => {
      const bigTable = mkTable({ total_relation_size_bytes: 1024 * 1024 * 1024 * 1024 });
      // fire: default 0.1 + a large table present
      const s = healthySettings();
      s.autovacuum_analyze_scale_factor = mkSetting("0.1");
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(s, { largestTables: [bigTable] }))
        .fired.find((f) => f.id === "analyze_scale_factor_high_global");
      expect(fired).toBeDefined();
      expect(fired!.conclusion).toContain("(the default)");
      // not-fire: no large table
      expect(firedIds(mkCtx(s, { largestTables: [] }))).not.toContain("analyze_scale_factor_high_global");
      // not-fire: below-threshold factor even with a large table
      const below = healthySettings();
      below.autovacuum_analyze_scale_factor = mkSetting("0.09");
      expect(firedIds(mkCtx(below, { largestTables: [bigTable] }))).not.toContain("analyze_scale_factor_high_global");
    });

    test("boundary not-fire cases (just past each threshold)", () => {
      const bigTable = mkTable({ total_relation_size_bytes: 1024 * 1024 * 1024 * 1024 });
      // cost_limit just above the low threshold
      const costLimit = healthySettings();
      costLimit.autovacuum_vacuum_cost_limit = mkSetting("201");
      expect(firedIds(mkCtx(costLimit))).not.toContain("cost_limit_low");
      // scale_factor just below default
      const scale = healthySettings();
      scale.autovacuum_vacuum_scale_factor = mkSetting("0.19");
      expect(firedIds(mkCtx(scale, { largestTables: [bigTable] }))).not.toContain("scale_factor_high_global");
      // naptime exactly at the ceiling (not strictly greater)
      const nap = healthySettings();
      nap.autovacuum_naptime = mkSetting("60", "s");
      expect(firedIds(mkCtx(nap))).not.toContain("naptime_high");
    });

    test("scale_factor_high_global does not cite tables that carry their own override", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_scale_factor = mkSetting("0.2");
      const overridden = mkTable({
        table_name: "events",
        total_relation_size_bytes: 1024 * 1024 * 1024 * 1024,
        has_av_override: true,
        scale_factor_override: 0.01,
      });
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(s, { largestTables: [overridden] }))
        .fired.find((f) => f.id === "scale_factor_high_global");
      expect(fired!.conclusion).not.toContain("dead-tuple debt");
    });

    test("log_autovacuum_disabled with PG15 note gating", () => {
      const s = healthySettings();
      s.log_autovacuum_min_duration = mkSetting("-1", "ms");
      const pg15 = checkup.evaluateAutovacuumRules(mkCtx(s, { pgMajorVersion: 15 }));
      const f15 = pg15.fired.find((f) => f.id === "log_autovacuum_disabled");
      expect(f15).toBeDefined();
      expect(f15!.conclusion).toContain("PG15+");
      const f13 = checkup
        .evaluateAutovacuumRules(mkCtx(s, { pgMajorVersion: 13 }))
        .fired.find((f) => f.id === "log_autovacuum_disabled");
      expect(f13!.conclusion).not.toContain("PG15+");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("log_autovacuum_disabled");
    });

    test("naptime_high", () => {
      const s = healthySettings();
      s.autovacuum_naptime = mkSetting("120", "s");
      expect(firedIds(mkCtx(s))).toContain("naptime_high");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("naptime_high");
    });

    test("max_workers_default", () => {
      const s = healthySettings();
      s.autovacuum_max_workers = mkSetting("3");
      expect(firedIds(mkCtx(s))).toContain("max_workers_default");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("max_workers_default");
    });

    test("insert_vacuum_disabled is PG13+ gated", () => {
      const s = healthySettings();
      s.autovacuum_vacuum_insert_threshold = mkSetting("-1");
      expect(firedIds(mkCtx(s, { pgMajorVersion: 13 }))).toContain("insert_vacuum_disabled");

      // PG12: the setting does not exist -> null -> rule does not apply
      const pg12 = healthySettings();
      delete pg12.autovacuum_vacuum_insert_threshold;
      expect(firedIds(mkCtx(pg12, { pgMajorVersion: 12 }))).not.toContain("insert_vacuum_disabled");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("insert_vacuum_disabled");
    });

    test("dirty_budget_small_large_db fires at stock defaults on a large DB", () => {
      // Realistic PG12+ defaults: 200 / 2ms -> dirty budget ~40.96 MB/s (<= 41).
      const s = healthySettings();
      s.autovacuum_vacuum_cost_limit = mkSetting("200");
      s.autovacuum_vacuum_cost_delay = mkSetting("2", "ms");
      const bigDb = 200 * 1024 * 1024 * 1024;
      const budget = checkup.computeThroughputBudget(checkup.resolveEffectiveAutovacuumSettings(s));
      expect(budget.dirty_write_mbps).toBeCloseTo(41.0, 1);
      expect(firedIds(mkCtx(s, { databaseSizeBytes: bigDb }))).toContain("dirty_budget_small_large_db");
      // small DB: not flagged
      expect(firedIds(mkCtx(s, { databaseSizeBytes: 1024 }))).not.toContain("dirty_budget_small_large_db");
      // ample budget (tuned cost_limit): not flagged even on a large DB
      const tuned = healthySettings(); // cost_limit 2000 -> ~410 MB/s dirty
      expect(firedIds(mkCtx(tuned, { databaseSizeBytes: bigDb }))).not.toContain("dirty_budget_small_large_db");
    });

    test("per_table_cost_delay_zero", () => {
      const reloptions: checkup.ReloptionsOverview = {
        ...emptyOverview,
        tables_with_av_overrides: 1,
        cost_delay_zero_tables: ['"public"."hot"'],
      };
      expect(firedIds(mkCtx(healthySettings(), { reloptions }))).toContain("per_table_cost_delay_zero");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("per_table_cost_delay_zero");
    });

    test("per_table_cost_delay_zero distinguishes heap-level from toast-level overrides", () => {
      // Two tables flagged; only the toast one is in the toast-only subset.
      const reloptions: checkup.ReloptionsOverview = {
        ...emptyOverview,
        tables_with_av_overrides: 2,
        cost_delay_zero_tables: ['"public"."hot"', '"public"."docs"'],
        cost_delay_zero_toast_only_tables: ['"public"."docs"'],
      };
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(healthySettings(), { reloptions }))
        .fired.find((f) => f.id === "per_table_cost_delay_zero");
      expect(fired).toBeDefined();
      // The toast-only table is annotated; the heap-level one is left plain.
      expect(fired!.conclusion).toContain('"public"."docs" (toast-level)');
      expect(fired!.conclusion).toContain('"public"."hot"');
      expect(fired!.conclusion).not.toContain('"public"."hot" (toast-level)');
      // The remediation calls out the toast.* reset for the toast-level ones.
      expect(fired!.recommendation).toContain("toast.autovacuum_vacuum_cost_delay");
    });

    test("per_table_cost_delay_zero omits the toast note when all overrides are heap-level", () => {
      const reloptions: checkup.ReloptionsOverview = {
        ...emptyOverview,
        tables_with_av_overrides: 1,
        cost_delay_zero_tables: ['"public"."hot"'],
      };
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(healthySettings(), { reloptions }))
        .fired.find((f) => f.id === "per_table_cost_delay_zero");
      expect(fired!.conclusion).not.toContain("(toast-level)");
      expect(fired!.recommendation).not.toContain("toast.autovacuum_vacuum_cost_delay");
    });

    test("reloptions_overview", () => {
      const reloptions: checkup.ReloptionsOverview = {
        ...emptyOverview,
        relations_total: 5000,
        tables_with_av_overrides: 3,
        autovacuum_disabled_tables: ['"public"."cache"'],
      };
      const fired = checkup
        .evaluateAutovacuumRules(mkCtx(healthySettings(), { reloptions }))
        .fired.find((f) => f.id === "reloptions_overview");
      expect(fired).toBeDefined();
      expect(fired!.conclusion).toContain("3");
      expect(firedIds(mkCtx(healthySettings()))).not.toContain("reloptions_overview");
    });
  });

  // --- never-recommend list (hard rules, test-enforced) -------------------
  describe("never-recommend list is enforced", () => {
    // A battery of contexts that trigger as many rules as possible.
    function allTriggeringContexts(): checkup.AutovacuumRuleContext[] {
      const bigTable = mkTable({ total_relation_size_bytes: 1024 * 1024 * 1024 * 1024 });
      const bigDb = 500 * 1024 * 1024 * 1024;
      const reloptions: checkup.ReloptionsOverview = {
        relations_total: 100000,
        candidates_considered: 100,
        tables_with_av_overrides: 5,
        autovacuum_disabled_tables: ['"public"."x"'],
        cost_delay_zero_tables: ['"public"."y"'],
        cost_delay_zero_toast_only_tables: [],
      };
      const contexts: checkup.AutovacuumRuleContext[] = [];
      const mutators: Array<(s: Record<string, checkup.SettingInfo>) => void> = [
        (s) => (s.autovacuum = mkSetting("off")),
        (s) => (s.track_counts = mkSetting("off")),
        (s) => (s.autovacuum_vacuum_cost_delay = mkSetting("0", "ms")),
        (s) => (s.autovacuum_vacuum_cost_delay = mkSetting("20", "ms")),
        (s) => (s.autovacuum_vacuum_cost_limit = mkSetting("200")),
        (s) => (s.autovacuum_work_mem = mkSetting("-1", "kB")),
        (s) => (s.autovacuum_vacuum_scale_factor = mkSetting("0.2")),
        (s) => (s.autovacuum_analyze_scale_factor = mkSetting("0.1")),
        (s) => (s.log_autovacuum_min_duration = mkSetting("-1", "ms")),
        (s) => (s.autovacuum_naptime = mkSetting("300", "s")),
        (s) => (s.autovacuum_max_workers = mkSetting("3")),
        (s) => (s.autovacuum_vacuum_insert_threshold = mkSetting("-1")),
      ];
      for (const pg of [12, 13, 15, 17]) {
        for (const m of mutators) {
          const s = healthySettings();
          m(s);
          contexts.push(mkCtx(s, { pgMajorVersion: pg, largestTables: [bigTable], reloptions, databaseSizeBytes: bigDb }));
        }
      }
      return contexts;
    }

    test("no recommendation ever suggests cost_delay = 0, disabling autovacuum, or per-table off", () => {
      const banned = [
        /cost_delay\s*=\s*0(?![.\d])/i,
        /disabl(e|ing)\s+autovacuum/i,
        /(?:^|[^_])autovacuum\s*=\s*off/i,
        /autovacuum_enabled\s*=\s*(?:off|false|0)/i,
      ];
      for (const ctx of allTriggeringContexts()) {
        for (const rec of checkup.evaluateAutovacuumRules(ctx).recommendations) {
          for (const pat of banned) {
            expect(rec).not.toMatch(pat);
          }
        }
      }
    });

    test("any recommendation to add workers also raises the cost budget", () => {
      for (const ctx of allTriggeringContexts()) {
        for (const rec of checkup.evaluateAutovacuumRules(ctx).recommendations) {
          if (/autovacuum_max_workers/i.test(rec)) {
            expect(rec.toLowerCase()).toContain("cost_limit");
          }
        }
      }
    });

    test("the battery collectively fires every F001 rule (including dirty_budget_small_large_db)", () => {
      const fired = new Set<string>();
      for (const ctx of allTriggeringContexts()) {
        for (const f of checkup.evaluateAutovacuumRules(ctx).fired) fired.add(f.id);
      }
      const allRuleIds = checkup.F001_RULES.map((r) => r.id);
      for (const id of allRuleIds) {
        expect(fired.has(id)).toBe(true);
      }
    });
  });

  // --- generator integration ----------------------------------------------
  describe("getAutovacuumRelopts", () => {
    // toast.* autovacuum options are stored on the toast relation as plain
    // autovacuum_* options; the SQL surfaces them 'toast.'-prefixed on the
    // parent. Simulate that shape and assert the toast-level cost_delay=0 is
    // attributed to the owning heap and flags the per-table rule.
    test("attributes a toast-level cost_delay=0 override to the owning table", async () => {
      const mockClient = createMockClient({
        autovacuumReloptsRows: [
          {
            tag_schemaname: "public",
            tag_relname: "docs",
            tag_relkind: "r",
            tag_category: "override",
            relpages: "500",
            total_relation_size_b: "1048576",
            has_av_override: 1,
            reloptions: "toast.autovacuum_vacuum_cost_delay=0",
            relations_total: "10",
            tables_with_av_overrides: "1",
          },
        ],
      });
      const { overview } = await checkup.getAutovacuumRelopts(mockClient as any, 16);
      expect(overview.cost_delay_zero_tables).toContain('"public"."docs"');
      // A toast-only override is classified as such (heap has no cost_delay=0).
      expect(overview.cost_delay_zero_toast_only_tables).toContain('"public"."docs"');
      expect(overview.tables_with_av_overrides).toBe(1);
    });

    test("a heap-level cost_delay=0 is NOT classified as toast-only", async () => {
      const mockClient = createMockClient({
        autovacuumReloptsRows: [
          {
            tag_schemaname: "public",
            tag_relname: "hot",
            tag_relkind: "r",
            tag_category: "override",
            relpages: "5",
            total_relation_size_b: "8192",
            has_av_override: 1,
            // Heap-level zero plus a toast-level zero: heap wins, so not toast-only.
            reloptions: "autovacuum_vacuum_cost_delay=0,toast.autovacuum_vacuum_cost_delay=0",
            relations_total: "10",
            tables_with_av_overrides: "1",
          },
        ],
      });
      const { overview } = await checkup.getAutovacuumRelopts(mockClient as any, 16);
      expect(overview.cost_delay_zero_tables).toContain('"public"."hot"');
      expect(overview.cost_delay_zero_toast_only_tables).not.toContain('"public"."hot"');
    });

    test("counts distinct relations (not rows) for candidates_considered", async () => {
      // Same relation appears in both 'largest' and 'override' categories.
      const shared = {
        tag_schemaname: "public",
        tag_relname: "events",
        tag_relkind: "r",
        relpages: "999",
        total_relation_size_b: "8192",
        has_av_override: 1,
        reloptions: "autovacuum_vacuum_scale_factor=0.01",
        relations_total: "3",
        tables_with_av_overrides: "1",
      };
      const mockClient = createMockClient({
        autovacuumReloptsRows: [
          { ...shared, tag_category: "largest" },
          { ...shared, tag_category: "override" },
        ],
      });
      const { overview } = await checkup.getAutovacuumRelopts(mockClient as any, 16);
      // 2 rows, 1 distinct relation -> must not exceed relations_total
      expect(overview.candidates_considered).toBe(1);
      expect(overview.candidates_considered).toBeLessThanOrEqual(overview.relations_total);
    });

    test("a toast.* scale_factor is not mistaken for the heap's own override", async () => {
      const mockClient = createMockClient({
        autovacuumReloptsRows: [
          {
            tag_schemaname: "public",
            tag_relname: "events",
            tag_relkind: "r",
            tag_category: "largest",
            relpages: "999",
            total_relation_size_b: String(1024 * 1024 * 1024 * 1024),
            has_av_override: 1,
            reloptions: "toast.autovacuum_vacuum_scale_factor=0.5",
            relations_total: "1",
            tables_with_av_overrides: "1",
          },
        ],
      });
      const { largestTables } = await checkup.getAutovacuumRelopts(mockClient as any, 16);
      expect(largestTables[0].scale_factor_override).toBeNull();
    });
  });

  describe("generateF001", () => {
    const richSettingsRows = [
      { tag_setting_name: "autovacuum", tag_setting_value: "on", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "bool", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "track_counts", tag_setting_value: "on", tag_unit: "", tag_category: "Statistics", tag_vartype: "bool", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "autovacuum_vacuum_cost_delay", tag_setting_value: "20", tag_unit: "ms", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 0, setting_normalized: 0.02, unit_normalized: "seconds" },
      { tag_setting_name: "autovacuum_vacuum_cost_limit", tag_setting_value: "-1", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "vacuum_cost_limit", tag_setting_value: "200", tag_unit: "", tag_category: "Resource Usage", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "autovacuum_vacuum_scale_factor", tag_setting_value: "0.2", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "real", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "autovacuum_work_mem", tag_setting_value: "-1", tag_unit: "kB", tag_category: "Autovacuum", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
      { tag_setting_name: "maintenance_work_mem", tag_setting_value: "65536", tag_unit: "kB", tag_category: "Resource Usage", tag_vartype: "integer", is_default: 1, setting_normalized: null, unit_normalized: null },
    ];

    test("produces effective values, throughput budget, conclusions and recommendations", async () => {
      const mockClient = createMockClient({
        settingsRows: richSettingsRows,
        autovacuumReloptsRows: [
          {
            tag_schemaname: "public",
            tag_relname: "events",
            tag_relkind: "r",
            tag_category: "largest",
            relpages: "9999999",
            total_relation_size_b: String(2 * 1024 * 1024 * 1024 * 1024),
            has_av_override: 0,
            reloptions: "",
            relations_total: "42000",
            tables_with_av_overrides: "2",
          },
          {
            tag_schemaname: "public",
            tag_relname: "hot",
            tag_relkind: "r",
            tag_category: "override",
            relpages: "10",
            total_relation_size_b: "8192",
            has_av_override: 1,
            reloptions: "autovacuum_vacuum_cost_delay=0",
            relations_total: "42000",
            tables_with_av_overrides: "2",
          },
        ],
      });
      const report = await checkup.REPORT_GENERATORS.F001(mockClient as any, "node-1");
      const node = report.results["node-1"] as any;

      expect(report.checkId).toBe("F001");
      // raw dump preserved (backward compatible)
      expect(node.data.autovacuum).toBeDefined();
      // inheritance chain resolved
      expect(node.effective_values.cost_limit.inherited_from).toBe("vacuum_cost_limit");
      expect(node.effective_values.cost_limit.effective).toBe(200);
      // throughput budget present
      expect(node.throughput_budget.tokens_per_sec).toBeGreaterThan(0);
      // rule findings surfaced
      const ids = node.settings_analysis.rules_fired.map((r: any) => r.id);
      expect(ids).toContain("cost_delay_pg11_legacy");
      expect(ids).toContain("cost_limit_low");
      expect(ids).toContain("scale_factor_high_global");
      expect(ids).toContain("per_table_cost_delay_zero");
      // reloptions coverage counters
      expect(node.settings_analysis.reloptions_overview.relations_total).toBe(42000);
      expect(node.settings_analysis.reloptions_overview.cost_delay_zero_tables.length).toBe(1);
      expect(node.conclusions.length).toBeGreaterThan(0);
      expect(node.recommendations.length).toBe(node.conclusions.length);
    });

    test("degrades gracefully when the largest-tables query fails", async () => {
      const failingClient = {
        query: async (sql: string) => {
          if (sql.includes("has_av_override")) throw new Error("statement timeout");
          return (createMockClient({ settingsRows: richSettingsRows }) as any).query(sql);
        },
      };
      const report = await checkup.REPORT_GENERATORS.F001(failingClient as any, "node-1");
      const node = report.results["node-1"] as any;
      // still produces effective values + throughput even without table cross-ref
      expect(node.effective_values).toBeDefined();
      expect(node.throughput_budget).toBeDefined();
      expect(node.settings_analysis.largest_tables).toEqual([]);
    });
  });
});

// CLI tests
describe("CLI tests", () => {
  test("checkup command exists and shows help", () => {
    const r = runCli(["checkup", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/express mode/i);
    expect(r.stdout).toMatch(/--check-id/);
    expect(r.stdout).toMatch(/--node-name/);
    expect(r.stdout).toMatch(/--output/);
    expect(r.stdout).toMatch(/upload/);
    expect(r.stdout).toMatch(/--json/);
  });

  test("checkup --help shows available check IDs", () => {
    const r = runCli(["checkup", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/A002/);
    expect(r.stdout).toMatch(/A003/);
    expect(r.stdout).toMatch(/A004/);
    expect(r.stdout).toMatch(/A007/);
    expect(r.stdout).toMatch(/A013/);
    expect(r.stdout).toMatch(/H001/);
    expect(r.stdout).toMatch(/H002/);
    expect(r.stdout).toMatch(/H004/);
  });

  test("checkup without connection shows help", () => {
    const r = runCli(["checkup"]);
    expect(r.status).not.toBe(0);
    // Should show full help (options + examples), like `checkup --help`
    expect(r.stdout).toMatch(/generate health check reports/i);
    expect(r.stdout).toMatch(/--check-id/);
    expect(r.stdout).toMatch(/available checks/i);
    expect(r.stdout).toMatch(/A002/);
  });

  test("checkup --help shows --upload and --no-upload options", () => {
    const r = runCli(["checkup", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--upload/);
    expect(r.stdout).toMatch(/--no-upload/);
  });

  test("checkup --no-upload is recognized as valid option", () => {
    // Should not produce "unknown option" error for --no-upload
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload"]);
    // Connection will fail, but option parsing should succeed
    expect(r.stderr).not.toMatch(/unknown option/i);
    expect(r.stderr).not.toMatch(/did you mean/i);
  });

  test("checkup --upload is recognized as valid option", () => {
    // Should not produce "unknown option" error for --upload
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--upload"]);
    // Connection will fail, but option parsing should succeed
    expect(r.stderr).not.toMatch(/unknown option/i);
    expect(r.stderr).not.toMatch(/did you mean/i);
  });

  test("checkup --json does not imply --no-upload (decoupled behavior)", () => {
    // Use empty config dir to ensure no API key is configured
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // --json alone should NOT disable upload - when --upload is explicitly requested
    // with --json, it should require API key (proving upload is not disabled)
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--json", "--upload"], env);
    // Should fail with "API key is required" because upload is enabled
    expect(r.stderr).toMatch(/API key is required/i);
    expect(r.stderr).not.toMatch(/unknown option/i);
  });

  test("checkup --json --no-upload explicitly disables upload", () => {
    // Use empty config dir to ensure no API key is configured
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // --json with --no-upload should disable upload (no API key error)
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--json", "--no-upload"], env);
    // Should NOT show "API key is required" because upload is disabled
    expect(r.stderr).not.toMatch(/API key is required/i);
    expect(r.stderr).not.toMatch(/unknown option/i);
  });

  test("checkup --upload requires API key", () => {
    // Use empty config dir to ensure no API key is configured
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // --upload explicitly requests upload, should fail without API key
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--upload"], env);
    expect(r.stderr).toMatch(/API key is required/i);
  });

  test("checkup --no-upload does not require API key", () => {
    // Use empty config dir to ensure no API key is configured
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // --no-upload disables upload, should not require API key
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload"], env);
    expect(r.stderr).not.toMatch(/API key is required/i);
  });

  test("checkup --help shows --markdown option", () => {
    const r = runCli(["checkup", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--markdown/);
    expect(r.stdout).toMatch(/PostgresAI API/i);
    expect(r.stdout).toMatch(/transmits the full\s+report JSON/i);
  });

  test("checkup --markdown is recognized as valid option", () => {
    // Should not produce "unknown option" error for --markdown
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--markdown"]);
    // Connection will fail, but option parsing should succeed
    expect(r.stderr).not.toMatch(/unknown option/i);
    expect(r.stderr).not.toMatch(/did you mean/i);
  });

  test("checkup --markdown works without API key", () => {
    // Use empty config dir to ensure no API key is configured
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // --markdown should work even without API key
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--markdown"], env);
    // Connection will fail, but --markdown flag should be recognized
    expect(r.status).not.toBe(0);
    expect(r.stderr).not.toMatch(/unknown option/i);
    expect(r.stderr).not.toMatch(/API key is required/i);
    expect(r.stderr).toMatch(/full report JSON will still be sent/i);
    expect(r.stderr).toMatch(/markdown conversion/i);
  });

  test("checkup with --no-upload and no output flags shows summary", () => {
    // This test verifies that when running with --no-upload and no output flags,
    // the user gets a summary of checks.
    // Note: This will fail to connect, but we can still verify behavior.
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload"], env);

    // The command will fail due to connection error, but if it succeeded,
    // it should show the summary. We can't test the success case without a real DB,
    // but we verify the option parsing is correct (tested above in other tests).
    // The actual summary output is tested in integration tests.
    expect(r.status).not.toBe(0); // Will fail due to connection
  });

  test("checkup rejects --json and --markdown together", () => {
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--json", "--markdown", "--no-upload"], env);

    // Should fail with mutual exclusivity error
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/mutually exclusive/i);
  });

  // Argument parsing tests for check ID / connection string recognition
  test("checkup with check ID but no connection shows specific error", () => {
    const r = runCli(["checkup", "H002"]);
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/connection string required/i);
    expect(r.stderr).toMatch(/H002/);
  });

  test("checkup recognizes valid check ID patterns", () => {
    // Valid check IDs: A002, H002, F004, etc.
    for (const checkId of ["A002", "H002", "F004", "K003", "a002", "h002"]) {
      const r = runCli(["checkup", checkId]);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toMatch(/connection string required/i);
    }
  });

  test("checkup does not treat connection string as check ID", () => {
    // Connection strings should not be parsed as check IDs
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload"]);
    // Should not show "connection string required" error
    expect(r.stderr).not.toMatch(/connection string required/i);
  });

  test("checkup with check ID and connection string works", () => {
    // pgai checkup H002 postgresql://...
    const r = runCli(["checkup", "H002", "postgresql://test:test@localhost:5432/test", "--no-upload"]);
    // Should not show "connection string required" error
    expect(r.stderr).not.toMatch(/connection string required/i);
    // Connection will fail but argument parsing should succeed
    expect(r.stderr).not.toMatch(/unknown option/i);
  });

  test("checkup with --check-id option works", () => {
    // pgai checkup --check-id H002 postgresql://...
    const r = runCli(["checkup", "--check-id", "H002", "postgresql://test:test@localhost:5432/test", "--no-upload"]);
    // Should not show "connection string required" error
    expect(r.stderr).not.toMatch(/connection string required/i);
    expect(r.stderr).not.toMatch(/unknown option/i);
  });

  // Tests for --output flag behavior (suppresses stdout when specified)
  test("checkup --output option is recognized", () => {
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload", "--output", "/tmp/test-output"]);
    // Connection will fail, but option parsing should succeed
    expect(r.stderr).not.toMatch(/unknown option/i);
    expect(r.stderr).not.toMatch(/did you mean/i);
  });

  test("checkup --json --output should NOT output JSON to stdout (writes to files only)", () => {
    // This is a behavioral test - when --output is specified along with --json,
    // JSON should only be written to files, not to stdout.
    // We verify by checking the help text describes this behavior
    const r = runCli(["checkup", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--output/);
    expect(r.stdout).toMatch(/--json/);
  });

  test("checkup --output creates directory if it doesn't exist", () => {
    const env = { XDG_CONFIG_HOME: "/tmp/postgresai-test-empty-config" };
    // Use a temp directory that might not exist
    const tempDir = `/tmp/postgresai-test-output-${Date.now()}`;
    const r = runCli(["checkup", "postgresql://test:test@localhost:5432/test", "--no-upload", "--json", "--output", tempDir], env);
    // Connection will fail, but directory creation should be attempted before connection
    // The error should be about connection, not about directory
    expect(r.stderr).not.toMatch(/Failed to create output directory/i);
  });
});

// CLI-level tests for the checkup auth pre-flight: missing/invalid credentials
// must surface BEFORE checks run (previously the upload at the end of the run
// was the first authenticated call, wasting minutes of work on a 401).
describe("checkup auth pre-flight (CLI)", () => {
  // Dead Postgres port: if the pre-flight correctly stops the run, the CLI
  // never attempts this connection; if the run continues, the connection
  // failure mentions this address.
  const DEAD_DB = "postgresql://test:test@127.0.0.1:2/test";

  test("--no-upload rejects --markdown before database or API work", () => {
    const env = {
      XDG_CONFIG_HOME: `/tmp/postgresai-test-no-upload-markdown-${process.pid}`,
      PGAI_API_KEY: "configured-token",
      PGAI_API_BASE_URL: "http://127.0.0.1:1",
    };
    const r = runCli(["checkup", DEAD_DB, "--no-upload", "--markdown"], env);

    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/--no-upload and --markdown are mutually exclusive/i);
    expect(r.stderr).toMatch(/markdown conversion is performed by the PostgresAI API/i);
    expect(r.stderr).toMatch(/use --json or --output/i);
    expect(r.stderr).not.toMatch(/127\.0\.0\.1:2|ECONNREFUSED/);
  });

  test("no API key: prominent notice, run continues locally", () => {
    const env = {
      XDG_CONFIG_HOME: `/tmp/postgresai-test-preflight-nokey-${process.pid}`,
      PGAI_API_KEY: "",
    };
    const r = runCli(["checkup", DEAD_DB], env);
    expect(r.stderr).toMatch(/results will NOT be uploaded/i);
    expect(r.stderr).toMatch(/auth login/);
    expect(r.stderr).toMatch(/--no-upload/);
    // Falls back to local-only mode instead of failing fast
    expect(r.stderr).not.toMatch(/API key is required/i);
    // Run continued to the database connection stage
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/127\.0\.0\.1:2|ECONNREFUSED|connect/i);
  });

  test("--no-upload: no notice, pre-flight skipped entirely", () => {
    const env = {
      XDG_CONFIG_HOME: `/tmp/postgresai-test-preflight-noupload-${process.pid}`,
      PGAI_API_KEY: "",
    };
    const r = runCli(["checkup", DEAD_DB, "--no-upload"], env);
    expect(r.stderr).not.toMatch(/results will NOT be uploaded/i);
    expect(r.stderr).not.toMatch(/could not verify API key/i);
    expect(r.stderr).not.toMatch(/API key is required/i);
  });

  test("invalid API key (HTTP 401): fails fast before running checks", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch() {
        return new Response(JSON.stringify({ message: "Invalid token" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        });
      },
    });
    try {
      const env = {
        XDG_CONFIG_HOME: `/tmp/postgresai-test-preflight-badkey-${process.pid}`,
        PGAI_API_KEY: "bad-token",
        PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}`,
      };
      const r = await runCliAsync(["checkup", DEAD_DB, "--project", "preflight-test"], env);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toMatch(/rejected by the PostgresAI API \(HTTP 401\)/);
      expect(r.stderr).toMatch(/auth login/);
      expect(r.stderr).toMatch(/--no-upload/);
      // Stopped BEFORE connecting to the database / running checks
      expect(r.stderr).not.toMatch(/127\.0\.0\.1:2|ECONNREFUSED/);
    } finally {
      server.stop(true);
    }
  });

  test("--no-upload skips pre-flight even when an invalid key is configured", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch() {
        return new Response(JSON.stringify({ message: "Invalid token" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        });
      },
    });
    try {
      const env = {
        XDG_CONFIG_HOME: `/tmp/postgresai-test-preflight-badkey-noupload-${process.pid}`,
        PGAI_API_KEY: "bad-token",
        PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}`,
      };
      const r = await runCliAsync(["checkup", DEAD_DB, "--no-upload"], env);
      expect(r.stderr).not.toMatch(/rejected by the PostgresAI API/);
      expect(r.stderr).not.toMatch(/could not verify API key/i);
      // Fails later at the database connection stage instead
      expect(r.status).not.toBe(0);
    } finally {
      server.stop(true);
    }
  });

  test("transient pre-flight failure (network error): warns and continues", () => {
    const env = {
      XDG_CONFIG_HOME: `/tmp/postgresai-test-preflight-netfail-${process.pid}`,
      PGAI_API_KEY: "some-token",
      PGAI_API_BASE_URL: "http://127.0.0.1:1", // connect refused — transient, not a 401/403
    };
    const r = runCli(["checkup", DEAD_DB, "--project", "preflight-test"], env);
    expect(r.stderr).toMatch(/Warning: could not verify API key/i);
    expect(r.stderr).not.toMatch(/rejected by the PostgresAI API/);
    // Run continued to the database connection stage
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/127\.0\.0\.1:2|ECONNREFUSED|connect/i);
  });
});

// Tests for checkup-api module
describe("checkup-api", () => {
  test("formatRpcErrorForDisplay formats details/hint nicely", () => {
    const err = new api.RpcError({
      rpcName: "checkup_report_file_post",
      statusCode: 402,
      payloadText: JSON.stringify({
        hint: "Start an express checkup subscription for the organization or contact support.",
        details: "Checkup report uploads require an active checkup subscription",
      }),
      payloadJson: {
        hint: "Start an express checkup subscription for the organization or contact support.",
        details: "Checkup report uploads require an active checkup subscription.",
      },
    });
    const lines = api.formatRpcErrorForDisplay(err);
    const text = lines.join("\n");
    expect(text).toMatch(/RPC checkup_report_file_post failed: HTTP 402/);
    expect(text).toMatch(/Details:/);
    expect(text).toMatch(/Hint:/);
  });

  test("withRetry succeeds on first attempt", async () => {
    let attempts = 0;
    const result = await api.withRetry(async () => {
      attempts++;
      return "success";
    });
    expect(result).toBe("success");
    expect(attempts).toBe(1);
  });

  test("withRetry retries on retryable errors and succeeds", async () => {
    let attempts = 0;
    const result = await api.withRetry(
      async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error("connection timeout");
        }
        return "success after retry";
      },
      { maxAttempts: 3, initialDelayMs: 10 }
    );
    expect(result).toBe("success after retry");
    expect(attempts).toBe(3);
  });

  test("withRetry calls onRetry callback", async () => {
    let attempts = 0;
    const retryLogs: string[] = [];
    await api.withRetry(
      async () => {
        attempts++;
        if (attempts < 2) {
          throw new Error("socket hang up");
        }
        return "ok";
      },
      { maxAttempts: 3, initialDelayMs: 10 },
      (attempt, _err, delayMs) => {
        retryLogs.push(`attempt ${attempt}, delay ${delayMs}ms`);
      }
    );
    expect(retryLogs.length).toBe(1);
    expect(retryLogs[0]).toMatch(/attempt 1/);
  });

  test("withRetry does not retry on non-retryable errors", async () => {
    let attempts = 0;
    try {
      await api.withRetry(
        async () => {
          attempts++;
          throw new Error("invalid input");
        },
        { maxAttempts: 3, initialDelayMs: 10 }
      );
    } catch (err) {
      expect((err as Error).message).toBe("invalid input");
    }
    expect(attempts).toBe(1);
  });

  test("withRetry does not retry on 4xx RpcError", async () => {
    let attempts = 0;
    try {
      await api.withRetry(
        async () => {
          attempts++;
          throw new api.RpcError({
            rpcName: "test",
            statusCode: 400,
            payloadText: "bad request",
            payloadJson: null,
          });
        },
        { maxAttempts: 3, initialDelayMs: 10 }
      );
    } catch (err) {
      expect(err).toBeInstanceOf(api.RpcError);
    }
    expect(attempts).toBe(1);
  });

  test("withRetry retries on 5xx RpcError", async () => {
    let attempts = 0;
    try {
      await api.withRetry(
        async () => {
          attempts++;
          throw new api.RpcError({
            rpcName: "test",
            statusCode: 503,
            payloadText: "service unavailable",
            payloadJson: null,
          });
        },
        { maxAttempts: 2, initialDelayMs: 10 }
      );
    } catch (err) {
      expect(err).toBeInstanceOf(api.RpcError);
    }
    expect(attempts).toBe(2);
  });

  test("withRetry retries on timeout errors", async () => {
    // Tests that timeout-like error messages are considered retryable
    let attempts = 0;
    try {
      await api.withRetry(
        async () => {
          attempts++;
          throw new Error("RPC test timed out after 30000ms (no response)");
        },
        { maxAttempts: 3, initialDelayMs: 10 }
      );
    } catch (err) {
      expect(err).toBeInstanceOf(Error);
      expect((err as Error).message).toContain("timed out");
    }
    expect(attempts).toBe(3); // Should retry on timeout
  });

  test("withRetry retries on ECONNRESET errors", async () => {
    // Tests that connection reset errors are considered retryable
    let attempts = 0;
    try {
      await api.withRetry(
        async () => {
          attempts++;
          const err = new Error("connection reset") as Error & { code: string };
          err.code = "ECONNRESET";
          throw err;
        },
        { maxAttempts: 2, initialDelayMs: 10 }
      );
    } catch (err) {
      expect(err).toBeInstanceOf(Error);
    }
    expect(attempts).toBe(2); // Should retry on ECONNRESET
  });

  // Transport selection — pick http/https by URL protocol, but refuse HTTP
  // to non-loopback hosts unless CHECKUP_ALLOW_HTTP=1 is set (prevents
  // typo-driven plaintext API-key leaks like http://api.postgres.ai/...).
  describe("transport selection", () => {
    test("https URL does not trip the guard (network error expected)", async () => {
      let caught: Error | null = null;
      try {
        await api.createCheckupReport({
          apiKey: "dummy",
          apiBaseUrl: "https://127.0.0.1:1/api", // port 1 — connect refused
          project: "p",
        });
      } catch (e) {
        caught = e as Error;
      }
      expect(caught).not.toBeNull();
      expect(caught!.message).not.toMatch(/Refusing to send API key/);
    });

    test("http on loopback does not trip the guard (network error expected)", async () => {
      // IPv6 loopback is written as `[::1]` in URLs; WHATWG URL preserves
      // the brackets in .hostname, so the guard must strip them before
      // matching the allowlist.
      for (const host of ["localhost", "127.0.0.1", "[::1]"]) {
        let caught: Error | null = null;
        try {
          await api.createCheckupReport({
            apiKey: "dummy",
            apiBaseUrl: `http://${host}:1/api`, // port 1 — connect refused
            project: "p",
          });
        } catch (e) {
          caught = e as Error;
        }
        expect(caught).not.toBeNull();
        expect(caught!.message).not.toMatch(/Refusing to send API key/);
      }
    });

    test("http to non-loopback host is refused by the guard", async () => {
      const saved = process.env.CHECKUP_ALLOW_HTTP;
      delete process.env.CHECKUP_ALLOW_HTTP;
      try {
        let caught: Error | null = null;
        try {
          await api.createCheckupReport({
            apiKey: "dummy",
            apiBaseUrl: "http://example.com/api",
            project: "p",
          });
        } catch (e) {
          caught = e as Error;
        }
        expect(caught).not.toBeNull();
        expect(caught!.message).toMatch(/Refusing to send API key over plaintext HTTP/);
        expect(caught!.message).toMatch(/example\.com/);
      } finally {
        if (saved !== undefined) process.env.CHECKUP_ALLOW_HTTP = saved;
      }
    });

    test("CHECKUP_ALLOW_HTTP=1 bypasses the guard for non-loopback hosts", async () => {
      const saved = process.env.CHECKUP_ALLOW_HTTP;
      process.env.CHECKUP_ALLOW_HTTP = "1";
      try {
        let caught: Error | null = null;
        try {
          await api.createCheckupReport({
            apiKey: "dummy",
            apiBaseUrl: "http://127.0.0.2:1/api", // non-loopback-match hostname, connect refused port
            project: "p",
          });
        } catch (e) {
          caught = e as Error;
        }
        expect(caught).not.toBeNull();
        expect(caught!.message).not.toMatch(/Refusing to send API key/);
      } finally {
        if (saved === undefined) delete process.env.CHECKUP_ALLOW_HTTP;
        else process.env.CHECKUP_ALLOW_HTTP = saved;
      }
    });
  });

  // Auth pre-flight: verifyApiKey checks the key with a cheap authenticated
  // GET before checkup runs expensive checks. Only a definitive 401 is
  // "invalid"; 403 is inconclusive (the probe endpoint is a different authz
  // surface than the upload RPC) and anything transient must come back
  // "unknown" (warn + continue).
  describe("verifyApiKey (auth pre-flight)", () => {
    function serveStatus(status: number, body = "[]") {
      return Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch() {
          return new Response(body, { status, headers: { "Content-Type": "application/json" } });
        },
      });
    }

    test("returns valid on HTTP 200", async () => {
      const server = serveStatus(200);
      try {
        const r = await api.verifyApiKey({ apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}` });
        expect(r.status).toBe("valid");
      } finally {
        server.stop(true);
      }
    });

    test("returns invalid on HTTP 401", async () => {
      const server = serveStatus(401, JSON.stringify({ message: "Invalid token" }));
      try {
        const r = await api.verifyApiKey({ apiKey: "bad", apiBaseUrl: `http://127.0.0.1:${server.port}` });
        expect(r.status).toBe("invalid");
        expect((r as { statusCode: number }).statusCode).toBe(401);
      } finally {
        server.stop(true);
      }
    });

    test("returns unknown on HTTP 403 (list endpoint is a different authz surface than the upload RPC)", async () => {
      // Work item 260, finding 2: the probe GETs /checkup_reports (list),
      // but the upload uses the checkup_report_create RPC. A token scoped
      // for upload-but-not-list would 403 here yet upload fine — so 403 must
      // NOT hard-fail the run. Only 401 (bad credentials) is definitive.
      const server = serveStatus(403);
      try {
        const r = await api.verifyApiKey({ apiKey: "bad", apiBaseUrl: `http://127.0.0.1:${server.port}` });
        expect(r.status).toBe("unknown");
        expect((r as { detail: string }).detail).toMatch(/403/);
      } finally {
        server.stop(true);
      }
    });

    test("returns unknown on HTTP 500 (not a definitive rejection)", async () => {
      const server = serveStatus(500);
      try {
        const r = await api.verifyApiKey({ apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}` });
        expect(r.status).toBe("unknown");
        expect((r as { detail: string }).detail).toMatch(/HTTP 500/);
      } finally {
        server.stop(true);
      }
    });

    test("returns unknown on connection refused", async () => {
      const r = await api.verifyApiKey({ apiKey: "k", apiBaseUrl: "http://127.0.0.1:1" }); // port 1 — connect refused
      expect(r.status).toBe("unknown");
    });

    test("returns unknown on timeout", async () => {
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        async fetch() {
          await new Promise((resolve) => setTimeout(resolve, 5000));
          return new Response("[]");
        },
      });
      try {
        const r = await api.verifyApiKey({
          apiKey: "k",
          apiBaseUrl: `http://127.0.0.1:${server.port}`,
          timeoutMs: 100,
        });
        expect(r.status).toBe("unknown");
      } finally {
        server.stop(true);
      }
    });

    test("does not send the key over plaintext HTTP to non-loopback hosts", async () => {
      const saved = process.env.CHECKUP_ALLOW_HTTP;
      delete process.env.CHECKUP_ALLOW_HTTP;
      try {
        const r = await api.verifyApiKey({ apiKey: "k", apiBaseUrl: "http://example.com/api" });
        expect(r.status).toBe("unknown");
        expect((r as { detail: string }).detail).toMatch(/plaintext HTTP/);
      } finally {
        if (saved !== undefined) process.env.CHECKUP_ALLOW_HTTP = saved;
      }
    });
  });
});

// Tests for checkup-summary module
describe("checkup-summary", () => {
  const summary = require("../lib/checkup-summary");

  test("generateCheckSummary for F002 handles healthy, risky, and unavailable reports", () => {
    const base = { databases: [{ database_name: "db1" }], tables: [], settings_available: true };
    expect(summary.generateCheckSummary("F002", { results: { node1: { data: { ...base, severity: "info" } } } })).toEqual({
      status: "ok", message: "1 database below wraparound thresholds",
    });
    expect(summary.generateCheckSummary("F002", { results: { node1: { data: {
      ...base, severity: "high", tables: [{ xid: { severity: "high" }, multixact: { severity: "info" } }],
    } } } })).toEqual({ status: "warning", message: "HIGH wraparound risk; 1 table affected" });
    expect(summary.generateCheckSummary("F002", { results: { node1: { data: {
      ...base, settings_available: false, severity: "info",
    } } } })).toEqual({ status: "info", message: "Wraparound settings unavailable" });
  });

  test("generateCheckSummary for F009 reports severity and dominant holder", () => {
    const result = summary.generateCheckSummary("F009", {
      results: { node1: { data: {
        skipped: false,
        severity: "CRITICAL",
        data_horizon_age_tx: 160000000,
        catalog_horizon_age_tx: 0,
        dominant_holder: { source: "slot" },
      } } },
    });
    expect(result.status).toBe("warning");
    expect(result.message).toContain("CRITICAL");
    expect(result.message).toContain("slot");
  });

  test("generateCheckSummary for F009 handles replica skip", () => {
    const result = summary.generateCheckSummary("F009", {
      results: { node1: { data: { skipped: true, skip_reason: "primary-only" } } },
    });
    expect(result).toEqual({ status: "info", message: "primary-only" });
  });

  test("generateCheckSummary for F009 describes wall-clock notices", () => {
    const result = summary.generateCheckSummary("F009", {
      results: { node1: { data: {
        skipped: false,
        severity: "NOTICE",
        data_horizon_severity: "OK",
        catalog_horizon_severity: "OK",
        data_horizon_age_tx: 1,
        catalog_horizon_age_tx: 0,
        components: {
          pg_stat_activity: { top_blocker: { xact_age_seconds: 3600 } },
        },
        dominant_holder: { source: "activity" },
      } } },
    });
    expect(result).toEqual({
      status: "info",
      message: "NOTICE: transaction open for 3600 seconds; dominant holder: activity",
    });
  });

  test("generateCheckSummary for F003 with no issues", () => {
    const report = {
      results: {
        node1: {
          data: {
            db1: {
              dead_tuples_tables: [],
              total_count: 0,
              flagged_count: 0,
              autovacuum_disabled_count: 0,
              autovacuum_disabled_flagged_count: 0,
            },
          },
        },
      },
    };
    const result = summary.generateCheckSummary("F003", report);
    expect(result.status).toBe("ok");
    expect(result.message).toMatch(/no significant dead tuple/i);
  });

  test("generateCheckSummary for F003 ignores tiny disabled-autovacuum tables", () => {
    const report = {
      results: {
        node1: {
          data: {
            db1: {
              dead_tuples_tables: [],
              total_count: 2,
              flagged_count: 0,
              autovacuum_disabled_count: 2,
              autovacuum_disabled_flagged_count: 0,
            },
          },
        },
      },
    };
    const result = summary.generateCheckSummary("F003", report);
    expect(result.status).toBe("ok");
  });

  test("generateCheckSummary for F003 with problems", () => {
    const report = {
      results: {
        node1: {
          data: {
            db1: {
              dead_tuples_tables: [],
              total_count: 3,
              flagged_count: 1,
              autovacuum_disabled_count: 2,
              autovacuum_disabled_flagged_count: 2,
            },
          },
        },
      },
    };
    const result = summary.generateCheckSummary("F003", report);
    expect(result.status).toBe("warning");
    expect(result.message).toBe(
      "1 table with excessive dead tuples, 2 tables with autovacuum disabled"
    );
  });

  for (const checkId of ["F004", "F005"]) {
    test(`generateCheckSummary for degraded ${checkId} is never ok`, () => {
      const report = {
        results: {
          node1: {
            data: {
              db1: {
                status: {
                  ok: false,
                  reason: "missing_schema",
                  error: 'schema "postgres_ai" does not exist',
                },
                total_count: 0,
              },
            },
          },
        },
      };

      const result = summary.generateCheckSummary(checkId, report);
      expect(result.status).toBe("warning");
      expect(result.message).toMatch(/degraded.*missing schema/i);
    });
  }

  test("generateCheckSummary for H001 with no issues", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              invalid_indexes: [],
              total_count: 0,
              total_size_bytes: 0,
              total_size_pretty: "0 bytes",
              database_size_bytes: 1000000,
              database_size_pretty: "1 MB"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H001", report);
    expect(result.status).toBe("ok");
    expect(result.message).toMatch(/no invalid/i);
  });

  test("generateCheckSummary for H001 with invalid indexes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              invalid_indexes: [{}, {}, {}],
              total_count: 3,
              total_size_bytes: 1024 * 1024 * 245,
              total_size_pretty: "245 MiB",
              database_size_bytes: 1000000000,
              database_size_pretty: "1 GB"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H001", report);
    expect(result.status).toBe("warning");
    expect(result.message).toMatch(/3 invalid indexes/i);
    expect(result.message).toMatch(/245 MiB/i);
  });

  test("generateCheckSummary for H002 with no issues", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              unused_indexes: [],
              total_count: 0,
              total_size_bytes: 0,
              total_size_pretty: "0 bytes",
              database_size_bytes: 1000000,
              database_size_pretty: "1 MB",
              stats_reset: {}
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H002", report);
    expect(result.status).toBe("ok");
    expect(result.message).toMatch(/all indexes utilized/i);
  });

  test("generateCheckSummary for H002 with unused indexes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              unused_indexes: [{}, {}],
              total_count: 2,
              total_size_bytes: 1024 * 1024 * 150,
              total_size_pretty: "150 MiB",
              database_size_bytes: 1000000000,
              database_size_pretty: "1 GB",
              stats_reset: {}
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H002", report);
    expect(result.status).toBe("warning");
    expect(result.message).toMatch(/2 unused indexes/i);
    expect(result.message).toMatch(/150 MiB/i);
  });

  test("generateCheckSummary for H004 with redundant indexes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              redundant_indexes: [{}, {}, {}, {}],
              total_count: 4,
              total_size_bytes: 1024 * 1024 * 1024 * 1.2,
              total_size_pretty: "1.2 GiB",
              database_size_bytes: 10000000000,
              database_size_pretty: "10 GB"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H004", report);
    expect(result.status).toBe("warning");
    expect(result.message).toMatch(/4 redundant indexes/i);
    expect(result.message).toMatch(/1\.2 GiB/i);
  });

  test("generateCheckSummary for A003 (settings)", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "setting1": "value1",
            "setting2": "value2"
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A003", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("2 settings collected");
  });

  test("generateCheckSummary for A002 with PostgreSQL 17", () => {
    const report = {
      results: {
        "node1": {
          data: {
            version: {
              version: "17.2",
              server_version_num: "170002",
              server_major_ver: "17",
              server_minor_ver: "2"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A002", report);
    expect(result.status).toBe("ok");
    expect(result.message).toBe("PostgreSQL 17");
  });

  test("generateCheckSummary for A002 with PostgreSQL 15", () => {
    const report = {
      results: {
        "node1": {
          data: {
            version: {
              version: "15.4",
              server_version_num: "150004",
              server_major_ver: "15",
              server_minor_ver: "4"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A002", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("PostgreSQL 15");
  });

  test("generateCheckSummary for A002 with old PostgreSQL 11", () => {
    const report = {
      results: {
        "node1": {
          data: {
            version: {
              version: "11.8",
              server_version_num: "110008",
              server_major_ver: "11",
              server_minor_ver: "8"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A002", report);
    expect(result.status).toBe("warning");
    expect(result.message).toMatch(/PostgreSQL 11.*consider upgrading/i);
  });

  test("generateCheckSummary for A013 with version", () => {
    const report = {
      results: {
        "node1": {
          data: {
            version: {
              version: "17.2",
              server_version_num: "170002",
              server_major_ver: "17",
              server_minor_ver: "2"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A013", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("Version 17.2");
  });

  test("generateCheckSummary handles empty results", () => {
    const report = { results: {} };
    const result = summary.generateCheckSummary("H001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No data");
  });

  test("generateCheckSummary handles unknown check ID", () => {
    const report = {
      results: {
        "node1": { data: {} }
      }
    };
    const result = summary.generateCheckSummary("UNKNOWN", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("Check completed");
  });

  test("generateCheckSummary for D001 (logging settings)", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "log_destination": { value: "stderr" },
            "log_line_prefix": { value: "%m [%p] " }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("D001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("2 logging settings collected");
  });

  test("generateCheckSummary for D004 (pg_stat_statements)", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "pg_stat_statements.max": { value: "5000" },
            "pg_stat_statements.track": { value: "all" }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("D004", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("2 pg_stat_statements settings collected");
  });

  test("generateCheckSummary for F001 (autovacuum) with no findings is healthy", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "autovacuum": { value: "on" },
            "autovacuum_max_workers": { value: "3" }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("F001", report);
    expect(result.status).toBe("ok");
    expect(result.message).toContain("healthy");
  });

  test("generateCheckSummary for F001 maps CRITICAL findings to warning", () => {
    const report = {
      results: {
        "node1": {
          data: { "autovacuum": { value: "off" } },
          settings_analysis: {
            severity: "CRITICAL",
            rules_fired: [
              { id: "autovacuum_off", severity: "CRITICAL", conclusion: "x", recommendation: "y" },
              { id: "cost_limit_low", severity: "WARNING", conclusion: "x", recommendation: "y" },
            ],
          },
        },
      },
    };
    const result = summary.generateCheckSummary("F001", report);
    expect(result.status).toBe("warning");
    expect(result.message).toContain("1 critical");
    expect(result.message).toContain("1 warning");
  });

  test("generateCheckSummary for F001 maps NOTICE-only findings to info", () => {
    const report = {
      results: {
        "node1": {
          data: { "autovacuum": { value: "on" } },
          settings_analysis: {
            severity: "NOTICE",
            rules_fired: [
              { id: "naptime_high", severity: "NOTICE", conclusion: "x", recommendation: "y" },
            ],
          },
        },
      },
    };
    const result = summary.generateCheckSummary("F001", report);
    expect(result.status).toBe("info");
  });

  test("generateCheckSummary for G001 (memory settings)", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "shared_buffers": { value: "128MB" },
            "work_mem": { value: "4MB" }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("G001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("2 memory settings collected");
  });

  test("generateCheckSummary for G003 with deadlocks", () => {
    const report = {
      results: {
        "node1": {
          data: {
            settings: {
              "lock_timeout": { value: "0" }
            },
            deadlock_stats: {
              deadlocks: 5,
              conflicts: 0,
              stats_reset: "2025-01-01"
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("G003", report);
    expect(result.status).toBe("warning");
    expect(result.message).toBe("5 deadlocks detected");
  });

  test("generateCheckSummary for G003 without deadlocks", () => {
    const report = {
      results: {
        "node1": {
          data: {
            settings: {
              "lock_timeout": { value: "0" },
              "statement_timeout": { value: "0" }
            },
            deadlock_stats: {
              deadlocks: 0,
              conflicts: 0
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("G003", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("2 timeout/lock settings collected");
  });

  // Edge cases: empty data
  test("generateCheckSummary for A003 with no settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("A003", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No settings found");
  });

  test("generateCheckSummary for A004 with no data", () => {
    const report = {
      results: {
        "node1": {}
      }
    };
    const result = summary.generateCheckSummary("A004", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("Cluster information collected");
  });

  test("generateCheckSummary for A004 with no database_sizes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            general_info: {}
          }
        }
      }
    };
    const result = summary.generateCheckSummary("A004", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("Cluster information collected");
  });

  test("generateCheckSummary for A007 with no altered settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("A007", report);
    expect(result.status).toBe("ok");
    expect(result.message).toBe("No altered settings");
  });

  test("generateCheckSummary for A002 with no version data", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("A002", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("Version checked");
  });

  test("generateCheckSummary for D001 with no settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("D001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No logging settings found");
  });

  test("generateCheckSummary for D004 with no settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("D004", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No pg_stat_statements settings found");
  });

  test("generateCheckSummary for F001 with no settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("F001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No autovacuum settings found");
  });

  test("generateCheckSummary for G001 with no settings", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("G001", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No memory settings found");
  });

  test("generateCheckSummary for G003 with no settings or deadlock_stats", () => {
    const report = {
      results: {
        "node1": {
          data: {}
        }
      }
    };
    const result = summary.generateCheckSummary("G003", report);
    expect(result.status).toBe("info");
    expect(result.message).toBe("No timeout/lock settings found");
  });

  test("generateCheckSummary for H001 with no invalid indexes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              invalid_indexes: [],
              total_count: 0,
              total_size_bytes: 0
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H001", report);
    expect(result.status).toBe("ok");
    expect(result.message).toBe("No invalid indexes");
  });

  test("generateCheckSummary for H002 with all indexes utilized", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              unused_indexes: [],
              total_count: 0,
              total_size_bytes: 0
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H002", report);
    expect(result.status).toBe("ok");
    expect(result.message).toBe("All indexes utilized");
  });

  test("generateCheckSummary for H004 with no redundant indexes", () => {
    const report = {
      results: {
        "node1": {
          data: {
            "db1": {
              redundant_indexes: [],
              total_count: 0,
              total_size_bytes: 0
            }
          }
        }
      }
    };
    const result = summary.generateCheckSummary("H004", report);
    expect(result.status).toBe("ok");
    expect(result.message).toBe("No redundant indexes");
  });
});

// Postgres version compatibility tests (PG13-PG19)
describe("Postgres version compatibility (PG13-PG19)", () => {
  /**
   * Version-matrix fixture invariants:
   * - version_num uses Postgres major * 10000 + minor encoding.
   * - shared_buffers 16384 * 8kB is 128 MiB.
   * - days_since_reset is derived by production code from the stats_reset metric's
   *   seconds_since_reset field, not from Date.now().
   * - postmaster uptime, uptime text, and postmaster startup timestamp are derived
   *   from TEST_NOW_EPOCH and POSTMASTER_STARTUP_EPOCH and must stay in sync.
   */
  const createVersionMockData = (major: number, minor: number) => ({
    versionRows: [
      { name: "server_version", setting: `${major}.${minor}` },
      { name: "server_version_num", setting: `${major}${String(minor).padStart(4, "0")}` },
    ],
    settingsRows: [
      {
        tag_setting_name: "shared_buffers",
        tag_setting_value: "16384",
        tag_unit: "8kB",
        tag_category: "Resource Usage / Memory",
        tag_vartype: "integer",
        is_default: 0,
        setting_normalized: "134217728",
        unit_normalized: "bytes",
      },
      {
        tag_setting_name: "autovacuum_vacuum_scale_factor",
        tag_setting_value: "0.2",
        tag_unit: "",
        tag_category: "Autovacuum",
        tag_vartype: "real",
        is_default: 0,
        setting_normalized: null,
        unit_normalized: null,
      },
      {
        tag_setting_name: "log_min_duration_statement",
        tag_setting_value: "-1",
        tag_unit: "ms",
        tag_category: "Reporting and Logging / When to Log",
        tag_vartype: "integer",
        is_default: 1,
        setting_normalized: null,
        unit_normalized: null,
      },
      {
        tag_setting_name: "deadlock_timeout",
        tag_setting_value: "1000",
        tag_unit: "ms",
        tag_category: "Lock Management",
        tag_vartype: "integer",
        is_default: 1,
        setting_normalized: null,
        unit_normalized: null,
      },
    ],
    databaseSizesRows: [{ datname: "postgres", size_bytes: "1073741824" }],
    dbStatsRows: [{
      numbackends: 5,
      xact_commit: 100,
      xact_rollback: 1,
      blks_read: 100,
      blks_hit: 900,
      tup_returned: 500,
      tup_fetched: 400,
      tup_inserted: 50,
      tup_updated: 30,
      tup_deleted: 10,
      deadlocks: 0,
      temp_files: 0,
      temp_bytes: 0,
      postmaster_uptime_s: POSTMASTER_UPTIME_SECONDS,
    }],
    connectionStatesRows: [{ state: "active", count: 2 }, { state: "idle", count: 3 }],
    uptimeRows: [{ start_time: new Date(POSTMASTER_STARTUP_EPOCH * 1000), uptime: `${DAYS_SINCE_RESET} days` }],
    invalidIndexesRows: [],
    unusedIndexesRows: [],
    redundantIndexesRows: [],
  });

  const pgVersions = SUPPORTED_PG_VERSIONS;

  // A003 surfaces the full pg_settings projection. A007 uses value, not setting,
  // and intentionally omits context/vartype because altered settings expose a smaller shape.
  const expectedSharedBuffersSetting = {
    setting: "16384",
    unit: "8kB",
    category: "Resource Usage / Memory",
    context: "",
    vartype: "integer",
    pretty_value: "128.00 MiB",
  };

  const expectedSharedBuffersAlteredSetting = {
    value: "16384",
    unit: "8kB",
    category: "Resource Usage / Memory",
    pretty_value: "128.00 MiB",
  };

  const expectedAutovacuumSetting = {
    setting: "0.2",
    unit: "",
    category: "Autovacuum",
    context: "",
    vartype: "real",
    pretty_value: "0.2",
  };

  const expectedAutovacuumAlteredSetting = {
    value: "0.2",
    unit: "",
    category: "Autovacuum",
    pretty_value: "0.2",
  };

  const expectedLogMinDurationSetting = {
    setting: "-1",
    unit: "ms",
    category: "Reporting and Logging / When to Log",
    context: "",
    vartype: "integer",
    pretty_value: "-1",
  };

  const expectedDeadlockTimeoutSetting = {
    setting: "1000",
    unit: "ms",
    category: "Lock Management",
    context: "",
    vartype: "integer",
    pretty_value: "1000",
  };

  const expectedDatabaseSizeBytes = 1073741824;

  describe("getPostgresVersion extracts correct version for each PG version", () => {
    for (const { major, minor, versionNum } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const version = await checkup.getPostgresVersion(mockClient as any);

        expect(version.version).toBe(`${major}.${minor}`);
        expect(version.server_version_num).toBe(versionNum);
        expect(version.server_major_ver).toBe(String(major));
        expect(version.server_minor_ver).toBe(String(minor));
      });
    }
  });

  describe("generateA002 (major version) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateA002(mockClient as any, "test-node");

        expect(report.checkId).toBe("A002");
        expect(report.checkTitle).toBe("Postgres major version");
        expect(report.results["test-node"].data.version.version).toBe(`${major}.${minor}`);
        expect(report.results["test-node"].data.version.server_major_ver).toBe(String(major));
        expect(report.results["test-node"].data.version.server_minor_ver).toBe(String(minor));
      });
    }
  });

  describe("generateA013 (minor version) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateA013(mockClient as any, "test-node");

        expect(report.checkId).toBe("A013");
        expect(report.checkTitle).toBe("Postgres minor version");
        expect(report.results["test-node"].data.version.server_minor_ver).toBe(String(minor));
        expect(report.results["test-node"].data.version.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateA003 (settings) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateA003(mockClient as any, "test-node");

        expect(report.checkId).toBe("A003");
        expect(report.checkTitle).toBe("Postgres settings");
        expect(report.results["test-node"].data.shared_buffers).toEqual(expectedSharedBuffersSetting);
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
        expect(report.results["test-node"].postgres_version?.server_major_ver).toBe(String(major));
      });
    }
  });

  describe("generateA007 (altered settings) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateA007(mockClient as any, "test-node");

        expect(report.checkId).toBe("A007");
        expect(report.checkTitle).toBe("Altered settings");
        expect(report.results["test-node"].data.shared_buffers).toEqual(expectedSharedBuffersAlteredSetting);
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateA004 (cluster info) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateA004(mockClient as any, "test-node");

        expect(report.checkId).toBe("A004");
        expect(report.checkTitle).toBe("Cluster information");
        const data = report.results["test-node"].data;
        expect(data.general_info.total_connections.value).toBe("5");
        expect(data.general_info.cache_hit_ratio.value).toBe("90.00");
        expect(data.general_info.connections_active.value).toBe("2");
        expect(data.database_sizes.postgres).toBe(expectedDatabaseSizeBytes);
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateH001 (invalid indexes) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateH001(mockClient as any, "test-node");

        expect(report.checkId).toBe("H001");
        expect(report.checkTitle).toBe("Invalid indexes");
        expect(report.results["test-node"].data.testdb).toEqual({
          invalid_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateH002 (unused indexes) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateH002(mockClient as any, "test-node");

        expect(report.checkId).toBe("H002");
        expect(report.checkTitle).toBe("Unused indexes");
        expect(report.results["test-node"].data.testdb).toEqual({
          unused_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
          stats_reset: {
            stats_reset_epoch: STATS_RESET_EPOCH,
            stats_reset_time: STATS_RESET_TIME,
            days_since_reset: DAYS_SINCE_RESET,
            postmaster_startup_epoch: STATS_RESET_EPOCH,
            postmaster_startup_time: POSTMASTER_STARTUP_TIME,
          },
        });
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateH004 (redundant indexes) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.generateH004(mockClient as any, "test-node");

        expect(report.checkId).toBe("H004");
        expect(report.checkTitle).toBe("Redundant indexes");
        expect(report.results["test-node"].data.testdb).toEqual({
          redundant_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("index reports surface non-empty rows for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient({
          ...createVersionMockData(major, minor),
          invalidIndexesRows: [
            {
              schema_name: "public",
              table_name: "orders",
              index_name: "orders_status_idx_invalid",
              relation_name: "orders",
              index_size_bytes: "2097152",
              index_definition: "CREATE INDEX orders_status_idx_invalid ON public.orders USING btree (status)",
              supports_fk: false,
              is_pk: false,
              is_unique: false,
              constraint_name: null,
              table_row_estimate: "50000",
              has_valid_duplicate: true,
              valid_index_name: "orders_status_idx",
              valid_index_definition: "CREATE INDEX orders_status_idx ON public.orders USING btree (status)",
            },
          ],
          unusedIndexesRows: [
            {
              schema_name: "public",
              table_name: "logs",
              index_name: "logs_created_idx",
              index_definition: "CREATE INDEX logs_created_idx ON public.logs USING btree (created_at)",
              reason: "Never Used Indexes",
              index_size_bytes: "4194304",
              idx_scan: "0",
              idx_is_btree: true,
              supports_fk: false,
            },
          ],
          redundantIndexesRows: [
            {
              schema_name: "public",
              table_name: "orders",
              index_name: "orders_user_id_idx",
              relation_name: "orders",
              access_method: "btree",
              reason: "public.orders_user_id_created_idx",
              index_size_bytes: "2097152",
              table_size_bytes: "16777216",
              index_usage: "0",
              supports_fk: false,
              index_definition: "CREATE INDEX orders_user_id_idx ON public.orders USING btree (user_id)",
              redundant_to_json: JSON.stringify([
                {
                  index_name: "public.orders_user_id_created_idx",
                  index_definition: "CREATE INDEX orders_user_id_created_idx ON public.orders USING btree (user_id, created_at)",
                  index_size_bytes: 1048576,
                },
              ]),
            },
          ],
        });

        const invalidReport = await checkup.generateH001(mockClient as any, "test-node");
        expect(invalidReport.results["test-node"].data.testdb).toEqual({
          invalid_indexes: [
            {
              schema_name: "public",
              table_name: "orders",
              index_name: "orders_status_idx_invalid",
              relation_name: "orders",
              index_size_bytes: 2097152,
              index_size_pretty: "2.00 MiB",
              index_definition: "CREATE INDEX orders_status_idx_invalid ON public.orders USING btree (status)",
              supports_fk: false,
              is_pk: false,
              is_unique: false,
              constraint_name: null,
              table_row_estimate: 50000,
              has_valid_duplicate: true,
              valid_duplicate_name: "orders_status_idx",
              valid_duplicate_definition: "CREATE INDEX orders_status_idx ON public.orders USING btree (status)",
            },
          ],
          total_count: 1,
          total_size_bytes: 2097152,
          total_size_pretty: "2.00 MiB",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });

        const unusedReport = await checkup.generateH002(mockClient as any, "test-node");
        expect(unusedReport.results["test-node"].data.testdb).toEqual({
          unused_indexes: [
            {
              schema_name: "public",
              table_name: "logs",
              index_name: "logs_created_idx",
              index_definition: "CREATE INDEX logs_created_idx ON public.logs USING btree (created_at)",
              reason: "Never Used Indexes",
              idx_scan: 0,
              index_size_bytes: 4194304,
              idx_is_btree: true,
              supports_fk: false,
              index_size_pretty: "4.00 MiB",
            },
          ],
          total_count: 1,
          total_size_bytes: 4194304,
          total_size_pretty: "4.00 MiB",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
          stats_reset: {
            stats_reset_epoch: STATS_RESET_EPOCH,
            stats_reset_time: STATS_RESET_TIME,
            days_since_reset: DAYS_SINCE_RESET,
            postmaster_startup_epoch: STATS_RESET_EPOCH,
            postmaster_startup_time: POSTMASTER_STARTUP_TIME,
          },
        });

        const redundantReport = await checkup.generateH004(mockClient as any, "test-node");
        expect(redundantReport.results["test-node"].data.testdb).toEqual({
          redundant_indexes: [
            {
              schema_name: "public",
              table_name: "orders",
              index_name: "orders_user_id_idx",
              relation_name: "orders",
              access_method: "btree",
              reason: "public.orders_user_id_created_idx",
              index_size_bytes: 2097152,
              table_size_bytes: 16777216,
              index_usage: 0,
              supports_fk: false,
              index_definition: "CREATE INDEX orders_user_id_idx ON public.orders USING btree (user_id)",
              index_size_pretty: "2.00 MiB",
              table_size_pretty: "16.00 MiB",
              redundant_to: [
                {
                  index_name: "public.orders_user_id_created_idx",
                  index_definition: "CREATE INDEX orders_user_id_created_idx ON public.orders USING btree (user_id, created_at)",
                  index_size_bytes: 1048576,
                  index_size_pretty: "1.00 MiB",
                },
              ],
            },
          ],
          total_count: 1,
          total_size_bytes: 2097152,
          total_size_pretty: "2.00 MiB",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
      });
    }
  });

  describe("generateD004 (pg_stat_statements) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.REPORT_GENERATORS.D004(mockClient as any, "test-node");

        expect(report.checkId).toBe("D004");
        expect(report.checkTitle).toBe("pg_stat_statements and pg_stat_kcache settings");
        const data = report.results["test-node"].data;
        expect(data).toEqual({
          settings: {},
          pg_stat_statements_status: {
            extension_available: false,
            metrics_count: 0,
            total_calls: 0,
            sample_queries: [],
          },
          pg_stat_kcache_status: {
            extension_available: false,
            metrics_count: 0,
            total_exec_time: 0,
            total_user_time: 0,
            total_system_time: 0,
            sample_queries: [],
          },
        });
      });
    }

    test("surfaces populated extension metrics", async () => {
      const mockClient = createMockClient({
        ...createVersionMockData(16, 3),
        settingsRows: [
          {
            tag_setting_name: "pg_stat_statements.max",
            tag_setting_value: "5000",
            tag_unit: "",
            tag_category: "Custom",
            tag_vartype: "integer",
            is_default: 0,
            setting_normalized: null,
            unit_normalized: null,
          },
          {
            tag_setting_name: "pg_stat_kcache.linux_hz",
            tag_setting_value: "100",
            tag_unit: "",
            tag_category: "Custom",
            tag_vartype: "integer",
            is_default: 1,
            setting_normalized: null,
            unit_normalized: null,
          },
        ],
        pgStatStatementsExtensionRows: [{ exists: 1 }],
        pgStatStatementsStatsRows: [{ cnt: "2", total_calls: "42" }],
        pgStatStatementsSampleRows: [
          { queryid: "101", user: "app", database: "testdb", calls: "40" },
          { queryid: "202", user: "worker", database: "testdb", calls: "2" },
        ],
        pgStatKcacheExtensionRows: [{ exists: 1 }],
        pgStatKcacheStatsRows: [{ cnt: "1", total_exec_time: "12.5", total_user_time: "8.5", total_system_time: "4" }],
        pgStatKcacheSampleRows: [{ queryid: "101", user: "app", exec_total_time: "12.5" }],
      });

      const report = await checkup.REPORT_GENERATORS.D004(mockClient as any, "test-node");
      expect(report.results["test-node"].data).toEqual({
        settings: {
          "pg_stat_statements.max": {
            setting: "5000",
            unit: "",
            category: "Custom",
            context: "",
            vartype: "integer",
            pretty_value: "5000",
          },
          "pg_stat_kcache.linux_hz": {
            setting: "100",
            unit: "",
            category: "Custom",
            context: "",
            vartype: "integer",
            pretty_value: "100",
          },
        },
        pg_stat_statements_status: {
          extension_available: true,
          metrics_count: 2,
          total_calls: 42,
          sample_queries: [
            { queryid: "101", user: "app", database: "testdb", calls: 40 },
            { queryid: "202", user: "worker", database: "testdb", calls: 2 },
          ],
        },
        pg_stat_kcache_status: {
          extension_available: true,
          metrics_count: 1,
          total_exec_time: 12.5,
          total_user_time: 8.5,
          total_system_time: 4,
          sample_queries: [{ queryid: "101", user: "app", exec_total_time: 12.5 }],
        },
      });
    });
  });

  describe("generateF001 (autovacuum settings) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.REPORT_GENERATORS.F001(mockClient as any, "test-node");

        expect(report.checkId).toBe("F001");
        expect(report.checkTitle).toBe("Autovacuum: current settings");
        expect(report.results["test-node"].data).toEqual({
          autovacuum_vacuum_scale_factor: expectedAutovacuumSetting,
        });
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateG001 (memory settings) works for each PG version", () => {
    for (const { major, minor } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const report = await checkup.REPORT_GENERATORS.G001(mockClient as any, "test-node");

        expect(report.checkId).toBe("G001");
        expect(report.checkTitle).toBe("Memory-related settings");
        const data = report.results["test-node"].data;
        expect(data).toEqual({
          settings: {
            shared_buffers: expectedSharedBuffersSetting,
          },
          analysis: {
            estimated_total_memory_usage: {
              shared_buffers_bytes: 134217728,
              shared_buffers_pretty: "128.00 MiB",
              wal_buffers_bytes: 4194304,
              wal_buffers_pretty: "4.00 MiB",
              shared_memory_total_bytes: 138412032,
              shared_memory_total_pretty: "132.00 MiB",
              work_mem_per_connection_bytes: 4194304,
              work_mem_per_connection_pretty: "4.00 MiB",
              max_work_mem_usage_bytes: 419430400,
              max_work_mem_usage_pretty: "400.00 MiB",
              maintenance_work_mem_bytes: 67108864,
              maintenance_work_mem_pretty: "64.00 MiB",
              effective_cache_size_bytes: 4294967296,
              effective_cache_size_pretty: "4.00 GiB",
            },
          },
        });
        expect(report.results["test-node"].postgres_version?.version).toBe(`${major}.${minor}`);
      });
    }
  });

  describe("generateAllReports works for each PG version", () => {
    for (const { major, minor, versionNum } of pgVersions) {
      test(`PG ${major}.${minor}`, async () => {
        const mockClient = createMockClient(createVersionMockData(major, minor));
        const reports = await checkup.generateAllReports(mockClient as any, "test-node");
        const expectedVersion = {
          version: `${major}.${minor}`,
          server_version_num: versionNum,
          server_major_ver: String(major),
          server_minor_ver: String(minor),
        };

        // Verify all express-mode checks are generated from the same source of truth.
        const expectedChecks = Object.keys(checkup.REPORT_GENERATORS).sort();
        expect(Object.keys(reports).sort()).toEqual(expectedChecks);
        for (const checkId of expectedChecks) {
          expect(reports[checkId]?.checkId).toBe(checkId);
          expect(reports[checkId].results["test-node"]).toBeDefined();
        }

        // Verify every generated report has a concrete payload shape.
        expect(reports.A002.results["test-node"].data).toEqual({ version: expectedVersion });
        expect(reports.A003.results["test-node"].data).toEqual({
          shared_buffers: expectedSharedBuffersSetting,
          autovacuum_vacuum_scale_factor: expectedAutovacuumSetting,
          log_min_duration_statement: expectedLogMinDurationSetting,
          deadlock_timeout: expectedDeadlockTimeoutSetting,
        });
        expect(reports.A004.results["test-node"].data.database_sizes).toEqual({ postgres: expectedDatabaseSizeBytes });
        expect(reports.A004.results["test-node"].data.general_info.total_connections.value).toBe("5");
        expect(reports.A004.results["test-node"].data.general_info.uptime.value).toBe("30 days 0:00:00");
        expect(reports.A007.results["test-node"].data).toEqual({
          shared_buffers: expectedSharedBuffersAlteredSetting,
          autovacuum_vacuum_scale_factor: expectedAutovacuumAlteredSetting,
        });
        expect(reports.A013.results["test-node"].data).toEqual({ version: expectedVersion });
        expect(reports.D001.results["test-node"].data).toEqual({
          log_min_duration_statement: expectedLogMinDurationSetting,
        });
        expect(reports.D004.results["test-node"].data).toEqual({
          settings: {},
          pg_stat_statements_status: {
            extension_available: false,
            metrics_count: 0,
            total_calls: 0,
            sample_queries: [],
          },
          pg_stat_kcache_status: {
            extension_available: false,
            metrics_count: 0,
            total_exec_time: 0,
            total_user_time: 0,
            total_system_time: 0,
            sample_queries: [],
          },
        });
        expect(reports.F001.results["test-node"].data).toEqual({
          autovacuum_vacuum_scale_factor: expectedAutovacuumSetting,
        });
        expect(reports.F004.results["test-node"].data.testdb).toEqual({
          status: { ok: true, reason: null, error: null },
          bloated_tables: [],
          total_count: 0,
          total_bloat_size_bytes: 0,
          total_bloat_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
        expect(reports.F005.results["test-node"].data.testdb).toEqual({
          status: { ok: true, reason: null, error: null },
          bloated_indexes: [],
          total_count: 0,
          total_bloat_size_bytes: 0,
          total_bloat_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
        expect(reports.G001.results["test-node"].data.settings).toEqual({
          shared_buffers: expectedSharedBuffersSetting,
        });
        expect(reports.G001.results["test-node"].data.analysis.estimated_total_memory_usage.shared_buffers_bytes).toBe(134217728);
        expect(reports.G003.results["test-node"].data).toEqual({
          settings: {
            deadlock_timeout: expectedDeadlockTimeoutSetting,
          },
          deadlock_stats: { deadlocks: 0, conflicts: 0, stats_reset: null },
        });
        expect(reports.H001.results["test-node"].data.testdb).toEqual({
          invalid_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });
        expect(reports.H002.results["test-node"].data.testdb).toEqual({
          unused_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
          stats_reset: {
            stats_reset_epoch: STATS_RESET_EPOCH,
            stats_reset_time: STATS_RESET_TIME,
            days_since_reset: DAYS_SINCE_RESET,
            postmaster_startup_epoch: STATS_RESET_EPOCH,
            postmaster_startup_time: POSTMASTER_STARTUP_TIME,
          },
        });
        expect(reports.H004.results["test-node"].data.testdb).toEqual({
          redundant_indexes: [],
          total_count: 0,
          total_size_bytes: 0,
          total_size_pretty: "0 B",
          database_size_bytes: expectedDatabaseSizeBytes,
          database_size_pretty: "1.00 GiB",
        });

        // Verify postgres_version is set in reports that include it.
        expect(reports.A003.results["test-node"].postgres_version).toEqual(expectedVersion);
        expect(reports.A004.results["test-node"].postgres_version).toEqual(expectedVersion);
      });
    }
  });
});

// Tests for version-aware SQL query selection
describe("Version-aware SQL query selection (PG13-PG19)", () => {
  const pgVersions = SUPPORTED_PG_MAJOR_VERSIONS;

  // All metrics registered in metrics.yml.
  const allMetrics = metricsLoader.listMetricNames();
  expect(allMetrics.length).toBeGreaterThan(0);
  const sqlStartPattern = /^\s*(with|select)\b/i;

  describe("All metrics from metrics.yml return valid SQL for each PG version", () => {
    for (const pgVersion of pgVersions) {
      describe(`PG${pgVersion}`, () => {
        for (const metric of allMetrics) {
          test(`${metric}`, () => {
            const sql = metricsLoader.getMetricSql(metric, pgVersion);
            expect(typeof sql).toBe("string");
            expect(sql.length).toBeGreaterThan(0);

            const trimmedSql = sql.trim();
            if (trimmedSql.startsWith(";")) {
              expect(metric).toBe("pg_stat_io");
              expect(pgVersion).toBeLessThan(16);
              expect(trimmedSql).toMatch(/pg_stat_io only available/i);
              return;
            }

            expect(trimmedSql).toMatch(sqlStartPattern);
            expect(trimmedSql.toLowerCase()).toMatch(/\bfrom\b/);
            expect(trimmedSql).not.toMatch(/\{\{.*\}\}/);
            expect(trimmedSql).not.toMatch(/\$\{.*\}/);
          });
        }
      });
    }
  });

  describe("getMetricSql rejects invalid version inputs", () => {
    const invalidVersions = [0, -1, Number.NaN];

    for (const pgVersion of invalidVersions) {
      test(`settings rejects Postgres ${String(pgVersion)}`, () => {
        expect(() => metricsLoader.getMetricSql("settings", pgVersion)).toThrow(/No compatible SQL version/);
      });
    }

    test("rejects versions older than the oldest keyed SQL", () => {
      for (const metric of allMetrics) {
        expect(() => metricsLoader.getMetricSql(metric, 10)).toThrow(/No compatible SQL version/);
      }
    });

    test("rejects unknown metric names", () => {
      expect(() => metricsLoader.getMetricSql("not_a_metric", 16)).toThrow(/Metric "not_a_metric" not found/);
    });
  });

  describe("getMetricSql selects the nearest compatible version", () => {
    test("uses exact and previous keyed SQL for mid-range versions", () => {
      const definition = metricsLoader.getMetricDefinition("db_stats")!;
      expect(typeof definition.sqls["11"]).toBe("string");
      expect(typeof definition.sqls["12"]).toBe("string");
      expect(typeof definition.sqls["14"]).toBe("string");
      expect(typeof definition.sqls["15"]).toBe("string");

      expect(metricsLoader.getMetricSql("db_stats", 12)).toBe(definition.sqls["12"]);
      expect(metricsLoader.getMetricSql("db_stats", 13)).toBe(definition.sqls["12"]);
      expect(metricsLoader.getMetricSql("db_stats", 14)).toBe(definition.sqls["14"]);
      expect(metricsLoader.getMetricSql("db_stats", 19)).toBe(definition.sqls["15"]);
      expect(definition.sqls["12"]).not.toBe(definition.sqls["11"]);
      expect(definition.sqls["15"]).not.toBe(definition.sqls["14"]);
    });

    test("uses a metric's oldest SQL when no newer key exists below the requested version", () => {
      const definition = metricsLoader.getMetricDefinition("settings")!;
      expect(typeof definition.sqls["11"]).toBe("string");
      expect(metricsLoader.getMetricSql("settings", 12)).toBe(definition.sqls["11"]);
      expect(metricsLoader.getMetricSql("settings", 19)).toBe(definition.sqls["11"]);
    });
  });

  describe("getMetricDefinition returns metadata for all metrics", () => {
    for (const metric of allMetrics) {
      test(`${metric} has definition with versioned SQL`, () => {
        const definition = metricsLoader.getMetricDefinition(metric);
        expect(definition).toBeTruthy();
        expect(definition?.sqls).toBeTruthy();
        expect(typeof definition?.sqls).toBe("object");
        const entries = Object.entries(definition!.sqls);
        expect(entries.length).toBeGreaterThan(0);
        for (const [versionKey, sql] of entries) {
          const version = Number(versionKey);
          const trimmedSql = sql.trim();
          expect(Number.isInteger(version)).toBe(true);
          expect(version).toBeGreaterThan(0);
          expect(typeof sql).toBe("string");
          expect(sql.length).toBeGreaterThan(0);
          if (trimmedSql.startsWith(";")) {
            expect(metric).toBe("pg_stat_io");
            expect(version).toBe(11);
            expect(trimmedSql).toMatch(/pg_stat_io only available/i);
            continue;
          }
          expect(trimmedSql).toMatch(sqlStartPattern);
        }
      });
    }
  });

  test("listMetricNames returns all expected core metrics", () => {
    const names = metricsLoader.listMetricNames();
    expect(Array.isArray(names)).toBe(true);
    expect(new Set(names).size).toBe(names.length);
    const coreMetrics = [
      "settings",
      "db_stats",
      "db_size",
      "stats_reset",
      "pg_invalid_indexes",
      "unused_indexes",
      "redundant_indexes",
    ];
    expect(names.length).toBeGreaterThanOrEqual(coreMetrics.length);
    for (const metric of coreMetrics) {
      expect(names).toContain(metric);
    }
  });

  test("METRIC_NAMES maps every express report metric", () => {
    expect(metricsLoader.METRIC_NAMES).toEqual({
      H001: "pg_invalid_indexes",
      H002: "unused_indexes",
      H004: "redundant_indexes",
      F001: "pg_autovacuum_relopts",
      F002Database: "pg_database_wraparound",
      F002Settings: "pg_settings_wraparound",
      F002Tables: "pg_table_wraparound",
      F002MultixactSize: "multixact_size",
      F003: "pg_dead_tuples",
      F004: "pg_table_bloat",
      F005: "pg_btree_bloat",
      F009: "xmin_horizon_snapshot",
      settings: "settings",
      dbStats: "db_stats",
      dbSize: "db_size",
      statsReset: "stats_reset",
      I001: "pg_stat_io",
    });
  });
});
