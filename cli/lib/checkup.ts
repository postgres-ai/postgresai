/**
 * Express Checkup Module
 * ======================
 * Generates JSON health check reports directly from PostgreSQL without Prometheus.
 *
 * ARCHITECTURAL DECISIONS
 * -----------------------
 *
 * 1. SINGLE SOURCE OF TRUTH FOR SQL QUERIES
 *    Complex metrics (index health, settings, db_stats) are loaded from
 *    config/pgwatch-prometheus/metrics.yml via getMetricSql() from metrics-loader.ts.
 *
 *    Simple queries (version, database list, connection states, uptime) use
 *    inline SQL as they're trivial and CLI-specific.
 *
 * 2. JSON SCHEMA COMPLIANCE
 *    All generated reports MUST comply with JSON schemas in reporter/schemas/.
 *    These schemas define the expected format for both:
 *    - Full-fledged monitoring reporter output
 *    - Express checkup output
 *
 *    Before adding or modifying a report, verify the corresponding schema exists
 *    and ensure the output matches. Every new or updated check must also emit
 *    local `conclusions` and `recommendations` in its JSON; server-side markdown
 *    may enrich those verdicts, but must not be their only source. Run schema
 *    validation tests to confirm.
 *
 * 3. ERROR HANDLING STRATEGY
 *    Functions follow two patterns based on criticality:
 *
 *    PROPAGATING (throws on error):
 *    - Core data functions: getPostgresVersion, getSettings, getAlteredSettings,
 *      getDatabaseSizes, getInvalidIndexes, getUnusedIndexes, getRedundantIndexes
 *    - If these fail, the entire report should fail (data is required)
 *    - Callers should handle errors at the report generation level
 *
 *    GRACEFUL DEGRADATION (catches errors, includes error in output):
 *    - Optional/supplementary queries: pg_stat_statements, pg_stat_kcache checks,
 *      memory calculations, postmaster startup time
 *    - These are nice-to-have; missing data shouldn't fail the whole report
 *    - Errors are logged and included in report output for visibility
 *
 * ADDING NEW REPORTS
 * ------------------
 * 1. Add/verify the metric exists in config/pgwatch-prometheus/metrics.yml
 * 2. Add the metric name mapping to METRIC_NAMES in metrics-loader.ts
 * 3. Verify JSON schema exists in reporter/schemas/{CHECK_ID}.schema.json
 * 4. Implement the generator function using getMetricSql()
 * 5. Emit local conclusions and recommendations from the generator
 * 6. Add schema validation test in test/schema-validation.test.ts
 */

import { Client } from "pg";
import * as fs from "fs";
import * as path from "path";
import * as pkg from "../package.json";
import { getMetricSql, transformMetricRow, METRIC_NAMES } from "./metrics-loader";
import { buildCheckInfoMap } from "./checkup-dictionary";
import { generateCheckSummary, CheckSummary } from "./checkup-summary";

/**
 * Version of the checkup JSON report contract (the report envelope + the
 * per-check JSON schemas in reporter/schemas/, plus the `checkup --no-upload
 * --json` stdout/stderr/exit-code ABI).
 *
 * This is the versioned public surface that host applications embedding express
 * checkup depend on. It is intentionally independent of the CLI/package version
 * (`version` in the envelope): the CLI can be released many times without the
 * contract changing.
 *
 * COMPATIBILITY POLICY (semver applied to the contract, not the code):
 *   - PATCH (x.y.Z): editorial/no-op changes that cannot affect a consumer
 *     (documentation, clarifications).
 *   - MINOR (x.Y.0): ADDITIVE, backward-compatible changes — new optional
 *     fields in the envelope or a report, new checks, new schema files. Existing
 *     valid reports stay valid; existing consumers keep working untouched.
 *   - MAJOR (X.0.0): BREAKING changes — removing/renaming a field, tightening a
 *     type, making an optional field required, or changing the CLI JSON ABI in a
 *     way that could break a consumer parsing the previous format.
 *
 * A consumer should accept any report whose contract_version has the same MAJOR
 * and a MINOR >= the minimum it was built against.
 *
 * The Python reporter (reporter/postgres_reports.py) emits the SAME value; a
 * cross-language test (cli/test/contract-version.test.ts) asserts the two
 * sources cannot drift.
 */
export const CONTRACT_VERSION = "1.0.0";

// Time constants
const SECONDS_PER_DAY = 86400;
const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;

/**
 * Convert various boolean representations to boolean.
 * PostgreSQL returns booleans as true/false, 1/0, 't'/'f', or 'true'/'false'
 * depending on context (query result, JDBC driver, etc.).
 */
function toBool(val: unknown): boolean {
  return val === true || val === 1 || val === "t" || val === "true";
}

/**
 * PostgreSQL version information
 */
export interface PostgresVersion {
  version: string;
  server_version_num: string;
  server_major_ver: string;
  server_minor_ver: string;
}

/**
 * Setting information from pg_settings
 */
export interface SettingInfo {
  setting: string;
  unit: string;
  category: string;
  context: string;
  vartype: string;
  pretty_value: string;
}

/**
 * Altered setting (A007) - subset of SettingInfo
 */
export interface AlteredSetting {
  value: string;
  unit: string;
  category: string;
  pretty_value: string;
}

/**
 * Cluster metric (A004)
 */
export interface ClusterMetric {
  value: string;
  unit: string;
  description: string;
}

/**
 * Invalid index entry (H001) - matches H001.schema.json invalidIndex
 *
 * Decision tree for remediation recommendations:
 * 1. has_valid_duplicate=true → DROP (valid duplicate exists, safe to remove)
 * 2. is_pk=true or is_unique=true → RECREATE (backs a constraint, must restore)
 * 3. table_row_estimate < 10000 → RECREATE (small table, quick rebuild)
 * 4. Otherwise → UNCERTAIN (needs manual analysis of query plans)
 */
export interface InvalidIndex {
  schema_name: string;
  table_name: string;
  index_name: string;
  relation_name: string;
  index_size_bytes: number;
  index_size_pretty: string;
  /** Full CREATE INDEX statement from pg_get_indexdef() - useful for DROP/RECREATE migrations */
  index_definition: string;
  supports_fk: boolean;
  /** True if this index backs a PRIMARY KEY constraint */
  is_pk: boolean;
  /** True if this is a UNIQUE index (includes PK indexes) */
  is_unique: boolean;
  /** Name of the constraint this index backs, or null if none */
  constraint_name: string | null;
  /** Estimated row count of the table from pg_class.reltuples */
  table_row_estimate: number;
  /** True if there is a valid index on the same column(s) */
  has_valid_duplicate: boolean;
  /** Name of the valid duplicate index if one exists */
  valid_duplicate_name: string | null;
  /** Full CREATE INDEX statement of the valid duplicate index */
  valid_duplicate_definition: string | null;
}

/** Recommendation for handling an invalid index */
export type InvalidIndexRecommendation = "DROP" | "RECREATE" | "UNCERTAIN";

/** Threshold for considering a table "small" (quick to rebuild) */
const SMALL_TABLE_ROW_THRESHOLD = 10000;

/**
 * Compute remediation recommendation for an invalid index using decision tree.
 *
 * Decision tree logic:
 * 1. If has_valid_duplicate is true → DROP (valid duplicate exists, safe to remove)
 * 2. If is_pk or is_unique is true → RECREATE (backs a constraint, must restore)
 * 3. If table_row_estimate < 10000 → RECREATE (small table, quick rebuild)
 * 4. Otherwise → UNCERTAIN (needs manual analysis of query plans)
 *
 * @param index - Invalid index with observation data
 * @returns Recommendation: "DROP", "RECREATE", or "UNCERTAIN"
 */
export function getInvalidIndexRecommendation(index: InvalidIndex): InvalidIndexRecommendation {
  // 1. Valid duplicate exists - safe to drop
  if (index.has_valid_duplicate) {
    return "DROP";
  }

  // 2. Backs a constraint - must recreate
  if (index.is_pk || index.is_unique) {
    return "RECREATE";
  }

  // 3. Small table - quick to recreate
  if (index.table_row_estimate < SMALL_TABLE_ROW_THRESHOLD) {
    return "RECREATE";
  }

  // 4. Large table without clear path - needs manual analysis
  return "UNCERTAIN";
}

/**
 * Unused index entry (H002) - matches H002.schema.json unusedIndex
 */
export interface UnusedIndex {
  schema_name: string;
  table_name: string;
  index_name: string;
  index_definition: string;
  reason: string;
  idx_scan: number;
  index_size_bytes: number;
  idx_is_btree: boolean;
  supports_fk: boolean;
  index_size_pretty: string;
}

/**
 * Stats reset info for H002 - matches H002.schema.json statsReset
 */
export interface StatsReset {
  stats_reset_epoch: number | null;
  stats_reset_time: string | null;
  days_since_reset: number | null;
  postmaster_startup_epoch: number | null;
  postmaster_startup_time: string | null;
  /** Set when postmaster startup time query fails - indicates data availability issue */
  postmaster_startup_error?: string;
}

/**
 * Redundant index entry (H004) - matches H004.schema.json redundantIndex
 */
/**
 * Index that makes another index redundant.
 * Used in redundant_to array to show which indexes this one is redundant to.
 */
export interface RedundantToIndex {
  index_name: string;
  index_definition: string;
  index_size_bytes: number;
  index_size_pretty: string;
}

export interface RedundantIndex {
  schema_name: string;
  table_name: string;
  index_name: string;
  relation_name: string;
  access_method: string;
  reason: string;
  index_size_bytes: number;
  table_size_bytes: number;
  index_usage: number;
  supports_fk: boolean;
  index_definition: string;
  index_size_pretty: string;
  table_size_pretty: string;
  redundant_to: RedundantToIndex[];
  /** Set when redundant_to_json parsing fails - indicates data quality issue */
  redundant_to_parse_error?: string;
}

/**
 * Dead tuples table entry (F003) - matches F003.schema.json deadTuplesTable
 *
 * Sourced from pg_stat_user_tables live counters (n_dead_tup / n_live_tup),
 * so dead tuples that have never been vacuumed are visible - unlike the
 * statistical bloat estimators (F004/F005), which miss them entirely.
 */
export interface DeadTuplesTable {
  schema_name: string;
  table_name: string;
  n_live_tup: number;
  n_dead_tup: number;
  /** Dead tuples as percentage of all tuples: n_dead_tup / (n_live_tup + n_dead_tup) * 100 */
  dead_pct: number;
  last_autovacuum: string | null;
  /** Epoch seconds of the last autovacuum; 0 = never */
  last_autovacuum_epoch: number;
  last_vacuum: string | null;
  /** Epoch seconds of the last manual vacuum; 0 = never */
  last_vacuum_epoch: number;
  autovacuum_count: number;
  vacuum_count: number;
  /** True when autovacuum is disabled per-table via reloptions (autovacuum_enabled=off/false/0/...) */
  autovacuum_disabled: boolean;
  table_size_bytes: number;
  table_size_pretty: string;
  /** True when BOTH F003_DEAD_TUPLES_MIN and F003_DEAD_PCT_MIN thresholds are exceeded */
  exceeds_dead_tuple_thresholds: boolean;
  /** True when autovacuum is disabled per-table on a non-tiny table (>= F003_AUTOVACUUM_DISABLED_MIN_ROWS tuples) */
  autovacuum_disabled_flagged: boolean;

  // --- WI #271: settings-aware trigger analysis. Optional/additive: present
  //     only when the table comes from getAutovacuumKeepup(); the legacy
  //     getDeadTuples() path leaves them undefined. ---
  /** pg_class.reltuples estimate used as the base for the trigger formulas (clamped to >= 0). */
  reltuples?: number;
  /** pg_class.relpages — catalog-only size proxy used for two-stage ranking. */
  relpages?: number;
  /** Modified tuples since the last analyze (drives the analyze trigger). */
  n_mod_since_analyze?: number;
  /** Inserted tuples since the last vacuum (drives the PG13+ insert trigger); null on PG12. */
  n_ins_since_vacuum?: number | null;
  last_autoanalyze?: string | null;
  last_autoanalyze_epoch?: number;
  /** Effective autovacuum_vacuum_threshold (reloption override or global GUC). */
  eff_vacuum_threshold?: number;
  /** Effective autovacuum_vacuum_scale_factor (reloption override or global GUC). */
  eff_vacuum_scale_factor?: number;
  /** True when the effective vacuum threshold/scale factor came from pg_class.reloptions. */
  vacuum_settings_from_reloptions?: boolean;
  eff_analyze_threshold?: number;
  eff_analyze_scale_factor?: number;
  /** Effective insert threshold/scale factor (PG13+); null on PG12. */
  eff_insert_threshold?: number | null;
  eff_insert_scale_factor?: number | null;
  insert_settings_from_reloptions?: boolean;
  /** vacuum_threshold + vacuum_scale_factor * reltuples. */
  vacuum_trigger_point?: number;
  analyze_trigger_point?: number;
  insert_trigger_point?: number | null;
  /** n_dead_tup / vacuum_trigger_point (1.0 = exactly at trigger; >=2 = long overdue). */
  over_trigger_ratio?: number;
  over_vacuum_trigger?: boolean;
  over_analyze_trigger?: boolean;
  over_insert_trigger?: boolean;
  /** True when autovacuum is disabled on the table's TOAST relation via toast.autovacuum_enabled. */
  toast_autovacuum_disabled?: boolean;
  /** Over vacuum trigger, last autovacuum older than F003_STARVATION_HOURS (or never), and no worker processing it now. */
  starving?: boolean;
}

/** Aggregate coverage counters from the pg_dead_tuples_keepup single-scan (WI #271). */
export interface AutovacuumKeepupAggregates {
  /** Total user relations seen in the single pg_stat_user_tables pass. */
  relations_total: number;
  /** Relations passing the noise gate (>= F003_KEEPUP_MIN_ROWS tuples). */
  candidates_considered: number;
  /** Tables past their vacuum trigger and above the noise gate = the autovacuum queue. */
  queue_length: number;
  /** Tables past their analyze trigger and above the noise gate. */
  analyze_queue_length: number;
  /** Tables past their insert trigger (PG13+); null on PG12. */
  insert_queue_length: number | null;
  /** Sum of n_dead_tup across all relations in the scan. */
  total_dead_tuples_all: number;
}

/** Single-snapshot autovacuum worker capacity vs demand (WI #271). */
export interface AutovacuumWorkerSnapshot {
  active_workers: number;
  /** autovacuum_max_workers; null if the setting could not be read. */
  max_workers: number | null;
  free_slots: number | null;
  /** Workers running "to prevent wraparound" (visible only to privileged roles; 0 otherwise). */
  anti_wraparound_workers: number;
}

/** An autovacuum worker blocked on a lock, with blocker info (WI #271). */
export interface BlockedAutovacuumWorker {
  worker_pid: number;
  blocker_pid: number | null;
  blocker_queryid: string | null;
  wait_seconds: number;
}

/** A running vacuum from pg_stat_progress_vacuum (WI #271). */
export interface VacuumProgressEntry {
  schema_name: string;
  table_name: string;
  /** autovacuum | aggressive_autovacuum | manual_vacuum | unknown */
  vacuum_mode: string;
  /** Human-readable phase name. */
  phase: string;
  /** Numeric phase code (1..7); null if unknown. */
  phase_code: number | null;
  heap_blks_total: number;
  heap_blks_scanned: number;
  heap_blks_vacuumed: number;
  index_vacuum_count: number;
  is_anti_wraparound: boolean;
}

/** Assembled keeping-up snapshot + judgment for the F003 report (WI #271). */
export interface AutovacuumKeepup extends AutovacuumKeepupAggregates {
  active_workers: number;
  max_workers: number | null;
  free_slots: number | null;
  anti_wraparound_workers: number;
  anti_wraparound_present: boolean;
  /** queue > 0 while every worker is busy: autovacuum cannot currently keep up. */
  saturated: boolean;
  /** queue > F003_QUEUE_SATURATION_MULTIPLIER * max_workers: chronic under-provisioning. */
  chronic_under_provisioning: boolean;
  starving_tables_count: number;
  blocked_workers: BlockedAutovacuumWorker[];
  vacuum_progress: VacuumProgressEntry[];
  judgment: string;
  status: "ok" | "warning" | "critical";
}

export type WraparoundSeverity = "info" | "warning" | "high" | "critical";

export interface WraparoundSettings {
  autovacuum_freeze_max_age: number;
  vacuum_freeze_min_age: number;
  vacuum_freeze_table_age: number;
  autovacuum_multixact_freeze_max_age: number;
  vacuum_multixact_freeze_min_age: number;
  vacuum_multixact_freeze_table_age: number;
  vacuum_failsafe_age: number | null;
  vacuum_multixact_failsafe_age: number | null;
}

export interface WraparoundRisk {
  age: number;
  emergency_age: number;
  failsafe_age: number | null;
  pct_towards_wraparound: number;
  pct_towards_emergency: number;
  pct_towards_failsafe: number | null;
  severity: WraparoundSeverity;
}

export interface WraparoundDatabase {
  database_name: string;
  xid: WraparoundRisk;
  multixact: WraparoundRisk;
}

export interface WraparoundTable {
  database_name: string;
  schema_name: string;
  table_name: string;
  ranked_by: string[];
  table_size_bytes: number;
  table_size_pretty: string;
  xid: WraparoundRisk;
  multixact: WraparoundRisk;
}

export interface MultixactSize {
  bytes: number | null;
  size_pretty: string | null;
  status_code: number;
}

/**
 * F002 transaction ID / MultiXact wraparound severity policy.
 *
 * Kept in sync with the full-mode implementation in
 * reporter/postgres_reports.py (same-named constants and `_wraparound_risk`).
 * The severity ladder:
 * - info (guard): emergencyAge <= 0. The emergency threshold is unknown
 *   (settings unavailable — a realistic full-mode monitoring gap where the
 *   pg_settings_wraparound series are missing and default to 0). Without this
 *   guard every `age >= 2*0`/`age >= 0` comparison is trivially true and a
 *   monitoring gap turns into a false-positive "everything is high" storm, so
 *   the risk is reported as info instead of being classified against a bogus 0.
 * - critical: age >= F002_CRITICAL_AGE (half of the 2^31 wrap limit).
 * - high: age >= 2 * emergencyAge, OR age >= F002_FAILSAFE_HIGH_PCT% of the
 *   failsafe age (PG14+). Twice the soft emergency threshold is where
 *   anti-wraparound autovacuum should already be running yet isn't keeping up.
 * - warning: age >= emergencyAge (the per-table/effective
 *   autovacuum_freeze_max_age at which anti-wraparound vacuum starts).
 */
export const F002_WRAPAROUND_LIMIT = 2_147_483_648;
export const F002_CRITICAL_AGE = 1_000_000_000;
export const F002_FAILSAFE_HIGH_PCT = 80;

const severityRank: Record<WraparoundSeverity, number> = {
  info: 0,
  warning: 1,
  high: 2,
  critical: 3,
};

export function evaluateWraparoundRisk(
  age: number,
  emergencyAge: number,
  failsafeAge: number | null,
): WraparoundRisk {
  let severity: WraparoundSeverity = "info";
  if (emergencyAge <= 0) {
    severity = "info";
  } else if (age >= F002_CRITICAL_AGE) {
    severity = "critical";
  } else if (
    age >= 2 * emergencyAge ||
    (failsafeAge !== null && age >= failsafeAge * (F002_FAILSAFE_HIGH_PCT / 100))
  ) {
    severity = "high";
  } else if (age >= emergencyAge) {
    severity = "warning";
  }

  const pct = (value: number, limit: number): number =>
    limit > 0 ? Math.round((value / limit) * 10_000) / 100 : 0;

  return {
    age,
    emergency_age: emergencyAge,
    failsafe_age: failsafeAge,
    pct_towards_wraparound: pct(age, F002_WRAPAROUND_LIMIT),
    pct_towards_emergency: pct(age, emergencyAge),
    pct_towards_failsafe: failsafeAge === null ? null : pct(age, failsafeAge),
    severity,
  };
}

function maxSeverity(...severities: WraparoundSeverity[]): WraparoundSeverity {
  return severities.reduce((max, value) => severityRank[value] > severityRank[max] ? value : max, "info");
}

/**
 * F003 thresholds.
 *
 * A table's dead-tuple accumulation is flagged only when it is high in BOTH
 * absolute and relative terms:
 * - F003_DEAD_TUPLES_MIN keeps small/noisy tables out (100k dead tuples is
 *   real work for vacuum regardless of table size);
 * - F003_DEAD_PCT_MIN = 20 mirrors the default autovacuum_vacuum_scale_factor
 *   of 0.2: with default settings autovacuum should have fired well before a
 *   table is 20% dead, so reaching this level in a snapshot is an unambiguous
 *   signal that vacuum is not keeping up (lagging, blocked, or disabled).
 *
 * Per-table disabled autovacuum is a classic footgun and is always flagged on
 * non-tiny tables (>= F003_AUTOVACUUM_DISABLED_MIN_ROWS total tuples; same
 * 10k-row "non-tiny" cutoff the classic postgres-checkup F003 uses).
 */
export const F003_DEAD_TUPLES_MIN = 100_000;
export const F003_DEAD_PCT_MIN = 20;
export const F003_AUTOVACUUM_DISABLED_MIN_ROWS = 10_000;

/**
 * F003 "is autovacuum keeping up?" analysis constants (WI #271).
 *
 * Part 1 evaluates each table against its *actual* trigger point under the
 * *effective* settings (global GUC overridden by pg_class.reloptions); Part 2
 * takes a single-snapshot queue/worker-saturation reading. Express has no time
 * series, so the judgment is deliberately snapshot-only — the back-to-back
 * vacuum timeline judgment belongs to full monitoring / Dashboard 7.
 */
/**
 * Top-K offenders returned by over_trigger_ratio. Mirrors the SQL `rn_ratio <= 50`
 * cap in the pg_dead_tuples_keepup metric. The metric's actual returned set is
 * `rn_dead <= 100 OR rn_ratio <= F003_TOP_K` (≤ ~150 rows): top-50 by
 * over_trigger_ratio UNION top-100 by n_dead_tup — the 100 retains parity with
 * the legacy pg_dead_tuples top-100 so express does not regress. Either way the
 * report stays bounded on 100k-relation databases (top-K + aggregates only).
 */
export const F003_TOP_K = 50;
/**
 * The noise gate (minimum total tuples) applied to the queue/trigger analysis.
 * Reuses the F003_AUTOVACUUM_DISABLED_MIN_ROWS semantics: a 1000-row table
 * sitting 51 dead tuples over its trigger is not a finding. The same value is
 * hard-coded in the pg_dead_tuples_keepup SQL (`>= 10000`); keep them in sync.
 */
export const F003_KEEPUP_MIN_ROWS = F003_AUTOVACUUM_DISABLED_MIN_ROWS;
/**
 * queue_length > F003_QUEUE_SATURATION_MULTIPLIER * autovacuum_max_workers is a
 * chronic under-provisioning signal regardless of the instantaneous worker
 * state: a single snapshot cannot see back-to-back vacuums, but a queue many
 * times deeper than the worker pool cannot be a transient blip.
 */
export const F003_QUEUE_SATURATION_MULTIPLIER = 5;
/**
 * A table over its vacuum trigger whose last (auto)vacuum is older than this
 * many hours (or which was never autovacuumed) and which no worker is
 * currently processing is flagged as starving.
 */
export const F003_STARVATION_HOURS = 24;

/**
 * I/O statistics by backend type (I001) - matches I001.schema.json backendIOStats
 */
export interface BackendIOStats {
  backend_type: string;
  reads: number;
  /** Read MiB. The historical `_mb` suffix is retained for schema compatibility. */
  read_bytes_mb: number;
  read_time_ms: number;
  writes: number;
  /** Written MiB. The historical `_mb` suffix is retained for schema compatibility. */
  write_bytes_mb: number;
  write_time_ms: number;
  writebacks: number;
  /** Writeback MiB. Always 0 on PG18+ (op_bytes removed, no writeback byte counts exposed). The historical `_mb` suffix is retained for schema compatibility. */
  writeback_bytes_mb: number;
  writeback_time_ms: number;
  fsyncs: number;
  fsync_time_ms: number;
  /** Relation extension operations reported by pg_stat_io for PostgreSQL 16+. */
  extends?: number;
  /** Extended MiB; PG16 derives extends * op_bytes, PG18+ uses native extend_bytes. */
  extend_bytes_mb?: number;
  hits: number;
  evictions: number;
  reuses: number;
}

/**
 * I/O statistics analysis summary (I001)
 */
export interface IOAnalysis {
  total_read_mb: number;
  total_write_mb: number;
  /** read_time_ms + write_time_ms across backends. Excludes writeback and fsync time. */
  total_io_time_ms: number;
  /** Buffer hit ratio: hits / (hits + reads) * 100. */
  read_hit_ratio_pct: number;
  /** Average read latency, or null when there are no reads. */
  avg_read_time_ms: number | null;
  /** Average write latency, or null when there are no writes. */
  avg_write_time_ms: number | null;
}

/**
 * Node result for reports
 */
export interface NodeResult {
  data: Record<string, any>;
  postgres_version?: PostgresVersion;
  // F001 autovacuum configuration linter (WI 274) — additive siblings of `data`.
  effective_values?: EffectiveAutovacuumSettings;
  throughput_budget?: ThroughputBudget;
  conclusions?: string[];
  recommendations?: string[];
  settings_analysis?: Record<string, any>;
}

/**
 * Report structure matching JSON schemas
 */
export interface Report {
  /** Version of the JSON report contract (see {@link CONTRACT_VERSION}). */
  contract_version: string;
  version: string | null;
  build_ts: string | null;
  generation_mode: string | null;
  checkId: string;
  checkTitle: string;
  timestamptz: string;
  nodes: {
    primary: string;
    standbys: string[];
  };
  results: Record<string, NodeResult>;
  /**
   * Severity summary for this check (status + human-readable message), derived
   * from the report data. Optional and additive: attached to the `--json`
   * output so embedders never reimplement severity logic. See
   * {@link withCheckSummary}.
   */
  summary?: CheckSummary;
}

/**
 * Parse PostgreSQL version number into major and minor components
 */
export function parseVersionNum(versionNum: string): { major: string; minor: string } {
  if (!versionNum || versionNum.length < 6) {
    return { major: "", minor: "" };
  }
  try {
    const num = parseInt(versionNum, 10);
    return {
      major: Math.floor(num / 10000).toString(),
      minor: (num % 10000).toString(),
    };
  } catch (err) {
    // parseInt shouldn't throw, but handle edge cases defensively
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[parseVersionNum] Warning: Failed to parse "${versionNum}": ${errorMsg}`);
    return { major: "", minor: "" };
  }
}

/**
 * Format bytes to human readable string using binary units (1024-based).
 * Uses IEC standard: KiB, MiB, GiB, etc.
 *
 * Note: PostgreSQL's pg_size_pretty() uses kB/MB/GB with 1024 base (technically
 * incorrect SI usage), but we follow IEC binary units per project style guide.
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  if (bytes < 0) return `-${formatBytes(-bytes)}`; // Handle negative values
  if (!Number.isFinite(bytes)) return `${bytes} B`; // Handle NaN/Infinity
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
}

/**
 * Format a setting's pretty value from the normalized value and unit.
 * The settings metric provides setting_normalized (bytes or seconds) and unit_normalized.
 */
function formatSettingPrettyValue(
  settingNormalized: number | null,
  unitNormalized: string | null,
  rawValue: string
): string {
  if (settingNormalized === null || unitNormalized === null) {
    return rawValue;
  }

  if (unitNormalized === "bytes") {
    return formatBytes(settingNormalized);
  }

  if (unitNormalized === "seconds") {
    // Format time values with appropriate units based on magnitude:
    // - Sub-second values (< 1s): show in milliseconds for precision
    // - Small values (< 60s): show in seconds
    // - Larger values (>= 60s): show in minutes for readability
    const MS_PER_SECOND = 1000;
    if (settingNormalized < 1) {
      return `${(settingNormalized * MS_PER_SECOND).toFixed(0)} ms`;
    } else if (settingNormalized < SECONDS_PER_MINUTE) {
      return `${settingNormalized} s`;
    } else {
      return `${(settingNormalized / SECONDS_PER_MINUTE).toFixed(1)} min`;
    }
  }

  return rawValue;
}

/**
 * Get PostgreSQL version information.
 * Uses simple inline SQL (trivial query, CLI-specific).
 *
 * @throws {Error} If database query fails (propagating - critical data)
 */
export async function getPostgresVersion(client: Client): Promise<PostgresVersion> {
  const result = await client.query(`
    select name, setting
    from pg_settings
    where name in ('server_version', 'server_version_num')
  `);

  let version = "";
  let serverVersionNum = "";

  for (const row of result.rows) {
    if (row.name === "server_version") {
      version = row.setting;
    } else if (row.name === "server_version_num") {
      serverVersionNum = row.setting;
    }
  }

  const { major, minor } = parseVersionNum(serverVersionNum);

  return {
    version,
    server_version_num: serverVersionNum,
    server_major_ver: major,
    server_minor_ver: minor,
  };
}

/**
 * Get all PostgreSQL settings
 * Uses 'settings' metric from metrics.yml
 */
export async function getSettings(client: Client, pgMajorVersion: number = 16): Promise<Record<string, SettingInfo>> {
  const sql = getMetricSql(METRIC_NAMES.settings, pgMajorVersion);
  const result = await client.query(sql);
  const settings: Record<string, SettingInfo> = {};

  for (const row of result.rows) {
    // The settings metric uses tag_setting_name, tag_setting_value, etc.
    const name = row.tag_setting_name;
    const settingValue = row.tag_setting_value;
    const unit = row.tag_unit || "";
    const category = row.tag_category || "";
    const vartype = row.tag_vartype || "";
    const settingNormalized = row.setting_normalized !== null ? parseFloat(row.setting_normalized) : null;
    const unitNormalized = row.unit_normalized || null;

    settings[name] = {
      setting: settingValue,
      unit,
      category,
      context: "", // Not available in the monitoring metric
      vartype,
      pretty_value: formatSettingPrettyValue(settingNormalized, unitNormalized, settingValue),
    };
  }

  return settings;
}

/**
 * Get altered (non-default) PostgreSQL settings
 * Uses 'settings' metric from metrics.yml and filters for non-default
 */
export async function getAlteredSettings(client: Client, pgMajorVersion: number = 16): Promise<Record<string, AlteredSetting>> {
  const sql = getMetricSql(METRIC_NAMES.settings, pgMajorVersion);
  const result = await client.query(sql);
  const settings: Record<string, AlteredSetting> = {};

  for (const row of result.rows) {
    // Filter for non-default settings (is_default = 0 means non-default)
    if (!toBool(row.is_default)) {
      const name = row.tag_setting_name;
      const settingValue = row.tag_setting_value;
      const unit = row.tag_unit || "";
      const category = row.tag_category || "";
      const settingNormalized = row.setting_normalized !== null ? parseFloat(row.setting_normalized) : null;
      const unitNormalized = row.unit_normalized || null;

      settings[name] = {
        value: settingValue,
        unit,
        category,
        pretty_value: formatSettingPrettyValue(settingNormalized, unitNormalized, settingValue),
      };
    }
  }

  return settings;
}

/**
 * Get database sizes (all non-template databases)
 * Uses simple inline SQL (lists all databases, CLI-specific)
 */
export async function getDatabaseSizes(client: Client): Promise<Record<string, number>> {
  const result = await client.query(`
    select
      datname,
      pg_database_size(datname) as size_bytes
    from pg_database
    where datistemplate = false
    order by size_bytes desc
  `);
  const sizes: Record<string, number> = {};

  for (const row of result.rows) {
    sizes[row.datname] = parseInt(row.size_bytes, 10);
  }

  return sizes;
}

/**
 * Get cluster general info metrics
 * Uses 'db_stats' metric and inline SQL for connection states/uptime
 */
export async function getClusterInfo(client: Client, pgMajorVersion: number = 16): Promise<Record<string, ClusterMetric>> {
  const info: Record<string, ClusterMetric> = {};

  // Get database statistics from db_stats metric
  const dbStatsSql = getMetricSql(METRIC_NAMES.dbStats, pgMajorVersion);
  const statsResult = await client.query(dbStatsSql);
  if (statsResult.rows.length > 0) {
    const stats = statsResult.rows[0];

    info.total_connections = {
      value: String(stats.numbackends || 0),
      unit: "connections",
      description: "Current database connections",
    };

    info.total_commits = {
      value: String(stats.xact_commit || 0),
      unit: "transactions",
      description: "Total committed transactions",
    };

    info.total_rollbacks = {
      value: String(stats.xact_rollback || 0),
      unit: "transactions",
      description: "Total rolled back transactions",
    };

    const blocksHit = parseInt(stats.blks_hit || "0", 10);
    const blocksRead = parseInt(stats.blks_read || "0", 10);
    const totalBlocks = blocksHit + blocksRead;
    const cacheHitRatio = totalBlocks > 0 ? ((blocksHit / totalBlocks) * 100).toFixed(2) : "0.00";

    info.cache_hit_ratio = {
      value: cacheHitRatio,
      unit: "%",
      description: "Buffer cache hit ratio",
    };

    info.blocks_read = {
      value: String(blocksRead),
      unit: "blocks",
      description: "Total disk blocks read",
    };

    info.blocks_hit = {
      value: String(blocksHit),
      unit: "blocks",
      description: "Total buffer cache hits",
    };

    info.tuples_returned = {
      value: String(stats.tup_returned || 0),
      unit: "rows",
      description: "Total rows returned by queries",
    };

    info.tuples_fetched = {
      value: String(stats.tup_fetched || 0),
      unit: "rows",
      description: "Total rows fetched by queries",
    };

    info.tuples_inserted = {
      value: String(stats.tup_inserted || 0),
      unit: "rows",
      description: "Total rows inserted",
    };

    info.tuples_updated = {
      value: String(stats.tup_updated || 0),
      unit: "rows",
      description: "Total rows updated",
    };

    info.tuples_deleted = {
      value: String(stats.tup_deleted || 0),
      unit: "rows",
      description: "Total rows deleted",
    };

    info.total_deadlocks = {
      value: String(stats.deadlocks || 0),
      unit: "deadlocks",
      description: "Total deadlocks detected",
    };

    info.temp_files_created = {
      value: String(stats.temp_files || 0),
      unit: "files",
      description: "Total temporary files created",
    };

    const tempBytes = parseInt(stats.temp_bytes || "0", 10);
    info.temp_bytes_written = {
      value: formatBytes(tempBytes),
      unit: "bytes",
      description: "Total temporary file bytes written",
    };

    // Uptime from db_stats
    if (stats.postmaster_uptime_s) {
      const uptimeSeconds = parseInt(stats.postmaster_uptime_s, 10);
      const days = Math.floor(uptimeSeconds / SECONDS_PER_DAY);
      const hours = Math.floor((uptimeSeconds % SECONDS_PER_DAY) / SECONDS_PER_HOUR);
      const minutes = Math.floor((uptimeSeconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
      info.uptime = {
        value: `${days} days ${hours}:${String(minutes).padStart(2, "0")}:${String(uptimeSeconds % SECONDS_PER_MINUTE).padStart(2, "0")}`,
        unit: "interval",
        description: "Server uptime",
      };
    }
  }

  // Get connection states (simple inline SQL)
  const connResult = await client.query(`
    select
      coalesce(state, 'null') as state,
      count(*) as count
    from pg_stat_activity
    group by state
  `);
  for (const row of connResult.rows) {
    const stateKey = `connections_${row.state.replace(/\s+/g, "_")}`;
    info[stateKey] = {
      value: String(row.count),
      unit: "connections",
      description: `Connections in '${row.state}' state`,
    };
  }

  // Get uptime info (simple inline SQL)
  const uptimeResult = await client.query(`
    select
      pg_postmaster_start_time() as start_time,
      current_timestamp - pg_postmaster_start_time() as uptime
  `);
  if (uptimeResult.rows.length > 0) {
    const uptime = uptimeResult.rows[0];
    const startTime = uptime.start_time instanceof Date
      ? uptime.start_time.toISOString()
      : String(uptime.start_time);
    info.start_time = {
      value: startTime,
      unit: "timestamp",
      description: "PostgreSQL server start time",
    };
    if (!info.uptime) {
      info.uptime = {
        value: String(uptime.uptime),
        unit: "interval",
        description: "Server uptime",
      };
    }
  }

  return info;
}

/**
 * Get invalid indexes from the database (H001).
 * Invalid indexes have indisvalid = false, typically from failed CREATE INDEX CONCURRENTLY.
 *
 * @param client - Connected PostgreSQL client
 * @param pgMajorVersion - PostgreSQL major version (default: 16)
 * @returns Array of invalid index entries with observation data for decision tree analysis
 */
export async function getInvalidIndexes(client: Client, pgMajorVersion: number = 16): Promise<InvalidIndex[]> {
  const sql = getMetricSql(METRIC_NAMES.H001, pgMajorVersion);
  const result = await client.query(sql);
  return result.rows.map((row) => {
    const transformed = transformMetricRow(row);
    const indexSizeBytes = parseInt(String(transformed.index_size_bytes || 0), 10);

    return {
      schema_name: String(transformed.schema_name || ""),
      table_name: String(transformed.table_name || ""),
      index_name: String(transformed.index_name || ""),
      relation_name: String(transformed.relation_name || ""),
      index_size_bytes: indexSizeBytes,
      index_size_pretty: formatBytes(indexSizeBytes),
      index_definition: String(transformed.index_definition || ""),
      supports_fk: toBool(transformed.supports_fk),
      is_pk: toBool(transformed.is_pk),
      is_unique: toBool(transformed.is_unique),
      constraint_name: transformed.constraint_name ? String(transformed.constraint_name) : null,
      table_row_estimate: parseInt(String(transformed.table_row_estimate || 0), 10),
      has_valid_duplicate: toBool(transformed.has_valid_duplicate),
      valid_duplicate_name: transformed.valid_index_name ? String(transformed.valid_index_name) : null,
      valid_duplicate_definition: transformed.valid_index_definition ? String(transformed.valid_index_definition) : null,
    };
  });
}

/**
 * Get unused indexes from the database (H002).
 * Unused indexes have zero scans since stats were last reset.
 *
 * @param client - Connected PostgreSQL client
 * @param pgMajorVersion - PostgreSQL major version (default: 16)
 * @returns Array of unused index entries with scan counts and FK support info
 */
export async function getUnusedIndexes(client: Client, pgMajorVersion: number = 16): Promise<UnusedIndex[]> {
  const sql = getMetricSql(METRIC_NAMES.H002, pgMajorVersion);
  const result = await client.query(sql);
  return result.rows.map((row) => {
    const transformed = transformMetricRow(row);
    const indexSizeBytes = parseInt(String(transformed.index_size_bytes || 0), 10);
    return {
      schema_name: String(transformed.schema_name || ""),
      table_name: String(transformed.table_name || ""),
      index_name: String(transformed.index_name || ""),
      index_definition: String(transformed.index_definition || ""),
      reason: String(transformed.reason || ""),
      idx_scan: parseInt(String(transformed.idx_scan || 0), 10),
      index_size_bytes: indexSizeBytes,
      idx_is_btree: toBool(transformed.idx_is_btree),
      supports_fk: toBool(transformed.supports_fk),
      index_size_pretty: formatBytes(indexSizeBytes),
    };
  });
}

/**
 * Get stats reset info (H002)
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (stats_reset)
 */
export async function getStatsReset(client: Client, pgMajorVersion: number = 16): Promise<StatsReset> {
  const sql = getMetricSql(METRIC_NAMES.statsReset, pgMajorVersion);
  const result = await client.query(sql);
  const row = result.rows[0] || {};

  // The stats_reset metric returns stats_reset_epoch and seconds_since_reset
  // We need to calculate additional fields
  const statsResetEpoch = row.stats_reset_epoch ? parseFloat(row.stats_reset_epoch) : null;
  const secondsSinceReset = row.seconds_since_reset ? parseInt(row.seconds_since_reset, 10) : null;

  // Calculate stats_reset_time from epoch
  const statsResetTime = statsResetEpoch
    ? new Date(statsResetEpoch * 1000).toISOString()
    : null;

  // Calculate days since reset
  const daysSinceReset = secondsSinceReset !== null
    ? Math.floor(secondsSinceReset / SECONDS_PER_DAY)
    : null;

  // Get postmaster startup time separately (simple inline SQL)
  // This is supplementary data - errors are captured in output, not propagated
  let postmasterStartupEpoch: number | null = null;
  let postmasterStartupTime: string | null = null;
  let postmasterStartupError: string | undefined;
  try {
    const pmResult = await client.query(`
      select
        extract(epoch from pg_postmaster_start_time()) as postmaster_startup_epoch,
        pg_postmaster_start_time()::text as postmaster_startup_time
    `);
    if (pmResult.rows.length > 0) {
      postmasterStartupEpoch = pmResult.rows[0].postmaster_startup_epoch
        ? parseFloat(pmResult.rows[0].postmaster_startup_epoch)
        : null;
      postmasterStartupTime = pmResult.rows[0].postmaster_startup_time || null;
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    postmasterStartupError = `Failed to query postmaster start time: ${errorMsg}`;
    console.error(`[getStatsReset] Warning: ${postmasterStartupError}`);
  }

  const statsResult: StatsReset = {
    stats_reset_epoch: statsResetEpoch,
    stats_reset_time: statsResetTime,
    days_since_reset: daysSinceReset,
    postmaster_startup_epoch: postmasterStartupEpoch,
    postmaster_startup_time: postmasterStartupTime,
  };

  // Only include error field if there was an error (keeps output clean)
  if (postmasterStartupError) {
    statsResult.postmaster_startup_error = postmasterStartupError;
  }

  return statsResult;
}

/**
 * Get current database name and size
 * Uses 'db_size' metric from metrics.yml
 */
export async function getCurrentDatabaseInfo(client: Client, pgMajorVersion: number = 16): Promise<{ datname: string; size_bytes: number }> {
  const sql = getMetricSql(METRIC_NAMES.dbSize, pgMajorVersion);
  const result = await client.query(sql);
  const row = result.rows[0] || {};

  // db_size metric returns tag_datname and size_b
  return {
    datname: row.tag_datname || "postgres",
    size_bytes: parseInt(row.size_b || "0", 10),
  };
}

/**
 * Type guard to validate redundant_to_json item structure.
 * Returns true if item is a valid object (may have expected properties).
 */
function isValidRedundantToItem(item: unknown): item is Record<string, unknown> {
  return typeof item === "object" && item !== null && !Array.isArray(item);
}

/**
 * Get redundant indexes from the database (H004).
 * Redundant indexes are covered by other indexes (same leading columns).
 *
 * @param client - Connected PostgreSQL client
 * @param pgMajorVersion - PostgreSQL major version (default: 16)
 * @returns Array of redundant index entries with covering index info
 */
export async function getRedundantIndexes(client: Client, pgMajorVersion: number = 16): Promise<RedundantIndex[]> {
  const sql = getMetricSql(METRIC_NAMES.H004, pgMajorVersion);
  const result = await client.query(sql);
  return result.rows.map((row) => {
    const transformed = transformMetricRow(row);
    const indexSizeBytes = parseInt(String(transformed.index_size_bytes || 0), 10);
    const tableSizeBytes = parseInt(String(transformed.table_size_bytes || 0), 10);

    // Parse redundant_to JSON array (indexes that make this one redundant)
    let redundantTo: RedundantToIndex[] = [];
    let parseError: string | undefined;
    try {
      const jsonStr = String(transformed.redundant_to_json || "[]");
      const parsed = JSON.parse(jsonStr);
      if (Array.isArray(parsed)) {
        redundantTo = parsed
          .filter(isValidRedundantToItem)
          .map((item) => {
            const sizeBytes = parseInt(String(item.index_size_bytes ?? 0), 10);
            return {
              index_name: String(item.index_name ?? ""),
              index_definition: String(item.index_definition ?? ""),
              index_size_bytes: sizeBytes,
              index_size_pretty: formatBytes(sizeBytes),
            };
          });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const indexName = String(transformed.index_name || "unknown");
      parseError = `Failed to parse redundant_to_json: ${errorMsg}`;
      console.error(`[H004] Warning: ${parseError} for index "${indexName}"`);
    }

    const result: RedundantIndex = {
      schema_name: String(transformed.schema_name || ""),
      table_name: String(transformed.table_name || ""),
      index_name: String(transformed.index_name || ""),
      relation_name: String(transformed.relation_name || ""),
      access_method: String(transformed.access_method || ""),
      reason: String(transformed.reason || ""),
      index_size_bytes: indexSizeBytes,
      table_size_bytes: tableSizeBytes,
      index_usage: parseInt(String(transformed.index_usage || 0), 10),
      supports_fk: toBool(transformed.supports_fk),
      index_definition: String(transformed.index_definition || ""),
      index_size_pretty: formatBytes(indexSizeBytes),
      table_size_pretty: formatBytes(tableSizeBytes),
      redundant_to: redundantTo,
    };

    // Only include parse error field if there was an error (keeps output clean)
    if (parseError) {
      result.redundant_to_parse_error = parseError;
    }

    return result;
  });
}

/**
 * Get per-table dead-tuple stats and per-table autovacuum overrides (F003).
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (pg_dead_tuples metric).
 *
 * Returns tables that carry dead tuples or have autovacuum disabled per-table,
 * with threshold flags precomputed (see F003_* constants).
 *
 * @param client - Connected PostgreSQL client
 * @param pgMajorVersion - PostgreSQL major version (default: 16)
 * @throws {Error} If database query fails (propagating - critical data)
 */
export async function getDeadTuples(client: Client, pgMajorVersion: number = 16): Promise<DeadTuplesTable[]> {
  const sql = getMetricSql(METRIC_NAMES.F003, pgMajorVersion);
  const result = await client.query(sql);
  return result.rows.map((row) => mapDeadTupleBaseRow(transformMetricRow(row)));
}

/**
 * Map one transformed metric row to the base DeadTuplesTable fields shared by
 * the legacy pg_dead_tuples metric and the WI #271 pg_dead_tuples_keepup
 * metric. Exported for unit testing of the mapping in isolation.
 */
export function mapDeadTupleBaseRow(t: Record<string, unknown>): DeadTuplesTable {
  const nLive = parseInt(String(t.n_live_tup || 0), 10);
  const nDead = parseInt(String(t.n_dead_tup || 0), 10);
  const deadPct = parseFloat(String(t.dead_pct)) || 0;
  const lastAutovacuumEpoch = parseInt(String(t.last_autovacuum || 0), 10);
  const lastVacuumEpoch = parseInt(String(t.last_vacuum || 0), 10);
  // The metric emits 0/1; be liberal in what we accept (driver may return strings)
  const autovacuumDisabled = parseInt(String(t.autovacuum_disabled || 0), 10) === 1 || toBool(t.autovacuum_disabled);
  const tableSizeBytes = parseInt(String(t.table_size_b || 0), 10);

  return {
    schema_name: String(t.schemaname || ""),
    table_name: String(t.relname || ""),
    n_live_tup: nLive,
    n_dead_tup: nDead,
    dead_pct: deadPct,
    last_autovacuum: lastAutovacuumEpoch > 0 ? new Date(lastAutovacuumEpoch * 1000).toISOString() : null,
    last_autovacuum_epoch: lastAutovacuumEpoch,
    last_vacuum: lastVacuumEpoch > 0 ? new Date(lastVacuumEpoch * 1000).toISOString() : null,
    last_vacuum_epoch: lastVacuumEpoch,
    autovacuum_count: parseInt(String(t.autovacuum_count || 0), 10),
    vacuum_count: parseInt(String(t.vacuum_count || 0), 10),
    autovacuum_disabled: autovacuumDisabled,
    table_size_bytes: tableSizeBytes,
    table_size_pretty: formatBytes(tableSizeBytes),
    exceeds_dead_tuple_thresholds: nDead >= F003_DEAD_TUPLES_MIN && deadPct >= F003_DEAD_PCT_MIN,
    autovacuum_disabled_flagged: autovacuumDisabled && nLive + nDead >= F003_AUTOVACUUM_DISABLED_MIN_ROWS,
  };
}

/** Parse a nullable numeric column that may arrive as string | number | null | undefined. */
function numOrNull(v: unknown): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

/**
 * Attach the WI #271 settings-aware trigger fields to a base DeadTuplesTable.
 * Exported for unit testing of the trigger math parsing/gating.
 */
export function mapDeadTupleTriggerFields(t: Record<string, unknown>): Partial<DeadTuplesTable> {
  const lastAutoanalyzeEpoch = parseInt(String(t.last_autoanalyze || 0), 10);
  const insertThreshold = numOrNull(t.eff_insert_threshold);
  const insertScale = numOrNull(t.eff_insert_scale_factor);
  const insertTrigger = numOrNull(t.insert_trigger_point);
  const nInsSinceVacuum = numOrNull(t.n_ins_since_vacuum);
  // over_insert_trigger only meaningful on PG13+ (insert columns present).
  const hasInsert = insertTrigger !== null || nInsSinceVacuum !== null;

  return {
    reltuples: Math.round(parseFloat(String(t.reltuples || 0)) || 0),
    relpages: parseInt(String(t.relpages || 0), 10) || 0,
    n_mod_since_analyze: parseInt(String(t.n_mod_since_analyze || 0), 10) || 0,
    n_ins_since_vacuum: nInsSinceVacuum,
    last_autoanalyze: lastAutoanalyzeEpoch > 0 ? new Date(lastAutoanalyzeEpoch * 1000).toISOString() : null,
    last_autoanalyze_epoch: lastAutoanalyzeEpoch,
    eff_vacuum_threshold: parseFloat(String(t.eff_vacuum_threshold || 0)) || 0,
    eff_vacuum_scale_factor: parseFloat(String(t.eff_vacuum_scale_factor || 0)) || 0,
    vacuum_settings_from_reloptions: toBool(t.vacuum_settings_from_reloptions),
    eff_analyze_threshold: parseFloat(String(t.eff_analyze_threshold || 0)) || 0,
    eff_analyze_scale_factor: parseFloat(String(t.eff_analyze_scale_factor || 0)) || 0,
    eff_insert_threshold: insertThreshold,
    eff_insert_scale_factor: insertScale,
    insert_settings_from_reloptions: toBool(t.insert_settings_from_reloptions),
    vacuum_trigger_point: parseFloat(String(t.vacuum_trigger_point || 0)) || 0,
    analyze_trigger_point: parseFloat(String(t.analyze_trigger_point || 0)) || 0,
    insert_trigger_point: insertTrigger,
    over_trigger_ratio: parseFloat(String(t.over_trigger_ratio || 0)) || 0,
    over_vacuum_trigger: toBool(t.over_vacuum_trigger),
    over_analyze_trigger: toBool(t.over_analyze_trigger),
    over_insert_trigger: hasInsert ? toBool(t.over_insert_trigger) : false,
    toast_autovacuum_disabled: toBool(t.toast_autovacuum_disabled),
  };
}

/** Parse the aggregate coverage counters carried on every keepup row (WI #271). */
export function parseKeepupAggregates(rows: Record<string, unknown>[]): AutovacuumKeepupAggregates {
  // All rows carry the same aggregate columns (cross join agg); read the first.
  // When there are zero rows the queue is empty and everything defaults to 0.
  const r = rows[0] || {};
  const insertQueue = numOrNull(r.insert_queue_length);
  return {
    relations_total: parseInt(String(r.relations_total || 0), 10) || 0,
    candidates_considered: parseInt(String(r.candidates_considered || 0), 10) || 0,
    queue_length: parseInt(String(r.queue_length || 0), 10) || 0,
    analyze_queue_length: parseInt(String(r.analyze_queue_length || 0), 10) || 0,
    insert_queue_length: insertQueue === null ? null : Math.round(insertQueue),
    total_dead_tuples_all: parseInt(String(r.total_dead_tuples_all || 0), 10) || 0,
  };
}

/**
 * Settings-aware autovacuum trigger analysis (WI #271, Part 1).
 *
 * Single materialized pass over pg_stat_user_tables joined to pg_class:
 * resolves each table's effective autovacuum settings (global GUC overridden by
 * pg_class.reloptions incl. toast.* variants) and computes the real per-table
 * vacuum/analyze/(PG13+)insert trigger points and over_trigger_ratio. Returns
 * the bounded top-K offenders plus the full-scan aggregate coverage counters.
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (pg_dead_tuples_keepup).
 */
export async function getAutovacuumKeepup(
  client: Client,
  pgMajorVersion: number = 16
): Promise<{ tables: DeadTuplesTable[]; aggregates: AutovacuumKeepupAggregates }> {
  const sql = getMetricSql(METRIC_NAMES.F003_KEEPUP, pgMajorVersion);
  const result = await client.query(sql);
  const rows = result.rows.map(transformMetricRow);
  // The metric always emits an aggregate-coverage row (agg LEFT JOIN finalists),
  // which carries the counters but no relation (relname empty) when there are no
  // finalists. Skip that synthetic row when building the per-table list; the
  // aggregates are still read from it via parseKeepupAggregates.
  const tables = rows
    .filter((t) => String(t.relname || "") !== "")
    .map((t) => ({ ...mapDeadTupleBaseRow(t), ...mapDeadTupleTriggerFields(t) }));
  return { tables, aggregates: parseKeepupAggregates(rows) };
}

/**
 * Single-snapshot autovacuum worker capacity vs demand (WI #271, Part 2).
 * SQL loaded from metrics.yml (pg_autovacuum_worker_snapshot). Degrades to a
 * safe empty snapshot on any error (the worker view is best-effort telemetry).
 */
export async function getAutovacuumWorkerSnapshot(
  client: Client,
  pgMajorVersion: number = 16
): Promise<AutovacuumWorkerSnapshot> {
  try {
    const sql = getMetricSql(METRIC_NAMES.F003_WORKER_SNAPSHOT, pgMajorVersion);
    const result = await client.query(sql);
    const r = transformMetricRow(result.rows[0] || {});
    const maxWorkers = numOrNull(r.max_workers);
    const activeWorkers = parseInt(String(r.active_workers || 0), 10) || 0;
    return {
      active_workers: activeWorkers,
      max_workers: maxWorkers === null ? null : Math.round(maxWorkers),
      free_slots: maxWorkers === null ? null : Math.max(0, Math.round(maxWorkers) - activeWorkers),
      anti_wraparound_workers: parseInt(String(r.anti_wraparound_workers || 0), 10) || 0,
    };
  } catch {
    return { active_workers: 0, max_workers: null, free_slots: null, anti_wraparound_workers: 0 };
  }
}

/**
 * Autovacuum workers currently blocked on locks, with blocker info
 * (WI #271, Part 2). SQL loaded from metrics.yml (pg_autovacuum_blocked).
 */
export async function getAutovacuumBlocked(
  client: Client,
  pgMajorVersion: number = 16
): Promise<BlockedAutovacuumWorker[]> {
  try {
    const sql = getMetricSql(METRIC_NAMES.F003_BLOCKED, pgMajorVersion);
    const result = await client.query(sql);
    return result.rows.map((row) => {
      const r = transformMetricRow(row);
      const blockerPid = parseInt(String(r.blocker_pid || 0), 10) || 0;
      const queryid = r.blocker_queryid !== undefined && r.blocker_queryid !== null ? String(r.blocker_queryid) : "";
      return {
        worker_pid: parseInt(String(r.worker_pid || 0), 10) || 0,
        blocker_pid: blockerPid > 0 ? blockerPid : null,
        blocker_queryid: queryid !== "" && queryid !== "0" ? queryid : null,
        wait_seconds: Math.round((parseFloat(String(r.wait_seconds || 0)) || 0) * 100) / 100,
      };
    });
  } catch {
    return [];
  }
}

const VACUUM_PHASE_NAMES: Record<number, string> = {
  1: "initializing",
  2: "scanning heap",
  3: "vacuuming indexes",
  4: "vacuuming heap",
  5: "cleaning up indexes",
  6: "truncating heap",
  7: "performing final cleanup",
};

/**
 * Running vacuums from pg_stat_progress_vacuum (WI #271, Part 2).
 * SQL loaded from metrics.yml (pg_vacuum_progress).
 */
export async function getVacuumProgress(
  client: Client,
  pgMajorVersion: number = 16
): Promise<VacuumProgressEntry[]> {
  try {
    const sql = getMetricSql(METRIC_NAMES.F003_PROGRESS, pgMajorVersion);
    const result = await client.query(sql);
    return result.rows.map((row) => {
      const r = transformMetricRow(row);
      const phaseCode = numOrNull(r.phase);
      return {
        schema_name: String(r.schema_name || ""),
        table_name: String(r.table_name || ""),
        vacuum_mode: String(r.vacuum_mode || "unknown"),
        phase: phaseCode !== null ? (VACUUM_PHASE_NAMES[phaseCode] || "unknown") : "unknown",
        phase_code: phaseCode === null ? null : Math.round(phaseCode),
        heap_blks_total: Math.round(parseFloat(String(r.heap_blks_total || 0)) || 0),
        heap_blks_scanned: Math.round(parseFloat(String(r.heap_blks_scanned || 0)) || 0),
        heap_blks_vacuumed: Math.round(parseFloat(String(r.heap_blks_vacuumed || 0)) || 0),
        index_vacuum_count: Math.round(parseFloat(String(r.index_vacuum_count || 0)) || 0),
        is_anti_wraparound:
          toBool(r.is_anti_wraparound) || String(r.vacuum_mode || "") === "aggressive_autovacuum",
      };
    });
  } catch {
    return [];
  }
}

/**
 * Single-snapshot "is autovacuum keeping up?" judgment (WI #271, Part 2).
 *
 * Deliberately snapshot-only (express has no time series). Flags:
 * - saturated: queue > 0 while every worker is busy — cannot currently keep up.
 * - chronic under-provisioning: queue many times deeper than the worker pool.
 * - anti-wraparound present: escalate (see F002 cross-reference).
 */
export function judgeKeepingUp(
  aggregates: AutovacuumKeepupAggregates,
  workers: AutovacuumWorkerSnapshot,
  blocked: BlockedAutovacuumWorker[],
  progress: VacuumProgressEntry[],
  starvingTablesCount: number
): AutovacuumKeepup {
  const antiWraparoundPresent =
    workers.anti_wraparound_workers > 0 || progress.some((p) => p.is_anti_wraparound);
  const saturated =
    aggregates.queue_length > 0 &&
    workers.max_workers !== null &&
    workers.active_workers >= workers.max_workers;
  const chronic =
    workers.max_workers !== null &&
    aggregates.queue_length > F003_QUEUE_SATURATION_MULTIPLIER * workers.max_workers;

  let status: "ok" | "warning" | "critical" = "ok";
  if (antiWraparoundPresent) {
    status = "critical";
  } else if (saturated || chronic || blocked.length > 0) {
    status = "warning";
  }

  const judgmentParts: string[] = [];
  // Worker counts are cluster-wide (pg_stat_activity), while the queue is scoped
  // to this database — make that explicit so the saturation reading isn't
  // misread as per-database worker capacity.
  const workerText =
    workers.max_workers !== null
      ? `${workers.active_workers}/${workers.max_workers} workers busy cluster-wide`
      : `${workers.active_workers} workers busy cluster-wide`;
  if (saturated) {
    judgmentParts.push(
      `Autovacuum cannot currently keep up: ${aggregates.queue_length} table(s) in this database past their vacuum trigger and ${workerText}.`
    );
  } else if (chronic) {
    judgmentParts.push(
      `Autovacuum queue (${aggregates.queue_length}) is more than ${F003_QUEUE_SATURATION_MULTIPLIER}x the worker pool (${workerText}): chronic under-provisioning.`
    );
  } else if (aggregates.queue_length > 0) {
    judgmentParts.push(
      `${aggregates.queue_length} table(s) past their vacuum trigger; ${workerText} (queue is being worked).`
    );
  } else {
    judgmentParts.push(`Autovacuum is keeping up: no tables past their vacuum trigger; ${workerText}.`);
  }
  if (antiWraparoundPresent) {
    judgmentParts.push(
      `${workers.anti_wraparound_workers || progress.filter((p) => p.is_anti_wraparound).length} anti-wraparound worker(s) running (see F002).`
    );
  }
  if (blocked.length > 0) {
    judgmentParts.push(`${blocked.length} autovacuum worker(s) blocked on locks.`);
  }
  if (starvingTablesCount > 0) {
    judgmentParts.push(`${starvingTablesCount} table(s) starving (over trigger, not vacuumed recently, not being processed).`);
  }

  return {
    ...aggregates,
    active_workers: workers.active_workers,
    max_workers: workers.max_workers,
    free_slots: workers.free_slots,
    anti_wraparound_workers: workers.anti_wraparound_workers,
    anti_wraparound_present: antiWraparoundPresent,
    saturated,
    chronic_under_provisioning: chronic,
    starving_tables_count: starvingTablesCount,
    blocked_workers: blocked,
    vacuum_progress: progress,
    judgment: judgmentParts.join(" "),
    status,
  };
}

/**
 * Quote a schema-qualified identifier for copy-paste DDL, doubling any embedded
 * double quotes (CWE-116). e.g. ("public", `weird"name`) -> "public"."weird""name".
 */
export function quoteIdent(schema: string, table: string): string {
  const q = (s: string) => `"${String(s).replace(/"/g, '""')}"`;
  return `${q(schema)}.${q(table)}`;
}

/**
 * Build keeping-up conclusions and recommendations for F003 (WI #271).
 *
 * Exported separately so the wording can be unit-tested without a database.
 * NEVER-RECOMMEND list (test-enforced): no recommendation ever proposes
 * autovacuum_vacuum_cost_delay = 0 (write-storm risk) or raising
 * autovacuum_max_workers without also raising the shared cost budget
 * (autovacuum_vacuum_cost_limit).
 */
export function buildKeepupConclusions(
  tables: DeadTuplesTable[],
  keepup: AutovacuumKeepup
): { conclusions: string[]; recommendations: string[] } {
  const conclusions: string[] = [];
  const recommendations: string[] = [];
  const fmt = (n: number) => n.toLocaleString("en-US");
  const rel = (t: DeadTuplesTable) => quoteIdent(t.schema_name, t.table_name);
  const overGate = (t: DeadTuplesTable) => (t.n_live_tup + t.n_dead_tup) >= F003_KEEPUP_MIN_ROWS;

  // Per-table over-trigger findings, most overdue first, bounded to avoid spam.
  const overTrigger = tables
    .filter((t) => t.over_vacuum_trigger && overGate(t))
    .sort((a, b) => (b.over_trigger_ratio || 0) - (a.over_trigger_ratio || 0))
    .slice(0, 5);

  for (const t of overTrigger) {
    const ratio = t.over_trigger_ratio || 0;
    const src = t.vacuum_settings_from_reloptions ? "per-table reloptions" : "global settings";
    conclusions.push(
      `Table ${rel(t)} has ${fmt(t.n_dead_tup)} dead tuples, ${ratio.toFixed(1)}x its vacuum trigger ` +
      `(~${fmt(Math.round(t.vacuum_trigger_point || 0))}; effective autovacuum_vacuum_scale_factor=` +
      `${t.eff_vacuum_scale_factor}, autovacuum_vacuum_threshold=${t.eff_vacuum_threshold}, from ${src}).`
    );
    // Per-table reloption recommendation: big table still on a large scale
    // factor and not already overridden per-table. Per-table overrides are
    // preferred over global changes.
    if (
      !t.vacuum_settings_from_reloptions &&
      (t.eff_vacuum_scale_factor || 0) >= 0.1 &&
      (t.reltuples || 0) >= 1_000_000
    ) {
      recommendations.push(
        `Lower the autovacuum trigger for the large table ${rel(t)} with a per-table override ` +
        `(preferred over changing the global setting): ` +
        `alter table ${rel(t)} set (autovacuum_vacuum_scale_factor = 0.02); ` +
        `for very large tables an absolute autovacuum_vacuum_threshold can be more predictable. ` +
        `Then run: vacuum (analyze) ${rel(t)}; to clear the current backlog.`
      );
    }
  }

  // Analyze-trigger findings (stale planner statistics): surface top offenders.
  const overAnalyze = tables
    .filter((t) => t.over_analyze_trigger && overGate(t))
    .sort((a, b) => (b.n_mod_since_analyze || 0) - (a.n_mod_since_analyze || 0))
    .slice(0, 3);
  for (const t of overAnalyze) {
    conclusions.push(
      `Table ${rel(t)} is past its autovacuum analyze trigger ` +
      `(${fmt(t.n_mod_since_analyze || 0)} modified tuples > ~${fmt(Math.round(t.analyze_trigger_point || 0))}); ` +
      `its planner statistics are stale.`
    );
    recommendations.push(
      `Run: analyze ${rel(t)}; and, if stale statistics keep recurring on it, lower its ` +
      `analyze scale factor: alter table ${rel(t)} set (autovacuum_analyze_scale_factor = 0.02);`
    );
  }

  // Insert-trigger findings (append-only tables). PG13+ only: over_insert_trigger
  // is always false on PG12 (the insert-trigger settings do not exist there).
  const overInsert = tables
    .filter((t) => t.over_insert_trigger && overGate(t))
    .sort((a, b) => (b.n_ins_since_vacuum || 0) - (a.n_ins_since_vacuum || 0))
    .slice(0, 3);
  for (const t of overInsert) {
    conclusions.push(
      `Append-only table ${rel(t)} is past its autovacuum insert trigger ` +
      `(${fmt(t.n_ins_since_vacuum || 0)} inserts since last vacuum > ~${fmt(Math.round(t.insert_trigger_point || 0))}); ` +
      `insert-driven vacuums keep the visibility and freeze maps current (index-only scans, cheaper freezing).`
    );
    recommendations.push(
      `Keep insert-triggered autovacuum current on ${rel(t)} ` +
      `(PG13+: autovacuum_vacuum_insert_scale_factor / autovacuum_vacuum_insert_threshold); ` +
      `a periodic vacuum (analyze) ${rel(t)}; also refreshes its visibility and freeze maps.`
    );
  }

  // Anti-wraparound escalation with F002 cross-reference.
  if (keepup.anti_wraparound_present) {
    const n = keepup.anti_wraparound_workers || keepup.vacuum_progress.filter((p) => p.is_anti_wraparound).length;
    conclusions.push(
      `${n} anti-wraparound autovacuum worker(s) are running (vacuuming "to prevent wraparound"). ` +
      `This is transaction-ID-wraparound prevention and must be allowed to complete.`
    );
    recommendations.push(
      `Do not cancel the anti-wraparound autovacuum worker(s); remove anything blocking them ` +
      `(long-running transactions, conflicting DDL) so they can finish. ` +
      `Cross-reference check F002 (transaction ID / MultiXact wraparound) for the XID age picture.`
    );
  }

  // Saturation / chronic under-provisioning: cost-budget-first recommendation.
  if (keepup.saturated || keepup.chronic_under_provisioning) {
    conclusions.push(keepup.judgment);
    recommendations.push(
      `Raise autovacuum throughput by increasing the shared cost budget: raise ` +
      `autovacuum_vacuum_cost_limit, and only together with it raise autovacuum_max_workers ` +
      `— all workers share one cost budget, so adding workers without also raising ` +
      `autovacuum_vacuum_cost_limit just makes each vacuum slower. Keep autovacuum's ` +
      `cost-delay throttle enabled. Also lower per-table autovacuum_vacuum_scale_factor on the ` +
      `busiest tables so they are vacuumed more frequently in smaller batches.`
    );
  }

  // Blocked workers: surface blocker PIDs/queries.
  for (const b of keepup.blocked_workers) {
    const blocker = b.blocker_pid ? `pid ${b.blocker_pid}` : "an unknown backend";
    const q = b.blocker_queryid ? ` (query id ${b.blocker_queryid})` : "";
    conclusions.push(
      `Autovacuum worker pid ${b.worker_pid} has been blocked on a lock held by ${blocker}${q} ` +
      `for ${b.wait_seconds}s.`
    );
    recommendations.push(
      `Investigate the lock holder ${blocker} blocking autovacuum worker pid ${b.worker_pid} ` +
      `(look it up in pg_stat_activity); autovacuum cannot make progress on that table until it is released.`
    );
  }

  return { conclusions, recommendations };
}

export async function getWraparoundData(client: Client, pgMajorVersion: number = 16): Promise<{
  settings: WraparoundSettings;
  databases: WraparoundDatabase[];
  tables: WraparoundTable[];
  multixact_size: MultixactSize;
  settings_available: boolean;
}> {
  const [settingsResult, databaseResult, tableResult, multixactSizeResult] = await Promise.all([
    client.query(getMetricSql(METRIC_NAMES.F002Settings, pgMajorVersion)),
    client.query(getMetricSql(METRIC_NAMES.F002Database, pgMajorVersion)),
    client.query(getMetricSql(METRIC_NAMES.F002Tables, pgMajorVersion)),
    client.query(getMetricSql(METRIC_NAMES.F002MultixactSize, pgMajorVersion)),
  ]);

  const numberValue = (value: unknown): number => parseInt(String(value ?? 0), 10) || 0;
  const rawSettings = settingsResult.rows[0] || {};
  const failsafe = pgMajorVersion >= 14 ? numberValue(rawSettings.vacuum_failsafe_age) : 0;
  const multixactFailsafe = pgMajorVersion >= 14 ? numberValue(rawSettings.vacuum_multixact_failsafe_age) : 0;
  const settings: WraparoundSettings = {
    autovacuum_freeze_max_age: numberValue(rawSettings.autovacuum_freeze_max_age),
    vacuum_freeze_min_age: numberValue(rawSettings.vacuum_freeze_min_age),
    vacuum_freeze_table_age: numberValue(rawSettings.vacuum_freeze_table_age),
    autovacuum_multixact_freeze_max_age: numberValue(rawSettings.autovacuum_multixact_freeze_max_age),
    vacuum_multixact_freeze_min_age: numberValue(rawSettings.vacuum_multixact_freeze_min_age),
    vacuum_multixact_freeze_table_age: numberValue(rawSettings.vacuum_multixact_freeze_table_age),
    vacuum_failsafe_age: failsafe > 0 ? failsafe : null,
    vacuum_multixact_failsafe_age: multixactFailsafe > 0 ? multixactFailsafe : null,
  };
  const settings_available = settings.autovacuum_freeze_max_age > 0 &&
    settings.autovacuum_multixact_freeze_max_age > 0;

  const databases = databaseResult.rows.map((row) => {
    const xidAge = numberValue(row.age_datfrozenxid);
    const multixactAge = numberValue(row.age_datminmxid);
    return {
      database_name: String(row.tag_datname || ""),
      xid: evaluateWraparoundRisk(xidAge, settings.autovacuum_freeze_max_age, settings.vacuum_failsafe_age),
      multixact: evaluateWraparoundRisk(
        multixactAge,
        settings.autovacuum_multixact_freeze_max_age,
        settings.vacuum_multixact_failsafe_age,
      ),
    };
  }).sort((a, b) => Math.max(b.xid.age, b.multixact.age) - Math.max(a.xid.age, a.multixact.age));

  const tables = tableResult.rows.map((row) => {
    const xidAge = numberValue(row.xid_age);
    const multixactAge = numberValue(row.multixact_age);
    const tableSizeBytes = numberValue(row.table_size_bytes);
    return {
      database_name: String(row.tag_datname || ""),
      schema_name: String(row.tag_schema_name || ""),
      table_name: String(row.tag_table_name || ""),
      ranked_by: String(row.tag_ranked_by || "").split(",").filter(Boolean),
      table_size_bytes: tableSizeBytes,
      table_size_pretty: formatBytes(tableSizeBytes),
      xid: evaluateWraparoundRisk(
        xidAge,
        numberValue(row.effective_freeze_max_age) || settings.autovacuum_freeze_max_age,
        settings.vacuum_failsafe_age,
      ),
      multixact: evaluateWraparoundRisk(
        multixactAge,
        numberValue(row.effective_multixact_freeze_max_age) || settings.autovacuum_multixact_freeze_max_age,
        settings.vacuum_multixact_failsafe_age,
      ),
    };
  }).sort((a, b) =>
    Math.max(b.xid.age, b.multixact.age) - Math.max(a.xid.age, a.multixact.age) ||
    a.database_name.localeCompare(b.database_name) ||
    a.schema_name.localeCompare(b.schema_name) ||
    a.table_name.localeCompare(b.table_name)
  );

  const rawMultixactSize = multixactSizeResult.rows[0] || {};
  const multixactBytes = rawMultixactSize.members_bytes === null || rawMultixactSize.members_bytes === undefined ||
    rawMultixactSize.offsets_bytes === null || rawMultixactSize.offsets_bytes === undefined
    ? null
    : numberValue(rawMultixactSize.members_bytes) + numberValue(rawMultixactSize.offsets_bytes);
  const multixact_size: MultixactSize = {
    bytes: multixactBytes,
    size_pretty: multixactBytes === null ? null : formatBytes(multixactBytes),
    status_code: numberValue(rawMultixactSize.status_code),
  };

  return { settings, databases, tables, multixact_size, settings_available };
}

export function buildWraparoundConclusions(
  databases: WraparoundDatabase[],
  tables: WraparoundTable[],
  settingsAvailable: boolean = true,
): { severity: WraparoundSeverity; conclusions: string[]; recommendations: string[] } {
  if (!settingsAvailable) {
    return {
      severity: "info",
      conclusions: ["Wraparound settings are unavailable; severity could not be evaluated."],
      recommendations: ["Verify pg_settings_wraparound collection and rerun F002 before assessing wraparound risk."],
    };
  }
  const offenders = [
    ...databases.flatMap((db) => [
      { name: `database "${db.database_name}"`, kind: "transaction ID", risk: db.xid },
      { name: `database "${db.database_name}"`, kind: "MultiXact", risk: db.multixact },
    ]),
    ...tables.flatMap((table) => [
      { name: `table "${table.database_name}"."${table.schema_name}"."${table.table_name}"`, kind: "transaction ID", risk: table.xid },
      { name: `table "${table.database_name}"."${table.schema_name}"."${table.table_name}"`, kind: "MultiXact", risk: table.multixact },
    ]),
  ].filter((item) => item.risk.severity !== "info")
    .sort((a, b) => severityRank[b.risk.severity] - severityRank[a.risk.severity] || b.risk.age - a.risk.age);

  const severity = maxSeverity(...offenders.map((item) => item.risk.severity));
  if (offenders.length === 0) {
    return {
      severity,
      conclusions: ["Transaction ID and MultiXact ages are below their emergency vacuum thresholds."],
      recommendations: [],
    };
  }

  const conclusions = offenders.slice(0, 10).map((item) =>
    `${item.name} has ${item.kind} age ${item.risk.age.toLocaleString("en-US")} ` +
    `(${item.risk.pct_towards_emergency}% of its emergency vacuum threshold; ${item.risk.severity}).`
  );
  const recommendations = [
    "Check pg_stat_activity for 'autovacuum: %to prevent wraparound%' workers and inspect pg_stat_progress_vacuum for progress.",
    "Check xmin-horizon holders (long transactions, stale replication slots, and prepared transactions) and the F003 autovacuum queue analysis; high age is usually a symptom of vacuum starvation or a blocked horizon.",
  ];
  if (severity === "high" || severity === "critical") {
    recommendations.push(
      "Run VACUUM (FREEZE, VERBOSE) on the highest-age offender tables after confirming operational impact. Do not raise freeze_max_age merely to silence the check; that reduces the remaining safety margin."
    );
  }
  return { severity, conclusions, recommendations };
}

/**
 * Build concrete, human-readable conclusions and recommendations for F003.
 *
 * Exported separately so the wording (which the console surfaces verbatim in
 * auto-created issues) can be unit-tested without a database.
 */
export function buildDeadTuplesConclusions(tables: DeadTuplesTable[]): {
  conclusions: string[];
  recommendations: string[];
} {
  const conclusions: string[] = [];
  const recommendations: string[] = [];

  const fmt = (n: number) => n.toLocaleString("en-US");

  for (const t of tables) {
    const rel = quoteIdent(t.schema_name, t.table_name);
    const lastAv = t.last_autovacuum
      ? `last autovacuum: ${t.last_autovacuum}`
      : "autovacuum has never vacuumed it";

    if (t.exceeds_dead_tuple_thresholds && t.autovacuum_disabled) {
      conclusions.push(
        `Table ${rel} has ${fmt(t.n_dead_tup)} dead tuples (${t.dead_pct}% of all tuples) ` +
        `and autovacuum is disabled on it via reloptions (${lastAv}).`
      );
      recommendations.push(
        `Re-enable autovacuum on ${rel}: alter table ${rel} reset (autovacuum_enabled); ` +
        `then run: vacuum (analyze) ${rel}; to clean up the accumulated dead tuples.`
      );
    } else if (t.exceeds_dead_tuple_thresholds) {
      conclusions.push(
        `Table ${rel} has ${fmt(t.n_dead_tup)} dead tuples (${t.dead_pct}% of all tuples; ${lastAv}).`
      );
      recommendations.push(
        `Run: vacuum (analyze) ${rel}; and review autovacuum settings ` +
        `(autovacuum_vacuum_scale_factor, autovacuum_vacuum_cost_delay, autovacuum_max_workers) ` +
        `if dead tuples keep accumulating on ${rel}.`
      );
    } else if (t.autovacuum_disabled_flagged) {
      conclusions.push(
        `Autovacuum is disabled via reloptions on table ${rel} ` +
        `(~${fmt(t.n_live_tup + t.n_dead_tup)} tuples); dead tuples and transaction ID age ` +
        `will accumulate unchecked.`
      );
      recommendations.push(
        `Re-enable autovacuum on ${rel}: alter table ${rel} reset (autovacuum_enabled); ` +
        `unless this table is managed by a carefully scheduled manual vacuum job.`
      );
    }
  }

  return { conclusions, recommendations };
}

/**
 * Create base report structure
 */
export function createBaseReport(
  checkId: string,
  checkTitle: string,
  nodeName: string
): Report {
  const buildTs = resolveBuildTs();
  return {
    contract_version: CONTRACT_VERSION,
    version: pkg.version || null,
    build_ts: buildTs,
    generation_mode: "express",
    checkId,
    checkTitle,
    timestamptz: new Date().toISOString(),
    nodes: {
      primary: nodeName,
      standbys: [],
    },
    results: {},
  };
}

function readTextFileSafe(p: string): string | null {
  try {
    const value = fs.readFileSync(p, "utf8").trim();
    return value || null;
  } catch {
    // Intentionally silent: this is a "safe" read that returns null on any error
    // (file not found, permission denied, etc.) - used for optional config files
    return null;
  }
}

function resolveBuildTs(): string | null {
  // Follow reporter.py approach: read BUILD_TS from filesystem, with env override.
  // Default: /BUILD_TS (useful in container images).
  const envPath = process.env.PGAI_BUILD_TS_FILE;
  const p = (envPath && envPath.trim()) ? envPath.trim() : "/BUILD_TS";

  const fromFile = readTextFileSafe(p);
  if (fromFile) return fromFile;

  // Fallback for packaged CLI: allow placing BUILD_TS next to dist/ (package root).
  // dist/lib/checkup.js => package root: dist/..
  try {
    const pkgRoot = path.resolve(__dirname, "..");
    const fromPkgFile = readTextFileSafe(path.join(pkgRoot, "BUILD_TS"));
    if (fromPkgFile) return fromPkgFile;
  } catch (err) {
    // Path resolution failing is unexpected - warn about it
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.warn(`[resolveBuildTs] Warning: path resolution failed: ${errorMsg}`);
  }

  // Last resort: use package.json mtime as an approximation (non-null, stable-ish).
  try {
    const pkgJsonPath = path.resolve(__dirname, "..", "package.json");
    const st = fs.statSync(pkgJsonPath);
    return st.mtime.toISOString();
  } catch (err) {
    // package.json not found is expected in some environments (e.g., bundled) - debug only
    if (process.env.DEBUG) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error(`[resolveBuildTs] Could not stat package.json, using current time: ${errorMsg}`);
    }
    return new Date().toISOString();
  }
}

// ============================================================================
// Unified Report Generator Helpers
// ============================================================================

/**
 * Generate a simple version report (A002, A013).
 * These reports only contain PostgreSQL version information.
 */
async function generateVersionReport(
  client: Client,
  nodeName: string,
  checkId: string,
  checkTitle: string
): Promise<Report> {
  const report = createBaseReport(checkId, checkTitle, nodeName);
  const postgresVersion = await getPostgresVersion(client);
  report.results[nodeName] = { data: { version: postgresVersion } };
  return report;
}

/**
 * Generate a settings-based report (A003, A007).
 * Fetches settings using provided function and includes postgres_version.
 */
async function generateSettingsReport(
  client: Client,
  nodeName: string,
  checkId: string,
  checkTitle: string,
  fetchSettings: (client: Client, pgMajorVersion: number) => Promise<Record<string, unknown>>
): Promise<Report> {
  const report = createBaseReport(checkId, checkTitle, nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const settings = await fetchSettings(client, pgMajorVersion);
  report.results[nodeName] = { data: settings, postgres_version: postgresVersion };
  return report;
}

/**
 * Generate an index report (H001, H002, H004).
 * Common structure: index list + totals + database info, keyed by database name.
 */
async function generateIndexReport<T extends { index_size_bytes: number }>(
  client: Client,
  nodeName: string,
  checkId: string,
  checkTitle: string,
  indexFieldName: string,
  fetchIndexes: (client: Client, pgMajorVersion: number) => Promise<T[]>,
  extraFields?: (client: Client, pgMajorVersion: number) => Promise<Record<string, unknown>>
): Promise<Report> {
  const report = createBaseReport(checkId, checkTitle, nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const indexes = await fetchIndexes(client, pgMajorVersion);
  const { datname: dbName, size_bytes: dbSizeBytes } = await getCurrentDatabaseInfo(client, pgMajorVersion);

  const totalCount = indexes.length;
  const totalSizeBytes = indexes.reduce((sum, idx) => sum + idx.index_size_bytes, 0);

  const dbEntry: Record<string, unknown> = {
    [indexFieldName]: indexes,
    total_count: totalCount,
    total_size_bytes: totalSizeBytes,
    total_size_pretty: formatBytes(totalSizeBytes),
    database_size_bytes: dbSizeBytes,
    database_size_pretty: formatBytes(dbSizeBytes),
  };

  // Add extra fields if provided (e.g., stats_reset for H002)
  if (extraFields) {
    Object.assign(dbEntry, await extraFields(client, pgMajorVersion));
  }

  report.results[nodeName] = { data: { [dbName]: dbEntry }, postgres_version: postgresVersion };
  return report;
}

// ============================================================================
// Report Generators (using unified helpers)
// ============================================================================

/** Generate A002 report - Postgres major version */
export const generateA002 = (client: Client, nodeName = "node-01") =>
  generateVersionReport(client, nodeName, "A002", "Postgres major version");

/** Generate A003 report - Postgres settings */
export const generateA003 = (client: Client, nodeName = "node-01") =>
  generateSettingsReport(client, nodeName, "A003", "Postgres settings", getSettings);

/** Generate A004 report - Cluster information (custom structure) */
export async function generateA004(client: Client, nodeName: string = "node-01"): Promise<Report> {
  const report = createBaseReport("A004", "Cluster information", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  report.results[nodeName] = {
    data: {
      general_info: await getClusterInfo(client, pgMajorVersion),
      database_sizes: await getDatabaseSizes(client),
    },
    postgres_version: postgresVersion,
  };
  return report;
}

/** Generate A007 report - Altered settings */
export const generateA007 = (client: Client, nodeName = "node-01") =>
  generateSettingsReport(client, nodeName, "A007", "Altered settings", getAlteredSettings);

/** Generate A013 report - Postgres minor version */
export const generateA013 = (client: Client, nodeName = "node-01") =>
  generateVersionReport(client, nodeName, "A013", "Postgres minor version");

/** Generate H001 report - Invalid indexes */
export const generateH001 = (client: Client, nodeName = "node-01") =>
  generateIndexReport(client, nodeName, "H001", "Invalid indexes", "invalid_indexes", getInvalidIndexes);

/** Generate H002 report - Unused indexes (includes stats_reset) */
export const generateH002 = (client: Client, nodeName = "node-01") =>
  generateIndexReport(client, nodeName, "H002", "Unused indexes", "unused_indexes", getUnusedIndexes,
    async (c, v) => ({ stats_reset: await getStatsReset(c, v) }));

/** Generate H004 report - Redundant indexes */
export const generateH004 = (client: Client, nodeName = "node-01") =>
  generateIndexReport(client, nodeName, "H004", "Redundant indexes", "redundant_indexes", getRedundantIndexes);

/**
 * Generate D004 report - pg_stat_statements and pg_stat_kcache settings.
 *
 * Uses graceful degradation: extension queries are wrapped in try-catch
 * because extensions may not be installed. Errors are included in the
 * report output rather than failing the entire report.
 */
async function generateD004(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("D004", "pg_stat_statements and pg_stat_kcache settings", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const allSettings = await getSettings(client, pgMajorVersion);

  // Filter settings related to pg_stat_statements and pg_stat_kcache
  const pgssSettings: Record<string, SettingInfo> = {};
  for (const [name, setting] of Object.entries(allSettings)) {
    if (name.startsWith("pg_stat_statements") || name.startsWith("pg_stat_kcache")) {
      pgssSettings[name] = setting;
    }
  }

  // Check pg_stat_statements extension
  let pgssAvailable = false;
  let pgssMetricsCount = 0;
  let pgssTotalCalls = 0;
  let pgssError: string | null = null;
  const pgssSampleQueries: Array<{ queryid: string; user: string; database: string; calls: number }> = [];

  try {
    const extCheck = await client.query(
      "select 1 from pg_extension where extname = 'pg_stat_statements'"
    );
    if (extCheck.rows.length > 0) {
      pgssAvailable = true;
      const statsResult = await client.query(`
        select count(*) as cnt, coalesce(sum(calls), 0) as total_calls
        from pg_stat_statements
      `);
      pgssMetricsCount = parseInt(statsResult.rows[0]?.cnt || "0", 10);
      pgssTotalCalls = parseInt(statsResult.rows[0]?.total_calls || "0", 10);

      // Get sample queries (top 5 by calls)
      const sampleResult = await client.query(`
        select
          queryid::text as queryid,
          coalesce(usename, 'unknown') as "user",
          coalesce(datname, 'unknown') as database,
          calls
        from pg_stat_statements s
        left join pg_database d on s.dbid = d.oid
        left join pg_user u on s.userid = u.usesysid
        order by calls desc
        limit 5
      `);
      for (const row of sampleResult.rows) {
        pgssSampleQueries.push({
          queryid: row.queryid,
          user: row.user,
          database: row.database,
          calls: parseInt(row.calls, 10),
        });
      }
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[D004] Error querying pg_stat_statements: ${errorMsg}`);
    pgssError = errorMsg;
  }

  // Check pg_stat_kcache extension
  let kcacheAvailable = false;
  let kcacheMetricsCount = 0;
  let kcacheTotalExecTime = 0;
  let kcacheTotalUserTime = 0;
  let kcacheTotalSystemTime = 0;
  let kcacheError: string | null = null;
  const kcacheSampleQueries: Array<{ queryid: string; user: string; exec_total_time: number }> = [];

  try {
    const extCheck = await client.query(
      "select 1 from pg_extension where extname = 'pg_stat_kcache'"
    );
    if (extCheck.rows.length > 0) {
      kcacheAvailable = true;
      const statsResult = await client.query(`
        select
          count(*) as cnt,
          coalesce(sum(exec_user_time + exec_system_time), 0) as total_exec_time,
          coalesce(sum(exec_user_time), 0) as total_user_time,
          coalesce(sum(exec_system_time), 0) as total_system_time
        from pg_stat_kcache
      `);
      kcacheMetricsCount = parseInt(statsResult.rows[0]?.cnt || "0", 10);
      kcacheTotalExecTime = parseFloat(statsResult.rows[0]?.total_exec_time || "0");
      kcacheTotalUserTime = parseFloat(statsResult.rows[0]?.total_user_time || "0");
      kcacheTotalSystemTime = parseFloat(statsResult.rows[0]?.total_system_time || "0");

      // Get sample queries (top 5 by exec time)
      const sampleResult = await client.query(`
        select
          queryid::text as queryid,
          coalesce(usename, 'unknown') as "user",
          (exec_user_time + exec_system_time) as exec_total_time
        from pg_stat_kcache k
        left join pg_user u on k.userid = u.usesysid
        order by (exec_user_time + exec_system_time) desc
        limit 5
      `);
      for (const row of sampleResult.rows) {
        kcacheSampleQueries.push({
          queryid: row.queryid,
          user: row.user,
          exec_total_time: parseFloat(row.exec_total_time),
        });
      }
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[D004] Error querying pg_stat_kcache: ${errorMsg}`);
    kcacheError = errorMsg;
  }

  report.results[nodeName] = {
    data: {
      settings: pgssSettings,
      pg_stat_statements_status: {
        extension_available: pgssAvailable,
        metrics_count: pgssMetricsCount,
        total_calls: pgssTotalCalls,
        sample_queries: pgssSampleQueries,
        ...(pgssError && { error: pgssError }),
      },
      pg_stat_kcache_status: {
        extension_available: kcacheAvailable,
        metrics_count: kcacheMetricsCount,
        total_exec_time: kcacheTotalExecTime,
        total_user_time: kcacheTotalUserTime,
        total_system_time: kcacheTotalSystemTime,
        sample_queries: kcacheSampleQueries,
        ...(kcacheError && { error: kcacheError }),
      },
    },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Generate D001 report - Logging settings
 *
 * Collects all PostgreSQL logging-related settings including:
 * - Log destination and collector settings
 * - Log file rotation and naming
 * - Log verbosity and filtering
 * - Statement and duration logging
 */
async function generateD001(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("D001", "Logging settings", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const allSettings = await getSettings(client, pgMajorVersion);

  // Filter logging-related settings (log_* and logging_*)
  const loggingSettings: Record<string, SettingInfo> = {};
  for (const [name, setting] of Object.entries(allSettings)) {
    if (name.startsWith("log_") || name.startsWith("logging_")) {
      loggingSettings[name] = setting;
    }
  }

  report.results[nodeName] = {
    data: loggingSettings,
    postgres_version: postgresVersion,
  };

  return report;
}

// ===========================================================================
// F001 autovacuum configuration linter (WI 274)
// ===========================================================================
//
// F001 analyses the EFFECTIVE runtime autovacuum settings from pg_settings
// (the same source generateF001 already reads via getSettings()) plus per-table
// pg_class.reloptions, and emits F003-style conclusions/recommendations from a
// curated rule set. The rule set is expressed as DATA (an array of predicate +
// severity + message-template objects) so tests can enumerate every rule and
// assert its fire / not-fire behaviour and its never-recommend guarantees.
//
// Scope here is Tier-1 (context-free) rules + the effective-throughput math +
// the largest-tables scale-factor cross-reference. Tier-2 hardware-aware rules
// (WI 269 environment context) are NOT implemented; where a recommendation's
// concrete value would need vCPU/RAM/disk facts, the rule degrades to
// conditional text and never fabricates a number.

/** Severity levels for F001 rules, in descending order of urgency. */
export type F001Severity = "CRITICAL" | "WARNING" | "NOTICE" | "INFO";

const F001_SEVERITY_ORDER: Record<F001Severity, number> = {
  CRITICAL: 4,
  WARNING: 3,
  NOTICE: 2,
  INFO: 1,
};

/**
 * F001 thresholds and reference constants (exported so tests and the report
 * can reference them without magic numbers).
 *
 * - PG12 raised the default autovacuum_vacuum_cost_delay from 20ms to 2ms and
 *   is the effective project minimum; the 20ms value is therefore a PG11-era
 *   leftover worth flagging.
 * - The default cost_limit (200) is widely considered too low for modern
 *   storage; EDB's autovacuum-tuning guidance suggests 1000–2000 as a starting
 *   point (hardware-aware refinement is Tier 2).
 * - The default scale_factor (0.2) means 20% of a table must die before
 *   autovacuum triggers; on large tables that is a large dead-tuple debt.
 */
export const F001_COST_LIMIT_DEFAULT = 200;
export const F001_COST_LIMIT_LOW_MAX = 200; // <= this is flagged as likely-too-low
export const F001_COST_DELAY_DEFAULT_MS = 2; // PG12+ default
export const F001_COST_DELAY_PG11_MS = 20; // pre-PG12 default (~8 MB/s)
export const F001_SCALE_FACTOR_DEFAULT = 0.2;
export const F001_ANALYZE_SCALE_FACTOR_DEFAULT = 0.1;
export const F001_NAPTIME_HIGH_S = 60; // naptime strictly greater than this is flagged
export const F001_MAX_WORKERS_DEFAULT = 3;
/** A large table for the scale-factor cross-reference (dead-tuple debt matters here). */
export const F001_LARGE_TABLE_BYTES = 10 * 1024 * 1024 * 1024; // 10 GiB
/** Cap on rows the largest-tables cross-reference reports (mirrors the SQL LIMIT). */
export const F001_LARGEST_TABLES_CAP = 20;
/** Default page costs (pg_settings defaults) used by the throughput math. */
export const F001_PAGE_COST_HIT_DEFAULT = 1;
export const F001_PAGE_COST_MISS_DEFAULT = 2;
export const F001_PAGE_COST_DIRTY_DEFAULT = 20;
/**
 * Dirty-write budget at or below this (MB/s) is "small" and, on a large
 * database, worth calling out: at this rate vacuum may never catch up with
 * write churn. Set to the PG12+ default budget itself (100000 tokens/s ÷
 * vacuum_cost_page_dirty=20 × 8 KiB ≈ 40.96 MB/s, emitted rounded as 41.0),
 * so a large database running at stock defaults IS flagged here — a message
 * distinct from the generic cost_limit_low finding.
 */
export const F001_DIRTY_WRITE_SMALL_MBPS = 41;
/** A "large" database for the small-dirty-budget conclusion. */
export const F001_LARGE_DB_BYTES = 100 * 1024 * 1024 * 1024; // 100 GiB
const PG_BLOCK_BYTES = 8192;

/**
 * An effective (inheritance-resolved) autovacuum value plus the chain that
 * produced it. `-1` sentinels fall back to a cluster-wide sibling setting;
 * the report shows the chain so the reader sees WHY the effective value is
 * what it is.
 */
export interface EffectiveAutovacuumValue {
  /** The setting's own raw value as configured (e.g. "-1", "2", "200"). */
  raw: string | null;
  /** Effective numeric value after resolving -1 inheritance; null if unknown. */
  effective: number | null;
  /** pg_settings unit (e.g. "ms", "kB", ""). */
  unit: string;
  /** Name of the setting the effective value was inherited from (when raw == -1). */
  inherited_from: string | null;
  /** Human-readable inheritance chain, e.g. "autovacuum_vacuum_cost_limit = -1 -> vacuum_cost_limit = 200". */
  inheritance_chain: string;
}

/** Resolved effective autovacuum settings used by the rule engine. */
export interface EffectiveAutovacuumSettings {
  autovacuum: boolean | null;
  track_counts: boolean | null;
  cost_delay_ms: EffectiveAutovacuumValue;
  cost_limit: EffectiveAutovacuumValue;
  work_mem_kb: EffectiveAutovacuumValue;
  naptime_s: number | null;
  scale_factor: number | null;
  analyze_scale_factor: number | null;
  /** PG13+; -1 means insert-driven autovacuum is disabled. null when unavailable (<PG13). */
  vacuum_insert_threshold: number | null;
  log_min_duration_ms: number | null;
  max_workers: number | null;
  page_cost_hit: number;
  page_cost_miss: number;
  page_cost_dirty: number;
}

/** Effective vacuum throughput budget derived from the cost-based delay model. */
export interface ThroughputBudget {
  effective_cost_delay_ms: number | null;
  effective_cost_limit: number | null;
  page_cost_hit: number;
  page_cost_miss: number;
  page_cost_dirty: number;
  /** cost tokens spent per second across ALL workers (shared budget); null when throttling disabled. */
  tokens_per_sec: number | null;
  /** MB/s ceilings for all-in-cache reads, cache-miss reads, and dirtied pages. null when unthrottled. */
  read_hit_mbps: number | null;
  read_miss_mbps: number | null;
  dirty_write_mbps: number | null;
  /** True when effective cost_delay is 0 → throttling fully disabled (budget is unbounded). */
  throttling_disabled: boolean;
}

/** A largest-table / override-carrying table row for the scale-factor cross-reference. */
export interface LargestTable {
  schema_name: string;
  table_name: string;
  relkind: string;
  relpages: number;
  total_relation_size_bytes: number;
  total_relation_size_pretty: string;
  /** Whether the table carries any autovacuum_* / toast.autovacuum_* reloption. */
  has_av_override: boolean;
  /** Raw reloptions text (comma-joined). */
  reloptions: string;
  /** Per-table autovacuum_vacuum_scale_factor override, if set. */
  scale_factor_override: number | null;
}

/** Per-table reloptions overview (counts + notable overrides). */
export interface ReloptionsOverview {
  relations_total: number;
  candidates_considered: number;
  tables_with_av_overrides: number;
  /** Tables with autovacuum disabled per-table (already flagged in detail by F003). */
  autovacuum_disabled_tables: string[];
  /** Tables with a per-table cost_delay=0 (write-storm risk, ardentperf). */
  cost_delay_zero_tables: string[];
  /**
   * Subset of cost_delay_zero_tables whose zero lives ONLY on the toast relation
   * (toast.autovacuum_vacuum_cost_delay=0), not on the heap itself. Tracked
   * separately so the per-table rule can name the toast-level ones distinctly —
   * their remediation targets the toast.* option, not the heap option.
   */
  cost_delay_zero_toast_only_tables: string[];
}

/** Context passed to every F001 rule predicate / message builder. */
export interface AutovacuumRuleContext {
  pgMajorVersion: number;
  effective: EffectiveAutovacuumSettings;
  throughput: ThroughputBudget;
  largestTables: LargestTable[];
  reloptions: ReloptionsOverview;
  databaseSizeBytes: number;
  /** WI 269 environment context (vCPU/RAM/disk/provider). Not implemented yet. */
  env: null;
}

/** A single fired rule's output. */
export interface FiredRule {
  id: string;
  severity: F001Severity;
  conclusion: string;
  recommendation: string;
}

/**
 * A data-driven F001 rule. `predicate` decides whether the rule fires from the
 * resolved context; `message` builds the conclusion + recommendation (only
 * called when the predicate is true). `appliesTo`, when present, version-gates
 * the rule (e.g. insert-driven vacuum is PG13+).
 */
export interface AutovacuumRule {
  id: string;
  severity: F001Severity;
  appliesTo?: (ctx: AutovacuumRuleContext) => boolean;
  predicate: (ctx: AutovacuumRuleContext) => boolean;
  message: (ctx: AutovacuumRuleContext) => { conclusion: string; recommendation: string };
}

// --- parse helpers ---------------------------------------------------------

function avNum(settings: Record<string, SettingInfo>, name: string): number | null {
  const s = settings[name];
  if (!s || s.setting === undefined || s.setting === null || s.setting === "") return null;
  const n = parseFloat(String(s.setting));
  return Number.isFinite(n) ? n : null;
}

function avBool(settings: Record<string, SettingInfo>, name: string): boolean | null {
  const s = settings[name];
  if (!s) return null;
  const v = String(s.setting).toLowerCase();
  if (v === "on" || v === "true" || v === "t" || v === "1" || v === "yes") return true;
  if (v === "off" || v === "false" || v === "f" || v === "0" || v === "no") return false;
  return null;
}

function avUnit(settings: Record<string, SettingInfo>, name: string): string {
  return settings[name]?.unit || "";
}

function avRaw(settings: Record<string, SettingInfo>, name: string): string | null {
  const s = settings[name];
  return s && s.setting !== undefined && s.setting !== null ? String(s.setting) : null;
}

/**
 * Resolve the effective autovacuum settings, following the `-1` inheritance
 * chains that Postgres uses:
 *   autovacuum_vacuum_cost_delay = -1 -> vacuum_cost_delay
 *   autovacuum_vacuum_cost_limit = -1 -> vacuum_cost_limit
 *   autovacuum_work_mem          = -1 -> maintenance_work_mem
 * Each resolved value carries the chain that produced it.
 */
export function resolveEffectiveAutovacuumSettings(
  settings: Record<string, SettingInfo>,
): EffectiveAutovacuumSettings {
  // cost_delay: autovacuum_vacuum_cost_delay = -1 inherits vacuum_cost_delay
  const avDelayRaw = avNum(settings, "autovacuum_vacuum_cost_delay");
  const baseDelay = avNum(settings, "vacuum_cost_delay");
  const delayInherits = avDelayRaw !== null && avDelayRaw < 0;
  const cost_delay_ms: EffectiveAutovacuumValue = {
    raw: avRaw(settings, "autovacuum_vacuum_cost_delay"),
    effective: delayInherits ? baseDelay : avDelayRaw,
    unit: avUnit(settings, "autovacuum_vacuum_cost_delay") || "ms",
    inherited_from: delayInherits ? "vacuum_cost_delay" : null,
    inheritance_chain: delayInherits
      ? `autovacuum_vacuum_cost_delay = -1 -> vacuum_cost_delay = ${baseDelay ?? "?"}ms`
      : `autovacuum_vacuum_cost_delay = ${avDelayRaw ?? "?"}ms`,
  };

  // cost_limit: autovacuum_vacuum_cost_limit = -1 inherits vacuum_cost_limit
  const avLimitRaw = avNum(settings, "autovacuum_vacuum_cost_limit");
  const baseLimit = avNum(settings, "vacuum_cost_limit");
  const limitInherits = avLimitRaw !== null && avLimitRaw < 0;
  const cost_limit: EffectiveAutovacuumValue = {
    raw: avRaw(settings, "autovacuum_vacuum_cost_limit"),
    effective: limitInherits ? baseLimit : avLimitRaw,
    unit: avUnit(settings, "autovacuum_vacuum_cost_limit"),
    inherited_from: limitInherits ? "vacuum_cost_limit" : null,
    inheritance_chain: limitInherits
      ? `autovacuum_vacuum_cost_limit = -1 -> vacuum_cost_limit = ${baseLimit ?? "?"}`
      : `autovacuum_vacuum_cost_limit = ${avLimitRaw ?? "?"}`,
  };

  // work_mem: autovacuum_work_mem = -1 inherits maintenance_work_mem
  const avWorkMemRaw = avNum(settings, "autovacuum_work_mem");
  const baseWorkMem = avNum(settings, "maintenance_work_mem");
  const workMemInherits = avWorkMemRaw !== null && avWorkMemRaw < 0;
  const work_mem_kb: EffectiveAutovacuumValue = {
    raw: avRaw(settings, "autovacuum_work_mem"),
    effective: workMemInherits ? baseWorkMem : avWorkMemRaw,
    unit: avUnit(settings, "autovacuum_work_mem") || "kB",
    inherited_from: workMemInherits ? "maintenance_work_mem" : null,
    inheritance_chain: workMemInherits
      ? `autovacuum_work_mem = -1 -> maintenance_work_mem = ${baseWorkMem ?? "?"}kB`
      : `autovacuum_work_mem = ${avWorkMemRaw ?? "?"}kB`,
  };

  return {
    autovacuum: avBool(settings, "autovacuum"),
    track_counts: avBool(settings, "track_counts"),
    cost_delay_ms,
    cost_limit,
    work_mem_kb,
    naptime_s: avNum(settings, "autovacuum_naptime"),
    scale_factor: avNum(settings, "autovacuum_vacuum_scale_factor"),
    analyze_scale_factor: avNum(settings, "autovacuum_analyze_scale_factor"),
    vacuum_insert_threshold: avNum(settings, "autovacuum_vacuum_insert_threshold"),
    log_min_duration_ms: avNum(settings, "log_autovacuum_min_duration"),
    max_workers: avNum(settings, "autovacuum_max_workers"),
    page_cost_hit: avNum(settings, "vacuum_cost_page_hit") ?? F001_PAGE_COST_HIT_DEFAULT,
    page_cost_miss: avNum(settings, "vacuum_cost_page_miss") ?? F001_PAGE_COST_MISS_DEFAULT,
    page_cost_dirty: avNum(settings, "vacuum_cost_page_dirty") ?? F001_PAGE_COST_DIRTY_DEFAULT,
  };
}

/**
 * Compute the effective vacuum throughput budget from the cost-based model
 * (see https://www.enterprisedb.com/blog/autovacuum-tuning-basics):
 *
 *   tokens_per_sec   = 1000 / cost_delay_ms * cost_limit
 *   read_hit_mbps    = tokens_per_sec / vacuum_cost_page_hit   * 8KB
 *   read_miss_mbps   = tokens_per_sec / vacuum_cost_page_miss  * 8KB
 *   dirty_write_mbps = tokens_per_sec / vacuum_cost_page_dirty * 8KB
 *
 * This budget is SHARED across all workers. cost_delay = 0 means throttling is
 * fully disabled (unbounded budget) — reported as such, never recommended.
 */
export function computeThroughputBudget(eff: EffectiveAutovacuumSettings): ThroughputBudget {
  const delay = eff.cost_delay_ms.effective;
  const limit = eff.cost_limit.effective;
  const throttlingDisabled = delay === 0;

  let tokens: number | null = null;
  let readHit: number | null = null;
  let readMiss: number | null = null;
  let dirty: number | null = null;

  if (delay !== null && limit !== null && delay > 0) {
    tokens = (1000 / delay) * limit;
    const toMbps = (cost: number) => (cost > 0 ? (tokens! / cost) * PG_BLOCK_BYTES / 1_000_000 : null);
    readHit = toMbps(eff.page_cost_hit);
    readMiss = toMbps(eff.page_cost_miss);
    dirty = toMbps(eff.page_cost_dirty);
  }

  const round1 = (n: number | null) => (n === null ? null : Math.round(n * 10) / 10);

  return {
    effective_cost_delay_ms: delay,
    effective_cost_limit: limit,
    page_cost_hit: eff.page_cost_hit,
    page_cost_miss: eff.page_cost_miss,
    page_cost_dirty: eff.page_cost_dirty,
    tokens_per_sec: tokens === null ? null : Math.round(tokens),
    read_hit_mbps: round1(readHit),
    read_miss_mbps: round1(readMiss),
    dirty_write_mbps: round1(dirty),
    throttling_disabled: throttlingDisabled,
  };
}

/**
 * Fetch the largest tables and per-table autovacuum reloptions overrides for
 * the F001 scale-factor cross-reference and reloptions overview. Catalog-only
 * two-stage prefilter (see the pg_autovacuum_relopts metric): rank by relpages,
 * exact size for the finalists, report <= F001_LARGEST_TABLES_CAP per category.
 */
export async function getAutovacuumRelopts(
  client: Client,
  pgMajorVersion: number = 16,
): Promise<{ largestTables: LargestTable[]; overview: ReloptionsOverview }> {
  const sql = getMetricSql(METRIC_NAMES.F001, pgMajorVersion);
  const result = await client.query(sql);

  // The heap's own scale_factor override (anchored to start/comma so a
  // toast.autovacuum_vacuum_scale_factor entry is NOT mistaken for the heap's).
  const parseScaleFactorOverride = (reloptions: string): number | null => {
    const m = reloptions.match(/(?:^|,)autovacuum_vacuum_scale_factor=([0-9.]+)/);
    return m ? parseFloat(m[1]) : null;
  };
  // cost_delay=0 on the heap itself (anchored so a toast.* entry is NOT matched).
  const hasHeapCostDelayZero = (reloptions: string): boolean =>
    /(?:^|,)\s*autovacuum_vacuum_cost_delay=0(?:\.0*)?(?:,|$)/.test(reloptions);
  // cost_delay=0 on the table's toast relation (SQL re-emits it 'toast.'-prefixed).
  const hasToastCostDelayZero = (reloptions: string): boolean =>
    /(?:^|,)\s*toast\.autovacuum_vacuum_cost_delay=0(?:\.0*)?(?:,|$)/.test(reloptions);
  const hasAutovacuumDisabled = (reloptions: string): boolean =>
    /(?:^|,)\s*autovacuum_enabled=(?:false|fals|fal|fa|f|no|n|off|of|0)(?:,|$)/i.test(reloptions);

  const byOid = new Map<string, LargestTable>();
  const largestTables: LargestTable[] = [];
  const distinctRelations = new Set<string>();
  const autovacuumDisabledTables = new Set<string>();
  const costDelayZeroTables = new Set<string>();
  const costDelayZeroToastOnlyTables = new Set<string>();
  let relationsTotal = 0;
  let tablesWithOverrides = 0;

  for (const row of result.rows) {
    const t = transformMetricRow(row);
    const schema = String(t.schemaname || "");
    const relname = String(t.relname || "");
    const rel = `"${schema}"."${relname}"`;
    const reloptions = String(t.reloptions || "");
    const category = String(t.category || "");
    const sizeBytes = parseInt(String(t.total_relation_size_b || 0), 10);

    relationsTotal = Math.max(relationsTotal, parseInt(String(t.relations_total || 0), 10));
    tablesWithOverrides = Math.max(tablesWithOverrides, parseInt(String(t.tables_with_av_overrides || 0), 10));
    // Count DISTINCT relations (a table can appear in both the 'largest' and
    // 'override' categories) so the coverage counter never exceeds the total.
    distinctRelations.add(rel);

    const table: LargestTable = {
      schema_name: schema,
      table_name: relname,
      relkind: String(t.relkind || ""),
      relpages: parseInt(String(t.relpages || 0), 10),
      total_relation_size_bytes: sizeBytes,
      total_relation_size_pretty: formatBytes(sizeBytes),
      has_av_override: parseInt(String(t.has_av_override || 0), 10) === 1,
      reloptions,
      scale_factor_override: parseScaleFactorOverride(reloptions),
    };

    if (hasAutovacuumDisabled(reloptions)) autovacuumDisabledTables.add(rel);
    const heapZero = hasHeapCostDelayZero(reloptions);
    const toastZero = hasToastCostDelayZero(reloptions);
    if (heapZero || toastZero) costDelayZeroTables.add(rel);
    // Track toast-only overrides distinctly (a heap-level zero on any row for
    // this relation wins, so the flag is retracted if the heap sets it too).
    if (heapZero) costDelayZeroToastOnlyTables.delete(rel);
    else if (toastZero) costDelayZeroToastOnlyTables.add(rel);

    // The 'largest' category feeds the scale-factor cross-reference; dedupe the
    // 'override' rows that also appear as largest.
    if (category === "largest" && !byOid.has(rel)) {
      byOid.set(rel, table);
      largestTables.push(table);
    }
  }

  largestTables.sort((a, b) => b.total_relation_size_bytes - a.total_relation_size_bytes);

  return {
    largestTables: largestTables.slice(0, F001_LARGEST_TABLES_CAP),
    overview: {
      relations_total: relationsTotal,
      candidates_considered: distinctRelations.size,
      tables_with_av_overrides: tablesWithOverrides,
      autovacuum_disabled_tables: [...autovacuumDisabledTables],
      cost_delay_zero_tables: [...costDelayZeroTables],
      cost_delay_zero_toast_only_tables: [...costDelayZeroToastOnlyTables],
    },
  };
}

// --- the rule set (data) ---------------------------------------------------

const fmtInt = (n: number) => n.toLocaleString("en-US");

/**
 * Tier-1 (context-free) F001 rules plus the throughput and scale-factor rules.
 *
 * Every rule's recommendation obeys the never-recommend list (test-enforced):
 * never recommend cost_delay=0, never recommend more workers without more cost
 * budget, never recommend disabling autovacuum. Tier-2 (hardware-aware) numbers
 * degrade to conditional text because WI 269 environment context is not wired in.
 */
export const F001_RULES: AutovacuumRule[] = [
  {
    id: "autovacuum_off",
    severity: "CRITICAL",
    predicate: (c) => c.effective.autovacuum === false,
    message: () => ({
      conclusion:
        "autovacuum is turned OFF cluster-wide. Dead tuples, bloat and transaction-ID age will accumulate unchecked, risking bloat and eventual wraparound shutdown.",
      recommendation:
        "Re-enable autovacuum now: set autovacuum = on (and reload). If it was disabled for a one-off bulk load, that window is over — turn it back on.",
    }),
  },
  {
    id: "track_counts_off",
    severity: "CRITICAL",
    predicate: (c) => c.effective.track_counts === false,
    message: () => ({
      conclusion:
        "track_counts is OFF, so the statistics collector is blind and autovacuum cannot see dead-tuple counts — autovacuum effectively does not run.",
      recommendation: "Re-enable statistics collection: set track_counts = on (requires reload).",
    }),
  },
  {
    id: "cost_delay_zero",
    severity: "CRITICAL",
    predicate: (c) => c.throughput.throttling_disabled,
    message: (c) => ({
      conclusion:
        `Effective autovacuum cost delay is 0ms (${c.effective.cost_delay_ms.inheritance_chain}), so autovacuum I/O throttling is fully disabled. ` +
        "Unthrottled hint-bit writes force full-page images after checkpoints → WAL bursts → replication lag and IPC:SyncRep commit stalls.",
      recommendation:
        `Restore a small cost delay (the PG12+ default is ${F001_COST_DELAY_DEFAULT_MS}ms): set autovacuum_vacuum_cost_delay = ${F001_COST_DELAY_DEFAULT_MS}ms. ` +
        "Even one millisecond of cost delay keeps autovacuum from overwhelming the system (https://ardentperf.com/2026/04/12/zero-autovacuum_cost_delay-write-storms-and-you/).",
    }),
  },
  {
    id: "cost_delay_pg11_legacy",
    severity: "WARNING",
    // Only when NOT inheriting: an explicit 20ms autovacuum_vacuum_cost_delay is the pre-PG12 default.
    predicate: (c) =>
      c.effective.cost_delay_ms.inherited_from === null &&
      c.effective.cost_delay_ms.effective === F001_COST_DELAY_PG11_MS,
    message: () => ({
      conclusion:
        `autovacuum_vacuum_cost_delay is ${F001_COST_DELAY_PG11_MS}ms — the pre-PG12 default that caps vacuum at roughly 8 MB/s of reads. ` +
        "This is a common leftover after a major-version upgrade.",
      recommendation: `Lower it to the modern default: set autovacuum_vacuum_cost_delay = ${F001_COST_DELAY_DEFAULT_MS}ms.`,
    }),
  },
  {
    id: "cost_limit_low",
    severity: "WARNING",
    predicate: (c) =>
      c.effective.cost_limit.effective !== null &&
      c.effective.cost_limit.effective > 0 &&
      c.effective.cost_limit.effective <= F001_COST_LIMIT_LOW_MAX,
    message: (c) => ({
      conclusion:
        `Effective autovacuum cost limit is ${c.effective.cost_limit.effective} (${c.effective.cost_limit.inheritance_chain}) — at or below the default of ${F001_COST_LIMIT_DEFAULT}, ` +
        "which is likely too low for modern SSD/NVMe storage.",
      recommendation:
        "Raise the vacuum cost budget as a starting point: set autovacuum_vacuum_cost_limit = 1000 (range 1000–2000) " +
        "(https://www.enterprisedb.com/blog/autovacuum-tuning-basics). Size it to your disk's real write bandwidth once hardware facts are available.",
    }),
  },
  {
    id: "work_mem_inherits",
    severity: "WARNING",
    predicate: (c) => c.effective.work_mem_kb.inherited_from !== null,
    message: (c) => {
      const pg17Note =
        c.pgMajorVersion >= 17
          ? " On PG17 the 1 GB per-worker cap was removed, so each worker can now consume the full maintenance_work_mem × autovacuum_max_workers — set autovacuum_work_mem explicitly to bound total memory."
          : " After a PG17 upgrade note that the 1 GB per-worker cap is removed, so workers can suddenly consume the full maintenance_work_mem each.";
      return {
        conclusion:
          `autovacuum_work_mem = -1, so each autovacuum worker inherits maintenance_work_mem (${c.effective.work_mem_kb.inheritance_chain}).${pg17Note}`,
        recommendation:
          "Set autovacuum_work_mem explicitly (e.g. a bounded fraction of maintenance_work_mem) so autovacuum memory is decoupled from one-off maintenance operations and total worker memory stays predictable.",
      };
    },
  },
  {
    id: "scale_factor_high_global",
    severity: "WARNING",
    // Per WI #274: flag the global default "on a database with large tables".
    // The gate (mirroring the analyze twin) keeps it from firing on a stock
    // instance that has no large tables to accumulate dead-tuple debt.
    predicate: (c) =>
      c.effective.scale_factor !== null &&
      c.effective.scale_factor >= F001_SCALE_FACTOR_DEFAULT &&
      c.largestTables.some((t) => t.total_relation_size_bytes >= F001_LARGE_TABLE_BYTES),
    message: (c) => {
      const big = c.largestTables.filter(
        (t) => t.total_relation_size_bytes >= F001_LARGE_TABLE_BYTES && t.scale_factor_override === null,
      );
      const pct = Math.round((c.effective.scale_factor ?? 0) * 100);
      const defaultNote = c.effective.scale_factor === F001_SCALE_FACTOR_DEFAULT ? ", the default" : "";
      let detail = "";
      if (big.length > 0) {
        const top = big[0];
        const debt = Math.round(top.total_relation_size_bytes * (c.effective.scale_factor ?? 0));
        detail =
          ` The largest table without a per-table override, ${top.schema_name}.${top.table_name} (${top.total_relation_size_pretty}), ` +
          `would accumulate roughly ${formatBytes(debt)} of dead-tuple debt before autovacuum triggers.`;
      }
      return {
        conclusion:
          `Global autovacuum_vacuum_scale_factor is ${c.effective.scale_factor} (${pct}%)${defaultNote}. ` +
          `On large tables that means ${pct}% of the table must die before autovacuum triggers.${detail}`,
        recommendation:
          "For OLTP workloads lower the global autovacuum_vacuum_scale_factor to 0.01–0.05 and/or add per-table overrides on the largest tables " +
          "(alter table … set (autovacuum_vacuum_scale_factor = 0.01)). Review autovacuum_analyze_scale_factor the same way.",
      };
    },
  },
  {
    id: "analyze_scale_factor_high_global",
    severity: "NOTICE",
    predicate: (c) =>
      c.effective.analyze_scale_factor !== null &&
      c.effective.analyze_scale_factor >= F001_ANALYZE_SCALE_FACTOR_DEFAULT &&
      c.largestTables.some((t) => t.total_relation_size_bytes >= F001_LARGE_TABLE_BYTES),
    message: (c) => {
      const defaultNote =
        c.effective.analyze_scale_factor === F001_ANALYZE_SCALE_FACTOR_DEFAULT ? " (the default)" : "";
      return {
        conclusion:
          `Global autovacuum_analyze_scale_factor is ${c.effective.analyze_scale_factor}${defaultNote}, so on large tables statistics are refreshed only after a large fraction of rows change — stale stats degrade planning.`,
        recommendation:
          "Lower autovacuum_analyze_scale_factor (e.g. 0.02–0.05) globally and/or per-table on the largest tables so the planner sees fresher statistics.",
      };
    },
  },
  {
    id: "log_autovacuum_disabled",
    severity: "WARNING",
    predicate: (c) => c.effective.log_min_duration_ms !== null && c.effective.log_min_duration_ms < 0,
    message: (c) => {
      const pg15Note =
        c.pgMajorVersion >= 15
          ? " (PG15+ ships a 10-minute default, so this was explicitly turned off.)"
          : "";
      return {
        conclusion:
          `log_autovacuum_min_duration = -1 (disabled)${pg15Note}. Without autovacuum logging there is no record of what autovacuum did — a precondition for any autovacuum root-cause analysis is missing.`,
        recommendation:
          "Enable autovacuum logging: set log_autovacuum_min_duration = '10s' (or 0 to log every run) so slow/looping autovacuum is visible in the logs.",
      };
    },
  },
  {
    id: "naptime_high",
    severity: "NOTICE",
    predicate: (c) => c.effective.naptime_s !== null && c.effective.naptime_s > F001_NAPTIME_HIGH_S,
    message: (c) => ({
      conclusion:
        `autovacuum_naptime is ${c.effective.naptime_s}s (> ${F001_NAPTIME_HIGH_S}s). The autovacuum launcher checks each database only that often, slowing autovacuum's reaction loop; this is rarely justified.`,
      recommendation: `Lower autovacuum_naptime back toward the ${F001_NAPTIME_HIGH_S}s default unless there is a specific reason it was raised.`,
    }),
  },
  {
    id: "max_workers_default",
    severity: "NOTICE",
    predicate: (c) => c.effective.max_workers === F001_MAX_WORKERS_DEFAULT,
    message: () => ({
      conclusion:
        `autovacuum_max_workers is at the default of ${F001_MAX_WORKERS_DEFAULT}. On large hosts with many tables this is often too few, but the right number depends on the vCPU count.`,
      recommendation:
        "On servers with many cores consider raising autovacuum_max_workers (a rule of thumb is up to ~30% of vCPUs). " +
        "Because the cost budget is SHARED across all workers, ALWAYS raise autovacuum_vacuum_cost_limit proportionally when you add workers — otherwise the workers simply go slower. " +
        "Re-run with environment context (vCPU count) for a concrete number.",
    }),
  },
  {
    id: "insert_vacuum_disabled",
    severity: "NOTICE",
    appliesTo: (c) => c.pgMajorVersion >= 13 && c.effective.vacuum_insert_threshold !== null,
    predicate: (c) =>
      c.effective.vacuum_insert_threshold !== null && c.effective.vacuum_insert_threshold < 0,
    message: () => ({
      conclusion:
        "autovacuum_vacuum_insert_threshold = -1 disables insert-driven autovacuum (PG13+). Append-only / insert-mostly tables will not be vacuumed for freezing or visibility-map maintenance, hurting index-only scans and inviting a wraparound-vacuum surprise.",
      recommendation:
        "Re-enable insert-driven autovacuum: set autovacuum_vacuum_insert_threshold to a positive value (the default is 1000) so insert-heavy tables get frozen and their visibility maps stay current.",
    }),
  },
  {
    id: "dirty_budget_small_large_db",
    severity: "NOTICE",
    predicate: (c) =>
      !c.throughput.throttling_disabled &&
      c.throughput.dirty_write_mbps !== null &&
      c.throughput.dirty_write_mbps <= F001_DIRTY_WRITE_SMALL_MBPS &&
      c.databaseSizeBytes >= F001_LARGE_DB_BYTES,
    message: (c) => ({
      conclusion:
        `At the configured budget autovacuum can dirty at most ~${c.throughput.dirty_write_mbps} MB/s across ALL workers combined, while the database is ${formatBytes(c.databaseSizeBytes)}. ` +
        "That write budget is small relative to the data volume, so vacuum may struggle to keep up with write churn.",
      recommendation:
        "Increase throughput by raising autovacuum_vacuum_cost_limit (keep the cost delay on). Size the new limit to a sane fraction of the disk's real write bandwidth once hardware facts are available.",
    }),
  },
  {
    id: "per_table_cost_delay_zero",
    severity: "WARNING",
    predicate: (c) => c.reloptions.cost_delay_zero_tables.length > 0,
    message: (c) => {
      // Annotate the toast-level overrides so operators know the zero lives on
      // the table's toast relation, not the heap — the remediation differs.
      const toastOnly = new Set(c.reloptions.cost_delay_zero_toast_only_tables);
      const labeled = c.reloptions.cost_delay_zero_tables
        .map((t) => (toastOnly.has(t) ? `${t} (toast-level)` : t))
        .join(", ");
      const hasToast = toastOnly.size > 0;
      return {
        conclusion:
          `Per-table autovacuum_vacuum_cost_delay = 0 is set on: ${labeled}. ` +
          "On those tables autovacuum runs unthrottled, risking the same write-storm / replication-lag pattern as the global setting.",
        recommendation:
          `Remove the per-table override so the table inherits a throttled delay: alter table … reset (autovacuum_vacuum_cost_delay)` +
          (hasToast ? " (use reset (toast.autovacuum_vacuum_cost_delay) for the toast-level ones)" : "") +
          "; or set it to a small non-zero value. Never leave cost_delay at 0 (https://ardentperf.com/2026/04/12/zero-autovacuum_cost_delay-write-storms-and-you/).",
      };
    },
  },
  {
    id: "reloptions_overview",
    severity: "INFO",
    predicate: (c) => c.reloptions.tables_with_av_overrides > 0,
    message: (c) => {
      const parts = [
        `${fmtInt(c.reloptions.tables_with_av_overrides)} relation(s) carry per-table autovacuum_* overrides (out of ${fmtInt(c.reloptions.relations_total)} scanned).`,
      ];
      if (c.reloptions.autovacuum_disabled_tables.length > 0) {
        parts.push(
          `autovacuum is disabled per-table on: ${c.reloptions.autovacuum_disabled_tables.join(", ")} (see F003 for details).`,
        );
      }
      return {
        conclusion: parts.join(" "),
        recommendation:
          "Per-table overrides are good practice for hot/large tables; review that each override is still intentional and that no non-tiny table has autovacuum disabled.",
      };
    },
  },
];

/**
 * Evaluate the F001 rule set against the resolved context. Returns the fired
 * rules (in the rule-set order) plus the aggregate conclusions/recommendations
 * and the highest severity fired.
 *
 * Exported so the wording (which the console surfaces verbatim in auto-created
 * issues) and the never-recommend guarantees can be unit-tested without a DB.
 */
export function evaluateAutovacuumRules(ctx: AutovacuumRuleContext): {
  fired: FiredRule[];
  conclusions: string[];
  recommendations: string[];
  severity: F001Severity | null;
} {
  const fired: FiredRule[] = [];
  for (const rule of F001_RULES) {
    if (rule.appliesTo && !rule.appliesTo(ctx)) continue;
    if (!rule.predicate(ctx)) continue;
    const { conclusion, recommendation } = rule.message(ctx);
    fired.push({ id: rule.id, severity: rule.severity, conclusion, recommendation });
  }

  let severity: F001Severity | null = null;
  for (const f of fired) {
    if (severity === null || F001_SEVERITY_ORDER[f.severity] > F001_SEVERITY_ORDER[severity]) {
      severity = f.severity;
    }
  }

  return {
    fired,
    conclusions: fired.map((f) => f.conclusion),
    recommendations: fired.map((f) => f.recommendation),
    severity,
  };
}

/**
 * Generate F001 report - Autovacuum: current settings + configuration linting.
 *
 * Keeps the raw autovacuum settings dump (backward compatible `data` map) and
 * adds effective-value resolution, the data-driven rule engine, the effective
 * throughput budget, and the largest-tables scale-factor cross-reference.
 * SQL for the largest-tables/reloptions cross-reference is loaded from
 * config/pgwatch-prometheus/metrics.yml (pg_autovacuum_relopts metric).
 */
async function generateF001(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F001", "Autovacuum: current settings", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const allSettings = await getSettings(client, pgMajorVersion);

  // Filter autovacuum-related settings (raw dump; unchanged, backward compatible)
  const autovacuumSettings: Record<string, SettingInfo> = {};
  for (const [name, setting] of Object.entries(allSettings)) {
    if (name.includes("autovacuum") || name.includes("vacuum")) {
      autovacuumSettings[name] = setting;
    }
  }

  // Effective-value resolution + throughput math (context-free; always runs).
  const effective = resolveEffectiveAutovacuumSettings(allSettings);
  const throughput = computeThroughputBudget(effective);

  // Largest-tables scale-factor cross-reference + reloptions overview.
  // Optional/supplementary: a failure here must not sink the whole report.
  let largestTables: LargestTable[] = [];
  let overview: ReloptionsOverview = {
    relations_total: 0,
    candidates_considered: 0,
    tables_with_av_overrides: 0,
    autovacuum_disabled_tables: [],
    cost_delay_zero_tables: [],
    cost_delay_zero_toast_only_tables: [],
  };
  let databaseSizeBytes = 0;
  try {
    const relopts = await getAutovacuumRelopts(client, pgMajorVersion);
    largestTables = relopts.largestTables;
    overview = relopts.overview;
    const dbInfo = await getCurrentDatabaseInfo(client, pgMajorVersion);
    databaseSizeBytes = dbInfo.size_bytes;
  } catch (err) {
    console.error(`[F001] Warning: largest-tables cross-reference failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  const ctx: AutovacuumRuleContext = {
    pgMajorVersion,
    effective,
    throughput,
    largestTables,
    reloptions: overview,
    databaseSizeBytes,
    env: null,
  };

  const { fired, conclusions, recommendations, severity } = evaluateAutovacuumRules(ctx);

  report.results[nodeName] = {
    data: autovacuumSettings,
    postgres_version: postgresVersion,
    effective_values: effective,
    throughput_budget: throughput,
    conclusions,
    recommendations,
    settings_analysis: {
      severity,
      rules_fired: fired,
      largest_tables: largestTables,
      reloptions_overview: overview,
      database_size_bytes: databaseSizeBytes,
      database_size_pretty: formatBytes(databaseSizeBytes),
      thresholds: {
        cost_limit_low_max: F001_COST_LIMIT_LOW_MAX,
        scale_factor_default: F001_SCALE_FACTOR_DEFAULT,
        large_table_bytes: F001_LARGE_TABLE_BYTES,
        dirty_write_small_mbps: F001_DIRTY_WRITE_SMALL_MBPS,
        large_db_bytes: F001_LARGE_DB_BYTES,
      },
    },
  };

  return report;
}

/** Generate F002 report - transaction ID and MultiXact wraparound risk. */
async function generateF002(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F002", "Autovacuum: transaction ID and MultiXact wraparound", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const { settings, databases, tables, multixact_size, settings_available } = await getWraparoundData(client, pgMajorVersion);
  const { severity, conclusions, recommendations } = buildWraparoundConclusions(
    databases, tables, settings_available,
  );

  report.results[nodeName] = {
    data: {
      settings,
      databases,
      tables,
      multixact_size,
      settings_available,
      severity,
      thresholds: {
        wraparound_limit: F002_WRAPAROUND_LIMIT,
        critical_age: F002_CRITICAL_AGE,
        failsafe_high_pct: F002_FAILSAFE_HIGH_PCT,
        table_limit_per_age: 50,
      },
      conclusions,
      recommendations,
    },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Generate F003 report - Autovacuum: dead tuples
 *
 * Reads per-table dead-tuple counters from pg_stat_user_tables and per-table
 * autovacuum overrides from pg_class.reloptions. Flags tables where dead
 * tuples are high both in absolute terms and relative to live tuples, and
 * tables where autovacuum is disabled per-table (a classic footgun).
 *
 * Unlike F004/F005 (statistical bloat estimators), this check sees dead
 * tuples that have never been vacuumed.
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (pg_dead_tuples metric).
 */
async function generateF003(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F003", "Autovacuum: dead tuples", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;

  // Part 1: single-scan settings-aware trigger analysis + dead-tuple detection.
  // This ONE pg_stat_user_tables pass serves both the legacy dead-tuple flags
  // and the WI #271 per-table trigger math + queue aggregates (they share the
  // scan; see the pg_dead_tuples_keepup metric).
  const { tables, aggregates } = await getAutovacuumKeepup(client, pgMajorVersion);
  const { datname: dbName, size_bytes: dbSizeBytes } = await getCurrentDatabaseInfo(client, pgMajorVersion);

  // Part 2: single-snapshot queue / worker-saturation reading. These read
  // pg_stat_activity and pg_stat_progress_vacuum (bounded, cheap) — not a
  // second pg_stat_user_tables scan.
  const workers = await getAutovacuumWorkerSnapshot(client, pgMajorVersion);
  const blocked = await getAutovacuumBlocked(client, pgMajorVersion);
  const progress = await getVacuumProgress(client, pgMajorVersion);

  // Per-table starvation flag: over the vacuum trigger, last (auto)vacuum older
  // than F003_STARVATION_HOURS (or never), and no worker currently processing.
  // A recent MANUAL vacuum counts as service too, so use the newer of the
  // autovacuum / manual vacuum timestamps.
  const processing = new Set(progress.map((p) => `${p.schema_name}.${p.table_name}`));
  const nowEpoch = Math.floor(Date.now() / 1000);
  const starvationCutoff = F003_STARVATION_HOURS * 3600;
  for (const t of tables) {
    const lastVacuumEpoch = Math.max(t.last_autovacuum_epoch, t.last_vacuum_epoch);
    const stale = lastVacuumEpoch === 0 || nowEpoch - lastVacuumEpoch > starvationCutoff;
    t.starving =
      Boolean(t.over_vacuum_trigger) &&
      (t.n_live_tup + t.n_dead_tup) >= F003_KEEPUP_MIN_ROWS &&
      stale &&
      !processing.has(`${t.schema_name}.${t.table_name}`);
  }
  const starvingCount = tables.filter((t) => t.starving).length;

  const keepup = judgeKeepingUp(aggregates, workers, blocked, progress, starvingCount);

  const flaggedCount = tables.filter((t) => t.exceeds_dead_tuple_thresholds).length;
  const autovacuumDisabledCount = tables.filter((t) => t.autovacuum_disabled).length;
  const autovacuumDisabledFlaggedCount = tables.filter((t) => t.autovacuum_disabled_flagged).length;
  const totalDeadTuples = tables.reduce((sum, t) => sum + t.n_dead_tup, 0);

  // Existing dead-tuple / disabled-autovacuum conclusions, then the WI #271
  // keeping-up conclusions (trigger, saturation, anti-wraparound, blocked).
  const base = buildDeadTuplesConclusions(tables);
  const keep = buildKeepupConclusions(tables, keepup);
  const conclusions = [...base.conclusions, ...keep.conclusions];
  const recommendations = [...base.recommendations, ...keep.recommendations];

  const dbEntry = {
    dead_tuples_tables: tables,
    total_count: tables.length,
    flagged_count: flaggedCount,
    autovacuum_disabled_count: autovacuumDisabledCount,
    autovacuum_disabled_flagged_count: autovacuumDisabledFlaggedCount,
    total_dead_tuples: totalDeadTuples,
    thresholds: {
      dead_tuples_min: F003_DEAD_TUPLES_MIN,
      dead_pct_min: F003_DEAD_PCT_MIN,
      autovacuum_disabled_min_rows: F003_AUTOVACUUM_DISABLED_MIN_ROWS,
    },
    trigger_thresholds: {
      top_k: F003_TOP_K,
      keepup_min_rows: F003_KEEPUP_MIN_ROWS,
      queue_saturation_multiplier: F003_QUEUE_SATURATION_MULTIPLIER,
      starvation_hours: F003_STARVATION_HOURS,
    },
    autovacuum_keepup: keepup,
    conclusions,
    recommendations,
    database_size_bytes: dbSizeBytes,
    database_size_pretty: formatBytes(dbSizeBytes),
  };

  report.results[nodeName] = {
    data: { [dbName]: dbEntry },
    postgres_version: postgresVersion,
  };

  return report;
}

export const F009_NOTICE_FRACTION = 0.10;
export const F009_WARNING_FRACTION = 0.50;
export const F009_CRITICAL_FRACTION = 0.80;
export const F009_ACTIVITY_NOTICE_SECONDS = SECONDS_PER_HOUR;

export type F009Severity = "OK" | "NOTICE" | "WARNING" | "CRITICAL";

export interface F009Component {
  age_tx: number;
  count: number;
  top_blocker: Record<string, unknown> | null;
}

export interface F009Components {
  pg_stat_activity: F009Component;
  pg_replication_slots: F009Component;
  pg_replication_slots_catalog: F009Component;
  pg_stat_replication: F009Component;
  pg_prepared_xacts: F009Component;
}

export interface F009DominantHolder {
  source: "slot" | "prepared" | "activity" | "replication";
  component: keyof F009Components;
  horizon_type: "data" | "catalog";
  age_tx: number;
  detail: Record<string, unknown>;
}

const F009_SEVERITY_RANK: Record<F009Severity, number> = {
  OK: 0,
  NOTICE: 1,
  WARNING: 2,
  CRITICAL: 3,
};

function numberValue(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function parseJsonObject(value: unknown): Record<string, any> {
  if (value && typeof value === "object") return value as Record<string, any>;
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      // Graceful fallback for drivers that return malformed JSON as text.
    }
  }
  return {};
}

function normalizeF009Components(value: unknown): F009Components {
  const raw = parseJsonObject(value);
  const component = (name: keyof F009Components): F009Component => {
    const item = parseJsonObject(raw[name]);
    const detail = item.top_blocker;
    return {
      age_tx: numberValue(item.age_tx),
      count: numberValue(item.count),
      top_blocker: detail && typeof detail === "object" ? detail : null,
    };
  };
  return {
    pg_stat_activity: component("pg_stat_activity"),
    pg_replication_slots: component("pg_replication_slots"),
    pg_replication_slots_catalog: component("pg_replication_slots_catalog"),
    pg_stat_replication: component("pg_stat_replication"),
    pg_prepared_xacts: component("pg_prepared_xacts"),
  };
}

export function getF009Severity(ageTx: number, freezeMaxAge: number): F009Severity {
  if (freezeMaxAge <= 0) return "OK";
  const fraction = ageTx / freezeMaxAge;
  if (fraction >= F009_CRITICAL_FRACTION) return "CRITICAL";
  if (fraction >= F009_WARNING_FRACTION) return "WARNING";
  if (fraction >= F009_NOTICE_FRACTION) return "NOTICE";
  return "OK";
}

export function selectF009DominantHolder(components: F009Components): F009DominantHolder | null {
  const candidates: Array<F009DominantHolder & { priority: number }> = [];
  const add = (
    source: F009DominantHolder["source"],
    component: keyof F009Components,
    horizonType: F009DominantHolder["horizon_type"],
    priority: number,
  ) => {
    const item = components[component];
    if (item.count > 0 && item.top_blocker) {
      candidates.push({ source, component, horizon_type: horizonType, age_tx: item.age_tx, detail: item.top_blocker, priority });
    }
  };
  // PFR-aligned tie-break: slot > prepared > activity > replication.
  add("slot", "pg_replication_slots", "data", 4);
  add("slot", "pg_replication_slots_catalog", "catalog", 4);
  add("prepared", "pg_prepared_xacts", "data", 3);
  add("activity", "pg_stat_activity", "data", 2);
  add("replication", "pg_stat_replication", "data", 1);
  candidates.sort((a, b) => b.age_tx - a.age_tx || b.priority - a.priority || a.component.localeCompare(b.component));
  if (candidates.length === 0) return null;
  const { priority: _priority, ...dominant } = candidates[0];
  return dominant;
}

function sqlLiteral(value: unknown): string {
  return String(value ?? "").replace(/'/g, "''");
}

export function buildF009Analysis(
  components: F009Components,
  dominantHolder: F009DominantHolder | null,
  dataSeverity: F009Severity,
  catalogSeverity: F009Severity,
  options: { limitedVisibility: boolean; historyDetected: boolean; skipped: boolean },
): { conclusions: string[]; recommendations: string[] } {
  const conclusions: string[] = [];
  const recommendations: string[] = [];

  if (options.skipped) {
    return { conclusions: ["INFO: xmin horizon analysis is primary-only; this replica was skipped."], recommendations };
  }
  if (options.limitedVisibility) {
    conclusions.push("NOTICE: the check ran with limited visibility; sessions owned by other users may hide query text or xmin details. Grant pg_monitor or pg_read_all_stats for complete RCA context.");
  }
  if (options.historyDetected) {
    conclusions.push("INFO: pg_flight_recorder detected — historical xmin timeline is available via pgfr_analyze.xmin_horizon_history().");
  }

  const dataAge = Math.max(
    components.pg_stat_activity.age_tx,
    components.pg_replication_slots.age_tx,
    components.pg_stat_replication.age_tx,
    components.pg_prepared_xacts.age_tx,
  );
  const catalogAge = components.pg_replication_slots_catalog.age_tx;
  if (dataSeverity !== "OK") conclusions.push(`${dataSeverity}: data xmin horizon age is ${dataAge} transactions.`);
  if (catalogSeverity !== "OK") conclusions.push(`${catalogSeverity}: catalog xmin horizon age is ${catalogAge} transactions.`);

  const activity = components.pg_stat_activity.top_blocker;
  const xactAgeSeconds = numberValue(activity?.xact_age_seconds);
  if (activity && xactAgeSeconds >= F009_ACTIVITY_NOTICE_SECONDS) {
    conclusions.push(`NOTICE: activity PID ${activity.pid} has held a transaction open for ${xactAgeSeconds} seconds.`);
  }

  if (!dominantHolder) {
    if (conclusions.length === 0) conclusions.push("OK: no xmin horizon blocker is currently visible.");
    return { conclusions, recommendations };
  }

  const d = dominantHolder.detail;
  conclusions.unshift(`${dataSeverity === "OK" && catalogSeverity === "OK" ? "INFO" : "CAUSE"}: dominant xmin holder is ${dominantHolder.source} (${dominantHolder.age_tx} transactions, ${dominantHolder.horizon_type} horizon).`);
  if (dominantHolder.source === "activity") {
    const pid = numberValue(d.pid);
    const backendType = String(d.backend_type || "").toLowerCase();
    const state = String(d.state || "").toLowerCase();
    if (backendType.includes("autovacuum")) {
      recommendations.push(`Inspect pg_stat_progress_vacuum for autovacuum worker PID ${pid}; do not kill it until its progress and table size are understood.`);
    } else if (state.includes("idle in transaction")) {
      recommendations.push(`Terminate the idle-in-transaction holder with SELECT pg_terminate_backend(${pid}); pg_cancel_backend() is a no-op while it is idle. Prevent recurrence with idle_in_transaction_session_timeout.`);
    } else {
      recommendations.push(`Cancel the active statement first with SELECT pg_cancel_backend(${pid}); if backend_xmin reappears, the query is inside an explicit BEGIN and SELECT pg_terminate_backend(${pid}) may be required. Prevent recurrence with statement_timeout and, on PostgreSQL 17+, transaction_timeout.`);
    }
  } else if (dominantHolder.source === "slot") {
    const slotName = sqlLiteral(d.slot_name);
    recommendations.push(`Confirm subscriber state for replication slot '${slotName}' before intervening. Then advance it with pg_replication_slot_advance() or, only if obsolete, SELECT pg_drop_replication_slot('${slotName}'). Slot status=${d.status || "unknown"}, wal_status=${d.wal_status || "unknown"}.`);
  } else if (dominantHolder.source === "replication") {
    recommendations.push(`On standby '${d.application_name || "unknown"}', find and cancel the long query responsible for hot_standby_feedback. Do not disable hot_standby_feedback as a first response; that trades xmin holdback for recovery-conflict cancellations.`);
  } else {
    const gid = sqlLiteral(d.gid);
    recommendations.push(`After confirming the transaction outcome, resolve the prepared transaction with ROLLBACK PREPARED '${gid}'. Owner=${d.owner || "unknown"}, database=${d.database || "unknown"}, prepared_at=${d.prepared_at || "unknown"}.`);
  }
  return { conclusions, recommendations };
}

/** Generate F009 - xmin horizon age and top blockers from one database snapshot. */
export async function generateF009(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F009", "Xmin horizon and blockers", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const result = await client.query(getMetricSql(METRIC_NAMES.F009, pgMajorVersion));
  const row = result.rows[0] || {};
  const components = normalizeF009Components(row.components);
  const skipped = toBool(row.is_in_recovery);
  const freezeMaxAge = numberValue(row.autovacuum_freeze_max_age);
  const dataAge = numberValue(row.data_horizon_age_tx);
  const catalogAge = numberValue(row.catalog_horizon_age_tx);
  const dataSeverity = skipped ? "OK" : getF009Severity(dataAge, freezeMaxAge);
  const catalogSeverity = skipped ? "OK" : getF009Severity(catalogAge, freezeMaxAge);
  const activityAgeSeconds = numberValue(components.pg_stat_activity.top_blocker?.xact_age_seconds);
  let severity = F009_SEVERITY_RANK[dataSeverity] >= F009_SEVERITY_RANK[catalogSeverity] ? dataSeverity : catalogSeverity;
  if (severity === "OK" && activityAgeSeconds >= F009_ACTIVITY_NOTICE_SECONDS) severity = "NOTICE";
  const dominantHolder = skipped ? null : selectF009DominantHolder(components);
  const limitedVisibility = !toBool(row.has_full_visibility);
  const historyDetected = toBool(row.pg_flight_recorder_detected);
  const { conclusions, recommendations } = buildF009Analysis(
    components, dominantHolder, dataSeverity, catalogSeverity,
    { limitedVisibility, historyDetected, skipped },
  );

  const data: Record<string, unknown> = {
    skipped,
    skip_reason: row.skip_reason || null,
    snapshot_xmin: row.snapshot_xmin === null || row.snapshot_xmin === undefined ? null : numberValue(row.snapshot_xmin),
    data_horizon_age_tx: dataAge,
    catalog_horizon_age_tx: catalogAge,
    autovacuum_freeze_max_age: freezeMaxAge,
    severity,
    data_horizon_severity: dataSeverity,
    catalog_horizon_severity: catalogSeverity,
    visibility: limitedVisibility ? "limited" : "full",
    query_preview_captured: toBool(row.query_preview_enabled) && !limitedVisibility,
    components,
    dominant_holder: dominantHolder,
    conclusions,
    recommendations,
  };
  if (historyDetected) {
    data.history = {
      available: true,
      source: "pg_flight_recorder",
      message: "Historical xmin timeline is available via pgfr_analyze.xmin_horizon_history().",
    };
  }
  report.results[nodeName] = { data, postgres_version: postgresVersion };
  return report;
}

/**
 * Generate F004 report - Autovacuum: heap bloat (estimated)
 *
 * Estimates table bloat based on statistical analysis of table pages vs expected pages.
 * Uses pg_stats for column statistics to estimate row sizes.
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (pg_table_bloat metric).
 */
type BloatCheckReason = "missing_schema" | "missing_view" | "missing_grant" | "query_error";

interface BloatCheckStatus {
  ok: boolean;
  reason: BloatCheckReason | null;
  error: string | null;
}

function bloatErrorStatus(err: unknown): BloatCheckStatus {
  const error = err instanceof Error ? err.message : String(err);
  const code = typeof err === "object" && err !== null && "code" in err
    ? String((err as { code?: unknown }).code || "")
    : "";
  const normalized = error.toLowerCase();

  let reason: BloatCheckReason = "query_error";
  if (code === "3F000" || normalized.includes('schema "postgres_ai" does not exist')) {
    reason = "missing_schema";
  } else if (code === "42P01" || normalized.includes('relation "postgres_ai.pg_statistic" does not exist')) {
    reason = "missing_view";
  } else if (code === "42501" || normalized.includes("permission denied")) {
    reason = "missing_grant";
  }

  return { ok: false, reason, error };
}

async function getBloatCheckStatus(client: Client): Promise<BloatCheckStatus> {
  try {
    const result = await client.query(`
      select
        to_regnamespace('postgres_ai') is not null as schema_exists,
        case
          when to_regnamespace('postgres_ai') is null then false
          else has_schema_privilege(current_user, 'postgres_ai', 'USAGE')
        end as schema_usage,
        case
          when to_regnamespace('postgres_ai') is null then false
          when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then false
          else to_regclass('postgres_ai.pg_statistic') is not null
        end as view_exists,
        case
          when to_regnamespace('postgres_ai') is null then false
          when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then false
          when to_regclass('postgres_ai.pg_statistic') is null then false
          else has_table_privilege(current_user, 'postgres_ai.pg_statistic', 'SELECT')
        end as view_select
    `);
    const capability = result.rows[0] || {};

    if (!capability.schema_exists) {
      return { ok: false, reason: "missing_schema", error: 'schema "postgres_ai" does not exist' };
    }
    if (!capability.schema_usage) {
      return { ok: false, reason: "missing_grant", error: "permission denied for schema postgres_ai" };
    }
    if (!capability.view_exists) {
      return { ok: false, reason: "missing_view", error: 'relation "postgres_ai.pg_statistic" does not exist' };
    }
    if (!capability.view_select) {
      return { ok: false, reason: "missing_grant", error: "permission denied for relation postgres_ai.pg_statistic" };
    }
    return { ok: true, reason: null, error: null };
  } catch (err) {
    return bloatErrorStatus(err);
  }
}

async function generateF004(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F004", "Autovacuum: heap bloat (estimated)", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10);

  interface BloatedTable {
    schema_name: string;
    table_name: string;
    real_size: number;
    extra_size: number;
    extra_pct: number;
    bloat_size: number;
    bloat_pct: number;
    fillfactor: number;
    last_vacuum: string | null;
    last_vacuum_epoch: number;
    real_size_pretty: string;
    extra_size_pretty: string;
    bloat_size_pretty: string;
  }

  let bloatedTables: BloatedTable[] = [];
  let status = await getBloatCheckStatus(client);

  try {
    if (!status.ok) throw Object.assign(new Error(status.error || "Bloat prerequisites unavailable"), {
      code: status.reason === "missing_schema" ? "3F000"
        : status.reason === "missing_view" ? "42P01"
        : status.reason === "missing_grant" ? "42501"
        : undefined,
    });
    // Get bloat data
    const sql = getMetricSql(METRIC_NAMES.F004, pgMajorVersion);
    const bloatResult = await client.query(sql);

    // Get vacuum stats for all tables
    const vacuumStatsResult = await client.query(`
      SELECT schemaname, relname, last_vacuum, last_autovacuum
      FROM pg_stat_user_tables
    `);
    const vacuumStats = new Map<string, { last_vacuum: string | null; last_vacuum_epoch: number }>();
    for (const row of vacuumStatsResult.rows) {
      const key = `${row.schemaname}.${row.relname}`;
      // Use last_autovacuum if last_vacuum is null, otherwise prefer last_vacuum
      const vacuumTime = row.last_vacuum || row.last_autovacuum;
      vacuumStats.set(key, {
        last_vacuum: vacuumTime ? new Date(vacuumTime).toISOString() : null,
        last_vacuum_epoch: vacuumTime ? Math.floor(new Date(vacuumTime).getTime() / 1000) : 0,
      });
    }

    bloatedTables = bloatResult.rows.map((row) => {
      const t = transformMetricRow(row);
      const schemaName = String(t.schemaname || "");
      const tableName = String(t.tblname || "");
      const realSizeBytes = Math.round((parseFloat(String(t.real_size_mib)) || 0) * 1024 * 1024);
      const extraSize = parseInt(String(t.extra_size || 0), 10);
      const bloatSize = parseInt(String(t.bloat_size || 0), 10);

      const vacuumInfo = vacuumStats.get(`${schemaName}.${tableName}`) || {
        last_vacuum: null,
        last_vacuum_epoch: 0,
      };

      return {
        schema_name: schemaName,
        table_name: tableName,
        real_size: realSizeBytes,
        extra_size: extraSize,
        extra_pct: parseFloat(String(t.extra_pct)) || 0,
        bloat_size: bloatSize,
        bloat_pct: parseFloat(String(t.bloat_pct)) || 0,
        fillfactor: parseInt(String(t.fillfactor || 100), 10),
        last_vacuum: vacuumInfo.last_vacuum,
        last_vacuum_epoch: vacuumInfo.last_vacuum_epoch,
        real_size_pretty: formatBytes(realSizeBytes),
        extra_size_pretty: formatBytes(extraSize),
        bloat_size_pretty: formatBytes(bloatSize),
      };
    });
  } catch (err) {
    status = bloatErrorStatus(err);
    const errorMsg = status.error || "Unknown error";
    console.error(`[F004] Error estimating table bloat: ${errorMsg}`);
    if (errorMsg.includes("postgres_ai.")) {
      console.error(`  Hint: Run "postgresai prepare-db <connection>" to create required objects.`);
    }
  }

  // Get database info
  const { datname: dbName, size_bytes: dbSizeBytes } = await getCurrentDatabaseInfo(client, pgMajorVersion);

  // Calculate totals
  const totalCount = bloatedTables.length;
  const totalBloatSizeBytes = bloatedTables.reduce((sum, t) => sum + t.bloat_size, 0);

  const dbEntry = {
    status,
    bloated_tables: bloatedTables,
    total_count: totalCount,
    total_bloat_size_bytes: totalBloatSizeBytes,
    total_bloat_size_pretty: formatBytes(totalBloatSizeBytes),
    database_size_bytes: dbSizeBytes,
    database_size_pretty: formatBytes(dbSizeBytes),
  };

  report.results[nodeName] = {
    data: { [dbName]: dbEntry },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Generate F005 report - Autovacuum: index bloat (estimated)
 *
 * Estimates B-tree index bloat based on statistical analysis of index pages vs expected pages.
 * SQL loaded from config/pgwatch-prometheus/metrics.yml (pg_btree_bloat metric).
 */
async function generateF005(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("F005", "Autovacuum: index bloat (estimated)", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10);

  interface BloatedIndex {
    schema_name: string;
    table_name: string;
    index_name: string;
    real_size: number;
    table_size: number;
    extra_size: number;
    extra_pct: number;
    bloat_size: number;
    bloat_pct: number;
    fillfactor: number;
    last_vacuum: string | null;
    last_vacuum_epoch: number;
    real_size_pretty: string;
    table_size_pretty: string;
    extra_size_pretty: string;
    bloat_size_pretty: string;
  }

  let bloatedIndexes: BloatedIndex[] = [];
  let status = await getBloatCheckStatus(client);

  try {
    if (!status.ok) throw Object.assign(new Error(status.error || "Bloat prerequisites unavailable"), {
      code: status.reason === "missing_schema" ? "3F000"
        : status.reason === "missing_view" ? "42P01"
        : status.reason === "missing_grant" ? "42501"
        : undefined,
    });
    // Get bloat data
    const sql = getMetricSql(METRIC_NAMES.F005, pgMajorVersion);
    const bloatResult = await client.query(sql);

    // Get vacuum stats for all tables (indexes inherit vacuum time from their table)
    const vacuumStatsResult = await client.query(`
      SELECT schemaname, relname, last_vacuum, last_autovacuum
      FROM pg_stat_user_tables
    `);
    const vacuumStats = new Map<string, { last_vacuum: string | null; last_vacuum_epoch: number }>();
    for (const row of vacuumStatsResult.rows) {
      const key = `${row.schemaname}.${row.relname}`;
      const vacuumTime = row.last_vacuum || row.last_autovacuum;
      vacuumStats.set(key, {
        last_vacuum: vacuumTime ? new Date(vacuumTime).toISOString() : null,
        last_vacuum_epoch: vacuumTime ? Math.floor(new Date(vacuumTime).getTime() / 1000) : 0,
      });
    }

    bloatedIndexes = bloatResult.rows.map((row) => {
      const t = transformMetricRow(row);
      const schemaName = String(t.schemaname || "");
      const tableName = String(t.tblname || "");
      const indexName = String(t.idxname || "");
      const realSizeBytes = Math.round((parseFloat(String(t.real_size_mib)) || 0) * 1024 * 1024);
      const tableSizeBytes = Math.round((parseFloat(String(t.table_size_mib)) || 0) * 1024 * 1024);
      const extraSize = parseInt(String(t.extra_size || 0), 10);
      const bloatSize = parseInt(String(t.bloat_size || 0), 10);

      const vacuumInfo = vacuumStats.get(`${schemaName}.${tableName}`) || {
        last_vacuum: null,
        last_vacuum_epoch: 0,
      };

      return {
        schema_name: schemaName,
        table_name: tableName,
        index_name: indexName,
        real_size: realSizeBytes,
        table_size: tableSizeBytes,
        extra_size: extraSize,
        extra_pct: parseFloat(String(t.extra_pct)) || 0,
        bloat_size: bloatSize,
        bloat_pct: parseFloat(String(t.bloat_pct)) || 0,
        fillfactor: parseInt(String(t.fillfactor || 90), 10),
        last_vacuum: vacuumInfo.last_vacuum,
        last_vacuum_epoch: vacuumInfo.last_vacuum_epoch,
        real_size_pretty: formatBytes(realSizeBytes),
        table_size_pretty: formatBytes(tableSizeBytes),
        extra_size_pretty: formatBytes(extraSize),
        bloat_size_pretty: formatBytes(bloatSize),
      };
    });
  } catch (err) {
    status = bloatErrorStatus(err);
    const errorMsg = status.error || "Unknown error";
    console.error(`[F005] Error estimating index bloat: ${errorMsg}`);
    if (errorMsg.includes("postgres_ai.")) {
      console.error(`  Hint: Run "postgresai prepare-db <connection>" to create required objects.`);
    }
  }

  // Get database info
  const { datname: dbName, size_bytes: dbSizeBytes } = await getCurrentDatabaseInfo(client, pgMajorVersion);

  // Calculate totals
  const totalCount = bloatedIndexes.length;
  const totalBloatSizeBytes = bloatedIndexes.reduce((sum, idx) => sum + idx.bloat_size, 0);

  const dbEntry = {
    status,
    bloated_indexes: bloatedIndexes,
    total_count: totalCount,
    total_bloat_size_bytes: totalBloatSizeBytes,
    total_bloat_size_pretty: formatBytes(totalBloatSizeBytes),
    database_size_bytes: dbSizeBytes,
    database_size_pretty: formatBytes(dbSizeBytes),
  };

  report.results[nodeName] = {
    data: { [dbName]: dbEntry },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Generate G001 report - Memory-related settings
 */
async function generateG001(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("G001", "Memory-related settings", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const allSettings = await getSettings(client, pgMajorVersion);

  // Memory-related setting names
  const memorySettingNames = [
    "shared_buffers",
    "work_mem",
    "maintenance_work_mem",
    "effective_cache_size",
    "wal_buffers",
    "temp_buffers",
    "max_connections",
    "autovacuum_work_mem",
    "hash_mem_multiplier",
    "logical_decoding_work_mem",
    "max_stack_depth",
    "max_prepared_transactions",
    "max_locks_per_transaction",
    "max_pred_locks_per_transaction",
  ];

  const memorySettings: Record<string, SettingInfo> = {};
  for (const name of memorySettingNames) {
    if (allSettings[name]) {
      memorySettings[name] = allSettings[name];
    }
  }

  // Calculate memory usage estimates
  interface MemoryUsage {
    shared_buffers_bytes: number;
    shared_buffers_pretty: string;
    wal_buffers_bytes: number;
    wal_buffers_pretty: string;
    shared_memory_total_bytes: number;
    shared_memory_total_pretty: string;
    work_mem_per_connection_bytes: number;
    work_mem_per_connection_pretty: string;
    max_work_mem_usage_bytes: number;
    max_work_mem_usage_pretty: string;
    maintenance_work_mem_bytes: number;
    maintenance_work_mem_pretty: string;
    effective_cache_size_bytes: number;
    effective_cache_size_pretty: string;
  }

  let memoryUsage: MemoryUsage | Record<string, never> = {};
  let memoryError: string | null = null;

  try {
    // Get actual byte values from PostgreSQL
    const memQuery = await client.query(`
      select
        pg_size_bytes(current_setting('shared_buffers')) as shared_buffers_bytes,
        pg_size_bytes(current_setting('wal_buffers')) as wal_buffers_bytes,
        pg_size_bytes(current_setting('work_mem')) as work_mem_bytes,
        pg_size_bytes(current_setting('maintenance_work_mem')) as maintenance_work_mem_bytes,
        pg_size_bytes(current_setting('effective_cache_size')) as effective_cache_size_bytes,
        current_setting('max_connections')::int as max_connections
    `);

    if (memQuery.rows.length > 0) {
      const row = memQuery.rows[0];
      const sharedBuffersBytes = parseInt(row.shared_buffers_bytes, 10);
      const walBuffersBytes = parseInt(row.wal_buffers_bytes, 10);
      const workMemBytes = parseInt(row.work_mem_bytes, 10);
      const maintenanceWorkMemBytes = parseInt(row.maintenance_work_mem_bytes, 10);
      const effectiveCacheSizeBytes = parseInt(row.effective_cache_size_bytes, 10);
      const maxConnections = row.max_connections;

      const sharedMemoryTotal = sharedBuffersBytes + walBuffersBytes;
      const maxWorkMemUsage = workMemBytes * maxConnections;

      memoryUsage = {
        shared_buffers_bytes: sharedBuffersBytes,
        shared_buffers_pretty: formatBytes(sharedBuffersBytes),
        wal_buffers_bytes: walBuffersBytes,
        wal_buffers_pretty: formatBytes(walBuffersBytes),
        shared_memory_total_bytes: sharedMemoryTotal,
        shared_memory_total_pretty: formatBytes(sharedMemoryTotal),
        work_mem_per_connection_bytes: workMemBytes,
        work_mem_per_connection_pretty: formatBytes(workMemBytes),
        max_work_mem_usage_bytes: maxWorkMemUsage,
        max_work_mem_usage_pretty: formatBytes(maxWorkMemUsage),
        maintenance_work_mem_bytes: maintenanceWorkMemBytes,
        maintenance_work_mem_pretty: formatBytes(maintenanceWorkMemBytes),
        effective_cache_size_bytes: effectiveCacheSizeBytes,
        effective_cache_size_pretty: formatBytes(effectiveCacheSizeBytes),
      };
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[G001] Error calculating memory usage: ${errorMsg}`);
    memoryError = errorMsg;
  }

  report.results[nodeName] = {
    data: {
      settings: memorySettings,
      analysis: {
        estimated_total_memory_usage: memoryUsage,
        ...(memoryError && { error: memoryError }),
      },
    },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Generate G003 report - Timeouts, locks, deadlocks
 *
 * Collects timeout and lock-related settings, plus deadlock statistics.
 */
async function generateG003(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("G003", "Timeouts, locks, deadlocks", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const pgMajorVersion = parseInt(postgresVersion.server_major_ver, 10) || 16;
  const allSettings = await getSettings(client, pgMajorVersion);

  // Timeout and lock-related setting names
  const lockTimeoutSettingNames = [
    "lock_timeout",
    "statement_timeout",
    "idle_in_transaction_session_timeout",
    "idle_session_timeout",
    "deadlock_timeout",
    "max_locks_per_transaction",
    "max_pred_locks_per_transaction",
    "max_pred_locks_per_relation",
    "max_pred_locks_per_page",
    "log_lock_waits",
    "transaction_timeout",
  ];

  const lockSettings: Record<string, SettingInfo> = {};
  for (const name of lockTimeoutSettingNames) {
    if (allSettings[name]) {
      lockSettings[name] = allSettings[name];
    }
  }

  // Get deadlock statistics from pg_stat_database
  let deadlockStats: {
    deadlocks: number;
    conflicts: number;
    stats_reset: string | null;
  } | null = null;
  let deadlockError: string | null = null;

  try {
    const statsResult = await client.query(`
      select
        coalesce(sum(deadlocks), 0)::bigint as deadlocks,
        coalesce(sum(conflicts), 0)::bigint as conflicts,
        min(stats_reset)::text as stats_reset
      from pg_stat_database
      where datname = current_database()
    `);
    if (statsResult.rows.length > 0) {
      const row = statsResult.rows[0];
      deadlockStats = {
        deadlocks: parseInt(row.deadlocks, 10),
        conflicts: parseInt(row.conflicts, 10),
        stats_reset: row.stats_reset || null,
      };
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[G003] Error querying deadlock stats: ${errorMsg}`);
    deadlockError = errorMsg;
  }

  report.results[nodeName] = {
    data: {
      settings: lockSettings,
      deadlock_stats: deadlockStats,
      ...(deadlockError && { deadlock_stats_error: deadlockError }),
    },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Get I/O statistics from pg_stat_io (PostgreSQL 16+).
 * Uses 'pg_stat_io' metric from metrics.yml.
 *
 * @param client - Connected PostgreSQL client
 * @param pgMajorVersion - PostgreSQL major version; defaults to 0 so omitted versions return unavailable
 * @param metricSqlOverride - Optional SQL override; empty or placeholder SQL returns [] without querying
 * @returns Array of I/O stats by backend type, or empty array if unavailable
 */
export async function getIOStatistics(
  client: Client,
  pgMajorVersion: number = 0,
  metricSqlOverride?: string
): Promise<BackendIOStats[]> {
  // pg_stat_io requires PostgreSQL 16+
  if (pgMajorVersion < 16) {
    return [];
  }

  try {
    const sql = metricSqlOverride ?? getMetricSql(METRIC_NAMES.I001, pgMajorVersion);
    // Skip if metric returns empty/placeholder SQL
    if (!sql || sql.trim().startsWith(";")) {
      return [];
    }

    const result = await client.query(sql);
    return result.rows.map((row) => {
      const transformed = transformMetricRow(row);
      return {
        backend_type: String(transformed.backend_type || "unknown"),
        reads: parseInt(String(transformed.reads || 0), 10),
        read_bytes_mb: parseInt(String(transformed.read_bytes_mb || 0), 10),
        read_time_ms: parseInt(String(transformed.read_time_ms || 0), 10),
        writes: parseInt(String(transformed.writes || 0), 10),
        write_bytes_mb: parseInt(String(transformed.write_bytes_mb || 0), 10),
        write_time_ms: parseInt(String(transformed.write_time_ms || 0), 10),
        writebacks: parseInt(String(transformed.writebacks || 0), 10),
        writeback_bytes_mb: parseInt(String(transformed.writeback_bytes_mb || 0), 10),
        writeback_time_ms: parseInt(String(transformed.writeback_time_ms || 0), 10),
        fsyncs: parseInt(String(transformed.fsyncs || 0), 10),
        fsync_time_ms: parseInt(String(transformed.fsync_time_ms || 0), 10),
        extends: parseInt(String(transformed.extends || 0), 10),
        extend_bytes_mb: parseInt(String(transformed.extend_bytes_mb || 0), 10),
        hits: parseInt(String(transformed.hits || 0), 10),
        evictions: parseInt(String(transformed.evictions || 0), 10),
        reuses: parseInt(String(transformed.reuses || 0), 10),
      };
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.log(`[I001] Error fetching I/O statistics: ${errorMsg}`);
    return [];
  }
}

/**
 * Generate I001 report - I/O statistics (pg_stat_io)
 *
 * This report collects I/O statistics from pg_stat_io (PostgreSQL 16+),
 * providing insights into read/write operations by backend type.
 *
 * @param client - Connected PostgreSQL client
 * @param nodeName - Node name for the report payload
 * @returns I001 report payload
 */
async function generateI001(client: Client, nodeName: string): Promise<Report> {
  const report = createBaseReport("I001", "I/O statistics (pg_stat_io)", nodeName);
  const postgresVersion = await getPostgresVersion(client);
  const parsedPgMajorVersion = parseInt(postgresVersion.server_major_ver, 10);
  const pgMajorVersion = Number.isFinite(parsedPgMajorVersion) ? parsedPgMajorVersion : 0;

  // pg_stat_io requires PostgreSQL 16+
  if (pgMajorVersion < 16) {
    report.results[nodeName] = {
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
      postgres_version: postgresVersion,
    };
    return report;
  }

  const ioStats = await getIOStatistics(client, pgMajorVersion);

  // Sort by backend_type, putting 'total' first if present
  ioStats.sort((a, b) => {
    if (a.backend_type === "total") return -1;
    if (b.backend_type === "total") return 1;
    return a.backend_type.localeCompare(b.backend_type);
  });

  // Find 'total' row for analysis, or sum all rows if not present
  let totalStats = ioStats.find((s) => s.backend_type === "total");
  if (!totalStats && ioStats.length > 0) {
    totalStats = {
      backend_type: "total",
      reads: ioStats.reduce((sum, s) => sum + s.reads, 0),
      read_bytes_mb: ioStats.reduce((sum, s) => sum + s.read_bytes_mb, 0),
      read_time_ms: ioStats.reduce((sum, s) => sum + s.read_time_ms, 0),
      writes: ioStats.reduce((sum, s) => sum + s.writes, 0),
      write_bytes_mb: ioStats.reduce((sum, s) => sum + s.write_bytes_mb, 0),
      write_time_ms: ioStats.reduce((sum, s) => sum + s.write_time_ms, 0),
      writebacks: ioStats.reduce((sum, s) => sum + s.writebacks, 0),
      writeback_bytes_mb: ioStats.reduce((sum, s) => sum + s.writeback_bytes_mb, 0),
      writeback_time_ms: ioStats.reduce((sum, s) => sum + s.writeback_time_ms, 0),
      fsyncs: ioStats.reduce((sum, s) => sum + s.fsyncs, 0),
      fsync_time_ms: ioStats.reduce((sum, s) => sum + s.fsync_time_ms, 0),
      extends: ioStats.reduce((sum, s) => sum + (s.extends || 0), 0),
      extend_bytes_mb: ioStats.reduce((sum, s) => sum + (s.extend_bytes_mb || 0), 0),
      hits: ioStats.reduce((sum, s) => sum + s.hits, 0),
      evictions: ioStats.reduce((sum, s) => sum + s.evictions, 0),
      reuses: ioStats.reduce((sum, s) => sum + s.reuses, 0),
    };
  }

  // Calculate analysis
  const totalReadMb = totalStats?.read_bytes_mb || 0;
  const totalWriteMb = totalStats?.write_bytes_mb || 0;
  const totalReadTime = totalStats?.read_time_ms || 0;
  const totalWriteTime = totalStats?.write_time_ms || 0;
  const totalIoTimeMs = totalReadTime + totalWriteTime;
  const totalReads = totalStats?.reads || 0;
  const totalWrites = totalStats?.writes || 0;
  const totalHits = totalStats?.hits || 0;

  // Hit ratio: hits / (hits + reads) * 100
  const totalRequests = totalHits + totalReads;
  const readHitRatioPct = totalRequests > 0 ? Math.round((totalHits / totalRequests) * 10000) / 100 : 0;

  // Average times
  const avgReadTimeMs = totalReads > 0 ? Math.round((totalReadTime / totalReads) * 1000) / 1000 : null;
  const avgWriteTimeMs = totalWrites > 0 ? Math.round((totalWriteTime / totalWrites) * 1000) / 1000 : null;

  // Direct-connect checkup queries stats_reset separately instead of reading it from pgwatch metrics.
  let statsResetS: number | null = null;
  try {
    const resetResult = await client.query(`
      select max(extract(epoch from now() - stats_reset)::int) as stats_reset_s
      from pg_stat_io
    `);
    if (resetResult.rows.length > 0 && resetResult.rows[0].stats_reset_s !== null) {
      const parsedStatsResetS = parseInt(resetResult.rows[0].stats_reset_s, 10);
      statsResetS = Number.isFinite(parsedStatsResetS) ? parsedStatsResetS : null;
    }
  } catch (err) {
    // Ignore errors getting stats_reset - not critical
  }

  report.results[nodeName] = {
    data: {
      available: ioStats.length > 0,
      by_backend_type: ioStats,
      analysis: {
        total_read_mb: totalReadMb,
        total_write_mb: totalWriteMb,
        total_io_time_ms: totalIoTimeMs,
        read_hit_ratio_pct: readHitRatioPct,
        avg_read_time_ms: avgReadTimeMs,
        avg_write_time_ms: avgWriteTimeMs,
      },
      stats_reset_s: statsResetS,
    },
    postgres_version: postgresVersion,
  };

  return report;
}

/**
 * Available report generators
 */
export const REPORT_GENERATORS: Record<string, (client: Client, nodeName: string) => Promise<Report>> = {
  A002: generateA002,
  A003: generateA003,
  A004: generateA004,
  A007: generateA007,
  A013: generateA013,
  D001: generateD001,
  D004: generateD004,
  F001: generateF001,
  F002: generateF002,
  F003: generateF003,
  F004: generateF004,
  F005: generateF005,
  F009: generateF009,
  G001: generateG001,
  G003: generateG003,
  H001: generateH001,
  H002: generateH002,
  H004: generateH004,
  I001: generateI001,
};

/**
 * Check IDs and titles.
 *
 * This mapping is built from the embedded checkup dictionary, which is
 * fetched from https://postgres.ai/api/general/checkup_dictionary at build time.
 *
 * For the full dictionary (all available checks), use the checkup-dictionary module.
 * CHECK_INFO is filtered to only include checks that have express-mode generators.
 */
export const CHECK_INFO: Record<string, string> = (() => {
  // Build the full dictionary map
  const fullMap = buildCheckInfoMap();

  // Filter to only include checks that have express-mode generators
  const expressCheckIds = Object.keys(REPORT_GENERATORS);
  const filtered: Record<string, string> = {};
  for (const checkId of expressCheckIds) {
    // Use dictionary title if available, otherwise use a fallback
    filtered[checkId] = fullMap[checkId] || checkId;
  }
  return filtered;
})();

/**
 * A single check that failed during generateAllReports. Reported through the
 * onCheckError callback so callers can warn, mark the run as partial, and
 * still keep every report that DID complete.
 */
export interface CheckGenerationFailure {
  checkId: string;
  checkTitle: string;
  error: Error;
}

/**
 * Generate all available health check reports.
 * This is the main entry point for express mode checkup generation.
 *
 * Per-check error isolation (work item 260, finding 4): one throwing check
 * must not abort the whole checkup and discard the reports that already
 * completed. Each generator runs in its own try/catch; failures are reported
 * through onCheckError and the remaining checks still run. A TOTAL failure
 * (every check failed — e.g. the connection died) still throws loudly, so a
 * fully broken run can never masquerade as an empty-but-successful one.
 *
 * Single-check runs (`--check-id <id>`) do NOT go through this function —
 * they call the generator directly and keep hard-fail (throw) semantics.
 *
 * @param client - Connected PostgreSQL client
 * @param nodeName - Node identifier for the report (default: "node-01")
 * @param onProgress - Optional callback for progress updates during generation
 * @param onCheckError - Optional callback invoked for each check whose
 *   generator threw; the run continues with the remaining checks
 * @returns Object mapping check IDs (e.g., "H001", "A002") to their reports;
 *   checks that failed are absent from the result
 * @throws {Error} Only when EVERY check fails (total failure)
 */
export async function generateAllReports(
  client: Client,
  nodeName: string = "node-01",
  onProgress?: (info: { checkId: string; checkTitle: string; index: number; total: number }) => void,
  onCheckError?: (failure: CheckGenerationFailure) => void
): Promise<Record<string, Report>> {
  const reports: Record<string, Report> = {};
  const failures: CheckGenerationFailure[] = [];

  const entries = Object.entries(REPORT_GENERATORS);
  const total = entries.length;
  let index = 0;

  for (const [checkId, generator] of entries) {
    index += 1;
    const checkTitle = CHECK_INFO[checkId] || checkId;
    onProgress?.({
      checkId,
      checkTitle,
      index,
      total,
    });
    try {
      reports[checkId] = withCheckSummary(await generator(client, nodeName));
    } catch (err) {
      const failure: CheckGenerationFailure = {
        checkId,
        checkTitle,
        error: err instanceof Error ? err : new Error(String(err)),
      };
      failures.push(failure);
      onCheckError?.(failure);
    }
  }

  // Preserve the loud-failure contract for a TOTAL failure: if nothing at all
  // succeeded (dead connection, catastrophic permission problem, ...), throw
  // instead of returning an empty result that downstream code would happily
  // upload / write as a "successful" checkup with zero findings.
  if (total > 0 && Object.keys(reports).length === 0) {
    const first = failures[0];
    throw new Error(
      `All ${total} checks failed — first error (${first.checkId}): ${first.error.message}`
    );
  }

  return reports;
}

/**
 * Attach the severity summary (status + message) to a report, mutating and
 * returning it. Idempotent. This folds the CLI's severity logic
 * (checkup-summary.ts) into the report envelope so that consumers of the JSON
 * contract — the `--json` CLI output and any host application embedding
 * checkup — get severity without reimplementing it.
 *
 * The summary is additive and optional in the schemas; existing consumers that
 * ignore it are unaffected.
 */
export function withCheckSummary(report: Report): Report {
  report.summary = generateCheckSummary(report.checkId, report);
  return report;
}
