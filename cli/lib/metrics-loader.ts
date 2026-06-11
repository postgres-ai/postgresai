/**
 * Metrics loader for express checkup reports
 *
 * Loads SQL queries from embedded metrics data (generated from metrics.yml at build time).
 * Provides version-aware query selection and row transformation utilities.
 */

import { METRICS, MetricDefinition } from "./metrics-embedded";

/**
 * Get SQL query for a specific metric, selecting the appropriate version.
 *
 * @param metricName - Name of the metric (e.g., "settings", "db_stats")
 * @param pgMajorVersion - PostgreSQL major version (default: 16)
 * @returns SQL query string
 * @throws Error if metric not found or no compatible version available
 */
export function getMetricSql(metricName: string, pgMajorVersion: number = 16): string {
  const metric = METRICS[metricName];

  if (!metric) {
    throw new Error(`Metric "${metricName}" not found. Available metrics: ${Object.keys(METRICS).join(", ")}`);
  }

  // Find the best matching version: highest version <= pgMajorVersion
  const availableVersions = Object.keys(metric.sqls)
    .map(v => parseInt(v, 10))
    .sort((a, b) => b - a); // Sort descending

  const matchingVersion = availableVersions.find(v => v <= pgMajorVersion);

  if (matchingVersion === undefined) {
    throw new Error(
      `No compatible SQL version for metric "${metricName}" with PostgreSQL ${pgMajorVersion}. ` +
      `Available versions: ${availableVersions.join(", ")}`
    );
  }

  return metric.sqls[matchingVersion];
}

/**
 * Get metric definition including all metadata.
 *
 * @param metricName - Name of the metric
 * @returns MetricDefinition or undefined if not found
 */
export function getMetricDefinition(metricName: string): MetricDefinition | undefined {
  return METRICS[metricName];
}

/**
 * List all available metric names.
 */
export function listMetricNames(): string[] {
  return Object.keys(METRICS);
}

/**
 * Metric names that correspond to express report checks.
 * Maps check IDs and logical names to metric names in the METRICS object.
 */
export const METRIC_NAMES = {
  // Index health checks
  H001: "pg_invalid_indexes",
  H002: "unused_indexes",
  H004: "redundant_indexes",
  // Dead tuples and per-table autovacuum overrides
  F003: "pg_dead_tuples",
  // Bloat estimation
  F004: "pg_table_bloat",
  F005: "pg_btree_bloat",
  // Settings and version info (A002, A003, A007, A013)
  settings: "settings",
  // Database statistics (A004)
  dbStats: "db_stats",
  dbSize: "db_size",
  // Stats reset info (H002)
  statsReset: "stats_reset",
  // I/O statistics (I001) - PostgreSQL 16+
  I001: "pg_stat_io",
} as const;

/**
 * Transform a row from metrics query output to JSON report format.
 * Metrics use `tag_` prefix for dimensions; we strip it for JSON reports.
 * Also removes Prometheus-specific fields like epoch_ns, num, tag_datname.
 */
export function transformMetricRow(row: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(row)) {
    // Skip Prometheus-specific fields
    if (key === "epoch_ns" || key === "num" || key === "tag_datname") {
      continue;
    }

    // Strip tag_ prefix
    const newKey = key.startsWith("tag_") ? key.slice(4) : key;
    result[newKey] = value;
  }

  return result;
}

/**
 * Transform settings metric row to the format expected by express reports.
 * The settings metric returns one row per setting with tag_setting_name as key.
 */
export function transformSettingsRow(row: Record<string, unknown>): {
  name: string;
  setting: string;
  unit: string;
  category: string;
  vartype: string;
  is_default: boolean;
} {
  return {
    name: String(row.tag_setting_name || ""),
    setting: String(row.tag_setting_value || ""),
    unit: String(row.tag_unit || ""),
    category: String(row.tag_category || ""),
    vartype: String(row.tag_vartype || ""),
    is_default: row.is_default === 1 || row.is_default === true,
  };
}

// Re-export types for convenience
export type { MetricDefinition } from "./metrics-embedded";

// Legacy export for backward compatibility
export function loadMetricsYml(): { metrics: Record<string, unknown> } {
  return { metrics: METRICS };
}
