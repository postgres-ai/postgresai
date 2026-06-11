#!/usr/bin/env bun
/**
 * Build script to embed metrics.yml into the CLI bundle.
 *
 * This script reads config/pgwatch-prometheus/metrics.yml and generates
 * cli/lib/metrics-embedded.ts with the metrics data embedded as TypeScript.
 *
 * The generated file is NOT committed to git - it's regenerated at build time.
 *
 * Usage: bun run scripts/embed-metrics.ts
 */

import * as fs from "fs";
import * as path from "path";
import * as yaml from "js-yaml";

// Resolve paths relative to cli/ directory
const CLI_DIR = path.resolve(__dirname, "..");
const METRICS_YML_PATH = path.resolve(CLI_DIR, "../config/pgwatch-prometheus/metrics.yml");
const OUTPUT_PATH = path.resolve(CLI_DIR, "lib/metrics-embedded.ts");

interface MetricDefinition {
  description?: string;
  // YAML parses numeric keys (e.g., 11:, 14:) as numbers, representing PG major versions
  sqls: Record<number, string>;
  gauges?: string[];
  statement_timeout_seconds?: number;
  is_instance_level?: boolean;
  node_status?: string;
}

interface MetricsYml {
  metrics: Record<string, MetricDefinition>;
}

// Metrics needed for express mode reports
const REQUIRED_METRICS = [
  // Settings and version (A002, A003, A007, A013)
  "settings",
  // Database stats (A004)
  "db_stats",
  "db_size",
  // Index health (H001, H002, H004)
  "pg_invalid_indexes",
  "unused_indexes",
  "redundant_indexes",
  // Stats reset info (H002)
  "stats_reset",
  // Dead tuples and per-table autovacuum overrides (F003)
  "pg_dead_tuples",
  // Bloat estimation (F004, F005)
  "pg_table_bloat",
  "pg_btree_bloat",
  // I/O statistics (I001)
  "pg_stat_io",
];

function main() {
  console.log(`Reading metrics from: ${METRICS_YML_PATH}`);

  if (!fs.existsSync(METRICS_YML_PATH)) {
    console.error(`ERROR: metrics.yml not found at ${METRICS_YML_PATH}`);
    process.exit(1);
  }

  const yamlContent = fs.readFileSync(METRICS_YML_PATH, "utf8");
  const parsed = yaml.load(yamlContent) as MetricsYml;

  if (!parsed.metrics) {
    console.error("ERROR: No 'metrics' section found in metrics.yml");
    process.exit(1);
  }

  // Extract only required metrics
  const extractedMetrics: Record<string, MetricDefinition> = {};
  const missingMetrics: string[] = [];

  for (const metricName of REQUIRED_METRICS) {
    if (parsed.metrics[metricName]) {
      extractedMetrics[metricName] = parsed.metrics[metricName];
    } else {
      missingMetrics.push(metricName);
    }
  }

  if (missingMetrics.length > 0) {
    console.error(`ERROR: Missing required metrics: ${missingMetrics.join(", ")}`);
    process.exit(1);
  }

  // Generate TypeScript code
  const tsCode = generateTypeScript(extractedMetrics);

  // Write output
  fs.writeFileSync(OUTPUT_PATH, tsCode, "utf8");
  console.log(`Generated: ${OUTPUT_PATH}`);
  console.log(`Embedded ${Object.keys(extractedMetrics).length} metrics`);
}

function generateTypeScript(metrics: Record<string, MetricDefinition>): string {
  const lines: string[] = [
    "// AUTO-GENERATED FILE - DO NOT EDIT",
    "// Generated from config/pgwatch-prometheus/metrics.yml by scripts/embed-metrics.ts",
    `// Generated at: ${new Date().toISOString()}`,
    "",
    "/**",
    " * Metric definition from metrics.yml",
    " */",
    "export interface MetricDefinition {",
    "  description?: string;",
    "  sqls: Record<number, string>;  // PG major version -> SQL query",
    "  gauges?: string[];",
    "  statement_timeout_seconds?: number;",
    "}",
    "",
    "/**",
    " * Embedded metrics for express mode reports.",
    " * Only includes metrics required for CLI checkup reports.",
    " */",
    "export const METRICS: Record<string, MetricDefinition> = {",
  ];

  for (const [name, metric] of Object.entries(metrics)) {
    lines.push(`  ${JSON.stringify(name)}: {`);

    if (metric.description) {
      // Escape description for TypeScript string
      const desc = metric.description.trim().replace(/\n/g, " ").replace(/\s+/g, " ");
      lines.push(`    description: ${JSON.stringify(desc)},`);
    }

    // sqls keys are PG major versions (numbers in YAML, but Object.entries returns strings)
    lines.push("    sqls: {");
    for (const [versionKey, sql] of Object.entries(metric.sqls)) {
      // YAML numeric keys may be parsed as numbers or strings depending on context;
      // explicitly convert to ensure consistent numeric keys in output
      const versionNum = typeof versionKey === "number" ? versionKey : parseInt(versionKey, 10);
      // Use JSON.stringify for robust escaping of all special characters
      lines.push(`      ${versionNum}: ${JSON.stringify(sql)},`);
    }
    lines.push("    },");

    if (metric.gauges) {
      lines.push(`    gauges: ${JSON.stringify(metric.gauges)},`);
    }

    if (metric.statement_timeout_seconds !== undefined) {
      lines.push(`    statement_timeout_seconds: ${metric.statement_timeout_seconds},`);
    }

    lines.push("  },");
  }

  lines.push("};");
  lines.push("");

  return lines.join("\n");
}

main();
