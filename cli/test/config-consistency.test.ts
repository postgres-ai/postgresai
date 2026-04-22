/**
 * Tests that config files are consistent with what the CLI expects.
 * Catches schema mismatches like pg_statistic in wrong schema.
 */
import { describe, test, expect } from "bun:test";
import { readdirSync, readFileSync } from "fs";
import { resolve } from "path";

type PgwatchConfig = {
  metrics: Record<string, unknown>;
  presets?: Record<string, { metrics?: Record<string, unknown> }>;
};

const configDir = resolve(import.meta.dir, "../../config");
const metricsPath = resolve(configDir, "pgwatch-prometheus/metrics.yml");
const dashboardDir = resolve(configDir, "grafana/dashboards");
const dashboardGeneratedMetricRefs = new Set([
  // Exported by monitoring_flask_backend/app.py and joined into query dashboards.
  "query_info",
]);
const pgwatchMetricRefPattern = /\bpgwatch_([a-zA-Z_:][a-zA-Z0-9_:]*)\b/g;

const loadPgwatchConfig = () =>
  Bun.YAML.parse(readFileSync(metricsPath, "utf8")) as PgwatchConfig;

const grafanaQueryStringKeys = new Set(["expr", "definition", "query"]);

const collectStringValuesByKeys = (
  value: unknown,
  keys: Set<string>
): string[] => {
  const matches: string[] = [];

  const visit = (node: unknown) => {
    if (Array.isArray(node)) {
      for (const item of node) {
        visit(item);
      }
      return;
    }

    if (node && typeof node === "object") {
      const record = node as Record<string, unknown>;
      for (const [recordKey, item] of Object.entries(record)) {
        if (keys.has(recordKey) && typeof item === "string") {
          matches.push(item);
        }

        visit(item);
      }
    }
  };

  visit(value);
  return matches;
};

const resolveMetricReference = (
  metricRef: string,
  metricNamesByLength: string[]
) =>
  metricNamesByLength.find((metricName) =>
    metricRef === metricName || metricRef.startsWith(`${metricName}_`)
  );

describe("Config consistency", () => {
  test("target-db/init.sql creates pg_statistic in postgres_ai schema", () => {
    const initSql = readFileSync(resolve(configDir, "target-db/init.sql"), "utf8");

    // Must create postgres_ai schema
    expect(initSql).toMatch(/create\s+schema\s+if\s+not\s+exists\s+postgres_ai/i);

    // Must create view in postgres_ai schema, not public
    expect(initSql).toMatch(/create\s+or\s+replace\s+view\s+postgres_ai\.pg_statistic/i);
    expect(initSql).not.toMatch(/create\s+or\s+replace\s+view\s+public\.pg_statistic/i);

    // Must grant on postgres_ai.pg_statistic
    expect(initSql).toMatch(/grant\s+select\s+on\s+postgres_ai\.pg_statistic/i);
  });

  test("pgwatch metrics.yml uses postgres_ai.pg_statistic", () => {
    const metricsYml = readFileSync(metricsPath, "utf8");

    // Should reference postgres_ai.pg_statistic, not public.pg_statistic
    expect(metricsYml).not.toMatch(/public\.pg_statistic/);
    expect(metricsYml).toMatch(/postgres_ai\.pg_statistic/);
  });

  test("pgwatch presets only reference configured metrics", () => {
    const pgwatchConfig = loadPgwatchConfig();
    const metricNames = new Set(Object.keys(pgwatchConfig.metrics));
    const presetNames = Object.keys(pgwatchConfig.presets ?? {});
    const unknownPresetMetrics = new Set<string>();

    expect(metricNames.size).toBeGreaterThan(0);
    expect(presetNames.length).toBeGreaterThan(0);

    for (const [presetName, preset] of Object.entries(pgwatchConfig.presets ?? {})) {
      for (const metricName of Object.keys(preset.metrics ?? {})) {
        if (!metricNames.has(metricName)) {
          unknownPresetMetrics.add(`${presetName}.${metricName}`);
        }
      }
    }

    expect([...unknownPresetMetrics].sort()).toEqual([]);
  });

  test("Grafana dashboard pgwatch metric references exist in metrics.yml", () => {
    const pgwatchConfig = loadPgwatchConfig();
    const metricNamesByLength = Object.keys(pgwatchConfig.metrics).sort(
      (a, b) => b.length - a.length
    );
    const dashboardFiles = readdirSync(dashboardDir).filter((file) =>
      file.endsWith(".json")
    );
    const unknownDashboardMetrics = new Set<string>();
    let observedDashboardMetricRefs = 0;

    expect(metricNamesByLength.length).toBeGreaterThan(0);
    expect(dashboardFiles.length).toBeGreaterThan(0);

    for (const dashboardFile of dashboardFiles) {
      const dashboard = JSON.parse(
        readFileSync(resolve(dashboardDir, dashboardFile), "utf8")
      );

      for (const queryString of collectStringValuesByKeys(
        dashboard,
        grafanaQueryStringKeys
      )) {
        for (const match of queryString.matchAll(pgwatchMetricRefPattern)) {
          observedDashboardMetricRefs += 1;
          const metricRef = match[1];
          if (dashboardGeneratedMetricRefs.has(metricRef)) {
            continue;
          }

          if (!resolveMetricReference(metricRef, metricNamesByLength)) {
            unknownDashboardMetrics.add(`${dashboardFile}: pgwatch_${metricRef}`);
          }
        }
      }
    }

    expect(observedDashboardMetricRefs).toBeGreaterThan(0);
    expect([...unknownDashboardMetrics].sort()).toEqual([]);
  });
});
