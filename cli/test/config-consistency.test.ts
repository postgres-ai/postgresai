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

type GrafanaDatasource = {
  name?: unknown;
  orgId?: unknown;
  uid?: unknown;
  editable?: unknown;
  basicAuth?: unknown;
};

type GrafanaDatasourceConfig = {
  deleteDatasources?: Array<{ name?: unknown; orgId?: unknown }>;
  datasources?: GrafanaDatasource[];
};

type DockerComposeConfig = {
  services?: Record<string, { restart?: unknown }>;
};

// These UIDs are referenced by provisioned Grafana dashboards.
// Changing them without updating dashboard JSON silently breaks panels.
const expectedGrafanaDatasourceUids = new Map([
  ["PGWatch-PostgreSQL", "P031DD592934B2F1F"],
  ["PGWatch-Prometheus", "P7A0D6631BB10B34F"],
  ["Infinity", "aerffb0z8rjlsc"],
]);

/**
 * Normalizes raw Grafana datasource orgId values to a positive integer.
 *
 * Grafana treats omitted/null orgId as org 1 in the default single-org setup.
 * Numeric strings are accepted because YAML or API callers may preserve them
 * as strings. Booleans, objects, empty strings, non-positive values, and
 * non-integers are rejected because Grafana cannot address those orgs.
 */
const normalizeGrafanaOrgId = (orgId: unknown) => {
  if (orgId === undefined || orgId === null) {
    return 1;
  }
  if (typeof orgId === "boolean" || typeof orgId === "object" || orgId === "") {
    throw new Error(`Invalid Grafana datasource orgId: ${orgId}`);
  }

  const normalizedOrgId = Number(orgId);
  if (!Number.isInteger(normalizedOrgId) || normalizedOrgId < 1) {
    throw new Error(`Invalid Grafana datasource orgId: ${orgId}`);
  }
  return normalizedOrgId;
};

// Normalize orgId so omitted/null orgId and explicit orgId: 1 map to the same
// datasource key, matching Grafana's implicit single-org behavior.
const datasourceKey = (name: unknown, orgId: unknown) =>
  `${name}:${normalizeGrafanaOrgId(orgId)}`;

const configDir = resolve(import.meta.dir, "../../config");
const metricsPath = resolve(configDir, "pgwatch-prometheus/metrics.yml");
const datasourcePath = resolve(
  configDir,
  "grafana/provisioning/datasources/datasources.yml"
);
const composePath = resolve(import.meta.dir, "../../docker-compose.yml");
const envExamplePath = resolve(import.meta.dir, "../../.env.example");
const dashboardDir = resolve(configDir, "grafana/dashboards");
const dashboardGeneratedMetricRefs = new Set([
  // Exported by monitoring_flask_backend/app.py and joined into query dashboards.
  "query_info",
]);
const pgwatchMetricRefPattern = /\bpgwatch_([a-zA-Z_:][a-zA-Z0-9_:]*)\b/g;

const loadPgwatchConfig = () =>
  Bun.YAML.parse(readFileSync(metricsPath, "utf8")) as PgwatchConfig;

const parseGrafanaDatasourceConfig = (content: string) =>
  Bun.YAML.parse(content) as GrafanaDatasourceConfig;

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

  test("Grafana datasource YAML parser rejects malformed config", () => {
    expect(() =>
      parseGrafanaDatasourceConfig("apiVersion: 1\ndatasources:\n  - name: [")
    ).toThrow();
  });

  test("Grafana orgId normalization rejects invalid values", () => {
    for (const orgId of [undefined, null, 1, "1"]) {
      expect(normalizeGrafanaOrgId(orgId)).toBe(1);
    }
    expect(normalizeGrafanaOrgId("2")).toBe(2);

    for (const orgId of [
      0,
      -1,
      true,
      "",
      1.5,
      "0",
      "-1",
      "1.5",
      "abc",
      NaN,
      Infinity,
    ]) {
      expect(() => normalizeGrafanaOrgId(orgId)).toThrow();
    }
  });

  test("datasources.yml delete entries match stable non-editable datasources", () => {
    const datasourceConfig = parseGrafanaDatasourceConfig(
      readFileSync(datasourcePath, "utf8")
    );
    const deleteDatasources = datasourceConfig.deleteDatasources ?? [];
    const datasources = datasourceConfig.datasources ?? [];
    const deletedKeys = new Set(
      deleteDatasources.map((datasource) =>
        datasourceKey(datasource.name, datasource.orgId)
      )
    );
    const datasourceByNameAndOrg = new Map(
      datasources.map((datasource) => [
        datasourceKey(datasource.name, datasource.orgId),
        datasource,
      ])
    );

    expect(deleteDatasources.length).toBeGreaterThan(0);
    expect(deleteDatasources.length).toBe(datasources.length);
    expect(datasourceByNameAndOrg.size).toBe(datasources.length);

    for (const deletedDatasource of deleteDatasources) {
      const datasource = datasourceByNameAndOrg.get(
        datasourceKey(deletedDatasource.name, deletedDatasource.orgId)
      );
      const expectedUid = expectedGrafanaDatasourceUids.get(
        String(deletedDatasource.name)
      );
      expect(datasource).toBeDefined();
      expect(expectedUid).toBeDefined();
      expect(datasource?.uid).toBe(expectedUid);
      expect(datasource?.editable).toBe(false);
    }

    for (const datasource of datasources) {
      const expectedUid = expectedGrafanaDatasourceUids.get(
        String(datasource.name)
      );
      expect(deletedKeys.has(datasourceKey(datasource.name, datasource.orgId))).toBe(
        true
      );
      expect(expectedUid).toBeDefined();
      expect(datasource.uid).toBe(expectedUid);
      expect(datasource.editable).toBe(false);
    }
  });

  test("Grafana Prometheus datasource requires VM auth environment", () => {
    const datasourceConfig = parseGrafanaDatasourceConfig(
      readFileSync(datasourcePath, "utf8")
    );
    const composeYml = readFileSync(composePath, "utf8");
    const prometheusDatasource = datasourceConfig.datasources?.find(
      (datasource) => datasource.name === "PGWatch-Prometheus"
    ) as
      | {
          basicAuth?: unknown;
          basicAuthUser?: unknown;
          secureJsonData?: { basicAuthPassword?: unknown };
        }
      | undefined;

    expect(prometheusDatasource?.basicAuth).toBe(true);
    expect(prometheusDatasource?.basicAuthUser).toBe("${VM_AUTH_USERNAME}");
    expect(prometheusDatasource?.secureJsonData?.basicAuthPassword).toBe(
      "${VM_AUTH_PASSWORD}"
    );
    expect(composeYml).toContain(
      "VM_AUTH_USERNAME: ${VM_AUTH_USERNAME:?VM_AUTH_USERNAME is required for Grafana datasource provisioning}"
    );
    expect(composeYml).toContain(
      "VM_AUTH_PASSWORD: ${VM_AUTH_PASSWORD:?VM_AUTH_PASSWORD is required for Grafana datasource provisioning}"
    );
  });

  test(".env.example documents direct Docker Compose credentials", () => {
    const envExample = readFileSync(envExamplePath, "utf8");

    expect(envExample).toContain("REPLICATOR_PASSWORD=");
    expect(envExample).toContain("VM_AUTH_USERNAME=vmauth");
    expect(envExample).toContain("VM_AUTH_PASSWORD=");
    expect(envExample).toContain("openssl rand -hex 32");
    expect(envExample).toContain("openssl rand -base64 18");
  });

  test("long-running Docker Compose services survive host reboot", () => {
    const composeConfig = Bun.YAML.parse(
      readFileSync(composePath, "utf8")
    ) as DockerComposeConfig;
    const services = composeConfig.services ?? {};

    // All services should survive host reboot unless they are one-shot setup jobs.
    const oneShotServices = new Set(["config-init", "sources-generator"]);
    const serviceEntries = Object.entries(services);

    expect(serviceEntries.length).toBeGreaterThan(0);

    for (const serviceName of oneShotServices) {
      expect(services[serviceName]).toBeDefined();
      expect(services[serviceName]?.restart).toBeUndefined();
    }

    for (const [serviceName, serviceConfig] of serviceEntries) {
      if (oneShotServices.has(serviceName)) {
        continue;
      }

      expect(serviceConfig.restart).toBe("unless-stopped");
    }
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
