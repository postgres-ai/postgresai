import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { dump, load } from "js-yaml";
import { requestTimeoutSignal } from "./util";

function validateName(name: string): void {
  if (!/^[A-Za-z0-9_-]+$/.test(name)) throw new Error("Invalid ClickHouse target name.");
}
function validateId(id: string): void {
  if (!/^[0-9a-f-]{36}$/.test(id)) throw new Error("Invalid ClickHouse organization or service id.");
}

export async function findService({ apiUrl, orgId, keyId, keySecret, hostname }: {
  apiUrl: string; orgId: string; keyId: string; keySecret: string; hostname: string;
}): Promise<{ id: string; name: string; state: string }> {
  validateId(orgId);
  const base = `${new URL(apiUrl).origin}/v1/organizations/${orgId}/postgres`;
  async function get<T>(url: string): Promise<T> {
    const response = await fetch(url, {
      headers: { Authorization: `Basic ${Buffer.from(`${keyId}:${keySecret}`).toString("base64")}` },
      signal: requestTimeoutSignal().signal,
      redirect: "error",
    });
    if (response.status === 401) throw new Error("ClickHouse Cloud rejected the API key (401). Check the key id and secret.");
    if (response.status === 403) throw new Error(`The API key cannot read Postgres services in organization ${orgId} (403). Give it read access to this organization.`);
    if (!response.ok) throw new Error(`ClickHouse Cloud API request failed (${response.status}).`);
    return ((await response.json()) as { result: T }).result;
  }
  type Service = { id: string; name: string; state: string; hostname?: string };
  for (const service of await get<Service[]>(base)) {
    validateId(service.id);
    const details = await get<Service>(`${base}/${service.id}`);
    if (details.hostname?.toLowerCase() === hostname.toLowerCase()) {
      return { id: service.id, name: details.name, state: details.state };
    }
  }
  throw new Error(`No ClickHouse Managed Postgres service in organization ${orgId} has hostname ${hostname}.`);
}

export function renderScrapeConfig({ name, cluster, nodeName = name, orgId, serviceId, keyId, passwordFile, apiUrl }: {
  name: string; cluster: string; nodeName?: string; orgId: string; serviceId: string; keyId: string; passwordFile: string; apiUrl: string;
}): string {
  validateName(name);
  validateId(orgId);
  validateId(serviceId);
  const url = new URL(apiUrl);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("Invalid ClickHouse API URL scheme.");
  const config = {
    job_name: `clickhouse-${name}`,
    scheme: url.protocol.slice(0, -1),
    metrics_path: `/v1/organizations/${orgId}/postgres/${serviceId}/prometheus`,
    scrape_interval: "60s", scrape_timeout: "30s",
    basic_auth: { username: keyId, password_file: passwordFile },
    static_configs: [{ targets: [url.host], labels: { cluster, node_name: nodeName } }],
    metric_relabel_configs: [{ source_labels: ["__name__"], regex: "PostgresServiceInfo|PostgresServer_.*", action: "keep" }],
  };
  // The "__" prefix keeps this label out of stored series, but the targets API
  // lists it, so a reload can be matched to this exact file.
  const revision = `r${createHash("sha256").update(dump([config], { lineWidth: -1 })).digest("hex").slice(0, 16)}`;
  Object.assign(config.static_configs[0].labels, { __pgai_rev: revision });
  return dump([config], { lineWidth: -1 });
}

export function scrapeRevision(projectDir: string, name: string): string {
  validateName(name);
  const [config] = load(readFileSync(join(projectDir, "host-metrics", `clickhouse-${name}.yml`), "utf8")) as any[];
  const revision = config?.static_configs?.[0]?.labels?.__pgai_rev;
  if (typeof revision !== "string" || !/^r[0-9a-f]{16}$/.test(revision)) throw new Error(`host-metrics/clickhouse-${name}.yml has no __pgai_rev label.`);
  return revision;
}

// Runs inside sink-prometheus: sh -c SCRIPT sh NEEDLE present|absent. VictoriaMetrics
// applies a SIGHUP asynchronously and skips a scrape file it cannot parse while
// still counting the reload as successful, so only the targets list shows the
// outcome. Polls it until NEEDLE is present (or absent), for at most ~10s.
export const HOST_METRICS_VERIFY_SCRIPT = `auth="Authorization: Basic $(printf '%s:%s' "$VM_AUTH_USERNAME" "$VM_AUTH_PASSWORD" | base64 | tr -d '\\n')"; end=$(($(date +%s) + 10)); while [ "$(date +%s)" -lt "$end" ]; do if t=$(wget -qO- -T 2 --header "$auth" http://127.0.0.1:9090/api/v1/targets); then case "$t" in '{"status":"success"'*) case "$t" in *"$1"*) [ "$2" = present ] && exit 0 ;; *) [ "$2" = absent ] && exit 0 ;; esac ;; esac; fi; sleep 0.5; done; exit 1`;

export async function addHostMetrics({ projectDir, name, conn, env, cluster = "default", nodeName = name }: {
  projectDir: string; name: string; conn: string; env: Record<string, string | undefined>; cluster?: string; nodeName?: string;
}): Promise<string> {
  validateName(name);
  const { CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: keyId, CLICKHOUSE_KEY_SECRET: keySecret } = env;
  if (!orgId || !keyId || !keySecret) return "Host metrics: set CLICKHOUSE_ORG_ID, CLICKHOUSE_KEY_ID and CLICKHOUSE_KEY_SECRET and re-run to collect CPU, memory, disk and I/O from ClickHouse Cloud";
  const apiUrl = env.CLICKHOUSE_API_URL || "https://api.clickhouse.cloud";
  const service = await findService({ apiUrl, orgId, keyId, keySecret, hostname: new URL(conn).hostname });
  if (service.state !== "running") {
    throw new Error(`ClickHouse Managed Postgres service is ${service.state}, not running. Start it in the ClickHouse Cloud console, then retry.`);
  }
  const prefix = `clickhouse-${name}`;
  const text = renderScrapeConfig({ name, cluster, nodeName, orgId, serviceId: service.id, keyId, passwordFile: `/etc/pgai/host-metrics/${prefix}.secret`, apiUrl });
  const dir = join(projectDir, "host-metrics");
  mkdirSync(dir, { recursive: true });
  const secretFile = join(dir, `${prefix}.secret`);
  const tmpFile = `${secretFile}.tmp`;
  // A previous run killed between write and rename leaves the tmp file behind; drop it so a retry does not fail with EEXIST.
  rmSync(tmpFile, { force: true });
  try {
    writeFileSync(tmpFile, keySecret, { mode: 0o600, flag: "wx" });
    renameSync(tmpFile, secretFile);
  } finally {
    rmSync(tmpFile, { force: true });
  }
  const configFile = join(dir, `${prefix}.yml`);
  writeFileSync(configFile, text, { mode: 0o644 });
  chmodSync(configFile, 0o644);
  return `Host metrics: ClickHouse Cloud Prometheus endpoint for service ${service.name} (scraped every 60s)`;
}

export function removeHostMetrics(projectDir: string, name: string): void {
  validateName(name);
  for (const extension of ["yml", "secret", "secret.tmp"]) rmSync(join(projectDir, "host-metrics", `clickhouse-${name}.${extension}`), { force: true });
}
