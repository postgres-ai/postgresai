/**
 * Hands-off AAS auto-onboarding for `mon local-install` (platform-all #338).
 *
 * After the monitoring stack is up and the instance is adopted, the CLI arms
 * AAS collection without an operator step:
 *   1. mint a `pgai-aas-collect` Grafana Viewer service-account token on the
 *      LOCAL Grafana (the CLI holds the admin password),
 *   2. resolve the numeric Prometheus datasource id,
 *   3. read the (cluster, node_name) labels straight from the pgwatch target
 *      config the CLI itself wrote (buildInstance's custom_tags) — no live
 *      series query, so no waiters>0 timing dependency,
 *   4. hand all of it to the platform via the API-token RPC
 *      v1.monitoring_instance_aas_register, which encrypts the token and stores
 *      the AAS state keys (it makes no outbound Grafana call of its own).
 *
 * Best-effort, exactly like registerMonitoringInstance: never throws, returns a
 * result the caller logs. The plaintext SA token only ever lives in locals.
 */

import { loadInstances } from "./instances";
import { resolveBaseUrls } from "./util";

const SA_NAME = "pgai-aas-collect";

/** Local Grafana base URL (published on the monitoring host). Overridable for tests/odd setups. */
function grafanaBaseUrl(): string {
  return (process.env.PGAI_GRAFANA_LOCAL_URL || "http://localhost:3000").replace(/\/+$/, "");
}

function grafanaAdminUser(): string {
  return process.env.GF_SECURITY_ADMIN_USER || "admin";
}

/** Parse a vcpus input (flag/env) to a non-negative integer; 0 = "unknown" fallback. */
export function parseVcpus(raw: string | number | undefined | null): number {
  if (raw === undefined || raw === null || raw === "") return 0;
  const n = typeof raw === "number" ? raw : parseInt(String(raw).trim(), 10);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
}

/**
 * Read the single enabled target's (cluster, node_name) from the pgwatch
 * instances file. Returns null when it can't be determined unambiguously
 * (0 or >1 enabled targets) — AAS onboards exactly one (cluster, node) pair.
 */
export function resolveAasLabels(instancesPath: string): { cluster: string; node: string } | null {
  let instances;
  try {
    instances = loadInstances(instancesPath);
  } catch {
    return null;
  }
  const enabled = instances.filter((i) => i.is_enabled !== false);
  if (enabled.length !== 1) return null;
  const tags = (enabled[0].custom_tags || {}) as Record<string, unknown>;
  const cluster = typeof tags.cluster === "string" && tags.cluster ? tags.cluster : "default";
  const node = typeof tags.node_name === "string" && tags.node_name ? tags.node_name : enabled[0].name;
  if (!cluster || !node) return null;
  return { cluster, node };
}

async function grafanaApi(
  method: string,
  pathPart: string,
  adminPassword: string,
  body?: unknown
): Promise<Response> {
  const auth = Buffer.from(`${grafanaAdminUser()}:${adminPassword}`).toString("base64");
  return fetch(`${grafanaBaseUrl()}${pathPart}`, {
    method,
    headers: { "Content-Type": "application/json", Authorization: `Basic ${auth}` },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/**
 * Find-or-create the pgai-aas-collect Viewer service account on the local
 * Grafana, prune any prior tokens (so they don't accumulate across re-installs),
 * and mint a fresh glsa_ token. Returns the token or null on any failure.
 */
export async function mintAasServiceAccountToken(
  adminPassword: string,
  debug = false
): Promise<string | null> {
  const log = (m: string) => debug && console.error(`Debug: AAS SA mint: ${m}`);
  try {
    let saId: number | null = null;

    const search = await grafanaApi("GET", `/api/serviceaccounts/search?query=${SA_NAME}`, adminPassword);
    if (search.ok) {
      const data = (await search.json().catch(() => null)) as { serviceAccounts?: Array<{ id?: unknown; name?: unknown }> } | null;
      const found = (data?.serviceAccounts || []).find((s) => s.name === SA_NAME);
      if (found && typeof found.id === "number") saId = found.id;
    }

    if (saId == null) {
      const created = await grafanaApi("POST", "/api/serviceaccounts", adminPassword, { name: SA_NAME, role: "Viewer" });
      if (!created.ok) {
        log(`create SA failed: HTTP ${created.status}`);
        return null;
      }
      const cj = (await created.json().catch(() => null)) as { id?: unknown } | null;
      if (typeof cj?.id !== "number") return null;
      saId = cj.id;
    }

    // Prune existing tokens so live Viewer tokens don't pile up across re-runs.
    const toks = await grafanaApi("GET", `/api/serviceaccounts/${saId}/tokens`, adminPassword);
    if (toks.ok) {
      const list = (await toks.json().catch(() => [])) as Array<{ id?: unknown }>;
      for (const t of list) {
        if (typeof t.id === "number") {
          await grafanaApi("DELETE", `/api/serviceaccounts/${saId}/tokens/${t.id}`, adminPassword).catch(() => undefined);
        }
      }
    }

    // Unique token name avoids a 409 on a pre-existing name.
    const mint = await grafanaApi("POST", `/api/serviceaccounts/${saId}/tokens`, adminPassword, {
      name: `aas-collect-${Date.now()}`,
      role: "Viewer",
    });
    if (!mint.ok) {
      log(`mint token failed: HTTP ${mint.status}`);
      return null;
    }
    const mj = (await mint.json().catch(() => null)) as { key?: unknown } | null;
    return typeof mj?.key === "string" ? mj.key : null;
  } catch (err) {
    log((err as Error).message);
    return null;
  }
}

/** Resolve the single Prometheus datasource's numeric id on the local Grafana. */
export async function resolveDatasourceId(adminPassword: string, debug = false): Promise<number | null> {
  try {
    const res = await grafanaApi("GET", "/api/datasources", adminPassword);
    if (!res.ok) return null;
    const list = (await res.json().catch(() => [])) as Array<{ id?: unknown; type?: unknown }>;
    const prom = list.filter((d) => d.type === "prometheus");
    if (prom.length !== 1) {
      if (debug) console.error(`Debug: AAS: expected 1 prometheus datasource, found ${prom.length}`);
      return null;
    }
    return typeof prom[0].id === "number" ? prom[0].id : null;
  } catch {
    return null;
  }
}

export interface AasRegisterResult {
  ok: boolean;
  reason?: string;
}

/**
 * Arm hands-off AAS collection for an adopted monitoring instance. Best-effort:
 * never throws; returns {ok:false, reason} on any failure so the caller can log
 * a non-fatal warning. Mirrors registerMonitoringInstance's API-call shape.
 */
export async function registerAasCollection(
  apiKey: string,
  instanceId: string,
  opts: {
    grafanaPassword: string;
    instancesPath: string;
    vcpus: number;
    apiBaseUrl?: string;
    debug?: boolean;
    fetchImpl?: typeof fetch;
  }
): Promise<AasRegisterResult> {
  const debug = !!opts.debug;
  try {
    if (!apiKey || !instanceId) return { ok: false, reason: "missing api key or instance id" };

    const labels = resolveAasLabels(opts.instancesPath);
    if (!labels) return { ok: false, reason: "could not determine a single (cluster, node_name) target" };

    const datasourceId = await resolveDatasourceId(opts.grafanaPassword, debug);
    if (datasourceId == null) return { ok: false, reason: "could not resolve the Prometheus datasource id" };

    const saToken = await mintAasServiceAccountToken(opts.grafanaPassword, debug);
    if (!saToken) return { ok: false, reason: "could not mint a Grafana service-account token" };

    const { apiBaseUrl } = resolveBaseUrls({ apiBaseUrl: opts.apiBaseUrl });
    const url = `${apiBaseUrl}/rpc/monitoring_instance_aas_register`;
    const doFetch = opts.fetchImpl || fetch;
    if (debug) console.error(`Debug: AAS: POST ${url} (cluster=${labels.cluster}, node=${labels.node}, vcpus=${opts.vcpus}, ds=${datasourceId})`);

    const res = await doFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_token: apiKey,
        instance_id: instanceId,
        sa_token: saToken,
        cluster_name: labels.cluster,
        node_name: labels.node,
        vcpus: opts.vcpus,
        datasource_id: datasourceId,
      }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      if (debug) console.error(`Debug: AAS register failed: HTTP ${res.status} ${body}`);
      return { ok: false, reason: `platform returned HTTP ${res.status}` };
    }
    return { ok: true };
  } catch (err) {
    if (debug) console.error(`Debug: AAS register error: ${(err as Error).message}`);
    return { ok: false, reason: (err as Error).message };
  }
}
