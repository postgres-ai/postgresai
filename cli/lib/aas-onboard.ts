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
import {
  resolveBaseUrls,
  redactTextSecrets,
  stripControlsToSpace,
  whitespaceTolerantSecretPattern,
} from "./util";
import { orgScopeHeaders, type OrgScope } from "./org-scope";

const SA_NAME = "pgai-aas-collect";

/** Local Grafana base URL (published on the monitoring host). Overridable for tests/odd setups. */
function grafanaBaseUrl(): string {
  return (process.env.PGAI_GRAFANA_LOCAL_URL || "http://localhost:3000").replace(/\/+$/, "");
}

function grafanaAdminUser(): string {
  // The monitoring stack's compose hardcodes the Grafana admin user to
  // "monitor" (GF_SECURITY_ADMIN_USER: monitor), so default to that rather than
  // Grafana's stock "admin" — otherwise AAS arming logs in as the wrong user
  // and every datasource lookup 401s. An explicit env override still wins.
  return process.env.GF_SECURITY_ADMIN_USER || "monitor";
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
 * Grafana and mint a fresh glsa_ token. Returns the token or null on any failure.
 *
 * We deliberately do NOT prune prior tokens: deleting them here is racy — a
 * concurrent or repeated install could delete the token the platform currently
 * holds (stored encrypted), silently 401-ing collection until the next register.
 * The unique mint name already avoids 409s, and orphaned Viewer tokens are
 * benign; token hygiene is left to a separate, non-racy mechanism.
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

    // Unique token name avoids a 409 on a pre-existing name (no prune needed).
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

/**
 * Resolve the single Prometheus-typed datasource's numeric id on the local
 * Grafana. The monitoring stack's VictoriaMetrics datasource is type
 * "prometheus" (VM speaks PromQL), and the stack registers exactly one such
 * datasource — the same one the collector queries. 0 / API-not-ready → null
 * (a provisioning transient — the readiness loop retries); >1 → "ambiguous"
 * (a permanent misconfiguration — the loop stops at once), matching
 * v1.aas_onboard's >1 skip.
 */
export async function resolveDatasourceId(adminPassword: string, debug = false): Promise<number | "ambiguous" | null> {
  try {
    const res = await grafanaApi("GET", "/api/datasources", adminPassword);
    if (!res.ok) return null;
    const list = (await res.json().catch(() => [])) as Array<{ id?: unknown; type?: unknown }>;
    const prom = list.filter((d) => d.type === "prometheus");
    if (prom.length > 1) {
      // >1 is a permanent misconfiguration, not a provisioning transient: the
      // datasource count only grows as Grafana provisions, so retrying can never
      // resolve it. Signal a definitive skip so the readiness loop bails at once.
      if (debug) console.error(`Debug: AAS: ${prom.length} prometheus datasources (ambiguous); not retrying`);
      return "ambiguous";
    }
    if (prom.length === 0) {
      if (debug) console.error(`Debug: AAS: no prometheus datasource resolvable yet`);
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
  /**
   * The EFFECTIVE vCPU count the platform stored, from the RPC reply -- not what
   * we sent. The platform owns this number and keeps its own whenever the agent
   * sends 0, so only the reply says what the producer will actually read. 0 means
   * neither side knows, and the producer skips with no_vcpus until one does.
   * undefined = the platform did not report it (older deployment).
   */
  vcpus?: number;
  /** Armed for pull, but the platform has no Grafana url for the instance yet. */
  urlPending?: boolean;
  /** Which path the platform says it armed: "pull" or "jobs". */
  armedFor?: string;
}

/**
 * The success line to print, given what the platform reported back.
 *
 * "Registered" on its own is not true enough: an instance armed with an unknown
 * vCPU count is skipped by the producer with no_vcpus indefinitely, and the
 * operator has no way to tell that from a working one (#348).
 *
 * Order matters. A pending Grafana url clears itself minutes later, when the
 * deploy notify reports it; a missing vCPU count never does, so when both are
 * true the vCPU count is the one worth a human's attention.
 */
export function aasSuccessMessage(r: AasRegisterResult): string {
  // Which channel the platform says it armed, when it says so. The two are not
  // interchangeable from an operator's seat: "pull" means the platform will dial
  // into this box's Grafana, so its url and the SA token have to keep working;
  // "jobs" means the box answers on the instance job channel and the platform
  // opens nothing. It was parsed and then dropped on the floor (#382 F4).
  const armed =
    r.armedFor === "pull"
      ? " (armed for pull: the platform will query this instance's Grafana)"
      : r.armedFor === "jobs"
        ? " (armed for the job channel: this instance answers its own queries)"
        : "";
  const base = `AAS auto-collection registered${armed}`;
  // Order matters: a pending url clears itself minutes later when the deploy
  // notify lands, a missing vCPU count never does. Report the one needing a human.
  if (r.vcpus === 0) {
    return `${base} — collection stays OFF until a source-DB vCPU count is known (the platform stamps it at provision time; --vcpus <n> sets it for a manual install)`;
  }
  if (r.urlPending) {
    return `${base} — collection starts once this deploy reports the Grafana url to the platform`;
  }
  return base;
}

/** Cap on platform-supplied error text admitted into a reason / log line. */
const PLATFORM_ERROR_MAX = 300;

/**
 * Turn a PostgREST error body into one short, safe line.
 *
 * Logging the bare status is what hid platform-all#778 for three months: every
 * console-provisioned box printed "platform returned HTTP 400" while the
 * platform was saying "this monitoring instance has no Grafana url yet". A
 * status with no reason is not actionable, and nobody goes looking for a
 * warning that says nothing.
 *
 * The body is NOT trusted. It is the platform's, this request carries the org
 * API token and a freshly minted glsa_ Grafana token, and the text is printed
 * to a terminal. So the ORDER below is load-bearing, and it is the order a
 * first version got wrong (#382 F1):
 *
 *   1. NORMALISE FIRST — strip control characters and flatten whitespace.
 *      Scrubbing first and flattening afterwards is exploitable: a secret the
 *      platform echoed with a line break inside it matches no literal, and the
 *      flatten then REASSEMBLES it on one line, one space-deletion from usable.
 *      Normalising first means every later matcher sees one canonical form.
 *   2. PATTERN scrub via the shared redactTextSecrets (cli/lib/util.ts), the
 *      same one formatHttpError uses — credential-named pairs and URL userinfo,
 *      i.e. secrets we did NOT send and cannot match by value. Reused rather
 *      than reimplemented; two redactors drifting apart is its own bug.
 *   3. BY-VALUE scrub of exactly what this request sent, with a
 *      whitespace-TOLERANT pattern so step 1 cannot be worked around by a body
 *      that was wrapped before we ever saw it.
 *   4. SHAPE scrub for a glsa_ token we did not send. Best-effort only: once an
 *      unknown token has whitespace in it, nothing distinguishes "the token
 *      continues" from "the next word", so this covers the contiguous case and
 *      step 3 covers every token we actually hold.
 *   5. CAP, with a marker, so a reader can tell the reason was cut short.
 *
 * `details` is preferred over `message`: a plpgsql `raise ... using detail = ...`
 * lands there, while `message` is usually just the status prose ("Bad Request").
 */
export function formatPlatformError(body: unknown, secrets: string[] = []): string {
  if (!body || typeof body !== "object") return "";
  const b = body as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === "string" && v.trim() !== "" ? v.trim() : "");
  const text = str(b.details) || str(b.message);
  if (!text) return "";
  const code = str(b.code);

  // 1. normalise
  let out = stripControlsToSpace(code ? `${code}: ${text}` : text)
    .replace(/\s+/g, " ")
    .trim();

  // 2. pattern scrub (shared with formatHttpError)
  out = redactTextSecrets(out);

  // 3. by-value scrub, whitespace-tolerant
  for (const secret of secrets) {
    // A short "secret" would match everywhere and redact the whole message; the
    // real ones are far longer than this floor.
    const compact = secret.replace(/\s+/g, "");
    if (compact.length < 8) continue;
    out = out.replace(whitespaceTolerantSecretPattern(compact), "[redacted]");
  }

  // 4. shape scrub
  out = out.replace(/glsa_[A-Za-z0-9_-]+/g, "[redacted]");

  // 5. cap, and say so
  return out.length > PLATFORM_ERROR_MAX
    ? `${out.slice(0, PLATFORM_ERROR_MAX - 1).trimEnd()}\u2026`
    : out;
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
    // Grafana-readiness polling for the datasource lookup (Grafana has just
    // been started by `compose up`). Defaults: 20 attempts × 3s.
    datasourceMaxAttempts?: number;
    datasourceRetryDelayMs?: number;
    orgScope?: OrgScope;
  }
): Promise<AasRegisterResult> {
  const debug = !!opts.debug;
  try {
    if (!apiKey || !instanceId) return { ok: false, reason: "missing api key or instance id" };

    // No client-side vcpus gate (#683; reverses the work-item-260 finding-3
    // gate, whose "confirm the RPC gates on vcpus > 0" premise did not hold).
    // No provisioning path passes --vcpus, so gating here disabled hands-off
    // onboarding everywhere. The RPC takes 0 as "unknown" and never clobbers
    // the platform's own value, so always register.
    const labels = resolveAasLabels(opts.instancesPath);
    if (!labels) return { ok: false, reason: "could not determine a single (cluster, node_name) target" };

    // Grafana was just started by `compose up`; it needs time to create its
    // admin user, provision datasources, and serve its API. Querying too early
    // makes the datasource lookup fail transiently, so poll until it resolves
    // (best-effort, capped — the install never blocks on this).
    const maxAttempts = opts.datasourceMaxAttempts ?? 20;
    const retryDelayMs = opts.datasourceRetryDelayMs ?? 3000;
    let datasourceId: number | null = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const resolved = await resolveDatasourceId(opts.grafanaPassword, debug);
      if (typeof resolved === "number") { datasourceId = resolved; break; }
      // "ambiguous" (>1 prometheus datasource) is permanent — retrying can't fix
      // it, so stop polling immediately instead of waiting out the whole budget.
      if (resolved === "ambiguous") break;
      if (attempt < maxAttempts) {
        if (debug) console.error(`Debug: AAS: datasource not resolvable yet (attempt ${attempt}/${maxAttempts}); waiting for Grafana…`);
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }
    }
    if (datasourceId == null) return { ok: false, reason: "could not resolve the Prometheus datasource id" };

    const saToken = await mintAasServiceAccountToken(opts.grafanaPassword, debug);
    if (!saToken) return { ok: false, reason: "could not mint a Grafana service-account token" };

    const { apiBaseUrl } = resolveBaseUrls({ apiBaseUrl: opts.apiBaseUrl });
    const url = `${apiBaseUrl}/rpc/monitoring_instance_aas_register`;
    const doFetch = opts.fetchImpl || fetch;
    if (debug) console.error(`Debug: AAS: POST ${url} (cluster=${labels.cluster}, node=${labels.node}, vcpus=${opts.vcpus}, ds=${datasourceId})`);

    const res = await doFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...orgScopeHeaders(opts.orgScope) },
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
      // Read the platform's own reason, scrubbed and capped by
      // formatPlatformError. Reporting the status alone is what made #778
      // invisible: "HTTP 400" gives an operator nothing to act on, while the
      // platform was naming the exact precondition that failed.
      const detail = formatPlatformError(
        await res.json().catch(() => null),
        [apiKey, saToken],
      );
      const suffix = detail ? ` — ${detail}` : "";
      if (debug) console.error(`Debug: AAS register failed: HTTP ${res.status}${suffix}`);
      return { ok: false, reason: `platform returned HTTP ${res.status}${suffix}` };
    }
    // Read the reply instead of discarding it (#348). The effective vcpus is the
    // platform's own value, which is the only one the producer will read, and
    // url_pending says whether anything can collect yet. A body we cannot parse
    // is not a failure -- the registration landed -- so every field stays
    // undefined and the caller falls back to the plain success line.
    const reply = (await res.json().catch(() => null)) as Record<string, unknown> | null;
    return {
      ok: true,
      vcpus: typeof reply?.vcpus === "number" ? reply.vcpus : undefined,
      urlPending: typeof reply?.url_pending === "boolean" ? reply.url_pending : undefined,
      armedFor: typeof reply?.armed_for === "string" ? reply.armed_for : undefined,
    };
  } catch (err) {
    if (debug) console.error(`Debug: AAS register error: ${(err as Error).message}`);
    return { ok: false, reason: (err as Error).message };
  }
}
