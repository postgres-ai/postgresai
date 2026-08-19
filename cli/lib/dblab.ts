/**
 * DBLab companion command surface (Joe API v2 · SPEC §8).
 *
 * Thin CLI client that **proxies the existing Platform DBLab API** — the very
 * same endpoints the Console (React) drives — to manage a project's own thin
 * clones, branches, and snapshots. Every verb goes through the generic proxy rpc
 * `v1.dblab_api_call(instance_id, method, action, data)`, mirroring
 * `packages/platform/src/api/{clones,branches,snapshots}/*` in the platform repo:
 *
 *   POST {base}/rpc/dblab_api_call
 *   body: { instance_id, action, method, data? }
 *
 * `method` is a lowercase HTTP verb, `action` is the leading-slash DBLab engine
 * path, and `data` (mutations only) is a nested JSON object — exactly the wire
 * shape the Console sends. Auth is the CLI's opaque org `access-token` header,
 * same as the joe/projects rpcs (`callRpc` in ./joe).
 *
 * Addressing is **project-centric**: callers pass `--project <id|alias>` and the
 * project's single DBLab instance is resolved to an `instance_id` (see
 * `resolveDblabInstanceId`). The destructive verbs — the HTTP DELETEs: clone
 * destroy (`DELETE /clone/<id>`), branch delete (`DELETE /branch/<name>`),
 * snapshot destroy (`DELETE /snapshot/<id>`) — are gated server-side: the
 * access token's OWNER must hold the Admin or AllFeaturesUser role in the
 * token org (the same role gate as `v1.joe_command_run`); clone reset is a
 * POST and is NOT gated. Read/list/create stay at plain org-token level. A
 * missing role on a DELETE surfaces here as a `PT403` → HTTP 403.
 */

import {
  HttpRequestTimeoutError,
  HttpStatusError,
  formatHttpError,
  maskSecret,
  normalizeBaseUrl,
  describeFetchError,
  isFetchTimeout,
  redactSecretsForLog,
  requestTimeoutSignal,
} from "./util";
import { listProjects, isNumericProjectRef, type ProjectListItem } from "./joe";
import { buildAuthHeaders } from "./org-scope";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DblabCommon {
  apiKey: string;
  apiBaseUrl: string;
  /** Resolved DBLab instance id (see `resolveDblabInstanceId`). */
  instanceId: string;
  debug?: boolean;
}

// ---------------------------------------------------------------------------
// Project → DBLab instance resolution
//
// Resolution rides the org-level projects listing rpc (`v1.projects_list`):
// its rows carry each project's `dblab_instance_id`, and the rpc authenticates
// with the CLI's opaque org `access-token` header (the same listing behind
// `pgai projects` and the joe verbs' `--project` resolution).
// ---------------------------------------------------------------------------

export interface ResolveDblabInstanceParams {
  apiKey: string;
  apiBaseUrl: string;
  /** `--project <id|alias>` — a numeric project id, or a project alias/name. */
  project: string;
  /** Optional org scope (narrows the projects listing). */
  orgId?: number;
  debug?: boolean;
}

/**
 * Resolve `--project <id|alias>` to the project's single DBLab `instance_id`.
 *
 * Lists the org's projects via `v1.projects_list` (the same call behind
 * `pgai projects`): a numeric ref matches `project_id`; anything else matches
 * `alias` / `name` (case-insensitive). Each project has at most one active
 * DBLab instance (`dblab_instance_id`). The returned id is a string so a
 * 64-bit id survives without JS number-precision loss.
 */
export async function resolveDblabInstanceId(params: ResolveDblabInstanceParams): Promise<string> {
  const { apiKey, apiBaseUrl, orgId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  const ref = String(params.project ?? "").trim();
  if (!ref) {
    throw new Error("project is required (--project <id|alias>)");
  }

  let projects: ProjectListItem[];
  try {
    projects = await listProjects({ apiKey, apiBaseUrl, orgId, debug });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to resolve project's DBLab instance: ${message}`);
  }

  const numeric = isNumericProjectRef(ref);
  const needle = ref.toLowerCase();
  const match = projects.find((p) => {
    if (numeric) {
      return String(p.project_id) === ref;
    }
    return (
      (p.alias !== null && p.alias.toLowerCase() === needle) ||
      (p.name !== null && p.name.toLowerCase() === needle)
    );
  });

  if (!match) {
    throw new Error(
      `No DBLab instance found for project '${ref}'. Run 'pgai projects' to see available projects.`
    );
  }
  if (match.dblab_instance_id == null) {
    throw new Error(
      `Project '${ref}' has no active DBLab instance. Register a DBLab instance for it in the Console first.`
    );
  }
  return String(match.dblab_instance_id);
}

// ---------------------------------------------------------------------------
// Low-level proxy caller — POST /rpc/dblab_api_call
// ---------------------------------------------------------------------------

interface DblabApiCallParams {
  apiKey: string;
  apiBaseUrl: string;
  instanceId: string;
  /** Leading-slash DBLab engine path, e.g. `/clone`, `/branches`, `/snapshots`. */
  action: string;
  /** Lowercase HTTP verb forwarded to the DBLab engine: get/post/patch/delete. */
  method: string;
  /** Optional request body (mutations only), forwarded as a nested JSON object. */
  data?: Record<string, unknown>;
  operation: string;
  debug?: boolean;
}

/**
 * Proxy a single call to the DBLab engine via `v1.dblab_api_call`, mirroring the
 * Console's `request('/rpc/dblab_api_call', { body: {instance_id, action, method,
 * data} })`. Returns the parsed JSON reply, or throws a formatted HTTP error.
 */
async function callDblabApi<T>(params: DblabApiCallParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, action, method, data, operation, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!instanceId) {
    throw new Error("instanceId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/dblab_api_call`);

  const bodyObj: Record<string, unknown> = {
    instance_id: instanceId,
    action,
    method,
  };
  if (data !== undefined) {
    bodyObj.data = data;
  }
  const body = JSON.stringify(bodyObj);

  // Org selector rides via buildAuthHeaders' activeOrgScope fallback (resolved
  // once in the CLI preAction hook), so every dblab_api_call — including the
  // role-gated DELETEs — carries x-pgai-org under a global token.
  const headers: Record<string, string> = buildAuthHeaders(apiKey);

  if (debug) {
    const debugHeaders = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    // Redact credential fields (clone create embeds a DB password) — the raw
    // body must never hit the log.
    console.error(`Debug: Request body: ${redactSecretsForLog(body)}`);
  }

  let response: Response;
  const requestTimeout = requestTimeoutSignal();
  try {
    response = await fetch(url.toString(), {
      method: "POST",
      headers,
      body,
      signal: requestTimeout.signal,
    });
  } catch (err) {
    if (isFetchTimeout(err)) {
      throw new HttpRequestTimeoutError(operation, requestTimeout.timeoutMs);
    }
    // Transport failure (connection refused, DNS, TLS, bad host/port) — surface
    // the real cause + URL rather than undici's opaque "fetch failed".
    throw new Error(describeFetchError(operation, base, err));
  }
  const text = await response.text();

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    // Clone create/status replies carry the live clone's db.password/connStr.
    console.error(`Debug: Response body: ${redactSecretsForLog(text)}`);
  }

  if (!response.ok) {
    // PostgREST maps custom `PTxyz` sqlstates to HTTP status `xyz`, so a
    // destructive verb denied by the backend's Admin/AllFeaturesUser role gate
    // surfaces here as HTTP 403. The RPC's user-facing message may ride in the
    // HTTP reason phrase (statusText) or the JSON body — pass both through.
    throw new HttpStatusError(
      formatHttpError(operation, response.status, text, response.statusText),
      response.status
    );
  }

  // Some DBLab actions (reset/destroy) reply with an empty body on success.
  if (text.trim() === "") {
    return null as unknown as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    // Non-JSON body — redact before embedding: this Error reaches CLI stderr
    // and must not bypass the debug-log redaction.
    throw new Error(`${operation}: failed to parse response: ${redactSecretsForLog(text)}`);
  }
}

// ===========================================================================
// Clones — create / list / status / reset / destroy
// ===========================================================================

export interface CreateCloneParams extends DblabCommon {
  /** Optional caller-chosen clone id (DBLab generates one when omitted). */
  cloneId?: string;
  /** Branch to clone from. */
  branch?: string;
  /** Snapshot id to clone from. */
  snapshotId?: string;
  /** Clone DB user (paired with `dbPassword`). */
  dbUser?: string;
  /** Clone DB password (paired with `dbUser`). */
  dbPassword?: string;
  /** Protect the clone from auto-deletion. */
  isProtected?: boolean;
}

/** Create a thin clone — `/clone` POST (Console `createClone`). */
export async function createClone<T = unknown>(params: CreateCloneParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, cloneId, branch, snapshotId, dbUser, dbPassword, isProtected, debug } = params;
  const data: Record<string, unknown> = { protected: Boolean(isProtected) };
  if (cloneId) data.id = cloneId;
  if (branch) data.branch = branch;
  if (snapshotId) data.snapshot = { id: snapshotId };
  if (dbUser && dbPassword) data.db = { username: dbUser, password: dbPassword };
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: "/clone", method: "post", data,
    operation: "Failed to create clone", debug,
  });
}

/** List clones — `/clones` GET. */
export async function listClones<T = unknown>(params: DblabCommon): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, debug } = params;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: "/clones", method: "get",
    operation: "Failed to list clones", debug,
  });
}

export interface CloneIdParams extends DblabCommon {
  cloneId: string;
}

/** Get a clone's status — `/clone/<id>` GET (Console `getClone`). */
export async function getClone<T = unknown>(params: CloneIdParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, cloneId, debug } = params;
  if (!cloneId) throw new Error("cloneId is required");
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: `/clone/${encodeURIComponent(cloneId)}`, method: "get",
    operation: "Failed to get clone", debug,
  });
}

export interface ResetCloneParams extends CloneIdParams {
  /** Snapshot to reset to; when omitted, resets to the latest snapshot. */
  snapshotId?: string;
  /** Reset to the latest snapshot (defaults true when no `snapshotId` is given). */
  latest?: boolean;
}

/** Reset a clone to a pristine snapshot — `/clone/<id>/reset` POST (Console `resetClone`). */
export async function resetClone<T = unknown>(params: ResetCloneParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, cloneId, snapshotId, latest, debug } = params;
  if (!cloneId) throw new Error("cloneId is required");
  const data: Record<string, unknown> = { latest: latest ?? !snapshotId };
  if (snapshotId) data.snapshotID = snapshotId;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: `/clone/${encodeURIComponent(cloneId)}/reset`, method: "post", data,
    operation: "Failed to reset clone", debug,
  });
}

/** Destroy a clone — `/clone/<id>` DELETE (Console `destroyClone`). */
export async function destroyClone<T = unknown>(params: CloneIdParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, cloneId, debug } = params;
  if (!cloneId) throw new Error("cloneId is required");
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: `/clone/${encodeURIComponent(cloneId)}`, method: "delete",
    operation: "Failed to destroy clone", debug,
  });
}

// ===========================================================================
// Branches — list / create / delete / log
// ===========================================================================

/** List branches — `/branches` GET (Console `getBranches`). */
export async function listBranches<T = unknown>(params: DblabCommon): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, debug } = params;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: "/branches", method: "get",
    operation: "Failed to list branches", debug,
  });
}

export interface CreateBranchParams extends DblabCommon {
  branchName: string;
  /** Parent branch to fork from. */
  baseBranch?: string;
  /** Snapshot id to base the branch on. */
  snapshotId?: string;
}

/** Create a branch — `/branch` POST (Console `createBranch`). */
export async function createBranch<T = unknown>(params: CreateBranchParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, branchName, baseBranch, snapshotId, debug } = params;
  if (!branchName) throw new Error("branchName is required");
  const data: Record<string, unknown> = { branchName };
  if (baseBranch) data.baseBranch = baseBranch;
  if (snapshotId) data.snapshotID = snapshotId;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: "/branch", method: "post", data,
    operation: "Failed to create branch", debug,
  });
}

export interface BranchNameParams extends DblabCommon {
  branchName: string;
}

/** Delete a branch — `/branch/<name>` DELETE (Console `deleteBranch`). */
export async function deleteBranch<T = unknown>(params: BranchNameParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, branchName, debug } = params;
  if (!branchName) throw new Error("branchName is required");
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: `/branch/${encodeURIComponent(branchName)}`, method: "delete",
    operation: "Failed to delete branch", debug,
  });
}

/** List a branch's snapshot log — `/branch/<name>/log` GET (Console `getSnapshotList`). */
export async function branchLog<T = unknown>(params: BranchNameParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, branchName, debug } = params;
  if (!branchName) throw new Error("branchName is required");
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: `/branch/${encodeURIComponent(branchName)}/log`, method: "get",
    operation: "Failed to fetch branch log", debug,
  });
}

// ===========================================================================
// Snapshots — list / create / destroy
// ===========================================================================

export interface ListSnapshotsParams extends DblabCommon {
  /** Filter snapshots by branch. */
  branchName?: string;
  /** Filter snapshots by dataset. */
  dataset?: string;
}

/** List snapshots — `/snapshots[?branch=&dataset=]` GET (Console `getSnapshots`). */
export async function listSnapshots<T = unknown>(params: ListSnapshotsParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, branchName, dataset, debug } = params;
  const qs = new URLSearchParams();
  const branch = branchName?.trim();
  if (branch) qs.append("branch", branch);
  if (dataset) qs.append("dataset", dataset);
  const action = `/snapshots${qs.toString() ? `?${qs.toString()}` : ""}`;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action, method: "get",
    operation: "Failed to list snapshots", debug,
  });
}

export interface CreateSnapshotParams extends DblabCommon {
  /** Clone to snapshot. */
  cloneId: string;
  /** Optional snapshot message. */
  message?: string;
}

/** Create a snapshot from a clone — `/branch/snapshot` POST (Console `createSnapshot`). */
export async function createSnapshot<T = unknown>(params: CreateSnapshotParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, cloneId, message, debug } = params;
  if (!cloneId) throw new Error("cloneId is required");
  const data: Record<string, unknown> = { cloneID: cloneId };
  if (message) data.message = message;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action: "/branch/snapshot", method: "post", data,
    operation: "Failed to create snapshot", debug,
  });
}

export interface DestroySnapshotParams extends DblabCommon {
  snapshotId: string;
  /** Force-delete even when dependent clones exist. */
  force?: boolean;
}

/** Destroy a snapshot — `/snapshot/<id>?force=<bool>` DELETE (Console `destroySnapshot`).
 *
 * The snapshot id is a MULTI-SEGMENT zfs path (`pool/branch/<b>/<clone>/r0@snap`),
 * which the DBLab engine routes as a wildcard path — it must be passed RAW,
 * exactly as the Console does. `encodeURIComponent` here turned `/`→`%2F` and
 * `@`→`%40`, which the engine rejects with 400 `invalid snapshot name given`
 * (verified live against DBLab CE 4.1.3). */
export async function destroySnapshot<T = unknown>(params: DestroySnapshotParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, snapshotId, force, debug } = params;
  if (!snapshotId) throw new Error("snapshotId is required");
  // DBLab expects the multi-segment ZFS snapshot name verbatim, but it is also
  // part of a URL. Restrict it to ZFS/path characters so a CLI caller cannot
  // inject query/fragment delimiters and alter the `force` parameter.
  if (!/^[a-zA-Z0-9_.@/:-]+$/.test(snapshotId)) {
    throw new Error("snapshotId contains invalid characters");
  }
  // `.` and `/` are legitimate inside a zfs snapshot name, but a dot-segment
  // (`.`/`..`) or empty segment could let the raw path traverse to a different
  // engine endpoint than `/snapshot/...` if any hop normalizes dot-segments.
  if (snapshotId.split("/").some((segment) => segment === "" || segment === "." || segment === "..")) {
    throw new Error("snapshotId contains an invalid path segment");
  }
  const action = `/snapshot/${snapshotId}?force=${Boolean(force)}`;
  return callDblabApi<T>({
    apiKey, apiBaseUrl, instanceId,
    action, method: "delete",
    operation: "Failed to destroy snapshot", debug,
  });
}
