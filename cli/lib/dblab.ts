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
  isRetryableHttpStatus,
  redactSecretsForLog,
  redactTextSecrets,
  requestTimeoutSignal,
  stripControlsToSpace,
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
 * One POST to an rpc, with this file's debug logging and transport handling.
 *
 * Split out of `callDblabApi` so the body can be sent twice without duplicating
 * any of that: once with `accept_async`, and once without it on a platform old
 * enough not to have the argument (see `isUnknownRpcSignature`).
 */
async function postDblabRpc(
  rpc: string,
  bodyObj: Record<string, unknown>,
  params: { apiKey: string; apiBaseUrl: string; operation: string; debug?: boolean }
): Promise<{ response: Response; text: string }> {
  const { apiKey, apiBaseUrl, operation, debug } = params;
  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/${rpc}`);
  const body = JSON.stringify(bodyObj);

  // Org selector rides via buildAuthHeaders' activeOrgScope fallback (resolved
  // once in the CLI preAction hook), so every call — including the role-gated
  // DELETEs and every poll below — carries x-pgai-org under a global token.
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
  let text: string;
  try {
    text = await response.text();
  } catch (err) {
    // The body read is transport too — the request timeout can fire here, and
    // outside this try it surfaced as a bare DOMException with no operation.
    if (isFetchTimeout(err)) {
      throw new HttpRequestTimeoutError(operation, requestTimeout.timeoutMs);
    }
    throw new Error(describeFetchError(operation, base, err));
  }

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    // Clone create/status replies carry the live clone's db.password/connStr.
    console.error(`Debug: Response body: ${redactSecretsForLog(text)}`);
  }
  return { response, text };
}

/**
 * Did PostgREST refuse to RESOLVE `rpc`, rather than run it?
 *
 * PostgREST picks an rpc by the exact set of body keys, so `accept_async`
 * against a platform predating platform-all#810 answers 404 out of the schema
 * cache — measured on v9.0.1, `{"hint":"If a new function was created …",
 * "message":"Could not find the v1.dblab_api_call(accept_async, action,
 * instance_id, method) function …"}`. Nothing ran, which is what makes the
 * retry safe for a write: a refused `POST /clone` never reached the function.
 *
 * It must therefore never match an error raised from INSIDE the function.
 * Requiring the message to name the rpc we sent is what enforces that here,
 * rather than relying on the platform repo's PT404 keeping its own wording.
 */
export function isUnknownRpcSignature(status: number, text: string, rpc: string): boolean {
  if (status !== 404) return false;
  try {
    const parsed = JSON.parse(text) as { message?: unknown };
    return (
      typeof parsed.message === "string" &&
      /Could not find the .* function/.test(parsed.message) &&
      parsed.message.includes(`${rpc}(`)
    );
  } catch {
    return false;
  }
}

/**
 * The handle `v1.dblab_api_call` returns (HTTP 202) for an instance that is
 * reached over the job channel rather than dialled.
 *
 * `retry_safe` is on the wire and deliberately NOT read: this client never
 * re-sends, so honouring it could only weaken that.
 */
interface DblabAsyncHandle {
  pgai_async?: unknown;
  job_id?: unknown;
  retry_safe?: unknown;
  first_answer_estimate_s?: unknown;
  expires_in_s?: unknown;
}

/** One `v1.dblab_call_result` reply. `result` is the engine's own body. */
interface DblabCallResult {
  status?: unknown;
  outcome?: unknown;
  result?: unknown;
  error?: unknown;
  failure_class?: unknown;
}

/** A usable positive number of seconds, or null for anything else. */
function positiveSeconds(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
}

/**
 * Poll delays, in ms — the same ladder `pgai promql` uses against the same job
 * channel: fast at the start for a box on a short interval, doubling to a 15s
 * ceiling because each poll is a full api_token_check (a bcrypt per candidate
 * token in the org) and the fleet default pacing is 600s.
 */
function dblabPollDelay(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 15000);
}

/** The platform holds the slot for `expires_in_s`; waiting past that is pointless. */
const DBLAB_ASYNC_FALLBACK_WAIT_MS = 15 * 60 * 1000;

/**
 * Ceiling on the wait whatever the handle says, mirroring the platform's own
 * `c_expires_ceiling`. The clamp lives in another repo, and a type guard at a
 * deserialisation boundary does not get to assume it: an absurd
 * `expires_in_s` would otherwise be an unkillable hang in a script.
 */
const DBLAB_ASYNC_MAX_WAIT_MS = 60 * 60 * 1000;

/**
 * Consecutive unanswered polls tolerated before the wait is abandoned.
 *
 * Five spans the first four rungs of the ladder, so the budget is 15s when the
 * failures come back instantly and 140s when each burns its 25s timeout —
 * measured, not the one figure it looks like. That is short for a proxy that
 * answers 502 at once, and a 429 is spent from the same budget as an outage;
 * both are arguments for an elapsed-time bound, which is its own change.
 */
const DBLAB_POLL_MAX_CONSECUTIVE_FAILURES = 5;

/** Statuses on which the job is still moving. Anything else has stopped. */
const DBLAB_JOB_PENDING = new Set(["queued", "running"]);

/**
 * The warning for the ENQUEUE leg, which has no job id to name.
 *
 * `v1.dblab_api_call` commits the instance_jobs insert before the reply is
 * written, so a 5xx or a lost connection can leave a job the box will still
 * run. "Gateway Timeout" on its own reads as "nothing happened, try again".
 */
function dblabMayHaveLandedAdvice(): string {
  return (
    "The platform may have accepted this call even though the reply was lost — " +
    "check the instance's current state before running this command again."
  );
}

/**
 * What to tell a user whose wait ended without an answer.
 *
 * Never "run it again": the call may have landed, and the platform holds one
 * in-flight call per instance for up to an hour with nothing that returns its
 * id, so repeating the command either duplicates a write or is refused.
 */
function dblabNotResentAdvice(jobId: string): string {
  return (
    `The call (job ${jobId}) was NOT re-sent and may still complete on the instance — ` +
    "check its current state (e.g. `pgai dblab clone list`) before running this command again."
  );
}

/**
 * Neutralise every control byte except the line breaks.
 *
 * `formatHttpError` puts the detail and the hint on their own lines, so a
 * blanket strip would flatten every error this file throws. A newline can
 * push text around; ESC, BEL and CR can overwrite what was already read.
 */
function stripControlsKeepingLines(text: string): string {
  return text.split("\n").map(stripControlsToSpace).join("\n");
}

/**
 * Text the customer's box wrote, on its way to an operator's terminal.
 *
 * Scrub BEFORE stripping controls, and scrub twice. Stripping first turns a
 * control byte inside a credential into a space, and both credential matchers
 * stop at whitespace -- measured, it leaks half the secret. The second pass is
 * `redactTextSecrets` because `redactSecretsForLog` takes a JSON branch when
 * the body parses, and that branch only redacts by KEY: a connection string
 * under a benign key survives it (`formatHttpError` reaches for the same
 * helper). Stripping last is still the final word, since neither scrub emits
 * a control character.
 */
function scrubBoxText(text: string): string {
  return stripControlsToSpace(redactTextSecrets(redactSecretsForLog(text)))
    .replace(/\s+/g, " ")
    .trim();
}

type DblabPollOutcome =
  | { kind: "result"; res: DblabCallResult & { status: string } }
  | { kind: "transient"; why: string };

/**
 * One `dblab_call_result` read.
 *
 * A 5xx/429 or a transport blip returns `transient` rather than throwing: the
 * poll is a READ, so repeating it is free, and aborting the wait would strand a
 * write the box may already have run. `joe result` takes the same line.
 *
 * Every throw here abandons a call that is still in flight — a token rotation
 * or a suspended org mid-wait reaches these — so each one carries the job id
 * and the advice, not just the status.
 */
async function pollDblabCallOnce(
  jobIdRaw: string,
  jobId: string,
  params: { apiKey: string; apiBaseUrl: string; operation: string; debug?: boolean }
): Promise<DblabPollOutcome> {
  const { operation } = params;
  const advice = dblabNotResentAdvice(jobId);
  let response: Response;
  let text: string;
  try {
    ({ response, text } = await postDblabRpc("dblab_call_result", { p_job_id: jobIdRaw }, params));
  } catch (err) {
    // `postDblabRpc` throws only for transport failures — a timeout, DNS, TLS,
    // connection refused. An HTTP status always comes back as a response.
    return { kind: "transient", why: err instanceof Error ? err.message : String(err) };
  }
  if (!response.ok) {
    if (isRetryableHttpStatus(response.status)) {
      return { kind: "transient", why: `HTTP ${response.status}` };
    }
    throw new HttpStatusError(
      `${stripControlsKeepingLines(formatHttpError(operation, response.status, text, response.statusText))}\n${advice}`,
      response.status
    );
  }
  let res: DblabCallResult;
  try {
    res = JSON.parse(text) as DblabCallResult;
  } catch {
    throw new Error(
      `${operation}: failed to parse response: ${stripControlsKeepingLines(redactSecretsForLog(text))}. ${advice}`
    );
  }
  if (!res || typeof res.status !== "string") {
    throw new Error(
      `${operation}: dblab_call_result returned an unexpected reply: ${stripControlsKeepingLines(redactSecretsForLog(text))}. ${advice}`
    );
  }
  return { kind: "result", res: res as DblabCallResult & { status: string } };
}

/**
 * Wait for one enqueued engine call and return the engine's reply.
 *
 * This never re-sends the call, whatever the failure looks like — only
 * `dblab_call_result` is polled. The agent answers a POST/DELETE as failed
 * after ONE attempt precisely because the write may have landed, so a client
 * that re-enqueued on a timeout could create a second clone on the customer's
 * disk. Every give-up path says so and names the job instead (#390).
 */
export async function awaitDblabCall<T>(
  handle: DblabAsyncHandle,
  params: {
    apiKey: string;
    apiBaseUrl: string;
    operation: string;
    debug?: boolean;
    sleep?: (ms: number) => Promise<void>;
    // Injectable so a test drives the deadline instead of real elapsed time.
    now?: () => number;
  }
): Promise<T> {
  const { operation } = params;
  // Two ids on purpose. `jobIdRaw` goes back on the wire as `p_job_id` and
  // must survive byte for byte; `jobId` is the printed one, scrubbed because
  // ESC in it would repaint the very line warning the call is in flight.
  // Scrubbing is lossy (it round-trips through JSON), so it must not be the
  // value polled with.
  const jobIdRaw = typeof handle.job_id === "string" ? handle.job_id : "";
  const jobId = jobIdRaw ? scrubBoxText(jobIdRaw) : "";
  if (!jobId) {
    throw new Error(
      `${operation}: the platform accepted the call but returned no job_id. ` +
        dblabMayHaveLandedAdvice()
    );
  }
  const now = params.now ?? (() => Date.now());
  const sleep = params.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const waitMs = Math.min(
    (positiveSeconds(handle.expires_in_s) ?? 0) * 1000 || DBLAB_ASYNC_FALLBACK_WAIT_MS,
    DBLAB_ASYNC_MAX_WAIT_MS
  );
  const deadline = now() + waitMs;
  const estimate = positiveSeconds(handle.first_answer_estimate_s);
  let announced = false;

  let failures = 0;
  let lastStatus = "unknown";

  for (let attempt = 0; ; attempt++) {
    const outcome = await pollDblabCallOnce(jobIdRaw, jobId, params);

    if (outcome.kind === "transient") {
      failures += 1;
      if (failures >= DBLAB_POLL_MAX_CONSECUTIVE_FAILURES) {
        throw new Error(
          `${operation}: lost contact with the platform while waiting (${failures} polls in a row failed, ` +
            `last: ${outcome.why}). ${dblabNotResentAdvice(jobId)}`
        );
      }
    } else {
      failures = 0;
      const res = outcome.res;
      lastStatus = scrubBoxText(res.status);

      if (res.status === "done" && res.outcome === "ok") {
        // `?? null` so both routes agree: the synchronous arm answers an empty
        // engine 200 (DELETE /clone) with null, and an absent `result` here
        // would otherwise be undefined.
        return (res.result ?? null) as T;
      }
      // Anything not queued/running has stopped, including a status this client
      // does not know: waiting out the window on one would only turn a report
      // the platform already has into a timeout.
      if (!DBLAB_JOB_PENDING.has(res.status)) {
        const why =
          typeof res.error === "string" && res.error
            ? res.error
            : res.status === "expired"
              ? "no agent claimed it before it expired"
              : `outcome ${String(res.outcome ?? "unknown")}`;
        const klass =
          typeof res.failure_class === "string" && res.failure_class
            ? ` [${scrubBoxText(res.failure_class)}]`
            : "";
        // `expired` means nothing ran; every other stop may have landed a
        // write, so it gets the job id and the do-not-repeat advice.
        const advice = res.status === "expired" ? "" : ` ${dblabNotResentAdvice(jobId)}`;
        throw new Error(
          `${operation}: the DBLab instance did not complete the call (status ${scrubBoxText(res.status)}) — ` +
            `${scrubBoxText(why)}${klass}.${advice}`
        );
      }
    }

    // Say once why nothing is happening: the box only learns about the job on
    // its NEXT poll, and the estimate is a MEAN — jitter spreads it ±20%. Not
    // on a transient: that means the PLATFORM is unreachable, and naming the
    // instance would point the operator at the wrong system.
    if (!announced && outcome.kind === "result") {
      announced = true;
      console.error(
        `Waiting for the DBLab instance to pick up the call (job ${jobId}` +
          (estimate ? `, typically ~${estimate}s` : "") +
          ")..."
      );
    }

    const delay = dblabPollDelay(attempt);
    if (now() + delay >= deadline) {
      throw new Error(
        `${operation}: timed out waiting for the DBLab instance (job ${jobId}, last status ${lastStatus}). ` +
          dblabNotResentAdvice(jobId)
      );
    }
    await sleep(delay);
  }
}

/**
 * The enqueue POST, with the one thing `postDblabRpc` cannot know: whether
 * losing this reply could have left a write running.
 *
 * A transport failure or a timeout here is not "nothing happened" — the rpc
 * commits before the reply is written.
 */
async function enqueueDblabCall(
  bodyObj: Record<string, unknown>,
  params: { apiKey: string; apiBaseUrl: string; operation: string; debug?: boolean },
  isWrite: boolean
): Promise<{ response: Response; text: string }> {
  try {
    return await postDblabRpc("dblab_api_call", bodyObj, params);
  } catch (err) {
    if (!isWrite) throw err;
    // Wrap, never assign: a DOMException IS an Error but its `message` is a
    // getter with no setter, and that is exactly what an abort/timeout during
    // the body read produces — the assignment threw where the advice was most
    // needed. Nothing here branches on the class of a postDblabRpc throw.
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`${message}\n${dblabMayHaveLandedAdvice()}`, { cause: err });
  }
}

/**
 * Proxy a single call to the DBLab engine via `v1.dblab_api_call`, mirroring the
 * Console's `request('/rpc/dblab_api_call', { body: {instance_id, action, method,
 * data} })`. Returns the parsed JSON reply, or throws a formatted HTTP error.
 *
 * TWO ROUTES, and the caller sees neither (platform-all#810): an instance whose
 * agent polls the job channel has no URL to dial, so the platform enqueues the
 * call and answers HTTP 202 with a handle; this polls it out and returns the
 * engine's reply exactly as the synchronous route does. Every other instance
 * answers synchronously as before.
 */
async function callDblabApi<T>(params: DblabApiCallParams): Promise<T> {
  const { apiKey, apiBaseUrl, instanceId, action, method, data, operation, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!instanceId) {
    throw new Error("instanceId is required");
  }

  const bodyObj: Record<string, unknown> = {
    instance_id: instanceId,
    action,
    method,
  };
  if (data !== undefined) {
    bodyObj.data = data;
  }

  // `accept_async` says this client understands a handle. Without it the
  // platform refuses a job-backed instance with PT426 rather than returning one,
  // because a client that treats any 2xx as the engine's reply would render the
  // handle as a successful result.
  const rpcParams = { apiKey, apiBaseUrl, operation, debug };
  // A GET changes nothing, so only a write needs the "it may have landed" line.
  const isWrite = method.toLowerCase() !== "get";
  let { response, text } = await enqueueDblabCall(
    { ...bodyObj, accept_async: true },
    rpcParams,
    isWrite
  );

  if (isUnknownRpcSignature(response.status, text, "dblab_api_call")) {
    // A platform that predates #810. Nothing ran, so this is safe for a write.
    ({ response, text } = await enqueueDblabCall(bodyObj, rpcParams, isWrite));
  }

  if (response.status === 202) {
    // Guarded like the synchronous reply below: a bare parse here throws a
    // SyntaxError carrying neither the operation nor the redaction, and an
    // empty or HTML 202 off a proxy is exactly when that matters.
    let handle: unknown;
    try {
      handle = JSON.parse(text);
    } catch {
      // A 202 is proof the enqueue committed -- the rpc sets that status only
      // after the INSERT returned a job id -- so every way of failing behind
      // one says the call may still run, write or not.
      throw new Error(
        `${operation}: failed to parse response: ${stripControlsKeepingLines(redactSecretsForLog(text))}. ` +
          dblabMayHaveLandedAdvice()
      );
    }
    if (!handle || typeof handle !== "object") {
      throw new Error(
        `${operation}: the platform accepted the call but returned no handle. ` +
          dblabMayHaveLandedAdvice()
      );
    }
    return await awaitDblabCall<T>(handle as DblabAsyncHandle, rpcParams);
  }

  if (!response.ok) {
    // PostgREST maps custom `PTxyz` sqlstates to HTTP status `xyz`, so a
    // destructive verb denied by the backend's Admin/AllFeaturesUser role gate
    // surfaces here as HTTP 403. The RPC's user-facing message may ride in the
    // HTTP reason phrase (statusText) or the JSON body — pass both through.
    // A 5xx is different: the rpc may have committed, so a write says so. NOT
    // `isRetryableHttpStatus` — that answers "is retrying free?", which is the
    // poll's question. A 429 is a gateway refusing to forward, so nothing ran,
    // and telling the user to check state would suppress the right recovery.
    const lost = isWrite && response.status >= 500;
    // Stripped because `formatHttpError` redacts credentials but not control
    // bytes, and the engine's own text rides in `details` on this route --
    // without it the async arm is safe and the sync arm is not, for the same
    // command. The shared helper wants the same treatment, separately.
    throw new HttpStatusError(
      stripControlsKeepingLines(
        formatHttpError(operation, response.status, text, response.statusText)
      ) + (lost ? `\n${dblabMayHaveLandedAdvice()}` : ""),
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
    throw new Error(
      `${operation}: failed to parse response: ${stripControlsKeepingLines(redactSecretsForLog(text))}`
    );
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
