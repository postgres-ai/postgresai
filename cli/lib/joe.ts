import {
  DEFAULT_HTTP_REQUEST_TIMEOUT_MS,
  HttpRequestTimeoutError,
  HttpStatusError,
  formatHttpError,
  isRetryableHttpStatus,
  maskSecret,
  normalizeBaseUrl,
  describeFetchError,
  isFetchTimeout,
  redactSecretsForLog,
  requestTimeoutSignal,
  isUnknownRpcSignature,
  positiveSeconds,
  asyncPollDelayMs,
} from "./util";
import { buildAuthHeaders } from "./org-scope";

/**
 * Joe API v2 client (`postgres-ai` CLI surface) — synchronous contract.
 *
 * Every Joe verb is a thin raw-text builder over the platform rpc
 * `v1.joe_command_run(instance_id, command, accept_async)`: the command text is sent RAW
 * (exactly what a console user could type at Joe — `plan select …`,
 * `exec create index …`, `\d users`), Joe dispatches the verb itself, and the
 * CLI polls `v1.joe_command_output(command_id)` until the status is terminal
 * (`ok`/`error`). No queue, no session store, no idempotency keys — a fresh
 * Joe session per run (Issue #438, supersedes the async !346 surface).
 */

/** The Joe verb set (mirrors Joe's own dispatcher; `describe` = the \d family). */
export const JOE_COMMANDS = [
  "plan",
  "explain",
  "exec",
  "hypo",
  "activity",
  "terminate",
  "reset",
  "describe",
] as const;

export type JoeCommand = (typeof JOE_COMMANDS)[number];

/** Output lifecycle: `pending` while Joe has not posted, then `ok`/`error`. */
export type JoeOutputStatus = "pending" | "ok" | "error";

/** Default one-shot poll budget (≤ 25 s, then resume by command id). */
export const DEFAULT_BUDGET_MS = 25_000;
const DEFAULT_POLL_INTERVAL_MS = 800;

/**
 * The job channel's budget, when the handle does not say.
 *
 * Derived from the platform's own clamp: public.joe_call_enqueue_job sizes
 * expires_in_s as 1.5x the box's poll interval bounded to
 * [c_expires_floor 10 minutes, c_expires_ceiling 1 hour] (read at platform-all
 * main). So a slot is held for AT LEAST ten minutes even when the field is
 * missing, and for at most an hour whatever the handle claims — past that the
 * sweep has expired the job and no answer can arrive.
 */
const JOE_ASYNC_FALLBACK_WAIT_MS = 10 * 60 * 1000;
const JOE_ASYNC_MAX_WAIT_MS = 60 * 60 * 1000;

/**
 * Consecutive unanswered polls tolerated on the job channel, where the window is
 * minutes rather than 25s: a platform answering every poll 5xx must not turn
 * into an hour-long hang. Five, as on the DBLab twin — on this ladder that is
 * 30s of pacing when the failures come back at once and ~155s when each burns
 * its 25s timeout. The dial path is unchanged: its budget already bounds it.
 */
const JOE_ASYNC_MAX_POLL_FAILURES = 5;

// ---------------------------------------------------------------------------
// Response shapes (the joe_command_output contract — mocked in tests)
// ---------------------------------------------------------------------------

/**
 * The FULL raw result row `v1.joe_command_output` returns once Joe has posted.
 * While `status` is `pending` only `command_id`/`status`/`created_at` are
 * present. `plan_json` and `plan_execution_json` arrive structured — the rpc
 * unwraps both from their stored jsonb-string form.
 */
export interface JoeCommandOutput {
  command_id: string;
  status: JoeOutputStatus;
  created_at?: string | null;
  command?: string | null;
  query?: string | null;
  queryid?: string | null;
  response?: string | null;
  plan_text?: string | null;
  plan_json?: unknown;
  plan_execution_text?: string | null;
  plan_execution_json?: unknown;
  stats?: string | null;
  recommendations?: string | null;
  error?: string | null;
}

export interface ProjectListItem {
  project_id: number | string;
  alias: string | null;
  name: string | null;
  /** Whether the project's single Joe instance is ready for Joe API v2. */
  joe_ready: boolean;
  /** Whether the project's DBLab tunnel is connected. */
  tunnel: boolean;
  /** The project's active JOE instance id — the `joe_command_run` target. */
  instance_id: number | string | null;
  /** The project's active DBLAB instance id (not used by the Joe verbs). */
  dblab_instance_id: number | string | null;
  /**
   * Every active MONITORING instance uuid of the project, newest first
   * (`pgai promql`). Undefined when the platform predates the column, so the
   * caller can tell "no instance" from "cannot know".
   */
  monitoring_instance_ids?: string[];
}

// ---------------------------------------------------------------------------
// Low-level rpc caller
// ---------------------------------------------------------------------------

interface RpcCallParams {
  apiKey: string;
  apiBaseUrl: string;
  fn: string;
  body: Record<string, unknown>;
  operation: string;
  debug?: boolean;
  timeoutMs?: number;
  /**
   * Re-sent ONCE, verbatim, when PostgREST could not RESOLVE `fn` with the keys
   * of `body` (an argument this platform predates — see isUnknownRpcSignature).
   * Nothing ran, so this is not a re-send of the command.
   */
  fallbackBody?: Record<string, unknown>;
}

async function callRpc<T>(params: RpcCallParams): Promise<T> {
  return (await callRpcRaw<T>(params)).data;
}

/** As {@link callRpc}, plus the reply's HTTP status (202 = an async handle). */
async function callRpcRaw<T>(params: RpcCallParams): Promise<{ data: T; status: number }> {
  const { apiKey, apiBaseUrl, fn, body, operation, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/${fn}`);

  // The org selector rides along via buildAuthHeaders' activeOrgScope fallback:
  // the CLI resolves it once in its preAction hook, so every joe rpc (run,
  // output, projects_list) carries x-pgai-org under a global token.
  const headers: Record<string, string> = buildAuthHeaders(apiKey);

  const send = async (bodyObj: Record<string, unknown>): Promise<{ response: Response; text: string }> => {
    const payload = JSON.stringify(bodyObj);

    if (debug) {
      const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
      console.error(`Debug: POST URL: ${url.toString()}`);
      console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
      // Redact credential-shaped fields before logging (mirrors the access-token
      // header masking above).
      console.error(`Debug: Request body: ${redactSecretsForLog(payload)}`);
    }

    let response: Response;
    const requestTimeout = requestTimeoutSignal(params.timeoutMs);
    try {
      response = await fetch(url.toString(), {
        method: "POST",
        headers,
        body: payload,
        signal: requestTimeout.signal,
      });
    } catch (err) {
      if (isFetchTimeout(err)) {
        throw new HttpRequestTimeoutError(operation, requestTimeout.timeoutMs);
      }
      // A transport failure (connection refused, DNS, TLS, bad host/port) never
      // reaches `response.ok`; undici throws with the real reason in `err.cause`.
      // Surface it — a bare "fetch failed" hides which URL/why. See util.describeFetchError.
      throw new Error(describeFetchError(operation, base, err));
    }

    const text = await response.text();

    if (debug) {
      console.error(`Debug: Response status: ${response.status}`);
      console.error(`Debug: Response body: ${redactSecretsForLog(text)}`);
    }
    return { response, text };
  };

  let { response, text } = await send(body);

  if (params.fallbackBody && isUnknownRpcSignature(response.status, text, fn)) {
    // PostgREST refused to resolve the rpc, so the function never ran: no row
    // was written and nothing was enqueued. Re-send the older body once.
    ({ response, text } = await send(params.fallbackBody));
  }

  if (!response.ok) {
    // PostgREST maps a custom `PTxyz` sqlstate to HTTP status `xyz`, so PT403 →
    // HTTP 403, PT404 → 404, etc. The RPC's user-facing message may ride in the
    // HTTP reason phrase (statusText) or the JSON body — pass both through.
    // The status rides on the Error so the poll loop can classify retryability.
    throw new HttpStatusError(
      formatHttpError(operation, response.status, text, response.statusText),
      response.status
    );
  }

  try {
    return { data: JSON.parse(text) as T, status: response.status };
  } catch {
    // Non-JSON body — redact before embedding: this Error reaches CLI stderr
    // and must not bypass the debug-log redaction.
    throw new Error(`${operation}: failed to parse response: ${redactSecretsForLog(text)}`);
  }
}

// ---------------------------------------------------------------------------
// Individual rpc client functions
// ---------------------------------------------------------------------------

export interface StartCommandParams {
  apiKey: string;
  apiBaseUrl: string;
  /** The project's Joe instance id (resolve via {@link resolveJoeInstanceId}). */
  instanceId: number | string;
  /** The RAW command text Joe dispatches (e.g. `plan select 1`, `\d users`). */
  command: string;
  debug?: boolean;
  timeoutMs?: number;
}

/**
 * The HTTP 202 handle. `retry_safe` is on the wire and deliberately NOT read:
 * this client never re-sends, so honouring it could only weaken that.
 */
interface JoeAsyncHandle {
  command_id?: unknown;
  job_id?: unknown;
  first_answer_estimate_s?: unknown;
  expires_in_s?: unknown;
}

/** What the run returned: the poll key, and how long the platform will wait. */
interface StartedCommand {
  commandId: string;
  /** The handle's clamped budget — null on the dial path, which has answered. */
  asyncWaitMs: number | null;
  /** When the command reaches Joe, per the handle; Joe's own time is extra. */
  firstAnswerEstimateS: number | null;
}

/** The wait the handle asks for: its expires_in_s, or the platform's floor. */
function joeAsyncWaitMs(handle: JoeAsyncHandle): number {
  const expires = positiveSeconds(handle.expires_in_s);
  return Math.min(
    expires ? expires * 1000 : JOE_ASYNC_FALLBACK_WAIT_MS,
    JOE_ASYNC_MAX_WAIT_MS
  );
}

/**
 * The command id out of the HTTP 202 handle (`pgai_async: "joe_call"`).
 *
 * An unrecognised shape is an ERROR, never a pending state: polling something
 * that is not a command id would report PT400 until the budget ran out with the
 * real id already thrown away. `retry_safe` is on the wire and deliberately NOT
 * read — this client never re-sends a command, and a delivery whose answer was
 * lost may well have reached Joe and run the user's query.
 */
function asyncHandleCommandId(reply: unknown, operation: string): string {
  const commandId = (reply as { command_id?: unknown } | null)?.command_id;
  if (typeof commandId !== "string" || !/^[0-9]+$/.test(commandId)) {
    throw new Error(
      `${operation}: the platform accepted the command over the job channel but ` +
        `returned no command id: ${redactSecretsForLog(JSON.stringify(reply))}`
    );
  }
  return commandId;
}

/**
 * Start a Joe command (`v1.joe_command_run`); returns the command id.
 * The rpc returns the id as a JSON string and it stays a string end-to-end —
 * a bigint id would lose precision beyond 2^53 as a JS number.
 *
 * TWO ROUTES since platform-all#840, and the caller sees neither: an instance
 * whose agent polls the job channel has no URL to dial, so the platform ENQUEUES
 * the command and answers HTTP 202 with a handle carrying the SAME command_id —
 * `joe_command_output` keeps working unchanged. `accept_async` is the opt-in
 * that route requires; without it such an instance is refused with PT426 and
 * nothing is enqueued. Every dialled instance answers synchronously as before.
 */
export async function startCommand(params: StartCommandParams): Promise<string> {
  return (await startCommandDetailed(params)).commandId;
}

/** As {@link startCommand}, plus what the job-channel handle says about waiting. */
async function startCommandDetailed(params: StartCommandParams): Promise<StartedCommand> {
  const { apiKey, apiBaseUrl, instanceId, command, debug, timeoutMs } = params;
  if (!String(command ?? "").trim()) {
    throw new Error("command text is required");
  }
  const operation = "Failed to run Joe command";
  const body: Record<string, unknown> = { instance_id: instanceId, command };
  const { data, status } = await callRpcRaw<unknown>({
    apiKey,
    apiBaseUrl,
    fn: "joe_command_run",
    // `accept_async` says this client understands a handle.
    body: { ...body, accept_async: true },
    // A platform predating #840 has no such argument, so PostgREST cannot
    // resolve the rpc at all and the command was never enqueued.
    fallbackBody: body,
    operation,
    debug,
    timeoutMs,
  });
  if (status === 202) {
    const handle = (data ?? {}) as JoeAsyncHandle;
    return {
      commandId: asyncHandleCommandId(data, operation),
      asyncWaitMs: joeAsyncWaitMs(handle),
      firstAnswerEstimateS: positiveSeconds(handle.first_answer_estimate_s),
    };
  }
  if (typeof data !== "string" || !/^[0-9]+$/.test(data)) {
    throw new Error(
      `${operation}: expected a command id string, got: ${redactSecretsForLog(JSON.stringify(data))}`
    );
  }
  return { commandId: data, asyncWaitMs: null, firstAnswerEstimateS: null };
}

export interface CommandOutputParams {
  apiKey: string;
  apiBaseUrl: string;
  commandId: string;
  debug?: boolean;
  timeoutMs?: number;
}

/**
 * Poll a command's output (`v1.joe_command_output`) — returns the status AND
 * the full result body in one call (`pending` until Joe posts the result).
 */
export async function getCommandOutput(params: CommandOutputParams): Promise<JoeCommandOutput> {
  const { apiKey, apiBaseUrl, commandId, debug, timeoutMs } = params;
  if (!commandId) {
    throw new Error("commandId is required");
  }
  return callRpc<JoeCommandOutput>({
    apiKey,
    apiBaseUrl,
    fn: "joe_command_output",
    body: { command_id: commandId },
    operation: "Failed to fetch command output",
    debug,
    timeoutMs,
  });
}

export interface ListProjectsParams {
  apiKey: string;
  apiBaseUrl: string;
  orgId?: number;
  debug?: boolean;
}

interface RawProjectRow {
  project_id?: number | string;
  alias?: string | null;
  name?: string | null;
  joe_ready?: boolean;
  tunnel?: boolean;
  instance_id?: number | string | null;
  dblab_instance_id?: number | string | null;
  monitoring_instance_ids?: string[] | null;
}

function preserveIntegerId(value: number | string): number | string {
  if (typeof value === "number") return value;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : value;
}

function normalizeProjectRow(row: RawProjectRow): ProjectListItem {
  return {
    project_id: preserveIntegerId(row.project_id ?? 0),
    alias: row.alias ?? null,
    name: row.name ?? null,
    joe_ready: Boolean(row.joe_ready ?? false),
    tunnel: Boolean(row.tunnel ?? false),
    instance_id: row.instance_id == null ? null : preserveIntegerId(row.instance_id),
    dblab_instance_id: row.dblab_instance_id == null ? null : preserveIntegerId(row.dblab_instance_id),
    ...(row.monitoring_instance_ids !== undefined && {
      monitoring_instance_ids: row.monitoring_instance_ids ?? [],
    }),
  };
}

/**
 * List the org's projects (org-level discovery — NOT a Joe endpoint).
 * Surfaces the per-project `joe_ready` + `tunnel` state and the Joe
 * `instance_id` the run rpc keys on.
 */
export async function listProjects(params: ListProjectsParams): Promise<ProjectListItem[]> {
  const { apiKey, apiBaseUrl, orgId, debug } = params;
  const body: Record<string, unknown> = {};
  if (typeof orgId === "number") {
    body.org_id = orgId;
  }
  const rows = await callRpc<RawProjectRow[]>({
    apiKey,
    apiBaseUrl,
    fn: "projects_list",
    body,
    operation: "Failed to list projects",
    debug,
  });
  if (!Array.isArray(rows)) {
    return [];
  }
  return rows.map(normalizeProjectRow);
}

// ---------------------------------------------------------------------------
// Project id-or-alias → Joe instance resolution
// ---------------------------------------------------------------------------

/** A bare numeric `--project` value is a project id; anything else is an alias/name. */
export function isNumericProjectRef(ref: string): boolean {
  return /^[0-9]+$/.test(ref.trim());
}

export interface ResolveInstanceParams {
  apiKey: string;
  apiBaseUrl: string;
  project: string;
  orgId?: number;
  debug?: boolean;
}

/**
 * Resolve `--project <id|alias>` to the project's Joe `instance_id` (what
 * `joe_command_run` keys on). Unlike a pure project-id resolver, a numeric ref
 * still needs the projects listing — the instance id lives there. Accepts a
 * numeric project id, or an alias/name (case-insensitive).
 */
export async function resolveJoeInstanceId(params: ResolveInstanceParams): Promise<number | string> {
  const ref = String(params.project ?? "").trim();
  if (!ref) {
    throw new Error("project is required (--project <id|alias>)");
  }
  const projects = await listProjects({
    apiKey: params.apiKey,
    apiBaseUrl: params.apiBaseUrl,
    orgId: params.orgId,
    debug: params.debug,
  });
  const needle = ref.toLowerCase();
  const match = isNumericProjectRef(ref)
    ? projects.find((p) => String(p.project_id) === String(preserveIntegerId(ref)))
    : projects.find(
        (p) =>
          (p.alias !== null && p.alias.toLowerCase() === needle) ||
          (p.name !== null && p.name.toLowerCase() === needle)
      );
  if (!match) {
    throw new Error(
      `Project not found for id/alias/name '${ref}'. Run 'pgai projects' to see available projects.`
    );
  }
  if (match.instance_id == null) {
    throw new Error(
      `Project '${ref}' has no Joe instance. Run 'pgai projects' to see which projects have Joe ready.`
    );
  }
  return match.instance_id;
}

// ---------------------------------------------------------------------------
// Raw command text builders (what Joe's /webui/command dispatches)
// ---------------------------------------------------------------------------

/** The \d-family variants Joe's psql allowlist accepts (`describe --variant`). */
export const DESCRIBE_VARIANTS = [
  "\\d",
  "\\d+",
  "\\dt",
  "\\dt+",
  "\\di",
  "\\di+",
  "\\l",
  "\\l+",
  "\\dv",
  "\\dv+",
  "\\dm",
  "\\dm+",
] as const;

export interface JoeVerbInput {
  /** The verb's positional payload: SQL, hypo tail, pid, or object name. */
  arg?: string | null;
  /** describe only: the \d-family variant (default `\d`). */
  variant?: string | null;
}

/**
 * Build the RAW command text for a verb — exactly what a console user could
 * type at Joe. The server adds no prefix and does no verb inspection, so this
 * string is the whole contract (`plan <sql>`, `terminate <pid>`, `\d+ users`).
 */
export function buildJoeCommandText(command: JoeCommand, input: JoeVerbInput = {}): string {
  const arg = String(input.arg ?? "").trim();
  switch (command) {
    case "plan":
    case "explain":
    case "exec":
    case "hypo": {
      if (!arg) {
        throw new Error(`${command} requires an argument`);
      }
      return `${command} ${arg}`;
    }
    case "activity":
    case "reset":
      return command;
    case "terminate": {
      // A pid must be a bare positive integer — parseInt() would silently
      // accept "12x"/"−5"/"1.5" and terminate the WRONG backend.
      if (!/^[1-9][0-9]*$/.test(arg)) {
        throw new Error("pid must be a positive integer");
      }
      return `terminate ${arg}`;
    }
    case "describe": {
      if (!arg) {
        throw new Error("describe requires an object name");
      }
      const variant = String(input.variant ?? "\\d").trim();
      if (!(DESCRIBE_VARIANTS as readonly string[]).includes(variant)) {
        throw new Error(
          `Unsupported describe variant '${variant}'. Supported: ${DESCRIBE_VARIANTS.join(" ")}`
        );
      }
      return `${variant} ${arg}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Run-then-poll one-shot
// ---------------------------------------------------------------------------

export interface RunCommandParams {
  apiKey: string;
  apiBaseUrl: string;
  instanceId: number | string;
  /** The RAW command text (see {@link buildJoeCommandText}). */
  command: string;
  budgetMs?: number;
  pollIntervalMs?: number;
  debug?: boolean;
  /** Injectable clock/sleep for deterministic tests. */
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

export interface RunCommandOutcome {
  commandId: string;
  status: JoeOutputStatus;
  /** Populated once the command reaches a terminal state (ok/error). */
  output: JoeCommandOutput | null;
  /** True when the ≤ budget one-shot expired before a terminal state — resume by id. */
  budgetExpired: boolean;
  /**
   * The budget actually used, in ms. Not always the caller's: an enqueued
   * command is given the platform's own window, so the expiry hint has to
   * report this rather than what `--budget` did or did not say.
   */
  budgetMs: number;
}

const defaultSleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/**
 * Run a raw command then poll `joe_command_output` within the one-shot budget.
 * On a terminal state returns the full output; on budget expiry returns a
 * resume handle (`budgetExpired: true`) so the caller can
 * `pgai joe result <command_id>` later.
 *
 * TWO BUDGETS, because the two routes answer on different clocks: a dialled Joe
 * already has the command and answers a poll or two later, while an enqueued one
 * waits for its box to poll the job channel. The job channel therefore gets the
 * window the platform itself holds the slot for (the handle's `expires_in_s`),
 * and its polls are paced by the shared ladder rather than every 800ms — each
 * poll is a full api_token_check. An explicit `budgetMs` wins over both.
 */
export async function runCommand(params: RunCommandParams): Promise<RunCommandOutcome> {
  const { apiKey, apiBaseUrl, instanceId, command, debug } = params;
  // Defensive: only a finite budget is honored. NaN survives `??` (it is
  // neither null nor undefined) and would make `deadline` NaN — `now() >= NaN`
  // is always false, i.e. an UNBOUNDED poll loop.
  const explicitBudgetMs =
    typeof params.budgetMs === "number" && Number.isFinite(params.budgetMs) && params.budgetMs >= 0
      ? params.budgetMs
      : null;
  const now = params.now ?? Date.now;
  const sleep = params.sleep ?? defaultSleep;

  const started = await startCommandDetailed({ apiKey, apiBaseUrl, instanceId, command, debug });
  const commandId = started.commandId;

  const budgetMs = explicitBudgetMs ?? started.asyncWaitMs ?? DEFAULT_BUDGET_MS;
  // The caller's pacing when given; otherwise 800ms on the dial path and the
  // backoff ladder on the job channel, whose window is minutes long.
  const pollDelayMs = (attempt: number): number =>
    params.pollIntervalMs ?? (started.asyncWaitMs === null ? DEFAULT_POLL_INTERVAL_MS : asyncPollDelayMs(attempt));

  const deadline = now() + budgetMs;
  let status: JoeOutputStatus = "pending";
  const remainingRequestMs = (): number =>
    Math.max(1, Math.min(DEFAULT_HTTP_REQUEST_TIMEOUT_MS, deadline - now()));

  // A zero/tiny budget may already be exhausted by the run round-trip. Do not
  // start an output request with a nominal 1ms timeout; return the valid
  // command handle immediately so the caller can resume deterministically.
  if (now() >= deadline) {
    return { commandId, status, output: null, budgetExpired: true, budgetMs };
  }

  // Said once, and only for a command that is still queued: an enqueued command
  // waits for its box to poll, and silence for minutes reads as a hang.
  let announced = false;
  const announce = (): void => {
    if (announced || started.asyncWaitMs === null) return;
    announced = true;
    const estimate = started.firstAnswerEstimateS;
    console.error(
      `Queued on the Joe job channel (command ${commandId}` +
        (estimate ? `, reaches Joe in about ${estimate}s` : "") +
        `); waiting up to ${Math.round(budgetMs / 1000)}s...`
    );
  };

  let transientFailures = 0;

  // Poll the output until terminal or the budget is exhausted.
  for (let attempt = 0; ; attempt++) {
    let output: JoeCommandOutput;
    try {
      output = await getCommandOutput({
        apiKey,
        apiBaseUrl,
        commandId,
        debug,
        timeoutMs: remainingRequestMs(),
      });
    } catch (err) {
      if (err instanceof HttpRequestTimeoutError) {
        return { commandId, status, output: null, budgetExpired: true, budgetMs };
      }
      if (err instanceof HttpStatusError && isRetryableHttpStatus(err.status)) {
        // Transient output failure (5xx proxy hiccup / Joe pod restart, or a
        // 429 rate limit) — the run rpc already returned a VALID command id,
        // so never throw it away: keep polling within the budget, then hand
        // back the resume handle (`pgai joe result <id>`) instead of failing.
        transientFailures += 1;
        if (
          now() >= deadline ||
          (started.asyncWaitMs !== null && transientFailures >= JOE_ASYNC_MAX_POLL_FAILURES)
        ) {
          return { commandId, status, output: null, budgetExpired: true, budgetMs };
        }
        await sleep(pollDelayMs(attempt));
        continue;
      }
      // Terminal (PT400/PT401/PT403/PT404 and other non-retryable errors):
      // abort loudly — polling on cannot succeed.
      throw err;
    }
    transientFailures = 0;
    status = output.status;
    if (status === "ok" || status === "error") {
      return { commandId, status, output, budgetExpired: false, budgetMs };
    }
    if (now() >= deadline) {
      return { commandId, status, output: null, budgetExpired: true, budgetMs };
    }
    announce();
    await sleep(pollDelayMs(attempt));
  }
}

// ---------------------------------------------------------------------------
// High-level orchestrator (the CLI verb surface)
// ---------------------------------------------------------------------------

export interface ExecuteJoeParams {
  apiKey: string;
  apiBaseUrl: string;
  command: JoeCommand;
  /** Raw `--project <id|alias>` value (resolved via `projects_list`). */
  project?: string;
  /**
   * Direct `--instance-id` value — skips project resolution entirely. Kept a
   * string end-to-end so a 64-bit id never rounds through a JS number;
   * PostgREST casts it to the rpc's bigint param. Wins over `project` when both
   * are given.
   */
  instanceId?: number | string;
  input?: JoeVerbInput;
  orgId?: number;
  budgetMs?: number;
  pollIntervalMs?: number;
  debug?: boolean;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

export interface ExecuteJoeOutcome extends RunCommandOutcome {
  command: JoeCommand;
  instanceId: number | string;
  /** The raw text that went on the wire (debugging/tests). */
  commandText: string;
}

/**
 * Build the raw command text from the verb, target the Joe instance (directly
 * via `instanceId`, or by resolving the project id-or-alias), and run the
 * one-shot. The text is built FIRST so a bad verb argument (e.g. a garbage
 * pid) fails before any network call.
 */
export async function executeJoeCommand(params: ExecuteJoeParams): Promise<ExecuteJoeOutcome> {
  const commandText = buildJoeCommandText(params.command, params.input);

  let instanceId: number | string;
  const directRef = String(params.instanceId ?? "").trim();
  if (directRef) {
    if (!/^[0-9]+$/.test(directRef)) {
      throw new Error("instanceId must be a numeric Joe instance id");
    }
    instanceId = directRef;
  } else if (String(params.project ?? "").trim()) {
    instanceId = await resolveJoeInstanceId({
      apiKey: params.apiKey,
      apiBaseUrl: params.apiBaseUrl,
      project: String(params.project),
      orgId: params.orgId,
      debug: params.debug,
    });
  } else {
    throw new Error("either instanceId or project is required");
  }

  const outcome = await runCommand({
    apiKey: params.apiKey,
    apiBaseUrl: params.apiBaseUrl,
    instanceId,
    command: commandText,
    budgetMs: params.budgetMs,
    pollIntervalMs: params.pollIntervalMs,
    debug: params.debug,
    now: params.now,
    sleep: params.sleep,
  });

  return { ...outcome, command: params.command, instanceId, commandText };
}

// ---------------------------------------------------------------------------
// Presentation helpers (pure — unit tested)
// ---------------------------------------------------------------------------

interface PlanNode {
  "Node Type"?: string;
  "Relation Name"?: string;
  Plans?: PlanNode[];
  [key: string]: unknown;
}

/**
 * Lightweight CLIENT-SIDE plan flagging: flag obvious issues (e.g. a Seq Scan)
 * from the returned structured plan_json itself.
 */
export function clientSidePlanFlags(planJson: unknown): string[] {
  const flags: string[] = [];
  const walk = (node: PlanNode | undefined): void => {
    if (!node || typeof node !== "object") {
      return;
    }
    const nodeType = node["Node Type"];
    if (nodeType === "Seq Scan") {
      const rel = node["Relation Name"];
      flags.push(
        `client-side: Seq Scan${rel ? ` on ${rel}` : ""} — no index serves this predicate; consider adding one.`
      );
    }
    if (Array.isArray(node.Plans)) {
      for (const child of node.Plans) {
        walk(child);
      }
    }
  };
  if (Array.isArray(planJson)) {
    // EXPLAIN (format json) returns an array: [{ "Plan": { … } }].
    for (const entry of planJson) {
      if (entry && typeof entry === "object") {
        walk((entry as { Plan?: PlanNode }).Plan ?? (entry as PlanNode));
      }
    }
    return flags;
  }
  if (planJson && typeof planJson === "object") {
    const root = planJson as { Plan?: PlanNode };
    walk(root.Plan ?? (planJson as PlanNode));
  }
  return flags;
}

/**
 * Format a terminal command output as human-readable text (non-JSON mode).
 * The sync contract returns one uniform row for every verb, so this prints
 * whichever sections are present rather than switching per command.
 */
export function formatJoeOutput(output: JoeCommandOutput): string {
  const lines: string[] = [];
  const section = (value: string | null | undefined, label?: string): void => {
    if (value == null || value.trim() === "") return;
    if (lines.length > 0) lines.push("");
    if (label) lines.push(`${label}:`);
    lines.push(value);
  };
  section(output.response);
  section(output.plan_text, "plan");
  const flags = clientSidePlanFlags(output.plan_json).map((flag) => `⚑ ${flag}`);
  if (flags.length > 0) {
    lines.push(...flags);
  }
  section(output.plan_execution_text, "execution plan (EXPLAIN ANALYZE)");
  section(output.stats, "stats");
  section(output.recommendations, "recommendations");
  if (output.queryid) {
    if (lines.length > 0) lines.push("");
    lines.push(`(queryid ${output.queryid})`);
  }
  return lines.join("\n");
}

/** Render `pgai projects` as a fixed-width table. */
export function formatProjectsTable(projects: ProjectListItem[]): string {
  const header = ["PROJECT_ID", "ALIAS", "PROJECT", "JOE", "TUNNEL"];
  const rows = projects.map((p) => [
    String(p.project_id),
    p.alias ?? "-",
    p.name ?? "-",
    p.joe_ready ? "ready" : "no",
    p.tunnel ? "yes" : "no",
  ]);
  const widths = header.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => r[i].length), 0)
  );
  const pad = (cells: string[]): string =>
    cells.map((c, i) => c.padEnd(i === cells.length - 1 ? 0 : widths[i])).join("  ").trimEnd();
  return [pad(header), ...rows.map(pad)].join("\n");
}
