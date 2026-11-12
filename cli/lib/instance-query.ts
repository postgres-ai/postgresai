/**
 * `pgai mon sql` — run a read-only SQL query on a monitoring instance's own metric
 * store (pgai-mon), over the instance job channel.
 *
 * The platform never connects to the instance: the query is enqueued as a job,
 * the box polls for it, runs it against its local pgai-mon (`POST /sql`) and
 * submits the result. So the first answer cannot be faster than the box's poll
 * interval, and this polls rather than blocking a connection open.
 *
 * Platform half: v1.instance_query_enqueue / v1.instance_query_result, with
 * the job kind `sql`.
 */

import { asyncPollDelayMs, resolveBaseUrls, SLOW_ANSWER_MS } from "./util";
import { orgScopeHeaders, getActiveOrgScope } from "./org-scope";
import { listProjects, isNumericProjectRef } from "./joe";

/** The job kind for a SQL query; the box runs it through pgai-mon's `POST /sql`. */
export const SQL_JOB_KIND = "sql";

/**
 * What the box submits for a `sql` job: pgai-mon's `/sql` reply in its row
 * format, rows cut to the job channel's byte budget and `truncated` set when
 * they were.
 */
export interface SqlPayload {
  columns: string[];
  types?: string[];
  rows: unknown[][];
  row_count?: number;
  truncated: boolean;
  elapsed_ms?: number;
}

export interface QueryJobResult {
  status: string;
  outcome: string | null;
  result: SqlPayload | null;
  error: string | null;
  failure_class: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface EnqueueArgs {
  instanceId: string;
  query: string;
}

/** A PostgREST error body, which carries the real reason in `message`/`details`. */
function describeRpcError(status: number, body: string): string {
  try {
    const parsed = JSON.parse(body) as { message?: string; details?: string; hint?: string };
    const parts = [parsed.message, parsed.details, parsed.hint].filter(Boolean);
    if (parts.length) return parts.join(" — ");
  } catch {
    // not JSON; fall through to the raw body
  }
  return body.trim() || `HTTP ${status}`;
}

async function callRpc(
  name: string,
  apiKey: string,
  body: Record<string, unknown>,
  opts?: { apiBaseUrl?: string; timeoutMs?: number },
): Promise<unknown> {
  const { apiBaseUrl } = resolveBaseUrls(opts);
  const res = await fetch(`${apiBaseUrl}/rpc/${name}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // The credential goes in the header, never the body: a body parameter is
      // a bind parameter, and Postgres writes those into the server log for any
      // statement over log_min_duration_statement.
      "access-token": apiKey,
      ...orgScopeHeaders(getActiveOrgScope()),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(opts?.timeoutMs ?? 30_000),
  });

  const text = await res.text().catch(() => "");
  if (!res.ok) {
    throw new Error(describeRpcError(res.status, text));
  }
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`${name} returned a body that is not JSON`);
  }
}

export interface EnqueuedQuery {
  jobId: string;
  /**
   * The instance's poll pacing in whole seconds, as the platform computes it:
   * `round(idle_ms / 1000.0)`.
   *
   * It is the MEAN gap, NOT a deadline — `jittered()` spreads the real gap
   * ±20%, so a 600 estimate means 480–720 s and an answer can legitimately
   * arrive after it. Say "roughly", never a flat figure, and size any wait
   * comfortably past it rather than to it.
   *
   * ABSENT ON THE WIRE; normalised to null here. The RPC cannot emit a null —
   * it builds the field from a non-null int — so do not add a branch for one.
   * What it can do is omit the field: any platform older than the one that
   * added it returns `{job_id}` alone, and the CLI upgrades independently, so
   * absent is the normal state during a rollout. Absent must mean "fall back
   * to the client default", never "estimate zero".
   */
  firstAnswerEstimateS: number | null;
}

export interface ResolveMonitoringInstanceParams {
  apiKey: string;
  apiBaseUrl: string;
  project: string;
  orgId?: number;
}

/**
 * Resolve `--project <id|alias>` to the project's monitoring instance uuid,
 * via `v1.projects_list` like the joe and dblab resolvers. A numeric ref
 * matches the project id; anything else the alias or name, case-insensitive.
 */
export async function resolveMonitoringInstanceId(params: ResolveMonitoringInstanceParams): Promise<string> {
  const ref = String(params.project ?? "").trim();
  if (!ref) {
    throw new Error("project is required (--project <id|alias>)");
  }
  const projects = await listProjects({
    apiKey: params.apiKey,
    apiBaseUrl: params.apiBaseUrl,
    orgId: params.orgId,
  });
  const needle = ref.toLowerCase();
  // BigInt, not the raw text: `--project 007` means project 7.
  const matches = isNumericProjectRef(ref)
    ? projects.filter((p) => String(p.project_id) === BigInt(ref).toString())
    : projects.filter(
        (p) =>
          (p.alias !== null && p.alias.toLowerCase() === needle) ||
          (p.name !== null && p.name.toLowerCase() === needle),
      );
  if (matches.length === 0) {
    throw new Error(`Project not found for id/alias/name '${ref}'. Run 'pgai projects' to see available projects.`);
  }
  // Names are not unique, and one project's name can be another's alias.
  if (matches.length > 1) {
    throw new Error(
      `'${ref}' matches ${matches.length} projects (ids ${matches.map((p) => p.project_id).join(", ")}). ` +
        "Pass --project <id> instead.",
    );
  }
  const match = matches[0];
  const ids = match.monitoring_instance_ids;
  if (ids === undefined) {
    throw new Error(
      "This platform does not report monitoring instances per project yet. Pass --instance <uuid> instead.",
    );
  }
  if (ids.length === 0) {
    throw new Error(`Project '${ref}' has no active monitoring instance.`);
  }
  // Nothing on the platform enforces one per project; guessing would query a
  // box the user did not mean.
  if (ids.length > 1) {
    throw new Error(
      `Project '${ref}' has ${ids.length} active monitoring instances: ${ids.join(", ")}. Pass --instance <uuid>.`,
    );
  }
  return ids[0];
}

/** Enqueue one SQL query. */
export async function enqueueQuery(
  apiKey: string,
  args: EnqueueArgs,
  opts?: { apiBaseUrl?: string },
): Promise<EnqueuedQuery> {
  // Only the keys a `sql` job uses are sent: PostgREST resolves an RPC by the
  // exact set of argument names in the body, so a stray null would not match.
  const body: Record<string, unknown> = {
    p_instance_id: args.instanceId,
    p_kind: SQL_JOB_KIND,
    p_query: args.query,
  };

  const reply = (await callRpc("instance_query_enqueue", apiKey, body, opts)) as
    | { job_id?: string; first_answer_estimate_s?: unknown }
    | null;
  const jobId = reply?.job_id;
  if (!jobId) {
    throw new Error("instance_query_enqueue returned no job_id");
  }
  return { jobId, firstAnswerEstimateS: positiveSeconds(reply?.first_answer_estimate_s) };
}

/**
 * A usable positive number of seconds, or null for anything else.
 *
 * One predicate, not a list of cases: it costs nothing to cover shapes the
 * current platform cannot emit, and it is not really guarding against our
 * platform at all. It guards the `as { first_answer_estimate_s?: unknown }`
 * above — an unchecked assertion about a JSON body off a network. A type guard
 * at a deserialisation boundary is right regardless of what today's server
 * happens to send, so do not prune it by counting producible cases.
 */
function positiveSeconds(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
}

export async function readQueryResult(
  apiKey: string,
  jobId: string,
  opts?: { apiBaseUrl?: string },
): Promise<QueryJobResult> {
  const reply = (await callRpc("instance_query_result", apiKey, { p_job_id: jobId }, opts)) as
    | QueryJobResult
    | null;
  if (!reply || typeof reply.status !== "string") {
    throw new Error("instance_query_result returned an unexpected reply");
  }
  return reply;
}

const TERMINAL = new Set(["done", "failed", "expired"]);

/** Whether a job has stopped moving. The caller branches on this. */
export function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}


/**
 * Poll until the job stops moving or the deadline passes.
 *
 * Always returns the LAST result read, terminal or not. The caller asks
 * `isTerminal(res.status)`; on a timeout `queued` and `running` are different
 * situations with different remedies, and the caller also gets `error` and
 * `failure_class` without a second path to fetch them.
 *
 * The first answer can never beat the box's own poll interval — the box only
 * learns a new pacing on its NEXT poll, so no client-side eagerness shortens it.
 */
export async function awaitQueryResult(
  apiKey: string,
  jobId: string,
  opts: {
    apiBaseUrl?: string;
    timeoutMs: number;
    sleep?: (ms: number) => Promise<void>;
    // Injectable so a test can drive the deadline deterministically instead of
    // depending on real elapsed time.
    now?: () => number;
    // Called once, when there is still no answer after SLOW_ANSWER_MS -- so a
    // caller says why it is waiting only when there is a wait to explain.
    onSlow?: () => void;
  },
): Promise<QueryJobResult> {
  const now = opts.now ?? (() => Date.now());
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const started = now();
  const deadline = started + opts.timeoutMs;
  let announced = false;

  for (let attempt = 0; ; attempt++) {
    const res = await readQueryResult(apiKey, jobId, opts);
    if (isTerminal(res.status)) return res;
    if (!announced && opts.onSlow && now() - started >= SLOW_ANSWER_MS) {
      announced = true;
      opts.onSlow();
    }

    const delay = asyncPollDelayMs(attempt);
    if (now() + delay >= deadline) return res;
    await sleep(delay);
  }
}

function pad(s: string, width: number): string {
  return s.length >= width ? s : s + " ".repeat(width - s.length);
}

/** One cell as text: SQL NULL as "NULL", objects and arrays as JSON. */
function cellText(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/**
 * Render a `sql` job's result as a table, with the row count under it.
 * `--json` prints the payload as the box sent it.
 */
export function renderSql(payload: SqlPayload): string {
  const lines: string[] = [];
  const rows = payload.rows.map((r) => r.map(cellText));
  if (!rows.length) {
    lines.push(payload.columns.length ? payload.columns.join("  ") : "", "Empty result set.");
  } else {
    const widths = payload.columns.map((c, i) => Math.max(c.length, ...rows.map((r) => (r[i] ?? "").length)));
    lines.push(payload.columns.map((c, i) => pad(c, widths[i])).join("  ").trimEnd());
    for (const r of rows) lines.push(r.map((v, i) => pad(v, widths[i] ?? 0)).join("  ").trimEnd());
  }
  lines.push(`\n${rows.length} ${rows.length === 1 ? "row" : "rows"}`);
  if (payload.truncated) {
    // Never let a partial answer look complete, and say what to do about it.
    lines.push("⚠ TRUNCATED: too large to return whole. Select fewer columns or rows (LIMIT, a narrower time range).");
  }
  return lines.filter((l, i) => !(i === 0 && l === "")).join("\n");
}

/**
 * `pgai mon sql --local`: one query straight to this box's pgai-mon
 * (`POST /sql`, row format), with the bearer token from the install's `.env`.
 * The token goes in the header only; error messages never contain it.
 */
export async function runLocalSql(
  baseUrl: string,
  token: string,
  query: string,
  timeoutMs: number,
): Promise<SqlPayload> {
  const url = `${baseUrl.replace(/\/+$/, "")}/sql`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "text/plain; charset=utf-8", Accept: "application/json" },
      body: query,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    const why = err instanceof Error && err.name === "TimeoutError" ? `no answer within ${Math.round(timeoutMs / 1000)}s` : "unreachable";
    throw new Error(`pgai-mon at ${baseUrl}: ${why}. Is the stack running? ('postgresai mon health')`);
  }
  const text = await res.text();
  if (res.status === 401 || res.status === 403) {
    throw new Error(`pgai-mon rejected the token (HTTP ${res.status}): PGAI_MON_TOKEN in .env differs from pgai-mon's`);
  }
  if (!res.ok) {
    let message = "";
    try {
      const body = JSON.parse(text) as { error?: { type?: string; message?: string } };
      message = [body.error?.type, body.error?.message].filter(Boolean).join(": ");
    } catch {
      // not JSON: report the status only
    }
    throw new Error(`Query failed (HTTP ${res.status})${message ? `: ${message}` : ""}`);
  }
  let payload: SqlPayload;
  try {
    payload = JSON.parse(text) as SqlPayload;
  } catch {
    throw new Error("pgai-mon's answer is not JSON");
  }
  if (!Array.isArray(payload?.columns) || !Array.isArray(payload?.rows)) {
    throw new Error("pgai-mon's answer is not a result set");
  }
  payload.truncated = !!payload.truncated;
  return payload;
}
