/**
 * `pgai promql` — run a PromQL query on a monitoring instance's own metric
 * store, over the instance job channel.
 *
 * The platform never connects to the instance: the query is enqueued as a job,
 * the box polls for it, runs it against its local store and submits the result.
 * So the first answer cannot be faster than the box's poll interval, and this
 * polls rather than blocking a connection open.
 *
 * Issue: https://gitlab.com/postgres-ai/postgresai/-/issues/378
 * Platform half: https://gitlab.com/postgres-ai/platform-all/-/merge_requests/809
 * (v1.instance_query_enqueue / v1.instance_query_result).
 */

import { resolveBaseUrls } from "./util";
import { orgScopeHeaders, getActiveOrgScope } from "./org-scope";

export type PromQLKind = "promql_instant" | "promql_range";

export type PromQLSample = [number, string];

export interface PromQLSeries {
  metric: Record<string, string>;
  value?: PromQLSample;
  values?: PromQLSample[];
}

export interface PromQLPayload {
  resultType: "vector" | "matrix" | string;
  result: PromQLSeries[];
  stats: { truncated: boolean };
}

export interface PromQLJobResult {
  status: string;
  outcome: string | null;
  result: PromQLPayload | null;
  error: string | null;
  failure_class: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface EnqueueArgs {
  instanceId: string;
  kind: PromQLKind;
  query: string;
  at?: string;
  start?: string;
  end?: string;
  stepS?: number;
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

/** Enqueue one query. */
export async function enqueueQuery(
  apiKey: string,
  args: EnqueueArgs,
  opts?: { apiBaseUrl?: string },
): Promise<EnqueuedQuery> {
  // Only the keys this kind uses are sent: PostgREST resolves an RPC by the
  // exact set of argument names in the body, so a stray null would not match.
  const body: Record<string, unknown> = {
    p_instance_id: args.instanceId,
    p_kind: args.kind,
    p_query: args.query,
  };
  if (args.kind === "promql_range") {
    body.p_start = args.start;
    body.p_end = args.end;
    body.p_step_s = args.stepS;
  } else if (args.at) {
    body.p_at = args.at;
  }

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
): Promise<PromQLJobResult> {
  const reply = (await callRpc("instance_query_result", apiKey, { p_job_id: jobId }, opts)) as
    | PromQLJobResult
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
 * Poll delays, in ms.
 *
 * The platform half fixes 1s then 2s (platform-all!809). That is right for the first few — a rig at a
 * 5s server interval answers inside the first three — but flat 2s against the
 * 600000 ms DEFAULT pacing is ~300 requests for one answer, each a full
 * api_token_check doing a bcrypt per candidate token in the org. Doubling to a
 * 15s ceiling keeps the early polls fast and makes the long wait ~45.
 */
function pollDelay(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 15000);
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
  },
): Promise<PromQLJobResult> {
  const now = opts.now ?? (() => Date.now());
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const deadline = now() + opts.timeoutMs;

  for (let attempt = 0; ; attempt++) {
    const res = await readQueryResult(apiKey, jobId, opts);
    if (isTerminal(res.status)) return res;

    const delay = pollDelay(attempt);
    if (now() + delay >= deadline) return res;
    await sleep(delay);
  }
}

/** The label set as Prometheus prints it: `name{a="1", b="2"}`. */
export function formatMetric(metric: Record<string, string>): string {
  const name = metric.__name__ ?? "";
  const labels = Object.entries(metric)
    .filter(([k]) => k !== "__name__")
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}="${v}"`);
  if (!labels.length) return name || "{}";
  return `${name}{${labels.join(", ")}}`;
}

function pad(s: string, width: number): string {
  return s.length >= width ? s : s + " ".repeat(width - s.length);
}

/**
 * Render a payload as a table.
 *
 * A matrix prints the first and last sample per series rather than every point:
 * a day at one-minute step is 1440 rows per series and would bury the answer.
 * `--json` is there for the whole thing, and the summary line says so.
 */
export function renderPromQL(payload: PromQLPayload): string {
  const lines: string[] = [];

  if (!payload.result.length) {
    lines.push("Empty result set.");
  } else if (payload.resultType === "vector") {
    const rows = payload.result.map((s) => [formatMetric(s.metric), s.value ? s.value[1] : ""]);
    const w = Math.max(6, ...rows.map((r) => r[0].length));
    lines.push(`${pad("SERIES", w)}  VALUE`);
    for (const [m, v] of rows) lines.push(`${pad(m, w)}  ${v}`);
  } else {
    const rows = payload.result.map((s) => {
      const vs = s.values ?? [];
      const first = vs.length ? vs[0][1] : "";
      const last = vs.length ? vs[vs.length - 1][1] : "";
      return [formatMetric(s.metric), String(vs.length), first, last];
    });
    const w = Math.max(6, ...rows.map((r) => r[0].length));
    lines.push(`${pad("SERIES", w)}  POINTS  FIRST  LAST`);
    for (const [m, n, f, l] of rows) lines.push(`${pad(m, w)}  ${pad(n, 6)}  ${f}  ${l}`);
    if (rows.length) lines.push("(first and last sample per series; use --json for every point)");
  }

  // Counted here rather than transmitted: both are derivable from the payload,
  // and sending them was the only reason the box had to maintain running totals.
  const series = payload.result.length;
  const points = payload.result.reduce(
    (n, s) => n + (s.values?.length ?? 0) + (s.value ? 1 : 0),
    0,
  );
  lines.push(`\n${series} series, ${points} points`);
  if (payload.stats.truncated) {
    // Never let a partial answer look complete, and say what to do about it:
    // with no point cap, truncation is a normal outcome on a wide range query.
    lines.push(
      "⚠ TRUNCATED: too large to return whole. Raise --step, narrow the time range, " +
        "or select fewer series.",
    );
  }
  return lines.join("\n");
}
