/** pgai-mon as published on the host (contract: 127.0.0.1:8765 only). */
export const PGAI_MON_URL = "http://127.0.0.1:8765";

export interface PgaiMonProbe {
  ok: boolean;
  /** One line per endpoint, ready to print; never contains the token. */
  lines: string[];
  /** /health's `collector` part, when /health answered. */
  collector?: unknown;
}

const describeError = (err: unknown) =>
  err instanceof Error && err.name === "TimeoutError" ? "no answer" : "unreachable";

/** One line of a collector error: no line breaks, and no password of a connection URL. */
const oneLine = (text: string) =>
  text.replace(/\s+/g, " ").replace(/(\/\/[^/\s:@]*):[^@\s]*@/g, "$1:***@").trim().slice(0, 300);

/**
 * The monitored databases as /health's `collector` part reports them
 * (`by_source.<source>.databases.<dbname>` with `up` and `last_error`): one
 * line per database that is down, else one "N/N up" line. Nothing when the
 * part is absent (an older pgai-mon) or no database is known yet. A database
 * that is not up and has no `last_error` has not finished its first attempt
 * (pgai-mon sets `last_error` on every failed connect, probe or job): it is
 * "connecting", not down, and does not fail the check.
 */
function targetLines(collector: unknown): { down: number; lines: string[] } {
  const bySource = (collector as { by_source?: unknown } | null | undefined)?.by_source;
  if (!bySource || typeof bySource !== "object") return { down: 0, lines: [] };
  let total = 0;
  const lines: string[] = [];
  const connecting: string[] = [];
  for (const [source, status] of Object.entries(bySource as Record<string, unknown>)) {
    const databases = (status as { databases?: unknown } | null)?.databases;
    if (!databases || typeof databases !== "object") continue;
    for (const [dbname, db] of Object.entries(databases as Record<string, unknown>)) {
      total++;
      const d = db as { up?: unknown; last_error?: unknown } | null;
      if (d?.up === true) continue;
      if (typeof d?.last_error !== "string" || !d.last_error) {
        connecting.push(`- target ${source}/${dbname}: connecting (first connection not finished yet)`);
        continue;
      }
      lines.push(`✗ target ${source}/${dbname}: down (${oneLine(d.last_error)})`);
    }
  }
  if (total === 0) return { down: 0, lines: [] };
  const down = lines.length;
  const up = total - down - connecting.length;
  const more = connecting.length > 0 ? `, ${connecting.length} connecting` : "";
  return {
    down,
    lines: [`${down > 0 ? "✗" : "✓"} targets: ${up}/${total} up${more}`, ...lines, ...connecting],
  };
}

/**
 * Asks pgai-mon itself whether it is alive (GET /livez, no credentials) and
 * whether it serves (GET /health with the bearer token). The container being
 * "running" says neither.
 */
export async function probePgaiMon(
  baseUrl: string,
  token: string | undefined,
  timeoutMs = 3000,
): Promise<PgaiMonProbe> {
  const base = baseUrl.replace(/\/+$/, "");
  const lines: string[] = [];
  let ok = true;
  let collector: unknown;

  try {
    const res = await fetch(`${base}/livez`, { signal: AbortSignal.timeout(timeoutMs) });
    await res.body?.cancel();
    if (res.ok) lines.push("✓ pgai-mon /livez: up");
    else {
      ok = false;
      lines.push(`✗ pgai-mon /livez: HTTP ${res.status}`);
    }
  } catch (err) {
    ok = false;
    lines.push(`✗ pgai-mon /livez: ${describeError(err)} at ${base}`);
  }

  if (!token) {
    return { ok: false, lines: [...lines, "✗ pgai-mon /health: no PGAI_MON_TOKEN in .env"] };
  }
  try {
    const res = await fetch(`${base}/health`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (res.status === 401 || res.status === 403) {
      await res.body?.cancel();
      ok = false;
      lines.push(`✗ pgai-mon /health: token rejected (HTTP ${res.status}); PGAI_MON_TOKEN in .env differs from pgai-mon's`);
    } else if (!res.ok) {
      await res.body?.cancel();
      ok = false;
      lines.push(`✗ pgai-mon /health: HTTP ${res.status}`);
    } else {
      const body = (await res.json().catch(() => null)) as Record<string, unknown> | null;
      const num = (key: string) => (typeof body?.[key] === "number" ? (body[key] as number) : "?");
      lines.push(`✓ pgai-mon /health: ${num("files")} Parquet files, ${num("series_known")} series, ${num("samples_in")} samples in`);
      collector = body?.collector;
      const targets = targetLines(collector);
      if (targets.down > 0) ok = false;
      lines.push(...targets.lines);
    }
  } catch (err) {
    ok = false;
    lines.push(`✗ pgai-mon /health: ${describeError(err)} at ${base}`);
  }
  return { ok, lines, collector };
}

/** Node's name-resolution failures, as pg passes them through. */
export function isNameResolutionError(err: unknown): boolean {
  const e = err as { code?: unknown; message?: unknown } | null;
  if (typeof e?.code === "string" && ["ENOTFOUND", "EAI_AGAIN", "ESERVFAIL"].includes(e.code)) return true;
  return typeof e?.message === "string" && e.message.includes("getaddrinfo");
}

/**
 * What `mon targets test` says when its own connection failed. Only a host
 * name that does not resolve here gets pgai-mon's view: such a name (a compose
 * service name) can resolve inside the Docker network, where pgai-mon
 * connects from. Every other error is reported as before. pgai-mon keys
 * /health by the instances.yml name; the main database has the same label.
 */
export function judgeTargetTestFailure(
  name: string,
  err: unknown,
  collector: unknown,
): { ok: boolean; lines: string[] } {
  const message = err instanceof Error ? err.message : String(err);
  const failed = `✗ Connection failed: ${message}`;
  if (!isNameResolutionError(err)) return { ok: false, lines: [failed] };

  const bySource = (collector as { by_source?: Record<string, unknown> } | null | undefined)?.by_source;
  const databases = (bySource?.[name] as { databases?: Record<string, unknown> } | null | undefined)?.databases;
  const db = (databases?.[name] ?? Object.values(databases ?? {})[0]) as { up?: unknown; last_error?: unknown } | undefined;
  const noDns = `- The host name of target '${name}' does not resolve from this machine (${oneLine(message)})`;
  if (db?.up === true) {
    return { ok: true, lines: [noDns, `✓ pgai-mon, which connects from inside the Docker network, reports '${name}' up`] };
  }
  if (typeof db?.last_error === "string" && db.last_error) {
    return { ok: false, lines: [noDns, `✗ pgai-mon reports '${name}' down: ${oneLine(db.last_error)}`] };
  }
  // Unreachable pgai-mon, a target it has not picked up yet, or its first
  // connection still running: nothing to confirm with.
  return {
    ok: false,
    lines: [failed, `The host name does not resolve from this machine, and pgai-mon could not confirm '${name}'`],
  };
}
