/** pgai-mon as published on the host (contract: 127.0.0.1:8765 only). */
export const PGAI_MON_URL = "http://127.0.0.1:8765";

export interface PgaiMonProbe {
  ok: boolean;
  /** One line per endpoint, ready to print; never contains the token. */
  lines: string[];
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
 * part is absent (an older pgai-mon) or no database is known yet.
 */
function targetLines(collector: unknown): { down: number; lines: string[] } {
  const bySource = (collector as { by_source?: unknown } | null | undefined)?.by_source;
  if (!bySource || typeof bySource !== "object") return { down: 0, lines: [] };
  let total = 0;
  const lines: string[] = [];
  for (const [source, status] of Object.entries(bySource as Record<string, unknown>)) {
    const databases = (status as { databases?: unknown } | null)?.databases;
    if (!databases || typeof databases !== "object") continue;
    for (const [dbname, db] of Object.entries(databases as Record<string, unknown>)) {
      total++;
      const d = db as { up?: unknown; last_error?: unknown } | null;
      if (d?.up === true) continue;
      const why = typeof d?.last_error === "string" && d.last_error ? oneLine(d.last_error) : "no successful connection yet";
      lines.push(`✗ target ${source}/${dbname}: down (${why})`);
    }
  }
  if (total === 0) return { down: 0, lines: [] };
  const down = lines.length;
  return { down, lines: down > 0 ? [`✗ targets: ${total - down}/${total} up`, ...lines] : [`✓ targets: ${total}/${total} up`] };
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
      const targets = targetLines(body?.collector);
      if (targets.down > 0) ok = false;
      lines.push(...targets.lines);
    }
  } catch (err) {
    ok = false;
    lines.push(`✗ pgai-mon /health: ${describeError(err)} at ${base}`);
  }
  return { ok, lines };
}
