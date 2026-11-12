/** pgai-mon as published on the host (contract: 127.0.0.1:8765 only). */
export const PGAI_MON_URL = "http://127.0.0.1:8765";

export interface PgaiMonProbe {
  ok: boolean;
  /** One line per endpoint, ready to print; never contains the token. */
  lines: string[];
}

const describeError = (err: unknown) =>
  err instanceof Error && err.name === "TimeoutError" ? "no answer" : "unreachable";

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
    }
  } catch (err) {
    ok = false;
    lines.push(`✗ pgai-mon /health: ${describeError(err)} at ${base}`);
  }
  return { ok, lines };
}
