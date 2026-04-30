/**
 * OOM (Out Of Memory) detector.
 *
 * Counts kernel OOM kill events in a recent time window. The canonical
 * signal is the "Out of memory: Killed process N (name)" line emitted by
 * the OOM killer. Each real OOM episode also emits "<task> invoked
 * oom-killer: ..." and often "oom_reaper: reaped process N (name) ...",
 * so naive line-counting double- or triple-counts a single event.
 *
 * Strategy: count distinct PIDs from the canonical "Killed process N"
 * line. Only when that line is missing entirely (e.g. rate-limited by
 * the kernel) do we fall back to counting "oom-killer" invocations.
 *
 * The parser is exposed separately from the journalctl runner so tests
 * can exercise it without root/dmesg access.
 */

const KILLED_PROCESS_RE = /\bOut of memory: Killed process (\d+)\b/g;
const OOM_KILLER_RE = /\boom-killer\b/gi;

/**
 * Counts kernel OOM kill *events* in a journalctl text dump. Returns
 * the number of distinct killed PIDs, or — when the canonical line is
 * absent — the number of "oom-killer" invocation lines.
 */
export function parseOomCount(journalOutput: string): number {
  if (!journalOutput) return 0;
  const pids = new Set<number>();
  for (const m of journalOutput.matchAll(KILLED_PROCESS_RE)) {
    if (m[1]) pids.add(Number(m[1]));
  }
  if (pids.size > 0) return pids.size;
  return journalOutput.match(OOM_KILLER_RE)?.length ?? 0;
}

export type ExecJournalctl = (args: string[]) => Promise<string>;

/**
 * Collects the OOM count over the given lookback window using journalctl.
 * Falls back to 0 on any error so a misconfigured host doesn't break the
 * heartbeat — but logs a warning so the failure is visible.
 */
export async function collectOomCount(opts: {
  lookback: string;
  exec?: ExecJournalctl;
}): Promise<number> {
  const exec = opts.exec ?? defaultExecJournalctl;
  try {
    const out = await exec([
      "-k",
      "--no-pager",
      "--since",
      opts.lookback,
    ]);
    return parseOomCount(out);
  } catch (err) {
    console.warn(
      `[telemetry] oom collector failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return 0;
  }
}

async function defaultExecJournalctl(args: string[]): Promise<string> {
  // Drain stdout AND stderr concurrently so a verbose child can't fill the
  // pipe buffer (~64 KiB on Linux) and deadlock on write. Then surface a
  // non-zero exit so the outer catch returns 0 AND ops sees a warning.
  const proc = Bun.spawn(["journalctl", ...args], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const code = await proc.exited;
  if (code !== 0) {
    throw new Error(`journalctl exited with code ${code}: ${stderr.trim().slice(0, 256)}`);
  }
  return stdout;
}
