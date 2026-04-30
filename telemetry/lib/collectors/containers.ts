/**
 * Faulty-container detector.
 *
 * Reads `docker ps -a --format '{{json .}}'` and selects containers whose
 * Status string indicates trouble: "Exited" (any non-zero or otherwise),
 * "Dead", or a parenthesized "(unhealthy)" health marker.
 *
 * Restarting containers are also considered faulty so flapping services
 * surface in the heartbeat.
 *
 * The parser is independent from docker invocation so tests don't need a
 * docker daemon.
 */

interface DockerPsRow {
  Names?: string;
  Status?: string;
  State?: string;
  Health?: string;
}

const FAULTY_STATUS_RE = /^(Exited|Dead|Restarting)\b/;

export function parseFaultyContainers(dockerOutput: string): string[] {
  if (!dockerOutput) return [];
  const out: string[] = [];
  for (const line of dockerOutput.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let row: DockerPsRow;
    try {
      row = JSON.parse(trimmed) as DockerPsRow;
    } catch {
      continue;
    }
    const name = row.Names ?? "";
    if (!name) continue;
    const status = row.Status ?? "";
    const state = row.State ?? "";
    const health = row.Health ?? "";
    const isUnhealthy =
      health.toLowerCase() === "unhealthy" ||
      /\(unhealthy\)/i.test(status);
    const isBadStatus =
      FAULTY_STATUS_RE.test(status) ||
      ["exited", "dead", "restarting"].includes(state.toLowerCase());
    if (isUnhealthy || isBadStatus) out.push(name);
  }
  return Array.from(new Set(out));
}

export type ExecDocker = (args: string[]) => Promise<string>;

export async function collectFaultyContainers(opts: {
  exec?: ExecDocker;
} = {}): Promise<string[]> {
  const exec = opts.exec ?? defaultExecDocker;
  try {
    const out = await exec(["ps", "-a", "--format", "{{json .}}"]);
    return parseFaultyContainers(out);
  } catch (err) {
    // Don't break the heartbeat, but leave a breadcrumb so on-call can tell
    // "docker is unreachable" apart from "all containers are healthy".
    console.warn(
      `[telemetry] containers collector failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return [];
  }
}

async function defaultExecDocker(args: string[]): Promise<string> {
  // Drain stdout AND stderr concurrently so a verbose child can't fill the
  // pipe buffer (~64 KiB on Linux) and deadlock on write. Then surface a
  // non-zero exit so the outer try/catch returns [] AND ops sees a warning.
  const proc = Bun.spawn(["docker", ...args], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const code = await proc.exited;
  if (code !== 0) {
    throw new Error(`docker exited with code ${code}: ${stderr.trim().slice(0, 256)}`);
  }
  return stdout;
}
