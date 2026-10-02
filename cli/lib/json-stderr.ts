import { format } from "node:util";

/** One JSON event a line on stderr: what `pgai connect` writes there with JSON output. */
export function writeEvent(event: object): void {
  process.stderr.write(`${JSON.stringify(event)}\n`);
}

/**
 * Runs `fn` with console.error and console.warn writing JSON events
 * ({"event":"log","level":...,"message":...}) instead of text, so a check's
 * error or the --debug request log does not break a stream an agent parses.
 */
export async function withJsonStderr<T>(fn: () => Promise<T>): Promise<T> {
  const { error, warn } = console;
  console.error = (...args: unknown[]) => writeEvent({ event: "log", level: "error", message: format(...args) });
  console.warn = (...args: unknown[]) => writeEvent({ event: "log", level: "warn", message: format(...args) });
  try {
    return await fn();
  } finally {
    console.error = error;
    console.warn = warn;
  }
}
