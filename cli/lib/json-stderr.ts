import { spawn, spawnSync } from "node:child_process";
import { createInterface } from "node:readline";
import { format } from "node:util";

/** One JSON event a line on stderr: what `pgai connect` writes there with JSON output. */
export function writeEvent(event: object): void {
  process.stderr.write(`${JSON.stringify(event)}\n`);
}

/**
 * Makes console.error and console.warn write JSON events
 * ({"event":"log","level":...,"message":...}) instead of text, so a check's
 * error or the --debug request log does not break a stream an agent parses.
 * The level is the console method's, unless the line names its own: `debug`
 * for "Debug: ..." (the --debug log), `warn` for "Warning: ...".
 * Returns the function that puts them back.
 */
export function jsonConsole(): () => void {
  const { error, warn } = console;
  const log = (level: string) => (...args: unknown[]) => {
    const message = format(...args);
    writeEvent({ event: "log", level: /^\s*Debug:/.test(message) ? "debug" : /^\s*Warning:/.test(message) ? "warn" : level, message });
  };
  console.error = log("error");
  console.warn = log("warn");
  return () => {
    console.error = error;
    console.warn = warn;
  };
}

// The logins mon local-install prints at its end: "user / password". Either may
// hold a space or " / ", so all of it after the label is masked.
const LOGIN = /^(\s*(?:Login|VictoriaMetrics Auth): ).+$/;

/**
 * Runs a child with stdin closed and resolves to its exit code (null when it
 * cannot start or is killed by a signal). Without JSON output, what it prints
 * goes to stderr as it is. With JSON output, each line is a log event naming
 * `source`: `info` from its stdout, `error` from its stderr; blank lines are dropped.
 * A log collector keeps those events: a login (user and password) is masked,
 * the line names the command that shows it.
 */
export function runChild(command: string, args: string[], env: NodeJS.ProcessEnv, json: boolean, source: string): Promise<number | null> {
  if (!json) return Promise.resolve(spawnSync(command, args, { stdio: ["ignore", 2, 2], env }).status);
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"], env });
    for (const [stream, level] of [[child.stdout!, "info"], [child.stderr!, "error"]] as const) {
      createInterface({ input: stream }).on("line", (line) => {
        const message = line.replace(LOGIN, "$1***** (pgai mon show-grafana-credentials)");
        if (line.trim() !== "") writeEvent({ event: "log", level, source, message });
      });
    }
    child.on("error", (err) => {
      writeEvent({ event: "log", level: "error", source, message: err.message });
      resolve(null);
    });
    child.on("close", (code) => resolve(code));
  });
}
