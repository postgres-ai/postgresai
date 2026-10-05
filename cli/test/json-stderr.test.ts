import { expect, test } from "bun:test";
import { jsonConsole, runChild } from "../lib/json-stderr";

/** What `fn` writes to stderr, a line each. */
async function stderrOf(fn: () => unknown): Promise<string[]> {
  const chunks: string[] = [];
  const write = process.stderr.write;
  process.stderr.write = ((chunk: string) => { chunks.push(String(chunk)); return true; }) as typeof process.stderr.write;
  try {
    await fn();
  } finally {
    process.stderr.write = write;
  }
  return chunks.join("").split("\n").filter(Boolean);
}

// What generateAllReports does when a check fails ([D004] Error querying
// pg_stat_statements: ...) must reach an agent as a JSON event, not as text.
test("after jsonConsole, console.error and console.warn are one JSON event a line", async () => {
  const lines = await stderrOf(() => {
    const restore = jsonConsole();
    try {
      console.error("[D004] Error querying pg_stat_statements: %s", "must be loaded via \"shared_preload_libraries\"");
      console.warn("two\nlines");
    } finally {
      restore();
    }
  });
  expect(lines.map((l) => JSON.parse(l))).toEqual([
    { event: "log", level: "error", message: "[D004] Error querying pg_stat_statements: must be loaded via \"shared_preload_libraries\"" },
    { event: "log", level: "warn", message: "two\nlines" },
  ]);
});

// The --debug request log and a config warning go through console.error: they
// are not errors an agent should count.
test("after jsonConsole, a line that names its level keeps it: Debug is debug, Warning is warn", async () => {
  const lines = await stderrOf(() => {
    const restore = jsonConsole();
    try {
      console.error("Debug: POST URL: http://127.0.0.1/rpc/cloud_monitoring_list");
      console.error("\nDebug: Response status: 200");
      console.error("Warning: Failed to read config from /x/config.json: bad JSON");
      console.warn("Debug: from warn");
      // The express checkup's warnings name where they come from first.
      console.error("[F001] Warning: largest-tables cross-reference failed: timeout");
      console.error("[F001] Error: Warning: not a warning");
    } finally {
      restore();
    }
  });
  expect(lines.map((l) => JSON.parse(l).level)).toEqual(["debug", "debug", "warn", "debug", "warn", "error"]);
});

test("the restore jsonConsole returns puts console.error and console.warn back", () => {
  const { error, warn } = console;
  jsonConsole()();
  expect(console.error).toBe(error);
  expect(console.warn).toBe(warn);
});

test("runChild with JSON output: stdout lines are info, stderr lines error, blank lines dropped", async () => {
  let status: number | null = -1;
  const script = "console.log('step 1'); console.log(''); console.error('✗ Connection failed'); process.exit(3)";
  const lines = await stderrOf(async () => { status = await runChild(process.execPath, ["-e", script], process.env, true, "child"); });
  expect(status).toBe(3);
  // Two pipes: the order between them is not kept.
  const events = lines.map((l) => JSON.parse(l));
  expect(events).toHaveLength(2);
  expect(events).toContainEqual({ event: "log", level: "info", source: "child", message: "step 1" });
  expect(events).toContainEqual({ event: "log", level: "error", source: "child", message: "✗ Connection failed" });
});

// The --self-hosted `mon local-install` wrapper: a child that cannot start is a
// failure with its reason, not a hang or a crash.
test("runChild with JSON output: a child that cannot start resolves null and says why", async () => {
  let status: number | null = -1;
  const lines = await stderrOf(async () => { status = await runChild("/nonexistent/pgai-child", [], process.env, true, "child"); });
  expect(status).toBeNull();
  expect(lines.map((l) => JSON.parse(l))).toEqual([
    { event: "log", level: "error", source: "child", message: expect.stringContaining("/nonexistent/pgai-child") },
  ]);
});

// mon local-install prints its logins at its end. A log event is kept by log
// collectors: it names the command that shows them instead of the password.
// VM_AUTH_USERNAME comes from the project's .env: it may hold a space or " / ".
test("runChild with JSON output: the logins mon local-install prints carry no password", async () => {
  const script = [
    "console.log('   Login: monitor / gr4fana-pw')",
    "console.log('   VictoriaMetrics Auth: vmauth / vm-pw')",
    "console.log('   VictoriaMetrics Auth: vm admin / s3cret pw')",
    "console.log('   VictoriaMetrics Auth: vm / admin / s3cret / pw')",
    "console.log('   Grafana Dashboard: http://localhost:3000')",
  ].join("; ");
  const lines = await stderrOf(() => runChild(process.execPath, ["-e", script], process.env, true, "child"));
  const messages = lines.map((l) => JSON.parse(l).message as string);
  expect(messages).toEqual([
    "   Login: ***** (pgai mon show-grafana-credentials)",
    "   VictoriaMetrics Auth: ***** (pgai mon show-grafana-credentials)",
    "   VictoriaMetrics Auth: ***** (pgai mon show-grafana-credentials)",
    "   VictoriaMetrics Auth: ***** (pgai mon show-grafana-credentials)",
    "   Grafana Dashboard: http://localhost:3000",
  ]);
  for (const secret of ["gr4fana-pw", "vm-pw", "s3cret"]) expect(messages.join("\n")).not.toContain(secret);
});

// A person at a terminal (or connect without JSON output): the child writes to
// stderr as it is; its exit code is what connect checks.
test("runChild without JSON output resolves the child's exit code", async () => {
  expect(await runChild(process.execPath, ["-e", "process.exit(3)"], process.env, false, "child")).toBe(3);
  expect(await runChild(process.execPath, ["-e", ""], process.env, false, "child")).toBe(0);
});
