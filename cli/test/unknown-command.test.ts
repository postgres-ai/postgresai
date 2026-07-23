import { describe, test, expect } from "bun:test";
import { resolve } from "path";

// Spawn the real CLI (source, via bun) the same way auth.test.ts does, so we
// exercise the actual Commander program wiring end-to-end, including exit codes
// and the stdout/stderr split.
function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin =
    typeof process.execPath === "string" && process.execPath.length > 0
      ? process.execPath
      : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

describe("unknown command handling", () => {
  test("unknown command errors on stderr and exits non-zero", () => {
    const r = runCli(["sdfasdf"]);

    // Must fail, not silently succeed.
    expect(r.status).not.toBe(0);

    // Must clearly name the unknown command (Commander's standard phrasing),
    // not a misleading message about some other command.
    expect(r.stderr).toContain("unknown command");
    expect(r.stderr).toContain("sdfasdf");

    // showHelpAfterError() must reprint the help text to stderr after the
    // error line, so the user sees the available commands without a second
    // invocation. This pins the setting: removing it would fail here.
    expect(r.stderr).toContain("Usage:");

    // The error belongs on stderr, never dumped to stdout as if it succeeded.
    expect(r.stdout).not.toContain("unknown command");
  });

  test("unknown option on a subcommand errors with usage appended", () => {
    // showHelpAfterError() is a program-wide setting inherited by subcommands,
    // so a Commander-level error anywhere in the tree (here: an unknown option
    // on `joe plan`) gets the same error-then-usage treatment on stderr.
    const r = runCli(["joe", "plan", "--bogus-flag"]);

    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain("unknown option");
    expect(r.stderr).toContain("--bogus-flag");
    expect(r.stderr).toContain("Usage:");
  });

  test("bare invocation still shows help and exits 0", () => {
    const r = runCli([]);

    expect(r.status).toBe(0);
    expect(r.stdout).toContain("Usage:");
  });

  test("explicit `help` still shows help and exits 0", () => {
    const r = runCli(["help"]);

    expect(r.status).toBe(0);
    expect(r.stdout).toContain("Usage:");
  });
});
