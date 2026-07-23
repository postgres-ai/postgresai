import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

import {
  FEEDBACK_URL,
  FEEDBACK_SUPPRESS_ENV,
  FEEDBACK_TIP_MIN_INTERVAL_MS,
  feedbackTipSuppressed,
  feedbackTipAllowed,
  isFeedbackTipDue,
  feedbackTipLine,
  feedbackJson,
  feedbackMessage,
  maybeEmitFeedbackTip,
} from "../lib/feedback";

// Run the real CLI as a subprocess. Bun.spawnSync gives it piped (non-TTY)
// stdio, which is exactly the environment agents/CI/scripts see — so these runs
// double as proof the occasional tip never leaks into non-TTY output.
function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin =
    typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

/** Isolate config so subprocess runs never touch real user state. */
function isolatedEnv(extra: Record<string, string> = {}) {
  const cfgHome = mkdtempSync(resolve(tmpdir(), "feedback-cli-test-"));
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome, ...extra };
}

const NOW = 1_700_000_000_000;

describe("feedbackTipSuppressed", () => {
  test("undefined / empty / falsey values do not suppress", () => {
    for (const value of [undefined, "", "0", "false", "FALSE", "no", "NO", " "]) {
      expect(feedbackTipSuppressed(value)).toBe(false);
    }
  });

  test("any other non-empty value suppresses", () => {
    for (const value of ["1", "true", "yes", "on", "please-stop"]) {
      expect(feedbackTipSuppressed(value)).toBe(true);
    }
  });
});

describe("feedbackTipAllowed", () => {
  test("allowed only on an interactive TTY without --json and without suppression", () => {
    expect(feedbackTipAllowed({ isTTY: true, json: false, suppress: undefined })).toBe(true);
  });

  test("never allowed with --json (machine-readable output stays byte-clean)", () => {
    expect(feedbackTipAllowed({ isTTY: true, json: true, suppress: undefined })).toBe(false);
  });

  test("never allowed on a non-TTY (agents / CI / pipes)", () => {
    expect(feedbackTipAllowed({ isTTY: false, json: false, suppress: undefined })).toBe(false);
  });

  test("never allowed when suppressed via env", () => {
    expect(feedbackTipAllowed({ isTTY: true, json: false, suppress: "1" })).toBe(false);
  });
});

describe("isFeedbackTipDue", () => {
  test("due when never shown", () => {
    expect(isFeedbackTipDue(null, NOW)).toBe(true);
    expect(isFeedbackTipDue(undefined, NOW)).toBe(true);
    expect(isFeedbackTipDue(Number.NaN, NOW)).toBe(true);
  });

  test("not due within the min interval, due after it", () => {
    expect(isFeedbackTipDue(NOW - 1000, NOW)).toBe(false);
    expect(isFeedbackTipDue(NOW - FEEDBACK_TIP_MIN_INTERVAL_MS + 1, NOW)).toBe(false);
    expect(isFeedbackTipDue(NOW - FEEDBACK_TIP_MIN_INTERVAL_MS, NOW)).toBe(true);
    expect(isFeedbackTipDue(NOW - FEEDBACK_TIP_MIN_INTERVAL_MS * 2, NOW)).toBe(true);
  });
});

describe("feedbackTipLine", () => {
  test("is a single dim line containing the URL and the silence hint", () => {
    const line = feedbackTipLine();
    expect(line).toContain(FEEDBACK_URL);
    expect(line).toContain(FEEDBACK_SUPPRESS_ENV);
    expect(line).toStartWith("\x1b[2m"); // dim
    expect(line).toEndWith("\x1b[0m"); // reset
    expect(line.includes("\n")).toBe(false); // one line
  });
});

describe("maybeEmitFeedbackTip", () => {
  function spyIO(overrides: Partial<Parameters<typeof maybeEmitFeedbackTip>[1]> = {}) {
    const writes: string[] = [];
    const persists: number[] = [];
    const io = {
      write: (line: string) => writes.push(line),
      readLastShown: () => null,
      persistLastShown: (n: number) => persists.push(n),
      now: () => NOW,
      ...overrides,
    };
    return { io, writes, persists };
  }

  test("emits, persists the timestamp, and writes exactly the tip line when allowed and due", () => {
    const { io, writes, persists } = spyIO();
    const shown = maybeEmitFeedbackTip({ isTTY: true, json: false, suppress: undefined }, io);
    expect(shown).toBe(true);
    expect(writes).toEqual([feedbackTipLine() + "\n"]);
    expect(persists).toEqual([NOW]);
  });

  test("suppressed under --json: no write, and no config mutation", () => {
    const { io, writes, persists } = spyIO();
    const shown = maybeEmitFeedbackTip({ isTTY: true, json: true, suppress: undefined }, io);
    expect(shown).toBe(false);
    expect(writes).toEqual([]);
    expect(persists).toEqual([]); // agents/CI must never even touch config
  });

  test("suppressed on non-TTY: no write, and no config mutation", () => {
    const { io, writes, persists } = spyIO();
    const shown = maybeEmitFeedbackTip({ isTTY: false, json: false, suppress: undefined }, io);
    expect(shown).toBe(false);
    expect(writes).toEqual([]);
    expect(persists).toEqual([]);
  });

  test("suppressed via env var: no write, and no config mutation", () => {
    const { io, writes, persists } = spyIO();
    const shown = maybeEmitFeedbackTip({ isTTY: true, json: false, suppress: "1" }, io);
    expect(shown).toBe(false);
    expect(writes).toEqual([]);
    expect(persists).toEqual([]);
  });

  test("does not re-emit within the min interval (weekly gate)", () => {
    const { io, writes } = spyIO({ readLastShown: () => NOW - 1000 });
    const shown = maybeEmitFeedbackTip({ isTTY: true, json: false, suppress: undefined }, io);
    expect(shown).toBe(false);
    expect(writes).toEqual([]);
  });

  test("still shows the tip even if persisting the timestamp fails (best-effort)", () => {
    const { io, writes } = spyIO({
      persistLastShown: () => {
        throw new Error("read-only config");
      },
    });
    const shown = maybeEmitFeedbackTip({ isTTY: true, json: false, suppress: undefined }, io);
    expect(shown).toBe(true);
    expect(writes).toEqual([feedbackTipLine() + "\n"]);
  });
});

describe("feedback command output helpers", () => {
  test("feedbackMessage points at the feedback URL and the silence hint", () => {
    const msg = feedbackMessage();
    expect(msg).toContain(FEEDBACK_URL);
    expect(msg).toContain(FEEDBACK_SUPPRESS_ENV);
  });

  test("feedbackJson is valid JSON exposing the URL", () => {
    expect(JSON.parse(feedbackJson())).toEqual({ url: FEEDBACK_URL });
  });
});

describe("pgai feedback (end-to-end)", () => {
  test("`pgai feedback` prints the feedback URL", () => {
    const { status, stdout } = runCli(["feedback"], isolatedEnv());
    expect(status).toBe(0);
    expect(stdout).toContain(FEEDBACK_URL);
  });

  test("`pgai feedback --json` is byte-clean JSON with no tip anywhere", () => {
    const { status, stdout, stderr } = runCli(["feedback", "--json"], isolatedEnv());
    expect(status).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ url: FEEDBACK_URL });
    // The occasional tip must never contaminate --json output on stdout or stderr.
    expect(stdout).not.toContain("💡");
    expect(stderr).not.toContain("💡");
    expect(stderr).not.toContain("Ideas / feedback");
  });

  test("top-level `--help` shows the feedback footer", () => {
    const { status, stdout } = runCli(["--help"], isolatedEnv());
    expect(status).toBe(0);
    expect(stdout).toContain("Ideas / feedback");
    expect(stdout).toContain(FEEDBACK_URL);
  });

  test("a normal command on a non-TTY (piped) run never prints the tip", () => {
    // set-default-project runs fully offline (just writes the isolated config).
    // Because stdio is piped here, the postAction tip hook must stay silent.
    const { status, stdout, stderr } = runCli(["set-default-project", "demo"], isolatedEnv());
    expect(status).toBe(0);
    expect(stdout).toContain("Default project saved");
    expect(stderr).not.toContain("Ideas / feedback");
    expect(stderr).not.toContain("💡");
  });
});
