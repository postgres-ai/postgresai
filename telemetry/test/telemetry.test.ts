import { describe, test, expect } from "bun:test";
import { runLoop } from "../bin/telemetry";
import type { ReporterConfig } from "../lib/types";

const config: ReporterConfig = {
  apiUrl: "https://platform.example/api/v1",
  apiToken: "tok",
  instanceId: "11111111-1111-1111-1111-111111111111",
  diskPath: "/",
  meminfoPath: "/proc/meminfo",
  oomLookback: "24 hours ago",
  intervalSec: 3600,
};

describe("runLoop", () => {
  test("calls tick once on startup, then once per interval until stopped", async () => {
    let tickCalls = 0;
    let signalHandler: (() => void) | null = null;
    const onSignal = (h: () => void) => {
      signalHandler = h;
    };
    const sleep = async (_ms: number, _signal: AbortSignal) => {
      // Resolve immediately so the loop iterates quickly. Stop after 3 ticks.
      if (tickCalls >= 3) signalHandler?.();
    };
    const tickFn = async () => {
      tickCalls += 1;
    };
    await runLoop(config, { tickFn, sleep, onSignal });
    expect(tickCalls).toBeGreaterThanOrEqual(3);
  });

  test("SIGTERM during sleep aborts immediately — does NOT wait for the full interval", async () => {
    // This is the regression test for the original "graceful shutdown waits
    // up to intervalSec" bug. The fake sleep registers an abort listener
    // and resolves only when the signal fires. The test would hang forever
    // if the loop kept calling tick instead of breaking on the abort.
    let tickCalls = 0;
    let signalHandler: (() => void) | null = null;
    const onSignal = (h: () => void) => {
      signalHandler = h;
    };
    const sleep = (_ms: number, signal: AbortSignal) =>
      new Promise<void>((resolve) => {
        if (signal.aborted) return resolve();
        signal.addEventListener("abort", () => resolve(), { once: true });
        // Trigger SIGTERM mid-sleep, after the abort listener is registered.
        queueMicrotask(() => signalHandler?.());
      });
    const tickFn = async () => {
      tickCalls += 1;
    };

    await runLoop(config, { tickFn, sleep, onSignal });

    // Only the startup tick ran; the sleep was canceled before the next tick.
    expect(tickCalls).toBe(1);
  });

  test("tick errors are caught and the loop continues", async () => {
    let tickCalls = 0;
    let signalHandler: (() => void) | null = null;
    const onSignal = (h: () => void) => {
      signalHandler = h;
    };
    const sleep = async (_ms: number, _signal: AbortSignal) => {
      if (tickCalls >= 2) signalHandler?.();
    };
    const tickFn = async () => {
      tickCalls += 1;
      throw new Error("intentional");
    };

    // Should not reject — tick errors must be caught inside the loop.
    await runLoop(config, { tickFn, sleep, onSignal });

    expect(tickCalls).toBeGreaterThanOrEqual(2);
  });

  test("passes the config through to tick on every call", async () => {
    const seen: ReporterConfig[] = [];
    let signalHandler: (() => void) | null = null;
    const onSignal = (h: () => void) => {
      signalHandler = h;
    };
    const sleep = async () => {
      if (seen.length >= 2) signalHandler?.();
    };
    const tickFn = async (cfg: ReporterConfig) => {
      seen.push(cfg);
    };
    await runLoop(config, { tickFn, sleep, onSignal });
    expect(seen.length).toBeGreaterThanOrEqual(2);
    for (const cfg of seen) expect(cfg).toBe(config);
  });
});
