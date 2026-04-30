#!/usr/bin/env bun
/**
 * Entry point for the PostgresAI monitoring instance telemetry agent.
 *
 * Runs forever, reporting once on startup and then every intervalSec.
 * SIGTERM / SIGINT cancel the in-flight sleep and exit cleanly, so
 * shutdown latency is bounded by the current tick (not by intervalSec).
 */

import { loadConfigFromEnv, ConfigError } from "../lib/config";
import { collectSnapshot } from "../lib/collect";
import { postReport } from "../lib/reporter";
import type { ReporterConfig } from "../lib/types";

const RESPONSE_LOG_CAP = 512;

export async function tick(config: ReporterConfig): Promise<void> {
  const snapshot = await collectSnapshot(config);
  const result = await postReport(config, snapshot);
  if (!result.ok) {
    const raw = JSON.stringify(result.body ?? null);
    const capped = raw.length > RESPONSE_LOG_CAP ? `${raw.slice(0, RESPONSE_LOG_CAP)}…` : raw;
    const safe = capped.split(config.apiToken).join("[REDACTED]");
    console.error(
      `[telemetry] report failed status=${result.status} err=${result.error ?? ""} body=${safe}`
    );
    return;
  }
  console.log(
    `[telemetry] reported snapshot oom=${snapshot.oomCount24h} faulty=${snapshot.faultyContainers.length} ram=${snapshot.freeRamBytes} disk=${snapshot.freeDiskBytes}`
  );
}

export interface RunLoopDeps {
  tickFn?: (config: ReporterConfig) => Promise<void>;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  onSignal?: (handler: () => void) => void;
}

const defaultSleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true }
    );
  });

const defaultOnSignal = (handler: () => void): void => {
  process.once("SIGTERM", handler);
  process.once("SIGINT", handler);
};

export async function runLoop(config: ReporterConfig, deps: RunLoopDeps = {}): Promise<void> {
  const tickFn = deps.tickFn ?? tick;
  const sleep = deps.sleep ?? defaultSleep;
  const onSignal = deps.onSignal ?? defaultOnSignal;

  const ac = new AbortController();
  let stopped = false;
  const stop = () => {
    stopped = true;
    ac.abort();
  };
  onSignal(stop);

  await tickFn(config).catch((err) => {
    console.error(`[telemetry] tick error: ${err instanceof Error ? err.message : String(err)}`);
  });

  while (!stopped) {
    await sleep(config.intervalSec * 1000, ac.signal);
    if (stopped) break;
    await tickFn(config).catch((err) => {
      console.error(`[telemetry] tick error: ${err instanceof Error ? err.message : String(err)}`);
    });
  }
}

async function main(): Promise<void> {
  let config: ReporterConfig;
  try {
    config = loadConfigFromEnv();
  } catch (err) {
    if (err instanceof ConfigError) {
      console.error(`[telemetry] config error: ${err.message}`);
      process.exit(2);
    }
    throw err;
  }

  await runLoop(config);
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(`[telemetry] fatal: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
    process.exit(1);
  });
}
