/**
 * High-level "run all collectors" helper. The entry script calls this on
 * each tick. Splitting this out keeps bin/telemetry.ts tiny and lets us
 * unit-test the orchestration with stubbed collectors.
 */

import { collectOomCount } from "./collectors/oom";
import { collectFaultyContainers } from "./collectors/containers";
import { collectFreeRamBytes } from "./collectors/memory";
import { collectFreeDiskBytes } from "./collectors/disk";
import type { ReporterConfig, TelemetrySnapshot } from "./types";

export interface CollectorOverrides {
  oom?: () => Promise<number>;
  containers?: () => Promise<string[]>;
  memory?: () => Promise<number>;
  disk?: () => Promise<number>;
}

export async function collectSnapshot(
  config: ReporterConfig,
  overrides: CollectorOverrides = {}
): Promise<TelemetrySnapshot> {
  const oomFn = overrides.oom ?? (() => collectOomCount({ lookback: config.oomLookback }));
  const containersFn = overrides.containers ?? (() => collectFaultyContainers());
  const memoryFn = overrides.memory ?? (() => collectFreeRamBytes({ meminfoPath: config.meminfoPath }));
  const diskFn = overrides.disk ?? (() => collectFreeDiskBytes({ path: config.diskPath }));

  const [oomCount24h, faultyContainers, freeRamBytes, freeDiskBytes] = await Promise.all([
    oomFn(),
    containersFn(),
    memoryFn(),
    diskFn(),
  ]);

  return {
    oomCount24h,
    faultyContainers,
    freeRamBytes,
    freeDiskBytes,
    metadata: {
      collected_at: new Date().toISOString(),
    },
  };
}
