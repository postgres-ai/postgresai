import { describe, test, expect } from "bun:test";
import { collectSnapshot } from "../lib/collect";
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

describe("collectSnapshot", () => {
  test("merges all collector results into a TelemetrySnapshot", async () => {
    const snap = await collectSnapshot(config, {
      oom: async () => 4,
      containers: async () => ["cadvisor", "node-exporter"],
      memory: async () => 2_000_000_000,
      disk: async () => 100_000_000_000,
    });
    expect(snap.oomCount24h).toBe(4);
    expect(snap.faultyContainers).toEqual(["cadvisor", "node-exporter"]);
    expect(snap.freeRamBytes).toBe(2_000_000_000);
    expect(snap.freeDiskBytes).toBe(100_000_000_000);
    expect(typeof snap.metadata?.collected_at).toBe("string");
  });

  test("runs collectors concurrently (overall duration <= max(individual))", async () => {
    const slow = (ms: number) =>
      new Promise((res) => setTimeout(res, ms));
    const t0 = Date.now();
    await collectSnapshot(config, {
      oom: async () => {
        await slow(60);
        return 1;
      },
      containers: async () => {
        await slow(60);
        return [];
      },
      memory: async () => {
        await slow(60);
        return 1;
      },
      disk: async () => {
        await slow(60);
        return 1;
      },
    });
    const elapsed = Date.now() - t0;
    // Sequential would be ~240ms; concurrent should land well under 200ms.
    expect(elapsed).toBeLessThan(200);
  });
});
