import { describe, test, expect } from "bun:test";
import { loadConfigFromEnv, ConfigError } from "../lib/config";

const minimalEnv = {
  PGAI_PLATFORM_API_URL: "https://postgres.ai/api/v1",
  PGAI_API_TOKEN: "tok",
  PGAI_MONITORING_INSTANCE_ID: "11111111-1111-1111-1111-111111111111",
};

describe("loadConfigFromEnv", () => {
  test("loads all required vars and applies sensible defaults", () => {
    const cfg = loadConfigFromEnv(minimalEnv);
    expect(cfg.apiUrl).toBe("https://postgres.ai/api/v1");
    expect(cfg.apiToken).toBe("tok");
    expect(cfg.instanceId).toBe("11111111-1111-1111-1111-111111111111");
    expect(cfg.diskPath).toBe("/");
    expect(cfg.meminfoPath).toBe("/proc/meminfo");
    expect(cfg.oomLookback).toBe("24 hours ago");
    expect(cfg.intervalSec).toBe(3600);
  });

  test("throws ConfigError when a required var is missing", () => {
    expect(() => loadConfigFromEnv({ ...minimalEnv, PGAI_API_TOKEN: undefined })).toThrow(ConfigError);
  });

  test("throws ConfigError when interval is non-numeric", () => {
    expect(() =>
      loadConfigFromEnv({ ...minimalEnv, PGAI_TELEMETRY_INTERVAL_SEC: "lots" })
    ).toThrow(ConfigError);
  });

  test("throws ConfigError when interval is below the 60s floor", () => {
    expect(() =>
      loadConfigFromEnv({ ...minimalEnv, PGAI_TELEMETRY_INTERVAL_SEC: "30" })
    ).toThrow(ConfigError);
  });

  test("respects overrides for paths and lookback", () => {
    const cfg = loadConfigFromEnv({
      ...minimalEnv,
      PGAI_TELEMETRY_DISK_PATH: "/var/lib/docker",
      PGAI_TELEMETRY_MEMINFO_PATH: "/host/proc/meminfo",
      PGAI_TELEMETRY_OOM_LOOKBACK: "48 hours ago",
      PGAI_TELEMETRY_INTERVAL_SEC: "1800",
    });
    expect(cfg.diskPath).toBe("/var/lib/docker");
    expect(cfg.meminfoPath).toBe("/host/proc/meminfo");
    expect(cfg.oomLookback).toBe("48 hours ago");
    expect(cfg.intervalSec).toBe(1800);
  });
});
