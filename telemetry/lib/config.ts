/**
 * Loads ReporterConfig from environment variables.
 *
 * Required:
 *   PGAI_PLATFORM_API_URL  -- e.g. "https://postgres.ai/api/v1"
 *   PGAI_API_TOKEN         -- double-base64 token (existing checkup format)
 *   PGAI_MONITORING_INSTANCE_ID -- uuid of this monitoring instance
 *
 * Optional:
 *   PGAI_TELEMETRY_DISK_PATH    -- defaults to "/"
 *   PGAI_TELEMETRY_MEMINFO_PATH -- defaults to "/proc/meminfo"
 *   PGAI_TELEMETRY_OOM_LOOKBACK -- defaults to "24 hours ago"
 *   PGAI_TELEMETRY_INTERVAL_SEC -- defaults to 3600
 */

import type { ReporterConfig } from "./types";

export class ConfigError extends Error {}

export function loadConfigFromEnv(env: Record<string, string | undefined> = process.env): ReporterConfig {
  const apiUrl = required(env, "PGAI_PLATFORM_API_URL");
  const apiToken = required(env, "PGAI_API_TOKEN");
  const instanceId = required(env, "PGAI_MONITORING_INSTANCE_ID");

  const intervalSecRaw = env.PGAI_TELEMETRY_INTERVAL_SEC ?? "3600";
  const intervalSec = Number.parseInt(intervalSecRaw, 10);
  if (!Number.isFinite(intervalSec) || intervalSec < 60) {
    throw new ConfigError(
      `PGAI_TELEMETRY_INTERVAL_SEC must be an integer >= 60, got "${intervalSecRaw}"`
    );
  }

  return {
    apiUrl,
    apiToken,
    instanceId,
    diskPath: env.PGAI_TELEMETRY_DISK_PATH ?? "/",
    meminfoPath: env.PGAI_TELEMETRY_MEMINFO_PATH ?? "/proc/meminfo",
    oomLookback: env.PGAI_TELEMETRY_OOM_LOOKBACK ?? "24 hours ago",
    intervalSec,
  };
}

function required(env: Record<string, string | undefined>, key: string): string {
  const v = env[key];
  if (v === undefined || v.trim() === "") {
    throw new ConfigError(`required env var ${key} is missing`);
  }
  return v;
}
