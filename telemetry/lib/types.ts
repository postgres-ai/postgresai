export interface TelemetrySnapshot {
  oomCount24h: number;
  faultyContainers: string[];
  freeRamBytes: number;
  freeDiskBytes: number;
  metadata?: Record<string, unknown>;
}

export interface ReporterConfig {
  apiUrl: string;
  apiToken: string;
  instanceId: string;
  // Path to monitor for free disk. Defaults to "/" if unset.
  diskPath: string;
  // Path to /proc/meminfo. Overridable for tests.
  meminfoPath: string;
  // How far back to scan the kernel log for OOMs. Defaults to "24 hours ago".
  oomLookback: string;
  // Loop interval in seconds. Defaults to 3600 (hourly).
  intervalSec: number;
}
