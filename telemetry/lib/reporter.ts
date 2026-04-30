/**
 * Reporter: posts a TelemetrySnapshot to the PostgresAI platform's
 * v1.monitoring_instance_telemetry_report RPC.
 */

import type { ReporterConfig, TelemetrySnapshot } from "./types";

export interface ReportRequestBody {
  api_token: string;
  instance_id: string;
  oom_count_24h: number;
  faulty_containers: string[];
  free_ram_bytes: number;
  free_disk_bytes: number;
  metadata: Record<string, unknown> | null;
}

export interface ReporterResult {
  ok: boolean;
  status: number;
  body?: unknown;
  error?: string;
}

export type FetchLike = (
  input: string,
  init: { method: string; headers: Record<string, string>; body: string }
) => Promise<{ ok: boolean; status: number; text: () => Promise<string> }>;

export function buildReportRequestBody(
  config: Pick<ReporterConfig, "apiToken" | "instanceId">,
  snapshot: TelemetrySnapshot
): ReportRequestBody {
  return {
    api_token: config.apiToken,
    instance_id: config.instanceId,
    oom_count_24h: snapshot.oomCount24h,
    faulty_containers: snapshot.faultyContainers,
    free_ram_bytes: snapshot.freeRamBytes,
    free_disk_bytes: snapshot.freeDiskBytes,
    metadata: snapshot.metadata ?? null,
  };
}

export async function postReport(
  config: ReporterConfig,
  snapshot: TelemetrySnapshot,
  fetchImpl?: FetchLike
): Promise<ReporterResult> {
  const url = `${config.apiUrl.replace(/\/+$/, "")}/rpc/monitoring_instance_telemetry_report`;
  const body = buildReportRequestBody(config, snapshot);
  const fn = fetchImpl ?? (fetch as unknown as FetchLike);
  try {
    const res = await fn(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: unknown = text;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = text;
      }
    }
    return { ok: res.ok, status: res.status, body: parsed };
  } catch (err) {
    return { ok: false, status: 0, error: err instanceof Error ? err.message : String(err) };
  }
}
