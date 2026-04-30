import { describe, test, expect } from "bun:test";
import { buildReportRequestBody, postReport } from "../lib/reporter";
import type { ReporterConfig, TelemetrySnapshot } from "../lib/types";

const baseConfig: ReporterConfig = {
  apiUrl: "https://platform.example/api/v1",
  apiToken: "tok-double-base64",
  instanceId: "11111111-1111-1111-1111-111111111111",
  diskPath: "/",
  meminfoPath: "/proc/meminfo",
  oomLookback: "24 hours ago",
  intervalSec: 3600,
};

const snapshot: TelemetrySnapshot = {
  oomCount24h: 2,
  faultyContainers: ["cadvisor"],
  freeRamBytes: 8589934592,
  freeDiskBytes: 50_000_000_000,
  metadata: { agent_version: "0.1.0" },
};

describe("buildReportRequestBody", () => {
  test("packs the snapshot into the platform's RPC argument shape", () => {
    const body = buildReportRequestBody(baseConfig, snapshot);
    expect(body).toEqual({
      api_token: baseConfig.apiToken,
      instance_id: baseConfig.instanceId,
      oom_count_24h: 2,
      faulty_containers: ["cadvisor"],
      free_ram_bytes: 8589934592,
      free_disk_bytes: 50_000_000_000,
      metadata: { agent_version: "0.1.0" },
    });
  });

  test("emits metadata: null when not provided", () => {
    const body = buildReportRequestBody(baseConfig, { ...snapshot, metadata: undefined });
    expect(body.metadata).toBeNull();
  });
});

describe("postReport", () => {
  test("POSTs to /rpc/monitoring_instance_telemetry_report with the right body", async () => {
    let calledUrl = "";
    let calledMethod = "";
    let calledBody = "";
    const fetchImpl = async (url: string, init: { method: string; headers: Record<string, string>; body: string }) => {
      calledUrl = url;
      calledMethod = init.method;
      calledBody = init.body;
      return {
        ok: true,
        status: 200,
        text: async () => '{"result":"OK"}',
      };
    };
    const result = await postReport(baseConfig, snapshot, fetchImpl);
    expect(result.ok).toBe(true);
    expect(result.status).toBe(200);
    expect(result.body).toEqual({ result: "OK" });
    expect(calledMethod).toBe("POST");
    expect(calledUrl).toBe("https://platform.example/api/v1/rpc/monitoring_instance_telemetry_report");
    const parsed = JSON.parse(calledBody);
    expect(parsed.instance_id).toBe(baseConfig.instanceId);
    expect(parsed.oom_count_24h).toBe(2);
  });

  test("strips a trailing slash from apiUrl", async () => {
    let calledUrl = "";
    const fetchImpl = async (url: string) => {
      calledUrl = url;
      return { ok: true, status: 200, text: async () => "" };
    };
    await postReport({ ...baseConfig, apiUrl: "https://platform.example/api/v1/" }, snapshot, fetchImpl);
    expect(calledUrl).toBe("https://platform.example/api/v1/rpc/monitoring_instance_telemetry_report");
  });

  test("returns ok=false on non-2xx without throwing", async () => {
    const fetchImpl = async () => ({
      ok: false,
      status: 401,
      text: async () => '{"message":"Unauthorized"}',
    });
    const result = await postReport(baseConfig, snapshot, fetchImpl);
    expect(result.ok).toBe(false);
    expect(result.status).toBe(401);
  });

  test("returns ok=false with error message on network failure", async () => {
    const fetchImpl = async () => {
      throw new Error("ECONNRESET");
    };
    const result = await postReport(baseConfig, snapshot, fetchImpl);
    expect(result.ok).toBe(false);
    expect(result.status).toBe(0);
    expect(result.error).toBe("ECONNRESET");
  });
});
