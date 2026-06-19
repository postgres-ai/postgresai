import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { addInstanceToFile, buildInstance } from "../lib/instances";
import {
  parseVcpus,
  resolveAasLabels,
  registerAasCollection,
} from "../lib/aas-onboard";

/** Minimal Response-like stub for mocking fetch. */
function res(ok: boolean, status: number, jsonBody: unknown, textBody = ""): Response {
  return {
    ok,
    status,
    json: async () => jsonBody,
    text: async () => textBody,
  } as unknown as Response;
}

describe("parseVcpus", () => {
  test("non-positive / junk → 0 (the 'unknown' fallback)", () => {
    expect(parseVcpus(undefined)).toBe(0);
    expect(parseVcpus(null)).toBe(0);
    expect(parseVcpus("")).toBe(0);
    expect(parseVcpus("0")).toBe(0);
    expect(parseVcpus("-4")).toBe(0);
    expect(parseVcpus("abc")).toBe(0);
  });
  test("positive values → integer", () => {
    expect(parseVcpus("16")).toBe(16);
    expect(parseVcpus(8)).toBe(8);
    expect(parseVcpus("12.9")).toBe(12);
    expect(parseVcpus("  4 ")).toBe(4);
  });
});

describe("resolveAasLabels", () => {
  let dir: string;
  let file: string;
  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "aas-labels-"));
    file = path.join(dir, "instances.yml");
  });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  test("single enabled target → its (cluster, node_name) from custom_tags", () => {
    addInstanceToFile(file, buildInstance("appdb", "postgresql://u@h:5432/db"));
    expect(resolveAasLabels(file)).toEqual({ cluster: "default", node: "appdb" });
  });

  test("no targets → null", () => {
    fs.writeFileSync(file, "# empty\n");
    expect(resolveAasLabels(file)).toBeNull();
  });

  test("more than one enabled target → null (cannot disambiguate)", () => {
    addInstanceToFile(file, buildInstance("a", "postgresql://u@h:5432/a"));
    addInstanceToFile(file, buildInstance("b", "postgresql://u@h:5432/b"));
    expect(resolveAasLabels(file)).toBeNull();
  });

  test("missing file → null (no throw)", () => {
    expect(resolveAasLabels(path.join(dir, "nope.yml"))).toBeNull();
  });
});

describe("registerAasCollection", () => {
  let dir: string;
  let instancesPath: string;
  let fetchSpy: ReturnType<typeof spyOn>;
  let calls: Array<{ url: string; method: string; body?: string }>;

  // Route a fetch by URL+method to canned Grafana/RPC responses. Options let a
  // test exercise the existing-SA branch, datasource ambiguity, a keyless mint,
  // and RPC success/failure.
  function installFetch(opts: {
    rpc?: { ok: boolean; status: number; text?: string };
    existingSa?: boolean; // search finds an existing pgai-aas-collect SA
    prometheusCount?: number; // # of prometheus-typed datasources (default 1)
    mintKey?: string | null; // token .key; null => mint returns no key
  } = {}) {
    const rpc = opts.rpc ?? { ok: true, status: 200 };
    const existingSa = opts.existingSa ?? false;
    const promCount = opts.prometheusCount ?? 1;
    const mintKey = opts.mintKey === undefined ? "glsa_mock_token_xyz" : opts.mintKey;
    calls = [];
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string; body?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      calls.push({ url, method, body: init?.body });
      if (url.includes("/api/serviceaccounts/search"))
        return res(true, 200, existingSa ? { serviceAccounts: [{ id: 99, name: "pgai-aas-collect" }] } : { serviceAccounts: [] });
      if (url.match(/\/tokens$/) && method === "POST") return res(true, 200, mintKey === null ? {} : { key: mintKey });
      if (url.endsWith("/api/serviceaccounts") && method === "POST") return res(true, 201, { id: 42, name: "pgai-aas-collect" });
      if (url.includes("/api/datasources")) {
        const dss: Array<Record<string, unknown>> = [];
        for (let i = 0; i < promCount; i++) dss.push({ id: 8 + i, uid: `prom${i}`, type: "prometheus" });
        dss.push({ id: 3, uid: "loki1", type: "loki" });
        return res(true, 200, dss);
      }
      if (url.includes("/rpc/monitoring_instance_aas_register")) return res(rpc.ok, rpc.status, {}, rpc.text || "");
      return res(false, 404, {});
    }) as unknown as typeof fetch);
  }

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "aas-reg-"));
    instancesPath = path.join(dir, "instances.yml");
    addInstanceToFile(instancesPath, buildInstance("appdb", "postgresql://u@h:5432/db"));
  });
  afterEach(() => {
    fetchSpy?.mockRestore();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("happy path: mints SA, resolves datasource, POSTs the RPC with the right body", async () => {
    installFetch();
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 16,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(true);

    const rpc = calls.find((c) => c.url.includes("/rpc/monitoring_instance_aas_register"));
    expect(rpc).toBeDefined();
    expect(rpc!.url).toBe("https://api.test/rpc/monitoring_instance_aas_register");
    const body = JSON.parse(rpc!.body!);
    expect(body).toMatchObject({
      api_token: "apikey-1",
      instance_id: "inst-123",
      sa_token: "glsa_mock_token_xyz",
      cluster_name: "default",
      node_name: "appdb",
      vcpus: 16,
      datasource_id: 8, // the prometheus one, not loki
    });
    // a fresh SA was created (search found none) and a token minted on its id.
    expect(calls.some((c) => c.url.endsWith("/api/serviceaccounts") && c.method === "POST")).toBe(true);
    expect(calls.some((c) => c.url.match(/\/serviceaccounts\/42\/tokens$/) && c.method === "POST")).toBe(true);
  });

  test("platform error → ok:false, reason carries the status (best-effort, no throw)", async () => {
    installFetch({ rpc: { ok: false, status: 403, text: "forbidden" } });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 16,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("403");
  });

  test("no resolvable target → ok:false and NO outbound calls (labels checked first)", async () => {
    installFetch();
    const empty = path.join(dir, "empty.yml");
    fs.writeFileSync(empty, "# none\n");
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw",
      instancesPath: empty,
      vcpus: 16,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("cluster");
    expect(calls.length).toBe(0); // bailed before any HTTP
  });

  test("missing api key / instance id → ok:false, no calls", async () => {
    installFetch();
    const r = await registerAasCollection("", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 0,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(calls.length).toBe(0);
  });

  test("existing service account is reused (no create), token minted on its id", async () => {
    installFetch({ existingSa: true });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(true);
    expect(calls.some((c) => c.url.endsWith("/api/serviceaccounts") && c.method === "POST")).toBe(false);
    expect(calls.some((c) => c.url.match(/\/serviceaccounts\/99\/tokens$/) && c.method === "POST")).toBe(true);
  });

  test("absent or ambiguous (>1) prometheus datasource → ok:false, no RPC call", async () => {
    for (const n of [0, 2]) {
      fetchSpy?.mockRestore();
      installFetch({ prometheusCount: n });
      const r = await registerAasCollection("apikey-1", "inst-123", {
        grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
        // 0/>1 is a definitive skip; cap the readiness retry so the test stays fast.
        datasourceMaxAttempts: 2, datasourceRetryDelayMs: 0,
      });
      expect(r.ok).toBe(false);
      expect(r.reason).toContain("datasource");
      expect(calls.some((c) => c.url.includes("/rpc/monitoring_instance_aas_register"))).toBe(false);
    }
  });

  test("polls the datasource until Grafana is ready, then registers", async () => {
    // Grafana isn't ready on the first probes (no prometheus datasource yet),
    // then it provisions — the readiness retry must keep going and then succeed.
    let dsProbes = 0;
    calls = [];
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string; body?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      calls.push({ url, method, body: init?.body });
      if (url.includes("/api/serviceaccounts/search")) return res(true, 200, { serviceAccounts: [] });
      if (url.match(/\/tokens$/) && method === "POST") return res(true, 200, { key: "glsa_mock" });
      if (url.endsWith("/api/serviceaccounts") && method === "POST") return res(true, 201, { id: 42 });
      if (url.includes("/api/datasources")) {
        dsProbes++;
        return dsProbes < 3
          ? res(true, 200, [{ id: 3, type: "loki" }]) // not ready yet
          : res(true, 200, [{ id: 8, type: "prometheus" }, { id: 3, type: "loki" }]);
      }
      if (url.includes("/rpc/monitoring_instance_aas_register")) return res(true, 200, {});
      return res(false, 404, {});
    }) as unknown as typeof fetch);

    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
      datasourceMaxAttempts: 6, datasourceRetryDelayMs: 0,
    });
    expect(r.ok).toBe(true);
    expect(dsProbes).toBeGreaterThanOrEqual(3); // kept polling past the not-ready probes
    const rpc = calls.find((c) => c.url.includes("/rpc/monitoring_instance_aas_register"));
    expect(rpc).toBeDefined();
    expect(JSON.parse(rpc!.body!).datasource_id).toBe(8);
  });

  test(">1 prometheus datasource is a definitive skip: one probe, no retry", async () => {
    // The >1 case is permanent (the datasource count only grows), so the
    // readiness loop must bail after a single probe, not burn its whole budget.
    let dsProbes = 0;
    calls = [];
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string; body?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      calls.push({ url, method, body: init?.body });
      if (url.includes("/api/datasources")) {
        dsProbes++;
        return res(true, 200, [{ id: 8, type: "prometheus" }, { id: 9, type: "prometheus" }, { id: 3, type: "loki" }]);
      }
      return res(false, 404, {});
    }) as unknown as typeof fetch);

    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
      datasourceMaxAttempts: 5, datasourceRetryDelayMs: 0,
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("datasource");
    expect(dsProbes).toBe(1); // bailed after one probe; did NOT retry 5x
    expect(calls.some((c) => c.url.includes("/rpc/monitoring_instance_aas_register"))).toBe(false);
  });

  test("never-ready datasource: polls exactly maxAttempts times, then ok:false", async () => {
    // Bounds the readiness loop: a never-appearing datasource must probe exactly
    // maxAttempts times (N probes, N-1 sleeps) and then give up — not loop forever.
    let dsProbes = 0;
    calls = [];
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string; body?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      calls.push({ url, method, body: init?.body });
      if (url.includes("/api/datasources")) { dsProbes++; return res(true, 200, [{ id: 3, type: "loki" }]); } // never a prometheus
      return res(false, 404, {});
    }) as unknown as typeof fetch);

    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
      datasourceMaxAttempts: 3, datasourceRetryDelayMs: 0,
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("datasource");
    expect(dsProbes).toBe(3); // bounded: exactly maxAttempts probes
    expect(calls.some((c) => c.url.includes("/rpc/monitoring_instance_aas_register"))).toBe(false);
  });

  test("mint returning no key → ok:false, no RPC call", async () => {
    installFetch({ mintKey: null });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 8, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("service-account token");
    expect(calls.some((c) => c.url.includes("/rpc/monitoring_instance_aas_register"))).toBe(false);
  });
});
