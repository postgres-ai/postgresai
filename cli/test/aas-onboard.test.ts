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

  // Route a fetch by URL+method to a canned Grafana/RPC response. `rpc`
  // controls the final RPC outcome so tests can exercise success and failure.
  function installFetch(rpc: { ok: boolean; status: number; text?: string }) {
    calls = [];
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string; body?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      calls.push({ url, method, body: init?.body });
      if (url.includes("/api/serviceaccounts/search")) return res(true, 200, { serviceAccounts: [] });
      if (url.match(/\/tokens$/) && method === "GET") return res(true, 200, []);
      if (url.match(/\/tokens$/) && method === "POST") return res(true, 200, { key: "glsa_mock_token_xyz" });
      if (url.endsWith("/api/serviceaccounts") && method === "POST") return res(true, 201, { id: 42, name: "pgai-aas-collect" });
      if (url.includes("/api/datasources")) return res(true, 200, [
        { id: 8, uid: "prom1", type: "prometheus" },
        { id: 3, uid: "loki1", type: "loki" },
      ]);
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
    installFetch({ ok: true, status: 200 });
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
    // pruned-then-minted: a DELETE-less run is fine (no prior tokens), but the
    // token GET (prune scan) + POST (mint) must both have happened.
    expect(calls.some((c) => c.url.match(/\/tokens$/) && c.method === "POST")).toBe(true);
  });

  test("platform error → ok:false, reason carries the status (best-effort, no throw)", async () => {
    installFetch({ ok: false, status: 403, text: "forbidden" });
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
    installFetch({ ok: true, status: 200 });
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
    installFetch({ ok: true, status: 200 });
    const r = await registerAasCollection("", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 0,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(calls.length).toBe(0);
  });
});
