import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { addInstanceToFile, buildInstance } from "../lib/instances";
import {
  parseVcpus,
  resolveAasLabels,
  registerAasCollection,
  aasSuccessMessage,
  formatPlatformError,
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
    rpc?: { ok: boolean; status: number; text?: string; json?: unknown };
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
      if (url.includes("/rpc/monitoring_instance_aas_register"))
        return res(rpc.ok, rpc.status, rpc.json ?? {}, rpc.text || "");
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

  test("platform error → the reason carries the platform's own detail (#778)", async () => {
    // "platform returned HTTP 400" with no reason is why platform-all#778 -- the
    // pull-path url guard failing the AAS register on EVERY provisioned box --
    // stayed invisible for three months. PostgREST renders a plpgsql RAISE as
    // {code, message, details, hint}; `details` is the only field that says what
    // to fix, and it must reach the operator.
    installFetch({ rpc: { ok: false, status: 400, json: {
      code: "PT400",
      message: "Bad Request",
      details: "This monitoring instance has no Grafana url yet; it must be set during provisioning before AAS registration.",
      hint: null,
    } } });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 16, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("400");
    expect(r.reason).toContain("PT400");
    expect(r.reason).toContain("no Grafana url yet");
  });

  test("an error body that echoes the request never leaks a secret (#778)", async () => {
    // The reason this used to log the status ONLY. The request carries a freshly
    // minted Grafana token and the org API key, so a platform that echoes the
    // payload into an error must not put either into the operator's terminal.
    // Scrubbed BY VALUE (we know exactly what we sent), plus the glsa_ shape for
    // a token we did not send.
    installFetch({ rpc: { ok: false, status: 500, json: {
      code: "XX000",
      message: "Internal Server Error",
      details: 'while handling {"api_token":"apikey-1","sa_token":"glsa_mock_token_xyz"} and glsa_some_other_token',
    } } });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 16, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).not.toContain("glsa_mock_token_xyz");
    expect(r.reason).not.toContain("glsa_some_other_token");
    expect(r.reason).not.toContain("apikey-1");
    expect(r.reason).toContain("[redacted]");
  });

  test("a huge or multi-line error body is capped and flattened to one line (#778)", async () => {
    installFetch({ rpc: { ok: false, status: 400, json: {
      code: "PT400",
      details: "line one\nline two" + " padding".repeat(200),
    } } });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 16, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason!.length).toBeLessThan(400);
    expect(r.reason).not.toContain("\n");
  });

  test("a non-JSON / empty error body still yields the status alone (#778)", async () => {
    installFetch({ rpc: { ok: false, status: 502, json: null } });
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 16, apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(false);
    expect(r.reason).toContain("502");
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

  test("vcpus 0/omitted → still registers, sending vcpus 0 (#683: the platform owns the value)", async () => {
    // No provisioning path passes --vcpus/PGAI_VCPUS, so a client-side
    // vcpus > 0 gate disabled hands-off onboarding everywhere. The RPC accepts
    // 0 as "unknown" and never clobbers a platform-stamped value (#346).
    installFetch();
    const r = await registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 0,
      apiBaseUrl: "https://api.test",
    });
    expect(r.ok).toBe(true);

    const rpc = calls.find((c) => c.url.includes("/rpc/monitoring_instance_aas_register"));
    expect(rpc).toBeDefined();
    expect(JSON.parse(rpc!.body!)).toMatchObject({
      instance_id: "inst-123",
      sa_token: "glsa_mock_token_xyz",
      cluster_name: "default",
      node_name: "appdb",
      vcpus: 0,
      datasource_id: 8,
    });
  });

  test("missing api key / instance id → ok:false, no calls", async () => {
    installFetch();
    const r = await registerAasCollection("", "inst-123", {
      grafanaPassword: "pw",
      instancesPath,
      vcpus: 8,
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

describe("the platform's reply is read, not discarded (postgres-ai/postgresai#348)", () => {
  let dir: string;
  let instancesPath: string;
  let fetchSpy: ReturnType<typeof spyOn> | null = null;

  function installRpcReply(reply: unknown) {
    fetchSpy = spyOn(globalThis, "fetch").mockImplementation((async (input: unknown, init?: { method?: string }) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();
      if (url.includes("/api/serviceaccounts/search")) return res(true, 200, { serviceAccounts: [] });
      if (url.match(/\/tokens$/) && method === "POST") return res(true, 200, { key: "glsa_mock_token_xyz" });
      if (url.endsWith("/api/serviceaccounts") && method === "POST") return res(true, 201, { id: 42 });
      if (url.includes("/api/datasources")) return res(true, 200, [{ id: 8, type: "prometheus" }]);
      if (url.includes("/rpc/monitoring_instance_aas_register")) return res(true, 200, reply);
      return res(false, 404, {});
    }) as unknown as typeof fetch);
  }

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "aas-reply-"));
    instancesPath = path.join(dir, "instances.yml");
    addInstanceToFile(instancesPath, buildInstance("appdb", "postgresql://u@h:5432/db"));
  });
  afterEach(() => {
    fetchSpy?.mockRestore();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const register = () =>
    registerAasCollection("apikey-1", "inst-123", {
      grafanaPassword: "pw", instancesPath, vcpus: 0, apiBaseUrl: "https://api.test",
    });

  test("the effective vcpus and url_pending come back on the result", async () => {
    installRpcReply({ armed_for: "pull", vcpus: 0, url_pending: true, debounce_reset: true });
    const r = await register();
    expect(r.ok).toBe(true);
    expect(r.vcpus).toBe(0);
    expect(r.urlPending).toBe(true);
    expect(r.armedFor).toBe("pull");
  });

  test("effective vcpus 0 => the success line says collection is still OFF", () => {
    // The platform accepts vcpus 0 as "unknown" and the producer then skips with
    // no_vcpus forever. A bare "registered" is a lie for that instance (#348),
    // and platform-all#778 makes this the COMMON outcome: with the url guard
    // gone, arming now succeeds and no_vcpus becomes the only thing left.
    const msg = aasSuccessMessage({ ok: true, vcpus: 0 });
    expect(msg).toContain("vCPU");
    expect(msg.toUpperCase()).toContain("OFF");
  });

  test("a pending url is reported, but a missing vcpus count outranks it", () => {
    // A pending url resolves by itself minutes later (the deploy notify writes
    // it); a missing vcpus never does. Report the one that needs a human.
    expect(aasSuccessMessage({ ok: true, vcpus: 8, urlPending: true })).toContain("Grafana url");
    expect(aasSuccessMessage({ ok: true, vcpus: 0, urlPending: true })).toContain("vCPU");
  });

  test("a real vcpus and a known url => the plain success line", () => {
    const msg = aasSuccessMessage({ ok: true, vcpus: 16, urlPending: false });
    expect(msg).toBe("AAS auto-collection registered");
  });

  test("an older platform that returns neither key gets no false warning", async () => {
    // "Neither key" means neither vcpus nor url_pending. armed_for HAS been in
    // the reply since #743, so a realistic old reply still carries it — what
    // matters is that nothing the platform did not say gets asserted at the
    // operator, in either direction.
    installRpcReply({ armed_for: "pull", cluster_name: "default" });
    const r = await register();
    expect(r.ok).toBe(true);
    expect(r.vcpus).toBeUndefined();
    expect(r.urlPending).toBeUndefined();
    const msg = aasSuccessMessage(r);
    expect(msg).toContain("AAS auto-collection registered");
    expect(msg).not.toContain("stays OFF");
    expect(msg).not.toContain("Grafana url");
  });

  test("a reply with no armed_for at all gets exactly the plain line", async () => {
    installRpcReply({ cluster_name: "default" });
    const r = await register();
    expect(r.ok).toBe(true);
    expect(r.armedFor).toBeUndefined();
    expect(aasSuccessMessage(r)).toBe("AAS auto-collection registered");
  });

  test("an unparseable 200 body does not fail the registration", async () => {
    installRpcReply(null);
    const r = await register();
    expect(r.ok).toBe(true);
    expect(aasSuccessMessage(r)).toBe("AAS auto-collection registered");
  });
});

describe("formatPlatformError never leaks, whatever the platform echoes (#382)", () => {
  // Shapes of an org API token and a minted Grafana SA token, both of which
  // registerAasCollection passes in `secrets` because it sent them. Assembled
  // from parts rather than written as one literal so the repo's gitleaks hook
  // does not flag a high-entropy string: only the LENGTH and charset matter
  // here, never the value.
  const API = ["pgai", "v1", "notarealtoken", "aaaabbbbccccdddd"].join("_");
  const SA = ["glsa", "notareal", "grafana", "serviceaccount", "token"].join("_");

  /** Insert `gap` into `secret` at `at`, i.e. what a wrapped error body does. */
  const splitAt = (secret: string, at: number, gap: string) =>
    secret.slice(0, at) + gap + secret.slice(at);

  /** The operator-facing criterion: not "is the secret present verbatim" but
   *  "is it one whitespace-deletion away from usable". A scrub that runs BEFORE
   *  the flatten passes the first check and fails this one. */
  const leaks = (out: string, secret: string) =>
    out.includes(secret) || out.replace(/\s+/g, "").includes(secret.replace(/\s+/g, ""));

  const fmt = (details: string) =>
    formatPlatformError({ code: "PT400", message: "Bad Request", details }, [API, SA]);

  test("a secret split by a newline is still redacted (ordering bug)", () => {
    // The body wraps the token across lines (JSON pretty-printing, a log line
    // break). A by-value scrub that runs first cannot match it, and the
    // whitespace flatten then REASSEMBLES it on one line, one space-deletion
    // from usable. Normalise first, and match whitespace-tolerantly.
    const out = fmt(`rejected token ${splitAt(API, 18, "\n")}`);
    expect(leaks(out, API)).toBe(false);
    expect(out).toContain("[redacted]");
  });

  test("a secret split by a tab or by several spaces is still redacted", () => {
    for (const gap of ["\t", "   ", "\r\n", "\u2028"]) {
      const out = fmt(`tok ${splitAt(API, 18, gap)}`);
      expect(leaks(out, API)).toBe(false);
    }
  });

  test("a secret split by a CONTROL character is still redacted (pins the ORDER)", () => {
    // This is the case that makes step 1 load-bearing rather than defensive, and
    // the suite did not have it until a mutation showed that reversing the order
    // changed nothing. A whitespace-tolerant matcher does NOT cover it: `\s*`
    // matches no part of NUL/ESC/BEL/C1, so a scrub running first misses, and the
    // control-strip afterwards turns the separator into a space and REASSEMBLES
    // the secret. Normalising first is the only ordering that closes it.
    for (const gap of ["\u0000", "\u001b", "\u0007", "\u009b", "\u007f"]) {
      const out = fmt(`tok ${splitAt(API, 18, gap)}`);
      expect(leaks(out, API)).toBe(false);
    }
  });

  test("a split glsa_ service-account token is redacted, tail included", () => {
    const cut = 9;
    const out = fmt(`minted ${splitAt(SA, cut, "\n")} ok`);
    expect(leaks(out, SA)).toBe(false);
    // The tail alone is most of a live credential; it must not survive either.
    expect(out).not.toContain(SA.slice(cut));
  });

  test("no control character reaches the terminal", () => {
    // Untrusted platform text printed straight to a TTY. ESC clears the screen
    // and repaints; BEL rings; the C1 byte 0x9b IS a single-character CSI, so
    // dropping ESC alone is not enough. Classification mirrors isControl() in
    // instance-jobs/internal/collect/promql.go.
    const nasty = "bad \u001b[2J\u001b[1;31mHIJACK\u0007 \u0000 \u007f \u009b6n \u2028 \u2029 req";
    const out = fmt(nasty);
    expect(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/.test(out)).toBe(false);
    // ...and the readable words survive rather than being fused together.
    expect(out).toContain("bad");
    expect(out).toContain("req");
  });

  test("a credential the caller did NOT send is still scrubbed by pattern", () => {
    const PW = ["not", "a", "real", "password"].join("-");
    // The by-value scrub only knows what we sent. Reuse of redactTextSecrets
    // (cli/lib/util.ts) is what covers the rest: credential-named pairs and URL
    // userinfo. Note it does NOT close the split-secret case above -- a bare
    // token in prose has no key name to match -- which is why both layers exist.
    const out = formatPlatformError(
      { code: "PT500", details: `upstream said password=${PW} for postgresql://mon:${PW}@db:5432/app` },
      [API, SA],
    );
    expect(out).not.toContain(PW);
  });

  test("a truncated reason says it was truncated", () => {
    const out = fmt("x".repeat(900));
    expect(out.length).toBeLessThanOrEqual(300);
    expect(out.endsWith("\u2026")).toBe(true);
  });

  test("a short reason is not marked truncated", () => {
    const out = fmt("short and complete");
    expect(out).toBe("PT400: short and complete");
    expect(out.endsWith("\u2026")).toBe(false);
  });
});

describe("the armed path is surfaced, not just parsed (#382 F4)", () => {
  test("armedFor rides out on the success line", () => {
    expect(aasSuccessMessage({ ok: true, vcpus: 16, armedFor: "pull" })).toContain("pull");
    expect(aasSuccessMessage({ ok: true, vcpus: 16, armedFor: "jobs" })).toContain("job channel");
  });

  test("an older platform that reports no armed_for still gets the plain line", () => {
    expect(aasSuccessMessage({ ok: true, vcpus: 16 })).toBe("AAS auto-collection registered");
  });

  test("the vcpus and url warnings still outrank the armed path", () => {
    expect(aasSuccessMessage({ ok: true, vcpus: 0, armedFor: "pull" })).toContain("vCPU");
    expect(aasSuccessMessage({ ok: true, vcpus: 8, urlPending: true, armedFor: "pull" })).toContain("Grafana url");
  });
});
