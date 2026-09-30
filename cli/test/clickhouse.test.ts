import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { findService, renderScrapeConfig, scrapeRevision } from "../lib/clickhouse";

const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
const hostname = "my-postgres.us-east-1.aws.pg.clickhouse.cloud";
const keyId = "test-key-id";
const keySecret = "fixture-only-secret";
const listPath = `/v1/organizations/${orgId}/postgres`;
let server: ReturnType<typeof Bun.serve> | undefined;
afterEach(() => server?.stop(true));

function api(fetch: (request: Request) => Response) {
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch });
  return { apiUrl: server.url.origin, orgId, keyId, keySecret, hostname };
}

const renderOptions = {
  name: "my-postgres", cluster: "default", orgId, serviceId, keyId,
  passwordFile: "/etc/pgai/host-metrics/clickhouse-my-postgres.secret",
  apiUrl: "https://api.clickhouse.cloud",
};

describe("ClickHouse service discovery", () => {
  test.each(["running", "creating", "stopped"])("matches hostname case-insensitively and returns %s without judging state", async (state) => {
    const requests: string[] = [];
    const options = api((request) => {
      expect(request.method).toBe("GET");
      expect(request.headers.get("authorization")).toBe(`Basic ${btoa(`${keyId}:${keySecret}`)}`);
      const path = new URL(request.url).pathname;
      requests.push(path);
      if (path === listPath) return Response.json({ result: [
        { id: "11111111-1111-1111-1111-111111111111", name: "other", state: "running" },
        { id: serviceId, name: "my-postgres", state },
        { id: "22222222-2222-2222-2222-222222222222", name: "unvisited", state: "running" },
      ], status: 200 });
      if (path === `${listPath}/11111111-1111-1111-1111-111111111111`) return Response.json({ result: { hostname: "other.pg.clickhouse.cloud" } });
      if (path === `${listPath}/${serviceId}`) return Response.json({ result: { id: serviceId, name: "my-postgres", state, hostname: hostname.toUpperCase() } });
      return new Response("unexpected request", { status: 404 });
    });
    expect(await findService(options)).toEqual({ id: serviceId, name: "my-postgres", state });
    expect(requests).toEqual([listPath, `${listPath}/11111111-1111-1111-1111-111111111111`, `${listPath}/${serviceId}`]);
  });

  for (const status of [401, 403]) {
    test(`service GET HTTP ${status} has the exact actionable error`, async () => {
      const options = api((request) => new URL(request.url).pathname === listPath
        ? Response.json({ result: [{ id: serviceId }] }) : new Response("denied", { status }));
      await expect(findService(options)).rejects.toEqual(new Error(status === 401
        ? "ClickHouse Cloud rejected the API key (401). Check the key id and secret."
        : `The API key cannot read Postgres services in organization ${orgId} (403). Use a key with the Basic Service API Reader role, or one that includes it.`));
    });
  }
  test("empty list has the exact no-match error", async () => {
    await expect(findService(api(() => Response.json({ result: [] })))).rejects.toEqual(
      new Error(`No ClickHouse Managed Postgres service in organization ${orgId} has hostname ${hostname}.`));
  });
});

describe("ClickHouse scrape config", () => {
  test("matches the list-form golden exactly", () => {
    expect(renderScrapeConfig(renderOptions)).toBe(readFileSync(`${import.meta.dir}/fixtures/clickhouse-scrape.golden.yml`, "utf8"));
  });
  for (const field of ["orgId", "serviceId"] as const) {
    test.each(["", "../escape", "g".repeat(36), "a".repeat(35), "a".repeat(37), `${orgId}\njob_name: injected`])(`rejects invalid ${field}: %s`, (value) => {
      expect(() => renderScrapeConfig({ ...renderOptions, [field]: value })).toThrow();
    });
  }
  test.each(["", "../escape", "a/b", "a b", "name\njob_name: injected", "x: y"])("rejects invalid name: %s", (name) => {
    expect(() => renderScrapeConfig({ ...renderOptions, name })).toThrow();
  });
  test("changes the reload revision only when the rendered config changes", () => {
    const revision = (options: typeof renderOptions) => (Bun.YAML.parse(renderScrapeConfig(options)) as any[])[0].static_configs[0].labels.__pgai_rev;
    expect(revision(renderOptions)).toBe(revision({ ...renderOptions }));
    expect(revision({ ...renderOptions, keyId: "other-key-id" })).not.toBe(revision(renderOptions));
  });
  test("reads the revision label, not revision-like text in another label", () => {
    const dir = mkdtempSync(`${tmpdir()}/clickhouse-revision-`);
    try {
      const text = renderScrapeConfig({ ...renderOptions, cluster: "x\n__pgai_rev: r0000000000000000" });
      mkdirSync(`${dir}/host-metrics`);
      writeFileSync(`${dir}/host-metrics/clickhouse-my-postgres.yml`, text);
      expect(scrapeRevision(dir, "my-postgres")).toBe((Bun.YAML.parse(text) as any[])[0].static_configs[0].labels.__pgai_rev);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  test("quotes YAML-sensitive scalar values without changing their meaning", () => {
    const values = { cluster: "default\nextra: injected", keyId: "key: #id", passwordFile: "/tmp/secret: #file" };
    const [config] = Bun.YAML.parse(renderScrapeConfig({ ...renderOptions, ...values, name: "CH_01-test" })) as any[];
    expect(config.static_configs).toEqual([{ targets: ["api.clickhouse.cloud"], labels: { cluster: values.cluster, node_name: "CH_01-test", __pgai_rev: expect.stringMatching(/^r[0-9a-f]{16}$/) } }]);
    expect(config.basic_auth).toEqual({ username: values.keyId, password_file: values.passwordFile });
  });
});
