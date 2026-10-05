import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { addTarget } from "../bin/postgres-ai";
import { addInstanceToFile, buildInstance, loadInstances } from "../lib/instances";

const hostname = "retry.pg.clickhouse.cloud";
const conn = `postgresql://monitor:password@${hostname}:5432/postgres`;
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
let dir: string, server: ReturnType<typeof Bun.serve>, requests: string[];
let credentials: NodeJS.ProcessEnv;
beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/clickhouse-retry-`);
  requests = [];
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const path = new URL(request.url).pathname;
    requests.push(path);
    if (request.headers.get("authorization") !== `Basic ${btoa("key:good-secret")}`) {
      return new Response("Unauthorized", { status: 401 });
    }
    const base = `/v1/organizations/${orgId}/postgres`;
    const service = { id: serviceId, name: "retry", state: "running", hostname };
    if (path === base) return Response.json({ result: [service] });
    if (path === `${base}/${serviceId}`) return Response.json({ result: service });
    return new Response("Not found", { status: 404 });
  } });
  credentials = { CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: "key", CLICKHOUSE_KEY_SECRET: "good-secret", CLICKHOUSE_API_URL: server.url.origin };
});
afterEach(() => { server.stop(true); rmSync(dir, { recursive: true, force: true }); });

async function run(env: NodeJS.ProcessEnv, connection = conn) {
  const stdout: string[] = [], stderr: string[] = [];
  const log = spyOn(console, "log").mockImplementation((message) => { stdout.push(String(message)); });
  const error = spyOn(console, "error").mockImplementation((message) => { stderr.push(String(message)); });
  const previous = process.exitCode ?? 0;
  process.exitCode = 0;
  try {
    await addTarget(`${dir}/instances.yml`, dir, connection, "retry", env, { apply: false });
    return { code: process.exitCode, stdout: stdout.join("\n"), stderr: stderr.join("\n") };
  } finally {
    process.exitCode = previous;
    log.mockRestore();
    error.mockRestore();
  }
}

function expectConfigured() {
  expect(loadInstances(`${dir}/instances.yml`).map(({ name, conn_str }) => ({ name, conn_str }))).toEqual([{ name: "retry", conn_str: conn }]);
  expect(existsSync(`${dir}/host-metrics/clickhouse-retry.yml`)).toBe(true);
  expect(readFileSync(`${dir}/host-metrics/clickhouse-retry.secret`, "utf8")).toBe("good-secret");
}

test("targets add resolves with a format error for a malformed channel_binding URL", async () => {
  await expect(run(credentials, "postgres://u:p@[bad/db?channel_binding=require")).resolves.toEqual({
    code: 1,
    stdout: "",
    stderr: "Invalid connection string format: use postgresql://user:password@host[:port]/database",
  });
  expect(requests).toEqual([]);
  expect(readdirSync(dir)).toEqual([]);
});

test("targets add retries host metrics after missing credentials", async () => {
  const first = await run({});
  expect(first.code).toBe(0);
  expect(first.stdout).toContain("set CLICKHOUSE_ORG_ID, CLICKHOUSE_KEY_ID and CLICKHOUSE_KEY_SECRET and re-run");
  expect(requests).toEqual([]);
  const retry = await run(credentials);
  expect(retry.code).toBe(0);
  expect(retry.stdout).toContain("Monitoring target 'retry' already exists");
  expect(retry.stderr).toBe("");
  expect(requests).toHaveLength(2);
  expectConfigured();
});

test("targets add retries host metrics after API 401", async () => {
  const first = await run({ ...credentials, CLICKHOUSE_KEY_SECRET: "bad-secret" });
  expect(first.code).toBe(1);
  expect(first.stderr).toContain("ClickHouse Cloud rejected the API key (401)");
  expect(requests).toHaveLength(1);
  const retry = await run(credentials);
  expect(retry.code).toBe(0);
  expect(retry.stderr).toBe("");
  expect(requests).toHaveLength(3);
  expectConfigured();
});

test("targets add rejects a conflicting connection before calling host metrics", async () => {
  expect((await run({})).code).toBe(0);
  const result = await run(credentials, conn.replace("password@", "different@"));
  expect(result.code).toBe(1);
  expect(result.stderr).toBe("Monitoring target 'retry' already exists");
  expect(requests).toEqual([]);
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(conn);
  expect(existsSync(`${dir}/host-metrics`)).toBe(false);
});

test("targets add accepts the postgres:// URL ClickHouse Cloud hands out", async () => {
  const url = `postgres://monitor:password@${hostname}:5432/postgres?sslmode=require`;
  const result = await run(credentials, url);
  expect(result.code).toBe(0);
  expect(result.stderr).toBe("");
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(url);
  expect(existsSync(`${dir}/host-metrics/clickhouse-retry.yml`)).toBe(true);
});

test("targets add drops channel_binding, which pgwatch rejects as a server parameter", async () => {
  const result = await run(credentials, `postgres://monitor:password@${hostname}:5432/postgres?sslmode=require&channel_binding=require`);
  expect(result.code).toBe(0);
  expect(result.stderr).toContain("Note: removed channel_binding from the connection string; the collector does not support it (TLS is kept)");
  expect(result.stdout).not.toContain("Note: removed channel_binding");
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(`postgres://monitor:password@${hostname}:5432/postgres?sslmode=require`);
});

test("targets add retries with the saved target labels and preserves instance data", async () => {
  const instance = { ...buildInstance("retry", conn), custom_tags: { cluster: "production-eu", node_name: "db-primary" } };
  addInstanceToFile(`${dir}/instances.yml`, instance);
  const saved = readFileSync(`${dir}/instances.yml`, "utf8");
  const result = await run(credentials);
  expect(result.code).toBe(0);
  expect(result.stderr).toBe("");
  const [scrape] = Bun.YAML.parse(readFileSync(`${dir}/host-metrics/clickhouse-retry.yml`, "utf8")) as any[];
  expect(scrape.static_configs[0].labels).toEqual({ cluster: "production-eu", node_name: "db-primary", __pgai_rev: expect.stringMatching(/^r[0-9a-f]{16}$/) });
  expect(scrape.job_name).toBe("clickhouse-retry");
  expect(readFileSync(`${dir}/instances.yml`, "utf8")).toBe(saved);
  expect(loadInstances(`${dir}/instances.yml`)).toEqual([instance]);
});
