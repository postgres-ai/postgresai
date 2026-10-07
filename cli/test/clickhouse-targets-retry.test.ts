import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { Client } from "pg";
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

async function run(env: NodeJS.ProcessEnv, connection = conn, verifyTls: ((url: string) => Promise<void>) | null = async (_url: string) => {}) {
  const stdout: string[] = [], stderr: string[] = [];
  const log = spyOn(console, "log").mockImplementation((message) => { stdout.push(String(message)); });
  const error = spyOn(console, "error").mockImplementation((message) => { stderr.push(String(message)); });
  const previous = process.exitCode ?? 0;
  process.exitCode = 0;
  try {
    await addTarget(`${dir}/instances.yml`, dir, connection, "retry", env, { apply: false, ...(verifyTls ? { verifyTls } : {}) });
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

test.each(["prefer", "disable"])("targets add drops channel_binding=%s with a note", async (mode) => {
  const result = await run({}, `postgres://monitor:password@${hostname}:5432/postgres?sslmode=require&channel_binding=${mode}`);
  expect(result.code).toBe(0);
  expect(result.stderr).toContain("Note: removed channel_binding from the connection string; the collector does not support it (TLS is kept)");
  expect(result.stdout).not.toContain("Note: removed channel_binding");
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(`postgres://monitor:password@${hostname}:5432/postgres?sslmode=require`);
  expect(requests).toEqual([]);
});

test.each(["disable", "allow"])("targets add refuses channel_binding=require with sslmode=%s before saving", async (sslmode) => {
  const result = await run({}, `${conn}?channel_binding=require&sslmode=${sslmode}`);
  expect(result.code).toBe(1);
  expect(result.stderr).toBe(`channel_binding=require needs TLS, but sslmode=${sslmode} is set`);
  expect(result.stdout).not.toContain("added");
  expect(requests).toEqual([]);
  expect(readdirSync(dir)).toEqual([]);
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

test("targets add strips channel_binding=require with verify-full and warns once", async () => {
  const result = await run({}, `${conn}?sslmode=verify-full&channel_binding=require`);
  expect(result.code).toBe(0);
  expect(result.stderr).toBe("Warning: the collector can't do channel binding; it connects with TLS and full certificate verification");
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(`${conn}?sslmode=verify-full`);
  expect(requests).toEqual([]);
});

const certErrors = ["DEPTH_ZERO_SELF_SIGNED_CERT", "SELF_SIGNED_CERT_IN_CHAIN", "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "UNABLE_TO_GET_ISSUER_CERT_LOCALLY", "ERR_TLS_CERT_ALTNAME_INVALID", "CERT_HAS_EXPIRED"];

test.each(["require", "prefer", "verify-ca", ""])("targets add upgrades channel_binding=require with sslmode=%s after verification", async (sslmode) => {
  const seen: string[] = [];
  const url = `${conn}?channel_binding=require${sslmode ? `&sslmode=${sslmode}` : ""}`;
  const result = await run({}, url, async (url) => { seen.push(url); });
  expect(result.code).toBe(0);
  expect(seen).toEqual([`${conn}?sslmode=verify-full`]);
  expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(`${conn}?sslmode=verify-full`);
  expect(result.stderr).toBe("Warning: the collector can't do channel binding; it connects with TLS and full certificate verification; upgraded to sslmode=verify-full");
});

test.each(certErrors)("targets add refuses a TLS verification failure (%s) before saving", async (code) => {
  const result = await run(credentials, `${conn}?sslmode=require&channel_binding=require`, async () => { throw Object.assign(new Error("certificate rejected"), { code }); });
  expect(result.code).toBe(1);
  expect(result.stderr).toContain("sslrootcert=<CA file>");
  expect(result.stderr).toContain("sslmode=verify-full");
  expect(result.stderr).toContain("removing channel_binding=require from the URL");
  expect(result.stdout).not.toContain("added");
  expect(requests).toEqual([]);
  expect(readdirSync(dir)).toEqual([]);
});

test("targets add verifies using the supplied CA and notes that the box needs it", async () => {
  const seen: string[] = [];
  const result = await run({}, `${conn}?sslmode=require&channel_binding=require&sslrootcert=%2Ftmp%2Fca.pem`, async (url) => { seen.push(url); });
  expect(result.code).toBe(0);
  expect(new URL(seen[0]).searchParams.get("sslrootcert")).toBe("/tmp/ca.pem");
  expect(new URL(seen[0]).searchParams.get("sslmode")).toBe("verify-full");
  expect(result.stderr).toContain("the monitoring box has no copy of the CA in sslrootcert");
  const saved = new URL(loadInstances(`${dir}/instances.yml`)[0].conn_str!);
  expect(saved.searchParams.get("sslrootcert")).toBe("/tmp/ca.pem");
  expect(saved.searchParams.get("sslmode")).toBe("verify-full");
  expect(saved.searchParams.has("channel_binding")).toBe(false);
});

test.each(["login", "28P01", "3D000", "post-handshake"])("the real TLS probe accepts %s only after certificate verification", async (answer) => {
  const configs: unknown[] = [];
  const connect = spyOn(Client.prototype, "connect").mockImplementation(async function(this: Client) {
    configs.push({ host: this.host, ssl: this.ssl });
    (this as any).connection.stream.authorized = true;
    if (answer !== "login") throw Object.assign(new Error("after handshake"), answer === "post-handshake" ? {} : { code: answer });
  });
  const end = spyOn(Client.prototype, "end").mockResolvedValue(undefined);
  try {
    const result = await run({}, `${conn}?sslmode=require&channel_binding=require`, null);
    expect(result.code).toBe(0);
    expect(configs).toEqual([{ host: hostname, ssl: { rejectUnauthorized: true, servername: hostname } }]);
    expect(loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(`${conn}?sslmode=verify-full`);
    expect(end).toHaveBeenCalledTimes(1);
  } finally { connect.mockRestore(); end.mockRestore(); }
});

test("the real TLS probe refuses an unverified handshake and closes the client", async () => {
  const connect = spyOn(Client.prototype, "connect").mockRejectedValue(Object.assign(new Error("self-signed"), { code: "DEPTH_ZERO_SELF_SIGNED_CERT" }));
  const end = spyOn(Client.prototype, "end").mockResolvedValue(undefined);
  try {
    const result = await run({}, `${conn}?sslmode=require&channel_binding=require`, null);
    expect(result.stderr).toContain("sslrootcert=<CA file>");
    expect(readdirSync(dir)).toEqual([]);
    expect(end).toHaveBeenCalledTimes(1);
  } finally { connect.mockRestore(); end.mockRestore(); }
});

test("the real TLS probe uses the CA file rather than the default roots", async () => {
  const ca = `${dir}/ca.pem`;
  writeFileSync(ca, "fixture CA");
  const configs: unknown[] = [];
  const connect = spyOn(Client.prototype, "connect").mockImplementation(async function(this: Client) { configs.push(this.ssl); });
  const end = spyOn(Client.prototype, "end").mockResolvedValue(undefined);
  try {
    const result = await run({}, `${conn}?sslmode=require&channel_binding=require&sslrootcert=${encodeURIComponent(ca)}`, null);
    expect(result.code).toBe(0);
    expect(configs).toEqual([{ rejectUnauthorized: true, servername: hostname, ca: "fixture CA" }]);
  } finally { connect.mockRestore(); end.mockRestore(); }
});

test("the real TLS probe refuses a connection failure before the handshake", async () => {
  const connect = spyOn(Client.prototype, "connect").mockRejectedValue(Object.assign(new Error("offline"), { code: "ECONNREFUSED" }));
  const end = spyOn(Client.prototype, "end").mockResolvedValue(undefined);
  try {
    const result = await run({}, `${conn}?sslmode=require&channel_binding=require`, null);
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("removing channel_binding=require from the URL");
    expect(readdirSync(dir)).toEqual([]);
    expect(end).toHaveBeenCalledTimes(1);
  } finally { connect.mockRestore(); end.mockRestore(); }
});
