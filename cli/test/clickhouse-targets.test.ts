import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { addHostMetrics, removeHostMetrics } from "../lib/clickhouse";
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
const hostname = "my-postgres.us-east-1.aws.pg.clickhouse.cloud";
const conn = `postgresql://monitor:fixture-password@${hostname}:5432/postgres`;
const keyId = "test-key-id", keySecret = "fixture-only-secret";
const listPath = `/v1/organizations/${orgId}/postgres`;
let dir: string, server: ReturnType<typeof Bun.serve>;
let status: number, state: string, match: boolean, requests: string[];
beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/clickhouse-targets-`);
  status = 200; state = "running"; match = true; requests = [];
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const path = new URL(request.url).pathname; requests.push(path);
    if (request.headers.get("authorization") !== `Basic ${btoa(`${keyId}:${keySecret}`)}`) return new Response("Unauthorized", { status: 401 });
    if (status !== 200) return new Response("Denied", { status });
    if (path === listPath) return Response.json({ result: [{ id: serviceId, name: "my-postgres", state }], status: 200 });
    if (path === `${listPath}/${serviceId}`) return Response.json({ result: { id: serviceId, name: "my-postgres", state, hostname: match ? hostname : "other.pg.clickhouse.cloud" } });
    return new Response("Not found", { status: 404 });
  } });
});
afterEach(() => { server.stop(true); rmSync(dir, { recursive: true, force: true }); });
function add(credentials: Record<string, string> = { CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: keyId, CLICKHOUSE_KEY_SECRET: keySecret }) {
  return addHostMetrics({ projectDir: dir, name: "ch-test", conn, env: { ...credentials, CLICKHOUSE_API_URL: server.url.origin } });
}
function expectNothingWritten() {
  expect(existsSync(`${dir}/host-metrics`) ? readdirSync(`${dir}/host-metrics`) : []).toEqual([]);
}
test("add discovers the service and writes config 0644 and secret 0600 without a newline", async () => {
  expect(await add()).toBe("Host metrics: ClickHouse Cloud Prometheus endpoint for service my-postgres (scraped every 60s)");
  expect(requests).toEqual([listPath, `${listPath}/${serviceId}`]);
  const configPath = `${dir}/host-metrics/clickhouse-ch-test.yml`;
  const secretPath = `${dir}/host-metrics/clickhouse-ch-test.secret`;
  expect(statSync(configPath).mode & 0o777).toBe(0o644);
  expect(statSync(secretPath).mode & 0o777).toBe(0o600);
  expect(readFileSync(secretPath, "utf8")).toBe(keySecret);
  const text = readFileSync(configPath, "utf8");
  expect(text).not.toContain(keySecret);
  const [config] = Bun.YAML.parse(text) as any[];
  expect(config.job_name).toBe("clickhouse-ch-test");
  expect(config.scheme).toBe("http");
  expect(config.metrics_path).toBe(`${listPath}/${serviceId}/prometheus`);
  expect(config.basic_auth).toEqual({ username: keyId, password_file: "/etc/pgai/host-metrics/clickhouse-ch-test.secret" });
  expect(config.static_configs[0].targets).toEqual([server.url.host]);
  expect(config.static_configs[0].labels.node_name).toBe("ch-test");
});
for (const missing of ["all", "CLICKHOUSE_ORG_ID", "CLICKHOUSE_KEY_ID", "CLICKHOUSE_KEY_SECRET"]) {
  test(`add without ${missing} returns guidance without requests or writes`, async () => {
    const credentials: Record<string, string> = { CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: keyId, CLICKHOUSE_KEY_SECRET: keySecret };
    if (missing === "all") for (const key of Object.keys(credentials)) delete credentials[key];
    else delete credentials[missing];
    expect(await add(credentials)).toBe("Host metrics: set CLICKHOUSE_ORG_ID, CLICKHOUSE_KEY_ID and CLICKHOUSE_KEY_SECRET and re-run to collect CPU, memory, disk and I/O from ClickHouse Cloud");
    expect(requests).toEqual([]);
    expectNothingWritten();
  });
}
for (const denied of [401, 403]) {
  test(`add HTTP ${denied} throws with the exact error and writes nothing`, async () => {
    status = denied;
    await expect(add()).rejects.toEqual(new Error(denied === 401
      ? "ClickHouse Cloud rejected the API key (401). Check the key id and secret."
      : `The API key cannot read Postgres services in organization ${orgId} (403). Give it read access to this organization.`));
    expectNothingWritten();
  });
}
test("add without a matching hostname throws and writes nothing", async () => {
  match = false;
  await expect(add()).rejects.toEqual(new Error(`No ClickHouse Managed Postgres service in organization ${orgId} has hostname ${hostname}.`));
  expectNothingWritten();
});
test.each(["creating", "stopped", "unknown"])("add with service %s throws and writes nothing", async (value) => {
  state = value;
  await expect(add()).rejects.toEqual(new Error(`ClickHouse Managed Postgres service is ${state}, not running. Start it in the ClickHouse Cloud console, then retry.`));
  expectNothingWritten();
});
test.each(["both", "config", "secret", "neither"])("remove cleans up optional host files (%s present)", async (present) => {
  mkdirSync(`${dir}/host-metrics`);
  for (const [kind, extension] of [["config", "yml"], ["secret", "secret"]]) {
    if (present === "both" || present === kind) writeFileSync(`${dir}/host-metrics/clickhouse-ch-test.${extension}`, "fixture");
  }
  writeFileSync(`${dir}/host-metrics/clickhouse-other.secret`, "keep");
  removeHostMetrics(dir, "ch-test");
  expect(readdirSync(`${dir}/host-metrics`)).toEqual(["clickhouse-other.secret"]);
  expect(requests).toEqual([]);
});
