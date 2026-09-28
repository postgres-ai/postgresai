import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// No Docker bypass exists in the CLI. These exercise the exported command
// handlers with only compose apply disabled; API requests and file writes are real.
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
const hostname = "my-postgres.us-east-1.aws.pg.clickhouse.cloud";
const conn = `postgresql://monitor:fixture-password@${hostname}:5432/postgres`;
const keyId = "test-key-id";
const keySecret = "fixture-only-secret";
const listPath = `/v1/organizations/${orgId}/postgres`;
const cliPath = resolve(import.meta.dir, "../bin/postgres-ai.ts");
let dir: string;
let server: ReturnType<typeof Bun.serve>;
let status: number;
let state: string;
let match: boolean;
let requests: string[];
const initial = "[]\n";

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/clickhouse-targets-`);
  writeFileSync(`${dir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${dir}/instances.yml`, initial);
  status = 200;
  state = "running";
  match = true;
  requests = [];
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const path = new URL(request.url).pathname;
    requests.push(path);
    if (request.headers.get("authorization") !== `Basic ${btoa(`${keyId}:${keySecret}`)}`) return new Response("Unauthorized", { status: 401 });
    if (status !== 200) return new Response("Denied", { status });
    if (path === listPath) return Response.json({ result: [{ id: serviceId, name: "my-postgres", state }], status: 200 });
    if (path === `${listPath}/${serviceId}`) return Response.json({ result: { id: serviceId, name: "my-postgres", state, hostname: match ? hostname : "other.pg.clickhouse.cloud" } });
    return new Response("Not found", { status: 404 });
  } });
});
afterEach(() => {
  server.stop(true);
  rmSync(dir, { recursive: true, force: true });
});

async function command(action: "add" | "remove", credentials: Record<string, string> = {
  CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: keyId, CLICKHOUSE_KEY_SECRET: keySecret,
}) {
  const handler = action === "add" ? "addMonitoringTarget" : "removeMonitoringTarget";
  const cli = await import("../bin/postgres-ai");
  expect(cli).toHaveProperty(handler);
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith("CLICKHOUSE_")) delete env[key];
  const args = action === "add" ? [conn, "ch-test", { apply: false }] : ["ch-test", { apply: false }];
  const script = `import { ${handler} } from ${JSON.stringify(cliPath)}; await ${handler}(...${JSON.stringify(args)});`;
  const child = Bun.spawn([process.execPath, "-e", script], {
    cwd: dir,
    env: { ...env, ...credentials, CLICKHOUSE_API_URL: server.url.origin, PGAI_PROJECT_DIR: dir, DO_NOT_TRACK: "1" },
    stdout: "pipe", stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect(stdout + stderr).not.toContain(keySecret);
  return { stdout, stderr, code };
}

function expectNothingWritten() {
  expect(readFileSync(`${dir}/instances.yml`, "utf8")).toBe(initial);
  expect(existsSync(`${dir}/host-metrics`) ? readdirSync(`${dir}/host-metrics`) : []).toEqual([]);
}

test("add discovers the service and writes config 0644 and secret 0600 without a newline", async () => {
  const result = await command("add");
  expect(result.code).toBe(0);
  expect(result.stdout.split("\n")).toContain("Host metrics: ClickHouse Cloud Prometheus endpoint for service my-postgres (scraped every 60s)");
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
  expect(config.scrape_interval).toBe("60s");
  expect(config.scrape_timeout).toBe("30s");
  expect(config.basic_auth).toEqual({ username: keyId, password_file: "/etc/pgai/host-metrics/clickhouse-ch-test.secret" });
  expect(config.static_configs).toEqual([{ targets: [server.url.host], labels: { cluster: "default", node_name: "ch-test" } }]);
  expect(config.metric_relabel_configs).toEqual([{ source_labels: ["__name__"], regex: "PostgresServiceInfo|PostgresServer_.*", action: "keep" }]);
  expect(readFileSync(`${dir}/instances.yml`, "utf8")).toContain("ch-test");
});

for (const missing of ["all", "CLICKHOUSE_ORG_ID", "CLICKHOUSE_KEY_ID", "CLICKHOUSE_KEY_SECRET"]) {
  test(`add without ${missing} prints guidance and still adds the Postgres target`, async () => {
    const credentials: Record<string, string> = { CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: keyId, CLICKHOUSE_KEY_SECRET: keySecret };
    if (missing === "all") for (const key of Object.keys(credentials)) delete credentials[key];
    else delete credentials[missing];
    const result = await command("add", credentials);
    expect(result.code).toBe(0);
    expect(result.stdout.split("\n")).toContain("Host metrics: set CLICKHOUSE_ORG_ID, CLICKHOUSE_KEY_ID and CLICKHOUSE_KEY_SECRET and re-run to collect CPU, memory, disk and I/O from ClickHouse Cloud");
    expect(readFileSync(`${dir}/instances.yml`, "utf8")).toContain("ch-test");
    expect(requests).toEqual([]);
    expect(existsSync(`${dir}/host-metrics`) ? readdirSync(`${dir}/host-metrics`) : []).toEqual([]);
  });
}

for (const denied of [401, 403]) {
  test(`add HTTP ${denied} exits 1 with the exact error and writes nothing`, async () => {
    status = denied;
    const result = await command("add");
    expect(result.code).toBe(1);
    expect(result.stderr.trim()).toBe(denied === 401
      ? "ClickHouse Cloud rejected the API key (401). Check the key id and secret."
      : `The API key cannot read Postgres services in organization ${orgId} (403). Give it read access to this organization.`);
    expectNothingWritten();
  });
}
test("add without a matching hostname exits 1 and writes nothing", async () => {
  match = false;
  const result = await command("add");
  expect(result.code).toBe(1);
  expect(result.stderr.trim()).toBe(`No ClickHouse Managed Postgres service in organization ${orgId} has hostname ${hostname}.`);
  expectNothingWritten();
});
test.each(["creating", "stopped"])("add with service %s exits 1 and writes nothing", async (value) => {
  state = value;
  const result = await command("add");
  expect(result.code).toBe(1);
  expect(result.stderr.trim()).toBe(`ClickHouse Managed Postgres service is ${state}, not running. Start it in the ClickHouse Cloud console, then retry.`);
  expectNothingWritten();
});
test.each(["both", "config", "secret", "neither"])("remove cleans up optional host files (%s present)", async (present) => {
  writeFileSync(`${dir}/instances.yml`, `- name: ch-test\n  conn_str: ${conn}\n  is_enabled: true\n`);
  mkdirSync(`${dir}/host-metrics`);
  for (const [kind, extension] of [["config", "yml"], ["secret", "secret"]]) {
    if (present === "both" || present === kind) writeFileSync(`${dir}/host-metrics/clickhouse-ch-test.${extension}`, "fixture");
  }
  writeFileSync(`${dir}/host-metrics/clickhouse-other.secret`, "keep");
  const result = await command("remove");
  expect(result.code).toBe(0);
  expect(readdirSync(`${dir}/host-metrics`)).toEqual(["clickhouse-other.secret"]);
  expect(readFileSync(`${dir}/instances.yml`, "utf8")).not.toContain("ch-test");
  expect(requests).toEqual([]);
});
