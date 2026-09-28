import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";

const vmBin = process.env.PGAI_TEST_VM_BIN;
let server: ReturnType<typeof Bun.serve> | undefined;
let vm: ReturnType<typeof Bun.spawn> | undefined;
let dir: string | undefined;
afterAll(async () => {
  if (vm) { vm.kill(); await vm.exited; }
  server?.stop(true);
  if (dir) rmSync(dir, { recursive: true, force: true });
});

test.skipIf(!vmBin)("VictoriaMetrics scrapes eight CPU modes, rejects bad auth, and drops non-Postgres metrics", async () => {
  const { renderScrapeConfig } = await import("../lib/clickhouse");
  const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
  const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
  const keyId = "test-key-id";
  const keySecret = "fixture-only-secret";
  const fixture = readFileSync(`${import.meta.dir}/fixtures/clickhouse-prometheus.txt`, "utf8");
  const scrapePath = `/v1/organizations/${orgId}/postgres/${serviceId}/prometheus`;
  let accepted = 0;
  let rejected = 0;
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    if (new URL(request.url).pathname !== scrapePath) return new Response("Not found", { status: 404 });
    if (request.headers.get("authorization") !== `Basic ${btoa(`${keyId}:${keySecret}`)}`) {
      rejected++;
      return new Response("Unauthorized", { status: 401 });
    }
    accepted++;
    return new Response(fixture, { headers: { "Content-Type": "text/plain; version=0.0.4" } });
  } });
  dir = mkdtempSync(`${tmpdir()}/clickhouse-vm-`);
  mkdirSync(`${dir}/scrapes`);
  for (const [name, secret] of [["ch-good", keySecret], ["ch-bad", "wrong-fixture-secret"]]) {
    const passwordFile = `${dir}/${name}.secret`;
    writeFileSync(passwordFile, secret, { mode: 0o600 });
    const text = renderScrapeConfig({ name, cluster: "default", orgId, serviceId, keyId, passwordFile, apiUrl: server.url.origin });
    expect((Bun.YAML.parse(text) as any[])[0].scrape_interval).toBe("60s");
    writeFileSync(`${dir}/scrapes/${name}.yml`, text);
  }
  const main = `${dir}/prometheus.yml`;
  writeFileSync(main, `scrape_config_files:\n  - '${dir}/scrapes/*.yml'\n`);
  const reservation = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() });
  const vmUrl = reservation.url.origin;
  const listenAddr = reservation.url.host;
  reservation.stop(true);
  const child = Bun.spawn([vmBin!, `-promscrape.config=${main}`, `-httpListenAddr=${listenAddr}`, `-storageDataPath=${dir}/data`, "-promscrape.configCheckInterval=1s", "-search.latencyOffset=0s"], { stdout: "pipe", stderr: "pipe" });
  vm = child;
  const vmOutput = Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text()]).then((parts) => parts.join("\n"));
  async function query(expression: string): Promise<Array<{ metric: Record<string, string>; value: [number, string] }>> {
    const response = await fetch(`${vmUrl}/api/v1/query?query=${encodeURIComponent(expression)}`);
    expect(response.ok).toBe(true);
    const body = await response.json() as any;
    expect(body.status).toBe("success");
    return body.data.result;
  }
  let ready = false;
  const deadline = Date.now() + 100_000;
  while (Date.now() < deadline) {
    if (vm.exitCode !== null) throw new Error(`VictoriaMetrics exited ${vm.exitCode}: ${await vmOutput}`);
    try {
      const body = await (await fetch(`${vmUrl}/api/v1/targets`)).json() as any;
      ready = accepted > 0 && rejected > 0 && body.data?.activeTargets?.length === 2
        && (await query('up{node_name="ch-good"}'))[0]?.value[1] === "1"
        && (await query('up{node_name="ch-bad"}'))[0]?.value[1] === "0"
        && (await query('PostgresServer_CPUSeconds_Total{node_name="ch-good",cluster="default"}')).length === 8;
      if (ready) break;
    } catch {
      // The HTTP listener and ingested samples become available asynchronously.
    }
    await Bun.sleep(250);
  }
  expect(ready).toBe(true);
  const cpu = await query('PostgresServer_CPUSeconds_Total{node_name="ch-good",cluster="default"}');
  expect(cpu).toHaveLength(8);
  expect(cpu.map(({ metric }) => metric.mode).sort()).toEqual(["user", "system", "iowait", "softirq", "steal", "irq", "nice", "idle"].sort());
  for (const { metric } of cpu) {
    expect(metric.clickhouse_org).toBe(orgId);
    expect(metric.postgres_service).toBe(serviceId);
    expect(metric.postgres_service_name).toBe("my-postgres");
  }
  expect((await query('up{node_name="ch-good"}')).map(({ value }) => value[1])).toEqual(["1"]);
  expect((await query('up{node_name="ch-bad"}')).map(({ value }) => value[1])).toEqual(["0"]);
  const badSeries = await query('{node_name="ch-bad"}');
  expect(badSeries.length).toBeGreaterThan(0);
  for (const { metric } of badSeries) expect(metric.__name__).toMatch(/^(up|scrape_.*)$/);
  expect(await query("go_goroutines")).toEqual([]);
}, 120_000);
