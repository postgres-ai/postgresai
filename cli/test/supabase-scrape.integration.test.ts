import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";

// The real VictoriaMetrics binary loads the scrape file `mon targets add` writes
// for Supabase, scrapes a stand-in for the instance-jobs relay, keeps only the
// node_* families the host_* rules read, and labels them with the target's
// cluster/node_name. CI: cli:host-metrics-scrape:tests.
const vmBin = process.env.PGAI_TEST_VM_BIN;
let relay: ReturnType<typeof Bun.serve> | undefined;
let vm: ReturnType<typeof Bun.spawn> | undefined;
let dir: string | undefined;
afterAll(async () => {
  if (vm) { vm.kill(); await vm.exited; }
  relay?.stop(true);
  if (dir) rmSync(dir, { recursive: true, force: true });
});

test.skipIf(!vmBin)("VictoriaMetrics scrapes the Supabase relay with the target's labels", async () => {
  const { renderSupabaseScrapeConfig } = await import("../lib/host-metrics");
  const exposition = readFileSync(`${import.meta.dir}/../../instance-jobs/internal/supabase/testdata/supabase_metrics.prom`, "utf8")
    + "go_goroutines 7\n";
  relay = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    if (new URL(request.url).pathname !== "/supabase/metrics") return new Response("Not found", { status: 404 });
    return new Response(exposition, { headers: { "Content-Type": "text/plain; version=0.0.4" } });
  } });
  dir = mkdtempSync(`${tmpdir()}/supabase-vm-`);
  mkdirSync(`${dir}/scrapes`);
  writeFileSync(`${dir}/prometheus.yml`, `scrape_config_files:\n  - '${dir}/scrapes/*.yml'\n`);
  writeFileSync(`${dir}/scrapes/supabase-main.yml`, renderSupabaseScrapeConfig({ cluster: "prod", nodeName: "main-db", target: relay.url.host }));
  const reservation = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response() });
  const vmUrl = reservation.url.origin;
  const listenAddr = reservation.url.host;
  reservation.stop(true);
  vm = Bun.spawn([vmBin!, `-promscrape.config=${dir}/prometheus.yml`, `-httpListenAddr=${listenAddr}`, `-storageDataPath=${dir}/data`, "-search.latencyOffset=0s"], { stdout: "ignore", stderr: "ignore" });
  async function query(expression: string): Promise<Array<{ metric: Record<string, string> }>> {
    const body = await (await fetch(`${vmUrl}/api/v1/query?query=${encodeURIComponent(expression)}`)).json() as any;
    return body.data.result;
  }
  let load: Array<{ metric: Record<string, string> }> = [];
  const deadline = Date.now() + 100_000;
  while (Date.now() < deadline && load.length === 0) {
    try {
      load = await query("node_load1");
    } catch {
      // VictoriaMetrics is still starting.
    }
    if (load.length === 0) await Bun.sleep(250);
  }
  expect(load).toHaveLength(1);
  expect(load[0].metric).toMatchObject({ job: "supabase-host-metrics", cluster: "prod", node_name: "main-db", supabase_project_ref: "abcdefghijklmnopqrst" });
  expect(load[0].metric).not.toHaveProperty("__pgai_rev");
  expect((await query('{__name__=~"node_.*", node_name!="main-db"}'))).toEqual([]);
  expect(await query("go_goroutines")).toEqual([]);
}, 120_000);
