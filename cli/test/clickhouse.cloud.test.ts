import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { randomBytes } from "crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { addHostMetrics } from "../lib/clickhouse";

const env = process.env;
const orgId = env.CLICKHOUSE_TEST_ORG_ID;
const keyId = env.CLICKHOUSE_TEST_KEY_ID;
const keySecret = env.CLICKHOUSE_TEST_KEY_SECRET;
const api = "https://api.clickhouse.cloud";
const auth = `Basic ${btoa(`${keyId}:${keySecret}`)}`;
const run = `pgai-ci-${Date.now().toString(36)}`;
const minutes = (n: number) => n * 60_000;

async function call(method: string, path: string, body?: unknown) {
  const res = await fetch(`${api}/v1/organizations/${orgId}/postgres${path}`, {
    method,
    headers: { authorization: auth, ...(body ? { "content-type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30_000),
  });
  const text = await res.text();
  return { status: res.status, text, json: text.startsWith("{") ? JSON.parse(text) : null };
}

async function until<T>(what: string, ms: number, probe: () => Promise<T | undefined>): Promise<T> {
  const deadline = Date.now() + ms;
  for (;;) {
    const value = await probe();
    if (value !== undefined) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await Bun.sleep(15_000);
  }
}

function cli(args: string[], extraEnv: Record<string, string> = {}) {
  const r = Bun.spawnSync([process.execPath, "./bin/postgres-ai.ts", ...args], {
    cwd: `${import.meta.dir}/..`,
    env: { ...env, ...extraEnv },
    timeout: minutes(5),
  });
  return { code: r.exitCode, stdout: r.stdout.toString(), stderr: r.stderr.toString() };
}

describe.skipIf(!orgId || !keyId || !keySecret)("ClickHouse Managed Postgres, real cloud", () => {
  let id = "";
  let adminUrl = "";
  let dir = "";

  beforeAll(async () => {
    const created = await call("POST", "", {
      name: run,
      provider: "aws",
      region: env.CLICKHOUSE_TEST_REGION ?? "us-east-1",
      size: env.CLICKHOUSE_TEST_SIZE ?? "r6gd.medium",
      postgresVersion: env.CLICKHOUSE_TEST_PG_VERSION ?? "17",
      tags: [{ key: "pgai-ci", value: run }, { key: "ttl", value: String(Math.floor((Date.now() + minutes(120)) / 1000)) }],
    });
    if (created.status !== 200) throw new Error(`create failed: HTTP ${created.status} ${created.text.slice(0, 300)}`);
    id = created.json.result.id;
    adminUrl = created.json.result.connectionString;
    if (env.CLICKHOUSE_TEST_ID_FILE) writeFileSync(env.CLICKHOUSE_TEST_ID_FILE, `${id}\n`);
    console.log(`created ${run} id=${id} at ${new Date().toISOString()}`);
    await until("state running", minutes(25), async () => {
      const got = await call("GET", `/${id}`);
      return got.json?.result?.state === "running" ? true : undefined;
    });
    console.log(`running at ${new Date().toISOString()}`);
    dir = mkdtempSync(`${tmpdir()}/clickhouse-cloud-`);
  }, minutes(30));

  afterAll(async () => {
    if (dir) rmSync(dir, { recursive: true, force: true });
    if (!id) return;
    await until("delete accepted", minutes(10), async () => {
      const del = await call("DELETE", `/${id}`);
      return del.status === 200 || del.status === 404 ? true : undefined;
    });
    await until("service gone", minutes(15), async () => {
      const got = await call("GET", `/${id}`);
      return got.status === 404 ? true : undefined;
    });
    console.log(`deleted ${run} id=${id} at ${new Date().toISOString()}`);
  }, minutes(30));

  test("prepare-db detects the provider, prints the scope and verifies", () => {
    const password = randomBytes(24).toString("hex");
    const prepared = cli(["prepare-db", adminUrl, "--password", password]);
    expect(prepared.code, prepared.stderr).toBe(0);
    const lines = prepared.stdout.split("\n");
    expect(lines).toContain("Provider: clickhouse (detected from host)");
    expect(lines).toContain("-- scope: role postgres_ai_mon gets pg_monitor, pg_read_all_stats; this admin connection is used for this run only and is not stored");
    const verified = cli(["prepare-db", adminUrl, "--verify"]);
    expect(verified.code, verified.stderr).toBe(0);
  }, minutes(6));

  test("host metrics: service found, endpoint serves the documented series", async () => {
    const line = await addHostMetrics({
      projectDir: dir,
      name: "ch-cloud",
      conn: adminUrl,
      env: { CLICKHOUSE_ORG_ID: orgId!, CLICKHOUSE_KEY_ID: keyId!, CLICKHOUSE_KEY_SECRET: keySecret! },
    });
    expect(line).toBe(`Host metrics: ClickHouse Cloud Prometheus endpoint for service ${run} (scraped every 60s)`);

    const names = (text: string) => new Set(text.split("\n").filter(l => l && !l.startsWith("#")).map(l => l.split(/[{ ]/)[0]));
    const documented = names(readFileSync(`${import.meta.dir}/fixtures/clickhouse-prometheus.txt`, "utf8"));
    documented.delete("go_goroutines");
    const body = await until("CPU series on the Prometheus endpoint", minutes(10), async () => {
      const res = await fetch(`${api}/v1/organizations/${orgId}/postgres/${id}/prometheus`, {
        headers: { authorization: auth }, signal: AbortSignal.timeout(30_000),
      });
      const text = await res.text();
      return res.status === 200 && text.includes("PostgresServer_CPUSeconds_Total") ? text : undefined;
    });
    if (env.CLICKHOUSE_TEST_RECORD_DIR) writeFileSync(`${env.CLICKHOUSE_TEST_RECORD_DIR}/prometheus.txt`, body);
    const live = names(body);
    expect([...documented].filter(n => !live.has(n)).sort()).toEqual([]);
    expect(body).toContain(`postgres_service="${id}"`);
    expect(body).toContain(`postgres_service_name="${run}"`);
  }, minutes(12));
});
