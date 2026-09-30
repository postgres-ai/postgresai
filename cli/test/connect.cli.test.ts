import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";

// The commands as a user or an agent runs them: stdout not a TTY (so JSON),
// a fake platform API, a throwaway HOME. Frozen: the JSON contract
// (status, next, ...) and the exit codes 0 / 1 / 3.

const CH = "postgresql://postgres:adminpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const NAME = "abc123.us-east-1.aws.pg.clickhouse.cloud/postgres";
const ROW = { id: "i-1", name: NAME, provider: "clickhouse", mode: "cloud", status: "active", dashboard_url: "https://abc.pgai.watch", host_metrics: true, created_at: "2026-09-30T00:00:00Z" };

async function run(args: string[], env: Record<string, string>) {
  const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-"));
  const proc = Bun.spawn([process.execPath, resolve(import.meta.dir, "..", "bin", "postgres-ai.ts"), ...args], {
    env: { ...process.env, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "", ...env },
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
  });
  const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  return { status, stdout, stderr, json: () => JSON.parse(stdout) };
}

async function withApi(fn: (env: Record<string, string>, calls: string[]) => Promise<void>) {
  const calls: string[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1", port: 0,
    async fetch(req) {
      const path = new URL(req.url).pathname;
      calls.push(`${path} ${req.headers.get("access-token")} ${await req.text()}`);
      if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([ROW]);
      if (path.endsWith("/rpc/cloud_monitoring_disconnect")) return Response.json({ id: "i-1", status: "deleting_launched" });
      return new Response("not found", { status: 404 });
    },
  });
  try {
    await fn({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}` }, calls);
  } finally {
    server.stop(true);
  }
}

describe("pgai connect / databases / status / disconnect", () => {
  test("not signed in and not interactive: action required, exit 3", async () => {
    const r = await run(["connect", CH], {});
    expect(r.status).toBe(3);
    expect(r.json()).toEqual({ status: "action_required", provider: "clickhouse", name: NAME, next: "Sign in: pgai auth login (agents: set PGAI_API_KEY), then re-run" });
    expect(r.stdout + r.stderr).not.toContain("adminpw");
  });

  test("not a URL: failed, exit 1", async () => {
    const r = await run(["connect", "host=db dbname=app"], {});
    expect(r.status).toBe(1);
    expect(r.json().next).toBe("Pass a URL: pgai connect postgresql://user:password@host:5432/dbname");
  });

  test("already connected: the status, exit 0, no database touched", async () => {
    await withApi(async (env, calls) => {
      const r = await run(["connect", CH], env);
      expect(r.status).toBe(0);
      expect(r.json()).toEqual({ status: "connected", provider: "clickhouse", name: NAME, id: "i-1", dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch" });
      expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}"]);
      expect(r.stdout + r.stderr).not.toContain("adminpw");
    });
  });

  test("databases and status", async () => {
    await withApi(async (env) => {
      expect((await run(["databases"], env)).json()).toEqual([ROW]);
      const s = await run(["status", NAME], env);
      expect(s.json()).toEqual([{ status: "connected", provider: "clickhouse", name: NAME, id: "i-1", dashboard_url: "https://abc.pgai.watch", host_metrics: true, next: "Open https://abc.pgai.watch" }]);
      const missing = await run(["status", "nope"], env);
      expect(missing.status).toBe(1);
      expect(missing.stderr).toContain("No database named nope. See: pgai databases");
    });
  });

  test("disconnect needs --yes when not interactive", async () => {
    await withApi(async (env, calls) => {
      const asked = await run(["disconnect", NAME], env);
      expect(asked.status).toBe(3);
      expect(asked.json().next).toBe(`pgai disconnect ${NAME} --yes`);
      expect(calls.some((c) => c.includes("disconnect"))).toBe(false);

      const done = await run(["disconnect", NAME, "--yes"], env);
      expect(done.status).toBe(0);
      expect(done.json()).toEqual({ status: "disconnected", name: NAME, id: "i-1", next: "Delete the ClickHouse Cloud API key you gave us" });
      expect(calls.at(-1)).toBe('/rpc/cloud_monitoring_disconnect test-key {"instance_id":"i-1"}');
    });
  });

  test("an unusable --wait is refused before anything is touched", async () => {
    const r = await run(["connect", CH, "--wait", "soon"], { PGAI_API_KEY: "k", PGAI_API_BASE_URL: "http://127.0.0.1:9" });
    expect(r.status).toBe(1);
    expect(r.json().next).toBe("--wait must be a number of minutes (0 = do not wait)");
  });

  test("init without a terminal points to pgai connect", async () => {
    const r = await run(["init"], {});
    expect(r.status).toBe(3);
    expect(r.json().next).toBe("pgai connect <database-url>");
  });
});
