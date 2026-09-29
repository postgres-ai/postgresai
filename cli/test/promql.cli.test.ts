import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

// Async spawn, not spawnSync: the fake API below is an in-process Bun.serve,
// and spawnSync would block the event loop that has to answer the CLI.
async function runCli(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [status, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { status, stdout, stderr };
}

function isolatedEnv() {
  const cfgHome = mkdtempSync(resolve(tmpdir(), "promql-cli-test-"));
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome };
}

const ONE = "01a0d3bc-629f-75a5-99e1-a3f34d01d813";
const TWO_A = "01a0d3bc-0000-7000-8000-00000000000a";
const TWO_B = "01a0d3bc-0000-7000-8000-00000000000b";

function startFakeApi() {
  const requests: { fn: string; body: Record<string, unknown>; org: string | null }[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const fn = new URL(req.url).pathname.split("/rpc/")[1] ?? "";
      const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;
      requests.push({ fn, body, org: req.headers.get("x-pgai-org") });
      if (fn === "projects_list") {
        return Response.json([
          { project_id: 1, alias: "one-box", name: "one-box", monitoring_instance_ids: [ONE] },
          { project_id: 2, alias: "two-boxes", name: "two-boxes", monitoring_instance_ids: [TWO_A, TWO_B] },
        ]);
      }
      if (fn === "instance_query_enqueue") {
        return Response.json({ job_id: "job-1", first_answer_estimate_s: 1 });
      }
      if (fn === "instance_query_result") {
        return Response.json({
          status: "done",
          outcome: "ok",
          result: { resultType: "vector", result: [{ metric: { __name__: "up" }, value: [1, "1"] }], stats: { truncated: false } },
          error: null,
          failure_class: null,
          started_at: null,
          finished_at: null,
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return { requests, baseUrl: `http://localhost:${server.port}`, stop: () => server.stop(true) };
}

describe("pgai promql targeting", () => {
  test("--project with exactly one instance queries that instance", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--project", "one-box"],
        isolatedEnv(),
      );
      expect(r.status).toBe(0);
      const enqueue = api.requests.find((x) => x.fn === "instance_query_enqueue");
      expect(enqueue?.body.p_instance_id).toBe(ONE);
    } finally {
      api.stop();
    }
  });

  test("--project with several instances refuses and queues nothing", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--project", "two-boxes"],
        isolatedEnv(),
      );
      expect(r.status).toBe(1);
      expect(r.stderr).toContain(`${TWO_A}, ${TWO_B}`);
      expect(api.requests.map((x) => x.fn)).toEqual(["projects_list"]);
    } finally {
      api.stop();
    }
  });

  test("--instance alone never lists projects", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--instance", ONE],
        isolatedEnv(),
      );
      expect(r.status).toBe(0);
      expect(api.requests.map((x) => x.fn)).not.toContain("projects_list");
      expect(api.requests.find((x) => x.fn === "instance_query_enqueue")?.body.p_instance_id).toBe(ONE);
    } finally {
      api.stop();
    }
  });

  test("--org scopes both the project lookup and the query", async () => {
    // A global token carries no org of its own: the selector must reach the
    // listing as a header, or --project could resolve in the wrong org.
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--org", "acme", "--project", "one-box"],
        isolatedEnv(),
      );
      expect(r.status).toBe(0);
      for (const fn of ["projects_list", "instance_query_enqueue"]) {
        expect(api.requests.find((x) => x.fn === fn)?.org).toBe("acme");
      }
    } finally {
      api.stop();
    }
  });

  test("--project and --instance together are refused before any request", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--project", "one-box", "--instance", ONE],
        isolatedEnv(),
      );
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Pass --instance or --project, not both.");
      expect(api.requests).toEqual([]);
    } finally {
      api.stop();
    }
  });
});
