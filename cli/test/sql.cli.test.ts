import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";

// Async spawn, not spawnSync: the fake API below is an in-process Bun.serve,
// and spawnSync would block the event loop that has to answer the CLI.
async function runCli(args: string[], env: Record<string, string>, cwd?: string) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    cwd,
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
  const cfgHome = mkdtempSync(resolve(tmpdir(), "sql-cli-test-"));
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome };
}

const ONE = "01a0d3bc-629f-75a5-99e1-a3f34d01d813";
const TWO_A = "01a0d3bc-0000-7000-8000-00000000000a";
const TWO_B = "01a0d3bc-0000-7000-8000-00000000000b";

type JobResult = Record<string, unknown>;
const OK_RESULT: JobResult = {
  status: "done",
  outcome: "ok",
  result: { columns: ["x"], types: ["Int32"], rows: [[1]], row_count: 1, truncated: false, elapsed_ms: 1 },
  error: null,
  failure_class: null,
  started_at: null,
  finished_at: null,
};

// The store's own bytes: JSON.parse would round both 64-bit ids. Everything
// else in it must come out exactly as before.
const BIG_ROWS =
  '{"columns":["queryid","n","f","e","s","labels"],"types":["BIGINT","BIGINT","DOUBLE","DOUBLE","VARCHAR","MAP"],' +
  '"rows":[[-8580166607628940136,9223372036854775806,0.25,1e-07,"12345678901234567890",{"job":"pg","n":3}],' +
  '[1,42,null,true,"x",{}]],"row_count":2,"truncated":false,"elapsed_ms":3}';
function expectExactIds(table: string, json: string) {
  expect(table).toContain('-8580166607628940136  9223372036854775806  0.25  1e-7  12345678901234567890  {"job":"pg","n":3}');
  expect(table).toContain("2 rows");
  // Bare numbers with every digit, and otherwise what JSON.stringify prints.
  expect(json).toContain("\n      -8580166607628940136,\n      9223372036854775806,\n      0.25,\n      1e-7,\n");
  const safe = JSON.parse(BIG_ROWS);
  safe.rows[0][0] = safe.rows[0][1] = "ID";
  expect(json.replace("-8580166607628940136", '"ID"').replace("9223372036854775806", '"ID"')).toBe(
    JSON.stringify(safe, null, 2) + "\n",
  );
}

function startFakeApi(jobResult: JobResult | string = OK_RESULT) {
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
        return typeof jobResult === "string" ? new Response(jobResult) : Response.json(jobResult);
      }
      return new Response("not found", { status: 404 });
    },
  });
  return { requests, baseUrl: `http://localhost:${server.port}`, stop: () => server.stop(true) };
}

describe("pgai mon sql targeting", () => {
  test("--project with exactly one instance queries that instance", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--project", "one-box"],
        isolatedEnv(),
      );
      expect(r.status).toBe(0);
      const enqueue = api.requests.find((x) => x.fn === "instance_query_enqueue");
      expect(enqueue?.body).toEqual({ p_instance_id: ONE, p_kind: "sql", p_query: "select 1" });
      expect(r.stdout).toContain("1 row");
    } finally {
      api.stop();
    }
  });

  test("--project with several instances refuses and queues nothing", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--project", "two-boxes"],
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
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--instance", ONE],
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
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--org", "acme", "--project", "one-box"],
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
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--project", "one-box", "--instance", ONE],
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

describe("pgai mon sql output", () => {
  test("--json prints the raw result payload on stdout", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--instance", ONE, "--json"],
        isolatedEnv(),
      );
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toEqual(OK_RESULT.result);
    } finally {
      api.stop();
    }
  });

  test("64-bit integers keep every digit from the platform's reply to the table and --json", async () => {
    const api = startFakeApi(`{"status":"done","outcome":"ok","result":${BIG_ROWS},"error":null,"failure_class":null}`);
    try {
      const args = ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select 1", "--instance", ONE];
      const table = await runCli(args, isolatedEnv());
      const json = await runCli([...args, "--json"], isolatedEnv());
      expect([table.status, json.status]).toEqual([0, 0]);
      expectExactIds(table.stdout, json.stdout);
    } finally {
      api.stop();
    }
  });

  test("a failed query exits non-zero with the box's error verbatim", async () => {
    const api = startFakeApi({
      ...OK_RESULT,
      outcome: "error",
      result: null,
      error: "Catalog Error: Table with name nope does not exist!",
      failure_class: "query",
    });
    try {
      const r = await runCli(
        ["--api-key", "k", "--api-base-url", api.baseUrl, "mon", "sql", "select * from nope", "--instance", ONE],
        isolatedEnv(),
      );
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Query done (query).");
      expect(r.stderr).toContain("Table with name nope does not exist");
      expect(r.stdout).toBe("");
    } finally {
      api.stop();
    }
  });

  test("the old top-level 'pgai sql' is not a command", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(["--api-key", "k", "--api-base-url", api.baseUrl, "sql", "select 1"], isolatedEnv());
      expect(r.status).not.toBe(0);
      expect(api.requests).toEqual([]);
    } finally {
      api.stop();
    }
  });
});

describe("pgai promql", () => {
  test("is gone: exits non-zero with a one-line pointer to pgai mon sql, sends nothing", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(["--api-key", "k", "--api-base-url", api.baseUrl, "promql", "up", "--range"], isolatedEnv());
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("pgai mon sql");
      expect(r.stderr.trim().split("\n").length).toBe(1);
      expect(api.requests).toEqual([]);
    } finally {
      api.stop();
    }
  });
});

// --local: straight to this box's pgai-mon with the .env token (F5). A fake
// pgai-mon on 127.0.0.1 stands in for the real one through PGAI_MON_URL.
describe("pgai mon sql --local", () => {
  const TOKEN = "tok-local-secret-1234";
  function project(env = `PGAI_TAG=1.0.0\nPGAI_MON_TOKEN=${TOKEN}\n`) {
    const dir = mkdtempSync(resolve(tmpdir(), "sql-local-"));
    writeFileSync(resolve(dir, "docker-compose.yml"), "services: {}\n");
    if (env) writeFileSync(resolve(dir, ".env"), env);
    return { ...isolatedEnv(), PGAI_PROJECT_DIR: dir, PGAI_API_KEY: "" };
  }
  function startFakeMon(reply: (body: string) => Response) {
    const seen: { auth: string | null; path: string; body: string }[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      async fetch(req) {
        const body = await req.text();
        seen.push({ auth: req.headers.get("authorization"), path: new URL(req.url).pathname, body });
        return reply(body);
      },
    });
    return { seen, url: `http://127.0.0.1:${server.port}`, stop: () => server.stop(true) };
  }
  // Run from inside the project, as on the box.
  const runLocal = (args: string[], env: Record<string, string>, extra: Record<string, string> = {}) =>
    runCli(args, { ...env, ...extra }, env.PGAI_PROJECT_DIR);
  const ROWS = { columns: ["metric", "n"], types: ["VARCHAR", "BIGINT"], rows: [["pgwatch_db_size_size_b", 12]], row_count: 1, truncated: false, elapsed_ms: 3 };

  test("posts the query with the .env token and prints the platform path's table; no API key needed", async () => {
    const mon = startFakeMon(() => Response.json(ROWS));
    try {
      const r = await runLocal(["mon", "sql", "--local", "select metric, count(*) n from series group by 1"], project(), { PGAI_MON_URL: mon.url });
      expect(r.status).toBe(0);
      expect(mon.seen).toEqual([{ auth: `Bearer ${TOKEN}`, path: "/sql", body: "select metric, count(*) n from series group by 1" }]);
      expect(r.stdout).toContain("pgwatch_db_size_size_b  12");
      expect(r.stdout).toContain("1 row");
      expect(r.stdout + r.stderr).not.toContain(TOKEN);
      const json = await runLocal(["mon", "sql", "--local", "--json", "select 1"], project(), { PGAI_MON_URL: mon.url });
      expect(JSON.parse(json.stdout)).toEqual(ROWS);
    } finally {
      mon.stop();
    }
  });

  test("64-bit integers keep every digit in the table and in --json", async () => {
    const mon = startFakeMon(() => new Response(BIG_ROWS, { headers: { "Content-Type": "application/json" } }));
    try {
      const table = await runLocal(["mon", "sql", "--local", "select 1"], project(), { PGAI_MON_URL: mon.url });
      const json = await runLocal(["mon", "sql", "--local", "--json", "select 1"], project(), { PGAI_MON_URL: mon.url });
      expect([table.status, json.status]).toEqual([0, 0]);
      expectExactIds(table.stdout, json.stdout);
    } finally {
      mon.stop();
    }
  });

  test("the store's error message is shown; a rejected token says so; neither prints the token", async () => {
    const mon = startFakeMon((body) =>
      body === "bad"
        ? Response.json({ error: { kind: "parse_error", message: "syntax error at or near bad" } }, { status: 400 })
        : new Response("", { status: 401 }),
    );
    try {
      const bad = await runLocal(["mon", "sql", "--local", "bad"], project(), { PGAI_MON_URL: mon.url });
      expect(bad.status).toBe(1);
      expect(bad.stderr).toContain("Query failed (HTTP 400): parse_error: syntax error at or near bad");
      const denied = await runLocal(["mon", "sql", "--local", "select 1"], project(), { PGAI_MON_URL: mon.url });
      expect(denied.status).toBe(1);
      expect(denied.stderr).toContain("pgai-mon rejected the token (HTTP 401)");
      expect(bad.stderr + denied.stderr + bad.stdout + denied.stdout).not.toContain(TOKEN);
    } finally {
      mon.stop();
    }
  });

  test("a refused query (HTTP 403) shows pgai-mon's reason, not a token problem", async () => {
    const mon = startFakeMon(() =>
      Response.json({ error: { kind: "not_read_only", message: "only one read-only statement is allowed" } }, { status: 403 }),
    );
    try {
      const r = await runLocal(["mon", "sql", "--local", "show tables"], project(), { PGAI_MON_URL: mon.url });
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Query refused (HTTP 403): not_read_only: only one read-only statement is allowed");
      expect(r.stderr).not.toContain("rejected the token");
      expect(r.stderr).not.toContain("PGAI_MON_TOKEN");
      expect(r.stderr + r.stdout).not.toContain(TOKEN);
    } finally {
      mon.stop();
    }
  });

  test("without a token in .env it stops before any request", async () => {
    const mon = startFakeMon(() => Response.json(ROWS));
    try {
      const r = await runLocal(["mon", "sql", "--local", "select 1"], project("PGAI_TAG=1.0.0\n"), { PGAI_MON_URL: mon.url });
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("No PGAI_MON_TOKEN in");
      expect(mon.seen).toEqual([]);
    } finally {
      mon.stop();
    }
  });

  test("--local takes no --project; without --local and without an API key the error points to --local", async () => {
    const both = await runLocal(["mon", "sql", "--local", "--project", "x", "select 1"], project());
    expect(both.status).toBe(1);
    expect(both.stderr).toContain("takes no --instance or --project");
    const noKey = await runLocal(["mon", "sql", "select 1"], project());
    expect(noKey.status).toBe(1);
    expect(noKey.stderr).toContain("API key is required");
    expect(noKey.stderr).toContain("pgai mon sql --local");
  });
});
