import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

// Sync runner for OFFLINE tests only (help / fail-fast). Never use it for a test
// that round-trips the in-process Bun.serve: spawnSync blocks the JS event loop, so
// the fake server can't answer and the two deadlock (see runCliAsync).
function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

// Async runner for tests that hit the fake server — keeps the event loop free.
async function runCliAsync(args: string[], env: Record<string, string> = {}) {
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

/** Isolate config so tests never touch real user state. */
function isolatedEnv(extra: Record<string, string> = {}) {
  const cfgHome = mkdtempSync(resolve(tmpdir(), "joe-cli-test-"));
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome, ...extra };
}

interface RecordedRequest {
  pathname: string;
  body: Record<string, unknown>;
}

/**
 * Fake PostgREST server for the sync Joe rpcs. Instance 55 ("cold-db") never
 * posts a result (stays `pending`, exercising the resume path); instance 66
 * ("no-role") returns the Joe-API-v2 role-gate PT403; instance 3 completes.
 * Command ids are ABOVE 2^53 so any parseInt round-trip in the CLI corrupts
 * them and the string-preservation assertions fail.
 */
function startFakeApi() {
  const requests: RecordedRequest[] = [];
  const COLD_INSTANCE = 55;
  const FORBIDDEN_INSTANCE = 66;
  const ERROR_INSTANCE = 33;
  const FLAKY_INSTANCE = 44;
  let nextId = 9007199254740993000n; // > Number.MAX_SAFE_INTEGER
  const commands = new Map<string, { command: string; instanceId: number }>();

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const bodyText = await req.text();
      let body: Record<string, unknown> = {};
      try {
        body = bodyText ? (JSON.parse(bodyText) as Record<string, unknown>) : {};
      } catch {
        body = {};
      }
      requests.push({ pathname: url.pathname, body });

      const respond = (obj: unknown, status = 200): Response =>
        new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

      if (req.method === "POST" && url.pathname.endsWith("/rpc/projects_list")) {
        return respond([
          { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: true, instance_id: 3 },
          { project_id: 77, alias: "cold-db", name: "Cold DB", joe_ready: true, tunnel: true, instance_id: COLD_INSTANCE },
          { project_id: 99, alias: "no-role", name: "No Role", joe_ready: true, tunnel: true, instance_id: FORBIDDEN_INSTANCE },
          { project_id: 14, alias: "no-joe", name: "No Joe", joe_ready: false, tunnel: false, instance_id: null },
          { project_id: 15, alias: "err-db", name: "Err DB", joe_ready: true, tunnel: true, instance_id: ERROR_INSTANCE },
          { project_id: 16, alias: "flaky-db", name: "Flaky DB", joe_ready: true, tunnel: true, instance_id: FLAKY_INSTANCE },
        ]);
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/joe_command_run")) {
        const instanceId = Number(body.instance_id);
        if (instanceId === FORBIDDEN_INSTANCE) {
          return respond(
            { code: "PT403", message: "Forbidden", details: "Joe API v2 requires the All Features role." },
            403
          );
        }
        const commandId = String(nextId++);
        commands.set(commandId, { command: String(body.command), instanceId });
        // The rpc returns the id as a bare JSON string.
        return respond(commandId);
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/joe_command_output")) {
        const commandId = String(body.command_id);
        if (commandId === "424242") {
          // Terminal failure without a structured row (e.g. clone init failed).
          return respond({ command_id: commandId, status: "error", created_at: "2026-07-22T10:00:00", error: "ERROR: something failed" });
        }
        if (commandId === "424244") {
          return respond({ command_id: commandId, status: "pending", created_at: "2026-07-22T10:00:00" });
        }
        const cmd = commands.get(commandId);
        if (!cmd) {
          return respond({ code: "PT404", message: "Not found", details: "Specified command not found." }, 404);
        }
        if (cmd.instanceId === COLD_INSTANCE) {
          return respond({ command_id: commandId, status: "pending", created_at: "2026-07-22T10:00:00" });
        }
        if (cmd.instanceId === ERROR_INSTANCE) {
          // Joe posted a terminal failure for this command.
          return respond({
            command_id: commandId,
            status: "error",
            created_at: "2026-07-22T10:00:00",
            queryid: null,
            plan_json: null,
            plan_text: null,
            error: "ERROR: relation \"nope\" does not exist",
          });
        }
        if (cmd.instanceId === FLAKY_INSTANCE) {
          // The output endpoint is down (proxy hiccup / pod restart) — every
          // poll 502s while the command id remains perfectly valid.
          return new Response("Bad Gateway", { status: 502 });
        }
        if (cmd.command.startsWith("exec ") || cmd.command.startsWith("\\d")) {
          return respond({
            command_id: commandId,
            status: "ok",
            created_at: "2026-07-22T10:00:00",
            command: cmd.command.split(" ")[0],
            query: null,
            queryid: null,
            response: cmd.command.startsWith("exec ") ? "CREATE INDEX" : "Table \"public.users\"",
            plan_text: null,
            plan_json: null,
            plan_execution_text: null,
            plan_execution_json: null,
            stats: null,
            recommendations: null,
            error: null,
          });
        }
        return respond({
          command_id: commandId,
          status: "ok",
          created_at: "2026-07-22T10:00:00",
          command: cmd.command.split(" ")[0],
          query: cmd.command.replace(/^\S+\s*/, "") || null,
          queryid: "7712349901234567890",
          response: null,
          plan_text: "Seq Scan on users  (cost=0.00..48210.00 rows=1 width=812)",
          plan_json: [{ Plan: { "Node Type": "Seq Scan", "Relation Name": "users" } }],
          plan_execution_text: "Seq Scan on users (actual time=0.05..12.30 rows=1 loops=1)",
          plan_execution_json: [{ Plan: {} }],
          stats: "Time: 12.4 ms",
          recommendations: null,
          error: null,
        });
      }

      return new Response("not found", { status: 404 });
    },
  });

  const baseUrl = `http://${server.hostname}:${server.port}/api/general`;
  return { baseUrl, requests, stop: () => server.stop(true) };
}

describe("CLI Joe command surface (grouped under `pgai joe …`)", () => {
  test("top-level help lists the joe group and top-level projects", () => {
    const r = runCli(["--help"], isolatedEnv());
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("joe");
    expect(out).toContain("projects");
  });

  test("pgai joe --help lists all the Joe verbs (and no async-era flags)", () => {
    const r = runCli(["joe", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    for (const verb of ["plan", "explain", "exec", "hypo", "activity", "terminate", "reset", "describe", "result"]) {
      expect(out).toContain(verb);
    }
    // The async surface's session flags and status/history verbs are gone.
    expect(out).not.toContain("--session");
    expect(out).not.toContain("new-session");
    expect(out).not.toContain("history");
  });

  test("pgai joe plan --help offers --instance-id and --budget but not --session", () => {
    const r = runCli(["joe", "plan", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("--instance-id");
    expect(out).toContain("--budget");
    expect(out).toContain("--project");
    expect(out).not.toContain("--session");
  });

  test("pgai projects prints the table", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(["projects"], isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl }));
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("PROJECT_ID");
      expect(r.stdout).toContain("ALIAS");
      expect(r.stdout).toContain("JOE");
      expect(r.stdout).toContain("TUNNEL");
      expect(r.stdout).toContain("main-db");
      expect(r.stdout).toContain("ready");
      const listReq = api.requests.find((x) => x.pathname.endsWith("/rpc/projects_list"));
      expect(listReq).toBeDefined();
    } finally {
      api.stop();
    }
  });

  test("pgai projects --json prints a machine-readable project array", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["projects", "--json"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const projects = JSON.parse(r.stdout) as Array<{ project_id: number; alias: string }>;
      expect(projects).toBeArray();
      expect(projects[0]).toMatchObject({ project_id: 12, alias: "main-db" });
    } finally {
      api.stop();
    }
  });

  test("pgai joe plan --project 12 sends the RAW text against the project's instance", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select * from users where email = 'x@acme.io'", "--project", "12"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("Seq Scan on users");
      // client-side plan flag from the structured plan_json
      expect(r.stdout).toContain("⚑");
      const run = api.requests.find((x) => x.pathname.endsWith("/rpc/joe_command_run"));
      // The run rpc keys on the INSTANCE id (3), resolved from project 12, and
      // the command is the raw text with the verb prefix — no structured body.
      expect(run?.body).toEqual({
        instance_id: 3,
        command: "plan select * from users where email = 'x@acme.io'",
      });
      // A numeric project still needs the listing — the instance id lives there.
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/projects_list"))).toBe(true);
    } finally {
      api.stop();
    }
  });

  test("pgai joe plan --json prints the full output row as JSON", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "12", "--json"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const result = JSON.parse(r.stdout) as {
        command_id: string;
        status: string;
        plan_text: string;
        plan_execution_text: string;
        stats: string;
        plan_json: unknown;
        queryid: string;
      };
      expect(result.command_id).toMatch(/^\d+$/);
      expect(result.status).toBe("ok");
      expect(result.plan_text).toContain("Seq Scan on users");
      // A >2^53 queryid survives the output rpc round-trip VERBATIM as a string.
      expect(result.queryid).toBe("7712349901234567890");
      // The sync contract surfaces the FULL body — execution plan and stats too.
      expect(result.plan_execution_text).toContain("actual time");
      expect(result.stats).toBe("Time: 12.4 ms");
      expect(Array.isArray(result.plan_json)).toBe(true);
    } finally {
      api.stop();
    }
  });

  test("pgai joe plan --project main-db resolves the alias via projects_list", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "main-db"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/projects_list"))).toBe(true);
      const run = api.requests.find((x) => x.pathname.endsWith("/rpc/joe_command_run"));
      expect(run?.body.instance_id).toBe(3);
    } finally {
      api.stop();
    }
  });

  test("every verb maps to its raw Joe command text", async () => {
    const cases: Array<{ argv: string[]; text: string }> = [
      { argv: ["joe", "explain", "select 1", "--project", "12"], text: "explain select 1" },
      { argv: ["joe", "exec", "create index on users (email)", "--project", "12"], text: "exec create index on users (email)" },
      { argv: ["joe", "hypo", "create index on users (email)", "--project", "12"], text: "hypo create index on users (email)" },
      { argv: ["joe", "activity", "--project", "12"], text: "activity" },
      { argv: ["joe", "reset", "--project", "12"], text: "reset" },
      { argv: ["joe", "terminate", "4711", "--project", "12"], text: "terminate 4711" },
      { argv: ["joe", "describe", "users", "--project", "12"], text: "\\d users" },
      { argv: ["joe", "describe", "users", "--project", "12", "--variant", "\\d+"], text: "\\d+ users" },
    ];
    for (const c of cases) {
      const api = startFakeApi();
      try {
        const r = await runCliAsync(c.argv, isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl }));
        expect(r.status).toBe(0);
        const run = api.requests.find((x) => x.pathname.endsWith("/rpc/joe_command_run"));
        expect(run?.body.command).toBe(c.text);
        expect(run?.body.instance_id).toBe(3);
      } finally {
        api.stop();
      }
    }
  });

  test("pgai joe explain --instance-id targets the instance directly (no projects_list)", async () => {
    // The v1 path: projects_list is not deployed on main/.green, so the joe
    // verbs must be fully usable with a manual instance id — resolution is
    // skipped entirely and the id goes on the wire as the given string.
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "explain", "select 1", "--instance-id", "1"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const run = api.requests.find((x) => x.pathname.endsWith("/rpc/joe_command_run"));
      expect(run?.body).toEqual({ instance_id: "1", command: "explain select 1" });
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/projects_list"))).toBe(false);
    } finally {
      api.stop();
    }
  });

  test("exec surfaces Joe's response body", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "exec", "create index on users (email)", "--project", "12"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("CREATE INDEX");
    } finally {
      api.stop();
    }
  });

  test("cold clone exceeds the budget -> resume handle (exit 0), then pgai joe result", async () => {
    const api = startFakeApi();
    const env = isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl });
    try {
      // cold-db (instance 55) stays `pending`; --budget 0 forces immediate resume.
      const r = await runCliAsync(["joe", "plan", "select 1", "--project", "cold-db", "--budget", "0"], env);
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("resume:");
      expect(r.stdout).toMatch(/pgai joe result \d+/);

      const match = r.stdout.match(/pgai joe result (\d+)/);
      const commandId = match ? match[1] : "";
      // The id is one the fake minted ABOVE 2^53 — it must be the exact string
      // the run rpc returned (a parseInt round-trip would corrupt it).
      expect(commandId).toMatch(/^900719925474099\d+$/);

      // `pgai joe result <id>` still returns pending for the cold instance and
      // exits 1 — the output rpc must receive the id VERBATIM as a string.
      const resumed = await runCliAsync(["joe", "result", commandId], env);
      expect(resumed.status).toBe(1);
      expect(resumed.stderr).toContain("result is not ready");
      const outputReq = api.requests.filter((x) => x.pathname.endsWith("/rpc/joe_command_output")).pop();
      expect(outputReq?.body.command_id).toBe(commandId);

      const jsonRun = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "cold-db", "--budget", "0", "--json"],
        env
      );
      expect(jsonRun.status).toBe(0);
      const jsonResume = JSON.parse(jsonRun.stdout) as {
        command_id: string;
        status: string;
        budget_expired: boolean;
        resume: string;
      };
      expect(jsonResume.budget_expired).toBe(true);
      expect(jsonResume.status).toBe("pending");
      expect(jsonResume.resume).toBe(`pgai joe result ${jsonResume.command_id}`);
    } finally {
      api.stop();
    }
  });

  test("pgai joe result renders a completed command's full output", async () => {
    const api = startFakeApi();
    const env = isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl });
    try {
      // Complete a command on the warm instance, then re-fetch it by id.
      const run = await runCliAsync(["joe", "plan", "select 1", "--project", "12", "--json"], env);
      expect(run.status).toBe(0);
      const commandId = (JSON.parse(run.stdout) as { command_id: string }).command_id;

      const resumed = await runCliAsync(["joe", "result", commandId], env);
      expect(resumed.status).toBe(0);
      expect(resumed.stdout).toContain("Seq Scan on users");
      expect(resumed.stdout).toContain("stats:");

      const resumedJson = await runCliAsync(["joe", "result", commandId, "--json"], env);
      expect(resumedJson.status).toBe(0);
      const jsonResult = JSON.parse(resumedJson.stdout) as { status: string; plan_text: string };
      expect(jsonResult.status).toBe("ok");
      expect(jsonResult.plan_text).toContain("Seq Scan on users");
    } finally {
      api.stop();
    }
  });

  test("pgai joe result exits 1 for a terminal error (JSON still printed with --json)", async () => {
    const api = startFakeApi();
    const env = isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl });
    try {
      const human = await runCliAsync(["joe", "result", "424242"], env);
      expect(human.status).toBe(1);
      expect(human.stderr).toContain("ERROR: something failed");

      const asJson = await runCliAsync(["joe", "result", "424242", "--json"], env);
      expect(asJson.status).toBe(1);
      expect((JSON.parse(asJson.stdout) as { status: string }).status).toBe("error");

      const pending = await runCliAsync(["joe", "result", "424244", "--json"], env);
      expect(pending.status).toBe(1);
      expect((JSON.parse(pending.stdout) as { status: string }).status).toBe("pending");
    } finally {
      api.stop();
    }
  });

  test("a terminal error from the one-shot exits 1 with the error message", async () => {
    // A plan against the role-gated instance surfaces the PT403 detail and exits 1.
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "no-role"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toContain("Joe API v2 requires the All Features role");
    } finally {
      api.stop();
    }
  });

  test("a command Joe fails (status=error) exits 1 with the error; --json still prints the row", async () => {
    const api = startFakeApi();
    const env = isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl });
    try {
      const human = await runCliAsync(["joe", "plan", "select * from nope", "--project", "err-db"], env);
      expect(human.status).toBe(1);
      expect(human.stderr).toContain('ERROR: relation "nope" does not exist');

      const asJson = await runCliAsync(["joe", "plan", "select * from nope", "--project", "err-db", "--json"], env);
      expect(asJson.status).toBe(1);
      const row = JSON.parse(asJson.stdout) as { status: string; error: string };
      expect(row.status).toBe("error");
      expect(row.error).toContain("does not exist");
    } finally {
      api.stop();
    }
  });

  test("a 502-ing output endpoint surfaces the resume handle instead of losing the id", async () => {
    // The run rpc succeeded (valid command id); every output poll 502s. The
    // one-shot must NOT throw the id away — it exits 0 with the resume line,
    // and the id is the exact string the run rpc returned.
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "flaky-db", "--budget", "0.2"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("resume:");
      const match = r.stdout.match(/pgai joe result (\d+)/);
      expect(match?.[1]).toMatch(/^900719925474099\d+$/);
      // The run rpc really was hit and the polls really did 502.
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/joe_command_run"))).toBe(true);
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/joe_command_output"))).toBe(true);
    } finally {
      api.stop();
    }
  });

  test("a terminal 404 on joe result aborts with a legible message", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "result", "111"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Failed to fetch command output");
      expect(r.stderr).toContain("404");
    } finally {
      api.stop();
    }
  });

  test("budget-expiry resume hint reports the ACTUAL --budget, not the hardcoded default", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "cold-db", "--budget", "0"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("budget 0s reached");
      expect(r.stdout).not.toContain("budget 25s reached");
    } finally {
      api.stop();
    }
  });

  test("negative --budget is rejected before any network call", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "12", "--budget", "-5"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("--budget must be a non-negative number");
      expect(api.requests.length).toBe(0);
    } finally {
      api.stop();
    }
  });

  test("joe terminate rejects a non-numeric / trailing-garbage pid (no network at all)", async () => {
    // A pid must be a bare positive integer. parseInt() would silently accept
    // "12x"→12, "-5"→-5, "1.5"→1 and terminate a WRONG backend. Those must be
    // a clean, typed rejection that never reaches any rpc.
    const api = startFakeApi();
    try {
      for (const bad of ["0", "12x", "abc", "1.5", "0x10", "  "]) {
        const r = await runCliAsync(
          ["joe", "terminate", bad, "--project", "12"],
          isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
        );
        expect(r.status).toBe(1);
        expect(`${r.stdout}\n${r.stderr}`).toContain("pid must be a positive integer");
      }
      expect(api.requests.length).toBe(0);
    } finally {
      api.stop();
    }
  });

  test("joe describe rejects a variant outside Joe's psql allowlist", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "describe", "users", "--project", "12", "--variant", "\\dx"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toContain("Unsupported describe variant");
      expect(api.requests.length).toBe(0);
    } finally {
      api.stop();
    }
  });

  test("a project without a Joe instance fails with a clear message", async () => {
    const api = startFakeApi();
    try {
      const r = await runCliAsync(
        ["joe", "plan", "select 1", "--project", "no-joe"],
        isolatedEnv({ PGAI_API_KEY: "k", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toContain("has no Joe instance");
      expect(api.requests.some((x) => x.pathname.endsWith("/rpc/joe_command_run"))).toBe(false);
    } finally {
      api.stop();
    }
  });

  test("neither --project nor --instance-id fails fast with the v1 hint", () => {
    const r = runCli(["joe", "plan", "select 1"], isolatedEnv({ PGAI_API_KEY: "k" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("--instance-id");
  });

  test("missing API key fails fast", () => {
    const r = runCli(["joe", "plan", "select 1", "--project", "12"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });
});
