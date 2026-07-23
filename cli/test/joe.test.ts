import { describe, test, expect, mock, afterEach, spyOn } from "bun:test";
import {
  startCommand,
  getCommandOutput,
  listProjects,
  resolveJoeInstanceId,
  isNumericProjectRef,
  buildJoeCommandText,
  runCommand,
  executeJoeCommand,
  clientSidePlanFlags,
  formatProjectsTable,
  formatJoeOutput,
  JOE_COMMANDS,
  DESCRIBE_VARIANTS,
  type JoeCommand,
  type JoeCommandOutput,
} from "../lib/joe";

const BASE = "https://api.example.com";
const originalFetch = globalThis.fetch;

interface Captured {
  url: string;
  body: Record<string, unknown>;
  headers: Record<string, string>;
}

/** Install a fetch mock that routes on the /rpc/<fn> suffix and records requests. */
function installFetch(routes: Record<string, (body: Record<string, unknown>) => Response>): Captured[] {
  const captured: Captured[] = [];
  globalThis.fetch = mock((url: string, options: RequestInit) => {
    const body = options.body ? (JSON.parse(options.body as string) as Record<string, unknown>) : {};
    const headers = (options.headers as Record<string, string>) || {};
    captured.push({ url, body, headers });
    const fn = new URL(url).pathname.split("/rpc/")[1] ?? "";
    const handler = routes[fn];
    if (!handler) {
      return Promise.resolve(new Response("not found", { status: 404 }));
    }
    return Promise.resolve(handler(body));
  }) as unknown as typeof fetch;
  return captured;
}

function json(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("startCommand (joe_command_run)", () => {
  test("throws when apiKey missing", async () => {
    await expect(
      startCommand({ apiKey: "", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" })
    ).rejects.toThrow("API key is required");
  });

  test("throws on empty command text before any network call", async () => {
    const captured = installFetch({ joe_command_run: () => json("1") });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "   " })
    ).rejects.toThrow("command text is required");
    expect(captured.length).toBe(0);
  });

  test("maps to /rpc/joe_command_run with {instance_id, command} and the access-token header", async () => {
    const captured = installFetch({ joe_command_run: () => json("4711") });
    const id = await startCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
    });
    expect(captured[0].url).toBe(`${BASE}/rpc/joe_command_run`);
    expect(captured[0].headers["access-token"]).toBe("k");
    // The RAW command text goes on the wire — no structured body, no prefixing.
    expect(captured[0].body).toEqual({ instance_id: 3, command: "plan select 1" });
    expect(id).toBe("4711");
  });

  test("the command id STAYS a string — a >2^53 id survives verbatim", async () => {
    // joe_command_run returns to_json(id::text). Any parseInt/Number round-trip
    // would corrupt 9007199254740993 to 9007199254740992.
    installFetch({ joe_command_run: () => json("9007199254740993") });
    const id = await startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "reset" });
    expect(id).toBe("9007199254740993");
    expect(typeof id).toBe("string");
  });

  test("a non-string rpc reply is rejected (contract violation)", async () => {
    installFetch({ joe_command_run: () => json({ command_id: "1" }) });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "reset" })
    ).rejects.toThrow(/expected a command id string/);
  });

  test("a bare-number id is rejected — the whole point of the precision guard", async () => {
    // If the backend ever returned to_json(id) instead of to_json(id::text),
    // JSON.parse would have ALREADY rounded 9007199254740993 → ...992; a
    // number must never be silently String()ed back.
    installFetch({ joe_command_run: () => json(9007199254740993) });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "reset" })
    ).rejects.toThrow(/expected a command id string/);
  });

  test("a 401 rejects with the auth remediation hint", async () => {
    installFetch({ joe_command_run: () => json({ message: "JWT expired" }, 401) });
    let thrown: Error | null = null;
    try {
      await startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" });
    } catch (err) {
      thrown = err as Error;
    }
    expect(thrown?.message).toContain("HTTP 401");
    expect(thrown?.message).toContain("JWT expired");
    expect(thrown?.message).toContain("postgresai auth");
  });

  test("a 502 on run rejects legibly (no id exists yet — nothing to resume)", async () => {
    installFetch({ joe_command_run: () => new Response("Bad Gateway", { status: 502 }) });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" })
    ).rejects.toThrow(/Failed to run Joe command: HTTP 502/);
  });

  test("surfaces the missing-role PT403 detail from the JSON body", async () => {
    installFetch({
      joe_command_run: () =>
        json({ code: "PT403", message: "Forbidden", details: "Joe API v2 requires the All Features role." }, 403),
    });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" })
    ).rejects.toThrow(/Joe API v2 requires the All Features role/);
  });

  test("surfaces a PT403 reason phrase when PostgREST returns no JSON body", async () => {
    installFetch({
      joe_command_run: () => new Response(null, {
        status: 403,
        statusText: "Joe API v2 requires the All Features role.",
      }),
    });
    await expect(
      startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" })
    ).rejects.toThrow("Joe API v2 requires the All Features role.");
  });

  test("a non-JSON body embedded in the parse-failure error is redacted", async () => {
    // The thrown Error flows into CLI stderr — a body echoing credentials must
    // not bypass redaction on this path.
    installFetch({
      joe_command_run: () => new Response("oops password=hunter2 dsn=postgresql://joe:pw-abc@h/db", { status: 200 }),
    });
    let thrown: Error | null = null;
    try {
      await startCommand({ apiKey: "k", apiBaseUrl: BASE, instanceId: 3, command: "plan select 1" });
    } catch (err) {
      thrown = err as Error;
    }
    expect(thrown?.message).toContain("failed to parse response");
    expect(thrown?.message).not.toContain("hunter2");
    expect(thrown?.message).not.toContain("pw-abc");
  });
});

describe("getCommandOutput (joe_command_output)", () => {
  test("maps to /rpc/joe_command_output with {command_id} and returns the FULL body", async () => {
    const captured = installFetch({
      joe_command_output: () =>
        json({
          command_id: "4711",
          status: "ok",
          created_at: "2026-07-22T10:00:00",
          command: "explain",
          query: "select * from users",
          queryid: "7712349901234567890",
          response: null,
          plan_text: "Seq Scan on users",
          plan_json: [{ Plan: { "Node Type": "Seq Scan", "Relation Name": "users" } }],
          plan_execution_text: "Seq Scan on users (actual time=0.1..12.3)",
          plan_execution_json: [{ Plan: {} }],
          stats: "Time: 12.4 ms",
          recommendations: ":warning: Seq Scan",
          error: null,
        }),
    });
    const output = await getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "4711" });
    expect(captured[0].url).toBe(`${BASE}/rpc/joe_command_output`);
    expect(captured[0].body).toEqual({ command_id: "4711" });
    expect(output.status).toBe("ok");
    expect(output.plan_text).toBe("Seq Scan on users");
    expect(output.plan_execution_text).toContain("actual time");
    expect(output.stats).toBe("Time: 12.4 ms");
    // plan_json arrives structured (the rpc unwraps the stored jsonb string).
    expect(Array.isArray(output.plan_json)).toBe(true);
  });

  test("a 'pending' body (no result columns yet) passes through", async () => {
    installFetch({
      joe_command_output: () => json({ command_id: "4711", status: "pending", created_at: "2026-07-22T10:00:00" }),
    });
    const output = await getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "4711" });
    expect(output.status).toBe("pending");
    expect(output.plan_text).toBeUndefined();
  });

  test("PT404 (not found / other org) surfaces as an error", async () => {
    installFetch({
      joe_command_output: () => json({ code: "PT404", message: "Not found", details: "Specified command not found." }, 404),
    });
    await expect(
      getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "9999" })
    ).rejects.toThrow(/Failed to fetch command output/);
  });

  test("requires a commandId", async () => {
    await expect(
      getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "" })
    ).rejects.toThrow("commandId is required");
  });

  test("a 502 rejects legibly (never mistaken for success)", async () => {
    installFetch({ joe_command_output: () => new Response("<html>Bad Gateway</html>", { status: 502 }) });
    await expect(
      getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "1" })
    ).rejects.toThrow(/Failed to fetch command output: HTTP 502/);
  });
});

describe("listProjects", () => {
  test("normalizes rows and defaults missing optional fields", async () => {
    installFetch({
      projects_list: () =>
        json([
          { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: true, instance_id: 3, dblab_instance_id: 7 },
          { project_id: 13, alias: "dw", name: "Warehouse", joe_ready: false, tunnel: false },
        ]),
    });
    const projects = await listProjects({ apiKey: "k", apiBaseUrl: BASE });
    expect(projects).toEqual([
      { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: true, instance_id: 3, dblab_instance_id: 7 },
      { project_id: 13, alias: "dw", name: "Warehouse", joe_ready: false, tunnel: false, instance_id: null, dblab_instance_id: null },
    ]);
  });

  test("passes org_id when provided", async () => {
    const captured = installFetch({ projects_list: () => json([]) });
    await listProjects({ apiKey: "k", apiBaseUrl: BASE, orgId: 7 });
    expect(captured[0].body).toEqual({ org_id: 7 });
  });

  test("preserves 64-bit project and instance ids without Number precision loss", async () => {
    installFetch({
      projects_list: () => json([{
        project_id: "9007199254740993",
        alias: "huge",
        instance_id: "9007199254740994",
        dblab_instance_id: "9007199254740995",
      }]),
    });
    const [project] = await listProjects({ apiKey: "k", apiBaseUrl: BASE });
    expect(project.project_id).toBe("9007199254740993");
    expect(project.instance_id).toBe("9007199254740994");
    expect(project.dblab_instance_id).toBe("9007199254740995");
  });
});

describe("resolveJoeInstanceId (project id-or-alias → Joe instance)", () => {
  const PROJECTS = [
    { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, instance_id: 3 },
    { project_id: 13, alias: "no-joe", name: "No Joe", joe_ready: false, instance_id: null },
  ];

  test("isNumericProjectRef distinguishes ids from aliases", () => {
    expect(isNumericProjectRef("12")).toBe(true);
    expect(isNumericProjectRef(" 12 ")).toBe(true);
    expect(isNumericProjectRef("main-db")).toBe(false);
    expect(isNumericProjectRef("12a")).toBe(false);
  });

  test("a numeric project id resolves to the project's instance_id (lookup required)", async () => {
    // Unlike a pure project-id resolver, the instance id only lives in the
    // projects listing — a numeric ref must still hit projects_list.
    const captured = installFetch({ projects_list: () => json(PROJECTS) });
    const id = await resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "12" });
    expect(id).toBe(3);
    expect(captured.length).toBe(1);
  });

  test("alias / name resolve case-insensitively to the same instance", async () => {
    installFetch({ projects_list: () => json(PROJECTS) });
    expect(await resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "MAIN-DB" })).toBe(3);
    expect(await resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "main db" })).toBe(3);
  });

  test("a 64-bit instance id survives as a string", async () => {
    installFetch({
      projects_list: () => json([{ project_id: 12, alias: "huge", instance_id: "9007199254740994" }]),
    });
    expect(await resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "huge" }))
      .toBe("9007199254740994");
  });

  test("unknown ref throws with a 'pgai projects' hint", async () => {
    installFetch({ projects_list: () => json(PROJECTS) });
    await expect(
      resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "nope" })
    ).rejects.toThrow(/Project not found for id\/alias\/name 'nope'/);
  });

  test("a project without a Joe instance throws", async () => {
    installFetch({ projects_list: () => json(PROJECTS) });
    await expect(
      resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "no-joe" })
    ).rejects.toThrow(/has no Joe instance/);
  });

  test("empty project ref throws before any network call", async () => {
    const captured = installFetch({ projects_list: () => json(PROJECTS) });
    await expect(
      resolveJoeInstanceId({ apiKey: "k", apiBaseUrl: BASE, project: "" })
    ).rejects.toThrow(/project is required/);
    expect(captured.length).toBe(0);
  });
});

describe("buildJoeCommandText — the raw wire text per verb", () => {
  test("SQL-carrying verbs prefix the verb, nothing else", () => {
    expect(buildJoeCommandText("plan", { arg: "select * from users" })).toBe("plan select * from users");
    expect(buildJoeCommandText("explain", { arg: "select 1" })).toBe("explain select 1");
    expect(buildJoeCommandText("exec", { arg: "create index on users (email)" })).toBe("exec create index on users (email)");
    expect(buildJoeCommandText("hypo", { arg: "create index on users (email)" })).toBe("hypo create index on users (email)");
  });

  test("bare verbs render as the verb alone", () => {
    expect(buildJoeCommandText("activity")).toBe("activity");
    expect(buildJoeCommandText("reset")).toBe("reset");
  });

  test("terminate renders `terminate <pid>` and rejects non-bare-digit pids", () => {
    expect(buildJoeCommandText("terminate", { arg: "4711" })).toBe("terminate 4711");
    for (const bad of ["0", "12x", "abc", "1.5", "0x10", "-5", "", "  "]) {
      expect(() => buildJoeCommandText("terminate", { arg: bad })).toThrow("pid must be a positive integer");
    }
  });

  test("describe defaults to \\d and honors an allowlisted --variant", () => {
    expect(buildJoeCommandText("describe", { arg: "users" })).toBe("\\d users");
    expect(buildJoeCommandText("describe", { arg: "users", variant: "\\d+" })).toBe("\\d+ users");
    expect(buildJoeCommandText("describe", { arg: "users_pkey", variant: "\\di+" })).toBe("\\di+ users_pkey");
  });

  test("describe rejects a variant outside Joe's psql allowlist", () => {
    for (const bad of ["\\dx", "\\copy", "d", "\\du"]) {
      expect(() => buildJoeCommandText("describe", { arg: "users", variant: bad })).toThrow(/Unsupported describe variant/);
    }
    // The allowlist mirrors Joe's own dispatcher table.
    expect([...DESCRIBE_VARIANTS]).toEqual(["\\d", "\\d+", "\\dt", "\\dt+", "\\di", "\\di+", "\\l", "\\l+", "\\dv", "\\dv+", "\\dm", "\\dm+"]);
  });

  test("empty required arguments throw", () => {
    for (const verb of ["plan", "explain", "exec", "hypo"] as const) {
      expect(() => buildJoeCommandText(verb, { arg: "  " })).toThrow(`${verb} requires an argument`);
    }
    expect(() => buildJoeCommandText("describe", { arg: "" })).toThrow("describe requires an object name");
  });

  test("JOE_COMMANDS is the Joe dispatcher verb set", () => {
    const expected: JoeCommand[] = ["plan", "explain", "exec", "hypo", "activity", "terminate", "reset", "describe"];
    expect([...JOE_COMMANDS]).toEqual(expected);
  });
});

describe("runCommand — run-then-poll", () => {
  test("polls the output until terminal (pending → pending → ok)", async () => {
    let outputCalls = 0;
    const captured = installFetch({
      joe_command_run: () => json("4711"),
      joe_command_output: () => {
        outputCalls += 1;
        if (outputCalls < 3) {
          return json({ command_id: "4711", status: "pending" });
        }
        return json({ command_id: "4711", status: "ok", plan_text: "Index Scan", error: null });
      },
    });
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.status).toBe("ok");
    expect(outcome.budgetExpired).toBe(false);
    expect(outcome.commandId).toBe("4711");
    expect(outcome.output?.plan_text).toBe("Index Scan");
    expect(outputCalls).toBe(3);
    // Every poll carried the id as the string the run rpc returned.
    for (const call of captured.slice(1)) {
      expect(call.body).toEqual({ command_id: "4711" });
    }
  });

  test("budget expiry returns a resume handle without an output", async () => {
    installFetch({
      joe_command_run: () => json("4712"),
      joe_command_output: () => json({ command_id: "4712", status: "pending" }),
    });
    let clock = 1000;
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      budgetMs: 5,
      pollIntervalMs: 0,
      now: () => (clock += 10),
      sleep: async () => {},
    });
    expect(outcome.budgetExpired).toBe(true);
    expect(outcome.status).toBe("pending");
    expect(outcome.output).toBeNull();
    expect(outcome.commandId).toBe("4712");
  });

  test("a stalled output request is aborted and returns a resume handle", async () => {
    let calls = 0;
    globalThis.fetch = mock((_url: string, options: RequestInit) => {
      calls += 1;
      if (calls === 1) {
        return Promise.resolve(json("4713"));
      }
      return new Promise<Response>((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => reject(options.signal?.reason), { once: true });
      });
    }) as unknown as typeof fetch;

    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      budgetMs: 10,
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome).toMatchObject({
      commandId: "4713",
      status: "pending",
      output: null,
      budgetExpired: true,
    });
  });

  test("a terminal error output is returned in full", async () => {
    installFetch({
      joe_command_run: () => json("5"),
      joe_command_output: () => json({ command_id: "5", status: "error", error: "ERROR: relation \"nope\" does not exist" }),
    });
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "explain select * from nope",
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.status).toBe("error");
    expect(outcome.budgetExpired).toBe(false);
    expect(outcome.output?.error).toContain("does not exist");
  });

  test("a transient 502 mid-poll never discards the id: keeps polling and recovers", async () => {
    // startCommand already returned a valid command id; a proxy hiccup on ONE
    // output poll must not throw it away — the loop retries within the budget.
    let outputCalls = 0;
    installFetch({
      joe_command_run: () => json("4714"),
      joe_command_output: () => {
        outputCalls += 1;
        if (outputCalls === 1) {
          return new Response("Bad Gateway", { status: 502 });
        }
        return json({ command_id: "4714", status: "ok", plan_text: "Index Scan", error: null });
      },
    });
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.status).toBe("ok");
    expect(outcome.commandId).toBe("4714");
    expect(outcome.output?.plan_text).toBe("Index Scan");
    expect(outputCalls).toBe(2);
  });

  test("a persistent 502 surfaces the id as a resume handle instead of throwing", async () => {
    installFetch({
      joe_command_run: () => json("4715"),
      joe_command_output: () => new Response("Bad Gateway", { status: 502 }),
    });
    let clock = 0;
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      budgetMs: 25,
      pollIntervalMs: 0,
      now: () => (clock += 10),
      sleep: async () => {},
    });
    expect(outcome).toMatchObject({
      commandId: "4715",
      status: "pending",
      output: null,
      budgetExpired: true,
    });
  });

  test("a persistent 429 (rate limit) also surfaces the id as a resume handle", async () => {
    installFetch({
      joe_command_run: () => json("4716"),
      joe_command_output: () => json({ message: "rate limited" }, 429),
    });
    let clock = 0;
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      budgetMs: 25,
      pollIntervalMs: 0,
      now: () => (clock += 10),
      sleep: async () => {},
    });
    expect(outcome).toMatchObject({
      commandId: "4716",
      status: "pending",
      output: null,
      budgetExpired: true,
    });
  });

  test("a terminal 404 mid-poll DOES abort (polling on cannot succeed)", async () => {
    installFetch({
      joe_command_run: () => json("4717"),
      joe_command_output: () => json({ code: "PT404", message: "Not found", details: "Specified command not found." }, 404),
    });
    await expect(
      runCommand({
        apiKey: "k",
        apiBaseUrl: BASE,
        instanceId: 3,
        command: "plan select 1",
        pollIntervalMs: 0,
        sleep: async () => {},
      })
    ).rejects.toThrow(/Failed to fetch command output: HTTP 404/);
  });

  test("a NaN budget is clamped to the default budget (never an unbounded poll)", async () => {
    // NaN survives `?? DEFAULT_BUDGET_MS` (it is neither null nor undefined), and
    // `now() >= NaN` is always false — without a clamp the poll loop never exits.
    // The output route trips a breaker well past the default-budget poll count so
    // a regression fails fast instead of hanging the test runner.
    let outputCalls = 0;
    installFetch({
      joe_command_run: () => json("9"),
      joe_command_output: () => {
        outputCalls += 1;
        if (outputCalls > 60) {
          throw new Error("unbounded poll loop: NaN budget never expired");
        }
        return json({ command_id: "9", status: "pending" });
      },
    });
    let clock = 0;
    const outcome = await runCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      instanceId: 3,
      command: "plan select 1",
      budgetMs: Number.NaN,
      pollIntervalMs: 0,
      now: () => (clock += 1000), // 1 s per observation → passes the 25 s default budget in <30 polls
      sleep: async () => {},
    });
    expect(outcome.budgetExpired).toBe(true);
    expect(outputCalls).toBeLessThanOrEqual(60);
  });
});

describe("executeJoeCommand — target → build → run", () => {
  test("resolves an alias to the project's instance_id and sends the raw text", async () => {
    const captured = installFetch({
      projects_list: () => json([{ project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, instance_id: 3 }]),
      joe_command_run: () => json("77"),
      joe_command_output: () => json({ command_id: "77", status: "ok", plan_text: "ok", error: null }),
    });
    const outcome = await executeJoeCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      command: "plan",
      project: "main-db",
      input: { arg: "select 1" },
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.instanceId).toBe(3);
    expect(outcome.commandText).toBe("plan select 1");
    expect(outcome.status).toBe("ok");
    const run = captured.find((c) => c.url.endsWith("/rpc/joe_command_run"));
    expect(run?.body).toEqual({ instance_id: 3, command: "plan select 1" });
  });

  test("a bad verb argument fails BEFORE any network call", async () => {
    const captured = installFetch({
      projects_list: () => json([]),
      joe_command_run: () => json("1"),
    });
    await expect(
      executeJoeCommand({
        apiKey: "k",
        apiBaseUrl: BASE,
        command: "terminate",
        project: "12",
        input: { arg: "12x" },
      })
    ).rejects.toThrow("pid must be a positive integer");
    expect(captured.length).toBe(0);
  });

  test("a direct instanceId skips project resolution entirely", async () => {
    // The direct path: --instance-id skips project resolution, so the id is
    // given directly and must go straight to joe_command_run — as the exact
    // string, never resolved, never Number()ed.
    const captured = installFetch({
      joe_command_run: () => json("80"),
      joe_command_output: () => json({ command_id: "80", status: "ok", response: "ok", error: null }),
    });
    const outcome = await executeJoeCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      command: "explain",
      instanceId: "9007199254740994001",
      input: { arg: "select 1" },
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.status).toBe("ok");
    expect(outcome.instanceId).toBe("9007199254740994001");
    expect(captured.some((c) => c.url.endsWith("/rpc/projects_list"))).toBe(false);
    const run = captured.find((c) => c.url.endsWith("/rpc/joe_command_run"));
    expect(run?.body).toEqual({ instance_id: "9007199254740994001", command: "explain select 1" });
  });

  test("instanceId wins over project when both are given (no projects_list call)", async () => {
    const captured = installFetch({
      projects_list: () => json([{ project_id: 12, alias: "main-db", joe_ready: true, instance_id: 3 }]),
      joe_command_run: () => json("81"),
      joe_command_output: () => json({ command_id: "81", status: "ok", response: "ok", error: null }),
    });
    await executeJoeCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      command: "reset",
      project: "main-db",
      instanceId: "7",
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(captured.some((c) => c.url.endsWith("/rpc/projects_list"))).toBe(false);
    const run = captured.find((c) => c.url.endsWith("/rpc/joe_command_run"));
    expect(run?.body.instance_id).toBe("7");
  });

  test("a malformed instanceId is rejected before any network call", async () => {
    const captured = installFetch({ joe_command_run: () => json("1") });
    await expect(
      executeJoeCommand({ apiKey: "k", apiBaseUrl: BASE, command: "reset", instanceId: "1; drop" })
    ).rejects.toThrow("instanceId must be a numeric Joe instance id");
    expect(captured.length).toBe(0);
  });

  test("neither instanceId nor project is a clear error before any network call", async () => {
    const captured = installFetch({ joe_command_run: () => json("1") });
    await expect(
      executeJoeCommand({ apiKey: "k", apiBaseUrl: BASE, command: "reset" })
    ).rejects.toThrow("either instanceId or project is required");
    expect(captured.length).toBe(0);
  });

  test("the full output (plan_execution_*, stats, recommendations) reaches the outcome", async () => {
    installFetch({
      projects_list: () => json([{ project_id: 12, alias: "main-db", joe_ready: true, instance_id: 3 }]),
      joe_command_run: () => json("78"),
      joe_command_output: () =>
        json({
          command_id: "78",
          status: "ok",
          command: "explain",
          query: "select 1",
          queryid: "991",
          plan_text: "Result",
          plan_json: [{ Plan: { "Node Type": "Result" } }],
          plan_execution_text: "Result (actual time=0.001..0.002)",
          stats: "Time: 0.5 ms",
          recommendations: "looks good",
          error: null,
        }),
    });
    const outcome = await executeJoeCommand({
      apiKey: "k",
      apiBaseUrl: BASE,
      command: "explain",
      project: "12",
      input: { arg: "select 1" },
      pollIntervalMs: 0,
      sleep: async () => {},
    });
    expect(outcome.output).toMatchObject({
      plan_execution_text: "Result (actual time=0.001..0.002)",
      stats: "Time: 0.5 ms",
      recommendations: "looks good",
      queryid: "991",
    });
  });
});

describe("presentation helpers", () => {
  test("clientSidePlanFlags flags Seq Scans in nested plans", () => {
    const flags = clientSidePlanFlags({
      Plan: {
        "Node Type": "Nested Loop",
        Plans: [{ "Node Type": "Seq Scan", "Relation Name": "users" }],
      },
    });
    expect(flags.length).toBe(1);
    expect(flags[0]).toContain("Seq Scan on users");
  });

  test("clientSidePlanFlags handles the EXPLAIN json array form the rpc returns", () => {
    // plan_json is unwrapped server-side into EXPLAIN's native array shape:
    // [{ "Plan": { … } }].
    const flags = clientSidePlanFlags([
      { Plan: { "Node Type": "Seq Scan", "Relation Name": "orders" } },
    ]);
    expect(flags.length).toBe(1);
    expect(flags[0]).toContain("Seq Scan on orders");
  });

  test("clientSidePlanFlags handles empty, root, and multiple nested plans", () => {
    expect(clientSidePlanFlags(null)).toEqual([]);
    expect(clientSidePlanFlags({})).toEqual([]);
    expect(clientSidePlanFlags({ "Node Type": "Seq Scan", "Relation Name": "root" })[0]).toContain("root");
    expect(clientSidePlanFlags({
      Plan: {
        "Node Type": "Append",
        Plans: [
          { "Node Type": "Seq Scan", "Relation Name": "a" },
          { "Node Type": "Seq Scan", "Relation Name": "b" },
        ],
      },
    })).toHaveLength(2);
  });

  test("formatJoeOutput prints every present section of the uniform row", () => {
    const output: JoeCommandOutput = {
      command_id: "1",
      status: "ok",
      plan_text: "Seq Scan on users",
      plan_json: [{ Plan: { "Node Type": "Seq Scan", "Relation Name": "users" } }],
      plan_execution_text: "Seq Scan on users (actual time=0.1..12.3)",
      stats: "Time: 12.4 ms",
      recommendations: ":warning: Seq Scan detected",
      queryid: "7712349901234567890",
      error: null,
    };
    const text = formatJoeOutput(output);
    expect(text).toContain("plan:");
    expect(text).toContain("Seq Scan on users");
    expect(text).toContain("⚑");
    expect(text).toContain("execution plan (EXPLAIN ANALYZE):");
    expect(text).toContain("stats:");
    expect(text).toContain("Time: 12.4 ms");
    expect(text).toContain("recommendations:");
    expect(text).toContain("(queryid 7712349901234567890)");
  });

  test("formatJoeOutput prints a bare response (exec/describe/activity) without labels", () => {
    const text = formatJoeOutput({ command_id: "2", status: "ok", response: "CREATE INDEX", error: null });
    expect(text).toBe("CREATE INDEX");
  });

  test("formatJoeOutput renders nothing for an empty row", () => {
    expect(formatJoeOutput({ command_id: "3", status: "ok", error: null })).toBe("");
  });

  test("formatProjectsTable renders the fixed-width columns", () => {
    const table = formatProjectsTable([
      { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: true, instance_id: 3, dblab_instance_id: 7 },
    ]);
    const [header, row] = table.split("\n");
    expect(header).toContain("PROJECT_ID");
    expect(header).toContain("ALIAS");
    expect(header).toContain("JOE");
    expect(header).toContain("TUNNEL");
    expect(row).toContain("12");
    expect(row).toContain("main-db");
    expect(row).toContain("ready");
    expect(row).toContain("yes");
  });
});

describe("--debug credential redaction (joe rpc surface)", () => {
  test("response bodies have password-named fields redacted before logging", async () => {
    // `--debug` writes the raw response body to stderr; any credential-shaped
    // field must be masked the same way the access-token header already is.
    const spy = spyOn(console, "error").mockImplementation(() => {});
    try {
      installFetch({
        joe_command_output: () =>
          json({
            command_id: "1",
            command: "exec",
            status: "ok",
            error: null,
            response: "ok",
            stats: null,
            plan_text: null,
            password: "row-secret-xyz",
          }),
      });
      await getCommandOutput({ apiKey: "k", apiBaseUrl: BASE, commandId: "1", debug: true });
      const logged = spy.mock.calls.map((c) => c.map(String).join(" ")).join("\n");
      expect(logged).toContain("Debug: Response body");
      expect(logged).not.toContain("row-secret-xyz");
      // The rest of the body still logs (redaction, not suppression).
      expect(logged).toContain("command_id");
    } finally {
      spy.mockRestore();
    }
  });
});
