import { describe, test, expect, afterEach } from "bun:test";
import {
  enqueueQuery,
  readQueryResult,
  awaitQueryResult,
  isTerminal,
  renderPromQL,
  formatMetric,
  type PromQLPayload,
} from "../lib/promql";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function stubFetch(handler: (url: string, init: RequestInit) => { status?: number; body: unknown }) {
  const calls: { url: string; body: any; headers: Record<string, string> }[] = [];
  globalThis.fetch = (async (url: any, init: any) => {
    const parsed = init?.body ? JSON.parse(init.body) : null;
    calls.push({ url: String(url), body: parsed, headers: init?.headers ?? {} });
    const { status = 200, body } = handler(String(url), init);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as any;
  return calls;
}

describe("enqueueQuery", () => {
  test("an instant query sends only its own keys", async () => {
    // PostgREST resolves an RPC by the exact set of argument names in the body,
    // so a stray null start/end/step would fail to match the function.
    const calls = stubFetch(() => ({ body: { job_id: "job-1" } }));
    const out = await enqueueQuery("pai-token", {
      instanceId: "i-1",
      kind: "promql_instant",
      query: "up",
    });

    expect(out.jobId).toBe("job-1");
    expect(Object.keys(calls[0].body).sort()).toEqual(["p_instance_id", "p_kind", "p_query"]);
    expect(calls[0].url).toContain("/rpc/instance_query_enqueue");
  });

  test("a range query adds start, end and step", async () => {
    const calls = stubFetch(() => ({ body: { job_id: "job-2" } }));
    await enqueueQuery("pai-token", {
      instanceId: "i-1",
      kind: "promql_range",
      query: "rate(x[5m])",
      start: "2026-09-18T11:00:00Z",
      end: "2026-09-18T12:00:00Z",
      stepS: 300,
    });
    expect(calls[0].body.p_step_s).toBe(300);
    expect(calls[0].body.p_start).toBe("2026-09-18T11:00:00Z");
  });

  test("the credential travels in the header, never the body", async () => {
    // A body parameter is a bind parameter, and Postgres writes those into the
    // server log for any statement over log_min_duration_statement.
    const calls = stubFetch(() => ({ body: { job_id: "job-3" } }));
    await enqueueQuery("pai-secret-value", { instanceId: "i-1", kind: "promql_instant", query: "up" });

    expect((calls[0].headers as any)["access-token"]).toBe("pai-secret-value");
    expect(JSON.stringify(calls[0].body)).not.toContain("pai-secret-value");
  });

  test("a PostgREST error surfaces its message rather than a bare status", async () => {
    stubFetch(() => ({
      status: 409,
      body: { message: "Conflict", details: "a query is already in flight for this instance" },
    }));
    await expect(
      enqueueQuery("pai-token", { instanceId: "i-1", kind: "promql_instant", query: "up" }),
    ).rejects.toThrow(/already in flight/);
  });

  test("a reply with no job_id is an error, not a silent success", async () => {
    stubFetch(() => ({ body: {} }));
    await expect(
      enqueueQuery("pai-token", { instanceId: "i-1", kind: "promql_instant", query: "up" }),
    ).rejects.toThrow(/no job_id/);
  });
});

describe("awaitQueryResult", () => {
  const running = {
    status: "running",
    outcome: null,
    result: null,
    error: null,
    failure_class: null,
    started_at: null,
    finished_at: null,
  };

  test("polls until a terminal state, starting at 1s then 2s", async () => {
    let n = 0;
    stubFetch(() => {
      n++;
      return { body: n < 3 ? running : { ...running, status: "done", outcome: "ok" } };
    });
    const slept: number[] = [];
    let clock = 0;
    const out = await awaitQueryResult("pai-token", "job-1", {
      timeoutMs: 600_000,
      now: () => clock,
      sleep: async (ms) => {
        slept.push(ms);
        clock += ms;
      },
    });

    expect(isTerminal(out.status)).toBe(true);
    expect(out.status).toBe("done");
    // The first two are the contract's 1s and 2s, so a rig at a 5s server
    // interval still answers inside the first few polls.
    expect(slept).toEqual([1000, 2000]);
  });

  test("the backoff is capped rather than polling 2s forever", async () => {
    // Flat 2s against the 600000ms DEFAULT pacing is ~300 requests for one
    // answer. Doubling to a ceiling keeps the early polls fast and the long
    // wait cheap.
    stubFetch(() => ({ body: running }));
    const slept: number[] = [];
    // A virtual clock advanced by the sleeps, so the deadline is reached
    // deterministically rather than by waiting ten real minutes.
    let clock = 0;
    await awaitQueryResult("pai-token", "job-1", {
      timeoutMs: 600_000,
      now: () => clock,
      sleep: async (ms) => {
        slept.push(ms);
        clock += ms;
      },
    });

    expect(slept.slice(0, 4)).toEqual([1000, 2000, 4000, 8000]);
    expect(Math.max(...slept)).toBeLessThanOrEqual(15000);
    // Far fewer than the ~300 a flat 2s would need for the same wait.
    expect(slept.length).toBeLessThan(80);
  });

  test("each terminal status stops the loop", async () => {
    for (const status of ["done", "failed", "expired"]) {
      stubFetch(() => ({ body: { ...running, status } }));
      const out = await awaitQueryResult("pai-token", "job-1", {
        timeoutMs: 60_000,
        sleep: async () => {},
      });
      expect(isTerminal(out.status)).toBe(true);
      expect(out.status).toBe(status);
    }
  });

  test("a timeout reports the LAST observed status, not a bare expiry", async () => {
    // queued and running are different situations for the user: nothing has
    // claimed the job (check the instance) versus the box is working on it
    // (wait). The poll already knows which; discarding it wastes the one fact
    // that decides what to do next.
    for (const status of ["queued", "running"]) {
      stubFetch(() => ({ body: { ...running, status } }));
      const out = await awaitQueryResult("pai-token", "job-1", {
        timeoutMs: 1,
        sleep: async () => {},
      });
      // The caller distinguishes a timeout by asking isTerminal, and gets the
      // whole result -- error and failure_class included -- without a second path.
      expect(isTerminal(out.status)).toBe(false);
      expect(out.status).toBe(status);
    }
  });

  test("a malformed reply is an error, not treated as still running", async () => {
    stubFetch(() => ({ body: { nonsense: true } }));
    await expect(readQueryResult("pai-token", "job-1")).rejects.toThrow(/unexpected reply/);
  });
});

describe("rendering", () => {
  test("formatMetric prints Prometheus's own shape with sorted labels", () => {
    expect(formatMetric({ __name__: "up", job: "pg", instance: "a" })).toBe(
      'up{instance="a", job="pg"}',
    );
    expect(formatMetric({})).toBe("{}");
    expect(formatMetric({ __name__: "up" })).toBe("up");
  });

  test("a truncated result says so, loudly", () => {
    // A silently partial answer to a question a human asked is worse than an
    // error: they would act on it believing it complete.
    const payload: PromQLPayload = {
      resultType: "vector",
      result: [{ metric: { __name__: "up" }, value: [1, "1"] }],
      stats: { truncated: true },
    };
    const out = renderPromQL(payload);
    expect(out).toContain("TRUNCATED");
    // And says what to DO about it: with no point cap, truncation is a normal
    // outcome on a wide range query, not an exotic failure.
    expect(out).toContain("--step");
  });

  test("an untruncated result does not cry wolf", () => {
    const payload: PromQLPayload = {
      resultType: "vector",
      result: [{ metric: { __name__: "up" }, value: [1, "1"] }],
      stats: { truncated: false },
    };
    expect(renderPromQL(payload)).not.toContain("TRUNCATED");
  });

  test("a matrix summarises per series and says where the rest is", () => {
    const payload: PromQLPayload = {
      resultType: "matrix",
      result: [
        {
          metric: { __name__: "x", job: "a" },
          values: [
            [1, "1"],
            [2, "2"],
            [3, "3"],
          ],
        },
      ],
      stats: { truncated: false },
    };
    const out = renderPromQL(payload);
    expect(out).toContain("POINTS");
    expect(out).toContain("--json");
    // First and last, not every point.
    expect(out).toMatch(/3\s+1\s+3/);
  });

  test("an empty result set is stated, not rendered as a blank table", () => {
    const out = renderPromQL({ resultType: "vector", result: [], stats: { truncated: false } });
    expect(out).toContain("Empty result set");
  });
});

/**
 * PostgREST resolution, which is the step both suites skip.
 *
 * PostgREST picks an overload by the exact SET of argument names in the body:
 * every argument without a default must be present, and every body key must be
 * a known argument. A signature mismatch is a 404, not a helpful error — which
 * is why a broken signature can ship green through tests that stub `fetch` at
 * the HTTP layer and never model the rule.
 */
interface PgFunction {
  name: string;
  required: string[];
  optional: string[];
}

function resolvePostgrest(fns: PgFunction[], name: string, bodyKeys: string[]): PgFunction | null {
  return (
    fns.find(
      (f) =>
        f.name === name &&
        f.required.every((k) => bodyKeys.includes(k)) &&
        bodyKeys.every((k) => f.required.includes(k) || f.optional.includes(k)),
    ) ?? null
  );
}

function stubPostgrest(fns: PgFunction[]) {
  globalThis.fetch = (async (url: any, init: any) => {
    const name = String(url).split("/rpc/")[1];
    const body = init?.body ? JSON.parse(init.body) : {};
    const fn = resolvePostgrest(fns, name, Object.keys(body));
    if (!fn) {
      return new Response(
        JSON.stringify({
          code: "PGRST202",
          message: `Could not find the function public.${name} in the schema cache`,
        }),
        { status: 404, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(JSON.stringify({ job_id: "job-1" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as any;
}

describe("the body resolves against the real RPC signature", () => {
  /**
   * Mirrors `db/api/v1/functions/instance_query_enqueue.sql` and
   * `instance_query_result.sql` in platform-all. That file is the truth; this
   * is a copy, and the two drifting apart is precisely what produced the 404 —
   * so if a signature changes there, this is what should go red here.
   *
   * Every argument carries `default null`, including `api_token`: the CLI sends
   * the credential in the access-token header and never in the body, so without
   * that default PostgREST cannot resolve the call at all.
   */
  const platformSignature: PgFunction[] = [
    {
      name: "instance_query_enqueue",
      required: [],
      optional: [
        "api_token",
        "p_instance_id",
        "p_kind",
        "p_query",
        "p_at",
        "p_start",
        "p_end",
        "p_step_s",
      ],
    },
    { name: "instance_query_result", required: [], optional: ["api_token", "p_job_id"] },
  ];

  test("every body shape the CLI sends resolves", async () => {
    stubPostgrest(platformSignature);
    await expect(
      enqueueQuery("pai-token", { instanceId: "i-1", kind: "promql_instant", query: "up" }),
    ).resolves.toMatchObject({ jobId: "job-1" });
    await expect(
      enqueueQuery("pai-token", { instanceId: "i-1", kind: "promql_instant", query: "up", at: "2026-09-18T12:00:00Z" }),
    ).resolves.toMatchObject({ jobId: "job-1" });
    await expect(
      enqueueQuery("pai-token", {
        instanceId: "i-1",
        kind: "promql_range",
        query: "up",
        start: "2026-09-18T11:00:00Z",
        end: "2026-09-18T12:00:00Z",
        stepS: 60,
      }),
    ).resolves.toMatchObject({ jobId: "job-1" });

    expect(
      resolvePostgrest(platformSignature, "instance_query_result", ["p_job_id"]),
    ).not.toBeNull();
  });

  test("losing the default on ARGUMENT ONE makes the CLI unreachable", () => {
    // The regression guard, and it only needs to watch one position: SQL
    // requires every parameter after a defaulted one to also have a default, so
    // arg 2..n cannot lose theirs without a syntax error. api_token is arg one
    // and is therefore the only place this can silently come back.
    const regressed: PgFunction[] = [
      {
        name: "instance_query_enqueue",
        required: ["api_token"],
        optional: ["p_instance_id", "p_kind", "p_query", "p_at", "p_start", "p_end", "p_step_s"],
      },
    ];
    // The CLI's body never carries api_token, so nothing resolves — an opaque
    // 404 on every single call, which is exactly what shipped.
    expect(
      resolvePostgrest(regressed, "instance_query_enqueue", ["p_instance_id", "p_kind", "p_query"]),
    ).toBeNull();
  });

  test("an argument the signature does not declare is also unresolvable", () => {
    // The mirror rule, and the only discriminator left once every declared
    // argument is optional: without it the positive assertions above would pass
    // against any body at all.
    expect(
      resolvePostgrest(platformSignature, "instance_query_enqueue", [
        "p_instance_id",
        "p_kind",
        "p_query",
        "p_unknown",
      ]),
    ).toBeNull();
  });
});

describe("first_answer_estimate_s", () => {
  test("is read when the platform sends it", async () => {
    stubFetch(() => ({ body: { job_id: "job-1", first_answer_estimate_s: 600 } }));
    const out = await enqueueQuery("pai-token", {
      instanceId: "i-1",
      kind: "promql_instant",
      query: "up",
    });
    expect(out.firstAnswerEstimateS).toBe(600);
  });

  test("absent means fall back, never zero", async () => {
    // Two of these the platform can actually produce: ABSENT (any platform
    // older than the one that added the field, which is the normal state
    // during a rollout) and 0 (enqueue floors its pacing at 1ms while the poll
    // floors the same setting at 1000, so sub-500ms pacing rounds to zero).
    // The rest cannot come from our RPC and are not here because they can --
    // they are here because the reply is an unchecked `as` over a JSON body
    // off a network, and one predicate covers all five for the price of one.
    // Reading a missing field as 0 would size the wait to nothing and time out
    // instantly on a healthy box.
    for (const body of [
      { job_id: "job-1" },
      { job_id: "job-1", first_answer_estimate_s: null },
      { job_id: "job-1", first_answer_estimate_s: 0 },
      { job_id: "job-1", first_answer_estimate_s: -5 },
      { job_id: "job-1", first_answer_estimate_s: "600" },
    ]) {
      stubFetch(() => ({ body }));
      const out = await enqueueQuery("pai-token", {
        instanceId: "i-1",
        kind: "promql_instant",
        query: "up",
      });
      expect(out.firstAnswerEstimateS).toBeNull();
    }
  });
});
