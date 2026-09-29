import { describe, test, expect, mock, afterEach, spyOn } from "bun:test";
import {
  resolveDblabInstanceId,
  createClone,
  listClones,
  getClone,
  resetClone,
  destroyClone,
  listBranches,
  createBranch,
  deleteBranch,
  branchLog,
  listSnapshots,
  createSnapshot,
  destroySnapshot,
  awaitDblabCall,
  isUnknownRpcSignature,
} from "../lib/dblab";

const originalFetch = globalThis.fetch;
const API = "https://api.example.com";

interface Captured {
  url: string;
  options: RequestInit;
}

interface DblabRequestBody extends Record<string, unknown> {
  data?: Record<string, unknown>;
}

/** Stub fetch, returning `body` for every call, capturing the last request. */
function stubFetch(body: unknown, status = 200): { get: () => Captured | null } {
  let captured: Captured | null = null;
  globalThis.fetch = mock((url: string, options: RequestInit) => {
    captured = { url, options };
    return Promise.resolve(
      new Response(typeof body === "string" ? body : JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      })
    );
  }) as unknown as typeof fetch;
  return { get: () => captured };
}

/** Route fetch by URL: `/rpc/projects_list` → projects, `/rpc/dblab_api_call` → reply. */
function routeFetch(projects: unknown[], reply: unknown): { calls: Captured[] } {
  const calls: Captured[] = [];
  globalThis.fetch = mock((url: string, options: RequestInit) => {
    calls.push({ url, options });
    if (String(url).includes("/rpc/projects_list")) {
      return Promise.resolve(new Response(JSON.stringify(projects), { status: 200 }));
    }
    return Promise.resolve(new Response(JSON.stringify(reply), { status: 200 }));
  }) as unknown as typeof fetch;
  return { calls };
}

function bodyOf(c: Captured | null): DblabRequestBody {
  return JSON.parse((c!.options.body as string) ?? "{}") as DblabRequestBody;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// resolveDblabInstanceId
// ---------------------------------------------------------------------------

describe("resolveDblabInstanceId", () => {
  const PROJECTS = [
    { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: false, instance_id: 1, dblab_instance_id: 7 },
    { project_id: 34, alias: "analytics", name: "Analytics", joe_ready: false, tunnel: false, instance_id: null, dblab_instance_id: 9 },
    { project_id: 56, alias: "no-dblab", name: "No DBLab", joe_ready: false, tunnel: false, instance_id: null, dblab_instance_id: null },
  ];

  test("throws when apiKey is missing", async () => {
    await expect(
      resolveDblabInstanceId({ apiKey: "", apiBaseUrl: API, project: "12" })
    ).rejects.toThrow("API key is required");
  });

  test("throws when project is missing", async () => {
    await expect(
      resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "  " })
    ).rejects.toThrow("project is required");
  });

  test("resolves a numeric project id to its dblab instance id (string) via projects_list", async () => {
    const cap = stubFetch(PROJECTS);
    const id = await resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "34" });
    expect(id).toBe("9");
    const c = cap.get()!;
    expect(c.url).toBe(`${API}/rpc/projects_list`);
    expect(c.options.method).toBe("POST");
    expect((c.options.headers as Record<string, string>)["access-token"]).toBe("k");
  });

  test("resolves an alias (case-insensitive) to its dblab instance id", async () => {
    stubFetch(PROJECTS);
    const id = await resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "Main-DB" });
    expect(id).toBe("7");
  });

  test("resolves a project name to its dblab instance id", async () => {
    stubFetch(PROJECTS);
    const id = await resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "analytics" });
    expect(id).toBe("9");
  });

  test("passes org_id in the rpc body when orgId is provided", async () => {
    const cap = stubFetch(PROJECTS);
    await resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "12", orgId: 5 });
    expect(bodyOf(cap.get()).org_id).toBe(5);
  });

  test("throws a helpful error when no project matches", async () => {
    stubFetch(PROJECTS);
    await expect(
      resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "nope" })
    ).rejects.toThrow(/No DBLab instance found for project 'nope'/);
  });

  test("throws when the project exists but has no active DBLab instance", async () => {
    stubFetch(PROJECTS);
    await expect(
      resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "no-dblab" })
    ).rejects.toThrow(/has no active DBLab instance/);
  });

  test("surfaces an HTTP error from the listing", async () => {
    stubFetch('{"message":"boom"}', 500);
    await expect(
      resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "12" })
    ).rejects.toThrow(/Failed to resolve project's DBLab instance/);
  });
});

// ---------------------------------------------------------------------------
// dblab_api_call proxy — action/method/data per verb
// ---------------------------------------------------------------------------

describe("clone verbs → dblab_api_call", () => {
  const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7" };

  test("createClone posts /clone with a minimal data body", async () => {
    const cap = stubFetch({ id: "c-1" });
    await createClone({ ...common });
    const c = cap.get()!;
    expect(c.url).toBe(`${API}/rpc/dblab_api_call`);
    expect(c.options.method).toBe("POST");
    const b = bodyOf(c);
    expect(b.instance_id).toBe("7");
    expect(b.action).toBe("/clone");
    expect(b.method).toBe("post");
    expect(b.data).toEqual({ protected: false });
  });

  test("createClone maps branch / snapshot / db / protected into data", async () => {
    const cap = stubFetch({ id: "c-1" });
    await createClone({
      ...common,
      cloneId: "c-1",
      branch: "feature-idx",
      snapshotId: "s-9",
      dbUser: "u",
      dbPassword: "p",
      isProtected: true,
    });
    expect(bodyOf(cap.get()).data).toEqual({
      protected: true,
      id: "c-1",
      branch: "feature-idx",
      snapshot: { id: "s-9" },
      db: { username: "u", password: "p" },
    });
  });

  test("listClones gets /clones with no data", async () => {
    const cap = stubFetch([]);
    await listClones({ ...common });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/clones");
    expect(b.method).toBe("get");
    expect(b.data).toBeUndefined();
  });

  test("getClone gets /clone/<id> (url-encoded)", async () => {
    const cap = stubFetch({ id: "c/1" });
    await getClone({ ...common, cloneId: "c/1" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/clone/c%2F1");
    expect(b.method).toBe("get");
  });

  test("resetClone posts /clone/<id>/reset with latest:true when no snapshot", async () => {
    const cap = stubFetch(true);
    await resetClone({ ...common, cloneId: "c-1" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/clone/c-1/reset");
    expect(b.method).toBe("post");
    expect(b.data).toEqual({ latest: true });
  });

  test("resetClone with a snapshot sends snapshotID + latest:false", async () => {
    const cap = stubFetch(true);
    await resetClone({ ...common, cloneId: "c-1", snapshotId: "s-3" });
    expect(bodyOf(cap.get()).data).toEqual({ latest: false, snapshotID: "s-3" });
  });

  test("destroyClone deletes /clone/<id>", async () => {
    const cap = stubFetch("");
    await destroyClone({ ...common, cloneId: "c-1" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/clone/c-1");
    expect(b.method).toBe("delete");
  });

  test("destroyClone requires cloneId", async () => {
    await expect(destroyClone({ ...common, cloneId: "" })).rejects.toThrow("cloneId is required");
  });
});

describe("branch verbs → dblab_api_call", () => {
  const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7" };

  test("listBranches gets /branches", async () => {
    const cap = stubFetch([]);
    await listBranches({ ...common });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/branches");
    expect(b.method).toBe("get");
  });

  test("createBranch posts /branch with branchName and optional base/snapshot", async () => {
    const cap = stubFetch({ name: "feature-idx" });
    await createBranch({ ...common, branchName: "feature-idx", baseBranch: "main", snapshotId: "s-1" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/branch");
    expect(b.method).toBe("post");
    expect(b.data).toEqual({ branchName: "feature-idx", baseBranch: "main", snapshotID: "s-1" });
  });

  test("createBranch omits base/snapshot when not given", async () => {
    const cap = stubFetch({ name: "b" });
    await createBranch({ ...common, branchName: "b" });
    expect(bodyOf(cap.get()).data).toEqual({ branchName: "b" });
  });

  test("deleteBranch deletes /branch/<name> (encoded)", async () => {
    const cap = stubFetch(true);
    await deleteBranch({ ...common, branchName: "feature/x" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/branch/feature%2Fx");
    expect(b.method).toBe("delete");
  });

  test("branchLog gets /branch/<name>/log", async () => {
    const cap = stubFetch([]);
    await branchLog({ ...common, branchName: "feature-idx" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/branch/feature-idx/log");
    expect(b.method).toBe("get");
  });
});

describe("snapshot verbs → dblab_api_call", () => {
  const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7" };

  test("listSnapshots gets /snapshots (no query when unfiltered)", async () => {
    const cap = stubFetch([]);
    await listSnapshots({ ...common });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/snapshots");
    expect(b.method).toBe("get");
  });

  test("listSnapshots appends branch and dataset query params", async () => {
    const cap = stubFetch([]);
    await listSnapshots({ ...common, branchName: "main", dataset: "ds1" });
    expect(bodyOf(cap.get()).action).toBe("/snapshots?branch=main&dataset=ds1");
  });

  test("createSnapshot posts /branch/snapshot with cloneID + message", async () => {
    const cap = stubFetch({ snapshotID: "s-1" });
    await createSnapshot({ ...common, cloneId: "c-1", message: "before idx" });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/branch/snapshot");
    expect(b.method).toBe("post");
    expect(b.data).toEqual({ cloneID: "c-1", message: "before idx" });
  });

  test("destroySnapshot deletes /snapshot/<id>?force=<bool>", async () => {
    const cap = stubFetch(true);
    await destroySnapshot({ ...common, snapshotId: "s-1", force: true });
    const b = bodyOf(cap.get());
    expect(b.action).toBe("/snapshot/s-1?force=true");
    expect(b.method).toBe("delete");
  });

  test("destroySnapshot passes a multi-segment zfs snapshot id RAW (no percent-encoding)", async () => {
    const cap = stubFetch(true);
    await destroySnapshot({ ...common, snapshotId: "dblab_pool/branch/main/c-1/r0@20260707065703" });
    // The engine 400s on %2F/%40 — the Console passes the id raw and so must we.
    expect(bodyOf(cap.get()).action).toBe("/snapshot/dblab_pool/branch/main/c-1/r0@20260707065703?force=false");
  });

  test("destroySnapshot defaults force=false", async () => {
    const cap = stubFetch(true);
    await destroySnapshot({ ...common, snapshotId: "s-1" });
    expect(bodyOf(cap.get()).action).toBe("/snapshot/s-1?force=false");
  });

  test("destroySnapshot rejects URL metacharacters before calling the proxy", async () => {
    for (const snapshotId of ["s-1?force=true", "s-1#fragment", "s-1&force=true"]) {
      await expect(destroySnapshot({ ...common, snapshotId })).rejects.toThrow(/invalid characters/);
    }
  });

  test("destroySnapshot rejects dot-segment / empty-segment path traversal", async () => {
    // The id is embedded raw in the action path; `../clone/c-1` would retarget
    // the DELETE at a different endpoint than the verb claims if any hop
    // normalizes dot-segments. Empty segments (leading `/`, `//`) are equally
    // outside the real zfs snapshot-name shape.
    for (const snapshotId of ["../clone/c-1", "..", ".", "a/../clone/c-1", "a/./b", "/clone/c-1", "a//b", "a/"]) {
      await expect(destroySnapshot({ ...common, snapshotId })).rejects.toThrow(/invalid path segment/);
    }
  });
});

// ---------------------------------------------------------------------------
// error / role-gating paths
// ---------------------------------------------------------------------------

describe("proxy error paths", () => {
  const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7" };

  test("a PT403 (destructive-verb role gate) surfaces as a formatted 403 error", async () => {
    stubFetch('{"message":"Forbidden: this token\'s owner lacks the Admin or AllFeaturesUser role required for destructive DBLab operations (clone/snapshot/branch destroy)"}', 403);
    await expect(destroyClone({ ...common, cloneId: "c-1" })).rejects.toThrow(/Failed to destroy clone/);
  });

  test("a 500 surfaces the operation label", async () => {
    stubFetch('{"message":"boom"}', 500);
    await expect(listClones({ ...common })).rejects.toThrow(/Failed to list clones/);
  });

  test("proxy requires an instanceId", async () => {
    await expect(listClones({ ...common, instanceId: "" })).rejects.toThrow("instanceId is required");
  });

  test("a non-JSON body embedded in the parse-failure error is redacted", async () => {
    // The thrown Error flows into CLI stderr — a DBLab reply echoing
    // credentials must not bypass redaction on this path.
    stubFetch("oops connStr=postgresql://joe:pw-abc@h:6002/db password=hunter2", 200);
    let thrown: Error | null = null;
    try {
      await getClone({ ...common, cloneId: "c-1" });
    } catch (err) {
      thrown = err as Error;
    }
    expect(thrown?.message).toContain("failed to parse response");
    expect(thrown?.message).not.toContain("pw-abc");
    expect(thrown?.message).not.toContain("hunter2");
  });
});

// ---------------------------------------------------------------------------
// --debug credential redaction
//
// Operator-side debug writes request/response bodies to stderr. Credential
// fields must still be masked like the access-token header: clone-create embeds
// `data.db.password`, and clone create/status replies carry `db.password` /
// `db.connStr`.
// ---------------------------------------------------------------------------

describe("--debug credential redaction", () => {
  function captureStderr() {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    return {
      logged: () => spy.mock.calls.map((c) => c.map(String).join(" ")).join("\n"),
      restore: () => spy.mockRestore(),
    };
  }

  test("clone create debug logs do not expose the DB password (request body redacted)", async () => {
    const cap = captureStderr();
    try {
      stubFetch({ id: "c1" });
      await createClone({
        apiKey: "k-0123456789abcdef",
        apiBaseUrl: API,
        instanceId: "7",
        cloneId: "c1",
        dbUser: "clone_user",
        dbPassword: "hunter2-cleartext",
        debug: true,
      });
      const logged = cap.logged();
      expect(logged).toContain("Debug: Request body");
      expect(logged).not.toContain("hunter2-cleartext");
      // The rest of the body still logs (redaction, not suppression).
      expect(logged).toContain("clone_user");
      // The access token rides the header line, which the poll leg now
      // reprints once per poll — it must be masked there too.
      expect(logged).toContain("Debug: Request headers");
      expect(logged).not.toContain("k-0123456789abcdef");
    } finally {
      cap.restore();
    }
  });

  test("clone status --debug does not log the clone's credentials from the response (password/connStr)", async () => {
    const cap = captureStderr();
    try {
      stubFetch({
        id: "c1",
        status: { code: "OK" },
        db: {
          connStr: "host=dblab port=6002 user=joe password=resp-secret-xyz",
          password: "resp-secret-xyz",
          username: "joe",
        },
      });
      await getClone({ apiKey: "k-0123456789abcdef", apiBaseUrl: API, instanceId: "7", cloneId: "c1", debug: true });
      const logged = cap.logged();
      expect(logged).toContain("Debug: Response body");
      expect(logged).not.toContain("resp-secret-xyz");
      expect(logged).toContain("username");
    } finally {
      cap.restore();
    }
  });
});

// ---------------------------------------------------------------------------
// resolve → proxy end-to-end (project → instance → call)
// ---------------------------------------------------------------------------

describe("project → instance → dblab_api_call", () => {
  test("a verb driven off a resolved instance sends both requests", async () => {
    const projects = [{ project_id: 12, alias: "main-db", dblab_instance_id: 7 }];
    const { calls } = routeFetch(projects, { id: "c-1", status: "ready" });
    const instanceId = await resolveDblabInstanceId({ apiKey: "k", apiBaseUrl: API, project: "main-db" });
    await createClone({ apiKey: "k", apiBaseUrl: API, instanceId });
    expect(calls.length).toBe(2);
    expect(calls[0].url).toContain("/rpc/projects_list");
    expect(calls[1].url).toBe(`${API}/rpc/dblab_api_call`);
    expect(bodyOf(calls[1]).instance_id).toBe("7");
  });
});

// ---------------------------------------------------------------------------
// The job channel (platform-all#810): 202 handle → poll → the engine's reply
// ---------------------------------------------------------------------------

/**
 * Route by rpc, replying from a per-rpc script. Each entry is consumed in
 * order; the last one repeats, so a poll ladder is written as a list.
 */
function scriptFetch(
  script: Record<
    string,
    Array<{
      status?: number;
      body?: unknown;
      reject?: string;
      rejectWith?: unknown;
      bodyRejects?: unknown;
    }>
  >
): {
  calls: Captured[];
} {
  const calls: Captured[] = [];
  const cursor: Record<string, number> = {};
  globalThis.fetch = mock((url: string, options: RequestInit) => {
    calls.push({ url, options });
    const rpc = String(url).split("/rpc/")[1] ?? "";
    const steps = script[rpc];
    if (!steps) throw new Error(`unscripted rpc: ${rpc}`);
    const i = Math.min(cursor[rpc] ?? 0, steps.length - 1);
    cursor[rpc] = (cursor[rpc] ?? 0) + 1;
    const step = steps[i];
    // A step may fail the transport itself, which is a different path from any
    // HTTP status: `fetch` rejects and never yields a Response.
    if (step.rejectWith) return Promise.reject(step.rejectWith);
    if (step.reject) return Promise.reject(new TypeError(step.reject));
    if (step.bodyRejects) {
      // Headers arrive, the body never does — the window the request timeout
      // can fire in after `fetch` has already resolved.
      const stream = new ReadableStream({
        start: (c) => c.error(step.bodyRejects),
      });
      return Promise.resolve(new Response(stream, { status: step.status ?? 200 }));
    }
    return Promise.resolve(
      new Response(typeof step.body === "string" ? step.body : JSON.stringify(step.body), {
        status: step.status ?? 200,
        headers: { "Content-Type": "application/json" },
      })
    );
  }) as unknown as typeof fetch;
  return { calls };
}

const HANDLE = {
  pgai_async: "dblab_call",
  status: "accepted",
  job_id: "job-810",
  poll_rpc: "dblab_call_result",
  retry_safe: false,
  first_answer_estimate_s: 600,
  // NOT 900: that is numerically DBLAB_ASYNC_FALLBACK_WAIT_MS in ms, so a
  // fixture of 900 cannot tell "reads expires_in_s" from "ignores it". Small,
  // because the tests that go through createClone/getClone inject no clock —
  // a classification regression there stalls CI for the whole budget.
  expires_in_s: 2,
};

describe("async handle", () => {
  const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7" };

  test("every call declares accept_async, so the platform may hand back a handle", async () => {
    const cap = stubFetch({ id: "c-1" });
    await getClone({ ...common, cloneId: "c-1" });
    expect(bodyOf(cap.get()).accept_async).toBe(true);
  });

  test("a 202 is polled out and the caller gets the engine's reply", async () => {
    const { calls } = scriptFetch({
      dblab_api_call: [{ status: 202, body: HANDLE }],
      dblab_call_result: [
        { body: { status: "done", outcome: "ok", result: { id: "c-1", status: { code: "OK" } } } },
      ],
    });
    const clone = await createClone({ ...common, cloneId: "c-1" });
    expect(clone).toEqual({ id: "c-1", status: { code: "OK" } });
    expect(calls.map((c) => String(c.url).split("/rpc/")[1])).toEqual([
      "dblab_api_call",
      "dblab_call_result",
    ]);
    expect(JSON.parse(calls[1].options.body as string)).toEqual({ p_job_id: "job-810" });
  });

  test("a failed job throws, and the write is NEVER re-sent", async () => {
    const { calls } = scriptFetch({
      dblab_api_call: [{ status: 202, body: HANDLE }],
      dblab_call_result: [
        {
          body: {
            status: "failed",
            outcome: "error",
            result: null,
            error: "engine refused the clone",
            failure_class: "engine",
          },
        },
      ],
    });
    await expect(createClone({ ...common, cloneId: "c-1" })).rejects.toThrow(
      /engine refused the clone/
    );
    // Exactly one enqueue. A second POST /clone would create a second clone.
    expect(calls.filter((c) => String(c.url).includes("/rpc/dblab_api_call")).length).toBe(1);
  });

  test("an expired job says so rather than reporting success", async () => {
    scriptFetch({
      dblab_api_call: [{ status: 202, body: HANDLE }],
      dblab_call_result: [{ body: { status: "expired", outcome: null, result: null, error: null } }],
    });
    await expect(getClone({ ...common, cloneId: "c-1" })).rejects.toThrow(/expired/);
  });

  test("a platform without accept_async is retried once WITHOUT it", async () => {
    const notFound = {
      hint: "If a new function was created in the database with this name and parameters, try reloading the schema cache.",
      message:
        "Could not find the v1.dblab_api_call(accept_async, action, instance_id, method) function in the schema cache",
    };
    const { calls } = scriptFetch({
      dblab_api_call: [
        { status: 404, body: notFound },
        { body: { id: "c-1" } },
      ],
    });
    const clone = await getClone({ ...common, cloneId: "c-1" });
    expect(clone).toEqual({ id: "c-1" });
    expect(calls.length).toBe(2);
    expect(bodyOf(calls[0]).accept_async).toBe(true);
    expect(bodyOf(calls[1]).accept_async).toBeUndefined();
  });

  test("a PT404 from inside the rpc is NOT retried as a signature miss", async () => {
    // PostgREST returns the function's own PT404 with `details`, not the
    // schema-cache `message` — retrying that would only repeat a write.
    const { calls } = scriptFetch({
      dblab_api_call: [
        { status: 404, body: { details: "Specified Database Lab instance not found or you have no access." } },
      ],
    });
    await expect(getClone({ ...common, cloneId: "c-1" })).rejects.toThrow(
      /Failed to get clone: HTTP 404/
    );
    expect(calls.length).toBe(1);
  });

  test("a 5xx on the ENQUEUE of a write says it may have landed anyway", async () => {
    // v1.dblab_api_call commits the instance_jobs insert before the reply is
    // written, so "Gateway Timeout" alone reads as "nothing happened".
    // 500 is the boundary AND the likely value: PostgREST maps PT500 to it.
    for (const status of [500, 504]) {
      scriptFetch({ dblab_api_call: [{ status, body: { message: "server error" } }] });
      await expect(
        createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      ).rejects.toThrow(/may have accepted this call even though the reply was lost/);
    }
  });

  test("a 4xx on the ENQUEUE of a write does not — the other side of the boundary", async () => {
    for (const status of [400, 409, 499]) {
      scriptFetch({ dblab_api_call: [{ status, body: { message: "refused" } }] });
      const err = (await createClone({
        apiKey: "k",
        apiBaseUrl: API,
        instanceId: "7",
        cloneId: "c-1",
      }).then(() => null, (e: Error) => e)) as Error;
      expect(err.message).not.toMatch(/may have accepted this call/);
    }
  });

  test("a transport failure on the ENQUEUE of a write says the same", async () => {
    scriptFetch({ dblab_api_call: [{ reject: "ECONNRESET" }] });
    await expect(
      createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
    ).rejects.toThrow(/may have accepted this call even though the reply was lost/);
  });

  test("a timeout during the BODY read is classified, not leaked as a DOMException", async () => {
    // `fetch` resolves as soon as the headers land, so the 25s signal can fire
    // while the body is still streaming. That read used to sit outside the
    // try, and the bare DOMException it threw carries no operation name — and
    // its `message` is a getter with no setter, so anything appending to it
    // threw a TypeError and destroyed the error.
    scriptFetch({
      dblab_api_call: [
        {
          status: 202,
          bodyRejects: new DOMException("The operation timed out.", "TimeoutError"),
        },
      ],
    });
    const err = (await createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err).not.toBeInstanceOf(TypeError);
    expect(err.message).not.toContain("readonly property");
    // Classified as the timeout it is, naming the budget — not as a generic
    // "could not reach", which tells the operator nothing to act on.
    // Literals, not a \d+ that `0ms` would satisfy, and the message must START
    // with the operation — `String(err)` would prefix the class name instead.
    expect(err.message.startsWith("Failed to create clone: request timed out after 25000ms")).toBe(
      true
    );
    expect(err.message).toMatch(/may have accepted this call even though the reply was lost/);
    // The wrap must keep the original, or the timeout class is simply lost.
    expect((err as { cause?: unknown }).cause).toBeInstanceOf(Error);
    expect(((err as { cause?: Error }).cause as Error).name).toBe("HttpRequestTimeoutError");
  });

  test("a transport failure during the BODY read of a READ is classified too", async () => {
    scriptFetch({
      dblab_api_call: [{ status: 200, bodyRejects: new TypeError("terminated") }],
    });
    const err = (await getClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/Failed to get clone: could not reach/);
    expect(err.message).not.toMatch(/may have accepted this call/);
  });

  test("a transport failure on a READ stays quiet — the enqueue never happened", async () => {
    // The 504 test below exercises the !response.ok predicate; this one is the
    // catch, which had no read counterpart at all.
    scriptFetch({ dblab_api_call: [{ reject: "ECONNRESET" }] });
    const err = (await getClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/ECONNRESET/);
    expect(err.message).not.toMatch(/may have accepted this call/);
  });

  test("a 429 on the enqueue of a write does NOT say it may have landed", async () => {
    // A 429 is a gateway refusing to forward: nothing ran, and the right
    // recovery is to back off and retry — which the advice would suppress.
    scriptFetch({ dblab_api_call: [{ status: 429, body: { message: "slow down" } }] });
    const err = (await createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/429/);
    expect(err.message).not.toMatch(/may have accepted this call/);
  });

  test("a sync-route engine error cannot repaint the terminal either", async () => {
    // The async route was fixed in round 3; the same command on the pull path
    // was still relaying the engine's ESC/BEL raw.
    scriptFetch({
      dblab_api_call: [
        {
          status: 500,
          body: {
            code: "PT500",
            message: "Internal Server Error",
            details: "clone failed\u001b[2K\u001b[1G  SUCCESS: clone created\u0007",
          },
        },
      ],
    });
    const err = (await createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    // Newlines stay — formatHttpError puts the detail on its own line. What
    // must not survive is anything that can overwrite text already read.
    expect(err.message).not.toMatch(/[\u0000-\u0009\u000b-\u001f\u007f]/);
    expect(err.message).toMatch(/SUCCESS: clone created/);
  });

  test("stripping controls does not flatten formatHttpError's own lines", async () => {
    // A GET, so no advice is appended — every newline in the message is one
    // formatHttpError put there, and a blanket strip would eat all of them.
    scriptFetch({
      dblab_api_call: [
        {
          status: 500,
          body: {
            code: "PT500",
            message: "Internal Server Error",
            details: "engine said no",
            hint: "check the engine log",
          },
        },
      ],
    });
    const err = (await getClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).not.toMatch(/may have accepted this call/);
    expect(err.message.split("\n").length).toBeGreaterThan(1);
    expect(err.message).toMatch(/engine said no/);
  });

  test("a READ stays quiet about it — a GET changes nothing", async () => {
    scriptFetch({ dblab_api_call: [{ status: 504, body: { message: "Gateway Timeout" } }] });
    const err = (await getClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/504/);
    expect(err.message).not.toMatch(/may have accepted this call/);
  });

  test("a NON-retryable enqueue failure stays quiet too — the rpc never ran", async () => {
    scriptFetch({ dblab_api_call: [{ status: 403, body: { message: "Forbidden" } }] });
    const err = (await createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
      .then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/403/);
    expect(err.message).not.toMatch(/may have accepted this call/);
  });

  test("a WRITE also takes the old-platform fallback, and takes it exactly twice", async () => {
    // The safety argument is about writes, so pin it on one: a schema-cache
    // miss never reached the function, so re-sending POST /clone without
    // `accept_async` cannot create a second clone.
    const { calls } = scriptFetch({
      dblab_api_call: [
        {
          status: 404,
          body: {
            hint: "If a new function was created in the database with this name and parameters, try reloading the schema cache.",
            message:
              "Could not find the v1.dblab_api_call(accept_async, action, data, instance_id, method) function in the schema cache",
          },
        },
        { body: { id: "c-1" } },
      ],
    });
    const clone = await createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" });
    expect(clone).toEqual({ id: "c-1" });
    expect(calls.length).toBe(2);
    expect(bodyOf(calls[0]).accept_async).toBe(true);
    expect(bodyOf(calls[1]).accept_async).toBeUndefined();
    expect(bodyOf(calls[1]).method).toBe("post");
    expect(bodyOf(calls[1]).action).toBe("/clone");
  });

  test("a write refused by the rpc's OWN PT404 is not re-sent", async () => {
    const { calls } = scriptFetch({
      dblab_api_call: [
        {
          status: 404,
          body: {
            code: "PT404",
            message: "Not found",
            details: "Specified Database Lab instance not found or you have no access.",
          },
        },
      ],
    });
    await expect(
      createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
    ).rejects.toThrow(/404/);
    // A second POST /clone would be a second clone on the customer's disk.
    expect(calls.length).toBe(1);
  });

  test("isUnknownRpcSignature only matches PostgREST's schema-cache miss", () => {
    const miss = JSON.stringify({
      hint: "If a new function was created in the database with this name and parameters, try reloading the schema cache.",
      message:
        "Could not find the v1.dblab_api_call(accept_async, action, instance_id, method) function in the schema cache",
    });
    expect(isUnknownRpcSignature(404, miss, "dblab_api_call")).toBe(true);
    expect(isUnknownRpcSignature(400, miss, "dblab_api_call")).toBe(false);
    expect(isUnknownRpcSignature(404, "not json", "dblab_api_call")).toBe(false);
    // The rpc's OWN PT404 -- re-sending this one would repeat a write that ran.
    expect(
      isUnknownRpcSignature(
        404,
        JSON.stringify({
          code: "PT404",
          message: "Not found",
          details: "Specified Database Lab instance not found or you have no access.",
        }),
        "dblab_api_call"
      )
    ).toBe(false);
    // A function-raised 404 that happens to borrow PostgREST's wording still
    // does not name the rpc we sent, so it is not a signature miss either.
    expect(
      isUnknownRpcSignature(
        404,
        JSON.stringify({ code: "PT404", message: "Could not find the requested clone function" }),
        "dblab_api_call"
      )
    ).toBe(false);
    // ...and one that DOES name the rpc but is not the schema-cache wording.
    // Without this the "Could not find the ... function" test is free.
    expect(
      isUnknownRpcSignature(
        404,
        JSON.stringify({ code: "PT404", message: "dblab_api_call(7, '/clone'): instance not found" }),
        "dblab_api_call"
      )
    ).toBe(false);
  });

  test("a schema-cache miss for a DIFFERENT rpc is not a reason to re-send", async () => {
    // Pins the name `callDblabApi` threads in: with an empty rpc the check
    // degrades to `message.includes("(")` and this write would be re-sent.
    const { calls } = scriptFetch({
      dblab_api_call: [
        {
          status: 404,
          body: {
            hint: "If a new function was created in the database with this name and parameters, try reloading the schema cache.",
            message: "Could not find the v1.some_other_rpc(a, b) function in the schema cache",
          },
        },
      ],
    });
    await expect(
      createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" })
    ).rejects.toThrow(/404/);
    expect(calls.length).toBe(1);
  });

  test("a timeout names the job and refuses to start over", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "queued", outcome: null, result: null } }] });
    captureAnnounce();
    let clock = 0;
    await expect(
      awaitDblabCall(
        { job_id: "job-810", expires_in_s: 10, first_answer_estimate_s: 600 },
        {
          apiKey: "k",
          apiBaseUrl: API,
          operation: "create clone",
          // Deterministic: the clock only moves when the code waits.
          sleep: async (ms: number) => {
            clock += ms;
          },
          now: () => clock,
        }
      )
    ).rejects.toThrow(/job-810.*last status queued/s);
  });

  test("the poll ladder backs off instead of hammering a bcrypt per poll", async () => {
    scriptFetch({
      dblab_call_result: [
        { body: { status: "queued", outcome: null, result: null } },
        { body: { status: "running", outcome: null, result: null } },
        { body: { status: "done", outcome: "ok", result: { ok: true } } },
      ],
    });
    captureAnnounce();
    const waits: number[] = [];
    let clock = 0;
    const out = await awaitDblabCall<{ ok: boolean }>(
      { job_id: "job-810", expires_in_s: 900 },
      {
        apiKey: "k",
        apiBaseUrl: API,
        operation: "status",
        sleep: async (ms: number) => {
          waits.push(ms);
          clock += ms;
        },
        now: () => clock,
      }
    );
    expect(out).toEqual({ ok: true });
    expect(waits).toEqual([1000, 2000]);
  });

  test("the ladder doubles to a 15s ceiling and stays there", async () => {
    scriptFetch({
      dblab_call_result: [
        { body: { status: "queued" } },
        { body: { status: "queued" } },
        { body: { status: "queued" } },
        { body: { status: "queued" } },
        { body: { status: "queued" } },
        { body: { status: "queued" } },
        { body: { status: "done", outcome: "ok", result: { ok: true } } },
      ],
    });
    const clock = fakeClock();
    await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params);
    // Past the ceiling, not just up to it: any ceiling from 2s to 120s passes
    // an assertion that stops at rung 1.
    expect(clock.waits).toEqual([1000, 2000, 4000, 8000, 15000, 15000]);
  });
});

// ---------------------------------------------------------------------------
// The poll leg. A blip here must not strand an enqueued write: the platform
// holds one in-flight call per instance and nothing returns its id, so every
// way of giving up has to say the call was not re-sent.
// ---------------------------------------------------------------------------

/**
 * `awaitDblabCall` params with a clock that only moves when the code waits,
 * and a captured stderr so the announce line does not leak into the suite's
 * own output. Call `restore()` — `afterEach` does it too, as a backstop.
 */
let announceSpy: ReturnType<typeof spyOn> | null = null;
afterEach(() => {
  announceSpy?.mockRestore();
  announceSpy = null;
});

/** Swallow and record the "Waiting for the DBLab instance..." line. */
function captureAnnounce() {
  announceSpy?.mockRestore();
  const spy = spyOn(console, "error").mockImplementation(() => {});
  announceSpy = spy;
  return () => spy.mock.calls.map((c) => c.map(String).join(" "));
}

function fakeClock(operation = "create clone") {
  const waits: number[] = [];
  let clock = 0;
  const announced = captureAnnounce();
  return {
    waits,
    announced,
    params: {
      apiKey: "k",
      apiBaseUrl: API,
      operation,
      sleep: async (ms: number) => {
        waits.push(ms);
        clock += ms;
      },
      now: () => clock,
    },
  };
}

const DONE_OK = { body: { status: "done", outcome: "ok", result: { ok: true } } };

describe("async handle — a 202 body that is not a handle", () => {
  test("a non-JSON 202 names the operation instead of throwing a bare SyntaxError", async () => {
    scriptFetch({ dblab_api_call: [{ status: 202, body: "<html>502 Bad Gateway</html>" }] });
    await expect(createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7" })).rejects.toThrow(
      /Failed to create clone: failed to parse response/
    );
  });

  test("an empty 202 body is refused rather than polled as a handle", async () => {
    const { calls } = scriptFetch({ dblab_api_call: [{ status: 202 }] });
    await expect(createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7" })).rejects.toThrow(
      /Failed to create clone: failed to parse response/
    );
    expect(calls.length).toBe(1);
  });

  test("a 202 carrying JSON that is not an object is refused", async () => {
    for (const body of ["null", '"accepted"', "7"]) {
      scriptFetch({ dblab_api_call: [{ status: 202, body }] });
      await expect(createClone({ apiKey: "k", apiBaseUrl: API, instanceId: "7" })).rejects.toThrow(
        /Failed to create clone: the platform accepted the call but returned no handle/
      );
    }
  });

});

describe("async handle — the poll leg: retries, give-ups and bounds", () => {
  test("a 503 mid-wait is polled again rather than failing the enqueued call", async () => {
    scriptFetch({ dblab_call_result: [{ status: 503, body: { message: "upstream" } }, DONE_OK] });
    const clock = fakeClock();
    const out = await awaitDblabCall<{ ok: boolean }>(
      { job_id: "job-810", expires_in_s: 900 },
      clock.params
    );
    expect(out).toEqual({ ok: true });
    expect(clock.waits).toEqual([1000]);
  });

  test("a 429 mid-wait is polled again", async () => {
    const { calls } = scriptFetch({
      dblab_call_result: [{ status: 429, body: { message: "slow down" } }, DONE_OK],
    });
    const clock = fakeClock();
    expect(
      await awaitDblabCall<{ ok: boolean }>({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).toEqual({ ok: true });
    expect(calls.length).toBe(2);
    expect(clock.waits).toEqual([1000]);
  });

  test("a transport failure mid-wait is polled again", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [{ reject: "ECONNRESET" }, DONE_OK] });
    const clock = fakeClock();
    expect(
      await awaitDblabCall<{ ok: boolean }>({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).toEqual({ ok: true });
    expect(calls.length).toBe(2);
    expect(clock.waits).toEqual([1000]);
  });

  test("five failures in a row give up, naming the job and refusing to repeat the command", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [{ status: 502, body: { m: "bad gw" } }] });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    expect(err.message).toMatch(/job-810.*was NOT re-sent/s);
    // The diagnostic is the only thing that says WHY the platform went away.
    expect(err.message).toContain("5 polls in a row failed");
    expect(err.message).toContain("HTTP 502");
    // The fifth failure aborts; a sixth poll would be one too many.
    expect(calls.length).toBe(5);
  });

  // Every give-up below abandons a call that is still in flight, so each must
  // name the job and refuse to tell the user to repeat the command — asserting
  // only the status would leave "re-run clone create" as the obvious next move.
  const STRANDED = /job-810.*was NOT re-sent/s;

  test("a non-retryable poll status aborts at once — polling on cannot help", async () => {
    const { calls } = scriptFetch({
      dblab_call_result: [{ status: 403, body: { message: "Forbidden" } }],
    });
    const clock = fakeClock();
    const err = await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params).then(
      () => null,
      (e: Error) => e
    );
    expect(err?.message).toMatch(/403/);
    expect(err?.message).toMatch(STRANDED);
    expect(calls.length).toBe(1);
  });

  test("an unexpected reply shape is refused, not read as still running", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [{ body: null }] });
    const clock = fakeClock();
    const err = await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params).then(
      () => null,
      (e: Error) => e
    );
    expect(err?.message).toMatch(/create clone: dblab_call_result returned an unexpected reply/);
    expect(err?.message).toMatch(STRANDED);
    expect(calls.length).toBe(1);
  });

  test("an unparseable poll body names the operation", async () => {
    scriptFetch({ dblab_call_result: [{ body: "<html>502</html>" }] });
    const clock = fakeClock();
    const err = await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params).then(
      () => null,
      (e: Error) => e
    );
    expect(err?.message).toMatch(/create clone: failed to parse response/);
    expect(err?.message).toMatch(STRANDED);
  });

  test("a FAILED job says the write may have landed; an EXPIRED one does not", async () => {
    // The whole point of the MR: the box answers a write as failed after ONE
    // attempt because it may have landed. `expired` means nothing ran.
    scriptFetch({
      dblab_call_result: [{ body: { status: "failed", outcome: "error", error: "engine 500" } }],
    });
    const failed = await awaitDblabCall({ job_id: "job-810", expires_in_s: 120 }, fakeClock().params)
      .then(() => null, (e: Error) => e);
    expect(failed?.message).toMatch(STRANDED);

    scriptFetch({ dblab_call_result: [{ body: { status: "expired" } }] });
    const expired = await awaitDblabCall({ job_id: "job-810", expires_in_s: 120 }, fakeClock().params)
      .then(() => null, (e: Error) => e);
    expect(expired?.message).toMatch(/no agent claimed it before it expired/);
    expect(expired?.message).not.toMatch(/was NOT re-sent/);
  });

  test("a status this client does not know has STOPPED — it is not waited out", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [{ body: { status: "cancelled" } }] });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).rejects.toThrow(/status cancelled/);
    expect(calls.length).toBe(1);
    expect(clock.waits).toEqual([]);
  });

  test("done with a non-ok outcome is a failure, not a null result", async () => {
    scriptFetch({
      dblab_call_result: [{ body: { status: "done", outcome: "error", result: null } }],
    });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).rejects.toThrow(/outcome error/);
  });

  test("both box-written fields are control-stripped and scrubbed, not just `error`", async () => {
    // `error` and `failure_class` are written by the customer's box and relayed
    // verbatim; both reach stderr via `err.message`.
    scriptFetch({
      dblab_call_result: [
        {
          body: {
            status: "failed",
            outcome: "error",
            // The control byte sits INSIDE the credential: stripping it first
            // turns it into a space, and both matchers stop at whitespace.
            error: "could not reach postgresql://joe:hunter2\u0001hunter2@db:5432/x\r\u001b[2Jok",
            failure_class: "engine token=aaaaaaaa\u0001aaaaaaaa",
          },
        },
      ],
    });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    // eslint-disable-next-line no-control-regex
    expect(err.message).not.toMatch(/[\u0000-\u001f\u007f]/);
    expect(err.message).not.toContain("hunter2");
    expect(err.message).not.toContain("aaaaaaaa");
  });

  test("a JSON error is scrubbed by PATTERN too, not only by key name", async () => {
    // `redactSecretsForLog` parses a JSON body and redacts by KEY, so a
    // credential under a benign key survives that pass entirely. This fixture
    // is the one the prose fixtures above cannot reach.
    scriptFetch({
      dblab_call_result: [
        {
          body: {
            status: "failed",
            outcome: "error",
            error: JSON.stringify({ upstream: "postgresql://joe:hunter2hunter2@db:5432/x" }),
            failure_class: "engine_unreachable",
          },
        },
      ],
    });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    expect(err.message).not.toContain("hunter2hunter2");
    // Positive control: without it, "not leaked" is satisfied by a
    // `failure_class` that is never rendered at all.
    expect(err.message).toContain("[engine_unreachable]");
  });

  test("a sensitive key whose value carries an escaped quote is still scrubbed", async () => {
    // The one class the JSON/key pass catches that the text pass does not.
    scriptFetch({
      dblab_call_result: [
        {
          body: {
            status: "failed",
            outcome: "error",
            error: JSON.stringify({ password: 'ab"cdEFGHIJKLMNOP' }),
          },
        },
      ],
    });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    expect(err.message).not.toContain("cdEFGHIJKLMNOP");
  });

  test("a job id carrying an escape sequence cannot repaint the warning", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }] });
    const clock = fakeClock();
    const err = (await awaitDblabCall(
      { job_id: "7f3\u001b[2K\u001b[1G  SUCCESS: clone created\u0007", expires_in_s: 2 },
      clock.params
    ).then(() => null, (e: Error) => e)) as Error;
    expect(err.message).not.toMatch(/[\u0000-\u001f\u007f]/);
    expect(clock.announced().join("")).not.toMatch(/[\u0000-\u001f\u007f]/);
  });

  test("a status carrying an escape sequence cannot repaint the failure line", async () => {
    scriptFetch({
      dblab_call_result: [{ body: { status: "queued\u001b[2K\u001b[1G DONE: ok\u0007" } }],
    });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    expect(err.message).not.toMatch(/[\u0000-\u001f\u007f]/);
  });

  test("the failure bound is CONSECUTIVE — a good poll in between resets it", async () => {
    const bad = { status: 502 as const, body: { m: "bad gw" } };
    const { calls } = scriptFetch({
      dblab_call_result: [bad, bad, bad, bad, { body: { status: "queued" } }, bad, bad, bad, bad, DONE_OK],
    });
    const clock = fakeClock();
    expect(
      await awaitDblabCall<{ ok: boolean }>({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).toEqual({ ok: true });
    // 8 failures in all, never 5 in a row, so the wait was never abandoned.
    expect(calls.length).toBe(10);
  });

  test("a transient poll still respects the deadline — it does not buy extra time", async () => {
    const { calls } = scriptFetch({
      dblab_call_result: [
        { body: { status: "queued" } },
        { status: 503, body: { m: "x" } },
        { status: 503, body: { m: "x" } },
        DONE_OK,
      ],
    });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 4 }, clock.params)
    ).rejects.toThrow(/timed out/);
    // 1000 + 2000 waited; the next rung (4000) would cross the 4000ms budget.
    expect(clock.waits).toEqual([1000, 2000]);
    expect(calls.length).toBe(3);
  });

  test("an absent expires_in_s takes the 15-minute fallback, not the hour ceiling", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }] });
    const clock = fakeClock();
    await expect(awaitDblabCall({ job_id: "job-810" }, clock.params)).rejects.toThrow(/timed out/);
    const total = clock.waits.reduce((a, b) => a + b, 0);
    expect(total).toBeLessThanOrEqual(15 * 60 * 1000);
    expect(total).toBeGreaterThan(14 * 60 * 1000);
  });

  test("a zero or non-numeric expires_in_s takes the fallback too, not a zero wait", async () => {
    for (const bad of [0, -5, "120", null]) {
      scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }] });
      const clock = fakeClock();
      await expect(
        awaitDblabCall({ job_id: "job-810", expires_in_s: bad }, clock.params)
      ).rejects.toThrow(/timed out/);
      expect(clock.waits.reduce((a, b) => a + b, 0)).toBeGreaterThan(14 * 60 * 1000);
    }
  });

  test("the wait is announced ONCE, with the estimate when the handle carries one", async () => {
    scriptFetch({
      dblab_call_result: [
        { body: { status: "queued" } },
        { body: { status: "running" } },
        DONE_OK,
      ],
    });
    const clock = fakeClock();
    await awaitDblabCall({ job_id: "job-810", expires_in_s: 120, first_answer_estimate_s: 600 }, clock.params);
    expect(clock.announced()).toEqual([
      "Waiting for the DBLab instance to pick up the call (job job-810, typically ~600s)...",
    ]);
  });

  test("with no estimate the announce drops the clause rather than printing undefined", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }, DONE_OK] });
    const clock = fakeClock();
    await awaitDblabCall({ job_id: "job-810", expires_in_s: 120 }, clock.params);
    expect(clock.announced()).toEqual([
      "Waiting for the DBLab instance to pick up the call (job job-810)...",
    ]);
  });

  test("an empty engine reply answers null on BOTH routes, so a DELETE prints the same thing", async () => {
    // `DELETE /clone` answers an empty 200. The synchronous arm turns that into
    // null; the async arm must not hand back undefined instead.
    const common = { apiKey: "k", apiBaseUrl: API, instanceId: "7", cloneId: "c-1" };
    stubFetch("");
    expect(await destroyClone(common)).toBeNull();

    scriptFetch({
      dblab_api_call: [{ status: 202, body: HANDLE }],
      dblab_call_result: [{ body: { status: "done", outcome: "ok" } }],
    });
    expect(await destroyClone(common)).toBeNull();
  });

  test("an absurd expires_in_s is clamped, so a script can never hang forever", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }] });
    captureAnnounce();
    const waits: number[] = [];
    let clock = 0;
    await expect(
      // 1e12 s is ~31000 years. The platform clamps to an hour; this client
      // does not get to assume the other repo still does.
      awaitDblabCall(
        { job_id: "job-810", expires_in_s: 1e12 },
        {
          apiKey: "k",
          apiBaseUrl: API,
          operation: "create clone",
          // Without the clamp the deadline is unreachable and the loop never
          // exits, so bail on virtual time rather than hanging the suite.
          sleep: async (ms: number) => {
            waits.push(ms);
            clock += ms;
            if (clock > 2 * 60 * 60 * 1000) throw new Error(`unclamped: still polling at ${clock}ms`);
          },
          now: () => clock,
        }
      )
    ).rejects.toThrow(/timed out/);
    const total = waits.reduce((a, b) => a + b, 0);
    expect(total).toBeLessThanOrEqual(60 * 60 * 1000);
    // And it is the hour ceiling, not the 15-minute absent-value fallback.
    expect(total).toBeGreaterThan(15 * 60 * 1000);
  });

  test("a transient poll does NOT announce — the instance is not the problem", async () => {
    scriptFetch({ dblab_call_result: [{ status: 503, body: { m: "x" } }] });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 900 }, clock.params)
    ).rejects.toThrow(/lost contact with the platform/);
    expect(clock.announced()).toEqual([]);
  });

  test("a poll-leg engine error cannot repaint the terminal either", async () => {
    // The sync twin of this is pinned below; the poll leg's own strip was not.
    scriptFetch({
      dblab_call_result: [
        {
          status: 403,
          body: {
            code: "PT403",
            message: "Forbidden",
            details: "denied\u001b[2K\u001b[1G  SUCCESS: clone created\u0007",
          },
        },
      ],
    });
    const clock = fakeClock();
    const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params).then(
      () => null,
      (e: Error) => e
    )) as Error;
    expect(err.message).not.toMatch(/[\u0000-\u0009\u000b-\u001f\u007f]/);
    expect(err.message).toMatch(/SUCCESS: clone created/);
  });

  test("an unparseable or misshapen poll body cannot repaint it either", async () => {
    // The second one must carry the controls INSIDE the refused shape, or the
    // strip on that branch has nothing to do and the assertion is free.
    for (const body of [
      "<html>\u001b[2J\u001b[1G done\u0007</html>",
      // C1 CSI and DEL, not ESC: JSON.stringify escapes C0 for free, so an
      // ESC here would make the strip a no-op and the assertion vacuous.
      { status: 7, note: "\u009b2J done\u007f" },
    ]) {
      scriptFetch({ dblab_call_result: [{ body }] });
      const clock = fakeClock();
      const err = (await awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params).then(
        () => null,
        (e: Error) => e
      )) as Error;
      expect(err.message).not.toMatch(/[\u0000-\u0009\u000b-\u001f\u007f]/);
    }
  });

  test("an empty-string error falls back to the status text rather than blanking it", async () => {
    scriptFetch({ dblab_call_result: [{ body: { status: "expired", error: "" } }] });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params)
    ).rejects.toThrow(/no agent claimed it before it expired/);
  });

  test("a reply that is an array or has a non-string status is refused too", async () => {
    for (const body of [{}, [{ status: "done", outcome: "ok" }], { status: 7 }]) {
      const { calls } = scriptFetch({ dblab_call_result: [{ body }] });
      const clock = fakeClock();
      await expect(
        awaitDblabCall({ job_id: "job-810", expires_in_s: 2 }, clock.params)
      ).rejects.toThrow(/unexpected reply/);
      expect(calls.length).toBe(1);
    }
  });

  test("a non-string job_id is refused, not coerced", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [DONE_OK] });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: 12345, expires_in_s: 2 }, clock.params)
    ).rejects.toThrow(/returned no job_id/);
    expect(calls.length).toBe(0);
  });

  test("a delay landing EXACTLY on the deadline is not taken", async () => {
    // 1000 + 2000 waited, next rung 4000, deadline 7000 -> 7000 >= 7000 stops.
    scriptFetch({ dblab_call_result: [{ body: { status: "queued" } }] });
    const clock = fakeClock();
    await expect(
      awaitDblabCall({ job_id: "job-810", expires_in_s: 7 }, clock.params)
    ).rejects.toThrow(/timed out/);
    expect(clock.waits).toEqual([1000, 2000]);
  });

  test("a 202 whose body is unusable still says the call may run — the 202 proved it", async () => {
    // v1.dblab_api_call sets 202 only after the INSERT returned a job id, and
    // PostgREST commits before writing the reply.
    for (const body of ["<html>502</html>", undefined, "null"]) {
      scriptFetch({ dblab_api_call: [{ status: 202, body }] });
      const err = (await createClone({
        apiKey: "k",
        apiBaseUrl: API,
        instanceId: "7",
        cloneId: "c-1",
      }).then(() => null, (e: Error) => e)) as Error;
      expect(err.message).toMatch(/may have accepted this call even though the reply was lost/);
    }
  });

  test("a 202 handle with no job_id says the same", async () => {
    scriptFetch({ dblab_api_call: [{ status: 202, body: { pgai_async: "dblab_call" } }] });
    const err = (await createClone({
      apiKey: "k",
      apiBaseUrl: API,
      instanceId: "7",
      cloneId: "c-1",
    }).then(() => null, (e: Error) => e)) as Error;
    expect(err.message).toMatch(/returned no job_id/);
    expect(err.message).toMatch(/may have accepted this call/);
  });

  test("the RAW job id is what goes back on the wire, not the scrubbed one", async () => {
    // Scrubbing round-trips through JSON, so it is lossy for anything but a
    // uuid; polling with the scrubbed value would query the wrong job.
    const raw = "  1e5  ";
    const { calls } = scriptFetch({ dblab_call_result: [DONE_OK] });
    const clock = fakeClock();
    await awaitDblabCall({ job_id: raw, expires_in_s: 2 }, clock.params);
    expect(JSON.parse(calls[0].options.body as string)).toEqual({ p_job_id: raw });
  });

  test("a handle with no job_id is refused before any poll", async () => {
    const { calls } = scriptFetch({ dblab_call_result: [DONE_OK] });
    const clock = fakeClock();
    await expect(awaitDblabCall({ expires_in_s: 900 }, clock.params)).rejects.toThrow(
      /returned no job_id/
    );
    expect(calls.length).toBe(0);
  });
});
