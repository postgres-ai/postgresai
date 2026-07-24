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
