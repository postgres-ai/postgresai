import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync, writeFileSync, existsSync, readFileSync } from "fs";
import { tmpdir } from "os";

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

function isolatedEnv(extra: Record<string, string> = {}) {
  // Ensure tests do not depend on any real user config on the machine running them.
  const cfgHome = mkdtempSync(resolve(tmpdir(), "postgresai-cli-test-"));
  return {
    XDG_CONFIG_HOME: cfgHome,
    HOME: cfgHome,
    ...extra,
  };
}

async function startFakeApi() {
  const requests: Array<{
    method: string;
    pathname: string;
    headers: Record<string, string>;
    bodyText: string;
    bodyJson: any | null;
    contentType?: string;
  }> = [];

  // For storage tests: tracks file uploads served at /storage/upload.
  const uploads: Array<{ filename: string; size: number; mimeType: string; url: string }> = [];

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const headers: Record<string, string> = {};
      for (const [k, v] of req.headers.entries()) headers[k.toLowerCase()] = v;

      // /storage/upload is multipart/form-data, not JSON. Branch on content type.
      const contentType = headers["content-type"] || "";
      const isMultipart = contentType.startsWith("multipart/form-data");

      // Storage upload endpoint — return a deterministic /files/N/<idx>_<name> URL.
      if (req.method === "POST" && url.pathname === "/storage/upload" && isMultipart) {
        const form = await req.formData();
        const file = form.get("file") as File | null;
        if (!file) return new Response("missing file", { status: 400 });
        const buf = new Uint8Array(await file.arrayBuffer());
        const idx = uploads.length;
        const fileUrl = `/files/test/${idx}_${file.name}`;
        uploads.push({ filename: file.name, size: buf.length, mimeType: file.type, url: fileUrl });
        requests.push({ method: req.method, pathname: url.pathname, headers, bodyText: "", bodyJson: null, contentType });
        return new Response(
          JSON.stringify({
            success: true,
            url: fileUrl,
            metadata: {
              originalName: file.name,
              size: buf.length,
              mimeType: file.type,
              uploadedAt: "2025-01-01T00:00:00.000Z",
              duration: 1,
            },
            requestId: `req-upload-${idx}`,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // Storage download endpoint — return whatever bytes were uploaded.
      if (req.method === "GET" && url.pathname.startsWith("/storage/files/")) {
        // Strip "/storage" so the lookup matches what we stored.
        const fileUrl = url.pathname.replace(/^\/storage/, "");
        const found = uploads.find((u) => u.url === fileUrl);
        if (!found) return new Response("not found", { status: 404 });
        // Re-upload-ish path: we don't keep bytes, but for test purposes return a known body.
        return new Response("FAKE_DOWNLOADED_BYTES", {
          status: 200,
          headers: { "Content-Type": found.mimeType || "application/octet-stream" },
        });
      }

      const bodyText = await req.text();
      let bodyJson: any | null = null;
      try {
        bodyJson = bodyText ? JSON.parse(bodyText) : null;
      } catch {
        bodyJson = null;
      }

      requests.push({
        method: req.method,
        pathname: url.pathname,
        headers,
        bodyText,
        bodyJson,
        contentType,
      });

      // Minimal fake PostgREST RPC endpoints used by our CLI.
      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_create")) {
        return new Response(
          JSON.stringify({
            id: "issue-1",
            title: bodyJson?.title ?? "",
            description: bodyJson?.description ?? null,
            created_at: "2025-01-01T00:00:00Z",
            status: 0,
            project_id: bodyJson?.project_id ?? null,
            labels: bodyJson?.labels ?? null,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_update")) {
        return new Response(
          JSON.stringify({
            id: bodyJson?.p_id ?? "issue-1",
            title: bodyJson?.p_title ?? "unchanged",
            description: bodyJson?.p_description ?? null,
            status: bodyJson?.p_status ?? 0,
            updated_at: "2025-01-02T00:00:00Z",
            labels: bodyJson?.p_labels ?? null,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_comment_update")) {
        return new Response(
          JSON.stringify({
            id: bodyJson?.p_id ?? "comment-1",
            issue_id: "issue-1",
            content: bodyJson?.p_content ?? "",
            updated_at: "2025-01-02T00:00:00Z",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_comment_create")) {
        return new Response(
          JSON.stringify({
            id: "comment-1",
            issue_id: bodyJson?.issue_id ?? "issue-1",
            author_id: 1,
            parent_comment_id: bodyJson?.parent_comment_id ?? null,
            content: bodyJson?.content ?? "",
            created_at: "2025-01-01T00:00:00Z",
            updated_at: "2025-01-01T00:00:00Z",
            data: null,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // GET /issues — used by `issues update --attach` to fetch existing description.
      if (req.method === "GET" && url.pathname.endsWith("/issues")) {
        const idParam = url.searchParams.get("id") || "";
        const issueId = idParam.replace("eq.", "");
        return new Response(
          JSON.stringify([
            {
              id: issueId,
              title: "Existing title",
              description: "Existing description body",
              status: 0,
              created_at: "2025-01-01T00:00:00Z",
              author_display_name: "tester",
              action_items: [],
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // Action Items endpoints
      if (req.method === "GET" && url.pathname.endsWith("/issue_action_items")) {
        const issueIdParam = url.searchParams.get("issue_id");
        const idParam = url.searchParams.get("id");
        if (issueIdParam) {
          // list_action_items
          return new Response(
            JSON.stringify([
              { id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", issue_id: issueIdParam.replace("eq.", ""), title: "Action 1", is_done: false, status: "waiting_for_approval" },
              { id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", issue_id: issueIdParam.replace("eq.", ""), title: "Action 2", is_done: true, status: "approved" },
            ]),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (idParam) {
          // view_action_item
          const actionId = idParam.replace("eq.", "").replace("in.(", "").replace(")", "").split(",")[0];
          return new Response(
            JSON.stringify([
              { id: actionId, issue_id: "11111111-1111-1111-1111-111111111111", title: "Test Action", description: "Test description", is_done: false, status: "waiting_for_approval", sql_action: "SELECT 1;", configs: [] },
            ]),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_action_item_create")) {
        return new Response(
          JSON.stringify("cccccccc-cccc-cccc-cccc-cccccccccccc"),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      if (req.method === "POST" && url.pathname.endsWith("/rpc/issue_action_item_update")) {
        return new Response("", { status: 200, headers: { "Content-Type": "application/json" } });
      }

      return new Response("not found", { status: 404 });
    },
  });

  const baseUrl = `http://${server.hostname}:${server.port}/api/general`;
  const storageBaseUrl = `http://${server.hostname}:${server.port}/storage`;

  return {
    baseUrl,
    storageBaseUrl,
    requests,
    uploads,
    stop: () => server.stop(true),
  };
}

describe("CLI issues command group", () => {
  test("issues help exposes the canonical subcommands and no legacy names", () => {
    const r = runCli(["issues", "--help"], isolatedEnv());
    expect(r.status).toBe(0);

    const out = `${r.stdout}\n${r.stderr}`;

    // Canonical subcommands
    expect(out).toContain("create [options] <title>");
    expect(out).toContain("update [options] <issueId>");
    expect(out).toContain("update-comment [options] <commentId> <content>");
    expect(out).toContain("post-comment [options] <issueId> <content>");

    // Legacy / removed names
    expect(out).not.toContain("create-issue");
    expect(out).not.toContain("update-issue");
    expect(out).not.toContain("update-issue-comment");
    expect(out).not.toContain("post_comment");
    expect(out).not.toContain("create_issue");
    expect(out).not.toContain("update_issue");
    expect(out).not.toContain("update_issue_comment");
  });

  test("issues create fails fast when API key is missing", () => {
    const r = runCli(["issues", "create", "Test issue"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues create fails fast when org id is missing (no config fallback)", () => {
    const r = runCli(["issues", "create", "Test issue"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("org_id is required");
  });

  test("issues update fails fast when API key is missing", () => {
    const r = runCli(["issues", "update", "00000000-0000-0000-0000-000000000000", "--title", "New title"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues update-comment fails fast when API key is missing", () => {
    const r = runCli(["issues", "update-comment", "00000000-0000-0000-0000-000000000000", "hello"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues post-comment fails fast when API key is missing", () => {
    const r = runCli(["issues", "post-comment", "00000000-0000-0000-0000-000000000000", "hello"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues create succeeds against a fake API and sends the expected request", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "create", "Hello", "--org-id", "123", "--description", "line1\\nline2", "--label", "a", "--label", "b"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out.id).toBe("issue-1");
      expect(out.title).toBe("Hello");
      expect(out.description).toBe("line1\nline2");
      expect(out.labels).toEqual(["a", "b"]);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_create"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.method).toBe("POST");
      expect(req!.bodyJson.org_id).toBe(123);
      expect(req!.bodyJson.title).toBe("Hello");
      expect(req!.bodyJson.description).toBe("line1\nline2");
      expect(req!.bodyJson.labels).toEqual(["a", "b"]);
    } finally {
      api.stop();
    }
  });

  test("issues update succeeds against a fake API (including status mapping)", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "update", "issue-1", "--title", "New title", "--status", "closed"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out.id).toBe("issue-1");
      expect(out.title).toBe("New title");
      expect(out.status).toBe(1);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_update"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.bodyJson.p_id).toBe("issue-1");
      expect(req!.bodyJson.p_title).toBe("New title");
      expect(req!.bodyJson.p_status).toBe(1);
    } finally {
      api.stop();
    }
  });

  test("issues update-comment succeeds against a fake API and decodes escapes", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "update-comment", "comment-1", "hello\\nworld"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out.id).toBe("comment-1");
      expect(out.content).toBe("hello\nworld");

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_update"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.bodyJson.p_id).toBe("comment-1");
      expect(req!.bodyJson.p_content).toBe("hello\nworld");
    } finally {
      api.stop();
    }
  });

  test("issues post-comment succeeds against a fake API and decodes escapes", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "post-comment", "issue-1", "hello\\nworld"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out.id).toBe("comment-1");
      expect(out.issue_id).toBe("issue-1");
      expect(out.content).toBe("hello\nworld");

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.bodyJson.issue_id).toBe("issue-1");
      expect(req!.bodyJson.content).toBe("hello\nworld");
    } finally {
      api.stop();
    }
  });
});

describe("CLI action items commands", () => {
  test("issues action-items fails fast when API key is missing", () => {
    const r = runCli(["issues", "action-items", "00000000-0000-0000-0000-000000000000"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues action-items fails when issue_id is not a valid UUID", () => {
    const r = runCli(["issues", "action-items", "invalid-id"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("issueId must be a valid UUID");
  });

  test("issues action-items succeeds against a fake API", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "action-items", "11111111-1111-1111-1111-111111111111"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(Array.isArray(out)).toBe(true);
      expect(out.length).toBe(2);
      expect(out[0].title).toBe("Action 1");
    } finally {
      api.stop();
    }
  });

  test("issues view-action-item fails fast when API key is missing", () => {
    const r = runCli(["issues", "view-action-item", "00000000-0000-0000-0000-000000000000"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues view-action-item fails when action_item_id is not a valid UUID", () => {
    const r = runCli(["issues", "view-action-item", "invalid-id"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("actionItemId is required and must be a valid UUID");
  });

  test("issues view-action-item succeeds against a fake API", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "view-action-item", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out[0].title).toBe("Test Action");
      expect(out[0].sql_action).toBe("SELECT 1;");
    } finally {
      api.stop();
    }
  });

  test("issues create-action-item fails fast when API key is missing", () => {
    const r = runCli(["issues", "create-action-item", "00000000-0000-0000-0000-000000000000", "Test title"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues create-action-item fails when issue_id is not a valid UUID", () => {
    const r = runCli(["issues", "create-action-item", "invalid-id", "Test title"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("issueId must be a valid UUID");
  });

  test("issues create-action-item succeeds against a fake API", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "create-action-item", "11111111-1111-1111-1111-111111111111", "New action item", "--description", "Test description"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(out.id).toBe("cccccccc-cccc-cccc-cccc-cccccccccccc");

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_action_item_create"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.bodyJson.issue_id).toBe("11111111-1111-1111-1111-111111111111");
      expect(req!.bodyJson.title).toBe("New action item");
      expect(req!.bodyJson.description).toBe("Test description");
    } finally {
      api.stop();
    }
  });

  test("issues create-action-item interprets escape sequences", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "create-action-item", "11111111-1111-1111-1111-111111111111", "Title\\nwith newline", "--description", "Desc\\twith tab"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_action_item_create"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.title).toBe("Title\nwith newline");
      expect(req!.bodyJson.description).toBe("Desc\twith tab");
    } finally {
      api.stop();
    }
  });

  test("issues update-action-item fails fast when API key is missing", () => {
    const r = runCli(["issues", "update-action-item", "00000000-0000-0000-0000-000000000000", "--done"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues update-action-item fails when action_item_id is not a valid UUID", () => {
    const r = runCli(["issues", "update-action-item", "invalid-id", "--done"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("actionItemId must be a valid UUID");
  });

  test("issues update-action-item fails when no update fields provided", () => {
    const r = runCli(["issues", "update-action-item", "00000000-0000-0000-0000-000000000000"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("At least one update option is required");
  });

  test("issues update-action-item succeeds with --done flag", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "update-action-item", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "--done"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_action_item_update"));
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
      expect(req!.bodyJson.action_item_id).toBe("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
      expect(req!.bodyJson.is_done).toBe(true);
    } finally {
      api.stop();
    }
  });

  test("issues update-action-item succeeds with --status flag", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["issues", "update-action-item", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "--status", "approved", "--status-reason", "LGTM"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_action_item_update"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.status).toBe("approved");
      expect(req!.bodyJson.status_reason).toBe("LGTM");
    } finally {
      api.stop();
    }
  });
});

async function startFakeStorageServer() {
  const requests: Array<{
    method: string;
    pathname: string;
    headers: Record<string, string>;
  }> = [];

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const headers: Record<string, string> = {};
      for (const [k, v] of req.headers.entries()) headers[k.toLowerCase()] = v;

      // Consume body to avoid warnings
      await req.arrayBuffer().catch(() => {});

      requests.push({
        method: req.method,
        pathname: url.pathname,
        headers,
      });

      // Upload endpoint
      if (req.method === "POST" && url.pathname === "/upload") {
        if (!headers["access-token"]) {
          return new Response(
            JSON.stringify({ code: "INVALID_API_TOKEN", message: "Missing token" }),
            { status: 401, headers: { "Content-Type": "application/json" } }
          );
        }

        return new Response(
          JSON.stringify({
            success: true,
            url: "/files/123/1707500000000_test-uuid.png",
            metadata: {
              originalName: "test-file.png",
              size: 16,
              mimeType: "image/png",
              uploadedAt: "2025-02-09T12:00:00.000Z",
              duration: 50,
            },
            requestId: "req-test-123",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // Download endpoint
      if (req.method === "GET" && url.pathname.startsWith("/files/")) {
        if (!headers["access-token"]) {
          return new Response(
            JSON.stringify({ code: "INVALID_API_TOKEN", message: "Missing token" }),
            { status: 401, headers: { "Content-Type": "application/json" } }
          );
        }

        return new Response(Buffer.from("fake-file-content"), {
          status: 200,
          headers: { "Content-Type": "image/png" },
        });
      }

      return new Response("not found", { status: 404 });
    },
  });

  const storageBaseUrl = `http://${server.hostname}:${server.port}`;

  return {
    storageBaseUrl,
    requests,
    stop: () => server.stop(true),
  };
}

describe("CLI issues files commands", () => {
  test("issues files upload fails fast when API key is missing", () => {
    const r = runCli(["issues", "files", "upload", "/tmp/test.png"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues files upload fails when file does not exist", () => {
    const r = runCli(
      ["issues", "files", "upload", "/tmp/nonexistent-file-99999.png"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("File not found");
  });

  test("issues files upload succeeds and shows URL and markdown", async () => {
    const storage = await startFakeStorageServer();
    const tmpFile = resolve(mkdtempSync(resolve(tmpdir(), "upload-test-")), "screenshot.png");
    writeFileSync(tmpFile, "fake-png-content");

    try {
      const r = await runCliAsync(
        ["issues", "files", "upload", tmpFile],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_STORAGE_BASE_URL: storage.storageBaseUrl,
        })
      );

      expect(r.status).toBe(0);
      const out = `${r.stdout}\n${r.stderr}`;
      expect(out).toContain("URL:");
      expect(out).toContain("/files/123/");
      expect(out).toContain("Markdown:");
      expect(out).toContain("![");

      const uploadReq = storage.requests.find((x) => x.pathname === "/upload");
      expect(uploadReq).toBeTruthy();
      expect(uploadReq!.headers["access-token"]).toBe("test-key");
      expect(uploadReq!.method).toBe("POST");
    } finally {
      storage.stop();
    }
  });

  test("issues files upload --json returns structured JSON", async () => {
    const storage = await startFakeStorageServer();
    const tmpFile = resolve(mkdtempSync(resolve(tmpdir(), "upload-json-test-")), "data.txt");
    writeFileSync(tmpFile, "test-content");

    try {
      const r = await runCliAsync(
        ["issues", "files", "upload", tmpFile, "--json"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_STORAGE_BASE_URL: storage.storageBaseUrl,
        })
      );

      expect(r.status).toBe(0);
      const out = JSON.parse(r.stdout.trim());
      expect(out.success).toBe(true);
      expect(out.url).toContain("/files/");
      expect(out.metadata.originalName).toBe("test-file.png");
      expect(out.metadata.mimeType).toBe("image/png");
    } finally {
      storage.stop();
    }
  });

  test("issues files download fails fast when API key is missing", () => {
    const r = runCli(["issues", "files", "download", "/files/123/test.png"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("issues files download succeeds and saves file", async () => {
    const storage = await startFakeStorageServer();
    const tmpOutDir = mkdtempSync(resolve(tmpdir(), "download-test-"));
    const outputPath = resolve(tmpOutDir, "downloaded.png");

    try {
      const r = await runCliAsync(
        ["issues", "files", "download", "/files/123/image.png", "-o", outputPath],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_STORAGE_BASE_URL: storage.storageBaseUrl,
        })
      );

      expect(r.status).toBe(0);
      const out = `${r.stdout}\n${r.stderr}`;
      expect(out).toContain("Saved:");
      expect(out).not.toContain("Size:");
      expect(out).not.toContain("Type:");

      expect(existsSync(outputPath)).toBe(true);
      expect(readFileSync(outputPath).toString()).toBe("fake-file-content");

      const downloadReq = storage.requests.find((x) => x.pathname.startsWith("/files/"));
      expect(downloadReq).toBeTruthy();
      expect(downloadReq!.headers["access-token"]).toBe("test-key");
      expect(downloadReq!.method).toBe("GET");
    } finally {
      storage.stop();
    }
  });

  test("issues help shows files subcommand", () => {
    const r = runCli(["issues", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("files");
  });

  test("issues files help shows upload and download", () => {
    const r = runCli(["issues", "files", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("upload");
    expect(out).toContain("download");
  });
});

describe("CLI set-storage-url command", () => {
  test("saves valid URL to config", () => {
    const env = isolatedEnv();
    const r = runCli(["set-storage-url", "https://v2.postgres.ai/storage"], env);
    expect(r.status).toBe(0);
    expect(`${r.stdout}\n${r.stderr}`).toContain("Storage URL saved: https://v2.postgres.ai/storage");

    // Verify persisted in config
    const cfgPath = resolve(env.XDG_CONFIG_HOME, "postgresai", "config.json");
    const cfg = JSON.parse(readFileSync(cfgPath, "utf-8"));
    expect(cfg.storageBaseUrl).toBe("https://v2.postgres.ai/storage");
  });

  test("normalizes trailing slash", () => {
    const env = isolatedEnv();
    const r = runCli(["set-storage-url", "https://example.com/storage/"], env);
    expect(r.status).toBe(0);
    expect(`${r.stdout}\n${r.stderr}`).toContain("Storage URL saved: https://example.com/storage");
  });

  test("rejects invalid URL", () => {
    const r = runCli(["set-storage-url", "not-a-url"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("invalid URL");
  });
});

describe("CLI issues --attach flag", () => {
  // Helper: write a small image-named tmp file (not a real PNG, but the .png
  // extension is what the CLI / storage helper read for MIME + markdown form).
  function writeTmpFile(name: string, body = "X"): string {
    const dir = mkdtempSync(resolve(tmpdir(), "pgai-attach-"));
    const p = resolve(dir, name);
    writeFileSync(p, body);
    return p;
  }

  test("post-comment --attach uploads then appends image markdown to comment", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("pic.png", "PNGBYTES");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "post-comment",
          "11111111-1111-1111-1111-111111111111",
          "see attached",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );

      expect(r.status).toBe(0);
      // Upload happened first.
      expect(api.uploads).toHaveLength(1);
      expect(api.uploads[0].filename).toBe("pic.png");

      // Comment-create request was sent with augmented content.
      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.content).toBe(
        `see attached\n\n![pic.png](${api.storageBaseUrl}${api.uploads[0].url})`
      );
      // Request order: upload before comment.
      const uploadIdx = api.requests.findIndex((x) => x.pathname === "/storage/upload");
      const commentIdx = api.requests.findIndex((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(uploadIdx).toBeGreaterThanOrEqual(0);
      expect(commentIdx).toBeGreaterThan(uploadIdx);
    } finally {
      api.stop();
    }
  });

  test("post-comment --attach with multiple files appends one link per line preserving order", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("a.png", "P");
    const log = writeTmpFile("b.log", "L");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "post-comment",
          "11111111-1111-1111-1111-111111111111",
          "ctx",
          "--attach",
          png,
          "--attach",
          log,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      expect(api.uploads.map((u) => u.filename)).toEqual(["a.png", "b.log"]);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(req).toBeTruthy();
      const expected =
        `ctx\n\n![a.png](${api.storageBaseUrl}/files/test/0_a.png)\n` +
        `[b.log](${api.storageBaseUrl}/files/test/1_b.log)`;
      expect(req!.bodyJson.content).toBe(expected);
    } finally {
      api.stop();
    }
  });

  test("create --attach appends markdown link to description", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("diagram.png", "PNG");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "create",
          "Reproduces under load",
          "--org-id",
          "1",
          "--description",
          "see chart",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_create"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.description).toBe(
        `see chart\n\n![diagram.png](${api.storageBaseUrl}/files/test/0_diagram.png)`
      );
    } finally {
      api.stop();
    }
  });

  test("create --attach without --description sets description to just the link", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("only.png", "P");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "create",
          "tinier example",
          "--org-id",
          "1",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_create"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.description).toBe(
        `![only.png](${api.storageBaseUrl}/files/test/0_only.png)`
      );
    } finally {
      api.stop();
    }
  });

  test("update --attach without --description fetches existing description and appends", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("evidence.png", "P");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "update",
          "issue-1",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      // GET /issues happens first to read the existing description, then upload, then update.
      const seq = api.requests.map((x) => `${x.method} ${x.pathname}`);
      const fetchIdx = seq.indexOf("GET /api/general/issues");
      const uploadIdx = seq.indexOf("POST /storage/upload");
      const updateIdx = seq.indexOf("POST /api/general/rpc/issue_update");
      expect(fetchIdx).toBeGreaterThanOrEqual(0);
      expect(uploadIdx).toBeGreaterThan(fetchIdx);
      expect(updateIdx).toBeGreaterThan(uploadIdx);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_update"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.p_description).toBe(
        `Existing description body\n\n![evidence.png](${api.storageBaseUrl}/files/test/0_evidence.png)`
      );
    } finally {
      api.stop();
    }
  });

  test("update --attach with --description appends to the new description (no fetch)", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("e2.png", "P");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "update",
          "issue-1",
          "--description",
          "Rewritten body",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      // No GET /issues happened — we already have the new description.
      const fetched = api.requests.find((x) => x.method === "GET" && x.pathname.endsWith("/issues"));
      expect(fetched).toBeFalsy();

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_update"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.p_description).toBe(
        `Rewritten body\n\n![e2.png](${api.storageBaseUrl}/files/test/0_e2.png)`
      );
    } finally {
      api.stop();
    }
  });

  test("update-comment --attach appends markdown link to comment content", async () => {
    const api = await startFakeApi();
    const png = writeTmpFile("after.png", "P");
    try {
      const r = await runCliAsync(
        [
          "issues",
          "update-comment",
          "comment-1",
          "now with a screenshot",
          "--attach",
          png,
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_update"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.p_content).toBe(
        `now with a screenshot\n\n![after.png](${api.storageBaseUrl}/files/test/0_after.png)`
      );
    } finally {
      api.stop();
    }
  });

  test("--attach with a missing file fails fast and never sends the comment-create request", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        [
          "issues",
          "post-comment",
          "11111111-1111-1111-1111-111111111111",
          "should not be posted",
          "--attach",
          "/tmp/this-path-does-not-exist-xyz123.png",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toMatch(/File not found/);
      // No comment-create request reached the server — we bailed before posting.
      const commentReq = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(commentReq).toBeFalsy();
    } finally {
      api.stop();
    }
  });

  test("post-comment without --attach still works (regression check)", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        [
          "issues",
          "post-comment",
          "11111111-1111-1111-1111-111111111111",
          "plain comment",
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_STORAGE_BASE_URL: api.storageBaseUrl,
        })
      );
      expect(r.status).toBe(0);
      expect(api.uploads).toHaveLength(0);
      const req = api.requests.find((x) => x.pathname.endsWith("/rpc/issue_comment_create"));
      expect(req).toBeTruthy();
      expect(req!.bodyJson.content).toBe("plain comment");
    } finally {
      api.stop();
    }
  });
});
