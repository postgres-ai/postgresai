import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
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

// Async spawn — MUST be used for any test that hits the in-process fake server:
// Bun.spawnSync would block this process's event loop, so the fake `Bun.serve`
// could never answer the subprocess's request (deadlock).
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
  const cfgHome = mkdtempSync(resolve(tmpdir(), "postgresai-dblab-test-"));
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome, ...extra };
}

interface DblabRequestBody {
  instance_id?: unknown;
  action?: unknown;
  method?: unknown;
  data?: Record<string, unknown>;
}

interface Recorded {
  method: string;
  pathname: string;
  search: string;
  bodyJson: DblabRequestBody;
}

/**
 * Fake Platform API: serves the `/rpc/projects_list` resolver listing (rows
 * carry `dblab_instance_id`) and echoes every `/rpc/dblab_api_call` proxy call
 * so tests can assert the action/method/payload the CLI produced.
 */
async function startFakeApi(projects?: unknown[]) {
  const requests: Recorded[] = [];
  const rows = projects ?? [
    { project_id: 12, alias: "main-db", name: "Main DB", joe_ready: true, tunnel: false, instance_id: 1, dblab_instance_id: 7 },
  ];

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const bodyText = req.method === "POST" ? await req.text() : "";
      let bodyJson: DblabRequestBody = {};
      try { bodyJson = bodyText ? (JSON.parse(bodyText) as DblabRequestBody) : {}; } catch { bodyJson = {}; }
      requests.push({ method: req.method, pathname: url.pathname, search: url.search, bodyJson });

      if (req.method === "POST" && url.pathname.endsWith("/rpc/projects_list")) {
        return new Response(JSON.stringify(rows), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (req.method === "POST" && url.pathname.endsWith("/rpc/dblab_api_call")) {
        // Echo a plausible reply keyed off the action.
        const action = String(bodyJson?.action ?? "");
        const reply = action === "/clones" || action.startsWith("/branches") || action.startsWith("/snapshots") || action.endsWith("/log")
          ? []
          : { instance_id: bodyJson?.instance_id, action, method: bodyJson?.method, ok: true };
        return new Response(JSON.stringify(reply), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response("not found", { status: 404 });
    },
  });

  const baseUrl = `http://${server.hostname}:${server.port}/api/general`;
  return {
    baseUrl,
    requests,
    proxyCalls: () => requests.filter((r) => r.pathname.endsWith("/rpc/dblab_api_call")),
    resolverCalls: () => requests.filter((r) => r.pathname.endsWith("/rpc/projects_list")),
    stop: () => server.stop(true),
  };
}

describe("CLI DBLab companion command groups (grouped under `pgai dblab …`)", () => {
  test("top-level help lists the dblab group", () => {
    const r = runCli(["--help"], isolatedEnv());
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("dblab");
  });

  test("dblab help exposes the clone, branch, and snapshot groups", () => {
    const r = runCli(["dblab", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("clone");
    expect(out).toContain("branch");
    expect(out).toContain("snapshot");
  });

  test("clone help exposes create/list/status/reset/destroy", () => {
    const r = runCli(["dblab", "clone", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("create");
    expect(out).toContain("list");
    expect(out).toContain("status");
    expect(out).toContain("reset");
    expect(out).toContain("destroy");
  });

  test("branch help exposes list/create/delete/log", () => {
    const r = runCli(["dblab", "branch", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("list");
    expect(out).toContain("create");
    expect(out).toContain("delete");
    expect(out).toContain("log");
  });

  test("snapshot help exposes list/create/destroy", () => {
    const r = runCli(["dblab", "snapshot", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("list");
    expect(out).toContain("create");
    expect(out).toContain("destroy");
  });

  test("clone create fails fast without an API key", () => {
    const r = runCli(["dblab", "clone", "create", "--project", "main-db"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("clone list without --project errors", () => {
    const r = runCli(["dblab", "clone", "list"], isolatedEnv({ PGAI_API_KEY: "test-key" }));
    expect(r.status).not.toBe(0);
    expect(`${r.stdout}\n${r.stderr}`.toLowerCase()).toContain("project");
  });

  test("clone create resolves the alias then proxies /clone POST", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "create", "--project", "main-db", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      // The resolver call happened.
      expect(api.resolverCalls().length).toBe(1);
      // Exactly one proxy call, with the right action/method/instance.
      const proxied = api.proxyCalls();
      expect(proxied.length).toBe(1);
      expect(proxied[0].bodyJson.instance_id).toBe("7");
      expect(proxied[0].bodyJson.action).toBe("/clone");
      expect(proxied[0].bodyJson.method).toBe("post");
      expect(proxied[0].bodyJson.data).toEqual({ protected: false });
    } finally {
      api.stop();
    }
  });

  test("clone create reads DB credentials from the environment, not argv", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "create", "--project", "main-db", "--db-user", "clone_u", "--json"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_CLONE_DB_PASSWORD: "env-only-secret",
        })
      );
      expect(r.status).toBe(0);
      expect(api.proxyCalls()[0].bodyJson.data?.db).toEqual({
        username: "clone_u",
        password: "env-only-secret",
      });
    } finally {
      api.stop();
    }
  });

  test("clone create maps branch/snapshot/protected into the payload", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "create", "--project", "main-db", "--branch", "feature-idx", "--snapshot", "s-9", "--protected", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.data).toEqual({ protected: true, branch: "feature-idx", snapshot: { id: "s-9" } });
    } finally {
      api.stop();
    }
  });

  test("clone reset proxies /clone/<id>/reset POST", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "reset", "c-8f21", "--project", "main-db", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/clone/c-8f21/reset");
      expect(body.method).toBe("post");
      expect(body.data).toEqual({ latest: true });
    } finally {
      api.stop();
    }
  });

  test("clone reset --snapshot maps snapshotID and disables latest", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "reset", "c-8f21", "--project", "main-db", "--snapshot", "s-9", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/clone/c-8f21/reset");
      expect(body.data).toEqual({ latest: false, snapshotID: "s-9" });
    } finally {
      api.stop();
    }
  });

  test("clone destroy proxies /clone/<id> DELETE and resolves a numeric project id", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "destroy", "c-8f21", "--project", "12", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.instance_id).toBe("7"); // resolved from project_id 12
      expect(body.action).toBe("/clone/c-8f21");
      expect(body.method).toBe("delete");
    } finally {
      api.stop();
    }
  });

  test("branch create proxies /branch POST with branchName + snapshotID", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "branch", "create", "feature-idx", "--project", "main-db", "--snapshot", "latest", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/branch");
      expect(body.method).toBe("post");
      expect(body.data).toEqual({ branchName: "feature-idx", snapshotID: "latest" });
    } finally {
      api.stop();
    }
  });

  test("branch log proxies /branch/<name>/log GET", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "branch", "log", "feature-idx", "--project", "main-db", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/branch/feature-idx/log");
      expect(body.method).toBe("get");
    } finally {
      api.stop();
    }
  });

  test("snapshot list proxies /snapshots GET", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "snapshot", "list", "--project", "main-db", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/snapshots");
      expect(body.method).toBe("get");
    } finally {
      api.stop();
    }
  });

  test("snapshot create proxies /branch/snapshot POST with cloneID", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "snapshot", "create", "--project", "main-db", "--clone", "c-8f21", "--message", "before idx", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/branch/snapshot");
      expect(body.data).toEqual({ cloneID: "c-8f21", message: "before idx" });
    } finally {
      api.stop();
    }
  });

  test("snapshot destroy proxies /snapshot/<id>?force=<bool> DELETE", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "snapshot", "destroy", "s-1", "--project", "main-db", "--force", "--json"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      const body = api.proxyCalls()[0].bodyJson;
      expect(body.action).toBe("/snapshot/s-1?force=true");
      expect(body.method).toBe("delete");
    } finally {
      api.stop();
    }
  });

  test("an unknown project alias exits 1 with a helpful message (no proxy call)", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["dblab", "clone", "list", "--project", "does-not-exist"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toContain("No DBLab instance found");
      expect(api.proxyCalls().length).toBe(0);
    } finally {
      api.stop();
    }
  });

  test("a backend role-gate error (403) surfaces and exits 1", async () => {
    // Fake server that 403s the proxy call to emulate the destructive-verb
    // gate: the token owner lacks Admin/AllFeaturesUser in the token org.
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      async fetch(req) {
        const url = new URL(req.url);
        if (req.method === "POST" && url.pathname.endsWith("/rpc/projects_list")) {
          return new Response(JSON.stringify([{ project_id: 12, alias: "main-db", dblab_instance_id: 7 }]), { status: 200 });
        }
        return new Response('{"message":"Forbidden: this token\'s owner lacks the Admin or AllFeaturesUser role required for destructive DBLab operations (clone/snapshot/branch destroy)"}', { status: 403 });
      },
    });
    try {
      const baseUrl = `http://${server.hostname}:${server.port}/api/general`;
      const r = await runCliAsync(
        ["dblab", "clone", "destroy", "c-1", "--project", "main-db"],
        isolatedEnv({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: baseUrl })
      );
      expect(r.status).toBe(1);
      expect(`${r.stdout}\n${r.stderr}`).toContain("Failed to destroy clone");
    } finally {
      server.stop(true);
    }
  });
});

// ---------------------------------------------------------------------------
// --help must accurately reflect the backend's destructive-verb gate: the HTTP
// DELETE verbs (clone destroy, branch delete, snapshot destroy) require the
// token owner to hold the Admin or AllFeaturesUser role in the token org
// (v1.dblab_api_call mirrors the v1.joe_command_run role gate); clone RESET
// (a POST) does NOT.
// ---------------------------------------------------------------------------
describe("dblab --help role-gate annotations match the backend gate", () => {
  test("clone destroy --help notes the Admin/AllFeaturesUser requirement", () => {
    const r = runCli(["dblab", "clone", "destroy", "--help"], isolatedEnv());
    expect(`${r.stdout}\n${r.stderr}`).toMatch(/Admin or AllFeaturesUser/);
  });

  test("branch delete --help notes the Admin/AllFeaturesUser requirement", () => {
    const r = runCli(["dblab", "branch", "delete", "--help"], isolatedEnv());
    expect(`${r.stdout}\n${r.stderr}`).toMatch(/Admin or AllFeaturesUser/);
  });

  test("snapshot destroy --help notes the Admin/AllFeaturesUser requirement", () => {
    const r = runCli(["dblab", "snapshot", "destroy", "--help"], isolatedEnv());
    expect(`${r.stdout}\n${r.stderr}`).toMatch(/Admin or AllFeaturesUser/);
  });

  test("clone reset --help does NOT claim a role requirement (reset is a POST, ungated)", () => {
    const r = runCli(["dblab", "clone", "reset", "--help"], isolatedEnv());
    expect(`${r.stdout}\n${r.stderr}`).not.toMatch(/Admin or AllFeaturesUser/);
  });
});
