import { describe, test, expect } from "bun:test";
import { existsSync, mkdtempSync, readFileSync } from "fs";
import { resolve } from "path";
import { tmpdir } from "os";

/**
 * `pgai orgs` is the discovery command every global-token user is sent to --
 * by the CLI's own "needs an organization" hint and by the server's PT403 --
 * so an empty or broken result strands them with no way to name an org.
 */

const GLOBAL_TOKEN = `pai_global_${"a".repeat(43)}`;

// Must be async: Bun.spawnSync blocks this process's event loop, so the
// in-process fake API below would never get to answer and every request would
// time out. The repo's other CLI tests carry a runCliAsync for the same reason.
async function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = process.execPath && process.execPath.length > 0 ? process.execPath : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, status] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { status, stdout, stderr };
}

function isolatedEnv(extra: Record<string, string> = {}) {
  const cfgHome = mkdtempSync(resolve(tmpdir(), "postgresai-orgs-test-"));
  // PGAI_API_KEY is cleared per-case: the developer's real environment must not
  // leak a key into a test asserting the no-key path.
  return { XDG_CONFIG_HOME: cfgHome, HOME: cfgHome, PGAI_API_KEY: "", ...extra };
}

async function startFakeApi(body: unknown, status = 200) {
  const requests: Array<{ pathname: string; headers: Record<string, string> }> = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const headers: Record<string, string> = {};
      for (const [k, v] of req.headers.entries()) headers[k.toLowerCase()] = v;
      requests.push({ pathname: url.pathname, headers });
      if (url.pathname.endsWith("/rpc/orgs_list")) {
        return new Response(typeof body === "string" ? body : JSON.stringify(body), {
          status,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return {
    baseUrl: `http://127.0.0.1:${server.port}/`,
    requests,
    stop: () => server.stop(true),
  };
}

function readStoredConfig(env: Record<string, string>): Record<string, unknown> {
  const path = resolve(env.XDG_CONFIG_HOME!, "postgresai", "config.json");
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
}

const ORGS = [
  { org_id: 1, alias: "acme", name: "Acme", is_active: true },
  { org_id: 5225, alias: "globex", name: "Globex", is_active: true },
];

describe("pgai orgs", () => {
  test("lists the reachable organizations", async () => {
    const api = await startFakeApi(ORGS);
    try {
      const r = await runCli(
        ["orgs"],
        isolatedEnv({ PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("acme");
      expect(r.stdout).toContain("globex");
    } finally {
      api.stop();
    }
  });

  test("--json emits the raw rows", async () => {
    const api = await startFakeApi(ORGS);
    try {
      const r = await runCli(
        ["orgs", "--json"],
        isolatedEnv({ PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toEqual(ORGS);
    } finally {
      api.stop();
    }
  });

  test("a selector in the environment neither breaks nor filters discovery", async () => {
    const api = await startFakeApi(ORGS);
    try {
      // PGAI_ORG is routinely exported in agent/CI shells, and `orgs` must keep
      // working there. This does NOT prove the no-leak property: `orgs` is not
      // org-scoped, so the preAction hook clears the scope before it runs and no
      // header could be added anyway. The real guard -- listOrgs sending no
      // header while a scope IS active, the case `issues create --org <alias>`
      // hits via resolveOrgIdForBody -- lives in org-scope-wire.test.ts.
      await runCli(
        ["orgs"],
        isolatedEnv({
          PGAI_API_KEY: GLOBAL_TOKEN,
          PGAI_API_BASE_URL: api.baseUrl,
          PGAI_ORG: "acme",
        })
      );
      const call = api.requests.find((r) => r.pathname.endsWith("/rpc/orgs_list"));
      expect(call).toBeDefined();
      expect(call!.headers["x-pgai-org"]).toBeUndefined();
      expect(call!.headers["x-pgai-org-id"]).toBeUndefined();
    } finally {
      api.stop();
    }
  });

  test("an empty list fails loudly rather than printing nothing", async () => {
    const api = await startFakeApi([]);
    try {
      const r = await runCli(
        ["orgs"],
        isolatedEnv({ PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).not.toBe(0);
      expect(r.stderr).toMatch(/cannot reach any organization/i);
    } finally {
      api.stop();
    }
  });

  test("a server error is reported, not swallowed", async () => {
    const api = await startFakeApi({ message: "boom" }, 500);
    try {
      const r = await runCli(
        ["orgs"],
        isolatedEnv({ PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.baseUrl })
      );
      expect(r.status).not.toBe(0);
      expect(r.stderr.length).toBeGreaterThan(0);
    } finally {
      api.stop();
    }
  });

  test("without an API key it says how to get one", async () => {
    const r = await runCli(["orgs"], isolatedEnv());
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/API key is required/i);
  });

  test("takes no --org flag", async () => {
    const r = await runCli(["orgs", "--help"], isolatedEnv());
    expect(r.stdout).not.toContain("--org ");
    expect(r.stdout).not.toContain("--org-id");
  });
});

/**
 * The two non-OAuth entry points that classify a token. Both are independent
 * code paths from the browser flow, and both must refuse to keep org state that
 * a global token cannot honour.
 */
describe("global tokens outside the OAuth flow", () => {
  test("auth --set-key <global> clears a stale orgId and defaultProject", async () => {
    const env = isolatedEnv();
    // Seed a per-org login, then replace it with a global token.
    await runCli(["auth", "--set-key", "legacy-per-org-key"], env);
    await runCli(["set-default-project", "prod-project"], env);
    expect(readStoredConfig(env)).toMatchObject({ defaultProject: "prod-project" });

    const r = await runCli(["auth", "--set-key", GLOBAL_TOKEN], env);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/global token/i);

    // Assert on the FILE, not on show-key: show-key tests isGlobalToken first,
    // so it never prints "Organization ID" for a global key and would stay
    // green with the stale value still on disk.
    const stored = readStoredConfig(env);
    expect(stored).toMatchObject({ apiKey: GLOBAL_TOKEN });
    expect(stored).not.toHaveProperty("orgId");
    expect(stored).not.toHaveProperty("defaultProject");

    const shown = await runCli(["auth", "show-key"], env);
    expect(shown.stdout).toMatch(/all organizations you belong to/i);
  });

  test("auth show-key reports per-org scope for a per-org token", async () => {
    const env = isolatedEnv();
    await runCli(["auth", "--set-key", "legacy-per-org-key"], env);
    const shown = await runCli(["auth", "show-key"], env);
    expect(shown.stdout).not.toMatch(/all organizations you belong to/i);
  });
});
