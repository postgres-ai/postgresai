import { describe, test, expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import { tmpdir } from "os";
import { normalizeOrgAlias } from "../lib/org-scope";

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

// printResult prints YAML on a TTY and JSON otherwise, so a --json test run
// through plain pipes passes whether or not the flag is read. `script` gives
// the CLI a real pseudo-terminal (util-linux on CI, BSD on macOS).
async function runCliOnTty(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const argv = [process.execPath || "bun", cliPath, ...args];
  const quoted = argv.map((a) => `'${a.replace(/'/g, `'\\''`)}'`).join(" ");
  const cmd = process.platform === "darwin"
    ? ["script", "-q", "/dev/null", ...argv]
    : ["script", "-qec", quoted, "/dev/null"];
  const proc = Bun.spawn(cmd, {env: {...process.env, ...env}, stdin: "ignore", stdout: "pipe", stderr: "pipe"});
  const [output, status] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
  return {status, output: output.replace(/\r/g, "")};
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


describe("pgai orgs create", () => {
  test("creates the first org without a selector and preserves config", async () => {
    const requests: Array<{headers: Headers; body: unknown}> = [];
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
      expect(new URL(req.url).pathname).toEndWith("/rpc/user_create_org");
      requests.push({headers: req.headers, body: await req.json()});
      return Response.json({id: 42, alias: "new-team", name: "New team"});
    }});
    try {
      const env = isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}/`, PGAI_ORG: "old-team"});
      const r = await runCli(["orgs", "create", "--name", "New team", "--alias", "new-team", "--json"], env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toEqual({id: 42, alias: "new-team", name: "New team"});
      expect(requests).toHaveLength(1);
      expect(requests[0]!.body).toEqual({name: "New team", alias: "new-team"});
      expect(requests[0]!.headers.get("access-token")).toBe(GLOBAL_TOKEN);
      expect(requests[0]!.headers.get("x-pgai-org")).toBeNull();
      expect(readStoredConfig(env)).toEqual({});
    } finally { server.stop(true); }
  });
  test("reports server duplicate/auth errors without leaking secrets", async () => {
    const server = Bun.serve({hostname:"127.0.0.1", port:0, fetch() {
      return Response.json({message:"Conflict", details:"Organization alias already exists"}, {status:409});
    }});
    try {
      const r = await runCli(["orgs", "create", "--name", "Team"], isolatedEnv({PGAI_API_KEY:GLOBAL_TOKEN, PGAI_API_BASE_URL:`http://127.0.0.1:${server.port}/`}));
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("409");
      expect(r.stderr).toContain("Organization alias already exists");
      expect(r.stderr).not.toContain(GLOBAL_TOKEN);
      expect(r.stdout).toBe("");
    } finally { server.stop(true); }
  });
  test("refuses a non-global token before calling the server", async () => {
    let calls = 0;
    const server = Bun.serve({hostname:"127.0.0.1", port:0, fetch() { calls++; return Response.json({id: 1, alias: "x"}); }});
    try {
      const r = await runCli(["orgs", "create", "--name", "Team"], isolatedEnv({PGAI_API_KEY:"legacy-per-org-key", PGAI_API_BASE_URL:`http://127.0.0.1:${server.port}/`}));
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("requires a global user token");
      expect(r.stderr).not.toContain("legacy-per-org-key");
      expect(calls).toBe(0);
    } finally { server.stop(true); }
  });
  async function createServer(reply: () => Response = () => Response.json({id: 7, alias: "team", name: "Team"})) {
    const requests: Array<{headers: Headers; body: unknown}> = [];
    const server = Bun.serve({hostname: "127.0.0.1", port: 0, async fetch(req) {
      requests.push({headers: req.headers, body: await req.json()});
      return reply();
    }});
    return {server, requests, url: `http://127.0.0.1:${server.port}/`};
  }

  test("sends alias null when --alias is omitted, and trims the name", async () => {
    const api = await createServer();
    try {
      const r = await runCli(["orgs", "create", "--name", "  Team  "], isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url}));
      expect(r.status).toBe(0);
      expect(api.requests).toHaveLength(1);
      expect(api.requests[0]!.body).toEqual({name: "Team", alias: null});
    } finally { api.server.stop(true); }
  });

  test("normalizes --alias like the server: lowercase, reserved characters to '-'", async () => {
    const api = await createServer();
    try {
      const r = await runCli(["orgs", "create", "--name", "Team", "--alias", " ACME Team/x "], isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url}));
      expect(r.status).toBe(0);
      expect(api.requests[0]!.body).toEqual({name: "Team", alias: "acme-team-x"});
    } finally { api.server.stop(true); }
  });

  test("refuses blank or unusable names and aliases without a request", async () => {
    const api = await createServer();
    try {
      const env = isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url});
      const cases: Array<[string[], string]> = [
        [["--name", "   "], "name must not be empty"],
        [["--name", "Team", "--alias", "  "], "alias must not be empty"],
        [["--name", "Team", "--alias", "!!!"], "alias must not be empty after normalization"],
        [["--name", "!!!"], "alias must not be empty after normalization"],
        [["--name", "Team", "--alias", "моя-команда"], "ASCII"],
        // QA r1: the server derives 'café-démo' from the name; --org cannot send it.
        [["--name", "Café Démo"], "Pass --alias with ASCII"],
      ];
      for (const [args, message] of cases) {
        const r = await runCli(["orgs", "create", ...args], env);
        expect(r.status).toBe(1);
        expect(r.stderr).toContain(message);
        expect(r.stdout).toBe("");
      }
      expect(api.requests).toHaveLength(0);
    } finally { api.server.stop(true); }
  });

  test("rejects an invalid success response", async () => {
    const bodies = ["{}", "[]", "null", JSON.stringify({id: 1}), JSON.stringify({id: 1, alias: ""}), "<html>oops</html>",
      JSON.stringify({alias: "x"}), JSON.stringify({id: null, alias: "x"}), JSON.stringify({id: 0, alias: "x"}),
      JSON.stringify({id: 1, alias: 5})];
    for (const body of bodies) {
      const api = await createServer(() => new Response(body, {headers: {"Content-Type": "application/json"}}));
      try {
        const r = await runCli(["orgs", "create", "--name", "Team"], isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url}));
        expect(r.status).toBe(1);
        expect(r.stderr).toContain("Create organization returned an invalid response.");
        expect(r.stdout).toBe("");
      } finally { api.server.stop(true); }
    }
  });

  test("sends no org selector and leaves a seeded config byte-identical", async () => {
    const api = await createServer();
    try {
      const env = isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url, PGAI_ORG: "old-team", PGAI_ORG_ID: "5225"});
      const dir = resolve(env.XDG_CONFIG_HOME!, "postgresai");
      mkdirSync(dir, {recursive: true});
      const seeded = JSON.stringify({apiKey: GLOBAL_TOKEN, orgId: 5225, defaultProject: "prod-project"}, null, 2) + "\n";
      writeFileSync(resolve(dir, "config.json"), seeded);
      const r = await runCli(["orgs", "create", "--name", "Team"], env);
      expect(r.status).toBe(0);
      expect(api.requests[0]!.headers.get("x-pgai-org")).toBeNull();
      expect(api.requests[0]!.headers.get("x-pgai-org-id")).toBeNull();
      expect(readFileSync(resolve(dir, "config.json"), "utf8")).toBe(seeded);
    } finally { api.server.stop(true); }
  });

  test("normalizeOrgAlias matches get_alias on reserved runs, edges and case", () => {
    const cases: Array<[string, string]> = [
      ["a..b", "a-b"],
      ["a~b", "a-b"],
      ["a -- b", "a-b"],
      ["--x--", "x"],
      ["My_Team", "my_team"],
    ];
    for (const [input, expected] of cases) expect([input, normalizeOrgAlias(input)]).toEqual([input, expected]);
  });

  test("--json prints JSON on a TTY, given to create or to orgs", async () => {
    const api = await createServer();
    try {
      const env = isolatedEnv({PGAI_API_KEY: GLOBAL_TOKEN, PGAI_API_BASE_URL: api.url, PGAI_NO_FEEDBACK_TIP: "1"});
      // Control: without --json the TTY gets YAML, so the pty is real.
      const plain = await runCliOnTty(["orgs", "create", "--name", "T"], env);
      expect(plain.status).toBe(0);
      expect(plain.output).toContain("alias: team");
      for (const args of [["orgs", "create", "--name", "T", "--json"], ["orgs", "--json", "create", "--name", "T"]]) {
        const r = await runCliOnTty(args, env);
        expect(r.status).toBe(0);
        const start = r.output.indexOf("{");
        const end = r.output.lastIndexOf("}");
        expect(start).toBeGreaterThanOrEqual(0);
        expect(JSON.parse(r.output.slice(start, end + 1))).toEqual({id: 7, alias: "team", name: "Team"});
      }
    } finally { api.server.stop(true); }
  }, 60000);

  test("help names the options and the token requirement", async () => {
    const r = await runCli(["orgs", "create", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("--name <name>");
    expect(r.stdout).toContain("--alias <alias>");
    expect(r.stdout).toContain("global user token");
    expect(r.stdout).toContain("All my organizations (global token)");
    expect(r.stdout).toContain("https://console.postgres.ai/profile/global-tokens");
  });

  test("requires credentials and a name", async () => {
    const noKey = await runCli(["orgs", "create", "--name", "Team"], isolatedEnv());
    expect(noKey.status).toBe(1);
    expect(noKey.stderr).toContain("API key is required");
    const noName = await runCli(["orgs", "create"], isolatedEnv());
    expect(noName.status).toBe(1);
    expect(noName.stderr).toContain("--name");
  });
});
