import { describe, test, expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";

import * as util from "../lib/util";
import * as pkce from "../lib/pkce";
import * as authServer from "../lib/auth-server";

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

describe("URL resolution", () => {
  test("resolveBaseUrls returns correct production defaults", () => {
    const result = util.resolveBaseUrls();
    expect(result.apiBaseUrl).toBe("https://postgres.ai/api/general");
    expect(result.uiBaseUrl).toBe("https://console.postgres.ai");
  });

  test("resolveBaseUrls strips trailing slashes", () => {
    const result = util.resolveBaseUrls({
      apiBaseUrl: "https://example.com/api/",
      uiBaseUrl: "https://example.com/",
    });
    expect(result.apiBaseUrl).toBe("https://example.com/api");
    expect(result.uiBaseUrl).toBe("https://example.com");
  });

  test("resolveBaseUrls respects environment variables", () => {
    const originalApiUrl = process.env.PGAI_API_BASE_URL;
    const originalUiUrl = process.env.PGAI_UI_BASE_URL;

    try {
      process.env.PGAI_API_BASE_URL = "https://custom-api.example.com/api/";
      process.env.PGAI_UI_BASE_URL = "https://custom-ui.example.com/";

      const result = util.resolveBaseUrls();
      expect(result.apiBaseUrl).toBe("https://custom-api.example.com/api");
      expect(result.uiBaseUrl).toBe("https://custom-ui.example.com");
    } finally {
      if (originalApiUrl === undefined) {
        delete process.env.PGAI_API_BASE_URL;
      } else {
        process.env.PGAI_API_BASE_URL = originalApiUrl;
      }
      if (originalUiUrl === undefined) {
        delete process.env.PGAI_UI_BASE_URL;
      } else {
        process.env.PGAI_UI_BASE_URL = originalUiUrl;
      }
    }
  });

  test("resolveBaseUrls prefers CLI options over env vars", () => {
    const originalApiUrl = process.env.PGAI_API_BASE_URL;

    try {
      process.env.PGAI_API_BASE_URL = "https://env.example.com/api/";

      const result = util.resolveBaseUrls({
        apiBaseUrl: "https://cli-option.example.com/api/",
      });
      expect(result.apiBaseUrl).toBe("https://cli-option.example.com/api");
    } finally {
      if (originalApiUrl === undefined) {
        delete process.env.PGAI_API_BASE_URL;
      } else {
        process.env.PGAI_API_BASE_URL = originalApiUrl;
      }
    }
  });

  test("resolveBaseUrls uses config baseUrl for API", () => {
    const result = util.resolveBaseUrls({}, { baseUrl: "https://config.example.com/api/" });
    expect(result.apiBaseUrl).toBe("https://config.example.com/api");
    // UI should still use default since config doesn't have uiBaseUrl
    expect(result.uiBaseUrl).toBe("https://console.postgres.ai");
  });

  test("normalizeBaseUrl throws on invalid URL", () => {
    expect(() => util.normalizeBaseUrl("not-a-url")).toThrow(/Invalid base URL/);
  });

  test("normalizeBaseUrl accepts valid URLs", () => {
    expect(util.normalizeBaseUrl("https://example.com")).toBe("https://example.com");
    expect(util.normalizeBaseUrl("https://example.com/")).toBe("https://example.com");
    expect(util.normalizeBaseUrl("https://example.com/api/")).toBe("https://example.com/api");
  });
});

describe("PKCE module", () => {
  test("generateCodeVerifier returns correct length string", () => {
    const verifier = pkce.generateCodeVerifier();
    expect(typeof verifier).toBe("string");
    expect(verifier.length).toBeGreaterThanOrEqual(43);
    expect(verifier.length).toBeLessThanOrEqual(128);
  });

  test("generateCodeChallenge returns base64url encoded SHA256", () => {
    const verifier = pkce.generateCodeVerifier();
    const challenge = pkce.generateCodeChallenge(verifier);
    expect(typeof challenge).toBe("string");
    expect(challenge.length).toBeGreaterThan(0);
    // Base64url encoding should not contain + or / characters
    expect(challenge).not.toMatch(/[+/]/);
  });

  test("generateState returns random string", () => {
    const state1 = pkce.generateState();
    const state2 = pkce.generateState();
    expect(typeof state1).toBe("string");
    expect(state1.length).toBeGreaterThan(0);
    expect(state1).not.toBe(state2); // Should be random
  });

  test("generatePKCEParams returns all required parameters", () => {
    const params = pkce.generatePKCEParams();
    expect(params.codeVerifier).toBeTruthy();
    expect(params.codeChallenge).toBeTruthy();
    expect(params.codeChallengeMethod).toBe("S256");
    expect(params.state).toBeTruthy();
  });
});

describe("Auth callback server", () => {
  test("createCallbackServer returns correct interface", () => {
    const server = authServer.createCallbackServer(0, "test-state", 1000);
    expect(server.server).toBeTruthy();
    expect(server.server.stop).toBeInstanceOf(Function);
    expect(server.promise).toBeInstanceOf(Promise);
    expect(server.ready).toBeInstanceOf(Promise);
    expect(server.getPort).toBeInstanceOf(Function);

    // Clean up
    server.server.stop();
  });

  test("createCallbackServer binds to a port", async () => {
    const server = authServer.createCallbackServer(0, "test-state", 5000);
    const port = await server.ready;
    expect(typeof port).toBe("number");
    expect(port).toBeGreaterThan(0);

    // Clean up
    server.server.stop();
  });

  test("createCallbackServer responds to callback requests", async () => {
    const testState = "test-state-" + Math.random().toString(36).substring(7);
    const server = authServer.createCallbackServer(0, testState, 5000);
    const port = await server.ready;

    // Simulate OAuth callback
    const testCode = "test-auth-code";
    const callbackUrl = `http://127.0.0.1:${port}/callback?code=${testCode}&state=${testState}`;

    const fetchPromise = fetch(callbackUrl);
    const result = await server.promise;

    expect(result.code).toBe(testCode);
    expect(result.state).toBe(testState);

    // Check response
    const response = await fetchPromise;
    expect(response.status).toBe(200);
    const text = await response.text();
    expect(text).toMatch(/Authentication successful/);
  });

  test("createCallbackServer rejects on state mismatch", async () => {
    const server = authServer.createCallbackServer(0, "expected-state", 5000);
    const port = await server.ready;

    const callbackUrl = `http://127.0.0.1:${port}/callback?code=test-code&state=wrong-state`;

    const fetchPromise = fetch(callbackUrl);

    await expect(server.promise).rejects.toThrow(/State mismatch/);

    const response = await fetchPromise;
    expect(response.status).toBe(400);
  });

  test("createCallbackServer handles OAuth errors", async () => {
    const server = authServer.createCallbackServer(0, "test-state", 5000);
    const port = await server.ready;

    const callbackUrl = `http://127.0.0.1:${port}/callback?error=access_denied&error_description=User%20denied%20access`;

    const fetchPromise = fetch(callbackUrl);

    await expect(server.promise).rejects.toThrow(/OAuth error: access_denied/);

    const response = await fetchPromise;
    expect(response.status).toBe(400);
  });

  test("createCallbackServer times out", async () => {
    const server = authServer.createCallbackServer(0, "test-state", 100); // 100ms timeout
    await server.ready;

    await expect(server.promise).rejects.toThrow(/timeout/i);
  });
});

describe("CLI auth commands", () => {
  test("cli: login --help shows all options", () => {
    const r = runCli(["login", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--set-key/);
    expect(r.stdout).toMatch(/--debug/);
  });

  test("cli: login --set-key aliases auth login", () => {
    const home = mkdtempSync(join(tmpdir(), "postgresai-login-"));

    try {
      const r = runCli(["login", "--set-key", "test-token"], {
        HOME: home,
        XDG_CONFIG_HOME: join(home, ".config"),
      });
      expect(r.status).toBe(0);
      expect(r.stdout).toMatch(/API key saved/);

      const saved = JSON.parse(
        readFileSync(join(home, ".config", "postgresai", "config.json"), "utf8")
      );
      expect(saved.apiKey).toBe("test-token");
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  test("cli: auth login --help shows all options", () => {
    const r = runCli(["auth", "login", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/--set-key/);
    expect(r.stdout).toMatch(/--debug/);
  });

  test("cli: auth show-key --help works", () => {
    const r = runCli(["auth", "show-key", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/show.*key/i);
  });

  test("cli: auth remove-key --help works", () => {
    const r = runCli(["auth", "remove-key", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/remove.*key/i);
  });
});

describe("maskSecret utility", () => {
  test("masks short secrets completely", () => {
    expect(util.maskSecret("abc")).toBe("****");
    expect(util.maskSecret("12345678")).toBe("****");
  });

  test("masks medium secrets with visible ends", () => {
    const masked = util.maskSecret("1234567890123456");
    // maskSecret shows first 4 chars, middle masked, last 4 chars for 16-char strings
    expect(masked).toMatch(/^1234\*+3456$/);
  });

  test("masks long secrets appropriately", () => {
    const secret = "abcdefghij1234567890klmnopqrstuvwxyz";
    const masked = util.maskSecret(secret);
    expect(masked.startsWith("abcdefghij12")).toBe(true);
    expect(masked.endsWith("wxyz")).toBe(true);
    expect(masked).toMatch(/\*+/);
  });

  test("handles empty string", () => {
    expect(util.maskSecret("")).toBe("");
  });
});

describe("OAuth login flow hardening", () => {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";

  // Bun startup + the CLI's full import graph is heavy, so give slow/loaded CI
  // runners generous margins. PORT_WAIT_MS waits for the callback server to bind;
  // EXIT_WAIT_MS is the hard kill deadline. TEST_TIMEOUT_MS bounds each test and
  // stays comfortably above PORT_WAIT_MS + EXIT_WAIT_MS.
  const PORT_WAIT_MS = 20000;
  const EXIT_WAIT_MS = 20000;
  const TEST_TIMEOUT_MS = 60000;

  interface LoginFlowResult {
    exitCode: number;
    stdout: string;
    stderr: string;
    configAfter: Record<string, unknown> | null;
  }

  /**
   * Drives a full `pgai auth login` OAuth flow end-to-end against a local mock API:
   * 1. starts a mock backend serving /rpc/oauth_init and /rpc/oauth_token_exchange
   * 2. spawns the CLI with HOME/XDG_CONFIG_HOME pointing at a temp dir
   * 3. captures the PKCE state from the oauth_init request and the callback port
   *    from CLI stdout, then simulates the browser redirect to the callback server
   * 4. returns the CLI exit code, output, and the resulting config.json contents
   *
   * PATH for the CLI subprocess controls the browser-opener lookup:
   * - default: fake `open`/`xdg-open` executables that succeed without opening anything
   * - breakOpener: an empty dir, so spawning the opener fails with ENOENT
   */
  async function runLoginFlow(options: {
    exchangeResponse: { status: number; body: string };
    breakOpener?: boolean;
    existingConfig?: Record<string, unknown>;
  }): Promise<LoginFlowResult> {
    const home = mkdtempSync(join(tmpdir(), "postgresai-oauth-"));
    const configDir = join(home, ".config", "postgresai");
    mkdirSync(configDir, { recursive: true });
    const configPath = join(configDir, "config.json");
    if (options.existingConfig) {
      writeFileSync(configPath, JSON.stringify(options.existingConfig, null, 2) + "\n");
    }

    const binDir = join(home, "fakebin");
    mkdirSync(binDir);
    if (!options.breakOpener) {
      for (const name of ["open", "xdg-open"]) {
        writeFileSync(join(binDir, name), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
      }
    }

    let capturedState: string | null = null;
    const mockApi = Bun.serve({
      port: 0,
      hostname: "127.0.0.1",
      fetch: async (req) => {
        const url = new URL(req.url);
        if (req.method === "POST" && url.pathname.endsWith("/rpc/oauth_init")) {
          const body = (await req.json()) as { state?: string };
          capturedState = body.state ?? null;
          return Response.json({});
        }
        if (req.method === "POST" && url.pathname.endsWith("/rpc/oauth_token_exchange")) {
          return new Response(options.exchangeResponse.body, {
            status: options.exchangeResponse.status,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response("not found", { status: 404 });
      },
    });

    let proc: ReturnType<typeof Bun.spawn> | null = null;
    try {
      proc = Bun.spawn([bunBin, cliPath, "auth", "login"], {
        env: {
          ...process.env,
          HOME: home,
          XDG_CONFIG_HOME: join(home, ".config"),
          PGAI_API_BASE_URL: `http://127.0.0.1:${mockApi.port}/api/general`,
          PATH: binDir,
        },
        stdout: "pipe",
        stderr: "pipe",
      });

      // Accumulate stdout incrementally so we can react mid-flow
      let stdoutBuf = "";
      const stdoutDone = (async () => {
        const decoder = new TextDecoder();
        for await (const chunk of proc.stdout as ReadableStream<Uint8Array>) {
          stdoutBuf += decoder.decode(chunk, { stream: true });
        }
      })();
      const stderrPromise = new Response(proc.stderr as ReadableStream<Uint8Array>).text();

      // Wait until the CLI registered with the mock backend and printed its callback port,
      // or until it exits early (e.g. crashes before reaching the callback wait)
      let exitedEarly = false;
      proc.exited.then(() => { exitedEarly = true; });
      const deadline = Date.now() + PORT_WAIT_MS;
      let callbackPort: number | null = null;
      while (Date.now() < deadline) {
        const portMatch = stdoutBuf.match(/listening on port (\d+)/);
        if (portMatch && capturedState !== null) {
          callbackPort = parseInt(portMatch[1], 10);
          break;
        }
        if (exitedEarly) break;
        await Bun.sleep(50);
      }

      // Simulate the browser hitting the local callback server. If the CLI already
      // crashed, the connection is refused — swallow that so the test can assert on it.
      if (callbackPort !== null && capturedState !== null) {
        try {
          await fetch(
            `http://127.0.0.1:${callbackPort}/callback?code=test-auth-code&state=${encodeURIComponent(capturedState)}`
          );
        } catch {
          // callback server is gone; the CLI process must have exited
        }
      }

      let timedOut = false;
      const exitCode = await Promise.race([
        proc.exited,
        Bun.sleep(EXIT_WAIT_MS).then(() => {
          timedOut = true;
          proc?.kill();
          return -1;
        }),
      ]);
      await stdoutDone.catch(() => {});
      const stderr = await stderrPromise.catch(() => "");

      // A timeout surfaces as `exitCode === -1`; without context that reads as a
      // bare "expected 0, got -1". Emit the captured output so a slow-runner flake
      // is diagnosable rather than mysterious.
      if (timedOut) {
        console.error(
          `runLoginFlow: CLI did not exit within ${EXIT_WAIT_MS}ms; killed.\n` +
            `--- stdout ---\n${stdoutBuf}\n--- stderr ---\n${stderr}`
        );
      }

      const configAfter = existsSync(configPath)
        ? (JSON.parse(readFileSync(configPath, "utf8")) as Record<string, unknown>)
        : null;

      return { exitCode, stdout: stdoutBuf, stderr, configAfter };
    } finally {
      try { proc?.kill(); } catch { /* already exited */ }
      mockApi.stop(true);
      rmSync(home, { recursive: true, force: true });
    }
  }

  test("exchange response without api_token fails and preserves existing config", async () => {
    const existingConfig = {
      apiKey: "existing-key",
      orgId: 42,
      defaultProject: "prod-project",
    };

    const result = await runLoginFlow({
      // Valid JSON, HTTP 200, but no api_token (e.g. error object or schema change)
      exchangeResponse: { status: 200, body: JSON.stringify({ hint: "no token here" }) },
      existingConfig,
    });

    expect(result.exitCode).not.toBe(0);
    expect(result.stdout).not.toMatch(/Authentication successful/);
    expect(result.stderr).toMatch(/API token/i);
    // The pre-existing config must be untouched
    expect(result.configAfter).toEqual(existingConfig);
  }, TEST_TIMEOUT_MS);

  test("exchange response with a token but no org_id fails and preserves existing config", async () => {
    const existingConfig = {
      apiKey: "existing-key",
      orgId: 42,
      defaultProject: "prod-project",
    };

    const result = await runLoginFlow({
      // Valid token but no org_id: writing orgId=undefined would drop the stored
      // orgId and spuriously clear defaultProject (undefined counts as an org change).
      exchangeResponse: { status: 200, body: JSON.stringify({ api_token: "new-token-123" }) },
      existingConfig,
    });

    expect(result.exitCode).not.toBe(0);
    expect(result.stdout).not.toMatch(/Authentication successful/);
    expect(result.stderr).toMatch(/organization ID/i);
    // The pre-existing config (including orgId and defaultProject) must be untouched
    expect(result.configAfter).toEqual(existingConfig);
  }, TEST_TIMEOUT_MS);

  test("login continues when the browser opener cannot be spawned", async () => {
    const result = await runLoginFlow({
      exchangeResponse: {
        status: 200,
        body: JSON.stringify({ api_token: "new-token-123", org_id: 7 }),
      },
      // PATH contains no `open`/`xdg-open`, so spawning the opener emits ENOENT
      breakOpener: true,
    });

    // The flow must fall back to the manually printed URL and complete normally
    expect(result.stdout).toMatch(/Waiting for authorization/);
    expect(result.stdout).toMatch(/Authentication successful/);
    expect(result.exitCode).toBe(0);
    expect(result.configAfter?.apiKey).toBe("new-token-123");
    expect(result.configAfter?.orgId).toBe(7);
  }, TEST_TIMEOUT_MS);

  test("login accepts the PostgREST array-shape exchange response", async () => {
    // PostgREST caching can wrap the RPC result in an array; the CLI unwraps
    // `result[0].result`. Exercise that fallback for both api_token and org_id.
    const result = await runLoginFlow({
      exchangeResponse: {
        status: 200,
        body: JSON.stringify([{ result: { api_token: "array-token-456", org_id: 9 } }]),
      },
    });

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toMatch(/Authentication successful/);
    expect(result.configAfter?.apiKey).toBe("array-token-456");
    expect(result.configAfter?.orgId).toBe(9);
  }, TEST_TIMEOUT_MS);
  test("a global token logs in without an org_id and clears stale org state", async () => {
    // The inverse of "a token but no org_id fails": a global token legitimately
    // carries no org, so the guard must NOT reject it -- and writeConfig merges,
    // so a leftover orgId/defaultProject would silently re-scope every later
    // command to an org the user never named.
    const globalToken = `pai_global_${"a".repeat(43)}`;
    const result = await runLoginFlow({
      exchangeResponse: { status: 200, body: JSON.stringify({ api_token: globalToken }) },
      existingConfig: { apiKey: "old-per-org-key", orgId: 42, defaultProject: "prod-project" },
    });

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toMatch(/Authentication successful/);
    expect(result.stdout).toMatch(/global token/i);
    expect(result.stdout).toMatch(/have been cleared/i);
    expect(result.configAfter?.apiKey).toBe(globalToken);
    expect(result.configAfter).not.toHaveProperty("orgId");
    expect(result.configAfter).not.toHaveProperty("defaultProject");
  }, TEST_TIMEOUT_MS);

  test("a per-org token with no org_id still fails (the global path is not a bypass)", async () => {
    // Guards the shape of the check: it must key off the token PREFIX, not
    // merely off org_id being absent, or every malformed response would be
    // accepted as global.
    const result = await runLoginFlow({
      exchangeResponse: { status: 200, body: JSON.stringify({ api_token: "pai_not_global_xyz" }) },
      existingConfig: { apiKey: "existing-key", orgId: 42 },
    });

    expect(result.exitCode).not.toBe(0);
    expect(result.stderr).toMatch(/organization ID/i);
    expect(result.configAfter).toEqual({ apiKey: "existing-key", orgId: 42 });
  }, TEST_TIMEOUT_MS);
});
