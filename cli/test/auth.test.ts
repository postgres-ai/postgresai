import { describe, test, expect } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "fs";
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
