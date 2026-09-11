import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, readFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import { createServer, type Server } from "http";

/**
 * The org selector must survive BUNDLING, not just module wiring (#357).
 *
 * Every other org-scope test spawns `bin/postgres-ai.ts` through bun, where each
 * module exists once. `bun build` emits `lib/org-scope.ts` more than once, so a
 * module-level binding is per-copy: the preAction hook wrote one copy and the
 * request path read another, and `x-pgai-org` was silently dropped from most
 * commands while every source-level test passed. These tests run the BUILT
 * artifact -- the thing users actually execute -- against a stub server.
 */

const CLI_DIR = resolve(import.meta.dir, "..");
const ALIAS = "acme-test-org";
const GLOBAL_TOKEN = "pai_global_0000000000000000000000000000000000000000";

let bundlePath: string;
let server: Server;
let port: number;
/** Headers of the first request of the current spawn, reset per command. */
let firstHeaders: Record<string, string> | null = null;

function buildBundle(): string {
  const outDir = mkdtempSync(join(tmpdir(), "pgai-bundle-"));
  const result = Bun.spawnSync(
    [process.execPath, "build", "./bin/postgres-ai.ts", "--outdir", outDir, "--target", "node"],
    { cwd: CLI_DIR }
  );
  if (result.exitCode !== 0) {
    throw new Error(`bun build failed: ${new TextDecoder().decode(result.stderr)}`);
  }
  return join(outDir, "postgres-ai.js");
}

/**
 * Async, never spawnSync: the stub server lives in THIS process, so a blocking
 * spawn would deadlock -- the CLI waits for a response the blocked event loop
 * cannot send.
 */
async function runBuilt(args: string[], env: Record<string, string> = {}): Promise<void> {
  firstHeaders = null;
  const proc = Bun.spawn([process.execPath, bundlePath, ...args, "--api-base-url", `http://127.0.0.1:${port}/`], {
    env: { ...process.env, PGAI_API_KEY: GLOBAL_TOKEN, PGAI_ORG: "", PGAI_ORG_ID: "", ...env },
    cwd: CLI_DIR,
    stdout: "ignore",
    stderr: "ignore",
  });
  await proc.exited;
}

beforeAll(async () => {
  bundlePath = buildBundle();
  server = createServer((req, res) => {
    if (firstHeaders === null) {
      firstHeaders = req.headers as Record<string, string>;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    // An empty array is a valid, terminal answer for every listing below, so
    // the CLI exits instead of following up with a second request.
    res.end("[]");
  });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  port = (server.address() as { port: number }).port;
});

afterAll(() => {
  server?.close();
});

/**
 * One command per org-scoped transport module. `issues list` used to be the
 * only passing command and was the one the feature's manual e2e sampled, so it
 * is deliberately NOT representative on its own -- each module gets an entry.
 */
const COMMANDS: Array<{ label: string; argv: string[] }> = [
  { label: "projects (lib/joe)", argv: ["projects"] },
  { label: "issues list (lib/issues)", argv: ["issues", "list"] },
  { label: "issues view (lib/issues)", argv: ["issues", "view", "1"] },
  { label: "reports list (lib/reports)", argv: ["reports", "list"] },
  { label: "joe activity (lib/joe)", argv: ["joe", "activity", "--project", "1"] },
  { label: "dblab clone list (lib/dblab)", argv: ["dblab", "clone", "list", "--project", "1"] },
];

describe("the org selector survives bundling", () => {
  for (const { label, argv } of COMMANDS) {
    test(`${label} sends x-pgai-org from the built bundle`, async () => {
      await runBuilt([...argv, "--org", ALIAS]);

      expect(firstHeaders).not.toBeNull();
      expect(firstHeaders?.["x-pgai-org"]).toBe(ALIAS);
    });
  }

  test("--org-id rides the same path", async () => {
    await runBuilt(["projects", "--org-id", "5225"]);

    expect(firstHeaders?.["x-pgai-org-id"]).toBe("5225");
  });

  test("PGAI_ORG reaches the wire from the built bundle too", async () => {
    await runBuilt(["projects"], { PGAI_ORG: ALIAS });

    expect(firstHeaders?.["x-pgai-org"]).toBe(ALIAS);
  });
});

describe("the scope is not held in a per-copy module binding", () => {
  test("the built bundle declares no module-level activeOrgScope", () => {
    const src = readFileSync(bundlePath, "utf8");

    // A bare `let activeOrgScope;` (or a renamed `activeOrgScope2`) means the
    // selection is back in module state, which the duplicate-module emit makes
    // per-copy. The registry symbol is the only form shared across copies.
    expect(src).not.toMatch(/\n\s*(?:let|var)\s+activeOrgScope\d*\s*[;=]/);
    expect(src).toContain("postgres-ai.cli.activeOrgScope");
  });
});
