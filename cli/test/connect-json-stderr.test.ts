import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Client } from "pg";

// With JSON output (--json, or stdout not a TTY), stderr is the progress
// stream an agent parses: every line of it is one JSON object. A check's
// error, the --debug request log and the output of a self-hosted
// `mon local-install` used to come out as plain text in between.

const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
let dir: string;

beforeAll(() => {
  dir = mkdtempSync(resolve(tmpdir(), "pgai-connect-json-"));
  for (const p of ["home", "bin", "project"]) mkdirSync(resolve(dir, p));
  // No Docker: no stack is running, and `mon local-install` stops at its first compose call.
  writeFileSync(resolve(dir, "bin", "docker"), "#!/bin/sh\necho 'docker: not here' >&2\nexit 1\n");
  chmodSync(resolve(dir, "bin", "docker"), 0o755);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

async function run(args: string[], env: Record<string, string> = {}) {
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    env: {
      ...process.env, HOME: resolve(dir, "home"), XDG_CONFIG_HOME: resolve(dir, "home"),
      PATH: `${resolve(dir, "bin")}:${process.env.PATH}`, PGAI_PROJECT_DIR: resolve(dir, "project"),
      PGAI_API_KEY: "", CLICKHOUSE_KEY_ID: "", CLICKHOUSE_KEY_SECRET: "", PGAI_NO_FEEDBACK_TIP: "1", ...env,
    },
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
  });
  const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  return { status, stdout, stderr, lines: stderr.split("\n").filter((l) => l !== "") };
}

function expectJsonLines(lines: string[]) {
  expect(lines.length).toBeGreaterThan(0);
  for (const line of lines) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      throw new Error(`stderr line is not JSON: ${line}`);
    }
    expect(typeof (parsed as { event?: unknown }).event).toBe("string");
  }
  return lines.map((l) => JSON.parse(l) as { event: string; level?: string; message?: string; source?: string });
}

async function closedPort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as { port: number };
  await new Promise<void>((r) => server.close(() => r()));
  return port;
}

test.each([[["--json"]], [[]]])("--debug request lines are JSON events on stderr (%p; stdout is a pipe)", async (flags) => {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => Response.json([]) });
  try {
    const url = `postgresql://postgres:pw@127.0.0.1:${await closedPort()}/postgres?sslmode=disable`;
    const r = await run(["connect", url, ...flags, "--debug", "--wait", "0"], {
      PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}`,
    });
    const events = expectJsonLines(r.lines);
    expect(events.some((e) => e.event === "log" && /Debug: POST URL: .*cloud_monitoring_list/.test(e.message ?? ""))).toBe(true);
    expect(JSON.parse(r.stdout).status).toBe("failed");
    expect(r.stderr).not.toContain("test-key");
  } finally {
    server.stop(true);
  }
});

// CI: the cli:clickhouse-like:tests job.
const ADMIN = process.env.PGAI_TEST_CLICKHOUSE_LIKE_URL;
describe.skipIf(!ADMIN)("real Postgres", () => {
  const dropMonRole = async () => {
    const c = new Client({ connectionString: ADMIN });
    await c.connect();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
  };
  beforeAll(dropMonRole);
  afterAll(dropMonRole);

  test("--self-hosted: the output of mon local-install comes as JSON events", async () => {
    const r = await run(["connect", ADMIN!, "--self-hosted", "--json"]);
    const events = expectJsonLines(r.lines);
    expect(events.some((e) => e.event === "log" && e.source === "mon local-install")).toBe(true);
    expect(JSON.parse(r.stdout).status).toBe("failed");
    expect(r.stderr).not.toContain(new URL(ADMIN!).password);
  });
});
