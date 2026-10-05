import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
  // The project mon local-install finds from its cwd: this one, not the checkout's (it writes .env there).
  writeFileSync(resolve(dir, "project", "docker-compose.yml"), "services: {}\n");
  // No Docker: no stack is running, and `mon local-install` stops at its first compose call.
  writeFileSync(resolve(dir, "bin", "docker"), "#!/bin/sh\necho 'docker: not here' >&2\nexit 1\n");
  chmodSync(resolve(dir, "bin", "docker"), 0o755);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

async function run(args: string[], env: Record<string, string> = {}, cwd = resolve(dir, "project")) {
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    cwd,
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
    // Not errors: an agent counting level "error" events finds none here.
    expect(events.filter((e) => /^\s*Debug:/.test(e.message ?? "")).every((e) => e.level === "debug")).toBe(true);
    expect(JSON.parse(r.stdout).status).toBe("failed");
    expect(r.stderr).not.toContain("test-key");
  } finally {
    server.stop(true);
  }
});

// Commander's own errors (a missing URL, an option without its value, an
// unknown option, at the root too) and the org check come before connect
// runs: each is one error event that names the cause, without the help after
// it. With --json, and with no flag when stdout is a pipe.
const URL_ = "postgresql://u:p@127.0.0.1:1/d";
const EARLY: [string, string[], Record<string, string>, RegExp][] = [
  ["no database URL", ["connect"], {}, /missing required argument 'database-url'/],
  ["a bad --org-id", ["connect", URL_, "--org-id", "abc"], { PGAI_API_KEY: "test-key" }, /--org-id must be a numeric organization id/],
  ["--api-key without its value", ["connect", URL_, "--api-key"], {}, /option '--api-key <key>' argument missing/],
  ["an unknown root option", ["--bogus", "connect", URL_], {}, /unknown option '--bogus'/],
  // Its value is not the command: connect is.
  ["an unknown root option with its value (--org before connect)", ["--org", "acme", "connect", URL_], {}, /unknown option '--org'/],
  // The value of a root option is not the command either.
  ["a root option with its value, no database URL", ["--api-key", "k", "connect"], {}, /missing required argument 'database-url'/],
  ["a root option with its value, an unknown option after the URL", ["--api-key", "k", "connect", URL_, "--bogus"], {}, /unknown option '--bogus'/],
  // Its value is not the command either: Commander fails at the root, the run is connect.
  ["an unknown option before connect whose value names a command", ["--org", "status", "connect", URL_], {}, /unknown option '--org'/],
  // The first command named is the command: "status" is one argument too many.
  ["a command name after the database URL", ["connect", URL_, "status"], {}, /too many arguments for 'connect'/],
  // A log collector keeps the event: an unknown option is named without its value.
  ["an unknown option holding a URL", ["connect", "--url=postgresql://postgres:s3cret-pw@db.example.com/app"], {}, /^error: unknown option '--url'$/],
  ["an unknown option holding a password", ["connect", URL_, "--password=MonPw-S3cret"], {}, /^error: unknown option '--password'$/],
  ["an unknown short option with its value", ["connect", URL_, "-pS3cret"], {}, /^error: unknown option '-p'$/],
  // Commander's suggestion after it names a known option: kept.
  ["an unknown option with its value, close to a known one", ["connect", URL_, "--debugs=1"], {}, /^error: unknown option '--debugs'\n\(Did you mean --debug\?\)$/],
  ["an unknown option holding a URL with a space", ["connect", "--url=postgresql://postgres:hunter two@db.example.com/app"], {}, /^error: unknown option '--url'$/],
];
describe.each([[["--json"]], [[]]])("with %p (stdout is a pipe)", (flags) => {
  test.each(EARLY)("%s: one error event names the cause", async (_case, args, env, cause) => {
    // The flags right after `connect`: after `--api-key`, --json would be the key.
    const r = await run(args.flatMap((a) => (a === "connect" ? [a, ...flags] : [a])), env);
    expect(r.status).toBe(1);
    expect(expectJsonLines(r.lines)).toEqual([{ event: "log", level: "error", message: expect.stringMatching(cause) }]);
  });
});

// Not a connect run: Commander's text and help, as before. Another command,
// even one given "connect" as its argument; "connect" after "--".
test.each([
  [["--api-key=k", "status", "connect", "x", "--json"], "error: too many arguments for 'status'", "Usage: postgres-ai status [options] [name]"],
  [["--bogus", "--", "connect", URL_, "--json"], "error: unknown option '--bogus'", "Usage: postgres-ai [options] [command]"],
])("%p: the error is text, the help follows", async (args, error, usage) => {
  const r = await run(args);
  expect(r.status).toBe(1);
  expect(r.stderr).toStartWith(error);
  expect(r.stderr).toContain(usage);
});

test("a bad API base URL: stdout says why, stderr is JSON only", async () => {
  const r = await run(["connect", URL_, "--json"], { PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: "not-a-url" });
  expect(r.status).toBe(1);
  expect(JSON.parse(r.stdout)).toMatchObject({ status: "failed", next: "Invalid base URL: not-a-url" });
  for (const line of r.lines) expect(() => JSON.parse(line), line).not.toThrow();
});

test("a config file that cannot be read: its warning is a JSON event, from the first line", async () => {
  const xdg = mkdtempSync(resolve(dir, "xdg-"));
  mkdirSync(resolve(xdg, "postgresai"));
  writeFileSync(resolve(xdg, "postgresai", "config.json"), "{bad");
  const r = await run(["connect", URL_], { XDG_CONFIG_HOME: xdg });
  const events = expectJsonLines(r.lines);
  expect(events[0]).toMatchObject({ event: "log", level: "warn", message: expect.stringContaining("Failed to read config") });
  expect(JSON.parse(r.stdout).status).toBe("action_required");
});

// A person at a terminal: Commander's errors are text, with the help after them,
// as before. --json given as the value of --wait (connect's) or of --api-key
// (the root's) does not ask for JSON.
const clean = (s: string) => s.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "");
async function runTty(args: string[]) {
  let out = "";
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    cwd: resolve(dir, "project"),
    env: { ...process.env, HOME: resolve(dir, "home"), XDG_CONFIG_HOME: resolve(dir, "home"), PGAI_API_KEY: "", PGAI_NO_FEEDBACK_TIP: "1" },
    terminal: { cols: 200, rows: 50, data: (_term, bytes) => void (out += new TextDecoder().decode(bytes)) },
  });
  return { status: await proc.exited, screen: clean(out) };
}
const NO_URL = "error: missing required argument 'database-url'";
const TTY_TEXT: [string[], string][] = [
  [["connect"], NO_URL],
  [["connect", "--wait", "--json"], NO_URL],
  [["connect", "--api-key", "--json"], NO_URL],
  // Commander takes the root's options first, wherever they are: --json is the value of --api-key, --coupon has none.
  [["connect", URL_, "--coupon", "--api-key", "--json"], "error: option '--coupon <code>' argument missing"],
  // After "--", --json is an argument: one too many.
  [["connect", URL_, "--", "--json"], "error: too many arguments for 'connect'. Expected 1 argument but got 2."],
];
test.each(TTY_TEXT)("at a terminal, %p: the error is text, the help follows", async (args, error) => {
  const r = await runTty(args);
  expect(r.status).toBe(1);
  expect(r.screen).toStartWith(`${error}\n`);
  expect(r.screen).toContain("Usage: postgres-ai connect [options] <database-url>");
  expect(r.screen).not.toMatch(/^\{"event"/m);
});

// With --json, at a terminal too: one error event, no help.
test.each([[["connect", "--json"]], [["--bogus", "connect", "--json"]]])("at a terminal, %p: one error event, no help", async (args) => {
  const r = await runTty(args);
  expect(r.status).toBe(1);
  expect(expectJsonLines(r.screen.split("\n").filter((l) => l !== ""))).toEqual([
    { event: "log", level: "error", message: expect.stringMatching(/^error: (missing required argument 'database-url'|unknown option '--bogus')$/) },
  ]);
  expect(r.screen).not.toContain("Usage:");
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
  beforeEach(dropMonRole);
  afterAll(dropMonRole);

  test("--self-hosted: the output of mon local-install comes as JSON events", async () => {
    const r = await run(["connect", ADMIN!, "--self-hosted", "--json"]);
    const events = expectJsonLines(r.lines);
    const child = events.filter((e) => e.event === "log" && e.source === "mon local-install");
    expect(child.length).toBeGreaterThan(0);
    expect(child.every((e) => e.message!.trim() !== "")).toBe(true);
    // Its stderr is where it says why it stopped: an error an agent can find.
    expect(child.some((e) => e.level === "error" && /docker/i.test(e.message!))).toBe(true);
    expect(JSON.parse(r.stdout).status).toBe("failed");
    expect(r.stderr).not.toContain(new URL(ADMIN!).password);
  });

  test("--self-hosted: the logins mon local-install prints at its end carry no password", async () => {
    // Docker that says yes to everything: the install runs to its end, in a project of its own.
    const sandbox = mkdtempSync(resolve(dir, "docker-ok-"));
    for (const p of ["bin", "project"]) mkdirSync(resolve(sandbox, p));
    writeFileSync(resolve(sandbox, "bin", "docker"), "#!/bin/sh\nexit 0\n");
    chmodSync(resolve(sandbox, "bin", "docker"), 0o755);
    writeFileSync(resolve(sandbox, "project", "docker-compose.yml"), "services: {}\n");
    const project = resolve(sandbox, "project");
    const r = await run(["connect", ADMIN!, "--self-hosted", "--json"], { PATH: `${resolve(sandbox, "bin")}:${process.env.PATH}`, PGAI_PROJECT_DIR: project }, project);
    expect(JSON.parse(r.stdout).status).toBe("connected");
    const logins = expectJsonLines(r.lines).filter((e) => e.source === "mon local-install" && /Login:|Auth:/.test(e.message!));
    expect(logins.map((e) => e.message!.trim())).toEqual([
      "Login: ***** (pgai mon show-grafana-credentials)",
      "VictoriaMetrics Auth: ***** (pgai mon show-grafana-credentials)",
    ]);
    const grafana = readFileSync(resolve(project, ".pgwatch-config"), "utf8").match(/^grafana_password=(.+)$/m)![1]!;
    const vm = readFileSync(resolve(project, ".env"), "utf8").match(/^VM_AUTH_PASSWORD=(.+)$/m)![1]!;
    expect(r.stderr).not.toContain(grafana);
    expect(r.stderr).not.toContain(vm);
  });
});
