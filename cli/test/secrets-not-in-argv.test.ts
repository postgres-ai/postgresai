import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// On the monitoring box ansible runs `mon local-install` and `prepare-db` with
// the database URL (password included) and the API key. In argv they show in
// `ps` and in the sudo log. Given as PGAI_DB_URL / PGAI_API_KEY, they must not
// reach argv: not the CLI's own, not any process it starts.
//
// Each test parks the CLI on a TCP server that accepts and does not answer (so
// the CLI is connecting to the URL it was given), takes a `ps` snapshot of every
// process, then drops the connection and lets the CLI run to its exit. The
// docker stub logs every call it gets, and the CLI's output is checked too.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
let dir: string, env: Record<string, string>, server: Bun.TCPSocketListener<undefined> | undefined;
let connected: Promise<void>, onConnect: () => void, sockets: Bun.Socket<undefined>[];
let dbPassword: string, apiKey: string;

beforeEach(() => {
  // Unique per test: `ps` sees every process, including other test files'.
  const tag = Math.random().toString(36).slice(2, 10);
  dbPassword = `Pw4argv${tag}`;
  apiKey = `key4argv${tag}`;
  dir = mkdtempSync(`${tmpdir()}/secrets-argv-`);
  for (const p of ["project", "project/.git", "bin", "home", "xdg"]) mkdirSync(`${dir}/${p}`);
  writeFileSync(`${dir}/project/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh\nprintf '%s\\n' "$*" >> "${dir}/docker.log"\nexit 1\n`);
  chmodSync(`${dir}/bin/docker`, 0o755);
  writeFileSync(`${dir}/docker.log`, "");
  env = {
    PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`,
    PGAI_PROJECT_DIR: `${dir}/project`, PGAI_API_BASE_URL: "http://127.0.0.1:9", PGSSLMODE: "disable",
  };
  sockets = [];
  connected = new Promise((r) => (onConnect = r));
  server = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { open: (s) => { sockets.push(s); onConnect(); }, data() {} } });
});
afterEach(() => {
  server?.stop(true);
  rmSync(dir, { recursive: true, force: true });
});

const dbUrl = () => `postgresql://admin:${dbPassword}@127.0.0.1:${server!.port}/postgres`;

/** Run the CLI; return every process's args while it connects, and its output and docker calls after it exits. */
async function run(args: string[], extra: Record<string, string>) {
  const child = Bun.spawn([process.execPath, cli, ...args], { cwd: dir, env: { ...env, ...extra }, stdout: "pipe", stderr: "pipe" });
  const output = Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text()]).then((o) => o.join(""));
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const exitedFirst = child.exited.then(async () => {
      throw new Error(`the CLI exited without connecting to the database URL: ${await output}`);
    });
    exitedFirst.catch(() => {}); // settles again after the race
    const timeout = new Promise<never>((_, rej) => (timer = setTimeout(() => rej(new Error("the CLI never connected to the database URL")), 30000)));
    await Promise.race([connected, exitedFirst, timeout]);
    clearTimeout(timer);
    const ps = Bun.spawnSync(["ps", "-A", "-o", "args="]).stdout.toString();
    for (const s of sockets) s.end();
    await Promise.race([child.exited, new Promise<never>((_, rej) => (timer = setTimeout(() => rej(new Error("the CLI did not exit")), 30000)))]);
    return { ps, output: await output, docker: readFileSync(`${dir}/docker.log`, "utf8") };
  } finally {
    clearTimeout(timer);
    child.kill();
    await child.exited;
  }
}

test("the ps check sees a password passed in argv (control)", async () => {
  const { ps } = await run(["prepare-db", dbUrl(), "--json"], {});
  expect(ps).toContain(dbPassword);
}, 70000);

test("prepare-db reads PGAI_DB_URL and keeps it out of argv", async () => {
  const { ps, output, docker } = await run(["prepare-db", "--json"], { PGAI_DB_URL: dbUrl() });
  expect(ps).toContain("prepare-db");
  for (const seen of [ps, output, docker]) expect(seen).not.toContain(dbPassword);
}, 70000);

test("mon local-install reads PGAI_DB_URL and PGAI_API_KEY and keeps both out of argv", async () => {
  const { ps, output, docker } = await run(["mon", "local-install", "-y"], { PGAI_DB_URL: dbUrl(), PGAI_API_KEY: apiKey });
  expect(ps).toContain("local-install");
  expect(output).toContain("Connection failed"); // it ran on past the connection test
  expect(output).toContain("Step 3");
  for (const seen of [ps, output, docker]) {
    expect(seen).not.toContain(dbPassword);
    expect(seen).not.toContain(apiKey);
  }
  expect(JSON.parse(readFileSync(`${dir}/xdg/postgresai/config.json`, "utf8")).apiKey).toBe(apiKey);
}, 70000);

test("an explicit prepare-db connection wins over PGAI_DB_URL", async () => {
  // Only the positional URL points at the parked server; run() fails if it is not dialled.
  await run(["prepare-db", dbUrl(), "--json"], { PGAI_DB_URL: "postgresql://other:x@127.0.0.1:1/postgres" });
}, 70000);

test("prepare-db --print-sql stays offline with PGAI_DB_URL set", () => {
  const r = Bun.spawnSync([process.execPath, cli, "prepare-db", "--print-sql"], {
    cwd: dir, env: { ...env, PGAI_DB_URL: dbUrl() }, timeout: 30000,
  });
  const out = r.stdout.toString() + r.stderr.toString();
  expect(out).toContain("SQL plan");
  expect(out).not.toContain(dbPassword);
  expect(sockets.length).toBe(0);
}, 40000);
