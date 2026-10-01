import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// On the monitoring box ansible runs `mon local-install` and `prepare-db` with
// the database URL (password included) and the API key. In argv they show in
// `ps` and in the sudo log. Given as PGAI_DB_URL / PGAI_API_KEY, they must not
// reach argv: not the CLI's own, not any process it starts.
//
// Each test parks the CLI on a TCP server that accepts and never answers (the
// CLI is then connecting to the URL it was given, so it did read it), takes a
// `ps` snapshot of every process, and looks for the secrets in it.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const DB_PASSWORD = "Pw4argvCheck9x";
const API_KEY = "key4argvCheck7q";
let dir: string, env: Record<string, string>, server: ReturnType<typeof Bun.listen> | undefined;
let connected: Promise<void>, onConnect: () => void;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/secrets-argv-`);
  for (const p of ["project", "project/.git", "bin", "home", "xdg"]) mkdirSync(`${dir}/${p}`);
  writeFileSync(`${dir}/project/docker-compose.yml`, "services: {}\n");
  // No Docker. The stub logs every call, so a secret passed to a child that has
  // already exited is caught too.
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh\nprintf '%s\\n' "$*" >> "${dir}/docker.log"\nexit 1\n`);
  chmodSync(`${dir}/bin/docker`, 0o755);
  writeFileSync(`${dir}/docker.log`, "");
  env = {
    PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`,
    PGAI_PROJECT_DIR: `${dir}/project`, PGAI_API_BASE_URL: "http://127.0.0.1:9", PGSSLMODE: "disable",
  };
  connected = new Promise((r) => (onConnect = r));
  server = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { open: () => onConnect(), data() {} } });
});
afterEach(() => {
  server?.stop(true);
  rmSync(dir, { recursive: true, force: true });
});

const dbUrl = () => `postgresql://admin:${DB_PASSWORD}@127.0.0.1:${server!.port}/postgres`;

/** Start the CLI, wait until it connects to the parked server, return every process's args. */
async function argvWhileConnecting(args: string[], extra: Record<string, string>): Promise<string> {
  const child = Bun.spawn([process.execPath, cli, ...args], { cwd: dir, env: { ...env, ...extra }, stdout: "pipe", stderr: "pipe" });
  try {
    const exitedFirst = child.exited.then(async () => {
      throw new Error(`the CLI exited without connecting to the database URL: ${await new Response(child.stdout).text()}${await new Response(child.stderr).text()}`);
    });
    exitedFirst.catch(() => {}); // the kill below settles it after the race
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_, rej) => (timer = setTimeout(() => rej(new Error("the CLI never connected to the database URL")), 30000)));
    await Promise.race([connected, exitedFirst, timeout]).finally(() => clearTimeout(timer));
    return Bun.spawnSync(["ps", "-A", "-o", "args="]).stdout.toString();
  } finally {
    child.kill();
    await child.exited;
  }
}

test("the ps check sees a password passed in argv (control)", async () => {
  const ps = await argvWhileConnecting(["prepare-db", dbUrl(), "--json"], {});
  expect(ps).toContain(DB_PASSWORD);
}, 40000);

test("prepare-db reads PGAI_DB_URL and keeps it out of argv", async () => {
  const ps = await argvWhileConnecting(["prepare-db", "--json"], { PGAI_DB_URL: dbUrl() });
  expect(ps).toContain("prepare-db");
  expect(ps).not.toContain(DB_PASSWORD);
  expect(readFileSync(`${dir}/docker.log`, "utf8")).not.toContain(DB_PASSWORD);
}, 40000);

test("mon local-install reads PGAI_DB_URL and PGAI_API_KEY and keeps both out of argv", async () => {
  const ps = await argvWhileConnecting(["mon", "local-install", "-y"], { PGAI_DB_URL: dbUrl(), PGAI_API_KEY: API_KEY });
  expect(ps).toContain("local-install");
  for (const secret of [DB_PASSWORD, API_KEY]) {
    expect(ps).not.toContain(secret);
    expect(readFileSync(`${dir}/docker.log`, "utf8")).not.toContain(secret);
  }
}, 40000);

test("an explicit prepare-db connection wins over PGAI_DB_URL", async () => {
  const ps = await argvWhileConnecting(["prepare-db", dbUrl(), "--json"], { PGAI_DB_URL: "postgresql://other:x@127.0.0.1:1/postgres" });
  expect(ps).toContain(DB_PASSWORD);
}, 40000);
