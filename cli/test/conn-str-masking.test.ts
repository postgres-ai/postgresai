import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { maskConnectionString } from "../lib/init";

// Every place the CLI echoes a connection string must hide the password:
// stdout/stderr of `mon local-install` ends up in CI logs and terminal scrollback.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const secret = "S3cretPw-7f2a";
let dir: string, projectDir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/conn-str-masking-`);
  projectDir = `${dir}/project`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  mkdirSync(`${projectDir}/.git`);
  // No Docker: local-install stops at its first compose call, after the target step.
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[]) {
  const result = Bun.spawnSync([process.execPath, cli, ...args], { cwd: dir, env, timeout: 60000 });
  return { exitCode: result.exitCode, out: result.stdout.toString() + result.stderr.toString() };
}

test("maskConnectionString hides a password given as a query parameter", () => {
  const masked = maskConnectionString(`postgresql://monitor@127.0.0.1:1/postgres?password=${secret}&sslmode=disable`);
  expect(masked).not.toContain(secret);
  expect(masked).toContain("sslmode=disable");
});

test("local-install --db-url does not print the password", () => {
  const dbUrl = `postgresql://monitor:${secret}@127.0.0.1:1/postgres?sslmode=disable`;
  const { out } = run(["mon", "local-install", "--db-url", dbUrl, "-y"]);
  expect(out).toContain("Adding PostgreSQL instance from: postgresql://monitor:*****@127.0.0.1:1/postgres");
  expect(out).toContain("Monitoring target '127-0-0-1-postgres' added");
  expect(out).not.toContain(secret);
});

test("mon targets add and list do not print the password", () => {
  const add = run(["mon", "targets", "add", `postgresql://monitor:${secret}@db.example:5432/app`, "app"]);
  expect(add.out).toContain("Monitoring target 'app' added");
  expect(add.out).not.toContain(secret);
  const list = run(["mon", "targets", "list"]);
  expect(list.out).toContain("Target: app");
  expect(list.out).not.toContain(secret);
});

test("a corrupted instances.yml error does not print stored passwords", () => {
  // The YAML parser's source snippet truncates long lines, so even a prefix
  // of the password counts as a leak.
  writeFileSync(`${projectDir}/instances.yml`, `- conn_str: postgresql://u:${secret}@db.example:5432/app\n  : [unclosed\n`);
  for (const args of [["mon", "targets", "list"], ["mon", "targets", "add", "postgresql://u:p@h:5432/d", "x"]]) {
    const { exitCode, out } = run(args);
    expect(exitCode).toBe(1);
    expect(out).toContain("Failed to parse");
    expect(out).toContain("(2:3)");
    expect(out).not.toContain(secret.slice(0, 4));
  }
});

test("maskConnectionString hides a query password in a string URL parsing rejects", () => {
  const masked = maskConnectionString(`postgresql://u@host:bad/db?password=${secret}`);
  expect(masked).not.toContain(secret);
});

test("mon targets add does not put a query password into the generated name", () => {
  const add = run(["mon", "targets", "add", `postgresql://u:p@db.example:5432/app?password=${secret}`]);
  expect(add.out).toContain("Monitoring target 'db-example-app' added");
  expect(add.out).not.toContain(secret);
});

test("a YAML error reason that quotes a connection string is masked", () => {
  writeFileSync(`${projectDir}/instances.yml`, `- conn_str: *postgresql://u:${secret}@db.example:5432/app\n`);
  const { exitCode, out } = run(["mon", "targets", "list"]);
  expect(exitCode).toBe(1);
  expect(out).toContain("Failed to parse");
  expect(out).not.toContain(secret);
});
