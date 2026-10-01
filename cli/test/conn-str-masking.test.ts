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

test.each([
  ["a '/' in the password", `postgresql://u:pa/${secret}@h:5432/db`],
  ["an '@' and a '/' in the password", `postgresql://u:p@${secret}/w@h:5432/db`],
  ["sslpassword", `postgresql://u:pw@h:5432/db?sslpassword=${secret}&sslmode=require`],
  // Digits first: URL parsing reads them as a port and the rest as a fragment or query.
  ["a '#' after digits in the password", `postgresql://u:5432#${secret}@h:5432/db`],
  ["a '?' after digits in the password", `postgresql://u:1234?${secret}@h:5432/db`],
  ["a '/' first in the password", `postgresql://u:/${secret}@h:5432/db`],
  ["digits and a '/' first in the password", `postgresql://u:1234/${secret}@h:5432/db`],
  ["an '@' in sslpassword", `postgresql://u:pw@h:5432/db?sslpassword=ab@${secret}&sslmode=require`],
  ["an '@' in sslpassword and a '/' after it", `postgresql://u:pw@h:5432/db?sslpassword=ab@${secret}/x&sslmode=require`],
])("maskConnectionString hides %s", (_shape, url) => {
  const masked = maskConnectionString(url);
  expect(masked).not.toContain(secret);
  expect(masked).toContain("@h:5432/db");
});

test.each([
  ["postgresql://h:5432/db@x"],
  ["postgresql://h/db@x"],
  ["postgresql://u@h:5432/db"],
  ["postgresql://h:5432?application_name=me@corp"],
  ["postgresql://[::1]:5432/db?application_name=me@corp"],
  ["postgresql://host:5432,host2:5433/db?x=a@b"],
])("maskConnectionString leaves %s, which has no password, as it is", (url) => {
  expect(maskConnectionString(url)).toBe(url);
});

test.each([
  ["a '/' in the password", `postgresql://u:pa/${secret}@h`],
  ["a '/' in the password and an '@' in the query", `postgresql://u:pa/${secret}@h?application_name=me@corp`],
  ["a '?' in the password", `postgresql://u:pw?${secret}@h`],
  ["a '#' first in the password", `postgresql://u:#${secret}@h`],
  ["a '?' first in the password", `postgresql://u:?${secret}@h`],
  ["a '/' and then a '#' in the password", `postgresql://u:Ab/${secret}#x@h`],
  ["a '#' and then a '/' in the password", `postgresql://u:Ab#${secret}/x@h`],
  ["a '?' and an '@' in the password", `postgresql://u:a?b@${secret}@h`],
  ["a '#' and a '/' first in the password", `postgresql://u:#/${secret}@h`],
  ["a '?password=' in the password", `postgresql://u:${secret}?password=@h`],
  ["an '@' in sslpassword", `postgresql://u:pw@h?sslpassword=ab@${secret}`],
])("maskConnectionString hides %s in a URL with no database path", (_shape, url) => {
  const masked = maskConnectionString(url);
  expect(masked).not.toContain(secret);
  expect(masked).toContain("@h");
});

test.each([
  ["a '?password=' in the password", `postgresql://u:a?password=${secret}@h/db`, "postgresql://u:*****@h/db"],
  ["a '#' after a query password", `postgresql://u:pw@h/db?PassWord=a#${secret}`, "postgresql://u:*****@h/db?PassWord=*****"],
  ["pwd", `postgresql://u@h/db?pwd=${secret}&sslmode=require`, "postgresql://u@h/db?pwd=*****&sslmode=require"],
])("maskConnectionString hides %s and keeps the host", (_shape, url, expected) => {
  expect(maskConnectionString(url)).toBe(expected);
});

test("maskConnectionString scans a long query key with no '=' once", () => {
  const started = performance.now();
  maskConnectionString(`postgresql://u@h/db?${"pass".repeat(25000)}`);
  expect(performance.now() - started).toBeLessThan(200);
});

test("maskConnectionString keeps host, db and a query '@' next to a hidden password", () => {
  expect(maskConnectionString(`postgresql://u:${secret}@h:5432/db?application_name=me@corp`)).toBe("postgresql://u:*****@h:5432/db?application_name=me@corp");
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
