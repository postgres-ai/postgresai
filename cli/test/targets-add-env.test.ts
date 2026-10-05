import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// `mon targets add` takes the DB URL from PGAI_DB_URL when it is not in argv,
// where `ps` and the sudo log would show the password. Same as prepare-db and
// local-install.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const secret = "S3cretPw-env-91c4";
const url = `postgresql://monitor:${secret}@db.example:5432/app`;
const argvSecret = "ArgvPw-222";
const other = `postgresql://monitor:${argvSecret}@other.example:5432/app`;
let dir: string, projectDir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/targets-add-env-`);
  projectDir = `${dir}/project`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${projectDir}/instances.yml`, "");
  // Logs the args and the environment of every docker call.
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh\nprintf '%s\\n' "$*" >> "${dir}/docker.log"\nenv >> "${dir}/docker.log"\nexit 0\n`);
  chmodSync(`${dir}/bin/docker`, 0o755);
  writeFileSync(`${dir}/docker.log`, "");
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], { cwd: dir, env: { ...env, ...extra }, timeout: 60000 });
  return { exitCode: result.exitCode, out: result.stdout.toString() + result.stderr.toString() };
}
const instances = () => readFileSync(`${projectDir}/instances.yml`, "utf8");
const docker = () => readFileSync(`${dir}/docker.log`, "utf8");

test("targets add reads the DB URL from PGAI_DB_URL, with the name as the only argument", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Using PGAI_DB_URL (user monitor)");
  expect(out).toContain("Monitoring target 'app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(instances()).toContain("name: app");
  expect(out).not.toContain(secret);
  // docker and compose run, without the URL in their args or environment.
  expect(docker()).toContain("compose");
  expect(docker()).not.toContain(secret);
  expect(docker()).not.toContain("PGAI_DB_URL");
});

test("targets add reads PGAI_DB_URL with no argument and uses the default name", () => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'db-example-app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(out).not.toContain(secret);
});

test("a name with '=' in it is a name", () => {
  const { exitCode, out } = run(["add", "region=west"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(instances()).toContain("name: region=west");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(out).not.toContain(secret);
});

test("a URL in argv still wins over PGAI_DB_URL", () => {
  const { exitCode, out } = run(["add", other, "app"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(instances()).toContain(`conn_str: ${other}`);
  expect(instances()).not.toContain(secret);
  expect(out).not.toContain(secret);
});

test("a URL alone in argv wins over PGAI_DB_URL and gets the default name", () => {
  const { exitCode, out } = run(["add", other], { PGAI_DB_URL: url });
  expect(exitCode).toBe(0);
  expect(out).toContain("Monitoring target 'other-example-app' added");
  expect(out).not.toContain("PGAI_DB_URL");
  expect(instances()).toContain(`conn_str: ${other}`);
  expect(instances()).toContain("name: other-example-app");
  expect(instances()).not.toContain(secret);
  expect(out).not.toContain(secret);
});

test("an upper-case scheme in argv is a connection string, refused as without PGAI_DB_URL", () => {
  const { exitCode, out } = run(["add", other.replace("postgresql", "POSTGRESQL")], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("Invalid connection string format");
  expect(out).not.toContain("pass only the target name");
  expect(instances()).toBe("");
});

test("two arguments with PGAI_DB_URL set: the first is the connection string", () => {
  const { exitCode, out } = run(["add", "foo", "bar"], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("Invalid connection string format");
  expect(instances()).toBe("");
});

// Each would otherwise be saved as the name of the PGAI_DB_URL target, and its
// password printed and written to instances.yml and the node_name label.
test.each([
  `host=other.example user=monitor password=${argvSecret} dbname=app`,
  `monitor:${argvSecret}@other.example:5432/app`,
  `postgresql:/other.example/app?password=${argvSecret}`,
  `postgresql+ssl://monitor:${argvSecret}@other.example:5432/app`,
  ` ${other}`,
])("with PGAI_DB_URL set, a lone argument like a connection string is refused: %p", (arg) => {
  const { exitCode, out } = run(["add", arg], { PGAI_DB_URL: url });
  expect(exitCode).toBe(1);
  expect(out).toContain("PGAI_DB_URL is set: pass only the target name");
  expect(out).not.toContain(argvSecret);
  expect(out).not.toContain(secret);
  expect(instances()).toBe("");
});

test("an empty PGAI_DB_URL is no connection string", () => {
  const { exitCode, out } = run(["add"], { PGAI_DB_URL: "" });
  expect(exitCode).toBe(1);
  expect(out).toContain("Connection string required");
  expect(instances()).toBe("");
});

test("an invalid PGAI_DB_URL is refused, named, and not printed", () => {
  const { exitCode, out } = run(["add", "app"], { PGAI_DB_URL: `monitor:${secret}@db.example:5432/app` });
  expect(exitCode).toBe(1);
  expect(out).toContain("PGAI_DB_URL");
  expect(out).toContain("Invalid connection string format");
  expect(out).not.toContain(secret);
  expect(instances()).toBe("");
});

test("targets add --help names PGAI_DB_URL, which automation probes for", () => {
  const { exitCode, out } = run(["add", "--help"]);
  expect(exitCode).toBe(0);
  expect(out).toContain("PGAI_DB_URL");
  expect(out).toContain("--preserve-env=PGAI_DB_URL");
});
