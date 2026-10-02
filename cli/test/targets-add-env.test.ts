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
let dir: string, projectDir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/targets-add-env-`);
  projectDir = `${dir}/project`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${projectDir}/instances.yml`, "");
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 0\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], { cwd: dir, env: { ...env, ...extra }, timeout: 60000 });
  return { exitCode: result.exitCode, out: result.stdout.toString() + result.stderr.toString() };
}
const instances = () => readFileSync(`${projectDir}/instances.yml`, "utf8");

test("targets add reads the DB URL from PGAI_DB_URL, with the name as the only argument", () => {
  const { out } = run(["add", "app"], { PGAI_DB_URL: url });
  expect(out).toContain("Monitoring target 'app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
  expect(instances()).toContain("name: app");
  expect(out).not.toContain(secret);
});

test("targets add reads PGAI_DB_URL with no argument and uses the default name", () => {
  const { out } = run(["add"], { PGAI_DB_URL: url });
  expect(out).toContain("Monitoring target 'db-example-app' added");
  expect(instances()).toContain(`conn_str: ${url}`);
});

test("a URL in argv still wins over PGAI_DB_URL", () => {
  const other = "postgresql://monitor:pw@other.example:5432/app";
  run(["add", other, "app"], { PGAI_DB_URL: url });
  expect(instances()).toContain(`conn_str: ${other}`);
  expect(instances()).not.toContain(secret);
});

test("targets add --help names PGAI_DB_URL, which automation probes for", () => {
  expect(run(["add", "--help"]).out).toContain("PGAI_DB_URL");
});
