import { afterEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { planUpdateTag, readEnvTag, writeEnvTag } from "../bin/postgres-ai";
import pkg from "../package.json";

// `mon update` used to keep PGAI_TAG, so an upgrade re-pulled the old images
// unless the user edited .env by hand first.
test.each([
  ["0.16.0", "0.17.0", "0.17.0", "PGAI_TAG: 0.16.0 -> 0.17.0"],
  [null, "0.17.0", "0.17.0", "PGAI_TAG: unset -> 0.17.0"],
  ["fix-branch-abc123", "0.17.0", "0.17.0", "PGAI_TAG: fix-branch-abc123 -> 0.17.0"],
  ["0.17.0", "0.17.0", null, "PGAI_TAG is 0.17.0, matching this CLI"],
  ["0.18.0", "0.17.0", null, "PGAI_TAG stays 0.18.0: it is newer than this CLI (0.17.0). Upgrade the CLI to move the stack: npm install -g postgresai@latest"],
  ["0.18.0-rc.1", "0.17.0", null, "PGAI_TAG stays 0.18.0-rc.1: it is newer than this CLI (0.17.0). Upgrade the CLI to move the stack: npm install -g postgresai@latest"],
  ["0.17.0-rc.1", "0.17.0", "0.17.0", "PGAI_TAG: 0.17.0-rc.1 -> 0.17.0"],
  ["0.16.0", "0.0.0-dev.0", null, "PGAI_TAG stays 0.16.0: this CLI (0.0.0-dev.0) is not a release. To pick a stack version, set PGAI_TAG=<version> in .env and re-run 'postgresai mon update'"],
])("planUpdateTag(%p, %p)", (current, cli, tag, note) => {
  expect(planUpdateTag(current, cli)).toEqual({ tag, note });
});

let dir: string | undefined;
afterEach(() => { if (dir) rmSync(dir, { recursive: true, force: true }); });

test("mon update rewrites PGAI_TAG in .env before anything else", () => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  const project = `${dir}/project`;
  mkdirSync(project); mkdirSync(`${dir}/bin`);
  // A git checkout with no origin: update fails at `git fetch`, after the .env step.
  mkdirSync(`${project}/.git`);
  writeFileSync(`${project}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${project}/.env`, "# keep me\nexport PGAI_TAG=0.16.0\nVM_AUTH_USERNAME=vmauth\n");
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  const result = Bun.spawnSync([process.execPath, resolve(import.meta.dir, "../bin/postgres-ai.ts"), "mon", "update"], {
    cwd: dir, timeout: 30000,
    env: { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: project, GIT_DIR: `${project}/.git` },
  });
  const expected = planUpdateTag("0.16.0", pkg.version);
  expect(result.stdout.toString()).toContain(expected.note);
  const env = readFileSync(`${project}/.env`, "utf8");
  expect(env).toContain("# keep me\n");
  expect(env).toContain(`export PGAI_TAG=${expected.tag ?? "0.16.0"}\n`);
  expect(env.match(/PGAI_TAG=/g)).toHaveLength(1);
});

test.each([
  ["PGAI_TAG=0.18.0 # pinned\n", { tag: "0.18.0", plain: true }],
  ['export PGAI_TAG = "0.18.0"\n', { tag: "0.18.0", plain: true }],
  ["PGAI_TAG='0.16.0'\r\nPGAI_TAG=0.18.0\r\n", { tag: "0.18.0", plain: true }],
  ["# PGAI_TAG=0.9.0\nOTHER=1\n", { tag: null, plain: true }],
  // Values compose resolves in ways we do not: never rewritten.
  ['PGAI_TAG="${STACK_VERSION:-0.18.0}"\n', { tag: null, plain: false }],
  ["PGAI_TAG: 0.18.0\n", { tag: null, plain: false }],
  ['PGAI_TAG="0.16.0"# pinned\n', { tag: null, plain: false }],
  ["PGAI_TAG=0.16.0\nPGAI_TAG=$OTHER\n", { tag: null, plain: false }],
])("readEnvTag reads what compose reads: %p", (content, expected) => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  writeFileSync(`${dir}/.env`, content);
  expect(readEnvTag(dir)).toEqual(expected);
});

test("writeEnvTag rewrites every assignment and keeps export, quotes, comments and CRLF", () => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  writeFileSync(`${dir}/.env`, "A=1\r\nexport PGAI_TAG = '0.16.0' # pin\r\nPGAI_TAG=0.16.0\r\n# PGAI_TAG=0.9.0\r\n");
  writeEnvTag(dir, "0.17.0");
  expect(readFileSync(`${dir}/.env`, "utf8")).toBe("A=1\r\nexport PGAI_TAG = '0.17.0' # pin\r\nPGAI_TAG=0.17.0\r\n# PGAI_TAG=0.9.0\r\n");
  expect(readEnvTag(dir)).toEqual({ tag: "0.17.0", plain: true });
});

test("writeEnvTag appends when .env has no PGAI_TAG", () => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  writeFileSync(`${dir}/.env`, "A=1");
  writeEnvTag(dir, "0.17.0");
  expect(readFileSync(`${dir}/.env`, "utf8")).toBe("A=1\nPGAI_TAG=0.17.0\n");
});

test("mon update leaves a PGAI_TAG it cannot resolve alone", () => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  const project = `${dir}/project`;
  mkdirSync(project); mkdirSync(`${dir}/bin`); mkdirSync(`${project}/.git`);
  writeFileSync(`${project}/docker-compose.yml`, "services: {}\n");
  const content = 'PGAI_TAG="${STACK_VERSION:-0.18.0}"\n';
  writeFileSync(`${project}/.env`, content);
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  const result = Bun.spawnSync([process.execPath, resolve(import.meta.dir, "../bin/postgres-ai.ts"), "mon", "update"], {
    cwd: dir, timeout: 30000,
    env: { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: project, GIT_DIR: `${project}/.git` },
  });
  expect(result.stdout.toString()).toContain("PGAI_TAG in .env is not a plain value, so it is left as is.");
  // Newly required keys are still appended; the PGAI_TAG line is untouched.
  const env = readFileSync(`${project}/.env`, "utf8");
  expect(env.startsWith(content)).toBe(true);
  expect(env.match(/PGAI_TAG/g)).toHaveLength(1);
});

// Compose prefers the environment over .env, so a PGAI_TAG exported in the
// shell would silently win over the one this command writes.
test("mon update warns when an exported PGAI_TAG overrides .env", () => {
  dir = mkdtempSync(`${tmpdir()}/mon-update-tag-`);
  const project = `${dir}/project`;
  mkdirSync(project); mkdirSync(`${dir}/bin`); mkdirSync(`${project}/.git`);
  writeFileSync(`${project}/docker-compose.yml`, "services: {}\n");
  writeFileSync(`${project}/.env`, "PGAI_TAG=0.16.0\n");
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  const result = Bun.spawnSync([process.execPath, resolve(import.meta.dir, "../bin/postgres-ai.ts"), "mon", "update"], {
    cwd: dir, timeout: 30000,
    env: { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: project, GIT_DIR: `${project}/.git`, PGAI_TAG: "0.15.0" },
  });
  expect(result.stderr.toString()).toContain("PGAI_TAG=0.15.0 is set in the environment, and docker compose prefers it over .env: unset it before 'postgresai mon stop && postgresai mon start'");
});
