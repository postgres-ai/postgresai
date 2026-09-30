import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// `pgai connect --self-hosted` hands local-install the key as PGAI_API_KEY next
// to PGAI_DB_URL. An exported PGAI_API_KEY alone (as `pgai connect` tells agents
// to set) must not reach a plain or --demo install.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
let dir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/local-install-api-key-`);
  for (const path of ["project", "project/.git", "bin", "home", "xdg"]) mkdirSync(`${dir}/${path}`);
  writeFileSync(`${dir}/project/docker-compose.yml`, "services: {}\n");
  // No Docker: local-install stops at its first compose call, after step 1.
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: `${dir}/project`, PGAI_API_KEY: "exported-key", PGAI_API_BASE_URL: "http://127.0.0.1:9" };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function localInstall(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "local-install", ...args, "-y"], { cwd: dir, env: { ...env, ...extra }, timeout: 60000 });
  return result.stdout.toString() + result.stderr.toString();
}

test("--demo ignores an exported PGAI_API_KEY", () => {
  const out = localInstall(["--demo"]);
  expect(out).not.toContain("Cannot use --api-key with --demo mode");
  expect(out).toContain("Step 1: Demo mode - API key configuration skipped");
});

test("a plain install ignores an exported PGAI_API_KEY and writes no key", () => {
  const out = localInstall([]);
  expect(out).toContain("Auto-yes mode: no API key provided, skipping API key setup");
  expect(existsSync(`${dir}/xdg/postgresai/config.json`)).toBe(false);
});

test("with PGAI_DB_URL (the pgai connect handoff) PGAI_API_KEY is used", () => {
  const out = localInstall([], { PGAI_DB_URL: "postgresql://u:pw@127.0.0.1:1/postgres" });
  expect(out).toContain("Using API key provided via --api-key parameter");
});
