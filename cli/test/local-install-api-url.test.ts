import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { resolveBaseUrls } from "../lib/util";

const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
let dir: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${import.meta.dir}/local-install-api-url-`);
  for (const path of ["project", "project/.git", "bin", "home", "xdg"]) mkdirSync(`${dir}/${path}`);
  writeFileSync(`${dir}/project/docker-compose.yml`, "services: {}\n");
  // No Docker: local-install stops at its first compose call, after step 1.
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: `${dir}/project` };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

for (const base of ["https://preview.example.test/api/general", undefined]) {
  test(`local-install saves the ${base ? "configured" : "production default"} API base beside the API key`, () => {
    if (base) env.PGAI_API_BASE_URL = base;
    const result = Bun.spawnSync([process.execPath, cli, "mon", "local-install", "--api-key", "test-key", "-y"], { cwd: `${dir}/project`, env, timeout: 60000 });
    expect(result.stdout.toString() + result.stderr.toString()).toContain("✓ API key saved");
    const config = readFileSync(`${dir}/project/.pgwatch-config`, "utf8");
    expect(config.split("\n")).toContain("api_key=test-key");
    const expected = resolveBaseUrls({ apiBaseUrl: base || "https://postgres.ai/api/general/" }).apiBaseUrl;
    expect(config.split("\n").find(line => line.startsWith("api_url="))).toBe(`api_url=${expected}`);
  });
}
