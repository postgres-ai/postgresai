import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
test("project initialization pre-creates host-metrics alongside bind-mount files", () => {
  const dir = mkdtempSync(`${tmpdir()}/clickhouse-init-`);
  try {
    const projectDir = `${dir}/project`;
    mkdirSync(projectDir);
    writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
    const result = Bun.spawnSync([process.execPath, `${root}/cli/bin/postgres-ai.ts`, "mon", "targets", "list"], {
      cwd: dir,
      env: { ...process.env, PGAI_PROJECT_DIR: projectDir },
      timeout: 15000,
    });
    expect(result.exitCode, result.stderr.toString()).toBe(0);
    expect(existsSync(`${projectDir}/host-metrics`)).toBe(true);
    expect(statSync(`${projectDir}/host-metrics`).isDirectory()).toBe(true);
    expect(statSync(`${projectDir}/instances.yml`).isFile()).toBe(true);
    expect(statSync(`${projectDir}/.pgwatch-config`).isFile()).toBe(true);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
