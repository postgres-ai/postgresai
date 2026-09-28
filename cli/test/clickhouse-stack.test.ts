import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
test("Prometheus loads host metric scrape files", () => {
  const config = Bun.YAML.parse(readFileSync(`${root}/config/prometheus/prometheus.yml`, "utf8")) as any;
  expect(config.scrape_config_files).toEqual(["/etc/pgai/host-metrics/*.yml"]);
});
test("sink-prometheus mounts host metrics read-only", () => {
  const compose = Bun.YAML.parse(readFileSync(`${root}/docker-compose.yml`, "utf8")) as any;
  expect(compose.services["sink-prometheus"].volumes).toContain("./host-metrics:/etc/pgai/host-metrics:ro");
});
test("project initialization pre-creates host-metrics alongside bind-mount files", async () => {
  const cli = await import("../bin/postgres-ai");
  expect(cli).toHaveProperty("ensureDefaultMonitoringProject");
  const dir = mkdtempSync(`${tmpdir()}/clickhouse-init-`);
  const previous = process.env.PGAI_PROJECT_DIR;
  try {
    const projectDir = `${dir}/project`;
    mkdirSync(projectDir);
    writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
    process.env.PGAI_PROJECT_DIR = projectDir;
    await (cli as any).ensureDefaultMonitoringProject();
    expect(existsSync(`${projectDir}/host-metrics`)).toBe(true);
    expect(statSync(`${projectDir}/host-metrics`).isDirectory()).toBe(true);
    expect(statSync(`${projectDir}/instances.yml`).isFile()).toBe(true);
    expect(statSync(`${projectDir}/.pgwatch-config`).isFile()).toBe(true);
  } finally {
    if (previous === undefined) delete process.env.PGAI_PROJECT_DIR;
    else process.env.PGAI_PROJECT_DIR = previous;
    rmSync(dir, { recursive: true, force: true });
  }
});
