import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// config/init-configs.sh skips when the volume's marker equals the image
// version. Dev builds all carry 0.0.0-dev*, so they re-provision every time.
const script = resolve(import.meta.dir, "../../config/init-configs.sh");
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/pgai-init-configs-`);
  mkdirSync(`${dir}/src/grafana/dashboards`, { recursive: true });
  mkdirSync(`${dir}/tgt`);
  writeFileSync(`${dir}/src/grafana/dashboards/new.json`, "{}\n");
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function provision(version: string) {
  writeFileSync(`${dir}/VERSION`, `${version}\n`);
  writeFileSync(`${dir}/tgt/.pgai-configs-version`, `${version}\n`);
  const r = Bun.spawnSync(["sh", script], {
    env: { PATH: "/usr/bin:/bin", SOURCE_DIR: `${dir}/src`, TARGET_DIR: `${dir}/tgt`, VERSION_FILE: `${dir}/VERSION`, BUILD_TS_FILE: `${dir}/none` },
  });
  return { exitCode: r.exitCode, stdout: r.stdout.toString() };
}

test("a dev build re-provisions although the marker matches", () => {
  const r = provision("0.0.0-dev.0");
  expect(r.exitCode).toBe(0);
  expect(r.stdout).not.toContain("skipping");
  expect(readFileSync(`${dir}/tgt/grafana/dashboards/new.json`, "utf8")).toBe("{}\n");
  expect(readFileSync(`${dir}/tgt/.pgai-configs-version`, "utf8").trim()).toBe("0.0.0-dev.0");
});

test("a release build with a matching marker still skips", () => {
  const r = provision("1.2.3");
  expect(r.exitCode).toBe(0);
  expect(r.stdout).toContain("skipping");
});
