import { expect, test } from "bun:test";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// The bundle reads its SQL from dist/sql at run time: a rebuild over an
// earlier dist must replace those files, not leave the old ones beside new JS.
test("a rebuild over an existing dist refreshes dist/sql", () => {
  const cliDir = resolve(import.meta.dir, "..");
  const build: string = JSON.parse(readFileSync(join(cliDir, "package.json"), "utf8")).scripts.build;
  const sqlSteps = build.split("&&").map((s) => s.trim()).filter((s) => s.includes("dist/sql"));
  expect(sqlSteps.length).toBeGreaterThan(0);
  const dir = mkdtempSync(join(tmpdir(), "pgai-build-sql-"));
  try {
    cpSync(join(cliDir, "sql"), join(dir, "sql"), { recursive: true });
    mkdirSync(join(dir, "dist"));
    const copy = () => {
      const r = Bun.spawnSync(["sh", "-c", sqlSteps.join(" && ")], { cwd: dir, stderr: "pipe" });
      expect(r.exitCode).toBe(0);
    };
    copy();
    writeFileSync(join(dir, "sql", "02.extensions.sql"), "-- changed\n");
    copy();
    expect(readFileSync(join(dir, "dist", "sql", "02.extensions.sql"), "utf8")).toBe("-- changed\n");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
