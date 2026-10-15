import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { resolve } from "path";

const cliSource = readFileSync(resolve(import.meta.dir, "../bin/postgres-ai.ts"), "utf8");

describe("mon targets configuration apply", () => {
  test("targets add regenerates sources and restarts pgwatch collectors after saving", () => {
    expect(cliSource).toContain("async function applyMonitoringTargetsConfig()");
    expect(cliSource).toContain('runCompose(["run", "--rm", "sources-generator"])');
    expect(cliSource).toContain(
      'runCompose(["up", "-d", "--force-recreate", "pgwatch-prometheus", "pgwatch-postgres"])'
    );

    const saveIndex = cliSource.indexOf("addInstanceToFile(file, instance)");
    const applyIndex = cliSource.indexOf("await applyMonitoringTargetsConfig()", saveIndex);
    expect(saveIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(saveIndex);
  });

  test("targets add and remove restart postgres-reports, so a new target's first report is not a day away", () => {
    const start = cliSource.indexOf("async function applyMonitoringTargetsConfig()");
    const body = cliSource.slice(start, cliSource.indexOf("\n}\n", start));
    const recreate = body.indexOf('"pgwatch-prometheus", "pgwatch-postgres"]');
    const restart = body.indexOf('runCompose(["restart", "postgres-reports"])');
    expect(recreate).toBeGreaterThan(-1);
    expect(restart).toBeGreaterThan(recreate);
  });

  test("targets remove regenerates sources and restarts pgwatch collectors after saving", () => {
    const removeIndex = cliSource.indexOf("removeInstanceFromFile(file, name)");
    const applyIndex = cliSource.indexOf("await applyMonitoringTargetsConfig()", removeIndex);
    expect(removeIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(removeIndex);
  });
});
