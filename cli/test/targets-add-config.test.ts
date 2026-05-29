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

    const saveIndex = cliSource.indexOf("addInstanceToFile(file, buildInstance(instanceName, connStr))");
    const applyIndex = cliSource.indexOf("await applyMonitoringTargetsConfig()", saveIndex);
    expect(saveIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(saveIndex);
  });

  test("targets remove regenerates sources and restarts pgwatch collectors after saving", () => {
    const removeIndex = cliSource.indexOf("removeInstanceFromFile(file, name)");
    const applyIndex = cliSource.indexOf("await applyMonitoringTargetsConfig()", removeIndex);
    expect(removeIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(-1);
    expect(applyIndex).toBeGreaterThan(removeIndex);
  });
});
