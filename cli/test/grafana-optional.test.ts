import { afterAll, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { buildLocalInstallEnv, grafanaEnabled, monitoringHealthServices } from "../bin/postgres-ai";
import { createCliSandbox } from "./cli-sandbox";

const scaleLines = (content: string) => content.split("\n").filter((l) => l.startsWith("GRAFANA_SCALE="));

describe("--no-grafana in .env", () => {
  test("false writes GRAFANA_SCALE=0, true removes it, undefined carries it over once", () => {
    expect(scaleLines(buildLocalInstallEnv("", "1.0.0", null, null, undefined, undefined, false).content)).toEqual(["GRAFANA_SCALE=0"]);
    expect(scaleLines(buildLocalInstallEnv("GRAFANA_SCALE=0\n", "1.0.0", null, null, undefined, undefined, true).content)).toEqual([]);
    expect(scaleLines(buildLocalInstallEnv("GRAFANA_SCALE=0\n", "1.0.0").content)).toEqual(["GRAFANA_SCALE=0"]);
    expect(scaleLines(buildLocalInstallEnv("", "1.0.0").content)).toEqual([]);
  });

  test("grafanaEnabled reads the value as compose does", () => {
    expect(grafanaEnabled(undefined)).toBe(true);
    expect(grafanaEnabled("1")).toBe(true);
    expect(grafanaEnabled("0")).toBe(false);
    expect(grafanaEnabled('"0"')).toBe(false);
  });

  test("mon health treats an absent Grafana as not enabled only when it is left out", () => {
    const grafana = (on: boolean) => monitoringHealthServices(false, false, on).find((s) => s.name === "Grafana")!;
    expect(grafana(true).optional).toBeUndefined();
    expect(grafana(false).optional).toBe(true);
  });
});

describe("--no-grafana on the command line", () => {
  const sandbox = createCliSandbox();
  afterAll(() => sandbox.cleanup());
  const envFile = path.join(sandbox.projectDir, ".env");

  test("mon local-install --no-grafana writes GRAFANA_SCALE=0; a later run without the flag keeps it", () => {
    sandbox.run(["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes", "--no-grafana"]);
    expect(scaleLines(fs.readFileSync(envFile, "utf8"))).toEqual(["GRAFANA_SCALE=0"]);
    sandbox.run(["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"]);
    expect(scaleLines(fs.readFileSync(envFile, "utf8"))).toEqual(["GRAFANA_SCALE=0"]);
  }, 60000);

  test("mon update --grafana removes the key, --no-grafana writes it back", () => {
    const r = sandbox.run(["mon", "update", "--grafana"]);
    expect(r.stdout).toContain("Grafana: on");
    expect(scaleLines(fs.readFileSync(envFile, "utf8"))).toEqual([]);
    sandbox.run(["mon", "update", "--no-grafana"]);
    expect(scaleLines(fs.readFileSync(envFile, "utf8"))).toEqual(["GRAFANA_SCALE=0"]);
  }, 60000);
});
