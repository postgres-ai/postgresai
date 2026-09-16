import { describe, test, expect } from "bun:test";
import { shouldScaleOutNodeExporter } from "../bin/postgres-ai";

/**
 * postgresai#359: `runCompose` appends `--scale self-node-exporter=0` on macOS,
 * because that container cannot mount the host root filesystem there. Every
 * call site used to pass a whole-stack `up`, so the flag was always valid.
 * Applying the VictoriaMetrics admin keys introduced the first service-scoped
 * `up`, and compose rejects `--scale` for a service that is not in the list:
 *
 *   docker compose up -d --no-deps sink-prometheus --scale self-node-exporter=0
 *   -> no such service: self-node-exporter
 *
 * which made `mon update` fail outright on macOS.
 */
describe("shouldScaleOutNodeExporter", () => {
  test("applies to a whole-stack up", () => {
    expect(shouldScaleOutNodeExporter(["up", "-d"])).toBe(true);
    expect(shouldScaleOutNodeExporter(["up"])).toBe(true);
    expect(shouldScaleOutNodeExporter(["up", "-d", "--remove-orphans"])).toBe(true);
  });

  test("does not apply to a service-scoped up", () => {
    expect(shouldScaleOutNodeExporter(["up", "-d", "--no-deps", "sink-prometheus"])).toBe(false);
    expect(shouldScaleOutNodeExporter(["up", "-d", "sink-prometheus"])).toBe(false);
    expect(shouldScaleOutNodeExporter(["up", "grafana"])).toBe(false);
  });

  test("does not apply to commands that are not up", () => {
    for (const args of [["pull"], ["restart"], ["ps", "-q", "sink-prometheus"], ["down"]]) {
      expect(shouldScaleOutNodeExporter(args)).toBe(false);
    }
  });

  test("the service name is what matters, not its position", () => {
    // `run --rm sources-generator` contains no `up`, so it is untouched.
    expect(shouldScaleOutNodeExporter(["run", "--rm", "sources-generator"])).toBe(false);
  });
});
