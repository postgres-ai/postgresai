import { describe, expect, test } from "bun:test";
import { judgeServiceHealth, monitoringHealthServices, type MonitoringService } from "../bin/postgres-ai";

const core: MonitoringService = { name: "Prometheus", container: "sink-prometheus" };
const gated: MonitoringService = {
  name: "Instance jobs",
  container: "instance-jobs",
  optional: true,
  readHealth: true,
};

// The boundary: these pin the decision, not the command around it. A refactor
// that spawned `docker inspect .State.Health.Status` for all seven services and
// passed the result in would still be ignored here for the six, so the claim
// "we do not even ask them" lives in the caller and is not covered.
describe("mon health: one service's verdict", () => {
  test("the six long-standing services keep the status-only check", () => {
    expect(judgeServiceHealth(core, true, "running", "")).toMatchObject({ ok: true, label: "healthy" });
    expect(judgeServiceHealth(core, true, "exited", "")).toMatchObject({ ok: false });
    expect(judgeServiceHealth(core, false, "", "")).toMatchObject({ ok: false, label: "unreachable" });
  });

  test("a container's health status is ignored unless the service opts in", () => {
    // Regression guard: reading .State.Health.Status for the pre-existing
    // services would flip boxes red fleet-wide, which is a separate decision.
    expect(judgeServiceHealth(core, true, "running", "unhealthy")).toMatchObject({ ok: true });
  });

  test("a profile-gated service that was never enabled is skipped, not failed", () => {
    const verdict = judgeServiceHealth(gated, false, "", "");
    expect(verdict.ok).toBe(true);
    expect(verdict.label).toBe("not enabled");
  });

  test("an absent container still fails when the service is not optional", () => {
    expect(judgeServiceHealth({ ...gated, optional: false }, false, "", "")).toMatchObject({ ok: false });
  });

  test("a self-reporting container that says unhealthy is unhealthy", () => {
    const verdict = judgeServiceHealth(gated, true, "running", "unhealthy");
    expect(verdict.ok).toBe(false);
    expect(verdict.label).toContain("unhealthy");
  });

  test("starting is not a failure", () => {
    // Docker reports `starting` until the first probe fires. Treating it as a
    // failure would make `mon health` red for the whole start period after
    // every `mon up`.
    expect(judgeServiceHealth(gated, true, "running", "starting")).toMatchObject({ ok: true });
  });

  test("a running container with no healthcheck at all is healthy", () => {
    expect(judgeServiceHealth(gated, true, "running", "")).toMatchObject({ ok: true, mark: "✓" });
  });

  test("a self-reporting container that says healthy is healthy", () => {
    // The ordinary green state, and the one this whole entry exists to show.
    expect(judgeServiceHealth(gated, true, "running", "healthy"))
      .toMatchObject({ ok: true, mark: "✓", label: "healthy" });
  });

  test("the mark an operator reads matches the verdict", () => {
    expect(judgeServiceHealth(gated, false, "", "").mark).toBe("-");
    expect(judgeServiceHealth(core, false, "", "").mark).toBe("✗");
    expect(judgeServiceHealth(core, true, "exited", "").mark).toBe("✗");
  });
});

describe("mon health: which services are reported on", () => {
  const services = monitoringHealthServices();

  test("the instance-jobs container is in the list, gated and self-reporting", () => {
    const entry = services.find(s => s.container === "instance-jobs");
    expect(entry).toBeDefined();
    // optional: it ships behind a compose profile, so on a stack that never
    // enabled it an absent container is "not enabled" rather than a fault --
    // that is the DEFAULT argument here. With the profile on it becomes a
    // fault; see instance-jobs-profile.test.ts (postgresai#381).
    // readHealth: it reports a real verdict.
    expect(entry!.optional).toBe(true);
    expect(entry!.readHealth).toBe(true);
  });

  test("the six long-standing services are unchanged", () => {
    const longStanding = services.filter(s => s.container !== "instance-jobs");
    expect(longStanding.map(s => s.container)).toEqual([
      "grafana-with-datasources",
      "sink-prometheus",
      "pgwatch-postgres",
      "pgwatch-prometheus",
      "target-db",
      "sink-postgres",
    ]);
    // Neither flag, so they keep the status-only check they have always had.
    for (const s of longStanding) {
      expect(s.optional).toBeUndefined();
      expect(s.readHealth).toBeUndefined();
    }
  });
});
