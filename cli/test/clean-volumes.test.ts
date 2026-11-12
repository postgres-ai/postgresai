import { describe, expect, test } from "bun:test";
import { CLEAN_ORPHAN_VOLUMES } from "../bin/postgres-ai";

describe("mon clean orphaned volumes", () => {
  test("covers pgai-mon's store under both older project names", () => {
    expect(CLEAN_ORPHAN_VOLUMES).toContain("monitoring_pgai_mon_data");
    expect(CLEAN_ORPHAN_VOLUMES).toContain("postgres_ai_configs_pgai_mon_data");
  });

  test("still covers the old stack's volumes", () => {
    for (const project of ["monitoring", "postgres_ai_configs"]) {
      for (const volume of ["victoria_metrics_data", "sink_postgres_data", "grafana_data", "target_db_data"]) {
        expect(CLEAN_ORPHAN_VOLUMES).toContain(`${project}_${volume}`);
      }
    }
  });

  test("never names a volume of the current default project", () => {
    expect(CLEAN_ORPHAN_VOLUMES.filter((v) => v.startsWith("postgres_ai_") && !v.startsWith("postgres_ai_configs_"))).toEqual([]);
  });
});
