import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { load } from "js-yaml";
import { renderScrapeConfig } from "../lib/clickhouse";
import { renderSupabaseScrapeConfig } from "../lib/host-metrics";

/**
 * pgai-mon's scraper is tested (mon/tests/scrape.rs) against copies of the
 * scrape files this CLI writes and of the expositions they scrape, kept in
 * mon/tests/fixtures/scrape/. These tests fail when the CLI's output or the
 * CLI-side fixtures change without those copies, so the Rust tests keep
 * exercising what the CLI writes today.
 */

const root = resolve(import.meta.dir, "../..");
const monFixture = (name: string) => readFileSync(resolve(root, "mon/tests/fixtures/scrape", name), "utf8");
const read = (rel: string) => readFileSync(resolve(root, rel), "utf8");

describe("the scrape files pgai-mon's tests read are the ones the CLI writes", () => {
  test("ClickHouse: renderScrapeConfig output equals mon's golden copy", () => {
    // The same options as the CLI's own golden test (test/clickhouse.test.ts).
    const text = renderScrapeConfig({
      name: "my-postgres",
      cluster: "default",
      orgId: "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f",
      serviceId: "0c330583-6396-86d0-82cd-ed0f23b0d38c",
      keyId: "test-key-id",
      passwordFile: "/etc/pgai/host-metrics/clickhouse-my-postgres.secret",
      apiUrl: "https://api.clickhouse.cloud",
    });
    expect(text).toBe(monFixture("clickhouse-scrape.golden.yml"));
  });

  test("Supabase: renderSupabaseScrapeConfig output equals mon's golden copy", () => {
    // The target `mon targets add <supabase url> sb` writes (test/host-metrics-providers.test.ts).
    const text = renderSupabaseScrapeConfig({ projectRef: "abcdefghijklmnopqrst", cluster: "default", nodeName: "sb" });
    expect(text).toBe(monFixture("supabase-scrape.golden.yml"));
  });

  test("the CLI's golden files and mon's copies are identical", () => {
    for (const name of ["clickhouse-scrape.golden.yml", "supabase-scrape.golden.yml"]) {
      expect(read(`cli/test/fixtures/${name}`)).toBe(monFixture(name));
    }
  });

  test("the recorded expositions are the same on both sides", () => {
    expect(read("cli/test/fixtures/clickhouse-prometheus.txt")).toBe(monFixture("clickhouse-prometheus.txt"));
    expect(read("instance-jobs/internal/supabase/testdata/supabase_metrics.prom")).toBe(monFixture("supabase_metrics.prom"));
  });
});


test("reporter and pgai-mon share the CLI's ClickHouse scrape files and key paths", () => {
  const compose = load(read("docker-compose.yml")) as any;
  for (const name of ["pgai-mon", "postgres-reports"]) {
    expect(compose.services[name].volumes).toContain("./host-metrics:/etc/pgai/host-metrics:ro");
  }
  const jobs = load(monFixture("clickhouse-scrape.golden.yml")) as any[];
  expect(jobs[0].basic_auth.password_file).toBe("/etc/pgai/host-metrics/clickhouse-my-postgres.secret");
});
