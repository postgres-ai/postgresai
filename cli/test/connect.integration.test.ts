import { afterAll, describe, expect, test } from "bun:test";
import { Client } from "pg";
import { prepareDatabase } from "../lib/connect";

// `pgai connect`'s prepare step against a real Postgres, with a superuser URL
// like the one ClickHouse Managed Postgres hands out. CI: the
// cli:clickhouse-like:tests job (PG 17 and 18).
const ADMIN = process.env.PGAI_TEST_CLICKHOUSE_LIKE_URL;

describe.skipIf(!ADMIN)("prepareDatabase (real Postgres)", () => {
  const admin = () => { const c = new Client({ connectionString: ADMIN }); return c.connect().then(() => c); };

  afterAll(async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.end();
  });

  test("an admin URL creates the monitoring role; its URL connects and verifies; a re-run still works", async () => {
    const first = await prepareDatabase(ADMIN!, "self-managed");
    if (!("monitoringUrl" in first)) throw new Error(`expected a URL, got SQL:\n${first.sql}`);
    const mon = new URL(first.monitoringUrl);
    expect(mon.username).toBe("postgres_ai_mon");
    expect(mon.password.length).toBeGreaterThanOrEqual(16);
    expect(mon.password).not.toBe(new URL(ADMIN!).password);

    const again = await prepareDatabase(first.monitoringUrl, "self-managed");
    expect(again).toEqual({ monitoringUrl: first.monitoringUrl });

    // A second admin run rotates the password; the new URL is the one that works.
    const second = await prepareDatabase(ADMIN!, "self-managed");
    if (!("monitoringUrl" in second)) throw new Error("expected a URL");
    const c = new Client({ connectionString: second.monitoringUrl });
    await c.connect();
    expect((await c.query("select current_user as u")).rows[0].u).toBe("postgres_ai_mon");
    await c.end();
  });

  test("a URL that can neither create roles nor is the monitoring role: the SQL, passwords redacted", async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.query("create role pgai_connect_app login password 'app-pw-123'");
    await c.end();
    const url = new URL(ADMIN!);
    url.username = "pgai_connect_app";
    url.password = "app-pw-123";
    const result = await prepareDatabase(url.toString(), "clickhouse");
    if (!("sql" in result)) throw new Error("expected SQL");
    expect(result.sql).toContain("-- 01.role");
    expect(result.sql).toContain("password '<redacted>'");
    expect(result.sql).not.toContain("app-pw-123");
  });
});
