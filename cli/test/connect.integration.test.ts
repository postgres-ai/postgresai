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

  test("an admin URL creates the monitoring role; its URL connects to the prepared database as postgres_ai_mon", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    // No database in the path and credentials in the query string: the monitoring
    // URL must still name the prepared database and carry only the new role.
    const tricky = new URL(ADMIN!);
    tricky.pathname = "";
    tricky.searchParams.set("user", tricky.username);
    tricky.searchParams.set("password", decodeURIComponent(tricky.password));
    const first = await prepareDatabase(tricky.toString(), "self-managed");
    if (!("monitoringUrl" in first)) throw new Error(`expected a URL, got: ${JSON.stringify(first)}`);
    const mon = new URL(first.monitoringUrl);
    expect(mon.username).toBe("postgres_ai_mon");
    expect(mon.searchParams.has("user") || mon.searchParams.has("password")).toBe(false);
    expect(first.monitoringUrl).not.toContain(new URL(ADMIN!).password);
    const m = new Client({ connectionString: first.monitoringUrl });
    await m.connect();
    expect((await m.query("select current_user as u, current_database() as d")).rows[0]).toEqual({ u: "postgres_ai_mon", d: "postgres" });
    await m.end();

    expect(await prepareDatabase(first.monitoringUrl, "self-managed")).toEqual({ monitoringUrl: first.monitoringUrl });
    // The monitoring role's own URL with the password in the query string: normalized the same way.
    const monQuery = new URL(first.monitoringUrl);
    monQuery.password = "";
    monQuery.searchParams.set("password", decodeURIComponent(mon.password));
    expect(await prepareDatabase(monQuery.toString(), "self-managed")).toEqual({ monitoringUrl: first.monitoringUrl });

    // The role already exists: another database on this server may use its
    // password, so it is never rotated silently.
    const again = await prepareDatabase(ADMIN!, "self-managed");
    expect("monitoringUrl" in again).toBe(false);
    expect((again as { next: string }).next).toContain("PGAI_MON_PASSWORD");
    process.env.PGAI_MON_PASSWORD = "   ";
    try {
      expect("monitoringUrl" in await prepareDatabase(ADMIN!, "self-managed")).toBe(false);
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
    const still = new Client({ connectionString: first.monitoringUrl });
    await still.connect();
    await still.end();

    process.env.PGAI_MON_PASSWORD = decodeURIComponent(mon.password);
    try {
      expect(await prepareDatabase(ADMIN!, "self-managed")).toHaveProperty("monitoringUrl");
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
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
    if (!("sql" in result) || !result.sql) throw new Error("expected SQL");
    expect(result.sql).toContain("-- 01.role");
    expect(result.sql).toContain("password '<redacted>'");
    expect(result.sql).not.toContain("app-pw-123");
  });
});
