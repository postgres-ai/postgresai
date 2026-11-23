import { mock } from "bun:test";
import { appendFileSync } from "node:fs";

// PGAI_TEST_PG_FAKE=1: a database that accepts every statement, for prepare-db runs.
// It answers the few reads prepare-db needs; PGAI_TEST_PG_ROLE_EXISTS=1 says the
// monitoring role exists. PGAI_TEST_PG_LOG names a file that gets each statement.
// PGAI_TEST_PG_FAIL: a regular expression; a statement it matches fails.
// prepare-db --supabase gets the same fake database through the Management API query endpoint.
// PGAI_TEST_PG_PERMS_MISSING=1: the checkup permission check finds pg_monitor missing.
// PGAI_TEST_API_URL: fetch may reach this local fake API (the checkup API key check).
function fakeRows(sql: string): unknown[] {
  if (/permission_checks/.test(sql) && process.env.PGAI_TEST_PG_PERMS_MISSING === "1") {
    return [{ permission_name: "pg_monitor role membership", status: "required", granted: false, fix_command: null }];
  }
  if (/current_database\(\) as db/.test(sql)) return [{ db: "appdb" }];
  if (/as role_exists/.test(sql)) return process.env.PGAI_TEST_PG_ROLE_EXISTS === "1" ? [{ role_exists: true }] : [];
  if (/scram_iterations/.test(sql)) return [{ iterations: "4096" }];
  return [];
}

function fakeQuery(sql: string): unknown[] {
  if (process.env.PGAI_TEST_PG_LOG) appendFileSync(process.env.PGAI_TEST_PG_LOG, `${sql}\n-- end statement\n`);
  if (process.env.PGAI_TEST_PG_FAIL && new RegExp(process.env.PGAI_TEST_PG_FAIL).test(sql)) {
    throw Object.assign(new Error("statement failed in the fake database"), { code: "XX000" });
  }
  return fakeRows(sql);
}

// Only CLI smoke children preload this file. Unit and integration tests keep pg.
mock.module("pg", () => ({
  Client: class {
    connection = { stream: { authorized: false } };
    constructor(private config: any) {}
    async connect() {
      if (process.env.PGAI_TEST_PG_FAKE === "1") return;
      if (process.env.PGAI_TEST_TLS_VERIFY && this.config.ssl?.rejectUnauthorized === true) {
        if (process.env.PGAI_TEST_TLS_VERIFY === "success") return;
        throw Object.assign(new Error("certificate rejected in offline fixture"), { code: process.env.PGAI_TEST_TLS_VERIFY });
      }
      throw new Error("Database unavailable in offline CLI smoke fixture");
    }
    async query(sql: string) {
      if (process.env.PGAI_TEST_PG_FAKE === "1") {
        const rows = fakeQuery(sql);
        return { rows, rowCount: rows.length };
      }
      throw new Error("Unexpected database query in offline CLI smoke fixture");
    }
    async end() {}
  },
}));

const fixtureFetch = globalThis.fetch;
globalThis.fetch = Object.assign(
  async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    if (process.env.PGAI_TEST_TLS_VERIFY && url.hostname === "127.0.0.1" && url.origin === process.env.CLICKHOUSE_API_URL) return fixtureFetch(input, init);
    if (process.env.PGAI_TEST_API_URL && url.origin === new URL(process.env.PGAI_TEST_API_URL).origin) return fixtureFetch(input, init);
    if (process.env.PGAI_TEST_PG_FAKE === "1" && url.hostname === "api.supabase.com") {
      const json = (body: unknown, status: number) =>
        new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
      if (!url.pathname.endsWith("/database/query")) return json({ message: "not found" }, 404);
      try {
        return json(fakeQuery(JSON.parse(String(init?.body ?? "{}")).query), 200);
      } catch (e) {
        return json({ message: (e as Error).message, code: "XX000" }, 400);
      }
    }
    throw new Error("Network disabled in offline CLI smoke fixture");
  },
  { preconnect() { throw new Error("Network disabled in offline CLI smoke fixture"); } },
);
