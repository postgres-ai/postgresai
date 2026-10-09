import { mock } from "bun:test";

// Only CLI smoke children preload this file. Unit and integration tests keep pg.
mock.module("pg", () => ({
  Client: class {
    connection = { stream: { authorized: false } };
    constructor(private config: any) {}
    async connect() {
      if (process.env.PGAI_TEST_TLS_VERIFY && this.config.ssl?.rejectUnauthorized === true) {
        if (process.env.PGAI_TEST_TLS_VERIFY === "success") return;
        throw Object.assign(new Error("certificate rejected in offline fixture"), { code: process.env.PGAI_TEST_TLS_VERIFY });
      }
      throw new Error("Database unavailable in offline CLI smoke fixture");
    }
    async query() { throw new Error("Unexpected database query in offline CLI smoke fixture"); }
    async end() {}
  },
}));

const fixtureFetch = globalThis.fetch;
globalThis.fetch = Object.assign(
  async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    if (process.env.PGAI_TEST_TLS_VERIFY && url.hostname === "127.0.0.1" && url.origin === process.env.CLICKHOUSE_API_URL) return fixtureFetch(input, init);
    throw new Error("Network disabled in offline CLI smoke fixture");
  },
  { preconnect() { throw new Error("Network disabled in offline CLI smoke fixture"); } },
);
