import { mock } from "bun:test";

// Only CLI smoke children preload this file. Unit and integration tests keep pg.
mock.module("pg", () => ({
  Client: class {
    async connect() { throw new Error("Database unavailable in offline CLI smoke fixture"); }
    async query() { throw new Error("Unexpected database query in offline CLI smoke fixture"); }
    async end() {}
  },
}));

globalThis.fetch = Object.assign(
  async (): Promise<Response> => { throw new Error("Network disabled in offline CLI smoke fixture"); },
  { preconnect() { throw new Error("Network disabled in offline CLI smoke fixture"); } },
);
