import { expect, test } from "bun:test";
import { withJsonStderr } from "../lib/json-stderr";

// What generateAllReports does when a check fails ([D004] Error querying
// pg_stat_statements: ...) must reach an agent as a JSON event, not as text.
test("console.error and console.warn inside withJsonStderr are one JSON event a line", async () => {
  const lines: string[] = [];
  const write = process.stderr.write;
  process.stderr.write = ((chunk: string) => { lines.push(String(chunk)); return true; }) as typeof process.stderr.write;
  let result: number;
  try {
    result = await withJsonStderr(async () => {
      console.error("[D004] Error querying pg_stat_statements: %s", "must be loaded via \"shared_preload_libraries\"");
      console.warn("two\nlines");
      return 7;
    });
  } finally {
    process.stderr.write = write;
  }
  expect(result).toBe(7);
  expect(lines.join("").split("\n").filter(Boolean).map((l) => JSON.parse(l))).toEqual([
    { event: "log", level: "error", message: "[D004] Error querying pg_stat_statements: must be loaded via \"shared_preload_libraries\"" },
    { event: "log", level: "warn", message: "two\nlines" },
  ]);
});

test("withJsonStderr puts console.error back, also after a throw", async () => {
  const original = console.error;
  await expect(withJsonStderr(async () => { throw new Error("boom"); })).rejects.toThrow("boom");
  expect(console.error).toBe(original);
});
