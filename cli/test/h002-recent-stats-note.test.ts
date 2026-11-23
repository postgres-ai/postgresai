// H002 summary note for recent index usage stats (#529).
// Imports only checkup-summary, so it runs without installed packages.
import { describe, expect, test } from "bun:test";
import { generateCheckSummary } from "../lib/checkup-summary";

const DAY = 86400;
const now = () => Math.floor(Date.now() / 1000);
type Stats = { stats_reset_epoch: number | null; postmaster_startup_epoch: number | null } | null;
const db = (stats: Stats) => ({ unused_indexes: [], total_count: 1, total_size_bytes: 8192, stats_reset: stats });
const h002 = (dbs: Record<string, ReturnType<typeof db>>) => ({ results: { node1: { data: dbs } } });
const message = (dbs: Record<string, ReturnType<typeof db>>) => generateCheckSummary("H002", h002(dbs)).message;
const NO_NOTE = "Found 1 unused index (8 KiB)";

describe("H002 recent-stats note: the 7-day limit", () => {
  test("stats reset 6.9 days ago get a note, in days", () => {
    const m = message({ db1: db({ stats_reset_epoch: now() - 6.9 * DAY, postmaster_startup_epoch: now() - 90 * DAY }) });
    expect(m).toMatch(/; note: usage stats may cover only \d+ d \(since stats reset\)/);
  });

  test("stats reset 7.1 days ago get no note", () => {
    const m = message({ db1: db({ stats_reset_epoch: now() - 7.1 * DAY, postmaster_startup_epoch: now() - 90 * DAY }) });
    expect(m).toBe(NO_NOTE);
  });
});

describe("H002 recent-stats note: the cause", () => {
  test("a server start newer than the reset says 'since server start'", () => {
    const m = message({ db1: db({ stats_reset_epoch: now() - 60 * DAY, postmaster_startup_epoch: now() - 3 * DAY }) });
    expect(m).toContain("since server start");
    expect(m).not.toContain("since stats reset");
  });

  test("null postmaster_startup_epoch: the reset alone decides", () => {
    const m = message({ db1: db({ stats_reset_epoch: now() - 7200, postmaster_startup_epoch: null }) });
    expect(m).toContain("(since stats reset)");
    expect(m).toContain("2 h");
  });

  test("both epochs null: no note", () => {
    expect(message({ db1: db({ stats_reset_epoch: null, postmaster_startup_epoch: null }) })).toBe(NO_NOTE);
  });
});

describe("H002 recent-stats note: more than one database", () => {
  test("the newest epoch of any database counts, not only the first database", () => {
    const m = generateCheckSummary("H002", h002({
      db1: { ...db({ stats_reset_epoch: now() - 60 * DAY, postmaster_startup_epoch: now() - 90 * DAY }), total_count: 0, total_size_bytes: 0 },
      db2: db({ stats_reset_epoch: now() - 7200, postmaster_startup_epoch: now() - 90 * DAY }),
    })).message;
    expect(m).toContain("2 h (since stats reset)");
  });

  test("two databases, both old: no note", () => {
    const m = generateCheckSummary("H002", h002({
      db1: { ...db({ stats_reset_epoch: now() - 60 * DAY, postmaster_startup_epoch: now() - 90 * DAY }), total_count: 0, total_size_bytes: 0 },
      db2: db({ stats_reset_epoch: now() - 30 * DAY, postmaster_startup_epoch: now() - 90 * DAY }),
    })).message;
    expect(m).toBe(NO_NOTE);
  });
});
