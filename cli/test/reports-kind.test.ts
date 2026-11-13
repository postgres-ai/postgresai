import { describe, test, expect, mock, afterEach } from "bun:test";
import {
  fetchReports,
  fetchAllReports,
  reportTypeLabel,
  formatReportsForTerminal,
  ReportKindUnsupportedError,
  type CheckupReport,
} from "../lib/reports";

// #504: health rows (Database health collections) carry created_at = a pin inside
// their period, in the future for an in-progress week or month. `reports list`
// shows display_at and a type label in a terminal, and filters by --kind.

const originalFetch = globalThis.fetch;

const row = (over: Partial<CheckupReport>): CheckupReport => ({
  id: 7,
  org_id: 1,
  org_name: "Acme",
  project_id: 3,
  project_name: "shop-db",
  created_at: "2030-03-31T23:59:30+00:00",
  created_formatted: "2030-03-31 23:59:30",
  epoch: 1,
  status: null,
  ...over,
});

const health = (over: Partial<CheckupReport>): CheckupReport =>
  row({
    kind: "health",
    health_period: "month",
    health_period_start: "2030-03-01T00:00:00+00:00",
    display_at: "2030-03-12T08:15:42.123456+00:00",
    ...over,
  });

describe("reportTypeLabel", () => {
  const now = new Date("2030-03-12T09:00:00Z");

  test("a checkup report", () => {
    expect(reportTypeLabel(row({ kind: "checkup" }), now)).toBe("Checkup");
  });

  test("an older API without kind has no label", () => {
    expect(reportTypeLabel(row({}), now)).toBeNull();
  });

  test("an in-progress month", () => {
    expect(reportTypeLabel(health({}), now)).toBe("Database health · month 2030-03 (in progress)");
  });

  test("a finished month", () => {
    expect(reportTypeLabel(health({ health_period_start: "2030-02-01T00:00:00+00:00" }), now))
      .toBe("Database health · month 2030-02");
  });

  test("a week inside one month, in progress until its Sunday ends", () => {
    const w = health({ health_period: "week", health_period_start: "2030-03-11T00:00:00+00:00" });
    expect(reportTypeLabel(w, now)).toBe("Database health · week 2030-W11 (Mar 11–17) (in progress)");
    expect(reportTypeLabel(w, new Date("2030-03-18T00:00:00Z"))).toBe("Database health · week 2030-W11 (Mar 11–17)");
    expect(reportTypeLabel(w, new Date("2030-03-17T23:59:59Z"))).toBe("Database health · week 2030-W11 (Mar 11–17) (in progress)");
  });

  test("a week across two months", () => {
    expect(reportTypeLabel(health({ health_period: "week", health_period_start: "2030-04-29T00:00:00+00:00" }), new Date("2030-06-01T00:00:00Z")))
      .toBe("Database health · week 2030-W18 (Apr 29–May 5)");
  });

  test("ISO weeks at the year boundary", () => {
    // 2026-12-28 starts ISO week 53 of 2026; 2029-12-31 starts week 1 of 2030.
    expect(reportTypeLabel(health({ health_period: "week", health_period_start: "2026-12-28T00:00:00+00:00" }), now))
      .toBe("Database health · week 2026-W53 (Dec 28–Jan 3)");
    expect(reportTypeLabel(health({ health_period: "week", health_period_start: "2029-12-31T00:00:00+00:00" }), now))
      .toBe("Database health · week 2030-W01 (Dec 31–Jan 6)");
  });

  test("a day stays in progress after its 10:00 pin until midnight UTC", () => {
    const d = health({ health_period: "day", created_at: "2030-03-12T10:00:00+00:00", health_period_start: "2030-03-12T00:00:00+00:00" });
    expect(reportTypeLabel(d, new Date("2030-03-12T23:00:00Z"))).toBe("Database health · day 2030-03-12 (in progress)");
    expect(reportTypeLabel(d, new Date("2030-03-13T00:00:00Z"))).toBe("Database health · day 2030-03-12");
  });

  test("a health row with no known period", () => {
    expect(reportTypeLabel(health({ health_period: null, health_period_start: null }), now)).toBe("Database health");
    // a period this CLI does not know yet is not mistaken for a month
    expect(reportTypeLabel(health({ health_period: "quarter" as never }), now)).toBe("Database health");
  });
});

describe("formatReportsForTerminal", () => {
  const now = new Date("2030-03-12T09:00:00Z");

  test("one block per report with the shown time in UTC; null status left out", () => {
    const text = formatReportsForTerminal(
      [
        health({ id: 1025 }),
        row({ id: 1045, kind: "checkup", created_at: "2030-03-11T19:01:00+00:00", display_at: "2030-03-11T19:01:00+00:00", status: "completed" }),
      ],
      now,
    );
    expect(text).toBe(
      [
        "- id: 1025",
        "  project: shop-db",
        "  type: Database health · month 2030-03 (in progress)",
        "  created: 2030-03-12 08:15:42 UTC",
        "",
        "- id: 1045",
        "  project: shop-db",
        "  type: Checkup",
        "  created: 2030-03-11 19:01:00 UTC",
        "  status: completed",
      ].join("\n"),
    );
    expect(text).not.toContain("2030-03-31");
  });

  test("an older API: created_at as the time, no type line", () => {
    const text = formatReportsForTerminal([row({ id: 9, created_at: "2030-03-10T05:06:07+00:00", status: "pending" })], now);
    expect(text).toBe(["- id: 9", "  project: shop-db", "  created: 2030-03-10 05:06:07 UTC", "  status: pending"].join("\n"));
  });

  test("a failed report shows why", () => {
    const text = formatReportsForTerminal(
      [row({ id: 4, kind: "checkup", status: "failed", status_reason: "H001: upload failed (HTTP 500)", display_at: "2030-03-10T05:06:07+00:00" })],
      now,
    );
    expect(text).toContain("  status: failed\n  status_reason: H001: upload failed (HTTP 500)");
  });

  test("a row without a project name shows the project id", () => {
    expect(formatReportsForTerminal([row({ id: 2, project_name: null as never, project_id: 3 })], now)).toContain("  project: 3");
  });

  test("server text cannot drive the terminal or break the block", () => {
    const text = formatReportsForTerminal(
      [row({ id: 5, project_name: "x\u001b[31mred\u001b]8;;http://evil\u0007", status: "fail\u009b31med\u0085", status_reason: "line one\nline two" })],
      now,
    );
    expect(text).not.toMatch(/[\u0000-\u0009\u000b-\u001f\u007f-\u009f]/);
    expect(text).toContain("  status: fail31med\n");
    expect(text).toContain("  status_reason: line one line two");
    expect(text.split("\n").every((l) => l.startsWith("- ") || l.startsWith("  "))).toBe(true);
  });

  test("an empty list", () => {
    expect(formatReportsForTerminal([], now)).toBe("No reports.");
  });

  test("the time is UTC whatever the local time zone", () => {
    const tz = process.env.TZ;
    process.env.TZ = "America/Los_Angeles";
    try {
      expect(formatReportsForTerminal([health({ id: 1 })], now)).toContain("created: 2030-03-12 08:15:42 UTC");
    } finally {
      if (tz === undefined) delete process.env.TZ;
      else process.env.TZ = tz;
    }
  });
});

describe("fetchReports --kind", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  const reply = (status: number, body: unknown) =>
    mock(() =>
      Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })),
    ) as unknown as typeof fetch;

  test("filters by kind and keeps id order", async () => {
    let seen = "";
    globalThis.fetch = mock((url: string) => {
      seen = url;
      return Promise.resolve(new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } }));
    }) as unknown as typeof fetch;
    await fetchReports({ apiKey: "k", apiBaseUrl: "https://api.example.com", kind: "health" });
    const u = new URL(seen);
    expect(u.searchParams.get("kind")).toBe("eq.health");
    expect(u.searchParams.get("order")).toBe("id.desc");
  });

  test("an API without report kinds gives a clear error", async () => {
    globalThis.fetch = reply(400, {
      code: "42703",
      message: "column checkup_reports.kind does not exist",
      hint: 'Perhaps you meant to reference the column "checkup_reports.id".',
    });
    const err = await fetchReports({ apiKey: "k", apiBaseUrl: "https://api.example.com", kind: "health" }).catch((e) => e);
    expect(err).toBeInstanceOf(ReportKindUnsupportedError);
    expect(err.message).toBe("--kind needs a newer PostgresAI API. Run without --kind to list all reports.");
  });

  test("the same reply without --kind keeps the generic error", async () => {
    globalThis.fetch = reply(400, { code: "42703", message: "column checkup_reports.kind does not exist" });
    const err = await fetchReports({ apiKey: "k", apiBaseUrl: "https://api.example.com" }).catch((e) => e);
    expect(err).not.toBeInstanceOf(ReportKindUnsupportedError);
    expect(err.message).toContain("Failed to fetch reports");
  });

  test("another unknown column with --kind keeps the generic error", async () => {
    globalThis.fetch = reply(400, { code: "42703", message: "column checkup_reports.status does not exist" });
    const err = await fetchReports({ apiKey: "k", apiBaseUrl: "https://api.example.com", kind: "checkup" }).catch((e) => e);
    expect(err).not.toBeInstanceOf(ReportKindUnsupportedError);
  });

  test("--all applies the kind filter on every page", async () => {
    const urls: string[] = [];
    let call = 0;
    globalThis.fetch = mock((url: string) => {
      urls.push(url);
      call++;
      const page = call === 1 ? [{ id: 5 }, { id: 4 }] : [];
      return Promise.resolve(new Response(JSON.stringify(page), { status: 200, headers: { "Content-Type": "application/json" } }));
    }) as unknown as typeof fetch;
    await fetchAllReports({ apiKey: "k", apiBaseUrl: "https://api.example.com", kind: "health", limit: 2 });
    expect(urls.length).toBe(2);
    for (const u of urls) expect(new URL(u).searchParams.get("kind")).toBe("eq.health");
    expect(new URL(urls[1]).searchParams.get("id")).toBe("lt.4");
  });
});
