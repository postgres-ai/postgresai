import { describe, expect, test } from "bun:test";
import {
  CHECKUP_BASELINE_FORMAT,
  CHECKUP_BASELINE_VERSION,
  compareCheckupBaseline,
  createCheckupBaseline,
  failsSeverityThreshold,
  parseCheckupBaseline,
} from "../lib/checkup-baseline";

function report(
  checkId: string,
  status: string,
  message: string,
  data: Record<string, unknown> = {}
) {
  return {
    checkId,
    checkTitle: `Check ${checkId}`,
    timestamptz: "2026-01-01T00:00:00.000Z",
    summary: { status, message },
    results: { node: { data } },
  };
}

describe("checkup baselines", () => {
  test("creates a deterministic, versioned baseline", () => {
    const first = createCheckupBaseline({
      H001: report("H001", "warning", "Invalid indexes", { count: 2 }),
    }, "2026-01-02T00:00:00.000Z");
    const second = createCheckupBaseline({
      H001: {
        ...report("H001", "warning", "Invalid indexes", { count: 2 }),
        timestamptz: "2026-01-03T00:00:00.000Z",
        build_ts: "2026-01-03T00:00:00.000Z",
      },
    }, "2026-01-04T00:00:00.000Z");

    expect(first.format).toBe(CHECKUP_BASELINE_FORMAT);
    expect(first.formatVersion).toBe(CHECKUP_BASELINE_VERSION);
    expect(first.reports.H001.fingerprint).toBe(second.reports.H001.fingerprint);
  });

  test("rejects unknown baseline versions and malformed entries", () => {
    expect(() => parseCheckupBaseline({
      format: CHECKUP_BASELINE_FORMAT,
      formatVersion: 99,
      createdAt: new Date().toISOString(),
      reports: {},
    })).toThrow("Unsupported checkup baseline version");

    expect(() => parseCheckupBaseline({
      format: CHECKUP_BASELINE_FORMAT,
      formatVersion: CHECKUP_BASELINE_VERSION,
      createdAt: new Date().toISOString(),
      reports: { H001: { checkId: "H001" } },
    })).toThrow('malformed report entry "H001"');
  });

  test("classifies new, resolved, regressed, improved, changed, and unchanged checks", () => {
    const baseline = createCheckupBaseline({
      A001: report("A001", "info", "all clear"),
      A002: report("A002", "warning", "warning"),
      A003: report("A003", "warning", "warning", { count: 1 }),
      A004: report("A004", "warning", "warning", { count: 1 }),
      A005: report("A005", "info", "gone"),
    }, "2026-01-01T00:00:00.000Z");

    const comparison = compareCheckupBaseline(baseline, {
      A001: report("A001", "warning", "now warning"),
      A002: report("A002", "info", "fixed"),
      A003: report("A003", "warning", "warning", { count: 2 }),
      A004: report("A004", "warning", "warning", { count: 1 }),
      A006: report("A006", "info", "new check"),
    }, "2026-01-02T00:00:00.000Z");

    expect(comparison.verdict).toBe("regression");
    expect(Object.fromEntries(comparison.checks.map((check) => [check.checkId, check.state]))).toEqual({
      A001: "regressed",
      A002: "improved",
      A003: "changed",
      A004: "unchanged",
      A005: "resolved",
      A006: "new",
    });
    expect(comparison.counts).toEqual({
      new: 1,
      resolved: 1,
      regressed: 1,
      improved: 1,
      changed: 1,
      unchanged: 1,
    });
  });

  test("supports warning, high, and critical thresholds", () => {
    const reports = {
      A001: report("A001", "warning", "warning"),
      A002: report("A002", "high", "high"),
    };
    expect(failsSeverityThreshold(reports, "warning")).toBe(true);
    expect(failsSeverityThreshold(reports, "high")).toBe(true);
    expect(failsSeverityThreshold(reports, "critical")).toBe(false);
  });

  test("falls back to the existing local summary when no summary is embedded", () => {
    const baseline = createCheckupBaseline({
      H001: {
        checkId: "H001",
        checkTitle: "Invalid indexes",
        results: {
          node: {
            data: {
              db: { total_count: 1, total_size_bytes: 1024 },
            },
          },
        },
      },
    });
    expect(baseline.reports.H001.severity).toBe("warning");
    expect(baseline.reports.H001.message).toContain("invalid index");
  });
});
