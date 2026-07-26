import { createHash } from "node:crypto";
import { generateCheckSummary, type CheckSummary } from "./checkup-summary";

export const CHECKUP_BASELINE_FORMAT = "postgresai-checkup-baseline";
export const CHECKUP_BASELINE_VERSION = 1;

export type FindingSeverity = "info" | "warning" | "high" | "critical";
export type CheckComparisonState =
  | "new"
  | "resolved"
  | "regressed"
  | "improved"
  | "changed"
  | "unchanged";

export interface BaselineCheck {
  checkId: string;
  title: string;
  severity: FindingSeverity;
  message: string;
  fingerprint: string;
}

export interface CheckupBaseline {
  format: typeof CHECKUP_BASELINE_FORMAT;
  formatVersion: typeof CHECKUP_BASELINE_VERSION;
  createdAt: string;
  reports: Record<string, BaselineCheck>;
}

export interface CheckComparison {
  checkId: string;
  title: string;
  state: CheckComparisonState;
  previousSeverity?: FindingSeverity;
  currentSeverity?: FindingSeverity;
  previousMessage?: string;
  currentMessage?: string;
}

export interface CheckupComparison {
  baselineCreatedAt: string;
  comparedAt: string;
  verdict: "pass" | "regression";
  counts: Record<CheckComparisonState, number>;
  checks: CheckComparison[];
}

const SEVERITY_RANK: Record<FindingSeverity, number> = {
  info: 0,
  warning: 1,
  high: 2,
  critical: 3,
};

const VOLATILE_KEYS = new Set([
  "build_ts",
  "created_at",
  "epoch_ns",
  "generated_at",
  "generation_time",
  "last_reset",
  "timestamptz",
  "timestamp",
]);

function normalizeSeverity(value: unknown): FindingSeverity | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === "ok") return "info";
  if (normalized in SEVERITY_RANK) return normalized as FindingSeverity;
  return undefined;
}

function summaryForReport(checkId: string, report: any): {
  severity: FindingSeverity;
  message: string;
} {
  const embedded = report?.summary;
  const embeddedSeverity = normalizeSeverity(embedded?.severity ?? embedded?.status);
  if (embeddedSeverity && typeof embedded?.message === "string") {
    return { severity: embeddedSeverity, message: embedded.message };
  }

  const generated: CheckSummary = generateCheckSummary(checkId, report);
  return {
    severity: normalizeSeverity(generated.status) ?? "info",
    message: generated.message,
  };
}

function stableProjection(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stableProjection);
  }
  if (value && typeof value === "object") {
    const projected: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      if (VOLATILE_KEYS.has(key.toLowerCase())) continue;
      projected[key] = stableProjection((value as Record<string, unknown>)[key]);
    }
    return projected;
  }
  return value;
}

function fingerprintReport(report: unknown): string {
  const stableJson = JSON.stringify(stableProjection(report));
  return createHash("sha256").update(stableJson).digest("hex");
}

export function createCheckupBaseline(
  reports: Record<string, any>,
  createdAt = new Date().toISOString()
): CheckupBaseline {
  const baselineReports: Record<string, BaselineCheck> = {};
  for (const checkId of Object.keys(reports).sort()) {
    const report = reports[checkId];
    const summary = summaryForReport(checkId, report);
    baselineReports[checkId] = {
      checkId,
      title: report?.checkTitle || checkId,
      severity: summary.severity,
      message: summary.message,
      fingerprint: fingerprintReport(report),
    };
  }

  return {
    format: CHECKUP_BASELINE_FORMAT,
    formatVersion: CHECKUP_BASELINE_VERSION,
    createdAt,
    reports: baselineReports,
  };
}

function isBaselineCheck(value: unknown, checkId: string): value is BaselineCheck {
  if (!value || typeof value !== "object") return false;
  const check = value as Partial<BaselineCheck>;
  return check.checkId === checkId
    && typeof check.title === "string"
    && normalizeSeverity(check.severity) !== undefined
    && typeof check.message === "string"
    && /^[a-f0-9]{64}$/.test(check.fingerprint ?? "");
}

export function parseCheckupBaseline(value: unknown): CheckupBaseline {
  if (!value || typeof value !== "object") {
    throw new Error("Invalid checkup baseline: expected a JSON object");
  }
  const baseline = value as Partial<CheckupBaseline>;
  if (baseline.format !== CHECKUP_BASELINE_FORMAT) {
    throw new Error(`Invalid checkup baseline: expected format "${CHECKUP_BASELINE_FORMAT}"`);
  }
  if (baseline.formatVersion !== CHECKUP_BASELINE_VERSION) {
    throw new Error(
      `Unsupported checkup baseline version ${String(baseline.formatVersion)}; expected ${CHECKUP_BASELINE_VERSION}`
    );
  }
  if (typeof baseline.createdAt !== "string" || Number.isNaN(Date.parse(baseline.createdAt))) {
    throw new Error("Invalid checkup baseline: createdAt must be an ISO timestamp");
  }
  if (!baseline.reports || typeof baseline.reports !== "object" || Array.isArray(baseline.reports)) {
    throw new Error("Invalid checkup baseline: reports must be an object");
  }
  for (const [checkId, check] of Object.entries(baseline.reports)) {
    if (!isBaselineCheck(check, checkId)) {
      throw new Error(`Invalid checkup baseline: malformed report entry "${checkId}"`);
    }
  }
  return baseline as CheckupBaseline;
}

export function compareCheckupBaseline(
  baseline: CheckupBaseline,
  currentReports: Record<string, any>,
  comparedAt = new Date().toISOString()
): CheckupComparison {
  const current = createCheckupBaseline(currentReports, comparedAt);
  const checkIds = [...new Set([
    ...Object.keys(baseline.reports),
    ...Object.keys(current.reports),
  ])].sort();
  const counts: Record<CheckComparisonState, number> = {
    new: 0,
    resolved: 0,
    regressed: 0,
    improved: 0,
    changed: 0,
    unchanged: 0,
  };
  const checks: CheckComparison[] = [];

  for (const checkId of checkIds) {
    const previous = baseline.reports[checkId];
    const next = current.reports[checkId];
    let state: CheckComparisonState;
    if (!previous) {
      state = "new";
    } else if (!next) {
      state = "resolved";
    } else if (SEVERITY_RANK[next.severity] > SEVERITY_RANK[previous.severity]) {
      state = "regressed";
    } else if (SEVERITY_RANK[next.severity] < SEVERITY_RANK[previous.severity]) {
      state = "improved";
    } else if (next.fingerprint !== previous.fingerprint || next.message !== previous.message) {
      state = "changed";
    } else {
      state = "unchanged";
    }
    counts[state] += 1;
    checks.push({
      checkId,
      title: next?.title ?? previous?.title ?? checkId,
      state,
      previousSeverity: previous?.severity,
      currentSeverity: next?.severity,
      previousMessage: previous?.message,
      currentMessage: next?.message,
    });
  }

  return {
    baselineCreatedAt: baseline.createdAt,
    comparedAt,
    verdict: counts.regressed > 0 ? "regression" : "pass",
    counts,
    checks,
  };
}

export function failsSeverityThreshold(
  reports: Record<string, any>,
  threshold: FindingSeverity
): boolean {
  const baseline = createCheckupBaseline(reports);
  return Object.values(baseline.reports)
    .some((check) => SEVERITY_RANK[check.severity] >= SEVERITY_RANK[threshold]);
}
