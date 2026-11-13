import { formatHttpError, maskSecret, normalizeBaseUrl } from "./util";
import { buildAuthHeaders, type OrgScope } from "./org-scope";

// ============================================================================
// Types
// ============================================================================

export interface CheckupReport {
  id: number;
  org_id: number;
  org_name: string;
  project_id: number;
  project_name: string;
  created_at: string;
  created_formatted: string;
  epoch: number;
  status: string | null;
  /** Why a report failed (when the reporter said so). */
  status_reason?: string | null;
  /**
   * Since platform-all #1208 (absent on older APIs): `checkup` is a checkup
   * report; `health` is a Database health collection (AAS, TEMPFILE) for one
   * period, whose created_at is a pin inside that period -- in the future for
   * an in-progress week or month.
   */
  kind?: "checkup" | "health" | null;
  health_period?: "day" | "week" | "month" | null;
  /** Start of the health row's period, UTC. */
  health_period_start?: string | null;
  /** The time to show: created_at for a checkup report; for a health row, when its data was last written. */
  display_at?: string | null;
}

export type ReportKind = "checkup" | "health";

/** `--kind` against an API that does not have report kinds yet (platform-all #1208). */
export class ReportKindUnsupportedError extends Error {
  constructor() {
    super("--kind needs a newer PostgresAI API. Run without --kind to list all reports.");
    this.name = "ReportKindUnsupportedError";
  }
}

export interface CheckupReportFile {
  id: number;
  checkup_report_id: number;
  filename: string;
  check_id: string;
  type: "json" | "md";
  created_at: string;
  created_formatted: string;
  project_id: number;
  project_name: string;
}

export interface CheckupReportFileData extends CheckupReportFile {
  data: string;
}

// ============================================================================
// Date parsing
// ============================================================================

/**
 * Parse a date string in various formats into an ISO 8601 string.
 * Supported formats:
 *   YYYY-MM-DD            2025-01-15
 *   YYYY-MM-DDTHH:mm:ss   2025-01-15T10:30:00
 *   YYYY-MM-DD HH:mm:ss   2025-01-15 10:30:00
 *   YYYY-MM-DD HH:mm      2025-01-15 10:30
 *   DD.MM.YYYY             15.01.2025
 *   DD.MM.YYYY HH:mm      15.01.2025 10:30
 *   DD.MM.YYYY HH:mm:ss   15.01.2025 10:30:00
 */
export function parseFlexibleDate(input: string): string {
  const s = input.trim();

  // DD.MM.YYYY [HH:mm[:ss]]
  const dotMatch = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$/);
  if (dotMatch) {
    const [, dd, mm, yyyy, hh, min, ss] = dotMatch;
    const iso = `${yyyy}-${mm.padStart(2, "0")}-${dd.padStart(2, "0")}T${(hh ?? "00").padStart(2, "0")}:${(min ?? "00").padStart(2, "0")}:${(ss ?? "00").padStart(2, "0")}Z`;
    const d = new Date(iso);
    if (isNaN(d.getTime())) throw new Error(`Invalid date: ${input}`);
    return d.toISOString();
  }

  // YYYY-MM-DD[T ]HH:mm[:ss] or YYYY-MM-DD
  const isoMatch = s.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?$/);
  if (isoMatch) {
    const [, yyyy, mm, dd, hh, min, ss] = isoMatch;
    const iso = `${yyyy}-${mm}-${dd}T${hh ?? "00"}:${min ?? "00"}:${ss ?? "00"}Z`;
    const d = new Date(iso);
    if (isNaN(d.getTime())) throw new Error(`Invalid date: ${input}`);
    return d.toISOString();
  }

  throw new Error(`Unrecognized date format: ${input}. Use YYYY-MM-DD or DD.MM.YYYY`);
}

// ============================================================================
// Params
// ============================================================================

export interface FetchReportsParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  projectId?: number;
  status?: string;
  limit?: number;
  beforeDate?: string;
  /** Only reports of this kind (`kind=eq.<kind>`); needs an API with report kinds. */
  kind?: ReportKind;
  /** @internal Used by fetchAllReports for keyset pagination */
  beforeId?: number;
  debug?: boolean;
}

export interface FetchReportFilesParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  reportId?: number;
  type?: "json" | "md";
  checkId?: string;
  debug?: boolean;
}

export interface FetchReportFileDataParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  reportId?: number;
  type?: "json" | "md";
  checkId?: string;
  debug?: boolean;
}

// ============================================================================
// API functions
// ============================================================================

export async function fetchReports(params: FetchReportsParams): Promise<CheckupReport[]> {
  const { apiKey, apiBaseUrl, orgScope, projectId, status, limit = 20, beforeDate, kind, beforeId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/checkup_reports`);
  url.searchParams.set("order", "id.desc");
  url.searchParams.set("limit", String(limit));
  if (typeof projectId === "number") {
    url.searchParams.set("project_id", `eq.${projectId}`);
  }
  if (status) {
    url.searchParams.set("status", `eq.${status}`);
  }
  if (kind) {
    url.searchParams.set("kind", `eq.${kind}`);
  }
  if (beforeDate) {
    url.searchParams.set("created_at", `lt.${beforeDate}`);
  }
  if (typeof beforeId === "number") {
    url.searchParams.set("id", `lt.${beforeId}`);
  }

  const headers: Record<string, string> = buildAuthHeaders(apiKey, orgScope, {
    Prefer: "return=representation",
  });

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), { method: "GET", headers });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as CheckupReport[];
    } catch {
      throw new Error(`Failed to parse reports response: ${data}`);
    }
  } else {
    if (kind && isMissingKindColumn(response.status, data)) {
      throw new ReportKindUnsupportedError();
    }
    throw new Error(formatHttpError("Failed to fetch reports", response.status, data));
  }
}

/** PostgREST's reply when the view has no `kind` column: HTTP 400, code 42703 naming it. */
function isMissingKindColumn(status: number, body: string): boolean {
  if (status !== 400) return false;
  try {
    const err = JSON.parse(body);
    return err?.code === "42703" && typeof err?.message === "string" && /\bkind\b/.test(err.message);
  } catch {
    return false;
  }
}

// ============================================================================
// Terminal view of `reports list` (#504)
// ============================================================================

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const pad2 = (n: number) => String(n).padStart(2, "0");

/** ISO-8601 week of a UTC date, e.g. "2026-W46". */
function isoWeek(date: Date): string {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7)); // the week's Thursday
  const jan1 = Date.UTC(d.getUTCFullYear(), 0, 1);
  const week = Math.floor((d.getTime() - jan1) / 86_400_000 / 7) + 1;
  return `${d.getUTCFullYear()}-W${pad2(week)}`;
}

function periodEnd(start: Date, period: "day" | "week" | "month"): Date {
  const end = new Date(start);
  if (period === "day") end.setUTCDate(end.getUTCDate() + 1);
  else if (period === "week") end.setUTCDate(end.getUTCDate() + 7);
  else end.setUTCMonth(end.getUTCMonth() + 1);
  return end;
}

function periodText(start: Date, period: "day" | "week" | "month"): string {
  const ym = `${start.getUTCFullYear()}-${pad2(start.getUTCMonth() + 1)}`;
  if (period === "month") return ym;
  if (period === "day") return `${ym}-${pad2(start.getUTCDate())}`;
  const last = new Date(start);
  last.setUTCDate(last.getUTCDate() + 6);
  const to = last.getUTCMonth() === start.getUTCMonth()
    ? String(last.getUTCDate())
    : `${MONTHS[last.getUTCMonth()]} ${last.getUTCDate()}`;
  return `${isoWeek(start)} (${MONTHS[start.getUTCMonth()]} ${start.getUTCDate()}–${to})`;
}

/**
 * The report's type as the Console names it: "Checkup", or "Database health"
 * with its period, e.g. "Database health · week 2026-W46 (Nov 9–15) (in
 * progress)" -- in progress while the period has not ended (UTC). Null when
 * the API does not say (no `kind`).
 */
export function reportTypeLabel(report: CheckupReport, now: Date = new Date()): string | null {
  if (report.kind === "checkup") return "Checkup";
  if (report.kind !== "health") return null;
  const period = report.health_period;
  const start = report.health_period_start ? new Date(report.health_period_start) : null;
  if (period !== "day" && period !== "week" && period !== "month") return "Database health";
  if (!start || Number.isNaN(start.getTime())) return "Database health";
  const inProgress = now.getTime() < periodEnd(start, period).getTime();
  return `Database health · ${period} ${periodText(start, period)}${inProgress ? " (in progress)" : ""}`;
}

/** A server string as one plain terminal line: control characters (ANSI/OSC escapes) dropped, newlines flattened. */
function plain(value: unknown): string {
  return String(value).replace(/[\r\n\t]+/g, " ").replace(/[\u0000-\u001f\u007f-\u009f]/g, "");
}

function utcText(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 19)} UTC`;
}

/**
 * `reports list` in a terminal: one block per report -- id, project, type,
 * created (display_at, else created_at, in UTC), status (when set) and
 * status_reason (when present). `--json` and piped output stay raw rows.
 */
export function formatReportsForTerminal(reports: CheckupReport[], now: Date = new Date()): string {
  if (reports.length === 0) return "No reports.";
  return reports
    .map((r) => {
      const lines = [`- id: ${r.id}`, `  project: ${plain(r.project_name ?? r.project_id)}`];
      const type = reportTypeLabel(r, now);
      if (type) lines.push(`  type: ${type}`);
      lines.push(`  created: ${plain(utcText(r.display_at || r.created_at))}`);
      if (r.status) lines.push(`  status: ${plain(r.status)}`);
      if (r.status_reason) lines.push(`  status_reason: ${plain(r.status_reason)}`);
      return lines.join("\n");
    })
    .join("\n\n");
}

const MAX_ALL_REPORTS = 10000;

export async function fetchAllReports(params: Omit<FetchReportsParams, "beforeId" | "beforeDate">): Promise<CheckupReport[]> {
  const pageSize = params.limit ?? 100;
  const all: CheckupReport[] = [];
  let beforeId: number | undefined;

  while (true) {
    const page = await fetchReports({ ...params, limit: pageSize, beforeId });
    if (page.length === 0) break;
    all.push(...page);
    if (all.length >= MAX_ALL_REPORTS) {
      console.warn(`Warning: reached maximum of ${MAX_ALL_REPORTS} reports, stopping pagination`);
      break;
    }
    beforeId = page[page.length - 1].id;
    if (page.length < pageSize) break;
  }

  return all;
}

export async function fetchReportFiles(params: FetchReportFilesParams): Promise<CheckupReportFile[]> {
  const { apiKey, apiBaseUrl, orgScope, reportId, type, checkId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (reportId === undefined && !checkId) {
    throw new Error("Either reportId or checkId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/checkup_report_files`);
  if (typeof reportId === "number") {
    url.searchParams.set("checkup_report_id", `eq.${reportId}`);
  }
  url.searchParams.set("order", "id.asc");
  if (type) {
    url.searchParams.set("type", `eq.${type}`);
  }
  if (checkId) {
    url.searchParams.set("check_id", `eq.${checkId}`);
  }

  const headers: Record<string, string> = buildAuthHeaders(apiKey, orgScope, {
    Prefer: "return=representation",
  });

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), { method: "GET", headers });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as CheckupReportFile[];
    } catch {
      throw new Error(`Failed to parse report files response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch report files", response.status, data));
  }
}

export async function fetchReportFileData(params: FetchReportFileDataParams): Promise<CheckupReportFileData[]> {
  const { apiKey, apiBaseUrl, orgScope, reportId, type, checkId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (reportId === undefined && !checkId) {
    throw new Error("Either reportId or checkId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/checkup_report_file_data`);
  if (typeof reportId === "number") {
    url.searchParams.set("checkup_report_id", `eq.${reportId}`);
  }
  url.searchParams.set("order", "id.asc");
  if (type) {
    url.searchParams.set("type", `eq.${type}`);
  }
  if (checkId) {
    url.searchParams.set("check_id", `eq.${checkId}`);
  }

  const headers: Record<string, string> = buildAuthHeaders(apiKey, orgScope, {
    Prefer: "return=representation",
  });

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), { method: "GET", headers });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as CheckupReportFileData[];
    } catch {
      throw new Error(`Failed to parse report file data response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch report file data", response.status, data));
  }
}

// ============================================================================
// Lightweight markdown terminal renderer
// ============================================================================

export function renderMarkdownForTerminal(md: string): string {
  if (!md) return "";

  const RESET = "\x1b[0m";
  const BOLD = "\x1b[1m";
  const BOLD_UNDERLINE = "\x1b[1;4m";
  const DIM = "\x1b[2m";
  const ITALIC = "\x1b[3m";
  const CYAN = "\x1b[36m";

  const lines = md.split("\n");
  const output: string[] = [];
  let inCodeBlock = false;

  for (const line of lines) {
    // Code block toggle
    if (line.trimStart().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      if (inCodeBlock) {
        output.push(`${DIM}${"─".repeat(40)}${RESET}`);
      } else {
        output.push(`${DIM}${"─".repeat(40)}${RESET}`);
      }
      continue;
    }

    // Inside code block — dim output
    if (inCodeBlock) {
      output.push(`${DIM}  ${line}${RESET}`);
      continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(line.trim()) || /^\*{3,}$/.test(line.trim()) || /^_{3,}$/.test(line.trim())) {
      output.push(`${DIM}${"─".repeat(60)}${RESET}`);
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.*)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      if (level === 1) {
        output.push(`${BOLD_UNDERLINE}${text}${RESET}`);
      } else {
        output.push(`${BOLD}${text}${RESET}`);
      }
      continue;
    }

    // Inline formatting
    let formatted = line;
    // Bold: **text** or __text__
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, `${BOLD}$1${RESET}`);
    formatted = formatted.replace(/__(.+?)__/g, `${BOLD}$1${RESET}`);
    // Italic: *text* (only single, not inside **)
    formatted = formatted.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, `${ITALIC}$1${RESET}`);
    // Italic: _text_ — only at word boundaries (not inside identifiers like foo_bar_baz)
    formatted = formatted.replace(/(?<=^|[\s(])_([^\s_](?:.*?[^\s_])?)_(?=$|[\s),.:;!?])/g, `${ITALIC}$1${RESET}`);
    // Inline code: `text`
    formatted = formatted.replace(/`([^`]+)`/g, `${CYAN}$1${RESET}`);

    output.push(formatted);
  }

  return output.join("\n");
}
