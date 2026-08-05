import { formatHttpError, maskSecret, normalizeBaseUrl } from "./util";

export interface CheckupReport {
  id: number;
  org_id: number;
  org_name: string;
  project_id: number;
  project_name: string;
  created_at: string;
  created_formatted: string;
  epoch: number;
  status: string;
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

export interface FetchReportsParams {
  apiKey: string;
  apiBaseUrl: string;
  projectId?: number;
  status?: string;
  limit?: number;
  beforeDate?: string;
  /** @internal Used by fetchAllReports for keyset pagination */
  beforeId?: number;
  debug?: boolean;
}

export interface FetchReportFilesParams {
  apiKey: string;
  apiBaseUrl: string;
  reportId?: number;
  type?: "json" | "md";
  checkId?: string;
  debug?: boolean;
}

export interface FetchReportFileDataParams {
  apiKey: string;
  apiBaseUrl: string;
  reportId?: number;
  type?: "json" | "md";
  checkId?: string;
  debug?: boolean;
}

export async function fetchReports(params: FetchReportsParams): Promise<CheckupReport[]> {
  const { apiKey, apiBaseUrl, projectId, status, limit = 20, beforeDate, beforeId, debug } = params;
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
  if (beforeDate) {
    url.searchParams.set("created_at", `lt.${beforeDate}`);
  }
  if (typeof beforeId === "number") {
    url.searchParams.set("id", `lt.${beforeId}`);
  }

  const headers: Record<string, string> = {
    "access-token": apiKey,
    "Prefer": "return=representation",
    "Content-Type": "application/json",
    "Connection": "close",
  };

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
    throw new Error(formatHttpError("Failed to fetch reports", response.status, data));
  }
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
  const { apiKey, apiBaseUrl, reportId, type, checkId, debug } = params;
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

  const headers: Record<string, string> = {
    "access-token": apiKey,
    "Prefer": "return=representation",
    "Content-Type": "application/json",
    "Connection": "close",
  };

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
  const { apiKey, apiBaseUrl, reportId, type, checkId, debug } = params;
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

  const headers: Record<string, string> = {
    "access-token": apiKey,
    "Prefer": "return=representation",
    "Content-Type": "application/json",
    "Connection": "close",
  };

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

/** Lightweight markdown terminal renderer. */
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
