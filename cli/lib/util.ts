/**
 * Map of HTTP status codes to human-friendly messages.
 */
const HTTP_STATUS_MESSAGES: Record<number, string> = {
  400: "Bad Request",
  401: "Unauthorized - check your API key",
  403: "Forbidden - access denied",
  404: "Not Found",
  408: "Request Timeout",
  429: "Too Many Requests - rate limited",
  500: "Internal Server Error",
  502: "Bad Gateway - server temporarily unavailable",
  503: "Service Unavailable - server temporarily unavailable",
  504: "Gateway Timeout - server temporarily unavailable",
};

/**
 * Check if a string looks like HTML content.
 */
function isHtmlContent(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.startsWith("<!DOCTYPE") || trimmed.startsWith("<html") || trimmed.startsWith("<HTML");
}

/**
 * Remediation hint appended to 401 errors so both humans and AI agents
 * (MCP tool callers) know how to recover from an invalid/stale API key.
 */
const AUTH_REMEDIATION_HINT = "Run 'postgresai auth' to (re)authenticate, or set/update PGAI_API_KEY.";

/**
 * Format an HTTP error response into a clean, developer-friendly message.
 * Handles HTML error pages (e.g., from Cloudflare) by showing just the status code and message.
 * For 401 responses, appends a remediation hint pointing at `postgresai auth`.
 */
export function formatHttpError(operation: string, status: number, responseBody?: string): string {
  const statusMessage = HTTP_STATUS_MESSAGES[status] || "Request failed";
  let errMsg = `${operation}: HTTP ${status} - ${statusMessage}`;
  const remediation = status === 401 ? `\n${AUTH_REMEDIATION_HINT}` : "";

  if (responseBody) {
    // If it's HTML (like Cloudflare error pages), don't dump the raw HTML
    if (isHtmlContent(responseBody)) {
      // Just use the status message, don't append HTML
      return errMsg + remediation;
    }

    // Try to parse as JSON for structured error info
    try {
      const errObj = JSON.parse(responseBody);
      // Extract common error message fields
      const message = errObj.message || errObj.error || errObj.detail;
      if (message && typeof message === "string") {
        errMsg += `\n${message}`;
      } else {
        errMsg += `\n${JSON.stringify(errObj, null, 2)}`;
      }
    } catch {
      // Plain text error - append it if it's short and useful
      const trimmed = responseBody.trim();
      if (trimmed.length > 0 && trimmed.length < 500) {
        errMsg += `\n${trimmed}`;
      }
    }
  }

  return errMsg + remediation;
}

export function maskSecret(secret: string): string {
  if (!secret) return "";
  if (secret.length <= 8) return "****";
  if (secret.length <= 16) return `${secret.slice(0, 4)}${"*".repeat(secret.length - 8)}${secret.slice(-4)}`;
  return `${secret.slice(0, Math.min(12, secret.length - 8))}${"*".repeat(Math.max(4, secret.length - 16))}${secret.slice(-4)}`;
}


export interface RootOptsLike {
  apiBaseUrl?: string;
  uiBaseUrl?: string;
  storageBaseUrl?: string;
}

export interface ConfigLike {
  baseUrl?: string | null;
  storageBaseUrl?: string | null;
}

export interface ResolvedBaseUrls {
  apiBaseUrl: string;
  uiBaseUrl: string;
  storageBaseUrl: string;
}

/**
 * Normalize a base URL by trimming a single trailing slash and validating.
 * @throws Error if the URL is invalid
 */
export function normalizeBaseUrl(value: string): string {
  const trimmed = (value || "").replace(/\/$/, "");
  try {
    // Validate
    // eslint-disable-next-line no-new
    new URL(trimmed);
  } catch {
    throw new Error(`Invalid base URL: ${value}`);
  }
  return trimmed;
}

/**
 * Resolve API and UI base URLs using precedence and normalize them.
 * Precedence (API): opts.apiBaseUrl → env.PGAI_API_BASE_URL → cfg.baseUrl → default
 * Precedence (UI):  opts.uiBaseUrl  → env.PGAI_UI_BASE_URL  → default
 */
export function resolveBaseUrls(
  opts?: RootOptsLike,
  cfg?: ConfigLike,
  defaults: { apiBaseUrl?: string; uiBaseUrl?: string; storageBaseUrl?: string } = {}
): ResolvedBaseUrls {
  const defApi = defaults.apiBaseUrl || "https://postgres.ai/api/general/";
  const defUi = defaults.uiBaseUrl || "https://console.postgres.ai";
  const defStorage = defaults.storageBaseUrl || "https://postgres.ai/storage";

  const apiCandidate = (opts?.apiBaseUrl || process.env.PGAI_API_BASE_URL || cfg?.baseUrl || defApi) as string;
  const uiCandidate = (opts?.uiBaseUrl || process.env.PGAI_UI_BASE_URL || defUi) as string;
  const storageCandidate = (opts?.storageBaseUrl || process.env.PGAI_STORAGE_BASE_URL || cfg?.storageBaseUrl || defStorage) as string;

  return {
    apiBaseUrl: normalizeBaseUrl(apiCandidate),
    uiBaseUrl: normalizeBaseUrl(uiCandidate),
    storageBaseUrl: normalizeBaseUrl(storageCandidate),
  };
}
