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
 * Standard HTTP reason phrases we should NOT treat as a server-authored
 * message: when the reason phrase equals the stock text for the status, it
 * carries no extra information, so we prefer the friendlier generic label.
 */
const STANDARD_REASON_PHRASES: Record<number, string> = {
  400: "Bad Request",
  401: "Unauthorized",
  403: "Forbidden",
  404: "Not Found",
  408: "Request Timeout",
  409: "Conflict",
  413: "Payload Too Large",
  429: "Too Many Requests",
  500: "Internal Server Error",
  502: "Bad Gateway",
  503: "Service Unavailable",
  504: "Gateway Timeout",
};

/**
 * Format an HTTP error response into a clean, developer-friendly message.
 * Handles HTML error pages (e.g., from Cloudflare) by showing just the status code and message.
 * For 401 responses, appends a remediation hint pointing at `postgresai auth`.
 *
 * The platform's PostgREST layer uses the `PTxyz` custom-status convention:
 * a raised `PT403`/`PT404`/… maps to the HTTP status and delivers the RPC's
 * user-facing message in the HTTP **reason phrase** (`response.statusText`),
 * NOT the JSON body — the body carries only `hint`/`details` (no `message`).
 * So callers pass `statusText` and it is preferred over the built-in generic
 * label. Behind an h2/h3 proxy the reason phrase is dropped entirely
 * (`statusText` is empty), so the JSON body's `code`, `details`, and `hint`
 * are all we get — they must be surfaced too (CLI half of
 * https://gitlab.com/postgres-ai/platform-all/-/issues/537).
 * Headline precedence: JSON body `message` → custom reason phrase → generic
 * label, suffixed with the JSON `code` (e.g. `PT403`) when present; the JSON
 * `details` (plural, PostgREST's spelling) is shown as a supplementary line,
 * and the JSON `hint` (the remediation) as a trailing `Hint:` line.
 */
export function formatHttpError(
  operation: string,
  status: number,
  responseBody?: string,
  statusText?: string
): string {
  const generic = HTTP_STATUS_MESSAGES[status] || "Request failed";
  const remediation = status === 401 ? `\n${AUTH_REMEDIATION_HINT}` : "";

  let bodyMessage: string | undefined;
  let bodyDetails: string | undefined;
  let bodyCode: string | undefined;
  let bodyHint: string | undefined;

  if (responseBody && !isHtmlContent(responseBody)) {
    // If it's HTML (like Cloudflare error pages), we fall through with no
    // parsed fields and never dump the raw HTML.
    try {
      const errObj = JSON.parse(responseBody);
      const message = errObj.message ?? errObj.error;
      if (typeof message === "string" && message.trim().length > 0) {
        bodyMessage = redactTextSecrets(message.trim());
      }
      // PostgREST spells it `details` (plural); accept `detail` too.
      const details = errObj.details ?? errObj.detail;
      if (typeof details === "string" && details.trim().length > 0) {
        bodyDetails = redactTextSecrets(details.trim());
      }
      if (typeof errObj.code === "string" && errObj.code.trim().length > 0) {
        bodyCode = errObj.code.trim();
      }
      if (typeof errObj.hint === "string" && errObj.hint.trim().length > 0) {
        bodyHint = redactTextSecrets(errObj.hint.trim());
      }
      if (
        bodyMessage === undefined &&
        bodyDetails === undefined &&
        bodyCode === undefined &&
        bodyHint === undefined
      ) {
        // A JSON body with none of the known fields (message/error/details/
        // detail/code/hint) still carries the only diagnostic there is —
        // surface it (redacted, compact) instead of silently dropping it.
        bodyDetails = redactSecretsForLog(JSON.stringify(errObj));
      }
    } catch {
      // Plain text error - treat it as the details line if short and useful.
      // Scrubbed: raw error bodies can echo credentials (e.g. a connStr) and
      // this string ends up in thrown Errors / MCP isError responses.
      const trimmed = responseBody.trim();
      if (trimmed.length > 0 && trimmed.length < 500) {
        bodyDetails = redactTextSecrets(trimmed);
      }
    }
  }

  // A custom reason phrase (PTxyz message) is meaningful only when it differs
  // from the stock HTTP reason phrase for this status.
  const trimmedReason = statusText?.trim();
  const reasonPhrase =
    trimmedReason &&
    trimmedReason !== STANDARD_REASON_PHRASES[status] &&
    trimmedReason !== generic
      ? trimmedReason
      : undefined;
  const safeReasonPhrase = reasonPhrase ? redactTextSecrets(reasonPhrase) : undefined;

  const headline = bodyMessage ?? safeReasonPhrase ?? generic;
  const codeSuffix = bodyCode && !headline.includes(bodyCode) ? ` (${bodyCode})` : "";
  let errMsg = `${operation}: HTTP ${status} - ${headline}${codeSuffix}`;
  if (bodyDetails && bodyDetails !== headline) {
    errMsg += `\n${bodyDetails}`;
  }
  if (bodyHint) {
    errMsg += `\nHint: ${bodyHint}`;
  }

  return errMsg + remediation;
}

/**
 * Turn a low-level `fetch` failure into an actionable message. Node's fetch
 * (undici) throws a `TypeError('fetch failed')` and stashes the real cause
 * (`ECONNREFUSED`, DNS failure, `bad port`, TLS error, …) in `err.cause` — the
 * opaque top-level message on its own tells the user nothing. This surfaces the
 * cause and the URL that could not be reached, e.g.
 *   "Failed to list projects: could not reach http://127.0.0.1:1 (ECONNREFUSED)"
 */
export function describeFetchError(operation: string, url: string, err: unknown): string {
  const cause = (err as { cause?: { code?: string; message?: string } } | null | undefined)?.cause;
  const detail =
    cause?.code ||
    cause?.message ||
    (err instanceof Error && err.message ? err.message : String(err));
  return `${operation}: could not reach ${url} (${detail})`;
}

/**
 * An HTTP error response with the status attached, so callers can classify
 * retryability without string-matching the formatted message.
 */
export class HttpStatusError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "HttpStatusError";
    this.status = status;
  }
}

/** Transient statuses worth retrying/resuming on: server errors and rate limits. */
export function isRetryableHttpStatus(status: number): boolean {
  return status >= 500 || status === 429;
}

/** Hard upper bound for a single platform request, even outside Joe polling. */
export const DEFAULT_HTTP_REQUEST_TIMEOUT_MS = 25_000;

export class HttpRequestTimeoutError extends Error {
  constructor(operation: string, timeoutMs: number) {
    super(`${operation}: request timed out after ${timeoutMs}ms`);
    this.name = "HttpRequestTimeoutError";
  }
}

/** Return a finite AbortSignal timeout acceptable to both Node and Bun fetch. */
export function requestTimeoutSignal(timeoutMs?: number): { signal: AbortSignal; timeoutMs: number } {
  const requested =
    typeof timeoutMs === "number" && Number.isFinite(timeoutMs) && timeoutMs > 0
      ? timeoutMs
      : DEFAULT_HTTP_REQUEST_TIMEOUT_MS;
  const bounded = Math.max(1, Math.min(DEFAULT_HTTP_REQUEST_TIMEOUT_MS, Math.floor(requested)));
  return { signal: AbortSignal.timeout(bounded), timeoutMs: bounded };
}

export function isFetchTimeout(err: unknown): boolean {
  const name = (err as { name?: unknown } | null)?.name;
  return name === "AbortError" || name === "TimeoutError";
}

export function maskSecret(secret: string): string {
  if (!secret) return "";
  if (secret.length <= 8) return "****";
  if (secret.length <= 16) return `${secret.slice(0, 4)}${"*".repeat(secret.length - 8)}${secret.slice(-4)}`;
  return `${secret.slice(0, Math.min(12, secret.length - 8))}${"*".repeat(Math.max(4, secret.length - 16))}${secret.slice(-4)}`;
}

/**
 * Credential-bearing field names. Match complete normalized names rather than
 * substrings: MCP results can contain ordinary SQL columns such as `author_id`,
 * `token_type`, `tokens`, and `credited_at`, and corrupting those values is worse
 * than leaving an unfamiliar key untouched. CamelCase is normalized so DBLab's
 * `connStr` and `dbPassword` remain covered.
 */
function isSensitiveLogKey(key: string): boolean {
  const normalized = key
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[-\s]+/g, "_")
    .toLowerCase();
  return /^(?:password|passwd|db_(?:pass|password)|conn_?str|secret|token|api_key|private_key|access_key|access_token|refresh_token|auth|auth_key|auth_token|authorization|credentials?|dsn)$/.test(normalized);
}

/**
 * Return a deep copy with DBLab credential fields removed.
 *
 * Best-effort hygiene only, NOT a security barrier: redaction is key-name
 * based, so secret VALUES under non-matching keys (e.g.
 * `select rolpassword as x from pg_authid`) pass through untouched. The
 * joe:exec scope and the org execution policy are the actual controls.
 */
export function redactSecrets(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(redactSecrets);
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      out[key] = isSensitiveLogKey(key) && child != null
        ? "[REDACTED]"
        : redactSecrets(child);
    }
    return out;
  }
  return value;
}

/**
 * Redact known credential fields from a serialized JSON payload before it is
 * written to a debug log — the body-side counterpart of the `maskSecret`
 * masking the `access-token` header already gets. Debug logging is reachable
 * by MCP callers (`debug: true` is a caller-controlled tool argument), and
 * DBLab bodies carry live credentials: the clone DB password rides in the clone
 * create request (`data.db.password`), and clone create/status replies return
 * the clone's `db.password` / `db.connStr`.
 *
 * A fixed placeholder is used instead of `maskSecret` because short passwords
 * would leak most of their characters through partial masking. Non-JSON input
 * cannot be redacted by key name, so it falls back to the pattern-based
 * `redactTextSecrets` scrub — error paths (parse failures, plain-text error
 * bodies) embed raw response text in thrown Errors that reach MCP `isError`
 * responses and CLI stderr, and must not bypass redaction.
 */
export function redactSecretsForLog(text: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return redactTextSecrets(text);
  }
  return JSON.stringify(redactSecrets(parsed));
}

/** URL userinfo credentials: `scheme://user:password@host` → password redacted. */
const TEXT_URL_USERINFO = /(\b[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+):[^\s/@]+@/gi;

/** `key=value` / `key: value` / `"key": "value"` pairs for credential-named keys. */
const TEXT_SENSITIVE_PAIR =
  /((?:password|passwd|db[-_]?pass|connstr|secret|token|api[-_]?key|private[-_]?key|access[-_]?key|cred(?:ential)?s?|dsn)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;&]+)/gi;

/**
 * Best-effort scrub of credential-looking patterns in plain (non-JSON) text.
 * Used for raw response text that ends up in thrown error messages, where the
 * key-based `redactSecrets` cannot apply. Hygiene only — not an egress control.
 */
export function redactTextSecrets(text: string): string {
  return text
    .replace(TEXT_URL_USERINFO, "$1:[REDACTED]@")
    .replace(TEXT_SENSITIVE_PAIR, "$1[REDACTED]");
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
