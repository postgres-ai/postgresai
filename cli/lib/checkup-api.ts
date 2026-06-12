import * as http from "http";
import * as https from "https";
import { URL } from "url";
import { normalizeBaseUrl } from "./util";

/**
 * Retry configuration for network operations
 */
export interface RetryConfig {
  maxAttempts: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  initialDelayMs: 1000,
  maxDelayMs: 10000,
  backoffMultiplier: 2,
};

/**
 * Check if an error is retryable (network errors, timeouts, 5xx errors)
 */
function isRetryableError(err: unknown): boolean {
  if (err instanceof RpcError) {
    // Retry on server errors (5xx), not on client errors (4xx)
    return err.statusCode >= 500 && err.statusCode < 600;
  }

  // Check for Node.js error codes (works on Error and Error-like objects)
  if (typeof err === "object" && err !== null && "code" in err) {
    const code = String((err as { code: unknown }).code);
    if (["ECONNRESET", "ECONNREFUSED", "ENOTFOUND", "ETIMEDOUT"].includes(code)) {
      return true;
    }
  }

  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    // Retry on network-related errors based on message content
    return (
      msg.includes("timeout") ||
      msg.includes("timed out") ||
      msg.includes("econnreset") ||
      msg.includes("econnrefused") ||
      msg.includes("enotfound") ||
      msg.includes("socket hang up") ||
      msg.includes("network")
    );
  }

  return false;
}

/**
 * Execute an async function with exponential backoff retry.
 * Retries on network errors, timeouts, and 5xx server errors.
 * Does not retry on 4xx client errors.
 *
 * @param fn - Async function to execute
 * @param config - Optional retry configuration (uses defaults if not provided)
 * @param onRetry - Optional callback invoked before each retry attempt
 * @returns Promise resolving to the function result
 * @throws The last error if all retry attempts fail or error is non-retryable
 *
 * @example
 * ```typescript
 * const result = await withRetry(
 *   () => fetchData(),
 *   { maxAttempts: 3 },
 *   (attempt, err, delay) => console.log(`Retry ${attempt}, waiting ${delay}ms`)
 * );
 * ```
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {},
  onRetry?: (attempt: number, error: unknown, delayMs: number) => void
): Promise<T> {
  const { maxAttempts, initialDelayMs, maxDelayMs, backoffMultiplier } = {
    ...DEFAULT_RETRY_CONFIG,
    ...config,
  };

  let lastError: unknown;
  let delayMs = initialDelayMs;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;

      if (attempt === maxAttempts || !isRetryableError(err)) {
        throw err;
      }

      if (onRetry) {
        onRetry(attempt, err, delayMs);
      }

      await new Promise((resolve) => setTimeout(resolve, delayMs));
      delayMs = Math.min(delayMs * backoffMultiplier, maxDelayMs);
    }
  }

  throw lastError;
}

/**
 * Error thrown when an RPC call to the PostgresAI API fails.
 * Contains detailed information about the failure for debugging and display.
 */
export class RpcError extends Error {
  /** Name of the RPC endpoint that failed */
  rpcName: string;
  /** HTTP status code returned by the server */
  statusCode: number;
  /** Raw response body text */
  payloadText: string;
  /** Parsed JSON response body, or null if parsing failed */
  payloadJson: any | null;

  constructor(params: { rpcName: string; statusCode: number; payloadText: string; payloadJson: any | null }) {
    const { rpcName, statusCode, payloadText, payloadJson } = params;
    super(`RPC ${rpcName} failed: HTTP ${statusCode}`);
    this.name = "RpcError";
    this.rpcName = rpcName;
    this.statusCode = statusCode;
    this.payloadText = payloadText;
    this.payloadJson = payloadJson;
  }
}

/**
 * Format an RpcError for human-readable console display.
 * Extracts message, details, and hint from the error payload if available.
 *
 * @param err - The RpcError to format
 * @returns Array of lines suitable for console output
 */
export function formatRpcErrorForDisplay(err: RpcError): string[] {
  const lines: string[] = [];
  lines.push(`Error: RPC ${err.rpcName} failed: HTTP ${err.statusCode}`);

  const obj = err.payloadJson && typeof err.payloadJson === "object" ? err.payloadJson : null;
  const details = obj && typeof (obj as any).details === "string" ? (obj as any).details : "";
  const hint = obj && typeof (obj as any).hint === "string" ? (obj as any).hint : "";
  const message = obj && typeof (obj as any).message === "string" ? (obj as any).message : "";

  if (message) lines.push(`Message: ${message}`);
  if (details) lines.push(`Details: ${details}`);
  if (hint) lines.push(`Hint: ${hint}`);

  // Fallback to raw payload if we couldn't extract anything useful.
  if (!message && !details && !hint) {
    const t = (err.payloadText || "").trim();
    if (t) lines.push(t);
  }
  return lines;
}

function unwrapRpcResponse(parsed: unknown): any {
  // Some deployments return a plain object, others return an array of rows,
  // and some wrap OUT params under a "result" key.
  if (Array.isArray(parsed)) {
    if (parsed.length === 1) return unwrapRpcResponse(parsed[0]);
    return parsed;
  }
  if (parsed && typeof parsed === "object") {
    const obj = parsed as any;
    if (obj.result !== undefined) return obj.result;
  }
  return parsed as any;
}

// Default timeout for HTTP requests (30 seconds)
const HTTP_TIMEOUT_MS = 30_000;

async function postRpc<T>(params: {
  apiKey: string;
  apiBaseUrl: string;
  rpcName: string;
  bodyObj: Record<string, unknown>;
  timeoutMs?: number;
}): Promise<T> {
  const { apiKey, apiBaseUrl, rpcName, bodyObj, timeoutMs = HTTP_TIMEOUT_MS } = params;

  // NOTE: API key validation removed intentionally to allow markdown conversion without auth.
  // When apiKey is empty, API returns partial markdown (observations only, no full reports).
  // API will return 401/403 for endpoints that require authentication.

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/${rpcName}`);
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = {
    // API key is sent in BOTH header and body (see bodyObj.access_token):
    // - Header: Used by the API gateway/proxy for HTTP authentication
    // - Body: Passed to PostgreSQL RPC function for in-database authorization
    // This is intentional for defense-in-depth; backend validates both.
    "access-token": apiKey,
    "Prefer": "return=representation",
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body).toString(),
  };

  // Use AbortController for clean timeout handling
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let settled = false;

  return new Promise((resolve, reject) => {
    const settledReject = (err: Error) => {
      if (settled) return;
      settled = true;
      if (timeoutId) clearTimeout(timeoutId);
      reject(err);
    };

    const settledResolve = (value: T) => {
      if (settled) return;
      settled = true;
      if (timeoutId) clearTimeout(timeoutId);
      resolve(value);
    };

    // Transport is picked from the URL protocol so the CLI can talk to a
    // local-dev PostgREST over plain HTTP. Production URLs are always HTTPS;
    // to guard against typos (e.g. a missing 's' in 'https://') silently
    // leaking the API key in cleartext, refuse HTTP to non-loopback hosts
    // unless the operator explicitly opts in via CHECKUP_ALLOW_HTTP=1.
    if (url.protocol === "http:") {
      // WHATWG URL keeps IPv6 literals bracketed in .hostname
      // (e.g. `[::1]`), so strip the brackets before matching the allowlist.
      const hostname = url.hostname.replace(/^\[|\]$/g, "");
      const isLoopback = ["localhost", "127.0.0.1", "::1"].includes(hostname);
      if (!isLoopback && process.env.CHECKUP_ALLOW_HTTP !== "1") {
        throw new Error(
          `Refusing to send API key over plaintext HTTP to '${url.host}'. ` +
          `Use https://, a loopback hostname, or set CHECKUP_ALLOW_HTTP=1.`
        );
      }
    }
    const transport = url.protocol === "http:" ? http : https;
    const req = transport.request(
      url,
      {
        method: "POST",
        headers,
        signal: controller.signal,
      },
      (res) => {
        // Response started (headers received) - clear the connection timeout.
        // Once the server starts responding, we let it complete rather than
        // timing out mid-response which would cause confusing errors.
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              const parsed = JSON.parse(data);
              settledResolve(unwrapRpcResponse(parsed) as T);
            } catch {
              settledReject(new Error(`Failed to parse RPC response: ${data}`));
            }
          } else {
            const statusCode = res.statusCode || 0;
            let payloadJson: any | null = null;
            if (data) {
              try {
                payloadJson = JSON.parse(data);
              } catch {
                payloadJson = null;
              }
            }
            settledReject(new RpcError({ rpcName, statusCode, payloadText: data, payloadJson }));
          }
        });
        res.on("error", (err) => {
          settledReject(err);
        });
      }
    );

    // Set up connection timeout - applies until response headers are received.
    // Once response starts, timeout is cleared (see response callback above).
    timeoutId = setTimeout(() => {
      controller.abort();
      req.destroy();  // Backup: ensure request is terminated
      settledReject(new Error(`RPC ${rpcName} timed out after ${timeoutMs}ms (no response)`));
    }, timeoutMs);

    req.on("error", (err: Error) => {
      // Handle abort as timeout (may already be rejected by timeout handler)
      if (err.name === "AbortError" || (err as any).code === "ABORT_ERR") {
        settledReject(new Error(`RPC ${rpcName} timed out after ${timeoutMs}ms`));
        return;
      }
      // Provide clearer error for common network issues
      if ((err as any).code === "ECONNREFUSED") {
        settledReject(new Error(`RPC ${rpcName} failed: connection refused to ${url.host}`));
      } else if ((err as any).code === "ENOTFOUND") {
        settledReject(new Error(`RPC ${rpcName} failed: DNS lookup failed for ${url.host}`));
      } else if ((err as any).code === "ECONNRESET") {
        settledReject(new Error(`RPC ${rpcName} failed: connection reset by server`));
      } else {
        settledReject(err);
      }
    });

    req.write(body);
    req.end();
  });
}

/**
 * Result of an API key pre-flight verification.
 * - "valid": the key was accepted by the API
 * - "invalid": the API definitively rejected the key (HTTP 401/403)
 * - "unknown": verification could not be completed (network error, timeout,
 *   unexpected status) — callers should warn and continue, not block the run
 */
export type ApiKeyVerification =
  | { status: "valid" }
  | { status: "invalid"; statusCode: number }
  | { status: "unknown"; detail: string };

// Timeout for the auth pre-flight (shorter than regular RPC timeout: this is
// an optional fast check and must not noticeably delay the run when the API
// is slow or unreachable).
const VERIFY_API_KEY_TIMEOUT_MS = 10_000;

/**
 * Verify an API key with a cheap, side-effect-free authenticated call
 * (GET /checkup_reports?limit=1 — the same endpoint the `reports` command
 * uses) so expensive work can fail fast on bad credentials.
 *
 * Only a definitive HTTP 401/403 is reported as "invalid". Network errors,
 * timeouts, and unexpected statuses are reported as "unknown" so a transient
 * pre-flight failure never blocks a run that might otherwise succeed.
 */
export async function verifyApiKey(params: {
  apiKey: string;
  apiBaseUrl: string;
  timeoutMs?: number;
}): Promise<ApiKeyVerification> {
  const { apiKey, apiBaseUrl, timeoutMs = VERIFY_API_KEY_TIMEOUT_MS } = params;
  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/checkup_reports`);
  url.searchParams.set("limit", "1");

  // Same plaintext-HTTP guard as postRpc: never send the API key over plain
  // HTTP to a non-loopback host. Report "unknown" rather than aborting the
  // run — the upload path raises the definitive, actionable error.
  if (url.protocol === "http:") {
    const hostname = url.hostname.replace(/^\[|\]$/g, "");
    const isLoopback = ["localhost", "127.0.0.1", "::1"].includes(hostname);
    if (!isLoopback && process.env.CHECKUP_ALLOW_HTTP !== "1") {
      return {
        status: "unknown",
        detail: `refusing to send API key over plaintext HTTP to '${url.host}'`,
      };
    }
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url.toString(), {
      method: "GET",
      headers: { "access-token": apiKey },
      signal: controller.signal,
    });
    // Drain the body so the connection is released cleanly.
    await response.text().catch(() => "");
    if (response.status === 401 || response.status === 403) {
      return { status: "invalid", statusCode: response.status };
    }
    if (response.ok) {
      return { status: "valid" };
    }
    return { status: "unknown", detail: `HTTP ${response.status}` };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { status: "unknown", detail: message };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Create a new checkup report in the PostgresAI backend.
 * This creates the parent report container; individual check results
 * are uploaded separately via uploadCheckupReportJson().
 *
 * @param params - Configuration for report creation
 * @param params.apiKey - PostgresAI API access token
 * @param params.apiBaseUrl - Base URL of the PostgresAI API
 * @param params.project - Project name or ID to associate the report with
 * @param params.status - Optional initial status for the report
 * @returns Promise resolving to the created report ID
 * @throws {RpcError} On API failures (4xx/5xx responses)
 * @throws {Error} On network errors or unexpected response format
 */
export async function createCheckupReport(params: {
  apiKey: string;
  apiBaseUrl: string;
  project: string;
  status?: string;
}): Promise<{ reportId: number }> {
  const { apiKey, apiBaseUrl, project, status } = params;
  const bodyObj: Record<string, unknown> = {
    access_token: apiKey,
    project,
  };
  if (status) bodyObj.status = status;

  const resp = await postRpc<any>({
    apiKey,
    apiBaseUrl,
    rpcName: "checkup_report_create",
    bodyObj,
  });
  const reportId = Number(resp?.report_id);
  if (!Number.isFinite(reportId) || reportId <= 0) {
    throw new Error(`Unexpected checkup_report_create response: ${JSON.stringify(resp)}`);
  }
  return { reportId };
}

/**
 * Upload a JSON check result to an existing checkup report.
 * Each check (e.g., H001, A003) is uploaded as a separate JSON file.
 *
 * @param params - Configuration for the upload
 * @param params.apiKey - PostgresAI API access token
 * @param params.apiBaseUrl - Base URL of the PostgresAI API
 * @param params.reportId - ID of the parent report (from createCheckupReport)
 * @param params.filename - Filename for the uploaded JSON (e.g., "H001.json")
 * @param params.checkId - Check identifier (e.g., "H001", "A003")
 * @param params.jsonText - JSON content as a string
 * @returns Promise resolving to the created report chunk ID
 * @throws {RpcError} On API failures (4xx/5xx responses)
 * @throws {Error} On network errors or unexpected response format
 */
export async function uploadCheckupReportJson(params: {
  apiKey: string;
  apiBaseUrl: string;
  reportId: number;
  filename: string;
  checkId: string;
  jsonText: string;
}): Promise<{ reportChunkId: number }> {
  const { apiKey, apiBaseUrl, reportId, filename, checkId, jsonText } = params;
  const bodyObj: Record<string, unknown> = {
    access_token: apiKey,
    checkup_report_id: reportId,
    filename,
    check_id: checkId,
    data: jsonText,
    type: "json",
    generate_issue: true,
  };

  const resp = await postRpc<any>({
    apiKey,
    apiBaseUrl,
    rpcName: "checkup_report_file_post",
    bodyObj,
  });
  // Backend has a typo: "report_chunck_id" (with 'ck') - handle both spellings for compatibility
  const chunkId = Number(resp?.report_chunck_id ?? resp?.report_chunk_id);
  if (!Number.isFinite(chunkId) || chunkId <= 0) {
    throw new Error(`Unexpected checkup_report_file_post response: ${JSON.stringify(resp)}`);
  }
  return { reportChunkId: chunkId };
}

/**
 * Convert a checkup report JSON to markdown format using the PostgresAI API.
 * This calls the v1.checkup_report_json_to_markdown RPC function.
 *
 * @param params - Configuration for the conversion
 * @param params.apiKey - PostgresAI API access token
 * @param params.apiBaseUrl - Base URL of the PostgresAI API
 * @param params.checkId - Check identifier (e.g., "H001", "A003")
 * @param params.jsonPayload - The JSON data from the check report
 * @param params.reportType - Optional report type parameter
 * @returns Promise resolving to the markdown content as JSON
 * @throws {RpcError} On API failures (4xx/5xx responses)
 * @throws {Error} On network errors or unexpected response format
 */
export async function convertCheckupReportJsonToMarkdown(params: {
  apiKey: string;
  apiBaseUrl: string;
  checkId: string;
  jsonPayload: any;
  reportType?: string;
}): Promise<any> {
  const { apiKey, apiBaseUrl, checkId, jsonPayload, reportType } = params;
  const bodyObj: Record<string, unknown> = {
    check_id: checkId,
    json_payload: jsonPayload,
    access_token: apiKey,
  };

  if (reportType) {
    bodyObj.report_type = reportType;
  }

  const resp = await postRpc<any>({
    apiKey,
    apiBaseUrl,
    rpcName: "checkup_report_json_to_markdown",
    bodyObj,
  });

  return resp;
}
