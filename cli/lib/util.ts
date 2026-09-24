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

/** Credential-named key names, shared by the pair matcher and by the guard
 *  that stops a wrapped-tail absorption from swallowing the NEXT pair. */
const SENSITIVE_KEY_SRC =
  "(?:password|passwd|db[-_]?pass|connstr|secret|token|api[-_]?key|private[-_]?key|access[-_]?key|cred(?:ential)?s?|dsn)";

/** That key followed by its `:`/`=` separator, with optional quote and spacing. */
const SENSITIVE_PAIR_SEPARATOR_SRC = `["']?\\s*[:=]`;

/** Characters a credential — or a fragment of one — is made of. */
const CREDENTIAL_CHAR_SRC = "[A-Za-z0-9_+/=~.-]";

/**
 * Shortest run we are willing to treat as credential material.
 *
 * It is both the absorption floor below and the "no usable run survives"
 * threshold the tests assert, deliberately the same number: a run long enough
 * to matter is a run long enough to absorb. Twelve characters is far below any
 * real credential and above the length at which a leftover fragment is worth
 * anything to an attacker.
 */
const MIN_CREDENTIAL_RUN = 12;

/**
 * `key=value` / `key: value` / `"key": "value"` pairs for credential-named keys.
 *
 * Group 1 is the key and separator, 2 a quoted value, 3 a BARE value, and 4 the
 * whitespace-separated runs that FOLLOW a bare value. A bare value can only run
 * to the first whitespace — that is what `[^\s,;&]+` means, and it is right for
 * the unwrapped case — so a value wrapped before we ever saw it (#383) leaves
 * its tail in group 4, where `replaceSensitivePair` decides run by run what is
 * still credential material and what is the next English word.
 *
 * The one thing group 4 will NOT consume is a run that itself opens a
 * credential pair. That guard is what keeps `password=a access_token=b`
 * working: the second pair is left for the next match instead of vanishing into
 * the first one's tail. Everything group 4 does consume is therefore
 * pair-free, so handing an unabsorbed run straight back costs nothing.
 */
const TEXT_SENSITIVE_PAIR = new RegExp(
  `(${SENSITIVE_KEY_SRC}${SENSITIVE_PAIR_SEPARATOR_SRC}\\s*)` +
    `(?:("[^"]*"|'[^']*')` +
    `|([^\\s,;&]+)` +
    `((?:\\s+(?![^\\s,;&]*${SENSITIVE_KEY_SRC}${SENSITIVE_PAIR_SEPARATOR_SRC})[^\\s,;&]+)*)` +
    `)`,
  "gi"
);

/** The whole run is credential characters — no prose punctuation anywhere. */
const CREDENTIAL_RUN_ONLY = new RegExp(`^${CREDENTIAL_CHAR_SRC}+$`);

/** Punctuation that can wrap a run without being part of it. A wrapped tail
 *  quoted or parenthesised by the surrounding text ends in one of these, and
 *  failing to strip them left 27 characters printing (#383 review F3). */
const RUN_LEADING_PUNCTUATION = /^[("'\[{<]+/;
const RUN_TRAILING_PUNCTUATION = /[)"'\]}>!?.,;:]+$/;

/** A consonant run English does not reach: it disqualifies a run as a word and
 *  qualifies it as a token, so the threshold has to be ONE number. */
const LONG_CONSONANT_RUN = /[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5,}/;

/** 8-4-4-4-12 hex. A correlation id a PT4xx names, never a token fragment. */
const UUID_SHAPE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

/**
 * English-word shape: letters only, a vowel, and no consonant run English does
 * not reach. Used to tell "the next word" from "more of the token".
 *
 * Exported because it is the ONE documented reason this scrub can still leave a
 * wrapped credential readable: a fragment that reads as a word breaks the
 * chain. The suite asserts that characterisation against this definition rather
 * than against a copy, so the two cannot drift.
 */
export function isWordShaped(s: string): boolean {
  if (!/^[A-Za-z]+$/.test(s)) return false;
  if (s.length <= 2) return true;
  if (hasInternalCapital(s)) return false; // camelCase is a token, not a word
  if (!/[aeiouyAEIOUY]/.test(s)) return false;
  return !LONG_CONSONANT_RUN.test(s);
}

/** An upper-case letter past the FIRST position, with lower-case present: the
 *  shape of camelCase, never of a capitalised word. One definition, because it
 *  both disqualifies a run as a word and qualifies it as a token. */
function hasInternalCapital(s: string): boolean {
  return /[a-z]/.test(s) && /.[A-Z]/.test(s);
}

/**
 * Does this run carry the shape of a DIAGNOSTIC rather than of a credential?
 *
 * This is the correction the first version of #383 needed. That version asked
 * only "is this English?", and it holds up against English — but almost
 * nothing in these messages is English. They are libpq connection strings,
 * `key=value` fields, paths and correlation ids, and `password=` / `connstr=` /
 * `dsn=` occur in the wild almost exclusively INSIDE a connection string, i.e.
 * surrounded by exactly this. Absorbing it deleted `sslmode=require`,
 * `host=db.example.com` and `request_id=…` on every call, which is a worse
 * outcome than the leak: #383 leaks a tail whose head is already redacted, and
 * needs the value to have been pre-wrapped, while this destroyed the diagnostic
 * every time.
 */
function hasHardStructure(core: string): boolean {
  // `key=value`. A trailing `=` is base64 padding, not an assignment.
  if (/=/.test(core.replace(/={1,2}$/, ""))) return true;
  // No url check is needed: a url carries a `:`, which is not a credential
  // character, so such a run is never a chain member to begin with.
  return UUID_SHAPE.test(core);
}

/**
 * A name made of words and small numbers joined by `/`, `_`, `-` or `.`.
 *
 * The separator cannot decide it on its own -- standard base64 is full of `/`,
 * base64url of `-` and `_`, and a JWT is dotted -- so every segment must read
 * as a word or a small number, and at least one must be a word. That is what
 * separates `platform-fix-383-redact-postgrest-1` from `R7qZmN4v-K1sT0xW8`.
 *
 * Exported for the same reason as `isWordShaped`: together they are the two
 * documented reasons a wrapped credential can still print, and the suite
 * asserts that characterisation against these definitions, not against a copy.
 */
export function isSeparatedName(core: string): boolean {
  const segments = core.split(/[/_.-]/).filter((part) => part.length > 0);
  if (segments.length < 2) return false;
  if (!segments.some(isNameSegment)) return false;
  return segments.every((part) => isNameSegment(part) || /^[0-9]{1,4}$/.test(part));
}

/** A word, or a short all-caps acronym of the kind a diagnostic is built from
 *  (`HTTP/1.1-429`, `PG/16`). The acronym allowance lives HERE rather than in
 *  `isWordShaped` so it only ever decides whether something is a name, and
 *  never whether a chain of credential fragments should break. */
function isNameSegment(part: string): boolean {
  return isWordShaped(part) || (/^[A-Z]+$/.test(part) && part.length <= 6);
}

/**
 * Does this run carry a signal a plain word does not?
 *
 * `-` and `.` are credential characters but NOT signals, so "rate-limited" and
 * a sentence-final "unauthorized." stay. Nor is `/` a signal: it appears in
 * relative paths, and standard base64 long enough to matter carries a digit or
 * mixed case anyway.
 */
function hasTokenSignal(core: string): boolean {
  if (/[0-9_+]/.test(core)) return true;
  if (/=$/.test(core)) return true; // base64 padding
  if (hasInternalCapital(core)) return true;
  return LONG_CONSONANT_RUN.test(core);
}

/** The run, without any punctuation the surrounding text wrapped it in. */
function runCore(run: string): string {
  return run.replace(RUN_LEADING_PUNCTUATION, "").replace(RUN_TRAILING_PUNCTUATION, "");
}

/** Long enough to matter, made of credential characters, not a diagnostic, and
 *  carrying a signal. All four, or it is not credential material. */
function looksLikeCredentialRun(core: string): boolean {
  // Charset is not re-checked: this only ever runs on a chain, and every chain
  // member already had to be credential characters only.
  return core.length >= MIN_CREDENTIAL_RUN && !hasHardStructure(core) && hasTokenSignal(core);
}

/**
 * Can this run be part of a chain of fragments of ONE wrapped credential?
 *
 * Deliberately weaker than `looksLikeCredentialRun`: it has no length floor,
 * because a value wrapped twice leaves a middle fragment that can be a single
 * character. The first version stopped absorbing at the first such fragment and
 * everything after it printed — 300 of 703 double-wrap offset pairs kept >= 12
 * characters, the worst 38 of 39 (#383 review F2). A chain breaks on a word, on
 * a diagnostic, or on prose punctuation, never on a short fragment.
 */
function isChainableFragment(core: string): boolean {
  return (
    core.length > 0 &&
    CREDENTIAL_RUN_ONLY.test(core) &&
    !hasHardStructure(core) &&
    !isWordShaped(core)
  );
}

/**
 * Replace one credential pair, absorbing a wrapped tail but not the diagnostic.
 *
 * The tail is cut into maximal chains of fragments, and a chain is absorbed
 * WHOLE when its concatenation looks like credential material. Judging the
 * chain rather than each run is what handles a value wrapped more than once:
 * the fragments are only a credential when put back together. Anything that
 * breaks a chain — a word, a `key=value`, a path — is never absorbed, so the
 * prose between two token-shaped runs survives both of them.
 */
function replaceSensitivePair(
  _match: string,
  keyPrefix: string,
  quotedValue: string | undefined,
  _bareValue: string | undefined,
  tail: string | undefined
): string {
  // A quoted value already spans whitespace, so it has no wrapped tail and
  // nothing past the closing quote belongs to the credential.
  if (quotedValue !== undefined) return `${keyPrefix}[REDACTED]`;

  const runs = [...(tail ?? "").matchAll(/(\s+)(\S+)/g)].map((m) => {
    const run = m[2] ?? "";
    return { gap: m[1] ?? "", run, core: runCore(run) };
  });

  const absorbed = runs.map(() => false);
  for (let i = 0; i < runs.length; ) {
    if (!isChainableFragment(runs[i]!.core)) {
      i++;
      continue;
    }
    let end = i;
    while (end < runs.length && isChainableFragment(runs[end]!.core)) end++;
    const members = runs.slice(i, end);
    const chain = members.map((r) => r.core).join("");
    // A chain whose every member already reads as a name is a run of adjacent
    // DIAGNOSTICS, not one wrapped token -- concatenating `instance-7-of-12`
    // with `HTTP/1.1-429` produces something that passes every token test.
    const allNames = members.every((r) => isSeparatedName(r.core));
    if (!allNames && looksLikeCredentialRun(chain)) for (let k = i; k < end; k++) absorbed[k] = true;
    i = end;
  }

  let kept = "";
  for (let k = 0; k < runs.length; k++) {
    const r = runs[k]!;
    if (!absorbed[k]) {
      kept += r.gap + r.run;
      continue;
    }
    // The fragment goes; the punctuation the text wrapped it in stays, so a
    // quoted or parenthesised message does not lose its closing mark.
    const lead = r.run.match(RUN_LEADING_PUNCTUATION)?.[0] ?? "";
    const trail = r.run.match(RUN_TRAILING_PUNCTUATION)?.[0] ?? "";
    if (lead || trail) kept += r.gap + lead + trail;
  }
  return `${keyPrefix}[REDACTED]${kept}`;
}

/**
 * Best-effort scrub of credential-looking patterns in plain (non-JSON) text.
 * Used for raw response text that ends up in thrown error messages, where the
 * key-based `redactSecrets` cannot apply. Hygiene only — not an egress control.
 *
 * KNOWN RESIDUALS, every one of them a case where nothing here can tell a
 * credential from text (#383). A single wrap has none: measured over 7
 * credential shapes, 3 key spellings, 4 gap characters and EVERY split offset,
 * 2268 cases leave no usable run. What is left:
 *   - an UNKEYED credential the caller never sent, in arbitrary prose, has no
 *     key name to match and no value to match by. Nothing can reach it;
 *   - a value wrapped MORE THAN ONCE, when a fragment reads as an English word
 *     or when the fragments all read as separated names. The chain of fragments
 *     ends there, because at that point the fragment is indistinguishable from
 *     the next word of the message or from `instance_identifier`. Measured at
 *     552 of 35352 double-wrap cases (1.6%), worst surviving run 16 characters.
 *     Bridging such a fragment was measured and rejected: allowing a
 *     5-character word through cuts the leak to 156 but takes the share of
 *     dictionary words eaten from 0.161% to 5.157%, which is the diagnostic
 *     content these messages exist to carry.
 */
export function redactTextSecrets(text: string): string {
  return text
    .replace(TEXT_URL_USERINFO, "$1:[REDACTED]@")
    .replace(TEXT_SENSITIVE_PAIR, replaceSensitivePair);
}

/**
 * True for C0, DEL, C1, and the two Unicode line separators.
 *
 * ONE definition, deliberately identical to `isControl` in
 * instance-jobs/internal/collect/promql.go, which faces the same problem from
 * the other side (a metric store echoing a submitted expression). U+009B is the
 * single-character CSI — `ESC [` in 8-bit form — so dropping ESC alone is not
 * enough, and U+2028/2029 terminate a line in some renderers.
 */
function isControlCodePoint(cp: number): boolean {
  return cp < 0x20 || cp === 0x7f || (cp >= 0x80 && cp <= 0x9f) || cp === 0x2028 || cp === 0x2029;
}

/**
 * Replace every control character with a space, for text a human will read.
 *
 * A space rather than nothing, matching `StripControlsToSpace` on the Go side
 * and for its reason: deleting them fuses tokens across a line break
 * ("select 1\nfrom t" -> "select 1from t"), which changes what the text says
 * instead of sanitising it.
 *
 * Use this on ANY text an external system supplied that will reach a terminal:
 * ESC can clear the screen and repaint it, BEL rings, and CR can overwrite the
 * line the operator just read.
 */
export function stripControlsToSpace(text: string): string {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    out += cp !== undefined && isControlCodePoint(cp) ? " " : ch;
  }
  return out;
}

/**
 * A regex matching `secret` even if whitespace was injected anywhere inside it.
 *
 * This is what makes a by-value scrub survive an error body that wrapped the
 * credential across lines. Matching the literal cannot: the scrub misses, and a
 * later whitespace-flatten reassembles the secret on one line, one deletion from
 * usable (#382 F1). Every character is regex-escaped and joined with `\s*`;
 * there is no nesting or alternation, so it cannot backtrack pathologically.
 */
export function whitespaceTolerantSecretPattern(secret: string): RegExp {
  const chars = Array.from(secret.replace(/\s+/g, ""));
  const body = chars.map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s*");
  return new RegExp(body, "g");
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
