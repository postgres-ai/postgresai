import { randomBytes } from "crypto";
import { SCRAM_DEFAULT_ITERATIONS, scramSha256Verifier } from "./scram";
import { URL, fileURLToPath } from "url";
import type { ConnectionOptions as TlsConnectionOptions } from "tls";
import type { Client as PgClient } from "pg";
import { parse as parseConnString } from "pg-connection-string";
import * as fs from "fs";
import * as path from "path";
import { requireChannelBinding, splitChannelBinding } from "./instances";
import { redactTextSecrets } from "./util";

export const DEFAULT_MONITORING_USER = "postgres_ai_mon";

/**
 * Database provider type. Affects which prepare-db steps are executed.
 * Known providers have specific behavior adjustments; unknown providers use default behavior.
 * TODO: Consider making this more flexible via a config that specifies which steps/checks to skip.
 */
export type DbProvider = string;

/** Known providers with special handling. Unknown providers are treated as self-managed. */
export const KNOWN_PROVIDERS = ["self-managed", "supabase", "clickhouse"] as const;

/** Providers where we skip role creation (users managed externally). */
const SKIP_ROLE_CREATION_PROVIDERS = ["supabase"];

/** Providers where we skip ALTER USER statements (restricted by provider). */
const SKIP_ALTER_USER_PROVIDERS = ["supabase"];

/** Providers where we skip search_path verification (not set via ALTER USER). */
const SKIP_SEARCH_PATH_CHECK_PROVIDERS = ["supabase"];

/** Check if a provider is known and return a warning message if not. */
export function validateProvider(provider: string | undefined): string | null {
  if (!provider || KNOWN_PROVIDERS.includes(provider as any)) return null;
  return `Unknown provider "${provider}". Known providers: ${KNOWN_PROVIDERS.join(", ")}. Treating as self-managed.`;
}

/** Detect a provider from the host of a URI, a libpq conninfo string, or a bare hostname. */
export function detectProvider(conn: string): "clickhouse" | null {
  try {
    const trimmed = conn.trim();
    const host = isLikelyUri(trimmed)
      ? new URL(trimmed).hostname
      : trimmed.includes("=")
        ? parseLibpqConninfo(trimmed).host
        : trimmed;
    const normalized = host?.toLowerCase().replace(/\.$/, "");
    return normalized?.endsWith(".pg.clickhouse.cloud") ? "clickhouse" : null;
  } catch {
    return null;
  }
}

/** Explicit --provider wins; otherwise detect from the connection host. `detected` is true only when auto-detection fired. */
export function resolveProvider(
  explicit: string | undefined,
  conn: string | undefined,
): { provider: string; detected: boolean } {
  // generic: the direct PostgreSQL path of pgai mon deploy, prepared as self-managed.
  if (explicit !== undefined) {
    const p = explicit.trim().toLowerCase();
    return { provider: p === "generic" ? "self-managed" : p, detected: false };
  }
  const provider = conn === undefined ? null : detectProvider(conn);
  return { provider: provider ?? "self-managed", detected: provider !== null };
}

export type PgClientConfig = {
  connectionString?: string;
  enableChannelBinding?: boolean;
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  database?: string;
  ssl?: boolean | TlsConnectionOptions;
};

/**
 * Convert PostgreSQL sslmode to node-postgres ssl config.
 */
function sslModeToConfig(mode: string): boolean | TlsConnectionOptions {
  if (mode.toLowerCase() === "disable") return false;
  if (mode.toLowerCase() === "verify-full" || mode.toLowerCase() === "verify-ca") return true;
  // For require/prefer/allow: encrypt without certificate verification
  return { rejectUnauthorized: false };
}

/** Extract sslmode from a PostgreSQL connection URI. */
function extractSslModeFromUri(uri: string): string | undefined {
  try {
    return new URL(uri).searchParams.get("sslmode") ?? undefined;
  } catch {
    return uri.match(/[?&]sslmode=([^&]+)/i)?.[1];
  }
}

/** Remove sslmode parameter from a PostgreSQL connection URI. */
function stripSslModeFromUri(uri: string): string {
  try {
    const u = new URL(uri);
    u.searchParams.delete("sslmode");
    return u.toString();
  } catch {
    // Fallback regex for malformed URIs
    return uri
      .replace(/[?&]sslmode=[^&]*/gi, "")
      .replace(/\?&/, "?")
      .replace(/\?$/, "");
  }
}

export type AdminConnection = {
  clientConfig: PgClientConfig;
  display: string;
  /** True if SSL fallback is enabled (try SSL first, fall back to non-SSL on failure). */
  sslFallbackEnabled?: boolean;
};

/**
 * Check if an error indicates SSL negotiation failed and fallback to non-SSL should be attempted.
 * This mimics libpq's sslmode=prefer behavior.
 *
 * IMPORTANT: This should NOT match certificate errors (expired, invalid, self-signed)
 * as those are real errors the user needs to fix, not negotiation failures.
 */
function isSslNegotiationError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as any;
  const msg = typeof e.message === "string" ? e.message.toLowerCase() : "";
  const code = typeof e.code === "string" ? e.code : "";

  // Specific patterns that indicate server doesn't support SSL (should fallback)
  const fallbackPatterns = [
    "the server does not support ssl",
    "ssl off",
    "server does not support ssl connections",
  ];

  for (const pattern of fallbackPatterns) {
    if (msg.includes(pattern)) return true;
  }

  // PostgreSQL error code 08P01 (protocol violation) during initial connection
  // often indicates SSL negotiation mismatch, but only if the message suggests it
  if (code === "08P01" && (msg.includes("ssl") || msg.includes("unsupported"))) {
    return true;
  }

  return false;
}

/**
 * The ssl setting and host pg uses: pg parses the connection string over the config, so its TLS
 * parameters (ssl, sslrootcert, sslcert, sslkey) replace `config.ssl`. Falsy ssl: no TLS.
 */
function effectiveConfig(config: PgClientConfig): { ssl: unknown; host?: string } {
  const host = config.host ?? process.env.PGHOST;
  if (!config.connectionString) return { ssl: config.ssl, host };
  try {
    const parsed = parseConnString(config.connectionString);
    // pg turns ssl=no-verify into TLS without a certificate check.
    const ssl = parsed.ssl === "no-verify" ? { rejectUnauthorized: false } : parsed.ssl;
    return { ssl: "ssl" in parsed ? ssl : config.ssl, host: parsed.host || host };
  } catch {
    // An unreadable sslrootcert: pg fails the same way before it connects.
    return { ssl: config.ssl, host };
  }
}

/**
 * Connect to PostgreSQL with sslmode=prefer-like behavior.
 * If sslFallbackEnabled is true, tries SSL first, then falls back to non-SSL on failure.
 * `warn` gets the TLS warnings (default: stderr).
 */
export async function connectWithSslFallback(
  ClientClass: new (config: PgClientConfig) => PgClient,
  adminConn: AdminConnection,
  verbose?: boolean,
  warn: (message: string) => void = (message) => console.error(message),
): Promise<{ client: PgClient; usedSsl: boolean }> {
  const tryConnect = async (config: PgClientConfig): Promise<PgClient> => {
    const client = new ClientClass({ ...config, connectionTimeoutMillis: 10_000 } as any);
    await client.connect();
    // Admin connections run installer and checkup SQL. A database owner can put
    // functions in public (or the role's schema) that shadow pg_catalog built-ins
    // when the server's default search_path is used. Pin it before the first
    // query that resolves SQL names, including on the plaintext fallback path.
    await client.query("set search_path = pg_catalog, pg_temp");
    // Set a default statement timeout to prevent runaway queries
    await client.query("SET statement_timeout = '30s'");
    return client;
  };

  // Warn before sending an administrator password or running setup SQL.
  // sslmode=require encrypts traffic but does not authenticate the server;
  // prefer/allow can additionally fall back to plaintext.
  const { ssl, host } = effectiveConfig(adminConn.clientConfig);
  // A Unix socket has no TLS and does not leave the machine.
  const socket = !!host?.startsWith("/");
  if (!socket && !ssl) {
    warn("Warning: this database connection uses plaintext, without TLS.");
  } else if (!socket && typeof ssl === "object" && (ssl as TlsConnectionOptions).rejectUnauthorized === false) {
    warn(
      "Warning: the database server certificate is not verified" +
      (adminConn.sslFallbackEnabled ? " and this connection may fall back to plaintext" : "") +
      ". Use sslmode=verify-full with a trusted CA to authenticate the server.",
    );
  }

  // If SSL was explicitly set or no SSL configured, just try once
  if (!adminConn.sslFallbackEnabled) {
    const client = await tryConnect(adminConn.clientConfig);
    return { client, usedSsl: !!ssl };
  }

  // sslmode=prefer behavior: try SSL first, fallback to non-SSL
  try {
    const client = await tryConnect(adminConn.clientConfig);
    return { client, usedSsl: !!ssl };
  } catch (sslErr) {
    if (!isSslNegotiationError(sslErr)) {
      // Not an SSL error, don't retry
      throw sslErr;
    }

    if (verbose) {
      console.error("SSL connection failed, retrying without SSL...");
    }

    // Retry without SSL (TLS parameters in the URI keep TLS on)
    const noSslConfig: PgClientConfig = { ...adminConn.clientConfig, ssl: false };
    const retrySsl = effectiveConfig(noSslConfig).ssl;
    try {
      const client = await tryConnect(noSslConfig);
      if (!retrySsl && !socket) warn("Warning: connected to the database without TLS after SSL negotiation failed.");
      return { client, usedSsl: !!retrySsl };
    } catch (noSslErr) {
      // If non-SSL also fails, check if it's "SSL required" - throw that instead
      if (isSslNegotiationError(noSslErr)) {
        const msg = (noSslErr as any)?.message || "";
        if (msg.toLowerCase().includes("ssl") && msg.toLowerCase().includes("required")) {
          // Server requires SSL but SSL attempt failed - throw original SSL error
          throw sslErr;
        }
      }
      // Throw the non-SSL error (it's more relevant since SSL attempt also failed)
      throw noSslErr;
    }
  }
}

export type InitStep = {
  name: string;
  sql: string;
  params?: unknown[];
  optional?: boolean;
};

export type InitPlan = {
  monitoringUser: string;
  database: string;
  steps: InitStep[];
};

/** What each prepare-db step grants the monitoring role, keyed by step name. Keep in sync with cli/sql/*.sql. */
const STEP_SCOPE: Record<string, string> = {
  "01.role": "create/update role",
  "02.extensions": "create extension pg_stat_statements",
  "03.permissions": "connect; pg_monitor, pg_read_all_stats; select on pg_catalog.pg_index; schema postgres_ai (view pg_statistic); usage on schema public; alter user set search_path",
  "06.helpers": "execute on postgres_ai.table_describe (SECURITY INVOKER, catalog only)",
  "04.optional_rds": "execute on rds_tools.pg_ls_multixactdir (RDS only)",
  "05.optional_self_managed": "execute on pg_catalog.pg_ls_dir, pg_catalog.pg_stat_file",
};

/**
 * One-line disclosure of what the plan about to run grants, derived from the steps actually
 * in the plan so it stays honest under --skip-optional-permissions, --reset-password, and
 * provider-specific step filtering.
 */
export function describeInitScope(plan: InitPlan): string {
  const grants = plan.steps.map(step => STEP_SCOPE[step.name] ?? step.name);
  return `-- scope: role ${plan.monitoringUser} gets: ${grants.join(" | ")}; this admin connection is used for this run only and is not stored`;
}

function sqlDir(): string {
  // Handle both development and production paths
  // Development: lib/init.ts -> ../sql
  // Production (bundled): dist/bin/postgres-ai.js -> ../sql (copied during build)
  //
  // IMPORTANT: Use import.meta.url instead of __dirname because bundlers (bun/esbuild)
  // bake in __dirname at build time, while import.meta.url resolves at runtime.
  const currentFile = fileURLToPath(import.meta.url);
  const currentDir = path.dirname(currentFile);

  const candidates = [
    path.resolve(currentDir, "..", "sql"),       // bundled: dist/bin -> dist/sql
    path.resolve(currentDir, "..", "..", "sql"), // dev from lib: lib -> ../sql
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(`SQL directory not found. Searched: ${candidates.join(", ")}`);
}

function loadSqlTemplate(filename: string): string {
  const p = path.join(sqlDir(), filename);
  return fs.readFileSync(p, "utf8");
}

function applyTemplate(sql: string, vars: Record<string, string>): string {
  return sql.replace(/\{\{([A-Z0-9_]+)\}\}/g, (_, key) => {
    const v = vars[key];
    if (v === undefined) throw new Error(`Missing SQL template var: ${key}`);
    return v;
  });
}

export function quoteIdent(ident: string): string {
  // Always quote. Escape embedded quotes by doubling.
  if (ident.includes("\0")) {
    throw new Error("Identifier cannot contain null bytes");
  }
  return `"${ident.replace(/"/g, "\"\"")}"`;
}

export function quoteLiteral(value: string): string {
  // Escape independently of the session's standard_conforming_strings setting.
  // E-strings are needed when the value contains a backslash.
  if (value.includes("\0")) {
    throw new Error("Literal cannot contain null bytes");
  }
  const escapedQuotes = value.replace(/'/g, "''");
  return value.includes("\\")
    ? `E'${escapedQuotes.replace(/\\/g, "\\\\")}'`
    : `'${escapedQuotes}'`;
}

function safeDoDelimiter(...values: string[]): string {
  // A role/database name is allowed to contain $$ or a dollar-quote tag.
  // Choose a tag absent from every interpolated value before forming DO SQL.
  for (let n = 0; ; n++) {
    const tag = `$pgai_do_${n}$`;
    if (values.every(value => !value.includes(tag))) return tag;
  }
}

export function redactPasswordsInSql(sql: string): string {
  // Replace PASSWORD '<literal>' (handles doubled quotes inside).
  return sql.replace(/password\s+'(?:''|[^'])*'/gi, "password '<redacted>'");
}

/** SQL for manual execution, including the mon deploy handoff. Required steps
 * must be atomic; optional failures roll back only their own subtransaction. */
export function renderInitPlanSql(plan: InitPlan): string {
  const lines = [
    "-- Run the complete plan in one session. For psql, use -v ON_ERROR_STOP=1.",
    "begin;",
  ];
  for (const step of plan.steps) {
    const sql = redactPasswordsInSql(step.sql);
    lines.push(`\n-- ${step.name}${step.optional ? " (optional)" : ""}`);
    if (!step.optional) {
      lines.push(sql);
      continue;
    }
    // Dollar tags must not occur in customer-supplied role/database names.
    const delimiter = (base: string): string => {
      let i = 0;
      while (sql.includes(`$${base}_${i}$`)) i++;
      return `$${base}_${i}$`;
    };
    const outer = delimiter("pgai_optional_outer");
    const inner = delimiter("pgai_optional_sql");
    lines.push(`do ${outer}
begin
  begin
    execute ${inner}
${sql}
${inner};
  exception when others then
    raise warning 'optional SQL step ${step.name} skipped: %', sqlerrm;
  end;
end ${outer};`);
  }
  lines.push("commit;");
  return lines.join("\n");
}

export function maskConnectionString(dbUrl: string): string {
  // Hide the password of postgresql://user:pass@host/db, then password-like
  // query parameters (password, sslpassword, pwd). An unencoded "/", "?",
  // "#" or "@" in the password makes the string ambiguous, and the rule
  // prefers hiding too much to printing a password: the password runs to the
  // last "@" that a "/" follows before the next "@" ("@host/db"); without one
  // (no "/db" path), to the first "@" when no "?" or "#" precedes it, else to
  // the last "@". Read as host:port with no password, and printed as given: a
  // "[" first (an IPv6 host), or digits and then "/", ",", "?" or "#"
  // (postgresql://h:5432/db@x, h:5432?application_name=me@corp): a password
  // of that shape in a URL with no "/db" path is the residual.
  let masked = dbUrl;
  const head = dbUrl.match(/^(\s*[a-z][a-z0-9+.-]*:\/\/)([^:@\/?#]*):([\s\S]*)$/i);
  if (head && !head[2].startsWith("[")) {
    // The end is found on a copy with query values blanked ("@" kept, length
    // kept), so a "/" in one (?sslpassword=ab@S3cret/x) is not read as "/db".
    const probe = head[3].replace(/([?&][^=&#?]*=)([^&]*)/g, (_m, key, value) => key + value.replace(/[^@]/g, "*"));
    const end = passwordEnd(probe);
    if (end >= 0) masked = `${head[1]}${head[2]}:*****@${head[3].slice(end + 1)}`;
  }
  // A key and its value run to "&" ("#" included: a fragment after a password
  // is hidden with it). The key is tested in a callback, so a long key with no
  // "=" is scanned once.
  masked = masked.replace(/([?&])([^=&#?]*)=([^&]*)/g, (m, sep, key) => (/pass|pwd/i.test(key) ? `${sep}${key}=*****` : m));
  try {
    const u = new URL(masked);
    if (u.password) u.password = "*****";
    return u.toString();
  } catch {
    return redactTextSecrets(masked);
  }
}

/** Index in `rest` (the text after "user:") of the "@" that ends the password, or -1 when there is none. */
function passwordEnd(rest: string): number {
  const first = rest.indexOf("@");
  if (first < 0) return -1;
  const atSlash = rest.match(/^([\s\S]*)@(?=[^@]*\/)/);
  if (atSlash) return atSlash[1].length;
  const candidate = rest.slice(0, first);
  if (/^[0-9]+[\/,?#]/.test(candidate)) return -1;
  return /[?#]/.test(candidate) ? rest.lastIndexOf("@") : first;
}

function isLikelyUri(value: string): boolean {
  return /^postgres(ql)?:\/\//i.test(value.trim());
}

function tokenizeConninfo(input: string): string[] {
  const s = input.trim();
  const tokens: string[] = [];
  let i = 0;

  const isSpace = (ch: string) => ch === " " || ch === "\t" || ch === "\n" || ch === "\r";

  while (i < s.length) {
    while (i < s.length && isSpace(s[i]!)) i++;
    if (i >= s.length) break;

    let tok = "";
    let inSingle = false;
    while (i < s.length) {
      const ch = s[i]!;
      if (!inSingle && isSpace(ch)) break;

      if (ch === "'" && !inSingle) {
        inSingle = true;
        i++;
        continue;
      }
      if (ch === "'" && inSingle) {
        inSingle = false;
        i++;
        continue;
      }

      if (ch === "\\" && i + 1 < s.length) {
        tok += s[i + 1]!;
        i += 2;
        continue;
      }

      tok += ch;
      i++;
    }

    tokens.push(tok);
    while (i < s.length && isSpace(s[i]!)) i++;
  }

  return tokens;
}

export function parseLibpqConninfo(input: string): PgClientConfig {
  const tokens = tokenizeConninfo(input);
  const cfg: PgClientConfig = {};
  let sslmode: string | undefined;

  for (const t of tokens) {
    const eq = t.indexOf("=");
    if (eq <= 0) continue;
    const key = t.slice(0, eq).trim();
    const rawVal = t.slice(eq + 1);
    const val = rawVal.trim();
    if (!key) continue;

    switch (key) {
      case "host":
        cfg.host = val;
        break;
      case "port": {
        const p = Number(val);
        if (Number.isFinite(p)) cfg.port = p;
        break;
      }
      case "user":
        cfg.user = val;
        break;
      case "password":
        cfg.password = val;
        break;
      case "dbname":
      case "database":
        cfg.database = val;
        break;
      case "sslmode":
        sslmode = val;
        break;
      case "channel_binding":
        if (val === "require") cfg.enableChannelBinding = true;
        break;
      // ignore everything else (options, application_name, etc.)
      default:
        break;
    }
  }

  // Apply SSL configuration based on sslmode
  if (sslmode) {
    cfg.ssl = sslModeToConfig(sslmode);
  }

  return cfg;
}

export function describePgConfig(cfg: PgClientConfig): string {
  if (cfg.connectionString) return maskConnectionString(cfg.connectionString);
  const user = cfg.user ? cfg.user : "<user>";
  const host = cfg.host ? cfg.host : "<host>";
  const port = cfg.port ? String(cfg.port) : "<port>";
  const db = cfg.database ? cfg.database : "<db>";
  // Don't include password
  return `postgresql://${user}:*****@${host}:${port}/${db}`;
}

export function resolveAdminConnection(opts: {
  conn?: string;
  dbUrlFlag?: string;
  host?: string;
  port?: string | number;
  username?: string;
  dbname?: string;
  adminPassword?: string;
  envPassword?: string;
}): AdminConnection {
  const conn = (opts.conn || "").trim();
  const dbUrlFlag = (opts.dbUrlFlag || "").trim();

  // Resolve explicit SSL setting from environment (undefined = auto-detect)
  const explicitSsl = process.env.PGSSLMODE;

  // NOTE: passwords alone (PGPASSWORD / --admin-password) do NOT constitute a connection.
  // We require at least some connection addressing (host/port/user/db) if no positional arg / --db-url is provided.
  const hasConnDetails = !!(opts.host || opts.port || opts.username || opts.dbname);

  if (conn && dbUrlFlag) {
    throw new Error("Provide either positional connection string or --db-url, not both");
  }

  if (conn || dbUrlFlag) {
    const v = conn || dbUrlFlag;
    if (isLikelyUri(v)) {
      const urlSslMode = extractSslModeFromUri(v);
      const effectiveSslMode = urlSslMode || explicitSsl;
      // SSL priority, as in libpq: URL param > PGSSLMODE env > auto (sslmode=prefer behavior)
      const sslConfig = effectiveSslMode
        ? sslModeToConfig(effectiveSslMode)
        : { rejectUnauthorized: false }; // Default: try SSL (with fallback)
      // Enable fallback for: no explicit mode OR explicit "prefer"/"allow"
      const shouldFallback = !effectiveSslMode ||
        effectiveSslMode.toLowerCase() === "prefer" ||
        effectiveSslMode.toLowerCase() === "allow";
      // Strip sslmode from URI so pg uses our ssl config object instead
      const { uri: cleanUri, value: channelBinding } = splitChannelBinding(stripSslModeFromUri(v));
      const enableChannelBinding = requireChannelBinding(channelBinding, sslConfig === false);
      return {
        clientConfig: {
          connectionString: cleanUri,
          ssl: sslConfig,
          ...(enableChannelBinding ? { enableChannelBinding: true } : {}),
        },
        display: maskConnectionString(v),
        // channel_binding=require never falls back to a plaintext connection
        sslFallbackEnabled: shouldFallback && !enableChannelBinding,
      };
    }
    // libpq conninfo (dbname=... host=...)
    const cfg = parseLibpqConninfo(v);
    if (opts.envPassword && !cfg.password) cfg.password = opts.envPassword;
    const cfgHadSsl = cfg.ssl !== undefined;
    if (cfg.ssl === undefined) {
      if (explicitSsl) cfg.ssl = sslModeToConfig(explicitSsl);
      else cfg.ssl = { rejectUnauthorized: false }; // Default: try SSL (with fallback)
    }
    // Enable fallback for: no explicit mode OR explicit "prefer"/"allow"
    const shouldFallback = (!explicitSsl && !cfgHadSsl) ||
      (!!explicitSsl && (explicitSsl.toLowerCase() === "prefer" || explicitSsl.toLowerCase() === "allow"));
    if (cfg.enableChannelBinding) requireChannelBinding("require", cfg.ssl === false);
    return {
      clientConfig: cfg,
      display: describePgConfig(cfg),
      sslFallbackEnabled: shouldFallback && !cfg.enableChannelBinding,
    };
  }

  if (!hasConnDetails) {
    // Keep this message short: the CLI prints full help (including examples) on this error.
    throw new Error("Connection is required.");
  }

  const cfg: PgClientConfig = {};
  if (opts.host) cfg.host = opts.host;
  if (opts.port !== undefined && opts.port !== "") {
    const p = Number(opts.port);
    if (!Number.isFinite(p) || !Number.isInteger(p) || p <= 0 || p > 65535) {
      throw new Error(`Invalid port value: ${String(opts.port)}`);
    }
    cfg.port = p;
  }
  if (opts.username) cfg.user = opts.username;
  if (opts.dbname) cfg.database = opts.dbname;
  if (opts.adminPassword) cfg.password = opts.adminPassword;
  if (opts.envPassword && !cfg.password) cfg.password = opts.envPassword;
  if (explicitSsl) {
    cfg.ssl = sslModeToConfig(explicitSsl);
    // Enable fallback for explicit "prefer"/"allow"
    const shouldFallback = explicitSsl.toLowerCase() === "prefer" || explicitSsl.toLowerCase() === "allow";
    return { clientConfig: cfg, display: describePgConfig(cfg), sslFallbackEnabled: shouldFallback };
  }
  // Default: try SSL with fallback (sslmode=prefer behavior)
  cfg.ssl = { rejectUnauthorized: false };
  return { clientConfig: cfg, display: describePgConfig(cfg), sslFallbackEnabled: true };
}

/**
 * Generate a cryptographically secure random password for the monitoring role.
 *
 * Encoding note — bytes vs output length:
 *   - hex:      N bytes → 2N characters  (24 bytes → 48 hex chars)
 *   - base64:   N bytes → ⌈4N/3⌉ chars   (24 bytes → 32 base64url chars, no padding)
 *
 * We use base64url (RFC 4648 §5) because it is shorter than hex and safe in URLs,
 * connection strings, and shell variables without quoting.
 */
function generateMonitoringPassword(): string {
  // 24 random bytes → 32 base64url characters (no padding).
  // Note: randomBytes() throws on failure; we add a tiny sanity check for unexpected output.
  const password = randomBytes(24).toString("base64url");
  if (password.length < 30) {
    throw new Error("Password generation failed: unexpected output length");
  }
  return password;
}

export async function resolveMonitoringPassword(opts: {
  passwordFlag?: string;
  passwordEnv?: string;
  monitoringUser: string;
}): Promise<{ password: string; generated: boolean }> {
  const fromFlag = (opts.passwordFlag || "").trim();
  if (fromFlag) return { password: fromFlag, generated: false };

  const fromEnv = (opts.passwordEnv || "").trim();
  if (fromEnv) return { password: fromEnv, generated: false };

  // Default: auto-generate (safer than prompting; works in non-interactive mode).
  return { password: generateMonitoringPassword(), generated: true };
}

export async function buildInitPlan(params: {
  database: string;
  monitoringUser?: string;
  monitoringPassword: string;
  /** Server SCRAM iteration count; never use fewer than the default 4096. */
  iterations?: number;
  includeOptionalPermissions: boolean;
  /** Provider type. Affects which steps are included. Defaults to "self-managed". */
  provider?: DbProvider;
  /** Create the role if missing, but never change an existing role's password. */
  keepExistingPassword?: boolean;
}): Promise<InitPlan> {
  // NOTE: kept async for API stability / potential future async template loading.
  const monitoringUser = params.monitoringUser || DEFAULT_MONITORING_USER;
  const database = params.database;
  const provider = params.provider ?? "self-managed";

  const qRole = quoteIdent(monitoringUser);
  const qDb = quoteIdent(database);
  // Ship a pre-computed SCRAM-SHA-256 verifier instead of the cleartext password
  // so the secret never reaches the server — keeping it out of pg_stat_activity,
  // the server log, and pg_stat_statements. Postgres stores the verifier verbatim.
  // Reject null bytes up front (they cannot appear in a password or a SQL literal).
  if (params.monitoringPassword.includes("\0")) {
    throw new Error("Password cannot contain null bytes");
  }
  const serverIterations = Number.isFinite(params.iterations) ? params.iterations! : 0;
  const iterations = Math.max(SCRAM_DEFAULT_ITERATIONS, serverIterations);
  const qPwVerifier = quoteLiteral(scramSha256Verifier(params.monitoringPassword, { iterations }));
  const qRoleNameLit = quoteLiteral(monitoringUser);
  const doQuote = safeDoDelimiter(qRole, qDb, qRoleNameLit, qPwVerifier);

  const steps: InitStep[] = [];

  const vars: Record<string, string> = {
    ROLE_IDENT: qRole,
    DB_IDENT: qDb,
    ROLE_LITERAL: qRoleNameLit,
    DO_QUOTE: doQuote,
  };

  // Some providers (e.g., Supabase) manage users externally - skip role creation.
  // TODO: Make this more flexible by allowing users to specify which steps to skip via config.
  if (!SKIP_ROLE_CREATION_PROVIDERS.includes(provider)) {
    // Role creation/update is done in one template file.
    // Always use a single DO block to avoid race conditions between "role exists?" checks and CREATE USER.
    // We:
    // - create role if missing (another session may create it concurrently: duplicate_object, or
    //   unique_violation when its CREATE raced this one past the exists check),
    // - then ALTER ROLE to ensure the password is set to the desired value.
    const roleStmt = `do ${doQuote} begin
  if not exists (select 1 from pg_catalog.pg_roles where rolname = ${qRoleNameLit}) then
    begin
      create user ${qRole} with password ${qPwVerifier};
    exception when duplicate_object or unique_violation then
      null;
    end;
  end if;${params.keepExistingPassword ? "" : `
  alter user ${qRole} with password ${qPwVerifier};`}
end ${doQuote};`;

    const roleSql = applyTemplate(loadSqlTemplate("01.role.sql"), { ...vars, ROLE_STMT: roleStmt });
    steps.push({ name: "01.role", sql: roleSql });
  }

  // Extensions should be created before permissions (so we can grant permissions on them)
  steps.push({
    name: "02.extensions",
    sql: loadSqlTemplate("02.extensions.sql"),
  });

  let permissionsSql = applyTemplate(loadSqlTemplate("03.permissions.sql"), vars);

  // Some providers restrict ALTER USER - remove those statements.
  // TODO: Make this more flexible by allowing users to specify which statements to skip via config.
  if (SKIP_ALTER_USER_PROVIDERS.includes(provider)) {
    // Remove the entire search_path DO block (marked with SEARCH_PATH_BLOCK_START/END)
    // since it contains ALTER USER and can't be line-filtered without breaking the DO block.
    permissionsSql = permissionsSql.replace(
      /-- \[SEARCH_PATH_BLOCK_START\][\s\S]*?-- \[SEARCH_PATH_BLOCK_END\]\n?/,
      ""
    );
  }

  steps.push({
    name: "03.permissions",
    sql: permissionsSql,
  });

  // Helper functions for table info
  steps.push({
    name: "06.helpers",
    sql: applyTemplate(loadSqlTemplate("06.helpers.sql"), vars),
  });

  if (params.includeOptionalPermissions) {
    steps.push(
      {
        name: "04.optional_rds",
        sql: applyTemplate(loadSqlTemplate("04.optional_rds.sql"), vars),
        optional: true,
      },
      {
        name: "05.optional_self_managed",
        sql: applyTemplate(loadSqlTemplate("05.optional_self_managed.sql"), vars),
        optional: true,
      }
    );
  }

  return { monitoringUser, database, steps };
}

export async function applyInitPlan(params: {
  client: PgClient;
  plan: InitPlan;
  verbose?: boolean;
}): Promise<{ applied: string[]; skippedOptional: string[] }> {
  const applied: string[] = [];
  const skippedOptional: string[] = [];

  // Helper to wrap a step execution in begin/commit
  const executeStep = async (step: InitStep): Promise<void> => {
    await params.client.query("begin;");
    try {
      await params.client.query(step.sql, step.params as any);
      await params.client.query("commit;");
    } catch (e) {
      // Rollback errors should never mask the original failure.
      try {
        await params.client.query("rollback;");
      } catch {
        // ignore
      }
      throw e;
    }
  };

  // Another session granting on the same catalog row at once (two deploys of one
  // database): the step is its own transaction and re-runnable, so it runs again.
  const concurrent = (e: unknown) => /tuple concurrently (updated|deleted)/.test(e instanceof Error ? e.message : String(e));
  const executeRetried = async (step: InitStep): Promise<void> => {
    for (let attempt = 1; ; attempt++) {
      try {
        return await executeStep(step);
      } catch (e) {
        if (attempt >= 3 || !concurrent(e)) throw e;
        await new Promise((r) => setTimeout(r, 200 * attempt));
      }
    }
  };

  // Apply non-optional steps, each in its own transaction
  for (const step of params.plan.steps.filter((s) => !s.optional)) {
    try {
      await executeRetried(step);
      applied.push(step.name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const errAny = e as any;
      const wrapped: any = new Error(`Failed at step "${step.name}": ${redactPasswordsInSql(msg)}`);
      // Preserve useful Postgres error fields so callers can provide better hints / diagnostics.
      const pgErrorFields = [
        "code",
        "detail",
        "hint",
        "position",
        "internalPosition",
        "internalQuery",
        "where",
        "schema",
        "table",
        "column",
        "dataType",
        "constraint",
        "file",
        "line",
        "routine",
      ] as const;
      if (errAny && typeof errAny === "object") {
        for (const field of pgErrorFields) {
          if (errAny[field] !== undefined) {
            wrapped[field] = typeof errAny[field] === "string"
              ? redactPasswordsInSql(errAny[field]) : errAny[field];
          }
        }
      }
      if (e instanceof Error && e.stack) {
        wrapped.stack = redactPasswordsInSql(e.stack);
      }
      throw wrapped;
    }
  }

  // Apply optional steps, each in its own transaction (failure doesn't abort)
  for (const step of params.plan.steps.filter((s) => s.optional)) {
    try {
      await executeStep(step);
      applied.push(step.name);
    } catch {
      skippedOptional.push(step.name);
      // best-effort: ignore
    }
  }

  return { applied, skippedOptional };
}

export type VerifyInitResult = {
  ok: boolean;
  missingRequired: string[];
  missingOptional: string[];
};

/** A single permission check result from the preflight query. */
export type PermissionCheckRow = {
  permission_name: string;
  status: "required" | "optional";
  /**
   * Whether the permission is granted.
   * - `true`  — permission is granted
   * - `false` — permission is explicitly denied
   * - `null`  — check was skipped (e.g., object does not exist, so the privilege
   *             check is inapplicable — such as SELECT on a view that hasn't been created)
   */
  granted: boolean | null;
  fix_command: string | null;
};

/**
 * Result of the preflight permission check for the current DB user.
 *
 * - `ok` is `true` when `missingRequired` is empty.
 * - `rows` contains every check (for inspection / logging).
 * - `missingRequired` / `missingOptional` are filtered subsets of `rows`
 *   where the permission is not granted (`granted !== true`).
 */
export type PreflightPermissionResult = {
  ok: boolean;
  rows: PermissionCheckRow[];
  missingRequired: PermissionCheckRow[];
  missingOptional: PermissionCheckRow[];
};

export type UninitPlan = {
  monitoringUser: string;
  database: string;
  steps: InitStep[];
  /** If true, also drop the monitoring role. If false, only revoke permissions. */
  dropRole: boolean;
};

export async function buildUninitPlan(params: {
  database: string;
  monitoringUser?: string;
  /** If true, drop the role entirely. If false, only revoke permissions/drop objects. */
  dropRole?: boolean;
  /** Provider type. Affects which steps are included. Defaults to "self-managed". */
  provider?: DbProvider;
}): Promise<UninitPlan> {
  const monitoringUser = params.monitoringUser || DEFAULT_MONITORING_USER;
  const database = params.database;
  const provider = params.provider ?? "self-managed";
  const dropRole = params.dropRole ?? true;

  const qRole = quoteIdent(monitoringUser);
  const qDb = quoteIdent(database);
  const qRoleLiteral = quoteLiteral(monitoringUser);
  const doQuote = safeDoDelimiter(qRole, qDb, qRoleLiteral);

  const steps: InitStep[] = [];

  const vars: Record<string, string> = {
    ROLE_IDENT: qRole,
    DB_IDENT: qDb,
    ROLE_LITERAL: qRoleLiteral,
    DO_QUOTE: doQuote,
  };

  // Step 1: Drop helper functions
  steps.push({
    name: "01.drop_helpers",
    sql: applyTemplate(loadSqlTemplate("uninit/01.helpers.sql"), vars),
  });

  // Revoke permissions while preserving a schema that may contain customer objects.
  steps.push({
    name: "02.revoke_permissions",
    sql: applyTemplate(loadSqlTemplate("uninit/02.permissions.sql"), vars),
  });

  // Schema provenance is unknown: prepare-db may have reused a customer schema.
  // Leave it in place; an owner can drop an empty schema explicitly.

  // Drop the role only after every earlier cleanup step succeeded.
  if (dropRole && !SKIP_ROLE_CREATION_PROVIDERS.includes(provider)) {
    steps.push({
      name: "03.drop_role",
      sql: applyTemplate(loadSqlTemplate("uninit/03.role.sql"), vars),
    });
  }

  return { monitoringUser, database, steps, dropRole };
}

export async function applyUninitPlan(params: {
  client: PgClient;
  plan: UninitPlan;
}): Promise<{ applied: string[]; errors: string[]; warnings: string[] }> {
  const applied: string[] = [];
  const errors: string[] = [];
  // Step 02 sends "unprepare-db:" WARNINGs for grants it cannot revoke.
  // Other server WARNINGs are no-op revoke noise and are not shown.
  const warnings: string[] = [];
  const onNotice = (notice: { message?: string }) => {
    if (notice.message?.startsWith("unprepare-db:")) warnings.push(notice.message);
  };
  params.client.on("notice", onNotice);

  // Helper to wrap a step execution in begin/commit
  const executeStep = async (step: InitStep): Promise<void> => {
    await params.client.query("begin;");
    try {
      await params.client.query(step.sql, step.params as any);
      await params.client.query("commit;");
    } catch (e) {
      try {
        await params.client.query("rollback;");
      } catch {
        // ignore
      }
      throw e;
    }
  };

  // Stop on a cleanup failure: later steps (especially role deletion) can
  // otherwise remove unrelated customer objects or hide incomplete revocation.
  for (const step of params.plan.steps) {
    try {
      await executeStep(step);
      applied.push(step.name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const detail = e && typeof e === "object" && "detail" in e && typeof e.detail === "string"
        ? e.detail.replace(/[\r\n]+/g, "; ")
        : "";
      errors.push(`${step.name}: ${msg}${detail ? ` (DETAIL: ${detail})` : ""}`);
      break;
    }
  }
  params.client.off("notice", onNotice);

  // A dropped role holds no grants.
  return { applied, errors, warnings: applied.includes("03.drop_role") ? [] : warnings };
}

/** Split a stored search_path value into schema names: quoted names keep case, others fold to lower case. */
function parseSearchPath(value: string): string[] {
  const names: string[] = [];
  let name = "";
  let quoted = false;
  for (let i = 0; i < value.length; i++) {
    const ch = value[i];
    if (quoted) {
      if (ch === '"' && value[i + 1] === '"') { name += '"'; i++; }
      else if (ch === '"') quoted = false;
      else name += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { names.push(name); name = ""; }
    else if (!/\s/.test(ch)) name += ch.toLowerCase();
  }
  names.push(name);
  return names;
}

/**
 * Check the search_path setting ("search_path=...") against the exact path that
 * 03.permissions.sql sets for this database. Returns the missing-requirement text, or null.
 */
export function roleSearchPathProblem(spLine: string, extSchema?: string | null): string | null {
  const ext = extSchema && extSchema !== "pg_catalog" && extSchema !== "public" ? [extSchema] : [];
  const expected = ["pg_catalog", "postgres_ai", ...ext, "public", "pg_temp"];
  const actual = parseSearchPath(spLine.slice("search_path=".length));
  if (actual.length === expected.length && actual.every((s, i) => s === expected[i])) return null;
  return `role search_path is ${expected.join(", ")}`;
}

/**
 * Security #387: step 06.helpers drops the retired SECURITY DEFINER explain_generic and
 * replaces table_describe with a SECURITY INVOKER version, but only their owner (or a
 * superuser) can. Report each legacy definer helper that survived, for any transport.
 */
export async function legacyHelperWarnings(client: {
  query: (sql: string) => Promise<{ rows?: Array<Record<string, unknown>> }>;
}): Promise<string[]> {
  // Qualified names and an oid = oid comparison: no shadowing object in public can match.
  const r = await client.query(`select
  pg_catalog.to_regprocedure('postgres_ai.explain_generic(text,text,text)') is not null as explain_generic,
  coalesce((select p.prosecdef from pg_catalog.pg_proc p
    where p.oid = pg_catalog.to_regprocedure('postgres_ai.table_describe(text)')::oid), false) as table_describe`);
  const row = r.rows?.[0];
  const legacy = [
    row?.explain_generic ? "postgres_ai.explain_generic(text, text, text)" : null,
    row?.table_describe ? "postgres_ai.table_describe(text)" : null,
  ].filter((f): f is string => f !== null);
  return legacy.map(
    (f) => `legacy SECURITY DEFINER helper ${f} is still installed; its owner or a superuser must drop it, then re-run prepare-db`
  );
}

export async function verifyInitSetup(params: {
  client: PgClient;
  database: string;
  monitoringUser: string;
  includeOptionalPermissions: boolean;
  /** Provider type. Affects which checks are performed. */
  provider?: DbProvider;
}): Promise<VerifyInitResult> {
  // Use a repeatable-read snapshot so all checks see a consistent view.
  await params.client.query("begin isolation level repeatable read;");
  try {
    const missingRequired: string[] = [];
    const missingOptional: string[] = [];

    const role = params.monitoringUser;
    const db = params.database;
    const provider = params.provider ?? "self-managed";

    const roleRes = await params.client.query("select 1 from pg_catalog.pg_roles where rolname = $1", [role]);
    const roleExists = (roleRes.rowCount ?? 0) > 0;
    if (!roleExists) {
      missingRequired.push(`role "${role}" does not exist`);
      // If role is missing, other checks will error or be meaningless.
      return { ok: false, missingRequired, missingOptional };
    }

    const connectRes = await params.client.query(
      "select has_database_privilege($1, $2, 'CONNECT') as ok",
      [role, db]
    );
    if (!connectRes.rows?.[0]?.ok) {
      missingRequired.push(`CONNECT on database "${db}"`);
    }

    const pgMonitorRes = await params.client.query(
      "select pg_has_role($1, 'pg_monitor', 'member') as ok",
      [role]
    );
    if (!pgMonitorRes.rows?.[0]?.ok) {
      missingRequired.push("membership in role pg_monitor");
    }

    const pgIndexRes = await params.client.query(
      "select has_table_privilege($1, 'pg_catalog.pg_index', 'SELECT') as ok",
      [role]
    );
    if (!pgIndexRes.rows?.[0]?.ok) {
      missingRequired.push("SELECT on pg_catalog.pg_index");
    }

    // Check postgres_ai schema exists and is usable
    const schemaExistsRes = await params.client.query(
      "select has_schema_privilege($1, 'postgres_ai', 'USAGE') as ok",
      [role]
    );
    if (!schemaExistsRes.rows?.[0]?.ok) {
      missingRequired.push("USAGE on schema postgres_ai");
    }

    const viewExistsRes = await params.client.query(`
      select case
        when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
        else to_regclass('postgres_ai.pg_statistic') is not null
      end as ok
    `);
    if (!viewExistsRes.rows?.[0]?.ok) {
      missingRequired.push("view postgres_ai.pg_statistic exists");
    } else {
      const viewPrivRes = await params.client.query(
        "select has_table_privilege($1, 'postgres_ai.pg_statistic', 'SELECT') as ok",
        [role]
      );
      if (!viewPrivRes.rows?.[0]?.ok) {
        missingRequired.push("SELECT on view postgres_ai.pg_statistic");
      }
    }

    const schemaUsageRes = await params.client.query(
      "select has_schema_privilege($1, 'public', 'USAGE') as ok",
      [role]
    );
    if (!schemaUsageRes.rows?.[0]?.ok) {
      missingRequired.push("USAGE on schema public");
    }

    // Check access to pg_stat_statements extension schema (may be 'extensions' on Supabase)
    const extSchemaRes = await params.client.query(`
      select n.nspname as schema
      from pg_extension e
      join pg_namespace n on e.extnamespace = n.oid
      where e.extname = 'pg_stat_statements'
    `);
    const extSchema = extSchemaRes.rows?.[0]?.schema;
    if (extSchema && extSchema !== "pg_catalog" && extSchema !== "public") {
      const extSchemaUsageRes = await params.client.query(
        "select has_schema_privilege($1, $2, 'USAGE') as ok",
        [role, extSchema]
      );
      if (!extSchemaUsageRes.rows?.[0]?.ok) {
        missingRequired.push(`USAGE on schema ${extSchema} (pg_stat_statements location)`);
      }
    }

    // Some providers don't allow setting search_path via ALTER USER - skip this check.
    // TODO: Make this more flexible by allowing users to specify which checks to skip via config.
    if (!SKIP_SEARCH_PATH_CHECK_PROVIDERS.includes(provider)) {
      // The role's path in this database: its own row if it sets search_path, else the role-wide row (setdatabase 0).
      const spRes = await params.client.query(`
        select cfg as setting
        from pg_catalog.pg_db_role_setting s, pg_catalog.unnest(s.setconfig) as cfg
        where s.setrole = (select oid from pg_catalog.pg_roles where rolname = $1)
          and s.setdatabase in (0, (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database()))
          and pg_catalog.starts_with(cfg, 'search_path=')
        order by s.setdatabase desc
        limit 1
      `, [role]);
      const spLine = spRes.rows?.[0]?.setting;
      if (typeof spLine !== "string" || !spLine) {
        missingRequired.push("role search_path is set");
      } else {
        const problem = roleSearchPathProblem(spLine, extSchema);
        if (problem) missingRequired.push(problem);
      }
    }

    // Check for helper functions. has_function_privilege raises on a missing function: ok is null then.
    const tableDescribeFnRes = await params.client.query(
      `select case when pg_catalog.to_regprocedure('postgres_ai.table_describe(text)') is not null
        then has_function_privilege($1, 'postgres_ai.table_describe(text)', 'EXECUTE') end as ok`,
      [role]
    );
    if (tableDescribeFnRes.rows?.[0]?.ok === null) {
      missingRequired.push("function postgres_ai.table_describe(text) exists");
    } else if (!tableDescribeFnRes.rows?.[0]?.ok) {
      missingRequired.push("EXECUTE on postgres_ai.table_describe(text)");
    }

    if (params.includeOptionalPermissions) {
      // Optional RDS/Aurora extras
      {
        const extRes = await params.client.query("select 1 from pg_extension where extname = 'rds_tools'");
        if ((extRes.rowCount ?? 0) === 0) {
          missingOptional.push("extension rds_tools");
        } else {
          const fnRes = await params.client.query(
            "select has_function_privilege($1, 'rds_tools.pg_ls_multixactdir()', 'EXECUTE') as ok",
            [role]
          );
          if (!fnRes.rows?.[0]?.ok) {
            missingOptional.push("EXECUTE on rds_tools.pg_ls_multixactdir()");
          }
        }
      }

      // Optional self-managed extras
      const optionalFns = [
        "pg_catalog.pg_stat_file(text)",
        "pg_catalog.pg_stat_file(text, boolean)",
        "pg_catalog.pg_ls_dir(text)",
        "pg_catalog.pg_ls_dir(text, boolean, boolean)",
      ];
      for (const fn of optionalFns) {
        const fnRes = await params.client.query("select has_function_privilege($1, $2, 'EXECUTE') as ok", [role, fn]);
        if (!fnRes.rows?.[0]?.ok) {
          missingOptional.push(`EXECUTE on ${fn}`);
        }
      }
    }

    return { ok: missingRequired.length === 0, missingRequired, missingOptional };
  } finally {
    // Read-only: rollback to release snapshot; do not mask original errors.
    try {
      await params.client.query("rollback;");
    } catch {
      // ignore
    }
  }
}

/**
 * Check that the currently connected DB user has sufficient permissions for
 * monitoring operations. Returns structured results with fix commands.
 *
 * Required permissions cause startup to fail; optional ones produce warnings.
 *
 * @param client  An already-connected PostgreSQL client.
 * @returns       A {@link PreflightPermissionResult} with per-check rows and
 *                filtered `missingRequired` / `missingOptional` arrays.
 * @throws        Propagates database errors (network, permission denied on catalog
 *                tables, timeout) to the caller.
 */
export async function checkCurrentUserPermissions(
  client: PgClient
): Promise<PreflightPermissionResult> {
  const sql = `
    with permission_checks as (
      select
        format('connect on database %I', current_database()) as permission_name,
        'required' as status,
        has_database_privilege(current_user, current_database(), 'connect') as granted

      union all

      select
        'pg_monitor role membership' as permission_name,
        'required' as status,
        -- CASE guarantees evaluation order: pg_has_role() is only called if the
        -- pg_monitor role exists, avoiding ERROR on PostgreSQL < 10 or when dropped.
        case
          when not exists (select from pg_roles where rolname = 'pg_monitor')
          then false
          else pg_has_role(current_user, 'pg_monitor', 'member')
        end as granted

      union all

      select
        'select on pg_catalog.pg_index' as permission_name,
        'required' as status,
        has_table_privilege(current_user, 'pg_catalog.pg_index', 'select') as granted

      union all

      select
        'postgres_ai schema exists' as permission_name,
        'optional' as status,
        to_regnamespace('postgres_ai') is not null as granted

      union all

      select
        'usage on postgres_ai schema' as permission_name,
        'optional' as status,
        case
          when to_regnamespace('postgres_ai') is null then null
          else has_schema_privilege(current_user, 'postgres_ai', 'USAGE')
        end as granted

      union all

      select
        'postgres_ai.pg_statistic view exists' as permission_name,
        'optional' as status,
        -- CASE guarantees evaluation order: has_schema_privilege() RAISES an
        -- error (it does not return false) when the schema does not exist, so
        -- the to_regnamespace guard must come first. A missing schema is
        -- reported by the 'postgres_ai schema exists' check; here it means
        -- skipped (null), not failed.
        case
          when to_regnamespace('postgres_ai') is null then null
          when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
          else to_regclass('postgres_ai.pg_statistic') is not null
        end as granted

      union all

      select
        'select on postgres_ai.pg_statistic' as permission_name,
        'optional' as status,
        -- Same lazy-CASE schema-existence guard as above; null = check skipped.
        case
          when to_regnamespace('postgres_ai') is null then null
          when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
          when to_regclass('postgres_ai.pg_statistic') is null then null
          else has_table_privilege(current_user, 'postgres_ai.pg_statistic', 'select')
        end as granted
    )
    select
      permission_name,
      status,
      granted,
      case
        when status = 'required' and not coalesce(granted, false) then
          case
            when permission_name like 'connect%' then
              format('grant connect on database %I to %I;', current_database(), current_user)
            when permission_name = 'pg_monitor role membership' then
              format('grant pg_monitor to %I;', current_user)
            when permission_name like 'select on pg_catalog.pg_index' then
              format('grant select on pg_catalog.pg_index to %I;', current_user)
          end
        when permission_name = 'postgres_ai schema exists' and granted = false then
          '-- run postgresai prepare-db or create the postgres_ai schema and pg_statistic view manually'
        when permission_name = 'usage on postgres_ai schema' and granted = false then
          format('grant usage on schema postgres_ai to %I;', current_user)
        when permission_name = 'postgres_ai.pg_statistic view exists' and granted = false then
          '-- create postgres_ai.pg_statistic view (see setup script)'
        when permission_name = 'select on postgres_ai.pg_statistic' and granted = false then
          format('grant select on postgres_ai.pg_statistic to %I;', current_user)
        else null
      end as fix_command
    from permission_checks
    order by
      case status when 'required' then 1 else 2 end,
      permission_name;
  `;

  const res = await client.query(sql);
  const rows: PermissionCheckRow[] = res.rows;

  // Required: treat null (skipped) as not-granted — fail safe.
  // Optional: only explicit false counts as missing; null means the check was
  //           skipped (e.g., view doesn't exist) and is not actionable.
  const missingRequired = rows.filter((r) => r.status === "required" && r.granted !== true);
  const missingOptional = rows.filter((r) => r.status === "optional" && r.granted === false);

  return {
    ok: missingRequired.length === 0,
    rows,
    missingRequired,
    missingOptional,
  };
}

/**
 * Format permission check results into user-facing error/warning lines.
 *
 * @returns An object with `warnings` (for optional misses), `errors` (for
 *          required misses including fix SQL), and `failed` (whether required
 *          permissions are missing).
 */
export function formatPermissionCheckMessages(result: PreflightPermissionResult): {
  failed: boolean;
  warnings: string[];
  errors: string[];
} {
  const warnings: string[] = [];
  const errors: string[] = [];

  for (const row of result.missingOptional) {
    if (row.permission_name === "postgres_ai schema exists") {
      warnings.push(
        "Warning: optional: postgres_ai schema not found — F004/F005 (bloat estimates) will be skipped; run prepare-db or create the view manually to enable them."
      );
      continue;
    }
    const fix = row.fix_command ? ` Fix: ${row.fix_command}` : "";
    warnings.push(`Warning: optional permission missing — ${row.permission_name}.${fix}`);
  }

  if (!result.ok) {
    errors.push("Error: the database user is missing required permissions.\n");
    errors.push("Missing permissions:");
    for (const row of result.missingRequired) {
      errors.push(`  - ${row.permission_name}`);
    }
    const fixes = result.missingRequired
      .map((r) => r.fix_command)
      .filter(Boolean);
    if (fixes.length > 0) {
      errors.push("\nTo fix, run the following as a superuser:\n");
      for (const fix of fixes) {
        errors.push(`  ${fix}`);
      }
    }
    errors.push("\nAlternatively, run 'postgresai prepare-db' to set up permissions automatically.");
  }

  return { failed: !result.ok, warnings, errors };
}
