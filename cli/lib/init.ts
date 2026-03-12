import { randomBytes } from "crypto";
import { URL, fileURLToPath } from "url";
import type { ConnectionOptions as TlsConnectionOptions } from "tls";
import type { Client as PgClient } from "pg";
import * as fs from "fs";
import * as path from "path";

export const DEFAULT_MONITORING_USER = "postgres_ai_mon";

/**
 * Database provider type. Affects which prepare-db steps are executed.
 * Known providers have specific behavior adjustments; unknown providers use default behavior.
 * TODO: Consider auto-detecting provider from connection string or server version string.
 * TODO: Consider making this more flexible via a config that specifies which steps/checks to skip.
 */
export type DbProvider = string;

/** Known providers with special handling. Unknown providers are treated as self-managed. */
export const KNOWN_PROVIDERS = ["self-managed", "supabase"] as const;

/** Providers where we skip role creation (users managed externally). */
const SKIP_ROLE_CREATION_PROVIDERS = ["supabase"];

/** Providers where we skip ALTER USER statements (restricted by provider). */
const SKIP_ALTER_USER_PROVIDERS = ["supabase"];

/** Providers where we skip search_path verification (not set via ALTER USER). */
const SKIP_SEARCH_PATH_CHECK_PROVIDERS = ["supabase"];

/** Check if a provider is known and return a warning message if not. */
export function validateProvider(provider: string | undefined): string | null {
  if (!provider || (KNOWN_PROVIDERS as readonly string[]).includes(provider)) return null;
  return `Unknown provider "${provider}". Known providers: ${KNOWN_PROVIDERS.join(", ")}. Treating as self-managed.`;
}

export type PgClientConfig = {
  connectionString?: string;
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
 * Connect to PostgreSQL with sslmode=prefer-like behavior.
 * If sslFallbackEnabled is true, tries SSL first, then falls back to non-SSL on failure.
 */
export async function connectWithSslFallback(
  ClientClass: new (config: PgClientConfig) => PgClient,
  adminConn: AdminConnection,
  verbose?: boolean
): Promise<{ client: PgClient; usedSsl: boolean }> {
  const tryConnect = async (config: PgClientConfig): Promise<PgClient> => {
    const client = new ClientClass(config);
    await client.connect();
    return client;
  };

  // If SSL was explicitly set or no SSL configured, just try once
  if (!adminConn.sslFallbackEnabled) {
    const client = await tryConnect(adminConn.clientConfig);
    return { client, usedSsl: !!adminConn.clientConfig.ssl };
  }

  // sslmode=prefer behavior: try SSL first, fallback to non-SSL
  try {
    const client = await tryConnect(adminConn.clientConfig);
    return { client, usedSsl: true };
  } catch (sslErr) {
    if (!isSslNegotiationError(sslErr)) {
      // Not an SSL error, don't retry
      throw sslErr;
    }

    if (verbose) {
      console.error("SSL connection failed, retrying without SSL...");
    }

    // Retry without SSL
    const noSslConfig: PgClientConfig = { ...adminConn.clientConfig, ssl: false };
    try {
      const client = await tryConnect(noSslConfig);
      return { client, usedSsl: false };
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

function quoteIdent(ident: string): string {
  // Always quote. Escape embedded quotes by doubling.
  if (ident.includes("\0")) {
    throw new Error("Identifier cannot contain null bytes");
  }
  return `"${ident.replace(/"/g, "\"\"")}"`;
}

function quoteLiteral(value: string): string {
  // Single-quote and escape embedded quotes by doubling.
  // This is used where Postgres grammar requires a literal (e.g., CREATE/ALTER ROLE PASSWORD).
  if (value.includes("\0")) {
    throw new Error("Literal cannot contain null bytes");
  }
  return `'${value.replace(/'/g, "''")}'`;
}

export function redactPasswordsInSql(sql: string): string {
  // Replace PASSWORD '<literal>' (handles doubled quotes inside).
  return sql.replace(/password\s+'(?:''|[^'])*'/gi, "password '<redacted>'");
}

export function maskConnectionString(dbUrl: string): string {
  // Hide password if present (postgresql://user:pass@host/db).
  try {
    const u = new URL(dbUrl);
    if (u.password) u.password = "*****";
    return u.toString();
  } catch {
    return dbUrl.replace(/\/\/([^:/?#]+):([^@/?#]+)@/g, "//$1:*****@");
  }
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
      const effectiveSslMode = explicitSsl || urlSslMode;
      // SSL priority: PGSSLMODE env > URL param > auto (sslmode=prefer behavior)
      const sslConfig = effectiveSslMode
        ? sslModeToConfig(effectiveSslMode)
        : { rejectUnauthorized: false }; // Default: try SSL (with fallback)
      // Enable fallback for: no explicit mode OR explicit "prefer"/"allow"
      const shouldFallback = !effectiveSslMode || 
        effectiveSslMode.toLowerCase() === "prefer" || 
        effectiveSslMode.toLowerCase() === "allow";
      // Strip sslmode from URI so pg uses our ssl config object instead
      const cleanUri = stripSslModeFromUri(v);
      return {
        clientConfig: { connectionString: cleanUri, ssl: sslConfig },
        display: maskConnectionString(v),
        sslFallbackEnabled: shouldFallback,
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
    return {
      clientConfig: cfg,
      display: describePgConfig(cfg),
      sslFallbackEnabled: shouldFallback,
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

function generateMonitoringPassword(): string {
  // URL-safe and easy to copy/paste; 24 bytes => 32 base64url chars (no padding).
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
  includeOptionalPermissions: boolean;
  /** Provider type. Affects which steps are included. Defaults to "self-managed". */
  provider?: DbProvider;
}): Promise<InitPlan> {
  // NOTE: kept async for API stability / potential future async template loading.
  const monitoringUser = params.monitoringUser || DEFAULT_MONITORING_USER;
  const database = params.database;
  const provider = params.provider ?? "self-managed";

  const qRole = quoteIdent(monitoringUser);
  const qDb = quoteIdent(database);
  const qPw = quoteLiteral(params.monitoringPassword);
  const qRoleNameLit = quoteLiteral(monitoringUser);

  const steps: InitStep[] = [];

  const vars: Record<string, string> = {
    ROLE_IDENT: qRole,
    DB_IDENT: qDb,
  };

  // Some providers (e.g., Supabase) manage users externally - skip role creation.
  // TODO: Make this more flexible by allowing users to specify which steps to skip via config.
  if (!SKIP_ROLE_CREATION_PROVIDERS.includes(provider)) {
    // Role creation/update is done in one template file.
    // Always use a single DO block to avoid race conditions between "role exists?" checks and CREATE USER.
    // We:
    // - create role if missing (and handle duplicate_object in case another session created it concurrently),
    // - then ALTER ROLE to ensure the password is set to the desired value.
    const roleStmt = `do $$ begin
  if not exists (select 1 from pg_catalog.pg_roles where rolname = ${qRoleNameLit}) then
    begin
      create user ${qRole} with password ${qPw};
    exception when duplicate_object then
      null;
    end;
  end if;
  alter user ${qRole} with password ${qPw};
end $$;`;

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

  // Helper functions (SECURITY DEFINER) for plan analysis and table info
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

  // Apply non-optional steps, each in its own transaction
  for (const step of params.plan.steps.filter((s) => !s.optional)) {
    try {
      await executeStep(step);
      applied.push(step.name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const errAny = e as Record<string, unknown>;
      const wrapped = new Error(`Failed at step "${step.name}": ${msg}`) as Error & Record<string, unknown>;
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
          if (errAny[field] !== undefined) wrapped[field] = errAny[field];
        }
      }
      if (e instanceof Error && e.stack) {
        wrapped.stack = e.stack;
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

  const steps: InitStep[] = [];

  const vars: Record<string, string> = {
    ROLE_IDENT: qRole,
    DB_IDENT: qDb,
    ROLE_LITERAL: qRoleLiteral,
  };

  // Step 1: Drop helper functions
  steps.push({
    name: "01.drop_helpers",
    sql: applyTemplate(loadSqlTemplate("uninit/01.helpers.sql"), vars),
  });

  // Step 2: Drop view, revoke permissions, drop schema
  steps.push({
    name: "02.revoke_permissions",
    sql: applyTemplate(loadSqlTemplate("uninit/02.permissions.sql"), vars),
  });

  // Step 3: Drop the role (only if requested and provider allows it)
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
}): Promise<{ applied: string[]; errors: string[] }> {
  const applied: string[] = [];
  const errors: string[] = [];

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

  // Apply steps in order - unlike init, uninit steps are not optional
  // but we continue on errors to clean up as much as possible
  for (const step of params.plan.steps) {
    try {
      await executeStep(step);
      applied.push(step.name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      errors.push(`${step.name}: ${msg}`);
      // Continue to try other steps
    }
  }

  return { applied, errors };
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

    const viewExistsRes = await params.client.query("select to_regclass('postgres_ai.pg_statistic') is not null as ok");
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
      const rolcfgRes = await params.client.query("select rolconfig from pg_catalog.pg_roles where rolname = $1", [role]);
      const rolconfig = rolcfgRes.rows?.[0]?.rolconfig;
      const spLine = Array.isArray(rolconfig) ? rolconfig.find((v: unknown) => String(v).startsWith("search_path=")) : undefined;
      if (typeof spLine !== "string" || !spLine) {
        missingRequired.push("role search_path is set");
      } else {
        // We accept any ordering as long as postgres_ai, public, and pg_catalog are included.
        // Also verify search_path includes the pg_stat_statements schema if in a non-standard location.
        const sp = spLine.toLowerCase();
        if (!sp.includes("postgres_ai") || !sp.includes("public") || !sp.includes("pg_catalog")) {
          missingRequired.push("role search_path includes postgres_ai, public and pg_catalog");
        }
        // If pg_stat_statements is in a non-standard schema (e.g., 'extensions' on Supabase), verify it's in search_path
        if (extSchema && extSchema !== "pg_catalog" && extSchema !== "public") {
          if (!sp.includes(extSchema.toLowerCase())) {
            missingRequired.push(`role search_path includes ${extSchema} (pg_stat_statements location)`);
          }
        }
      }
    }

    // Check for helper functions
    const explainFnRes = await params.client.query(
      "select has_function_privilege($1, 'postgres_ai.explain_generic(text, text, text)', 'EXECUTE') as ok",
      [role]
    );
    if (!explainFnRes.rows?.[0]?.ok) {
      missingRequired.push("EXECUTE on postgres_ai.explain_generic(text, text, text)");
    }

    const tableDescribeFnRes = await params.client.query(
      "select has_function_privilege($1, 'postgres_ai.table_describe(text)', 'EXECUTE') as ok",
      [role]
    );
    if (!tableDescribeFnRes.rows?.[0]?.ok) {
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


