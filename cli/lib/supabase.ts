/**
 * Supabase Management API client for database operations.
 *
 * This module provides an alternative to direct PostgreSQL connections by using
 * the Supabase Management API to execute SQL queries.
 *
 * API Reference: https://supabase.com/docs/reference/api/introduction
 * Endpoint: POST /v1/projects/{ref}/database/query
 */

const SUPABASE_API_BASE = "https://api.supabase.com";

export type SupabaseConfig = {
  /** Supabase project reference (e.g., "abc123xyz") */
  projectRef: string;
  /** Supabase Management API access token (Personal Access Token) */
  accessToken: string;
};

/**
 * PostgreSQL-compatible error structure.
 * Mirrors the error fields from node-postgres for consistent error handling.
 */
export type PgCompatibleError = Error & {
  code?: string;
  detail?: string;
  hint?: string;
  position?: string;
  internalPosition?: string;
  internalQuery?: string;
  where?: string;
  schema?: string;
  table?: string;
  column?: string;
  dataType?: string;
  constraint?: string;
  file?: string;
  line?: string;
  routine?: string;
  // Supabase-specific fields (mapped to pg-compatible structure)
  supabaseErrorCode?: string;
  httpStatus?: number;
};

/**
 * Result from Supabase Management API query endpoint.
 */
export type SupabaseQueryResult = {
  rows: Record<string, unknown>[];
  rowCount: number;
};

/**
 * Raw response from Supabase Management API.
 */
type SupabaseApiResponse = {
  // Success case: array of rows
  // Error case: { code, message, ... }
  error?: {
    code?: string;
    message?: string;
    details?: string;
    hint?: string;
  };
  // The API returns the result directly (array) on success
} | Record<string, unknown>[];

/**
 * Validate Supabase project reference format.
 * Project refs are typically 20 lowercase alphanumeric characters.
 */
function isValidProjectRef(ref: string): boolean {
  // Supabase project refs are alphanumeric, typically 20 chars, lowercase
  return /^[a-z0-9]{10,30}$/i.test(ref);
}

/**
 * Supabase Management API client for executing SQL queries.
 */
export class SupabaseClient {
  private config: SupabaseConfig;

  constructor(config: SupabaseConfig) {
    if (!config.projectRef) {
      throw new Error("Supabase project reference is required");
    }
    if (!config.accessToken) {
      throw new Error("Supabase access token is required");
    }
    // Validate project ref format to prevent path traversal
    if (!isValidProjectRef(config.projectRef)) {
      throw new Error(`Invalid Supabase project reference format: "${config.projectRef}". Expected 10-30 alphanumeric characters.`);
    }
    this.config = config;
  }

  /**
   * Execute a SQL query via the Supabase Management API.
   *
   * @param sql The SQL query to execute
   * @param readOnly If true, uses read_only flag in API request (default: false for DDL/DML operations)
   * @returns Query result with rows and rowCount (rowCount is array length for SELECT queries)
   * @throws PgCompatibleError on failure
   */
  async query(sql: string, readOnly = false): Promise<SupabaseQueryResult> {
    // URL-encode projectRef for safety (validated in constructor, but defense in depth)
    const url = `${SUPABASE_API_BASE}/v1/projects/${encodeURIComponent(this.config.projectRef)}/database/query`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.accessToken}`,
      },
      body: JSON.stringify({
        query: sql,
        read_only: readOnly,
      }),
    });

    const body = await response.text();
    let data: SupabaseApiResponse;

    try {
      data = JSON.parse(body);
    } catch {
      // If we can't parse JSON, create an error with the raw body
      throw this.createPgError({
        message: `Supabase API returned non-JSON response: ${body.slice(0, 200)}`,
        httpStatus: response.status,
      });
    }

    // Handle HTTP errors
    if (!response.ok) {
      throw this.parseApiError(data, response.status);
    }

    // Handle explicit error response
    if (data && typeof data === "object" && "error" in data && data.error) {
      throw this.parseApiError(data, response.status);
    }

    // Success: API returns array of rows directly
    const rows = Array.isArray(data) ? data : [];
    return {
      rows: rows as Record<string, unknown>[],
      rowCount: rows.length,
    };
  }

  /**
   * Test connection by executing a simple query.
   */
  async testConnection(): Promise<{ database: string; version: string }> {
    const result = await this.query(
      "SELECT current_database() as db, version() as version",
      true
    );
    const row = result.rows[0] ?? {};
    return {
      database: String(row.db ?? ""),
      version: String(row.version ?? ""),
    };
  }

  /**
   * Get current database name.
   */
  async getCurrentDatabase(): Promise<string> {
    const result = await this.query("SELECT current_database() as db", true);
    const row = result.rows[0] ?? {};
    return String(row.db ?? "");
  }

  /**
   * Parse Supabase API error and convert to PostgreSQL-compatible error.
   */
  private parseApiError(
    data: SupabaseApiResponse,
    httpStatus: number
  ): PgCompatibleError {
    // Handle different error formats from Supabase API
    if (data && typeof data === "object" && !Array.isArray(data)) {
      const errObj = "error" in data && data.error ? data.error : data;

      // Check for PostgreSQL error embedded in the response
      // Supabase forwards PostgreSQL errors with their original structure
      const pgCode = this.extractPgErrorCode(errObj);
      const message = this.extractErrorMessage(errObj);
      const detail = this.extractField(errObj, ["details", "detail"]);
      const hint = this.extractField(errObj, ["hint"]);

      return this.createPgError({
        message,
        code: pgCode,
        detail,
        hint,
        httpStatus,
        supabaseErrorCode:
          typeof errObj === "object" && errObj && "code" in errObj
            ? String((errObj as Record<string, unknown>).code ?? "")
            : undefined,
      });
    }

    return this.createPgError({
      message: `Supabase API error (HTTP ${httpStatus})`,
      httpStatus,
    });
  }

  /**
   * Extract PostgreSQL error code from various error formats.
   * Supabase may return errors as:
   * - { code: "42501", ... } (PostgreSQL error code)
   * - { code: "PGRST...", ... } (PostgREST error code)
   * - { error: { code: "...", ... } }
   */
  private extractPgErrorCode(errObj: unknown): string | undefined {
    if (!errObj || typeof errObj !== "object") return undefined;

    const obj = errObj as Record<string, unknown>;

    // Direct code field
    if (typeof obj.code === "string") {
      const code = obj.code;
      // PostgreSQL error codes are 5 characters (e.g., "42501")
      if (/^\d{5}$/.test(code)) {
        return code;
      }
      // Map common Supabase/PostgREST error codes to PostgreSQL equivalents
      return this.mapSupabaseCodeToPg(code);
    }

    return undefined;
  }

  /**
   * Map Supabase/PostgREST error codes to PostgreSQL equivalents.
   */
  private mapSupabaseCodeToPg(code: string): string | undefined {
    // PostgREST error codes: https://postgrest.org/en/stable/references/errors.html
    const mapping: Record<string, string> = {
      // Authentication/Authorization
      PGRST301: "28000", // invalid_authorization_specification
      PGRST302: "28P01", // invalid_password
      // Permission errors
      "42501": "42501", // insufficient_privilege (pass through)
      PGRST000: "42501", // permission denied (generic)
      // Syntax errors
      "42601": "42601", // syntax_error (pass through)
      // Object errors
      "42P01": "42P01", // undefined_table (pass through)
      PGRST200: "42P01", // table not found
      "42883": "42883", // undefined_function (pass through)
      // Connection errors
      "08000": "08000", // connection_exception (pass through)
      "08003": "08003", // connection_does_not_exist (pass through)
      "08006": "08006", // connection_failure (pass through)
      // Duplicate object
      "42710": "42710", // duplicate_object (pass through)
    };

    return mapping[code];
  }

  /**
   * Extract error message from various error formats.
   */
  private extractErrorMessage(errObj: unknown): string {
    if (!errObj || typeof errObj !== "object") {
      return "Unknown Supabase API error";
    }

    const obj = errObj as Record<string, unknown>;

    // Try common message fields
    for (const field of ["message", "error", "msg", "description"]) {
      if (typeof obj[field] === "string" && obj[field]) {
        return obj[field] as string;
      }
    }

    // If error is nested, try to extract from it
    if (obj.error && typeof obj.error === "object") {
      return this.extractErrorMessage(obj.error);
    }

    return "Unknown Supabase API error";
  }

  /**
   * Extract a field from error object, trying multiple possible field names.
   */
  private extractField(
    errObj: unknown,
    fieldNames: string[]
  ): string | undefined {
    if (!errObj || typeof errObj !== "object") return undefined;

    const obj = errObj as Record<string, unknown>;

    for (const field of fieldNames) {
      if (typeof obj[field] === "string" && obj[field]) {
        return obj[field] as string;
      }
    }

    return undefined;
  }

  /**
   * Create a PostgreSQL-compatible error object.
   */
  private createPgError(opts: {
    message: string;
    code?: string;
    detail?: string;
    hint?: string;
    httpStatus?: number;
    supabaseErrorCode?: string;
  }): PgCompatibleError {
    const err = new Error(opts.message) as PgCompatibleError;

    if (opts.code) err.code = opts.code;
    if (opts.detail) err.detail = opts.detail;
    if (opts.hint) err.hint = opts.hint;
    if (opts.httpStatus) err.httpStatus = opts.httpStatus;
    if (opts.supabaseErrorCode) err.supabaseErrorCode = opts.supabaseErrorCode;

    return err;
  }
}

/**
 * Fetch the database pooler connection string from Supabase Management API.
 * Returns a postgresql:// URL with the specified username but no password.
 *
 * Note: The username will be automatically suffixed with `.<projectRef>` if not
 * already present, as required by Supabase pooler connections.
 *
 * @param config Supabase configuration with projectRef and accessToken
 * @param username Username to include in the URL (e.g., monitoring user).
 *                 Will be transformed to `<username>.<projectRef>` format.
 * @returns Database URL without password (e.g., "postgresql://user.project@host:port/postgres"),
 *          or null if the API call fails or returns no pooler config.
 */
export async function fetchPoolerDatabaseUrl(
  config: SupabaseConfig,
  username: string
): Promise<string | null> {
  // Validate projectRef format to prevent SSRF via crafted project references
  if (!isValidProjectRef(config.projectRef)) {
    throw new Error(`Invalid Supabase project reference format: "${config.projectRef}". Expected 10-30 alphanumeric characters.`);
  }
  const url = `${SUPABASE_API_BASE}/v1/projects/${encodeURIComponent(config.projectRef)}/config/database/pooler`;

  // For Supabase pooler connections, the username must include the project ref:
  //   <user>.<project_ref>
  // Example:
  //   postgresql://postgres_ai_mon.xhaqmsvczjkkvkgdyast@aws-1-eu-west-1.pooler.supabase.com:6543/postgres
  const suffix = `.${config.projectRef}`;
  const effectiveUsername = username.endsWith(suffix) ? username : `${username}${suffix}`;
  // URL-encode the username to handle special characters safely
  const encodedUsername = encodeURIComponent(effectiveUsername);
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${config.accessToken}`,
      },
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();

    // The API returns an array of pooler configurations
    // Look for a connection string in the response
    if (Array.isArray(data) && data.length > 0) {
      const pooler = data[0];
      // Build URL from components if available
      if (pooler.db_host && pooler.db_port && pooler.db_name) {
        return `postgresql://${encodedUsername}@${pooler.db_host}:${pooler.db_port}/${pooler.db_name}`;
      }
      // Fallback: try to extract from connection_string if present
      if (typeof pooler.connection_string === "string") {
        try {
          const connUrl = new URL(pooler.connection_string);
          // Use provided username; handle empty port for default ports (e.g., 5432)
          const portPart = connUrl.port ? `:${connUrl.port}` : "";
          return `postgresql://${encodedUsername}@${connUrl.hostname}${portPart}${connUrl.pathname}`;
        } catch {
          return null;
        }
      }
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Resolve Supabase configuration from options and environment variables.
 */
export function resolveSupabaseConfig(opts: {
  accessToken?: string;
  projectRef?: string;
}): SupabaseConfig {
  const accessToken =
    opts.accessToken?.trim() ||
    process.env.SUPABASE_ACCESS_TOKEN?.trim() ||
    "";

  const projectRef =
    opts.projectRef?.trim() || process.env.SUPABASE_PROJECT_REF?.trim() || "";

  if (!accessToken) {
    throw new Error(
      "Supabase access token is required.\n" +
        "Provide it via --supabase-access-token or SUPABASE_ACCESS_TOKEN environment variable.\n" +
        "Generate a token at: https://supabase.com/dashboard/account/tokens"
    );
  }

  if (!projectRef) {
    throw new Error(
      "Supabase project reference is required.\n" +
        "Provide it via --supabase-project-ref or SUPABASE_PROJECT_REF environment variable.\n" +
        "Find your project ref in the Supabase dashboard URL: https://supabase.com/dashboard/project/<ref>"
    );
  }

  return { accessToken, projectRef };
}

/**
 * Extract project reference from a Supabase database URL.
 * Supabase database URLs typically look like:
 *   - Direct: postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
 *   - Pooler (modern): postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
 *   - Pooler (legacy): postgresql://postgres:[PASSWORD]@[PROJECT_REF].pooler.supabase.com:6543/postgres
 *
 * @param dbUrl PostgreSQL connection URL
 * @returns Project reference if found, undefined otherwise
 */
export function extractProjectRefFromUrl(dbUrl: string): string | undefined {
  try {
    const url = new URL(dbUrl);
    const host = url.hostname;

    // Match db.<ref>.supabase.co or <ref>.supabase.co patterns (direct connection)
    const match = host.match(/^(?:db\.)?([^.]+)\.supabase\.co$/i);
    if (match && match[1]) {
      return match[1];
    }

    // Modern pooler URLs: project ref is in the username as postgres.<ref>
    // Example: postgresql://postgres.abcdefghij:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres
    if (host.includes("pooler.supabase.com")) {
      const username = url.username;
      const userMatch = username.match(/^postgres\.([a-z0-9]+)$/i);
      if (userMatch && userMatch[1]) {
        return userMatch[1];
      }
    }

    // Legacy pooler URLs: <project-ref>.pooler.supabase.com (fallback)
    const poolerMatch = host.match(/^([a-z0-9]+)\.pooler\.supabase\.com$/i);
    if (poolerMatch && poolerMatch[1] && !poolerMatch[1].startsWith("aws-")) {
      return poolerMatch[1];
    }

    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Apply init plan steps via Supabase Management API.
 * Mirrors the behavior of applyInitPlan() in init.ts but uses Supabase API.
 */
export async function applyInitPlanViaSupabase(params: {
  client: SupabaseClient;
  plan: {
    monitoringUser: string;
    database: string;
    steps: Array<{
      name: string;
      sql: string;
      params?: unknown[];
      optional?: boolean;
    }>;
  };
  verbose?: boolean;
}): Promise<{ applied: string[]; skippedOptional: string[] }> {
  const applied: string[] = [];
  const skippedOptional: string[] = [];

  // Helper to execute a step (each step is wrapped in BEGIN/COMMIT)
  const executeStep = async (step: {
    name: string;
    sql: string;
    optional?: boolean;
  }): Promise<void> => {
    // Wrap in explicit transaction for atomic execution.
    // Note: Supabase API uses pooled connections, so if the transaction fails,
    // PostgreSQL automatically rolls it back - no separate ROLLBACK needed.
    const wrappedSql = `BEGIN;\n${step.sql}\nCOMMIT;`;
    await params.client.query(wrappedSql, false);
  };

  // Apply non-optional steps first
  for (const step of params.plan.steps.filter((s) => !s.optional)) {
    try {
      if (params.verbose) {
        console.log(`Executing step: ${step.name}`);
      }
      await executeStep(step);
      applied.push(step.name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const errAny = e as PgCompatibleError;
      const wrapped: PgCompatibleError = new Error(
        `Failed at step "${step.name}": ${msg}`
      ) as PgCompatibleError;

      // Preserve PostgreSQL error fields for consistent error handling
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
        "httpStatus",
        "supabaseErrorCode",
      ] as const;

      for (const field of pgErrorFields) {
        if (errAny[field] !== undefined) {
          (wrapped as unknown as Record<string, unknown>)[field] = errAny[field];
        }
      }

      if (e instanceof Error && e.stack) {
        wrapped.stack = e.stack;
      }

      throw wrapped;
    }
  }

  // Apply optional steps (failures don't abort)
  for (const step of params.plan.steps.filter((s) => s.optional)) {
    try {
      if (params.verbose) {
        console.log(`Executing optional step: ${step.name}`);
      }
      await executeStep(step);
      applied.push(step.name);
    } catch {
      skippedOptional.push(step.name);
      // best-effort: ignore errors for optional steps
    }
  }

  return { applied, skippedOptional };
}

/**
 * Verify init setup via Supabase Management API.
 * Mirrors the behavior of verifyInitSetup() in init.ts but uses Supabase API.
 *
 * @param params.client - Supabase client for API calls
 * @param params.database - Database name to verify
 * @param params.monitoringUser - Role name to check permissions for
 * @param params.includeOptionalPermissions - Whether to check optional permissions
 * @returns Object with ok status and arrays of missing required/optional items
 */
export async function verifyInitSetupViaSupabase(params: {
  client: SupabaseClient;
  database: string;
  monitoringUser: string;
  includeOptionalPermissions: boolean;
}): Promise<{
  ok: boolean;
  missingRequired: string[];
  missingOptional: string[];
}> {
  const missingRequired: string[] = [];
  const missingOptional: string[] = [];

  const role = params.monitoringUser;
  const db = params.database;

  // Validate role name to prevent SQL injection
  if (!isValidIdentifier(role)) {
    throw new Error(`Invalid monitoring user name: "${role}". Must be a valid PostgreSQL identifier (letters, digits, underscores, max 63 chars, starting with letter or underscore).`);
  }

  // Check if role exists
  const roleRes = await params.client.query(
    `SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '${escapeLiteral(role)}'`,
    true
  );
  const roleExists = roleRes.rowCount > 0;

  if (!roleExists) {
    missingRequired.push(`role "${role}" does not exist`);
    return { ok: false, missingRequired, missingOptional };
  }

  // Check CONNECT privilege
  const connectRes = await params.client.query(
    `SELECT has_database_privilege('${escapeLiteral(role)}', '${escapeLiteral(db)}', 'CONNECT') as ok`,
    true
  );
  if (!connectRes.rows?.[0]?.ok) {
    missingRequired.push(`CONNECT on database "${db}"`);
  }

  // Check pg_monitor membership
  const pgMonitorRes = await params.client.query(
    `SELECT pg_has_role('${escapeLiteral(role)}', 'pg_monitor', 'member') as ok`,
    true
  );
  if (!pgMonitorRes.rows?.[0]?.ok) {
    missingRequired.push("membership in role pg_monitor");
  }

  // Check SELECT on pg_index
  const pgIndexRes = await params.client.query(
    `SELECT has_table_privilege('${escapeLiteral(role)}', 'pg_catalog.pg_index', 'SELECT') as ok`,
    true
  );
  if (!pgIndexRes.rows?.[0]?.ok) {
    missingRequired.push("SELECT on pg_catalog.pg_index");
  }

  // Check postgres_ai schema exists and has USAGE privilege
  // First check if schema exists to avoid has_schema_privilege throwing error
  const schemaExistsRes = await params.client.query(
    "SELECT nspname FROM pg_namespace WHERE nspname = 'postgres_ai'",
    true
  );
  if (schemaExistsRes.rowCount === 0) {
    missingRequired.push("schema postgres_ai exists");
  } else {
    const schemaPrivRes = await params.client.query(
      `SELECT has_schema_privilege('${escapeLiteral(role)}', 'postgres_ai', 'USAGE') as ok`,
      true
    );
    if (!schemaPrivRes.rows?.[0]?.ok) {
      missingRequired.push("USAGE on schema postgres_ai");
    }
  }

  // Check pg_statistic view
  const viewExistsRes = await params.client.query(
    "SELECT to_regclass('postgres_ai.pg_statistic') IS NOT NULL as ok",
    true
  );
  if (!viewExistsRes.rows?.[0]?.ok) {
    missingRequired.push("view postgres_ai.pg_statistic exists");
  } else {
    const viewPrivRes = await params.client.query(
      `SELECT has_table_privilege('${escapeLiteral(role)}', 'postgres_ai.pg_statistic', 'SELECT') as ok`,
      true
    );
    if (!viewPrivRes.rows?.[0]?.ok) {
      missingRequired.push("SELECT on view postgres_ai.pg_statistic");
    }
  }

  // Check USAGE on public schema (check existence first to avoid has_schema_privilege throwing)
  const publicSchemaExistsRes = await params.client.query(
    "SELECT nspname FROM pg_namespace WHERE nspname = 'public'",
    true
  );
  if (publicSchemaExistsRes.rowCount === 0) {
    missingRequired.push("schema public exists");
  } else {
    const schemaUsageRes = await params.client.query(
      `SELECT has_schema_privilege('${escapeLiteral(role)}', 'public', 'USAGE') as ok`,
      true
    );
    if (!schemaUsageRes.rows?.[0]?.ok) {
      missingRequired.push("USAGE on schema public");
    }
  }

  // Check search_path
  const rolcfgRes = await params.client.query(
    `SELECT rolconfig FROM pg_catalog.pg_roles WHERE rolname = '${escapeLiteral(role)}'`,
    true
  );
  const rolconfig = rolcfgRes.rows?.[0]?.rolconfig as string[] | null;
  const spLine = Array.isArray(rolconfig)
    ? rolconfig.find((v: string) => String(v).startsWith("search_path="))
    : undefined;
  if (typeof spLine !== "string" || !spLine) {
    missingRequired.push("role search_path is set");
  } else {
    const sp = spLine.toLowerCase();
    if (
      !sp.includes("postgres_ai") ||
      !sp.includes("public") ||
      !sp.includes("pg_catalog")
    ) {
      missingRequired.push(
        "role search_path includes postgres_ai, public and pg_catalog"
      );
    }
  }

  // Check helper functions - first verify they exist to avoid has_function_privilege errors
  const explainFnExistsRes = await params.client.query(
    "SELECT oid FROM pg_proc WHERE proname = 'explain_generic' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'postgres_ai')",
    true
  );
  if (explainFnExistsRes.rowCount === 0) {
    missingRequired.push("function postgres_ai.explain_generic exists");
  } else {
    const explainFnRes = await params.client.query(
      `SELECT has_function_privilege('${escapeLiteral(role)}', 'postgres_ai.explain_generic(text, text, text)', 'EXECUTE') as ok`,
      true
    );
    if (!explainFnRes.rows?.[0]?.ok) {
      missingRequired.push(
        "EXECUTE on postgres_ai.explain_generic(text, text, text)"
      );
    }
  }

  const tableDescribeFnExistsRes = await params.client.query(
    "SELECT oid FROM pg_proc WHERE proname = 'table_describe' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'postgres_ai')",
    true
  );
  if (tableDescribeFnExistsRes.rowCount === 0) {
    missingRequired.push("function postgres_ai.table_describe exists");
  } else {
    const tableDescribeFnRes = await params.client.query(
      `SELECT has_function_privilege('${escapeLiteral(role)}', 'postgres_ai.table_describe(text)', 'EXECUTE') as ok`,
      true
    );
    if (!tableDescribeFnRes.rows?.[0]?.ok) {
      missingRequired.push("EXECUTE on postgres_ai.table_describe(text)");
    }
  }

  // Optional permissions
  if (params.includeOptionalPermissions) {
    // RDS tools extension
    const extRes = await params.client.query(
      "SELECT 1 FROM pg_extension WHERE extname = 'rds_tools'",
      true
    );
    if (extRes.rowCount === 0) {
      missingOptional.push("extension rds_tools");
    } else {
      try {
        const fnRes = await params.client.query(
          `SELECT has_function_privilege('${escapeLiteral(role)}', 'rds_tools.pg_ls_multixactdir()', 'EXECUTE') as ok`,
          true
        );
        if (!fnRes.rows?.[0]?.ok) {
          missingOptional.push("EXECUTE on rds_tools.pg_ls_multixactdir()");
        }
      } catch {
        missingOptional.push("EXECUTE on rds_tools.pg_ls_multixactdir()");
      }
    }

    // Self-managed extras (these are hardcoded constants, safe to use directly)
    const optionalFns = [
      "pg_catalog.pg_stat_file(text)",
      "pg_catalog.pg_stat_file(text, boolean)",
      "pg_catalog.pg_ls_dir(text)",
      "pg_catalog.pg_ls_dir(text, boolean, boolean)",
    ];
    for (const fn of optionalFns) {
      try {
        const fnRes = await params.client.query(
          `SELECT has_function_privilege('${escapeLiteral(role)}', '${fn}', 'EXECUTE') as ok`,
          true
        );
        if (!fnRes.rows?.[0]?.ok) {
          missingOptional.push(`EXECUTE on ${fn}`);
        }
      } catch {
        // Function may not exist on this PostgreSQL version
        missingOptional.push(`EXECUTE on ${fn}`);
      }
    }
  }

  return {
    ok: missingRequired.length === 0,
    missingRequired,
    missingOptional,
  };
}

/**
 * Validate that a string is a valid PostgreSQL identifier.
 * PostgreSQL identifiers can contain letters, digits, and underscores,
 * must start with a letter or underscore, and are max 63 characters.
 */
function isValidIdentifier(name: string): boolean {
  return /^[a-zA-Z_][a-zA-Z0-9_]{0,62}$/.test(name);
}

/**
 * Escape a string literal for use in SQL.
 * Handles null bytes and single quotes for safe SQL interpolation.
 * Note: This is for dynamic query building where parameterized queries aren't possible.
 */
function escapeLiteral(value: string): string {
  // Reject null bytes which can cause string truncation
  if (value.includes("\0")) {
    throw new Error("SQL literal cannot contain null bytes");
  }
  // Escape single quotes by doubling them
  return value.replace(/'/g, "''");
}
