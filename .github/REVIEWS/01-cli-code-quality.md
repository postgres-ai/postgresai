# CLI Code Quality Review

**Date:** 2026-03-12
**Scope:** `cli/lib/` and `cli/bin/` (all `.ts` files), `cli/sql/` (all `.sql` files)
**Reviewer:** Automated code review

---

## 1. Code Organization

**Rating: Good**

Module boundaries are clean. The dependency graph is acyclic:

```
bin/postgres-ai.ts (entry point)
  -> lib/checkup.ts -> lib/metrics-loader.ts -> lib/metrics-embedded.ts
                    -> lib/checkup-dictionary.ts -> lib/checkup-dictionary-embedded.ts
  -> lib/checkup-api.ts -> lib/util.ts
  -> lib/checkup-summary.ts (standalone)
  -> lib/issues.ts -> lib/util.ts
  -> lib/reports.ts -> lib/util.ts
  -> lib/storage.ts -> lib/util.ts
  -> lib/config.ts (standalone)
  -> lib/init.ts (standalone)
  -> lib/supabase.ts (standalone)
  -> lib/mcp-server.ts -> lib/issues.ts, lib/reports.ts, lib/config.ts, lib/util.ts
  -> lib/pkce.ts (standalone)
  -> lib/auth-server.ts (standalone)
```

No circular dependency risk. `util.ts` is a proper leaf module used by multiple API clients. The split between `checkup-api.ts` (backend RPC) and `checkup.ts` (local SQL generation) is clear.

**Observation (low):** `bin/postgres-ai.ts` is very large (~52K tokens). Consider extracting command handlers into separate files (e.g., `lib/commands/checkup.ts`, `lib/commands/issues.ts`) to improve maintainability.

---

## 2. Error Handling

**Rating: Good with minor inconsistencies**

`checkup.ts` documents two error patterns (lines 26-38):
- **Propagating:** Core data functions throw errors (e.g., `getPostgresVersion`, `getSettings`, `getInvalidIndexes`)
- **Graceful degradation:** Optional queries catch errors and include them in output (e.g., `getStatsReset` postmaster query, `generateD004` extension queries)

### Findings

| Severity | File | Line | Description |
|----------|------|------|-------------|
| **Low** | `checkup.ts` | 1325-1331 | `generateF004` catches bloat estimation errors (graceful degradation) but still calls `getCurrentDatabaseInfo` outside the try/catch. If bloat estimation fails AND `getCurrentDatabaseInfo` fails, the second error masks the first. Same pattern in `generateF005` (line 1443-1449). Acceptable since `getCurrentDatabaseInfo` is a core data function that should propagate. |
| **Low** | `init.ts` | 608-638 | Error wrapping in `applyInitPlan` uses `const wrapped: any = new Error(...)` -- loses type safety. The Supabase variant (`supabase.ts` line 530) correctly types it as `PgCompatibleError`. |
| **Low** | `reports.ts`, `issues.ts`, `storage.ts` | Various | API client functions (`fetchIssues`, `fetchReports`, etc.) all follow a consistent pattern: throw on HTTP error, throw on JSON parse failure. No retry logic, unlike `checkup-api.ts`. This is acceptable for CLI commands (user can retry manually) but noted for consistency. |

---

## 3. TypeScript Strictness

**Rating: Moderate -- `any` usage is pervasive in checkup-summary.ts**

`tsconfig.json` has `"strict": true`, which is good. However, `any` is used in several places.

### `any` Usage Inventory

| Severity | File | Lines | Count | Assessment |
|----------|------|-------|-------|------------|
| **Medium** | `checkup-summary.ts` | 19, 50, 64, 78, 100, 107, 122, 129, 144, 151, 166, 190, 204, 218, 232, 246, 260 | 17 | All summarize functions take `nodeData: any`. The file header (line 6-8) acknowledges this as technical debt. A `NodeData` type interface would improve safety. |
| **Medium** | `checkup-api.ts` | 123, 125, 148-150, 164, 172, 175, 256 | ~10 | `RpcError.payloadJson: any`, `unwrapRpcResponse` returns `any`, `formatRpcErrorForDisplay` casts to `any`. These are inherent to handling dynamic API responses. |
| **Low** | `init.ts` | 32, 96, 163, 590, 610, 729, 860 | 7 | Most are `as any` casts for error code inspection or `KNOWN_PROVIDERS.includes(provider as any)`. |
| **Low** | `mcp-server.ts` | 540 | 1 | `server.setRequestHandler(CallToolRequestSchema, async (req: any): Promise<any>)` -- SDK typing limitation. |
| **Low** | `checkup.ts` | 250 (NodeResult) | 1 | `data: Record<string, any>` in `NodeResult` interface. |

### Unsafe Type Cast

| Severity | File | Line | Description |
|----------|------|------|-------------|
| **Low** | `init.ts` | 611 | `const wrapped: any = new Error(...)` should be typed as `Error & Record<string, unknown>` or similar. |

---

## 4. SQL Injection

**Rating: Good**

### Direct PostgreSQL Queries (via `pg` client)

All queries in `checkup.ts` use either:
- **Static SQL** (inline queries with no user input): `getPostgresVersion`, `getDatabaseSizes`, `getClusterInfo`, `generateD004`, `generateG003`, etc.
- **Metric SQL from embedded data** (loaded from build-time embedded metrics): `getSettings`, `getInvalidIndexes`, `getUnusedIndexes`, `getRedundantIndexes`, etc.

No user-supplied values are interpolated into SQL strings in `checkup.ts`. All queries are safe.

### `init.ts` -- SQL Template System

User-supplied identifiers (`monitoringUser`, `database`) are escaped via:
- `quoteIdent()` (line 224-230): double-quotes and escapes embedded quotes, rejects null bytes
- `quoteLiteral()` (line 232-239): single-quotes and escapes embedded quotes, rejects null bytes

These are used to fill `{{ROLE_IDENT}}`, `{{DB_IDENT}}`, `{{ROLE_STMT}}` templates. The approach is sound.

### `verifyInitSetup()` (init.ts)

Uses parameterized queries throughout (`$1`, `$2` placeholders). Safe.

### `verifyInitSetupViaSupabase()` (supabase.ts)

The Supabase Management API does not support parameterized queries (SQL is sent as a string). The module handles this with:
- `isValidIdentifier()` (line 825-827): validates role names match `^[a-zA-Z_][a-zA-Z0-9_]{0,62}$`
- `escapeLiteral()` (line 834-841): escapes single quotes, rejects null bytes
- Validation is applied at line 613 before any queries

**This is the correct approach** given the API limitation. The identifier validation is strict enough to prevent injection.

### `explain_generic` SQL Function (06.helpers.sql)

Concatenates user input into EXPLAIN statement (line 96): `execute 'explain ... ' || v_clean_query`. This is inherently necessary (EXPLAIN cannot use parameterized input). Mitigations:
1. EXPLAIN is read-only (plans but doesn't execute)
2. Semicolons are rejected (line 67-74), preventing statement stacking
3. The function runs as SECURITY DEFINER with a restricted `search_path`

**Acceptable risk.** The semicolon check has a documented limitation (rejects valid queries containing semicolons in string literals), which is noted in comments.

### SQL Template Files (`cli/sql/`)

| File | Findings |
|------|----------|
| `03.permissions.sql:49` | `execute format('grant usage on schema %I to {{ROLE_IDENT}}', ext_schema)` -- `ROLE_IDENT` is pre-quoted via `quoteIdent()`, safe. |
| `03.permissions.sql:73` | `execute format('alter user {{ROLE_IDENT}} set search_path = %s', sp)` -- `sp` is built from `quote_ident()` (PL/pgSQL), `ROLE_IDENT` pre-quoted, safe. |
| `uninit/03.role.sql:9,16,23` | `execute format('... %I', {{ROLE_LITERAL}})` -- `ROLE_LITERAL` is a string literal from `quoteLiteral()`, used as `%I` argument to `format()`, safe. |

---

## 5. API Clients

### `checkup-api.ts`

**Rating: Good**

- **Retry logic:** `withRetry()` (line 76-109) with exponential backoff, configurable max attempts, retries on 5xx and network errors only.
- **Timeout:** 30-second connection timeout (line 179), cleared when response headers arrive (line 242-243). Uses `AbortController` for clean cancellation.
- **Error surfaces:** `RpcError` class with status code, payload text, and parsed JSON. `formatRpcErrorForDisplay()` for user-friendly output.
- **Defense-in-depth:** API key sent in both header and body (line 199-202).

### `issues.ts`, `reports.ts`, `storage.ts`

**Rating: Acceptable with caveats**

| Severity | Finding |
|----------|---------|
| **Medium** | No timeout on `fetch()` calls. If the server hangs after accepting the connection, the CLI will hang indefinitely. The `checkup-api.ts` module has proper timeouts, but `issues.ts`, `reports.ts`, and `storage.ts` do not. |
| **Medium** | No retry logic. Unlike `checkup-api.ts` which has `withRetry()`, these modules throw immediately on any failure. For a CLI tool, this is acceptable (user retries manually), but the inconsistency is notable. |
| **Low** | `storage.ts` reads entire files into memory (line 102: `fs.readFileSync`). The 500MB size check (line 87) prevents OOM, but streaming would be more efficient for large files. The code comments explain why streaming isn't used (FormData API limitation). |

### `supabase.ts`

**Rating: Good**

- Clean error mapping from Supabase API errors to PostgreSQL-compatible errors
- Project ref validation prevents SSRF (line 72-75, 91-93)
- URL encoding for path parameters (line 107, 357)

---

## 6. Dependencies

**Rating: Good**

`package.json` dependencies:

| Dependency | Version | Assessment |
|------------|---------|------------|
| `@modelcontextprotocol/sdk` | ^1.20.2 | Required for MCP server. |
| `commander` | ^12.1.0 | Standard CLI framework. |
| `js-yaml` | ^4.1.0 | Used for YAML parsing (instances config). |
| `pg` | ^8.16.3 | PostgreSQL client. |

Dev dependencies are appropriate (`@types/*`, `ajv` for schema validation tests, `typescript`).

**No unused dependencies detected.** All four runtime deps are imported in the codebase.

**No concerning dependencies.** All are well-maintained, widely-used packages with no known security issues.

---

## 7. Additional Findings

### Duplicate `formatBytes` implementation

| Severity | File | Line | Description |
|----------|------|------|-------------|
| **Low** | `checkup.ts` | 299-306 | `formatBytes()` with IEC units (KiB, MiB), 2 decimal places |
| **Low** | `checkup-summary.ts` | 92-98 | `formatBytes()` with IEC units (KiB, MiB), 0-1 decimal places |

Two different implementations with slightly different formatting. Consider exporting from one location.

### `__dirname` usage in bundled code

| Severity | File | Line | Description |
|----------|------|------|-------------|
| **Low** | `checkup.ts` | 890, 901 | Uses `__dirname` for `resolveBuildTs()`. The comment in `init.ts` (line 193-194) explicitly warns that `__dirname` is baked in at build time by bundlers and recommends `import.meta.url`. `checkup.ts` uses `__dirname` anyway for the `BUILD_TS` fallback path. This works but may resolve incorrectly if the bundled output is moved. |

### Missing `access_token` in body for some `checkup-api.ts` calls

The `convertCheckupReportJsonToMarkdown` function (line 406-432) sends `access_token` in the body but the comment at line 190-192 notes that for markdown conversion, auth is optional. The `apiKey` is still sent in headers for gateway auth. No issue -- just documenting the intentional asymmetry.

### Config file permissions

| Severity | File | Line | Description |
|----------|------|------|-------------|
| **Info** | `config.ts` | 112, 132 | Config directory created with `mode: 0o700`, config file with `mode: 0o600`. Good security practice for files that may contain API keys. |

---

## Summary

| Category | Rating | Critical | High | Medium | Low |
|----------|--------|----------|------|--------|-----|
| Code Organization | Good | 0 | 0 | 0 | 1 |
| Error Handling | Good | 0 | 0 | 0 | 3 |
| TypeScript Strictness | Moderate | 0 | 0 | 2 | 4 |
| SQL Injection | Good | 0 | 0 | 0 | 0 |
| API Clients | Good | 0 | 0 | 2 | 1 |
| Dependencies | Good | 0 | 0 | 0 | 0 |
| Other | -- | 0 | 0 | 0 | 3 |
| **Total** | | **0** | **0** | **4** | **12** |

### Top Recommendations

1. **Add timeouts to `fetch()` calls** in `issues.ts`, `reports.ts`, `storage.ts` (Medium) -- these can hang indefinitely if the server stops responding mid-connection.
2. **Type the `checkup-summary.ts` functions** with proper interfaces instead of `any` (Medium) -- the file header already acknowledges this as planned work.
3. **Fix `init.ts` line 611** -- type the `wrapped` error variable instead of using `any` (Low).
4. **Deduplicate `formatBytes`** -- export from `checkup.ts` and import in `checkup-summary.ts` (Low).
