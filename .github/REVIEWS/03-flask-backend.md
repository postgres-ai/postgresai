# Flask Backend Code Review: `monitoring_flask_backend/`

**Date:** 2026-03-12
**Scope:** All files in `monitoring_flask_backend/` (app.py, test_app.py, Dockerfile, requirements.txt)

---

## Summary

The Flask backend serves as a Prometheus/PostgreSQL bridge for monitoring dashboards. It provides query text lookups, CSV metric exports, and Prometheus exposition format endpoints. The code is functional but has several security, architecture, and robustness issues.

---

## Findings

### CRITICAL -- Security

#### S1. PromQL Injection in All Metric Endpoints (FIXED)
- **File:** `monitoring_flask_backend/app.py`, lines 390-394, 704-714, 842-851
- **Severity:** Critical
- **Description:** User-supplied query parameters (`cluster_name`, `node_name`, `db_name`, `schemaname`, `tblname`, `idxname`) were interpolated directly into PromQL label selectors without sanitization. An attacker could inject `"}` to break out of the label matcher and execute arbitrary PromQL, potentially exfiltrating metric data from other tenants or causing denial of service.
- **Example:** `GET /btree_bloat/csv?cluster_name=foo"} or vector(1) or pgwatch{x="` would inject arbitrary PromQL.
- **Fix applied:** Added `_sanitize_promql_label()` that escapes backslashes and double-quotes. Applied to all 14 filter interpolation sites across three endpoints (`/pgss_metrics/csv`, `/btree_bloat/csv`, `/table_info/csv`).

#### S2. Error Messages Leak Internal Details
- **File:** `monitoring_flask_backend/app.py`, lines 341, 502, 650, 797, 1009, 1240, 1393
- **Severity:** Medium
- **Description:** Exception messages are returned directly to the client via `jsonify({"error": str(e)})`. This can expose database connection strings, file paths, and stack traces. The `/query_info_metrics` endpoint is worst -- line 1393 returns `f"# Error: {str(e)}\n"`.
- **Recommendation:** Return generic error messages to clients; log the real exception server-side (which is already done).

#### S3. Debug Endpoint Exposed in Production
- **File:** `monitoring_flask_backend/app.py`, lines 652-684
- **Severity:** Medium
- **Description:** `/debug/metrics` is unconditionally registered. It exposes all available Prometheus metrics and sample label structures. Should be gated behind `FLASK_DEBUG` or removed for production.

### HIGH -- Architecture

#### A1. No App Factory Pattern
- **File:** `monitoring_flask_backend/app.py`, line 195
- **Severity:** Medium
- **Description:** The app is created at module level (`app = Flask(__name__)`). This prevents proper test isolation (tests mutate the same singleton) and makes it impossible to run multiple configurations. Flask's recommended pattern is `create_app()`.
- **Impact:** Test isolation, configuration management, and future extension are all hampered.

#### A2. No Blueprints -- All Routes in One File
- **File:** `monitoring_flask_backend/app.py` (1398 lines)
- **Severity:** Low
- **Description:** All endpoints are defined in a single file with no blueprint organization. As the app grows, this becomes difficult to maintain. Logical groupings would be: health/version, pgss metrics, btree bloat, table info, query text.

#### A3. Massive Code Duplication in SQL Queries
- **File:** `monitoring_flask_backend/app.py`, lines 258-281, 1184-1207, 1295-1318
- **Severity:** Medium
- **Description:** The same `SELECT DISTINCT ON (data->>'queryid') ...` query is copy-pasted three times (in `get_query_texts_from_sink()`, `get_query_texts()`, and `get_query_info_metrics()`). Each copy has the same db_name filter logic. This should be refactored into a single shared function.

#### A4. No Connection Pooling for PostgreSQL
- **File:** `monitoring_flask_backend/app.py`, lines 253, 1179, 1290
- **Severity:** Medium
- **Description:** Every request to a query-text endpoint calls `psycopg2.connect()` and then `conn.close()`. With 4 gunicorn workers and frequent scraping (e.g., `/query_info_metrics` every 15s), this creates significant connection churn. Should use `psycopg2.pool` or SQLAlchemy's pool.

### MEDIUM -- Query Truncation (`smart_truncate_query`)

#### T1. CTE Parenthesis Tracking Can Be Fooled by String Literals
- **File:** `monitoring_flask_backend/app.py`, lines 75-86
- **Severity:** Low
- **Description:** The parenthesis-depth tracker counts `(` and `)` characters without considering SQL string literals. A CTE containing `'value with ) paren'` would throw off the depth count. This is acceptable for a display-name heuristic but worth documenting.

#### T2. FROM Clause Extraction Misses Subqueries
- **File:** `monitoring_flask_backend/app.py`, lines 93-120
- **Severity:** Low
- **Description:** Subqueries in the FROM clause like `FROM (SELECT ...) AS sub` are filtered out (line 119), but complex nested subqueries could still produce incorrect table names. Again, acceptable for a heuristic.

#### T3. Truncation of Non-SELECT/INSERT/UPDATE/DELETE Falls Through
- **File:** `monitoring_flask_backend/app.py`, lines 143-168
- **Severity:** Low
- **Description:** If a query starts with INSERT/UPDATE/DELETE but the regex match fails (e.g., `INSERT SELECT ...`), execution falls through to the fallback truncation at line 171. This is correct behavior but the `elif` chain means a WITH...INSERT would be handled as a SELECT (line 125 `or cte_names`), which may produce confusing output for CTE inserts.

### MEDIUM -- Performance

#### P1. N+1 Prometheus Queries in `/pgss_metrics/csv`
- **File:** `monitoring_flask_backend/app.py`, lines 433-452
- **Severity:** High
- **Description:** The endpoint iterates over 16 metrics and makes 2 Prometheus HTTP requests per metric (start + end), resulting in **32 sequential HTTP calls** per request. This should use a single PromQL query with `{__name__=~"pgwatch_pg_stat_statements_.*"}` or batch queries.

#### P2. N+1 Prometheus Queries in `/table_info/csv`
- **File:** `monitoring_flask_backend/app.py`, lines 884-905
- **Severity:** High
- **Description:** Same pattern as P1: 14 metrics x 2 time points = **28 sequential HTTP calls**. Same fix applies.

#### P3. No Caching for Query Texts
- **File:** `monitoring_flask_backend/app.py`, lines 238-304, 1164-1240, 1261-1393
- **Severity:** Medium
- **Description:** Query texts change infrequently but are fetched from PostgreSQL on every request. The `/query_info_metrics` endpoint is designed for Prometheus scraping (every 15-60s), making caching particularly beneficial. A simple TTL cache (e.g., `functools.lru_cache` with TTL or `cachetools.TTLCache`) would reduce DB load significantly.

### LOW -- Code Quality

#### C1. Unused Import: `csv` Used But `io` Should Use `contextmanager`
- **File:** `monitoring_flask_backend/app.py`, lines 3-4
- **Severity:** Low
- **Description:** `io.StringIO` objects are created and manually closed. Using `with io.StringIO() as output:` would be safer, though the current approach works fine since StringIO.close() just marks the buffer as closed.

#### C2. `FLASK_ENV` is Deprecated
- **File:** `monitoring_flask_backend/Dockerfile`, line 41
- **Severity:** Low
- **Description:** `ENV FLASK_ENV=production` is deprecated since Flask 2.3. Use `ENV FLASK_DEBUG=0` instead, though in production with gunicorn this has no effect anyway.

#### C3. Missing `database` Field in Rate-Calculated Table Stats CSV
- **File:** `monitoring_flask_backend/app.py`, lines 1047-1052
- **Severity:** Low
- **Description:** When `calculate_rates` is True, the result rows include `schema` and `table_name` but not `database`. This is inconsistent with the instant query path which does include it (before removing at line 978). Not a bug since `database` isn't in the `fieldnames` list, but the row construction is asymmetric.

#### C4. Inconsistent Endpoint URL Patterns
- **File:** `monitoring_flask_backend/app.py`
- **Severity:** Low
- **Description:** Endpoint URLs mix styles: `/pgss_metrics/csv` vs `/btree_bloat/csv` vs `/query_texts` vs `/query_info_metrics`. The CSV endpoints use path segments for format, while the JSON endpoints don't. Minor inconsistency.

#### C5. `process_pgss_data` and `process_table_stats_with_rates` Not Route-Protected
- **File:** `monitoring_flask_backend/app.py`, lines 504, 1011
- **Severity:** Low
- **Description:** These helper functions are defined between route handlers rather than grouped with other utility functions at the top. This is a readability issue, not a bug.

### LOW -- Test Coverage

#### TC1. No Tests for CSV Endpoints
- **File:** `monitoring_flask_backend/test_app.py`
- **Severity:** Medium
- **Description:** `/pgss_metrics/csv`, `/btree_bloat/csv`, and `/table_info/csv` have zero test coverage. These are the most complex endpoints with time parsing, Prometheus queries, and CSV generation.

#### TC2. No Tests for `_sanitize_promql_label`
- **File:** `monitoring_flask_backend/test_app.py`
- **Severity:** Medium
- **Description:** The newly added PromQL sanitization function needs test coverage to prevent regressions.

#### TC3. No Integration/Contract Tests for Prometheus Queries
- **File:** `monitoring_flask_backend/test_app.py`
- **Severity:** Low
- **Description:** All Prometheus interactions are mocked. There are no contract tests verifying that the PromQL queries are syntactically valid or that the expected metric names exist.

---

## Fixes Applied in This Review

| ID | Description | Lines Changed |
|----|-------------|---------------|
| S1 | Added `_sanitize_promql_label()` and applied to all 14 PromQL filter interpolation sites | app.py: +15 lines (function), 14 call sites updated |

---

## Recommended Next Steps (Priority Order)

1. **Add tests for `_sanitize_promql_label`** -- verify injection attempts are neutralized
2. **Batch Prometheus queries** (P1, P2) -- reduce 32 HTTP calls to 2 per request
3. **Add TTL cache for query texts** (P3) -- reduce PostgreSQL connection churn
4. **Extract duplicate SQL into shared function** (A3) -- single source of truth
5. **Sanitize error responses** (S2) -- return generic messages to clients
6. **Gate `/debug/metrics`** (S3) -- behind env var or remove
7. **Add connection pooling** (A4) -- use `psycopg2.pool.ThreadedConnectionPool`
8. **Adopt app factory pattern** (A1) -- for better test isolation
