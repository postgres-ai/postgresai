# Reporter Module Review

**Reviewer:** Claude Code
**Date:** 2026-03-12
**Scope:** `reporter/` (4 Python files) + `tests/reporter/` (33 test files)
**Files reviewed:**
- `reporter/__init__.py`
- `reporter/logger.py`
- `reporter/report_schemas.py`
- `reporter/postgres_reports.py` (~5127 lines)
- `reporter/schemas/*.schema.json` (27 schemas)
- `tests/reporter/conftest.py` + 32 test modules

---

## Summary

The reporter module is a well-structured, single-class system (`PostgresReportGenerator`) that
queries Prometheus/PromQL for pgwatch metrics and produces JSON health check reports. Code quality
is generally good with consistent patterns, proper docstrings, and solid test coverage. The issues
found are mostly low-medium severity with a few clear bugs fixed in this review.

---

## Findings

### 1. Bugs Fixed

#### BUG-01: `query_text_limit` typo truncates query text at 65 KB instead of 640 KB (FIXED)
- **File:** `reporter/postgres_reports.py`, line 4991
- **Severity:** Medium
- **Details:** The comment says "640 KB should be enough for anybody" but the value passed was
  `66560` (65 KB) instead of `655360` (640 KB). This silently truncates long query texts in the
  main CLI codepath. The method signature default (`655360`) and the
  `generate_per_query_jsons` default were both correct; only the `main()` call site was wrong.

#### BUG-02: Unreachable `sys.exit(1)` after `raise e` (FIXED)
- **File:** `reporter/postgres_reports.py`, lines 5117-5119
- **Severity:** Low
- **Details:** `raise e` on line 5118 unconditionally re-raises, making `sys.exit(1)` on line 5119
  dead code. Fixed to bare `raise` (preserves traceback) and removed unreachable exit.

#### BUG-03: Mixed `print()` and `logger` usage (FIXED)
- **File:** `reporter/postgres_reports.py`, lines 3683, 3717, 3738, 4126, 4139, 4151, 5028
- **Severity:** Low
- **Details:** Seven `print()` calls bypass the structured logging pipeline. These were in
  `generate_d004_from_a003`, `generate_f001_from_a003`, `generate_g001_from_a003`, and `main()`.
  All converted to `logger.info()` or `logger.warning()` as appropriate.

---

### 2. PromQL Query Correctness

#### PROMQL-01: No sanitization of label values interpolated into PromQL
- **File:** `reporter/postgres_reports.py`, throughout (e.g., lines 481, 531-540, 673, 834)
- **Severity:** Low (internal use only)
- **Details:** Cluster names, node names, and database names are interpolated directly into PromQL
  f-strings: `cluster="{cluster}"`. If any value contains a double-quote or backslash, the query
  would be malformed. The `_build_qid_regex` method (line 3289) validates queryids via regex, but
  cluster/node/db names are not validated. This is low risk because values originate from Prometheus
  labels (not user input), but a defensive sanitization function would be prudent.

#### PROMQL-02: Label inconsistency between `datname` and `dbname`
- **File:** `reporter/postgres_reports.py`, lines 673 vs 925
- **Severity:** Info
- **Details:** Some pgwatch metrics use `datname` (e.g., unused indexes at line 819) while others
  use `dbname` (e.g., redundant indexes at line 925). The code handles this correctly by querying
  the right label for each metric, and `get_all_databases()` (line 4637) unifies both. This is
  not a bug but warrants a comment explaining the inconsistency is intentional.

#### PROMQL-03: Queries are well-formed and use appropriate functions
- **Severity:** Info (positive finding)
- **Details:** `last_over_time(...[3h])` is used consistently for instant gauge lookups.
  `increase(...[1h])` is used for hourly delta calculations in topk logic.
  `topk(k, ...)` is used for per-hour ranking. The `sum by (queryid)` grouping is correct.

---

### 3. Report Schema Compliance

#### SCHEMA-01: Schema coverage is comprehensive
- **Severity:** Info (positive finding)
- **Details:** 26 check-specific schemas + 1 query schema exist under `reporter/schemas/`.
  All use Draft 2020-12 with `additionalProperties: false`, which catches unexpected fields.
  Schemas share a common envelope (`checkId`, `checkTitle`, `timestamptz`, `nodes`, `results`).

#### SCHEMA-02: No schema versioning strategy
- **File:** `reporter/schemas/`
- **Severity:** Low
- **Details:** Schemas have no `$id` with version, and there is no migration/versioning mechanism.
  If the report format changes, old consumers will break without warning. Consider adding a
  `schemaVersion` field to the report envelope.

#### SCHEMA-03: Schema validation only runs on explicit calls
- **File:** `reporter/report_schemas.py`, lines 24-30
- **Severity:** Low
- **Details:** `validate_report()` exists but is not called during report generation. It is only
  used in tests. This means production could emit schema-invalid reports silently. Consider
  adding validation in `format_report_data()` at least in debug mode.

---

### 4. Memory Management

#### MEM-01: `gc.collect()` usage is defensive but reasonable
- **File:** `reporter/postgres_reports.py`, lines 4104-4106, 4540-4541, 4544-4545, 5001, 5014-5015
- **Severity:** Info
- **Details:** Manual `gc.collect()` is called periodically during report generation and after
  `del` statements. This is a pragmatic approach for a long-running process that generates many
  large JSON reports. The periodic collection (every 5 reports, every 10 queries) prevents
  accumulation of cyclic references. The `import gc` on line 30 is justified.

#### MEM-02: Server-side cursors used correctly for Postgres queries
- **File:** `reporter/postgres_reports.py`, lines 239, 302
- **Severity:** Info (positive finding)
- **Details:** Named cursors (`name='index_defs_cursor'`, `name='queryid_cursor'`) are used
  for potentially large result sets, enabling server-side iteration without loading all rows
  into memory.

#### MEM-03: `generate_per_query_jsons` can accumulate large dicts
- **File:** `reporter/postgres_reports.py`, line 4443
- **Severity:** Low
- **Details:** `get_queryid_queries_from_sink` is called with `db_names=None` (fetches ALL
  databases) to work around db name mismatches. For clusters with many databases and queries,
  this could be significant. The `write_immediately=True` mode mitigates per-query accumulation.

---

### 5. Error Handling

#### ERR-01: `except Exception` without logging in `_read_text_file`
- **File:** `reporter/postgres_reports.py`, line 172
- **Severity:** Low
- **Details:** `_read_text_file` catches all exceptions silently. This is intentional (build
  metadata files may not exist in dev), but logging at DEBUG level would aid troubleshooting.

#### ERR-02: Silent exception swallowing in `get_query_metrics_from_prometheus`
- **File:** `reporter/postgres_reports.py`, lines 4370-4372
- **Severity:** Low
- **Details:** `except Exception: pass` silently drops all errors for individual metric fetches.
  The comment explains this ("some may not exist for older PG versions"), but a debug-level log
  would help distinguish missing metrics from actual failures.

#### ERR-03: No retry logic for Prometheus queries
- **File:** `reporter/postgres_reports.py`, `query_instant()` and `query_range()`
- **Severity:** Low
- **Details:** All Prometheus queries use a single attempt with 30s timeout. Transient network
  errors will cause silent data gaps in reports (logged as errors but not retried). Consider
  adding a simple retry with backoff for transient HTTP errors (5xx, connection errors).

#### ERR-04: `connect_postgres_sink` has no connection timeout
- **File:** `reporter/postgres_reports.py`, line 210
- **Severity:** Low
- **Details:** `psycopg2.connect(self.postgres_sink_url)` uses no explicit `connect_timeout`.
  If the Postgres sink is unreachable, this could block for the OS-level TCP timeout (often
  2+ minutes). Consider passing `connect_timeout=10` in the connection string or as a parameter.

---

### 6. Connection Management

#### CONN-01: No connection pooling
- **File:** `reporter/postgres_reports.py`, lines 202-220
- **Severity:** Info
- **Details:** A single `psycopg2` connection is reused across all sink queries. This is
  acceptable for the current use case (single-threaded batch processing). Connection pooling
  would only be needed if the module were used in a multi-threaded context.

#### CONN-02: Postgres connection not used as context manager
- **File:** `reporter/postgres_reports.py`, lines 202-220
- **Severity:** Low
- **Details:** The Postgres connection is stored as `self.pg_conn` and manually closed in
  `close_postgres_sink()`. If an exception occurs between connect and close, the connection
  leaks. The `finally` block in `main()` (line 5120-5122) handles this, but library consumers
  who use `PostgresReportGenerator` directly could miss it. Consider implementing `__enter__`
  and `__exit__` for context manager support.

#### CONN-03: No `requests.Session` reuse for Prometheus
- **File:** `reporter/postgres_reports.py`, lines 370-379
- **Severity:** Low
- **Details:** Each `query_instant()` and `query_range()` call creates a new HTTP connection
  via `requests.get()`. Using a `requests.Session` would enable HTTP keep-alive and connection
  reuse, reducing latency for the hundreds of queries made during report generation.

---

### 7. Python Code Quality

#### PY-01: Timezone-naive `datetime.now()` used in multiple places
- **File:** `reporter/postgres_reports.py`, lines 797, 816, 1872, 2013, 2132, 2254, 2382
- **Severity:** Medium
- **Details:** `datetime.now()` returns a timezone-naive datetime. When mixed with
  `datetime.fromtimestamp(epoch)` (also naive, uses local timezone), the results are
  consistent but not UTC-aware. If the server timezone changes or differs between environments,
  calculations like `days_since_reset` could be off. The `format_report_data` method correctly
  uses `datetime.now(timezone.utc)` (line 3551), creating an inconsistency. All `datetime.now()`
  calls should use `datetime.now(timezone.utc)` and all `datetime.fromtimestamp()` should use
  `datetime.fromtimestamp(v, tz=timezone.utc)`.

#### PY-02: Type hint `str | None` syntax requires Python 3.10+
- **File:** `reporter/postgres_reports.py`, line 3520
- **Severity:** Low
- **Details:** `def format_epoch_timestamp(self, epoch_value: float) -> str | None:` uses PEP 604
  union syntax. The file uses `from __future__ import annotations` only in `report_schemas.py`,
  not in `postgres_reports.py`. If this module needs to support Python 3.9, this should be
  `Optional[str]`.

#### PY-03: Duplicated settings lists
- **File:** `reporter/postgres_reports.py`, lines 54-117 vs 1035-1047, 1214-1237, 1490-1514
- **Severity:** Low
- **Details:** The class-level constants `D004_SETTINGS`, `F001_SETTINGS`, `G001_SETTINGS` are
  defined at lines 54-117 for the `_from_a003` methods. But the standalone `generate_d004_*`,
  `generate_f001_*`, `generate_g001_*` methods define their own identical local lists. These
  should reference the class constants to avoid drift.

#### PY-04: `_parse_memory_value` assumes bare numbers are KB
- **File:** `reporter/postgres_reports.py`, lines 1641-1648
- **Severity:** Low
- **Details:** The fallback for bare numeric values assumes KB (`return numeric_value * 1024`).
  PostgreSQL settings use different base units: `shared_buffers` uses 8kB blocks, `work_mem`
  uses kB. This simplification could cause incorrect memory estimates in `_analyze_memory_settings`.
  The comment acknowledges this ("This is a simplified assumption").

#### PY-05: Deprecated `generate_queries_json` method still present
- **File:** `reporter/postgres_reports.py`, lines 4165-4186
- **Severity:** Info
- **Details:** Marked as `DEPRECATED` but still present and callable. Consider removing it in
  the next major version.

#### PY-06: `boto3` import is unconditional but only needed for AMP
- **File:** `reporter/postgres_reports.py`, lines 43-44
- **Severity:** Low
- **Details:** `import boto3` and `from requests_aws4auth import AWS4Auth` are top-level imports
  that will fail if these packages are not installed, even when AMP is not used. These should be
  guarded with try/except like `psycopg2` is (lines 37-41).

---

### 8. Test Coverage Observations

- 33 test files provide solid coverage of the reporter module.
- Tests use well-structured fixtures (`conftest.py`) with mock Prometheus responses.
- Schema validation tests exist (`test_report_schemas.py`, `test_report_schemas_hourly.py`).
- Edge cases, error paths, and connection errors are covered.
- Test file naming suggests iterative coverage pushes (`test_final_push_to_85.py`,
  `test_final_coverage_push.py`), which could be consolidated.

---

## Changes Made

| Change | File | Lines |
|--------|------|-------|
| Fixed `query_text_limit` typo: 66560 -> 655360 | `reporter/postgres_reports.py` | 4991 |
| Removed unreachable `sys.exit(1)` after `raise` | `reporter/postgres_reports.py` | 5117-5119 |
| Replaced 7 `print()` calls with `logger` calls | `reporter/postgres_reports.py` | 3683, 3717, 3738, 4126, 4139, 4151, 5028 |
