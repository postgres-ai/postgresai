# Review: Index Pilot Component

**Component:** `components/index_pilot/`
**Date:** 2026-03-12
**Scope:** SQL correctness, security, test coverage, install/uninstall, shell script, documentation

---

## Summary

Index Pilot is a well-architected PL/pgSQL-based automated index lifecycle manager for PostgreSQL. It uses a control database pattern with `postgres_fdw`/`dblink` to perform `REINDEX INDEX CONCURRENTLY` on target databases without deadlock risk. The codebase is thorough in version checks, bloat estimation, and configuration hierarchy. Several issues were found and fixed; others are noted for future work.

## Findings

### Security

#### S1: SQL injection via unquoted literals in `_cleanup_our_not_valid_indexes` [FIXED]
- **File:** `index_pilot_functions.sql`, `_cleanup_our_not_valid_indexes` procedure
- **Issue:** Used `format('%1$s', ...)` (unquoted string interpolation) for schema/table/index names in SQL sent over dblink. While the data comes from internally-populated `current_processed_index`, using `%L` (quote_literal) is the correct practice for string comparisons.
- **Also fixed:** `format('drop index concurrently %I.%I_ccnew', ...)` produced `"indexname"_ccnew` rather than `"indexname_ccnew"` because `_ccnew` was outside the `%I` specifier.
- **Fix:** Changed `%s` to `%L` for value comparisons, changed `_ccnew` suffix to use string concatenation before `%I` quoting.

#### S2: Shell script SQL injection in `index_pilot.sh` [FIXED]
- **File:** `index_pilot.sh`, `ensure_control_db()`
- **Issue:** `${CONTROL_DB}` was interpolated directly into `create database` and `select ... where datname =` SQL statements. If the control DB name contained quotes or special characters, this could produce malformed SQL.
- **Fix:** Applied single-quote escaping for the `datname` comparison and double-quote (identifier) quoting for the `CREATE DATABASE` statement.

#### S3: Shell script string interpolation in `register_target` [NOTED]
- **File:** `index_pilot.sh`, `register_target()`
- **Issue:** `${TARGET_DB}`, `${DB_USER}`, `${DB_PASS}`, and other variables are interpolated directly into SQL strings (e.g., `values ('${TARGET_DB}', ...)` and `password '${DB_PASS}'`). A password or database name containing single quotes would break the SQL or potentially allow injection.
- **Risk:** Moderate. The `sanitize_server_name` function handles server names, but database names, usernames, and passwords are not sanitized.
- **Recommendation:** Use `psql -v` variables with `:'varname'` quoting, or escape single quotes in all interpolated values.

#### S4: No `search_path` pinning on functions [NOTED]
- **Issue:** Functions do not use `SET search_path` in their definitions. While most internal references are schema-qualified (`index_pilot.`, `pg_catalog.`), a manipulated `search_path` could in theory shadow operators or implicit casts.
- **Risk:** Low, because the functions run in the control database where access is restricted.
- **Recommendation:** Add `SET search_path = pg_catalog, index_pilot` to function definitions for defense-in-depth.

#### S5: FDW credential model is sound
- Passwords are stored in `pg_user_mappings` (catalog-managed), not in application tables.
- `_connect_securely` correctly prevents self-connection (deadlock protection).
- `check_fdw_security_status()` provides comprehensive validation.

### SQL Correctness

#### C1: `_reindex_index` function is dead code [NOTED]
- **File:** `index_pilot_functions.sql`, lines 901-1004
- **Issue:** The function is defined but never called. The `do_reindex` procedure performs reindexing directly via `dblink_exec`. The function also inserts its own `reindex_history` record, which would cause double-logging if ever called alongside `do_reindex`.
- **Recommendation:** Remove or consolidate.

#### C2: `_pattern_convert` does not escape regex metacharacters [NOTED]
- **File:** `index_pilot_functions.sql`, line 434
- **Issue:** Only `*` and `?` are converted to regex equivalents. Characters like `.`, `+`, `(`, `)`, `[`, `]`, `^`, `$` in config pattern values will be interpreted as regex metacharacters rather than literals.
- **Risk:** Low, since PostgreSQL identifier names rarely contain these characters, but config entries like `event_*` technically match `event.` (any char) not just `event_`.
- **Recommendation:** Escape all regex metacharacters before applying `*`/`?` conversion.

#### C3: Advisory lock not released on exception in `periodic` [NOTED]
- **File:** `index_pilot_functions.sql`, `periodic` procedure
- **Issue:** If an exception occurs after `_check_lock()` acquires the advisory lock but before `pg_advisory_unlock(_id)`, the lock remains held for the session lifetime.
- **Risk:** Low for cron usage (session ends, lock released). Higher if running interactively with retries in the same session.
- **Recommendation:** Use `pg_try_advisory_xact_lock` where possible, or document the behavior.

#### C4: `do_reindex` connection cleanup commented out [NOTED]
- **File:** `index_pilot_functions.sql`, lines 1179-1186
- **Issue:** The exception handler for dblink cleanup is commented out because it conflicts with `commit` inside the procedure body. If an unhandled error occurs, the dblink connection may leak until session end.
- **Risk:** Low, since dblink connections are session-scoped and cleaned up on disconnect.

#### C5: DDL is properly transactional
- `index_pilot_tables.sql` and `index_pilot_functions.sql` are wrapped in `begin`/`commit`.
- Version check (`server_version_num >= 130000`) runs at install time.
- Unique indexes on `config` correctly enforce hierarchical uniqueness with partial index predicates.

#### C6: History view handles NULLs correctly
- The `history` view uses `nullif(indexsize_after, 0)` to prevent division by zero in ratio calculation.
- In-progress records with NULL `indexsize_after` produce NULL ratio/duration as expected.

### Install/Uninstall

#### I1: Uninstall drops FDW servers before schema -- correct order
- `uninstall.sql` reads `fdw_server_name` from `target_databases` before dropping the schema, which is correct.
- Invalid `_ccnew` indexes are reported but not auto-dropped (safe).
- Extensions (`postgres_fdw`, `dblink`) are intentionally left in place.

#### I2: Tables are not idempotent (`CREATE` without `IF NOT EXISTS`) [NOTED]
- **File:** `index_pilot_tables.sql`
- **Issue:** Uses `CREATE SCHEMA index_pilot` and `CREATE TABLE` without `IF NOT EXISTS`. Re-running the tables script will fail.
- **Recommendation:** The current design relies on `tables_version` for upgrades, and `check_update_structure_version()` handles migrations. Document that re-running `index_pilot_tables.sql` requires uninstall first.

#### I3: Version upgrade mechanism exists but is skeletal
- `check_update_structure_version()` calls `_structure_version_X_Y()` functions that don't yet exist (version is 1). The framework is ready for future migrations.

### Shell Script (`index_pilot.sh`)

#### H1: Input validation is thorough
- Port validation (`^[0-9]+$`), required field checks, and `psql` availability check are all present.
- `set -euo pipefail` ensures early failure.
- `PGPASSWORD` environment variable is preferred over command-line password.

#### H2: Unknown positional arguments are caught
- The parser collects unknown arguments and fails with a message.

#### H3: Password required for all subcommands [NOTED]
- All subcommands (`install-control`, `register-target`, `verify`, `uninstall`) require a password. This prevents use with `.pgpass` file or peer/cert authentication.
- **Recommendation:** Make password optional when other auth methods are available.

### Test Coverage

#### T1: Tests use `do $$` blocks, not pgTAP [NOTED]
- Despite the task description mentioning pgTAP, the tests use plain `do $$` blocks with `raise exception` for failures and `raise notice` for passes. This works but lacks pgTAP's structured output and assertion library.

#### T2: Test numbering gap in `02_functionality.sql`
- Tests jump from number 5 to 7 (test 6 is missing). Cosmetic issue only.

#### T3: Typo in `01_basic_installation.sql` [FIXED]
- Double closing parenthesis in notice message: `'PASS: Version function works (%))'` should be `'PASS: Version function works (%)'`.

#### T4: Missing test coverage [NOTED]
- No tests for `_pattern_convert` wildcard matching edge cases.
- No tests for `set_or_replace_setting` conflict resolution at different hierarchy levels.
- No tests for `_cleanup_old_records` retention behavior.
- No negative test for connecting to self (deadlock prevention).
- No test for concurrent `periodic` execution (advisory lock).

#### T5: Security test SQL injection check is weak
- **File:** `test/03_security.sql`, test 3
- **Issue:** The injection test appends `'; drop table ...; --'` to a database name, which will fail at connection time (no such database) rather than testing actual injection resistance in SQL construction. The test passes regardless of whether injection protection exists.

### Documentation

#### D1: Function reference documents non-existent functions [NOTED]
- **File:** `docs/function_reference.md`
- **Issue:** Documents `setup_fdw_self_connection()`, `setup_user_mapping()`, `setup_connection()`, and `setup_fdw_complete()`. These functions do not exist in any SQL file. They appear to be from a previous iteration that was refactored.
- **Impact:** The CI non-superuser test (`.gitlab-ci.yml` lines 174-176) also calls these functions and will fail.

#### D2: Table name mismatch in docs [FIXED]
- **Files:** `docs/runbook.md`, `docs/function_reference.md`
- **Issue:** Referenced `index_current_state` instead of the actual table name `index_latest_state`.

#### D3: `faq.md` uses `select` instead of `call` for procedure [FIXED]
- **File:** `docs/faq.md`
- **Issue:** `cron.schedule_in_database` example used `'select index_pilot.periodic(...)'` but `periodic` is a procedure requiring `CALL`.

#### D4: Installation docs reference non-existent `setup_connection` function [NOTED]
- **File:** `docs/installation.md` line 125, `CONTRIBUTING.md` line 44
- **Issue:** Self-hosted example calls `index_pilot.setup_connection(...)` which does not exist.

#### D5: `README.md` external cron example runs in wrong database [NOTED]
- **File:** `README.md`, "Using external cron" section
- **Issue:** The script connects to `postgres` database and calls `index_pilot.periodic(true)`, but `index_pilot` schema is in the control database, not `postgres`.

#### D6: Documentation is otherwise comprehensive
- Architecture diagram, FAQ, runbook, function reference, and installer CLI reference are all well-organized.
- Security notes about `PGPASSWORD` usage are present.
- Troubleshooting section covers common FDW issues.

---

## Changes Made

| File | Change |
|------|--------|
| `index_pilot_functions.sql` | Fixed SQL injection: `%s` to `%L` in `_cleanup_our_not_valid_indexes` dblink queries |
| `index_pilot_functions.sql` | Fixed `%I_ccnew` quoting bug in DROP INDEX statement |
| `index_pilot.sh` | Fixed unquoted `CONTROL_DB` in SQL statements in `ensure_control_db` |
| `docs/runbook.md` | Fixed `index_current_state` to `index_latest_state` |
| `docs/function_reference.md` | Fixed `index_current_state` to `index_latest_state` |
| `docs/faq.md` | Fixed `select` to `call` for `periodic` procedure in cron example |
| `test/01_basic_installation.sql` | Fixed double parenthesis typo in notice message |

## Recommendations (not fixed, for future work)

1. **Remove or consolidate `_reindex_index` dead code** to avoid confusion with `do_reindex`.
2. **Add `search_path` pinning** to function definitions for defense-in-depth.
3. **Remove or update documentation** for non-existent `setup_fdw_self_connection`, `setup_user_mapping`, `setup_connection`, `setup_fdw_complete` functions. Update CI accordingly.
4. **Fix `README.md` external cron example** to use the control database.
5. **Make password optional** in `index_pilot.sh` to support `.pgpass` and cert-based auth.
6. **Escape regex metacharacters** in `_pattern_convert` before wildcard conversion.
7. **Add tests** for config hierarchy, pattern matching, retention cleanup, and advisory lock contention.
8. **Sanitize all shell variables** interpolated into SQL in `register_target` (especially `DB_PASS`).
