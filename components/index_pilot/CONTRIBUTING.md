## Contributing to pg_index_pilot

Thank you for your interest in contributing! This document describes how to set up your environment, coding standards, how to run tests, and our commit/PR conventions.

### Ways to contribute
- Report bugs and edge cases
- Improve docs and examples
- Add tests and CI improvements
- Implement features and performance improvements
- Triage issues and review MRs/PRs

## Development setup

### Prerequisites
- PostgreSQL 13 or higher (psql client required)
- Bash shell (Linux/macOS)
- Docker (optional, for local PG quickly)

### Quick local Postgres (optional)
```bash
docker run --rm -d --name pg_dev -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:17-alpine
# wait ~5–10s until healthy
```

### Manual installation (control DB)
```bash
# 1) Create control database
psql -h <host> -U <admin_user> -c "create database index_pilot_control;"

# 2) Install required extensions in control DB
psql -h <host> -U <admin_user> -d index_pilot_control -c "create extension if not exists postgres_fdw;"
psql -h <host> -U <admin_user> -d index_pilot_control -c "create extension if not exists dblink;"

# 3) Load schema and functions
psql -h <host> -U <admin_user> -d index_pilot_control -f index_pilot_tables.sql
psql -h <host> -U <admin_user> -d index_pilot_control -f index_pilot_functions.sql
psql -h <host> -U <admin_user> -d index_pilot_control -f index_pilot_fdw.sql

# Note: at this step you may see a WARNING from the internal permissions self-check.
# It is expected before FDW self-connection is configured.

# 4) Configure secure FDW self-connection (password stored in PG catalog)
psql -h <host> -U <admin_user> -d index_pilot_control \
  -c "select index_pilot.setup_connection('<host>', 5432, '<user>', '<password>');"

# 5) Verify environment
psql -h <host> -U <admin_user> -d index_pilot_control -c "select * from index_pilot.check_permissions();"
psql -h <host> -U <admin_user> -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status();"
```

### Register a target database (inventory row + FDW server)
```sql
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name)
values ('<target_db>', '<target_host>', 5432, 'target_<target_db>');
```

### Run
```sql
-- dry run (no reindex)
call index_pilot.periodic(false);

-- real run (may reindex eligible indexes)
call index_pilot.periodic(true);
```

## Testing

- End-to-end SQL tests live in `test/`
- Run locally:
```bash
cd test
./run_tests.sh
```
- The suite installs into a temporary database, runs core checks (installation, functionality, security, in-progress handling), and prints a short report.

When adding features/bug fixes:
- Add or update tests to cover behavior
- Keep tests deterministic and fast

## Coding standards

### SQL
- Use lowercase keywords
- snake_case identifiers
- Be explicit: always use `as` for aliases; explicit join types
- Prefer CTEs over deeply nested queries
- Meaningful aliases (no single-letter unless obvious)
- One argument per line in multi-arg clauses
- Use ISO 8601 for timestamps
- Include comments for non-trivial logic

### Shell (Bash)
- We adhere to the Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html
- Bash only (`#!/bin/bash`), set `set -euo pipefail`
- Two-space indentation; keep line length reasonable
- Quote variables consistently; prefer `"${var}"`
- Prefer `$(...)` command substitution and `[[ ... ]]` tests
- Run ShellCheck for new/changed scripts

## Commit message convention and MR/PR titles

We use simplified Conventional Commits. A single line is enough. Types drive versioning and changelog generation.

Supported types:
- `feat:` new functionality (MINOR)
- `fix:` bug fix (PATCH)
- `perf:` performance improvement (PATCH)
- `docs:` documentation only (no release)
- `chore:` infra, CI, dependencies (no release)
- `test:` tests only (no release)
- `refactor:` code refactor without API changes (no release)

Breaking changes: add an exclamation mark after the type, e.g. `feat!: drop support for Postgres 13`.

Examples:
- `feat: add dry-run mode for automatic reindex`
- `fix: avoid deadlock on REINDEX CONCURRENTLY`
- `perf: reduce lock time by 20%`
- `docs: update PG 13–17 support matrix`
- `refactor!: remove legacy flag`

MR/PR titles should follow the same convention as the main commit.

Additional guidance:
- Keep PRs focused and reasonably small
- Include rationale, migration notes (if any), and benchmarks when relevant
- Reference related issues (e.g., `Closes #123`)

## Review checklist (for authors and reviewers)
- Tests added/updated and pass locally
- Backward compatibility considered; breaking changes clearly marked (`type!:`)
- Security and credentials: no secrets in code; FDW user mappings only
- Docs updated (README/runbook/installation) if behavior or UX changed
- Performance impact measured for non-trivial changes

## Reporting issues
Please include:
- PostgreSQL version (`select current_setting('server_version');`)
- `select index_pilot.version();`
- Output of `select * from index_pilot.check_fdw_security_status();` and `select * from index_pilot.check_permissions();`
- Recent failures if relevant
- Minimal reproduction steps and expected vs actual behavior
