# Critical Failure Modes Registry

> These are the top failure modes that would lose customer trust. Each one MUST
> have dedicated automated coverage. This registry is the source of truth for
> what we cannot allow to ship broken.

## FM-1: Data Loss or Corruption During Clone/Snapshot

**Severity**: P0 — Catastrophic
**Component**: DBLab Engine (thin cloning)
**Customer impact**: Permanent data loss, broken clones, stale snapshots

### What Could Go Wrong

- Clone created from corrupted snapshot
- Snapshot taken during active checkpoint, producing inconsistent state
- Disk full during clone creation, leaving partial state
- OOM kill during snapshot, leaving dangling references
- Network partition during distributed clone operation

### Required Test Coverage

- [ ] Clone from healthy snapshot produces valid, queryable database
- [ ] Clone operation handles disk-full gracefully (cleanup, actionable error)
- [ ] Snapshot interrupted mid-operation doesn't corrupt existing snapshots
- [ ] Concurrent clone requests don't interfere with each other
- [ ] Clone from snapshot taken during heavy write load is consistent

### Automated Checks

- Integration test: create snapshot → clone → run pgbench validation queries
- Destructive test: fill disk to 95% → attempt clone → verify graceful failure
- Concurrent test: 5 simultaneous clones from same snapshot → all valid

---

## FM-2: Incorrect Diagnostic Recommendation

**Severity**: P0 — Critical
**Component**: Health checks (reporter + CLI checkup), SAMO Analyst
**Customer impact**: Customer acts on wrong advice, makes database worse

### What Could Go Wrong

- Index reported as "unused" when it's actually used by a critical query
  (e.g., index used only during monthly batch job, outside observation window)
- Bloat estimate wildly inaccurate due to TOAST tables or expression indexes
- Query identified as "top by time" is actually a monitoring query, not app query
- Redundant index detection flags a covering index that serves different queries
- Version-specific catalog differences cause wrong data interpretation

### Required Test Coverage

- [ ] H002 (unused indexes): index used in last 24h is NOT flagged
- [ ] H002: index only used by pg_stat_statements reset is handled correctly
- [ ] H004 (redundant indexes): covering index with additional columns is NOT flagged
- [ ] H004: partial indexes with different WHERE clauses are NOT flagged as redundant
- [ ] F004 (bloat): TOAST-heavy tables report accurate bloat estimates
- [ ] F004: tables with >50% dead tuples are correctly identified
- [ ] K003 (top queries): monitoring/system queries can be excluded
- [ ] K003: queryid consistency across PostgreSQL versions (14-17)
- [ ] All checks: empty database returns valid (empty) results, not errors
- [ ] All checks: database with no pg_stat_statements returns graceful degradation

### Automated Checks

- Schema validation: every check output conforms to `reporter/schemas/*.schema.json`
- Snapshot tests: check outputs against known-good baselines (syrupy)
- Cross-version tests: same dataset, same checks, PG14 vs PG15 vs PG16 produce
  consistent results (nightly matrix)

---

## FM-3: Silent Monitoring Failure

**Severity**: P0 — Critical
**Component**: Monitoring stack (pgwatch, Prometheus, Grafana)
**Customer impact**: Thinks monitoring is working, but metrics stopped flowing;
misses critical issues

### What Could Go Wrong

- pgwatch collector crashes silently, no metrics collected
- Prometheus scrape target becomes unreachable, no alert generated
- Grafana dashboard shows stale data without "last updated" indicator
- Sink database fills up, new metrics silently dropped
- TLS certificate expires, metric push fails silently
- Database password rotation breaks collector, no notification

### Required Test Coverage

- [ ] `mon health` detects when pgwatch has stopped collecting metrics
- [ ] `mon health` detects when Prometheus has no recent scrape results
- [ ] `mon health` detects when sink database is unreachable
- [ ] `mon status` shows accurate state of all services
- [ ] Adding/removing targets properly updates pgwatch configuration
- [ ] Target test (`mon targets test`) catches connection failures

### Automated Checks

- E2E test: `local-install --demo` → wait → verify metrics flowing in Prometheus
- E2E test: stop target-db → verify `mon health` reports unhealthy
- E2E test: `mon targets add` → verify metrics appear within 2 collection cycles

---

## FM-4: Security Exposure

**Severity**: P0 — Critical
**Component**: All (CLI auth, monitoring stack, database connections)
**Customer impact**: Unauthorized access to database credentials or data

### What Could Go Wrong

- API key logged in plaintext during auth flow
- Database credentials stored in config file with world-readable permissions
- Connection string with password visible in `docker inspect` or process list
- HTTPS not enforced for API communication
- Monitoring stack endpoints exposed without authentication
- SQL injection in dynamically constructed queries

### Required Test Coverage

- [ ] Auth flow: API key is masked in all log output (`show-key` shows `****`)
- [ ] Config file: `config.json` created with 0600 permissions
- [ ] Docker: no credentials visible in `docker inspect` environment section
- [ ] All SQL: parameterized queries (no string interpolation for user input)
- [ ] VictoriaMetrics: basic auth enabled, unauthenticated requests return 401
- [ ] Grafana: default credentials are randomized on `local-install`
- [ ] Pre-commit: gitleaks catches secrets before they reach the repository

### Automated Checks

- SAST: GitLab SAST pipeline scans every PR
- gitleaks: pre-commit hook + CI check
- CLI tests: auth flow tests verify masking behavior
- E2E tests: VM basic auth verification (already in `cli:node:full:dind`)

---

## FM-5: Performance Regression in Tooling

**Severity**: P1 — High
**Component**: CLI checkup, reporter
**Customer impact**: Checkup times out on large databases, monitoring becomes
the bottleneck

### What Could Go Wrong

- New check adds sequential scan on pg_class without filter, O(n) on table count
- Memory usage grows linearly with number of indexes/tables (no pagination)
- Connection held open for entire checkup duration instead of per-check
- JSON report generation creates massive string for databases with many objects
- N+1 query pattern: one query per table/index instead of batch query

### Required Test Coverage

- [ ] Checkup completes within 60s on database with 1000+ tables
- [ ] Memory usage stays under 256MB during checkup of large database
- [ ] Individual checks have statement_timeout (30s default)
- [ ] Report JSON stays under 10MB for databases with 10k objects
- [ ] No N+1 queries: each check category uses at most 3 queries

### Automated Checks

- Performance benchmark: nightly run against reference database, compare to baseline
- Query count assertion: integration tests verify expected number of SQL roundtrips
- Memory profiling: periodic check that peak RSS stays within bounds

---

## Failure Mode Ownership

| FM | Primary Owner | Backup | Last Reviewed |
|----|--------------|--------|---------------|
| FM-1 | DBLab team | — | — |
| FM-2 | Reporter/CLI engineer | — | — |
| FM-3 | Monitoring engineer | — | — |
| FM-4 | All engineers | — | — |
| FM-5 | CLI engineer | — | — |

> **Action**: Assign owners and schedule initial review dates during the next
> team meeting. Each failure mode should be reviewed quarterly.
