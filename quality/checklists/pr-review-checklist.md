# PR Review Checklist

Use this checklist when reviewing pull requests. Not every item applies to every
PR — use judgment. Items marked **(required)** must be verified for all PRs.

## General

- [ ] **(required)** PR has a clear title and description explaining the "why"
- [ ] **(required)** Changes match the stated intent (no scope creep)
- [ ] Tests cover the happy path AND at least 2 error/edge cases
- [ ] No unrelated changes bundled into the PR

## SQL & Database

- [ ] **(required)** All SQL uses parameterized queries (no string interpolation)
- [ ] **(required)** Database connections are closed/returned in finally blocks
- [ ] New queries have been reviewed with EXPLAIN ANALYZE in mind
- [ ] Queries include appropriate statement_timeout
- [ ] No queries that acquire AccessExclusiveLock without justification
- [ ] No N+1 query patterns (queries inside loops)
- [ ] Results are bounded (LIMIT or WHERE clause prevents unbounded returns)

## Health Checks

- [ ] New check has JSON schema in `reporter/schemas/<ID>.schema.json`
- [ ] Check output conforms to schema (validated in tests)
- [ ] Check handles empty database gracefully (no errors, returns empty results)
- [ ] Check handles missing pg_stat_statements gracefully
- [ ] Check works on PostgreSQL 14-17 (version-specific SQL handled)
- [ ] Check title and description added to checkup dictionary

## Security

- [ ] **(required)** No hardcoded credentials, tokens, or connection strings
- [ ] **(required)** Sensitive data not logged in plaintext
- [ ] SSL/TLS options handled correctly
- [ ] File permissions appropriate for sensitive config files (0600)

## Error Handling

- [ ] **(required)** Error messages are actionable (what failed, why, what to do)
- [ ] No bare catch blocks that swallow errors silently
- [ ] Connection errors distinguished from query errors
- [ ] Graceful degradation when optional features are unavailable

## Compatibility

- [ ] No new Node.js features requiring version >18
- [ ] No new Python features requiring version >3.11
- [ ] Schema changes are backward-compatible (additive only)
- [ ] No new dependencies without justification in PR description

## Monitoring Stack

- [ ] Docker Compose changes tested with `mon local-install --demo`
- [ ] Resource limits (CPU/memory) are appropriate
- [ ] Health check endpoints remain accessible
- [ ] Configuration changes are backward-compatible with existing installations
