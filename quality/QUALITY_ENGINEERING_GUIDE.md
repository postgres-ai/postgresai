# Quality Engineering Guide

> Living document — last updated: 2026-03-15
>
> This guide defines PostgresAI's quality standards, processes, and automated
> gates. It serves as the "constitution" that both AI agents and human engineers
> reference when building, reviewing, and releasing software.

## Core Philosophy: Quality as Code

PostgresAI products touch production PostgreSQL instances — mistakes can mean
data loss, incorrect diagnostics, or silent monitoring failures. Traditional QA
departments don't fit a small, distributed team. Instead, quality is embedded
into the development workflow itself:

| Layer | Purpose | Catches |
|-------|---------|---------|
| **1. Automated Gates** | CI/CD pipelines, pre-commit hooks, schema validation | ~80% of issues before any human sees them |
| **2. AI-Assisted Review** | PostgreSQL-specific PR review, test generation, spec gap analysis | Edge cases, combinatorial scenarios, domain-specific bugs |
| **3. Human Judgment** | Architecture decisions, customer scenarios, risk assessment | Design flaws, UX issues, safety-critical decisions |

---

## Layer 1: Automated Foundation

### 1.1 Pre-Commit Hooks

Every developer must have pre-commit hooks installed (`pre-commit install`).
Current hooks:

- **gitleaks** — Prevents secrets from being committed
- **TypeScript typecheck** — Catches type errors before push
- **pytest (unit)** — Runs fast unit tests on changed Python files

### 1.2 CI Pipeline Quality Gates

Every PR must pass these gates before merge:

| Gate | Tool | Blocks Merge |
|------|------|:------------:|
| Python unit + integration tests | pytest + pytest-postgresql | Yes |
| CLI unit tests + coverage | Bun test runner | Yes |
| CLI smoke tests | Node.js + built CLI | Yes |
| E2E monitoring stack tests | Docker-in-Docker | Yes |
| Helm/config validation | pytest + helm template | Yes |
| SAST security scanning | GitLab SAST | Yes |
| Secret detection | gitleaks | Yes |
| JSON schema validation | ajv / jsonschema | Yes |
| Performance regression check | Benchmark comparison | Warning |

### 1.3 PostgreSQL Version Matrix

Products must be tested against supported PostgreSQL versions:

| Version | Status | CI Coverage |
|---------|--------|:-----------:|
| 14 | Supported | Nightly |
| 15 | Supported | Every PR |
| 16 | Supported | Every PR |
| 17 | Supported | Nightly |
| 18 | Preview | Weekly |

### 1.4 Test Categories

Tests are organized by execution speed and infrastructure requirements:

```
pytest markers:
  unit          — Fast, mocked, no external services (~seconds)
  integration   — Requires PostgreSQL (~30s)
  requires_postgres — Alias for integration
  e2e           — Full monitoring stack (~minutes)
  enable_socket — Allow network access

Bun test tags:
  *.test.ts              — Unit tests (default)
  *.integration.test.ts  — Integration tests
```

### 1.5 Coverage Requirements

| Component | Minimum | Target |
|-----------|:-------:|:------:|
| Reporter (Python) | 70% | 85% |
| CLI (TypeScript) | 60% | 80% |
| New code (any) | 80% | 95% |

Coverage is reported automatically in CI and visible in MR/PR comments.

### 1.6 Schema Validation

All health check outputs must conform to JSON schemas in `reporter/schemas/`.
Schema compliance is enforced at two levels:

1. **Build time** — `test_report_schemas.py` validates all check outputs
2. **Runtime** — `checkup.ts` validates against embedded schemas before upload

When adding a new check:
1. Create `reporter/schemas/<CHECK_ID>.schema.json`
2. Add test cases in `tests/reporter/`
3. Add CLI implementation in `cli/lib/checkup.ts`
4. Validate output matches schema in both Python and TypeScript paths

---

## Layer 2: AI-Assisted Quality

### 2.1 AI PR Review

Every PR is reviewed by an AI agent with the PostgreSQL-specific system prompt
defined in `quality/pr-review-prompt.md`. The review focuses on:

- **SQL safety** — Injection paths, raw string concatenation, missing parameterization
- **Connection handling** — Unclosed connections, missing timeouts, pool exhaustion
- **Transaction safety** — Incorrect isolation assumptions, long-running transactions
- **Resource leaks** — Unreleased advisory locks, unclosed cursors, temp table accumulation
- **PostgreSQL version compatibility** — Features not available in all supported versions
- **Error handling** — Missing error paths on database operations
- **Lock awareness** — DDL that acquires AccessExclusive locks, missing `CONCURRENTLY`

### 2.2 AI Test Generation

When implementing a new health check or analyzer, use AI to generate test
scaffolding:

1. Write the spec/implementation
2. Feed to AI with prompt: *"Generate test cases for this PostgreSQL analyzer.
   Cover: normal case, empty table, table with no indexes, partial indexes,
   expression indexes, concurrent DDL during analysis, permission errors,
   PostgreSQL version differences."*
3. Developer reviews, adjusts, and commits the tests

### 2.3 Spec Gap Analysis

Before implementation begins, feed the spec to AI for gap analysis:

- *"What failure modes aren't addressed in this spec?"*
- *"What PostgreSQL version-specific behaviors could affect this?"*
- *"What happens if this runs concurrently with vacuum/reindex/DDL?"*

### 2.4 Automated Issue Triage

When a bug report arrives:
1. AI agent classifies severity (P0-P3)
2. Identifies likely affected components (reporter, CLI, monitoring stack)
3. Searches for related past issues
4. Drafts initial investigation path
5. Human picks up with context already assembled

---

## Layer 3: Human Quality Decisions

### 3.1 Architecture Reviews

Required for:
- New health checks that modify database state
- Changes to the Analyst/Auditor/Actor pipeline
- New autonomous actions (anything that writes to production databases)
- Changes to connection pooling or authentication flows
- New PostgreSQL extension dependencies

### 3.2 Customer Scenario Testing

Before each release, one engineer walks through key customer workflows:

| Scenario | What to verify |
|----------|---------------|
| Express checkup on fresh PostgreSQL | All checks run, report is valid JSON, upload succeeds |
| Monitoring stack install (demo mode) | `local-install --demo` completes, Grafana accessible, metrics flowing |
| Add external target database | Target added, metrics collected, checkup runs against it |
| Large database checkup | No timeouts, memory stays bounded, results are accurate |
| Extension-heavy database | Common extensions (PostGIS, pg_partman, pg_stat_statements) don't cause failures |

### 3.3 Risk Classification for Autonomous Actions

Every autonomous action (current or future) must have a risk classification:

| Risk Level | Description | Gate |
|------------|-------------|------|
| **Read-only** | Queries, EXPLAIN, pg_stat views | Automated |
| **Advisory** | Recommendations shown to user | AI review + human spot-check |
| **Reversible write** | CREATE INDEX CONCURRENTLY, config changes with reload | Human approval required |
| **Irreversible write** | DROP, TRUNCATE, ALTER TABLE rewrite | Human approval + confirmation prompt |

---

## PostgreSQL-Specific Quality Standards

### SQL Query Standards

- All queries generated by the product must be tested with `EXPLAIN ANALYZE`
- No sequential scans on tables expected to have >10k rows
- No queries that acquire `AccessExclusiveLock` without explicit documentation
- All SQL uses parameterized queries (`$1`, `$2`) — never string concatenation
- Queries must specify `statement_timeout` for safety

### Extension Compatibility

First-class CI coverage for these extensions (used by most customers):

| Extension | Why |
|-----------|-----|
| pg_stat_statements | Core dependency for K-series checks |
| pg_stat_kcache | CPU/IO metrics in D004 |
| auto_explain | Query plan analysis |
| pg_buffercache | Buffer analysis |
| PostGIS | Common in customer deployments |
| pg_partman | Partition management |
| pgvector | Growing adoption |

### Connection Handling Standards

- All connections must have a `statement_timeout` (default: 30s for checks)
- All connections must have a `connect_timeout` (default: 10s)
- Connections must be returned to pool or closed in `finally` blocks
- Connection errors must produce actionable error messages
- Maximum connection count must be configurable and bounded

### WAL and Replication Safety

- Features touching WAL or replication need tests for:
  - Replica lag scenarios
  - Failover during operation
  - WAL segment cleanup interaction
- Never hold connections across WAL switch boundaries unnecessarily

---

## Process: Feature Development Workflow

### For Every Feature

```
1. Spec written
   └─→ Spec reviewed by engineer + AI gap analysis

2. Implementation + tests
   └─→ Developer writes code
   └─→ AI generates test scaffolding from spec
   └─→ Developer refines tests

3. PR opened
   └─→ CI runs fast suite (unit + lint + typecheck)
   └─→ AI runs PostgreSQL-specific review
   └─→ Human reviewer focuses on design + correctness

4. Merge to main
   └─→ Nightly: full PostgreSQL version matrix
   └─→ Nightly: performance benchmarks vs baseline

5. Release candidate
   └─→ AI produces release readiness report
   └─→ Human does scenario walkthrough
   └─→ Go/no-go decision
```

### PR Review Checklist

Before approving any PR, verify:

- [ ] Tests cover the happy path AND at least 2 error paths
- [ ] New SQL queries are parameterized (no string concatenation)
- [ ] Database connections are properly closed/returned
- [ ] New checks have corresponding JSON schema
- [ ] Schema changes are backward-compatible
- [ ] No new dependencies without justification
- [ ] Error messages are actionable (not just "something went wrong")
- [ ] PostgreSQL version-specific behavior is handled
- [ ] No hardcoded credentials, tokens, or connection strings

---

## Quality Metrics

Track these metrics to measure quality system effectiveness:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Test coverage (Python) | `pytest --cov` in CI | >70% overall, >80% new code |
| Test coverage (CLI) | Bun coverage in CI | >60% overall, >80% new code |
| CI pipeline pass rate | GitLab CI analytics | >90% on main |
| Mean time bug-intro → detection | Git blame + issue timestamps | <1 sprint |
| Performance benchmark trend | Nightly benchmark results | No regression >5% |
| Schema validation failures | CI artifact count | 0 on main |
| Security findings (SAST) | GitLab security dashboard | 0 critical/high |

---

## Weekly Quality Rhythm

| Day | Activity |
|-----|----------|
| **Monday** | Review nightly test failures, triage new issues |
| **Wednesday** | Mid-week check: any flaky tests? CI pipeline health? |
| **Friday** | Quality retro: what slipped through? New test needed? CI tightening? |

---

## What We Don't Do

- **Dedicated QA team** — Quality ownership stays with engineers, amplified by AI
- **Manual test plans in spreadsheets** — Everything is code
- **Separate staging that drifts** — Use monitoring stack's own Docker setup to mirror real environments
- **100% coverage targets** — Diminishing returns; focus on critical paths and failure modes
