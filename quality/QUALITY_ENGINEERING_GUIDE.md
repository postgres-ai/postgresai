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

## Layer 3: Human Judgment

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
- [ ] **Visual changes**: if the MR touches any user-visible surface (Grafana
      dashboards, Console UI, Joe UI, marketing or landing pages), the
      description contains **before/after screenshots** for each visual fix.

See the [Visual Change Verification](#visual-change-verification) section
below for the full standard, and rule 11 in
[`quality/pr-review-prompt.md`](./pr-review-prompt.md) for the canonical
file-path globs the AI reviewer uses.

### Visual Change Verification

For every MR that modifies a user-visible surface, the description must include
a **Visual changes** section with one before/after pair per fix:

```markdown
## Visual changes

### Fix 1 — Dashboard 3 first panel "No data"
**Before:**
![before](before-d3-panel.png)
**After:**
![after](after-d3-panel.png)
```

This is the visual analogue of red/green TDD:

- A failing unit test proves the bug existed and is now fixed *in code*.
- A before/after screenshot pair proves the bug existed and is now fixed
  *in the rendered output*.

Text descriptions like "I fixed the layout" or "I changed the color" are not
sufficient — reviewers should not have to spin up the demo to confirm a visual
claim.

**Scope (which surfaces trigger this rule):** Grafana dashboards
(`config/grafana/dashboards/*.json`,
`postgres_ai_helm/.../dashboards/*.json`), the Console UI (`console/` or
similar frontend directories), Joe UI, and marketing or landing pages. See
rule 11 in [`quality/pr-review-prompt.md`](./pr-review-prompt.md) for the
authoritative list of path globs the AI reviewer uses.

**Capture mechanics:**

- `before` shots from the deployed staging/demo environment (the broken state
  users currently see).
- `after` shots from either the redeployed environment OR a local stack
  running the patched JSON / build.
- Upload to GitLab using a multipart `POST` to the project's `/uploads`
  endpoint (the `%2F` between `<namespace>` and `<project>` is the
  URL-encoded slash GitLab requires in the project identifier):

  ```bash
  curl --request POST \
    --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
    --form "file=@shot.png" \
    "https://gitlab.com/api/v4/projects/<namespace>%2F<project>/uploads"
  ```

  The response JSON contains a `markdown` field with the ready-to-paste
  `![](/uploads/...)` reference — copy that string into the MR description.

**Exception — render too expensive:**

If running the AFTER state would take more than ~10 minutes of setup (e.g., a
fix that requires the full monitoring stack with live data), the AFTER may be
replaced by a citation of the lint/unit test that proves the change. The
citation must name the test file path and the asserted invariant:

```markdown
**After (verified by test):** `tests/grafana_dashboards/test_xyz.py`
asserts the invariant; see line 42.
```

Use this exception only when (a) the AFTER state cannot be rendered in
under ~10 minutes, and (b) the cited test directly asserts the user-visible
invariant the fix establishes. The default is screenshots.

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


---

## UX/DX Quality Standards

The `postgresai` tool is used by engineers and DBAs who may be troubleshooting a production incident, running their first checkup, or installing the monitoring stack in an unfamiliar environment. Every user-facing surface — CLI output, error messages, report output, install flow, Grafana dashboards — must meet explicit standards. These are as non-negotiable as SQL safety.

Surfaces covered by this section:
- CLI (`postgresai checkup`, `postgresai init`, `postgresai monitoring`, subcommands)
- Report output (JSON schema + human-readable summary)
- Error messages (CLI, Python reporter, monitoring stack)
- Monitoring stack install and first-run experience
- Grafana dashboard UX

---

### CLI DX Standards

The CLI is the primary interface for most users. It must feel like a professional tool.

#### Help Text

Every command and subcommand must have:
- A one-line description that says what it does (not just what it is)
- All flags documented with types, defaults, and purpose
- At least one concrete usage example
- An explicit note on required vs optional flags

**Test:** `postgresai --help` and `postgresai <command> --help` must pass review before merge. New commands without complete `--help` are blocked.

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | User error (bad input, missing flag, invalid config) |
| 2 | System error (connection failed, permission denied, timeout) |
| 3 | Not found (target database unreachable, schema not present) |

Exit codes must be tested: every CLI integration test must assert the exit code, not just the output.

#### Stderr / Stdout Separation

- **Stdout:** Structured data, report output, machine-parseable results
- **Stderr:** Progress messages, warnings, errors, diagnostic output

Users pipe stdout to files or downstream tools. Mixing error messages into stdout breaks pipelines silently. This is a blocking violation.

#### Output Formats

Commands that produce structured output must support:
- Human-readable default (summary table or formatted output)
- `--output json` for machine consumption
- `--quiet` to suppress progress output (for scripting)

#### Progress Indication

Any operation expected to take >2 seconds must emit progress to stderr:
- Show what is happening ("Running K001: top query analysis...")
- Show completion or failure for each step
- Show a summary at the end ("12 checks complete, 2 warnings, 0 errors")

Long checkups with no output look like hangs. Users Ctrl-C and file bugs.

#### Actionable Errors

CLI errors must be actionable. The pattern is:

```
Error: <what failed>: <why>
Hint: <what to do>
```

Examples:

| ❌ Don't | ✅ Do |
|----------|-------|
| `Connection failed` | `Error: could not connect to postgres://user@host:5432/db: connection refused\nHint: Is PostgreSQL running? Check that the host and port are reachable.` |
| `Error: undefined` | `Error: missing required flag --connection-string\nHint: Run 'postgresai checkup --help' for usage.` |
| `Something went wrong` | `Error: checkup failed at K001: pg_stat_statements not found\nHint: Enable pg_stat_statements in postgresql.conf and reload PostgreSQL.` |

---

### Report Output DX Standards

Health check reports are the core product output. They must be readable by both humans and machines.

#### Human-Readable Summary

Every checkup run must produce a summary that a DBA can scan in 30 seconds:

```
PostgresAI Checkup — db-prod (PostgreSQL 16.2)
Ran 12 checks | 3 warnings | 0 critical

⚠  K001  Top queries: 4 queries using >80% of total query time
⚠  U001  Unused indexes: 7 indexes totaling 2.3 GB (13% of total DB size)
⚠  D004  Bloat: 3 tables with >30% dead tuple ratio
✓  9 other checks passed
```

Issues must be sorted by severity, not alphabetically. Critical issues surface first. The user's eye goes to the top.

#### Check Result Clarity

Each check result in CLI output must include:
- Check ID and name
- A one-sentence finding ("7 unused indexes totaling 2.3 GB")
- The recommended action, if any ("Run REINDEX CONCURRENTLY or DROP INDEX after verifying no app dependencies")
- A link to extended documentation

Raw numbers without context are not a finding. "Index size: 2.3 GB" tells a DBA nothing. "7 unused indexes consuming 2.3 GB (13% of total DB size) — candidates for removal" tells them what to do.

#### JSON Schema Compliance (Already Required)

This is already enforced (see §1.6). DX addition: every schema field must have a `description` property explaining what it contains. Schemas without field descriptions are not mergeable.

---

### Error Message Standards

Error messages in both CLI and Python reporter must follow the same standard:

1. **Be specific.** Name the failing operation, not just the category.
2. **Be actionable.** Give the user something to do.
3. **Be appropriate.** CLI errors are concise. Use structured format above.

**Anti-patterns — never ship:**
- `"error": null` with a non-zero exit code
- Exception class names as user-facing messages (`AttributeError: 'NoneType' object has no attribute 'get'`)
- Partial stack traces in user-visible output
- Silent failures (operation fails, exit 0, no output)
- "Please contact support" with no other guidance

**PostgreSQL error translation:** When surfacing `psycopg2` or `asyncpg` errors to users, translate them. `FATAL: password authentication failed for user "postgres"` → `Error: authentication failed for user "postgres" at host:5432\nHint: Check the password in your connection string.`

---

### Monitoring Stack Install DX

The `local-install` flow and demo mode are often the first experience with the full monitoring stack. Standards:

**Install flow must:**
- Print a clear step-by-step progress log (not silent Docker output)
- Detect and report common failures before they happen (Docker not running, port already in use, missing env vars)
- End with a clear "what to do next" summary: URLs, credentials, first steps
- Complete in under 5 minutes on a standard machine (benchmark this)

**Demo mode specifically:**
- Must work out of the box with zero configuration
- The README's "getting started in 5 minutes" must be tested in CI — if the steps are stale, they're broken
- First-run experience: when Grafana opens, the default dashboard must have data visible immediately (no "No data" panels)

**Common failure modes that must be caught early:**

| Failure | Current behavior | Required behavior |
|---------|-----------------|-------------------|
| Docker not running | Cryptic Docker error | Clear: "Docker is not running. Start Docker Desktop and retry." |
| Port 3000 already in use | `address already in use` buried in logs | Early check: "Port 3000 is in use. Pass --grafana-port to use a different port." |
| PostgreSQL unreachable | Timeout after 30s | Early check: "Cannot reach PostgreSQL at host:port. Check connection string." |
| Missing env var | Python KeyError | Startup validation: "Missing required env var DB_CONNECTION_STRING. See .env.example." |

---

### Grafana Dashboard UX

The monitoring stack ships Grafana dashboards. These are a user-facing product surface.

Standards for dashboards added or modified in this repo:
- Every panel must have a description (tooltip icon) explaining what the metric means and when to be concerned
- Thresholds must be set on relevant metrics (don't leave panels without color coding)
- Panel titles must be plain English ("Buffer Cache Hit Rate" not "bgwriter_buffers_alloc / checkpoints")
- Dashboards must have a top-level "status summary" row visible without scrolling
- Default time range must be useful for first open (last 1 hour, not last 6 months)
- Variables (database selector, etc.) must default to a useful value, not blank

**"No data" panels are launch blockers.** A dashboard with panels showing "No data" on first install looks broken even when it isn't. Either the metric must be collected or the panel must be hidden until data is available.

---

### UX/DX Quality Gates in CI

These automated checks must pass on every PR touching user-facing surfaces:

| Check | What it verifies | Blocks merge |
|-------|-----------------|:------------:|
| CLI help text presence | Every new command/subcommand has `--help` output | Yes |
| Exit code assertions | CLI integration tests assert exit codes, not just output | Yes |
| Error message format | stderr output matches `Error: ...\nHint: ...` pattern for known failures | Warning |
| Report summary presence | Every checkup run emits a human-readable summary block | Yes |
| JSON schema descriptions | All schema fields have `description` properties | Warning |

---

### UX/DX AI Review

Use the UX/DX review prompt alongside the PostgreSQL-specific prompt for any PR touching:
- CLI command structure or output format
- Error messages (Python or TypeScript)
- Report output templates or summary logic
- Install scripts or first-run flows
- Grafana dashboard JSON

The reviewer checks:
- Is there a human-readable summary, or just raw data?
- Are error messages specific and actionable?
- Does every long operation show progress?
- Does the install flow handle the top 5 failure modes gracefully?
- Are Grafana panel descriptions present?

---

### UX/DX Release Checklist Additions

Before releasing any version, verify:

- [ ] All new CLI commands have complete `--help` output with examples
- [ ] All new error messages follow the `Error: ... \nHint: ...` format
- [ ] `local-install --demo` completes cleanly and Grafana shows data on first open
- [ ] README "getting started" steps are tested and accurate
- [ ] No new "No data" panels in shipped Grafana dashboards
- [ ] Human-readable checkup summary is generated for all check combinations
- [ ] Exit codes are correct for all failure scenarios

---

*CLI DX and report readability are not secondary concerns. When a DBA runs `postgresai checkup` during an incident, they need answers fast. Every second of confusion is a second closer to an outage.*
