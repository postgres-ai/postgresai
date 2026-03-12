# Review 09: Documentation & Developer Experience

**Date:** 2026-03-12
**Scope:** All `.md` files, CLI help text, scripts, inline documentation
**Status:** Complete

---

## Summary

The documentation is generally comprehensive and well-structured. The README provides a clear on-ramp for new users, CONTRIBUTING.md gives a detailed local development workflow, and component docs (index_pilot, terraform) are thorough. Several concrete issues were found and fixed directly; remaining items are noted below.

---

## Fixes Applied

### 1. Missing backtick in README.md (typo)
**File:** `README.md` line 211
**Was:** `- Full monitoring: omit --api-key\``
**Now:** `- Full monitoring: omit \`--api-key\``
The opening backtick was missing, causing broken inline code rendering.

### 2. Inconsistent GitLab repo URLs (broken links)
**Files:** `README.md`, `cli/README.md`
Several links pointed to `postgres-ai/postgres_ai` (underscore) while the actual repo appears to be `postgres-ai/postgresai` (no underscore), matching the badges and the Issues link in the Links table. Fixed:
- `README.md` line 464: Contributing link
- `README.md` line 482: Report issues link
- `cli/README.md` line 401: Issues link

### 3. Missing blank line before `## Links` heading (README.md)
**File:** `README.md` line 213
The `## Links` heading was run together with the preceding bullet point, violating Markdown spec (heading needs a blank line before it).

### 4. Skipped step numbers in index_pilot README
**File:** `components/index_pilot/README.md`
Both the RDS and self-hosted installation sections numbered steps 1-5 then jumped to 7, skipping step 6. Fixed to use consecutive numbering (step 7 renamed to step 6).

---

## Findings (Not Fixed — Require Discussion)

### HIGH

#### H1. TMP_MON_PREVIEWS.md should be removed or relocated
**File:** `TMP_MON_PREVIEWS.md` (64KB)
This is a detailed internal planning document ("Version 1.7.0, Status: Implementation Complete - Awaiting VM Provisioning") for preview environments. It contains architecture decisions, CI/CD pipeline designs, and implementation phases. The implementation is done and the operational docs already live in `CONTRIBUTING.md` (Preview Environments section) and `preview-infra/RUNBOOK.md`. This file is stale planning material that should either be:
- Deleted (the implementation is complete and documented elsewhere), or
- Moved to a `docs/adr/` or `docs/design/` directory if historical record is desired

#### H2. README has two distinct sections with different style
The README has a clean, focused top section (lines 1-221) and then a much longer bottom section (lines 222-485) inside a `<div align="center">` that repeats and expands on the same content with emoji-prefixed headings. This creates confusion:
- "Use cases" repeats "Get Started"
- "Management commands" repeats commands from the CLI README
- "Checkup reports" is a useful reference table but duplicates `CLAUDE.md`
- "Testing" section appears at the bottom of README rather than in CONTRIBUTING.md

Consider extracting the lower half into dedicated docs or consolidating.

#### H3. `instances.yml` referenced in CONTRIBUTING.md does not exist in repo
CONTRIBUTING.md references editing `instances.yml` directly, but this file does not exist in the repository (it is generated at runtime by the CLI into `~/.config/postgresai/monitoring/`). A contributor following the CONTRIBUTING.md instructions would need to understand this is created during `docker compose` setup. This should be clarified.

### MEDIUM

#### M1. CONTRIBUTING.md references `.git/info/exclude` for ignoring local files
CONTRIBUTING.md states "This repo is already configured (locally) to ignore these files via `.git/info/exclude`" for `.env`, `.pgwatch-config`, etc. However, `.git/info/exclude` is local to each clone and not shared — so this is only true if someone manually sets it up. These patterns should be in `.gitignore` instead, or the instructions should say "add these to your local `.git/info/exclude`".

#### M2. CLI command name mismatch: `postgres-ai` vs `postgresai`
The CLI source (`cli/bin/postgres-ai.ts`) uses `.name("postgres-ai")` for the Commander program name, but the npm package and all documentation use `postgresai` (no hyphen). While both work due to npm bin aliasing, the `--help` output shows `postgres-ai` which could confuse users.

#### M3. No CHANGELOG or release notes process documented
Neither the README, CONTRIBUTING.md, nor any other doc describes how releases are versioned or where changelogs are maintained. The cli/README.md mentions "placeholder version (0.0.0-dev.0)" set by CI, but there is no CHANGELOG.md or documented release process.

#### M4. Terraform docs reference version `0.10` which may be stale
`terraform/aws/README.md` and `QUICKSTART.md` both default to `postgres_ai_version = "0.10"`. If the project has moved past this version, new deployments would get an outdated version by default.

#### M5. Helm chart version hardcoded in INSTALLATION_GUIDE.md
`postgres_ai_helm/INSTALLATION_GUIDE.md` references `postgres-ai-monitoring-0.12.tgz` in install and upgrade commands. This will become stale with each release. Consider using a generic placeholder or variable.

#### M6. `docker-compose` vs `docker compose` inconsistency
Some docs use the legacy `docker-compose` (hyphenated) command while others use the modern `docker compose` (space). The CONTRIBUTING.md consistently uses `docker compose`, but `terraform/aws/README.md` and `tests/lock_waits/README.md` use the legacy form.

### LOW

#### L1. Scripts lack inline documentation headers
`scripts/demo.sh` and `scripts/demo-record.sh` have no usage comments or documentation. `scripts/run_reporter_local.sh` has good inline docs and could serve as a template for the others.

#### L2. `CLAUDE.md` check table is minimal
The `.claude/CLAUDE.md` lists only 4 checks (H002, H004, F004, K003) while the README lists 25+ checks. Consider either expanding the CLAUDE.md table or linking to the full list in the README.

#### L3. `pgai/README.md` is minimal but adequate
The `pgai/` wrapper package README is functional but very brief. This is acceptable since it is a thin wrapper.

#### L4. `config/grafana/dashboards/README.md` is a naming convention doc, not a dashboards listing
This is fine for its purpose but a contributor looking for "what dashboards exist" would need to look at the JSON files directly.

#### L5. Recommendation for `wal_keep_segments` in index_pilot README is outdated
`components/index_pilot/README.md` recommends setting `wal_keep_segments` which was renamed to `wal_keep_size` in PostgreSQL 13. Since the tool requires PG 13+, this recommendation should be updated.

#### L6. Missing `scripts/demos/` documentation
The `scripts/demos/` directory exists but its contents and purpose are not documented anywhere.

---

## Discoverability Assessment

**Strengths:**
- Clear top-level README with "Get Started" section
- CONTRIBUTING.md is thorough for local development
- Component docs (index_pilot) have a good docs/ subdirectory structure
- CLI `--help` text is detailed with examples (especially `prepare-db`)
- `cli/README.md` has comprehensive command reference

**Gaps:**
- No top-level docs/ directory linking all documentation together
- No "architecture overview" document for the main project (only index_pilot has one)
- The relationship between components (CLI, reporter, Flask backend, pgwatch, Grafana) is not documented in one place
- Preview environments are documented across three files (CONTRIBUTING.md, TMP_MON_PREVIEWS.md, preview-infra/RUNBOOK.md)

---

## Consistency Check

| Topic | README | CONTRIBUTING | CLI README | CLAUDE.md |
|-------|--------|-------------|------------|-----------|
| Check IDs | 25+ listed | Not listed | Not listed | 4 listed |
| Node.js version | 18+ | "optional" | 18+ | Not mentioned |
| Docker requirement | For monitoring | Yes | Yes | Not mentioned |
| GitLab repo URL | Fixed (was inconsistent) | Not linked | Fixed | Not linked |
| `npx postgresai` usage | Yes | No (uses bare `postgresai`) | Both | Bare `postgresai` |
