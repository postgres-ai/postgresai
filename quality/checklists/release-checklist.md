# Release Checklist

## Pre-Release (1 week before)

### Automated Checks
- [ ] Run `./quality/scripts/release-readiness.sh --full`
- [ ] All CI pipelines green on main
- [ ] Nightly PostgreSQL version matrix passed (PG 14, 15, 16, 17)
- [ ] Performance benchmarks show no regression >5%
- [ ] Coverage meets thresholds (Python ≥70%, CLI ≥60%)
- [ ] Zero critical/high SAST findings

### Manual Verification
- [ ] Fresh `postgresai checkup` against PostgreSQL 15 — valid report generated
- [ ] `postgresai mon local-install --demo` — stack starts, Grafana accessible
- [ ] `postgresai mon targets add` — external target works, metrics flow
- [ ] Auth flow works: `postgresai auth` → login → `show-key` shows masked key
- [ ] Check `prepare-db --print-sql` output is correct for current PG version

### Code Review
- [ ] All merged PRs since last release have been reviewed
- [ ] No TODO/FIXME/HACK comments added without tracking issues
- [ ] CHANGELOG or release notes drafted

## Release Day

- [ ] Tag created: `git tag v<version>`
- [ ] CI publish pipeline triggered and completed
- [ ] npm package published: `npm view postgresai@<version>`
- [ ] Docker images published and pullable
- [ ] Smoke test: `npx postgresai --help` works with published version
- [ ] Release notes published

## Post-Release (24 hours)

- [ ] Monitor error reporting for new issues
- [ ] Verify npm download counts are non-zero
- [ ] Check community channels for immediate feedback
- [ ] Close related issues/milestones

---

## Weekly Quality Rhythm

### Monday — Triage & Review
- Review nightly test failures from the weekend
- Triage new bug reports (assign severity P0-P3)
- Check CI pipeline health (flaky tests? timeouts?)
- Review any security alerts (Dependabot, SAST findings)

### Wednesday — Mid-Week Check
- Are any tests consistently flaky? Root cause and fix or quarantine
- Check coverage trends — any significant drops?
- Review in-progress PRs — any blocked on quality issues?

### Friday — Quality Retrospective (15 min)
- What slipped through this week? (bugs found in main, customer reports)
- Does a new test need to be added?
- Does a CI check need tightening?
- Update failure modes registry if new risks identified
- Celebrate: what quality wins happened this week?

### Monthly — Quality Health Report
- Test coverage trends (up/down/stable?)
- CI pipeline pass rate
- Mean time from bug introduction to detection
- Number of critical failure modes with coverage gaps
- Review and update the Quality Engineering Guide
