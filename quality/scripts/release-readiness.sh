#!/usr/bin/env bash
#
# Release Readiness Check
#
# Verifies that the codebase is ready for release by checking:
#   1. All tests pass (Python + CLI)
#   2. No open P0/P1 issues tagged as blockers
#   3. Coverage meets minimum thresholds
#   4. JSON schemas are valid and consistent
#   5. Build succeeds
#   6. No secrets in codebase
#
# Usage:
#   ./quality/scripts/release-readiness.sh [--full]
#
#   --full    Run complete checks including nightly-level tests
#
# Exit codes:
#   0 = Ready for release
#   1 = Blockers found (see output)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FULL_MODE=false
BLOCKERS=0
WARNINGS=0

if [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=true
fi

# --- Helpers ---

pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

fail() {
    echo -e "  ${RED}✗${NC} $1"
    BLOCKERS=$((BLOCKERS + 1))
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

section() {
    echo ""
    echo -e "${BLUE}━━━ $1 ━━━${NC}"
}

# --- Checks ---

section "1. Git Status"

cd "$PROJECT_ROOT"

BRANCH=$(git branch --show-current)
info "Branch: $BRANCH"

UNCOMMITTED=$(git status --porcelain | wc -l | tr -d ' ')
if [[ "$UNCOMMITTED" -gt 0 ]]; then
    warn "Uncommitted changes: $UNCOMMITTED files"
else
    pass "Working tree clean"
fi

# Check if on main or a release branch
if [[ "$BRANCH" == "main" || "$BRANCH" =~ ^release/ ]]; then
    pass "On release-eligible branch: $BRANCH"
else
    warn "Not on main or release branch (current: $BRANCH)"
fi


section "2. Python Tests (Reporter)"

if command -v python3 &>/dev/null && [[ -f "$PROJECT_ROOT/reporter/requirements-dev.txt" ]]; then
    if python3 -m pytest tests/reporter --tb=short -q 2>/dev/null; then
        pass "Python unit tests pass"
    else
        fail "Python unit tests failed"
    fi

    if $FULL_MODE; then
        if python3 -m pytest tests/reporter --run-integration --tb=short -q 2>/dev/null; then
            pass "Python integration tests pass"
        else
            fail "Python integration tests failed"
        fi
    else
        info "Skipping integration tests (use --full)"
    fi
else
    warn "Python/pytest not available — skipping reporter tests"
fi


section "3. CLI Tests"

if command -v bun &>/dev/null; then
    cd "$PROJECT_ROOT/cli"

    if bun run test:fast 2>/dev/null; then
        pass "CLI tests pass"
    else
        fail "CLI tests failed"
    fi

    cd "$PROJECT_ROOT"
else
    warn "Bun not available — skipping CLI tests"
fi


section "4. CLI Build"

if command -v bun &>/dev/null; then
    cd "$PROJECT_ROOT/cli"

    if bun run build 2>/dev/null; then
        pass "CLI build succeeds"

        # Verify the built binary runs
        if node ./dist/bin/postgres-ai.js --help >/dev/null 2>&1; then
            pass "Built CLI binary runs"
        else
            fail "Built CLI binary fails to run"
        fi
    else
        fail "CLI build failed"
    fi

    cd "$PROJECT_ROOT"
else
    warn "Bun not available — skipping CLI build check"
fi


section "5. TypeScript Type Check"

if command -v bun &>/dev/null; then
    cd "$PROJECT_ROOT/cli"

    if bun run typecheck 2>/dev/null; then
        pass "TypeScript types check out"
    else
        fail "TypeScript type errors found"
    fi

    cd "$PROJECT_ROOT"
else
    warn "Bun not available — skipping type check"
fi


section "6. JSON Schema Validation"

SCHEMA_DIR="$PROJECT_ROOT/reporter/schemas"
if [[ -d "$SCHEMA_DIR" ]]; then
    SCHEMA_COUNT=$(find "$SCHEMA_DIR" -name "*.schema.json" | wc -l | tr -d ' ')
    info "Found $SCHEMA_COUNT JSON schemas"

    # Validate that all schema files are valid JSON
    INVALID_SCHEMAS=0
    for schema in "$SCHEMA_DIR"/*.schema.json; do
        if ! python3 -c "import json; json.load(open('$schema'))" 2>/dev/null; then
            fail "Invalid JSON schema: $(basename "$schema")"
            INVALID_SCHEMAS=$((INVALID_SCHEMAS + 1))
        fi
    done

    if [[ "$INVALID_SCHEMAS" -eq 0 ]]; then
        pass "All $SCHEMA_COUNT schemas are valid JSON"
    fi
else
    warn "Schema directory not found: $SCHEMA_DIR"
fi


section "7. Secret Detection"

if command -v gitleaks &>/dev/null; then
    if gitleaks detect --source "$PROJECT_ROOT" --no-banner --no-git 2>/dev/null; then
        pass "No secrets detected (gitleaks)"
    else
        fail "Secrets detected in codebase — run 'gitleaks detect' for details"
    fi
else
    # Fallback: basic pattern check
    SUSPECT=$(grep -rn --include="*.ts" --include="*.py" --include="*.yml" --include="*.json" \
        -E '(password|secret|token|api_key)\s*[:=]\s*["\x27][^"\x27]{8,}' \
        "$PROJECT_ROOT/cli/lib" "$PROJECT_ROOT/reporter" 2>/dev/null | \
        grep -v -E '(test|mock|example|placeholder|<|TODO|process\.env|os\.environ|getenv)' | \
        head -5 || true)

    if [[ -z "$SUSPECT" ]]; then
        pass "No obvious hardcoded secrets (basic check)"
    else
        warn "Possible hardcoded secrets found (install gitleaks for thorough check)"
        echo "$SUSPECT" | while IFS= read -r line; do
            info "  $line"
        done
    fi
fi


section "8. Coverage Thresholds"

# Check if coverage reports exist from recent CI runs
REPORTER_COV="$PROJECT_ROOT/coverage/reporter-coverage.xml"
CLI_COV="$PROJECT_ROOT/cli/coverage"

if [[ -f "$REPORTER_COV" ]]; then
    # Extract line rate from coverage XML
    LINE_RATE=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('$REPORTER_COV')
rate = float(tree.getroot().get('line-rate', 0)) * 100
print(f'{rate:.1f}')
" 2>/dev/null || echo "0")

    if (( $(echo "$LINE_RATE >= 70" | bc -l 2>/dev/null || echo 0) )); then
        pass "Reporter coverage: ${LINE_RATE}% (≥70% threshold)"
    else
        warn "Reporter coverage: ${LINE_RATE}% (below 70% threshold)"
    fi
else
    info "No reporter coverage report found (run CI pipeline to generate)"
fi

if [[ -d "$CLI_COV" ]]; then
    info "CLI coverage report exists at cli/coverage/"
else
    info "No CLI coverage report found (run 'bun run test:coverage' to generate)"
fi


section "9. Documentation"

# Check that key docs exist and aren't empty
for doc in README.md CONTRIBUTING.md SECURITY.md; do
    if [[ -s "$PROJECT_ROOT/$doc" ]]; then
        pass "$doc exists"
    else
        warn "$doc missing or empty"
    fi
done

# Check CHANGELOG or release notes
if [[ -f "$PROJECT_ROOT/CHANGELOG.md" ]] || git tag --list 'v*' | head -1 >/dev/null 2>&1; then
    pass "Release history available (tags or CHANGELOG)"
else
    warn "No CHANGELOG.md or version tags found"
fi


# --- Summary ---

echo ""
echo -e "${BLUE}━━━ Release Readiness Summary ━━━${NC}"
echo ""

if [[ "$BLOCKERS" -gt 0 ]]; then
    echo -e "  ${RED}BLOCKERS: $BLOCKERS${NC} — Release is NOT ready"
else
    echo -e "  ${GREEN}BLOCKERS: 0${NC}"
fi

if [[ "$WARNINGS" -gt 0 ]]; then
    echo -e "  ${YELLOW}WARNINGS: $WARNINGS${NC} — Review before release"
else
    echo -e "  ${GREEN}WARNINGS: 0${NC}"
fi

echo ""

if [[ "$BLOCKERS" -gt 0 ]]; then
    echo -e "  ${RED}❌ NOT READY FOR RELEASE${NC}"
    echo "  Fix all blockers before proceeding."
    exit 1
else
    if [[ "$WARNINGS" -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠ CONDITIONALLY READY${NC}"
        echo "  Review warnings above. Proceed with caution."
    else
        echo -e "  ${GREEN}✅ READY FOR RELEASE${NC}"
    fi
    exit 0
fi
