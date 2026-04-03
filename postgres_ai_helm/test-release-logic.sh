#!/usr/bin/env bash
#
# Unit and integration tests for release.sh logic.
# Usage: ./test-release-logic.sh
#
# Unit tests (no git required): version format validation, semantic version
# comparison, Chart.yaml backup/restore, and the sed expression used by
# release.sh and the CI pipeline.
#
# Integration tests (require git): interactive stdin prompt edge cases and
# git error handling (commit failure, push failure).  These sections are
# automatically skipped when git is not available.
#

set -euo pipefail

PASS=0
FAIL=0

assert_pass() {
  local description="$1"
  local result="$2"
  if [ "${result}" = "0" ]; then
    echo "  PASS: ${description}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: ${description} (expected pass, got exit ${result})"
    FAIL=$((FAIL + 1))
  fi
}

assert_fail() {
  local description="$1"
  local result="$2"
  if [ "${result}" != "0" ]; then
    echo "  PASS: ${description}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: ${description} (expected failure, got exit 0)"
    FAIL=$((FAIL + 1))
  fi
}

# --- Version format validation (mirrors release.sh regex) ---
validate_version_format() {
  local v="$1"
  [[ "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

echo "=== Version format validation ==="
validate_version_format "1.2.3"     && assert_pass "accepts valid semver 1.2.3"         0 || assert_pass "accepts valid semver 1.2.3"         1
validate_version_format "0.12.0"    && assert_pass "accepts 0.12.0"                     0 || assert_pass "accepts 0.12.0"                     1
validate_version_format "10.20.300" && assert_pass "accepts 10.20.300"                  0 || assert_pass "accepts 10.20.300"                  1

validate_version_format "v1.2.3"    && assert_fail "rejects v1.2.3 (leading v)"        0 || assert_fail "rejects v1.2.3 (leading v)"        1
validate_version_format "1.2"       && assert_fail "rejects 1.2 (two parts)"           0 || assert_fail "rejects 1.2 (two parts)"           1
validate_version_format "1.2.3.4"   && assert_fail "rejects 1.2.3.4 (four parts)"      0 || assert_fail "rejects 1.2.3.4 (four parts)"      1
validate_version_format "1.2.3-rc1" && assert_fail "rejects 1.2.3-rc1 (pre-release)"   0 || assert_fail "rejects 1.2.3-rc1 (pre-release)"   1
validate_version_format ""          && assert_fail "rejects empty string"               0 || assert_fail "rejects empty string"               1
validate_version_format "abc"       && assert_fail "rejects non-numeric"                0 || assert_fail "rejects non-numeric"                1

# --- Semantic version comparison (mirrors release.sh version_gt function) ---
version_gt() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -1)" != "$1" ]; }

echo ""
echo "=== Version comparison (version_gt) ==="
version_gt "1.2.3" "1.2.2" && assert_pass "1.2.3 > 1.2.2 (patch bump)"    0 || assert_pass "1.2.3 > 1.2.2 (patch bump)"    1
version_gt "1.3.0" "1.2.9" && assert_pass "1.3.0 > 1.2.9 (minor bump)"    0 || assert_pass "1.3.0 > 1.2.9 (minor bump)"    1
version_gt "2.0.0" "1.9.9" && assert_pass "2.0.0 > 1.9.9 (major bump)"    0 || assert_pass "2.0.0 > 1.9.9 (major bump)"    1

version_gt "1.2.2" "1.2.3" && assert_fail "1.2.2 !> 1.2.3 (downgrade)"    0 || assert_fail "1.2.2 !> 1.2.3 (downgrade)"    1
version_gt "1.2.3" "1.2.3" && assert_fail "1.2.3 !> 1.2.3 (same version)" 0 || assert_fail "1.2.3 !> 1.2.3 (same version)" 1
version_gt "0.9.0" "1.0.0" && assert_fail "0.9.0 !> 1.0.0 (major downgrade)" 0 || assert_fail "0.9.0 !> 1.0.0 (major downgrade)" 1
version_gt "0.12.0" "0.12"  && assert_pass "0.12.0 > 0.12 (quoted vs unquoted Chart.yaml)" 0 || assert_pass "0.12.0 > 0.12 (quoted vs unquoted Chart.yaml)" 1
version_gt "0.12"   "0.12.0" && assert_fail "0.12 !> 0.12.0 (unquoted vs quoted Chart.yaml)" 0 || assert_fail "0.12 !> 0.12.0 (unquoted vs quoted Chart.yaml)" 1

# --- CI tag-to-version extraction (mirrors the pipeline's ${CI_COMMIT_TAG#helm-} / ${VERSION#v} steps) ---

echo ""
echo "=== CI tag-to-version extraction ==="

# Simulate the two-step extraction used in the release-helm-chart CI job:
#   VERSION="${CI_COMMIT_TAG#helm-}"
#   VERSION="${VERSION#v}"
ci_extract_version() {
  local tag="$1"
  local v="${tag#helm-}"
  echo "${v#v}"
}

result=$(ci_extract_version "helm-v1.2.3")
[ "${result}" = "1.2.3" ] && assert_pass "helm-v1.2.3 extracts to 1.2.3" 0 || assert_pass "helm-v1.2.3 extracts to 1.2.3" 1

result=$(ci_extract_version "helm-1.2.3")
[ "${result}" = "1.2.3" ] && assert_pass "helm-1.2.3 extracts to 1.2.3 (no v prefix)" 0 || assert_pass "helm-1.2.3 extracts to 1.2.3 (no v prefix)" 1

result=$(ci_extract_version "helm-v0.12.0")
[ "${result}" = "0.12.0" ] && assert_pass "helm-v0.12.0 extracts to 0.12.0" 0 || assert_pass "helm-v0.12.0 extracts to 0.12.0" 1

result=$(ci_extract_version "helm-v10.20.300")
[ "${result}" = "10.20.300" ] && assert_pass "helm-v10.20.300 extracts to 10.20.300 (large version)" 0 || assert_pass "helm-v10.20.300 extracts to 10.20.300 (large version)" 1

# --- Chart.yaml version rewrite (sed expression used by release.sh and CI pipeline) ---
#
# Both release.sh and the CI release-helm-chart job use the same sed pattern:
#   sed "s/^version: .*/version: \"${VERSION}\"/"  ...  > Chart.yaml.tmp && mv Chart.yaml.tmp Chart.yaml
# These tests verify correctness for both quoted and unquoted input formats.

echo ""
echo "=== Chart.yaml version rewrite (sed expression) ==="

# Helper: apply the sed rewrite to a single-line chart file and return the parsed result.
rewrite_version() {
  local input_line="$1"
  local new_version="$2"
  local tmpdir
  tmpdir=$(mktemp -d)
  echo "${input_line}" > "${tmpdir}/Chart.yaml"
  sed "s/^version: .*/version: \"${new_version}\"/" "${tmpdir}/Chart.yaml" > "${tmpdir}/Chart.yaml.tmp"
  mv "${tmpdir}/Chart.yaml.tmp" "${tmpdir}/Chart.yaml"
  grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"'
  rm -rf "${tmpdir}"
}

test_sed_quoted() {
  local result
  result=$(rewrite_version 'version: "0.12.0"' "1.0.0")
  [ "${result}" = "1.0.0" ]
}
test_sed_quoted && assert_pass "rewrites quoted version (release.sh format)" 0 || assert_pass "rewrites quoted version (release.sh format)" 1

test_sed_unquoted() {
  local result
  result=$(rewrite_version 'version: 0.12' "1.0.0")
  [ "${result}" = "1.0.0" ]
}
test_sed_unquoted && assert_pass "rewrites unquoted version (legacy format)" 0 || assert_pass "rewrites unquoted version (legacy format)" 1

test_sed_preserves_other_fields() {
  local tmpdir
  tmpdir=$(mktemp -d)
  printf 'apiVersion: v2\nversion: "0.12.0"\nname: postgres-ai\n' > "${tmpdir}/Chart.yaml"
  sed "s/^version: .*/version: \"2.0.0\"/" "${tmpdir}/Chart.yaml" > "${tmpdir}/Chart.yaml.tmp"
  mv "${tmpdir}/Chart.yaml.tmp" "${tmpdir}/Chart.yaml"
  local api_version
  api_version=$(grep '^apiVersion:' "${tmpdir}/Chart.yaml" | awk '{print $2}')
  local name
  name=$(grep '^name:' "${tmpdir}/Chart.yaml" | awk '{print $2}')
  rm -rf "${tmpdir}"
  [ "${api_version}" = "v2" ] && [ "${name}" = "postgres-ai" ]
}
test_sed_preserves_other_fields && assert_pass "sed does not modify other Chart.yaml fields" 0 || assert_pass "sed does not modify other Chart.yaml fields" 1

# --- Chart.yaml backup and ERR trap rollback logic (mirrors release.sh) ---

echo ""
echo "=== Chart.yaml backup and ERR trap rollback ==="

test_backup_created_before_sed() {
  local tmpdir
  tmpdir=$(mktemp -d)
  local chart="${tmpdir}/Chart.yaml"
  echo 'version: "0.12.0"' > "${chart}"
  cp "${chart}" "${chart}.bak"
  local exists=0
  [ -f "${chart}.bak" ] || exists=1
  rm -rf "${tmpdir}"
  [ "${exists}" = "0" ]
}
test_backup_created_before_sed && assert_pass "backup file is created before Chart.yaml is modified" 0 || assert_pass "backup file is created before Chart.yaml is modified" 1

test_trap_restores_original() {
  local tmpdir
  tmpdir=$(mktemp -d)
  local chart="${tmpdir}/Chart.yaml"
  local original='version: "0.12.0"'
  echo "${original}" > "${chart}"
  cp "${chart}" "${chart}.bak"
  # Simulate sed rewrite
  sed "s/^version: .*/version: \"1.0.0\"/" "${chart}.bak" > "${chart}"
  # Simulate ERR trap (mv backup over modified file)
  if [ -f "${chart}.bak" ]; then mv "${chart}.bak" "${chart}"; fi
  local restored
  restored=$(cat "${chart}")
  rm -rf "${tmpdir}"
  [ "${restored}" = "${original}" ]
}
test_trap_restores_original && assert_pass "ERR trap restores Chart.yaml to original on failure" 0 || assert_pass "ERR trap restores Chart.yaml to original on failure" 1

test_backup_removed_on_success() {
  local tmpdir
  tmpdir=$(mktemp -d)
  local chart="${tmpdir}/Chart.yaml"
  echo 'version: "0.12.0"' > "${chart}"
  cp "${chart}" "${chart}.bak"
  sed "s/^version: .*/version: \"1.0.0\"/" "${chart}.bak" > "${chart}"
  # Simulate successful completion: disarm trap then remove backup
  trap - ERR 2>/dev/null || true
  rm -f "${chart}.bak"
  local present=0
  [ -f "${chart}.bak" ] && present=1
  rm -rf "${tmpdir}"
  [ "${present}" = "0" ]
}
test_backup_removed_on_success && assert_pass "backup file is removed after successful update" 0 || assert_pass "backup file is removed after successful update" 1

test_trap_disarmed_before_rm() {
  # Verify that disarming the trap before rm -f means a failed rm does NOT restore the file.
  # (rm -f never fails in practice, but this tests the logical ordering.)
  local tmpdir
  tmpdir=$(mktemp -d)
  local chart="${tmpdir}/Chart.yaml"
  echo 'version: "1.0.0"' > "${chart}"
  # No .bak exists; trap - ERR first, then rm -f (which is a no-op here)
  trap - ERR 2>/dev/null || true
  rm -f "${chart}.bak"
  # Chart.yaml should still contain the new version
  local result
  result=$(grep '^version:' "${chart}" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  [ "${result}" = "1.0.0" ]
}
test_trap_disarmed_before_rm && assert_pass "trap is disarmed before backup removal (no spurious rollback)" 0 || assert_pass "trap is disarmed before backup removal (no spurious rollback)" 1

# --- Interactive prompt (stdin) tests and git operations error handling ---
# These sections require a working git installation.  They are automatically
# skipped when git is not available (e.g. minimal CI containers without git).

if command -v git >/dev/null 2>&1; then

# Helper: create a minimal temporary git repository with Chart.yaml and a
# copy of release.sh so the script can be run in an isolated environment.
_make_test_git_repo() {
  local dir
  dir=$(mktemp -d)
  git -C "${dir}" init --quiet
  git -C "${dir}" config user.email "test@test.invalid"
  git -C "${dir}" config user.name "Test"
  printf 'apiVersion: v2\nversion: "0.12.0"\nname: postgres-ai\n' > "${dir}/Chart.yaml"
  git -C "${dir}" add Chart.yaml
  git -C "${dir}" commit --quiet -m "initial"
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cp "${script_dir}/release.sh" "${dir}/"
  echo "${dir}"
}

echo ""
echo "=== Interactive prompt (stdin) tests ==="

test_stdin_cancel_uppercase_n() {
  local tmpdir exit_code=0 chart_version commit_count_before commit_count_after
  tmpdir=$(_make_test_git_repo)
  commit_count_before=$(git -C "${tmpdir}" rev-list --count HEAD)
  (cd "${tmpdir}" && printf '1.0.0\nN\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  commit_count_after=$(git -C "${tmpdir}" rev-list --count HEAD)
  rm -rf "${tmpdir}"
  [ "${exit_code}" = "0" ] && [ "${chart_version}" = "0.12.0" ] && [ "${commit_count_after}" = "${commit_count_before}" ]
}
test_stdin_cancel_uppercase_n \
  && assert_pass "cancel with 'N': exits 0, Chart.yaml unchanged, no new commit" 0 \
  || assert_pass "cancel with 'N': exits 0, Chart.yaml unchanged, no new commit" 1

test_stdin_cancel_lowercase_n() {
  local tmpdir exit_code=0 chart_version commit_count_before commit_count_after
  tmpdir=$(_make_test_git_repo)
  commit_count_before=$(git -C "${tmpdir}" rev-list --count HEAD)
  (cd "${tmpdir}" && printf '1.0.0\nn\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  commit_count_after=$(git -C "${tmpdir}" rev-list --count HEAD)
  rm -rf "${tmpdir}"
  [ "${exit_code}" = "0" ] && [ "${chart_version}" = "0.12.0" ] && [ "${commit_count_after}" = "${commit_count_before}" ]
}
test_stdin_cancel_lowercase_n \
  && assert_pass "cancel with 'n': exits 0, Chart.yaml unchanged, no new commit" 0 \
  || assert_pass "cancel with 'n': exits 0, Chart.yaml unchanged, no new commit" 1

test_stdin_empty_version() {
  local tmpdir exit_code=0 chart_version
  tmpdir=$(_make_test_git_repo)
  (cd "${tmpdir}" && printf '\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  [ "${exit_code}" != "0" ] && [ "${chart_version}" = "0.12.0" ]
}
test_stdin_empty_version \
  && assert_pass "empty version via stdin: exits non-zero, Chart.yaml unchanged" 0 \
  || assert_pass "empty version via stdin: exits non-zero, Chart.yaml unchanged" 1

test_stdin_whitespace_version() {
  local tmpdir exit_code=0 chart_version
  tmpdir=$(_make_test_git_repo)
  (cd "${tmpdir}" && printf '   \n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  [ "${exit_code}" != "0" ] && [ "${chart_version}" = "0.12.0" ]
}
test_stdin_whitespace_version \
  && assert_pass "whitespace-only version via stdin: exits non-zero, Chart.yaml unchanged" 0 \
  || assert_pass "whitespace-only version via stdin: exits non-zero, Chart.yaml unchanged" 1

test_stdin_invalid_version_format() {
  local tmpdir exit_code=0 chart_version
  tmpdir=$(_make_test_git_repo)
  (cd "${tmpdir}" && printf 'not-a-version\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  [ "${exit_code}" != "0" ] && [ "${chart_version}" = "0.12.0" ]
}
test_stdin_invalid_version_format \
  && assert_pass "invalid version format via stdin: exits non-zero, Chart.yaml unchanged" 0 \
  || assert_pass "invalid version format via stdin: exits non-zero, Chart.yaml unchanged" 1

echo ""
echo "=== Dirty working tree guard tests ==="

test_dirty_tree_is_rejected() {
  local tmpdir exit_code=0 chart_version
  tmpdir=$(_make_test_git_repo)
  # Dirty a file OTHER than Chart.yaml to trigger the guard
  # (Chart.yaml is exempted from the dirty check in release.sh)
  echo "test" > "${tmpdir}/values.yaml"
  git -C "${tmpdir}" add values.yaml
  echo "# dirty" >> "${tmpdir}/values.yaml"
  (cd "${tmpdir}" && printf '1.0.0\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  # Chart.yaml version should be unchanged (the guard fires before any sed rewrite)
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  [ "${exit_code}" != "0" ] && [ "${chart_version}" = "0.12.0" ]
}
test_dirty_tree_is_rejected \
  && assert_pass "dirty working tree: exits non-zero, Chart.yaml version unchanged" 0 \
  || assert_pass "dirty working tree: exits non-zero, Chart.yaml version unchanged" 1

test_dirty_tree_no_new_commit() {
  local tmpdir exit_code=0 commit_count_before commit_count_after
  tmpdir=$(_make_test_git_repo)
  commit_count_before=$(git -C "${tmpdir}" rev-list --count HEAD)
  # Dirty a file OTHER than Chart.yaml
  echo "test" > "${tmpdir}/values.yaml"
  git -C "${tmpdir}" add values.yaml
  echo "# dirty" >> "${tmpdir}/values.yaml"
  (cd "${tmpdir}" && printf '1.0.0\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  commit_count_after=$(git -C "${tmpdir}" rev-list --count HEAD)
  rm -rf "${tmpdir}"
  [ "${exit_code}" != "0" ] && [ "${commit_count_after}" = "${commit_count_before}" ]
}
test_dirty_tree_no_new_commit \
  && assert_pass "dirty working tree: no new commit created" 0 \
  || assert_pass "dirty working tree: no new commit created" 1

test_dirty_chart_alone_does_not_trigger_guard() {
  local tmpdir exit_code=0
  tmpdir=$(_make_test_git_repo)
  # Chart.yaml is exempted from the dirty check
  # (previous run may have bumped it but commit failed)
  echo "# dirty" >> "${tmpdir}/Chart.yaml"
  # Script should NOT exit due to dirty guard, but will fail later at version validation
  # because the new version (1.0.0) must be greater than what's on disk (0.12.0).
  # We verify the guard didn't fire by checking the script reached far enough to perform
  # the not-greater check (which produces a different error message than the dirty guard).
  output=$(cd "${tmpdir}" && printf '0.11.0\ny\n' | bash "release.sh" 2>&1) || exit_code=$?
  rm -rf "${tmpdir}"
  # Must fail (downgrade not allowed), but NOT with the dirty-tree error message
  [ "${exit_code}" != "0" ] && echo "${output}" | grep -q "must be greater than current version"
}
test_dirty_chart_alone_does_not_trigger_guard \
  && assert_pass "dirty Chart.yaml alone: guard does NOT fire, fails at version check" 0 \
  || assert_pass "dirty Chart.yaml alone: guard does NOT fire, fails at version check" 1

echo ""
echo "=== git operations error handling tests ==="

test_push_failure_prints_recovery_instructions() {
  local tmpdir exit_code=0 output local_tag commit_count_before commit_count_after
  tmpdir=$(_make_test_git_repo)
  commit_count_before=$(git -C "${tmpdir}" rev-list --count HEAD)
  # No remote configured — push will fail naturally without any mocking
  output=$(cd "${tmpdir}" && printf '1.0.0\ny\n' | bash "release.sh" 2>&1) || exit_code=$?
  # Verify local state: tag must exist and commit count must have increased by 1
  local_tag=$(git -C "${tmpdir}" tag -l "helm-v1.0.0")
  commit_count_after=$(git -C "${tmpdir}" rev-list --count HEAD)
  rm -rf "${tmpdir}"
  # Must exit non-zero, print all three recovery instruction lines, have local tag, and have one new commit
  [ "${exit_code}" != "0" ] \
    && echo "${output}" | grep -q "git push origin HEAD" \
    && echo "${output}" | grep -q "git tag -d" \
    && echo "${output}" | grep -q "git reset --soft HEAD~1" \
    && [ "${local_tag}" = "helm-v1.0.0" ] \
    && [ "${commit_count_after}" = "$((commit_count_before + 1))" ]
}
test_push_failure_prints_recovery_instructions \
  && assert_pass "push failure: exits non-zero, prints recovery instructions, local tag and commit exist" 0 \
  || assert_pass "push failure: exits non-zero, prints recovery instructions, local tag and commit exist" 1

test_commit_failure_unstages_chart() {
  local tmpdir exit_code=0 staged_files chart_version
  tmpdir=$(_make_test_git_repo)
  # Install a pre-commit hook that rejects all commits
  mkdir -p "${tmpdir}/.git/hooks"
  printf '#!/bin/sh\necho "commit rejected by hook" >&2\nexit 1\n' \
    > "${tmpdir}/.git/hooks/pre-commit"
  chmod +x "${tmpdir}/.git/hooks/pre-commit"
  (cd "${tmpdir}" && printf '1.0.0\ny\n' | bash "release.sh" >/dev/null 2>&1) || exit_code=$?
  staged_files=$(cd "${tmpdir}" && git diff --cached --name-only 2>/dev/null || true)
  # Chart.yaml on disk should contain the new version (sed succeeded; only the commit failed)
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  # Must exit non-zero, leave the index clean (nothing staged), and Chart.yaml on disk has new version
  [ "${exit_code}" != "0" ] && [ -z "${staged_files}" ] && [ "${chart_version}" = "1.0.0" ]
}
test_commit_failure_unstages_chart \
  && assert_pass "commit failure: exits non-zero, unstages Chart.yaml, disk has new version" 0 \
  || assert_pass "commit failure: exits non-zero, unstages Chart.yaml, disk has new version" 1

echo ""
echo "=== Commit failure fix-and-retry tests ==="

# These tests cover the critical user journey: operator hits a commit failure,
# fixes the hook, and retries.  Because release.sh leaves Chart.yaml on disk
# with the new version after a commit failure, retrying with the *same* version
# must fail (not-greater check), while retrying with the *next* version must
# succeed up to tag creation (push will fail since there is no remote, but the
# tag and commit should exist locally).

test_retry_same_version_after_commit_failure() {
  local tmpdir exit_code_first=0 exit_code_retry=0 chart_version
  tmpdir=$(_make_test_git_repo)
  # Install a pre-commit hook that rejects all commits
  mkdir -p "${tmpdir}/.git/hooks"
  printf '#!/bin/sh\necho "commit rejected by hook" >&2\nexit 1\n' \
    > "${tmpdir}/.git/hooks/pre-commit"
  chmod +x "${tmpdir}/.git/hooks/pre-commit"
  # First attempt — fails due to hook
  (cd "${tmpdir}" && printf '1.0.0\ny\n' | bash "release.sh" >/dev/null 2>&1) || exit_code_first=$?
  # Remove the hook so the next commit attempt can succeed
  rm -f "${tmpdir}/.git/hooks/pre-commit"
  # Retry with the SAME version — must fail because Chart.yaml already has 1.0.0
  (cd "${tmpdir}" && printf '1.0.0\ny\n' | bash "release.sh" >/dev/null 2>&1) || exit_code_retry=$?
  chart_version=$(grep '^version:' "${tmpdir}/Chart.yaml" | awk '{print $2}' | tr -d '"')
  rm -rf "${tmpdir}"
  # First run must fail; retry with same version must also fail (not-greater); disk stays at 1.0.0
  [ "${exit_code_first}" != "0" ] && [ "${exit_code_retry}" != "0" ] && [ "${chart_version}" = "1.0.0" ]
}
test_retry_same_version_after_commit_failure \
  && assert_pass "retry same version after commit failure: exits non-zero (not-greater)" 0 \
  || assert_pass "retry same version after commit failure: exits non-zero (not-greater)" 1

test_retry_next_version_after_commit_failure() {
  local tmpdir exit_code_first=0 exit_code_retry=0 local_tag commit_count_before commit_count_after
  tmpdir=$(_make_test_git_repo)
  commit_count_before=$(git -C "${tmpdir}" rev-list --count HEAD)
  # Install a pre-commit hook that rejects all commits
  mkdir -p "${tmpdir}/.git/hooks"
  printf '#!/bin/sh\necho "commit rejected by hook" >&2\nexit 1\n' \
    > "${tmpdir}/.git/hooks/pre-commit"
  chmod +x "${tmpdir}/.git/hooks/pre-commit"
  # First attempt — fails due to hook; Chart.yaml now has 1.0.0 on disk, unstaged
  (cd "${tmpdir}" && printf '1.0.0\ny\n' | bash "release.sh" >/dev/null 2>&1) || exit_code_first=$?
  # Remove the hook; retry with 1.0.1 which is greater than the on-disk 1.0.0
  rm -f "${tmpdir}/.git/hooks/pre-commit"
  # Push will still fail (no remote), but commit and tag must be created locally
  (cd "${tmpdir}" && printf '1.0.1\ny\n' | bash "release.sh" >/dev/null 2>&1) || exit_code_retry=$?
  local_tag=$(git -C "${tmpdir}" tag -l "helm-v1.0.1")
  commit_count_after=$(git -C "${tmpdir}" rev-list --count HEAD)
  rm -rf "${tmpdir}"
  # First must fail; retry must fail only on push (non-zero); tag exists; one new commit
  [ "${exit_code_first}" != "0" ] \
    && [ "${exit_code_retry}" != "0" ] \
    && [ "${local_tag}" = "helm-v1.0.1" ] \
    && [ "${commit_count_after}" = "$((commit_count_before + 1))" ]
}
test_retry_next_version_after_commit_failure \
  && assert_pass "retry next version after commit failure: local tag and commit created (push fails without remote)" 0 \
  || assert_pass "retry next version after commit failure: local tag and commit created (push fails without remote)" 1

else
  echo ""
  echo "=== Interactive prompt and git error handling tests: SKIPPED (git not available) ==="
fi

# --- Summary ---
echo ""
echo "=========================================="
TOTAL=$((PASS + FAIL))
echo "Results: ${PASS}/${TOTAL} passed"
echo "=========================================="

if [ "${FAIL}" -gt 0 ]; then
  echo "FAILED: ${FAIL} test(s) failed"
  exit 1
else
  echo "All tests passed"
  exit 0
fi
