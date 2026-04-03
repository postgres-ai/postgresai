#!/usr/bin/env bash
#
# Helm chart release script
# Creates a git tag to trigger automated helm chart release via GitLab CI/CD
# Usage: ./release.sh <version>   e.g.: ./release.sh 0.14.0
#

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_FILE="${SCRIPT_DIR}/Chart.yaml"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Print functions
print_error() {
  echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
  echo -e "${GREEN}$1${NC}"
}

print_info() {
  echo -e "${YELLOW}$1${NC}"
}

# Returns true (0) if $1 is strictly greater than $2 (semver comparison)
version_gt() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -1)" != "$1" ]; }

main() {
  # Check if we're in a git repository
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
  fi

  # Check for uncommitted changes, but allow Chart.yaml to be dirty.
  # A previous release.sh run that failed at the commit stage (e.g. rejected by a
  # pre-commit hook) leaves Chart.yaml on disk with the bumped version but does not
  # commit it.  Blocking on that specific file would prevent the operator from simply
  # fixing the hook and re-running the script with the next version.
  local _chart_relpath _dirty_except_chart
  _chart_relpath=$(git ls-files --full-name "${CHART_FILE}" 2>/dev/null || basename "${CHART_FILE}")
  _dirty_except_chart=$(git diff-index --name-only HEAD -- | grep -Fxv "${_chart_relpath}" | head -1 || true)
  if [ -n "${_dirty_except_chart}" ]; then
    print_error "You have uncommitted changes. Please commit or stash them first."
    exit 1
  fi

  # Get current version from Chart.yaml
  local current_version
  current_version=$(grep '^version:' "${CHART_FILE}" | awk '{print $2}' | tr -d '"')
  if [ -z "${current_version}" ]; then
    print_error "Cannot read version from Chart.yaml — 'version:' field missing or blank"
    exit 1
  fi
  print_info "Current helm chart version: ${current_version}"

  # Get version argument or ask for it
  local new_version
  if [ $# -eq 0 ]; then
    echo
    echo "Enter new version (e.g., 0.14.0):"
    read -r new_version
  else
    new_version="$1"
  fi

  # Validate version format (semantic versioning)
  if ! [[ "${new_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format. Use semantic versioning (e.g., 0.14.0)"
    exit 1
  fi

  # Check if new version is strictly greater than current (prevents same version and downgrades)
  if ! version_gt "${new_version}" "${current_version}"; then
    print_error "New version (${new_version}) must be greater than current version (${current_version})"
    exit 1
  fi

  # Confirm the release
  echo
  print_info "This will:"
  print_info "  1. Update Chart.yaml to version ${new_version}"
  print_info "  2. Commit the change"
  print_info "  3. Create and push tag 'helm-v${new_version}'"
  print_info "  4. Trigger automated GitLab CI/CD pipeline to:"
  print_info "     - Package the helm chart"
  print_info "     - Create a GitLab release"
  print_info "     - Attach the packaged chart"
  echo
  echo "Continue? (y/N)"
  read -r confirm

  if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    print_info "Release cancelled"
    exit 0
  fi

  # Update Chart.yaml (only chart version; appVersion tracks the deployed app, not the chart)
  print_info "Updating Chart.yaml..."
  cp "${CHART_FILE}" "${CHART_FILE}.bak"
  # ERR trap covers the file-write phase only; disarmed after sed succeeds, before any git
  # operations. Keeping the trap active through git would cause a misleading Chart.yaml
  # rollback even when the file content has already been committed.
  trap 'trap - ERR; if [ -f "${CHART_FILE}.bak" ]; then mv "${CHART_FILE}.bak" "${CHART_FILE}"; print_error "Rolled back Chart.yaml"; fi' ERR
  sed "s/^version: .*/version: \"${new_version}\"/" "${CHART_FILE}.bak" > "${CHART_FILE}"
  # sed succeeded — disarm file-write trap and clean up backup before git operations
  trap - ERR
  rm -f "${CHART_FILE}.bak"

  # Verify the substitution succeeded
  if ! grep -q "^version: \"${new_version}\"" "${CHART_FILE}"; then
    print_error "Failed to update version in Chart.yaml (pattern may not match)"
    exit 1
  fi

  # Show the diff
  print_info "Changes to Chart.yaml:"
  git diff "${CHART_FILE}"

  # Stage and commit; if commit fails (e.g. pre-commit hook), unstage the file so the working
  # tree is left clean and the operator can fix the issue before re-running.
  print_info "Committing changes..."
  git add "${CHART_FILE}"
  if ! git commit -m "chore(helm): bump chart version to ${new_version}"; then
    git reset HEAD "${CHART_FILE}" 2>/dev/null || true
    print_error "Commit failed. Chart.yaml has been updated on disk but the change was unstaged."
    print_info "Fix the commit issue, then re-run this script to retry."
    exit 1
  fi

  # Create and push commit + tag atomically.
  # If push fails, local commit and tag exist but are not on the remote; recovery instructions
  # are printed so the operator can push manually or clean up.
  local tag_name="helm-v${new_version}"
  print_info "Creating tag ${tag_name}..."
  git tag -a "${tag_name}" -m "Helm chart release ${new_version}"

  print_info "Pushing commit and tag..."
  if ! git push origin HEAD "${tag_name}"; then
    print_error "Push failed. Local commit and tag '${tag_name}' exist but were not pushed to remote."
    print_info "To retry the push:   git push origin HEAD ${tag_name}"
    print_info "To undo local state: git tag -d ${tag_name} && git reset --soft HEAD~1"
    exit 1
  fi

  echo
  print_success "Release initiated successfully!"
  print_success "Tag ${tag_name} has been pushed."
  echo
  print_info "GitLab CI/CD will now:"
  print_info "  - Package the helm chart"
  print_info "  - Create a release at: https://gitlab.com/postgres-ai/postgresai/-/releases/${tag_name}"
  echo
  print_info "Monitor the pipeline at:"
  print_info "  https://gitlab.com/postgres-ai/postgresai/-/pipelines"
}

main "$@"
