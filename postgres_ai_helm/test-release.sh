#!/usr/bin/env bash
#
# Validates helm chart integrity locally before creating a release.
# Simulates the helm-specific steps of the GitLab CI/CD validate pipeline
# (dependency update, lint, template render, package).
# Does not simulate the CI version-bump sed or release artifact creation steps.
# Usage: ./test-release.sh
#

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_FILE=$(mktemp /tmp/helm-test.XXXXXX.log)
PKG_DIR=$(mktemp -d /tmp/helm-test-pkg.XXXXXX)
trap 'rm -f "${LOG_FILE}"; rm -rf "${PKG_DIR}"' EXIT

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

print_error() {
  echo -e "${RED}[ERROR] $1${NC}" >&2
}

print_success() {
  echo -e "${GREEN}[OK] $1${NC}"
}

print_info() {
  echo -e "${BLUE}[INFO] $1${NC}"
}

print_step() {
  echo -e "${YELLOW}[STEP] $1${NC}"
}

run_check() {
  local description="$1"
  shift

  print_step "${description}"

  if "$@" > "${LOG_FILE}" 2>&1; then
    print_success "${description}"
  else
    print_error "${description}"
    echo "  Error output:"
    sed 's/^/    /' "${LOG_FILE}"
    errors=$((errors + 1))
  fi
  echo
}

main() {
  # Track errors
  local errors=0

  echo "=========================================="
  echo "Helm chart release test"
  echo "=========================================="
  echo

  # Check requirements
  print_step "Checking requirements"
  if ! command -v helm &> /dev/null; then
    print_error "helm not found. Please install helm."
    exit 1
  fi
  print_success "helm found: $(helm version --short)"
  echo

  # Extract version
  print_step "Extracting version from Chart.yaml"
  local version
  local app_version
  version=$(grep '^version:' Chart.yaml | awk '{print $2}' | tr -d '"')
  app_version=$(grep '^appVersion:' Chart.yaml | awk '{print $2}' | tr -d '"')
  print_success "Chart version: ${version}"
  print_success "App version: ${app_version}"
  echo

  # Update dependencies
  run_check "Updating helm dependencies" \
    helm dependency update

  # Lint chart
  run_check "Linting helm chart" \
    helm lint .

  # Validate Chart.yaml version format (strict 3-part semver, matching release.sh)
  print_step "Validating Chart.yaml format"
  if [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_success "Version format is valid"
  else
    print_error "Invalid version format: ${version} (must be X.Y.Z)"
    errors=$((errors + 1))
  fi
  echo

  # Test template rendering
  run_check "Rendering templates with default values" \
    helm template test-release .

  # Test with custom values if they exist
  if [ -f custom-values.yaml ]; then
    run_check "Rendering templates with custom values" \
      helm template test-release . -f custom-values.yaml
  fi

  # Package chart into a dedicated temp directory to avoid stale artifact confusion
  print_step "Packaging helm chart"
  local package_file=""
  if helm package . -d "${PKG_DIR}" > "${LOG_FILE}" 2>&1; then
    # Use a glob array instead of ls to avoid parsing ls output
    local pkg_files=("${PKG_DIR}"/postgres-ai-monitoring-*.tgz)
    if [ ${#pkg_files[@]} -gt 0 ] && [ -f "${pkg_files[0]}" ]; then
      package_file="${pkg_files[0]}"
      local package_size
      package_size=$(du -h "${package_file}" | cut -f1)
      print_success "Chart packaged: $(basename "${package_file}") (${package_size})"
    else
      print_error "Package file not found after packaging"
      errors=$((errors + 1))
    fi
  else
    print_error "Failed to package chart"
    cat "${LOG_FILE}"
    errors=$((errors + 1))
  fi
  echo

  # Verify package
  if [ -n "${package_file}" ]; then
    print_step "Verifying packaged chart"
    if helm show chart "${package_file}" > /dev/null 2>&1; then
      print_success "Package is valid"
    else
      print_error "Package is invalid"
      errors=$((errors + 1))
    fi
    echo
  fi

  # Check for common issues
  print_step "Checking for common issues"

  # Hardcoded namespaces
  if grep -r "namespace: " templates/ | grep -v "{{" | grep -v "#" > /dev/null 2>&1; then
    print_info "Warning: Found potentially hardcoded namespaces in templates"
    grep -r "namespace: " templates/ | grep -v "{{" | grep -v "#" | sed 's/^/  /'
  else
    print_success "No hardcoded namespaces found"
  fi

  # Missing required labels
  local label_count
  label_count=$(grep -r "app.kubernetes.io/name" templates/ | wc -l)
  if [ "${label_count}" -gt 0 ]; then
    print_success "Found ${label_count} resources with recommended labels"
  else
    print_info "Warning: No recommended labels found in templates"
  fi

  echo

  # Summary
  echo "=========================================="
  echo "Test summary"
  echo "=========================================="
  echo

  if [ "${errors}" -eq 0 ]; then
    print_success "All tests passed!"
    echo
    print_info "The chart is ready for release."
    echo
    print_info "To create a release:"
    echo "  ./release.sh ${version}"
    echo
    exit 0
  else
    print_error "Found ${errors} error(s)"
    echo
    print_info "Please fix the errors before releasing."
    echo
    exit 1
  fi
}

main "$@"
