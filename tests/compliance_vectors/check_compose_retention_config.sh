#!/usr/bin/env bash
# Verify Docker Compose renders monitoring retention knobs without dropping VM safety flags.
set -Eeuo pipefail
IFS=$'\n\t'

extract_vm_command() {
  jq -er '.services["sink-prometheus"].command | if type == "array" then join(" ") else . end'
}

extract_vm_retention() {
  jq -er '.services["sink-prometheus"].environment.VM_RETENTION_PERIOD'
}

extract_queryid_retention() {
  jq -er '.services.monitoring_flask_backend.environment.QUERYID_RETENTION_HOURS'
}

render_compose_config() {
  env \
    PGAI_TAG=0.14.0 \
    REPLICATOR_PASSWORD=test-replicator-password \
    VM_AUTH_USERNAME=vmauth \
    VM_AUTH_PASSWORD=test-vm-password \
    "$@" \
    docker compose config --format json
}

assert_vm_flag_present() {
  local label="$1"
  local expected_flag="$2"
  local config_json="$3"
  local vm_command

  vm_command="$(extract_vm_command <<< "$config_json")"
  if [[ "$vm_command" != *"$expected_flag"* ]]; then
    printf '%s\n' "FAIL: $label VM command missing $expected_flag" >&2
    exit 1
  fi
}

assert_vm_safety_flags() {
  local label="$1"
  local config_json="$2"

  assert_vm_flag_present "$label" '-retentionPeriod="$$VM_RETENTION_PERIOD"' "$config_json"
  assert_vm_flag_present "$label" "-promscrape.maxScrapeSize=128000000" "$config_json"
  assert_vm_flag_present "$label" "-search.maxQueryDuration=30s" "$config_json"
  assert_vm_flag_present "$label" "-search.maxConcurrentRequests=16" "$config_json"
}

assert_retention_config() {
  local label="$1"
  local expected_vm_retention="$2"
  local expected_queryid_retention="$3"
  local config_json="$4"
  local vm_retention
  local queryid_retention

  vm_retention="$(extract_vm_retention <<< "$config_json")"
  if [[ "$vm_retention" != "$expected_vm_retention" ]]; then
    printf '%s\n' "FAIL: $label VM retention expected $expected_vm_retention, got $vm_retention" >&2
    exit 1
  fi

  queryid_retention="$(extract_queryid_retention <<< "$config_json")"
  if [[ "$queryid_retention" != "$expected_queryid_retention" ]]; then
    printf '%s\n' "FAIL: $label queryid retention expected $expected_queryid_retention, got $queryid_retention" >&2
    exit 1
  fi

  assert_vm_safety_flags "$label" "$config_json"
  printf '%s\n' "PASS: $label retention config renders VM=$vm_retention queryid=$queryid_retention"
}

main() {
  cd "$(dirname "$0")/../.."

  local default_config
  local override_config
  local bare_int_config
  local long_retention_config
  local empty_config

  default_config="$(render_compose_config)"
  assert_retention_config "default" "336h" "720" "$default_config"

  override_config="$(render_compose_config VM_RETENTION_PERIOD=168h QUERYID_RETENTION_HOURS=168)"
  assert_retention_config "override" "168h" "168" "$override_config"

  bare_int_config="$(render_compose_config VM_RETENTION_PERIOD=6 QUERYID_RETENTION_HOURS=4380)"
  assert_retention_config "bare-int-months" "6" "4380" "$bare_int_config"

  long_retention_config="$(render_compose_config VM_RETENTION_PERIOD=4380h QUERYID_RETENTION_HOURS=4380)"
  assert_retention_config "long-retention" "4380h" "4380" "$long_retention_config"

  # ${VAR:-default} must fall back for explicitly empty values, not just unset values.
  empty_config="$(render_compose_config VM_RETENTION_PERIOD= QUERYID_RETENTION_HOURS=)"
  assert_retention_config "empty-fallback" "336h" "720" "$empty_config"
}

main "$@"
