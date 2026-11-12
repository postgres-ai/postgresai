#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Generates pgwatch sources.yml files based on instances.yaml template.
#
# Expected inputs:
# - /app/instances.yaml (mounted from ./instances.yml)
# - /postgres_ai_configs volume
#
# Output:
# - /postgres_ai_configs/pgwatch/sources.yml
# - /postgres_ai_configs/pgwatch-prometheus/sources.yml

INSTANCES_PATH="${INSTANCES_PATH:-/app/instances.yaml}"
CONFIGS_DIR="${CONFIGS_DIR:-/postgres_ai_configs}"

write_default_instances() {
  cat <<'YAML'
- name: target-database
  conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database
  preset_metrics: full
  custom_metrics:
  is_enabled: true
  group: default
  custom_tags:
    env: demo
    cluster: local
    node_name: node-01
    sink_type: ~sink_type~
YAML
}

write_sources() {
  local sink_type="$1"
  local out_path="$2"
  local instances_path="$3"

  {
    echo "# PGWatch Sources Configuration - ${sink_type} Instance"
    sed "s/~sink_type~/${sink_type}/g" "${instances_path}"
  } > "${out_path}"
}

main() {
  local instances_path
  instances_path="${INSTANCES_PATH}"

  if [[ ! -f "${instances_path}" ]]; then
    echo "generate-pgwatch-sources: instances file not found: ${instances_path}; using demo default" >&2
    instances_path="$(mktemp)"
    write_default_instances > "${instances_path}"
  fi

  mkdir -p -- "${CONFIGS_DIR}/pgwatch" "${CONFIGS_DIR}/pgwatch-prometheus"

  write_sources "postgresql" "${CONFIGS_DIR}/pgwatch/sources.yml" "${instances_path}"
  write_sources "prometheus" "${CONFIGS_DIR}/pgwatch-prometheus/sources.yml" "${instances_path}"

  echo "generate-pgwatch-sources: generated sources.yml files"
}

main "$@"


