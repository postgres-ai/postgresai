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
# - /postgres_ai_configs/prometheus/supabase-host-metrics.json

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

# Writes the file_sd target of the supabase-host-metrics scrape job. The
# relay serves one Supabase project, so its series get the pgwatch cluster and
# node_name tags of the one enabled Supabase target in instances.yml (a
# db.<ref>.supabase.co or *.pooler.supabase.com host). With none or several,
# the target is written without labels. Parses the instances.yml layout that
# the CLI and provisioning write: list items, 2-space fields, 4-space tags.
write_supabase_targets() {
  local instances_path="$1"
  local out_path="$2"
  awk '
    function unquote(v) {
      sub(/^[ \t]+/, "", v); sub(/[ \t]+$/, "", v)
      if (v ~ /^".*"$/) { v = substr(v, 2, length(v) - 2); gsub(/\\"/, "\"", v) }
      else if (v ~ /^\047.*\047$/) { v = substr(v, 2, length(v) - 2); gsub(/\047\047/, "\047", v) }
      return v
    }
    function json(v) { gsub(/\\/, "\\\\", v); gsub(/"/, "\\\"", v); gsub(/\t/, "\\t", v); return "\"" v "\"" }
    function finish() {
      if (seen && enabled != "false" && conn ~ /@[^\/]*(\.supabase\.co|\.pooler\.supabase\.com)(:[0-9]+)?([\/?]|$)/) {
        n++; cluster_out = cluster; node_out = node
      }
      seen = 0; conn = ""; enabled = ""; cluster = ""; node = ""; tags = 0
    }
    /^- / { finish(); seen = 1; line = substr($0, 3); if (line ~ /^conn_str:/) conn = unquote(substr(line, 10)); next }
    /^  [a-z_]+:/ { tags = ($0 ~ /^  custom_tags:/) }
    /^  conn_str:/ { conn = unquote(substr($0, 12)) }
    /^  is_enabled:/ { enabled = unquote(substr($0, 14)) }
    tags && /^    cluster:/ { cluster = unquote(substr($0, 14)) }
    tags && /^    node_name:/ { node = unquote(substr($0, 16)) }
    END {
      finish()
      labels = "{}"
      if (n == 1 && cluster_out != "" && node_out != "") labels = "{\"cluster\": " json(cluster_out) ", \"node_name\": " json(node_out) "}"
      if (n > 1) print "generate-pgwatch-sources: " n " Supabase targets in instances.yml; Supabase host metrics get no cluster/node_name labels" > "/dev/stderr"
      print "[{\"targets\": [\"instance-jobs:9188\"], \"labels\": " labels "}]"
    }
  ' "${instances_path}" > "${out_path}.tmp"
  mv -f -- "${out_path}.tmp" "${out_path}"
}

main() {
  local instances_path
  instances_path="${INSTANCES_PATH}"

  if [[ ! -f "${instances_path}" ]]; then
    echo "generate-pgwatch-sources: instances file not found: ${instances_path}; using demo default" >&2
    instances_path="$(mktemp)"
    write_default_instances > "${instances_path}"
  fi

  mkdir -p -- "${CONFIGS_DIR}/pgwatch" "${CONFIGS_DIR}/pgwatch-prometheus" "${CONFIGS_DIR}/prometheus"

  write_sources "postgresql" "${CONFIGS_DIR}/pgwatch/sources.yml" "${instances_path}"
  write_sources "prometheus" "${CONFIGS_DIR}/pgwatch-prometheus/sources.yml" "${instances_path}"
  write_supabase_targets "${instances_path}" "${CONFIGS_DIR}/prometheus/supabase-host-metrics.json"

  echo "generate-pgwatch-sources: generated sources.yml files"
}

main "$@"


