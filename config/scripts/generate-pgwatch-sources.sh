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
# the CLI (js-yaml) and provisioning write: list items, 2-space fields, 4-space
# tags, plain, quoted or block (>-, |) scalars. Strings are built character by
# character, never with gsub, because BusyBox awk (bash:5.2) and other awks
# disagree on backslashes in replacements.
write_supabase_targets() {
  local instances_path="$1"
  local out_path="$2"
  awk -v q="'" '
    function trim(v) { sub(/^[ \t]+/, "", v); sub(/[ \t]+$/, "", v); return v }
    function unquote(v,   n, i, c, out) {
      v = trim(v); n = length(v); out = ""
      if (n >= 2 && substr(v, 1, 1) == "\"" && substr(v, n, 1) == "\"") {
        for (i = 2; i < n; i++) {
          c = substr(v, i, 1)
          if (c == "\\" && i < n - 1) { i++; c = substr(v, i, 1); if (c == "t") c = "\t"; else if (c == "n") c = " " }
          out = out c
        }
        return out
      }
      if (n >= 2 && substr(v, 1, 1) == q && substr(v, n, 1) == q) {
        for (i = 2; i < n; i++) { c = substr(v, i, 1); if (c == q && substr(v, i + 1, 1) == q) i++; out = out c }
        return out
      }
      sub(/[ \t]+#.*$/, "", v)
      return v
    }
    function json(v,   i, c, out) {
      out = ""
      for (i = 1; i <= length(v); i++) {
        c = substr(v, i, 1)
        if (c == "\\") out = out "\\\\"
        else if (c == "\"") out = out "\\\""
        else if (c == "\t") out = out "\\t"
        else if (c >= " ") out = out c
      }
      return "\"" out "\""
    }
    function assign(field, v) {
      if (field == "f:conn_str") conn = v
      else if (field == "f:is_enabled") enabled = v
      else if (field == "t:cluster") cluster = v
      else if (field == "t:node_name") node = v
    }
    function finish() {
      if (pending != "") { assign(pending, block); pending = "" }
      if (seen && enabled != "false" && conn ~ /@[^\/]*(\.supabase\.co|\.pooler\.supabase\.com)(:[0-9]+)?([\/?]|$)/) {
        n++; cluster_out = cluster; node_out = node
      }
      seen = 0; conn = ""; enabled = ""; cluster = ""; node = ""; tags = 0
    }
    # field: "f:<key>" at indent 2, "t:<key>" under custom_tags at indent 4.
    function field(line, indent, prefix,   key, val) {
      match(line, /^ *[A-Za-z_]+:/)
      key = trim(substr(line, 1, RLENGTH - 1)); val = trim(substr(line, RLENGTH + 1))
      if (prefix == "f:") tags = (key == "custom_tags")
      if (val ~ /^[>|][-+]?$/) { pending = prefix key; pending_indent = indent; block = "" }
      else assign(prefix key, unquote(val))
    }
    { line = $0; sub(/\r$/, "", line) }
    pending != "" {
      if (line ~ /^[ \t]*$/) next
      match(line, /^ */)
      if (RLENGTH > pending_indent) { block = (block == "" ? trim(line) : block " " trim(line)); next }
      assign(pending, block); pending = ""
    }
    /^- / { finish(); seen = 1; line = "  " substr(line, 3) }
    line ~ /^  [A-Za-z_]+:/ { field(line, 2, "f:"); next }
    tags && line ~ /^    [A-Za-z_]+:/ { field(line, 4, "t:"); next }
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


