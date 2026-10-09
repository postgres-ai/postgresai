#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

initial_delay_seconds="${REPORTER_INITIAL_DELAY_SECONDS:-1800}"
interval_seconds="${REPORTER_INTERVAL_SECONDS:-86400}"

prometheus_url="${PROMETHEUS_URL:-http://sink-prometheus:9090}"
output_template="${REPORTER_OUTPUT_TEMPLATE:-/app/all_reports_%Y%m%d_%H%M%S.json}"
use_current_time="${USE_CURRENT_TIME:-false}"

pgwatch_config_path="${REPORTER_PGWATCH_CONFIG_PATH:-/app/.pgwatch-config}"
api_url="${REPORTER_API_URL:-}"
# Project name: env var takes priority, then config file. No default — uploads
# require a project name (the hardcoded "postgres-ai-monitoring" default was removed).
project_name="${REPORTER_PROJECT_NAME:-}"

sleep_seconds() {
  local s
  s="$1"
  if [[ "${s}" =~ ^[0-9]+$ ]] && (( s > 0 )); then
    sleep "${s}"
  fi
}

read_api_key() {
  local key
  if [[ ! -f "${pgwatch_config_path}" ]]; then
    return 1
  fi
  # Extract everything after first '=' to keep compatibility with values containing '='.
  # Suppress stderr to avoid noise when file is unreadable.
  key="$(
    grep -E '^api_key=' "${pgwatch_config_path}" 2>/dev/null \
      | head -n 1 \
      | cut -d'=' -f2- \
      | tr -d '\r'
  )"
  if [[ -n "${key:-}" ]]; then
    printf '%s' "${key}"
    return 0
  fi
  return 1
}

read_api_url() {
  local url
  if [[ ! -f "${pgwatch_config_path}" ]]; then
    return 1
  fi
  url="$(
    grep -E '^api_url=' "${pgwatch_config_path}" 2>/dev/null \
      | head -n 1 \
      | cut -d'=' -f2- \
      | tr -d '\r'
  )"
  if [[ -n "${url:-}" ]]; then
    printf '%s' "${url%/}"
    return 0
  fi
  return 1
}

read_project_name() {
  local name
  if [[ ! -f "${pgwatch_config_path}" ]]; then
    return 1
  fi
  name="$(
    grep -E '^project_name=' "${pgwatch_config_path}" 2>/dev/null \
      | head -n 1 \
      | cut -d'=' -f2- \
      | tr -d '\r'
  )"
  if [[ -n "${name:-}" ]]; then
    printf '%s' "${name}"
    return 0
  fi
  return 1
}

# Resolve API base: non-empty env var > config file > production default.
if [[ -z "${api_url}" ]]; then
  if config_api_url="$(read_api_url)"; then
    api_url="${config_api_url}"
  else
    api_url="https://postgres.ai/api/general"
  fi
fi

export REPORTER_API_URL="${api_url}"

# Resolve project name: env var > config file. No default — uploads require a
# project name (the hardcoded "postgres-ai-monitoring" default was removed).
if [[ -z "${project_name}" ]]; then
  if config_project_name="$(read_project_name)"; then
    project_name="${config_project_name}"
  fi
fi

echo "postgres-reports: initial_delay_seconds=${initial_delay_seconds}, interval_seconds=${interval_seconds}"
sleep_seconds "$initial_delay_seconds"

while true; do
  output_path="$(date -u +"${output_template}")"

  # Build optional args
  use_current_time_arg=""
  if [[ "${use_current_time}" == "true" ]]; then
    use_current_time_arg="--use-current-time"
  fi

  if api_key="$(read_api_key)"; then
    # A box token lives 90 days; renew it each cycle (internal#354). On stdin,
    # not argv. A failure is logged and the reports go on.
    printf '%s\n' "${api_key}" | python -m reporter.token_renew "${api_url}" || true
    if [[ -z "${project_name}" ]]; then
      # Upload requires a project name; there is no default. Still generate
      # reports locally (like the no-api-key branch) and only skip the upload.
      echo "postgres-reports: ERROR project name is required for upload — set REPORTER_PROJECT_NAME or project_name in .pgwatch-config; skipping upload this cycle" >&2
      echo "postgres-reports: generating reports (no upload) -> ${output_path}"
      python -m reporter.postgres_reports \
        --prometheus-url "${prometheus_url}" \
        --output "${output_path}" \
        --no-upload \
        ${use_current_time_arg}
    else
      echo "postgres-reports: generating reports (upload enabled) -> ${output_path}"
      python -m reporter.postgres_reports \
        --prometheus-url "${prometheus_url}" \
        --output "${output_path}" \
        --api-url "${api_url}" \
        --project-name "${project_name}" \
        --token "${api_key}" \
        ${use_current_time_arg}
    fi
  else
    echo "postgres-reports: generating reports (no upload) -> ${output_path}"
    python -m reporter.postgres_reports \
      --prometheus-url "${prometheus_url}" \
      --output "${output_path}" \
      --no-upload \
      ${use_current_time_arg}
  fi

  echo "postgres-reports: sleeping ${interval_seconds}s"
  sleep_seconds "${interval_seconds}"
done


