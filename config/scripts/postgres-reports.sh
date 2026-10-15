#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# First run: wait REPORTER_INITIAL_DELAY_SECONDS (180 s), then poll Prometheus
# every REPORTER_READINESS_POLL_SECONDS (15 s) until each target in
# instances.yml has `settings` and `db_stats` samples, at most
# REPORTER_READINESS_TIMEOUT_SECONDS (30 min; 0 skips the wait). Collection
# time has no fixed bound (some metrics run for minutes, a failed first
# connect retries after 120 s), so the wait checks the data, not the clock.
# If no target has data the reporter does not upload; later runs retry.
initial_delay_seconds="${REPORTER_INITIAL_DELAY_SECONDS:-180}"
readiness_timeout_seconds="${REPORTER_READINESS_TIMEOUT_SECONDS:-1800}"
readiness_poll_seconds="${REPORTER_READINESS_POLL_SECONDS:-15}"
sources_path="${REPORTER_SOURCES_PATH:-/postgres_ai_configs/pgwatch-prometheus/sources.yml}"
# A failed run (e.g. the platform is down) is retried after this, not daily
# and not by a container restart every few minutes.
retry_seconds="${REPORTER_RETRY_SECONDS:-1800}"
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

echo "postgres-reports: initial_delay_seconds=${initial_delay_seconds}, interval_seconds=${interval_seconds}, readiness_timeout_seconds=${readiness_timeout_seconds}"
sleep_seconds "$initial_delay_seconds"

if [[ "${readiness_timeout_seconds}" =~ ^[0-9]+$ ]] && (( readiness_timeout_seconds > 0 )); then
  # Always exits 0: on timeout the run goes ahead.
  python -m reporter.wait_ready "${prometheus_url}" "${sources_path}" \
    "${readiness_timeout_seconds}" "${readiness_poll_seconds}" || true
fi

while true; do
  output_path="$(date -u +"${output_template}")"
  reporter_args=(--prometheus-url "${prometheus_url}" --output "${output_path}")
  if [[ "${use_current_time}" == "true" ]]; then
    reporter_args+=(--use-current-time)
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
      reporter_args+=(--no-upload)
    else
      echo "postgres-reports: generating reports (upload enabled) -> ${output_path}"
      reporter_args+=(--api-url "${api_url}" --project-name "${project_name}" --token "${api_key}")
    fi
  else
    echo "postgres-reports: generating reports (no upload) -> ${output_path}"
    reporter_args+=(--no-upload)
  fi

  if python -m reporter.postgres_reports "${reporter_args[@]}"; then
    echo "postgres-reports: sleeping ${interval_seconds}s"
    sleep_seconds "${interval_seconds}"
  else
    echo "postgres-reports: run failed; retrying in ${retry_seconds}s" >&2
    sleep_seconds "${retry_seconds}"
  fi
done
