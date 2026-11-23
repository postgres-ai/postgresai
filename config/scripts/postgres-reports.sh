#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# First run: wait REPORTER_INITIAL_DELAY_SECONDS (180 s), then poll pgai-mon
# every REPORTER_READINESS_POLL_SECONDS (15 s) until each enabled target in
# instances.yml has `settings` and `db_stats` samples, at most
# REPORTER_READINESS_TIMEOUT_SECONDS (30 min; 0 skips the wait). Collection
# time has no fixed bound (some metrics run for minutes, a failed first
# connect retries after 120 s), so the wait checks the data, not the clock.
# If no target has data the reporter does not upload; later runs retry.
initial_delay_seconds="${REPORTER_INITIAL_DELAY_SECONDS:-180}"
readiness_timeout_seconds="${REPORTER_READINESS_TIMEOUT_SECONDS:-1800}"
readiness_poll_seconds="${REPORTER_READINESS_POLL_SECONDS:-15}"
# The target list: the instances.yml pgai-mon collects (docker-compose.yml mounts it here).
sources_path="${REPORTER_SOURCES_PATH:-/app/instances.yml}"
# A failed run (e.g. the platform is down) is retried after this, not daily
# and not by a container restart every few minutes.
retry_seconds="${REPORTER_RETRY_SECONDS:-1800}"
interval_seconds="${REPORTER_INTERVAL_SECONDS:-86400}"

# pgai-mon's SQL endpoint; the bearer token comes from PGAI_MON_TOKEN in the
# environment (the reporter reads it there, never from argv).
mon_url="${PGAI_MON_URL:-http://pgai-mon:8765}"
export PGAI_MON_URL="${mon_url}"
output_template="${REPORTER_OUTPUT_TEMPLATE:-/app/all_reports_%Y%m%d_%H%M%S.json}"
use_current_time="${USE_CURRENT_TIME:-false}"

pgwatch_config_path="${REPORTER_PGWATCH_CONFIG_PATH:-/app/.pgwatch-config}"
api_url="${REPORTER_API_URL:-}"

# docker-compose.yml mounts instances.yml and .pgwatch-config as single files, which keep the
# inode they were mounted with: a tool that replaces a file on the host (sed -i, Ansible, mv)
# leaves this container on the old one, whose link count (by path) is then 0 (#523). Before
# each run: exit so Docker restarts the reporter with the new file. A file at link count 0 when
# first seen is not checked (file sharing that always says 0 would restart it forever).
replaced_exit_code=75
declare -A replaced_armed=()
check_replaced_files() {
  local f n
  for f in "$@"; do
    n="$(stat -L -c %h "${f}" 2>/dev/null)" || continue
    if [[ -z "${replaced_armed[${f}]+x}" ]]; then
      if [[ "${n}" == "0" ]]; then
        echo "postgres-reports: ${f} has link count 0 at start; replaced-file detection is off for it" >&2
        replaced_armed[${f}]=0
      else
        echo "postgres-reports: watching ${f} for replacement (link count ${n})"
        replaced_armed[${f}]=1
      fi
    elif [[ "${replaced_armed[${f}]}" == "1" && "${n}" == "0" ]]; then
      echo "postgres-reports: ${f} was replaced or removed on the host; the container still sees the old file." \
        "Exiting so Docker restarts postgres-reports with the new one. If the file was removed: remove the" \
        "directory Docker creates at that path (sudo rmdir), restore the file, then docker compose up -d." >&2
      exit "${replaced_exit_code}"
    fi
  done
}

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

echo "postgres-reports: initial_delay_seconds=${initial_delay_seconds}, interval_seconds=${interval_seconds}, readiness_timeout_seconds=${readiness_timeout_seconds}"
check_replaced_files "${sources_path}" "${pgwatch_config_path}"
sleep_seconds "$initial_delay_seconds"

if [[ "${readiness_timeout_seconds}" =~ ^[0-9]+$ ]] && (( readiness_timeout_seconds > 0 )); then
  # Always exits 0: on timeout the run goes ahead.
  # wait_ready's URL argument is the Prometheus fallback; with PGAI_MON_URL
  # exported the generator queries pgai-mon.
  python -m reporter.wait_ready "${mon_url}" "${sources_path}" \
    "${readiness_timeout_seconds}" "${readiness_poll_seconds}" || true
fi

# Until a report succeeds, end its window now: the previous full hour can predate collection.
# Later runs keep the hour-aligned window unless USE_CURRENT_TIME=true.
first_run=true
while true; do
  check_replaced_files "${sources_path}" "${pgwatch_config_path}"
  output_path="$(date -u +"${output_template}")"
  reporter_args=(--mon-url "${mon_url}" --output "${output_path}")
  if [[ "${use_current_time}" == "true" || "${first_run}" == "true" ]]; then
    reporter_args+=(--use-current-time)
  fi

  # Project name: env var > config file, read each cycle: a platform-provisioned
  # box learns its project only when the platform adopts the instance, after
  # this container has started. No default: uploads require a project name.
  project_name="${REPORTER_PROJECT_NAME:-}"
  if [[ -z "${project_name}" ]] && config_project_name="$(read_project_name)"; then
    project_name="${config_project_name}"
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
    first_run=false
    echo "postgres-reports: sleeping ${interval_seconds}s"
    sleep_seconds "${interval_seconds}"
  else
    echo "postgres-reports: run failed; retrying in ${retry_seconds}s" >&2
    sleep_seconds "${retry_seconds}"
  fi
done
