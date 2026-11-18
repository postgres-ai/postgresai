#!/usr/bin/env bash
set -Eeuo pipefail

# Run the reporter on the host against the Docker Compose stack.
#
# Defaults:
# - pgai-mon: http://127.0.0.1:${PGAI_MON_PORT:-8765}, token from the project .env
# - Sources (node connections, for A005): ./instances.yml
# - Output directory: ./dev_reports/dev_report_<UTC timestamp>/
#   (one JSON file per check, e.g. A002.json, K003.json, ...)
#
# Overrides (env vars):
# - PGAI_MON_PORT (environment, then project .env, then 8765)
# - PGAI_MON_URL
# - PGAI_MON_TOKEN
# - REPORTER_SOURCES_PATH
# - OUTPUT_DIR
# - EXTRA_ARGS (appended as-is)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PGAI_MON_PORT+x}" && -f "${REPO_ROOT}/.env" ]]; then
  PGAI_MON_PORT="$(awk '
    /^[[:space:]]*(export[[:space:]]+)?PGAI_MON_PORT[[:space:]]*=/ {
      value = substr($0, index($0, "=") + 1)
      sub(/[[:space:]]*#.*/, "", value)
      gsub(/^[[:space:]"\047]+|[[:space:]"\047]+$/, "", value)
    }
    END { print value }
   ' "${REPO_ROOT}/.env")"
fi
PGAI_MON_URL="${PGAI_MON_URL:-http://127.0.0.1:${PGAI_MON_PORT:-8765}}"
if [[ -z "${PGAI_MON_TOKEN:-}" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
  PGAI_MON_TOKEN="$(sed -n 's/^PGAI_MON_TOKEN=//p' "${REPO_ROOT}/.env" | tail -n 1)"
fi
export PGAI_MON_TOKEN="${PGAI_MON_TOKEN:-}"
REPORTER_SOURCES_PATH="${REPORTER_SOURCES_PATH:-${REPO_ROOT}/instances.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dev_reports/dev_report_$(date -u +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # Convenience: auto-activate local venv if present.
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"

# Reporter always writes per-check files to the current working directory.
# If we pass --output <file>, it ALSO writes a combined JSON.
# For local dev, we want only per-check files in a timestamped directory.
cd "${OUTPUT_DIR}"

# Run as a module so imports like "from reporter.logger import logger" work
# regardless of the current working directory.
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m reporter.postgres_reports \
  --mon-url "${PGAI_MON_URL}" \
  --sources-path "${REPORTER_SOURCES_PATH}" \
  --no-upload \
  --output "-" \
  ${EXTRA_ARGS}

echo "Wrote reports to: ${OUTPUT_DIR}"


