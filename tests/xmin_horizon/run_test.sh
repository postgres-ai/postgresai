#!/usr/bin/env bash
# Run the xmin_horizon integration test against a pgwatch-monitored database.
# Requires Bash; CI invokes this script with bash because it uses arrays.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DB_URL="${TARGET_DB_URL:-postgresql://postgres:postgres@localhost:55432/target_database}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:59090}"
PROMETHEUS_TEST_USERNAME="${PROMETHEUS_USERNAME-}"
PROMETHEUS_TEST_PASSWORD="${PROMETHEUS_PASSWORD-}"
if [ -z "${PROMETHEUS_USERNAME+x}" ]; then
  PROMETHEUS_TEST_USERNAME="${VM_AUTH_USERNAME-}"
fi
if [ -z "${PROMETHEUS_PASSWORD+x}" ]; then
  PROMETHEUS_TEST_PASSWORD="${VM_AUTH_PASSWORD-}"
fi
COLLECTION_WAIT="${COLLECTION_WAIT_SECONDS:-480}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if [ -n "${PROMETHEUS_TEST_USERNAME}" ] && [ -n "${PROMETHEUS_TEST_PASSWORD}" ]; then
  prometheus_auth_status="configured"
else
  prometheus_auth_status="none"
fi

echo "=========================================="
echo "xmin horizon metric test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target DB: ${TARGET_DB_URL}"
echo "  Standby DB: ${STANDBY_DB_URL:-<not set>}"
echo "  Prometheus URL: ${PROMETHEUS_URL}"
echo "  Prometheus Auth: ${prometheus_auth_status}"
echo "  Collection Wait: ${COLLECTION_WAIT}s"
echo "  Require replication slot path: ${REQUIRE_REPLICATION_SLOT_TEST:-0}"
echo "  Require prepared xacts path: ${REQUIRE_PREPARED_XACTS_TEST:-0}"
echo "  Require standby feedback path: ${REQUIRE_STANDBY_FEEDBACK_TEST:-0}"
echo ""

# Safe to repeat in CI containers; local users can preinstall in a venv.
python3 -m pip install --quiet -r "${REQUIREMENTS_FILE}"
python3 -m unittest discover -s "${SCRIPT_DIR}" -p 'test_metrics_sql_static.py'

args=(
  --target-db-url "${TARGET_DB_URL}"
  --prometheus-url "${PROMETHEUS_URL}"
  --collection-wait "${COLLECTION_WAIT}"
)
if [[ -n "${STANDBY_DB_URL:-}" ]]; then
  args+=(--standby-db-url "${STANDBY_DB_URL}")
fi

python3 "${SCRIPT_DIR}/test_xmin_horizon_metric.py" "${args[@]}"
