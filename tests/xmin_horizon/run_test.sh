#!/usr/bin/env bash
# Run the xmin_horizon integration test against a database pgai-mon collects from.
# Requires Bash; CI invokes this script with bash because it uses arrays.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DB_URL="${TARGET_DB_URL:-postgresql://postgres:postgres@localhost:55432/target_database}"
PGAI_MON_URL="${PGAI_MON_URL:-http://localhost:8765}"
COLLECTION_WAIT="${COLLECTION_WAIT_SECONDS:-480}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

# The token is read from PGAI_MON_TOKEN by the Python test; it is never passed in argv.
if [ -n "${PGAI_MON_TOKEN:-}" ]; then
  mon_auth_status="configured"
else
  mon_auth_status="none"
fi

echo "=========================================="
echo "xmin horizon metric test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target DB: ${TARGET_DB_URL}"
echo "  Standby DB: ${STANDBY_DB_URL:-<not set>}"
echo "  pgai-mon URL: ${PGAI_MON_URL}"
echo "  pgai-mon token: ${mon_auth_status}"
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
  --mon-url "${PGAI_MON_URL}"
  --collection-wait "${COLLECTION_WAIT}"
)
if [[ -n "${STANDBY_DB_URL:-}" ]]; then
  args+=(--standby-db-url "${STANDBY_DB_URL}")
fi

python3 "${SCRIPT_DIR}/test_xmin_horizon_metric.py" "${args[@]}"
