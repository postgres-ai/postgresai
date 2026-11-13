#!/bin/bash
# Simple wrapper script to run the lock_waits metric test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values (can be overridden by environment variables)
TARGET_DB_URL="${TARGET_DB_URL:-postgresql://postgres:postgres@localhost:55432/target_database}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:59090}"
TEST_DBNAME="${TEST_DBNAME:-target_database}"
COLLECTION_WAIT="${COLLECTION_WAIT_SECONDS:-60}"

echo "=========================================="
echo "Lock Waits Metric Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target DB: $TARGET_DB_URL"
echo "  Prometheus URL: $PROMETHEUS_URL"
echo "  Test DB Name: $TEST_DBNAME"
echo "  Collection Wait: ${COLLECTION_WAIT}s"
echo ""

# Check if required packages are installed
if ! python3 -c "import psycopg" 2>/dev/null; then
    echo "Installing psycopg..."
    pip3 install psycopg
fi

if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing requests..."
    pip3 install requests
fi

# Run the test
cd "$PROJECT_ROOT"
python3 tests/lock_waits/test_lock_waits_metric.py \
    --target-db-url "$TARGET_DB_URL" \
    --prometheus-url "$PROMETHEUS_URL" \
    --test-dbname "$TEST_DBNAME" \
    --collection-wait "$COLLECTION_WAIT"

