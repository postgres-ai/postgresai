#!/bin/bash
# E2E tests for postgres_ai CLI (Node.js)
# Usage: ./tests/e2e.cli.sh

set -e

CLI_CMD="node ./cli/dist/bin/postgres-ai.js"
MON_CMD="$CLI_CMD mon"

# Ensure instances.yml exists as a file (instances.demo.yml is the tracked template)
if [[ ! -f instances.yml ]]; then
  cp instances.demo.yml instances.yml || { echo "ERROR: instances.demo.yml missing — cannot seed instances.yml" >&2; exit 1; }
fi

echo "=== Testing service commands ==="
$MON_CMD check || true
$MON_CMD config || true
$MON_CMD update-config
$MON_CMD start
sleep 10
$MON_CMD status
$MON_CMD logs --tail 5 grafana || true
$MON_CMD health --wait 60 || true

echo ""
echo "=== Testing instance commands ==="
$MON_CMD targets list
$MON_CMD targets add "postgresql://monitor:monitor_pass@target-db:5432/target_database" ci-test
$MON_CMD targets list | grep -q ci-test
sleep 5
$MON_CMD targets test ci-test || true
$MON_CMD targets remove ci-test

echo ""
echo "=== Testing API key commands ==="
$CLI_CMD auth login --set-key "test_api_key_12345"
$CLI_CMD auth show-key | grep -q "test_api"
$CLI_CMD auth remove-key

echo ""
echo "=== Testing Grafana commands ==="
$MON_CMD show-grafana-credentials || true
$MON_CMD generate-grafana-password || true
$MON_CMD show-grafana-credentials || true

echo ""
echo "=== Testing service management ==="
$MON_CMD restart grafana
sleep 3
$MON_CMD status
$MON_CMD stop
$MON_CMD clean || true

echo ""
echo "✓ All E2E tests passed"

