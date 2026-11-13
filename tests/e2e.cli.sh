#!/bin/bash
# E2E tests for postgres_ai CLI (Node.js)
# Usage: ./tests/e2e.cli.sh

set -Eeuo pipefail

CLI_CMD="node ./cli/dist/bin/postgres-ai.js"
MON_CMD="${CLI_CMD} mon"
GRAFANA_CONTAINER="grafana-with-datasources"
GRAFANA_DEADLINE_SECONDS=120
GRAFANA_DATASOURCE_POLL_SECONDS=3
GRAFANA_RESTART_POLL_SECONDS=2
readonly CLI_CMD MON_CMD GRAFANA_CONTAINER GRAFANA_DEADLINE_SECONDS \
  GRAFANA_DATASOURCE_POLL_SECONDS GRAFANA_RESTART_POLL_SECONDS

# Return the host where Docker publishes ports for the current runner.
docker_host_for_ports() {
  local host="${DOCKER_HOST:-}"
  case "${host}" in
    ""|unix://*|npipe://*|fd://*)
      echo "localhost"
      ;;
    tcp://*)
      host="${host#tcp://}"
      echo "${host%%:*}"
      ;;
    *)
      echo "ERROR: unsupported DOCKER_HOST for port checks: ${host}" >&2
      return 1
      ;;
  esac
}

# The Grafana admin password: the one local-install generated (.pgwatch-config, which the
# CLI passes to Compose), else the one in .env. There is no default.
grafana_password() {
  local password=""
  if [[ -f .pgwatch-config ]]; then
    password="$(awk -F= '$1 == "grafana_password" { value = substr($0, index($0, "=") + 1) } END { print value }' .pgwatch-config)"
  fi
  if [[ -z "${password}" && -f .env ]]; then
    password="$(awk -F= '$1 == "GF_SECURITY_ADMIN_PASSWORD" { value = substr($0, index($0, "=") + 1) } END { print value }' .env)"
  fi
  if [[ -z "${password}" ]]; then
    echo "ERROR: no Grafana password in .pgwatch-config or .env" >&2
    return 1
  fi
  echo "${password}"
}

# Build a Grafana Basic auth header without putting the raw password in curl argv.
grafana_auth_header() {
  local password
  local token
  password="$(grafana_password)"
  token="$(GRAFANA_PASSWORD="${password}" node -e 'process.stdout.write(Buffer.from(`monitor:${process.env.GRAFANA_PASSWORD}`).toString("base64"))')"
  printf 'Authorization: Basic %s\n' "${token}"
}

# Fetch Grafana datasources without exposing raw credentials in curl argv.
# curl --config reads the header from a process-substitution FIFO; using -H
# would put the basic-auth token in argv where ps/proc can expose it.
grafana_datasources_json() {
  local grafana_url
  grafana_url="http://$(docker_host_for_ports):3000/api/datasources"
  curl -fsS --config <(printf 'header = "%s"\n' "$(grafana_auth_header)") "${grafana_url}"
}

# Verify provisioned datasource names, UIDs, orgs, defaults, and read-only state.
assert_grafana_datasources() {
  local datasources_json
  datasources_json="$(grafana_datasources_json)"

  DATASOURCES_JSON="${datasources_json}" node <<'NODE'
const datasources = JSON.parse(process.env.DATASOURCES_JSON || "[]");
// config/grafana/provisioning/datasources/pgai-mon.yml provisions exactly one
// datasource; the old stack's PGWatch-* and Infinity ones must be gone (an
// upgraded volume deletes them through deleteDatasources).
const expected = new Map([
  ["pgai-mon", { uid: "pgai-mon", orgId: 1, isDefault: false, type: "yesoreyeram-infinity-datasource" }],
]);
const staleUids = new Set(["stale-pgai-mon"]);
const byName = new Map(datasources.map((datasource) => [datasource.name, datasource]));

if (datasources.length !== expected.size) {
  throw new Error(`Expected ${expected.size} Grafana datasources, got ${datasources.length}`);
}

for (const datasource of datasources) {
  if (staleUids.has(datasource.uid)) {
    throw new Error(`Stale datasource uid survived provisioning: ${datasource.name} -> ${datasource.uid}`);
  }
}

for (const [name, expectedConfig] of expected) {
  const datasource = byName.get(name);
  if (!datasource) {
    throw new Error(`Missing Grafana datasource: ${name}`);
  }
  if (datasource.uid !== expectedConfig.uid) {
    throw new Error(`${name} uid drifted: expected ${expectedConfig.uid}, got ${datasource.uid}`);
  }
  if (datasource.orgId !== expectedConfig.orgId) {
    throw new Error(`${name} orgId drifted: expected ${expectedConfig.orgId}, got ${datasource.orgId}`);
  }
  if (datasource.type !== expectedConfig.type) {
    throw new Error(`${name} type drifted: expected ${expectedConfig.type}, got ${datasource.type}`);
  }
  if (datasource.isDefault !== expectedConfig.isDefault) {
    throw new Error(`${name} default flag drifted: expected ${expectedConfig.isDefault}, got ${datasource.isDefault}`);
  }
  // Grafana API versions expose provisioned immutability as readOnly, editable, or both.
  const hasReadOnly = Object.prototype.hasOwnProperty.call(datasource, "readOnly");
  const hasEditable = Object.prototype.hasOwnProperty.call(datasource, "editable");
  if (!hasReadOnly && !hasEditable) {
    throw new Error(`${name} exposes neither readOnly nor editable after provisioning`);
  }
  if (hasReadOnly && datasource.readOnly !== true) {
    throw new Error(`${name} readOnly flag drifted: expected true, got ${datasource.readOnly}`);
  }
  if (hasEditable && datasource.editable !== false) {
    throw new Error(`${name} editable flag drifted: expected false, got ${datasource.editable}`);
  }
}
NODE
}

# Poll Grafana until provisioning has converged after startup or restart.
wait_for_grafana_datasources() {
  local deadline
  deadline=$((SECONDS + GRAFANA_DEADLINE_SECONDS))
  until assert_grafana_datasources; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: Grafana datasource provisioning did not converge" >&2
      assert_grafana_datasources || true
      return 1
    fi
    sleep "${GRAFANA_DATASOURCE_POLL_SECONDS}"
  done
}

# Return the current Grafana container start timestamp.
grafana_started_at() {
  docker inspect "${GRAFANA_CONTAINER}" --format '{{.State.StartedAt}}'
}

# Read an environment variable from the Grafana container config.
grafana_env_value() {
  local key="$1"
  docker inspect "${GRAFANA_CONTAINER}" \
    --format '{{range .Config.Env}}{{println .}}{{end}}' |
    sed -n "s/^${key}=//p" | tail -n 1
}

# Derive Grafana's SQLite path from GF_DATABASE_PATH when set; otherwise use
# GF_PATHS_DATA/grafana.db, falling back to Grafana's image default
# /var/lib/grafana/grafana.db.
grafana_db_path() {
  local db_path
  local data_path
  db_path="$(grafana_env_value GF_DATABASE_PATH)"
  if [[ -n "${db_path}" ]]; then
    echo "${db_path}"
    return 0
  fi
  data_path="$(grafana_env_value GF_PATHS_DATA)"
  echo "${data_path:-/var/lib/grafana}/grafana.db"
}

# Wait until Docker reports a new Grafana process after compose restart.
wait_for_grafana_restart() {
  local previous_started_at="$1"
  local deadline
  local current_started_at
  deadline=$((SECONDS + GRAFANA_DEADLINE_SECONDS))

  while true; do
    current_started_at="$(grafana_started_at 2>/dev/null || true)"
    if [[ -n "${current_started_at}" && "${current_started_at}" != "${previous_started_at}" ]]; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: Grafana container did not restart" >&2
      return 1
    fi
    sleep "${GRAFANA_RESTART_POLL_SECONDS}"
  done
}

# Corrupt the persisted Grafana datasource rows, then restart Grafana before return.
seed_conflicting_grafana_datasource_state() {
  local db_path
  local restart_needed="false"
  db_path="$(grafana_db_path)"

  # restart_needed avoids a double start: normal exit sets it false after the
  # explicit docker start, while early errors leave it true so RETURN restarts
  # Grafana before the function exits.
  trap 'if [[ "${restart_needed}" == "true" ]]; then docker start "${GRAFANA_CONTAINER}" >/dev/null 2>&1 || true; fi' RETURN
  docker stop "${GRAFANA_CONTAINER}" >/dev/null
  restart_needed="true"
  docker run --rm --volumes-from "${GRAFANA_CONTAINER}" \
    -e GRAFANA_DB_PATH="${db_path}" alpine:3.20 sh -s <<'SH'
set -eu
apk add --no-cache sqlite
changed="$(sqlite3 "${GRAFANA_DB_PATH}" <<'SQL'
UPDATE data_source
SET uid = 'stale-pgai-mon',
  is_default = 1,
  updated = datetime('now')
WHERE name = 'pgai-mon';
SELECT changes();
SQL
)"
if [ "${changed}" != "1" ]; then
  echo "ERROR: expected to corrupt one Grafana datasource, changed=${changed}" >&2
  exit 1
fi
stale_count="$(sqlite3 "${GRAFANA_DB_PATH}" "SELECT count(*) FROM data_source WHERE uid LIKE 'stale-%';")"
if [ "${stale_count}" != "1" ]; then
  echo "ERROR: expected one stale Grafana datasource UID, got ${stale_count}" >&2
  exit 1
fi
SH
  docker start "${GRAFANA_CONTAINER}" >/dev/null
  restart_needed="false"
  trap - RETURN
}

# Ensure instances.yml exists as a file (instances.demo.yml is the tracked template)
if [[ ! -f instances.yml ]]; then
  cp instances.demo.yml instances.yml || { echo "ERROR: instances.demo.yml missing — cannot seed instances.yml" >&2; exit 1; }
fi

echo "=== Testing service commands ==="
$MON_CMD check
$MON_CMD config
$MON_CMD update-config
$MON_CMD start
sleep 10
$MON_CMD status
$MON_CMD logs --tail 5 grafana
# Probes pgai-mon at PGAI_MON_URL (default http://127.0.0.1:8765): set it where
# the docker host is not this host (docker-in-docker).
$MON_CMD health --wait 120

echo ""
echo "=== Testing Grafana datasource provisioning after fresh start ==="
wait_for_grafana_datasources

echo ""
echo "=== Testing instance commands ==="
$MON_CMD targets list
$MON_CMD targets add "postgresql://monitor:monitor_pass@target-db:5432/target_database" ci-test
$MON_CMD targets list | grep -q ci-test
sleep 5
# Tolerated: target-db resolves on the compose network only, so the test
# connects only where this script runs on that network.
$MON_CMD targets test ci-test || true
$MON_CMD targets remove ci-test

echo ""
echo "=== Testing API key commands ==="
$CLI_CMD auth login --set-key "test_api_key_12345"
$CLI_CMD auth show-key | grep -q "test_api"
$CLI_CMD auth remove-key

echo ""
echo "=== Testing Grafana commands ==="
$MON_CMD show-grafana-credentials
$MON_CMD generate-grafana-password
$MON_CMD show-grafana-credentials

echo ""
echo "=== Testing service management ==="
seed_conflicting_grafana_datasource_state
wait_for_grafana_datasources
grafana_started_before_restart="$(grafana_started_at)"
$MON_CMD restart grafana
wait_for_grafana_restart "${grafana_started_before_restart}"
wait_for_grafana_datasources
$MON_CMD status
$MON_CMD stop
$MON_CMD clean

echo ""
echo "✓ All E2E tests passed"

