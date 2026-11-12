#!/usr/bin/env bash
# Smoke test of docker-compose.yml: config-init, the demo target-db, pgai-mon and Grafana come up;
# pgai-mon turns healthy, refuses SQL without the token, answers SQL with it, and has collected
# series from target-db; Grafana answers its health check. Then everything is removed.
#
#   PGAI_REGISTRY=registry.example.com/postgres-ai/postgres_ai PGAI_TAG=my-branch tests/e2e/compose_smoke.sh
#
# Runs in its own project directory (SMOKE_DIR, default <repo>/.compose-smoke) with its own .env
# and instances.yml, under its own compose project name, so a developer's stack is untouched. In
# GitLab's docker-in-docker the directory must be under the build directory (shared with the
# docker service), which the default is. Images must be pullable or present locally.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PGAI_TAG:?PGAI_TAG is required}"
registry="${PGAI_REGISTRY:-postgresai}"
smoke_dir="${SMOKE_DIR:-${root}/.compose-smoke}"
project="${SMOKE_PROJECT:-pgai-smoke}"
# The owner instances.yml gets, as `mon local-install` writes it (not the image's own uid).
owner="${SMOKE_OWNER:-1000:1000}"
collect_timeout="${SMOKE_COLLECT_TIMEOUT:-240}"
curl_image="curlimages/curl:8.11.1"

rand_hex() { od -An -N"$1" -tx1 /dev/urandom | tr -d ' \n'; }

mkdir -p "${smoke_dir}/host-metrics"
token="$(rand_hex 24)"
umask 077
cat > "${smoke_dir}/.env" <<EOF
PGAI_REGISTRY=${registry}
PGAI_TAG=${PGAI_TAG}
PGAI_MON_TOKEN=${token}
REPLICATOR_PASSWORD=$(rand_hex 32)
INSTANCE_JOBS_USER=${owner}
EOF
cp "${root}/instances.demo.yml" "${smoke_dir}/instances.yml"
chmod 0600 "${smoke_dir}/instances.yml"
chmod 0700 "${smoke_dir}/host-metrics"
chown "${owner}" "${smoke_dir}/instances.yml" "${smoke_dir}/host-metrics" 2>/dev/null \
  || echo "smoke: cannot chown to ${owner} (not root); pgai-mon may not read instances.yml" >&2

dc() { docker compose -p "${project}" -f "${root}/docker-compose.yml" --project-directory "${smoke_dir}" "$@"; }

cleanup() {
  local rc=$?
  if (( rc != 0 )); then
    dc ps -a || true
    dc logs --no-color --tail 200 config-init pgai-mon grafana || true
  fi
  dc --profile demo down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "${smoke_dir}"
  exit "${rc}"
}
trap cleanup EXIT

# health CONTAINER TIMEOUT_S: wait for Docker's healthcheck to say healthy.
wait_healthy() {
  local c="$1" deadline=$((SECONDS + $2)) s
  while :; do
    s="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$c" 2>/dev/null || true)"
    [[ "$s" == "healthy" ]] && { echo "smoke: $c healthy"; return 0; }
    if (( SECONDS >= deadline )); then
      echo "smoke: FAIL: $c is '$s' after $2 s" >&2
      return 1
    fi
    sleep 3
  done
}

# req ARGS...: curl on the compose network (ports are published on the docker host's
# 127.0.0.1 only, which a job container in docker-in-docker cannot reach). Prints
# "<http code> <body>"; the token travels in a header file, not in argv.
req() {
  docker run --rm --network "${project}_default" -e TOKEN="${token}" --entrypoint sh "${curl_image}" -c '
    printf "Authorization: Bearer %s\n" "$TOKEN" > /tmp/auth
    curl -sS -o /tmp/body -w "%{http_code} " "$@" && cat /tmp/body' sh "$@" 2>&1 \
    | tr -d '\n'
}
req_noauth() {
  docker run --rm --network "${project}_default" "${curl_image}" -sS -o /dev/null -w "%{http_code}" "$@" 2>&1
}
sql() { req -H @/tmp/auth -H 'Content-Type: text/plain' --data-binary "$1" "http://pgai-mon:8765/sql?format=objects"; }

echo "smoke: docker compose config"
dc --profile demo config -q

echo "smoke: up (registry ${registry}, tag ${PGAI_TAG})"
dc --profile demo up -d target-db pgai-mon grafana
docker pull -q "${curl_image}" >/dev/null

wait_healthy pgai-mon 180
wait_healthy grafana-with-datasources 180

# Liveness needs no token; data needs it.
code="$(req_noauth http://pgai-mon:8765/livez)"
[[ "${code}" == "200" ]] || { echo "smoke: FAIL: /livez -> ${code}" >&2; exit 1; }
code="$(req_noauth -X POST --data-binary 'select 1' http://pgai-mon:8765/sql)"
[[ "${code}" == "401" ]] || { echo "smoke: FAIL: /sql without token -> ${code}, want 401" >&2; exit 1; }
code="$(req_noauth -X POST -H 'Authorization: Bearer wrong-token-wrong-token' --data-binary 'select 1' http://pgai-mon:8765/sql)"
[[ "${code}" == "401" ]] || { echo "smoke: FAIL: /sql with a wrong token -> ${code}, want 401" >&2; exit 1; }
out="$(sql 'select 42 as answer')"
[[ "${out}" == 200\ * && "${out}" == *'"answer":42'* ]] || { echo "smoke: FAIL: select 42 -> ${out}" >&2; exit 1; }
echo "smoke: SQL with the token: ${out}"

# The collector reached target-db through instances.yml (owner-only, read as INSTANCE_JOBS_USER).
deadline=$((SECONDS + collect_timeout))
while :; do
  # Series of the target's own metrics (pgai-mon's self-metrics name the source too: excluded).
  out="$(sql "select count(*) as n from series where metric not like 'pgai_mon%' and cast(labels as varchar) like '%target_database%'")"
  n="$(printf '%s' "${out}" | sed -n 's/.*"n":\([0-9][0-9]*\).*/\1/p')"
  if [[ -n "${n}" && "${n}" -gt 0 ]]; then
    echo "smoke: ${n} series collected from target_database"
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "smoke: FAIL: no series from target_database after ${collect_timeout} s (last: ${out})" >&2
    exit 1
  fi
  sleep 10
done

echo "smoke: OK"
