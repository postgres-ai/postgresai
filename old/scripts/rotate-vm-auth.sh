#!/usr/bin/env bash
# Rotate VictoriaMetrics basic auth and restart dependent services together.

set -Eeuo pipefail
IFS=$'\n\t'

ENV_FILE="${ENV_FILE:-.env}"
USERNAME="${VM_AUTH_USERNAME:-}"
PASSWORD="${VM_AUTH_PASSWORD:-}"

read_env_value() {
  local key="$1"
  if [[ ! -f "${ENV_FILE}" ]]; then
    return 0
  fi
  awk -F= -v key="${key}" '$1 == key { value = substr($0, index($0, "=") + 1) } END { print value }' "${ENV_FILE}"
}

write_env_value() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  if [[ -f "${ENV_FILE}" ]]; then
    awk -v key="${key}" -v value="${value}" '
      BEGIN { written = 0 }
      $0 ~ "^" key "=" { print key "=" value; written = 1; next }
      { print }
      END { if (!written) print key "=" value }
    ' "${ENV_FILE}" > "${tmp}"
  else
    printf '%s=%s\n' "${key}" "${value}" > "${tmp}"
  fi
  chmod 600 "${tmp}"
  mv "${tmp}" "${ENV_FILE}"
}

if [[ ! -f docker-compose.yml ]]; then
  echo "ERROR: run from the monitoring directory containing docker-compose.yml" >&2
  exit 1
fi

if [[ -z "${USERNAME}" ]]; then
  USERNAME="$(read_env_value VM_AUTH_USERNAME)"
fi
USERNAME="${USERNAME:-vmauth}"

if [[ -z "${PASSWORD}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    PASSWORD="$(openssl rand -base64 18)"
  else
    PASSWORD="$(dd if=/dev/urandom bs=18 count=1 2>/dev/null | base64 | tr -d '\n')"
  fi
fi

if [[ -z "${PASSWORD}" ]]; then
  echo "ERROR: VM_AUTH_PASSWORD is empty" >&2
  exit 1
fi

write_env_value VM_AUTH_USERNAME "${USERNAME}"
write_env_value VM_AUTH_PASSWORD "${PASSWORD}"

docker compose up -d --force-recreate sink-prometheus grafana

echo "VM auth rotated; sink-prometheus and grafana were recreated with the same .env credentials."
