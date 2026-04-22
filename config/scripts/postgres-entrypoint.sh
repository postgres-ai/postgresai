#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Wrapper around the official Postgres image entrypoint.
#
# Copies optional init files from a config subdirectory into
# /docker-entrypoint-initdb.d/ and then execs docker-entrypoint.sh.
#
# Usage:
#   postgres-entrypoint.sh <config_subdir> [postgres args...]
#
# Example:
#   postgres-entrypoint.sh target-db -c shared_preload_libraries=pg_stat_statements

CONFIGS_ROOT="${PGAI_CONFIGS_ROOT:-/postgres_ai_configs}"
INITDB_DIR="${PGAI_INITDB_DIR:-/docker-entrypoint-initdb.d}"

copy_init_files() {
  local src_dir="$1"
  local pattern

  if [[ ! -d "${src_dir}" ]]; then
    return 0
  fi

  shopt -s nullglob
  for pattern in "${src_dir}"/*.sh "${src_dir}"/*.sql; do
    cp -f -- "${pattern}" "${INITDB_DIR}/"
  done
  shopt -u nullglob
}

# Fail fast with an actionable message if the on-disk PGDATA major version
# does not match the container image. Without this, postgres crash-loops
# with only "database files are incompatible with server" buried in the
# logs, which hides the real upgrade-path instruction from operators
# running `postgres-ai mon update` against an existing PG 15 volume.
check_pgdata_version() {
  local data_dir="${PGDATA:-/var/lib/postgresql/data}"
  local version_file="${data_dir}/PG_VERSION"

  if [[ ! -f "${version_file}" ]]; then
    return 0  # empty PGDATA: initdb will run on first start
  fi

  local on_disk image
  on_disk="$(tr -d '[:space:]' < "${version_file}")"
  image="$(postgres --version | awk '{print $3}' | cut -d. -f1)"

  if [[ -n "${on_disk}" && "${on_disk}" != "${image}" ]]; then
    cat >&2 <<EOF
=======================================================================
PostgreSQL version mismatch — refusing to start.

  Data directory PG version: ${on_disk}
  Container image PG version: ${image}

PostgreSQL on-disk format is not compatible across major versions.
Starting postgres now would crash-loop. The data is NOT lost; it is
simply unreadable by this image version.

To upgrade an existing PG ${on_disk} data directory to PG ${image}:
  1. Stop this container.
  2. Run pg_upgrade per the PostgresAI runbook (tracked in MR !145),
     OR start the previous PG ${on_disk} image, take a logical dump,
     then restore into a fresh PG ${image} volume.

For standbys: do NOT pg_upgrade the replica. Instead, pg_upgrade the
primary first, then re-clone the standby via pg_basebackup from the
upgraded primary.
=======================================================================
EOF
    exit 3
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    echo "postgres-entrypoint: missing <config_subdir>" >&2
    exit 2
  fi

  local config_subdir="$1"
  shift

  check_pgdata_version
  copy_init_files "${CONFIGS_ROOT}/${config_subdir}"

  exec docker-entrypoint.sh postgres "$@"
}

main "$@"
