#!/usr/bin/env bash
# Docker initdb hook: creates the 'replicator' role used by target-standby for
# streaming replication. The password is supplied via the REPLICATOR_PASSWORD
# environment variable and forwarded to psql as a session variable
# (`-v replicator_password=...`), so it is interpolated only inside the psql
# session and never appears on the Postgres command line, in postgresql.conf,
# or as a queryable server setting.
set -Eeuo pipefail
IFS=$'\n\t'

: "${REPLICATOR_PASSWORD:?REPLICATOR_PASSWORD is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"

psql \
  -v ON_ERROR_STOP=1 \
  -v "replicator_password=${REPLICATOR_PASSWORD}" \
  --no-psqlrc \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" <<'SQL'
create user replicator with replication password :'replicator_password';
SQL
