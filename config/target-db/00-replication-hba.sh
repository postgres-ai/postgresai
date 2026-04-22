#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

hba_rule="host replication replicator samenet scram-sha-256"
grep -qF "${hba_rule}" "${PGDATA}/pg_hba.conf" || echo "${hba_rule}" >> "${PGDATA}/pg_hba.conf"
