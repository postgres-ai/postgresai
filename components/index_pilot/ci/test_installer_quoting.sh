#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

export PGPASSWORD="${POSTGRES_PASSWORD:?}"
db_host="${POSTGRES_HOST:-postgres}"
db_port="${POSTGRES_PORT:-5432}"
db_user="${POSTGRES_USER:-postgres}"

for _ in $(seq 1 30); do
  if pg_isready -h "$db_host" -p "$db_port" -U "$db_user" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
pg_isready -h "$db_host" -p "$db_port" -U "$db_user" >/dev/null

real_psql=$(command -v psql)
admin_user="$db_user"
db_user="ci'\"role"
password_hex=$(printf '%s' "$POSTGRES_PASSWORD" | od -An -v -tx1 | tr -d '[:space:]')
"$real_psql" -X -v ON_ERROR_STOP=on -h "$db_host" -p "$db_port" -U "$admin_user" -d postgres \
  -v "quoted_user=$db_user" -f - <<SQL
select pg_catalog.format('create role %I login superuser password %L',
                         :'quoted_user', pg_catalog.convert_from(pg_catalog.decode('${password_hex}', 'hex'), 'UTF8'))
\gexec
SQL

capture_dir=$(mktemp -d)
trap 'rm -rf "$capture_dir"' EXIT
cat > "$capture_dir/psql" <<'SH'
#!/bin/sh
printf '%s\0' "$@" >> "$PSQL_ARGV_CAPTURE"
exec "$REAL_PSQL" "$@"
SH
chmod +x "$capture_dir/psql"
export REAL_PSQL="$real_psql" PSQL_ARGV_CAPTURE="$capture_dir/argv"
export PATH="$capture_dir:$PATH"

installer="$(dirname "$0")/../index_pilot.sh"
control_db="ci'\"control"
# In raw SQL this name runs "drop server victim". The quotes after "--" are in
# the SQL comment, so the injection still runs, and the name keeps both quotes.
live_server="x; drop server victim; --'\""

control_sql() {
  "$real_psql" -X -v ON_ERROR_STOP=on -h "$db_host" -p "$db_port" -U "$db_user" -d "$control_db" -At \
    -v "live_server=$live_server" "$@" -f -
}

register_live() {
  bash "$installer" register-target \
    -H "$db_host" -P "$db_port" -U "$db_user" -C "$control_db" \
    -T postgres --server-name "$live_server" --fdw-host localhost "$@"
}

bash "$installer" install-control \
  -H "$db_host" -P "$db_port" -U "$db_user" -C "$control_db"
# install-control does not reinstall the schema. Drop it, so the second run
# tests the "control database exists" check.
control_sql <<'SQL'
drop schema index_pilot cascade;
SQL
bash "$installer" install-control \
  -H "$db_host" -P "$db_port" -U "$db_user" -C "$control_db"

control_sql <<'SQL'
create server victim foreign data wrapper postgres_fdw;
SQL

# This path must use the installed _connect_securely function and a reachable FDW
# target, so quoted server names and passwords are tested by a real connection.
# The second run finds the server and the user mapping already present.
register_live
register_live
# --force must drop and create the named server again.
control_sql <<'SQL'
alter server :"live_server" options (set dbname 'stale');
SQL
register_live --force

target_db="db'; DROP TABLE index_pilot.target_databases; --"
server='srv"; DROP SERVER victim; --'
fdw_host="host'; DROP SCHEMA public; --"

# The crafted host cannot resolve. Registration should reach the real connection
# check and fail there, after safely storing the quoted SQL values.
if bash "$installer" register-target \
  -H "$db_host" -P "$db_port" -U "$db_user" -C "$control_db" \
  -T "$target_db" --server-name "$server" --fdw-host "$fdw_host" --force; then
  echo 'Expected the crafted FDW host connection to fail' >&2
  exit 1
fi

if grep -Fq "$POSTGRES_PASSWORD" "$PSQL_ARGV_CAPTURE"; then
  echo 'FDW password leaked into child psql arguments' >&2
  exit 1
fi

result=$(control_sql -v "control_db=$control_db" -v "db_user=$db_user" \
  -v "target_db=$target_db" -v "server=$server" -v "fdw_host=$fdw_host" -v "password=$POSTGRES_PASSWORD" <<'SQL'
select
  (select count(*) from pg_catalog.pg_database where datname = :'control_db') = 1
  and (select count(*) from index_pilot.target_databases
   where database_name = :'target_db' and host = :'fdw_host' and fdw_server_name = :'server') = 1
  and (select count(*) from pg_catalog.pg_foreign_server where srvname = :'server') = 1
  and (select count(*)
       from pg_catalog.pg_foreign_server s,
            pg_catalog.pg_options_to_table(s.srvoptions) o
       where s.srvname = :'server'
         and ((o.option_name = 'host' and o.option_value = :'fdw_host')
           or (o.option_name = 'dbname' and o.option_value = :'target_db'))) = 2
  and pg_catalog.to_regnamespace('public') is not null
  and (select count(*) from pg_catalog.pg_user_mappings
       where srvname = :'server' and usename = :'db_user'
         and umoptions @> array['password=' || :'password']) = 1
  and (select count(*) from pg_catalog.pg_foreign_server where srvname = 'victim') = 1
  and (select count(*)
       from pg_catalog.pg_foreign_server as s
       cross join lateral pg_catalog.pg_options_to_table(s.srvoptions) as o
       where s.srvname = :'live_server'
         and o.option_name = 'dbname' and o.option_value = 'postgres') = 1
  and (select count(*) from pg_catalog.pg_user_mappings
       where srvname = :'live_server' and usename = :'db_user'
         and umoptions @> array['password=' || :'password']) = 1
  and exists (
    select 1 from pg_catalog.pg_foreign_data_wrapper f,
                  pg_catalog.aclexplode(f.fdwacl) acl
    where f.fdwname = 'postgres_fdw'
      and acl.grantee = (select oid from pg_catalog.pg_roles where rolname = :'db_user')
      and acl.privilege_type = 'USAGE');
SQL
)
test "$result" = t
echo 'Installer SQL quoting integration test passed'
