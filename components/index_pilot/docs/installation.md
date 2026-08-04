## Installation

### Quick install via index_pilot.sh

```bash
# Clone the repository
git clone https://gitlab.com/postgres-ai/pg_index_pilot
cd pg_index_pilot

# 1) Install into control database (auto-creates DB, installs extensions/objects)
PGPASSWORD='your_password' \
  ./index_pilot.sh install-control \
  -H your_host -U your_user -C your_control_db_name

# 2) Register a target database via FDW (secure user mapping)
PGPASSWORD='your_password' \
  ./index_pilot.sh register-target \
  -H your_host -U your_user -C your_control_db_name \
  -T your_database --fdw-host your_host

# 3) Verify installation and environment
PGPASSWORD='your_password' \
  ./index_pilot.sh verify \
  -H your_host -U your_user -C your_control_db_name

# (Optional) Uninstall
PGPASSWORD='your_password' \
  ./index_pilot.sh uninstall \
  -H your_host -U your_user -C your_control_db_name --drop-servers
```

Notes:
- Use `PGPASSWORD` to avoid echoing secrets; the script won’t print passwords.
- `--fdw-host` should be reachable from the database server itself (in Docker/CI it might be `postgres`, `127.0.0.1`, or the container IP).
- For self-hosted deployments, replace the host with `127.0.0.1`. For managed services, ensure the admin user can `create database` and `create extension`.

Security notes:
- Prefer `PGPASSWORD` over putting passwords on the command line to avoid shell history leaks.
- Restrict access to foreign servers and user mappings to admins only.

### Before you start (checklist)
- PostgreSQL ≥ 13 and ability to create database/extensions (control DB).
- Decide: CONTROL_DB, TARGET_DB, TARGET_HOST.
- If using pg_cron: add to `shared_preload_libraries`, reboot; `create extension pg_cron` in `cron.database_name`.
- Remember: FDW `user mapping` must be for `current_user` in control DB.

### Placeholders
- CONTROL_DB, TARGET_DB, TARGET_HOST, SERVER_NAME (`target_<target_db>`), CONTROL_USER/PASS, TARGET_USER/PASS

### Installer CLI reference

For the full reference of installer subcommands, options, defaults, and examples, see `docs/installer_cli_reference.md`.

### Manual installation

#### Control database setup (required)

```bash
# Clone the repository
git clone https://gitlab.com/postgres-ai/pg_index_pilot
cd pg_index_pilot

# 1. Create control database (as admin user)
psql -h your-instance.region.rds.amazonaws.com -U postgres -c "create database index_pilot_control;"

# 2. Install required extensions in control database
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "CREATE EXTENSION IF NOT EXISTS postgres_fdw;"
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "CREATE EXTENSION IF NOT EXISTS dblink;"

# 3. Install schema and functions in control database
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -f index_pilot_tables.sql
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -f index_pilot_functions.sql
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -f index_pilot_fdw.sql

# 4. Create FDW server and user mapping for the TARGET database
#    fdw_server_name must refer to a foreign server that points to the TARGET DB
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control <<'SQL'
create server if not exists target_your_database foreign data wrapper postgres_fdw
  options (host 'your-instance.region.rds.amazonaws.com', port '5432', dbname 'your_database');

-- dblink_connect(server_name) uses current_user mapping; create mapping for the user running control DB (often postgres or index_pilot)
create user mapping if not exists for current_user server target_your_database
  options (user 'remote_owner_or_role', password 'remote_password');

SQL

# 5. Register the TARGET database (links index_pilot.target_databases to your FDW server)
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control <<'SQL'
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
values ('your_database', 'your-instance.region.rds.amazonaws.com', 5432, 'target_your_database', true)
on conflict (database_name) do update
  set
    host = excluded.host,
    port = excluded.port,
    fdw_server_name = excluded.fdw_server_name,
    enabled = true;
SQL

# 6. Verify FDW and environment
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status()"
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "select * from index_pilot.check_environment();"
```

#### Self-hosted PostgreSQL example

```bash
# Clone the repository
git clone https://gitlab.com/postgres-ai/pg_index_pilot
cd pg_index_pilot

# 1. Create control database (as superuser)
psql -U postgres -c "create database index_pilot_control;"

# 2. Install required extensions in control database (as superuser)
psql -U postgres -d index_pilot_control -c "CREATE EXTENSION IF NOT EXISTS postgres_fdw;"
psql -U postgres -d index_pilot_control -c "CREATE EXTENSION IF NOT EXISTS dblink;"

# 3. Install schema and functions in control database (as superuser)
psql -U postgres -d index_pilot_control -f index_pilot_tables.sql
psql -U postgres -d index_pilot_control -f index_pilot_functions.sql
psql -U postgres -d index_pilot_control -f index_pilot_fdw.sql

# 4. Setup FDW connection infrastructure (as superuser; self-connection in control DB)
psql -U postgres -d index_pilot_control \
  -c "select index_pilot.setup_connection('127.0.0.1', 5432, 'postgres', 'postgres');"  # Use actual password

# 5. Create FDW server and user mapping for the TARGET database
psql -U postgres -d index_pilot_control <<'SQL'
create server if not exists target_your_database foreign data wrapper postgres_fdw
  options (host '127.0.0.1', port '5432', dbname 'your_database');

create user mapping if not exists for current_user server target_your_database
  options (user 'remote_owner_or_role', password 'remote_password');
SQL

# 6. Register the TARGET database (links index_pilot.target_databases to your FDW server)
psql -U postgres -d index_pilot_control <<'SQL'
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
values ('your_database', '127.0.0.1', 5432, 'target_your_database', true)
on conflict (database_name) do update
  set host=excluded.host, port=excluded.port, fdw_server_name=excluded.fdw_server_name, enabled=true;
SQL

# 7. Verify
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status()"
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_environment();"
```
### Verification checklist

Run in CONTROL_DB after installation/registration:

```sql
-- FDW basic status
select * from index_pilot.check_fdw_security_status();

-- Environment and targets
select * from index_pilot.check_environment();
select * from index_pilot.target_databases;

-- psql helpers
\des+   -- foreign servers
\deu+   -- user mappings
```

### Post‑install: initialize baseline, list candidates, and exclusions

After registering targets, you may want to initialize a bloat baseline and inspect candidates before running real reindexing.

```sql
-- Initialize baseline without reindexing (sets best_ratio for sufficiently large indexes)
select index_pilot.do_force_populate_index_stats('<TARGET_DB>', null, null, null);

-- List indexes that periodic(true) would process under current thresholds
select
  schemaname, relname, indexrelname,
  pg_size_pretty(indexsize) as size,
  round(estimated_bloat::numeric, 2) as bloat_x
from index_pilot.get_index_bloat_estimates('<TARGET_DB>')
where indexsize >= pg_size_bytes(index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'index_size_threshold'))
  and coalesce(index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'skip')::boolean, false) = false
  and (estimated_bloat is null
       or estimated_bloat >= index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'index_rebuild_scale_factor')::float)
order by estimated_bloat desc nulls first
limit 50;

-- Run a real pass when ready
call index_pilot.periodic(true,false);
```

Notes:
- Baseline means best_ratio is set to current size/tuples. Immediately after baseline, estimated_bloat is ~1.0 by definition; new bloat is detected as indexes grow from that baseline.
- Small indexes below `minimum_reliable_index_size` (default 128kB) are not assigned best_ratio by baseline to avoid noise. This does not affect candidates because `index_size_threshold` (default 10MB) filters them anyway.

Exclude service schemas if desired (e.g., TOAST or TimescaleDB internals):

```sql
-- Skip TOAST indexes in target DB
select index_pilot.set_or_replace_setting('<TARGET_DB>','pg_toast',null,null,'skip','true',null);

-- Skip TimescaleDB internal chunks
select index_pilot.set_or_replace_setting('<TARGET_DB>','_timescaledb_internal',null,null,'skip','true',null);
```

### Troubleshooting

- ERROR: user mapping not found for "postgres"
  - dblink uses mapping for control DB current_user (often `postgres`). Create mapping for that user on the relevant server (self or target), or run the function as the user that has a mapping, or use SECURITY INVOKER functions.

- ERROR: could not establish connection
  - `host` in FDW must be reachable from the PostgreSQL server (not your client). In Docker/CI use `postgres` or container IP; on RDS use the instance endpoint.

- ERROR: pg_cron can only be loaded via shared_preload_libraries
  - Add `pg_cron` to `shared_preload_libraries` (RDS: parameter group), reboot the instance, then `create extension pg_cron` in the database shown by `show cron.database_name;`.

- How to see reindex progress
  - On target DB: `select * from pg_stat_progress_create_index where command='REINDEX CONCURRENTLY';`
  - History in control DB: `select * from index_pilot.history order by ts desc limit 20;`


### AWS RDS / Aurora specifics

The core flow above works the same on AWS. The following items are specific to RDS/Aurora:

- Parameter groups and pg_cron
  - Add `pg_cron` to the instance parameter group `shared_preload_libraries` (keep existing entries, e.g. `rdsutils,pg_tle,pg_stat_statements,pg_cron`). Apply and reboot.
  - In the database shown by `show cron.database_name;`, run `create extension if not exists pg_cron;` and schedule using `cron.schedule_in_database(..., '<CONTROL_DB>')`.

- FDW endpoints and networking
  - Use the RDS instance endpoint for FDW servers (both self and target). The FDW host must be reachable from the PostgreSQL server itself, not only from your client.
  - If the control DB and target DB are on the same instance, prefer creating the control DB there for simpler networking.

- Roles and user mappings
  - `dblink_connect(server_name)` uses the mapping for the control DB `current_user`. Ensure a user mapping exists for that user to each `target_<db>` server.
  - Ensure the current_user has a user mapping for the FDW server.

- Monitoring and logs
  - Progress on target DB: `select * from pg_stat_progress_create_index where command='REINDEX CONCURRENTLY';`
  - History in control DB: `select * from index_pilot.history order by ts desc limit 20;`
  - Instance logs: use CloudWatch Logs.


