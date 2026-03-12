# pg_index_pilot – autonomous index lifecycle management for Postgres

The purpose of `pg_index_pilot` is to provide all tools needed to manage indexes in Postgres in most automated fashion.

This project is in its very early stage. We start with most boring yet extremely important task: automatic reindexing ("AR") to mitigate index bloat, supporting any types of indexes, and then expand to other areas of index health. And then expand to two other big areas – automated index removal ("AIR") and, finally, automated index creation and optimization ("AIC&O"). It is a part of the Self‑driving Postgres, but can be used independently as a standalone tool.

Docs: [Installation](docs/installation.md) | [Runbook](docs/runbook.md) | [FAQ](docs/faq.md) | [Function reference](docs/function_reference.md) | [Architecture](docs/architecture.md)

## What it is for

- Automated index lifecycle management for PostgreSQL, starting with automatic reindexing to keep index bloat under control without manual work.

## Key principles

- Simplicity and full embed: implemented entirely inside PostgreSQL (PL/pgSQL), no external services required to rebuild indexes.
- Works everywhere PostgreSQL runs (including managed platforms) — all logic lives in the database.
- No superuser requirement for day‑to‑day operations; designed to run under owner/privileged roles in control DB and target DBs.
- Scheduling inside the database via `pg_cron` — no EC2/Lambda or other external orchestrators needed.
- Supports reindexing of all common index types (btree, hash, gin, gist, spgist); brin is currently excluded.
- Control DB orchestrates multiple target databases via `postgres_fdw`/`dblink`; reindexing is executed with `reindex concurrently` to minimize locking.

See [Architecture](docs/architecture.md) for detailed design decisions and requirements.

## Table of contents

- [Roadmap](#roadmap)
- [Automated reindexing](#automated-reindexing)
- [Requirements](#requirements)
- [Recommendations](#recommendations)
- [Installation](#installation)
  - [Control database setup (required)](#control-database-setup-required)
  - [Self-hosted PostgreSQL Example](#self-hosted-postgresql-example)
- [Initial launch](#initial-launch)
- [Scheduling automated maintenance](#scheduling-automated-maintenance)
  - [Choosing the right schedule](#choosing-the-right-schedule)
  - [Using pg_cron (Recommended)](#using-pg_cron-recommended)
  - [Using external cron](#using-external-cron)
- [Uninstalling pg_index_pilot](#uninstalling-pg_index_pilot)
- [Updating pg_index_pilot](#updating-pg_index_pilot)
- [Monitoring and Analysis](#monitoring-and-analysis)
  - [View reindexing history](#view-reindexing-history)
  - [Check current bloat status](#check-current-bloat-status)

## Roadmap

The roadmap covers three big areas:

1. [ ] "AR": Automated Reindexing
    1. [x] Maxim Boguk's bloat estimation formula – works with any type of index, not only btree
        1. [x] original implementation (`pg_index_pilot`) – requires initial full reindex
        2. [x] non-superuser mode for cloud databases (AWS RDS, Google Cloud SQL, Azure)
        3. [x] flexible connection management for dblink
        4. [ ] API for stats obtained on a clone (to avoid full reindex on prod primary)
    2. [ ] Traditional bloat estimation (ioguix; btree only)
    3. [ ] Exact bloat analysis (pgstattuple; analysis on clones)
    4. [x] Tested on managed services
        - [x] RDS and Aurora (see AWS specifics in Installation: docs/installation.md#aws-rds--aurora-specifics)
        - [ ] CloudSQL
        - [x] Supabase
        - [ ] Crunchy Bridge
        - [ ] Azure
    5. [ ] Integration with postgres_ai monitoring
    6. [ ] Resource-aware scheduling, predictive maintenance windows (when will load be lowest?)
    7. [ ] Coordination with other ops (backups, vacuums, upgrades)
    8. [ ] Parallelization and throttling (adaptive)
    9. [ ] Predictive bloat modeling
    10. [ ] Learning & Feedback Loops: learning from past actions, A/B testing and "what-if" simulation (DBLab)
    11. [ ] Impact estimation before scheduling
    12. [ ] RCA of fast degraded index health (why it gets bloated fast?) and mitigation (tune autovacuum, avoid xmin horizon getting stuck)
    13. [ ] Self-adjusting thresholds
2. [ ] "AIR": Automated Index Removal
    1. [ ] Unused indexes
    2. [ ] Redundant indexes
    3. [ ] Invalid indexes (or, per configuration, rebuilding them)
    4. [ ] Advanced scoring; suboptimal / rarely used indexes cleanup; self-adjusting thresholds
    5. [ ] Forecasting of index usage; seasonal pattern recognition
    6. [ ] Impact estimation before removal; "what-if" simulation (DBLab)
3. [ ] "AIC&O": Automated Index Creation & Optimization
    1. [ ] Index recommendations (including multi-column, expression, partial, hybrid, and covering indexes)
    2. [ ] Index optimization according to configured goals (latency, size, WAL, write/HOT overhead, read overhead)
    3. [ ] Experimentation (hypothetical with HypoPG, real with DBLab)
    4. [ ] Query pattern classification
    5. [ ] Advanced scoring; cost/benefit analysis
    6. [ ] Impact estimation before operations; "what-if" simulation (DBLab)

## Automated reindexing

The framework of reindexing is implemented entirely inside Postgres, using:
- PL/pgSQL functions and stored procedures with transaction control
- [dblink](https://www.postgresql.org/docs/current/contrib-dblink-function.html) to execute `REINDEX CONCURRENTLY` – because it cannot be inside a transaction block)
- [pg_cron](https://github.com/citusdata/pg_cron) for scheduling

---


## Requirements

- PostgreSQL version 13.0 or higher
- **IMPORTANT:** Requires ability to create database (not supported on TigerData, formerly Timescale Cloud)
- Separate control database (`index_pilot_control`) to manage target databases
- `dblink` and `postgres_fdw` extensions installed in control database
- Database owner or user with appropriate permissions
- Works with AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL (where database creation is allowed)
- Manages multiple target databases from single control database
- Uses REINDEX CONCURRENTLY from control database (avoids deadlocks)

## Recommendations 
- If server resources allow set non-zero `max_parallel_maintenance_workers` (exact amount depends on server parameters).
- To set `wal_keep_segments` to at least `5000`, unless the WAL archive is used to support streaming replication.

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
- For self-hosted replace host with `127.0.0.1`. For managed services ensure the admin user can `create database` and `create extension`.

### Before you start (checklist)
- PostgreSQL ≥ 13 and ability to create database/extensions (control DB).
- Decide: CONTROL_DB name, TARGET_DB name, TARGET_HOST (reachable from Postgres server, not only from client).
- If you plan to use pg_cron: ensure it’s in `shared_preload_libraries` (RDS: parameter group + reboot), and `create extension pg_cron` in `cron.database_name`.
- The FDW user mapping is looked up for the `current_user` in the control DB session. Create mapping for that user.

### Placeholders used below
- CONTROL_DB, TARGET_DB, TARGET_HOST, SERVER_NAME (e.g. `target_<target_db>`)
- CONTROL_USER/PASS (user running commands in control DB)
- TARGET_USER/PASS (user in the target DB; typically an owner or a role with owner rights)

### Key concepts
- `target_<db>`: FDW server that points to the target database. This name goes to `index_pilot.target_databases.fdw_server_name`.
- A user mapping must exist for `current_user` (in the control DB) to each `target_<db>` server you intend to use.

### Security Note

**CRITICAL**: Never use hardcoded passwords in production. The `setup_01_user.psql` script requires a secure password to be provided via psql variable:

```bash
# Generate secure random password
RANDOM_PWD=$(openssl rand -base64 32)

# Use the secure setup script (recommended)
./setup_user_secure.sh

# Or run manually with secure password
psql -f setup_01_user.psql -v index_pilot_password="$RANDOM_PWD"
echo "Generated password: $RANDOM_PWD"
```

### Manual installation

#### Control database setup (Required)

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
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control <<'SQL'
create server if not exists target_<your_database> foreign data wrapper postgres_fdw
  options (host 'your-instance.region.rds.amazonaws.com', port '5432', dbname 'your_database');

-- dblink_connect(server_name) uses current_user user mapping; create mapping for the user running control DB (often postgres or index_pilot)
create user mapping if not exists for current_user server target_<your_database>
  options (user 'remote_owner_or_role', password 'remote_password');
SQL

# 5. Register the TARGET database (links index_pilot.target_databases to your FDW server)
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control <<'SQL'
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
values ('your_database', 'your-instance.region.rds.amazonaws.com', 5432, 'target_your_database', true)
on conflict (database_name) do update
  set host=excluded.host, port=excluded.port, fdw_server_name=excluded.fdw_server_name, enabled=true;
SQL

# 6. Verify FDW and environment
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status();"
psql -h your-instance.region.rds.amazonaws.com -U postgres -d index_pilot_control -c "select * from index_pilot.check_environment();"
```

#### Self-hosted PostgreSQL Example

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

# 4. Create FDW server and user mapping for the TARGET database
psql -U postgres -d index_pilot_control <<'SQL'
create server if not exists target_your_database foreign data wrapper postgres_fdw
  options (host '127.0.0.1', port '5432', dbname 'your_database');

create user mapping if not exists for current_user server target_your_database
  options (user 'remote_owner_or_role', password 'remote_password');
SQL

# 5. Register the TARGET database
psql -U postgres -d index_pilot_control <<'SQL'
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
values ('your_database', '127.0.0.1', 5432, 'target_your_database', true)
on conflict (database_name) do update
  set host=excluded.host, port=excluded.port, fdw_server_name=excluded.fdw_server_name, enabled=true;
SQL

# 6. Verify
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status();"
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_environment();"
```

## Initial launch

**⚠️ IMPORTANT:** During the first run, all indexes larger than index_size_threshold (default: 10MB) will be analyzed and potentially rebuilt. This process may take hours or days on large databases.

For manual initial run:

```bash
# Set credentials
export PGSSLMODE=require
export PGPASSWORD='your_index_pilot_password'

# Run initial analysis and reindexing
nohup psql -h your_host -U index_pilot -d your_database \
  -qXt -c "call index_pilot.periodic(true)" >> index_pilot.log 2>&1
```

## Scheduling automated maintenance

### Choosing the right schedule

The optimal maintenance schedule depends on your database characteristics:

**Daily maintenance (recommended for):**
- High-traffic databases with frequent updates
- Databases where index bloat accumulates quickly
- Systems with sufficient maintenance windows each night
- When you want to catch and fix bloat early

**Weekly maintenance (recommended for):**
- Stable databases with predictable workloads
- Systems where index bloat accumulates slowly
- Production systems where daily maintenance might be disruptive
- Databases with limited maintenance windows

### Using pg_cron (Recommended)

**Step 1: Check where pg_cron is installed**
```sql
-- Find which database has pg_cron
show cron.database_name;
```

**Step 2: Schedule jobs from the pg_cron database**

```sql
-- Connect to the database shown in step 1
\c postgres_ai  -- or whatever cron.database_name shows

-- Daily maintenance at 2 AM
select cron.schedule_in_database(
    'pg_index_pilot_daily',
    '0 2 * * *',
    'call index_pilot.periodic(real_run := true);',
    'index_pilot_control'  -- Run in control database
);

-- Monitoring every 6 hours (no actual reindex)
select cron.schedule_in_database(
    'pg_index_pilot_monitor',
    '0 */6 * * *',
    'call index_pilot.periodic(real_run := false);',
    'index_pilot_control'
);

-- OR weekly maintenance on Sunday at 2 AM
select cron.schedule_in_database(
    'pg_index_pilot_weekly',
    '0 2 * * 0',
    'call index_pilot.periodic(real_run := true);',
    'index_pilot_control'
);
```

**Step 3: Verify and manage schedules**
```sql
-- View scheduled jobs
select jobname, schedule, command, database, active 
from cron.job 
where jobname like 'pg_index_pilot%';

-- Disable a schedule
select cron.unschedule('pg_index_pilot_daily');

-- Change schedule time
select cron.unschedule('pg_index_pilot_daily');
select cron.schedule_in_database(
    'pg_index_pilot_daily', 
    '0 3 * * *',  -- New time: 3 AM
    'call index_pilot.periodic(real_run := true);',
    'index_pilot_control'
);
```

### Using external cron

Create a maintenance script:
```bash
# Runs reindexing only on primary (all databases)
psql -d postgres -AtqXc "select not pg_is_in_recovery()" | grep -qx t || exit; psql -d postgres -qt -c "call index_pilot.periodic(true);"
```

Add to crontab:
```cron
# Runs reindexing daily at 2 AM (only on primary)
0 2 * * * /usr/local/bin/index_maintenance.sh
```

**💡 Best Practices:**
- Schedule during low-traffic periods
- Avoid overlapping with backup or other IO-intensive operations
- Consider hourly runs for high-write workloads
- Monitor resource usage during initial runs (first of all, both disk IO and CPU usage)

## Uninstalling pg_index_pilot

To completely remove pg_index_pilot from your database:

```bash
# Uninstall the tool (this will delete all collected statistics!)
psql -h your-instance.region.rds.amazonaws.com -U postgres -d your_database -f uninstall.sql

# Check for any leftover invalid indexes from failed reindexes
psql -h your-instance.region.rds.amazonaws.com -U postgres -d your_database \
  -c "select format('drop index concurrently if exists %I.%I;', n.nspname, i.relname) 
      from pg_index idx
      join pg_class i on i.oid = idx.indexrelid
      join pg_namespace n on n.oid = i.relnamespace
      where i.relname ~ '_ccnew[0-9]*$'
      and not idx.indisvalid;"

# Run any drop index commands from the previous query manually
```

**Note:** The uninstall script will:
- Remove the `index_pilot` schema and all its objects
- Remove the FDW server configuration
- List any invalid `_ccnew*` indexes that need manual cleanup
- Preserve the `postgres_fdw` extension (may be used by other tools)

## Updating pg_index_pilot

To update to the latest version:
```bash
cd pg_index_pilot
git pull

# Reload the updated functions (or reinstall completely)
psql -1 -d your_database -f index_pilot_functions.sql
psql -1 -d your_database -f index_pilot_fdw.sql
```

## Monitoring and Analysis

### View Reindexing History
```sql
-- Show recent reindexing operations with status
select 
    schemaname, relname, indexrelname,
    pg_size_pretty(indexsize_before::bigint) as size_before,
    pg_size_pretty(indexsize_after::bigint) as size_after,
    reindex_duration,
    status,
    case when error_message is not null then left(error_message, 50) else null end as error,
    entry_timestamp
from index_pilot.reindex_history 
order by entry_timestamp desc 
limit 20;

-- Show only failed reindexes for debugging
select 
    schemaname, relname, indexrelname,
    pg_size_pretty(indexsize_before::bigint) as size_before,
    reindex_duration,
    error_message,
    entry_timestamp
from index_pilot.reindex_history 
where status = 'failed'
order by entry_timestamp desc;
```

**💡 Tip:** Use the convenient `index_pilot.history` view for formatted output:
```sql
-- View recent operations with formatted sizes and status
select * from index_pilot.history limit 20;

-- View only failed operations
select * from index_pilot.history where status = 'failed';
```

### Check Current Bloat Status
```sql
-- Check bloat estimates for current database
select 
    indexrelname,
    pg_size_pretty(indexsize::bigint) as current_size,
    round(estimated_bloat::numeric, 1)||'x' as bloat_now
from index_pilot.get_index_bloat_estimates(current_database()) 
order by estimated_bloat desc nulls last 
limit 40;
```

### Baseline, candidates, and exclusions (quick reference)

```sql
-- Initialize baseline without reindex (sets best_ratio for large indexes)
select index_pilot.do_force_populate_index_stats('<TARGET_DB>', null, null, null);

-- List what periodic(true) would take under current thresholds
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

-- Exclude service schemas if desired
select index_pilot.set_or_replace_setting('<TARGET_DB>','pg_toast',null,null,'skip','true',null);
select index_pilot.set_or_replace_setting('<TARGET_DB>','_timescaledb_internal',null,null,'skip','true',null);
```

Notes:
- Baseline sets best_ratio to current size/tuples; immediately after, bloat_x ≈ 1.0 and will grow as indexes bloat.
- Small indexes (< minimum_reliable_index_size, default 128kB) skip best_ratio to avoid noise; candidates are still gated by index_size_threshold (default 10MB).


