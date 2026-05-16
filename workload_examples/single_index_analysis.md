## workload example for single index analysis dashboard

This example prepares and runs a repeatable workload designed for the postgres_ai monitoring “Single index analysis” dashboard. It also shows how to deploy `pg_index_pilot`, generate controlled index bloat, and let `pg_index_pilot` automatically rebuild indexes when bloat exceeds the configured threshold during periodic runs.

### prerequisites

- Postgres instance
- `pg_cron` extension available for scheduling periodic execution
- `pgbench` installed for workload generation

## prepare the dataset in the target database

Create a table with several indexes and populate 10 million rows in the target database (e.g., `workloaddb`). This schema uses `test_pilot` schema and `items` table.

```bash
psql -U postgres -d workloaddb <<'SQL'
drop table if exists test_pilot.items cascade;
drop sequence if exists test_pilot.items_id_seq;

create sequence test_pilot.items_id_seq as bigint;

create table test_pilot.items (
  id bigint primary key default nextval('test_pilot.items_id_seq'::regclass),
  email text not null,
  status text not null,
  data jsonb,
  created_at timestamptz not null default now(),
  amount numeric(12,2) not null default 0,
  category integer not null default 0,
  updated_at timestamptz
);

alter sequence test_pilot.items_id_seq owned by test_pilot.items.id;

create index items_updated_at_idx on test_pilot.items(updated_at);
create index items_category_idx on test_pilot.items(category);
create index items_status_idx on test_pilot.items(status);
create index items_created_at_idx on test_pilot.items(created_at);
create index items_email_idx on test_pilot.items(email);
create index idx_items_data_gin on test_pilot.items using gin (data);

insert into test_pilot.items (email, status, data, created_at, amount, category, updated_at)
select
  'user'||g||'@ex',
  (g % 10)::text,
  jsonb_build_object('k', g),
  now() - (g % 1000) * interval '1 sec',
  (g % 1000) / 10.0,
  (g % 10),
  now()
from generate_series(1, 10000000) g;

select setval('test_pilot.items_id_seq', (select coalesce(max(id),0) from test_pilot.items));
SQL
```

### deploy pg_index_pilot

```bash
# Clone the repository
git clone https://gitlab.com/postgres-ai/pg_index_pilot
cd pg_index_pilot

# 1) Create the control database
psql -U postgres -c "create database index_pilot_control;"

# 2) Install required extensions in the control database
psql -U postgres -d index_pilot_control -c "create extension if not exists postgres_fdw;"
psql -U postgres -d index_pilot_control -c "create extension if not exists dblink;"

# 3) Install schema and functions in the control database
psql -U postgres -d index_pilot_control -f index_pilot_tables.sql
psql -U postgres -d index_pilot_control -f index_pilot_functions.sql
psql -U postgres -d index_pilot_control -f index_pilot_fdw.sql
```

### register the target database via FDW

Replace placeholders with actual connection details for your target database (the database where workload and indexes live; in examples below it is `workloaddb`).

```sql
psql -U postgres -d index_pilot_control <<'SQL'
create server if not exists target_workloaddb foreign data wrapper postgres_fdw
  options (host '127.0.0.1', port '5432', dbname 'workloaddb');

create user mapping if not exists for current_user server target_workloaddb
  options (user 'postgres', password 'your_password');

insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
values ('workloaddb', '127.0.0.1', 5432, 'target_workloaddb', true)
on conflict (database_name) do update
  set host = excluded.host,
      port = excluded.port,
      fdw_server_name = excluded.fdw_server_name,
      enabled = true;
SQL
```

Verify environment readiness:

```bash
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_fdw_security_status();"
psql -U postgres -d index_pilot_control -c "select * from index_pilot.check_environment();"
```

### schedule periodic runs with pg_cron (run from the primary database)

Install `pg_cron` in the primary database (e.g., `postgres`) and schedule execution of `index_pilot.periodic` in the control database using `cron.schedule_in_database`:

```sql
select cron.schedule_in_database(
    'pg_index_pilot_daily',
    '0 2 * * *',
    'call index_pilot.periodic(real_run := true);',
    'index_pilot_control'  -- run in control database
);
```

Behavior: when `index_pilot.periodic(true)` runs, it evaluates index bloat in the registered target database(s). If bloat for an index exceeds the configured `index_rebuild_scale_factor` at the time of a run, an index rebuild is initiated.

### run the workload with pgbench

Use two concurrent pgbench jobs: one generates updates that touch ranges of `id` and another performs point-lookups by `id`. This mix creates index bloat over time; when bloat exceeds the configured threshold during a periodic run, `pg_index_pilot` triggers a rebuild.

1) Create workload scripts on the machine where `pgbench` runs:

```bash
cat >/root/workload/update.sql <<'SQL'
\set id random(1,10000000)
update test_pilot.items
set updated_at = clock_timestamp()
where id between :id and (:id + 100);
SQL

cat >/root/workload/longselect.sql <<'SQL'
\set id random(1,10000000)

select 1 from test_pilot.items where id = :id;

\sleep 300s
SQL
```

2) Start pgbench sessions against the target database (example: `workloaddb`):

```bash
# Updates at limited rate (-R 50), 4 clients, 4 threads
pgbench -n -h 127.0.0.1 -U postgres -d workloaddb -c 4 -j 4 -R 50 -P 10 -T 1000000000 -f /root/workload/update.sql

# Long selects, 2 clients, 2 threads
pgbench -n -h 127.0.0.1 -U postgres -d workloaddb -c 2 -j 2 -P 10 -T 1000000000 -f /root/workload/longselect.sql
```

Tip: run pgbench in tmux so workloads continue running after disconnects. Example:

```bash
tmux new -d -s pgbench_updates 'env PGPASSWORD=<password> pgbench -n -h 127.0.0.1 -U postgres -d workloaddb -c 4 -j 4 -R 50 -P 10 -T 1000000000 -f /root/workload/update.sql'
# Optional: run long selects in another tmux session
tmux new -d -s pgbench_selects 'env PGPASSWORD=<password> pgbench -n -h 127.0.0.1 -U postgres -d workloaddb -c 2 -j 2 -P 10 -T 1000000000 -f /root/workload/longselect.sql'
```

Let these processes run continuously. The updates will steadily create index bloat; on each scheduled run, `index_pilot.periodic(true)` evaluates bloat and, if thresholds are exceeded, initiates index rebuilds.

### monitor results

- In the postgres_ai monitoring included with this repository, use:
  - `Single index analysis` for targeted inspection


