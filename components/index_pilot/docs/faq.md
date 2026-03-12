## FAQ

### Installer (index_pilot.sh)

- **How do I install quickly?** - Use the installer.
```bash
PGPASSWORD='your_password' ./index_pilot.sh install-control -H <host> -U <user> -C <control_db>
PGPASSWORD='your_password' ./index_pilot.sh register-target -H <host> -U <user> -C <control_db> -T <db> --fdw-host <target_host>
PGPASSWORD='your_password' ./index_pilot.sh verify -H <host> -U <user> -C <control_db>
```

- **What are the common flags and defaults?**
  - `-H/--host`: PostgreSQL host
  - `-P/--port`: Port 
  - `-U/--user`: User 
  - `-W/--password`: Password (prefer `PGPASSWORD` env)
  - `-C/--control-db`: Control DB
  - `--fdw-host`: Host used inside FDW (default same as `--host`)
  - `--no-create-db`: Do not auto-create control DB
  - `-q/--quiet`: Quieter `psql` output

- **Managed services specifics (RDS/Aurora, etc.)?**
  - Installer creates user mappings for the current user on the FDW server.
  - Use `--fdw-host` reachable from the DB server/container network.

- **How do I uninstall?**
```bash
PGPASSWORD='your_password' ./index_pilot.sh uninstall -H <host> -U <user> -C <control_db> --drop-servers
```

### Bloat

- **How is bloat measured?** - Using the Boguk formula: factor (x) = current index size over the best-known size-per-tuple baseline (see [Architecture](architecture.md#bloat-detection-formula)).
```sql
-- Factor, not percent (2.0 = 2x)
-- estimated_bloat = indexsize / (best_ratio * estimated_tuples)
-- where:
--   indexsize         = pg_relation_size(index)
--   estimated_tuples  = greatest(1, reltuples)
--   best_ratio        = baseline size-per-tuple learned after REINDEX
```
- Notes: `best_ratio` is updated from reindex history (`indexsize_after / greatest(1, estimated_tuples)`) and can be initialized via `do_force_populate_index_stats` for indexes larger than `minimum_reliable_index_size`. If `best_ratio` is unknown, `estimated_bloat` is null and such indexes are prioritized for reindex. Reindexing is triggered when `estimated_bloat ≥ index_rebuild_scale_factor`.

- **Why wasn't an index reindexed?** - Below `index_rebuild_scale_factor`, under `index_size_threshold`, explicitly `skip=true`, or target DB `enabled=false`.

- **How do I see current bloat for a DB?** - Query estimates and sort by bloat.
```sql
select 
  indexrelname,
  pg_size_pretty(indexsize) as current_size,
  round(estimated_bloat::numeric, 2) as bloat_x
from index_pilot.get_index_bloat_estimates('<db>')
order by estimated_bloat desc nulls last
limit 50;  -- adjust as needed
```

- **How do I change the reindex threshold?** - Set `index_rebuild_scale_factor` at the desired scope.
```sql
-- Global: rebuild when bloat ≥ 2x
select index_pilot.set_or_replace_setting(null, null, null, null, 'index_rebuild_scale_factor', '2', 'default policy');
```

- **How do I ignore small indexes?** - Raise `index_size_threshold` per DB.
```sql
select index_pilot.set_or_replace_setting('<db>', null, null, null, 'index_size_threshold', '100MB', 'skip small indexes in this DB');
```

- **How do I initialize baseline from current sizes (no reindex)?** - Pre-populate stats.
```sql
select index_pilot.do_force_populate_index_stats('<db>', null, null, null);
```

- **How do I check the effective value for a setting?** - Use the resolver.
```sql
select index_pilot.get_setting('<db>', '<schema>', '<table>', '<index>', 'index_rebuild_scale_factor');
```

### Behavior

- **What does `index_pilot.periodic()` do?** - Scans registered targets, estimates bloat, runs REINDEX CONCURRENTLY for eligible indexes, and records history.
```sql
call index_pilot.periodic(false);  -- dry run
call index_pilot.periodic(true);   -- real run
```

- **Does it operate in the control DB or target DB?** - Control DB only; it never reindexes the current DB by design.

- **Are BRIN/TOAST indexes handled?** - BRIN are skipped; TOAST handled only on safe PostgreSQL versions.

- **Can I reindex a single index?** - Yes, use manual procedure.
```sql
call index_pilot.do_reindex('<db>', '<schema>', '<table>', '<index>', false);  -- set last arg true to force
```

### Security

- **How are credentials stored?** - In `postgres_fdw` user mappings (catalog-managed), not in plain text.

- **How do I verify FDW security status?** - Run the check.
```sql
select * from index_pilot.check_fdw_security_status();
```

- **Can non-superusers operate it?** - Yes, if grants/mappings are set; verify readiness.
```sql
select * from index_pilot.check_permissions();
```

### Operations

- **How do I register target databases?** - Insert into inventory.
```sql
insert into index_pilot.target_databases(database_name, host, port, fdw_server_name)
values ('<target_db_name>', '<target_host>', 5432, 'target_<target_db_name>');
```

- **How do I schedule runs with pg_cron?** - Schedule in the cron database.
```sql
select cron.schedule_in_database(
  'pg_index_pilot_daily', '0 2 * * *',
  'call index_pilot.periodic(real_run := true);',
  '<index_pilot_control_db>'
);
```

- **How do I pause all reindexing immediately?** - Set global skip.
```sql
select index_pilot.set_or_replace_setting(null, null, null, null, 'skip', 'true', 'global pause');
```

- **How do I disable or enable a specific target DB?** - Toggle `enabled`.
```sql
update index_pilot.target_databases set enabled = false where database_name = '<db>';
update index_pilot.target_databases set enabled = true  where database_name = '<db>';
```

- **How do I uninstall from the control DB?** - Run the provided script.
```bash
# WARNING: removes schema and history in the specified database
psql -h <host> -U <user> -d <index_pilot_control_db> -f uninstall.sql
```

### Performance

- **How do I throttle work?** - Increase `index_size_threshold`, lower frequency via scheduler, or set per-DB `skip=true` temporarily.

- **How do I minimize lock contention?** - Run off-peak, pause specific targets during peak, and use default REINDEX CONCURRENTLY behavior.

- **How do I monitor progress and recent actions?** - Check current work and history.
```sql
select * from index_pilot.current_processed_index order by mtime desc; -- in-progress
select * from index_pilot.history order by ts desc limit 50;  -- recent events
```

### Roadmap

- **Where is the roadmap?** - See the Roadmap section in the root `README.md`.

- **How to propose features or report issues?** - Open an issue or PR with diagnostics from `docs/runbook.md` Escalation section.

