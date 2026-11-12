# SQL dashboards for pgai-mon

`dashboards/` holds the 14 ported dashboards (every dashboard in
`old/config/grafana/dashboards/` except `Dashboard_99_Host_stats` and `Self_Monitoring_Dashboard`).
They are generated; do not edit them by hand. Each keeps the original's panels, layout, units
and legends, with a new uid (`pgai-mon-*`) and the title suffix ` (pgai-mon)`.

## Datasource

Every target uses the Infinity datasource `pgai-mon` (`yesoreyeram-infinity-datasource`, uid
`pgai-mon`, provisioned by `config/grafana/provisioning/datasources/pgai-mon.yml` with the bearer
token from `PGAI_MON_TOKEN`). A target POSTs its SQL as a `text/plain` body to
`http://pgai-mon:8765/sql?format=objects`, with Grafana variables (`${__from}`, `${__to}`,
`${__interval_ms}`, `${cluster_name:sqlstring}`, `${db_name:regex}`, ...) interpolated into the
SQL. A variable holding one value is written `${x:sqlstring}`, so Grafana quotes it and doubles a
quote inside it; a filter matched as a regex is written `'${x:regex}'`. No panel calls the Flask backend
or the pgwatch Postgres sink: the four tables that did (Dashboard 2's pg_stat_statements table,
Dashboard 3's query text, Dashboard 8's table stats, Dashboard 10's B-tree bloat) run the SQL in
`../sql/reports/panels/` (`TABLE_PANELS` in `compile.py`), keeping their column names.

## Inputs

- `old/config/grafana/dashboards/Dashboard_<n>_*.json`: the PromQL originals (read only);
- `cases.py`: maps each PromQL target to its ported SQL file and parameters;
- `../sql/reports/`: the ported SQL (`panels/`: the tables that were not PromQL).

## Recompile

`compile.py` needs a running pgai-mon with some data (any store): it runs each target's SQL
once with `LIMIT 0` to learn its output columns, and reads a metric's label keys for table
panels. Standard library Python only.

```bash
# from the repository root, with pgai-mon serving on 127.0.0.1:8765
# (its bearer token in PGAI_MON_TOKEN, or pgai-mon started with --no-auth)
python3 mon/grafana/compile.py --probe-url http://127.0.0.1:8765
#   --target-url URL   the URL baked into targets (default http://pgai-mon:8765/sql?format=objects)
#   --out DIR          default mon/grafana/dashboards

# also run every emitted query and variable against the probe and compare with the
# reference inlining of cases.py (exit 1 on any failure)
python3 mon/grafana/compile.py --probe-url http://127.0.0.1:8765 --check \
    [--instance cluster,node,db,second_db]

# offline checks of the emitted JSON (no pgai-mon, no Grafana)
python3 mon/grafana/test_compile.py
```

Ranges of 7 days or more pin the step to the 1 h rollup tier (90 days or more: 1 d), so long
ranges read rollups instead of raw samples; there `$__rate_interval` equals the step.

## Check every panel

`check_panels.py` runs each panel the way Grafana would: the dashboard's default range ending at
`--now-ms` (default: the newest sample), `$__interval` from the range and the panel's
`maxDataPoints` / min interval, the template variables in order (each query variable through
`/sql`, its current value if among the options, else the first one; All as Grafana expands it),
then every target's SQL with the variables interpolated. Each panel is classified `data`,
`empty` or `error`; the exit status is 1 on any error (or on a mismatch with `--expect`).

```bash
python3 mon/grafana/check_panels.py --url http://127.0.0.1:8765 [--range 7d] \
    [--var db_name=mydb --var query_id=123] [--only Dashboard_8_:45] [--json out.json]
```

CI runs it against a small fixture store (`tests/grafana_dashboards/test_sql_panels_run.py`,
with `PGAI_MON_BIN` pointing at a built binary); every panel must show data there.

