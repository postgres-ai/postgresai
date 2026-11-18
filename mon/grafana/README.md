# SQL dashboards for pgai-mon

`dashboards/` holds the 16 ported dashboards, one per dashboard in `old/config/grafana/dashboards/`.
They are generated; do not edit them by hand. Each file is one line of JSON, so a recompile
does not add thousands of lines to a diff; `python3 -m json.tool <file>` prints one for reading. Each keeps the original's panels, layout, units
and legends, and the original's uid and title, so links and bookmarks keep working. The exception is
the self-monitoring dashboard (below).

## Self-monitoring

The original charted components the pgai-mon stack does not run (postgres-exporter,
VictoriaMetrics, the sink Postgres, the Flask backend), so `self_monitoring.py` writes `Self_Monitoring_Dashboard_pgai_mon.json` (uid
`self-monitoring`) from what pgai-mon records about itself: `pgai_mon_source_up`,
`pgai_mon_source_last_success_epoch_s`, `pgai_mon_connect_failures_total` {cluster, dbname, node_name};
`pgai_mon_errors_total`, `pgai_mon_timeouts_total`, `pgai_mon_skipped_slots_total` {cluster, dbname, node_name,
metric};
`pgai_mon_metric_duration_ms`, `pgai_mon_series_cut_total`, `pgai_mon_rows_cut_total` {base labels,
metric}; `up`, `scrape_duration_seconds`, `scrape_samples_scraped` per scrape target; and the
`series` view (series and metric counts). Kept from the original: "Time series in store" (was
Victoria Metrics time series) and "Scrape duration by target". Dropped (`DROPPED` there): the
version banner, the VictoriaMetrics and sink Postgres panels.

The process itself (selfmon.rs: GET /health written as series every
`collector.self_metrics_interval_s`, no labels) is charted in the "pgai-mon process" row: ingest rate
(`pgai_mon_ingest_samples_total`, `pgai_mon_ingest_records_total`), flush duration, WAL and Parquet
bytes, store files and known series, process and reader RSS, process CPU, reader restarts and memory
kills, late flushes, S3 upload lag, backlog and failed rounds (`pgai_mon_s3_*`, S3 stores only), cache
hit rate by cache (`pgai_mon_cache_hits_total`, `pgai_mon_cache_misses_total`) and query admission. Kept
for them: "Application memory usage" (3), "Victoria Metrics ingestion rate" (20) and "storage size" (22).
The monitoring host itself is charted in the "Host" and "Containers" rows, with the original's
panels, from the compose services `self-node-exporter` and `self-cadvisor`, which pgai-mon scrapes
every 30 s (jobs of the same names in `../docker/pgai-mon.yml`, keeping only the series charted):
CPU by mode, memory, filesystems and disk I/O from `node_*`; the host's network from cAdvisor's
root container (`id="/"`), since node-exporter outside the host network namespace sees only its
own; per container CPU, memory, network and disk I/O from `container_*`. `pgai_mon_source_*` also carry the source's
`cluster` and `node_name` (when set), so two sources with the same database label are told apart.

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

The grouped ASH charts use `ash_by.sql`'s `r(k,t,value)` and `names(k,labels)` relations.
The compiler formats each group's legend in a materialized `rendered` relation before joining
the time grid. Keep that work before the join: formatting every point is costly for many query IDs.
The SQL template's final SELECT is also used directly by the evaluator comparison cases.

## Inputs

- `old/config/grafana/dashboards/Dashboard_<n>_*.json`: the PromQL originals (read only);
- `cases.py`: maps each PromQL target to its ported SQL file and parameters;
- `../sql/reports/`: the ported SQL (`panels/`: the tables that were not PromQL).

## Recompile

`compile.py` needs a running pgai-mon with some data (any store): it runs each target's SQL
once with `LIMIT 0` to learn its output columns, and reads a metric's label keys for table
panels. Standard library Python only. A table panel's columns are the label keys its metric
has in the probe store (plus the fields its transformations name), so the emitted JSON depends
on the probe: collect with the tags a deployment has (`instances.yml` adds
`sink_type: prometheus`, which Dashboards 5 and 6 show as a column), and a metric without
series in the probe (e.g. Dashboard 7's blocked autovacuum workers) gets only the named fields.

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

# compile.py also writes the self-monitoring dashboard; alone (no pgai-mon needed):
python3 mon/grafana/self_monitoring.py

# offline checks of the emitted JSON (no pgai-mon, no Grafana)
python3 mon/grafana/test_compile.py
```

Ranges of 7 days or more pin the step to the 1 h rollup tier (90 days or more: 1 d), so long
ranges read rollups instead of raw samples; there `$__rate_interval` equals the step.

Instant targets (stat and table panels) evaluate one point at the range end (aligned down to
`$__interval`) with a step of 5 minutes: Grafana's Prometheus datasource sends an instant query
without a step and VictoriaMetrics then uses 5 minutes, so a bare selector looks back 5 minutes
(settings are sampled every 5 minutes, most metrics every 30-60 s). `$__interval` and
`$__rate_interval` written in the PromQL keep their values.

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

