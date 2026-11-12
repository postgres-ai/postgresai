# SQL dashboards for pgai-mon

`dashboards/` holds the 14 ported dashboards (every dashboard in
`config/grafana/dashboards/` except `Dashboard_99_Host_stats` and `Self_Monitoring_Dashboard`).
They are generated; do not edit them by hand. Each keeps the original's panels, layout, units
and legends, with a new uid (`pgai-mon-*`) and the title suffix ` (pgai-mon)`.

## Datasource

Every target uses the Infinity datasource (`yesoreyeram-infinity-datasource`, uid
`aerffb0z8rjlsc`, already provisioned in `config/grafana/provisioning/datasources/datasources.yml`).
A target POSTs its SQL as a `text/plain` body to `http://pgai-mon:8765/sql?format=objects`,
with Grafana variables (`${__from}`, `${__to}`, `${__interval_ms}`, `${cluster_name}`,
`${node_name}`, `${db_name}`, ...) interpolated into the SQL. Three panels (Dashboards 2, 8
and 10) still call the Flask backend at `http://flask-pgss-api:8000`.

## Inputs

- `config/grafana/dashboards/Dashboard_<n>_*.json`: the PromQL originals (read only);
- `cases.py`: maps each PromQL target to its ported SQL file and parameters;
- `../sql/reports/`: the ported SQL.

## Recompile

`compile.py` needs a running pgai-mon with some data (any store): it runs each target's SQL
once with `LIMIT 0` to learn its output columns, and reads a metric's label keys for table
panels. Standard library Python only.

```bash
# from the repository root, with pgai-mon serving on 127.0.0.1:8765
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
