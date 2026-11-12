#!/usr/bin/env python3
"""The self-monitoring dashboard for pgai-mon (dashboards/Self_Monitoring_Dashboard_pgai_mon.json).

The original (old/config/grafana/dashboards/Self_Monitoring_Dashboard.json) charted the old stack's
components: cAdvisor containers, node-exporter host stats, VictoriaMetrics, the sink Postgres and
the Flask backend. None of them exists any more, and its PromQL has no case in cases.py, so this
dashboard is not compiled from it: it is written here from the series pgai-mon writes about itself
(collector.rs: pgai_mon_*; scrape.rs: up, scrape_*) and from the `series` view, in the same target
shape as compile.py's (time, value, series, label_set). Standard library Python only; it needs no
pgai-mon.

  python3 mon/grafana/self_monitoring.py [--out DIR]   # compile.py also writes it

Panels of the original with no counterpart (DROPPED) are listed in mon/grafana/README.md.
"""
import argparse
import copy
import json
import os

import compile as C

SRC = "Self_Monitoring_Dashboard.json"
OUT = "Self_Monitoring_Dashboard_pgai_mon.json"
UID = "pgai-mon-self-monitoring"

E = "CAST([] AS VARCHAR[][])"  # no matchers: pgai-mon's own series carry no cluster/node_name
R = "getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms')"
RI = "getvariable('rate_interval_ms')"

# the original panels with no counterpart in the new stack, and why (README.md lists them too)
DROPPED = {
    200: "PostgresAI version banner: read the Flask backend's /version (pgai-mon's version is in GET /health only)",
    1: "Active monitoring services: cAdvisor, node-exporter, postgres-exporter, VictoriaMetrics, pgwatch",
    2: "Running containers: cAdvisor",
    3: "Application memory usage: node-exporter (pgai-mon's RSS is in GET /health only)",
    4: "System CPU usage: node-exporter",
    5: "Victoria Metrics storage size (the store's Parquet bytes are in GET /health only)",
    40: "System CPU usage breakdown: node-exporter", 41: "System memory usage breakdown: node-exporter",
    42: "System network I/O: node-exporter", 43: "System disk usage: node-exporter",
    50: "Disk I/O operations: node-exporter", 51: "Disk I/O throughput: node-exporter",
    52: "Disk I/O utilization: node-exporter", 53: "Disk I/O average latency: node-exporter",
    10: "Container CPU usage: cAdvisor", 11: "Container memory usage: cAdvisor",
    12: "Container network I/O: cAdvisor", 13: "Container disk I/O: cAdvisor",
    20: "Victoria Metrics ingestion rate (samples_in is in GET /health only)",
    22: "Victoria Metrics storage size (GET /health only)", 23: "Victoria Metrics rows count (GET /health only)",
    30: "Sink Postgres connections: no sink", 31: "Sink Postgres transactions: no sink",
    32: "Sink Postgres database size: no sink", 33: "Sink Postgres block I/O: no sink",
}
# kept: 6 (time series in the store, from the series view), 21 (scrape duration by target)
KEPT = (6, 21)


def ts_defaults(unit, stack=False, **extra):
    d = {"color": {"mode": "palette-classic"},
         "custom": {"axisBorderShow": False, "axisCenteredZero": False, "axisColorMode": "text", "axisLabel": "",
                    "axisPlacement": "auto", "barAlignment": 0, "barWidthFactor": 0.6, "drawStyle": "line",
                    "fillOpacity": 10, "gradientMode": "none",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False}, "insertNulls": False,
                    "lineInterpolation": "linear", "lineWidth": 1, "pointSize": 5,
                    "scaleDistribution": {"type": "linear"}, "showPoints": "never", "spanNulls": False,
                    "stacking": {"group": "A", "mode": "normal" if stack else "none"},
                    "thresholdsStyle": {"mode": "off"}},
         "mappings": [], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
         "unit": unit}
    d.update(extra)
    return d


def stat_defaults(unit, steps, no_value=None):
    d = {"color": {"mode": "thresholds"}, "mappings": [], "thresholds": {"mode": "absolute", "steps": steps},
         "unit": unit}
    if no_value is not None:
        d["noValue"] = no_value
    return d


STAT_OPTIONS = {"colorMode": "value", "graphMode": "none", "justifyMode": "auto", "orientation": "auto",
                "percentChangeColorMode": "standard",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "showPercentChange": False, "textMode": "auto", "wideLayout": True}
GREEN = [{"color": "green", "value": None}]
RED_FROM_1 = [{"color": "green", "value": None}, {"color": "red", "value": 1}]


def legend(sort_by, calcs):
    return {"legend": {"calcs": calcs, "displayMode": "table", "placement": "bottom", "showLegend": True,
                       "sortBy": sort_by, "sortDesc": True},
            "tooltip": {"hideZeros": False, "mode": "multi", "sort": "desc"}}


# (id, type, title, description, gridPos (h, w, x, y), field defaults, options,
#  [(refId, sql, legendFormat, instant)])
PANELS = [
    ("row", 100, "Overview", 0),
    (1001, "stat", "Databases up", "Monitored databases whose last collection connected "
     "(pgai_mon_source_up = 1).", (4, 4, 0, 1), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", f"SELECT t, sum(value) AS value FROM prom_last('pgai_mon_source_up', {E}, {R}) GROUP BY t",
       "up", True)]),
    (1002, "stat", "Databases down", "Monitored databases pgai-mon cannot connect to "
     "(pgai_mon_source_up = 0).", (4, 4, 4, 1), stat_defaults("short", RED_FROM_1, "0"), STAT_OPTIONS,
     [("A", f"SELECT t, sum(1 - value) AS value FROM prom_last('pgai_mon_source_up', {E}, {R}) GROUP BY t",
       "down", True)]),
    (1003, "stat", "Collection errors in range", "Failed metric queries over the dashboard range, all "
     "databases (increase of pgai_mon_errors_total).", (4, 4, 8, 1), stat_defaults("short", RED_FROM_1, "0"),
     STAT_OPTIONS,
     [("A", f"SELECT t, value FROM prom_agg('sum', 'increase', 'pgai_mon_errors_total', {E}, {R}, "
            "getvariable('range_ms'))", "errors", True)]),
    (1004, "stat", "Scrape targets up", "Scrape targets and collector sources whose last scrape or "
     "collection succeeded (up = 1).", (4, 4, 12, 1), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", f"SELECT t, sum(value) AS value FROM prom_last('up', {E}, {R}) GROUP BY t", "up", True)]),
    (6, "stat", "Time series in store", "Distinct series the store knows (the series view), all metrics.",
     (4, 4, 16, 1), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", "SELECT getvariable('end_ms') AS t, count(DISTINCT series_id) AS value FROM series", "series",
       True)]),
    (1006, "stat", "Metrics in store", "Distinct metric names the store knows (the series view).",
     (4, 4, 20, 1), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", "SELECT getvariable('end_ms') AS t, count(DISTINCT metric) AS value FROM series", "metrics",
       True)]),
    ("row", 101, "Collection", 5),
    (1010, "timeseries", "Database collection status", "1: the database's last collection connected; "
     "0: it did not (pgai_mon_source_up).", (8, 12, 0, 6),
     ts_defaults("short", min=0, max=1), legend("Last", ["min", "last"]),
     [("A", f"SELECT labels, t, value FROM prom_last('pgai_mon_source_up', {E}, {R})", "{{dbname}}", False)]),
    (1011, "timeseries", "Time since last successful collection", "Seconds since a metric query last "
     "succeeded on the database (now - pgai_mon_source_last_success_epoch_s).", (8, 12, 12, 6),
     ts_defaults("s"), legend("Max", ["mean", "max", "last"]),
     [("A", f"SELECT labels, t, t / 1000.0 - value AS value "
            f"FROM prom_last('pgai_mon_source_last_success_epoch_s', {E}, {R})", "{{dbname}}", False)]),
    (1012, "timeseries", "Collection errors /s", "Failed metric queries per second, by database and "
     "metric (rate of pgai_mon_errors_total).", (8, 12, 0, 14), ts_defaults("short"),
     legend("Mean", ["mean", "max"]),
     [("A", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_errors_total', {E}, {R}, {RI}, "
            "p_by := ['dbname', 'metric'])", "{{dbname}}: {{metric}}", False)]),
    (1013, "timeseries", "Timeouts, skipped runs and connection failures /s", "Metric queries cancelled by "
     "statement or lock timeout, runs skipped because the previous one still ran, and failed connection "
     "attempts, per second (rates of pgai_mon_timeouts_total, pgai_mon_skipped_slots_total, "
     "pgai_mon_connect_failures_total).", (8, 12, 12, 14), ts_defaults("short"), legend("Mean", ["mean", "max"]),
     [("A", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_timeouts_total', {E}, {R}, {RI}, "
            "p_by := ['dbname', 'metric'])", "timeouts {{dbname}}: {{metric}}", False),
      ("B", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_skipped_slots_total', {E}, {R}, "
            f"{RI}, p_by := ['dbname', 'metric'])", "skipped {{dbname}}: {{metric}}", False),
      ("C", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_connect_failures_total', {E}, {R}, "
            f"{RI}, p_by := ['dbname'])", "connection failures {{dbname}}", False)]),
    (1014, "timeseries", "Metric query duration (top 10)", "Wall time of each metric's collection query, "
     "the 10 slowest series over the range (pgai_mon_metric_duration_ms).", (8, 12, 0, 22), ts_defaults("ms"),
     legend("Max", ["mean", "max", "last"]),
     [("A", "WITH x AS (\n"
            f"  SELECT labels, t, value FROM prom_last('pgai_mon_metric_duration_ms', {E}, {R})\n"
            "), top AS (\n"
            "  SELECT labels['cluster'] AS c, labels['node_name'] AS n, labels['dbname'] AS d, labels['metric'] AS m\n"
            "  FROM x GROUP BY ALL ORDER BY max(value) DESC, c, n, d, m LIMIT 10\n"
            ")\n"
            "SELECT x.labels, x.t, x.value FROM x JOIN top ON x.labels['cluster'] IS NOT DISTINCT FROM top.c\n"
            "  AND x.labels['node_name'] IS NOT DISTINCT FROM top.n AND x.labels['dbname'] IS NOT DISTINCT FROM top.d\n"
            "  AND x.labels['metric'] IS NOT DISTINCT FROM top.m", "{{dbname}}: {{metric}}", False)]),
    (1015, "timeseries", "Series and rows cut at caps (since start)", "Series dropped over "
     "collector.max_series_per_query and results cut at collector.max_rows_per_query, counted since pgai-mon "
     "started (pgai_mon_series_cut_total, pgai_mon_rows_cut_total).", (8, 12, 12, 22), ts_defaults("short"),
     legend("Last", ["max", "last"]),
     [("A", f"SELECT labels, t, value FROM prom_last('pgai_mon_series_cut_total', {E}, {R})",
       "series cut {{dbname}}: {{metric}}", False),
      ("B", f"SELECT labels, t, value FROM prom_last('pgai_mon_rows_cut_total', {E}, {R})",
       "rows cut {{dbname}}: {{metric}}", False)]),
    ("row", 102, "Scrape targets", 30),
    (1020, "timeseries", "Scrape target status", "1: the target's last scrape (or the collector source's "
     "last run) succeeded; 0: it failed (up).", (8, 8, 0, 31), ts_defaults("short", min=0, max=1),
     legend("Last", ["min", "last"]),
     [("A", f"SELECT labels, t, value FROM prom_last('up', {E}, {R})", "{{job}} {{instance}}", False)]),
    (21, "timeseries", "Scrape duration by target", "Average scrape duration by job (scrape_duration_seconds).",
     (8, 8, 8, 31), ts_defaults("s"), legend("Max", ["mean", "max", "last"]),
     [("A", "SELECT map(['job'], [labels['job']]) AS labels, t, avg(value) AS value "
            f"FROM prom_last('scrape_duration_seconds', {E}, {R}) GROUP BY ALL", "{{job}}", False)]),
    (1022, "timeseries", "Samples scraped by target", "Samples per scrape by job (scrape_samples_scraped).",
     (8, 8, 16, 31), ts_defaults("short"), legend("Last", ["mean", "max", "last"]),
     [("A", "SELECT map(['job'], [labels['job']]) AS labels, t, sum(value) AS value "
            f"FROM prom_last('scrape_samples_scraped', {E}, {R}) GROUP BY ALL", "{{job}}", False)]),
]


def target(ref, sql, fmt, instant, gp_range, gp_inst):
    cols = ["labels", "t", "value"] if "labels" in sql else ["t", "value"]  # a stat's sum has no labels
    body = C.wrap(C.inline(sql, {}, gp_inst if instant else gp_range), cols, fmt, [])
    return C.infinity_target(ref, body, False, [])


def build():
    src = json.load(open(os.path.join(C.DASH_DIR, SRC)))
    d = {k: copy.deepcopy(v) for k, v in src.items() if k not in ("panels", "templating")}
    d.update(uid=UID, title=src["title"] + " (pgai-mon)", id=None, version=1, templating={"list": []})
    gp_range, gp_inst = C.dashboard_params(d)
    panels = []
    for p in PANELS:
        if p[0] == "row":
            _, pid, title, y = p
            panels.append({"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "id": pid,
                           "panels": [], "title": title, "type": "row"})
            continue
        pid, typ, title, desc, (h, w, x, y), defaults, options, targets = p
        overrides = [{"matcher": {"id": "byFrameRefID", "options": ref},
                      "properties": [{"id": "displayName", "value": "${__field.labels.series}"}]}
                     for ref, *_ in targets]
        panels.append({"datasource": dict(C.INFINITY), "description": desc,
                       "fieldConfig": {"defaults": copy.deepcopy(defaults), "overrides": overrides},
                       "gridPos": {"h": h, "w": w, "x": x, "y": y}, "id": pid, "options": copy.deepcopy(options),
                       "pluginVersion": "12.0.2", "title": title, "type": typ,
                       "targets": [target(ref, sql, fmt, inst, gp_range, gp_inst)
                                   for ref, sql, fmt, inst in targets]})
    d["panels"] = panels
    return d


def write(out_dir):
    path = os.path.join(out_dir, OUT)
    with open(path, "w") as f:
        json.dump(build(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(C.HERE, "dashboards"))
    print(write(ap.parse_args().out))
