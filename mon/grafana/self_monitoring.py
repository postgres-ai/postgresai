#!/usr/bin/env python3
"""The self-monitoring dashboard for pgai-mon (dashboards/Self_Monitoring_Dashboard_pgai_mon.json).

The original (old/config/grafana/dashboards/Self_Monitoring_Dashboard.json) charted the old stack's
components: cAdvisor containers, node-exporter host stats, VictoriaMetrics, the sink Postgres and
the Flask backend. Only cAdvisor and node-exporter remain, and its PromQL has no case in cases.py,
so this dashboard is not compiled from it: it is written here from the series pgai-mon writes about
itself (collector.rs: pgai_mon_* per database; selfmon.rs: pgai_mon_* of the process, from GET
/health; scrape.rs: up, scrape_*), from the `series` view, and from the self-node-exporter and
self-cadvisor scrape jobs (mon/docker/pgai-mon.yml), in the same target shape as compile.py's
(time, value, __series, __sid), sent as CSV. Standard library Python only; it needs no
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
UID = "self-monitoring"

E = "CAST([] AS VARCHAR[][])"  # no matchers: pgai-mon's process series carry no cluster/node_name
R = "getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms')"
RI = "getvariable('rate_interval_ms')"
# the monitoring host's own exporters (jobs in mon/docker/pgai-mon.yml); Supabase relays node_* too
NE = "[['job', '=', 'self-node-exporter']]"
CA = "[['job', '=', 'self-cadvisor'], ['name', '!=', '']]"  # containers; the root cgroup has no name
DISK = "[['job', '=', 'self-node-exporter'], ['device', '!~', 'dm-.*']]"

# the original panels with no counterpart in the new stack, and why (README.md lists them too)
DROPPED = {
    1: "Active monitoring services: cAdvisor, node-exporter, postgres-exporter, VictoriaMetrics, pgwatch "
       "(the scrape targets are counted in panel 1004)",
    5: "Victoria Metrics storage size stat: the store size is charted in panel 22",
    23: "Victoria Metrics rows count: the store keeps no row count (its known series are charted)",
    30: "Sink Postgres connections: no sink", 31: "Sink Postgres transactions: no sink",
    32: "Sink Postgres database size: no sink", 33: "Sink Postgres block I/O: no sink",
}
# kept: 200 (the version banner: pgai_mon_build_info instead of the Flask backend's /version),
# 3 (memory: pgai-mon's RSS), 6 (time series in the store, from the series view),
# 20 (ingestion rate), 21 (scrape duration by target), 22 (storage size: WAL and Parquet), and the
# host and container panels (2, 4, 40-43, 50-53, 10-13), from the self-* scrape jobs
KEPT = (200, 3, 6, 20, 21, 22, 2, 4, 40, 41, 42, 43, 50, 51, 52, 53, 10, 11, 12, 13)
# panels drawn without a background or frame, as the original's banner
TRANSPARENT = {200}


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


def last(ref, metric, fmt):
    """A target: the metric's series as they are."""
    return (ref, f"SELECT labels, t, value FROM prom_last('{metric}', {E}, {R})", fmt, False)


def over(fn, ref, metric, fmt, m=E):
    """A target: rate or increase of a counter over the rate interval."""
    return (ref, f"SELECT labels, t, value FROM prom_{fn}('{metric}', {m}, {R}, {RI})", fmt, False)


def by_name(ref, metric, fmt):
    """A target: a cAdvisor counter's rate per container, summed over its devices or interfaces."""
    return (ref, f"SELECT labels, t, value FROM prom_agg('sum', 'rate', '{metric}', {CA}, {R}, {RI}, "
                 "p_by := ['name'])", fmt, False)


def disk_latency(ref, op, fmt):
    """A target: seconds per read or write, by block device (rate of the time / rate of the count)."""
    t, c = (f"prom_rate('node_disk_{m}_total', {DISK}, {R}, {RI})" for m in (f"{op}_time_seconds", f"{op}s_completed"))
    return (ref, f"SELECT a.labels, a.t, a.value / b.value AS value FROM {t} a JOIN {c} b\n"
                 "  ON a.t = b.t AND a.labels['device'] = b.labels['device'] WHERE b.value > 0", fmt, False)


MEM = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable")


def memory(sql):
    """`sql` over p: node-exporter's memory gauges, one column each, per time point."""
    return ("WITH m AS (\n" + "\n  UNION ALL ".join(
        f"SELECT '{k}' AS k, t, value FROM prom_last('node_memory_{k}_bytes', {NE}, {R})" for k in MEM) +
        "\n), p AS (\n  SELECT t, " + ", ".join(f"max(value) FILTER (WHERE k = '{k}') AS {k}" for k in MEM) +
        " FROM m GROUP BY t\n)\n" + sql)


# (id, type, title, description, gridPos (h, w, x, y), field defaults, options,
#  [(refId, sql, legendFormat, instant)])
PANELS = [
    ("row", 100, "Overview", 0),
    # the original's banner (a transparent stat with no title, TRANSPARENT): the release shows as the
    # series' name, "PostgresAI <version>", with the value (1) hidden
    (200, "stat", "", "The PostgresAI monitoring release that pgai-mon runs (the version label of "
     "pgai_mon_build_info).", (2, 24, 0, 1), stat_defaults("short", [{"color": "blue", "value": None}]),
     dict(STAT_OPTIONS, colorMode="background", justifyMode="center", orientation="horizontal", textMode="name"),
     [("A", f"SELECT labels, t, value FROM prom_last('pgai_mon_build_info', {E}, {R})", "PostgresAI {{version}}",
       True)]),
    (1001, "stat", "Databases up", "Monitored databases whose last collection connected "
     "(pgai_mon_source_up = 1).", (4, 4, 0, 3), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", f"SELECT t, sum(value) AS value FROM prom_last('pgai_mon_source_up', {E}, {R}) GROUP BY t",
       "up", True)]),
    (1002, "stat", "Databases down", "Monitored databases pgai-mon cannot connect to "
     "(pgai_mon_source_up = 0).", (4, 4, 4, 3), stat_defaults("short", RED_FROM_1, "0"), STAT_OPTIONS,
     [("A", f"SELECT t, sum(1 - value) AS value FROM prom_last('pgai_mon_source_up', {E}, {R}) GROUP BY t",
       "down", True)]),
    (1003, "stat", "Collection errors in range", "Failed metric queries over the dashboard range, all "
     "databases (increase of pgai_mon_errors_total).", (4, 4, 8, 3), stat_defaults("short", RED_FROM_1, "0"),
     STAT_OPTIONS,
     [("A", f"SELECT t, value FROM prom_agg('sum', 'increase', 'pgai_mon_errors_total', {E}, {R}, "
            "getvariable('range_ms'))", "errors", True)]),
    (1004, "stat", "Scrape targets up", "Scrape targets and collector sources whose last scrape or "
     "collection succeeded (up = 1).", (4, 4, 12, 3), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", f"SELECT t, sum(value) AS value FROM prom_last('up', {E}, {R}) GROUP BY t", "up", True)]),
    (6, "stat", "Time series in store", "Distinct series the store knows (the series view), all metrics.",
     (4, 4, 16, 3), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", "SELECT getvariable('end_ms') AS t, count(DISTINCT series_id) AS value FROM series", "series",
       True)]),
    (1006, "stat", "Metrics in store", "Distinct metric names the store knows (the series view).",
     (4, 4, 20, 3), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", "SELECT getvariable('end_ms') AS t, count(DISTINCT metric) AS value FROM series", "metrics",
       True)]),
    ("row", 101, "Collection", 7),
    (1010, "timeseries", "Database collection status", "1: the database's last collection connected; "
     "0: it did not (pgai_mon_source_up).", (8, 12, 0, 8),
     ts_defaults("short", min=0, max=1), legend("Last", ["min", "last"]),
     [last("A", "pgai_mon_source_up", "{{dbname}}")]),
    (1011, "timeseries", "Time since last successful collection", "Seconds since a metric query last "
     "succeeded on the database (now - pgai_mon_source_last_success_epoch_s).", (8, 12, 12, 8),
     ts_defaults("s"), legend("Max", ["mean", "max", "last"]),
     [("A", f"SELECT labels, t, t / 1000.0 - value AS value "
            f"FROM prom_last('pgai_mon_source_last_success_epoch_s', {E}, {R})", "{{dbname}}", False)]),
    (1012, "timeseries", "Collection errors /s", "Failed metric queries per second, by database and "
     "metric (rate of pgai_mon_errors_total).", (8, 12, 0, 16), ts_defaults("short"),
     legend("Mean", ["mean", "max"]),
     [("A", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_errors_total', {E}, {R}, {RI}, "
            "p_by := ['dbname', 'metric'])", "{{dbname}}: {{metric}}", False)]),
    (1013, "timeseries", "Timeouts, skipped runs and connection failures /s", "Metric queries cancelled by "
     "statement or lock timeout, runs skipped because the previous one still ran, and failed connection "
     "attempts, per second (rates of pgai_mon_timeouts_total, pgai_mon_skipped_slots_total, "
     "pgai_mon_connect_failures_total).", (8, 12, 12, 16), ts_defaults("short"), legend("Mean", ["mean", "max"]),
     [("A", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_timeouts_total', {E}, {R}, {RI}, "
            "p_by := ['dbname', 'metric'])", "timeouts {{dbname}}: {{metric}}", False),
      ("B", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_skipped_slots_total', {E}, {R}, "
            f"{RI}, p_by := ['dbname', 'metric'])", "skipped {{dbname}}: {{metric}}", False),
      ("C", f"SELECT labels, t, value FROM prom_agg('sum', 'rate', 'pgai_mon_connect_failures_total', {E}, {R}, "
            f"{RI}, p_by := ['dbname'])", "connection failures {{dbname}}", False)]),
    (1014, "timeseries", "Metric query duration (top 10)", "Wall time of each metric's collection query, "
     "the 10 slowest series over the range (pgai_mon_metric_duration_ms).", (8, 12, 0, 24), ts_defaults("ms"),
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
     "started (pgai_mon_series_cut_total, pgai_mon_rows_cut_total).", (8, 12, 12, 24), ts_defaults("short"),
     legend("Last", ["max", "last"]),
     [("A", f"SELECT labels, t, value FROM prom_last('pgai_mon_series_cut_total', {E}, {R})",
       "series cut {{dbname}}: {{metric}}", False),
      ("B", f"SELECT labels, t, value FROM prom_last('pgai_mon_rows_cut_total', {E}, {R})",
       "rows cut {{dbname}}: {{metric}}", False)]),
    ("row", 102, "Scrape targets", 32),
    (1020, "timeseries", "Scrape target status", "1: the target's last scrape (or the collector source's "
     "last run) succeeded; 0: it failed (up).", (8, 8, 0, 33), ts_defaults("short", min=0, max=1),
     legend("Last", ["min", "last"]),
     [last("A", "up", "{{job}} {{instance}}")]),
    (21, "timeseries", "Scrape duration by target", "Average scrape duration by job (scrape_duration_seconds).",
     (8, 8, 8, 33), ts_defaults("s"), legend("Max", ["mean", "max", "last"]),
     [("A", "SELECT map(['job'], [labels['job']]) AS labels, t, avg(value) AS value "
            f"FROM prom_last('scrape_duration_seconds', {E}, {R}) GROUP BY ALL", "{{job}}", False)]),
    (1022, "timeseries", "Samples scraped by target", "Samples per scrape by job (scrape_samples_scraped).",
     (8, 8, 16, 33), ts_defaults("short"), legend("Last", ["mean", "max", "last"]),
     [("A", "SELECT map(['job'], [labels['job']]) AS labels, t, sum(value) AS value "
            f"FROM prom_last('scrape_samples_scraped', {E}, {R}) GROUP BY ALL", "{{job}}", False)]),
    ("row", 103, "pgai-mon process", 41),
    (20, "timeseries", "Ingestion rate", "Samples and records the store takes in per second, from the collector, "
     "scrapes, pushes and pgai-mon itself (rates of pgai_mon_ingest_samples_total, "
     "pgai_mon_ingest_records_total).", (8, 8, 0, 42), ts_defaults("short"), legend("Mean", ["mean", "max"]),
     [over("rate", "A", "pgai_mon_ingest_samples_total", "samples/s"),
      over("rate", "B", "pgai_mon_ingest_records_total", "records/s")]),
    (1031, "timeseries", "Flush duration", "Wall time of the last WAL to Parquet flush before each write "
     "(pgai_mon_flush_duration_seconds).", (8, 8, 8, 42), ts_defaults("s"), legend("Max", ["mean", "max", "last"]),
     [last("A", "pgai_mon_flush_duration_seconds", "flush")]),
    (22, "timeseries", "WAL and store size", "Bytes in the write-ahead log not flushed yet and in the "
     "store's Parquet files (pgai_mon_wal_bytes, pgai_mon_store_parquet_bytes).", (8, 8, 16, 42),
     ts_defaults("bytes"), legend("Last", ["max", "last"]),
     [last("A", "pgai_mon_wal_bytes", "WAL"),
      last("B", "pgai_mon_store_parquet_bytes", "Parquet")]),
    (3, "timeseries", "Memory", "Resident memory of the pgai-mon process and of its SQL reader process, when "
     "the reader runs as one (pgai_mon_process_resident_memory_bytes, pgai_mon_reader_resident_memory_bytes).",
     (8, 8, 0, 50), ts_defaults("bytes"), legend("Max", ["mean", "max", "last"]),
     [last("A", "pgai_mon_process_resident_memory_bytes", "pgai-mon"),
      last("B", "pgai_mon_reader_resident_memory_bytes", "reader")]),
    (1033, "timeseries", "CPU", "CPU time of the pgai-mon process per second, user and system; 100% is one "
     "core (rate of pgai_mon_process_cpu_seconds_total).", (8, 8, 8, 50), ts_defaults("percentunit"),
     legend("Mean", ["mean", "max"]),
     [over("rate", "A", "pgai_mon_process_cpu_seconds_total", "CPU")]),
    (1034, "timeseries", "Reader restarts, memory kills and late flushes", "SQL reader process restarts, "
     "the restarts its memory cap forced, and flushes that started late, per interval (increases of "
     "pgai_mon_reader_restarts_total, pgai_mon_reader_memory_kills_total, pgai_mon_late_flushes_total).",
     (8, 8, 16, 50), ts_defaults("short"), legend("Max", ["max", "last"]),
     [over("increase", "A", "pgai_mon_reader_restarts_total", "reader restarts"),
      over("increase", "B", "pgai_mon_reader_memory_kills_total", "reader memory kills"),
      over("increase", "C", "pgai_mon_late_flushes_total", "late flushes")]),
    (1035, "timeseries", "S3 upload lag", "Seconds since the store's catalog was last committed to the bucket "
     "(pgai_mon_s3_last_commit_age_seconds; S3 stores only).", (8, 8, 0, 58), ts_defaults("s"),
     legend("Max", ["mean", "max", "last"]),
     [last("A", "pgai_mon_s3_last_commit_age_seconds", "commit age {{store}}")]),
    (1036, "timeseries", "S3 upload backlog", "Bytes of local Parquet files not uploaded yet "
     "(pgai_mon_s3_pending_bytes; S3 stores only).", (8, 8, 8, 58), ts_defaults("bytes"),
     legend("Max", ["mean", "max", "last"]),
     [last("A", "pgai_mon_s3_pending_bytes", "pending {{store}}")]),
    (1037, "timeseries", "S3 upload errors", "Failed upload rounds per interval (increase of "
     "pgai_mon_s3_upload_errors_total; S3 stores only). Data stays local and the round is retried.",
     (8, 8, 16, 58), ts_defaults("short"), legend("Max", ["max", "last"]),
     [over("increase", "A", "pgai_mon_s3_upload_errors_total", "failed rounds {{store}}")]),
    (1038, "timeseries", "Cache hit rate", "Hits / (hits + misses) per second, by cache: query (HTTP result "
     "cache), footer (Parquet footers), row (evaluated rows) (pgai_mon_cache_hits_total, "
     "pgai_mon_cache_misses_total).", (8, 8, 0, 66), ts_defaults("percentunit", min=0, max=1),
     legend("Mean", ["mean", "min", "last"]),
     [("A", "WITH h AS (\n"
            f"  SELECT labels, t, value FROM prom_rate('pgai_mon_cache_hits_total', {E}, {R}, {RI})\n"
            "), m AS (\n"
            f"  SELECT labels, t, value FROM prom_rate('pgai_mon_cache_misses_total', {E}, {R}, {RI})\n"
            ")\n"
            "SELECT h.labels, h.t, h.value / (h.value + m.value) AS value FROM h JOIN m\n"
            "  ON h.labels['cache'] = m.labels['cache'] AND h.t = m.t WHERE h.value + m.value > 0",
       "{{cache}}", False)]),
    (1039, "timeseries", "Query admission", "SQL queries running and waiting for a slot "
     "(pgai_mon_admission_running, pgai_mon_admission_waiting).", (8, 8, 8, 66), ts_defaults("short"),
     legend("Max", ["mean", "max", "last"]),
     [last("A", "pgai_mon_admission_running", "running"),
      last("B", "pgai_mon_admission_waiting", "waiting")]),
    (1040, "timeseries", "Store files and known series", "Parquet files in the store's catalog and series "
     "the writer knows (pgai_mon_store_files, pgai_mon_series_known).", (8, 8, 16, 66), ts_defaults("short"),
     legend("Last", ["max", "last"]),
     [last("A", "pgai_mon_store_files", "files"),
      last("B", "pgai_mon_series_known", "known series")]),
    ("row", 104, "Host", 74),
    (2, "stat", "Running containers", "Containers cAdvisor sees on the monitoring host (container_last_seen).",
     (4, 8, 0, 75), stat_defaults("short", GREEN), STAT_OPTIONS,
     [("A", f"SELECT t, CAST(count(*) AS DOUBLE) AS value FROM prom_last('container_last_seen', {CA}, {R}) GROUP BY t",
       "containers", True)]),
    # a 5 minute window, as the original's [5m]: the rate interval is shorter than the 30 s scrape
    (4, "stat", "System CPU usage", "Busy share of the monitoring host's CPUs over the last 5 minutes, all "
     "cores (1 - rate of node_cpu_seconds_total{mode=\"idle\"}).", (4, 8, 8, 75), stat_defaults("percent", GREEN),
     STAT_OPTIONS,
     [("A", "SELECT t, 100 * (1 - avg(value)) AS value FROM prom_rate('node_cpu_seconds_total', "
            f"[['job', '=', 'self-node-exporter'], ['mode', '=', 'idle']], {R}, 300000) GROUP BY t", "CPU", True)]),
    (1041, "stat", "System memory usage", "Memory of the monitoring host not available to new processes "
     "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes).", (4, 8, 16, 75),
     stat_defaults("percent", GREEN), STAT_OPTIONS,
     [("A", memory("SELECT t, 100 * (1 - MemAvailable / MemTotal) AS value FROM p WHERE MemTotal > 0"), "memory",
       True)]),
    (40, "timeseries", "System CPU usage breakdown", "Share of the monitoring host's CPU time by mode, "
     "averaged over cores (rate of node_cpu_seconds_total).", (8, 12, 0, 79), ts_defaults("percentunit"),
     legend("Mean", ["mean", "max"]),
     [("A", "SELECT labels, t, value FROM prom_agg('avg', 'rate', 'node_cpu_seconds_total', "
            "[['job', '=', 'self-node-exporter'], ['mode', '=~', 'user|system|iowait|nice|irq|softirq|steal']], "
            f"{R}, {RI}, p_by := ['mode'])", "{{mode}}", False)]),
    (41, "timeseries", "System memory usage breakdown", "The monitoring host's memory: application (total - "
     "free - buffers - cache), buffers, cache (page cache and reclaimable slab) and free (node_memory_*_bytes).",
     (8, 12, 12, 79), ts_defaults("bytes", stack=True), legend("Last", ["mean", "max", "last"]),
     [("A", memory("SELECT map(['part'], [part]) AS labels, t, value FROM (\n"
                   "  SELECT t, MemTotal - MemFree - Buffers - Cached - SReclaimable AS Application, Buffers, "
                   "Cached + SReclaimable AS Cache, MemFree AS Free FROM p\n"
                   ") UNPIVOT (value FOR part IN (Application, Buffers, Cache, Free))"), "{{part}}", False)]),
    (42, "timeseries", "System network I/O", "Bytes per second the monitoring host receives and sends, by "
     "interface, without loopback, bridges and container ends (rates of cAdvisor's root-container "
     "container_network_*_bytes_total).", (8, 12, 0, 87), ts_defaults("Bps"), legend("Mean", ["mean", "max"]),
     [over("rate", "A", "container_network_receive_bytes_total", "{{interface}} receive",
           "[['job', '=', 'self-cadvisor'], ['id', '=', '/']]"),
      over("rate", "B", "container_network_transmit_bytes_total", "{{interface}} transmit",
           "[['job', '=', 'self-cadvisor'], ['id', '=', '/']]")]),
    (43, "timeseries", "System disk usage", "Used share of each filesystem of the monitoring host "
     "(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes), without tmpfs, squashfs and vfat.",
     (8, 12, 12, 87), ts_defaults("percent", min=0, max=100), legend("Last", ["max", "last"]),
     [("A", "WITH f AS (\n"
            "  SELECT labels, t, value FROM prom_last('node_filesystem_size_bytes', "
            f"[['job', '=', 'self-node-exporter'], ['fstype', '!~', 'tmpfs|fuse.lxcfs|squashfs|vfat']], {R})\n"
            "), a AS (\n"
            f"  SELECT labels, t, value FROM prom_last('node_filesystem_avail_bytes', {NE}, {R})\n"
            ")\n"
            "SELECT f.labels, f.t, 100 * (1 - a.value / f.value) AS value FROM f JOIN a ON f.t = a.t\n"
            "  AND f.labels['mountpoint'] = a.labels['mountpoint'] AND f.labels['device'] = a.labels['device']\n"
            "  WHERE f.value > 0", "{{mountpoint}}", False)]),
    (50, "timeseries", "Disk I/O operations (IOPS)", "Reads and writes completed per second, by block device "
     "(rates of node_disk_reads_completed_total, node_disk_writes_completed_total).", (8, 12, 0, 95),
     ts_defaults("iops"), legend("Mean", ["mean", "max"]),
     [over("rate", "A", "node_disk_reads_completed_total", "{{device}} read", DISK),
      over("rate", "B", "node_disk_writes_completed_total", "{{device}} write", DISK)]),
    (51, "timeseries", "Disk I/O throughput", "Bytes read and written per second, by block device (rates of "
     "node_disk_read_bytes_total, node_disk_written_bytes_total).", (8, 12, 12, 95), ts_defaults("Bps"),
     legend("Mean", ["mean", "max"]),
     [over("rate", "A", "node_disk_read_bytes_total", "{{device}} read", DISK),
      over("rate", "B", "node_disk_written_bytes_total", "{{device}} write", DISK)]),
    (52, "timeseries", "Disk I/O utilization", "Share of time each block device was busy (rate of "
     "node_disk_io_time_seconds_total).", (8, 12, 0, 103), ts_defaults("percentunit"),
     legend("Max", ["mean", "max"]),
     [over("rate", "A", "node_disk_io_time_seconds_total", "{{device}}", DISK)]),
    (53, "timeseries", "Disk I/O average latency", "Average time of a read and of a write, by block device "
     "(rate of node_disk_read_time_seconds_total / rate of node_disk_reads_completed_total; writes alike).",
     (8, 12, 12, 103), ts_defaults("s"), legend("Max", ["mean", "max"]),
     [disk_latency("A", "read", "{{device}} read"), disk_latency("B", "write", "{{device}} write")]),
    ("row", 105, "Containers", 111),
    (10, "timeseries", "Container CPU usage", "CPU time of each container per second; 100% is one core (rate "
     "of container_cpu_usage_seconds_total).", (8, 12, 0, 112), ts_defaults("percent"),
     legend("Mean", ["mean", "max"]),
     [("A", f"SELECT labels, t, 100 * value AS value FROM prom_agg('sum', 'rate', 'container_cpu_usage_seconds_total', "
            f"{CA}, {R}, {RI}, p_by := ['name'])", "{{name}}", False)]),
    (11, "timeseries", "Container memory usage", "Memory of each container, page cache included "
     "(container_memory_usage_bytes).", (8, 12, 12, 112), ts_defaults("bytes"), legend("Last", ["mean", "max", "last"]),
     [("A", f"SELECT labels, t, value FROM prom_last('container_memory_usage_bytes', {CA}, {R})", "{{name}}",
       False)]),
    (12, "timeseries", "Container network I/O", "Bytes each container receives and sends per second, all "
     "its interfaces (rates of container_network_*_bytes_total).", (8, 12, 0, 120), ts_defaults("Bps"),
     legend("Mean", ["mean", "max"]),
     [by_name("A", "container_network_receive_bytes_total", "{{name}} rx"),
      by_name("B", "container_network_transmit_bytes_total", "{{name}} tx")]),
    (13, "timeseries", "Container disk I/O", "Bytes each container reads and writes per second, all its "
     "devices (rates of container_fs_reads_bytes_total, container_fs_writes_bytes_total).", (8, 12, 12, 120),
     ts_defaults("Bps"), legend("Mean", ["mean", "max"]),
     [by_name("A", "container_fs_reads_bytes_total", "{{name}} read"),
      by_name("B", "container_fs_writes_bytes_total", "{{name}} write")]),
]


def target(ref, sql, fmt, instant, gp_range, gp_inst):
    cols = ["labels", "t", "value"] if "labels" in sql else ["t", "value"]  # a stat's sum has no labels
    body = C.wrap(C.inline(sql, {}, gp_inst if instant else gp_range), cols, fmt, [])
    return C.infinity_target(ref, C.window_header(body, instant), False, [])


def build():
    src = json.load(open(os.path.join(C.DASH_DIR, SRC)))
    d = {k: copy.deepcopy(v) for k, v in src.items() if k not in ("panels", "templating")}
    d.update(uid=UID, title=src["title"], id=None, version=1, templating={"list": []})
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
                      "properties": [{"id": "displayName", "value": "${__field.labels.__series}"}]}
                     for ref, *_ in targets]
        panels.append({"datasource": dict(C.INFINITY), "description": desc,
                       "fieldConfig": {"defaults": copy.deepcopy(defaults), "overrides": overrides},
                       "gridPos": {"h": h, "w": w, "x": x, "y": y}, "id": pid, "options": copy.deepcopy(options),
                       "pluginVersion": "12.0.2", "title": title, "type": typ,
                       "targets": [target(ref, sql, fmt, inst, gp_range, gp_inst)
                                   for ref, sql, fmt, inst in targets]})
        if pid in TRANSPARENT:
            panels[-1]["transparent"] = True
    d["panels"] = panels
    return d


def write(out_dir):
    path = os.path.join(out_dir, OUT)
    C.write_dashboard(build(), path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(C.HERE, "dashboards"))
    print(write(ap.parse_args().out))
