#!/usr/bin/env python3
"""Build the panel-check fixture (panel_store.jsonl.gz, a VictoriaMetrics /api/v1/export JSON-lines
file that `pgai-mon load` imports) from a pgai-mon serving a recorded store. Standard library only.

  make_panel_store.py --url http://127.0.0.1:8765 --end-ms 1794206700000 [--out ...]

It keeps 130 minutes before --end-ms (the dashboards' default 1 h range, plus the 1 h subquery
window of Dashboard 6's threshold panel and rate look-back), at the recorded 30 s resolution, of
a few series per metric: database target_database, schema app, table t119 and its index
t119_unused, a handful of statements. Then it adds synthetic series for what the recording has
none of (host metrics, WAL archiving, replication and slots, lock waits, autovacuum blocked on a
lock, index_pilot, the settings columns metrics.yml gained later), so that every panel has data
to run on. Values are rounded to 6 significant digits to keep the file small.
"""
import argparse
import gzip
import json
import math
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPAN_MS = 130 * 60_000
STEP_MS = 30_000
MAX_SERIES = 25

KEEP = {  # label -> allowed values (a series without the label passes)
    "datname": {"target_database"}, "real_dbname": {"target_database"}, "dbname": {"target_database"},
    "schemaname": {"app"}, "schema": {"app"}, "schema_name": {"app"},
    "relname": {"t119", "t119_unused", "t119_pkey", "app.t119"}, "table_name": {"t119"}, "tblname": {"t119"},
    "table_full_name": {"app.t119"}, "indexrelname": {"t119_pkey"}, "idxname": {"t119_unused"},
    "setting_name": {"block_size", "max_connections", "wal_sender_timeout", "archive_mode", "archive_command",
                     "archive_library", "wal_keep_size", "min_wal_size", "max_wal_size", "max_slot_wal_keep_size",
                     "checkpoint_timeout", "wal_level", "max_wal_senders", "max_replication_slots", "synchronous_commit",
                     "synchronous_standby_names", "autovacuum_freeze_max_age", "autovacuum_multixact_freeze_max_age",
                     "autovacuum_vacuum_scale_factor", "autovacuum_vacuum_threshold", "autovacuum_max_workers"},
}
QUERYIDS = {"3821806407744617563", "-1106401531938326102", "5691194137505178511", "2713108609228007267"}
QUERY_ID = "3821806407744617563"  # the statement Dashboard 3 is checked with; it also gets ASH samples


def post(url, sql):
    req = urllib.request.Request(url.rstrip("/") + "/sql?format=objects", data=sql.encode(), method="POST",
                                 headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    if "error" in j or j.get("truncated"):
        raise SystemExit(f"{j.get('error', 'result truncated')}\n{sql}")
    return j["rows"]


def keep_matchers():
    """keep() as pgai_samples matchers (an absent label matches the empty alternative), so the
    reader returns only the kept series' samples."""
    def alt(vs):
        return "|".join(re.escape(v) for v in sorted(vs)) + "|"
    ms = [[k, "=~", alt(vs)] for k, vs in KEEP.items()] + [[k, "=~", alt(QUERYIDS)] for k in ("queryid", "query_id")]
    return "[" + ", ".join("[" + ", ".join("'" + x.replace("'", "''") + "'" for x in m) + "]" for m in ms) + "]"


def keep(labels):
    for k, allowed in KEEP.items():
        if k in labels and labels[k] not in allowed:
            return False
    for k in ("queryid", "query_id"):
        if k in labels and labels[k] not in QUERYIDS:
            return False
    return True


def r6(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(f"{v:.6g}")


def recorded(url, t0, t1):
    out = []
    metrics = [r["metric"] for r in post(url, "SELECT DISTINCT metric FROM series ORDER BY 1")]
    for m in metrics:
        rows, page = [], 20000
        while True:  # pages: a response is capped in rows and bytes
            got = post(url, f"SELECT series_id, CAST(labels AS JSON) AS labels, CAST(epoch_ms(ts) AS BIGINT) AS t, value "
                            f"FROM pgai_samples('{m}', {keep_matchers()}, {t0}, {t1}) WHERE NOT stale "
                            f"ORDER BY series_id, ts LIMIT {page} OFFSET {len(rows)}")
            rows += got
            if len(got) < page:
                break
        by = {}
        for r in rows:
            lab = r["labels"] if isinstance(r["labels"], dict) else json.loads(r["labels"])
            if not keep(lab):
                continue
            s = by.setdefault(r["series_id"], {"labels": lab, "t": [], "v": []})
            v = r6(r["value"])
            if v is not None:
                s["t"].append(r["t"])
                s["v"].append(v)
        best = sorted(by.values(), key=lambda s: (-len(s["t"]), json.dumps(s["labels"], sort_keys=True)))[:MAX_SERIES]
        for s in best:
            out.append({"metric": {"__name__": m, **s["labels"]}, "values": s["v"], "timestamps": s["t"]})
    return out


def synthetic(base, t0, t1):
    """Series for the metrics the recording lacks; base: the labels of an instance-level series."""
    ts = list(range(t0 - t0 % STEP_MS + STEP_MS, t1 + 1, STEP_MS))
    n = len(ts)
    out = []

    def add(name, extra, values):
        out.append({"metric": {"__name__": name, **base, **extra}, "values": [r6(v) for v in values], "timestamps": ts})

    def counter(rate, start=1000.0):
        return [start + rate * i * STEP_MS / 1000 for i in range(n)]

    def wave(lo, hi, period=40):
        return [lo + (hi - lo) * (0.5 + 0.5 * math.sin(2 * math.pi * i / period)) for i in range(n)]

    for m, vals in (("host_cpu_utilization_percent", wave(20, 60)), ("host_os_cpu_percent", wave(20, 60)),
                    ("host_os_cpu_user_percent", wave(10, 30)), ("host_os_cpu_system_percent", wave(5, 10)),
                    ("host_os_cpu_iowait_percent", wave(1, 5)), ("host_os_cpu_steal_percent", wave(0, 1)),
                    ("host_os_memory_total_bytes", [8 << 30] * n), ("host_os_memory_free_bytes", wave(1 << 30, 2 << 30)),
                    ("host_os_memory_cached_bytes", wave(2 << 30, 3 << 30)), ("host_os_memory_buffers_bytes", [1 << 28] * n),
                    ("host_memory_available_bytes", wave(3 << 30, 4 << 30)), ("host_os_swap_used_bytes", [0] * n),
                    ("host_os_process_max_rss_bytes", wave(1 << 29, 1 << 30)), ("host_disk_size_bytes", [100 << 30] * n),
                    ("host_disk_free_bytes", wave(40 << 30, 41 << 30)), ("host_local_storage_free_bytes", [20 << 30] * n),
                    ("host_volume_used_bytes", wave(59 << 30, 60 << 30)), ("host_disk_read_iops", wave(10, 200)),
                    ("host_disk_write_iops", wave(50, 300)), ("host_volume_read_iops", wave(10, 200)),
                    ("host_volume_write_iops", wave(50, 300)), ("host_network_receive_bytes_per_second", wave(1e5, 1e6)),
                    ("host_network_transmit_bytes_per_second", wave(1e5, 1e6)), ("host_db_load", wave(0.5, 3))):
        add(m, {}, vals)
    add("host_provider_info", {"provider": "self-managed"}, [1] * n)
    # WAL archiving (archive_lag) and replication slots
    add("pgwatch_archive_lag_archived_count", {}, counter(0.1))
    add("pgwatch_archive_lag_failed_count", {}, counter(0.002, 3))
    lsn = counter(2e6, 5e10)
    add("pgwatch_archive_lag_current_lsn_numeric", {}, lsn)
    add("pgwatch_archive_lag_archived_wal_finish_lsn_numeric", {}, [x - 3.2e7 for x in lsn])
    add("pgwatch_archive_lag_seconds_since_archive", {}, wave(1, 60))
    add("pgwatch_archive_lag_wal_files_behind", {}, wave(0, 3))
    slot = {"slot_name": "standby1_slot", "plugin": ""}
    add("pgwatch_replication_slots_restart_lsn_lag_b", slot, wave(1e6, 5e7))
    add("pgwatch_replication_slots_non_active_int", slot, [1 if i > n // 2 else 0 for i in range(n)])
    # one streaming standby
    rep = {"application_name": "standby1", "client_info": "10.0.0.2", "usename": "replicator"}
    for m, vals in (("replay_lag_b", wave(1e5, 1e7)), ("replay_lag_ms", wave(10, 120_000)), ("write_lag_b", wave(1e4, 1e6)),
                    ("write_lag_ms", wave(1, 50)), ("flush_lag_b", wave(1e4, 2e6)), ("flush_lag_ms", wave(1, 80)),
                    ("sent_lag_b", wave(0, 1e5)), ("reply_time_lag_ms", wave(100, 1000)), ("is_sync_int", [1] * n),
                    ("backend_xmin_reported_int", [1] * n), ("backend_xmin_age_tx", wave(10, 5000))):
        add(f"pgwatch_replication_{m}", rep, vals)
    # lock waits: one blocked session
    lw = {"blocked_locktype": "relation", "blocked_mode": "RowExclusiveLock", "blocker_locktype": "relation",
          "blocker_mode": "AccessExclusiveLock", "blocked_table": "app.t119", "blocker_table": "app.t119",
          "blocked_pid": "4242", "blocker_pid": "4141", "blocked_appname": "app", "blocker_appname": "psql",
          "blocked_user": "app", "blocker_user": "admin", "blocked_query_id": QUERY_ID, "blocker_query_id": "0"}
    add("pgwatch_lock_waits_blocked_ms", lw, wave(100, 30_000))
    add("pgwatch_lock_waits_blocker_tx_ms", lw, wave(1000, 60_000))
    add("pgwatch_pg_autovacuum_blocked_wait_seconds", {"worker_pid": "5151", "blocker_pid": "4141", "blocker_queryid": "0"},
        wave(1, 120))
    # index_pilot (Dashboard 11's Boguk ratio and rebuild scale factor)
    ip = {"schemaname": "app", "relname": "t119", "indexrelname": "t119_pkey"}
    add("pgwatch_pg_index_pilot_best_ratio", ip, [0.8] * n)
    add("pgwatch_pg_index_pilot_config_config_value_normalized",
        {"schemaname": "*", "indexrelname": "*", "datname": "*", "config_key": "index_rebuild_scale_factor"}, [2] * n)
    # settings columns newer than the recording
    for name, v in (("wal_keep_size", 0), ("min_wal_size", 80 << 20), ("max_wal_size", 1 << 30),
                    ("max_slot_wal_keep_size", -1), ("checkpoint_timeout", 300)):
        add("pgwatch_settings_setting_normalized", {"setting_name": name}, [v] * n)
    add("pgwatch_settings_max_connections", {}, [100] * n)
    return out


def supplements(base, t0, t1):
    """Series the recording has for other objects only (its per-relation metrics keep a top-N, and
    no vacuum ran in the last hour): table t119 and its index t119_pkey get them."""
    ts = list(range(t0 - t0 % STEP_MS + STEP_MS, t1 + 1, STEP_MS))
    n = len(ts)
    out = []

    def add(name, extra, values, times=ts):
        out.append({"metric": {"__name__": name, **base, **extra}, "values": [r6(v) for v in values], "timestamps": times})

    add("pgwatch_pg_class_relation_size_bytes", {"relkind": "i", "relname": "t119_pkey", "schemaname": "app"},
        [2_260_992 + 8192 * (i // 20) for i in range(n)])
    tb = {"schemaname": "app", "tblname": "t119"}
    add("pgwatch_pg_table_bloat_bloat_pct", tb, [12.5] * n)
    add("pgwatch_pg_table_bloat_bloat_size", tb, [688_128] * n)
    vac = ts[-40:-10]  # an autovacuum of t119, 15 minutes long, in the last half hour
    vl = {"phase": "4", "relid": "16840", "relname": "app.t119", "schema_name": "app", "table_name": "t119",
          "vacuum_mode": "autovacuum"}
    for m, vals in (("heap_blks_total", [675] * len(vac)), ("heap_blks_scanned", [20 * i for i in range(len(vac))]),
                    ("heap_blks_vacuumed", [15 * i for i in range(len(vac))]), ("index_vacuum_count", [1] * len(vac)),
                    ("max_dead_tuples_mb", [64] * len(vac)), ("num_dead_item_ids", [500 + i for i in range(len(vac))]),
                    ("is_anti_wraparound", [0] * len(vac))):
        add(f"pgwatch_pg_vacuum_progress_{m}", vl, vals, vac)
    return out


def ash_for(ash):
    """ASH samples for Dashboard 3's statement: a recorded wait-event series under its query_id."""
    out = []
    if ash and ash["metric"].get("query_id") != QUERY_ID:
        out.append({"metric": {**ash["metric"], "query_id": QUERY_ID}, "values": ash["values"],
                    "timestamps": ash["timestamps"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--end-ms", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "panel_store.jsonl.gz"))
    a = ap.parse_args()
    t0, t1 = a.end_ms - SPAN_MS, a.end_ms
    rec = recorded(a.url, t0, t1)
    base = next(s["metric"] for s in rec if s["metric"]["__name__"] == "pgwatch_db_stats_numbackends")
    base = {k: v for k, v in base.items() if k != "__name__"}
    ash = next((s for s in rec if s["metric"]["__name__"] == "pgwatch_wait_events_total"), None)
    have = {s["metric"]["__name__"] for s in rec}
    syn = [s for s in synthetic(base, t0, t1) if s["metric"]["__name__"] not in have]
    syn += supplements(base, t0, t1) + ash_for(ash)
    with gzip.open(a.out, "wt", compresslevel=9) as f:
        for s in rec + syn:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    print(f"{a.out}: {len(rec)} recorded + {len(syn)} synthetic series, "
          f"{sum(len(s['timestamps']) for s in rec + syn)} samples, {os.path.getsize(a.out)} bytes")


if __name__ == "__main__":
    main()
