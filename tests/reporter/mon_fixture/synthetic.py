#!/usr/bin/env python3
"""Write synthetic.jsonl.gz: deterministic series for the metrics the real slice in
samples.jsonl.gz does not have (dev tool; the output is committed, the tests do not run this).

The real demo box has no host metrics, no lock waits, no invalid indexes, no pg_stat_kcache, no
stats_reset readings, no B-tree bloat bytes, no archiving, no deadlocks and no `up` series, so
without these the reporter queries for them would compare empty with empty. It has no
setting_normalized either (not stored before #340), which the dashboard compiler needs for the
label columns of dashboard 05 panel 14. Same line format as samples.jsonl.gz (null = staleness
marker); loaded next to it by mon_harness.build_store.

    python tests/reporter/mon_fixture/synthetic.py
"""
import gzip
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AT_MS = 1794236820 * 1000  # mon_harness.FIXTURE_AT
DAY = 86_400_000
NODE = {"cluster": "local", "node_name": "node-01"}
PGWATCH = dict(NODE, env="demo", instance="pgwatch:9091", job="pgwatch-prometheus", sink_type="prometheus",
               sys_id="7705963719539601447")


def db_labels(datname):
    return dict(PGWATCH, datname=datname, real_dbname=datname,
                dbname="target_database" if datname == "target_database" else f"target_database_{datname}")


def grid(start_ms, end_ms, step_ms, jitter_ms=7_000):
    """Readings like the real slice: every 30 min before the last 26 h, then every `step_ms`."""
    dense_from = AT_MS - 26 * 3_600_000
    out, t = [], start_ms
    while t <= end_ms:
        out.append(t + jitter_ms)
        t += step_ms if t >= dense_from else 1_800_000
    return [t for t in out if t <= AT_MS]


def series(name, labels, ts, values):
    # no empty labels: an empty tag is no label in Prometheus, and pgai-mon's collector drops it too
    return {"metric": {k: v for k, v in dict(labels, __name__=name).items() if v != ""}, "timestamps": ts,
            "values": values}


def wave(i, lo, hi, period=97, phase=0):
    """A deterministic value between lo and hi with two decimals."""
    return round(lo + (hi - lo) * (0.5 + 0.5 * math.sin(2 * math.pi * (i + phase) / period)), 2)


def host():
    ts = grid(AT_MS - 8 * DAY, AT_MS, 60_000)
    total = 8 * 1024 ** 3
    low_at = len(ts) - 400  # the lowest memory reading, inside the last 26 h
    avail = [total * (0.08 if i == low_at else wave(i, 0.3, 0.7) * 1.0) for i in range(len(ts))]
    yield series("host_os_memory_total_bytes", NODE, ts, [total] * len(ts))
    yield series("host_memory_available_bytes", NODE, ts, [round(v) for v in avail])
    yield series("host_cpu_utilization_percent", NODE, ts, [wave(i, 5, 95, 61) for i in range(len(ts))])
    yield series("host_disk_free_bytes", NODE, ts, [200 * 1024 ** 3 - i * 4096 for i in range(len(ts))])
    for name, lo, hi, ph in (("host_disk_read_bytes_per_second", 1e5, 9e7, 0),
                             ("host_disk_write_bytes_per_second", 5e4, 4e7, 13),
                             ("host_disk_read_iops", 10, 3000, 29), ("host_disk_write_iops", 5, 1500, 41)):
        yield series(name, NODE, ts, [wave(i, lo, hi, 53, ph) for i in range(len(ts))])


def resource_sizing_and_spills():
    """RDS/ClickHouse resizes, credit balances, and one statement that spills."""
    ts = grid(AT_MS - 8 * DAY, AT_MS, 60_000)
    cut = AT_MS - 12 * 3_600_000
    for cls, times in (("db.t4g.medium", [t for t in ts if t < cut]),
                       ("db.t4g.large", [t for t in ts if t >= cut])):
        labels = dict(NODE, instance_class=cls, engine="postgres", storage_type="gp2",
                      multi_az="true", region="eu-central-1")
        yield series("host_instance_info", labels, times, [1] * len(times))
    for name, value in (("host_disk_size_bytes", 300 * 2 ** 30),
                        ("host_disk_max_size_bytes", 600 * 2 ** 30),
                        ("host_disk_provisioned_iops", 3000),
                        ("host_disk_provisioned_throughput_bytes_per_second", 125 * 2 ** 20)):
        yield series(name, NODE, ts, [value] * len(ts))
    balances = grid(AT_MS - 8 * DAY, AT_MS, 300_000)
    for name, value in (("host_cpu_credit_balance", 96), ("host_cpu_surplus_credit_balance", 0),
                        ("host_disk_burst_balance_percent", 90), ("host_ebs_io_balance_percent", 99),
                        ("host_ebs_byte_balance_percent", 97.5)):
        yield series(name, NODE, balances, [value] * len(balances))
    ch = dict(NODE, node_name="clickhouse")
    yield series("host_provider_info", dict(ch, provider="ClickHouse"), ts, [1] * len(ts))
    for name, before, after in (("host_cpu_cores", 2, 4),
                                ("host_os_memory_total_bytes", 8 * 2 ** 30, 16 * 2 ** 30),
                                ("host_disk_size_bytes", 100 * 2 ** 30, 200 * 2 ** 30)):
        yield series(name, ch, ts, [before if t < cut else after for t in ts])
    for name, scale in (("temp_files", 1), ("temp_bytes", 2 ** 20)):
        yield series(f"pgwatch_db_stats_{name}", dict(db_labels("target_database"), env="spill"),
                     ts, [i * scale for i in range(len(ts))])
    # A unique statement avoids overlapping the recorded counters with synthetic ones.
    lab = dict(db_labels("target_database"), queryid="900000000001")
    fields = ("calls", "rows", "shared_bytes_hit_total", "shared_bytes_read_total", "wal_bytes",
              "temp_bytes_written", "exec_time_total", "temp_bytes_read")
    for n, name in enumerate(fields, 1):
        yield series(f"pgwatch_pg_stat_statements_{name}", lab, ts, [i * n * 8192 for i in range(len(ts))])


def stats_reset():
    """pgwatch's tag is database_name; an older sink had datname: one of each."""
    ts = grid(AT_MS - 8 * DAY, AT_MS, 3_600_000)
    epoch = (AT_MS - 20 * DAY) // 1000
    lab = dict(PGWATCH, database_name="target_database", dbname="target_database")
    yield series("pgwatch_stats_reset_stats_reset_epoch", lab, ts, [epoch] * len(ts))
    yield series("pgwatch_stats_reset_stats_reset_epoch", db_labels("postgres"), ts, [epoch + 3600] * len(ts))


def lock_waits():
    """Three episodes in the G003 window: a reading every 30 s while the wait lasts, then a
    staleness marker. The value is the blocked statement's age, so it grows within an episode."""
    pairs = [  # (start, readings, blocked pid/qid/table, blocker pid/qid/table)
        (AT_MS - 5 * DAY + 3_600_000, 6, ("4101", "-6445724505443009296", "pgbench_branches"),
         ("4102", "-7648445788243534873", "pgbench_tellers")),
        (AT_MS - 2 * DAY + 600_000, 9, ("5201", "2713108609228007267", "pgbench_accounts"),
         ("5202", "-6445724505443009296", "pgbench_branches")),
        (AT_MS - 10 * 3_600_000 + 45_000, 4, ("5301", "2713108609228007267", "pgbench_accounts"),
         ("5302", "-6445724505443009296", "pgbench_branches")),
    ]
    for start, n, (bd_pid, bd_qid, bd_tbl), (bk_pid, bk_qid, bk_tbl) in pairs:
        lab = dict(db_labels("target_database"), blocked_pid=bd_pid, blocked_user="app", blocked_appname="pgbench",
                   blocked_table=bd_tbl, blocked_query_id=bd_qid, blocker_pid=bk_pid, blocker_user="app",
                   blocker_appname="pgbench", blocker_table=bk_tbl, blocker_query_id=bk_qid)
        ts = [start + 30_000 * i for i in range(n + 1)]
        yield series("pgwatch_lock_waits_blocked_ms", lab, ts, [1500 + 30_000 * i for i in range(n)] + [None])


INVALID = [  # (schema, table, index, valid duplicate, readings from, readings to; None = still there)
    ("public", "pgbench_accounts", "pgbench_accounts_abalance_idx_ccnew", "pgbench_accounts_abalance_idx",
     8 * DAY, None),
    ("app", "t10", "t10_c_idx", "", 3 * DAY, None),
    ("app", "t30", "t30_gone_idx", "", 6 * DAY, 4 * DAY),  # dropped since
    ("pg_catalog", "pg_class", "pg_class_bogus_idx", "", 8 * DAY, None),  # a system schema: skipped
]


def invalid_indexes():
    """pg_invalid_indexes runs every 2 h (metrics.yml): one series per field, labelled by index."""
    for n, (schema, table, index, valid, since, until) in enumerate(INVALID):
        ts = [t for t in range(AT_MS - since + 1_000, AT_MS - (until or 0), 7_200_000)]
        if until:
            ts.append(ts[-1] + 1)
        size = (n + 1) * 8 * 1024 ** 2
        lab = dict(db_labels("target_database"), schema_name=schema, table_name=table, index_name=index,
                   relation_name=f"{schema}.{table}",
                   index_definition=f"CREATE INDEX {index} ON {schema}.{table} USING btree (c)",
                   valid_index_name=valid,
                   valid_index_definition=f"CREATE INDEX {valid} ON {schema}.{table} USING btree (c)" if valid else "",
                   constraint_name="")
        k = len(ts) - bool(until)
        stale = [None] if until else []
        fields = {"index_size_bytes": size, "supports_fk": n % 2, "is_pk": 0, "is_unique": int(n == 1),
                  "has_valid_duplicate": int(bool(valid)), "table_row_estimate": 100_000 * (n + 1),
                  "is_ready": int(n != 1), "build_in_progress": 0}
        for field, value in fields.items():
            yield series(f"pgwatch_pg_invalid_indexes_{field}", lab, ts, [value] * k + stale)


def kcache():
    ts = grid(AT_MS - 26 * 3_600_000, AT_MS, 60_000)
    for n, qid in enumerate(("-6445724505443009296", "-7648445788243534873", "2713108609228007267")):
        lab = dict(db_labels("target_database"), queryid=qid, tag_user="app")
        user = [round(10 * (n + 1) + 0.25 * i, 2) for i in range(len(ts))]
        system = [round(2 * (n + 1) + 0.05 * i, 2) for i in range(len(ts))]
        yield series("pgwatch_pg_stat_kcache_exec_user_time", lab, ts, user)
        yield series("pgwatch_pg_stat_kcache_exec_system_time", lab, ts, system)
        yield series("pgwatch_pg_stat_kcache_exec_total_time", lab, ts, [round(u + s, 2) for u, s in zip(user, system)])


def btree_bloat_bytes():
    """The older pgwatch config's btree bloat gauges (real_size / table_size in bytes, no *_mib),
    in the postgres database: F005 falls back to them when the MiB gauges are missing."""
    ts = grid(AT_MS - 8 * DAY, AT_MS, 3_600_000)
    for n, (table, index) in enumerate((("audit_log", "audit_log_pkey"), ("audit_log", "audit_log_ts_idx"))):
        lab = dict(db_labels("postgres"), schemaname="public", tblname=table, idxname=index)
        real = [(64 + 16 * n) * 1024 ** 2 + i * 65536 for i in range(len(ts))]
        bloat = [round(r * (0.30 + 0.1 * n)) for r in real]
        yield series("pgwatch_pg_btree_bloat_real_size", lab, ts, real)
        yield series("pgwatch_pg_btree_bloat_table_size", lab, ts, [512 * 1024 ** 2] * len(ts))
        yield series("pgwatch_pg_btree_bloat_extra_size", lab, ts, bloat)
        yield series("pgwatch_pg_btree_bloat_extra_pct", lab, ts, [30.0 + 10 * n] * len(ts))
        yield series("pgwatch_pg_btree_bloat_bloat_size", lab, ts, bloat)
        yield series("pgwatch_pg_btree_bloat_bloat_pct", lab, ts, [30.0 + 10 * n] * len(ts))
        yield series("pgwatch_pg_btree_bloat_fillfactor", lab, ts, [90] * len(ts))
        yield series("pgwatch_pg_btree_bloat_is_na", lab, ts, [0] * len(ts))


def settings_normalized():
    """setting_normalized (bytes or seconds) for the five settings dashboard 05 panel 14 shows.
    The recording lacks it: pgai-mon did not store it before #340. Each series has the labels of
    its pgwatch_settings_configured series in samples.jsonl.gz, on grid(): every 30 minutes, then
    every 5 minutes in the last 26 h."""
    ts = grid(AT_MS - 8 * DAY, AT_MS, 300_000)
    for name, value, unit, category, normalized in (
            ("wal_keep_size", "0", "MB", "Replication / Sending Servers", 0),
            ("max_slot_wal_keep_size", "-1", "MB", "Replication / Sending Servers", -(1 << 20)),
            ("min_wal_size", "80", "MB", "Write-Ahead Log / Checkpoints", 80 << 20),
            ("max_wal_size", "1024", "MB", "Write-Ahead Log / Checkpoints", 1 << 30),
            ("checkpoint_timeout", "300", "s", "Write-Ahead Log / Checkpoints", 300)):
        lab = dict(db_labels("target_database"), category=category, setting_name=name, setting_value=value,
                   unit=unit, unit_normalized="seconds" if unit == "s" else "bytes", vartype="integer")
        yield series("pgwatch_settings_setting_normalized", lab, ts, [normalized] * len(ts))


def archiver_deadlocks_up():
    """K005's archiver (the real slice has archive_mode off), a deadlock counter that rises and is
    reset (the real one is all zeros), and the monitoring stack's `up` (the real slice has none):
    one target of another cluster, which A005 and K006-K008 leave out, and one that went down."""
    ts = grid(AT_MS - 2 * DAY, AT_MS, 300_000)
    n, half = len(ts), len(ts) // 2
    lab = dict(db_labels("target_database"), status="ok", wal_segment_size="16 MB")
    yield series("pgwatch_archive_lag_archived_count", lab, ts, [1000 + 3 * i for i in range(n)])
    yield series("pgwatch_archive_lag_failed_count", lab, ts, [0] * half + [2] * (n - half))
    yield series("pgwatch_archive_lag_seconds_since_archive", lab, ts, [wave(i, 5, 290, 37) for i in range(n)])
    yield series("pgwatch_db_stats_deadlocks", dict(db_labels("postgres"), env="deadlocks"), ts,
                 [i // 20 for i in range(half)] + [(i - half) // 15 for i in range(half, n)])
    for labels, values in (({"job": "pgwatch-prometheus", "instance": "pgwatch:9091"}, [1] * n),
                           ({"job": "pgwatch-prometheus", "instance": "other:9091", "cluster": "other"}, [1] * n),
                           ({"job": "self-node-exporter", "instance": "self-node-exporter:9100"}, [1] * (n - 3) + [0] * 3)):
        yield series("up", labels, ts, values)


def main():
    out = HERE / "synthetic.jsonl.gz"
    rows = [*host(), *resource_sizing_and_spills(), *stats_reset(), *lock_waits(), *invalid_indexes(), *kcache(), *btree_bloat_bytes(),
            *settings_normalized(), *archiver_deadlocks_up()]
    with gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode())
    print(f"{len(rows)} series, {sum(len(r['timestamps']) for r in rows)} samples -> {out}")


if __name__ == "__main__":
    main()
