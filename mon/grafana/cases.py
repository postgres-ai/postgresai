"""Dashboard cases: each PromQL dashboard target with its ported SQL (DuckDB) and parameters.
compile.py reads them to map every PromQL target of the source dashboards to SQL.

Each case: id, kind (instant|range), promql (string.Template: $cluster, $node, $db, plus any
`params`/`prepare` output), and either `sql` (inline) or `sql_file` (relative to mon/).
SQL reads its parameters with getvariable(): start_ms, end_ms, step_ms (instant: start=end=T,
step=300000), cluster, node, db, and the case's own params.

Result contract for SQL: a `value` column, a `t` column (epoch ms) for range queries, and labels
either as a `labels` MAP or as plain VARCHAR columns (columns starting with '_' are ignored).
`keyed=True` (as dash/ash_by.sql): the SQL ends with `SELECT <its columns> FROM r JOIN names USING
(k)`, points r(k, t, value) and one names row per series (every other column); compile.py builds
legends from `names` once per series instead of once per point.
"""

CASES = []
# ============================================================================================
# Dashboards
# ============================================================================================
DASH_WINDOWS = ["10m:15s", "1h:30s", "1h:20s:7", "6h:1m", "24h:5m", "14d:1h"]
D1_METRICS = [("calls", False), ("exec_time_total", True), ("plan_time_total", True), ("rows", True),
              ("shared_bytes_hit_total", True), ("shared_bytes_read_total", True),
              ("shared_bytes_written_total", True), ("shared_bytes_dirtied_total", True),
              ("wal_bytes", True), ("wal_fpi", True), ("temp_bytes_read", True), ("temp_bytes_written", True)]
_INFO = "(topk by (queryid) (1, tlast_over_time(pgwatch_query_info[7d])) > bool 0)"
_GL = ("group_left(displayname, displayname_medium, displayname_long, displayname_raw_short, "
       "displayname_raw_medium, displayname_raw_long, displayname_full, displayname_queryid)")
for m, has_per_call in D1_METRICS:
    for per_call in ((False, True) if has_per_call else (False,)):
        if m == "calls" and per_call:
            continue
        sel = "{cluster='$cluster', node_name='$node', datname=~\"$db_re\"}"
        ir = f"irate(pgwatch_pg_stat_statements_{m}{sel}[${{rate_interval_s}}s])"
        if per_call:
            ir = f"{ir}/irate(pgwatch_pg_stat_statements_calls{sel}[${{rate_interval_s}}s])"
        top = f"topk($top_n, {ir})"
        CASES.append(dict(
            id=f"D1/{m}{'_per_call' if per_call else '_per_s'}", kind="range", windows=DASH_WINDOWS, mode="topk",
            sql_file="sql/reports/d1_query_analysis_topn.sql", keyed=True,
            params={"metric": f"pgwatch_pg_stat_statements_{m}", "per_call": per_call, "top_n": 10,
                    "db_re": "target_database|postgres"},
            promql=f"({top} * on(queryid) {_GL} {_INFO}) or ({top} unless on(queryid) {_INFO})"))

# --- Dashboard_1_Node_performance_overview.json (census D2) ---
_D2 = "sql/reports/d2/"
_dsel = "{cluster='$cluster', node_name='$node', datname=~\"$db_re\"}"
_dsel2 = '{cluster="$cluster", node_name="$node", datname=~"$db_re"}'
_RI = "[${rate_interval_s}s]"
_I = "[${interval_s}s]"
_DB = {"db_re": "target_database"}
_CNv = [["cluster", "=", "local"], ["node_name", "=", "node-01"]]
_CNDv = _CNv + [["datname", "=~", "target_database"]]


def d2(id_, file, promql, **params):
    p = dict(_DB)
    extra = {k: params.pop(k) for k in ("mode", "note", "kind", "windows") if k in params}
    p.update(params)
    c = dict(id="D2/" + id_, kind=extra.get("kind", "range"), windows=extra.get("windows", DASH_WINDOWS),
             sql_file=_D2 + file if "/" not in file else file, params=p, promql=promql)
    c.update({k: v for k, v in extra.items() if k not in ("kind", "windows")})
    CASES.append(c)


_ash = ("sum by (wait_event_type) ((sum by (wait_event_type, instance) (sum_over_time(pgwatch_wait_events_total{%s}" + _I +
        ")) default 0) / on (instance) group_left() (sum by (instance) (sum_over_time(up{job=\"pgwatch-prometheus\"}" + _I + ")) > 0))")
d2("38A_ash", "ash.sql", _ash % 'cluster="$cluster", node_name="$node", datname=~"$db_re"', variant="a")
d2("38B_ash_server", "ash.sql", _ash % 'cluster="$cluster", node_name="$node", datname="server_process", wait_event_type!="Activity"', variant="b")
d2("38C_ash_server_activity", "ash.sql", _ash % 'cluster="$cluster", node_name="$node", datname="server_process", wait_event_type="Activity"', variant="c")
d2("4A_sessions", "sum_by_state_gt0.sql", f"sum by (state) (pgwatch_pg_stat_activity_count{_dsel}) > 0", non_idle=False)
d2("39A_non_idle", "sum_by_state_gt0.sql",
   "sum by (state) (pgwatch_pg_stat_activity_count{cluster='$cluster', node_name='$node', datname=~\"$db_re\", state!=\"idle\"}) > 0",
   non_idle=True)
d2("4D_max_connections", "raw_selector.sql", 'pgwatch_settings_numeric_value{setting_name="max_connections"}',
   metric="pgwatch_settings_numeric_value", matchers=[["setting_name", "=", "max_connections"]])
d2("39D_max_connections_legacy", "raw_selector.sql", "pgwatch_settings_max_connections",
   metric="pgwatch_settings_max_connections", matchers=[])
for ref, m, div in (("7B", "db_stats_xact_rollback", 1), ("7A", "db_stats_xact_commit", 1), ("6A", "pg_stat_statements_calls", 1),
                    ("10A", "pg_stat_statements_exec_time_total", 1000), ("10B", "pg_stat_statements_plan_time_total", 1000),
                    ("25A", "wal_xlog_location_b", 1)):
    d2(f"{ref}_sum_irate_{m}", "sum_irate.sql",
       f"sum(irate(pgwatch_{m}{_dsel}{_RI}))" + (f"/{div}" if div != 1 else ""), metric="pgwatch_" + m, divisor=div)
for ref, m in (("48A", "exec_time_total"), ("48B", "plan_time_total")):
    d2(f"{ref}_latency_{m}", "latency_ratio.sql",
       f"sum(irate(pgwatch_pg_stat_statements_{m}{_dsel}{_RI}))/1000/ sum(irate(pgwatch_pg_stat_statements_calls{_dsel}{_RI}))",
       metric="pgwatch_pg_stat_statements_" + m)
for ref, m in (("12A", "tup_returned"), ("12B", "tup_fetched"), ("47A", "tup_inserted"), ("47B", "tup_updated"),
               ("47C", "tup_deleted"), ("14A", "blks_read"), ("14B", "blks_hit"), ("49A", "blk_read_time"),
               ("49B", "blk_write_time"), ("29A", "temp_bytes")):
    d2(f"{ref}_irate_{m}", "irate_series.sql", f"irate(pgwatch_db_stats_{m}{_dsel2}{_RI})", metric="pgwatch_db_stats_" + m)
for ref, metric, m, w in (("14C", "pgwatch_settings_numeric_value", [["setting_name", "=", "block_size"]], "1h"),
                          ("46A", "pgwatch_multixact_size_members_bytes", _CNv, "1h"),
                          ("51A", "pgwatch_pg_wal_size_bytes", _CNv, "10m"),
                          ("52A", "pgwatch_pg_wal_size_status_code", _CNv, "10m")):
    sel = "{" + ", ".join(f'{a}{op}"{v}"' for a, op, v in m) + "}"
    from diff import dur as _dur
    d2(f"{ref}_last_over_time", "sql/reports/last_over_time.sql", f"last_over_time({metric}{sel}[{w}])",
       metric=metric, matchers=m, window_ms=_dur(w) * 1000)
d2("32A_locks_by_mode", "locks_rate.sql", f"sum by (lockmode) (rate(pgwatch_locks_mode_count{_dsel}{_RI}) * 60)")
for ref, metric in (("33A", "pgwatch_pg_long_running_transactions_age_in_seconds"),
                    ("34A", "pgwatch_pg_database_wraparound_age_datfrozenxid"),
                    ("35A", "pgwatch_pg_database_wraparound_age_datminmxid")):
    d2(f"{ref}_raw", "raw_selector.sql", f"{metric}{_dsel}", metric=metric, matchers=_CNDv)
for ref, metric in (("50A", "pgwatch_xmin_horizon_data_horizon_age_tx"), ("50B", "pgwatch_xmin_horizon_catalog_horizon_age_tx")):
    d2(f"{ref}_raw", "raw_selector.sql", f'{metric}{{cluster="$cluster", node_name="$node"}}', metric=metric, matchers=_CNv)
for ref, metric, gt0 in (("34B", "pgwatch_pg_settings_wraparound_autovacuum_freeze_max_age", False),
                         ("34C", "pgwatch_pg_settings_wraparound_vacuum_failsafe_age", True),
                         ("35B", "pgwatch_pg_settings_wraparound_autovacuum_multixact_freeze_max_age", False),
                         ("35C", "pgwatch_pg_settings_wraparound_vacuum_multixact_failsafe_age", True)):
    d2(f"{ref}_max_last", "max_last_over_time.sql",
       f'max(last_over_time({metric}{{cluster="$cluster", node_name="$node"}}[5m]))' + (" > 0" if gt0 else ""),
       metric=metric, gt0=gt0)
d2("34D_const", "constant.sql", "vector(2137483648)", value=2137483648)
d2("35D_const", "constant.sql", "vector(2137483648)", value=2137483648)
d2("46B_const", "constant.sql", "10737418240", value=10737418240)
_bs = 'pgwatch_settings_numeric_value{datname=~"$db_re", setting_name="block_size"}'
d2("41A_checkpointer_bytes", "irate_times_setting.sql",
   f'irate(pgwatch_checkpointer_buffers_written{{datname=~"$db_re"}}{_RI}) * on(datname) {_bs}',
   metric="pgwatch_checkpointer_buffers_written", matchers=[["datname", "=~", "target_database"]])
d2("41B_bgwriter_clean_bytes", "irate_times_setting.sql",
   f"irate(pgwatch_bgwriter_buffers_clean{_RI}) * on(datname) {_bs}", metric="pgwatch_bgwriter_buffers_clean", matchers=[])
d2("41C_backend_bytes", "irate_times_setting.sql",
   f"irate(pg_stat_bgwriter_buffers_backend_total{_RI}) * on(datname) {_bs}", metric="pg_stat_bgwriter_buffers_backend_total", matchers=[])
_vac = " or ".join(
    f'group by (schema_name, table_name, vacuum_mode) (pgwatch_pg_vacuum_progress_index_vacuum_count{{cluster="$cluster", '
    f'node_name="$node", datname="$db", phase="{p}"}}) * 0 + {p}' for p in range(7, 0, -1))
d2("42A_vacuum_timeline", "vacuum_timeline.sql", f"({_vac})")
for ref, m in (("44A", "total_toast_size_b"), ("44B", "table_main_size_b"), ("44C", "table_indexes_size_b"),
               ("44D", "table_vm_size_b"), ("44E", "table_fsm_size_b")):
    d2(f"{ref}_sum_{m}", "sum_series.sql",
       f'sum(pgwatch_table_size_detailed_{m}{{datname=~"$db_re", cluster="$cluster", node_name="$node"}})',
       metric="pgwatch_table_size_detailed_" + m)

# host_* panels: absent in the reference (EMPTY there); run against the synthetic VM with --cases D2/host
_hs = '{cluster="$cluster", node_name="$node"}'
for ref, metric in (("63A", "host_provider_info"), ("55F", "host_os_memory_total_bytes"), ("55G", "host_os_swap_used_bytes"),
                    ("55H", "host_os_process_max_rss_bytes"), ("56B", "host_disk_free_bytes"), ("56C", "host_disk_size_bytes"),
                    ("56D", "host_local_storage_free_bytes"), ("56E", "host_volume_used_bytes"), ("64A", "host_disk_read_iops"),
                    ("64B", "host_volume_read_iops"), ("65A", "host_disk_write_iops"), ("65B", "host_volume_write_iops"),
                    ("66A", "host_network_receive_bytes_per_second"), ("67A", "host_network_transmit_bytes_per_second"),
                    ("62A", "host_db_load")):
    d2(f"host/{ref}_raw", "raw_selector.sql", f"{metric}{_hs}", metric=metric, matchers=_CNv)
_cpu4 = ["host_os_cpu_user_percent", "host_os_cpu_system_percent", "host_os_cpu_iowait_percent", "host_os_cpu_steal_percent"]
_mem4 = ["host_os_memory_total_bytes", "host_os_memory_free_bytes", "host_os_memory_cached_bytes", "host_os_memory_buffers_bytes"]


def _and_chain(ms):
    return " and on(cluster, node_name) ".join(f"{m}{_hs}" for m in ms)


for ref, metric in zip(("54A", "54B", "54C", "54D"), _cpu4):
    d2(f"host/{ref}_and", "host_and_unless.sql", f"{metric}{_hs} and on(cluster, node_name) ({_and_chain(_cpu4)})",
       metric=metric, others=_cpu4, op="and")
for ref, metric in (("55B", "host_os_memory_buffers_bytes"), ("55C", "host_os_memory_cached_bytes"), ("55D", "host_os_memory_free_bytes")):
    d2(f"host/{ref}_and", "host_and_unless.sql", f"{metric}{_hs} and on(cluster, node_name) ({_and_chain(_mem4)})",
       metric=metric, others=_mem4, op="and")
d2("host/55E_unless", "host_and_unless.sql",
   f"host_memory_available_bytes{_hs} unless on(cluster, node_name) ({_and_chain(_mem4)})",
   metric="host_memory_available_bytes", others=_mem4, op="unless")
_total = (f"max by (cluster, node_name) (host_os_cpu_percent{_hs} or on(cluster, node_name) "
          f"(host_cpu_utilization_percent{_hs} unless on(cluster, node_name) host_os_cpu_percent{_hs}))")
d2("host/54E_cpu_other", "host_cpu.sql", f"clamp_min({_total} - " + " - ".join(f"{m}{_hs}" for m in _cpu4) + ", 0)", variant="other")
d2("host/54F_cpu_total", "host_cpu.sql", f"{_total} unless on(cluster, node_name) ({_and_chain(_cpu4)})", variant="total_unless_split")
d2("host/54G_cpu_idle", "host_cpu.sql", f"clamp_min(100 - {_total}, 0)", variant="idle")
d2("host/55A_mem_used", "host_mem_used.sql",
   f"(host_os_memory_total_bytes{_hs} - host_os_memory_free_bytes{_hs} - host_os_memory_cached_bytes{_hs} - "
   f"host_os_memory_buffers_bytes{_hs}) or on(cluster, node_name) (host_os_memory_total_bytes{_hs} - host_memory_available_bytes{_hs})")
d2("host/56A_disk_used", "host_disk_used.sql", f"host_disk_size_bytes{_hs} - host_disk_free_bytes{_hs}")

# Dashboard_99 (Host stats): the same host_* shapes as Dashboard_1's Host row, with its own panel ids
# (101 CPU, 102 memory, 103 storage, 104 network, 105-110 disk read/write IOPS, throughput,
# latency, 62 DB load), plus the per-direction disk series and the provisioned limits


def d99(id_, file, promql, **params):
    d2(id_, file, promql, **params)
    CASES[-1]["id"] = "D99/" + id_


for ref, metric in (("102F", "host_os_memory_total_bytes"), ("102G", "host_os_swap_used_bytes"),
                    ("102H", "host_os_process_max_rss_bytes"), ("103B", "host_disk_free_bytes"),
                    ("103C", "host_disk_size_bytes"), ("103D", "host_local_storage_free_bytes"),
                    ("103E", "host_volume_used_bytes"), ("104A", "host_network_receive_bytes_per_second"),
                    ("104B", "host_network_transmit_bytes_per_second"), ("105A", "host_disk_read_iops"),
                    ("105B", "host_volume_read_iops"), ("105C", "host_disk_provisioned_iops"),
                    ("106A", "host_disk_read_bytes_per_second"),
                    ("106B", "host_disk_provisioned_throughput_bytes_per_second"),
                    ("107A", "host_disk_read_latency_seconds"), ("108A", "host_disk_write_iops"),
                    ("108B", "host_volume_write_iops"), ("108C", "host_disk_provisioned_iops"),
                    ("109A", "host_disk_write_bytes_per_second"),
                    ("109B", "host_disk_provisioned_throughput_bytes_per_second"),
                    ("110A", "host_disk_write_latency_seconds"), ("62A", "host_db_load")):
    d99(f"{ref}_raw", "raw_selector.sql", f"{metric}{_hs}", metric=metric, matchers=_CNv)
for ref, metric in zip(("101A", "101B", "101C", "101D"), _cpu4):
    d99(f"{ref}_and", "host_and_unless.sql", f"{metric}{_hs} and on(cluster, node_name) ({_and_chain(_cpu4)})",
        metric=metric, others=_cpu4, op="and")
for ref, metric in (("102B", "host_os_memory_buffers_bytes"), ("102C", "host_os_memory_cached_bytes"),
                    ("102D", "host_os_memory_free_bytes")):
    d99(f"{ref}_and", "host_and_unless.sql", f"{metric}{_hs} and on(cluster, node_name) ({_and_chain(_mem4)})",
        metric=metric, others=_mem4, op="and")
d99("102E_unless", "host_and_unless.sql",
    f"host_memory_available_bytes{_hs} unless on(cluster, node_name) ({_and_chain(_mem4)})",
    metric="host_memory_available_bytes", others=_mem4, op="unless")
d99("101E_cpu_other", "host_cpu.sql", f"clamp_min({_total} - " + " - ".join(f"{m}{_hs}" for m in _cpu4) + ", 0)",
    variant="other")
d99("101F_cpu_total", "host_cpu.sql", f"{_total} unless on(cluster, node_name) ({_and_chain(_cpu4)})",
    variant="total_unless_split")
d99("101G_cpu_idle", "host_cpu.sql", f"clamp_min(100 - {_total}, 0)", variant="idle")
d99("102A_mem_used", "host_mem_used.sql",
    f"(host_os_memory_total_bytes{_hs} - host_os_memory_free_bytes{_hs} - host_os_memory_cached_bytes{_hs} - "
    f"host_os_memory_buffers_bytes{_hs}) or on(cluster, node_name) (host_os_memory_total_bytes{_hs} - "
    f"host_memory_available_bytes{_hs})")
d99("103A_disk_used", "host_disk_used.sql", f"host_disk_size_bytes{_hs} - host_disk_free_bytes{_hs}")

# ============================================================================================
# Dashboards 4, 3, 7, 13 (census ids D4/, D3/, D7/, D13/: "<panel id><refId>_<what>", the key
# grafana/compile.py matches targets on). Shapes shared with Dashboard_1 reuse sql/reports/d2/;
# new shapes are in sql/reports/dash/.
# ============================================================================================
_DASH = "sql/reports/dash/"
_CNDq = [["cluster", "=", "local"], ["node_name", "=", "node-01"], ["datname", "=~", "target_database"]]


def dash(census, id_, file, promql, **params):
    p = dict(_DB)
    extra = {k: params.pop(k) for k in ("mode", "note", "kind", "windows") if k in params}
    p.update(params)
    path = file if "/" in file else _DASH + file
    c = dict(id=f"{census}/{id_}", kind=extra.get("kind", "range"), windows=extra.get("windows", DASH_WINDOWS),
             sql_file=path, params=p, promql=promql)
    c.update({k: v for k, v in extra.items() if k not in ("kind", "windows")})
    CASES.append(c)


def _ash_by(by, sel):
    b = ", ".join(by)
    return (f"sum by ({b}) ((sum by ({b}, instance) (sum_over_time(pgwatch_wait_events_total{{{sel}}}{_I})) default 0)"
            f" / on (instance) group_left() (sum by (instance) (sum_over_time(up{{job=\"pgwatch-prometheus\"}}{_I})) > 0))")


_wsel = 'cluster="$cluster", node_name="$node", datname=~"$db_re"'
# The reference values for the dashboards' own variables: a real subset, so the filters filter.
D4_WET = "Client|IO|LWLock|Lock"
D4_WE = "DataFileRead|ClientRead|WALWrite|WalSync|tuple"

# --- Dashboard_4_Wait_Sampling_Dashboard.json ---
# 1 A/B/C: the Dashboard_1 panel 38 shapes (1A adds sort_by_label, which orders series only)
dash("D4", "1A_ash", "sql/reports/d2/ash.sql", "sort_by_label(" + _ash % _wsel + ', "wait_event_type")', variant="a")
dash("D4", "1B_ash_server", "sql/reports/d2/ash.sql",
     _ash % 'cluster="$cluster", node_name="$node", datname="server_process", wait_event_type!="Activity"', variant="b")
dash("D4", "1C_ash_server_activity", "sql/reports/d2/ash.sql",
     _ash % 'cluster="$cluster", node_name="$node", datname="server_process", wait_event_type="Activity"', variant="c")
dash("D4", "2A_ash_by_event", "ash_by.sql",
     "sort_by_label(" + _ash_by(["wait_event_type", "wait_event"], _wsel + ', wait_event_type=~"$wet_re"') + ', "wait_event_type")',
     by=["wait_event_type", "wait_event"], filter="wet", wet_re=D4_WET, we_re="", query_id="")
dash("D4", "3A_ash_by_event_query", "ash_by.sql",
     "sort_by_label(" + _ash_by(["wait_event_type", "wait_event", "query_id"],
                                _wsel + ', wait_event_type=~"$wet_re", wait_event=~"$we_re"') + ', "wait_event_type", "query_id")',
     by=["wait_event_type", "wait_event", "query_id"], filter="wet_we", wet_re=D4_WET, we_re=D4_WE, query_id="")

# --- Dashboard_3_Single_query_analysis.json ($query_id: a query present in the reference store) ---
D3_QUERY_IDS = {"pgss": "5788417122087653992", "ash": "5788417122087653992"}  # a1: 7 wait-event series, 14 d of pgss
dash("D3", "19A_ash_query", "ash_by.sql",
     _ash_by(["wait_event_type", "wait_event"], _wsel + ', query_id="$query_id", query_id!=""'),
     by=["wait_event_type", "wait_event"], filter="query", wet_re="", we_re="", query_id=D3_QUERY_IDS["ash"])
_qsel = '{cluster="$cluster", node_name="$node", datname=~"$db_re", queryid="$query_id"}'
for ref, m, per_call in (("2A", "calls", False), ("3A", "exec_time_total", False), ("4A", "exec_time_total", True),
                         ("6A", "rows", False), ("7A", "rows", True), ("8A", "shared_bytes_hit_total", False),
                         ("9A", "shared_bytes_hit_total", True), ("10A", "wal_bytes", False), ("13A", "wal_bytes", True),
                         ("11A", "wal_fpi", False), ("14A", "wal_fpi", True), ("12A", "temp_bytes_read", False),
                         ("15A", "temp_bytes_read", True), ("16A", "temp_bytes_written", False),
                         ("17A", "temp_bytes_written", True)):
    ir = f"irate(pgwatch_pg_stat_statements_{m}{_qsel}{_RI})"
    if per_call:
        ir += f"/irate(pgwatch_pg_stat_statements_calls{_qsel}{_RI})"
    dash("D3", f"{ref}_{m}{'_per_call' if per_call else ''}", "irate_query.sql", ir,
         metric=f"pgwatch_pg_stat_statements_{m}", per_call=per_call, query_id=D3_QUERY_IDS["pgss"])

# --- Dashboard_7_Autovacuum_and_xmin_horizon.json ---
_SYS = [["schema", "!~", "pg_.*|information_schema|_timescaledb.*"]]
for ref, metric, extra in (("54A", "pgwatch_table_stats_tx_freeze_age", _SYS), ("55A", "pgwatch_table_stats_mxid_freeze_age", _SYS),
                           ("91A", "pgwatch_pg_autovacuum_queue_autovacuum_overdue_factor", [])):
    sel = '{cluster="$cluster", node_name="$node", datname=~"$db_re"' + \
          ("".join(f', {a}{op}"{v}"' for a, op, v in extra)) + "}"
    dash("D7", f"{ref}_topk_last", "topk_last_over_time.sql", f"topk($top_n, last_over_time({metric}{sel}[5m]))",
         metric=metric, matchers=_CNDq + extra, window_ms=300000, top_n=10, mode="topk")
for ref, metric, gt0 in (("54B", "pgwatch_pg_settings_wraparound_autovacuum_freeze_max_age", False),
                         ("54C", "pgwatch_pg_settings_wraparound_vacuum_failsafe_age", True),
                         ("55B", "pgwatch_pg_settings_wraparound_autovacuum_multixact_freeze_max_age", False),
                         ("55C", "pgwatch_pg_settings_wraparound_vacuum_multixact_failsafe_age", True)):
    dash("D7", f"{ref}_max_last", "sql/reports/d2/max_last_over_time.sql",
         f'max(last_over_time({metric}{{cluster="$cluster", node_name="$node"}}[5m]))' + (" > 0" if gt0 else ""),
         metric=metric, gt0=gt0)
dash("D7", "54D_const", "sql/reports/d2/constant.sql", "vector(2137483648)", value=2137483648)
dash("D7", "55D_const", "sql/reports/d2/constant.sql", "vector(2137483648)", value=2137483648)
dash("D7", "91B_const", "sql/reports/d2/constant.sql", "vector(1)", value=1)
_cn = '{cluster="$cluster", node_name="$node"}'
for ref, metric in (("11A", "pgwatch_xmin_horizon_data_horizon_age_tx"), ("11B", "pgwatch_xmin_horizon_catalog_horizon_age_tx"),
                    ("12A", "pgwatch_xmin_horizon_pg_stat_activity_count"),
                    ("12B", "pgwatch_xmin_horizon_pg_replication_slots_count"),
                    ("12C", "pgwatch_xmin_horizon_pg_replication_slots_catalog_count"),
                    ("12D", "pgwatch_xmin_horizon_pg_stat_replication_count"),
                    ("12E", "pgwatch_xmin_horizon_pg_prepared_xacts_count"),
                    ("92A", "pgwatch_pg_autovacuum_workers_active_workers"),
                    ("92B", "pgwatch_pg_autovacuum_workers_max_workers")):
    dash("D7", f"{ref}_raw", "sql/reports/d2/raw_selector.sql", f"{metric}{_cn}", metric=metric, matchers=_CNv)
for ref, comp in (("11C", "pg_stat_activity"), ("11D", "pg_replication_slots"), ("11E", "pg_replication_slots_catalog"),
                  ("11F", "pg_stat_replication"), ("11G", "pg_prepared_xacts")):
    dash("D7", f"{ref}_raw_blockers", "sql/reports/d2/raw_selector.sql",
         f'pgwatch_xmin_horizon_blockers_age_tx{{cluster="$cluster", node_name="$node", component="{comp}"}}',
         metric="pgwatch_xmin_horizon_blockers_age_tx", matchers=_CNv + [["component", "=", comp]])
dash("D7", "3A_raw", "sql/reports/d2/raw_selector.sql",
     "pgwatch_pg_long_running_transactions_age_in_seconds{cluster='$cluster', node_name='$node', datname=~\"$db_re\"}",
     metric="pgwatch_pg_long_running_transactions_age_in_seconds", matchers=_CNDq)
dash("D7", "92C_pct", "scaled_ratio.sql",
     f"100 * pgwatch_pg_autovacuum_workers_active_workers{_cn} / clamp_min(pgwatch_pg_autovacuum_workers_max_workers{_cn}, 1)",
     metric="pgwatch_pg_autovacuum_workers_active_workers", den_metric="pgwatch_pg_autovacuum_workers_max_workers",
     matchers=_CNv, scale=100, den_min=1)
dash("D7", "93A_instant", "sql/reports/last_over_time.sql",
     'last_over_time(pgwatch_pg_autovacuum_blocked_wait_seconds{cluster="$cluster", node_name="$node", datname=~"$db_re"}[1m])',
     metric="pgwatch_pg_autovacuum_blocked_wait_seconds", matchers=_CNDq, window_ms=60000, kind="instant")
_vac7 = " or ".join(
    f'group by (schema_name, table_name, vacuum_mode) (pgwatch_pg_vacuum_progress_index_vacuum_count{{cluster="$cluster", '
    f'node_name="$node", datname="$db", phase="{p}", schema_name=~"$schema_re", table_name=~"$table_re"}}) * 0 + {p}'
    for p in range(7, 0, -1))
dash("D7", "2A_vacuum_timeline", "vacuum_timeline_st.sql", f"({_vac7})", schema_re=".*", table_re=".*")

# --- Dashboard_13_Lock_waits.json ---
_lw = '{cluster="$cluster", node_name="$node", datname=~"$db_re"}'
_LWB = "pgwatch_lock_waits_blocked_ms"
dash("D13", "1A_count_or_0", "count_or_zero.sql", f"count({_LWB}{_lw}) or vector(0)", metric=_LWB, matchers=_CNDq)
for ref, op, metric in (("2A", "avg", _LWB), ("2B", "max", _LWB),
                        ("5A", "avg", "pgwatch_lock_waits_blocker_tx_ms"), ("5B", "max", "pgwatch_lock_waits_blocker_tx_ms")):
    dash("D13", f"{ref}_{op}", "agg_last.sql", f"{op}({metric}{_lw})", op=op, metric=metric, matchers=_CNDq, by=[])
dash("D13", "3A_count_by_locktype", "agg_last.sql", f"count by (blocked_locktype) ({_LWB}{_lw})",
     op="count", metric=_LWB, matchers=_CNDq, by=["blocked_locktype"])
dash("D13", "6A_count_by_table", "agg_last.sql",
     f'count by (blocked_table) ({_LWB}{{cluster="$cluster", node_name="$node", datname=~"$db_re", blocked_table!=""}})',
     op="count", metric=_LWB, matchers=_CNDq + [["blocked_table", "!=", ""]], by=["blocked_table"])
dash("D13", "4A_instant", "sql/reports/d2/raw_selector.sql", f"{_LWB}{_lw}", metric=_LWB, matchers=_CNDq, kind="instant")

# --- Dashboard_12_SLRU.json: the Dashboard_1 irate_series shape ---
for ref, m in (("46A", "blks_hit"), ("52A", "blks_exists"), ("47A", "blks_read"), ("48A", "blks_written"),
               ("49A", "blks_zeroed"), ("50A", "truncates"), ("51A", "flushes")):
    dash("D12", f"{ref}_irate_{m}", "sql/reports/d2/irate_series.sql",
         f"irate(pgwatch_pg_stat_slru_{m}{_dsel2}{_RI})", metric="pgwatch_pg_stat_slru_" + m)

# --- Dashboard_14_IO_Statistics.json (no db variable) ---
for ref, m, total in (("2A", "read_bytes_mb", True), ("2B", "write_bytes_mb", True), ("3A", "read_time_ms", True),
                      ("3B", "write_time_ms", True), ("3C", "fsync_time_ms", True), ("5A", "reads", True),
                      ("5B", "writes", True), ("5C", "hits", True), ("11A", "reads", False), ("12A", "writes", False),
                      ("21A", "writebacks", True), ("22A", "fsyncs", True), ("23A", "extends", True),
                      ("31A", "evictions", True), ("31B", "reuses", True), ("32A", "hits", False)):
    op = "=" if total else "!="
    dash("D14", f"{ref}_irate_{m}{'' if total else '_by_backend'}", "irate_matchers.sql",
         f'irate(pgwatch_pg_stat_io_{m}{{cluster="$cluster", node_name="$node", backend_type{op}"total"}}{_RI})',
         metric="pgwatch_pg_stat_io_" + m, matchers=_CNv + [["backend_type", op, "total"]])
_tot = '{cluster="$cluster", node_name="$node", backend_type="total"}'
dash("D14", "4A_hit_ratio", "pct_of_sum.sql",
     f"100 * pgwatch_pg_stat_io_hits{_tot} / (pgwatch_pg_stat_io_hits{_tot} + pgwatch_pg_stat_io_reads{_tot})",
     metric="pgwatch_pg_stat_io_hits", other="pgwatch_pg_stat_io_reads", matchers=_CNv + [["backend_type", "=", "total"]])

# ============================================================================================
# Dashboards 5, 9, 11 and Dashboard_6's $__range targets. Selectors are matcher lists whose
# values are the reference store's; compile.py gives each matcher the dashboard variable that
# the target's own expression has for that label and operator (expr_matcher_vars).
# ============================================================================================
_CNmv = [["cluster", "=", "local"], ["node_name", "=", "node-01"]]
_cn5 = '{cluster="$cluster", node_name="$node"}'

# --- Dashboard_5_Backup_stats.json ---
dash("D5", "6A_last_1h", "sql/reports/last_over_time.sql", f"last_over_time(pgwatch_pg_wal_size_bytes{_cn5}[1h])",
     metric="pgwatch_pg_wal_size_bytes", matchers=_CNmv, window_ms=3600000)
dash("D5", "7A_sum_irate", "agg_fn.sql", f"sum(irate(pgwatch_wal_xlog_location_b{_cn5}{_RI}))",
     op="sum", fn="irate", metric="pgwatch_wal_xlog_location_b", matchers=_CNmv, by=[])
for ref, m in (("2B", "archived_count"), ("2A", "failed_count")):
    dash("D5", f"{ref}_irate_{m}", "irate_matchers.sql", f"irate(pgwatch_archive_lag_{m}{_cn5}{_RI})",
         metric=f"pgwatch_archive_lag_{m}", matchers=_CNmv)
_ok, _ko = f"sum(irate(pgwatch_archive_lag_archived_count{_cn5}{_RI}))", f"sum(irate(pgwatch_archive_lag_failed_count{_cn5}{_RI}))"
dash("D5", "3A_success_ratio", "sum_ratio_clamp.sql", f"clamp_max({_ok} / ({_ok} + {_ko}), 1)",
     metric="pgwatch_archive_lag_archived_count", other="pgwatch_archive_lag_failed_count", matchers=_CNmv)
dash("D5", "1A_lag_bytes", "binop_last.sql",
     f"pgwatch_archive_lag_current_lsn_numeric{_cn5} - pgwatch_archive_lag_archived_wal_finish_lsn_numeric{_cn5}",
     op="-", metric="pgwatch_archive_lag_current_lsn_numeric", other="pgwatch_archive_lag_archived_wal_finish_lsn_numeric",
     matchers=_CNmv)
for ref, m in (("9A", "seconds_since_archive"), ("9B", "wal_files_behind")):
    dash("D5", f"{ref}_raw", "sql/reports/d2/raw_selector.sql", f"pgwatch_archive_lag_{m}{_cn5}",
         metric=f"pgwatch_archive_lag_{m}", matchers=_CNmv)
dash("D5", "11A_max_by_slot", "agg_last.sql", f"max by (slot_name, plugin) (pgwatch_replication_slots_restart_lsn_lag_b{_cn5})",
     op="max", metric="pgwatch_replication_slots_restart_lsn_lag_b", matchers=_CNmv, by=["slot_name", "plugin"])
dash("D5", "12A_sum", "agg_last.sql", f"sum(pgwatch_replication_slots_non_active_int{_cn5})",
     op="sum", metric="pgwatch_replication_slots_non_active_int", matchers=_CNmv, by=[])
for ref, m, names in (("14A", "pgwatch_settings_configured",
                       "archive_mode|archive_command|archive_library|wal_keep_size|min_wal_size|max_wal_size|max_slot_wal_keep_size|checkpoint_timeout"),
                      ("14B", "pgwatch_settings_setting_normalized",
                       "wal_keep_size|min_wal_size|max_wal_size|max_slot_wal_keep_size|checkpoint_timeout")):
    dash("D5", f"{ref}_instant_range", "last_over_range.sql",
         f'last_over_time({m}{{cluster="$cluster", node_name="$node", setting_name=~"{names}"}}[${{range_s}}s])',
         metric=m, matchers=_CNmv + [["setting_name", "=~", names]], kind="instant")
dash("D5", "15A_topk_by_queryid", "topk_by.sql",
     f"topk(10, sum by (queryid) (irate(pgwatch_pg_stat_statements_wal_bytes{_dsel2}{_RI})))",
     k=10, fn="irate", metric="pgwatch_pg_stat_statements_wal_bytes", matchers=_CNDq, by=["queryid"], mode="topk")

# --- Dashboard_9_Single_table_analysis.json ($schema_name, $table_name: app.t112 on a1) ---
D9_SCHEMA, D9_TABLE = "app", "t135"  # a1: full 14 d, gaps in its TOAST statio series
_t9 = _CNDq + [["schema", "=", D9_SCHEMA], ["table_name", "=", D9_TABLE]]
_p9 = '{datname=~"$db_re", cluster="$cluster", node_name="$node", schema="' + D9_SCHEMA + '", table_name="' + D9_TABLE + '"}'
dash("D9", "38A_raw", "sql/reports/d2/raw_selector.sql", f"pgwatch_table_size_detailed_total_toast_size_b{_p9}",
     metric="pgwatch_table_size_detailed_total_toast_size_b", matchers=_t9)
for ref, m in (("38B", "table_main_size_b"), ("38C", "table_indexes_size_b"), ("38D", "table_vm_size_b"),
               ("38E", "table_fsm_size_b")):
    dash("D9", f"{ref}_sum", "agg_last.sql", f"sum(pgwatch_table_size_detailed_{m}{_p9})",
         op="sum", metric=f"pgwatch_table_size_detailed_{m}", matchers=_t9, by=[])
for ref, m in (("39A", "total_toast_size_b"), ("39B", "table_main_size_b"), ("39C", "table_indexes_size_b"),
               ("39D", "table_vm_size_b"), ("39E", "table_fsm_size_b")):
    dash("D9", f"{ref}_irate", "irate_matchers.sql", f"irate(pgwatch_table_size_detailed_{m}{_p9}{_RI})",
         metric=f"pgwatch_table_size_detailed_{m}", matchers=_t9)
_b9 = [["schemaname", "=", D9_SCHEMA], ["tblname", "=", D9_TABLE]]
for ref, m in (("33A", "bloat_pct"), ("32A", "bloat_size")):
    dash("D9", f"{ref}_raw", "sql/reports/d2/raw_selector.sql",
         f'pgwatch_pg_table_bloat_{m}{{schemaname="{D9_SCHEMA}", tblname="{D9_TABLE}"}}',
         metric=f"pgwatch_pg_table_bloat_{m}", matchers=_b9)
for ref, m in (("21A", "n_tup_ins"), ("21B", "n_tup_upd"), ("21C", "n_tup_hot_upd"), ("21D", "n_tup_del")):
    dash("D9", f"{ref}_sum_by_irate", "agg_fn.sql",
         f"sum by (node_name, table_full_name, dbname, cluster) (irate(pgwatch_table_stats_{m}{_p9}{_RI}))",
         op="sum", fn="irate", metric=f"pgwatch_table_stats_{m}", matchers=_t9,
         by=["node_name", "table_full_name", "dbname", "cluster"])
_t9re = _CNDq + [["schema", "=~", D9_SCHEMA], ["table_name", "=~", D9_TABLE]]
_p9re = '{datname=~"$db_re", cluster="$cluster", node_name="$node", schema=~"' + D9_SCHEMA + '", table_name=~"' + D9_TABLE + '"}'
for ref, m in (("37A", "n_tup_ins"), ("37B", "n_tup_upd"), ("37C", "n_tup_hot_upd"), ("37D", "n_tup_del")):
    dash("D9", f"{ref}_sum_irate", "agg_fn.sql", f"sum(irate(pgwatch_table_stats_{m}{_p9re}{_RI}))",
         op="sum", fn="irate", metric=f"pgwatch_table_stats_{m}", matchers=_t9re, by=[])
_s9 = _CNDq + [["schemaname", "=~", D9_SCHEMA], ["relname", "=", D9_TABLE]]
_ps9 = '{schemaname=~"' + D9_SCHEMA + '", datname=~"$db_re", cluster="$cluster", node_name="$node", relname="' + D9_TABLE + '"}'
for ref, m in (("42A", "seq_tup_read"), ("42B", "idx_tup_fetch")):
    dash("D9", f"{ref}_irate", "irate_matchers.sql", f"irate(pgwatch_pg_stat_all_tables_{m}{_ps9}{_RI})",
         metric=f"pgwatch_pg_stat_all_tables_{m}", matchers=_s9)
_bs9 = _CNDq + [["setting_name", "=", "block_size"]]
_pbs9 = 'last_over_time(pgwatch_settings_numeric_value{datname=~"$db_re", cluster="$cluster", node_name="$node", setting_name="block_size"}[1d])'
for ref, m in (("44B", "heap_blks_hit"), ("44C", "idx_blks_hit"), ("44D", "toast_blks_hit"), ("44E", "tidx_blks_hit"),
               ("45B", "heap_blks_read"), ("45C", "idx_blks_read"), ("45D", "toast_blks_read"), ("45E", "tidx_blks_read")):
    dash("D9", f"{ref}_irate_x_block_size", "irate_times_scalar.sql",
         f"irate(pgwatch_pg_statio_all_tables_{m}{_ps9}{_RI}) * on() group_left() {_pbs9}",
         metric=f"pgwatch_pg_statio_all_tables_{m}", matchers=_s9, smetric="pgwatch_settings_numeric_value",
         smatchers=_bs9, swindow_ms=86400000)


def _st(kind):
    return " + ".join(f"irate(pgwatch_pg_statio_all_tables_{x}_blks_{kind}{_ps9}{_RI})" for x in ("heap", "idx", "toast", "tidx"))


dash("D9", "46A_hit_ratio", "statio_ratio.sql", f"({_st('hit')})/({_st('read')} + ({_st('hit')}))", num="hit", matchers=_s9)
dash("D9", "46B_read_ratio", "statio_ratio.sql", f"({_st('read')})/({_st('read')} + ({_st('hit')}))", num="read", matchers=_s9)

# relkind: metrics.yml exports it as text (c.relkind::text: "r", "i"); the PromQL dashboards match
# the char's code ("114", "105"), which pgwatch exported before that cast and no store written since
# has. Both are matched, so stores with either keep working.
RELKIND_TABLE, RELKIND_INDEX = ["relkind", "=~", "114|r"], ["relkind", "=~", "105|i"]

# --- Dashboard_11_Single_index_analysis.json ($datname, $schema_name, $index_name) ---
D11_SCHEMA, D11_INDEX = "public", "pgbench_accounts_pkey"
_i11 = _CNmv + [["datname", "=", "target_database"], ["schemaname", "=", D11_SCHEMA]]
_pi11 = '{cluster="$cluster", node_name="$node", datname="$db", schemaname="' + D11_SCHEMA + '", indexrelname="' + D11_INDEX + '"}'
dash("D11", "3A_raw", "sql/reports/d2/raw_selector.sql",
     f'pgwatch_pg_class_relation_size_bytes{{relkind=~"105|i", cluster="$cluster", node_name="$node", datname="$db", '
     f'schemaname="{D11_SCHEMA}", relname="{D11_INDEX}"}}',
     metric="pgwatch_pg_class_relation_size_bytes", matchers=[RELKIND_INDEX] + _i11 + [["relname", "=", D11_INDEX]])
for ref, m in (("5A", "idx_scan"), ("6A", "idx_tup_read"), ("6B", "idx_tup_fetch")):
    dash("D11", f"{ref}_irate", "irate_matchers.sql", f"irate(pgwatch_pg_stat_all_indexes_{m}{_pi11}{_RI})",
         metric=f"pgwatch_pg_stat_all_indexes_{m}", matchers=_i11 + [["indexrelname", "=", D11_INDEX]])
for ref, m in (("4A", "idx_blks_read"), ("4B", "idx_blks_hit")):
    dash("D11", f"{ref}_irate_x_block_size", "irate_times_scalar.sql",
         f"irate(pgwatch_pg_statio_all_indexes_{m}{_pi11}{_RI}) * on() group_left() last_over_time(pgwatch_settings_numeric_value"
         '{cluster="$cluster", node_name="$node", datname="$db", setting_name="block_size"}[1d])',
         metric=f"pgwatch_pg_statio_all_indexes_{m}", matchers=_i11 + [["indexrelname", "=", D11_INDEX]],
         smetric="pgwatch_settings_numeric_value",
         smatchers=_CNmv + [["datname", "=", "target_database"], ["setting_name", "=", "block_size"]], swindow_ms=86400000)
_ri11 = [["relname", "=", D11_INDEX], ["schemaname", "=", D11_SCHEMA]]
dash("D11", "1A_boguk_ratio", "binop_last.sql",
     f'pgwatch_pg_class_total_relation_size_bytes{{relname="{D11_INDEX}", schemaname="{D11_SCHEMA}"}} / '
     f'pgwatch_pg_class_reltuples{{relname="{D11_INDEX}", schemaname="{D11_SCHEMA}"}}',
     op="/", metric="pgwatch_pg_class_total_relation_size_bytes", other="pgwatch_pg_class_reltuples", matchers=_ri11)
dash("D11", "1B_raw", "sql/reports/d2/raw_selector.sql",
     f'pgwatch_pg_index_pilot_best_ratio{{indexrelname="{D11_INDEX}", schemaname="{D11_SCHEMA}"}}',
     metric="pgwatch_pg_index_pilot_best_ratio", matchers=[["indexrelname", "=", D11_INDEX], ["schemaname", "=", D11_SCHEMA]])
_ip = "pgwatch_pg_index_pilot_config_config_value_normalized"
_pr = []
for prio, (s_, i_, d_) in zip(range(8, 0, -1), [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 0)]):
    sel = ", ".join([f'config_key="index_rebuild_scale_factor"',
                     'schemaname=~"$schema_re"' if s_ else 'schemaname="*"',
                     'indexrelname=~"$index_re"' if i_ else 'indexrelname="*"',
                     'datname=~"$datname_re"' if d_ else 'datname="*"'])
    _pr.append((prio, f"last_over_time({_ip}{{{sel}}}[1w])"))
dash("D11", "1C_scale_factor", "index_pilot_scale.sql",
     "max by () (" + " or ".join(f"{x} * {p}" for p, x in _pr) + ") / max by () ("
     + " or ".join(f"{p} * ({x} * 0 + 1)" for p, x in _pr) + ")",
     schema_re=D11_SCHEMA, index_re=D11_INDEX, datname_re="target_database")

# --- Dashboard_6_Replication_and_HA.json: the $__range targets that need no subquery ---
_rep = _CNmv + [["application_name", "=~", ".*"]]
_prep = '{cluster="$cluster", node_name="$node", application_name=~".*"}'
_cond = f"count(count by (application_name) (max_over_time(pgwatch_replication_replay_lag_b{_prep}[${{range_s}}s]))) == 1"
dash("D6", "221A_in_flight", "replica_stage.sql",
     f"(clamp_min(max_over_time(pgwatch_replication_sent_lag_b{_prep}{_RI}), 0)) and on() ({_cond})",
     metric="pgwatch_replication_sent_lag_b", matchers=_rep, divisor=1, clamp0=True,
     cond_metric="pgwatch_replication_replay_lag_b", cond_by="application_name")
for ref, m in (("222A", "write_lag_ms"), ("222B", "flush_lag_ms"), ("222C", "replay_lag_ms")):
    dash("D6", f"{ref}_{m}", "replica_stage.sql",
         f"(max_over_time(pgwatch_replication_{m}{_prep}{_RI}) / 1000) and on() ({_cond})",
         metric=f"pgwatch_replication_{m}", matchers=_rep, divisor=1000, clamp0=False,
         cond_metric="pgwatch_replication_replay_lag_b", cond_by="application_name")
dash("D6", "243B_max_last_range", "max_last_range.sql",
     'max(last_over_time(pgwatch_settings_numeric_value{cluster="$cluster", node_name="$node", setting_name="wal_sender_timeout", '
     'unit="ms"}[${range_s}s])) / 1000 > 0',
     metric="pgwatch_settings_numeric_value", divisor=1000,
     matchers=_CNmv + [["setting_name", "=", "wal_sender_timeout"], ["unit", "=", "ms"]])
dash("D6", "251A_instant_range", "last_over_range.sql",
     f"last_over_time(pgwatch_replication_replay_lag_b{_cn5}[${{range_s}}s])",
     metric="pgwatch_replication_replay_lag_b", matchers=_CNmv, kind="instant")
_w6 = "wal_level|max_wal_senders|max_replication_slots|wal_keep_size|max_slot_wal_keep_size|wal_sender_timeout|synchronous_commit|synchronous_standby_names"
dash("D6", "252A_instant_range", "last_over_range.sql",
     f'last_over_time(pgwatch_settings_configured{{cluster="$cluster", node_name="$node", setting_name=~"{_w6}"}}[${{range_s}}s])',
     metric="pgwatch_settings_configured", matchers=_CNmv + [["setting_name", "=~", _w6]], kind="instant")

# --- Dashboard_6, the rest: aggregations that fit existing shapes ---
for ref, op, m in (("102A", "count", "replay_lag_b"), ("105A", "sum", "is_sync_int"),
                   ("106A", "sum", "backend_xmin_reported_int")):
    dash("D6", f"{ref}_{op}", "agg_last.sql", f"{op}(pgwatch_replication_{m}{_cn5})",
         op=op, metric=f"pgwatch_replication_{m}", matchers=_CNmv, by=[])
dash("D6", "107A_max", "agg_last.sql", f"max(pgwatch_wal_timeline{_cn5})",
     op="max", metric="pgwatch_wal_timeline", matchers=_CNmv, by=[])
dash("D6", "104A_max_max", "agg_fn.sql", f"max(max_over_time(pgwatch_replication_replay_lag_b{_cn5}{_RI}))",
     op="max", fn="max_over_time", metric="pgwatch_replication_replay_lag_b", matchers=_CNmv, by=[])
dash("D6", "103A_max_max_s", "agg_fn_div.sql", f"max(max_over_time(pgwatch_replication_replay_lag_ms{_cn5}{_RI})) / 1000",
     op="max", fn="max_over_time", metric="pgwatch_replication_replay_lag_ms", matchers=_CNmv, by=[], divisor=1000)
dash("D6", "211A_max_by_app", "agg_fn.sql",
     f"max by (application_name) (max_over_time(pgwatch_replication_replay_lag_b{_cn5}{_RI}))",
     op="max", fn="max_over_time", metric="pgwatch_replication_replay_lag_b", matchers=_CNmv, by=["application_name"])
dash("D6", "212A_max_by_app_s", "agg_fn_div.sql",
     f"max by (application_name) (max_over_time(pgwatch_replication_replay_lag_ms{_cn5}{_RI})) / 1000",
     op="max", fn="max_over_time", metric="pgwatch_replication_replay_lag_ms", matchers=_CNmv, by=["application_name"],
     divisor=1000)
dash("D6", "231A_sum_irate", "agg_fn.sql", f"sum(irate(pgwatch_wal_xlog_location_b{_cn5}{_RI}))",
     op="sum", fn="irate", metric="pgwatch_wal_xlog_location_b", matchers=_CNmv, by=[])
for ref, m in (("241A", "backend_xmin_reported_int"), ("242A", "backend_xmin_age_tx")):
    dash("D6", f"{ref}_max_by_app", "agg_fn.sql",
         f"max by (application_name) (max_over_time(pgwatch_replication_{m}{_prep}{_RI}))",
         op="max", fn="max_over_time", metric=f"pgwatch_replication_{m}", matchers=_rep, by=["application_name"])
dash("D6", "243A_max_by_app_s", "agg_fn_div.sql",
     f"max by (application_name) (max_over_time(pgwatch_replication_reply_time_lag_ms{_prep}{_RI})) / 1000",
     op="max", fn="max_over_time", metric="pgwatch_replication_reply_time_lag_ms", matchers=_rep, by=["application_name"],
     divisor=1000)

# --- Dashboard_6: rollups over expressions and the [1h:1m] subquery, on the subquery grid ---
for ref, a_, b_ in (("221B", "write_lag_b", "sent_lag_b"), ("221C", "flush_lag_b", "write_lag_b"),
                    ("221D", "replay_lag_b", "flush_lag_b")):
    dash("D6", f"{ref}_sq_diff_max", "sq_binop_max.sql",
         f"(clamp_min(max_over_time((pgwatch_replication_{a_}{_prep} - pgwatch_replication_{b_}{_prep}){_RI}), 0))"
         f" and on() ({_cond})",
         metric=f"pgwatch_replication_{a_}", other=f"pgwatch_replication_{b_}", matchers=_rep,
         cond_metric="pgwatch_replication_replay_lag_b", cond_by="application_name")
dash("D6", "231B_sq_max_max_s", "sq_agg_max.sql",
     f"max_over_time((max(pgwatch_replication_replay_lag_ms{_cn5})){_RI}) / 1000",
     metric="pgwatch_replication_replay_lag_ms", matchers=_CNmv, divisor=1000)
dash("D6", "233A_sq_minutes_above", "sq_threshold_sum.sql",
     f"sum_over_time(((max by (application_name) (pgwatch_replication_replay_lag_ms{_cn5})) > bool ($lag_alert_s * 1000))[1h:1m])",
     metric="pgwatch_replication_replay_lag_ms", matchers=_CNmv, lag_alert_s=1, sq_window_ms=3600000, sq_step_ms=60000)

# --- Dashboard_8_Table_Stats.json ($schema_name, $top_n): topk as a per-step rank ---
D8_SCHEMA = "app|public"
_t8 = _CNDq + [["schemaname", "=~", D8_SCHEMA]]
_p8 = '{schemaname=~"' + D8_SCHEMA + '", datname=~"$db_re", cluster="$cluster", node_name="$node"}'
_t8s = _CNDq + [["schema", "=~", D8_SCHEMA]]
_p8s = '{schema=~"' + D8_SCHEMA + '", datname=~"$db_re", cluster="$cluster", node_name="$node"}'
_t8k = _t8 + [RELKIND_TABLE]
_p8k = '{relkind=~"114|r", schemaname=~"' + D8_SCHEMA + '", datname=~"$db_re", cluster="$cluster", node_name="$node"}'
for ref, m, ms, ps in (("1A", "pg_class_total_relation_size_bytes", _t8k, _p8k),
                       ("23A", "table_size_detailed_table_main_size_b", _t8s, _p8s),
                       ("28A", "table_size_detailed_toast_indexes_size_b", _t8s, _p8s),
                       ("25A", "table_size_detailed_table_indexes_size_b", _t8s, _p8s),
                       ("8A", "pg_table_bloat_bloat_pct", _t8, _p8), ("2A", "pg_table_bloat_bloat_size", _t8, _p8)):
    dash("D8", f"{ref}_topk_last", "topk_last.sql", f"topk($top_n, pgwatch_{m}{ps})",
         metric=f"pgwatch_{m}", matchers=ms, top_n=10, mode="topk")
for ref, m, ms, ps in (("7A", "pg_class_total_relation_size_bytes", _t8k, _p8k),
                       ("24A", "table_size_detailed_table_main_size_b", _t8s, _p8s),
                       ("29A", "table_size_detailed_total_toast_size_b", _t8s, _p8s),
                       ("26A", "table_size_detailed_table_indexes_size_b", _t8s, _p8s),
                       ("40A", "pg_stat_all_tables_seq_tup_read", _t8, _p8),
                       ("41A", "pg_stat_all_tables_idx_tup_fetch", _t8, _p8),
                       ("32A", "pg_statio_all_tables_heap_blks_read", _t8, _p8),
                       ("39A", "pg_statio_all_tables_toast_blks_read", _t8, _p8)):
    dash("D8", f"{ref}_topk_irate", "topk_fn.sql", f"topk($top_n, irate(pgwatch_{m}{ps}{_RI}))",
         fn="irate", metric=f"pgwatch_{m}", matchers=ms, by=[], top_n=10, mode="topk")
_by8 = ["schema", "table_name", "table_full_name"]
for ref, m in (("3A", "n_tup_ins"), ("5A", "n_tup_del"), ("6A", "n_tup_hot_upd")):
    dash("D8", f"{ref}_topk_sum_by_irate", "topk_fn.sql",
         f"topk($top_n, sum by (schema, table_name, table_full_name) (irate(pgwatch_table_stats_{m}{_p8s}{_RI})))",
         fn="irate", metric=f"pgwatch_table_stats_{m}", matchers=_t8s, by=_by8, top_n=10, mode="topk")
dash("D8", "10A_topk_sum_by_irate_diff", "topk_irate_diff_by.sql",
     f"topk($top_n, sum by (schema, table_name, table_full_name) (irate(pgwatch_table_stats_n_tup_upd{_p8s}{_RI})"
     f" - irate(pgwatch_table_stats_n_tup_hot_upd{_p8s}{_RI})))",
     metric="pgwatch_table_stats_n_tup_upd", other="pgwatch_table_stats_n_tup_hot_upd", matchers=_t8s, by=_by8,
     top_n=10, mode="topk")
_bs8 = _CNDq + [["setting_name", "=", "block_size"]]
_pbs8 = ('last_over_time(pgwatch_settings_numeric_value{datname=~"$db_re", cluster="$cluster", node_name="$node", '
         'setting_name="block_size"}[1d])')


def _st8(kind):
    return " + ".join(f"irate(pgwatch_pg_statio_all_tables_{x}_blks_{kind}{_p8}{_RI})" for x in ("heap", "idx", "toast", "tidx"))


for ref, num in (("31A", "hit"), ("37A", "read")):
    dash("D8", f"{ref}_topk_statio_sum_x_block_size", "topk_statio_sum_x_scalar.sql",
         f"topk($top_n, {_st8(num)}) * on() group_left() {_pbs8}",
         num=num, matchers=_t8, smetric="pgwatch_settings_numeric_value", smatchers=_bs8, swindow_ms=86400000,
         top_n=10, mode="topk")
for ref, num in (("42A", "hit"), ("43A", "read")):
    dash("D8", f"{ref}_topk_statio_ratio", "topk_statio_ratio.sql",
         f"topk($top_n, ({_st8(num)}) / ({_st8('read')} + ({_st8('hit')})))",
         num=num, matchers=_t8, top_n=10, mode="topk")
for ref, m in (("36A", "heap_blks_hit"), ("38A", "toast_blks_hit"), ("33A", "idx_blks_hit"), ("34A", "idx_blks_read")):
    dash("D8", f"{ref}_topk_irate_x_block_size", "topk_irate_x_scalar.sql",
         f"topk($top_n, irate(pgwatch_pg_statio_all_tables_{m}{_p8}{_RI})) * on() group_left() {_pbs8}",
         metric=f"pgwatch_pg_statio_all_tables_{m}", matchers=_t8, smetric="pgwatch_settings_numeric_value",
         smatchers=_bs8, swindow_ms=86400000, top_n=10, mode="topk")

# --- Dashboard_10_Index health.json ($top_n) ---
_p10 = '{cluster="$cluster", node_name="$node", datname=~"$db_re"}'
dash("D10", "23A_topk_last", "topk_last.sql",
     '''topk($top_n, pgwatch_pg_class_relation_size_bytes{relkind=~"105|i", cluster="$cluster", node_name="$node", datname=~"$db_re"})''',
     metric="pgwatch_pg_class_relation_size_bytes", matchers=_CNDq + [RELKIND_INDEX], top_n=10, mode="topk")
for ref, m in (("2A", "bloat_pct"), ("22A", "bloat_size")):
    dash("D10", f"{ref}_topk_last", "topk_last.sql", f"topk($top_n, pgwatch_pg_btree_bloat_{m}{_p10})",
         metric=f"pgwatch_pg_btree_bloat_{m}", matchers=_CNDq, top_n=10, mode="topk")
for ref, m in (("27A", "idx_tup_read"), ("28A", "idx_tup_fetch")):
    dash("D10", f"{ref}_topk_irate", "topk_fn.sql", f"topk($top_n, irate(pgwatch_pg_stat_all_indexes_{m}{_p10}{_RI}))",
         fn="irate", metric=f"pgwatch_pg_stat_all_indexes_{m}", matchers=_CNDq, by=[], top_n=10, mode="topk")
for ref, m in (("25A", "idx_blks_read"), ("26A", "idx_blks_hit")):
    dash("D10", f"{ref}_topk_irate_x_block_size", "topk_irate_x_scalar.sql",
         f"topk($top_n, irate(pgwatch_pg_statio_all_indexes_{m}{_p10}{_RI})) * on() group_left() "
         'last_over_time(pgwatch_settings_numeric_value{cluster="$cluster", node_name="$node", datname=~"$db_re", '
         'setting_name="block_size"}[1d])',
         metric=f"pgwatch_pg_statio_all_indexes_{m}", matchers=_CNDq, smetric="pgwatch_settings_numeric_value",
         smatchers=_bs8, swindow_ms=86400000, top_n=10, mode="topk")
