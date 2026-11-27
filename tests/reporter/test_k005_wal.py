"""K005: the statements' window facts and the server's WAL facts (reporter/k005.py)."""
from __future__ import annotations

import re
from typing import Any, List

import pytest

from reporter import k005
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report
from tests.reporter.test_g003 import Conn

H = 3600
TIMELINE = [H * h for h in range(470_000, 470_003)]  # three hours, each stamp the end of its hour
PG = {"version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"}
SEG = 16 << 20


def _readings(values: List[float], start: int = TIMELINE[0] - H, step: int = k005.STEP_S) -> List[List[Any]]:
    return [[start + i * step, str(v)] for i, v in enumerate(values)]


# ── facts ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_peak_span_is_the_most_wal_in_fifteen_minutes() -> None:
    values = [i * SEG for i in range(37)]
    values = values[:20] + [v + 10 * SEG for v in values[20:]]  # +10 segments between readings 19 and 20
    out = k005.peak_span([{"metric": {}, "values": _readings(values)}])
    assert out["bytes"] == 13 * SEG and out["seconds"] == 900
    assert out["end"] == "2023-08-14T08:40:00+00:00"
    assert k005.peak_span([{"metric": {}, "values": _readings([0, 1])}]) is None  # never 15 minutes apart


@pytest.mark.unit
def test_peak_span_skips_a_span_with_a_reset() -> None:
    values = [0, SEG, 2 * SEG, 5, SEG + 5, 2 * SEG + 5, 3 * SEG + 5]  # the counter went back after the third
    out = k005.peak_span([{"metric": {}, "values": _readings(values)}])
    assert out["bytes"] == 3 * SEG  # only the span after the fall; never one across it


@pytest.mark.unit
def test_segment_size_from_the_archiver_label() -> None:
    assert k005.segment_bytes([{"metric": {"wal_segment_size": "16 MB"}}]) == SEG
    assert k005.segment_bytes([{"metric": {"wal_segment_size": "64MB"}}]) == 64 << 20
    assert k005.segment_bytes([{"metric": {}}]) is None


@pytest.mark.unit
def test_wait_share_counts_only_the_wal_write_events() -> None:
    samples = [{"metric": {"wait_event": "WalSync"}, "values": [[1, "3"], [2, "1"]]},
               {"metric": {"wait_event": "WALWrite"}, "values": [[1, "2"]]},
               {"metric": {"wait_event": "CPU*"}, "values": [[1, "94"]]}]
    out = k005.wait_share(samples)
    assert out == {"wal_samples": 6.0, "all_samples": 100.0, "wal_events": {"WalSync": 4.0, "WALWrite": 2.0}}
    assert k005.wait_share([]) is None  # no samples: not known, never 0%


@pytest.mark.unit
def test_gauge_ends_first_last_and_highest() -> None:
    out = k005.gauge_ends([{"metric": {}, "values": _readings([300, 500, 400])}])
    assert (out["first"], out["last"], out["max"]) == (300.0, 400.0, 500.0)
    assert k005.gauge_ends([]) is None


INDEXES = [{"schema": "app", "table": "events", "index": "events_pkey", "bytes": 8192, "is_unique": True,
            "is_primary": True, "predicate": None, "scans": 10},
           {"schema": "app", "table": "events", "index": "events_open_idx", "bytes": 4096, "is_unique": False,
            "is_primary": False, "predicate": "(status <> 0)", "scans": 0},
           {"schema": "other", "table": "events", "index": "x", "bytes": 1, "is_unique": False,
            "is_primary": False, "predicate": None, "scans": 1}]
ANSWERS = {
    "select current_database()": [{"db": "app"}],
    "from pg_settings": [{"name": "checkpoint_timeout", "setting": "300", "unit": "s", "category": "c",
                          "context": "sighup", "vartype": "integer", "source": "default", "boot_val": "300",
                          "pending_restart": False}],
    "select slot_name": [{"slot_name": "sub", "slot_type": "logical", "active": False, "wal_status": "extended",
                          "retained_bytes": 3 * SEG}],
    "from pg_index x": INDEXES,
    "pg_inherits": [{"schema": "app", "table": "events", "plain": True}],
}


@pytest.mark.unit
def test_node_facts_reads_settings_slots_and_the_top_tables_indexes() -> None:
    tables = [{"database": "app", "schema": "app", "table": "events"},
              {"database": "other_db", "schema": "x", "table": "y"}]
    out = k005.node_facts("host=h dbname=app", tables, opener=lambda c: Conn(ANSWERS))
    assert out["settings"]["checkpoint_timeout"] == {"setting": "300", "unit": "s", "context": "sighup"}
    assert out["slots"] == [{"slot_name": "sub", "slot_type": "logical", "active": False, "wal_status": "extended",
                             "retained_bytes": float(3 * SEG)}]
    # only this database's tables, only the asked pairs (other.events is not app.events)
    assert [i["name"] for i in out["indexes"]["app.events"]] == ["events_open_idx", "events_pkey"] or \
        [i["name"] for i in out["indexes"]["app.events"]] == ["events_pkey", "events_open_idx"]
    assert "other.events" not in out["indexes"]
    partial = next(i for i in out["indexes"]["app.events"] if i["name"] == "events_open_idx")
    assert partial["predicate"] == "(status <> 0)" and partial["scans"] == 0.0


@pytest.mark.unit
def test_node_facts_keeps_the_other_facts_when_one_read_fails() -> None:
    out = k005.node_facts("postgresql://mon:s3cr3t@db/app", [],
                          opener=lambda c: Conn(ANSWERS, broken=("select slot_name",)))
    assert "settings" in out and "slots" not in out and "s3cr3t" not in out["slots_error"]
    assert k005.node_facts(None) == {"node_error": "no connection for this node in the sources file"}


# ── the generator ─────────────────────────────────────────────────────────────

WINDOW = {
    "pgwatch_pg_stat_statements_calls": {"111": [60, 60, 60], "-222": [100, 100, 100]},
    "pgwatch_pg_stat_statements_wal_fpi": {"111": [10, 10, 10]},
    "pgwatch_pg_stat_statements_wal_records": {"111": [100, 100, 100]},
}
TEXTS = {"app": {"111": "update app.events set status = $1 where id = $2",
                 "-222": "select /* pgwatch_generated */ 1 from pg_class"}}


@pytest.fixture(name="generator")
def fixture_generator(monkeypatch: pytest.MonkeyPatch) -> PostgresReportGenerator:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["app"])

    def hourly(cluster, node_name, db_name, metric_name="", hours=24, step_s=3600, k=3):
        return ({"111": [SEG, SEG, SEG], "-222": [1000, 0, 1000], "333": [0.0, 0.0, 0.0]}, [0.0, 0.0, 0.0],
                TIMELINE)

    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", hourly)
    monkeypatch.setattr(gen, "get_queryid_queries_from_sink", lambda limit=655360, db_names=None, cluster=None, node_name=None: TEXTS)

    def query_instant(query):
        if query.startswith("topk("):
            return {"status": "success", "data": {"result": [
                {"metric": {"datname": "app", "schema": "app", "table_name": "events"}, "value": [0, "36"]}]}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", query_instant)

    def query_range(query, start, end, step="30s"):
        five = [i for i in range(37)]
        if query.startswith("pgwatch_wal_xlog_location_b"):
            return [{"metric": {}, "values": _readings([i * SEG for i in five])}]
        if query.startswith("max by (datname) (pgwatch_db_size_size_b"):
            return [{"metric": {"datname": "app"}, "values": _readings([100 << 20, 101 << 20])}]
        if query.startswith("pgwatch_checkpointer_num_timed"):
            return [{"metric": {}, "values": _readings(five)}]
        if query.startswith("pgwatch_checkpointer_num_requested"):
            return [{"metric": {}, "values": _readings([0] * 37)}]
        if query.startswith("pgwatch_archive_lag_archived_count"):
            return [{"metric": {"wal_segment_size": "16 MB"}, "values": _readings(five)}]
        if query.startswith("pgwatch_archive_lag_failed_count"):
            return [{"metric": {}, "values": _readings([0] * 37)}]
        if query.startswith("pgwatch_archive_lag_seconds_since_archive"):
            return [{"metric": {}, "values": _readings([30, 117, 60])}]
        if query.startswith("max(pgwatch_pg_wal_size_bytes"):
            return [{"metric": {}, "values": _readings([128 << 20])}]
        if query.startswith("sum by (wait_event)"):
            assert 'wait_event_type!~"Activity|Client|Timeout"' in query
            return [{"metric": {"wait_event": "CPU*"}, "values": [[TIMELINE[0], "50"]]}]
        if query.startswith("pgwatch_table_stats_n_tup_upd"):
            return [{"metric": {"table_size_cardinality_mb": "1"}, "values": _readings([0, 12, 24])},
                    {"metric": {"table_size_cardinality_mb": "10"}, "values": _readings([36], start=TIMELINE[0] - H + 3 * k005.STEP_S)}]
        if query.startswith("pgwatch_table_stats_n_tup_hot_upd"):
            return [{"metric": {}, "values": _readings([0, 0, 1, 1])}]
        if query.startswith("pgwatch_table_stats_"):
            return []
        if query.startswith("max(pgwatch_table_stats_total_relation_size_b"):
            return [{"metric": {}, "values": _readings([9 << 20])}]
        assert step == "3600s", query
        if query.startswith("count(count by (queryid)"):
            return [{"metric": {}, "values": [[ts, "40"] for ts in TIMELINE]}]
        if query.startswith("count(resets("):
            return [{"metric": {}, "values": [[ts, "0"] for ts in TIMELINE]}]
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_calls"):
            return [{"metric": {}, "values": [[ts, "1000"] for ts in TIMELINE]}]
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_wal_fpi"):
            return [{"metric": {}, "values": [[ts, "40"] for ts in TIMELINE]}]
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_wal_records"):
            return [{"metric": {}, "values": [[ts, "400"] for ts in TIMELINE]}]
        metric = re.search(r"increase\((pgwatch_[a-z_]+)\{", query).group(1)
        assert "333" not in query  # a statement with no WAL gets no window facts
        return [{"metric": {"queryid": q}, "values": [[ts, str(v)] for ts, v in zip(TIMELINE, vals)]}
                for q, vals in WINDOW.get(metric, {}).items()]

    monkeypatch.setattr(gen, "query_range", query_range)
    monkeypatch.setattr("reporter.postgres_reports.read_sources", lambda path: [])
    return gen


@pytest.mark.unit
def test_k005_rows_carry_window_facts_and_the_server_facts(generator) -> None:
    report = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)
    validate_report(report)
    node = report["results"]["node-1"]
    app = node["data"]["app"]
    rows = {q["queryid"]: q for q in app["top_queries"]}
    assert list(rows) == ["111", "-222"]  # "333" wrote no WAL: not a top statement
    assert rows["111"]["calls"] == 180 and rows["111"]["wal_fpi"] == 30 and rows["111"]["wal_records"] == 300
    assert rows["-222"]["is_monitoring"] is True and "wal_fpi" not in rows["-222"]  # no series: not known, never 0
    s = app["summary"]
    assert s["statements_wal_fpi"] == 120 and s["statements_wal_records"] == 1200
    assert node["wal"]["total"] == 36 * SEG and node["wal"]["hourly"] == [12 * SEG] * 3
    assert node["wal"]["peak_15m"]["bytes"] == 3 * SEG
    assert node["database_sizes"]["app"]["first"] == 100 << 20 and node["database_sizes"]["app"]["last"] == 101 << 20
    assert node["checkpoints"]["timed"]["total"] == 36 and node["checkpoints"]["requested"]["total"] == 0
    assert node["archiver"]["segment_bytes"] == SEG and node["archiver"]["archived"]["total"] == 36
    assert node["archiver"]["failed"]["total"] == 0 and node["archiver"]["longest_gap_s"] == 117
    assert node["pg_wal"]["max"] == 128 << 20
    assert node["wal_waits"] == {"wal_samples": 0.0, "all_samples": 50.0, "wal_events": {}}
    # the updates counter merged over the series a size-label change started: 36, not 36 + 36
    assert node["tables"] == [{"database": "app", "schema": "app", "table": "events", "updates": 36.0,
                               "hot_updates": 1.0, "total_bytes": float(9 << 20)}]
    assert node["node_error"] == "no connection for this node in the sources file"
    assert node["settings_error"] == "monitoring has no reading of these settings"


@pytest.mark.unit
def test_k005_a_failed_metric_is_said_never_zero(generator, monkeypatch: pytest.MonkeyPatch) -> None:
    real = generator.query_range

    def no_checkpoints(query, start, end, step="30s"):
        if "checkpoint" in query:
            return []
        return real(query, start, end, step)

    monkeypatch.setattr(generator, "query_range", no_checkpoints)
    node = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)["results"]["node-1"]
    assert "checkpoints" not in node
    assert "fewer than two readings" in node["checkpoints_timed_error"]


@pytest.mark.unit
def test_wait_share_counts_the_pre_17_event_names() -> None:
    """PostgreSQL 13-16 name the IO events WALSync and WALWrite; 12 and older the LWLock WALWriteLock (REV r1)."""
    samples = [{"metric": {"wait_event": "WALSync"}, "values": [[1, "4"]]},
               {"metric": {"wait_event": "WALWriteLock"}, "values": [[1, "1"]]},
               {"metric": {"wait_event": "CPU*"}, "values": [[1, "95"]]}]
    assert k005.wait_share(samples)["wal_samples"] == 5.0


@pytest.mark.unit
def test_node_facts_matches_quoted_names_and_says_a_table_has_no_index() -> None:
    """pgwatch labels a schema and a table with quote_ident(): a mixed-case table matches; a table read with no index
    is an empty list (REV r1)."""
    answers = dict(ANSWERS, **{"pg_inherits": [{"schema": "app", "table": '"Orders"', "plain": True},
                                               {"schema": "app", "table": "bare", "plain": True}],
                               "from pg_index x": [
        {"schema": "app", "table": '"Orders"', "index": "orders_pkey", "bytes": 8192, "is_unique": True,
         "is_primary": True, "predicate": None, "scans": 3}]})
    tables = [{"database": "app", "schema": "app", "table": '"Orders"'}, {"database": "app", "schema": "app", "table": "bare"}]
    conn = Conn(answers)
    out = k005.node_facts("host=h dbname=app", tables, opener=lambda c: conn)
    assert [i["name"] for i in out["indexes"]['app."Orders"']] == ["orders_pkey"]
    assert out["indexes"]["app.bare"] == []
    sql = next(q for q in conn.executed if "pg_index" in q)
    assert "quote_ident(n.nspname) = any(" in sql and "quote_ident(c.relname) = any(" in sql


@pytest.mark.unit
def test_k005_excludes_the_other_bucket_ranks_at_the_window_end_and_counts_waits_over_the_window(generator,
                                                                                                    monkeypatch) -> None:
    asked = {}
    real_instant, real_range = generator.query_instant, generator.query_range

    def instant(query):
        asked.setdefault("instant", []).append(query)
        return real_instant(query)

    def ranged(query, start, end, step="30s"):
        if query.startswith("sum by (wait_event)"):
            asked["waits_start"] = start.timestamp()
        return real_range(query, start, end, step)

    monkeypatch.setattr(generator, "query_instant", instant)
    monkeypatch.setattr(generator, "query_range", ranged)
    generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)
    topk = next(q for q in asked["instant"] if q.startswith("topk("))
    assert 'schema!="$other$"' in topk and f"@ {TIMELINE[-1]}" in topk
    assert asked["waits_start"] == TIMELINE[0]  # the first point sums the window's first hour, not the hour before


@pytest.mark.unit
def test_k005_checkpoints_from_pg_stat_bgwriter_before_17(generator, monkeypatch) -> None:
    real = generator.query_range

    def pre17(query, start, end, step="30s"):
        if query.startswith("pgwatch_checkpointer_"):
            return []
        if query.startswith("pgwatch_bgwriter_checkpoints_timed"):
            return [{"metric": {}, "values": _readings(list(range(37)))}]
        if query.startswith("pgwatch_bgwriter_checkpoints_req"):
            return [{"metric": {}, "values": _readings([0] * 36 + [2])}]
        return real(query, start, end, step)

    monkeypatch.setattr(generator, "query_range", pre17)
    node = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)["results"]["node-1"]
    assert node["checkpoints"]["timed"]["total"] == 36 and node["checkpoints"]["requested"]["total"] == 2
    assert "checkpoints_timed_error" not in node


@pytest.mark.unit
def test_k005_an_empty_reply_is_said_never_left_out(generator, monkeypatch) -> None:
    real = generator.query_range

    def empty(query, start, end, step="30s"):
        if query.startswith(("max(pgwatch_pg_wal_size_bytes", "sum by (wait_event)", "pgwatch_archive_lag_failed",
                             "pgwatch_archive_lag_seconds")):
            return []
        return real(query, start, end, step)

    monkeypatch.setattr(generator, "query_range", empty)
    monkeypatch.setattr(generator, "query_instant", lambda q: {"status": "success", "data": {"result": []}})
    node = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)["results"]["node-1"]
    for key in ("pg_wal_error", "wal_waits_error", "archiver_failed_error", "archiver_gap_error", "tables_error"):
        assert key in node, key
    assert "pg_wal" not in node and "wal_waits" not in node and "tables" not in node


@pytest.mark.unit
def test_node_facts_leaves_out_a_partition_root_and_a_dropped_table_with_a_reason() -> None:
    """A partition root keeps its indexes on the partitions and a table dropped since the window is not found: neither
    reads as "no index" (REV r2)."""
    answers = dict(ANSWERS, **{"pg_inherits": [{"schema": "app", "table": "events", "plain": True},
                                               {"schema": "app", "table": "measurements", "plain": False}]})
    tables = [{"database": "app", "schema": "app", "table": "events"},
              {"database": "app", "schema": "app", "table": "measurements"},
              {"database": "app", "schema": "app", "table": "gone"}]
    out = k005.node_facts("host=h dbname=app", tables, opener=lambda c: Conn(answers))
    assert set(out["indexes"]) == {"app.events"} and "partition root" in out["indexes_error"]


@pytest.mark.unit
def test_k005_rows_take_the_texts_of_their_own_node(generator, monkeypatch: pytest.MonkeyPatch) -> None:
    """v4 reads statement texts per cluster and node (#406): K005 asks for its own node's, not those of
    the last node K003 read (generate_all_reports runs K003 for every node first)."""
    asked = []

    def texts(limit=655360, db_names=None, cluster=None, node_name=None):
        asked.append((cluster, node_name))
        return TEXTS if node_name == "node-1" else {}

    monkeypatch.setattr(generator, "get_queryid_queries_from_sink", texts)
    generator._k003_text_scope = ("local", "standby-1")  # what K003 of the unit's last node left before the fix
    report = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=180)
    rows = {q["queryid"]: q for q in report["results"]["node-1"]["data"]["app"]["top_queries"]}
    assert rows["111"]["query_text"].startswith("update app.events") and rows["-222"]["is_monitoring"] is True
    assert set(asked) == {("local", "node-1")}
