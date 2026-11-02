"""F001: worker history and per-table autovacuum facts in the box reporter's F001 file.

CheckupLoop's F001 report locks them as [fixed] blocks (protocols
checkup-acceptance-criteria/F001: §1 assessed scope, F001-O2 each table
against its own trigger, F001-O3 workers mean and peak with the sample count).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

GiB = 1024 ** 3


def _rows(values: List[tuple], schema_label: str = "schemaname", table_label: str = "relname") -> List[Dict]:
    return [{"metric": {"datname": "app", schema_label: s, table_label: t}, "value": [0, str(v)]}
            for s, t, v in values]


def _gen(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, Any]) -> PostgresReportGenerator:
    gen = PostgresReportGenerator("http://prom", use_current_time=True)

    def fake(query: str) -> Dict[str, Any]:
        for needle, rows in responses.items():
            if needle in query:
                if rows is None:
                    return {"status": "error"}
                return {"status": "success", "data": {"result": rows}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", fake)
    return gen


TABLE_STATS = ("schema", "table_name")
RESPONSES: Dict[str, Any] = {
    "avg_over_time(pgwatch_pg_autovacuum_workers_active_workers": [{"metric": {}, "value": [0, "0.0104"]}],
    "max_over_time(pgwatch_pg_autovacuum_workers_active_workers": [{"metric": {}, "value": [0, "2"]}],
    "count_over_time(pgwatch_pg_autovacuum_workers_active_workers": [{"metric": {}, "value": [0, "20159"]}],
    "max_over_time(pgwatch_pg_autovacuum_workers_max_workers": [{"metric": {}, "value": [0, "3"]}],
    "pgwatch_pg_autovacuum_queue_n_dead_tup": _rows([("public", "small", 5000), ("public", "big", 0),
                                                     ("pg_catalog", "pg_class", 10), ("public", "clean", 0)]),
    "pgwatch_pg_autovacuum_queue_autovacuum_threshold": _rows([("public", "small", 1000), ("public", "big", 2_000_050),
                                                               ("pg_catalog", "pg_class", 50), ("public", "clean", 60)]),
    "pgwatch_pg_autovacuum_queue_n_live_tup": _rows([("public", "small", 20000), ("public", "big", 10_000_000)]),
    "pgwatch_table_stats_total_relation_size_b": _rows([("public", "small", 10_000_000), ("public", "big", 2 * GiB),
                                                        ("pg_catalog", "pg_class", 3 * GiB)], *TABLE_STATS),
    "pgwatch_table_stats_autovacuum_count": _rows([("public", "small", 0), ("public", "big", 4)], *TABLE_STATS),
    "pgwatch_table_stats_seconds_since_last_vacuum": _rows([("public", "big", 7200)], *TABLE_STATS),
    "pgwatch_table_stats_no_autovacuum": _rows([("public", "small", 1), ("public", "big", 0)], *TABLE_STATS),
}


@pytest.mark.unit
def test_f001_carries_workers_over_the_window_with_the_sample_count(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    assert report["results"]["node-1"]["autovacuum_workers"] == {
        "window_seconds": 604800, "mean_active": 0.0104, "peak_active": 2.0, "samples": 20159, "max_workers": 3.0}


@pytest.mark.unit
def test_f001_lists_tables_in_scope_and_tables_with_dead_tuples_most_overdue_first(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Tables of 1 GiB or more, and smaller ones with dead tuples; system
    catalogs (F006) and clean small tables are left out."""
    report = _gen(monkeypatch, RESPONSES).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    tables = report["results"]["node-1"]["autovacuum_tables"]
    assert [(t["schema_name"], t["table_name"]) for t in tables] == [("public", "small"), ("public", "big")]
    small, big = tables
    assert small == {
        "database": "app", "schema_name": "public", "table_name": "small", "total_relation_size_bytes": 10_000_000,
        "in_scope_by_size": False, "n_live_tup": 20000, "n_dead_tup": 5000,
        "n_dead_tup_24h_ago": 5000,  # the stub answers the offset query the same
        "vacuum_threshold": 1000,
        "autovacuum_runs_in_window": 0, "seconds_since_last_vacuum": None, "vacuum_recorded": False,
        "autovacuum_disabled": True}
    assert big["in_scope_by_size"] is True and big["seconds_since_last_vacuum"] == 7200
    assert big["autovacuum_runs_in_window"] == 4 and big["autovacuum_disabled"] is False


@pytest.mark.unit
def test_f001_unreadable_queue_metrics_leave_the_tables_out_never_an_empty_list(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown is not "no tables": without the queue metrics the key is absent."""
    responses = dict(RESPONSES, **{"pgwatch_pg_autovacuum_queue_n_dead_tup": None})
    report = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    assert "autovacuum_tables" not in report["results"]["node-1"]


@pytest.mark.unit
def test_f001_reads_the_source_settings_the_settings_table_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = [{"metric": {"setting_name": name, "setting_value": value, "unit": unit, "category": "c",
                            "context": "sighup", "vartype": "integer"}, "value": [0, "1"]}
                for name, value, unit in (("maintenance_work_mem", "65536", "kB"),
                                          ("log_autovacuum_min_duration", "1000", "ms"),
                                          ("autovacuum_vacuum_insert_threshold", "1000", ""),
                                          ("shared_buffers", "16384", "8kB"))]
    report = _gen(monkeypatch, dict(RESPONSES, pgwatch_settings_configured=settings)) \
        .generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    data = report["results"]["node-1"]["data"]
    assert {"maintenance_work_mem", "log_autovacuum_min_duration", "autovacuum_vacuum_insert_threshold"} <= set(data)
    assert "shared_buffers" not in data  # not an autovacuum setting


@pytest.mark.unit
def test_f001_queue_metrics_with_no_series_are_unknown_not_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """A box whose pgwatch exports no queue metric answers with no series."""
    responses = dict(RESPONSES, **{"pgwatch_pg_autovacuum_queue_n_dead_tup": []})
    report = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")
    assert "autovacuum_tables" not in report["results"]["node-1"]


@pytest.mark.unit
def test_f001_window_facts_name_changed_settings_and_the_most_vacuumed_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """F001-O1: settings unchanged across the window, or which changed (each
    value is its own series); F001-O3: completed runs per table, top N."""
    responses = {  # the stub answers the first needle a query contains: these first
        "count by (setting_name) (count by (setting_name, setting_value)": [
            {"metric": {"setting_name": "autovacuum_naptime"}, "value": [0, "2"]},
            {"metric": {"setting_name": "autovacuum_vacuum_scale_factor"}, "value": [0, "1"]}],
        "topk(10, increase(pgwatch_table_stats_autovacuum_count": _rows(
            [("public", "a", 2), ("public", "b", 7)], *TABLE_STATS),
        **RESPONSES,
    }
    report = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    node = report["results"]["node-1"]
    assert node["settings_changed_in_window"] == ["autovacuum_naptime"]
    assert node["settings_window_seconds"] == 604800
    assert node["autovacuum_runs_top"] == [
        {"database": "app", "schema_name": "public", "table_name": "b", "autovacuum_runs": 7},
        {"database": "app", "schema_name": "public", "table_name": "a", "autovacuum_runs": 2}]


@pytest.mark.unit
def test_f001_from_a003_the_normal_path_carries_the_dynamics(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_all_reports and --check-id F001 build F001 from A003 when A003
    exists (REV round 1: the fields were only on the fallback path)."""
    gen = _gen(monkeypatch, RESPONSES)
    a003 = {"results": {"node-1": {"data": {
        "maintenance_work_mem": {"setting": "65536", "unit": "kB", "category": "c", "context": "user",
                                 "vartype": "integer", "pretty_value": "64 MB"},
        "autovacuum_naptime": {"setting": "60", "unit": "s", "category": "c", "context": "sighup",
                               "vartype": "integer", "pretty_value": "1 min"}}}}}
    report = gen.generate_f001_from_a003(a003, "node-1", "local")
    validate_report(report)
    node = report["results"]["node-1"]
    assert "maintenance_work_mem" in node["data"]  # in F001_SETTINGS now
    assert node["autovacuum_workers"]["samples"] == 20159
    assert [t["table_name"] for t in node["autovacuum_tables"]] == ["small", "big"]
    assert node["autovacuum_scope"] == {"scope_bytes": GiB, "tables_in_scope_by_size": 1,
                                        "smaller_tables_with_dead_tuples": 1,
                                        "smaller_tables_past_threshold": 1, "rows_shown": 2}


@pytest.mark.unit
def test_f001_joins_quoted_table_stats_names_and_skips_other_and_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """table_stats labels are quote_ident()ed, the queue labels raw; "$other$" is
    pgwatch's sum of the tail tables; an excluded database is left out."""
    responses = {
        "pgwatch_pg_autovacuum_queue_n_dead_tup": _rows([("public", "MyTable", 900), ("public", "x", 5)])
        + [{"metric": {"datname": "template1", "schemaname": "public", "relname": "t"}, "value": [0, "9"]}],
        "pgwatch_pg_autovacuum_queue_autovacuum_threshold": _rows([("public", "MyTable", 100), ("public", "x", 50)]),
        "pgwatch_table_stats_total_relation_size_b": _rows(
            [('"public"', '"MyTable"', 2 * GiB), ("$other$", "$other$", 5 * GiB)], *TABLE_STATS),
        "pgwatch_table_stats_seconds_since_last_vacuum": _rows([('"public"', '"MyTable"', 60)], *TABLE_STATS),
    }
    tables = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")[
        "results"]["node-1"]["autovacuum_tables"]
    assert [(t["table_name"], t["in_scope_by_size"], t["seconds_since_last_vacuum"]) for t in tables] == [
        ("MyTable", True, 60), ("x", False, None)]


@pytest.mark.unit
def test_f001_nan_and_inf_samples_are_dropped_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = dict(RESPONSES, **{
        "pgwatch_pg_autovacuum_queue_n_dead_tup": _rows([("public", "small", "NaN"), ("public", "big", "+Inf"),
                                                         ("public", "ok", 10)]),
        "avg_over_time(pgwatch_pg_autovacuum_workers_active_workers": [{"metric": {}, "value": [0, "NaN"]}],
    })
    report = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    node = report["results"]["node-1"]
    assert node["autovacuum_workers"]["mean_active"] is None
    assert "ok" in [t["table_name"] for t in node["autovacuum_tables"]]


@pytest.mark.unit
def test_f001_never_vacuumed_only_where_table_stats_covers_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """table_stats covers its top 100 tables per database and partition roots; a
    queue-only table (no size row) is unknown, not "never vacuumed" (REV round 2)."""
    responses = dict(RESPONSES, **{
        "pgwatch_pg_autovacuum_queue_n_dead_tup": _rows([("public", "small", 5000), ("public", "part_2026_10", 700)]),
        "pgwatch_pg_autovacuum_queue_autovacuum_threshold": _rows([("public", "small", 1000), ("public", "part_2026_10", 100)]),
    })
    tables = {t["table_name"]: t for t in _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report(
        "local", "node-1")["results"]["node-1"]["autovacuum_tables"]}
    assert tables["small"]["vacuum_recorded"] is False   # covered (size row), no vacuum series
    assert tables["part_2026_10"]["vacuum_recorded"] is None  # not covered by table_stats


@pytest.mark.unit
def test_f001_top_runs_leave_excluded_databases_out_in_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, RESPONSES)
    asked = []
    original = gen.query_instant
    monkeypatch.setattr(gen, "query_instant", lambda q: asked.append(q) or original(q))
    gen.generate_f001_autovacuum_settings_report("local", "node-1")
    [top] = [q for q in asked if q.startswith("topk(10, increase(")]
    assert 'datname!~"' in top and "template0" in top


@pytest.mark.unit
def test_f001_run_counts_carry_their_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """autovacuum_runs_in_window and autovacuum_runs_top are counted over
    F001_WINDOW; without the window beside them a report cannot source "N runs
    in 7 days"."""
    responses = {"topk(10, increase(pgwatch_table_stats_autovacuum_count": _rows(
        [("public", "a", 2)], *TABLE_STATS), **RESPONSES}
    report = _gen(monkeypatch, responses).generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)
    node = report["results"]["node-1"]
    assert node["autovacuum_tables"] and node["autovacuum_runs_top"]
    assert node["autovacuum_runs_window_seconds"] == 604800
