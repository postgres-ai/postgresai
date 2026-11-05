"""F004: per-table vacuum facts and the estimate's history in the box reporter's F004 file.

CheckupLoop's F004 report needs them to tell dead tuples not yet vacuumed from
reusable free space (protocols checkup-acceptance-criteria/F004: F004-O1 the
previous value or the min-max across the window, F004-O2 dead tuples, write
counters and last autovacuum of the top tables, F004-C2 the kind of space).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

DB = "app"


def _rows(values: List[tuple], schema_label: str, table_label: str) -> List[Dict]:
    return [{"metric": {"datname": DB, schema_label: s, table_label: t}, "value": [0, str(v)]}
            for s, t, v in values]


def _stats(values: List[tuple]) -> List[Dict]:
    # table_stats labels are quote_ident()ed
    return _rows([(f'"{s}"' if s != s.lower() else s, t, v) for s, t, v in values], "schema", "table_name")


def _bloat(values: List[tuple]) -> List[Dict]:
    return _rows(values, "schemaname", "tblname")


BLOAT = [("public", "victim", 38_000_000), ("Sales", "orders", 32_768)]
RESPONSES: Dict[str, Any] = {
    # ordered: the first needle a query contains answers it
    "pgwatch_db_size_size_b": [{"metric": {"datname": DB}, "value": [0, "70000000"]}],
    "pgwatch_settings_configured": [
        {"metric": {"setting_name": "autovacuum_vacuum_threshold", "setting_value": "50"}, "value": [0, "1"]},
        {"metric": {"setting_name": "autovacuum_vacuum_scale_factor", "setting_value": "0.2"}, "value": [0, "1"]}],
    "pgwatch_db_stats_numbackends": [{"metric": {"datname": DB}, "value": [0, "1"]}],
    "min_over_time(timestamp(pgwatch_pg_table_bloat_bloat_size": [{"metric": {}, "value": [0, "1793400586"]}],
    "[3h] offset 1d": [{"metric": {}, "value": [0, "38032768"]}],
    "[3h] offset 2d": [{"metric": {}, "value": [0, "30000000"]}],
    "[3h] offset 3d": None,  # failed query: that day is left out
    "[3h] offset": [],
    "sum(last_over_time(pgwatch_pg_table_bloat_bloat_size": [{"metric": {}, "value": [0, "38032768"]}],
    "min_over_time(pgwatch_pg_table_bloat_bloat_size": _bloat([("public", "victim", 30_000_000), ("Sales", "orders", 32_768)]),
    "max_over_time(pgwatch_pg_table_bloat_bloat_size": _bloat([("public", "victim", 38_000_000), ("Sales", "orders", 40_960)]),
    "pgwatch_pg_table_bloat_real_size_mib": _bloat([("public", "victim", 40), ("Sales", "orders", 2)]),
    "pgwatch_pg_table_bloat_bloat_size": _bloat(BLOAT),
    "pgwatch_pg_table_bloat_bloat_pct": _bloat([("public", "victim", 95.0), ("Sales", "orders", 1.6)]),
    "pgwatch_pg_autovacuum_queue_autovacuum_threshold": _rows(
        [("public", "victim", 3250), ("Sales", "orders", 4050)], "schemaname", "relname"),
    "pgwatch_table_stats_n_live_tup": _stats([("public", "victim", 16000), ("Sales", "orders", 20000)]),
    "pgwatch_table_stats_n_dead_tup": _stats([("public", "victim", 304000), ("Sales", "orders", 0)]),
    "pgwatch_table_stats_autovacuum_count": _stats([("public", "victim", 0), ("Sales", "orders", 1)]),
    "pgwatch_table_stats_vacuum_count": _stats([("public", "victim", 0), ("Sales", "orders", 0)]),
    "pgwatch_table_stats_n_tup_ins": _stats([("public", "victim", 0), ("Sales", "orders", 12)]),
    "pgwatch_table_stats_n_tup_upd": _stats([("public", "victim", 0), ("Sales", "orders", 7.2)]),
    "pgwatch_table_stats_n_tup_del": _stats([("public", "victim", 0), ("Sales", "orders", 0)]),
    "pgwatch_table_stats_n_tup_hot_upd": _stats([("public", "victim", 0), ("Sales", "orders", 3)]),
    "min_over_time(pgwatch_table_stats_table_size_b": _stats([("public", "victim", 40_000_000), ("Sales", "orders", 2_000_000)]),
    "max_over_time(pgwatch_table_stats_table_size_b": _stats([("public", "victim", 40_378_368), ("Sales", "orders", 2_000_000)]),
    "pgwatch_table_stats_table_size_b": _stats([("public", "victim", 40_378_368), ("Sales", "orders", 2_000_000)]),
    "pgwatch_table_stats_toast_size_b": _stats([("public", "victim", 8192)]),
    # victim never vacuumed: no series; orders vacuumed 20 days ago
    "pgwatch_table_stats_seconds_since_last_vacuum": _stats([("Sales", "orders", 20 * 86400)]),
    "pgwatch_table_stats_no_autovacuum": _stats([("public", "victim", 1), ("Sales", "orders", 0)]),
}


def _gen(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, Any]) -> PostgresReportGenerator:
    gen = PostgresReportGenerator("http://prom", use_current_time=True)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: [DB])

    def fake(query: str) -> Dict[str, Any]:
        for needle, rows in responses.items():
            if needle in query:
                if rows is None:
                    return {"status": "error"}
                return {"status": "success", "data": {"result": rows}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", fake)
    return gen


def _db(report: Dict[str, Any]) -> Dict[str, Any]:
    return report["results"]["node-1"]["data"][DB]


@pytest.mark.unit
def test_f004_tables_carry_dead_tuples_trigger_runs_writes_and_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES).generate_f004_heap_bloat_report("local", "node-1")
    validate_report(report)
    victim, orders = _db(report)["bloated_tables"]
    assert {k: victim[k] for k in (
        "n_live_tup", "n_dead_tup", "vacuum_threshold", "autovacuum_runs_in_window", "vacuum_runs_in_window",
        "n_tup_ins_in_window", "n_tup_upd_in_window", "n_tup_del_in_window", "n_tup_hot_upd_in_window",
        "table_size_bytes", "toast_size_bytes", "table_size_min_in_window_bytes", "table_size_max_in_window_bytes",
        "bloat_size_min_in_window_bytes", "bloat_size_max_in_window_bytes", "seconds_since_last_vacuum", "vacuum_recorded",
        "autovacuum_disabled")} == {
        "n_live_tup": 16000, "n_dead_tup": 304000, "vacuum_threshold": 3250, "autovacuum_runs_in_window": 0,
        "vacuum_runs_in_window": 0, "n_tup_ins_in_window": 0, "n_tup_upd_in_window": 0, "n_tup_del_in_window": 0,
        "n_tup_hot_upd_in_window": 0, "table_size_bytes": 40_378_368, "toast_size_bytes": 8192,
        "table_size_min_in_window_bytes": 40_000_000, "table_size_max_in_window_bytes": 40_378_368,
        "bloat_size_min_in_window_bytes": 30_000_000, "bloat_size_max_in_window_bytes": 38_000_000,
        "seconds_since_last_vacuum": None, "vacuum_recorded": False, "autovacuum_disabled": True}
    # quoted table_stats labels match the bloat metric's raw names; window counts are rounded
    assert (orders["schema_name"], orders["n_tup_upd_in_window"], orders["toast_size_bytes"]) == ("Sales", 7, None)
    assert orders["vacuum_recorded"] is True and orders["autovacuum_disabled"] is False


@pytest.mark.unit
def test_f004_no_run_in_the_window_when_the_last_vacuum_is_older(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table whose last vacuum is older than the window has no run in it."""
    report = _gen(monkeypatch, RESPONSES).generate_f004_heap_bloat_report("local", "node-1")
    orders = _db(report)["bloated_tables"][1]
    assert orders["seconds_since_last_vacuum"] == 20 * 86400
    assert orders["autovacuum_runs_in_window"] == 0


@pytest.mark.unit
def test_f004_history_is_one_total_a_day_with_the_first_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES).generate_f004_heap_bloat_report("local", "node-1")
    history = _db(report)["history"]
    assert [(d["days_ago"], d["total_bloat_size_bytes"]) for d in history["daily"]] == [
        (0, 38_032_768), (1, 38_032_768), (2, 30_000_000)]
    assert all(d["at"] for d in history["daily"])
    assert history["first_reading_at"].startswith("2026-10-30T22:49")
    assert history["window_seconds"] == 7 * 86400


@pytest.mark.unit
def test_f004_unknown_facts_are_null_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without table_stats and queue series every fact is null and there is no history."""
    responses = {"sum(last_over_time": [], "timestamp(": [], "table_stats": [], "pgwatch_pg_autovacuum": [],
                 "min_over_time(pgwatch_pg_table_bloat": [], "max_over_time(pgwatch_pg_table_bloat": []}
    responses.update({k: v for k, v in RESPONSES.items() if k not in responses})
    report = _gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1")
    validate_report(report)
    victim = _db(report)["bloated_tables"][0]
    assert victim["n_dead_tup"] is None and victim["vacuum_recorded"] is None and victim["autovacuum_disabled"] is None
    assert "history" not in _db(report)


@pytest.mark.unit
def test_f004_carries_the_server_settings_behind_the_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES).generate_f004_heap_bloat_report("local", "node-1")
    validate_report(report)
    assert _db(report)["autovacuum_settings"] == {"autovacuum_vacuum_threshold": 50.0,
                                                  "autovacuum_vacuum_scale_factor": 0.2}


@pytest.mark.unit
def test_f004_other_row_gets_no_facts_from_another_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgwatch's '$other$' rows of the bloat metric and table_stats sum different tables."""
    responses = dict(RESPONSES)
    responses["pgwatch_pg_table_bloat_bloat_size"] = _bloat(BLOAT + [("$other$", "$other$", 1000)])
    responses["pgwatch_table_stats_n_dead_tup"] = _stats([("public", "victim", 304000), ("$other$", "$other$", 99)])
    report = _gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1")
    other = next(t for t in _db(report)["bloated_tables"] if t["table_name"] == "$other$")
    assert other["n_dead_tup"] is None and other["autovacuum_disabled"] is None


@pytest.mark.unit
def test_f004_table_stats_facts_are_one_series_per_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """table_stats carries table_size_cardinality_mb, so a table that crosses a power of ten in
    the window has two series; every per-table query must collapse them (REV !555 round 1)."""
    gen = _gen(monkeypatch, RESPONSES)
    issued: List[str] = []
    answer = gen.query_instant

    def record(query: str) -> Dict[str, Any]:
        issued.append(query)
        return answer(query)

    monkeypatch.setattr(gen, "query_instant", record)
    gen.generate_f004_heap_bloat_report("local", "node-1")
    stats = [q for q in issued if "pgwatch_table_stats_" in q]
    assert stats and all("by (schema, table_name) (pgwatch_table_stats_" in q or
                         q.startswith(("min by (schema, table_name)", "max by (schema, table_name)"))
                         for q in stats), stats
    # VictoriaMetrics' increase() counts a series' first value from 0 when it starts inside the
    # window (a new box, a size crossing): window counts sum the merged series' per-step rises.
    assert not any("increase(" in q for q in stats), stats
    # ...and a counter reset (pg_stat_reset, crash restart, failover) must not count the whole
    # pre-reset total: window counts sum the clamped per-step deltas (REV !555 round 3).
    window = [q for q in stats if "[7d:5m]" in q and "_count" in q or "n_tup_" in q]
    assert window and all(q.startswith("sum_over_time(clamp_min(") and "offset 5m" in q for q in window), window
    # ...and a data gap longer than a step (box down, sink down) must not drop the rise across it:
    # the previous value comes from the raw samples over the window (REV !555 round 4).
    assert all("[7d] offset 5m" in q for q in window), window
    # ...and the previous value comes only from series present at the step: a stale series (from
    # before a size crossing) must not hold the counts at 0 after a reset (REV !555 round 5).
    assert all("[7d] offset 5m) and pgwatch_table_stats_" in q for q in window), window


@pytest.mark.unit
@pytest.mark.parametrize("rows,expected", [
    # the older pgwatch label schema
    ([{"metric": {"tag_setting_name": "autovacuum_vacuum_threshold", "tag_setting_value": "50"}, "value": [0, "1"]},
      {"metric": {"tag_setting_name": "autovacuum_vacuum_scale_factor", "tag_setting_value": "0.2"}, "value": [0, "1"]}],
     {"autovacuum_vacuum_threshold": 50.0, "autovacuum_vacuum_scale_factor": 0.2}),
    # one of the two: no partial block (the schema requires both)
    ([{"metric": {"setting_name": "autovacuum_vacuum_threshold", "setting_value": "50"}, "value": [0, "1"]}], None),
    (None, None),  # a failed query
])
def test_f004_trigger_settings_both_label_schemas_and_never_partial(monkeypatch, rows, expected) -> None:
    responses = dict(RESPONSES)
    responses["pgwatch_settings_configured"] = rows
    report = _gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1")
    validate_report(report)
    assert _db(report).get("autovacuum_settings") == expected


@pytest.mark.unit
def test_f004_failed_last_vacuum_query_is_unknown_not_never(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {"pgwatch_table_stats_seconds_since_last_vacuum": None}
    responses.update({k: v for k, v in RESPONSES.items() if k not in responses})
    report = _gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1")
    victim = _db(report)["bloated_tables"][0]
    assert victim["vacuum_recorded"] is None and victim["seconds_since_last_vacuum"] is None


@pytest.mark.unit
def test_f004_quoted_table_names_match_the_bloat_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """table_stats labels are quote_ident()ed (schema and table), pg_table_bloat's are raw."""
    responses = dict(RESPONSES)
    responses["pgwatch_pg_table_bloat_bloat_size"] = _bloat([("public", "Order Items", 38_000_000)])
    responses["pgwatch_table_stats_n_dead_tup"] = _rows([("public", '"Order Items"', 1234)], "schema", "table_name")
    report = _gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1")
    table = next(t for t in _db(report)["bloated_tables"] if t["table_name"] == "Order Items")
    assert table["n_dead_tup"] == 1234


@pytest.mark.unit
def test_f004_window_cap_zeroes_both_run_counts_only_for_an_old_vacuum(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {"pgwatch_table_stats_vacuum_count": _stats([("public", "victim", 2), ("Sales", "orders", 5)]),
                 "pgwatch_table_stats_seconds_since_last_vacuum": _stats([("public", "victim", 3600),
                                                                         ("Sales", "orders", 20 * 86400)])}
    responses.update({k: v for k, v in RESPONSES.items() if k not in responses})
    victim, orders = _db(_gen(monkeypatch, responses).generate_f004_heap_bloat_report("local", "node-1"))["bloated_tables"]
    assert victim["vacuum_runs_in_window"] == 2  # vacuumed an hour ago: the count stays
    assert (orders["vacuum_runs_in_window"], orders["autovacuum_runs_in_window"]) == (0, 0)
