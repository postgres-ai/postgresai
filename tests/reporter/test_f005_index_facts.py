"""F005: the facts CheckupLoop's rebuild-or-drop decision reads, in the box reporter's F005 file.

Protocols checkup-acceptance-criteria/F005: F005-O1 the previous value and whether the top estimates
changed, F005-O2 the unused and redundant list overlap and the top tables' writes and dead tuples,
F005-C2 one decision per top index, F005-R2 an observed low-traffic window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

DB = "app"


def _btree(values: List[tuple]) -> List[Dict]:
    return [{"metric": {"datname": DB, "schemaname": s, "tblname": t, "idxname": i}, "value": [0, str(v)]}
            for s, t, i, v in values]


def _stats(values: List[tuple]) -> List[Dict]:
    return [{"metric": {"datname": DB, "schema": s, "table_name": t}, "value": [0, str(v)]} for s, t, v in values]


INDEXES = [("public", "orders", "orders_pkey", 5_000_000), ("public", "orders", "orders_n_idx", 4_000_000),
           ("public", "items", "items_pkey", 1_000), ("Sales", "Big", "Big_idx", 2_000), ("public", "items", "v1.2_idx", 500),
           ("my.schema", "t", "dotted_idx", 400)]
RESPONSES: Dict[str, Any] = {
    # ordered: the first needle a query contains answers it
    "pgwatch_db_size_size_b": [{"metric": {"datname": DB}, "value": [0, "70000000"]}],
    "pgwatch_db_stats_numbackends": [{"metric": {"datname": DB}, "value": [0, "1"]}],
    "min_over_time(timestamp(pgwatch_pg_btree_bloat_bloat_size": [{"metric": {}, "value": [0, "1793400586"]}],
    "[3h] offset 1d": [{"metric": {}, "value": [0, "9000000"]}],
    "[3h] offset": [],
    "sum(last_over_time(pgwatch_pg_btree_bloat_bloat_size": [{"metric": {}, "value": [0, "9001000"]}],
    "min_over_time(pgwatch_pg_btree_bloat_bloat_size": _btree([
        ("public", "orders", "orders_pkey", 4_000_000), ("public", "orders", "orders_n_idx", 4_000_000),
        ("public", "items", "items_pkey", 1_000)]),
    "max_over_time(pgwatch_pg_btree_bloat_bloat_size": _btree(INDEXES),
    "min_over_time(pgwatch_pg_btree_bloat_real_size_mib": _btree([("public", "orders", "orders_pkey", 5)]),
    "max_over_time(pgwatch_pg_btree_bloat_real_size_mib": _btree([("public", "orders", "orders_pkey", 6)]),
    "pgwatch_pg_btree_bloat_real_size_mib": _btree([(s, t, i, 6) for s, t, i, _ in INDEXES]),
    "pgwatch_pg_btree_bloat_bloat_size": _btree(INDEXES),
    "pgwatch_pg_btree_bloat_bloat_pct": _btree([(s, t, i, 80) for s, t, i, _ in INDEXES]),
    "pgwatch_unused_indexes_index_size_bytes": [
        {"metric": {"datname": DB, "schema_name": "public", "table_name": "orders", "index_name": "orders_n_idx"},
         "value": [0, "1"]},
        {"metric": {"datname": DB, "schema_name": "public", "table_name": "items", "index_name": "v1.2_idx"},
         "value": [0, "1"]}],
    "pgwatch_redundant_indexes_index_size_bytes": [
        # the metric's name is formatted: no schema for public, schema-qualified and quoted otherwise;
        # and another database's row
        {"metric": {"dbname": "source-1", "real_dbname": DB, "schema_name": "public", "table_name": "items",
                    "index_name": "items_pkey"}, "value": [0, "1"]},
        {"metric": {"dbname": "source-1", "real_dbname": DB, "schema_name": "Sales", "table_name": "Big",
                    "index_name": '"Sales"."Big_idx"'}, "value": [0, "1"]},
        {"metric": {"dbname": "source-1", "real_dbname": DB, "schema_name": "my.schema", "table_name": "t",
                    "index_name": '"my.schema".dotted_idx'}, "value": [0, "1"]},
        {"metric": {"dbname": "source-2", "real_dbname": "other", "schema_name": "public", "table_name": "orders",
                    "index_name": "orders_pkey"}, "value": [0, "1"]}],
    "pgwatch_pg_autovacuum_queue_autovacuum_threshold": [
        {"metric": {"datname": DB, "schemaname": "public", "relname": "orders"}, "value": [0, "3250"]}],
    "pgwatch_table_stats_n_live_tup": _stats([("public", "orders", 16000), ("public", "items", 100)]),
    "pgwatch_table_stats_n_dead_tup": _stats([("public", "orders", 304000), ("public", "items", 0)]),
    "pgwatch_table_stats_autovacuum_count": _stats([("public", "orders", 0)]),
    "pgwatch_table_stats_vacuum_count": _stats([("public", "orders", 0)]),
    "pgwatch_table_stats_n_tup_upd": _stats([("public", "orders", 200)]),
    "pgwatch_table_stats_n_tup_hot_upd": _stats([("public", "orders", 50)]),
    "pgwatch_table_stats_no_autovacuum": _stats([("public", "orders", 1)]),
}


def _hourly(values: List[float]) -> List[Dict]:
    base = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    return [{"metric": {}, "values": [[base + n * 3600, str(v)] for n, v in enumerate(values)]}]


def _gen(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, Any], traffic: List[Dict]) -> PostgresReportGenerator:
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
    monkeypatch.setattr(gen, "query_range", lambda *a, **k: traffic)
    return gen


def _db(report: Dict[str, Any]) -> Dict[str, Any]:
    return report["results"]["node-1"]["data"][DB]


def _by_name(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {i["index_name"]: i for i in entry["bloated_indexes"]}


@pytest.mark.unit
def test_f005_window_queries_give_one_value_per_index(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, RESPONSES, [])
    issued: List[str] = []
    answer = gen.query_instant

    def record(query: str) -> Dict[str, Any]:
        issued.append(query)
        return answer(query)

    monkeypatch.setattr(gen, "query_instant", record)
    gen.generate_f005_btree_bloat_report("local", "node-1")
    window = [q for q in issued if "pgwatch_pg_btree_bloat" in q and "_over_time" in q and "[7d]" in q]
    assert len(window) >= 4 and all("by (schemaname, tblname, idxname)" in q for q in window), window


@pytest.mark.unit
def test_f005_indexes_carry_their_lists_and_their_window(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    indexes = _by_name(_db(report))
    assert indexes["orders_n_idx"]["on_unused_list"] is True and indexes["orders_n_idx"]["on_redundant_list"] is False
    assert indexes["items_pkey"]["on_redundant_list"] is True and indexes["items_pkey"]["on_unused_list"] is False
    # another database's redundant row is not this database's
    assert indexes["orders_pkey"]["on_redundant_list"] is False
    pkey = indexes["orders_pkey"]
    assert (pkey["bloat_size_min_in_window_bytes"], pkey["bloat_size_max_in_window_bytes"]) == (4_000_000, 5_000_000)
    assert (pkey["real_size_min_in_window_bytes"], pkey["real_size_max_in_window_bytes"]) == (5 * 1024 ** 2, 6 * 1024 ** 2)
    assert indexes["items_pkey"]["real_size_min_in_window_bytes"] is None
    assert indexes["Big_idx"]["on_redundant_list"] is True  # a schema-qualified, quoted label
    assert indexes["v1.2_idx"]["on_unused_list"] is True  # a raw name with a dot
    assert indexes["dotted_idx"]["on_redundant_list"] is True  # a quoted schema with a dot


@pytest.mark.unit
def test_f005_a_capped_list_without_the_name_is_unknown_not_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgwatch folds the indexes past its top 100 into one $other$ row, and F005 ranks differently."""
    other = {"metric": {"datname": DB, "schema_name": "$other$", "table_name": "$other$", "index_name": "$other$"},
             "value": [0, "1"]}
    named = RESPONSES["pgwatch_unused_indexes_index_size_bytes"][0]
    responses = dict(RESPONSES, pgwatch_unused_indexes_index_size_bytes=[named, other])
    report = _gen(monkeypatch, responses, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    indexes = _by_name(_db(report))
    assert indexes["orders_n_idx"]["on_unused_list"] is True
    assert indexes["orders_pkey"]["on_unused_list"] is None


@pytest.mark.unit
def test_f005_the_older_bytes_gauge_gives_the_index_size(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {k: v for k, v in RESPONSES.items() if "real_size_mib" not in k}
    responses = {"min_over_time(pgwatch_pg_btree_bloat_real_size{": _btree([("public", "orders", "orders_pkey", 7_000_000)]),
                 "max_over_time(pgwatch_pg_btree_bloat_real_size{": _btree([("public", "orders", "orders_pkey", 8_000_000)]),
                 **responses}
    report = _gen(monkeypatch, responses, []).generate_f005_btree_bloat_report("local", "node-1")
    pkey = _by_name(_db(report))["orders_pkey"]
    assert (pkey["real_size_min_in_window_bytes"], pkey["real_size_max_in_window_bytes"]) == (7_000_000, 8_000_000)


@pytest.mark.unit
def test_f005_a_failed_list_query_is_unknown_not_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = dict(RESPONSES, pgwatch_unused_indexes_index_size_bytes=None)
    report = _gen(monkeypatch, responses, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    assert all(i["on_unused_list"] is None for i in _db(report)["bloated_indexes"])


@pytest.mark.unit
def test_f005_the_aggregate_row_gets_no_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = INDEXES + [("$other$", "$other$", "$other$", 9_000_000)]
    responses = dict(RESPONSES, pgwatch_pg_btree_bloat_bloat_size=_btree(rows))
    responses["max_over_time(pgwatch_pg_btree_bloat_bloat_size"] = _btree(rows)
    report = _gen(monkeypatch, responses, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    other = _by_name(_db(report))["$other$"]
    assert other["on_unused_list"] is None and other["on_redundant_list"] is None
    assert other["bloat_size_max_in_window_bytes"] is None
    assert "$other$" not in {t["table_name"] for t in _db(report)["table_facts"]}


@pytest.mark.unit
def test_f005_table_facts_are_those_of_the_top_indexes_once_each(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    facts = _db(report)["table_facts"]
    assert [t["table_name"] for t in facts] == ["orders", "Big", "items", "t"]  # heaviest index first
    orders = facts[0]
    assert (orders["n_dead_tup"], orders["n_live_tup"], orders["vacuum_threshold"]) == (304000, 16000, 3250)
    assert orders["autovacuum_disabled"] is True
    # the table's heap estimate is F004's, not part of this file
    assert "bloat_size_min_in_window_bytes" not in orders


@pytest.mark.unit
def test_f005_history_reads_the_btree_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, RESPONSES, []).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    history = _db(report)["history"]
    assert [(d["days_ago"], d["total_bloat_size_bytes"]) for d in history["daily"]] == [(0, 9_001_000), (1, 9_000_000)]
    assert history["first_reading_at"].startswith("2026-10-30T22:49")


@pytest.mark.unit
def test_f005_traffic_is_the_mean_per_utc_hour_of_a_full_day(monkeypatch: pytest.MonkeyPatch) -> None:
    values = [float(n % 24) for n in range(48)]  # sample n, at n:00, is the rate over the hour before it
    report = _gen(monkeypatch, RESPONSES, _hourly(values)).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    traffic = _db(report)["traffic_by_hour_utc"]
    assert len(traffic) == 24 and traffic[5] == {"hour": 5, "xact_commit_per_s": 6.0, "readings": 2}


@pytest.mark.unit
@pytest.mark.parametrize("rows", [[], _hourly([1.0] * 10), _hourly(["nan"] * 48)])
def test_f005_traffic_is_left_out_without_a_full_day(monkeypatch: pytest.MonkeyPatch, rows: List[Dict]) -> None:
    report = _gen(monkeypatch, RESPONSES, rows).generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)
    assert "traffic_by_hour_utc" not in _db(report)
