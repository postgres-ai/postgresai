"""H004: the facts a redundant-index decision rests on, as the box's metric store has them.

- the definitions of each candidate and of each index that covers it, with the
  covering index's size, come from the metric's own labels (index_definition,
  redundant_to_json) when the sink has none: a pair can be checked only with both;
- the database size is matched by the series' real_dbname (the redundant_indexes
  series carry pgwatch's source name in dbname, not the database's name), and the
  entry is keyed by the database's name;
- each candidate's scan counter at its first monitored reading in the lookback and
  its latest (protocols checkup-acceptance-criteria/H004: H004-O2), with a restart
  or a statistics reset between them flagged;
- candidates and covering indexes also on the unused-index list (H004-O3, C4).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

NOW = 1_793_660_000.0
FIRST = NOW - 4 * 86400
LAST = NOW - 3600
SOURCE = "pooler-host-postgres"


def _labels(index: str, reason: str, covers: List[Dict[str, Any]] | None = None) -> Dict[str, str]:
    labels = {"dbname": SOURCE, "real_dbname": "app", "schema_name": "demo", "table_name": "accounts",
              "index_name": f"demo.{index}", "access_method": "btree", "reason": reason,
              "index_definition": f"CREATE INDEX {index} ON demo.accounts USING btree (email)"}
    if covers is not None:
        labels["redundant_to_json"] = json.dumps(covers)
    return labels


COVER = {"index_name": "demo.accounts_email_status_idx",
         "index_definition": "CREATE INDEX accounts_email_status_idx ON demo.accounts USING btree (email, status)",
         "index_size_bytes": 1007616}
DUP = _labels("accounts_email_dup_idx", "demo.accounts_email_status_idx", [COVER])
OLD = _labels("accounts_email_old_idx", "demo.accounts_email_status_idx")  # an older box: no redundant_to_json


def _gen(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, Any]) -> PostgresReportGenerator:
    gen = PostgresReportGenerator("http://prom", use_current_time=True)
    seen: List[str] = []

    def fake(query: str) -> Dict[str, Any]:
        seen.append(query)
        for needle, rows in responses.items():
            if needle in query:
                rows = rows(query) if callable(rows) else rows
                return {"status": "success", "data": {"result": rows}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", fake)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["app", SOURCE])
    monkeypatch.setattr(gen, "get_index_definitions_from_sink", lambda db: {})
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: {
        "version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"})
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: NOW)
    gen.seen = seen  # type: ignore[attr-defined]
    return gen


def _only_source(rows: List[Dict[str, Any]]):
    """Series of the redundant_indexes metric answer only for the source's dbname."""
    return lambda q: rows if f'dbname="{SOURCE}"' in q else []


def _responses(**over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "pgwatch_db_size_size_b": [{"metric": {"datname": "app", "dbname": SOURCE}, "value": [NOW, "100000000"]}],
        "pgwatch_db_stats_postmaster_uptime_s": [{"metric": {}, "value": [NOW, str(20 * 86400)]}],
        "pgwatch_stats_reset_stats_reset_epoch{": lambda q: (
            [{"metric": {"database_name": "app"}, "value": [NOW, str(NOW - 30 * 86400)]}]
            if 'database_name="app"' in q else []),
        "min_over_time(timestamp(pgwatch_redundant_indexes_index_usage": _only_source([
            {"metric": DUP, "value": [NOW, str(FIRST)]}, {"metric": OLD, "value": [NOW, str(FIRST + 86400)]}]),
        "max_over_time(timestamp(pgwatch_redundant_indexes_index_usage": _only_source([
            {"metric": DUP, "value": [NOW, str(LAST)]}, {"metric": OLD, "value": [NOW, str(LAST)]}]),
        "min_over_time(pgwatch_redundant_indexes_index_usage": _only_source([
            {"metric": DUP, "value": [NOW, "5"]}, {"metric": OLD, "value": [NOW, "0"]}]),
        "pgwatch_redundant_indexes_index_size_bytes": _only_source([
            {"metric": DUP, "value": [NOW, "827392"]}, {"metric": OLD, "value": [NOW, "409600"]}]),
        "pgwatch_redundant_indexes_index_usage{": lambda q: (
            [{"metric": DUP, "value": [NOW, "9"]}] if "accounts_email_dup_idx" in q else
            [{"metric": OLD, "value": [NOW, "0"]}]),
        # the unused list: raw names, by datname
        "pgwatch_unused_indexes_index_size_bytes": lambda q: (
            [{"metric": {"datname": "app", "schema_name": "demo", "index_name": "accounts_email_dup_idx"},
              "value": [NOW, "827392"]},
             {"metric": {"datname": "app", "schema_name": "demo", "index_name": "accounts_email_status_idx"},
              "value": [NOW, "1007616"]}] if 'datname="app"' in q else []),
    }
    base.update(over)
    return base


def _entry(report: Dict[str, Any], db: str = "app") -> Dict[str, Any]:
    return next(iter(report["results"].values()))["data"][db]


def test_definitions_sizes_and_database_name(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, _responses()).generate_h004_redundant_indexes_report("c", "n")
    validate_report(report)
    data = next(iter(report["results"].values()))["data"]
    assert list(data) == ["app"]  # keyed by the database's name, not pgwatch's source name
    entry = data["app"]
    assert entry["database_size_bytes"] == 100000000
    dup, old = entry["redundant_indexes"]
    assert dup["index_definition"].startswith("CREATE INDEX accounts_email_dup_idx")
    assert dup["redundant_to"] == [dict(COVER, index_size_bytes=1007616.0, index_size_pretty="984 KiB",
                                        also_unused=True)]
    # no redundant_to_json: the names from the reason, definition and size unknown
    assert old["redundant_to"][0]["index_name"] == "demo.accounts_email_status_idx"
    assert old["redundant_to"][0]["index_definition"] == "Definition not available"
    assert old["redundant_to"][0]["index_size_bytes"] == 0


def test_two_dated_readings_and_window(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry(_gen(monkeypatch, _responses()).generate_h004_redundant_indexes_report("c", "n"))
    dup, old = entry["redundant_indexes"]
    assert (dup["index_usage_first"], dup["index_usage"]) == (5, 9)  # rose
    assert dup["index_usage_first_time"] == PostgresReportGenerator._h002_utc(FIRST)
    assert dup["index_usage_last_time"] == PostgresReportGenerator._h002_utc(LAST)
    assert old["index_usage_first_epoch"] == FIRST + 86400  # its own first reading
    readings = entry["usage_readings"]
    assert (readings["start_epoch"], readings["end_epoch"]) == (FIRST, LAST)
    assert readings["restart_in_window"] is False and readings["reset_in_window"] is False


def test_restart_between_readings_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{
        "pgwatch_db_stats_postmaster_uptime_s": [{"metric": {}, "value": [NOW, str(2 * 86400)]}]}))
    readings = _entry(gen.generate_h004_redundant_indexes_report("c", "n"))["usage_readings"]
    assert readings["restart_in_window"] is True and readings["reset_in_window"] is False


def test_no_readings_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{
        "min_over_time(timestamp(pgwatch_redundant_indexes_index_usage": [],
        "max_over_time(timestamp(pgwatch_redundant_indexes_index_usage": []}))
    report = gen.generate_h004_redundant_indexes_report("c", "n")
    validate_report(report)
    entry = _entry(report)
    assert "usage_readings" not in entry
    assert all("index_usage_first" not in i for i in entry["redundant_indexes"])


def test_also_unused_for_candidates_and_covering_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses())
    dup, old = _entry(gen.generate_h004_redundant_indexes_report("c", "n"))["redundant_indexes"]
    assert dup["also_unused"] is True and old["also_unused"] is False
    assert old["redundant_to"][0]["also_unused"] is True  # the covering index is itself unused
    assert any('datname="app"' in q and "pgwatch_unused_indexes_index_size_bytes" in q for q in gen.seen)


def test_unused_list_unreadable_leaves_overlap_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses())
    real = gen.query_instant
    monkeypatch.setattr(gen, "query_instant", lambda q: {"status": "error"}
                        if "pgwatch_unused_indexes_index_size_bytes" in q else real(q))
    entry = _entry(gen.generate_h004_redundant_indexes_report("c", "n"))
    assert all("also_unused" not in i for i in entry["redundant_indexes"])  # unknown, never "no"


def test_qualified_names() -> None:
    q = PostgresReportGenerator._h004_qualified
    assert q("demo.accounts_email_idx", "demo") == "demo.accounts_email_idx"
    assert q("accounts_email_idx", "public") == "public.accounts_email_idx"
    assert q('"My Schema"."Idx.x"', "x") == "My Schema.Idx.x"


def test_readings_aggregate_each_candidates_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """redundant_to_json is a label holding the covering index's size, so a candidate has a new
    series each time that index grows: each reading must be one value per candidate, taken across
    its series, not whichever series is read last."""
    gen = _gen(monkeypatch, _responses())
    gen.generate_h004_redundant_indexes_report("c", "n")
    readings = [q for q in gen.seen if "over_time" in q and "pgwatch_redundant_indexes_index_usage" in q
                and "[3h]" not in q]
    assert len(readings) == 6  # three readings for each of the two databases
    for q in readings:
        assert q.startswith(("min by (schema_name, table_name, index_name) (",
                             "max by (schema_name, table_name, index_name) (")), q
