"""H002: the facts a drop decision rests on, as the box's metric store has them.

- supports_fk and idx_is_btree are value series of the unused_indexes metric
  (config/pgwatch-prometheus/metrics.yml), not labels; an index the series does
  not cover is unknown (None), never "no foreign key".
- the definition comes from the metric's index_definition label when the sink
  has none (the undo statement needs it).
- stats_reset is labelled database_name on pgwatch's stats_reset metric.
- the window: 0 scans in every monitored reading from the first one in the
  lookback to now (protocols checkup-acceptance-criteria/H002: H002-O2,
  checklist Q1), with restarts and resets inside it flagged; per table, the
  updates and HOT share over that window (H002-O5).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

NOW = 1_793_660_000.0
FIRST = NOW - 3 * 86400  # first monitored reading, three days ago


def _idx(index: str, table: str = "accounts", extra: Dict[str, str] | None = None) -> Dict[str, str]:
    return {"datname": "app", "schema_name": "public", "table_name": table, "index_name": index,
            "reason": "Never Used Indexes", **(extra or {})}


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
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["app"])
    monkeypatch.setattr(gen, "get_index_definitions_from_sink", lambda db: {})
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: {
        "version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"})
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: NOW)
    gen.seen = seen  # type: ignore[attr-defined]
    return gen


def _responses(**over: Any) -> Dict[str, Any]:
    labels_a = _idx("idx_a", extra={"index_definition": "CREATE INDEX idx_a ON public.accounts USING btree (a)"})
    labels_b = _idx("idx_b", table="orders")
    base: Dict[str, Any] = {
        "pgwatch_db_size_size_b": [{"metric": {"datname": "app"}, "value": [NOW, "100000000"]}],
        "pgwatch_db_stats_postmaster_uptime_s": [{"metric": {}, "value": [NOW, str(20 * 86400)]}],
        # pgwatch labels stats_reset with database_name, not datname
        "pgwatch_stats_reset_stats_reset_epoch{": lambda q: (
            [{"metric": {"database_name": "app"}, "value": [NOW, str(NOW - 30 * 86400)]}]
            if 'database_name="app"' in q else []),
        "pgwatch_unused_indexes_index_size_bytes": [
            {"metric": labels_a, "value": [NOW, "2097152"]},
            {"metric": labels_b, "value": [NOW, "1048576"]},
        ],
        "pgwatch_unused_indexes_supports_fk": [
            {"metric": labels_a, "value": [NOW, "0"]},
            {"metric": labels_b, "value": [NOW, "1"]},
        ],
        "pgwatch_unused_indexes_idx_is_btree": [
            {"metric": labels_a, "value": [NOW, "1"]},
            {"metric": labels_b, "value": [NOW, "1"]},
        ],
        "min_over_time(timestamp(pgwatch_unused_indexes_idx_scan": [
            {"metric": labels_a, "value": [NOW, str(FIRST)]},
            {"metric": labels_b, "value": [NOW, str(FIRST + 86400)]},
        ],
        "pgwatch_unused_indexes_idx_scan{": [{"metric": {}, "value": [NOW, "0"]}],
        "increase(pgwatch_table_stats_n_tup_upd{": [
            {"metric": {"schema": "public", "table_name": "accounts"}, "value": [NOW, "1000"]},
            {"metric": {"schema": "public", "table_name": "orders"}, "value": [NOW, "0"]},
        ],
        "increase(pgwatch_table_stats_n_tup_hot_upd{": [
            {"metric": {"schema": "public", "table_name": "accounts"}, "value": [NOW, "250"]},
        ],
        "increase(pgwatch_table_stats_n_tup_ins{": [
            {"metric": {"schema": "public", "table_name": "accounts"}, "value": [NOW, "40"]},
        ],
    }
    base.update(over)
    return base


def _db(report: Dict[str, Any]) -> Dict[str, Any]:
    return report["results"]["node-1"]["data"]["app"]


@pytest.mark.unit
def test_h002_flags_come_from_their_series(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _db(_gen(monkeypatch, _responses()).generate_h002_unused_indexes_report("local", "node-1"))
    by = {i["index_name"]: i for i in db["unused_indexes"]}
    assert by["idx_a"]["supports_fk"] is False and by["idx_b"]["supports_fk"] is True
    assert by["idx_a"]["idx_is_btree"] is True


@pytest.mark.unit
def test_h002_flag_unknown_without_its_series(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _db(_gen(monkeypatch, _responses(pgwatch_unused_indexes_supports_fk=[]))
             .generate_h002_unused_indexes_report("local", "node-1"))
    assert all(i["supports_fk"] is None for i in db["unused_indexes"])


@pytest.mark.unit
def test_h002_definition_from_the_metric_label(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _db(_gen(monkeypatch, _responses()).generate_h002_unused_indexes_report("local", "node-1"))
    by = {i["index_name"]: i for i in db["unused_indexes"]}
    assert by["idx_a"]["index_definition"] == "CREATE INDEX idx_a ON public.accounts USING btree (a)"
    assert by["idx_b"]["index_definition"] == "Definition not available"


@pytest.mark.unit
def test_h002_stats_reset_read_by_database_name(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _db(_gen(monkeypatch, _responses()).generate_h002_unused_indexes_report("local", "node-1"))
    assert db["stats_reset"]["stats_reset_epoch"] == NOW - 30 * 86400


@pytest.mark.unit
def test_h002_window_from_monitored_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _gen(monkeypatch, _responses()).generate_h002_unused_indexes_report("local", "node-1")
    db = _db(report)
    w = db["scan_window"]
    assert w["start_epoch"] == FIRST and w["end_epoch"] == NOW
    assert w["days"] == 3.0
    assert w["start_time"].endswith("Z") and w["reading_step"] == "1h"
    assert "missing_hours" not in w  # changes no decision; the counter only grows
    assert w["restart_in_window"] is False and w["reset_in_window"] is False
    by = {i["index_name"]: i for i in db["unused_indexes"]}
    assert by["idx_a"]["zero_scans_since_epoch"] == FIRST
    assert by["idx_b"]["zero_scans_since_epoch"] == FIRST + 86400
    validate_report(report)


@pytest.mark.unit
def test_h002_restart_inside_the_window_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _responses(pgwatch_db_stats_postmaster_uptime_s=[{"metric": {}, "value": [NOW, str(86400)]}])
    w = _db(_gen(monkeypatch, resp).generate_h002_unused_indexes_report("local", "node-1"))["scan_window"]
    assert w["restart_in_window"] is True


@pytest.mark.unit
def test_h002_table_writes_over_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses())
    db = _db(gen.generate_h002_unused_indexes_report("local", "node-1"))
    writes = {(t["schema_name"], t["table_name"]): t for t in db["table_writes"]}
    acc = writes[("public", "accounts")]
    assert acc["n_tup_upd"] == 1000 and acc["n_tup_hot_upd"] == 250 and acc["n_tup_ins"] == 40
    assert acc["hot_share_pct"] == 25.0
    assert writes[("public", "orders")]["hot_share_pct"] is None  # no updates: no share
    assert any("[72h]" in q for q in gen.seen if "increase(pgwatch_table_stats_n_tup_upd" in q)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_h002_restart_before_the_window_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A restart a minute before the first reading: the report's clock alone decides it.
    resp = _responses(pgwatch_db_stats_postmaster_uptime_s=[{"metric": {}, "value": [NOW, str(3 * 86400 + 60)]}])
    w = _db(_gen(monkeypatch, resp).generate_h002_unused_indexes_report("local", "node-1"))["scan_window"]
    assert w["restart_in_window"] is False


@pytest.mark.unit
def test_h002_table_writes_for_quoted_table_names(monkeypatch: pytest.MonkeyPatch) -> None:
    # table_stats labels are quote_ident()ed; unused_indexes labels are raw names.
    resp = _responses(**{
        "pgwatch_unused_indexes_index_size_bytes": [{"metric": _idx("idx_u", table="User"), "value": [NOW, "2097152"]}],
        "increase(pgwatch_table_stats_n_tup_upd{": [
            {"metric": {"schema": "public", "table_name": '"User"'}, "value": [NOW, "10"]}],
        "increase(pgwatch_table_stats_n_tup_hot_upd{": [
            {"metric": {"schema": "public", "table_name": '"User"'}, "value": [NOW, "5"]}],
        "increase(pgwatch_table_stats_n_tup_ins{": [
            {"metric": {"schema": "public", "table_name": '"User"'}, "value": [NOW, "3"]}],
    })
    db = _db(_gen(monkeypatch, resp).generate_h002_unused_indexes_report("local", "node-1"))
    assert [(t["table_name"], t["n_tup_upd"], t["n_tup_hot_upd"], t["n_tup_ins"]) for t in db["table_writes"]] == [
        ("User", 10, 5, 3)]
