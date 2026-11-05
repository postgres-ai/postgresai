"""H001: the facts an invalid-index decision rests on, as the box's metric store has them.

- the definition comes from the metric's own index_definition label when the sink has none;
- is_ready (pg_index.indisready) and build_in_progress come from their own series, and are
  left out when the box does not collect them (an older box), never defaulted to false;
- each index's first and last monitored reading in the lookback, and how many hourly steps
  listed it against how many had a database-size reading (protocols
  checkup-acceptance-criteria/H001: H001-O2, one reading alone is not a leftover);
- the database's NOT VALID constraints (H001-O4), with "not collected" (no `$total$`
  series) kept apart from "none".
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

NOW = 1_793_660_000.0
FIRST = NOW - 6 * 86400
START = NOW - 7 * 86400
DEF = "CREATE UNIQUE INDEX accounts_status_uniq_bad ON demo.accounts USING btree (status)"


def _index(**extra: str) -> Dict[str, str]:
    labels = {"datname": "postgres", "schema_name": "demo", "table_name": "accounts",
              "index_name": "accounts_status_uniq_bad", "relation_name": "demo.accounts",
              "index_definition": DEF}
    labels.update(extra)
    return labels


def _row(labels: Dict[str, str], value: Any) -> Dict[str, Any]:
    return {"metric": labels, "value": [NOW, str(value)]}


def _constraint(schema: str, name: str, ctype: str = "c", definition: str = "CHECK ((x > 0)) NOT VALID") -> Dict[str, str]:
    return {"datname": "postgres", "schema_name": schema, "table_name": "t", "constraint_name": name,
            "constraint_type": ctype, "constraint_definition": definition}


TOTAL = {"datname": "postgres", "schema_name": "$total$", "table_name": "$total$", "constraint_name": "$total$",
         "constraint_type": "$total$", "constraint_definition": "$total$"}


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
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["postgres"])
    monkeypatch.setattr(gen, "get_index_definitions_from_sink", lambda db: {})
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: {
        "version": "17.11", "server_version_num": "170011", "server_major_ver": "17", "server_minor_ver": "11"})
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: NOW)
    gen.seen = seen  # type: ignore[attr-defined]
    return gen


def _responses(**over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        # order matters: the first needle found in the query answers it
        "min by (schema_name, table_name, index_name) (min_over_time(timestamp(pgwatch_pg_invalid_indexes":
            [_row(_index(), FIRST)],
        "max by (schema_name, table_name, index_name) (max_over_time(timestamp(pgwatch_pg_invalid_indexes":
            [_row(_index(), NOW - 1800)],
        "count_over_time((max by (schema_name, table_name, index_name) (max_over_time(pgwatch_pg_invalid_indexes":
            [_row(_index(), 144)],
        "count_over_time((max(max_over_time(pgwatch_db_size_size_b": [_row({}, 144)],
        "min(min_over_time(timestamp(pgwatch_db_size_size_b": [_row({}, START)],
        "pgwatch_db_size_size_b": [_row({"datname": "postgres"}, 70530739)],
        "pgwatch_pg_invalid_indexes_index_size_bytes": [_row(_index(), 0)],
        "pgwatch_pg_invalid_indexes_is_unique": [_row(_index(), 1)],
        "pgwatch_pg_invalid_indexes_table_row_estimate": [_row(_index(), 20000)],
        "pgwatch_pg_invalid_indexes_is_ready": [_row(_index(), 0)],
        "pgwatch_pg_invalid_indexes_build_in_progress": [_row(_index(), 0)],
        "pgwatch_pg_not_valid_constraints_not_valid": [
            _row(_constraint("app", "orders_amount_positive"), 1),
            _row(_constraint("app", "items_order_fk", "f", "FOREIGN KEY (order_id) REFERENCES app.orders(id) NOT VALID"), 1),
            _row(TOTAL, 2)],
        "pgwatch_pg_not_valid_constraints_is_partition_copy": [
            _row(_constraint("app", "orders_amount_positive"), 0),
            _row(_constraint("app", "items_order_fk", "f", "FOREIGN KEY (order_id) REFERENCES app.orders(id) NOT VALID"), 1),
            _row(TOTAL, 1)],
    }
    base.update(over)
    return base


def _entry(report: Dict[str, Any]) -> Dict[str, Any]:
    return next(iter(report["results"].values()))["data"]["postgres"]


def test_definition_readiness_and_persistence_from_the_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses())
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    validate_report(report)
    idx = _entry(report)["invalid_indexes"][0]
    assert idx["index_definition"] == DEF  # the label, not "Definition not available"
    assert idx["is_unique"] is True and idx["is_ready"] is False and idx["build_in_progress"] is False
    assert idx["first_seen_epoch"] == FIRST and idx["last_seen_epoch"] == NOW - 1800
    assert idx["first_seen_time"] == gen._h002_utc(FIRST)
    assert idx["listed_hours"] == 144 and idx["reading_hours"] == 144
    entry = _entry(report)
    assert entry["monitoring_start_epoch"] == START
    # both counts cover the same window, from the index's first sighting (never the whole lookback for one
    # and not the other: a listed count above the reading count read as a contradiction)
    windows = [q for q in gen.seen if "count_over_time(" in q]  # type: ignore[attr-defined]
    assert len(windows) == 2 and all("[144h:1h]" in q for q in windows)


def test_an_older_box_leaves_readiness_out(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{"pgwatch_pg_invalid_indexes_is_ready": [],
                                          "pgwatch_pg_invalid_indexes_build_in_progress": [],
                                          "pgwatch_pg_not_valid_constraints_not_valid": [],
                                          "pgwatch_pg_not_valid_constraints_is_partition_copy": []}))
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    validate_report(report)
    idx = _entry(report)["invalid_indexes"][0]
    assert "is_ready" not in idx and "build_in_progress" not in idx  # unknown is not "false"
    assert "not_valid_constraints" not in _entry(report)  # not collected is not "none"


def test_not_valid_constraints_with_partition_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses())
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    nv = _entry(report)["not_valid_constraints"]
    assert nv["total_count"] == 2
    by_name = {c["constraint_name"]: c for c in nv["constraints"]}
    assert by_name["orders_amount_positive"]["constraint_type"] == "c"
    assert by_name["orders_amount_positive"]["is_partition_copy"] is False
    assert by_name["items_order_fk"]["is_partition_copy"] is True
    assert "$total$" not in by_name


def test_none_collected_reads_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{"pgwatch_pg_not_valid_constraints_not_valid": [_row(TOTAL, 0)]}))
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    assert _entry(report)["not_valid_constraints"] == {"total_count": 0, "constraints": []}


def test_a_database_with_only_a_not_valid_constraint_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{"pgwatch_pg_invalid_indexes_index_size_bytes": []}))
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    validate_report(report)
    entry = _entry(report)
    assert entry["total_count"] == 0 and entry["not_valid_constraints"]["total_count"] == 2


def test_system_schema_constraints_are_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, _responses(**{"pgwatch_pg_not_valid_constraints_not_valid": [
        _row(_constraint("pg_catalog", "x"), 1), _row(TOTAL, 1)]}))
    report = gen.generate_h001_invalid_indexes_report(cluster="default", node_name="node-01")
    assert _entry(report)["not_valid_constraints"]["constraints"] == []


def test_pg18_not_enforced_constraints_are_not_listed() -> None:
    """PG 18 stores a NOT ENFORCED constraint with convalidated = false: a design, not a half-done change,
    and VALIDATE CONSTRAINT fails on it. The PG 18 SQL skips it; conenforced does not exist before 18."""
    import pathlib
    import yaml
    metrics = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / "config" / "pgwatch-prometheus"
                              / "metrics.yml").read_text())["metrics"]
    sqls = metrics["pg_not_valid_constraints"]["sqls"]
    assert "con.conenforced" in sqls[18] and "conenforced" not in sqls[11]
    import re
    # otherwise the same query: only the comment and the conenforced line differ
    assert re.sub(r"\n[ ]*-- PG 18:[^\n]*\n[ ]*and con\.conenforced", "", sqls[18]) == sqls[11]
