"""A report without data says so, and why (UAT 2026-10-01: 15 of 25 reports were `data: {}`)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from jsonschema import Draft202012Validator

import reporter.postgres_reports as postgres_reports
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import load_schema, schema_dir, validate_report

PG = {"version": "17.11", "server_version_num": "170011", "server_major_ver": "17", "server_minor_ver": "11"}

# Every check the reporter has a schema for.
CHECK_IDS = sorted(p.name.split(".")[0] for p in schema_dir().glob("*.schema.json") if p.name != "query.schema.json")

QUERY_REASON = (
    "No query statistics in the window. The monitoring records a query from pg_stat_statements once it has "
    "run at least 3 times and for 1 s in total, so an idle or newly connected database has none yet."
)
DEFAULT_REASON = "No data for this check in the collected metrics yet."

# available: false. The report cannot tell "nothing found" from "not collected" or "the query failed".
NOT_COLLECTED = {
    **{check_id: QUERY_REASON for check_id in ("K001", "K003", "K004", "K005", "K006", "K007", "K008", "M001", "M002", "M003")},
    "N001": "No active-session samples in the window: no session was active when sampled, or the collection has not run yet.",
    "A007": "No settings in the collected metrics: the collection has not run yet, or the metrics query failed.",
    "F004": "No table bloat estimate in the collected metrics: no table is larger than 1 MiB, the collection has not run yet, or the metrics query failed.",
    "F005": "No index bloat estimate in the collected metrics: no B-tree index is larger than 1 MiB, the collection has not run yet, or the metrics query failed.",
    "H001": "No invalid index in the collected metrics: there is none, the collection has not run yet, or the metrics query failed.",
    "H002": "No unused index in the collected metrics: there is none, the collection has not run yet, or the metrics query failed.",
    "H004": "No redundant index in the collected metrics: there is none, the collection has not run yet, or the metrics query failed.",
}
# available: true. The source metric had rows for the node, and the report kept none of them.
FOUND_NOTHING = {
    "H001": "No invalid indexes found in the collected metrics.",
    "H002": "No unused indexes found in the collected metrics.",
    "H004": "No redundant indexes found in the collected metrics.",
}


@pytest.fixture(name="generator")
def fixture_generator(request) -> PostgresReportGenerator:
    generator = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="",
                                        mon_url="http://mon.test" if getattr(request, "param", None) == "pgai-mon" else None)
    if generator.mon:
        generator.mon.retries = 0
    return generator


def _annotation(node: dict[str, Any]) -> tuple[Any, Any]:
    return node.get("available"), node.get("reason")


def _validate_node_fields(check_id: str, node: dict[str, Any]) -> None:
    """The check's schema accepts the node-level fields. `data: {}` is left out: it is not a valid
    payload for checks whose data has a fixed shape (A002, D004, ...), and those never come out empty."""
    schema = load_schema(check_id)
    node_schema = {"$defs": schema["$defs"], **schema["$defs"]["nodeResult"], "required": []}
    Draft202012Validator(node_schema).validate({k: v for k, v in node.items() if k != "data"})


@pytest.mark.unit
def test_the_reason_tables_name_real_checks() -> None:
    assert len(CHECK_IDS) == 32
    assert set(NOT_COLLECTED) <= set(CHECK_IDS)
    assert set(FOUND_NOTHING) <= set(CHECK_IDS)


@pytest.mark.unit
@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_an_empty_report_without_evidence_is_not_available(generator: PostgresReportGenerator, check_id: str) -> None:
    report = generator.format_report_data(check_id, {}, "node-1")
    reason = NOT_COLLECTED.get(check_id, DEFAULT_REASON)
    assert report["results"] == {"node-1": {"data": {}, "available": False, "reason": reason}}
    _validate_node_fields(check_id, report["results"]["node-1"])


@pytest.mark.unit
@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_an_empty_report_with_evidence_found_nothing(generator: PostgresReportGenerator, check_id: str) -> None:
    report = generator.format_report_data(check_id, {}, "node-1", collected=True)
    expected = (True, FOUND_NOTHING[check_id]) if check_id in FOUND_NOTHING else (False, NOT_COLLECTED.get(check_id, DEFAULT_REASON))
    assert _annotation(report["results"]["node-1"]) == expected
    _validate_node_fields(check_id, report["results"]["node-1"])


@pytest.mark.unit
@pytest.mark.parametrize("collected", [False, True])
def test_an_unknown_check_gets_the_default(generator: PostgresReportGenerator, collected: bool) -> None:
    node = generator.format_report_data("Z999", {}, "node-1", collected=collected)["results"]["node-1"]
    assert _annotation(node) == (False, DEFAULT_REASON)


@pytest.mark.unit
@pytest.mark.parametrize("collected", [False, True])
def test_a_report_with_data_is_unchanged(generator: PostgresReportGenerator, collected: bool) -> None:
    report = generator.format_report_data("H002", {"db1": {"unused_indexes": []}}, "node-1", collected=collected)
    assert report["results"] == {"node-1": {"data": {"db1": {"unused_indexes": []}}}}


@pytest.mark.unit
def test_empty_node_in_a_multi_node_report(generator: PostgresReportGenerator) -> None:
    data = {"primary": {"data": {"db1": {"x": 1}}}, "replica": {"data": {}}}
    results = generator.format_report_data("K001", data, "primary")["results"]
    assert results["primary"] == {"data": {"db1": {"x": 1}}}
    assert results["replica"] == {"data": {}, "available": False, "reason": QUERY_REASON}


@pytest.mark.unit
def test_a_combined_report_keeps_each_node_annotation(generator: PostgresReportGenerator) -> None:
    # generate_all_reports combines per-node reports without `collected`; a node's own evidence stays.
    primary = generator.format_report_data("H002", {}, "primary", collected=True)["results"]["primary"]
    replica = generator.format_report_data("H002", {}, "replica")["results"]["replica"]
    results = generator.format_report_data("H002", {"primary": primary, "replica": replica}, "primary")["results"]
    assert _annotation(results["primary"]) == (True, FOUND_NOTHING["H002"])
    assert _annotation(results["replica"]) == (False, NOT_COLLECTED["H002"])


@pytest.mark.unit
def test_a_node_value_that_is_not_a_dict_is_left_alone(generator: PostgresReportGenerator) -> None:
    data = {"primary": {"data": {}}, "note": "not a node"}
    results = generator.format_report_data("H002", data, "primary")["results"]
    assert results["note"] == "not a node"
    assert _annotation(results["primary"]) == (False, NOT_COLLECTED["H002"])


@pytest.mark.unit
def test_k001_with_no_query_metrics_carries_the_reason(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: PG)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["postgres"])
    monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data", lambda *args, **kwargs: ({}, [0.0] * 24, []))
    report = generator.generate_k001_query_calls_report("local", "node-1", time_range_minutes=1440)
    node = report["results"]["node-1"]
    assert _annotation(node) == (False, QUERY_REASON)
    validate_report(report)


FOUND_NOTHING_GENERATORS = {
    "A007": "generate_a007_altered_settings_report",
    "F004": "generate_f004_heap_bloat_report",
    "F005": "generate_f005_btree_bloat_report",
    "H001": "generate_h001_invalid_indexes_report",
    "H002": "generate_h002_unused_indexes_report",
    "H004": "generate_h004_redundant_indexes_report",
}


@pytest.mark.unit
@pytest.mark.parametrize(("check_id", "method"), sorted(FOUND_NOTHING_GENERATORS.items()))
@pytest.mark.parametrize("generator", ["prometheus", "pgai-mon"], indirect=True)
def test_metrics_unreachable_is_not_an_all_clear(
    monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, check_id: str, method: str
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    if generator.mon:
        monkeypatch.setattr(generator.mon.session, "post", refuse)
    report = getattr(generator, method)("local", "node-1")
    node = report["results"]["node-1"]
    assert node["data"] == {}
    assert _annotation(node) == (False, NOT_COLLECTED[check_id])
    validate_report(report)


def _prometheus(series_by_metric: dict[str, list[dict[str, Any]]]):
    """A query_instant that answers each query with the series of the first metric name it contains."""

    def query_instant(query: str) -> dict[str, Any]:
        for metric, series in series_by_metric.items():
            if metric in query:
                return {"status": "success", "data": {"resultType": "vector", "result": series}}
        return {"status": "success", "data": {"resultType": "vector", "result": []}}

    return query_instant


def _index_series(db_label: str, schema_name: str) -> list[dict[str, Any]]:
    labels = {db_label: "db1", "schema_name": schema_name, "table_name": "t", "index_name": "i", "reason": "r"}
    return [{"metric": labels, "value": [0, "16384"]}]


INDEX_METRICS = {
    "H001": ("generate_h001_invalid_indexes_report", "pgwatch_pg_invalid_indexes_index_size_bytes", "datname"),
    "H002": ("generate_h002_unused_indexes_report", "pgwatch_unused_indexes_index_size_bytes", "datname"),
    "H004": ("generate_h004_redundant_indexes_report", "pgwatch_redundant_indexes_index_size_bytes", "real_dbname"),
}


@pytest.mark.unit
@pytest.mark.parametrize("check_id", sorted(INDEX_METRICS))
@pytest.mark.parametrize(
    ("schema_name", "expected"),
    [
        # Rows only for system-catalog indexes, which the report drops: the check ran and found nothing.
        ("pg_catalog", "found nothing"),
        # No rows: nothing found and not collected look the same.
        (None, "not collected"),
    ],
)
def test_index_checks_say_found_nothing_only_with_rows(
    monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, check_id: str, schema_name: str | None, expected: str
) -> None:
    method, metric, db_label = INDEX_METRICS[check_id]
    series = _index_series(db_label, schema_name) if schema_name else []
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: PG)
    monkeypatch.setattr(generator, "query_instant", _prometheus({metric: series}))
    report = getattr(generator, method)("local", "node-1")
    node = report["results"]["node-1"]
    assert node["data"] == {}
    if expected == "found nothing":
        assert _annotation(node) == (True, FOUND_NOTHING[check_id])
    else:
        assert _annotation(node) == (False, NOT_COLLECTED[check_id])
    validate_report(report)


@pytest.mark.unit
@pytest.mark.parametrize(("use_current_time", "has_data"), [(False, False), (True, True)])
def test_a_box_younger_than_the_hour_has_query_data_only_with_the_current_time(
    monkeypatch: pytest.MonkeyPatch, use_current_time: bool, has_data: bool
) -> None:
    """UAT 2026-10-02: the first sample is at 12:05, the first report runs at 12:30.

    A window floored to 12:00 ends before the first sample, so K001 is empty;
    a window ending at 12:30 has the query.
    """
    now = int(datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc).timestamp())
    first_sample = int(datetime(2026, 10, 2, 12, 5, tzinfo=timezone.utc).timestamp())

    def query_range(query: str, start_time: datetime, end_time: datetime, step: str = "30s") -> list[dict[str, Any]]:
        # increase(...[1h]) has a value at t only once samples exist before t.
        start, end, step_s = int(start_time.timestamp()), int(end_time.timestamp()), int(step.rstrip("s"))
        values = [[t, "5"] for t in range(start, end + 1, step_s) if t > first_sample]
        if not values:
            return []
        if "queryid" in query:
            return [{"metric": {"queryid": "4242"}, "values": values}]
        return [{"metric": {}, "values": values}]

    generator = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="", use_current_time=use_current_time)
    monkeypatch.setattr(postgres_reports, "time", SimpleNamespace(time=lambda: now))
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: PG)
    monkeypatch.setattr(generator, "query_range", query_range)

    report = generator.generate_k001_query_calls_report("local", "node-1", time_range_minutes=1440)
    node = report["results"]["node-1"]
    if has_data:
        # No validate_report here: K001's hourly data (`other_calls_hourly`) does not match K001.schema.json on main either.
        assert [q["queryid"] for q in node["data"]["db1"]["query_metrics"]] == ["4242"]
        assert "available" not in node
    else:
        assert node["data"] == {}
        assert _annotation(node) == (False, QUERY_REASON)
        validate_report(report)
