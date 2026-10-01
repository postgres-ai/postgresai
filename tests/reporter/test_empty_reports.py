"""A report without data says so, and why (UAT 2026-10-01: 15 of 25 reports were `data: {}`)."""
from __future__ import annotations

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

PG = {"version": "17.11", "server_version_num": "170011", "server_major_ver": "17", "server_minor_ver": "11"}

QUERY_REASON = (
    "No query statistics in the window. The monitoring records a query from pg_stat_statements once it has "
    "run at least 3 times and for 1 s in total, so an idle or newly connected database has none yet."
)


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("check_id", "available", "reason"),
    [
        # Not collected: the report cannot say anything yet.
        ("K001", False, QUERY_REASON),
        ("M003", False, QUERY_REASON),
        ("N001", False, "No active-session samples in the window: no session was active when sampled, or the collection has not run yet."),
        # Collected, and nothing found: an answer, said plainly.
        ("A007", True, "No setting differs from its default."),
        ("H002", True, "No unused indexes found in the collected metrics."),
        ("F004", True, "No table bloat estimate in the collected metrics (tables of 1 MiB or more are estimated)."),
    ],
)
def test_empty_report_says_why(generator: PostgresReportGenerator, check_id: str, available: bool, reason: str) -> None:
    report = generator.format_report_data(check_id, {}, "node-1", postgres_version=PG)
    assert report["results"] == {"node-1": {"data": {}, "available": available, "reason": reason, "postgres_version": PG}}
    validate_report(report)


@pytest.mark.unit
def test_every_check_has_a_reason(generator: PostgresReportGenerator) -> None:
    for check_id in ["A002", "A003", "A004", "A007", "D004", "F001", "F002", "F004", "F005", "G001",
                     "H001", "H002", "H004", "I001", "K001", "K003", "K004", "K005", "K006", "K007",
                     "K008", "M001", "M002", "M003", "N001"]:
        node = generator.format_report_data(check_id, {}, "node-1")["results"]["node-1"]
        assert isinstance(node["available"], bool)
        assert node["reason"].startswith("No "), check_id


@pytest.mark.unit
def test_empty_node_in_a_multi_node_report(generator: PostgresReportGenerator) -> None:
    data = {"primary": {"data": {"db1": {"x": 1}}}, "replica": {"data": {}}}
    results = generator.format_report_data("K001", data, "primary")["results"]
    assert results["primary"] == {"data": {"db1": {"x": 1}}}
    assert results["replica"] == {"data": {}, "available": False, "reason": QUERY_REASON}


@pytest.mark.unit
def test_a_report_with_data_is_unchanged(generator: PostgresReportGenerator) -> None:
    report = generator.format_report_data("K001", {"db1": {"query_metrics": []}}, "node-1")
    assert report["results"] == {"node-1": {"data": {"db1": {"query_metrics": []}}}}


@pytest.mark.unit
def test_k001_with_no_query_metrics_carries_the_reason(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: PG)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["postgres"])
    monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data", lambda *args, **kwargs: ({}, [0.0] * 24, []))
    report = generator.generate_k001_query_calls_report("local", "node-1", time_range_minutes=1440)
    node = report["results"]["node-1"]
    assert (node["available"], node["reason"]) == (False, QUERY_REASON)
    validate_report(report)
