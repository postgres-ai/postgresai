from __future__ import annotations

import json
from typing import Any

import pytest

from reporter import postgres_reports as postgres_reports_module
from reporter.postgres_reports import PostgresReportGenerator


@pytest.mark.unit
def test_upload_report_file_extracts_check_id_from_json(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")

    report_path = tmp_path / "cluster_A002.json"
    report_payload = {
        "checkId": "A002",
        "checkTitle": "Postgres major version",
        "timestamptz": "2025-01-01T00:00:00+00:00",
        "nodes": {"primary": "node-1", "standbys": []},
        "results": {"node-1": {"data": {}}},
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_make_request(_api_url: str, _endpoint: str, request_data: dict[str, Any]) -> dict[str, Any]:
        captured["request_data"] = request_data
        return {"report_chunck_id": 1}

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    generator.upload_report_file("http://api.test", "tok", 123, str(report_path))

    req = captured["request_data"]
    assert req["check_id"] == "A002"
    assert req["generate_issue"] is True


@pytest.mark.unit
def test_upload_report_file_query_json_has_no_check_id(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")

    query_path = tmp_path / "prod_query_123.json"
    query_payload = {
        "cluster_id": "prod",
        "query_id": "123",
        "query_text": "select 1",
        "nodes": {"primary": "main", "standbys": ["replica-1"]},
        "results": {"main": {"db1": {"metrics": {"calls": 1}}}},
        "timestamptz": "2025-01-01T00:00:00+00:00",
    }
    query_path.write_text(json.dumps(query_payload), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_make_request(_api_url: str, _endpoint: str, request_data: dict[str, Any]) -> dict[str, Any]:
        captured["request_data"] = request_data
        return {"report_chunck_id": 1}

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    generator.upload_report_file("http://api.test", "tok", 123, str(query_path))

    req = captured["request_data"]
    assert req["check_id"] == ""
    assert req["generate_issue"] is False


