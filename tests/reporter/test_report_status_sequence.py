"""The reporter closes every report it creates (#400).

checkup_report_create leaves a report `pending`. After its uploads, the
reporter must call checkup_report_status_update: `completed` when every
file went up, `failed` when an upload failed or generation raised.

Each test runs main() against a recording make_request and compares the
whole request sequence with a golden list.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest
import requests

from reporter import postgres_reports
from reporter.postgres_reports import PostgresReportGenerator

API = "http://api.test"
TOKEN = "tok"
REPORT_ID = 77

ARGS_ALL = [
    "postgres_reports.py",
    "--prometheus-url", "http://prom.test",
    "--postgres-sink-url", "postgresql://u@h:5432/db",
    "--cluster", "c1",
    "--api-url", API,
    "--token", TOKEN,
    "--project-name", "proj",
]

CREATE = ("/rpc/checkup_report_create", {"access_token": TOKEN, "project": "proj", "epoch": "1"})


def file_post(name: str) -> tuple[str, dict[str, Any]]:
    return ("/rpc/checkup_report_file_post", {"checkup_report_id": REPORT_ID, "filename": name})


def status(value: str) -> tuple[str, dict[str, Any]]:
    return ("/rpc/checkup_report_status_update",
            {"access_token": TOKEN, "report_id": REPORT_ID, "status": value})


def _http_error(code: int) -> requests.exceptions.HTTPError:
    resp = requests.models.Response()
    resp.status_code = code
    return requests.exceptions.HTTPError(f"{code} Server Error", response=resp)


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """Run main() with stubbed generation; return the recorded requests."""
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, dict[str, Any]]] = []

    def run(argv: list[str], *, reports=None, fail_upload_of: str | None = None,
            generation_error: Exception | None = None,
            status_error: Exception | None = None) -> list[tuple[str, dict[str, Any]]]:
        def fake_make_request(api_url: str, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
            assert api_url == API
            if endpoint == "/rpc/checkup_report_file_post":
                calls.append((endpoint, {"checkup_report_id": data["checkup_report_id"],
                                         "filename": data["filename"]}))
                if data["filename"] == fail_upload_of:
                    raise _http_error(500)
                return {}
            calls.append((endpoint, dict(data)))
            if endpoint == "/rpc/checkup_report_create":
                return {"report_id": REPORT_ID}
            if status_error is not None:
                raise status_error
            return {"report_id": REPORT_ID, "status": data.get("status")}

        def fake_generate_all(self, *_a, **_k):
            if generation_error is not None:
                raise generation_error
            return dict(reports if reports is not None else {"A002": {"checkId": "A002"}})

        monkeypatch.setattr(postgres_reports, "make_request", fake_make_request)
        monkeypatch.setattr(PostgresReportGenerator, "test_connection", lambda self: True)
        monkeypatch.setattr(PostgresReportGenerator, "generate_all_reports", fake_generate_all)
        monkeypatch.setattr(PostgresReportGenerator, "generate_per_query_jsons",
                            lambda self, *a, **k: [])
        monkeypatch.setattr(PostgresReportGenerator, "generate_a002_version_report",
                            lambda self, *a, **k: {"checkId": "A002"})
        monkeypatch.setattr(sys, "argv", argv)
        postgres_reports.main()
        return calls

    run.calls = calls
    return run


@pytest.mark.unit
def test_all_uploads_ok_marks_report_completed(run_main) -> None:
    calls = run_main(ARGS_ALL, reports={"A002": {"checkId": "A002"}, "H001": {"checkId": "H001"}})
    assert calls == [
        CREATE,
        file_post("A002.json"),
        file_post("H001.json"),
        status("completed"),
    ]


@pytest.mark.unit
def test_failed_upload_marks_report_failed(run_main) -> None:
    calls = run_main(ARGS_ALL, reports={"A002": {"checkId": "A002"}, "H001": {"checkId": "H001"}},
                     fail_upload_of="A002.json")
    assert calls == [
        CREATE,
        file_post("A002.json"),
        file_post("H001.json"),
        status("failed"),
    ]


@pytest.mark.unit
def test_generation_error_marks_report_failed_and_still_raises(run_main) -> None:
    with pytest.raises(RuntimeError, match="prometheus went away"):
        run_main(ARGS_ALL, generation_error=RuntimeError("prometheus went away"))
    assert run_main.calls == [CREATE, status("failed")]


@pytest.mark.unit
def test_single_check_upload_marks_report_completed(run_main) -> None:
    argv = ARGS_ALL + ["--check-id", "A002", "--output", "out.json"]
    calls = run_main(argv)
    assert calls == [CREATE, file_post("out.json"), status("completed")]


@pytest.mark.unit
def test_status_update_failure_does_not_fail_the_run(run_main) -> None:
    """An API without the status RPC (404) must not turn a good run into a crash."""
    calls = run_main(ARGS_ALL, status_error=_http_error(404))
    assert calls == [CREATE, file_post("A002.json"), status("completed")]
