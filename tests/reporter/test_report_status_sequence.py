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
UPLOAD = {"report_chunck_id": 1}

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


def status(value: str, reason: str | None = None) -> tuple[str, dict[str, Any]]:
    """The status call; a failed report says why (platform-all !960)."""
    data: dict[str, Any] = {"access_token": TOKEN, "report_id": REPORT_ID, "status": value}
    if reason is not None:
        data["reason"] = reason
    return ("/rpc/checkup_report_status_update", data)


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
            status_error: Exception | None = None,
            reason_unsupported: bool = False,
            upload_response=UPLOAD) -> list[tuple[str, dict[str, Any]]]:
        def fake_make_request(api_url: str, endpoint: str, data: dict[str, Any]) -> Any:
            assert api_url == API
            if endpoint == "/rpc/checkup_report_file_post":
                calls.append((endpoint, {"checkup_report_id": data["checkup_report_id"],
                                         "filename": data["filename"]}))
                if data["filename"] == fail_upload_of:
                    raise _http_error(500)
                if isinstance(upload_response, Exception):
                    raise upload_response
                return upload_response
            calls.append((endpoint, dict(data)))
            if endpoint == "/rpc/checkup_report_create":
                return {"report_id": REPORT_ID}
            if status_error is not None:
                raise status_error
            if reason_unsupported and "reason" in data:
                raise _http_error(404)
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
@pytest.mark.parametrize("ack", [
    {"report_chunck_id": 1},
    {"report_chunk_id": 2},
    {"report_chunck_id": None, "report_chunk_id": 2},
    {"report_chunck_id": "3"},
])
def test_all_uploads_ok_marks_report_completed(run_main, ack) -> None:
    calls = run_main(ARGS_ALL, reports={"A002": {"checkId": "A002"}, "H001": {"checkId": "H001"}},
                     upload_response=ack)
    assert calls == [
        CREATE,
        file_post("A002.json"),
        file_post("H001.json"),
        status("completed"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("ack", [
    {}, None,
    requests.exceptions.JSONDecodeError("Invalid JSON", "not JSON", 0),
    {"report_chunck_id": "invalid"},
    {"report_chunck_id": {}},
    {"report_chunck_id": True},
    {"report_chunck_id": 1.5},
    {"report_chunck_id": 0},
    {"report_chunck_id": None},
])
def test_unconfirmed_upload_marks_report_failed(run_main, ack) -> None:
    calls = run_main(ARGS_ALL, upload_response=ack)
    assert calls == [
        CREATE,
        file_post("A002.json"),
        status("failed", "1 file(s) failed to upload: A002.json"),
    ]


@pytest.mark.unit
def test_failed_upload_marks_report_failed(run_main) -> None:
    calls = run_main(ARGS_ALL, reports={"A002": {"checkId": "A002"}, "H001": {"checkId": "H001"}},
                     fail_upload_of="A002.json")
    assert calls == [
        CREATE,
        file_post("A002.json"),
        file_post("H001.json"),
        status("failed", "1 file(s) failed to upload: A002.json"),
    ]


@pytest.mark.unit
def test_generation_error_marks_report_failed_and_still_raises(run_main) -> None:
    with pytest.raises(RuntimeError, match="prometheus went away"):
        run_main(ARGS_ALL, generation_error=RuntimeError("prometheus went away"))
    assert run_main.calls == [CREATE, status("failed", "the reporter raised: prometheus went away")]


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


@pytest.mark.unit
def test_single_check_upload_crash_marks_report_failed_and_still_raises(run_main, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(self, *_a, **_k):
        raise PermissionError("out.json is not readable")

    monkeypatch.setattr(PostgresReportGenerator, "upload_report_file", unreadable)
    argv = ARGS_ALL + ["--check-id", "A002", "--output", "out.json"]
    with pytest.raises(PermissionError):
        run_main(argv)
    assert run_main.calls == [CREATE, status("failed", "the reporter raised: out.json is not readable")]


@pytest.mark.unit
def test_platform_without_reason_still_gets_the_status(run_main) -> None:
    """An API older than the reason parameter (404 for that call) still gets `failed`."""
    calls = run_main(ARGS_ALL, reports={"A002": {"checkId": "A002"}}, fail_upload_of="A002.json",
                     reason_unsupported=True)
    assert calls == [
        CREATE,
        file_post("A002.json"),
        status("failed", "1 file(s) failed to upload: A002.json"),
        status("failed"),
    ]
