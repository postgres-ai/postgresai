"""First-run window regression ported from !446."""
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from reporter import postgres_reports
from reporter.postgres_reports import PostgresReportGenerator

PG = {"version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"}


@pytest.mark.parametrize("method", [
    "generate_k001_query_calls_report", "generate_k003_top_queries_report",
    "generate_k004_temp_bytes_report", "generate_k005_wal_bytes_report",
    "generate_k006_shared_read_report", "generate_k007_shared_hit_report",
    "generate_k008_shared_hit_read_report", "generate_m001_mean_time_report",
    "generate_m002_rows_report", "generate_m003_io_time_report",
    "generate_n001_wait_events_report",
])
@pytest.mark.unit
@pytest.mark.parametrize(("use_current_time", "has_data"), [(False, False), (True, True)])
def test_a_box_younger_than_the_hour_has_query_data_only_with_the_current_time(
    monkeypatch: pytest.MonkeyPatch, use_current_time: bool, has_data: bool, method: str
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
        if "wait_events" in query:
            return [{"metric": {"query_id": "4242", "wait_event_type": "CPU", "wait_event": "CPU"}, "values": values}]
        if "queryid" in query:
            return [{"metric": {"queryid": "4242"}, "values": values}]
        return [{"metric": {}, "values": values}]

    generator = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="", use_current_time=use_current_time)
    monkeypatch.setattr(postgres_reports, "time", SimpleNamespace(time=lambda: now))
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: PG)
    monkeypatch.setattr(generator, "query_range", query_range)

    monkeypatch.setattr(generator, "get_queryid_queries_from_sink", lambda *a, **k: {})
    monkeypatch.setattr(generator, "query_instant", lambda *a, **k: {"status": "success", "data": {"result": []}})
    kwargs = {"hours": 24} if method.startswith("generate_n001") else {"time_range_minutes": 1440}
    report = getattr(generator, method)("local", "node-1", **kwargs)
    node = report["results"]["node-1"]
    if has_data:
        data = node["data"]["db1"]
        summary = data["summary"]
        assert summary["hourly_timestamps"] == list(range(now - 23 * 3600, now + 1, 3600))
        assert summary["time_range_hours"] == 24
        if method.startswith("generate_k001"):
            assert [q["queryid"] for q in data["query_metrics"]] == ["4242"]
        if method.startswith("generate_k003"):
            assert datetime.fromisoformat(summary["window_end"]).timestamp() == now
            assert datetime.fromisoformat(summary["window_start"]).timestamp() == now - 24 * 3600
        if method.startswith("generate_n001"):
            assert datetime.fromisoformat(summary["end_time"]).timestamp() == now
            assert datetime.fromisoformat(summary["start_time"]).timestamp() == now - 24 * 3600
    else:
        assert node["data"] == {}
