from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


@pytest.fixture(name="fixed_pg_version")
def fixture_fixed_pg_version() -> dict[str, str]:
    return {
        "version": "15.3",
        "server_version_num": "150003",
        "server_major_ver": "15",
        "server_minor_ver": "3",
    }


def _stub_hourly_topk_single_metric(
    metric_name_to_data: dict[str, tuple[dict[str, list[float]], list[float], list[int]]]
):
    def _stub(
        cluster: str,
        node_name: str,
        db_name: str,
        metric_name: str = "pgwatch_pg_stat_statements_calls",
        hours: int = 24,
        step_s: int = 3600,
        k: int = 3,
    ):
        _ = (cluster, node_name, db_name, hours, step_s, k)
        if metric_name not in metric_name_to_data:
            raise AssertionError(f"Unexpected metric_name: {metric_name}")
        return metric_name_to_data[metric_name]

    return _stub


@pytest.mark.unit
def test_generate_k004_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [1.0, 2.0], "2": [0.0, 4.0]}
    other = [10.0, 0.0]
    timeline = [100, 200]
    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric(
            {"pgwatch_pg_stat_statements_temp_bytes_written": (per_query, other, timeline)}
        ),
    )

    report = generator.generate_k004_temp_bytes_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    assert db["summary"]["total_temp_bytes_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_temp_bytes_tracked_queries"] == pytest.approx(3.0 + 4.0)
    assert db["summary"]["total_temp_bytes"] == pytest.approx((3.0 + 4.0) + sum(other))


@pytest.mark.unit
def test_generate_k005_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [5.0, 5.0]}
    other = [1.0, 2.0]
    timeline = [100, 200]
    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric({"pgwatch_pg_stat_statements_wal_bytes": (per_query, other, timeline)}),
    )

    report = generator.generate_k005_wal_bytes_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    assert db["summary"]["total_wal_bytes_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_wal_bytes_tracked_queries"] == pytest.approx(10.0)
    assert db["summary"]["total_wal_bytes"] == pytest.approx(10.0 + sum(other))


@pytest.mark.unit
def test_generate_k006_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [1.0, 1.0], "2": [0.0, 3.0]}
    other = [0.0, 10.0]
    timeline = [100, 200]
    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric(
            {"pgwatch_pg_stat_statements_shared_bytes_read_total": (per_query, other, timeline)}
        ),
    )

    report = generator.generate_k006_shared_read_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    assert db["summary"]["total_shared_read_bytes_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_shared_read_bytes_tracked_queries"] == pytest.approx(2.0 + 3.0)
    assert db["summary"]["total_shared_read_bytes"] == pytest.approx((2.0 + 3.0) + sum(other))


@pytest.mark.unit
def test_generate_k007_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [2.0, 0.0]}
    other = [2.0, 2.0]
    timeline = [100, 200]
    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric(
            {"pgwatch_pg_stat_statements_shared_bytes_hit_total": (per_query, other, timeline)}
        ),
    )

    report = generator.generate_k007_shared_hit_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    assert db["summary"]["total_shared_hit_bytes_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_shared_hit_bytes_tracked_queries"] == pytest.approx(2.0)
    assert db["summary"]["total_shared_hit_bytes"] == pytest.approx(2.0 + sum(other))


@pytest.mark.unit
def test_generate_k008_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [6.0, 1.0], "2": [0.0, 3.0], "3": [2.0, 2.0]}
    other = [1.0, 10.0]
    timeline = [100, 200]

    monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data_sum2", lambda *args, **kwargs: (per_query, other, timeline))

    report = generator.generate_k008_shared_hit_read_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    tracked = sum(sum(v) for v in per_query.values())
    assert db["summary"]["total_shared_hit_read_bytes_tracked_queries"] == pytest.approx(tracked)
    assert db["summary"]["total_shared_hit_read_bytes_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_shared_hit_read_bytes"] == pytest.approx(tracked + sum(other))


@pytest.mark.unit
def test_generate_m001_computes_mean(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    timeline = [100, 200]
    time_per_query = {"1": [100.0, 100.0], "2": [10.0, 0.0]}
    calls_per_query = {"1": [1.0, 1.0], "2": [2.0, 0.0]}
    time_other = [0.0, 0.0]
    calls_other = [0.0, 0.0]

    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric(
            {
                "pgwatch_pg_stat_statements_exec_time_total": (time_per_query, time_other, timeline),
                "pgwatch_pg_stat_statements_calls": (calls_per_query, calls_other, timeline),
            }
        ),
    )

    report = generator.generate_m001_mean_time_report("local", "node-1", time_range_minutes=120, limit=50)
    top = report["results"]["node-1"]["data"]["db1"]["top_queries"]

    # query 1 mean: 200/2 = 100; query 2 mean: 10/2 = 5
    assert top[0]["queryid"] == "1"
    assert top[0]["mean_time_ms"] == pytest.approx(100.0)
    assert top[1]["queryid"] == "2"
    assert top[1]["mean_time_ms"] == pytest.approx(5.0)


@pytest.mark.unit
def test_generate_m002_computes_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    per_query = {"1": [1.0, 2.0], "2": [3.0, 0.0]}
    other = [10.0, 0.0]
    timeline = [100, 200]
    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric({"pgwatch_pg_stat_statements_rows": (per_query, other, timeline)}),
    )

    report = generator.generate_m002_rows_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    tracked = (1.0 + 2.0) + (3.0 + 0.0)
    assert db["summary"]["total_rows_tracked_queries"] == pytest.approx(tracked)
    assert db["summary"]["total_rows_other"] == pytest.approx(sum(other))
    assert db["summary"]["total_rows"] == pytest.approx(tracked + sum(other))


@pytest.mark.unit
def test_generate_m003_computes_io_totals(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    timeline = [100, 200]
    read_per_query = {"1": [1.0, 2.0]}
    write_per_query = {"1": [3.0, 4.0]}
    read_other = [0.0, 10.0]
    write_other = [1.0, 1.0]

    monkeypatch.setattr(
        generator,
        "_get_hourly_topk_pgss_data",
        _stub_hourly_topk_single_metric(
            {
                "pgwatch_pg_stat_statements_block_read_total": (read_per_query, read_other, timeline),
                "pgwatch_pg_stat_statements_block_write_total": (write_per_query, write_other, timeline),
            }
        ),
    )

    report = generator.generate_m003_io_time_report("local", "node-1", time_range_minutes=120, limit=50)
    db = report["results"]["node-1"]["data"]["db1"]

    assert db["top_queries"][0]["total_io_time_ms"] == pytest.approx((1.0 + 2.0) + (3.0 + 4.0))
    assert db["summary"]["total_io_time_other_ms"] == pytest.approx(sum([r + w for r, w in zip(read_other, write_other)]))


@pytest.mark.unit
def test_generate_n001_groups_wait_events(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    # Fix timeline deterministically: end_s=7200 for hours=3 -> [0,3600,7200]
    monkeypatch.setattr(generator, "_floor_hour", lambda *_: 7200)

    captured: dict[str, Any] = {}

    def fake_query_range(query: str, start, end, step: str = "3600s") -> list[dict[str, Any]]:
        captured.update(query=query, start=start, end=end, step=step)
        # Values are already aggregated sampled active-session ticks per hour.
        # Missing timestamps must remain zero in the initialized output arrays.
        return [
            {
                "metric": {
                    "wait_event_type": "IO",
                    "wait_event": "DataFileRead",
                    "query_id": "123",
                },
                "values": [[0, "3"], [7200, "2"]],
            },
            {
                "metric": {
                    "wait_event_type": "IO",
                    "wait_event": "DataFileRead",
                    "query_id": "456",
                },
                "values": [[3600, "4"]],
            },
        ]

    monkeypatch.setattr(generator, "query_range", fake_query_range)

    report = generator.generate_n001_wait_events_report("local", "node-1", hours=3)
    db = report["results"]["node-1"]["data"]["db1"]
    io = db["wait_event_types"]["IO"]

    query = captured["query"]
    assert query.startswith("sum by (wait_event_type, wait_event, query_id) (")
    assert 'sum_over_time(pgwatch_wait_events_total{cluster="local",node_name="node-1",datname="db1"}[1h])' in query
    assert captured["step"] == "3600s"
    assert int(captured["start"].timestamp()) == 0
    assert int(captured["end"].timestamp()) == 7200

    summary = db["summary"]
    assert int(datetime.fromisoformat(summary["start_time"]).timestamp()) == -3600
    assert int(datetime.fromisoformat(summary["end_time"]).timestamp()) == 7200
    assert summary["time_range_hours"] == 3
    assert summary["hourly_timestamps"] == [0, 3600, 7200]
    assert summary["sampling_semantics"] == "sampled_active_session_ticks"
    assert "collection cadence" in summary["sampling_note"]
    assert "missed scrapes" in summary["sampling_note"]

    assert io["unique_queries"] == 2
    assert io["total_occurrences"] == 9
    # Sorted by sampled ticks desc: q123 has 5, q456 has 4.
    assert io["queries_list"][0]["query_id"] == "123"
    assert io["queries_list"][0]["hourly_occurrences"] == [3, 0, 2]
    assert io["queries_list"][1]["query_id"] == "456"
    assert io["queries_list"][1]["hourly_occurrences"] == [0, 4, 0]
