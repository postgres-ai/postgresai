"""A014 Resource sizing: usage measurements for the platform's verdict (platform-all#1051)."""
from __future__ import annotations

import time
from typing import Any

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

PG_VERSION = {"version": "15.3", "server_version_num": "150003", "server_major_ver": "15", "server_minor_ver": "3"}


def _gen(monkeypatch: pytest.MonkeyPatch, ranges: dict[str, list[float]], instants: dict[str, Any]) -> PostgresReportGenerator:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG_VERSION)
    now = int(time.time()) // 300 * 300

    def fake_range(query: str, start, end, step: str = "30s") -> list[dict[str, Any]]:
        for needle, values in ranges.items():
            if needle in query:
                # oldest first, one point per 5 minutes, ending now
                return [{"metric": {}, "values": [[now - 300 * (len(values) - 1 - i), str(v)]
                                                  for i, v in enumerate(values)]}]
        return []

    def fake_instant(query: str) -> dict[str, Any]:
        for needle, payload in instants.items():
            if needle in query:
                return {"status": "success", "data": {"result": [payload]}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_range", fake_range)
    monkeypatch.setattr(gen, "query_instant", fake_instant)
    return gen


@pytest.mark.unit
def test_a014_with_host_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = [float(v) for v in range(1, 101)]  # 100 points: p95 = 95.05, max 100
    gen = _gen(monkeypatch, {
        "host_cpu_utilization_percent": cpu,
        "host_memory_available_bytes": [50.0] * 99 + [7.0],
        "host_disk_read_bytes_per_second": [2.0] * 100,
        "host_disk_read_iops": [100.0] * 100,
        "pgwatch_pg_stat_activity_count": [10.0] * 100,
    }, {
        'setting_name="max_connections"': {"metric": {"setting_name": "max_connections", "setting_value": "60"}, "value": [0, "1"]},
        "host_os_memory_total_bytes": {"metric": {}, "value": [0, "1000000000"]},
        "host_disk_free_bytes": {"metric": {}, "value": [0, "5000000000"]},
        "pgwatch_db_size_size_b": {"metric": {}, "value": [0, "123456"]},
    })
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    data = report["results"]["node-1"]["data"]
    assert report["checkId"] == "A014" and report["checkTitle"] == "Resource sizing"
    assert data["host_metrics"] is True
    assert data["cpu"]["p95_pct"] == pytest.approx(95.05) and data["cpu"]["max_pct"] == 100
    assert data["memory"]["available_min_pct"] == 7 and data["memory"]["total_bytes"] == 1000000000
    assert data["disk"]["mbps_p95"] == 2 and data["disk"]["iops_max"] == 100 and data["disk"]["free_bytes"] == 5000000000
    assert data["connections"] == {"backends_p95": 10, "backends_max": 10, "max_connections": 60}
    assert data["db_size_bytes"] == 123456
    assert 8.0 <= data["window_hours"] <= 8.4  # 100 five-minute points
    s = data["series"]
    assert s["step_seconds"] == 300 and len(s["cpu_pct"]) == 2016
    assert s["cpu_pct"][-1] == 100 and s["cpu_pct"][0] is None  # oldest first, gaps are null


@pytest.mark.unit
def test_a014_without_host_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _gen(monkeypatch, {"pgwatch_pg_stat_activity_count": [3.0] * 12}, {})
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    data = report["results"]["node-1"]["data"]
    assert data["host_metrics"] is False
    assert data["cpu"]["p95_pct"] is None and data["memory"]["available_p5_pct"] is None
    assert data["disk"]["mbps_p95"] is None
    assert data["connections"]["backends_p95"] == 3 and data["connections"]["max_connections"] is None
    assert data["window_hours"] > 0


@pytest.mark.unit
def test_a014_runs_with_every_report() -> None:
    import inspect
    src = inspect.getsource(PostgresReportGenerator.generate_all_reports)
    assert "('A014', self.generate_a014_resource_sizing_report)" in src


@pytest.mark.unit
def test_a014_window_is_the_hours_with_host_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # pgwatch for ~8 h, host metrics allowed 1 h ago: the window is 1 h, so the
    # platform's "oversized" gate cannot pass on hours the box never measured.
    gen = _gen(monkeypatch, {
        "host_cpu_utilization_percent": [4.0] * 12,
        "host_memory_available_bytes": [80.0] * 12,
        "pgwatch_pg_stat_activity_count": [3.0] * 100,
    }, {})
    data = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]
    assert data["host_metrics"] is True
    assert data["window_hours"] == pytest.approx(1.0)


@pytest.mark.unit
def test_a014_disk_series_is_sustained_throughput(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each 5-minute point is the platform's "5 minutes at this rate": it must
    # be the average over the 5 minutes, not the busiest minute.
    queries: list[str] = []
    gen = _gen(monkeypatch, {}, {})
    real_range = gen.query_range

    def spy(query: str, *a: Any, **k: Any) -> list[dict[str, Any]]:
        queries.append(query)
        return real_range(query, *a, **k)

    monkeypatch.setattr(gen, "query_range", spy)
    gen.generate_a014_resource_sizing_report("local", "node-1")
    disk = [q for q in queries if "host_disk_read_bytes_per_second" in q]
    assert len(disk) == 1 and disk[0].startswith("avg_over_time(") and "max_over_time" not in disk[0]


@pytest.mark.unit
def test_a014_disk_iops_series_is_the_5_minute_average(monkeypatch: pytest.MonkeyPatch) -> None:
    # The platform judges IOPS against the sustained limit the same way as
    # throughput: a point is "5 minutes at this rate", so it must be the
    # 5-minute average, and the report says so (series.disk_iops_kind), since
    # older boxes sent the busiest minute (platform-all#1086).
    queries: list[str] = []
    gen = _gen(monkeypatch, {
        "host_cpu_utilization_percent": [4.0] * 12,
        "host_disk_read_iops": [100.0] * 11 + [300.0],
    }, {})
    real_range = gen.query_range

    def spy(query: str, *a: Any, **k: Any) -> list[dict[str, Any]]:
        queries.append(query)
        return real_range(query, *a, **k)

    monkeypatch.setattr(gen, "query_range", spy)
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    iops = [q for q in queries if "host_disk_read_iops" in q]
    assert len(iops) == 1 and iops[0].startswith("avg_over_time(") and "max_over_time" not in iops[0]
    data = report["results"]["node-1"]["data"]
    assert data["series"]["disk_iops_kind"] == "avg_5m"
    assert data["disk"]["iops_max"] == 300 and data["disk"]["iops_p95"] == pytest.approx(190)
    assert data["series"]["disk_iops"][-1] == 300


@pytest.mark.unit
def test_a014_schema_iops_kind_is_optional_and_only_avg_5m(monkeypatch: pytest.MonkeyPatch) -> None:
    from jsonschema import ValidationError

    gen = _gen(monkeypatch, {"host_cpu_utilization_percent": [4.0] * 12}, {})
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    series = report["results"]["node-1"]["data"]["series"]
    series.pop("disk_iops_kind", None)
    validate_report(report)  # reports from released reporters stay valid (optional field)
    series["disk_iops_kind"] = "max_1m"
    with pytest.raises(ValidationError):
        validate_report(report)
