"""A014 Resource sizing: usage measurements for the platform's verdict (platform-all#1051)."""
from __future__ import annotations

import re
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


# --- Resource sizing for RDS and ClickHouse (platform-all#1101) -------------

def _sizing_gen(monkeypatch: pytest.MonkeyPatch, range_items: dict[str, list[dict[str, Any]]],
                instants: dict[str, float]) -> PostgresReportGenerator:
    """Like _gen, but a range query returns the given items (labels and values) as they are."""
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG_VERSION)
    monkeypatch.setattr(gen, "query_range",
                        lambda query, *a, **k: next((v for n, v in range_items.items() if n in query), []))
    monkeypatch.setattr(gen, "query_instant", lambda query: next(
        ({"status": "success", "data": {"result": [{"metric": {}, "value": [0, str(v)]}]}}
         for n, v in instants.items() if n in query), {"status": "success", "data": {"result": []}}))
    return gen


def _points(values: list[Any], end: int) -> list[list[Any]]:
    return [[end - 300 * (len(values) - 1 - i), str(v)] for i, v in enumerate(values)]


RDS_LABELS = {"instance_class": "db.t4g.micro", "engine": "postgres", "storage_type": "gp3", "multi_az": "false",
              "region": "eu-central-1"}


@pytest.mark.unit
def test_a014_rds_observed_size_and_resize_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    now = int(time.time()) // 300 * 300
    micro = [[now - 300 * i, "1"] for i in range(11, 5, -1)]   # 6 points, oldest first
    xlarge = [[now - 300 * i, "1"] for i in range(5, -1, -1)]  # the 6 newest
    gen = _sizing_gen(monkeypatch, {
        "host_instance_info": [{"metric": dict(RDS_LABELS), "values": micro},
                               {"metric": dict(RDS_LABELS, instance_class="db.t4g.xlarge"), "values": xlarge}],
        "host_cpu_utilization_percent": [{"metric": {}, "values": _points([10.0] * 12, now)}],
    }, {
        "host_disk_size_bytes": 20 * 2 ** 30, "host_disk_max_size_bytes": 40 * 2 ** 30,
        "host_disk_provisioned_iops": 3000, "host_disk_provisioned_throughput_bytes_per_second": 125 * 2 ** 20,
    })
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    assert observed == {
        "provider": "rds", "size": "db.t4g.xlarge", "region": "eu-central-1", "engine": "postgres", "multi_az": False,
        "storage": {"type": "gp3", "size_gb": 20.0, "max_size_gb": 40.0, "iops": 3000, "throughput_mbps": 131.1},
        "seen_at_epoch": now, "since_epoch": now - 300 * 5,
    }


@pytest.mark.unit
def test_a014_rds_storage_change_is_not_a_resize_and_aurora_is_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    now = int(time.time()) // 300 * 300
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": dict(RDS_LABELS, storage_type="gp2"), "values": [[now - 600, "1"]]},
        {"metric": dict(RDS_LABELS), "values": [[now - 300, "1"], [now, "1"]]}]}, {})
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["since_epoch"] is None and observed["storage"]["type"] == "gp3"
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": dict(RDS_LABELS, engine="aurora-postgresql", instance_class="db.r6g.large"), "values": [[now, "1"]]}]}, {})
    assert gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]["provider"] == "aurora"


@pytest.mark.unit
def test_a014_sizing_series_only_where_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    now = int(time.time()) // 300 * 300
    plain = _sizing_gen(monkeypatch, {"host_cpu_utilization_percent": [{"metric": {}, "values": _points([10.0, 30.0], now)}]}, {})
    data = plain.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]
    # A Supabase-shaped box: no new series keys, no observed_size; the average CPU is there.
    assert not {"mem_available_bytes", "cpu_credit_balance", "cpu_surplus_balance", "burst_balance_pct",
                "ebs_io_balance_pct", "ebs_byte_balance_pct"} & set(data["series"])
    assert "observed_size" not in data and data["cpu"]["avg_pct"] == 20
    rds = _sizing_gen(monkeypatch, {
        "host_cpu_surplus_credit_balance": [{"metric": {}, "values": _points([0.0, 3.5], now)}],
        "host_cpu_credit_balance": [{"metric": {}, "values": _points([12.0, 0.0], now)}],
        "host_memory_available_bytes": [{"metric": {}, "values": _points([5e8, 4e8], now)}],
    }, {})
    report = rds.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    s = report["results"]["node-1"]["data"]["series"]
    assert s["cpu_surplus_balance"][-2:] == [0, 3.5] and s["cpu_credit_balance"][-1] == 0
    assert s["mem_available_bytes"][-1] == 4e8 and "burst_balance_pct" not in s


CH_SCRAPE = """- job_name: clickhouse-ch1
  scheme: https
  metrics_path: /v1/organizations/00000000-0000-4000-8000-000000000001/postgres/00000000-0000-4000-8000-000000000002/prometheus
  scrape_interval: 60s
  basic_auth:
    username: keyid123
    password_file: /etc/pgai/host-metrics/clickhouse-ch1.secret
  static_configs:
    - targets:
        - api.clickhouse.cloud
      labels:
        cluster: default
        node_name: ch1
"""


class _Reply:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code, self._body = status, body

    def json(self) -> Any:
        return self._body


def _clickhouse(monkeypatch: pytest.MonkeyPatch, tmp_path, replies: dict[str, _Reply]) -> tuple[PostgresReportGenerator, list]:
    (tmp_path / "clickhouse-ch1.yml").write_text(CH_SCRAPE)
    (tmp_path / "clickhouse-ch1.secret").write_text("s3cret\n")
    monkeypatch.setenv("REPORTER_HOST_METRICS_DIR", str(tmp_path))
    calls: list = []

    def fake_get(url: str, **kwargs: Any) -> _Reply:
        calls.append((url, kwargs.get("auth")))
        return next((r for n, r in replies.items() if url.endswith(n)), _Reply(404, {}))

    import reporter.postgres_reports as module
    monkeypatch.setattr(module.requests, "get", fake_get)
    now = int(time.time()) // 300 * 300
    gen = _sizing_gen(monkeypatch, {
        "host_cpu_cores": [{"metric": {}, "values": _points([1 * 1e15 + 8 * 2 ** 30] * 3 + [2 * 1e15 + 16 * 2 ** 30] * 3, now)}],
    }, {'provider="ClickHouse"': 1, "host_cpu_cores": 2, "host_os_memory_total_bytes": 16 * 2 ** 30,
        "host_disk_size_bytes": 118 * 2 ** 30})
    return gen, calls


@pytest.mark.unit
def test_a014_clickhouse_size_from_the_api_with_the_box_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    gen, calls = _clickhouse(monkeypatch, tmp_path, {
        "/postgres/00000000-0000-4000-8000-000000000002": _Reply(200, {"result": {
            "size": "r6gd.large", "region": "eu-central-1", "haType": "async", "storageSize": 118}}),
        "/usageCost": _Reply(200, {"result": {"costs": [{"organizationTier": "SCALE"}]}}),
    })
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    now = int(time.time()) // 300 * 300
    assert observed == {
        "provider": "clickhouse", "size": "r6gd.large", "region": "eu-central-1", "standbys": 1, "org_tier": "scale",
        "vcpus": 2, "memory_bytes": 16 * 2 ** 30, "size_source": "api",
        "storage": {"type": "nvme", "size_gb": 118.0, "max_size_gb": None, "iops": None, "throughput_mbps": None},
        "seen_at_epoch": now, "since_epoch": now - 600,
    }
    base = "https://api.clickhouse.cloud/v1/organizations/00000000-0000-4000-8000-000000000001"
    assert [c[0] for c in calls] == [f"{base}/postgres/00000000-0000-4000-8000-000000000002", f"{base}/usageCost"]
    assert all(c[1] == ("keyid123", "s3cret") for c in calls)


@pytest.mark.unit
def test_a014_clickhouse_api_refused_falls_back_to_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    gen, _ = _clickhouse(monkeypatch, tmp_path, {"/postgres/00000000-0000-4000-8000-000000000002": _Reply(403, {})})
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] is None and observed["size_source"] == "metrics" and observed["org_tier"] is None
    assert observed["vcpus"] == 2 and observed["memory_bytes"] == 16 * 2 ** 30
    assert observed["storage"]["size_gb"] == 118.0


@pytest.mark.unit
def test_a014_no_observed_size_without_rds_info_or_clickhouse(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _sizing_gen(monkeypatch, {}, {"host_cpu_cores": 2})
    assert "observed_size" not in gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]


@pytest.mark.unit
def test_a014_balances_keep_the_lowest_and_the_surplus_the_highest(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two series in one 5-minute bucket (e.g. a restarted poller): a balance
    # counts at its lowest, the surplus at its highest.
    now = int(time.time()) // 300 * 300
    gen = _sizing_gen(monkeypatch, {
        "host_cpu_credit_balance": [{"metric": {}, "values": [[now, "5"]]}, {"metric": {}, "values": [[now, "2"]]}],
        "host_ebs_byte_balance_percent": [{"metric": {}, "values": [[now, "40"]]}, {"metric": {}, "values": [[now, "0"]]}],
        "host_cpu_surplus_credit_balance": [{"metric": {}, "values": [[now, "1"]]}, {"metric": {}, "values": [[now, "4"]]}],
    }, {})
    s = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["series"]
    assert (s["cpu_credit_balance"][-1], s["ebs_byte_balance_pct"][-1], s["cpu_surplus_balance"][-1]) == (2, 0, 4)


@pytest.mark.unit
def test_a014_rds_info_newer_than_the_grid_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first report runs 3 minutes after the box starts: the instance info
    # exists only after the last 5-minute grid point, and must still be read.
    instants = {"host_instance_info": 1}
    gen = _sizing_gen(monkeypatch, {}, instants)
    real = gen.query_instant

    def instant(query: str) -> dict[str, Any]:
        if "host_instance_info" in query:
            # tlast_over_time: the value is the last sample's time.
            return {"status": "success", "data": {"result": [{"metric": dict(RDS_LABELS), "value": [time.time(), str(time.time() - 20)]}]}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "db.t4g.micro" and observed["since_epoch"] is None
    assert observed["seen_at_epoch"] >= int(time.time()) - 600


@pytest.mark.unit
def test_a014_clickhouse_size_from_this_nodes_scrape_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # A box monitoring two services: each node gets its own service's size, and
    # a refused key on the other one does not matter.
    other = CH_SCRAPE.replace("clickhouse-ch1", "clickhouse-a0").replace("node_name: ch1", "node_name: other") \
        .replace("00000000-0000-4000-8000-000000000002", "11111111-1111-1111-1111-111111111111")
    (tmp_path / "clickhouse-a0.yml").write_text(other)
    (tmp_path / "clickhouse-a0.secret").write_text("bad\n")
    gen, calls = _clickhouse(monkeypatch, tmp_path, {
        "/postgres/11111111-1111-1111-1111-111111111111": _Reply(401, {}),
        "/postgres/00000000-0000-4000-8000-000000000002": _Reply(200, {"result": {
            "size": "r6gd.large", "region": "eu-central-1", "haType": "none", "storageSize": 118}}),
        "/usageCost": _Reply(200, {"result": {"costs": [{"organizationTier": "SCALE"}]}}),
    })
    observed = gen.generate_a014_resource_sizing_report("default", "ch1")["results"]["ch1"]["data"]["observed_size"]
    assert observed["size"] == "r6gd.large" and observed["size_source"] == "api"
    assert not any("11111111" in c[0] for c in calls)


@pytest.mark.unit
def test_a014_rds_newest_class_right_after_a_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    # Within 10 minutes of a resize both label sets answer last_over_time; the
    # one whose last sample is newest is the current class.
    now = int(time.time())
    gen = _sizing_gen(monkeypatch, {}, {})

    def instant(query: str) -> dict[str, Any]:
        if "host_instance_info" in query:
            old = {"metric": dict(RDS_LABELS, instance_class="db.r6g.xlarge"), "value": [now, str(now - 420)]}
            new = {"metric": dict(RDS_LABELS, instance_class="db.r6g.large"), "value": [now, str(now - 30)]}
            # tlast_over_time: the value is the last sample's time.
            return {"status": "success", "data": {"result": [old, new]}} if "tlast" in query else \
                {"status": "success", "data": {"result": [dict(old, value=[now, "1"]), dict(new, value=[now, "1"])]}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "db.r6g.large"


@pytest.mark.unit
def test_a014_max_connections_after_a_resize_is_the_newest(monkeypatch: pytest.MonkeyPatch) -> None:
    # max_connections is a label: after a resize the window holds the old and
    # the new value. The one whose last sample is newest is current.
    gen = _gen(monkeypatch, {"pgwatch_pg_stat_activity_count": [5.0] * 12}, {})
    now = time.time()

    def instant(query: str) -> dict[str, Any]:
        if 'setting_name="max_connections"' in query:
            stamp = "tlast_over_time" in query
            return {"status": "success", "data": {"result": [
                {"metric": {"setting_name": "max_connections", "setting_value": "1713"},
                 "value": [now, str(now - 3000) if stamp else "1"]},
                {"metric": {"setting_name": "max_connections", "setting_value": "78"},
                 "value": [now, str(now - 20) if stamp else "1"]},
            ]}}
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(gen, "query_instant", instant)
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    assert report["results"]["node-1"]["data"]["connections"]["max_connections"] == 78


@pytest.mark.unit
def test_a014_rds_resize_tie_on_the_last_grid_point(monkeypatch: pytest.MonkeyPatch) -> None:
    # last_over_time[10m] keeps the old class on the grid for 10 minutes after
    # its last sample, so both classes share the newest grid points. The class
    # sampled last is current, and the resize is where it starts.
    now = int(time.time()) // 300 * 300
    old = dict(RDS_LABELS, instance_class="db.r6g.xlarge")
    new = dict(RDS_LABELS, instance_class="db.r6g.large")
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": new, "values": [[now - 300, "1"], [now, "1"]]},
        {"metric": old, "values": [[now - 300 * i, "1"] for i in range(4, -1, -1)]}]}, {})
    real = gen.query_instant

    def instant(query: str) -> dict[str, Any]:
        if "tlast_over_time(host_instance_info" in query:
            return {"status": "success", "data": {"result": [
                {"metric": old, "value": [now, str(now - 420)]}, {"metric": new, "value": [now, str(now - 30)]}]}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "db.r6g.large"
    assert observed["since_epoch"] == now - 300 and observed["seen_at_epoch"] == now


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["timeout", "bad json", "500"])
def test_a014_clickhouse_org_tier_failure_keeps_the_size(monkeypatch: pytest.MonkeyPatch, tmp_path, failure: str) -> None:
    class _Bad(_Reply):
        def json(self) -> Any:
            raise ValueError("not json")

    gen, _ = _clickhouse(monkeypatch, tmp_path, {
        "/postgres/00000000-0000-4000-8000-000000000002": _Reply(200, {"result": {
            "size": "r6gd.large", "region": "eu-central-1", "haType": "none", "storageSize": 118}}),
        "/usageCost": _Reply(500, {}) if failure == "500" else _Bad(200, None),
    })
    if failure == "timeout":
        import reporter.postgres_reports as module
        good = module.requests.get

        def get(url: str, **kwargs: Any) -> Any:
            if url.endswith("/usageCost"):
                raise module.requests.exceptions.Timeout()
            return good(url, **kwargs)
        monkeypatch.setattr(module.requests, "get", get)
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "r6gd.large" and observed["size_source"] == "api" and observed["org_tier"] is None


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["timeout", "bad json", "500", "no secret"])
def test_a014_clickhouse_service_read_failure_falls_back_to_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path,
                                                                     failure: str) -> None:
    class _Bad(_Reply):
        def json(self) -> Any:
            raise ValueError("not json")

    service = "/postgres/00000000-0000-4000-8000-000000000002"
    gen, _ = _clickhouse(monkeypatch, tmp_path, {service: _Reply(500, {}) if failure == "500" else _Bad(200, None)})
    import reporter.postgres_reports as module
    if failure == "timeout":
        monkeypatch.setattr(module.requests, "get", lambda url, **k: (_ for _ in ()).throw(module.requests.exceptions.Timeout()))
    if failure == "no secret":
        (tmp_path / "clickhouse-ch1.secret").unlink()
    report = gen.generate_a014_resource_sizing_report("local", "node-1")
    validate_report(report)
    observed = report["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] is None and observed["size_source"] == "metrics" and observed["vcpus"] == 2


@pytest.mark.unit
def test_a014_clickhouse_no_file_for_this_node_reads_no_other_service(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Two services on the box and neither file is this node's: another
    # service's size must not be reported as this one's.
    other = CH_SCRAPE.replace("clickhouse-ch1", "clickhouse-a0").replace("node_name: ch1", "node_name: other")
    (tmp_path / "clickhouse-a0.yml").write_text(other)
    (tmp_path / "clickhouse-a0.secret").write_text("s3cret\n")
    gen, calls = _clickhouse(monkeypatch, tmp_path, {
        "/postgres/00000000-0000-4000-8000-000000000002": _Reply(200, {"result": {"size": "r6gd.large"}})})
    observed = gen.generate_a014_resource_sizing_report("default", "ch9")["results"]["ch9"]["data"]["observed_size"]
    assert observed["size_source"] == "metrics" and calls == []


@pytest.mark.unit
def test_a014_clickhouse_seen_at_is_the_last_sample_when_no_grid_point(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Metrics last scraped 2 h ago (no point in the last 10 minutes) and the API
    # down: the size was seen 2 h ago, not now.
    gen, _ = _clickhouse(monkeypatch, tmp_path, {})
    monkeypatch.setattr(gen, "query_range", lambda *a, **k: [])
    real = gen.query_instant
    two_hours_ago = int(time.time()) - 7200

    def instant(query: str) -> dict[str, Any]:
        if query.startswith("tlast_over_time(host_os_memory_total_bytes"):
            return {"status": "success", "data": {"result": [{"metric": {}, "value": [0, str(two_hours_ago)]}]}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["seen_at_epoch"] == two_hours_ago and observed["since_epoch"] is None


@pytest.mark.unit
@pytest.mark.parametrize("age", [610, 700, 1500])
def test_a014_rds_resize_older_than_the_carry_over(monkeypatch: pytest.MonkeyPatch, age: int) -> None:
    # The old class's last sample can be older than 10 minutes while its
    # last_over_time[10m] carry-over is still on the grid: the resize is where
    # the new class starts, even when the newest sample is not newer than the
    # last grid point.
    now = int(time.time()) // 300 * 300
    resized = now - age
    old = dict(RDS_LABELS, instance_class="db.r6g.xlarge")
    new = dict(RDS_LABELS, instance_class="db.r6g.large")
    grid = lambda lo, hi: [[t, "1"] for t in range(now - 3600, now + 1, 300) if lo <= t <= hi]
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": old, "values": grid(0, resized + 600)},
        {"metric": new, "values": grid(resized, now)}]}, {})
    real = gen.query_instant

    def instant(query: str) -> dict[str, Any]:
        if "tlast_over_time(host_instance_info" in query:
            window = int(re.search(r"\[(\d+)s\]", query).group(1)) if re.search(r"\[(\d+)s\]", query) else 600
            result = [{"metric": new, "value": [now, str(now - 20)]}]
            if resized - 60 >= now - window:
                result.insert(0, {"metric": old, "value": [now, str(resized - 60)]})
            return {"status": "success", "data": {"result": result}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "db.r6g.large"
    assert observed["since_epoch"] == min(t for t, _ in grid(resized, now))


@pytest.mark.unit
def test_a014_rds_storage_change_carry_over_does_not_win(monkeypatch: pytest.MonkeyPatch) -> None:
    # gp2 -> gp3 on the same class: the label set sampled last is current.
    now = int(time.time()) // 300 * 300
    gp2 = dict(RDS_LABELS, storage_type="gp2")
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": dict(RDS_LABELS), "values": [[now - 300, "1"], [now, "1"]]},
        {"metric": gp2, "values": [[now - 900 + 300 * i, "1"] for i in range(4)]}]}, {})
    real = gen.query_instant

    def instant(query: str) -> dict[str, Any]:
        if "tlast_over_time(host_instance_info" in query:
            return {"status": "success", "data": {"result": [
                {"metric": gp2, "value": [now, str(now - 420)]}, {"metric": dict(RDS_LABELS), "value": [now, str(now - 30)]}]}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["storage"]["type"] == "gp3" and observed["since_epoch"] is None


@pytest.mark.unit
def test_a014_rds_resize_at_the_window_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resized just before the 7-day window: the old class's carry-over is still
    # on the first grid points, so its last sample must be read too, and the
    # window holds only the new class.
    now = int(time.time()) // 300 * 300
    start = now - 7 * 86400
    resized = start - 120
    old = dict(RDS_LABELS, instance_class="db.r6g.xlarge")
    new = dict(RDS_LABELS, instance_class="db.r6g.large")
    gen = _sizing_gen(monkeypatch, {"host_instance_info": [
        {"metric": old, "values": [[t, "1"] for t in range(start, resized + 601, 300)]},
        {"metric": new, "values": [[t, "1"] for t in range(start, now + 1, 300)]}]}, {})
    real = gen.query_instant

    def instant(query: str) -> dict[str, Any]:
        if "tlast_over_time(host_instance_info" in query:
            window = int(re.search(r"\[(\d+)s\]", query).group(1))
            result = [{"metric": new, "value": [now, str(now - 20)]}]
            if resized - 60 >= now - window:
                result.insert(0, {"metric": old, "value": [now, str(resized - 60)]})
            return {"status": "success", "data": {"result": result}}
        return real(query)

    monkeypatch.setattr(gen, "query_instant", instant)
    observed = gen.generate_a014_resource_sizing_report("local", "node-1")["results"]["node-1"]["data"]["observed_size"]
    assert observed["size"] == "db.r6g.large" and observed["since_epoch"] is None
