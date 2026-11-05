"""G001 memory settings: the memory estimate's units and the window facts."""
import pytest

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    return PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")


def setting(value, unit):
    return {"setting": value, "unit": unit, "category": "Resource Usage / Memory", "context": "",
            "vartype": "integer", "pretty_value": ""}


@pytest.mark.unit
def test_the_estimate_applies_each_setting_unit(generator) -> None:
    """pg_settings gives shared_buffers in 8kB pages: 28672 pages are 224 MiB, not 28 MiB
    (a Supabase report showed the 8x error in the analysis block)."""
    memory = {
        "shared_buffers": setting("28672", "8kB"),
        "effective_cache_size": setting("49152", "8kB"),
        "wal_buffers": setting("492", "8kB"),
        "work_mem": setting("2184", "kB"),
        "maintenance_work_mem": setting("32768", "kB"),
        "max_connections": setting("60", ""),
    }
    usage = generator._analyze_memory_settings(memory)["estimated_total_memory_usage"]
    assert usage["shared_buffers_bytes"] == 224 * 1024 ** 2
    assert usage["effective_cache_size_bytes"] == 384 * 1024 ** 2
    assert usage["wal_buffers_bytes"] == 3936 * 1024
    assert usage["work_mem_per_connection_bytes"] == 2184 * 1024
    assert usage["maintenance_work_mem_bytes"] == 32 * 1024 ** 2
    assert usage["max_work_mem_usage_bytes"] == 60 * 2184 * 1024
    assert usage["shared_buffers_pretty"] == "224 MiB"


@pytest.mark.unit
def test_values_with_a_suffix_and_missing_settings_still_parse(generator) -> None:
    memory = {"shared_buffers": {"setting": "128MB"}, "work_mem": {"setting": "4MB"},
              "autovacuum_work_mem": setting("-1", "kB")}
    usage = generator._analyze_memory_settings(memory)["estimated_total_memory_usage"]
    assert usage["shared_buffers_bytes"] == 128 * 1024 ** 2
    assert usage["work_mem_per_connection_bytes"] == 4 * 1024 ** 2
    assert usage["effective_cache_size_bytes"] == 4 * 1024 ** 3  # the default


def answer(value, metric=None):
    return {"status": "success", "data": {"result": [{"metric": metric or {}, "value": [0, str(value)]}]}}


class FakeProm:
    """query_instant by the first pattern the query contains; empty otherwise."""

    def __init__(self, rules):
        self.rules, self.queries = rules, []

    def __call__(self, query):
        self.queries.append(query)
        for pattern, result in self.rules:
            if pattern in query:
                return result
        return {"status": "success", "data": {"result": []}}


NOW = 1793700000

RULES = [
    ("count by (datname)", {"status": "success", "data": {"result": [
        {"metric": {"datname": "postgres"}, "value": [0, "1"]},
        {"metric": {"datname": "template1"}, "value": [0, "1"]},
        {"metric": {}, "value": [0, "1"]}]}}),
    ("min(min_over_time(timestamp(pgwatch_db_stats_numbackends", answer(NOW - 3 * 86400)),
    ("count_over_time(pgwatch_db_stats_numbackends", answer(8640)),
    ('max_over_time(timestamp((sum(pgwatch_pg_stat_activity_count{cluster="c", node_name="n", state="active"})) >= 4.0)',
     answer(NOW - 3600)),
    ('max_over_time((sum(pgwatch_pg_stat_activity_count{cluster="c", node_name="n", state="active"}))', answer(4)),
    ('avg_over_time((sum(pgwatch_pg_stat_activity_count{cluster="c", node_name="n", state="active"}))', answer(1.049)),
    ("max_over_time(timestamp((sum(pgwatch_pg_stat_activity_count", answer(NOW - 7200)),
    ("topk(5", {"status": "success", "data": {"result": [
        {"metric": {"application_name": "PostgREST"}, "value": [0, "3"]},
        {"metric": {"application_name": "Supavisor"}, "value": [0, "7"]},
        {"metric": {"application_name": "idle app"}, "value": [0, "0"]}]}}),
    ("max_over_time((sum(pgwatch_pg_stat_activity_count", answer(15)),
    ("avg_over_time((sum(pgwatch_pg_stat_activity_count", answer(10.917)),
    ("last_over_time(pgwatch_db_stats_postmaster_uptime_s", answer(86400)),
    ("resets(pgwatch_db_stats_postmaster_uptime_s", answer(0)),
    ("resets(pgwatch_db_stats_sessions{", answer(1)),
    ("pgwatch_db_stats_sessions_fatal", answer(0)),
    ("pgwatch_db_stats_sessions_killed", answer(3.9)),
    ("max(last_over_time(host_os_memory_total_bytes", answer(454955008)),
    ("max_over_time(timestamp((100 * host_memory_available_bytes", answer(NOW - 600)),
    ("quantile_over_time(0.05", answer(30.468)),
    ("min_over_time((100 * host_memory_available_bytes", answer(17.064)),
]


@pytest.mark.unit
def test_activity_gives_peaks_starts_and_host_memory(generator, monkeypatch) -> None:
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: NOW)
    prom = FakeProm(RULES)
    monkeypatch.setattr(generator, "query_instant", prom)
    out = generator._g001_activity("c", "n")
    assert out["window"] == {"start": "2026-10-31T10:00:00Z", "end": "2026-11-03T10:00:00Z",
                             "seconds": 3 * 86400, "samples": 8640, "sample_seconds": 30}
    assert out["databases"] == ["postgres"]  # template databases and an empty name left out
    assert out["connections"] == {"peak": 15, "peak_at": "2026-11-03T08:00:00Z", "mean": 10.92}
    assert out["active_statements"] == {"peak": 4, "peak_at": "2026-11-03T09:00:00Z", "mean": 1.05}
    assert out["applications"] == [{"application": "Supavisor", "peak": 7}, {"application": "PostgREST", "peak": 3}]
    assert out["server"] == {"last_start": "2026-11-02T10:00:00Z", "starts_in_window": 0, "stats_resets": 1,
                             "sessions_fatal": 0}  # operator-ended sessions are no memory signal
    assert out["host_memory"] == {"total_bytes": 454955008, "available_min_pct": 17.06,
                                  "available_min_at": "2026-11-03T09:50:00Z", "available_p5_pct": 30.47}
    assert all('cluster="c", node_name="n"' in q for q in prom.queries if "pgwatch" in q or "host_" in q)
    # Host memory on a 1-minute grid: a shorter dip would be missed on 5 minutes (REV r2).
    assert all(":1m])" in q for q in prom.queries if "host_memory_available_bytes" in q)


@pytest.mark.unit
def test_activity_leaves_out_what_it_cannot_read(generator, monkeypatch) -> None:
    monkeypatch.setattr(generator, "query_instant", FakeProm([]))
    assert generator._g001_activity("c", "n") == {}
    # No host metrics (no Supabase scope, a self-managed box without them): no host_memory.
    monkeypatch.setattr(generator, "query_instant", FakeProm([r for r in RULES if "host_" not in r[0]]))
    out = generator._g001_activity("c", "n")
    assert "host_memory" not in out and out["connections"]["peak"] == 15


@pytest.mark.unit
def test_g001_reads_the_memory_model_inputs_and_adds_activity(generator, monkeypatch) -> None:
    for name in ("shared_memory_size", "autovacuum_max_workers", "max_parallel_workers",
                 "max_parallel_maintenance_workers", "max_worker_processes", "max_parallel_workers_per_gather"):
        assert name in generator.G001_SETTINGS
    calls = []
    monkeypatch.setattr(generator, "_g001_activity", lambda c, n: calls.append((c, n)) or {"connections": {"peak": 2}})
    a003 = {"results": {"n": {"data": {"shared_buffers": {"setting": "128MB"}}}}}
    report = generator.generate_g001_from_a003(a003, node_name="n", cluster="c")
    assert report["results"]["n"]["data"]["activity"] == {"connections": {"peak": 2}} and calls == [("c", "n")]
    # Without a cluster (an older caller): no queries, no activity key.
    report = generator.generate_g001_from_a003(a003, node_name="n")
    assert "activity" not in report["results"]["n"]["data"] and len(calls) == 1



@pytest.mark.unit
def test_thresholds_are_the_exact_extremes(generator, monkeypatch) -> None:
    """REV r1: a :g threshold kept 6 digits, so `<= 17.0641` matched nothing for 17.064123456."""
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: NOW)
    rules = [r for r in RULES if "min_over_time((100 * host_memory_available_bytes" not in r[0]
             and "max_over_time((sum(pgwatch_pg_stat_activity_count" not in r[0]]
    rules = [("min_over_time((100 * host_memory_available_bytes", answer(17.064123456)),
             ('max_over_time((sum(pgwatch_pg_stat_activity_count{cluster="c", node_name="n"}))', answer(1234567))] + rules
    prom = FakeProm(rules)
    monkeypatch.setattr(generator, "query_instant", prom)
    generator._g001_activity("c", "n")
    assert any("<= 17.064123456)" in q for q in prom.queries)
    assert any(">= 1234567.0)" in q for q in prom.queries)
    assert not any("e+06" in q or "<= 17.0641)" in q for q in prom.queries)
