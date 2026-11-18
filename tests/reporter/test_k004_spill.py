"""K004: the statements' window facts, the database's temporary-file counters and the spill settings."""
from __future__ import annotations

import re
from typing import Any, List

import pytest

from reporter import k004
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report
from tests.reporter.test_g003 import Conn

H = 3600
TIMELINE = [H * h for h in range(470_000, 470_003)]  # three hours, each stamp the end of its hour
PG = {"version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"}


def _readings(values: List[float], start: int = TIMELINE[0] - H, step: int = k004.STEP_S) -> List[List[Any]]:
    return [[start + i * step, str(v)] for i, v in enumerate(values)]


# ── counter_hours ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_counter_hours_sums_rises_per_hour_and_finds_the_busiest_five_minutes() -> None:
    # 37 readings five minutes apart over the three hours: +10 each step, +600 in one step of the second hour.
    values, total = [], 1000.0
    for i in range(37):
        if i:
            total += 600 if i == 18 else 10
        values.append(total)
    out = k004.counter_hours([{"metric": {"datname": "app"}, "values": _readings(values)}], TIMELINE)
    assert out["total"] == 35 * 10 + 600
    assert out["hourly"] == [120.0, 110.0 + 600, 120.0]
    assert out["peak_per_s"] == 600 / k004.STEP_S
    assert out["peak_at"] == "2023-08-14T08:30:00+00:00"


@pytest.mark.unit
def test_counter_hours_counts_from_a_reset_never_the_first_reading_and_merges_a_relabelled_series() -> None:
    first = {"metric": {"a": "1"}, "values": _readings([5000, 5100, 5200])}
    # The counter restarted (a stats reset): 10 after 5200 is a rise of 10, never -5190; then +40.
    after = {"metric": {"a": "2"}, "values": _readings([10, 50], start=TIMELINE[0] - H + 3 * k004.STEP_S)}
    out = k004.counter_hours([first, after], TIMELINE)
    assert out["total"] == 100 + 100 + 10 + 40  # the 10 written since the reset count


@pytest.mark.unit
def test_counter_hours_gives_no_rate_across_a_gap_and_none_without_two_readings() -> None:
    gap = [[TIMELINE[0] - H, "0"], [TIMELINE[0] - H + 1800, "900"]]  # 30 minutes: a rise, not a 5-minute rate
    out = k004.counter_hours([{"metric": {}, "values": gap}], TIMELINE)
    assert out["total"] == 900 and "peak_per_s" not in out
    assert k004.counter_hours([{"metric": {}, "values": [[TIMELINE[0], "1"]]}], TIMELINE) is None
    assert k004.counter_hours([], TIMELINE) is None


@pytest.mark.unit
def test_counter_hours_with_no_files_is_measured_zero_not_unknown() -> None:
    out = k004.counter_hours([{"metric": {}, "values": _readings([7] * 37)}], TIMELINE)
    assert out["total"] == 0 and out["hourly"] == [0.0, 0.0, 0.0] and out["peak_per_s"] == 0


# ── settings and the node ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_settings_from_monitoring_keeps_only_the_spill_settings() -> None:
    reply = {"data": {"result": [
        {"metric": {"setting_name": "work_mem", "setting_value": "4096", "unit": "kB"}},
        {"metric": {"tag_setting_name": "log_temp_files", "tag_setting_value": "-1", "unit": "kB"}},
        {"metric": {"setting_name": "shared_buffers", "setting_value": "16384", "unit": "8kB"}},
    ]}}
    assert k004.settings_from_monitoring(reply) == ({
        "work_mem": {"setting": "4096", "unit": "kB", "context": ""},
        "log_temp_files": {"setting": "-1", "unit": "kB", "context": ""}}, [])


@pytest.mark.unit
def test_settings_from_monitoring_leaves_out_a_setting_whose_series_disagree() -> None:
    # One series per monitored database: a database's own value (ALTER DATABASE ... SET) is not the node's.
    reply = {"data": {"result": [
        {"metric": {"setting_name": "work_mem", "setting_value": "4096", "unit": "kB", "datname": "a"}},
        {"metric": {"setting_name": "work_mem", "setting_value": "65536", "unit": "kB", "datname": "b"}},
        {"metric": {"setting_name": "temp_file_limit", "setting_value": "-1", "unit": "kB", "datname": "a"}},
        {"metric": {"setting_name": "temp_file_limit", "setting_value": "-1", "unit": "kB", "datname": "b"}},
    ]}}
    settings, disagree = k004.settings_from_monitoring(reply)
    assert disagree == ["work_mem"] and list(settings) == ["temp_file_limit"]


@pytest.mark.unit
def test_settings_from_monitoring_reads_the_unit_of_either_label_form() -> None:
    # Both forms in the range (a config change): one setting, one value, the unit kept whichever comes first.
    reply = {"data": {"result": [
        {"metric": {"tag_setting_name": "work_mem", "tag_setting_value": "4096", "tag_unit": "kB"}},
        {"metric": {"setting_name": "work_mem", "setting_value": "4096"}},
    ]}}
    assert k004.settings_from_monitoring(reply) == ({"work_mem": {"setting": "4096", "unit": "kB", "context": ""}}, [])


@pytest.mark.unit
def test_counter_hours_a_small_dip_of_merged_series_is_no_reset() -> None:
    # Two series of one counter merged by their highest reading; the higher one misses one stamp.
    high = {"metric": {"s": "1"}, "values": [v for i, v in enumerate(_readings([1000, 1010, 1020, 1030])) if i != 2]}
    low = {"metric": {"s": "2"}, "values": _readings([990, 1000, 1009, 1020])}
    assert k004.counter_hours([high, low], TIMELINE)["total"] == 10 + 21


def _setting(name, value, unit, context, boot):
    return {"name": name, "setting": value, "unit": unit, "category": "c", "context": context, "vartype": "integer",
            "source": "configuration file", "boot_val": boot, "pending_restart": False}


ANSWERS = {
    "from pg_settings": [_setting("work_mem", "2184", "kB", "user", "4096"),
                         _setting("temp_file_limit", "-1", "kB", "superuser", "-1")],
    "from pg_db_role_setting": [{"role": "batch", "database": None, "config": ["work_mem=256MB", "search_path=app"]},
                                {"role": None, "database": "app", "config": ["log_temp_files=0"]}],
    "privileged_role_allowed_configs": [{"allowed": "log_temp_files, pg_stat_statements.*, auto_explain.*"}],
}


@pytest.mark.unit
def test_node_facts_reads_context_overrides_and_the_platform_list(monkeypatch: pytest.MonkeyPatch) -> None:
    out = k004.node_facts("host=h dbname=d user=u", opener=lambda c: Conn(ANSWERS))
    assert out["settings"]["work_mem"] == {"setting": "2184", "unit": "kB", "context": "user"}
    assert out["settings"]["temp_file_limit"]["context"] == "superuser"
    assert [(v["role"], v["database"], v["setting"], v["value"]) for v in out["overrides"]["values"]] == [
        ("batch", None, "work_mem", "256MB"), (None, "app", "log_temp_files", "0")]
    assert out["platform_allowed"] == ["log_temp_files", "pg_stat_statements.track"]


@pytest.mark.unit
def test_node_facts_without_a_platform_list_or_a_node(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = dict(ANSWERS, **{"from pg_db_role_setting": [], "privileged_role_allowed_configs": [{"allowed": None}]})
    out = k004.node_facts("host=h", opener=lambda c: Conn(answers))
    assert "platform_allowed" not in out and out["overrides"] == {"rows_read": 0, "values": []}
    assert k004.node_facts(None) == {"node_error": "no connection for this node in the sources file"}

    def refuse(conn_str: str) -> Any:
        raise RuntimeError("could not connect to host=h password=secret")

    assert "secret" not in k004.node_facts("host=h password=secret", opener=refuse)["node_error"]


@pytest.mark.unit
def test_node_facts_keeps_the_other_facts_when_one_read_fails() -> None:
    out = k004.node_facts("postgresql://mon:s3cr3t@db/app", opener=lambda c: Conn(ANSWERS, broken=("from pg_db_role_setting",)))
    assert "settings" in out and "overrides" not in out and "s3cr3t" not in out["overrides_error"]


# ── the generator ─────────────────────────────────────────────────────────────

WINDOW = {
    "pgwatch_pg_stat_statements_calls": {"111": [4, 4, 4], "-222": [100, 100, 100]},
    "pgwatch_pg_stat_statements_exec_time_total": {"111": [40000, 40000, 40000], "-222": [500, 500, 500]},
    "pgwatch_pg_stat_statements_temp_bytes_read": {"111": [1 << 20, 1 << 20, 1 << 20]},
}
TEXTS = {"app": {"111": "call app.rebuild($1)", "-222": "select /* pgwatch_generated */ 1 from pg_class"}}


@pytest.fixture(name="generator")
def fixture_generator(monkeypatch: pytest.MonkeyPatch) -> PostgresReportGenerator:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["app", "quiet"])

    def hourly(cluster, node_name, db_name, metric_name="", hours=24, step_s=3600, k=3):
        if db_name == "quiet":  # an idle database: its hours' top 3 rank zeros
            return {"9": [0.0, 0.0, 0.0]}, [0.0, 0.0, 0.0], TIMELINE
        return ({"111": [3 << 20, 3 << 20, 3 << 20], "-222": [1 << 20, 0, 1 << 20], "333": [0.0, 0.0, 0.0]},
                [0.0, 0.0, 0.0], TIMELINE)

    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", hourly)
    monkeypatch.setattr(gen, "get_queryid_queries_from_sink", lambda limit=655360, db_names=None, cluster=None, node_name=None: TEXTS)
    gen.instant: List[str] = []

    def query_instant(query):
        gen.instant.append(query)
        return {"status": "success", "data": {"result": [
            {"metric": {"setting_name": "work_mem", "setting_value": "4096", "unit": "kB"}}]}}

    monkeypatch.setattr(gen, "query_instant", query_instant)

    def query_range(query, start, end, step="30s"):
        if query.startswith("pgwatch_db_stats_temp_"):
            assert step == f"{k004.STEP_S}s" and 'datname="' in query
            if 'datname="quiet"' in query:
                return []  # monitoring has no reading: not known
            grows = 1 << 20 if "temp_bytes" in query else 1
            return [{"metric": {"datname": "app"},
                     "values": _readings([i * grows for i in range(37)])}]
        assert step == "3600s"
        if query.startswith("count(count by (queryid)"):
            return [{"metric": {}, "values": [[ts, "40"] for ts in TIMELINE]}]
        if query.startswith("count(resets("):
            return [{"metric": {}, "values": [[ts, "0"] for ts in TIMELINE]}]
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_calls"):
            return [{"metric": {}, "values": [[ts, "1000"] for ts in TIMELINE]}]
        metric = re.search(r"increase\((pgwatch_[a-z_]+)\{", query).group(1)
        assert "333" not in query  # a statement with no temporary files gets no window facts
        return [{"metric": {"queryid": q}, "values": [[ts, str(v)] for ts, v in zip(TIMELINE, vals)]}
                for q, vals in WINDOW.get(metric, {}).items()]

    monkeypatch.setattr(gen, "query_range", query_range)
    monkeypatch.setattr("reporter.postgres_reports.read_sources", lambda path: [])
    return gen


@pytest.mark.unit
def test_k004_rows_carry_window_facts_and_the_database_counters(generator) -> None:
    report = generator.generate_k004_temp_bytes_report("local", "node-1", time_range_minutes=180)
    validate_report(report)
    node = report["results"]["node-1"]
    app = node["data"]["app"]
    rows = {q["queryid"]: q for q in app["top_queries"]}
    assert list(rows) == ["111", "-222"]  # "333" wrote nothing: not a top statement
    assert rows["111"]["calls"] == 12 and rows["111"]["exec_time_ms"] == 120000
    assert rows["111"]["temp_bytes_read"] == 3 << 20 and rows["111"]["is_monitoring"] is False
    assert rows["-222"]["is_monitoring"] is True
    assert "temp_bytes_read" not in rows["-222"]  # no series: not known, never 0
    s = app["summary"]
    assert s["window_seconds"] == 3 * H and s["export_rule"]["top_n_per_database"] == 100
    assert s["database_temp"]["bytes"]["total"] == 36 * (1 << 20)
    assert s["database_temp"]["files"]["total"] == 36
    assert s["database_temp"]["bytes"]["hourly"] == [12 * (1 << 20)] * 3


@pytest.mark.unit
def test_k004_an_idle_database_has_no_rows_and_says_why_it_has_no_counter(generator) -> None:
    report = generator.generate_k004_temp_bytes_report("local", "node-1", time_range_minutes=180)
    quiet = report["results"]["node-1"]["data"]["quiet"]
    assert quiet["top_queries"] == []
    temp = quiet["summary"]["database_temp"]  # query_range answers [] on an error too: not known, never zero
    assert set(temp) == {"bytes_error", "files_error"} and "fewer than two readings" in temp["bytes_error"]


@pytest.mark.unit
def test_k004_settings_from_monitoring_when_the_node_is_not_reachable(generator) -> None:
    node = generator.generate_k004_temp_bytes_report("local", "node-1", time_range_minutes=180)["results"]["node-1"]
    assert node["settings"] == {"work_mem": {"setting": "4096", "unit": "kB", "context": ""}}
    assert node["node_error"] == "no connection for this node in the sources file"
    # No backslash in the PromQL string literal: a dot matches as [.].
    assert "\\" not in generator.instant[-1] and "pg_stat_statements[.]track" in generator.instant[-1]
    assert "tag_setting_name=~" in generator.instant[-1]  # older pgwatch configs keep the tag_ prefix
