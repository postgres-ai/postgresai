"""K006–K008 and the chart data of A005 and K006–K008: what CheckupLoop asked the box for (postgresai#534)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pytest

from reporter import k004, kblocks
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

H = 3600
TIMELINE = [H * h for h in range(470_000, 470_003)]  # three hours, each stamp the end of its hour
END = TIMELINE[-1]
PG = {"version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"}
TEXTS = {"app": {"111": "select * from orders where customer_id = $1", "-222": "update stock set n = n - $1"}}
# Per statement, per hour of the window: the K003-style facts (the sum is the window's value).
WINDOW = {
    "pgwatch_pg_stat_statements_calls": {"111": [4, 4, 4], "-222": [10, 10, 10]},
    "pgwatch_pg_stat_statements_exec_time_total": {"111": [40000, 40000, 40000], "-222": [100, 100, 100]},
    "pgwatch_pg_stat_statements_rows": {"111": [8, 8, 8], "-222": [10, 10, 10]},
    "pgwatch_pg_stat_statements_shared_bytes_hit_total": {"111": [1 << 20] * 3, "-222": [1 << 10] * 3},
    "pgwatch_pg_stat_statements_shared_bytes_read_total": {"111": [3 << 20] * 3, "-222": [0, 0, 0]},
    "pgwatch_pg_stat_statements_block_read_total": {"111": [900, 900, 900]},
    "pgwatch_pg_stat_statements_wal_bytes": {"-222": [512, 512, 512]},
}


def _readings(values: List[float], start: int = TIMELINE[0] - H, step: int = k004.STEP_S) -> List[List[Any]]:
    return [[start + i * step, str(v)] for i, v in enumerate(values)]


def _ok(result: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"status": "success", "data": {"result": result}}


class FakeMonitoring:
    """Prometheus answers for one node: `fail` names metrics whose queries fail (an empty range
    reply, a failed instant reply), as a broken or filtered metrics store answers."""

    def __init__(self, timing: str = "on", timing_before: Optional[str] = None, reset: Optional[float] = 1_690_000_000.0,
                 reset_label: str = "database_name", before_at: int = END - 2 * H):
        self.timing, self.timing_before, self.reset, self.reset_label = timing, timing_before, reset, reset_label
        self.before_at = before_at
        self.fail: set = set()
        self.instant: List[str] = []
        self.ranges: List[str] = []

    def settings(self, lookback_h: int) -> List[Dict[str, Any]]:
        """tlast_over_time: each series' value is the time of its last reading; a replaced value has its
        own series, seen only by a lookback that reaches its last reading."""
        rows = [{"metric": {"setting_name": "block_size", "setting_value": "8192", "unit": "8kB"}, "value": [END, END - 60]},
                {"metric": {"setting_name": "track_io_timing", "setting_value": self.timing}, "value": [END, END - 50]}]
        if self.timing_before and self.before_at >= END - lookback_h * H:
            rows.append({"metric": {"setting_name": "track_io_timing", "setting_value": self.timing_before},
                         "value": [END, self.before_at]})
        return rows

    def query_instant(self, query: str) -> Dict[str, Any]:
        self.instant.append(query)
        ours = (kblocks.SETTING_PATTERN, "pgwatch_db_size_size_b", "pgwatch_stats_reset_stats_reset_epoch", "up ")
        if not any(m in query or query.startswith(m) for m in ours):
            return _ok([])  # another report's query (generate_all_reports)
        assert query.endswith(f" @ {END}"), query  # evaluated at the window's end, not at the call
        if any(m in query for m in self.fail):
            return {}
        if kblocks.SETTING_PATTERN in query:
            operands = query.split(" or ")
            assert len(operands) == 2 and all(o.startswith("tlast_over_time(") and o.endswith(f" @ {END}")
                                              for o in operands), query  # each operand at the window's end
            return _ok(self.settings(int(re.search(r"\[(\d+)h\]", query).group(1))))
        if "pgwatch_db_size_size_b" in query:
            return _ok([{"metric": {"datname": "app"}, "value": [END, "123456789"]}])
        if "pgwatch_stats_reset_stats_reset_epoch" in query:
            if f'{self.reset_label}="app"' not in query or self.reset is None:
                return _ok([])  # the metric has a row only after a reset
            return _ok([{"metric": {self.reset_label: "app"}, "value": [END, str(self.reset)]}])
        if query.startswith("up "):
            return _ok([{"metric": {"job": "victoriametrics", "instance": "localhost:9090"}, "value": [END, "1"]},
                        {"metric": {"job": "pgwatch-prometheus", "instance": "pgwatch-prometheus:9091"},
                         "value": [END, "1"]},
                        {"metric": {"job": "self-cadvisor", "instance": "self-cadvisor:8080"}, "value": [END, "0"]}])
        return _ok([])

    def query_range(self, query: str, start, end, step: str = "30s") -> List[Dict[str, Any]]:
        self.ranges.append(query)
        if any(m in query for m in self.fail):
            return []  # query_range answers [] on an error
        if query.startswith("pgwatch_db_stats_blks_"):
            assert step == f"{k004.STEP_S}s" and int(start.timestamp()) == TIMELINE[0] - H
            grows = 100 if "blks_read" in query else 1000
            return [{"metric": {"datname": "app"}, "values": _readings([i * grows for i in range(37)])}]
        if query.startswith("pgwatch_db_stats_deadlocks"):
            assert 'datname="' not in query  # every database of the node
            one = [0] * 20 + [1] * 17  # one deadlock in the second hour
            return [{"metric": {"datname": "app"}, "values": _readings(one)},
                    {"metric": {"datname": "postgres"}, "values": _readings([0] * 37)}]
        assert step == "3600s", query
        if query.startswith("count(count by (queryid)"):
            return [{"metric": {}, "values": [[ts, "40"] for ts in TIMELINE]}]
        if query.startswith("count(resets("):
            return [{"metric": {}, "values": [[ts, "0"] for ts in TIMELINE]}]
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_calls"):
            return [{"metric": {}, "values": [[ts, "1000"] for ts in TIMELINE]}]
        metric = re.search(r"increase\((pgwatch_[a-z_]+)\{", query).group(1)
        out = []
        for qid, vals in WINDOW.get(metric, {}).items():
            pts = [[ts, str(v)] for ts, v in zip(TIMELINE, vals)]
            if metric == "pgwatch_pg_stat_statements_calls" and qid == "-222":
                pts = pts[1:]  # not exported in the first hour
            out.append({"metric": {"queryid": qid}, "values": pts})
        return out


def _generator(monkeypatch: pytest.MonkeyPatch, fake: FakeMonitoring) -> PostgresReportGenerator:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    gen._window_end_s = END  # the run's window
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["app"])
    ranked = {"111": [3 << 20] * 3, "-222": [1 << 10] * 3}

    def hourly(*a, **k):
        return ranked, [0.0, 0.0, 0.0], TIMELINE

    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", hourly)
    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data_sum2", hourly)
    monkeypatch.setattr(gen, "get_queryid_queries_from_sink", lambda limit=655360, db_names=None: TEXTS)
    monkeypatch.setattr(gen, "query_instant", fake.query_instant)
    monkeypatch.setattr(gen, "query_range", fake.query_range)
    monkeypatch.setattr("reporter.postgres_reports.read_sources", lambda path: [])
    return gen


GENERATORS = {"K006": "generate_k006_shared_read_report", "K007": "generate_k007_shared_hit_report",
              "K008": "generate_k008_shared_hit_read_report"}


def _report(gen: PostgresReportGenerator, check_id: str) -> Dict[str, Any]:
    report = getattr(gen, GENERATORS[check_id])("local", "node-1", time_range_minutes=180)
    validate_report(report)
    return report


# ── S1–S3: the statements, the database and the node ────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("check_id", ["K006", "K007", "K008"])
def test_rows_carry_k003_facts_their_time_and_calls_per_hour(monkeypatch, check_id) -> None:
    report = _report(_generator(monkeypatch, FakeMonitoring()), check_id)
    rows = {q["queryid"]: q for q in report["results"]["node-1"]["data"]["app"]["top_queries"]}
    one = rows["111"]
    assert one["calls"] == 12 and one["exec_time_ms"] == 120000 and one["rows"] == 24
    assert one["shared_blks_hit_bytes"] == 3 << 20 and one["shared_blks_read_bytes"] == 9 << 20
    assert one["query_text"].startswith("select * from orders") and one["relations"] == ["orders"]
    assert one["hourly_calls"] == [4, 4, 4]
    assert "hourly_calls_error" not in report["results"]["node-1"]["data"]["app"]["summary"]
    assert rows["-222"]["hourly_calls"] == [None, 10, 10]  # not exported in the first hour: not 0
    assert "wal_bytes" not in one and rows["-222"]["wal_bytes"] == 1536  # no series: not known, never 0
    assert ("blk_read_time_ms" in one) is (check_id == "K006")
    if check_id == "K006":
        assert one["blk_read_time_ms"] == 2700


@pytest.mark.unit
@pytest.mark.parametrize("check_id", ["K006", "K007", "K008"])
def test_the_summary_carries_the_window_and_the_database(monkeypatch, check_id) -> None:
    s = _report(_generator(monkeypatch, FakeMonitoring()), check_id)["results"]["node-1"]["data"]["app"]["summary"]
    assert s["window_seconds"] == 3 * H and s["total_calls"] == 3000 and s["statements_exported_max"] == 40
    assert s["database_blocks"]["read"]["total"] == 3600 and s["database_blocks"]["read"]["hourly"] == [1200.0] * 3
    assert s["database_blocks"]["hit"]["total"] == 36000
    assert s["database_size_bytes"] == 123456789
    assert s["stats_reset"] == "2023-07-22T04:26:40+00:00"


@pytest.mark.unit
@pytest.mark.parametrize("check_id", ["K006", "K007", "K008"])
def test_the_node_carries_its_settings_from_monitoring(monkeypatch, check_id) -> None:
    fake = FakeMonitoring()
    node = _report(_generator(monkeypatch, fake), check_id)["results"]["node-1"]
    assert node["settings"] == {"block_size": {"setting": "8192", "unit": "8kB"},
                                "track_io_timing": {"setting": "on", "unit": "", "changed_in_window": False}}
    settings = [q for q in fake.instant if "pgwatch_settings_configured" in q]
    assert settings and all('setting_name=~"^(block_size|track_io_timing)$"' in q for q in settings)


# ── S4, S5: the chart data, one window per run ────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("check_id", ["K006", "K007", "K008"])
def test_k_reports_carry_scrape_targets_and_deadlocks(monkeypatch, check_id) -> None:
    report = _report(_generator(monkeypatch, FakeMonitoring()), check_id)
    assert report["scrape_targets"] == [
        {"job": "pgwatch-prometheus", "instance": "pgwatch-prometheus:9091", "up": 1},
        {"job": "self-cadvisor", "instance": "self-cadvisor:8080", "up": 0},
        {"job": "victoriametrics", "instance": "localhost:9090", "up": 1}]
    deadlocks = report["results"]["node-1"]["deadlocks"]
    assert deadlocks["app"]["total"] == 1 and deadlocks["app"]["hourly"] == [0.0, 1.0, 0.0]
    assert deadlocks["postgres"]["total"] == 0  # measured zero, kept


@pytest.mark.unit
def test_a005_gets_the_same_chart_data_beside_its_catalog_facts(monkeypatch) -> None:
    fake = FakeMonitoring()
    gen = _generator(monkeypatch, fake)
    k006 = gen.generate_k006_shared_read_report("local", "node-1", time_range_minutes=1440)  # the run's 24 hours
    validate_report(k006)
    a005 = gen.generate_a005_extensions_report("local", "node-1")
    validate_report(a005)
    node = a005["results"]["node-1"]
    assert node["deadlocks"] == k006["results"]["node-1"]["deadlocks"]
    # The hours those counts are for (the end of each, the run's window; platform-all#1306)
    stamps = node["hourly_timestamps"]
    assert stamps == gen._run_timeline() and stamps[-1] == gen._run_window_end()
    assert k006["results"]["node-1"]["hourly_timestamps"] == stamps  # a K node too, data or not
    assert all(len(window["hourly"]) == len(stamps) for window in node["deadlocks"].values())
    assert a005["scrape_targets"] == k006["scrape_targets"]
    assert "deadlocks" not in node["data"]  # A005's data is the CLI's twin: unchanged
    # Computed once per run: one deadlock range query and one `up` query for all files.
    assert sum(q.startswith("pgwatch_db_stats_deadlocks") for q in fake.ranges) == 1
    assert sum(q.startswith("up ") for q in fake.instant) == 1


@pytest.mark.unit
def test_one_run_has_one_window_even_across_an_hour_boundary(monkeypatch) -> None:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    clock = iter([10 * H - 5, 10 * H + 5])  # the run starts at 09:59:55 and goes on past 10:00
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: next(clock))
    assert gen._run_window_end() == 9 * H
    assert gen._run_window_end() == 9 * H and gen._run_timeline(3) == [7 * H, 8 * H, 9 * H]


@pytest.mark.unit
def test_each_cluster_of_a_run_starts_its_own_window(monkeypatch) -> None:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    gen._window_end_s, gen._chart_cache = 9 * H, {("scrape_targets", "c1"): ([], None)}  # cluster 1's

    def stop(cluster):
        raise RuntimeError("stop after the reset")

    monkeypatch.setattr(gen, "get_all_nodes", stop)
    with pytest.raises(RuntimeError):
        gen.generate_all_reports("c2")
    assert gen._window_end_s is None and gen._chart_cache == {}


def _all_reports(monkeypatch, nodes: Dict[str, Any]) -> tuple:
    fake = FakeMonitoring()
    gen = _generator(monkeypatch, fake)
    monkeypatch.setattr(gen, "get_all_nodes", lambda cluster: nodes)
    keep = {"generate_a005_extensions_report", "generate_k006_shared_read_report",
            "generate_k007_shared_hit_report", "generate_k008_shared_hit_read_report"}
    for name in dir(gen):
        if name.startswith("generate_") and name.endswith("_report") and name not in keep:
            monkeypatch.setattr(gen, name, lambda *a, **k: {"results": {}})
    reports = gen.generate_all_reports("local")
    gen._window_end_s = END  # generate_all_reports starts a new window: keep the fake's hours
    return gen, fake, reports


@pytest.mark.unit
@pytest.mark.parametrize("nodes", [{"primary": "p", "standbys": ["s"]}, {"primary": "p", "standbys": []}],
                         ids=["primary and standby", "single node"])
def test_a_run_gives_every_chart_file_the_same_scrape_targets(monkeypatch, nodes) -> None:
    monkeypatch.setattr(PostgresReportGenerator, "_run_window_end", lambda self: END)
    gen, fake, reports = _all_reports(monkeypatch, nodes)
    targets = [reports[cid].get("scrape_targets") for cid in ("A005", "K006", "K007", "K008")]
    assert targets[0] and all(t == targets[0] for t in targets)  # A005 is rebuilt from its nodes: attached again
    for cid in ("A005", "K006", "K007", "K008"):
        validate_report(reports[cid])
        assert set(reports[cid]["results"]) == {"p", *nodes["standbys"]}
        for node in reports[cid]["results"].values():
            assert node["deadlocks"]["app"]["total"] == 1
            assert cid == "A005" or node["settings"]["track_io_timing"]["setting"] == "on"
    assert sum(q.startswith("up ") for q in fake.instant) == 1


@pytest.mark.unit
def test_the_top_k_timeline_is_the_run_window_even_after_the_clock_moves(monkeypatch) -> None:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    clock = iter([10 * H - 5] + [11 * H + 5] * 10)  # the run starts at 09:59:55 and goes on past 10:00
    monkeypatch.setattr("reporter.postgres_reports.time.time", lambda: next(clock))
    monkeypatch.setattr(gen, "query_range", lambda *a, **k: [])
    assert gen._run_window_end() == 9 * H
    _, _, timeline = gen._get_hourly_topk_pgss_data_multi("c", "n", "db", ["pgwatch_pg_stat_statements_calls"], hours=3)
    assert timeline == gen._run_timeline(3) == [7 * H, 8 * H, 9 * H]


@pytest.mark.unit
def test_a_combined_report_keeps_the_scrape_targets(monkeypatch) -> None:
    gen = _generator(monkeypatch, FakeMonitoring())
    combined = gen.format_report_data("K006", {"node-1": {"data": {}}}, "node-1")
    assert "scrape_targets" not in combined
    assert gen._attach_scrape_targets(combined)["scrape_targets"][0]["job"] == "pgwatch-prometheus"


# ── S6: track_io_timing ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_k006_has_no_read_time_when_track_io_timing_is_off(monkeypatch) -> None:
    fake = FakeMonitoring(timing="off")
    node = _report(_generator(monkeypatch, fake), "K006")["results"]["node-1"]
    assert node["settings"]["track_io_timing"]["setting"] == "off"
    assert all("blk_read_time_ms" not in q for q in node["data"]["app"]["top_queries"])
    assert not any("block_read_total" in q for q in fake.ranges)  # not even asked


@pytest.mark.unit
def test_track_io_timing_changed_within_the_window_is_flagged_and_the_read_time_kept(monkeypatch) -> None:
    node = _report(_generator(monkeypatch, FakeMonitoring(timing="on", timing_before="off")), "K006")["results"]["node-1"]
    assert node["settings"]["track_io_timing"]["changed_in_window"] is True
    assert node["data"]["app"]["top_queries"][0]["blk_read_time_ms"] == 2700


@pytest.mark.unit
def test_k006_has_no_read_time_when_the_setting_is_not_known(monkeypatch) -> None:
    fake = FakeMonitoring()
    fake.fail = {"pgwatch_settings_configured"}
    node = _report(_generator(monkeypatch, fake), "K006")["results"]["node-1"]
    assert node["settings_error"] == "the monitoring query failed" and "settings" not in node
    assert all("blk_read_time_ms" not in q for q in node["data"]["app"]["top_queries"])


# ── S7, S13: a fact that cannot be read ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("metric,key", [
    ("pgwatch_db_stats_blks_read", ("database_blocks", "read_error")),
    ("pgwatch_db_size_size_b", ("database_size_error",)),
    ("pgwatch_stats_reset_stats_reset_epoch", ("stats_reset_error",)),
])
def test_one_failed_metric_is_left_out_with_its_reason_and_the_rest_written(monkeypatch, metric, key) -> None:
    fake = FakeMonitoring()
    fake.fail = {metric}
    report = _report(_generator(monkeypatch, fake), "K006")
    s = report["results"]["node-1"]["data"]["app"]["summary"]
    value = s
    for part in key:
        value = value[part]
    assert isinstance(value, str) and value
    assert s["database_blocks"].get("hit", {}).get("total") == 36000  # the others are written
    assert report["scrape_targets"] and report["results"]["node-1"]["deadlocks"]
    assert 0 not in (s.get("database_size_bytes"), s.get("stats_reset"))  # never a zero for "not known"


@pytest.mark.unit
def test_chart_data_that_cannot_be_read_says_why(monkeypatch) -> None:
    fake = FakeMonitoring()
    fake.fail = {"pgwatch_db_stats_deadlocks", "up "}
    report = _report(_generator(monkeypatch, fake), "K007")
    assert report["scrape_targets_error"] == "the monitoring query failed" and "scrape_targets" not in report
    assert report["results"]["node-1"]["deadlocks_error"] == kblocks.NO_COUNTER


# ── S12: statistics reset ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_never_reset_is_null_where_this_database_is_collected(monkeypatch) -> None:
    s = _report(_generator(monkeypatch, FakeMonitoring(reset=None)), "K008")["results"]["node-1"]["data"]["app"]["summary"]
    assert s["stats_reset"] is None and "stats_reset_error" not in s


@pytest.mark.unit
def test_a_database_monitoring_does_not_collect_is_not_known_never_null(monkeypatch) -> None:
    # pgwatch reads pg_stat_database of the database it connects to: another database of the node has
    # no blocks, no size and no reset row, and "never reset" would be a guess.
    fake = FakeMonitoring(reset=None)
    fake.fail = {"pgwatch_db_stats_blks_", "pgwatch_db_size_size_b"}
    s = _report(_generator(monkeypatch, fake), "K006")["results"]["node-1"]["data"]["app"]["summary"]
    assert "stats_reset" not in s and "no statistics of this database" in s["stats_reset_error"]


@pytest.mark.unit
def test_stats_reset_from_the_older_datname_label_and_a_zero_epoch(monkeypatch) -> None:
    s = _report(_generator(monkeypatch, FakeMonitoring(reset_label="datname")), "K006")["results"]["node-1"]["data"]["app"]["summary"]
    assert s["stats_reset"] == "2023-07-22T04:26:40+00:00"
    s = _report(_generator(monkeypatch, FakeMonitoring(reset=0)), "K006")["results"]["node-1"]["data"]["app"]["summary"]
    assert s["stats_reset"] is None and "stats_reset_error" not in s


@pytest.mark.unit
def test_blocks_after_a_reset_count_the_reading_after_it_never_a_negative() -> None:
    before = _readings([1000, 1100, 1200])
    after = _readings([5, 25], start=TIMELINE[0] - H + 3 * k004.STEP_S)  # pg_stat_reset(): back near zero
    window = k004.counter_hours([{"metric": {"datname": "app"}, "values": before + after}], TIMELINE)
    assert window["total"] == 100 + 100 + 5 + 20 and min(window["hourly"]) >= 0


# ── S10, S11: compatibility and size ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("check_id", ["K006", "K007", "K008", "A005"])
def test_a_file_of_the_released_reporter_is_still_valid(monkeypatch, check_id) -> None:
    gen = _generator(monkeypatch, FakeMonitoring())
    if check_id == "A005":
        report = gen.generate_a005_extensions_report("local", "node-1")
        report["results"]["node-1"].pop("deadlocks", None)
        report.pop("scrape_targets", None)
    else:
        report = _report(gen, check_id)
        node = report["results"]["node-1"]
        for key in ("settings", "settings_error", "deadlocks", "deadlocks_error"):
            node.pop(key, None)
        report.pop("scrape_targets", None)
        app = node["data"]["app"]
        app["summary"] = {k: v for k, v in app["summary"].items()
                          if k in ("queries_returned", "time_range_hours", "hourly_timestamps", "limit")
                          or k.startswith("total_")}
        app["top_queries"] = [{k: v for k, v in q.items() if k == "queryid" or k.startswith(("total_", "hourly_s"))}
                              for q in app["top_queries"]]
    report["contract_version"] = "1.1.0"
    validate_report(report)


#: The bytes one K006 row may take with every new fact at its largest (a 200-character text, three
#: relations, 24 hours of calls): the input for CheckupLoop's size limit (platform-all#1306).
K006_ROW_BYTES_MAX = 1700


@pytest.mark.unit
def test_k006_size_per_row_and_the_worst_case_file() -> None:
    row = {"queryid": "-9223372036854775808", "total_shared_read_bytes": 1.2345678901234567e15,
           "hourly_shared_read_bytes": [1.2345678901234567e13] * 24}
    row.update({key: 1.2345678901234567e15 for key in ("calls", "exec_time_ms", "blk_read_time_ms", "rows",
                                                         "shared_blks_hit_bytes", "shared_blks_read_bytes",
                                                         "wal_bytes", "temp_bytes_written")})
    row.update(hourly_calls=[123456789.0] * 24, query_text="x" * 200, is_monitoring=False,
               relations=["schema_name.table_name_" + str(i) for i in range(3)])
    size = len(json.dumps(row, separators=(",", ":")))
    assert size <= K006_ROW_BYTES_MAX, size
    # 3 databases × 72 rows (the hourly top 3 union of 24 hours), as compact JSON.
    worst = 3 * 72 * size
    assert worst < 400_000  # recorded for #1306: the bot's new limit must exceed it


# ── the parts ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_scrape_targets_and_settings_parsers() -> None:
    assert kblocks.scrape_targets({}) == (None, "the monitoring query failed")
    assert kblocks.scrape_targets(_ok([])) == (None, "monitoring has no scrape targets")
    out = kblocks.settings_facts(_ok([{"metric": {"tag_setting_name": "track_io_timing", "tag_setting_value": "on"},
                                       "value": [1, "1000"]},
                                      {"metric": {"setting_name": "block_size", "setting_value": "8192"}, "value": [1, "1000"]},
                                      {"metric": {"setting_name": "block_size", "setting_value": "16384"},
                                       "value": [1, "990"]}]),  # read at once (one value per database)
                                 {})
    assert out == {"settings": {"track_io_timing": {"setting": "on", "unit": ""}}}  # block_size disagrees: left out
    assert kblocks.timing_on(out["settings"]) is True and kblocks.timing_on({}) is None
    assert kblocks.hourly_values({3600: 2.0}, [3600, 7200]) == [2.0, None]
    # --use-current-time: stamps at :17:33, the store's reply on the hour: the nearest reading counts.
    assert kblocks.hourly_values({3600: 2.0, 7200: 3.0}, [3600 + 1053, 7200 + 1053]) == [2.0, 3.0]
    up = _ok([{"metric": {"job": "a", "instance": "x", "cluster": "other"}, "value": [1, "1"]},
              {"metric": {"job": "b", "instance": "y"}, "value": [1, "1"]}])
    assert kblocks.scrape_targets(up, "mine")[0] == [{"job": "b", "instance": "y", "up": 1}]
    assert kblocks.deadlocks_by_database([{"metric": {"datname": "x"}, "values": [[1, "1"]]}], TIMELINE) == {}


@pytest.mark.unit
def test_settings_are_read_in_either_label_form(monkeypatch) -> None:
    fake = FakeMonitoring()
    _report(_generator(monkeypatch, fake), "K006")
    settings = [q for q in fake.instant if kblocks.SETTING_PATTERN in q]
    assert settings and all('setting_name=~' in q and 'tag_setting_name=~' in q and " or " in q for q in settings)


@pytest.mark.unit
def test_a_failed_hourly_calls_query_says_why(monkeypatch) -> None:
    fake = FakeMonitoring()
    fake.fail = {"sum by (queryid) (increase(pgwatch_pg_stat_statements_calls"}
    s = _report(_generator(monkeypatch, fake), "K007")["results"]["node-1"]["data"]["app"]["summary"]
    assert s["hourly_calls_error"] == "the monitoring query failed or answered no series"


@pytest.mark.unit
def test_a_change_hours_ago_is_seen_by_the_window_not_by_the_value_now(monkeypatch) -> None:
    # A 24-hour run, the replaced value read last 10 hours back: outside "now" (3 h), inside the window.
    fake = FakeMonitoring(timing="on", timing_before="off", before_at=END - 10 * H)
    gen = _generator(monkeypatch, fake)
    report = gen.generate_k006_shared_read_report("local", "node-1", time_range_minutes=1440)
    timing = report["results"]["node-1"]["settings"]["track_io_timing"]
    assert timing == {"setting": "on", "unit": "", "changed_in_window": True}
    lookbacks = sorted(re.search(r"\[(\d+)h\]", q).group(1) for q in fake.instant if kblocks.SETTING_PATTERN in q)
    assert lookbacks == ["24", "3"]


@pytest.mark.unit
@pytest.mark.parametrize("failing", ["pgwatch_db_size_size_b", "pgwatch_db_stats_blks_"])
def test_never_reset_needs_only_one_of_this_databases_readings(monkeypatch, failing) -> None:
    fake = FakeMonitoring(reset=None)
    fake.fail = {failing}
    s = _report(_generator(monkeypatch, fake), "K006")["results"]["node-1"]["data"]["app"]["summary"]
    assert s["stats_reset"] is None and "stats_reset_error" not in s


@pytest.mark.unit
def test_stats_reset_is_looked_up_over_the_window(monkeypatch) -> None:
    fake = FakeMonitoring()
    gen = _generator(monkeypatch, fake)
    day = [END - (23 - i) * H for i in range(24)]  # a 24-hour window: the lookback is the window, not 3 h
    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", lambda *a, **k: ({"111": [1.0] * 24}, [0.0] * 24, day))
    gen.generate_k006_shared_read_report("local", "node-1", time_range_minutes=1440)
    resets = [q for q in fake.instant if q.startswith("last_over_time(pgwatch_stats_reset_stats_reset_epoch")]
    assert resets and all("[24h]" in q for q in resets)


@pytest.mark.unit
def test_a_database_with_no_ranked_statement_has_no_hourly_calls_error(monkeypatch) -> None:
    gen = _generator(monkeypatch, FakeMonitoring())
    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", lambda *a, **k: ({}, [5.0, 5.0, 5.0], TIMELINE))
    s = _report(gen, "K007")["results"]["node-1"]["data"]["app"]
    assert s["top_queries"] == [] and "hourly_calls_error" not in s["summary"]


# ── A005 database sizes (platform-all#1306) ───────────────────────────────────────


@pytest.mark.unit
def test_a005_carries_each_databases_size_at_the_window_end(monkeypatch) -> None:
    fake = FakeMonitoring()
    gen = _generator(monkeypatch, fake)
    original = fake.query_instant

    def sizes(query):
        if query.startswith("last_over_time(pgwatch_db_size_size_b") and 'datname="' not in query:
            assert query.endswith(f" @ {END}")
            return _ok([{"metric": {"datname": "app"}, "value": [END, "300"]},
                        {"metric": {"datname": "app"}, "value": [END, "200"]},  # a renamed source: the largest
                        {"metric": {"datname": "postgres"}, "value": [END, "100"]}])
        return original(query)

    monkeypatch.setattr(gen, "query_instant", sizes)
    a005 = gen.generate_a005_extensions_report("local", "node-1")
    validate_report(a005)
    node = a005["results"]["node-1"]
    assert node["database_sizes"] == {"app": 300.0, "postgres": 100.0}
    assert "database_sizes" not in node["data"]


@pytest.mark.unit
def test_a005_sizes_that_cannot_be_read_say_why(monkeypatch) -> None:
    fake = FakeMonitoring()
    fake.fail = {"pgwatch_db_size_size_b"}
    a005 = _generator(monkeypatch, fake).generate_a005_extensions_report("local", "node-1")
    validate_report(a005)
    node = a005["results"]["node-1"]
    assert node["database_sizes_error"] == "the monitoring query failed" and "database_sizes" not in node
