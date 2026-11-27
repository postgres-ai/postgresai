"""G001 carries what CheckupLoop asked the box for (postgresai#537, platform-all#1303): every memory
setting of each node, and the window's temporary files per database."""
import re

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report


@pytest.fixture
def generator():
    return PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")


def s(value, category, unit=""):
    return {"setting": value, "unit": unit, "category": category, "context": "postmaster",
            "vartype": "integer", "pretty_value": value}


#: A PostgreSQL 17 node's settings, as A003 holds them: G001's categories and names, and others.
NODE = {
    "shared_buffers": s("16384", "Resource Usage / Memory", "8kB"),
    "work_mem": s("4096", "Resource Usage / Memory", "kB"),
    "commit_timestamp_buffers": s("256", "Resource Usage / Memory", "8kB"),
    "huge_page_size": s("0", "Resource Usage / Memory", "kB"),
    "temp_file_limit": s("-1", "Resource Usage / Disk", "kB"),
    "effective_io_concurrency": s("1", "Resource Usage / Asynchronous Behavior"),
    "bgwriter_flush_after": s("64", "Resource Usage / Background Writer", "8kB"),
    "autovacuum_naptime": s("60", "Autovacuum", "s"),
    "vacuum_cost_limit": s("200", "Resource Usage / Cost-Based Vacuum Delay"),
    "autovacuum_vacuum_cost_limit": s("-1", "Vacuuming / Automatic Vacuuming"),
    "checkpoint_timeout": s("300", "Write-Ahead Log / Checkpoints", "s"),
    "deadlock_timeout": s("1000", "Lock Management", "ms"),
    "session_preload_libraries": s("", "Client Connection Defaults / Shared Library Preloading"),
    "max_connections": s("100", "Connections and Authentication / Connection Settings"),
    # Memory settings outside those categories: by name.
    "effective_cache_size": s("524288", "Query Tuning / Planner Cost Constants", "8kB"),
    "wal_buffers": s("512", "Write-Ahead Log / Settings", "8kB"),
    "max_wal_senders": s("10", "Replication / Sending Servers"),
    "huge_pages_status": s("off", "Preset Options"),
    "track_activity_query_size": s("1024", "Statistics / Cumulative Query and Index Statistics", "B"),
    # Not G001's.
    "random_page_cost": s("4", "Query Tuning / Planner Cost Constants"),
    "log_min_duration_statement": s("-1", "Reporting and Logging / When to Log", "ms"),
    "server_version_num": s("170004", "Preset Options"),
}
NOT_G001 = {"random_page_cost", "log_min_duration_statement", "server_version_num"}


def a003(**nodes):
    return {"checkId": "A003", "results": {name: {"data": data} for name, data in nodes.items()}}


@pytest.mark.unit
def test_g001_carries_every_setting_of_its_categories_and_names(generator) -> None:
    data = generator.generate_g001_from_a003(a003(n1=NODE), node_name="n1")["results"]["n1"]["data"]
    assert set(data["settings"]) == set(NODE) - NOT_G001
    assert data["settings"]["commit_timestamp_buffers"] == NODE["commit_timestamp_buffers"]  # as A003 holds it
    # The file says what it covers, so CheckupLoop can tell it from an older file (29 fixed names).
    scope = data["settings_scope"]
    assert scope["categories"] == generator.G001_CATEGORIES
    assert set(generator.G001_SETTINGS) <= set(scope["names"]) and "max_wal_senders" in scope["names"]
    assert scope["names"] == sorted(scope["names"])


@pytest.mark.unit
def test_the_categories_are_checkup_loops_live_set(generator) -> None:
    """The pattern and names CheckupLoop's G001 live set reads, copied from platform-all
    ace/bot/checkup_loop_promql.py (_G001_RESOURCES, _G001_OTHERS, _G001_NAMES at f07ef964a): a change
    there must change these. Matched whole, as PromQL's =~ does."""
    pattern = re.compile(generator.G001_CATEGORIES)
    assert pattern.fullmatch("Resource Usage / Memory")
    assert pattern.fullmatch("Vacuuming / Cost-Based Vacuum Delay")  # PostgreSQL 18
    assert not pattern.fullmatch("Vacuuming / Freezing")
    assert not pattern.fullmatch("Autovacuum and more")
    assert generator.G001_NAMES == (
        "effective_cache_size", "wal_buffers", "shared_memory_size", "shared_memory_size_in_huge_pages",
        "huge_pages_status", "wal_decode_buffer_size", "track_activity_query_size", "max_wal_senders",
        "max_replication_slots", "max_logical_replication_workers", "max_active_replication_origins")


@pytest.mark.unit
def test_each_node_gets_its_own_values(generator) -> None:
    """Before: the filter merged every node's settings, so each node carried the last node's values."""
    other = {**NODE, "shared_buffers": s("32768", "Resource Usage / Memory", "8kB")}
    del other["huge_page_size"]
    report = a003(primary=NODE, replica=other)
    primary = generator.generate_g001_from_a003(report, node_name="primary")["results"]["primary"]["data"]
    replica = generator.generate_g001_from_a003(report, node_name="replica")["results"]["replica"]["data"]
    assert primary["settings"]["shared_buffers"]["setting"] == "16384"
    assert replica["settings"]["shared_buffers"]["setting"] == "32768"
    assert "huge_page_size" in primary["settings"] and "huge_page_size" not in replica["settings"]
    assert primary["analysis"]["estimated_total_memory_usage"]["shared_buffers_bytes"] == 128 * 1024 ** 2
    assert replica["analysis"]["estimated_total_memory_usage"]["shared_buffers_bytes"] == 256 * 1024 ** 2


@pytest.mark.unit
def test_the_direct_path_selects_the_same_settings(generator, monkeypatch) -> None:
    """Without A003, G001 reads the node's settings from monitoring and keeps the same ones."""
    rows = [{"metric": {"setting_name": name, "setting_value": v["setting"], "unit": v["unit"],
                        "category": v["category"], "context": v["context"], "vartype": v["vartype"]},
             "value": [0, "1"]} for name, v in NODE.items()]
    monkeypatch.setattr(generator, "query_instant",
                        lambda q: {"status": "success", "data": {"result": rows}})
    monkeypatch.setattr(generator, "_g001_activity", lambda c, n: {})
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *a, **k: {})
    data = generator.generate_g001_memory_settings_report("c", "n1")["results"]["n1"]["data"]
    assert set(data["settings"]) == set(NODE) - NOT_G001
    assert data["settings_scope"]["categories"] == generator.G001_CATEGORIES


def answer_by_db(values):
    return {"status": "success", "data": {"result": [
        {"metric": {"datname": db}, "value": [0, str(v)]} for db, v in values.items()]}}


NOT_KNOWN = "monitoring has fewer than two readings of the counter at least 5 minutes apart in the window"
NOT_ANSWERED = "monitoring did not answer the query"


@pytest.mark.unit
def test_activity_gives_temp_files_per_database(generator, monkeypatch) -> None:
    """protocols G001-R3: a work_mem raise needs temp-file evidence. Per database over the window; a
    database with fewer than two readings is not known, never 0."""
    queries = []

    def prom(query):
        queries.append(query)
        if query.startswith("count by (datname) (last_over_time(pgwatch_db_stats_numbackends"):
            return answer_by_db({"app": 1, "quiet": 1, "new": 1, "template1": 1})
        if "count_over_time(pgwatch_db_stats_temp_bytes" in query:
            return answer_by_db({"app": 2016, "quiet": 2016, "new": 1, "template1": 2016, "dropped": 900})
        if "clamp_min(max by (datname) (pgwatch_db_stats_temp_bytes" in query:
            return answer_by_db({"app": 1073741824.4, "quiet": 0, "new": 0, "template1": 5, "dropped": 5e10})
        if "count_over_time(pgwatch_db_stats_temp_files" in query:
            return answer_by_db({"app": 2016, "quiet": 2016, "new": 1})
        if "clamp_min(max by (datname) (pgwatch_db_stats_temp_files" in query:
            return answer_by_db({"app": 41.7, "quiet": 0, "new": 0})
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(generator, "query_instant", prom)
    out = generator._g001_activity("c", "n")
    assert out["temp_files"] == [
        {"database": "app", "bytes": 1073741824, "files": 42},
        # Dropped before the list was read: its spill still counts in the window.
        {"database": "dropped", "bytes": 50000000000, "files_error": NOT_KNOWN},
        {"database": "new", "bytes_error": NOT_KNOWN, "files_error": NOT_KNOWN},
        {"database": "quiet", "bytes": 0, "files": 0},
    ]  # template1 left out, as the databases list
    temp = [q for q in queries if "temp_" in q]
    assert len(temp) == 4 and all('cluster="c", node_name="n"' in q and "[7d" in q for q in temp)
    # Rises per step, never increase(): on VictoriaMetrics increase() counts a series that starts in
    # the window from 0 (a counter flat at 3 on a 2-day-old box read 3; checked on v1.140.0).
    assert not any("increase(" in q for q in temp)
    sel = 'cluster="c", node_name="n"'
    assert {q for q in temp if "clamp_min(" in q} == {
        generator._g001_rise(m, sel, "7d") for m in ("pgwatch_db_stats_temp_bytes", "pgwatch_db_stats_temp_files")}


@pytest.mark.unit
def test_the_rise_expression(generator) -> None:
    """Each step adds what the counter gained since the last raw sample a step earlier (the offset equals
    the step, so steps neither overlap nor leave a gap; the lookback spans the window, so a gap in
    monitoring is bridged); only series present at the step; a drop counts as 0. Fatal session ends
    are counted the same way, summed over databases."""
    m, sel = "pgwatch_db_stats_temp_files", 'cluster="c", node_name="n"'
    assert generator._g001_rise(m, sel, "7d") == (
        f'sum_over_time(clamp_min(max by (datname) ({m}{{{sel}}}) - max by (datname) '
        f'(last_over_time({m}{{{sel}}}[7d] offset 5m) and {m}{{{sel}}}), 0)[7d:5m])')
    queries = []
    generator.query_instant = lambda q: queries.append(q) or {"status": "success", "data": {"result": []}}
    generator._g001_activity("c", "n")
    fatal = [q for q in queries if "sessions_fatal" in q]
    assert fatal == [f'sum({generator._g001_rise("pgwatch_db_stats_sessions_fatal", sel, "7d")})']


@pytest.mark.unit
def test_temp_files_not_known_when_monitoring_does_not_answer(generator, monkeypatch) -> None:
    def prom(query):
        if query.startswith("count by (datname) (last_over_time(pgwatch_db_stats_numbackends"):
            return answer_by_db({"app": 1})
        if "temp_bytes" in query:
            return {"status": "error", "error": "timeout"}
        if "count_over_time(pgwatch_db_stats_temp_files" in query:
            return answer_by_db({"app": 3})
        if "clamp_min(max by (datname) (pgwatch_db_stats_temp_files" in query:
            return answer_by_db({"app": 0})
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(generator, "query_instant", prom)
    # A failed query says so: it is not "fewer than two readings".
    assert generator._g001_activity("c", "n")["temp_files"] == [
        {"database": "app", "bytes_error": NOT_ANSWERED, "files": 0}]
    # No database known: no temp_files key.
    monkeypatch.setattr(generator, "query_instant", lambda q: {"status": "success", "data": {"result": []}})
    assert "temp_files" not in generator._g001_activity("c", "n")


@pytest.mark.unit
def test_the_new_g001_file_matches_its_schema(generator, monkeypatch) -> None:
    monkeypatch.setattr(generator, "_g001_activity", lambda c, n: {
        "databases": ["app"],
        "temp_files": [{"database": "app", "bytes": 0, "files": 0},
                       {"database": "new", "bytes_error": NOT_KNOWN, "files_error": NOT_KNOWN}]})
    report = generator.generate_g001_from_a003(a003(n1=NODE), node_name="n1", cluster="c")
    validate_report(report)


@pytest.mark.unit
def test_no_scope_when_the_settings_were_not_read_whole(generator, monkeypatch) -> None:
    """settings_scope tells CheckupLoop the file has every setting of its categories, so it
    is left out when that is not known. CheckupLoop then asks the box, as for an older file."""
    def scope(a003_report, node="n1"):
        return "settings_scope" in generator.generate_g001_from_a003(a003_report, node_name=node)["results"][node]["data"]

    assert scope(a003(n1=NODE))
    assert not scope(a003(n1={}))  # nothing read
    assert not scope(a003(n1={**NODE, "work_mem": {"setting": "4096", "unit": "kB", "category": "Other"}}))
    assert not scope(a003(n1={**NODE, "temp_buffers": {"setting": "1024"}}))  # a row without its category
    # A003 without this node: the other nodes' values, as before, never claimed as this node's whole set.
    assert not scope(a003(primary=NODE, other=NODE), node="replica")
    # The direct path: an empty or failed read writes no scope either.
    monkeypatch.setattr(generator, "_g001_activity", lambda c, n: {})
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *a, **k: {})
    monkeypatch.setattr(generator, "query_instant", lambda q: {"status": "error", "error": "timeout"})
    assert "settings_scope" not in generator.generate_g001_memory_settings_report("c", "n1")["results"]["n1"]["data"]


@pytest.mark.unit
def test_no_scope_when_a_monitoring_row_has_no_category(generator, monkeypatch) -> None:
    """The direct path gives a row without a category label 'Memory': the setting is kept by name, and
    the file does not claim to hold every setting of its categories."""
    rows = [{"metric": {"setting_name": name, "setting_value": v["setting"], "unit": v["unit"],
                        **({} if name == "work_mem" else {"category": v["category"]})},
             "value": [0, "1"]} for name, v in NODE.items()]
    monkeypatch.setattr(generator, "query_instant", lambda q: {"status": "success", "data": {"result": rows}})
    monkeypatch.setattr(generator, "_g001_activity", lambda c, n: {})
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *a, **k: {})
    data = generator.generate_g001_memory_settings_report("c", "n1")["results"]["n1"]["data"]
    assert "work_mem" in data["settings"] and "settings_scope" not in data


@pytest.mark.unit
@pytest.mark.parametrize("failing", ["count_over_time(pgwatch_db_stats_temp_bytes",
                                     "clamp_min(max by (datname) (pgwatch_db_stats_temp_bytes"])
def test_temp_files_one_failed_query_of_the_pair(generator, monkeypatch, failing) -> None:
    """Either query of a pair failing (query_instant answers {} on an HTTP error) makes that key not
    answered; the other key is still read. Two readings with a rise are enough for a value."""
    def prom(query):
        if query.startswith("count by (datname) (last_over_time(pgwatch_db_stats_numbackends"):
            return answer_by_db({"app": 1})
        if failing in query:
            return {}
        if "count_over_time(" in query:
            return answer_by_db({"app": 2})
        if "clamp_min(" in query:
            return answer_by_db({"app": 7})
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(generator, "query_instant", prom)
    assert generator._g001_activity("c", "n")["temp_files"] == [
        {"database": "app", "bytes_error": NOT_ANSWERED, "files": 7}]
