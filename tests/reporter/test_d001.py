"""D001 on a monitoring box: the logging settings with their context, the role and database
values, the login roles, the longest calls per role from the node, and the temporary files
and autovacuum runs of a window from monitoring (reporter/d001.py)."""
import pytest
from psycopg2 import sql as pgsql

from reporter import d001
from reporter.report_schemas import validate_report
from tests.reporter.test_g003 import Box, Conn


def counter(*values, start=1_000_000.0, step=3600, **labels):
    return {"metric": labels or {"datname": "app"},
            "values": [[start + i * step, str(v)] for i, v in enumerate(values)]}


def test_the_window_counts_rises_never_the_first_value_or_a_reset():
    assert d001.counter_window([counter(100, 100, 103, 0, 2)])["total"] == 5
    assert d001.counter_window([counter(7)]) is None
    assert d001.counter_window([]) is None


def test_series_merge_by_the_given_labels_and_add_up_otherwise():
    a = counter(1, 2, datname="app", sys_id="1")
    b = counter(4, 6, start=1_000_000.0 + 7200, datname="app", sys_id="2")
    assert d001.counter_window([a, b], labels=("datname",))["total"] == 5  # one counter, 1 -> 6
    tables = [counter(0, 1, relname="a"), counter(5, 7, relname="b")]
    assert d001.counter_window(tables)["total"] == 3


def test_a_role_row_keeps_logging_settings_and_the_statement_limit_only():
    assert d001._config(["search_path=a", "log_min_duration_statement=1s", "statement_timeout=3s",
                         "auto_explain.log_min_duration=1s", "work_mem=1GB"], d001.OVERRIDE_PATTERN) == {
        "log_min_duration_statement": "1s", "statement_timeout": "3s", "auto_explain.log_min_duration": "1s"}


def test_thresholds_add_the_servers_own_positive_ones():
    settings = {"log_min_duration_statement": {"setting": "250", "unit": "ms"},
                "auto_explain.log_min_duration": {"setting": "-1", "unit": "ms"}}
    assert d001.thresholds_of(settings) == [250, 1000, 10000]


SETTINGS = [
    {"name": "log_min_duration_statement", "setting": "-1", "unit": "ms", "category": "Reporting and Logging / When to Log",
     "context": "superuser", "vartype": "integer", "source": "default", "boot_val": "-1", "pending_restart": False},
    {"name": "auto_explain.log_min_duration", "setting": "10000", "unit": "ms", "category": "Customized Options",
     "context": "superuser", "vartype": "integer", "source": "configuration file", "boot_val": "-1", "pending_restart": False},
]
ANSWERS = {
    "current_user as role": [{"role": "mon", "database": "app"}],
    "from pg_settings": SETTINGS,
    "from pg_db_role_setting": [
        {"role": "app", "database": None, "config": ["statement_timeout=8s", "search_path=public"]},
        {"role": "admin", "database": None, "config": ["log_statement=none"]},
        {"role": None, "database": "app", "config": ["app.x=1"]}],
    "r.rolcanlogin": [{"role": "app", "superuser": False, "member_of": [], "sessions": 2,
                       "oldest_session_start": None, "applications": ["api"]}],
    "from pg_extension": [{"schema": "extensions"}],
    "with s as": [{"role": "app", "groups": 40, "calls": 900, "longest_ms": 12_500.0, "stats_since": "2026-10-01",
                   "groups_at_or_over": {"1000": 3, "10000": 1}, "calls_of_groups_at_or_over": {"1000": 70, "10000": 2}}],
    "pg_stat_statements_info": [{"dealloc": 0, "stats_reset": "2026-10-01"}],
}


@pytest.fixture(autouse=True)
def composed_sql_without_a_server(monkeypatch):
    """A composed query needs a real connection to become text; the fake answers by fragments."""
    monkeypatch.setattr(pgsql.Composed, "as_string", lambda self, context=None: repr(self))


def facts(conn):
    return d001.catalog_facts("postgresql://mon:s3cr3t@db/app", opener=lambda _dsn: conn)


def test_catalog_facts_read_settings_overrides_roles_and_calls():
    out = facts(Conn(ANSWERS))
    assert out["settings"]["log_min_duration_statement"]["context"] == "superuser"
    assert out["settings"]["log_min_duration_statement"]["is_default"] is True
    assert out["settings"]["auto_explain.log_min_duration"]["is_default"] is False
    assert out["overrides"]["rows_read"] == 3
    assert {(v["role"], v["setting"]) for v in out["overrides"]["values"]} == {
        ("app", "statement_timeout"), ("admin", "log_statement")}
    assert out["login_roles"][0]["sessions"] == 2
    role = out["statements"]["by_role"][0]
    assert role["groups_at_or_over_ms"] == {"1000": 3, "10000": 1}
    assert role["calls_of_groups_at_or_over_ms"]["1000"] == 70
    assert out["statements"]["thresholds_ms"] == [1000, 10000]


def test_no_pg_stat_statements_leaves_the_calls_unknown_never_zero():
    out = facts(Conn({k: v for k, v in ANSWERS.items() if k != "from pg_extension"}))
    assert "statements" not in out
    assert "pg_stat_statements" in out["statements_error"]


def test_a_fact_that_fails_is_left_out_with_its_reason_and_no_password():
    out = facts(Conn(ANSWERS, broken=("from pg_db_role_setting",)))
    assert "overrides" not in out
    assert "s3cr3t" not in out["overrides_error"]
    assert out["login_roles"]


A003 = {"results": {"n1": {"data": {
    "log_min_duration_statement": {"setting": "-1", "unit": "ms", "category": "c", "context": "", "vartype": "integer",
                                   "pretty_value": "-1 ms"},
    "work_mem": {"setting": "4096", "unit": "kB", "category": "c", "context": "", "vartype": "integer", "pretty_value": "4 MB"},
}, "postgres_version": {"version": "17.1", "server_version_num": "170001", "server_major_ver": "17", "server_minor_ver": "1"}}}}


def box(monkeypatch, conn, ranges):
    read = d001.catalog_facts
    monkeypatch.setattr("reporter.postgres_reports.read_sources",
                        lambda path: [{"cluster": "c", "node_name": "n1", "conn_str": "postgresql://mon:pw@db/app"}])
    monkeypatch.setattr(d001, "catalog_facts", lambda dsn: read(dsn, opener=lambda _dsn: conn))
    return Box({}, ranges)


def test_the_report_is_schema_valid_with_the_nodes_settings_and_the_windows(monkeypatch):
    report = box(monkeypatch, Conn(ANSWERS), [counter(10, 12)]).generate_d001_logging_report(A003, "c", "n1")
    validate_report(report)
    node = report["results"]["n1"]
    assert set(node["data"]) == {"log_min_duration_statement", "auto_explain.log_min_duration"}  # the node's, not A003's
    assert node["data"]["log_min_duration_statement"]["context"] == "superuser"
    assert node["logging"]["setting_sources"]["auto_explain.log_min_duration"]["source"] == "configuration file"
    assert node["logging"]["temp_files"]["total"] == 2
    assert node["postgres_version"]["server_major_ver"] == "17"


def test_without_the_node_the_settings_come_from_a003_and_say_why(monkeypatch):
    monkeypatch.setattr("reporter.postgres_reports.read_sources", lambda path: [])
    report = Box({}, []).generate_d001_logging_report(A003, "c", "n1")
    validate_report(report)
    node = report["results"]["n1"]
    assert set(node["data"]) == {"log_min_duration_statement"}
    assert node["logging"]["node_error"].startswith("no connection")
    assert node["logging"]["temp_files_error"]


def test_a_value_on_all_roles_and_databases_has_its_own_scope():
    answers = dict(ANSWERS)
    answers["from pg_db_role_setting"] = [{"role": None, "database": None, "config": ["log_statement=ddl"]}]
    assert facts(Conn(answers))["overrides"]["values"][0]["scope"] == "all"


def test_a_reader_that_cannot_see_other_sessions_writes_unknown_not_zero():
    answers = dict(ANSWERS)
    answers["pg_has_role"] = [{"visible": False}]
    out = facts(Conn(answers))
    assert out["login_roles"][0]["sessions"] is None and out["login_roles"][0]["applications"] == []
    assert "pg_read_all_stats" in out["sessions_error"]


def test_autovacuum_runs_are_asked_per_table_without_roots_or_other(monkeypatch):
    asked = []
    # A table the metric skipped for one reading, beside a quiet one: merged per database the dip
    # and the rise would read 1000 as new runs (round 2's overcount); merged per table it is 2.
    start = 1_000_000.0
    series = [{"metric": {"datname": "app", "schema": "s", "table_name": "hot"},
               "values": [[start, "1000"], [start + 7200, "1002"]]},
              {"metric": {"datname": "app", "schema": "s", "table_name": "b"},
               "values": [[start, "5"], [start + 3600, "5"], [start + 7200, "5"]]}]
    generator = box(monkeypatch, Conn(ANSWERS), series)
    plain = generator.query_range

    def recording(query, start_time, end_time, step="30s"):
        asked.append(query)
        return plain(query, start_time, end_time, step)

    generator.query_range = recording
    report = generator.generate_d001_logging_report(A003, "c", "n1")
    validate_report(report)
    where = 'cluster="c", node_name="n1"'
    assert (f'pgwatch_table_stats_autovacuum_count{{{where}, schema!="$other$"}} unless on (datname, schema, table_name) '
            f'(pgwatch_table_stats_is_part_root{{{where}}} == 1)') in asked
    runs = report["results"]["n1"]["logging"]["autovacuum_runs"]
    assert runs["total"] == 2 and runs["floor"] is True
    assert "floor" not in report["results"]["n1"]["logging"]["temp_files"]


def test_a_table_the_metric_skips_leaves_a_gap_not_a_lifetime_of_new_runs():
    """Under a lock the metric omits the table. Per table it is a gap; a per-database sum would dip and rise."""
    start = 1_000_000.0
    hot = {"metric": {"datname": "app", "schema": "s", "table_name": "hot"},
           "values": [[start, "1000"], [start + 7200, "1002"]]}  # no reading at +3600
    other = {"metric": {"datname": "app", "schema": "s", "table_name": "b"},
             "values": [[start, "5"], [start + 3600, "5"], [start + 7200, "5"]]}
    window = d001.counter_window([hot, other], labels=("datname", "schema", "table_name"))
    assert window["total"] == 2


def test_a_size_label_flip_is_one_table_when_merged_by_name():
    a = {"metric": {"datname": "app", "schema": "s", "table_name": "t", "table_size_cardinality_mb": "10"},
         "values": [[1000.0, "7"], [4600.0, "8"]]}
    b = {"metric": {"datname": "app", "schema": "s", "table_name": "t", "table_size_cardinality_mb": "100"},
         "values": [[8200.0, "9"]]}
    assert d001.counter_window([a, b], labels=("datname", "schema", "table_name"))["total"] == 2


def test_a_failed_session_check_keeps_the_facts_and_says_unknown():
    out = facts(Conn(ANSWERS, broken=("pg_has_role",)))
    assert out["login_roles"][0]["sessions"] is None
    assert "failed at pg_has_role" in out["sessions_error"] and "s3cr3t" not in out["sessions_error"]
    assert out["statements"]["by_role"]  # the rest stands


def test_a_node_that_raises_is_recorded_without_its_password(monkeypatch):
    monkeypatch.setattr("reporter.postgres_reports.read_sources",
                        lambda path: [{"cluster": "c", "node_name": "n1", "conn_str": "postgresql://mon:s3cr3t@db/app"}])

    def fails(dsn):
        raise RuntimeError("connection failed for postgresql://mon:s3cr3t@db/app")

    monkeypatch.setattr(d001, "catalog_facts", fails)
    report = Box({}, []).generate_d001_logging_report(A003, "c", "n1")
    validate_report(report)
    error = report["results"]["n1"]["logging"]["node_error"]
    assert "connection failed" in error and "s3cr3t" not in error


def test_a_value_stored_for_a_role_is_a_threshold_too():
    settings = {"log_min_duration_statement": {"setting": "-1", "unit": "ms"}}
    overrides = {"values": [{"setting": "log_min_duration_statement", "value": "5s"},
                            {"setting": "auto_explain.log_min_duration", "value": "250"},
                            {"setting": "log_min_duration_sample", "value": "1min"},
                            {"setting": "log_min_duration_statement", "value": "-1"},
                            {"setting": "log_min_duration_statement", "value": "0"},
                            {"setting": "statement_timeout", "value": "3s"}]}
    assert d001.thresholds_of(settings, overrides) == [250, 1000, 5000, 10000, 60000]


def test_a_stored_value_reads_as_postgres_reads_it():
    for text, ms in (("1.5s", 1500), ("1h", 3_600_000), ("1d", 86_400_000), ("1.005s", 1005), ("1.1h", 3_960_000), ("1500us", 2)):
        assert d001._stored_ms(text) == ms, text
    for text in ("5S", "garbage", "", "-1", "25d", "9" * 400):
        assert d001._stored_ms(text) is None, text


def test_stored_thresholds_are_capped():
    overrides = {"values": [{"setting": "log_min_duration_statement", "value": f"{n}ms"} for n in range(1, 100)]}
    assert len(d001.thresholds_of({}, overrides)) == d001.MAX_STORED_THRESHOLDS + 2


def test_a_stored_value_reaches_the_statement_read(monkeypatch):
    seen = {}
    plain = d001.read_statements
    monkeypatch.setattr(d001, "read_statements", lambda conn, thresholds: seen.update(t=list(thresholds)) or plain(conn, thresholds))
    stored = {"role": "app", "database": None, "config": ["log_min_duration_statement=5s"]}
    facts(Conn({**ANSWERS, "from pg_db_role_setting": [stored]}))
    assert 5000 in seen["t"]
