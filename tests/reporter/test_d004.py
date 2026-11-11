"""D004 on a monitoring box: the observability extensions' state per database, the
pg_stat_statements entries, deallocation rounds and reset from the node, and whether the
monitored statements' calls grow over a window (reporter/d004.py)."""
from datetime import datetime, timezone

import pytest
from psycopg2 import sql as pgsql

from reporter import a005, d001, d004
from reporter.report_schemas import validate_report
from tests.reporter.test_d001 import counter
from tests.reporter.test_g003 import Box, Conn

SETTINGS = [
    {"name": "pg_stat_statements.max", "setting": "5000", "unit": "", "category": "Customized Options",
     "context": "postmaster", "vartype": "integer", "source": "default", "boot_val": "5000", "pending_restart": False},
    {"name": "pg_stat_statements.track", "setting": "top", "unit": "", "category": "Customized Options",
     "context": "superuser", "vartype": "enum", "source": "default", "boot_val": "top", "pending_restart": False},
    {"name": "shared_preload_libraries", "setting": "pg_stat_statements", "unit": "", "category": "Preloading",
     "context": "postmaster", "vartype": "string", "source": "configuration file", "boot_val": "",
     "pending_restart": False},
]
DATABASES = [
    {"datname": "app", "is_current": True, "can_connect": True, "roles_with_create": 0},
    {"datname": "other", "is_current": False, "can_connect": True, "roles_with_create": 0},
    {"datname": "locked", "is_current": False, "can_connect": False, "roles_with_create": 0},
]
ANSWERS = {
    "current_user as role": [{"role": "mon", "database": "app"}],
    "from pg_settings": SETTINGS,
    "from pg_available_extensions": [{"name": "pg_stat_statements", "default_version": "1.11", "installed_version": "1.10"},
                                     {"name": "pg_wait_sampling", "default_version": "1.1", "installed_version": None}],
    "from pg_db_role_setting": [
        {"role": "app", "database": None, "config": ["pg_stat_statements.track=all", "work_mem=1GB"]},
        {"role": None, "database": "app", "config": ["statement_timeout=5s"]}],
    "where d.datallowconn": DATABASES,
    "from pg_extension as e": [{"name": "pg_stat_statements", "version": "1.10", "schema": "extensions"}],
    "called_once": [{"database": "app", "entries": 4800, "calls": 90_000, "called_once": 3000, "temp_table_text": 2500,
                     "texts_visible": True},
                    {"database": "other", "entries": 185, "calls": 10, "called_once": 5, "temp_table_text": 0,
                     "texts_visible": True}],
    "pg_stat_statements_info": [{"dealloc": 2577, "stats_reset": datetime(2026, 7, 31, 21, 31, tzinfo=timezone.utc),
                                 "read_at": datetime(2026, 9, 23, 8, 42, tzinfo=timezone.utc)}],
}
OTHER = {"from pg_extension as e": []}


@pytest.fixture(autouse=True)
def composed_sql_without_a_server(monkeypatch):
    """A composed query needs a real connection to become text; the fake answers by fragments."""
    monkeypatch.setattr(pgsql.Composed, "as_string", lambda self, context=None: repr(self))


def opener(conns):
    return lambda _dsn, database=None: conns[database or "app"]


def facts(conns):
    return d004.catalog_facts("postgresql://mon:s3cr3t@db/app", opener=opener(conns))


def test_the_node_read_gives_states_per_database_entries_and_rounds():
    out = facts({"app": Conn(ANSWERS), "other": Conn(OTHER)})
    assert out["available"] == {"pg_stat_statements": {"default_version": "1.11"},
                                "pg_wait_sampling": {"default_version": "1.1"}}
    assert out["databases"]["app"]["installed"]["pg_stat_statements"]["version"] == "1.10"
    assert out["databases"]["other"]["installed"] == {}
    assert "CONNECT" in out["databases"]["locked"]["error"]
    assert out["settings"]["pg_stat_statements.max"]["context"] == "postmaster"
    assert out["settings"]["shared_preload_libraries"]["is_default"] is False
    assert out["overrides"]["values"] == [{"scope": "role", "role": "app", "database": None,
                                           "setting": "pg_stat_statements.track", "value": "all"}]
    st = out["statements"]
    assert st["entries"] == 4985 and st["dealloc"] == 2577 and st["database"] == "app"
    assert st["days_since_reset"] == 53.47 and st["stats_reset"] == "2026-07-31T21:31:00+00:00"
    assert st["by_database"][0] == {"database": "app", "entries": 4800, "calls": 90_000, "called_once": 3000,
                                    "temp_table_text": 2500}


def test_the_view_is_read_where_it_is_created():
    here = dict(ANSWERS, **{"from pg_extension as e": []})
    there = dict(ANSWERS, **{"current_user as role": []})
    out = facts({"app": Conn(here), "other": Conn(there)})
    assert out["statements"]["database"] == "other"


def test_no_pg_stat_statements_anywhere_is_said_never_zero_entries():
    out = facts({"app": Conn(dict(ANSWERS, **{"from pg_extension as e": []})), "other": Conn(OTHER)})
    assert "statements" not in out
    assert "not created" in out["statements_error"]


def test_hidden_texts_leave_the_temporary_table_count_out():
    rows = [dict(ANSWERS["called_once"][0], texts_visible=False)]
    out = facts({"app": Conn(dict(ANSWERS, called_once=rows)), "other": Conn(OTHER)})
    assert "temp_table_text" not in out["statements"]["by_database"][0]
    assert "pg_read_all_stats" in out["statements"]["temp_table_text_error"]


def test_a_failed_fact_is_left_out_with_its_reason_and_no_password():
    out = facts({"app": Conn(ANSWERS, broken=("from pg_available_extensions", "pg_stat_statements_info")),
                 "other": Conn(OTHER)})
    assert "available" not in out and "s3cr3t" not in out["available_error"]
    assert "dealloc" not in out["statements"] and "s3cr3t" not in out["statements"]["dealloc_error"]
    assert out["statements"]["entries"] == 4985


@pytest.mark.parametrize("entries,removed", [(5000, 250), (4785, 239), (100, 10), (7, 7), (0, 0)])
def test_a_round_removes_five_percent_but_at_least_ten_and_at_most_all(entries, removed):
    assert d004.per_round(entries) == removed


def test_the_calls_window_is_per_database_and_counts_statements():
    series = [counter(10, 15, datname="app", queryid="1"), counter(3, 3, datname="app", queryid="2"),
              counter(0, 0, datname="idle", queryid="3")]
    out = d004.monitored_calls_window(series, d001.counter_window)
    assert [(w["database"], w["total"], w["statements"]) for w in out] == [("app", 5, 2), ("idle", 0, 1)]


def box(monkeypatch, conns, ranges):
    read = d004.catalog_facts
    monkeypatch.setattr("reporter.postgres_reports.read_sources",
                        lambda path: [{"cluster": "c", "node_name": "n1", "conn_str": "postgresql://mon:pw@db/app"}])
    monkeypatch.setattr(d004, "catalog_facts", lambda dsn: read(dsn, opener=opener(conns)))
    calls = [{"metric": {"datname": "app", "queryid": str(i)}, "value": [0, str(v)]} for i, v in enumerate((5, 7, 1, 1, 1, 9))]
    return Box({"pgwatch_pg_stat_statements_calls": calls}, ranges)


def test_the_report_is_schema_valid_with_the_node_facts_and_the_window(monkeypatch):
    generator = box(monkeypatch, {"app": Conn(ANSWERS), "other": Conn(OTHER)}, [counter(10, 12, datname="app", queryid="1")])
    report = generator.generate_d004_pgstat_settings_report("c", "n1")
    validate_report(report)
    node = report["results"]["n1"]
    assert node["data"]["pg_stat_statements_status"]["total_calls"] == 24  # every series, not the first five
    obs = node["observability"]
    assert obs["statements"]["per_round_at_max"] == 250  # a round runs at the maximum (5,000), not at today's count
    assert obs["monitored_calls_window"][0]["total"] == 2


def test_a_node_that_raises_is_recorded_without_its_password(monkeypatch):
    monkeypatch.setattr("reporter.postgres_reports.read_sources",
                        lambda path: [{"cluster": "c", "node_name": "n1", "conn_str": "postgresql://mon:pw@db/app"}])

    def refuse(dsn):
        raise RuntimeError("connection refused password=pw")
    monkeypatch.setattr(d004, "catalog_facts", refuse)
    report = Box({}, []).generate_d004_pgstat_settings_report("c", "n1")
    validate_report(report)
    obs = report["results"]["n1"]["observability"]
    assert "pw" not in obs["node_error"].replace("password", "")
    assert "monitored_calls_window_error" in obs


def test_a_failed_database_list_keeps_the_facts_read_before_it():
    out = facts({"app": Conn(ANSWERS, broken=("where d.datallowconn",)), "other": Conn(OTHER)})
    assert out["settings"] and out["available"] and out["overrides"]
    assert "databases" not in out and "statements" not in out and "statements_error" not in out
    assert "failed at" in out["databases_error"] and "s3cr3t" not in out["databases_error"]


def test_monitoring_that_raises_costs_only_the_window(monkeypatch):
    generator = box(monkeypatch, {"app": Conn(ANSWERS), "other": Conn(OTHER)}, [])

    def broken(*args, **kwargs):
        raise RuntimeError("bad range reply")
    monkeypatch.setattr(generator, "query_range", broken)
    report = generator.generate_d004_pgstat_settings_report("c", "n1")
    validate_report(report)
    obs = report["results"]["n1"]["observability"]
    assert obs["statements"]["entries"] == 4985 and "bad range reply" in obs["monitored_calls_window_error"]


def test_past_the_database_cap_and_the_time_budget_the_rest_are_not_read(monkeypatch):
    many = [dict(DATABASES[0])] + [{"datname": f"db{i:03d}", "is_current": False, "can_connect": True}
                                   for i in range(a005.MAX_DATABASES + 2)]
    conns = {"app": Conn(dict(ANSWERS, **{"where d.datallowconn": many}))}
    out = d004.catalog_facts("postgresql://mon:pw@db/app", opener=lambda _dsn, database=None: conns.get(database or "app")
                             or Conn(OTHER))
    errors = [db for db, entry in out["databases"].items() if "more than" in (entry.get("error") or "")]
    assert len(errors) == 2 and out["databases"]["db000"]["installed"] == {}
    ticks = iter([0.0] + [a005.TIME_BUDGET_S + 1.0] * 500)
    out = d004.catalog_facts("postgresql://mon:pw@db/app", opener=lambda _dsn, database=None: conns.get(database or "app")
                             or Conn(OTHER), clock=lambda: next(ticks))
    assert all("time budget" in entry["error"] for db, entry in out["databases"].items() if db != "app")
    assert out["statements"]["entries"] == 4985  # the connected database's view still read


def test_the_view_in_another_database_is_read_there_and_its_failure_said():
    here = dict(ANSWERS, **{"from pg_extension as e": []})
    there = dict(ANSWERS, called_once=[dict(ANSWERS["called_once"][0], entries=7)])
    out = facts({"app": Conn(here), "other": Conn(there)})
    assert out["statements"]["database"] == "other" and out["statements"]["entries"] == 7

    calls = {"n": 0}

    def opener(_dsn, database=None):
        if database == "other":
            calls["n"] += 1
            if calls["n"] == 2:  # the second open, for the view, fails
                raise RuntimeError("too many connections password=s3cr3t")
            return Conn(there)
        return Conn(here)
    out = d004.catalog_facts("postgresql://mon:s3cr3t@db/app", opener=opener)
    assert "statements" not in out and "too many connections" in out["statements_error"]
    assert "s3cr3t" not in out["statements_error"] and out["databases"]["other"]["installed"]


def test_both_databases_with_the_view_read_the_connected_one():
    # "other" listed first: the connected database must still win, not the first one found.
    here = dict(ANSWERS, **{"where d.datallowconn": [DATABASES[1], DATABASES[0]]})
    there = dict(ANSWERS, called_once=[dict(ANSWERS["called_once"][0], entries=7)])
    out = facts({"app": Conn(here), "other": Conn(there)})
    assert out["statements"]["database"] == "app" and out["statements"]["entries"] == 4985


@pytest.mark.parametrize("change", ["no rounds", "info failed", "no max"])
def test_per_round_at_max_only_with_rounds_and_the_maximum(monkeypatch, change):
    answers = dict(ANSWERS)
    broken = ()
    if change == "no rounds":
        answers["pg_stat_statements_info"] = [dict(ANSWERS["pg_stat_statements_info"][0], dealloc=0)]
    elif change == "info failed":
        broken = ("pg_stat_statements_info",)
    else:
        answers["from pg_settings"] = [s for s in SETTINGS if s["name"] != "pg_stat_statements.max"]
    report = box(monkeypatch, {"app": Conn(answers, broken=broken), "other": Conn(OTHER)}, []).generate_d004_pgstat_settings_report("c", "n1")
    validate_report(report)
    assert "per_round_at_max" not in report["results"]["n1"]["observability"]["statements"]


def test_an_empty_window_says_it_may_be_a_failed_query(monkeypatch):
    report = box(monkeypatch, {"app": Conn(ANSWERS), "other": Conn(OTHER)}, []).generate_d004_pgstat_settings_report("c", "n1")
    assert "or the query failed" in report["results"]["n1"]["observability"]["monitored_calls_window_error"]
