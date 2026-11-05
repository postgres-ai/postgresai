"""G003 on a monitoring box: role values and longest calls from the node, deadlocks over
a window and lock waits from monitoring (reporter/g003.py)."""
import json
import os
import uuid

import psycopg2
import pytest

from reporter import g003
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report


def counter(database, *values, start=1_000_000.0, step=3600):
    return {"metric": {"real_dbname": database},
            "values": [[start + i * step, str(v)] for i, v in enumerate(values)]}


def test_the_window_counts_rises_not_the_counter_or_its_first_value():
    window = g003.deadlock_window([counter("app", 370, 370, 373, 373)])
    assert window["deadlocks"] == 3
    assert window["days_with_deadlocks"] == 1
    assert window["by_database"] == {"app": 3}
    assert window["days"] == round(3 * 3600 / 86400, 1)


def test_a_statistics_reset_is_not_counted_as_deadlocks():
    assert g003.deadlock_window([counter("app", 5, 9, 0, 2)])["deadlocks"] == 6


def test_a_series_that_changes_labels_goes_on_with_the_same_counter():
    first = {"metric": {"real_dbname": "app", "sys_id": "1"}, "values": [[1000.0, "5"], [4600.0, "6"]]}
    second = {"metric": {"real_dbname": "app", "sys_id": "2"}, "values": [[8200.0, "8"]]}
    assert g003.deadlock_window([first, second])["deadlocks"] == 3


def test_the_peak_hour_is_the_hour_of_day_with_the_most():
    # 86400 * 10 is midnight UTC: hours 1 and 2 of the same day
    series = counter("app", 0, 1, 11, 11, start=86400.0 * 10)
    window = g003.deadlock_window([series])
    assert window["peak_hour_utc"] == 2
    assert window["peak_hour_deadlocks"] == 10
    assert window["per_day"] == [{"date": "1970-01-11", "deadlocks": 11}]


def test_one_reading_is_no_window():
    assert g003.deadlock_window([counter("app", 3)]) is None
    assert g003.deadlock_window([]) is None


def lock_series(blocked_query="1", blocker_query="2", age=100.0, pid="10", user="app"):
    return {"metric": {"__name__": "pgwatch_lock_waits_blocked_ms", "blocked_pid": pid, "blocked_query_id": blocked_query,
                       "blocked_user": "app", "blocked_table": "t", "blocker_query_id": blocker_query,
                       "blocker_user": user, "blocker_table": "t"}, "value": [0, str(age)]}


def test_lock_waits_group_by_statement_pair_worst_first_with_the_time_of_the_longest():
    longest = [lock_series(age=100.0, pid="10"), lock_series(age=900.0, pid="11"), lock_series(blocker_query="3", age=50.0)]
    at = [dict(lock_series(pid="11"), value=[0, "86400"]), dict(lock_series(pid="10"), value=[0, "86000"])]
    texts = {"1": "update  t\n set a = 1", "2": "select pg_sleep($1)"}
    out = g003.lock_waits(longest, at, window={"start": "s", "end": "e", "days": 1.0}, query_text=texts.get)
    assert out["pairs_total"] == 2 and out["session_pairs_total"] == 3
    top = out["pairs"][0]
    assert (top["sessions"], top["longest_blocked_statement_age_ms"]) == (2, 900)
    assert top["longest_at"] == "1970-01-02T00:00:00Z"
    assert top["blocked_query"] == "update t set a = 1"
    assert out["pairs"][1]["blocker_query"] is None


def test_no_series_is_no_pairs():
    assert g003.lock_waits([], [], window={"start": "s", "end": "e", "days": 1.0})["pairs"] == []


def test_a_role_row_keeps_only_timeouts_never_other_settings():
    assert g003._settings(["search_path=a, b", "statement_timeout=3s", "app.secret=x", "lock_timeout=8s"]) == {
        "statement_timeout": "3s", "lock_timeout": "8s"}
    assert g003._settings(None) == {}


class Conn:
    """Answers by the first words of each query; a query in ``broken`` fails."""

    def __init__(self, answers, broken=()):
        self.answers, self.broken, self.closed = answers, broken, False
        self.executed = []

    def cursor(self):
        return Cursor(self)

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class Column:
    def __init__(self, name):
        self.name = name


class Cursor:
    def __init__(self, conn):
        self.conn, self.rows, self.description = conn, [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def mogrify(self, sql, params):
        return str(sql).encode()

    def execute(self, sql, params=None):
        self.conn.executed.append(str(sql))
        for key in self.conn.broken:
            if key in str(sql):
                raise RuntimeError(f"failed at {key} password=s3cr3t")
        for key, rows in self.conn.answers.items():
            if key in str(sql):
                self.rows = rows
                self.description = [Column(k) for k in (rows[0] if rows else {})]
                return
        self.rows, self.description = [], []

    def fetchall(self):
        return [tuple(r.values()) for r in self.rows]


ROLES = [{"role": "anon", "can_login": False, "superuser": False, "bypass_rls": False, "member_of": [],
          "config": ["statement_timeout=3s", "search_path=public"]},
         {"role": "app", "can_login": True, "superuser": False, "bypass_rls": True, "member_of": ["anon"], "config": None}]
ANSWERS = {
    "current_user as role": [{"role": "mon", "database": "app"}],
    "from pg_settings": [{"name": "statement_timeout", "setting": "120000", "unit": "ms", "source": "configuration file"}],
    "where s.setrole = 0": [{"database": "app", "config": ["statement_timeout=5s", "work_mem=1GB"]}],
    "s.setdatabase = 0": ROLES,
    "from pg_extension": [{"schema": "extensions"}],
    "from pg_proc": [{"schema": "public", "name": "job", "kind": "p", "owner": "app", "commits": True,
                      "catches_cancel": False, "config": ["statement_timeout=1h"]}],
}


def facts(conn):
    return g003.catalog_facts("postgresql://mon:s3cr3t@db/app", opener=lambda _dsn: conn)


def test_catalog_facts_read_the_roles_database_values_and_routines():
    out = facts(Conn(ANSWERS))
    assert out["read_as"] == {"role": "mon", "database": "app"}
    assert out["roles"][0]["settings"] == {"statement_timeout": "3s"}
    assert out["database_settings"] == {"app": {"statement_timeout": "5s"}}
    assert out["routines"][0]["kind"] == "procedure" and out["routines"][0]["settings"] == {"statement_timeout": "1h"}


def test_the_server_values_are_read_without_our_own_statement_timeout():
    """a005._run sets statement_timeout locally, which read back as the server's value."""
    conn = Conn(ANSWERS)
    facts(conn)
    index = next(i for i, sql in enumerate(conn.executed) if "from pg_settings" in sql)
    assert "set local" not in conn.executed[index - 1]


def test_a_fact_that_fails_is_left_out_with_its_reason_and_no_password():
    out = facts(Conn(ANSWERS, broken=("from pg_proc",)))
    assert "routines" not in out
    assert out["routines_error"].startswith("failed at from pg_proc")
    assert "s3cr3t" not in out["routines_error"]


def test_no_pg_stat_statements_leaves_calls_unknown_never_zero():
    answers = {k: v for k, v in ANSWERS.items() if k != "from pg_extension"}
    out = facts(Conn(answers))
    assert "pg_stat_statements" in out["calls_error"]
    assert all("calls" not in role for role in out["roles"])


class Box(PostgresReportGenerator):
    """Monitoring answers from a table of query fragments."""

    def __init__(self, instant, ranges):
        super().__init__(prometheus_url="http://prom.test", postgres_sink_url="")
        self.instant, self.ranges = instant, ranges

    def query_instant(self, query):
        for key, rows in self.instant.items():
            if key in query:
                return {"status": "success", "data": {"result": rows}}
        return {"status": "success", "data": {"result": []}}

    def query_range(self, query, start_time, end_time, step="30s"):
        return self.ranges

    def get_queryid_queries_from_sink(self, query_text_limit=655360, db_names=None):
        return {"app": {"1": "update t set a = 1"}}


A003 = {"results": {"n1": {"data": {
    "statement_timeout": {"setting": "120000", "unit": "ms", "category": "c", "context": "", "vartype": "integer", "pretty_value": "120 s"},
    "work_mem": {"setting": "4096", "unit": "kB", "category": "c", "context": "", "vartype": "integer", "pretty_value": "4 MB"},
}, "postgres_version": {"version": "17.1", "server_version_num": "170001", "server_major_ver": "17", "server_minor_ver": "1"}}}}


READ_CATALOG = g003.catalog_facts


def box(monkeypatch, conn):
    monkeypatch.setattr("reporter.postgres_reports.read_sources",
                        lambda path: [{"cluster": "c", "node_name": "n1", "conn_str": "postgresql://mon:pw@db/app"}])
    monkeypatch.setattr(g003, "catalog_facts", lambda dsn: READ_CATALOG(dsn, opener=lambda _dsn: conn))
    return Box({"max_over_time(pgwatch_lock_waits": [lock_series(age=900.0)],
                "tmax_over_time": [dict(lock_series(), value=[0, "86400"])],
                "pgwatch_db_stats_deadlocks": [{"value": [0, "370"]}],
                "pgwatch_db_stats_conflicts": [{"value": [0, "1"]}],
                "pgwatch_stats_reset": [{"value": [0, "86400"]}]},
               [counter("app", 370, 372)])


def test_the_report_is_schema_valid_and_keeps_only_g003_settings(monkeypatch):
    report = box(monkeypatch, Conn(ANSWERS)).generate_g003_timeouts_report(A003, "c", "n1")
    validate_report(report)
    data = report["results"]["n1"]["data"]
    assert set(data["settings"]) == {"statement_timeout"}
    assert data["deadlock_stats"]["deadlocks"] == 370
    assert data["deadlock_window"]["deadlocks"] == 2
    assert data["lock_waits"]["pairs"][0]["blocked_query"] == "update t set a = 1"
    assert data["roles"][1]["role"] == "app"
    assert report["results"]["n1"]["postgres_version"]["server_major_ver"] == "17"


def test_a_database_named_data_does_not_break_the_file(monkeypatch):
    answers = dict(ANSWERS)
    answers["where s.setrole = 0"] = [{"database": "data", "config": ["statement_timeout=5s"]}]
    report = box(monkeypatch, Conn(answers)).generate_g003_timeouts_report(A003, "c", "n1")
    validate_report(report)
    assert report["results"]["n1"]["data"]["database_settings"] == {"data": {"statement_timeout": "5s"}}


def test_a_statement_that_is_not_a_plain_query_is_left_out():
    assert g003._short("create role a password 'x'") is None
    assert g003._short("  update t set a = $1") == "update t set a = $1"
    assert g003._short("<insufficient privilege>") is None
    assert g003._short("select 'x' as a into t1") is None
    assert g003._short("insert into t values ($1)") == "insert into t values ($1)"
    assert g003._short("with a as (select 1) select * into t2 from a") is None


def test_a_failed_lock_wait_query_is_not_read_as_none(monkeypatch):
    generator = box(monkeypatch, Conn(ANSWERS))
    plain = generator.query_instant

    def failing(query):
        return {} if "max_over_time(pgwatch_lock_waits" in query else plain(query)

    generator.query_instant = failing
    report = generator.generate_g003_timeouts_report(A003, "c", "n1")
    validate_report(report)
    data = report["results"]["n1"]["data"]
    assert "lock_waits" not in data and data["lock_waits_error"]


def test_below_14_the_lock_wait_metric_does_not_exist(monkeypatch):
    a003 = {"results": {"n1": {"data": A003["results"]["n1"]["data"], "postgres_version": {
        "version": "13.14", "server_version_num": "130014", "server_major_ver": "13", "server_minor_ver": "14"}}}}
    data = box(monkeypatch, Conn(ANSWERS)).generate_g003_timeouts_report(a003, "c", "n1")["results"]["n1"]["data"]
    assert "lock_waits" not in data and "14" in data["lock_waits_error"]


def test_a_failed_conflicts_counter_leaves_the_stats_out_with_the_reason(monkeypatch):
    generator = box(monkeypatch, Conn(ANSWERS))
    generator.instant = {k: v for k, v in generator.instant.items() if k != "pgwatch_db_stats_conflicts"}
    data = generator.generate_g003_timeouts_report(A003, "c", "n1")["results"]["n1"]["data"]
    assert data["deadlock_stats"] is None and data["deadlock_stats_error"]


def test_a_node_that_cannot_be_read_keeps_the_monitoring_facts(monkeypatch):
    generator = box(monkeypatch, Conn(ANSWERS))

    def refuse(_dsn):
        raise RuntimeError("no route to postgresql://mon:pw@db/app")

    monkeypatch.setattr(g003, "catalog_facts", refuse)
    report = generator.generate_g003_timeouts_report(A003, "c", "n1")
    validate_report(report)
    data = report["results"]["n1"]["data"]
    assert "roles" not in data
    assert "pw" not in data["roles_error"]
    assert data["deadlock_window"]["deadlocks"] == 2


def test_no_monitoring_data_writes_no_window_and_no_lock_waits(monkeypatch):
    generator = box(monkeypatch, Conn(ANSWERS))
    generator.instant, generator.ranges = {}, []
    data = generator.generate_g003_timeouts_report(A003, "c", "n1")["results"]["n1"]["data"]
    assert data["deadlock_stats"] is None
    assert "lock_waits" not in data and data["deadlock_window_error"]


@pytest.fixture
def server():
    """A real server: a role with limits, a procedure that commits, a function with its own timeout."""
    host = os.getenv("PG_TEST_HOST", "localhost")
    port = int(os.getenv("PG_TEST_PORT", "5432"))
    user, password = os.getenv("PG_TEST_USER", "postgres"), os.getenv("PG_TEST_PASSWORD", "postgres")
    dsn = f"host={host} port={port} user={user} password={password} dbname=postgres"
    try:
        admin = psycopg2.connect(dsn, connect_timeout=3)
    except psycopg2.OperationalError as err:
        pytest.skip(f"no server: {err}")
    admin.autocommit = True
    tag = uuid.uuid4().hex[:8]
    role = f"g003_{tag}"
    with admin.cursor() as cur:
        cur.execute(f"create role {role} login password 'pw_{tag}' in role pg_monitor")
        cur.execute(f"alter role {role} set statement_timeout = '3s'")
        cur.execute(f"alter role {role} set search_path = public")
        cur.execute(f"create procedure public.g003_{tag}_job() language plpgsql as $$ begin commit; "
                    "exception when query_canceled then null; end $$")
        cur.execute(f"create procedure public.g003_{tag}_nocommit() language plpgsql as $$ begin -- no commit here\n"
                    "create temp table t (a int) on commit drop; end $$")
        cur.execute(f"create procedure public.g003_{tag}_between() language plpgsql as $$ begin /* one */ commit; /* two */ "
                    "commit; /* three */ end $$")
        cur.execute(f"create function public.g003_{tag}_f() returns int language sql set statement_timeout = '4s' as 'select 1'")
    try:
        yield f"host={host} port={port} user={role} password=pw_{tag} dbname=postgres", role, tag
    finally:
        with admin.cursor() as cur:
            cur.execute(f"drop procedure public.g003_{tag}_job()")
            cur.execute(f"drop procedure public.g003_{tag}_nocommit()")
            cur.execute(f"drop procedure public.g003_{tag}_between()")
            cur.execute(f"drop function public.g003_{tag}_f()")
            cur.execute(f"drop role {role}")
        admin.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_a_real_server_gives_the_role_value_and_the_routines(server):
    dsn, role, tag = server
    out = g003.catalog_facts(dsn)
    assert out["read_as"]["role"] == role
    row = next(r for r in out["roles"] if r["role"] == role)
    assert row["settings"] == {"statement_timeout": "3s"} and row["can_login"] is True
    # The server value, not this role's or the reader's set local.
    assert out["server_settings"]["statement_timeout"]["source"] == "user"
    assert out["server_settings"]["statement_timeout"]["setting"] == "3000"
    routines = {r["name"]: r for r in out["routines"]}
    job = routines[f"g003_{tag}_job"]
    assert job["kind"] == "procedure" and job["commits"] and job["catches_cancel"]
    assert routines[f"g003_{tag}_nocommit"]["commits"] is False
    assert routines[f"g003_{tag}_between"]["commits"] is True
    assert routines[f"g003_{tag}_f"]["settings"] == {"statement_timeout": "4s"}
    json.dumps(out)


def test_g003_failing_costs_no_other_report(monkeypatch):
    gen = PostgresReportGenerator(prometheus_url="http://unused")
    monkeypatch.setattr(gen, "get_all_nodes", lambda cluster: {"primary": "node-01", "standbys": []})

    def boom(*a, **k):
        raise RuntimeError("postgresql://mon:pw@h/db exploded")

    for name in dir(gen):
        if name.startswith("generate_") and name.endswith("_report"):
            check = name.split("_")[1].upper()
            monkeypatch.setattr(gen, name, lambda *a, check=check, **k: gen.format_report_data(check, {"x": 1}, "node-01"))
    monkeypatch.setattr(gen, "generate_g003_timeouts_report", boom)
    reports = gen.generate_all_reports("default")
    assert "G003" not in reports and "A003" in reports and "H002" in reports
