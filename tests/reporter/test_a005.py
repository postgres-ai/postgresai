"""A005 "Extensions" on a monitoring box (postgresai#474): the reporter reads
the catalog facts of every database of each node, with the node's conn_str."""
import json
import os
import uuid
from pathlib import Path

import pytest

from reporter import a005
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report
from reporter.wait_ready import read_sources

VECTORS = Path(__file__).resolve().parents[1] / "compliance_vectors" / "a005_conclusions.json"

DB_ROW = {"extensions": {"plpgsql": "1.0"}, "index_access_methods": [{"am": "btree", "extension": None, "indexes": 3}],
          "row_security_policies": 0, "extended_statistics": 0, "deferrable_uniques": 1}


class FakeColumn:
    def __init__(self, name):
        self.name = name


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []
        self.description = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        rows = self.conn.answer(sql)
        if isinstance(rows, Exception):
            raise rows
        self.rows = rows
        self.description = [FakeColumn(k) for k in (rows[0].keys() if rows else [])]

    def fetchone(self):
        return tuple(self.rows[0].values()) if self.rows else None

    def fetchall(self):
        return [tuple(r.values()) for r in self.rows]


class FakeConn:
    def __init__(self, answers, database):
        self.answers = answers
        self.database = database
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    def rollback(self):
        self.rollbacks = getattr(self, "rollbacks", 0) + 1

    def answer(self, sql):
        if sql.startswith("set local "):
            self.settings = getattr(self, "settings", []) + [sql]
            return []
        for needle, rows in self.answers.items():
            if needle in sql:
                return rows(self.database) if callable(rows) else rows
        raise AssertionError(f"unexpected SQL: {sql[:60]}")


def fake_opener(databases, *, fail=None, hashes=None, opened=None):
    """databases: [(name, can_connect)]; the first one is current. fail: {name: Exception}."""
    answers = {
        "pg_settings where name in": [{"name": "server_version", "setting": "17.6"},
                                      {"name": "server_version_num", "setting": "170006"}],
        "from pg_database as d": [{"datname": n, "is_current": i == 0, "can_connect": c, "roles_with_create": 0}
                                  for i, (n, c) in enumerate(databases)],
        "indclass::oid[]": lambda db: [dict(DB_ROW)],
        "from pg_replication_slots": [{"database": "app1", "plugin": "pgoutput", "slots": 1, "active": 0}],
        "rds_replication": [{"replication": 0}],
        "from pg_authid": hashes or Exception("permission denied for table pg_authid"),
        "scram_iterations": [{"v": "4096"}],
        "from pg_subscription": [{"n": 0}],
        "from pg_hba_file_rules": Exception("permission denied for view pg_hba_file_rules"),
    }

    def opener(conn_str, database=None):
        name = database or databases[0][0]
        if opened is not None:
            opened.append(name)
        if fail and name in fail:
            raise fail[name]
        return FakeConn(answers, name)
    return opener


def test_every_database_is_read_and_one_without_connect_is_listed_with_the_reason():
    opened = []
    result = a005.node_facts("postgresql://mon:secret@db:5432/postgres",
                             opener=fake_opener([("postgres", True), ("app1", True), ("noconn", False)], opened=opened))
    dbs = result["data"]["databases"]
    assert dbs["postgres"]["read"] and dbs["app1"]["read"]
    assert dbs["app1"]["deferrable_uniques"] == 1
    assert dbs["noconn"] == {"read": False, "roles_with_create": 0,
                             "error": "the checkup role has no CONNECT privilege on this database"}
    assert opened == ["postgres", "app1"]
    assert result["postgres_version"]["server_minor_ver"] == "6"


def test_a_box_role_sees_hashes_and_hba_as_unknown_with_the_reason_never_zero():
    data = a005.node_facts("postgresql://mon@db/postgres", opener=fake_opener([("postgres", True)]))["data"]
    assert data["roles"]["md5_passwords"] is None
    assert data["roles"]["password_hashes_error"] == "permission denied for table pg_authid"
    assert data["hba_rules"] is None and data["hba_rules_error"] == "permission denied for view pg_hba_file_rules"
    assert data["logical_slots"] == [{"database": "app1", "plugin": "pgoutput", "slots": 1, "active": 0}]


def test_a_database_that_fails_to_open_is_listed_and_its_error_has_no_password():
    err = Exception('connection to "postgresql://mon:s3cr3t@db:5432/app1" failed: password=s3cr3t rejected\n  again')
    data = a005.node_facts("postgresql://mon:s3cr3t@db:5432/postgres",
                           opener=fake_opener([("postgres", True), ("app1", True)], fail={"app1": err}))["data"]
    entry = data["databases"]["app1"]
    assert entry["read"] is False
    assert "s3cr3t" not in entry["error"] and "\n" not in entry["error"]
    assert len(entry["error"]) <= a005.ERROR_CHARS


def test_at_most_max_databases_are_opened_the_rest_listed_as_not_read(monkeypatch):
    monkeypatch.setattr(a005, "MAX_DATABASES", 2)
    names = [("postgres", True)] + [(f"db{i}", True) for i in range(4)]
    data = a005.node_facts("postgresql://mon@db/postgres", opener=fake_opener(names))["data"]
    assert [n for n, d in data["databases"].items() if d["read"]] == ["postgres", "db0", "db1"]
    assert data["databases"]["db3"]["error"] == "not read: more than 2 databases"


def test_the_time_budget_stops_opening_databases():
    ticks = iter([0, 61])
    data = a005.node_facts("postgresql://mon@db/postgres", opener=fake_opener([("postgres", True), ("app1", True)]),
                           clock=lambda: next(ticks))["data"]
    assert data["databases"]["app1"]["error"] == "not read: the time budget ran out"


def test_an_unreachable_node_lists_no_database_and_every_fact_unknown():
    report = PostgresReportGenerator(prometheus_url="http://unused").format_report_data(
        "A005", {"n1": a005.unreachable_node(Exception("timeout to postgresql://mon:pw@h/db"), "postgresql://mon:pw@h/db")}, "n1")
    validate_report(report)
    data = report["results"]["n1"]["data"]
    assert data["databases"] == {} and data["roles"]["md5_passwords"] is None
    assert "pw@" not in json.dumps(report)


def test_the_reports_json_is_schema_valid_and_carries_the_conclusions():
    gen = PostgresReportGenerator(prometheus_url="http://unused")
    result = a005.node_facts("postgresql://mon@db/postgres", opener=fake_opener([("postgres", True), ("noconn", False)]))
    report = gen.format_report_data("A005", {"node-01": result}, "node-01")
    validate_report(report)
    assert report["checkId"] == "A005" and report["checkTitle"] == "Extensions"
    assert "Not read: noconn." in report["results"]["node-01"]["data"]["conclusions"][1]


@pytest.mark.parametrize("case", json.loads(VECTORS.read_text())["valid_cases"], ids=lambda c: c["id"])
def test_conclusions_match_the_cli_word_for_word(case):
    """The same vectors run in cli/test/checkup-a005.test.ts against a005Conclusions."""
    assert a005.conclusions(case["input"]) == case["expected"]


def test_sources_keep_the_conn_str_of_each_enabled_target(tmp_path):
    path = tmp_path / "sources.yml"
    path.write_text("""# PGWatch Sources Configuration - prometheus Instance
- name: main
  conn_str: >-
    postgresql://mon:pw@db.example:5432/postgres
  is_enabled: true
  custom_tags:
    cluster: default
    node_name: node-a
- name: off
  conn_str: postgresql://x@y/z
  is_enabled: false
""")
    assert read_sources(str(path)) == [{"name": "main", "cluster": "default", "node_name": "node-a",
                                        "conn_str": "postgresql://mon:pw@db.example:5432/postgres"}]


def test_a_node_missing_from_the_sources_file_is_listed_as_not_read(tmp_path):
    gen = PostgresReportGenerator(prometheus_url="http://unused", sources_path=str(tmp_path / "missing.yml"))
    report = gen.generate_a005_extensions_report("default", "node-a")
    validate_report(report)
    assert report["results"]["node-a"]["data"]["databases"] == {}
    assert "no connection for this node" in report["results"]["node-a"]["data"]["logical_slots_error"]


def test_a005_failing_costs_no_other_report(monkeypatch):
    """S9: the A005 generator raises; generate_all_reports still returns the other reports."""
    gen = PostgresReportGenerator(prometheus_url="http://unused")
    monkeypatch.setattr(gen, "get_all_nodes", lambda cluster: {"primary": "node-01", "standbys": []})

    def boom(*a, **k):
        raise RuntimeError("postgresql://mon:pw@h/db exploded")
    monkeypatch.setattr(gen, "generate_a005_extensions_report", boom)
    stub = lambda check_id: (lambda *a, **k: gen.format_report_data(check_id, {"x": 1}, "node-01"))
    for name in dir(gen):
        if name.startswith("generate_") and name.endswith("_report") and name != "generate_a005_extensions_report":
            check = name.split("_")[1].upper()
            monkeypatch.setattr(gen, name, stub(check))
    reports = gen.generate_all_reports("default")
    assert "A005" not in reports and "A002" in reports and "H002" in reports


# --- against a real server (CI: the system PostgreSQL) -----------------------

@pytest.fixture
def cluster():
    psycopg2 = pytest.importorskip("psycopg2")
    host, port = os.environ.get("PGHOST", "localhost"), int(os.environ.get("PGPORT", "5432"))
    user, password = os.environ.get("PGUSER", "postgres"), os.environ.get("PGPASSWORD")
    try:
        admin = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres", connect_timeout=3)
    except Exception as err:
        pytest.skip(f"no PostgreSQL: {err}")
    admin.autocommit = True
    tag = uuid.uuid4().hex[:8]
    role, ok_db, no_db = f"a005_mon_{tag}", f"a005_ok_{tag}", f"a005_no_{tag}"
    with admin.cursor() as cur:
        cur.execute(f"create role {role} login password 'pw_{tag}' in role pg_monitor")
        cur.execute(f"create database {ok_db}")
        cur.execute(f"create database {no_db}")
        cur.execute(f"revoke connect on database {no_db} from public")
    other = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=ok_db)
    other.autocommit = True
    with other.cursor() as cur:
        cur.execute("create table t (id int, constraint t_id_key unique (id) deferrable)")
    other.close()
    yield f"postgresql://{role}:pw_{tag}@{host}:{port}/postgres", ok_db, no_db, f"pw_{tag}"
    with admin.cursor() as cur:
        cur.execute(f"drop database if exists {ok_db}")
        cur.execute(f"drop database if exists {no_db}")
        cur.execute(f"drop role if exists {role}")
    admin.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_a_real_server_every_database_one_without_connect(cluster):
    conn_str, ok_db, no_db, password = cluster
    result = a005.node_facts(conn_str)
    dbs = result["data"]["databases"]
    assert dbs[ok_db]["read"] is True and dbs[ok_db]["deferrable_uniques"] == 1
    assert dbs[no_db]["read"] is False and "CONNECT" in dbs[no_db]["error"]
    assert "template1" in dbs
    assert result["data"]["roles"]["md5_passwords"] is None  # pg_monitor cannot read pg_authid
    report = PostgresReportGenerator(prometheus_url="http://unused").format_report_data("A005", {"n": result}, "n")
    validate_report(report)
    assert password not in json.dumps(report)
