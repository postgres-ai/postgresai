"""A005 on a box (postgresai#474): how the reporter connects to a node and reads
the sources file, and what reaches the report and the log (REV round 1 of
postgresai!552)."""
import json

import pytest

from reporter import a005
from reporter.postgres_reports import PostgresReportGenerator
from reporter.wait_ready import read_sources
from tests.reporter.test_a005 import FakeConn, fake_opener


def test_open_database_keeps_the_conn_str_options_and_trusts_system_roots_for_verify(monkeypatch):
    """pgwatch's own options stay; verify-* without sslrootcert uses the system CA roots."""
    calls = []

    class Conn:
        def set_session(self, **kw):
            calls.append(("session", kw))

    monkeypatch.setattr(a005.psycopg2, "connect", lambda dsn, **kw: calls.append((dsn, kw)) or Conn())
    a005.open_database("postgresql://u:p@h/db?options=-c%20work_mem%3D7MB&sslmode=verify-full", "other")
    dsn, kw = calls[0]
    assert "options" not in kw and kw["dbname"] == "other"
    # The system CA bundle as a file: libpq's sslrootcert=system finds no roots in this libpq.
    assert kw["sslrootcert"] == a005.system_ca_file() and kw["sslrootcert"] != "system"
    assert calls[1] == ("session", {"readonly": True, "autocommit": False})
    calls.clear()
    a005.open_database("postgresql://u:p@h/db?sslmode=require")
    assert "sslrootcert" not in calls[0][1]


def test_each_query_runs_in_its_own_read_only_transaction_with_local_settings():
    """Nothing is left on a pooled backend: SET LOCAL, then rollback, per query."""
    conns = []

    def opener(conn_str, database=None):
        conn = fake_opener([("postgres", True)])(conn_str, database)
        conns.append(conn)
        return conn
    a005.node_facts("postgresql://mon@db/postgres", opener=opener)
    conn = conns[0]
    assert set(conn.settings) == {"set local statement_timeout = '30s'", "set local search_path = pg_catalog, pg_temp"}
    assert conn.rollbacks >= 8  # one per query, failed ones included


@pytest.mark.parametrize("bad", ["postgresql://u:pa%zzss@127.0.0.1:1/db", "host=h password=sec ret"])
def test_a_conn_str_libpq_cannot_parse_is_reported_without_its_text(bad):
    with pytest.raises(a005.InvalidConnection) as err:
        a005.open_database(bad)
    assert str(err.value) == "the connection string is not valid"
    data = json.dumps(a005.unreachable_node(err.value, bad)["data"])
    assert "zz" not in data and "sec ret" not in data


@pytest.mark.parametrize("message,password", [
    ("failed for postgresql://mon:p%40ss@h/db", "p@ss"),
    ("password='a b' rejected", "a b"),
    ("fe_sendauth: password=s3cr3t", "s3cr3t"),
    ("auth p%40ss rejected", "p@ss"),  # the URL-encoded password on its own
])
def test_error_text_scrubs_passwords_in_every_form(message, password):
    secrets = a005._secrets(f"postgresql://mon:{a005.quote(password, safe='')}@h/db")
    out = a005.error_text(Exception(message), secrets)
    assert password not in out and a005.quote(password, safe="") not in out


def test_versions_before_10_are_unknown_as_in_the_cli():
    conn = FakeConn({"pg_settings where name in": [{"name": "server_version", "setting": "9.6.24"},
                                                   {"name": "server_version_num", "setting": "90624"}]}, "postgres")
    assert a005.postgres_version(conn)["server_major_ver"] == ""


@pytest.mark.parametrize("text,expected", [
    ("- name: a\n  conn_str: 'host=h password=''a #b'''\n", "host=h password='a #b'"),
    ('- name: a\n  conn_str: "postgresql://u:p@h/db?x=1#frag"\n', "postgresql://u:p@h/db?x=1#frag"),
    ("- name: a\n  conn_str: |-\n    postgresql://u:p@h/db\n", "postgresql://u:p@h/db"),
    ("- name: a\n", None),
])
def test_read_sources_conn_str_forms(tmp_path, text, expected):
    path = tmp_path / "s.yml"
    path.write_text(text)
    [src] = read_sources(str(path))
    assert src["conn_str"] == expected and src["cluster"] == "default" and src["node_name"] == "a"


def test_read_sources_an_item_may_open_with_any_key(tmp_path):
    path = tmp_path / "s.yml"
    path.write_text("- name: a\n  conn_str: postgresql://A\n- conn_str: postgresql://B\n  name: b\n"
                    "  custom_tags:\n    node_name: nb\n")
    assert [(s["name"], s["node_name"], s["conn_str"]) for s in read_sources(str(path))] == \
        [("a", "a", "postgresql://A"), ("b", "nb", "postgresql://B")]
    assert read_sources(str(tmp_path / "missing.yml")) == []


def test_the_report_reads_the_matching_node_with_its_conn_str_and_logs_no_password(tmp_path, monkeypatch, caplog):
    path = tmp_path / "sources.yml"
    path.write_text("- name: a\n  conn_str: postgresql://mon:pwA@hA/db\n  custom_tags:\n    node_name: na\n"
                    "- name: b\n  conn_str: postgresql://mon:pwB@hB/db\n  is_enabled: false\n"
                    "  custom_tags:\n    node_name: nb\n")
    seen = []

    def facts(conn_str):
        seen.append(conn_str)
        raise RuntimeError(f"connect to {conn_str} failed")
    monkeypatch.setattr(a005, "node_facts", facts)
    gen = PostgresReportGenerator(prometheus_url="http://unused", sources_path=str(path))
    with caplog.at_level("WARNING"):
        report = gen.generate_a005_extensions_report("default", "na")
        missing = gen.generate_a005_extensions_report("default", "nb")  # disabled
    assert seen == ["postgresql://mon:pwA@hA/db"]
    assert "pwA" not in json.dumps(report) and "pwA" not in caplog.text
    assert "no connection for this node" in missing["results"]["nb"]["data"]["logical_slots_error"]


def test_a005_covers_every_node_of_a_multi_node_cluster(monkeypatch):
    gen = PostgresReportGenerator(prometheus_url="http://unused")
    monkeypatch.setattr(gen, "get_all_nodes", lambda cluster: {"primary": "p", "standbys": ["s"]})
    for name in dir(gen):
        if name.startswith("generate_") and name.endswith("_report") and name != "generate_a005_extensions_report":
            monkeypatch.setattr(gen, name, lambda *a, **k: {"results": {}})
    monkeypatch.setattr(gen, "generate_a005_extensions_report", lambda cluster, node: gen.format_report_data(
        "A005", {node: a005.unreachable_node(Exception("down"))}, node))
    report = gen.generate_all_reports("default")["A005"]
    assert set(report["results"]) == {"p", "s"} and report["nodes"] == {"primary": "p", "standbys": ["s"]}


def test_the_system_ca_bundle_is_a_real_file():
    ca = a005.system_ca_file()
    assert ca is None or __import__("os").path.isfile(ca)


def test_the_conn_str_connect_timeout_is_kept(monkeypatch):
    calls = []

    class Conn:
        def set_session(self, **kw):
            pass

    monkeypatch.setattr(a005.psycopg2, "connect", lambda dsn, **kw: calls.append(kw) or Conn())
    a005.open_database("postgresql://u:p@h/db?connect_timeout=30")
    assert "connect_timeout" not in calls[0]


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("mode", ["verify-ca", "verify-full"])
def test_verify_modes_reach_the_server(mode):
    """libpq accepts the CA file with both modes (sslrootcert=system refused verify-ca).
    The local test server has no SSL, so reaching it ends in exactly that error."""
    import os
    host, port = os.environ.get("PGHOST", "localhost"), os.environ.get("PGPORT", "5432")
    with pytest.raises(Exception) as err:
        a005.open_database(f"postgresql://x:y@{host}:{port}/postgres?sslmode={mode}")
    assert "weak sslmode" not in str(err.value) and "root certificate" not in str(err.value)


def test_a_nested_name_key_does_not_rename_the_item(tmp_path):
    path = tmp_path / "s.yml"
    path.write_text("- name: a\n  conn_str: postgresql://A\n  custom_tags:\n    name: tagname\n"
                    "  custom_metrics:\n    name: 60\n")
    [src] = read_sources(str(path))
    assert (src["name"], src["node_name"]) == ("a", "a")


def test_double_quoted_escapes_js_yaml_writes():
    from reporter.wait_ready import _scalar
    assert _scalar('"p\\x41\\u00e9\\r\\\\z"') == "pA\u00e9\r\\z"
