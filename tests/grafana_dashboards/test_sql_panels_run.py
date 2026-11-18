"""Every panel of the SQL dashboards (mon/grafana/dashboards) runs on pgai-mon and shows data.

The end-to-end test loads tests/grafana_dashboards/fixtures/panel_store.jsonl.gz (2 hours of a
recorded store plus synthetic series for what it lacks; see make_panel_store.py there) into a
temporary store, serves it with the pgai-mon binary named by PGAI_MON_BIN, and runs
mon/grafana/check_panels.py: Grafana's variable resolution and interpolation, then each panel's
SQL. It fails on any error and on any panel without data. Without PGAI_MON_BIN it is skipped, or fails when CI is set.

The other tests cover check_panels' reproduction of Grafana (formats, intervals, filters) and
need nothing.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mon" / "grafana"))
import check_panels as CP  # noqa: E402
import compile as DC  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "panel_store.jsonl.gz"
FIXTURE_END_MS = 1794206700000  # make_panel_store.py --end-ms
QUERY_ID = "3821806407744617563"  # make_panel_store.QUERY_ID


# --------------------------------------------------------------------------------------------
# Grafana's part, reproduced
# --------------------------------------------------------------------------------------------
def test_formats_match_grafana():
    assert CP.fmt_value(["a.b"], "regex", "v") == r"a\.b"
    assert CP.fmt_value(["a", "b-c"], "regex", "v") == r"(a|b\-c)"
    assert CP.fmt_value(["a", "b"], None, "v") == "{a,b}"
    assert CP.fmt_value(["a"], None, "v") == "a"
    assert CP.fmt_value(["o'k", "x"], "sqlstring", "v") == "'o''k','x'"
    assert CP.fmt_value(["a", "b"], "queryparam", "db") == "var-db=a&var-db=b"
    assert CP.fmt_value(["a", "b"], "csv", "v") == "a,b"
    # a custom allValue is not escaped (Grafana skips formatting it)
    assert CP.fmt_value([CP.CustomAll(".+")], "regex", "v") == ".+"


def test_interpolation():
    b = CP.builtin_vars(1_000_000, 4_600_000, 5000)
    out, left = CP.interpolate("${__from} ${__to} ${__interval_ms} $db [[db]] ${db:regex} ${__from:date} ${x}",
                               {"db": ["a.b"]}, b)
    assert out.split()[:6] == ["1000000", "4600000", "5000", "a.b", "a.b", r"a\.b"]
    assert out.split()[6] == "1970-01-01T00:16:40.000Z"
    assert left == ["${x}"]
    # panel field references are not query variables
    out, left = CP.interpolate("${__field.labels.series}", {}, b)
    assert out == "${__field.labels.series}" and not left


def test_interval_is_grafanas():
    assert CP.round_interval(3600) == 5000          # 1 h / 1000 points
    assert CP.round_interval(86_400) == 60_000      # 24 h
    assert CP.round_interval(604_800) == 600_000    # 7 d
    assert CP.interval_ms(3_600_000, {}) == 5000
    assert CP.interval_ms(3_600_000, {"interval": "20s"}) == 20_000  # the panel's min interval
    assert CP.interval_ms(3_600_000, {"maxDataPoints": 100}) == 30_000


def test_filter_expression():
    f = CP.row_filter("is_na<1 && bloat_pct>0")
    assert f({"is_na": 0, "bloat_pct": 3.5})
    assert not f({"is_na": 1, "bloat_pct": 3.5})
    assert not f({"is_na": 0, "bloat_pct": None})
    with pytest.raises(ValueError):
        CP.row_filter("__import__('os')")


def test_variable_resolution_without_a_server():
    d = {"templating": {"list": [
        {"name": "top_n", "type": "custom", "query": "5, 10, 20", "current": {"value": "10"}},
        {"name": "mode", "type": "custom", "query": "Smart : a, Raw : b", "current": {"value": "zz"}},
        {"name": "q", "type": "textbox", "query": "", "current": {"value": "42"}},
    ]}}
    vars_, report = CP.resolve_variables(d, "http://127.0.0.1:9", 0, 3_600_000, {})
    assert vars_ == {"top_n": ["10"], "mode": ["a"], "q": ["42"]} and report == []


# --------------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------------
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pgai_mon(tmp_path_factory):
    binary = os.environ.get("PGAI_MON_BIN")
    if not binary:
        # CI builds the binary (mon:build) for this test: a missing one is an error there.
        (pytest.fail if os.environ.get("CI") else pytest.skip)(
            "PGAI_MON_BIN is not set (path to a built pgai-mon)")
    store = tmp_path_factory.mktemp("panel_store") / "store"
    with gzip.open(FIXTURE, "rb") as src:
        subprocess.run([binary, "load", "--store", str(store), "--input", "-"], input=src.read(),
                       check=True, capture_output=True, timeout=300)
    port = _free_port()
    log = open(store.parent / "serve.log", "wb")
    proc = subprocess.Popen(
        [binary, "serve", "--store", str(store), "--listen", f"127.0.0.1:{port}", "--no-compact", "--no-auth",
         "--metrics-yml", str(REPO_ROOT / "mon" / "metrics.yml"),
         "--macros-sql", str(REPO_ROOT / "mon" / "sql" / "macros.sql")],
        stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            urllib.request.urlopen(url + "/health", timeout=2).read()
            break
        except OSError:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                pytest.fail("pgai-mon did not start: " + (store.parent / "serve.log").read_text()[-2000:])
            time.sleep(0.2)
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    shutil.rmtree(store, ignore_errors=True)


def test_every_panel_has_data(pgai_mon, tmp_path):
    out = tmp_path / "panels.json"
    rc = CP.main(["--url", pgai_mon, "--now-ms", str(FIXTURE_END_MS), "--var", f"query_id={QUERY_ID}",
                  "--json", str(out), "--quiet"])
    results = json.loads(out.read_text())
    assert len(results) == 16
    bad = [(r["file"], p["panel"], p["title"], p["class"],
            [t.get("error") for t in p["targets"] if t.get("error")])
           for r in results for p in r["panels"] if p["class"] not in ("data", "none")]
    assert not bad, json.dumps(bad, indent=1)
    verr = [(r["file"], v) for r in results for v in r["variable_queries"] if v["class"] != "data"]
    assert not verr, verr
    assert rc == 0
    # every query panel ran (only rows and text panels have no targets)
    assert sum(1 for r in results for p in r["panels"] if p["class"] == "data") >= 180


def test_ash_label_dictionary_matches_per_row_formatting(pgai_mon):
    # Empty, missing and unusual labels must keep exactly the same Grafana series identity.
    labels = """SELECT row_number() OVER () AS k, labels FROM (VALUES
      (map(['wait_event_type','wait_event','query_id'], ['IO','DataFileRead','42'])),
      (map(['wait_event_type','wait_event'], ['IO','DataFileRead'])),
      (map(['wait_event_type','wait_event'], ['', 'é\"\\event'])),
      (map(['query_id'], ['42'])),
      (map()), (NULL::MAP(VARCHAR,VARCHAR))
    ) s(labels)"""
    sql = f"""WITH names AS MATERIALIZED ({labels}), r AS MATERIALIZED (
      SELECT k, t, value FROM names
      CROSS JOIN (VALUES (30000::BIGINT, 1.0), (60000, 0.0)) v(t,value)
    )
    SELECT labels, t, value FROM r JOIN names USING (k)"""
    cols = ["labels", "t", "value"]
    legend = "{{wait_event_type}} - {{wait_event}} - {{query_id}}"
    plain = CP.post(pgai_mon, DC.wrap(sql, cols, legend, []))
    once = CP.post(pgai_mon, DC.wrap(sql, cols, legend, [], ash=True))
    assert "error" not in plain and "error" not in once, (plain, once)
    assert plain["columns"] == once["columns"]
    ordered = lambda rows: sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    assert len(plain["rows"]) == 12
    assert ordered(plain["rows"]) == ordered(once["rows"])


def test_ash_panel_identical_http_request_hits_cache(pgai_mon):
    path = next((REPO_ROOT / "mon/grafana/dashboards").glob("Dashboard_4_*.json"))
    panel = next(p for p in CP.walk(json.loads(path.read_text())["panels"]) if p.get("id") == 3)
    variables = {k: [v] for k, v in dict(
        cluster_name="local", node_name="node-01", db_name="target_database",
        wait_event_type=CP.CustomAll(".*"), wait_event=CP.CustomAll(".*"),
    ).items()}
    sql, left = CP.interpolate(panel["targets"][0]["url_options"]["data"], variables,
                               CP.builtin_vars(FIXTURE_END_MS - 6420000, FIXTURE_END_MS, 30000))
    assert not left
    req = urllib.request.Request(pgai_mon + "/sql?format=objects", sql.encode(),
                                 {"Content-Type": "text/plain"})
    with urllib.request.urlopen(req) as response:
        first = response.read()
    with urllib.request.urlopen(req) as response:
        assert response.headers["x-pgai-cache"] == "hit"
        assert response.read() == first
    assert json.loads(first)["row_count"] > 0


def test_ash_group_names_preserve_sparse_instance_normalization(pgai_mon):
    sql = DC.case_sql({"sql_file": "sql/reports/dash/ash_by.sql"})
    # Feed the aggregation stage directly; evaluator bucket semantics have their own tests.
    for metric, relation in (("pgwatch_wait_events_total", "waits"), ("up", "up_rows")):
        sql, count = re.subn(
            rf"prom_sum_over_time\('{metric}',[\s\S]*?getvariable\('interval_ms'\)\)",
            f"(SELECT labels, t, value FROM {relation})", sql,
        )
        assert count == 1
    inputs = """WITH waits AS (
      SELECT map(['instance','wait_event_type','wait_event','query_id'],
                 [instance,wet,'DataFileRead','42']) AS labels, t, value
      FROM (VALUES ('a','IO',0,1.0), ('a','IO',20,2.0), ('b','IO',10,9.0),
                   ('orphan','Lock',0,100.0)) v(instance,wet,t,value)
    ), up_rows AS (
      SELECT map(['instance'], [instance]) AS labels, t, value
      FROM (VALUES ('a',0,2.0), ('a',10,2.0), ('a',20,0.0),
                   ('b',0,4.0), ('b',10,3.0), ('b',20,2.0)) v(instance,t,value)
    ), w AS ("""
    sql = sql.replace("WITH w AS (", inputs, 1)
    sql = DC.inline(sql, dict(by=["wait_event_type", "wait_event", "query_id"],
                              start_ms=0, end_ms=30, step_ms=10), None)
    result = CP.post(pgai_mon, DC.wrap(sql, ["labels", "t", "value"], "{{wait_event_type}}", [], ash=True))
    assert "error" not in result, result
    assert [(r["time"], r["value"], r["series"]) for r in result["rows"]] == [
        (0, 0.5, "IO"), (10, 3.0, "IO"), (20, 0.0, "IO"),
    ]
