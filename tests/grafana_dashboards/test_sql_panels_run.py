"""Every panel of the SQL dashboards (mon/grafana/dashboards) runs on pgai-mon and shows data.

The end-to-end test loads tests/grafana_dashboards/fixtures/panel_store.jsonl.gz (2 hours of a
recorded store plus synthetic series for what it lacks; see make_panel_store.py there) into a
temporary store, serves it with the pgai-mon binary named by PGAI_MON_BIN, and runs
mon/grafana/check_panels.py: Grafana's variable resolution and interpolation, then each panel's
SQL. It fails on any error and on any panel without data. Without PGAI_MON_BIN it is skipped.

The other tests cover check_panels' reproduction of Grafana (formats, intervals, filters) and
need nothing.
"""
from __future__ import annotations

import gzip
import json
import os
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
        pytest.skip("PGAI_MON_BIN is not set (path to a built pgai-mon)")
    store = tmp_path_factory.mktemp("panel_store") / "store"
    with gzip.open(FIXTURE, "rb") as src:
        subprocess.run([binary, "load", "--store", str(store), "--input", "-"], input=src.read(),
                       check=True, capture_output=True, timeout=300)
    port = _free_port()
    log = open(store.parent / "serve.log", "wb")
    proc = subprocess.Popen(
        [binary, "serve", "--store", str(store), "--listen", f"127.0.0.1:{port}", "--no-compact",
         "--metrics-yml", str(REPO_ROOT / "config" / "pgwatch-prometheus" / "metrics.yml"),
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
    assert len(results) == 14
    bad = [(r["file"], p["panel"], p["title"], p["class"],
            [t.get("error") for t in p["targets"] if t.get("error")])
           for r in results for p in r["panels"] if p["class"] not in ("data", "none")]
    assert not bad, json.dumps(bad, indent=1)
    verr = [(r["file"], v) for r in results for v in r["variable_queries"] if v["class"] != "data"]
    assert not verr, verr
    assert rc == 0
    # every query panel ran (only rows and text panels have no targets)
    assert sum(1 for r in results for p in r["panels"] if p["class"] == "data") >= 180
