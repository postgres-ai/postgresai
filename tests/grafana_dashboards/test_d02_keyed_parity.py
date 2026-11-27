"""Dashboard 02's keyed top-k SQL (#516) answers exactly what its per-point predecessor did.

The 23 D1 cases moved from one row per point carrying its labels and cleaned text to points keyed
by series id plus one names row per series (mon/sql/reports/d1_query_analysis_topn.sql). This
runs both on pgai-mon over the reporter's recorded store, for every D1 case, and compares the rows
(labels, t, value, display text) as multisets. The store has one text per statement, so the text
choice (newest by first appearance now, by last refresh before) cannot differ here; that choice
is pinned by mon/tests/cache_516.rs.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

from tests.reporter import mon_harness

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mon" / "grafana"))
import compile as DC  # noqa: E402

# The D1 case SQL as it was before #516 (prom_topk macro, text cleaned on every point).
PER_POINT = """
SELECT top.labels, top.t, top.value,
       left(regexp_replace(qt.query, '\\s+', ' ', 'g'), 80) AS _displayname_long
FROM prom_topk(getvariable('top_n'), 'irate', getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_den_metric := CASE WHEN getvariable('per_call') THEN 'pgwatch_pg_stat_statements_calls' ELSE '' END) top
LEFT JOIN (SELECT queryid, dbname, arg_max(query, last_seen) AS query FROM query_texts GROUP BY ALL) qt
  ON qt.queryid = TRY_CAST(top.labels['queryid'] AS BIGINT) AND qt.dbname = top.labels['datname']
"""


def _rows(url: str, sql: str) -> list[str]:
    req = urllib.request.Request(url + "/sql?format=rows", data=sql.encode())
    with urllib.request.urlopen(req, timeout=120) as f:
        body = json.loads(f.read())
    assert "error" not in body, body
    return sorted(json.dumps(r, sort_keys=True) for r in body["rows"])


def _wrapped(sql: str) -> str:
    # labels as text, so maps compare by value
    return ("SELECT CAST(labels AS VARCHAR) AS labels, t, value, _displayname_long FROM (\n"
            f"{sql}\n) x")


def test_keyed_d1_sql_equals_the_per_point_sql(tmp_path):
    binary = mon_harness.find_binary()
    if binary is None:
        if os.environ.get("CI"):
            pytest.fail("no pgai-mon binary (PGAI_MON_BIN)")
        pytest.skip("no pgai-mon binary: build mon/ or set PGAI_MON_BIN")
    store = mon_harness.build_store(binary, tmp_path / "store")
    cases = [c for c in DC.load_cases() if c["id"].startswith("D1/")]
    assert len(cases) == 23 and all(c.get("keyed") for c in cases)
    with mon_harness.MonServer(binary, store) as mon:
        req = urllib.request.Request(mon.url + "/sql?format=rows", data=(
            "SELECT labels['cluster'], labels['node_name'], labels['datname'], max(t) FROM prom_last("
            "'pgwatch_pg_stat_statements_calls', [], 0, 9999999999999, 3600000) GROUP BY ALL ORDER BY 4 DESC LIMIT 1"
        ).encode())
        with urllib.request.urlopen(req, timeout=120) as f:
            cluster, node, db, last = json.loads(f.read())["rows"][0]
        compared = 0
        for case in cases:
            for span_ms, step_ms in ((3_600_000, 30_000), (6 * 3_600_000, 60_000)):
                end = last - last % step_ms
                params = dict(case["params"], cluster=cluster, node=node, db_re=db, start_ms=end - span_ms,
                              end_ms=end, step_ms=step_ms, rate_interval_ms=max(step_ms + 5000, 20_000))
                new = _rows(mon.url, _wrapped(DC.inline(DC.case_sql(case), params, None)))
                old = _rows(mon.url, _wrapped(DC.inline(PER_POINT, params, None)))
                assert new == old, f"{case['id']} {span_ms // 3_600_000} h"
                compared += len(new)
        assert compared > 0, "no rows compared: the store has no pg_stat_statements data"
