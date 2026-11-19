"""Every query of the SQL dashboards goes to pgai-mon, and reads the selected instance.

The PromQL dashboards mixed three datasources (Prometheus, the pgwatch Postgres sink, the Flask
backend through Infinity CSV); a sink panel without a pinned database rendered a silent "No data".
The SQL dashboards have one: Grafana's Infinity datasource POSTing a single SELECT to pgai-mon's
/sql. A panel left on another datasource would query a service the pgai-mon stack does not run.

A target that reads a metric must also select the dashboard's cluster and node, or with several
monitored instances it blends them. The targets listed in UNSCOPED are faithful ports of source
PromQL that lacks those matchers; the list may only shrink. The self-monitoring dashboard reads
pgai-mon's own series and the scrape targets', which have no cluster or node: it has no instance
to select.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import (
    INFINITY_TYPE,
    INFINITY_UID,
    PGAI_MON_SQL_URL,
    is_dashboard,
    iter_panels,
    target_query,
)

INFINITY = {"type": INFINITY_TYPE, "uid": INFINITY_UID}
# Grafana's built-in source, which the "Annotations & Alerts" annotation uses
BUILTIN = {"type": "grafana", "uid": "-- Grafana --"}

# (dashboard number, panel id, refId) of metric reads without cluster/node matchers, as in the
# PromQL source (settings joins on Dashboard 1, the bloat and index panels of Dashboards 9 and 11)
UNSCOPED = {
    (1, 4, "D"), (1, 14, "C"), (1, 39, "D"), (1, 41, "A"), (1, 41, "B"), (1, 41, "C"),
    (9, 32, "A"), (9, 33, "A"),
    (11, 1, "A"), (11, 1, "B"), (11, 1, "C"),
}
SELF_MONITORING = "Self_Monitoring_Dashboard_pgai_mon.json"
SCOPE = ("${cluster_name:sqlstring}", "${node_name:sqlstring}")
METRIC_READ = re.compile(r"\b(?:prom_\w+|pgai_samples)\(\(?'(\w+)'|\bmetric = '(\w+)'")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _datasources(node, where="dashboard"):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "datasource":
                yield where, v
            yield from _datasources(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _datasources(v, f"{where}[{i}]")


def test_only_the_infinity_datasource(dashboard_path: Path) -> None:
    bad = [f"{where}: {ds}" for where, ds in _datasources(_load(dashboard_path))
           if ds not in (INFINITY, BUILTIN) and not (where.startswith("dashboard.annotations") and ds == BUILTIN)]
    assert not bad, f"{dashboard_path.name}:\n  " + "\n  ".join(bad)


def test_every_target_posts_one_select_to_pgai_mon(dashboard_path: Path) -> None:
    offenders = []
    for panel in iter_panels(_load(dashboard_path)):
        for target in panel.get("targets", []) or []:
            where = f"panel id={panel.get('id')!r} title={panel.get('title')!r} refId={target.get('refId')!r}"
            opts = target.get("url_options") or {}
            if target.get("expr") or target.get("rawSql"):
                offenders.append(f"{where}: still carries PromQL or sink SQL")
            csv = target.get("type") == "csv"
            url = PGAI_MON_SQL_URL.replace("format=objects", "format=csv") if csv else PGAI_MON_SQL_URL
            if (target.get("datasource"), target.get("url"), target.get("source"), target.get("type")) != \
                    (INFINITY, url, "url", "csv" if csv else "json"):
                offenders.append(f"{where}: not an Infinity target on {url}")
            if (opts.get("method"), opts.get("body_type"), opts.get("body_content_type"), target.get("root_selector")) \
                    != ("POST", "raw", "text/plain", "" if csv else "rows"):
                offenders.append(f"{where}: not a raw text/plain POST with the expected root")
            body = re.sub(r"'(?:[^']|'')*'", "''", opts.get("data") or "")  # string literals aside
            body = re.sub(r"^-- pgai-window-v1 \$\{__from\} \$\{__to\} \$\{__interval_ms\} (?:range|instant)\n",
                          "", body)
            if not re.match(r"\s*(WITH|SELECT)\b", body) or re.search(r";\s*\S", body):
                offenders.append(f"{where}: the body is not one SELECT")
    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def test_query_variables_read_pgai_mon(dashboard_path: Path) -> None:
    for v in _load(dashboard_path).get("templating", {}).get("list", []) or []:
        if v.get("type") != "query":
            continue
        inner = v["query"]["infinityQuery"]
        assert v["datasource"] == INFINITY, v["name"]
        assert inner["url"] == PGAI_MON_SQL_URL and inner["url_options"]["method"] == "POST", v["name"]
        assert inner["url_options"]["data"] == v["definition"], v["name"]
        assert v["definition"].startswith("SELECT "), v["name"]


def _unscoped(dashboard_path: Path) -> set[tuple[int, int, str]]:
    number = int(re.match(r"Dashboard_(\d+)_", dashboard_path.name).group(1))
    out = set()
    for panel in iter_panels(_load(dashboard_path)):
        for target in panel.get("targets", []) or []:
            sql = target_query(target)
            if METRIC_READ.search(sql) and not all(s in sql for s in SCOPE):
                out.add((number, panel["id"], target["refId"]))
    return out


def test_metric_reads_select_the_instance(dashboard_path: Path) -> None:
    if dashboard_path.name == SELF_MONITORING:
        pytest.skip("pgai-mon's own series: no cluster/node to select")
    number = int(re.match(r"Dashboard_(\d+)_", dashboard_path.name).group(1))
    known = {k for k in UNSCOPED if is_dashboard(dashboard_path, k[0])}
    found = _unscoped(dashboard_path)
    assert found <= known, f"{dashboard_path.name}: reads without cluster/node: {sorted(found - known)}"
    assert found == known, (f"Dashboard {number}: now scoped, drop from UNSCOPED: {sorted(known - found)}")


def test_metric_read_pattern() -> None:
    assert METRIC_READ.search("FROM prom_last(('pgwatch_x'), [])")
    assert METRIC_READ.search("FROM prom_irate('pgwatch_x', [])")
    assert METRIC_READ.search("FROM series WHERE metric = 'pgwatch_x'")
    assert not METRIC_READ.search("SELECT 2137483648 AS value")
