"""Sparse wait-event gauge aggregation in the SQL dashboards.

``pgwatch_wait_events_total`` is a snapshot gauge with high-cardinality series that disappear
when a query/wait combination is not active. Carrying each series forward independently would
sum observations from different collection ticks as if they were concurrent.

Every ASH target therefore integrates the raw samples inside each rendered bucket
(``prom_sum_over_time`` with a window equal to the step), zero-fills a wait series that is absent
from a bucket, and divides by the number of successful pgwatch scrapes of the same instance in that
bucket (``up{job="pgwatch-prometheus"}``), dropping buckets without one. Per-series absence is then
zero, while complete telemetry gaps remain gaps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tests.grafana_dashboards.conftest import call_args, dashboard_file, iter_panels, target_query

WAIT_METRIC = "'pgwatch_wait_events_total'"

# (dashboard number, panel id, refId) -> the labels each bucket is summed by before normalization
# (always with the instance, which the scrape count divides by)
EXPECTED_TARGETS = {
    (1, 38, "A"): ["wait_event_type"],
    (1, 38, "B"): ["wait_event_type"],
    (1, 38, "C"): ["wait_event_type"],
    (3, 19, "A"): ["wait_event_type", "wait_event"],
    (4, 1, "A"): ["wait_event_type"],
    (4, 1, "B"): ["wait_event_type"],
    (4, 1, "C"): ["wait_event_type"],
    (4, 2, "A"): ["wait_event_type", "wait_event"],
    (4, 3, "A"): ["wait_event_type", "wait_event", "query_id"],
}
SCOPE = ("['cluster','=',(${cluster_name:sqlstring})]", "['node_name','=',(${node_name:sqlstring})]")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_event_targets(dashboard: dict):
    for panel in iter_panels(dashboard):
        for target in panel.get("targets", []) or []:
            sql = target_query(target)
            if "pgwatch_wait_events_total" in sql:
                yield panel, target, sql


def variant(sql: str) -> str:
    """The branch of the target's `CASE ('x')` matcher switch the compiler selected."""
    return re.search(r"\|\| CASE \('(\w+)'\)", sql).group(1)


def branch(sql: str, name: str) -> str:
    m = re.search(rf"WHEN '{name}' THEN (\[.*?\]\])", sql)
    if m:
        return m.group(1)
    return re.search(r"ELSE (\[.*?\]\]) END", sql).group(1)


def problems_of(panel: dict, sql: str) -> list[str]:
    problems = []
    w = call_args(sql, "prom_sum_over_time")
    if w[0] != WAIT_METRIC:
        problems.append("does not integrate raw wait-event samples (prom_sum_over_time)")
    elif w[4] != w[5]:
        problems.append("the integration window is not the bucket (window != step)")
    if "prom_sum_over_time('up', [['job','=','pgwatch-prometheus']]" not in sql:
        problems.append("does not count successful pgwatch scrapes")
    else:
        up = call_args(sql[sql.index("prom_sum_over_time('up'"):], "prom_sum_over_time")
        if up[2:] != w[2:]:
            problems.append("counts scrapes on another grid than the wait events")
    if "HAVING sum(value) > 0" not in sql:
        problems.append("does not drop buckets without a successful scrape")
    if not re.search(r"coalesce\(w\.v, 0\) AS v", sql):
        problems.append("does not zero-fill sparse wait-series gaps")
    if not re.search(r"FROM filled JOIN up USING \(instance, t\)", sql) or "v / n" not in sql:
        problems.append("does not normalize per exporter instance after the zero-fill")
    if not all(s in w[1] for s in SCOPE):
        problems.append("is not scoped to the selected cluster and node")
    if panel.get("interval") != "30s":
        problems.append("panel minimum interval is not 30s")
    for rollup in ("prom_last_over_time", "prom_last", "prom_tlast_over_time"):
        if f"{rollup}({WAIT_METRIC}" in sql or f"{rollup}(({WAIT_METRIC})" in sql:
            problems.append(f"carries wait series forward ({rollup})")
    return problems


def test_all_wait_event_targets_use_scrape_aware_average(dashboard_path: Path) -> None:
    offenders = [
        f"panel id={panel.get('id')!r} title={panel.get('title')!r} target={target.get('refId')!r}: "
        + "; ".join(problems)
        for panel, target, sql in _wait_event_targets(_load(dashboard_path))
        if (problems := problems_of(panel, sql))
    ]
    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def grouping(sql: str) -> list[str]:
    """The labels the bucket sum is grouped by besides the instance."""
    head = sql[: sql.index("FROM prom_sum_over_time")]
    if "coalesce(labels['instance'], '') AS instance" not in head:
        return []
    m = re.search(r"list_transform\(\(\[([^\]]*)\]\)", head)
    if m:
        return re.findall(r"'(\w+)'", m.group(1))
    return re.findall(r"coalesce\(labels\['(\w+)'\], ''\) AS (?!instance)\w+", head)


def test_every_known_wait_event_target_is_covered() -> None:
    actual: dict[tuple[int, int, str], list[str]] = {}
    for number in (1, 3, 4):
        for panel, target, sql in _wait_event_targets(_load(dashboard_file(number))):
            actual[(number, panel["id"], target["refId"])] = grouping(sql)
    assert set(actual) == set(EXPECTED_TARGETS)
    for key, labels in EXPECTED_TARGETS.items():
        assert actual[key] == labels, f"{key} groups by {actual[key]} before scrape normalization"


def _panel(number: int, panel_id: int) -> dict:
    return next(p for p in iter_panels(_load(dashboard_file(number))) if p.get("id") == panel_id)


def test_server_process_targets_are_scoped_and_split_activity() -> None:
    for number, panel_id in ((1, 38), (4, 1)):
        targets = {t["refId"]: target_query(t) for t in _panel(number, panel_id)["targets"]}
        assert variant(targets["A"]) == "a"
        assert branch(targets["A"], "a") == "[['datname','=~',('${db_name:regex}')]]"
        for ref, want in (("B", "b"), ("C", "c")):
            assert variant(targets[ref]) == want
        assert branch(targets["B"], "b") == \
            "[['datname','=','server_process'],['wait_event_type','!=','Activity']]"
        assert branch(targets["C"], "c") == \
            "[['datname','=','server_process'],['wait_event_type','=','Activity']]"


def test_single_query_ash_is_fully_scoped() -> None:
    sql = target_query(_panel(3, 19)["targets"][0])
    assert "['datname','=~',('${db_name:regex}')]" in call_args(sql, "prom_sum_over_time")[1]
    assert variant(sql) == "query"
    assert branch(sql, "query") == "[['query_id','=',(${query_id:sqlstring})],['query_id','!=','']]"


def test_ash_labels_are_built_and_formatted_once_per_group() -> None:
    for number, panel_id in ((3, 19), (4, 2), (4, 3)):
        sql = target_query(_panel(number, panel_id)["targets"][0])
        assert "FROM (SELECT DISTINCT k FROM w)" in sql
        assert "rendered AS MATERIALIZED" in sql
        assert "FROM names q" in sql
        assert "FROM r JOIN rendered n USING (k)" in sql


def test_wait_event_variables_follow_scope_dependency_order() -> None:
    variables = _load(dashboard_file(4))["templating"]["list"]
    assert [v["name"] for v in variables] == [
        "cluster_name", "node_name", "db_name", "wait_event_type", "wait_event",
    ]
    scope = ("metric = 'pgwatch_wait_events_total'",
             "coalesce(labels['cluster'], '') = ${cluster_name:sqlstring}",
             "coalesce(labels['node_name'], '') = ${node_name:sqlstring}",
             "regexp_full_match(coalesce(labels['datname'], ''), '${db_name:regex}')")
    by_name = {v["name"]: v for v in variables}
    for name in ("wait_event_type", "wait_event"):
        v = by_name[name]
        assert v["definition"] == v["query"]["infinityQuery"]["url_options"]["data"]
        assert v["definition"].startswith(f"SELECT DISTINCT labels['{name}'] AS value FROM series")
        for s in scope:
            assert s in v["definition"], (name, s)
    assert "regexp_full_match(coalesce(labels['wait_event_type'], ''), '${wait_event_type:regex}')" \
        in by_name["wait_event"]["definition"]
    assert "wait_event_type:regex" not in by_name["wait_event_type"]["definition"]


def test_call_args_parser() -> None:
    assert call_args("x f(a, [b, 'c,)'], (d, e)) y", "f") == ["a", "[b, 'c,)']", "(d, e)"]


def test_guard_catches_each_regression() -> None:
    """Each regression, applied to a shipped ASH target, is reported (the guard is not vacuous)."""
    panel = _panel(4, 1)
    sql = target_query(panel["targets"][0])
    assert problems_of(panel, sql) == []
    w = call_args(sql, "prom_sum_over_time")
    mutations = {
        "no scrape count": sql.replace("prom_sum_over_time('up', [['job','=','pgwatch-prometheus']]",
                                       "prom_sum_over_time('up', [['job','=','other']]"),
        "zero-success buckets kept": sql.replace("HAVING sum(value) > 0", ""),
        "no zero-fill": sql.replace("coalesce(w.v, 0) AS v", "w.v AS v"),
        "not normalized": sql.replace("v / n", "v"),
        "unscoped": sql.replace("['node_name','=',(${node_name:sqlstring})]", "['node_name','=~','.*']", 1),
        "carried forward": sql.replace("prom_sum_over_time('pgwatch_wait_events_total'",
                                       "prom_last_over_time('pgwatch_wait_events_total'", 1),
        "window wider than the bucket": sql.replace(", ".join(w[4:6]), w[4] + ", 60000", 1),
    }
    for name, mutated in mutations.items():
        assert mutated != sql, name
        try:
            found = problems_of(panel, mutated)
        except ValueError:
            found = ["unparseable"]
        assert found, name
    assert problems_of(dict(panel, interval="1m"), sql)
