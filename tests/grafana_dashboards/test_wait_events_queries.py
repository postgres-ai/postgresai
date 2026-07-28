"""Regression tests for sparse wait-event gauge aggregation.

``pgwatch_wait_events_total`` is a snapshot gauge with high-cardinality
series that disappear when a query/wait combination is not active.  A bare
range query lets the TSDB carry each series independently and can therefore
sum observations from different collection ticks as if they were concurrent.

Every ASH target must integrate raw samples inside the rendered bucket and
divide by successful pgwatch scrape ticks.  Per-series absence is then zero,
while complete telemetry gaps remain gaps.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.grafana_dashboards.conftest import REPO_ROOT, iter_panels


DASHBOARD_DIR = REPO_ROOT / "config" / "grafana" / "dashboards"

EXPECTED_TARGETS = {
    ("Dashboard_1_Node_performance_overview.json", 38, "A"): "sum by (wait_event_type, instance)",
    ("Dashboard_1_Node_performance_overview.json", 38, "B"): "sum by (wait_event_type, instance)",
    ("Dashboard_1_Node_performance_overview.json", 38, "C"): "sum by (wait_event_type, instance)",
    ("Dashboard_3_Single_query_analysis.json", 19, "A"): "sum by (wait_event_type, wait_event, instance)",
    ("Dashboard_4_Wait_Sampling_Dashboard.json", 1, "A"): "sum by (wait_event_type, instance)",
    ("Dashboard_4_Wait_Sampling_Dashboard.json", 1, "B"): "sum by (wait_event_type, instance)",
    ("Dashboard_4_Wait_Sampling_Dashboard.json", 1, "C"): "sum by (wait_event_type, instance)",
    ("Dashboard_4_Wait_Sampling_Dashboard.json", 2, "A"): "sum by (wait_event_type, wait_event, instance)",
    ("Dashboard_4_Wait_Sampling_Dashboard.json", 3, "A"): "sum by (wait_event_type, wait_event, query_id, instance)",
}


def _load_dashboard(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_event_targets(path: Path):
    dashboard = _load_dashboard(path)
    for panel in iter_panels(dashboard):
        for target in panel.get("targets", []) or []:
            expr = target.get("expr", "")
            if "pgwatch_wait_events_total" in expr:
                yield panel, target, expr


def test_all_wait_event_targets_use_scrape_aware_average(dashboard_path: Path) -> None:
    offenders: list[str] = []

    for panel, target, expr in _wait_event_targets(dashboard_path):
        problems: list[str] = []
        division_at = expr.find("/ on (instance) group_left()")

        if "sum_over_time(" not in expr or "[$__interval]" not in expr:
            problems.append("does not integrate raw samples over $__interval")
        if 'sum_over_time(up{job="pgwatch-prometheus"}[$__interval])' not in expr:
            problems.append("does not count successful pgwatch scrapes")
        if division_at < 0:
            problems.append("does not normalize per exporter instance")
        if "default 0" not in expr:
            problems.append("does not zero-fill sparse wait-series gaps")
        elif division_at >= 0:
            if expr.find("default 0") > division_at or "default 0" in expr[division_at:]:
                problems.append("zero-fills telemetry gaps after normalization")
        if "> 0" not in expr or " bool " in expr:
            problems.append("does not filter zero-success scrape buckets")
        if 'cluster="$cluster_name"' not in expr or 'node_name="$node_name"' not in expr:
            problems.append("is not scoped to the selected cluster and node")
        if panel.get("interval") != "30s":
            problems.append("panel minimum interval is not 30s")
        if "last_over_time(pgwatch_wait_events_total" in expr:
            problems.append("still uses the high-cardinality 30s subquery")

        if problems:
            offenders.append(
                f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                f"target={target.get('refId')!r}: " + "; ".join(problems)
            )

    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def test_every_known_wait_event_target_is_covered() -> None:
    actual: dict[tuple[str, int, str], str] = {}

    for path in DASHBOARD_DIR.glob("*.json"):
        for panel, target, expr in _wait_event_targets(path):
            key = (path.name, panel["id"], target["refId"])
            actual[key] = expr

    assert set(actual) == set(EXPECTED_TARGETS)
    for key, inner_group in EXPECTED_TARGETS.items():
        assert inner_group in actual[key], f"{key} drops labels required before scrape normalization"


def test_server_process_targets_are_scoped_and_split_activity() -> None:
    for filename, panel_id in (
        ("Dashboard_1_Node_performance_overview.json", 38),
        ("Dashboard_4_Wait_Sampling_Dashboard.json", 1),
    ):
        dashboard = _load_dashboard(DASHBOARD_DIR / filename)
        panel = next(panel for panel in iter_panels(dashboard) if panel.get("id") == panel_id)
        targets = {target["refId"]: target["expr"] for target in panel["targets"]}

        for ref_id in ("B", "C"):
            expr = targets[ref_id]
            assert 'cluster="$cluster_name"' in expr
            assert 'node_name="$node_name"' in expr
            assert 'datname="server_process"' in expr

        assert 'wait_event_type!="Activity"' in targets["B"]
        assert 'wait_event_type="Activity"' in targets["C"]
        assert ".*- Activity.*" not in targets["B"] + targets["C"]


def test_single_query_ash_is_fully_scoped() -> None:
    dashboard = _load_dashboard(DASHBOARD_DIR / "Dashboard_3_Single_query_analysis.json")
    panel = next(panel for panel in iter_panels(dashboard) if panel.get("id") == 19)
    expr = panel["targets"][0]["expr"]

    assert 'datname=~"$db_name"' in expr
    assert 'query_id="$query_id"' in expr
    assert 'query_id!=""' in expr


def test_wait_event_variables_follow_scope_dependency_order() -> None:
    dashboard = _load_dashboard(DASHBOARD_DIR / "Dashboard_4_Wait_Sampling_Dashboard.json")
    variables = dashboard["templating"]["list"]

    assert [variable["name"] for variable in variables] == [
        "cluster_name",
        "node_name",
        "db_name",
        "wait_event_type",
        "wait_event",
    ]

    expected_queries = {
        "wait_event_type": (
            'label_values(pgwatch_wait_events_total{cluster="$cluster_name", '
            'node_name="$node_name", datname=~"$db_name"},wait_event_type)'
        ),
        "wait_event": (
            'label_values(pgwatch_wait_events_total{cluster="$cluster_name", '
            'node_name="$node_name", datname=~"$db_name", '
            'wait_event_type=~"$wait_event_type"},wait_event)'
        ),
    }

    variables_by_name = {variable["name"]: variable for variable in variables}
    for name, expected_query in expected_queries.items():
        variable = variables_by_name[name]
        assert variable["definition"] == expected_query
        assert variable["query"]["query"] == expected_query
