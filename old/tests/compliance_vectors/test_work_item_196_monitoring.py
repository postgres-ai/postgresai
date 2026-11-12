"""Regression coverage for work item 196 monitoring release fixes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIRS = (
    PROJECT_ROOT / "config" / "grafana" / "dashboards",
    PROJECT_ROOT / "postgres_ai_helm" / "config" / "grafana" / "dashboards",
)
DASHBOARD_1 = DASHBOARD_DIRS[0] / "Dashboard_1_Node_performance_overview.json"
DASHBOARD_7 = DASHBOARD_DIRS[0] / "Dashboard_7_Autovacuum_and_xmin_horizon.json"


def load_dashboard(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_panels(value: Any):
    if isinstance(value, dict):
        if "id" in value and "type" in value:
            yield value
        for child in value.values():
            yield from iter_panels(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_panels(child)


def panel_by_title(dashboard: dict[str, Any], title: str) -> dict[str, Any]:
    for panel in iter_panels(dashboard):
        if panel.get("title") == title:
            return panel
    raise AssertionError(f"Panel {title!r} not found")


def variable_by_name(dashboard: dict[str, Any], name: str) -> dict[str, Any]:
    for variable in dashboard.get("templating", {}).get("list", []):
        if variable.get("name") == name:
            return variable
    raise AssertionError(f"Variable {name!r} not found")


def target_expr(panel: dict[str, Any]) -> str:
    targets = panel.get("targets") or []
    assert targets, f"Panel {panel.get('title')!r} has no targets"
    return targets[0].get("expr", "")


def test_dashboard_panel_ids_are_unique() -> None:
    for dashboard_dir in DASHBOARD_DIRS:
        for path in dashboard_dir.glob("*.json"):
            ids = [panel["id"] for panel in iter_panels(load_dashboard(path))]
            duplicates = sorted(
                panel_id for panel_id, count in Counter(ids).items() if count > 1
            )
            assert not duplicates, f"{path.relative_to(PROJECT_ROOT)} duplicate ids: {duplicates}"


def test_helm_dashboard_parity_is_preserved() -> None:
    source_dir, helm_dir = DASHBOARD_DIRS
    for source_path in source_dir.glob("*.json"):
        helm_path = helm_dir / source_path.name
        assert helm_path.exists(), f"Missing helm dashboard {helm_path.name}"
        assert source_path.read_text() == helm_path.read_text(), source_path.name


def test_vacuum_timeline_queries_are_scoped() -> None:
    expectations = {
        DASHBOARD_1: (),
        DASHBOARD_7: ('schema_name=~"$schema_name"', 'table_name=~"$table_name"'),
    }
    for path, extra_filters in expectations.items():
        expr = target_expr(panel_by_title(load_dashboard(path), "Vacuum timeline"))
        for phase in range(1, 8):
            fragment = f'pgwatch_pg_vacuum_progress_index_vacuum_count{{cluster="$cluster_name", node_name="$node_name", datname="$db_name", phase="{phase}"'
            assert fragment in expr
        for extra_filter in extra_filters:
            assert extra_filter in expr
        assert 'pgwatch_pg_vacuum_progress_index_vacuum_count{phase="' not in expr


def test_dashboard_7_schema_and_table_variables_are_scoped() -> None:
    dashboard = load_dashboard(DASHBOARD_7)
    schema_query = variable_by_name(dashboard, "schema_name")["query"]["query"]
    table_query = variable_by_name(dashboard, "table_name")["query"]["query"]

    for query in (schema_query, table_query):
        assert 'cluster="$cluster_name"' in query
        assert 'node_name="$node_name"' in query
        assert 'datname="$db_name"' in query

    assert 'schema!~"pg_.*|information_schema|_timescaledb.*"' in schema_query
    assert "allValue" not in variable_by_name(dashboard, "schema_name")
    assert 'schema=~"$schema_name"' in table_query


def test_dashboard_7_wraparound_topn_excludes_system_schemas() -> None:
    dashboard = load_dashboard(DASHBOARD_7)
    system_schema_filter = 'schema!~"pg_.*|information_schema|_timescaledb.*"'

    xid_expr = target_expr(panel_by_title(dashboard, "Top-N tables by XID age (relfrozenxid)"))
    mxid_expr = target_expr(panel_by_title(dashboard, "Top-N tables by MultiXID age (relminmxid)"))

    for expr in (xid_expr, mxid_expr):
        assert 'cluster="$cluster_name"' in expr
        assert 'node_name="$node_name"' in expr
        assert 'datname=~"$db_name"' in expr
        assert system_schema_filter in expr


def test_dashboard_1_surfaces_pg_wal_size_status_code() -> None:
    dashboard = load_dashboard(DASHBOARD_1)
    size_panel = panel_by_title(dashboard, "pg_wal directory size")
    status_panel = panel_by_title(dashboard, "pg_wal size collection status")

    assert size_panel["id"] != status_panel["id"]
    assert size_panel["gridPos"]["y"] == 113
    assert status_panel["gridPos"]["y"] == 123
    assert "[10m]" in target_expr(size_panel)
    status_expr = target_expr(status_panel)
    assert "pgwatch_pg_wal_size_status_code" in status_expr
    assert "[10m]" in status_expr
    mapping_text = json.dumps(status_panel.get("fieldConfig", {}))
    assert "pg_ls_waldir() unavailable" in mapping_text
    assert "EXECUTE missing" in mapping_text
