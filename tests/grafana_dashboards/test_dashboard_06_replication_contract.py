"""Dashboard 06 contract: shared crosshair, logical standby identity, the
replica selector, and the "exactly one standby" detail gate.

The 2026-08-20 live review found the detail panels indistinguishable from the
summary panels and silently rendering a blend of every standby when the
selector was left on All. The fix is structural (a `replica` template variable
plus a range-wide PromQL gate that erases the detail series unless the selection
resolves to exactly one application name), so it is worth pinning. In the SQL
dashboard the gate is a CTE keeping only the steps where the $__range-wide
window holds exactly one application_name, joined to every detail series. Logical
per-standby graphs are also keyed only by application_name so client-port churn
does not create duplicate legends; the connection table deliberately retains
client_info as its physical-connection identity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import is_dashboard, iter_panels, target_query

DETAIL_MARKER = "select one standby"
REPLICA_FILTER = "['application_name', '=~', '${replica:regex}']"
# the gate: steps whose $__range-wide window holds exactly one standby
GATE_MARKER = "HAVING count(DISTINCT coalesce(labels[('application_name')], '')) = 1"
RANGE_WINDOW = "$__pgai_range_ms"
BY_APPLICATION = re.compile(
    r"p_by := CAST\(\(\['application_name'\]\) AS VARCHAR\[\]\)|prom_by\(labels, \['application_name'\]\)")
SERIES_BY_APPLICATION = ", coalesce(q.labels['application_name'], '') AS __series"
PROMPT = "select one standby to analyze replication stages"
DETAIL_PANEL_TITLES = {
    221: "Replay lag by stage (bytes)",
    222: "Replay lag by stage (time)",
}
REPLICA_FILTER_PANEL_IDS = {221, 222, 241, 242, 243}
APPLICATION_IDENTITY_PANEL_IDS = {211, 212, 233, 241, 242, 243}
CONNECTION_IDENTITY_PANEL_ID = 251


def _load_d6(dashboard_path: Path) -> dict | None:
    if not is_dashboard(dashboard_path, 6):
        return None
    with open(dashboard_path) as f:
        return json.load(f)


def test_shared_crosshair(dashboard_path: Path):
    d = _load_d6(dashboard_path)
    if d is None:
        pytest.skip("not dashboard 06")
    assert d.get("graphTooltip") == 1, (
        "dashboard 06 correlates lag, WAL rate and feedback across panels; it "
        "must declare graphTooltip=1 (shared crosshair) like dashboards 1/8/9/"
        "12/13/14."
    )


def test_replica_variable_declared(dashboard_path: Path):
    d = _load_d6(dashboard_path)
    if d is None:
        pytest.skip("not dashboard 06")
    variables = {v["name"]: v for v in d["templating"]["list"]}
    assert "replica" in variables, "dashboard 06 must expose a `replica` selector"
    replica = variables["replica"]
    assert replica.get("includeAll") is True, (
        "the selector must default to All standbys, like every db_name variable "
        "in the repo"
    )


def test_replica_filter_is_limited_to_detail_panels(dashboard_path: Path):
    d = _load_d6(dashboard_path)
    if d is None:
        pytest.skip("not dashboard 06")

    filtered_panel_ids = {
        panel["id"]
        for panel in iter_panels(d)
        for target in panel.get("targets") or []
        if REPLICA_FILTER in target_query(target)
    }
    assert filtered_panel_ids == REPLICA_FILTER_PANEL_IDS, (
        "the Standby selector must filter only the detail and per-standby "
        "health panels; overview and comparison panels stay cluster-wide"
    )


def test_detail_panels_are_gated_and_prompt(dashboard_path: Path):
    d = _load_d6(dashboard_path)
    if d is None:
        pytest.skip("not dashboard 06")

    panels_by_id = {p.get("id"): p for p in iter_panels(d)}
    detail_panels = [panels_by_id[panel_id] for panel_id in DETAIL_PANEL_TITLES]
    assert {
        panel["id"]: panel.get("title") for panel in detail_panels
    } == DETAIL_PANEL_TITLES, (
        "dashboard 06 detail panel titles must remain short house-style noun "
        "phrases"
    )
    assert any(
        panel.get("type") == "row"
        and DETAIL_MARKER in (panel.get("title") or "").lower()
        for panel in iter_panels(d)
    ), (
        f"expected the detail row title to retain the hint {DETAIL_MARKER!r}"
    )

    problems: list[str] = []
    for panel in detail_panels:
        where = f"panel {panel.get('id')} ({panel.get('title')})"
        if (panel.get("fieldConfig", {}).get("defaults", {}).get("noValue")
                != PROMPT):
            problems.append(f"{where}: fieldConfig.defaults.noValue must be {PROMPT!r}")
        for target in panel.get("targets") or []:
            sql = target_query(target)
            # The reference-line target is intentionally ungated: it is a
            # constant, not a per-standby series.
            if "pgwatch_settings_" in sql:
                continue
            gate = sql[sql.find("), c AS ("):]
            if (GATE_MARKER not in gate or RANGE_WINDOW not in gate
                    or not re.search(r"FROM \w+ JOIN c USING \(t\)", sql)):
                problems.append(
                    f"{where} target {target.get('refId')}: missing the "
                    f"'exactly one standby' gate"
                )

    assert not problems, "\n  ".join(["dashboard 06 detail contract:"] + problems)


def test_per_standby_graphs_key_series_by_application_name(dashboard_path: Path):
    d = _load_d6(dashboard_path)
    if d is None:
        pytest.skip("not dashboard 06")

    panels_by_id = {p.get("id"): p for p in iter_panels(d)}
    problems: list[str] = []
    for panel_id in APPLICATION_IDENTITY_PANEL_IDS:
        panel = panels_by_id[panel_id]
        where = f"panel {panel_id} ({panel.get('title')})"
        if "identity is `application_name`" not in (panel.get("description") or ""):
            problems.append(f"{where}: description must state the logical identity")
        for target in panel.get("targets") or []:
            sql = target_query(target)
            if "pgwatch_replication_" not in sql:
                continue
            if not BY_APPLICATION.search(sql):
                problems.append(
                    f"{where} target {target.get('refId')}: must aggregate "
                    "reconnects by application_name"
                )
            if "client_info" in sql:
                problems.append(
                    f"{where} target {target.get('refId')}: client_info must "
                    "not split a logical standby"
                )
            if SERIES_BY_APPLICATION not in sql:
                problems.append(
                    f"{where} target {target.get('refId')}: series must be "
                    "named by application_name"
                )

    connection_panel = panels_by_id[CONNECTION_IDENTITY_PANEL_ID]
    connection_description = connection_panel.get("description") or ""
    if "one per-connection panel" not in connection_description:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: description must state "
            "why client_info is part of physical connection identity"
        )
    included_fields = next(
        t for t in connection_panel["transformations"] if t["id"] == "filterFieldsByName"
    )["options"]["include"]["pattern"]
    if "client_info" not in included_fields:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: connection table must "
            "retain client_info"
        )
    connection_sql = target_query(connection_panel["targets"][0])
    if """q.labels['client_info'] AS "client_info\"""" not in connection_sql:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: the query must return client_info"
        )
    if BY_APPLICATION.search(connection_sql):
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: physical connections "
            "must not be collapsed by application_name"
        )

    assert not problems, "\n  ".join(
        ["dashboard 06 standby identity contract:"] + problems
    )
