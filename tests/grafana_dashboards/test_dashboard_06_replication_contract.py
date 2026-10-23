"""Dashboard 06 contract: shared crosshair, logical standby identity, the
replica selector, and the "exactly one standby" detail gate.

The 2026-08-20 live review found the detail panels indistinguishable from the
summary panels and silently rendering a blend of every standby when the
selector was left on All. The fix is structural (a `replica` template variable
plus a range-wide PromQL gate that erases the detail series unless the selection
resolves to exactly one application name), so it is worth pinning. Logical
per-standby graphs are also keyed only by application_name so client-port churn
does not create duplicate legends; the connection table deliberately retains
client_info as its physical-connection identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import iter_panels

DASHBOARD_FILENAME = "Dashboard_6_Replication_and_HA.json"
DETAIL_MARKER = "select one standby"
GATE_MARKER = "count by (application_name)"
PROMPT = "select one standby to analyze replication stages"
DETAIL_PANEL_TITLES = {
    221: "Replay lag by stage (bytes)",
    222: "Replay lag by stage (time)",
}
REPLICA_FILTER_PANEL_IDS = {221, 222, 241, 242, 243}
APPLICATION_IDENTITY_PANEL_IDS = {211, 212, 233, 241, 242, 243}
CONNECTION_IDENTITY_PANEL_ID = 251


def _load_d6(dashboard_path: Path) -> dict | None:
    if dashboard_path.name != DASHBOARD_FILENAME:
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
        if 'application_name=~"$replica"' in (target.get("expr") or "")
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
            expr = target.get("expr") or ""
            # The reference-line target is intentionally ungated: it is a
            # constant, not a per-standby series.
            if "pgwatch_settings_" in expr:
                continue
            if (GATE_MARKER not in expr or "max_over_time" not in expr
                    or "[$__range]" not in expr or "== 1" not in expr):
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
            expr = target.get("expr") or ""
            if "pgwatch_replication_" not in expr:
                continue
            if "max by (application_name)" not in expr:
                problems.append(
                    f"{where} target {target.get('refId')}: must aggregate "
                    "reconnects by application_name"
                )
            if "client_info" in expr:
                problems.append(
                    f"{where} target {target.get('refId')}: client_info must "
                    "not split a logical standby"
                )
            if target.get("legendFormat") != "{{application_name}}":
                problems.append(
                    f"{where} target {target.get('refId')}: legend must be "
                    "{{application_name}}"
                )

    connection_panel = panels_by_id[CONNECTION_IDENTITY_PANEL_ID]
    connection_description = connection_panel.get("description") or ""
    if "one per-connection panel" not in connection_description:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: description must state "
            "why client_info is part of physical connection identity"
        )
    included_fields = connection_panel["transformations"][0]["options"][
        "include"
    ]["pattern"]
    if "client_info" not in included_fields:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: connection table must "
            "retain client_info"
        )
    connection_expr = connection_panel["targets"][0].get("expr") or ""
    if "max by (application_name)" in connection_expr:
        problems.append(
            f"panel {CONNECTION_IDENTITY_PANEL_ID}: physical connections "
            "must not be collapsed by application_name"
        )

    assert not problems, "\n  ".join(
        ["dashboard 06 standby identity contract:"] + problems
    )
