"""Every table target shows its series' labels, not only Time and Value (#340).

compile.py takes a table target's label columns from the label keys its metric has in the probe
store (mon/grafana/README.md). A metric without series there compiles to Time and Value alone:
dashboard 05 panel 14 target B listed five normalized settings with no setting names until the
fixture gained pgwatch_settings_setting_normalized. test_compiled_dashboards_current cannot see
that, because the committed JSON and the recompile lose the columns together.
"""
from __future__ import annotations

import json

import pytest

from tests.grafana_dashboards.conftest import dashboard_file, iter_panels


def _table_targets(dashboard: dict):
    for panel in iter_panels(dashboard):
        if panel.get("type") != "table":
            continue
        for target in panel.get("targets") or []:
            if target.get("format") == "table" and target.get("columns"):
                yield panel, target


@pytest.mark.unit
def test_no_table_target_is_only_time_and_value(dashboard_path):
    dashboard = json.loads(dashboard_path.read_text())
    bare = [
        f"panel {panel.get('id')} ({panel.get('title')}) target {target.get('refId')}"
        for panel, target in _table_targets(dashboard)
        if {c["selector"] for c in target["columns"]} <= {"Time", "Value"}
    ]
    assert not bare, (
        f"{dashboard_path.name}: table targets with no label columns; add their metric's series to "
        f"tests/reporter/mon_fixture/synthetic.py, run it, and recompile against the fixture store "
        f"(mon/grafana/README.md, Recompile): {bare}"
    )


@pytest.mark.unit
def test_wal_settings_table_names_each_normalized_value():
    """Dashboard 05 panel 14: the normalized values (B) carry the same labels as the settings (A)."""
    dashboard = json.loads(dashboard_file(5).read_text())
    (panel,) = [p for p in iter_panels(dashboard) if p.get("id") == 14]
    targets = {t["refId"]: t for t in panel["targets"]}
    columns = {ref: [c["selector"] for c in t["columns"]] for ref, t in targets.items()}
    assert "pgwatch_settings_setting_normalized" in targets["B"]["url_options"]["data"]
    assert columns["B"] == columns["A"]
    assert {"setting_name", "setting_value", "unit", "unit_normalized"} <= set(columns["B"])
