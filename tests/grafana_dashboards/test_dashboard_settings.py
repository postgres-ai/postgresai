"""Customer issue screenshots must use UTC, with a shared crosshair on every dashboard."""
from __future__ import annotations

import json
from pathlib import Path


def test_dashboard_timezone_and_tooltip(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    assert dashboard.get("timezone") == "utc", f"{dashboard_path.name}: timezone must be utc"
    assert type(dashboard.get("graphTooltip")) is int and dashboard["graphTooltip"] == 1, (
        f"{dashboard_path.name}: graphTooltip must be 1 (shared crosshair)"
    )
