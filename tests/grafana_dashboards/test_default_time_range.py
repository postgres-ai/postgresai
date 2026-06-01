"""Bug 5 — Dashboard default time ranges must be no wider than 1 hour.

On a fresh-deployed demo, dashboards default to "Last 24 hours" render as
thin bars at the far right of the chart — they look broken. The ASH panel
is particularly affected. Narrow the default window so newly populated
data fills the canvas.

Allowed `time.from` values are the Grafana relative shortcuts that resolve
to at most 1 hour: `now-30m`, `now-45m`, `now-1h`, `now-3600s`, etc.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Grafana relative-time tokens. Map each unit to seconds.
_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 86400 * 7,
    "M": 86400 * 30,
    "y": 86400 * 365,
}

_NOW_REL_RE = re.compile(r"^now(?:-(\d+)([smhdwMy]))?(?:/[smhdwMy])?$")

ONE_HOUR_SECONDS = 3600


def _parse_relative_seconds(spec: str) -> int | None:
    """Return how many seconds before "now" the spec refers to.

    `now` -> 0
    `now-30m` -> 1800
    `now-1h` -> 3600
    `now-24h` -> 86400
    Anything that doesn't match the relative grammar returns None.
    """
    m = _NOW_REL_RE.match(spec)
    if not m:
        return None
    if m.group(1) is None:
        return 0
    qty = int(m.group(1))
    unit = m.group(2)
    return qty * _UNIT_SECONDS[unit]


def test_default_time_window_at_most_one_hour(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    time_cfg = dashboard.get("time") or {}
    frm = time_cfg.get("from") or ""
    to = time_cfg.get("to") or ""

    seconds = _parse_relative_seconds(frm)
    assert seconds is not None, (
        f"{dashboard_path.name}: time.from={frm!r} is not a relative "
        f"`now-<N><unit>` spec; cannot bound it for demo readability."
    )
    assert seconds <= ONE_HOUR_SECONDS, (
        f"{dashboard_path.name}: time.from={frm!r} resolves to {seconds}s "
        f"({seconds // 60}m), wider than the 1-hour demo cap. On fresh "
        f"deployments wide windows render as thin bars at the right of the "
        f"chart and look broken. Use now-30m or now-1h."
    )
    # Sanity: `time.to` must be `now` (or a sub-now ref). We only enforce
    # that it's a relative token, not its exact form.
    assert _parse_relative_seconds(to) is not None, (
        f"{dashboard_path.name}: time.to={to!r} is not a relative spec."
    )
