"""Bug 2 — Dashboard titles must not leak TODO/WIP/FIXME/XXX markers or `--`.

The rc.6 demo surfaced Dashboard 6 with the literal title:

    06. Replication and HA  -- "Metrics are collected (part of health check);
    dashboard – TODO"

This is a developer note that shipped to production. Clean titles only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Markers that indicate an unfinished / draft title (case-insensitive)
FORBIDDEN_SUBSTRINGS = ("TODO", "WIP", "XXX", "FIXME")

# A double dash usually signals an inline aside; if a future title legitimately
# needs an em-dash, use the unicode em-dash ("—") instead.
DOUBLE_DASH_PATTERN = re.compile(r"\s--\s|^--\s|\s--$")


def test_dashboard_title_clean(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    title = dashboard.get("title") or ""

    found: list[str] = []
    upper = title.upper()
    for marker in FORBIDDEN_SUBSTRINGS:
        if marker in upper:
            found.append(marker)
    if DOUBLE_DASH_PATTERN.search(title):
        found.append("` -- ` (inline double-dash aside; use em-dash if needed)")

    assert not found, (
        f"{dashboard_path.name}: dashboard title leaks draft markers "
        f"{found!r}: {title!r}"
    )
