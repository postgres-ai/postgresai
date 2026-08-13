"""Every table-mode legend must declare a default sort that Grafana can apply.

Two failure modes this catches, both seen in the shipped dashboards:

  1. No `sortBy` at all — the legend renders in datasource order, so a top-N
     panel does not put the top offender first. 106 of 150 legends were in this
     state before the convention was applied.

  2. `sortBy` naming a column that `calcs` does not display — Grafana silently
     ignores the sort, so the panel looks configured but renders unsorted.
     Dashboard 08 panel 2 shipped this way: `sortBy: "Last"` with
     `calcs: [min, max, mean]`.

Note `lastNotNull` renders as the column `Last *`, not `Last` — verified
against the reducer registry in the shipped grafana/grafana bundle
(`id: "lastNotNull", name: "Last *"`).

Which key a panel should use (Mean for rates, Max for per-call/latency, Last
for levels) is a judgement call documented in
config/grafana/dashboards/README.md § "Legend sorting"; this test only enforces
that *some* valid sort is declared.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.grafana_dashboards.conftest import iter_panels

# Grafana reducer id -> the legend column name it renders.
REDUCER_COLUMNS = {
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
    "last": "Last",
    "lastNotNull": "Last *",
    "first": "First",
    "firstNotNull": "First *",
    "sum": "Total",
    "count": "Count",
    "range": "Range",
    "diff": "Difference",
    "variance": "Variance",
    "stdDev": "StdDev",
}

CONVENTION = "config/grafana/dashboards/README.md — see 'Legend sorting'"


def test_table_legends_declare_a_usable_sort(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    problems: list[str] = []
    for panel in iter_panels(dashboard):
        legend = (panel.get("options") or {}).get("legend") or {}
        if legend.get("displayMode") != "table":
            continue

        where = f"panel {panel.get('id')} ({panel.get('title') or 'untitled'})"
        sort_by = legend.get("sortBy")
        shown = [
            REDUCER_COLUMNS[c]
            for c in (legend.get("calcs") or [])
            if c in REDUCER_COLUMNS
        ]

        if not sort_by:
            problems.append(
                f"{where}: no 'sortBy' — the legend will render in datasource "
                f"order. Pick a key from {shown or 'the panel calcs'}."
            )
            continue

        if sort_by not in shown:
            problems.append(
                f"{where}: sortBy={sort_by!r} is not a displayed column "
                f"{shown!r}, so Grafana ignores the sort. Add the matching "
                f"reducer to 'calcs' (lastNotNull renders as 'Last *')."
            )

        if legend.get("sortDesc") is not True:
            problems.append(
                f"{where}: sortDesc must be true — the worst offender belongs "
                f"in the first row."
            )

    assert not problems, "{}: {} legend problem(s)\n  {}\n\nConvention: {}".format(
        dashboard_path.name, len(problems), "\n  ".join(problems), CONVENTION
    )
