"""Shipped dashboards must not be "Coming soon" placeholders.

Dashboard 6 ("06. Replication and HA") shipped for several releases as a
single markdown text panel reading `# Coming soon...`, even though every
`pgwatch_replication_*` metric it needs was already being collected. On a
customer engagement that meant replication lag had to be investigated with
ad-hoc PromQL against VictoriaMetrics because the dashboard that should
have answered the question was empty.

A dashboard file that exists in the provisioning directory is shipped to
every deployment and shows up in the Grafana dashboard list. Either it has
real panels or it should not be provisioned at all.

Rules enforced here:

1. No panel may render placeholder copy ("coming soon", "todo", "tbd",
   "under construction", "work in progress") as its content or title.
2. Every dashboard must have at least one panel that queries a data
   source. A dashboard made only of `text` / `row` panels is a
   placeholder regardless of what its copy says.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tests.grafana_dashboards.conftest import iter_panels

PLACEHOLDER_RE = re.compile(
    r"coming\s+soon|under\s+construction|work\s+in\s+progress|\btbd\b|\btodo\b",
    re.IGNORECASE,
)

# Panel types that never carry a query; a dashboard built only from these
# has nothing to show.
NON_QUERY_PANEL_TYPES = {"row", "text"}


def test_no_placeholder_copy_in_panels(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    offenders: list[str] = []
    for panel in iter_panels(dashboard):
        haystacks = [
            panel.get("title") or "",
            (panel.get("options") or {}).get("content") or "",
        ]
        for text in haystacks:
            match = PLACEHOLDER_RE.search(text)
            if match:
                offenders.append(
                    f"panel id={panel.get('id')!r} "
                    f"title={panel.get('title')!r} contains "
                    f"placeholder copy {match.group(0)!r}"
                )
                break

    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} placeholder panel(s); a "
        f"provisioned dashboard must be finished or not shipped:\n  "
        + "\n  ".join(offenders)
    )


def test_dashboard_has_at_least_one_query_panel(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    query_panels = [
        panel
        for panel in iter_panels(dashboard)
        if panel.get("type") not in NON_QUERY_PANEL_TYPES
        and (panel.get("targets") or [])
    ]

    assert query_panels, (
        f"{dashboard_path.name}: no panel queries a data source — the "
        f"dashboard is a placeholder but is provisioned to every deployment."
    )
