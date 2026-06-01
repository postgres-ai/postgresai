"""Bug 1 + main(0916c30) compose — D3 first panel must keep BOTH fixes.

Dashboard 3 panel id=18 (the "Query text" panel at the top) carries two
independent fixes that must coexist:

1. ``${db_name}`` template-variable pin in the predicate. Original Bug 1
   fix on this MR's branch — without it Grafana renders a silent "No
   data" with a pink/magenta warning triangle when the provisioned
   Postgres data source has no default database (see
   ``test_postgres_panels_pin_database.py``).

2. Graceful "query text not yet collected" fallback that renders when
   the requested queryid has not been scraped yet. Landed on ``main``
   in commit 0916c30 (``fix(dashboards): emit clean var-db_name in
   drill-down links``) via a UNION ALL against a CTE.

The naive resolution of the rebase conflict between these two changes
would have taken just one side and silently lost the other. This test
asserts both invariants on D3 panel id=18 specifically so future
maintainers cannot accidentally simplify the query and lose either
behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

DASHBOARD = (
    Path(__file__).parent.parent.parent
    / "config"
    / "grafana"
    / "dashboards"
    / "Dashboard_3_Single_query_analysis.json"
)
QUERY_TEXT_PANEL_ID = 18

# Markers we expect in the composed rawSql.
DB_NAME_PIN = "${db_name}"
FALLBACK_MARKER = "query text not yet collected"


def _query_text_panel():
    with open(DASHBOARD) as f:
        dashboard = json.load(f)
    for panel in dashboard.get("panels", []) or []:
        if panel.get("id") == QUERY_TEXT_PANEL_ID:
            return panel
    raise AssertionError(
        f"D3 panel id={QUERY_TEXT_PANEL_ID} not found in {DASHBOARD.name}; "
        "did the panel layout change?"
    )


def test_d3_query_text_panel_pins_db_name():
    """The composed rawSql must reference ${db_name} so the partition
    lookup uses the right index AND Grafana does not fall back to a
    missing default database."""
    panel = _query_text_panel()
    raw_sql = (panel.get("targets") or [{}])[0].get("rawSql") or ""
    assert DB_NAME_PIN in raw_sql, (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql lost the ${{db_name}} pin: "
        f"{raw_sql!r}"
    )


def test_d3_query_text_panel_keeps_graceful_fallback():
    """The composed rawSql must keep the 'query text not yet collected'
    fallback so the panel never silently shows 'No data' when the
    queryid is valid but hasn't been scraped yet."""
    panel = _query_text_panel()
    raw_sql = (panel.get("targets") or [{}])[0].get("rawSql") or ""
    assert FALLBACK_MARKER in raw_sql, (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql lost the graceful "
        f"'query text not yet collected' fallback: {raw_sql!r}"
    )
