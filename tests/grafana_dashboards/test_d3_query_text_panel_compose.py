"""D3 first panel must keep every fix its rawSql now carries.

Dashboard 3 panel id=18 (the "Query text" panel at the top) accumulated four
independent invariants that must coexist. A naive simplification of the query
— or a bad rebase resolution — would silently drop one, so each is asserted
separately against panel id=18:

1. A ``db_name`` reference in the predicate, so the panel reads rows for the
   selected database rather than whichever databases happen to be in the
   sink. Note this does NOT fix the "no default database configured" error
   the original Bug 1 report blamed it for — a rawSql predicate is a row
   filter, not a connection database; that error is fixed at the datasource
   level via ``jsonData.database`` (postgresai#314, and see
   ``test_postgres_panels_pin_database.py``).

2. Graceful "query text not yet collected" fallback that renders when the
   requested queryid has not been scraped yet. Landed on ``main`` in commit
   0916c30 (``fix(dashboards): emit clean var-db_name in drill-down links``)
   via a UNION ALL against a CTE.

3. The db_name predicate spelled ``= ANY(ARRAY[...]::text[])``. ``db_name``
   is multi-value, so quoting it doubles the quoting (postgresai#314 F-1)
   and ``IN (...)`` is a syntax error when "All" resolves to no options —
   the state of a fresh install before the first scrape.

4. ``query_id`` interpolated as ``${query_id:sqlstring}``. It is a
   free-form ``textbox`` any Viewer can set via ``?var-query_id=``, and
   Grafana adds no quotes of its own for a single-value variable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tests.grafana_dashboards.conftest import DB_NAME_VAR_PATTERN

DASHBOARD = (
    Path(__file__).parent.parent.parent
    / "config"
    / "grafana"
    / "dashboards"
    / "Dashboard_3_Single_query_analysis.json"
)
QUERY_TEXT_PANEL_ID = 18

FALLBACK_MARKER = "query text not yet collected"

# The db_name predicate must be anchored to the real_dbname column and test
# set membership via = ANY(ARRAY[...]::type[]), with the variable spelled
# EXACTLY ${db_name:sqlstring}.
#
# ANY(ARRAY[...]) is required over IN (...) because "All" resolves to the
# variable's *options*, which are empty on a fresh install (the Prometheus
# label_values query has nothing to return yet). IN () is a PostgreSQL syntax
# error, which would replace the graceful fallback below with a red panel
# error — the same class of out-of-the-box breakage this fix removes.
# ARRAY[]::text[] is valid and simply matches nothing.
#
# The format spec is pinned rather than left optional (`(?::\w+)?`) because
# swapping it while keeping the ARRAY shape is a real regression this guard
# must catch. `?var-db_name=` accepts arbitrary values — db_name is a query
# variable with `regex: ""`, so Grafana does not validate a URL value against
# its options — and with `:raw` the value is interpolated with no escaping at
# all, giving a working SQL-injection sink. `:csv`, `:json` and `:doublequote`
# render `column "appdb" does not exist`, and `:singlequote` escapes
# apostrophes with a BACKSLASH, which is inert under
# standard_conforming_strings=on and so is also injectable. Only the shipped
# `:sqlstring` (quotes plus '' doubling) is both valid and safe.
DB_NAME_PREDICATE = re.compile(
    r"real_dbname'\)\s+=\s+ANY\(ARRAY\[\$\{db_name:sqlstring\}\]::text\[\]\)"
)

# query_id must be the bare ${query_id:sqlstring} token, anchored to its own
# predicate. Anchoring matters in both directions: the spec makes the token
# self-quoting, so wrapping it in literal quotes renders ''value'' — two
# adjacent literals, a syntax error — and an unanchored substring check would
# still find the token inside those quotes and pass.
QUERY_ID_PREDICATE = re.compile(
    r"queryid'\)\s+=\s+\$\{query_id:sqlstring\}"
)
# Any literal quote immediately before the reference, whatever spec follows.
QUERY_ID_QUOTED = re.compile(r"'\$\{?query_id\b")


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
    """The composed rawSql must reference the db_name variable at all, so the
    panel filters rows to the selected database.

    Deliberately redundant: DB_NAME_PREDICATE below requires the reference in
    a specific position, which implies this. It is kept only because it gives
    a clearer failure when the predicate is deleted outright rather than
    mis-spelled. Do not mistake it for independent coverage — reference
    detection uses conftest's shared matcher, so it also cannot drift from
    test_postgres_panels_pin_database.py."""
    panel = _query_text_panel()
    raw_sql = (panel.get("targets") or [{}])[0].get("rawSql") or ""
    assert DB_NAME_VAR_PATTERN.search(raw_sql), (
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


def test_d3_query_text_panel_db_name_uses_any_array():
    """``db_name`` is multi-value with "All", so Grafana interpolates it as a
    comma-separated list — and to *nothing* when All resolves to no options,
    which is the state of a fresh install before the first scrape. Only
    ``= ANY(ARRAY[...]::text[])`` is valid for both: ``IN ()`` is a
    PostgreSQL syntax error and ``'${db_name}'`` doubles the quoting
    (postgresai#314, F-1). Anchored to the real_dbname column so a regression
    on a different clause cannot slip through."""
    panel = _query_text_panel()
    raw_sql = (panel.get("targets") or [{}])[0].get("rawSql") or ""
    assert "'${db_name}'" not in raw_sql, (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql wraps the multi-value "
        f"${{db_name}} variable in literal single quotes, which doubles the "
        f"quoting after interpolation and breaks the query: {raw_sql!r}"
    )
    assert DB_NAME_PREDICATE.search(raw_sql), (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql must filter real_dbname via "
        f"= ANY(ARRAY[${{db_name:sqlstring}}]::text[]) — IN (...) is a syntax "
        f"error when All resolves to no options: {raw_sql!r}"
    )


def test_d3_query_text_panel_query_id_is_escaped():
    """``query_id`` is a single-value ``textbox`` holding arbitrary
    viewer-supplied input (any Viewer can set it via ``?var-query_id=``).
    Grafana hands such a variable to SQL with apostrophes doubled but no
    surrounding quotes, so the panel's own spelling decides whether that is
    safe: a bare ``$query_id`` would splice raw SQL tokens, and the quoted
    ``'$query_id'`` is one config change from the F-1 breakage (flip the
    variable to multi-value and Grafana adds its own quotes).
    ``${query_id:sqlstring}`` supplies both the quotes and the doubling, so it
    is correct either way."""
    panel = _query_text_panel()
    raw_sql = (panel.get("targets") or [{}])[0].get("rawSql") or ""
    assert QUERY_ID_QUOTED.search(raw_sql) is None, (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql wraps ${{query_id}} in a "
        f"literal quote. Bare '$query_id' splices raw input; and with the "
        f"self-quoting :sqlstring spec it renders ''value'', two adjacent "
        f"literals: {raw_sql!r}"
    )
    assert QUERY_ID_PREDICATE.search(raw_sql), (
        f"D3 panel id={QUERY_TEXT_PANEL_ID} rawSql must match the queryid via "
        f"= ${{query_id:sqlstring}}: {raw_sql!r}"
    )
