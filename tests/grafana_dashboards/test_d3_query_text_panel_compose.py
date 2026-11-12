"""Dashboard 3 panel 18 (the statement's text) must keep each of its invariants.

The panel runs mon/sql/reports/panels/query_text.sql on pgai-mon (it was raw SQL on the pgwatch
Postgres sink). Four independent properties have to coexist, and a simplification of the query
would silently drop one, so each is asserted separately:

1. The selected databases filter the rows: ``regexp_full_match(dbname, '${db_name:regex}')``.
   db_name is multi-value with All; the regex format renders one value, several as (a|b), and
   All with no options yet (a fresh install) as a regex that matches nothing, all valid SQL.

2. A "query text not yet collected" row when the queryid has no stored text, so the panel never
   shows a bare "No data" for a valid queryid that was not scraped yet.

3. One row: the newest text (``ORDER BY last_seen DESC ... LIMIT 1``), since query_texts keeps
   one row per stored text and per database.

4. ``query_id`` interpolated as ``${query_id:sqlstring}`` with no quotes of the query's own. It is
   a free-form textbox any Viewer can set with ``?var-query_id=``; sqlstring makes Grafana add the
   quotes and double a quote inside the value. Wrapped in quotes as well it would render
   ''value'', two adjacent literals.
"""
from __future__ import annotations

import json
import re

from tests.grafana_dashboards.conftest import dashboard_file, iter_panels, target_query

QUERY_TEXT_PANEL_ID = 18
FALLBACK_MARKER = "query text not yet collected"

DB_NAME_PREDICATE = re.compile(r"regexp_full_match\(dbname, \('\$\{db_name:regex\}'\)\)")
QUERY_ID_PREDICATE = re.compile(r"CAST\(queryid AS VARCHAR\) = \(\$\{query_id:sqlstring\}\)")
# a literal quote right before the reference, whatever format follows
QUERY_ID_QUOTED = re.compile(r"'\$\{?query_id\b")


def _sql() -> str:
    dashboard = json.loads(dashboard_file(3).read_text(encoding="utf-8"))
    for panel in iter_panels(dashboard):
        if panel.get("id") == QUERY_TEXT_PANEL_ID:
            targets = panel.get("targets") or []
            assert len(targets) == 1 and targets[0].get("columns") == [
                {"selector": "Query text", "text": "Query text", "type": "string"}]
            return target_query(targets[0])
    raise AssertionError(f"Dashboard 3 panel id={QUERY_TEXT_PANEL_ID} not found; did the layout change?")


def test_d3_query_text_panel_filters_the_selected_databases():
    sql = _sql()
    assert DB_NAME_PREDICATE.search(sql), f"the db_name filter is gone or not a regex match: {sql!r}"
    assert "'${db_name}'" not in sql and "${db_name:sqlstring}" not in sql, sql


def test_d3_query_text_panel_keeps_graceful_fallback():
    sql = _sql()
    assert FALLBACK_MARKER in sql, f"lost the 'query text not yet collected' row: {sql!r}"
    assert "WHERE NOT EXISTS (SELECT 1 FROM q)" in sql, f"the fallback row is not conditional: {sql!r}"


def test_d3_query_text_panel_returns_the_newest_text_only():
    sql = _sql()
    assert re.search(r"FROM query_texts\b.*ORDER BY last_seen DESC.*LIMIT 1\b", sql, re.S), sql


def test_d3_query_text_panel_query_id_is_escaped():
    sql = _sql()
    assert QUERY_ID_QUOTED.search(sql) is None, f"${{query_id}} sits inside a literal quote: {sql!r}"
    assert QUERY_ID_PREDICATE.search(sql), f"queryid must be matched to ${{query_id:sqlstring}}: {sql!r}"


def test_every_query_id_reference_is_sqlstring():
    """The other Dashboard 3 panels filter series by query_id too: same rule."""
    dashboard = json.loads(dashboard_file(3).read_text(encoding="utf-8"))
    refs = [m.group(0) for p in iter_panels(dashboard) for t in p.get("targets") or []
            for m in re.finditer(r".\$\{query_id[^}]*\}", target_query(t))]
    assert len(refs) >= 30
    assert set(refs) == {"(${query_id:sqlstring}"}, set(refs)
