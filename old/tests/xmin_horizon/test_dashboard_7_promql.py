"""Old stack: Dashboard 7 PromQL guards (old/config/grafana/dashboards, the input of
mon/grafana/compile.py). The live checks on the pgai-mon Dashboard 7 are in
tests/xmin_horizon/test_metrics_sql_static.py."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_7_PATH = (
    ROOT / "old" / "config" / "grafana" / "dashboards" / "Dashboard_7_Autovacuum_and_xmin_horizon.json"
)


def dashboard_panel(title: str) -> dict[str, Any]:
    with DASHBOARD_7_PATH.open() as dashboard_file:
        dashboard = json.load(dashboard_file)

    for panel in dashboard["panels"]:
        if panel.get("title") == title:
            return panel
    raise AssertionError(f"Dashboard panel {title!r} not found")


class Dashboard7BlockedPanelTest(unittest.TestCase):
    def test_blocker_dashboard_table_uses_recent_scrape_window(self) -> None:
        panel = dashboard_panel("Autovacuum workers blocked on lock")
        targets = panel.get("targets", [])

        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].get("instant"))
        self.assertIn(
            "last_over_time(pgwatch_pg_autovacuum_blocked_wait_seconds",
            targets[0]["expr"],
        )
        self.assertIn("[1m]", targets[0]["expr"])


class Dashboard7TemplateVarsTest(unittest.TestCase):
    """Pin Dashboard 7 template variables that the new top-N panels rely on.

    `top_n` controls every `topk()` panel and `db_name` scopes per-database
    series; renaming or removing either silently breaks the dashboard with
    no other test catching the regression.
    """

    @staticmethod
    def _template_var(name: str) -> dict[str, Any]:
        with DASHBOARD_7_PATH.open() as dashboard_file:
            dashboard = json.load(dashboard_file)
        for variable in dashboard.get("templating", {}).get("list", []):
            if variable.get("name") == name:
                return variable
        raise AssertionError(f"Template variable {name!r} not found")

    def test_top_n_variable_definition(self) -> None:
        variable = self._template_var("top_n")
        self.assertEqual(variable.get("type"), "custom")
        option_values = {opt.get("value") for opt in variable.get("options", [])}
        # Pin the canonical breakpoints used by every top-N panel; a smaller
        # set would limit operator triage, a larger one would inflate the
        # dropdown without value.
        self.assertEqual(
            option_values,
            {"5", "10", "15", "20", "25", "50", "100"},
        )
        # Default selection must match the documented value (20).
        current = variable.get("current") or {}
        self.assertEqual(current.get("value"), "20")

    def test_db_name_variable_definition(self) -> None:
        variable = self._template_var("db_name")
        self.assertEqual(variable.get("type"), "query")
        # The dropdown must filter to monitored databases visible to the
        # selected cluster/node, excluding the always-empty `template1`.
        query_text = variable.get("query", {}).get("query", "")
        self.assertIn("pgwatch_db_size_size_b", query_text)
        self.assertIn("datname!=\"template1\"", query_text)

    def test_top_n_panels_apply_topk_with_variable(self) -> None:
        with DASHBOARD_7_PATH.open() as dashboard_file:
            dashboard = json.load(dashboard_file)

        expected_panels = {
            "Top-N tables by XID age (relfrozenxid)",
            "Top-N tables by MultiXID age (relminmxid)",
            "Autovacuum debt — top-N overdue tables",
        }
        seen: set[str] = set()
        for panel in dashboard.get("panels", []):
            title = panel.get("title")
            if title not in expected_panels:
                continue
            seen.add(title)
            targets = panel.get("targets", [])
            self.assertTrue(
                any("topk($top_n," in (t.get("expr") or "") for t in targets),
                f"Panel {title!r} does not apply `topk($top_n, ...)`",
            )

        missing = expected_panels - seen
        self.assertFalse(missing, f"Expected top-N panels missing: {sorted(missing)}")



if __name__ == "__main__":
    unittest.main()
