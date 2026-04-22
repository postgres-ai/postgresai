"""Static regression tests for xmin horizon pgwatch SQL/dashboard definitions."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "config" / "pgwatch-prometheus" / "metrics.yml"
DASHBOARD_7_PATH = (
    ROOT / "config" / "grafana" / "dashboards" / "Dashboard_7_Autovacuum_and_bloat.json"
)


def metric_def(metric_name: str) -> dict[str, Any]:
    with METRICS_PATH.open() as metrics_file:
        metrics = yaml.safe_load(metrics_file)
    return metrics["metrics"][metric_name]


def sql_for_version(sqls: dict[int, str], version: int = 11) -> str:
    if version not in sqls:
        raise AssertionError(f"SQL version {version} not found; got {sorted(sqls)}")
    return sqls[version]


def metric_sql(metric_name: str, version: int = 11) -> str:
    return sql_for_version(metric_def(metric_name)["sqls"], version)


def dashboard_panel(title: str) -> dict[str, Any]:
    with DASHBOARD_7_PATH.open() as dashboard_file:
        dashboard = json.load(dashboard_file)

    for panel in dashboard["panels"]:
        if panel.get("title") == title:
            return panel
    raise AssertionError(f"Dashboard panel {title!r} not found")


def normalized(sql_text: str) -> str:
    return re.sub(r"\s+", " ", sql_text).lower()


def postgres_greatest(*values: int | None) -> int | None:
    populated = [value for value in values if value is not None]
    return max(populated) if populated else None


def split_top_level_args(args: str) -> list[str]:
    values: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(args):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            values.append(args[start:index].strip())
            start = index + 1
    values.append(args[start:].strip())
    return values


def function_args_for_alias(
    sql_text: str,
    function_name: str,
    alias: str,
) -> list[str]:
    sql_text = normalized(sql_text)
    alias_marker = f" as {alias}"
    alias_position = sql_text.find(alias_marker)
    if alias_position < 0:
        raise AssertionError(f"Alias {alias} not found")

    function_marker = f"{function_name}("
    function_position = sql_text.rfind(function_marker, 0, alias_position)
    if function_position < 0:
        raise AssertionError(f"Function {function_name} not found before {alias}")

    start = function_position + len(function_marker)
    depth = 1
    for index in range(start, len(sql_text)):
        character = sql_text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return split_top_level_args(sql_text[start:index])

    raise AssertionError(f"Could not parse {function_name} arguments before {alias}")


def cte_body(metric_name: str, cte_name: str) -> str:
    sql_text = normalized(metric_sql(metric_name))
    marker = f"{cte_name} as ("
    marker_position = sql_text.find(marker)
    if marker_position < 0:
        raise AssertionError(f"CTE {cte_name} not found in {metric_name}")

    start = marker_position + len(marker)
    depth = 1
    for index in range(start, len(sql_text)):
        character = sql_text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return sql_text[start:index].strip()

    raise AssertionError(f"Could not parse CTE {cte_name} in {metric_name}")


class xminHorizonSqlStaticTest(unittest.TestCase):
    def test_summary_sql_coalesces_every_component(self) -> None:
        sql_text = normalized(metric_sql("xmin_horizon"))

        for component in (
            "pg_stat_activity",
            "pg_replication_slots",
            "pg_replication_slots_catalog",
            "pg_stat_replication",
            "pg_prepared_xacts",
        ):
            self.assertIn(f"coalesce((select {component}_age_tx", sql_text)
            self.assertIn(f"coalesce((select {component}_count", sql_text)

        self.assertIn("txid_snapshot_xmin(txid_current_snapshot())", sql_text)
        self.assertNotIn("xmin_horizon_age_tx", sql_text)
        self.assertEqual(
            set(
                function_args_for_alias(
                    metric_sql("xmin_horizon"),
                    "greatest",
                    "data_horizon_age_tx",
                )
            ),
            {
                "pg_stat_activity_age_tx",
                "pg_replication_slots_age_tx",
                "pg_stat_replication_age_tx",
                "pg_prepared_xacts_age_tx",
            },
        )
        self.assertEqual(
            set(
                function_args_for_alias(
                    metric_sql("xmin_horizon"),
                    "greatest",
                    "catalog_horizon_age_tx",
                )
            ),
            {
                "pg_stat_activity_age_tx",
                "pg_replication_slots_age_tx",
                "pg_stat_replication_age_tx",
                "pg_prepared_xacts_age_tx",
                "pg_replication_slots_catalog_age_tx",
            },
        )

    def test_slot_summary_separates_xmin_and_catalog_xmin(self) -> None:
        slots_body = cte_body("xmin_horizon", "slots")
        slots_catalog_body = cte_body("xmin_horizon", "slots_catalog")
        summary_body = cte_body("xmin_horizon", "summary")

        self.assertIn("max(age(xmin))", slots_body)
        self.assertIn("where xmin is not null", slots_body)
        self.assertNotIn("catalog_xmin", slots_body)
        self.assertIn("max(age(catalog_xmin))", slots_catalog_body)
        self.assertIn("where catalog_xmin is not null", slots_catalog_body)
        self.assertIn(
            "coalesce((select pg_replication_slots_count from slots), 0)",
            summary_body,
        )
        self.assertIn(
            "coalesce((select pg_replication_slots_catalog_count from slots_catalog), 0)",
            summary_body,
        )

    def test_blocker_detail_has_five_union_branches(self) -> None:
        sql_text = normalized(metric_sql("xmin_horizon_blockers"))
        expected_sources = {
            "activity": "from pg_stat_activity",
            "slots": "from pg_replication_slots",
            "slots_catalog": "from pg_replication_slots",
            "replication": "from pg_stat_replication",
            "prepared": "from pg_prepared_xacts",
        }

        self.assertEqual(sql_text.count(" union all "), 4)
        for branch, source in expected_sources.items():
            self.assertIn(source, cte_body("xmin_horizon_blockers", branch))

    def test_summary_activity_filters_client_backends_with_xmin(self) -> None:
        body = cte_body("xmin_horizon", "activity")

        self.assertIn("count(*)::int8 as pg_stat_activity_count", body)
        self.assertIn("pid <> pg_backend_pid()", body)
        self.assertIn("backend_type = 'client backend'", body)
        self.assertIn("backend_xmin is not null", body)

    def test_blocker_activity_filters_client_backends_with_xmin(self) -> None:
        body = cte_body("xmin_horizon_blockers", "activity")

        self.assertIn("from pg_stat_activity", body)
        self.assertIn("pid <> pg_backend_pid()", body)
        self.assertIn("backend_type = 'client backend'", body)
        self.assertIn("backend_xmin is not null", body)

    def test_blocker_slot_detail_separates_xmin_and_catalog_xmin(self) -> None:
        slots_body = cte_body("xmin_horizon_blockers", "slots")
        catalog_body = cte_body("xmin_horizon_blockers", "slots_catalog")

        self.assertIn("from pg_replication_slots", slots_body)
        self.assertIn("'xmin'::text as tag_slot_xmin_source", slots_body)
        self.assertIn("where xmin is not null", slots_body)
        self.assertIn("age(xmin)::int8 as age_tx", slots_body)
        self.assertNotIn("catalog_xmin", slots_body)
        self.assertIn("from pg_replication_slots", catalog_body)
        self.assertIn("'catalog_xmin'::text as tag_slot_xmin_source", catalog_body)
        self.assertIn("where catalog_xmin is not null", catalog_body)
        self.assertIn("age(catalog_xmin)::int8 as age_tx", catalog_body)

    def test_blocker_detail_selects_oldest_row_deterministically(self) -> None:
        expected_ordering = {
            "activity": "order by age(backend_xmin) desc, pid asc limit 1",
            "slots": "order by age(xmin) desc, slot_name asc limit 1",
            "slots_catalog": "order by age(catalog_xmin) desc, slot_name asc limit 1",
            "replication": "order by age(backend_xmin) desc, pid asc limit 1",
            "prepared": "order by age(transaction) desc, gid asc limit 1",
        }

        for branch, order_clause in expected_ordering.items():
            self.assertIn(order_clause, cte_body("xmin_horizon_blockers", branch))

    def test_blocker_detail_uses_actionable_labels_without_query_text(self) -> None:
        sql_text = normalized(metric_sql("xmin_horizon_blockers"))

        self.assertIn("to_jsonb(a)->>'query_id'", sql_text)
        self.assertIn("'data'::text as tag_horizon_type", sql_text)
        self.assertIn("'catalog'::text as tag_horizon_type", sql_text)
        self.assertIn("slot_name::text as tag_slot_name", sql_text)
        self.assertIn("slot_type::text as tag_slot_type", sql_text)
        self.assertIn("coalesce(plugin, '')::text as tag_slot_plugin", sql_text)
        self.assertIn("'xmin'::text as tag_slot_xmin_source", sql_text)
        self.assertIn("'catalog_xmin'::text as tag_slot_xmin_source", sql_text)
        self.assertIn("to_jsonb(s)->>'wal_status'", sql_text)
        self.assertIn("to_jsonb(s)->>'inactive_since'", sql_text)
        self.assertIn("to_jsonb(s)->>'conflicting'", sql_text)
        self.assertIn("to_jsonb(s)->>'invalidation_reason'", sql_text)
        self.assertIn("coalesce(application_name, '')::text as tag_standby_name", sql_text)
        self.assertIn("gid::text as tag_prepared_gid", sql_text)
        self.assertIn("owner::text as tag_owner", sql_text)

        for unsafe_fragment in (
            "query::text as tag",
            "client_addr",
            "client_hostname",
            "client_port",
        ):
            self.assertNotIn(unsafe_fragment, sql_text)

    def test_xmin_metrics_are_primary_only(self) -> None:
        self.assertEqual(metric_def("xmin_horizon").get("node_status"), "primary")
        self.assertEqual(
            metric_def("xmin_horizon_blockers").get("node_status"),
            "primary",
        )

    def test_xmin_metrics_are_instance_level(self) -> None:
        # Both metrics aggregate cluster-wide state (pg_stat_activity,
        # pg_replication_slots, pg_stat_replication, pg_prepared_xacts).
        # Without is_instance_level, pgwatch would scrape them once per
        # database, producing duplicate Prometheus series.
        self.assertTrue(metric_def("xmin_horizon").get("is_instance_level"))
        self.assertTrue(metric_def("xmin_horizon_blockers").get("is_instance_level"))

    def test_xmin_metrics_expose_expected_gauges(self) -> None:
        # Pin the gauges contract: xmin_horizon exports every column via the
        # ['*'] wildcard; xmin_horizon_blockers exports only age_tx because
        # the rest of its columns are bounded identity labels, not gauges.
        self.assertEqual(metric_def("xmin_horizon").get("gauges"), ["*"])
        self.assertEqual(metric_def("xmin_horizon_blockers").get("gauges"), ["age_tx"])

    def test_blocker_dashboard_table_uses_recent_scrape_window(self) -> None:
        panel = dashboard_panel("Current top blockers")
        targets = panel.get("targets", [])

        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].get("instant"))
        self.assertIn(
            "last_over_time(pgwatch_xmin_horizon_blockers_age_tx",
            targets[0]["expr"],
        )
        self.assertIn("[5m]", targets[0]["expr"])

    def test_replication_slots_metric_preserves_null_xmin_age(self) -> None:
        sql_text = metric_sql("replication_slots")
        sql_normalized = normalized(sql_text)

        self.assertEqual(
            set(function_args_for_alias(sql_text, "greatest", "xmin_age_tx")),
            {"age(xmin)", "age(catalog_xmin)"},
        )
        self.assertNotIn("coalesce(age(xmin)", sql_normalized)
        self.assertNotIn("coalesce(age(catalog_xmin)", sql_normalized)

    def test_metric_sql_allows_additive_versions(self) -> None:
        self.assertEqual(
            sql_for_version({11: "select 11", 17: "select 17"}),
            "select 11",
        )

    def test_replication_slots_null_xmin_contract_semantics(self) -> None:
        replication_slots_sql = normalized(metric_sql("replication_slots"))
        xmin_horizon_slots_body = cte_body("xmin_horizon", "slots")
        xmin_horizon_catalog_body = cte_body("xmin_horizon", "slots_catalog")

        self.assertNotIn("coalesce(age(xmin)", replication_slots_sql)
        self.assertNotIn("coalesce(age(catalog_xmin)", replication_slots_sql)
        self.assertNotIn("coalesce(age(xmin)", xmin_horizon_slots_body)
        self.assertNotIn("coalesce(age(catalog_xmin)", xmin_horizon_catalog_body)
        self.assertIn("where xmin is not null", xmin_horizon_slots_body)
        self.assertIn("where catalog_xmin is not null", xmin_horizon_catalog_body)
        self.assertIsNone(postgres_greatest(None, None))
        self.assertEqual(postgres_greatest(None, 7), 7)
        self.assertEqual(postgres_greatest(11, None), 11)


if __name__ == "__main__":
    unittest.main()
