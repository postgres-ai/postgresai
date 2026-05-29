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
    ROOT / "config" / "grafana" / "dashboards" / "Dashboard_7_Autovacuum_and_xmin_horizon.json"
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
        self.assertIn("usename <> current_user", body)

    def test_blocker_activity_filters_client_backends_with_xmin(self) -> None:
        body = cte_body("xmin_horizon_blockers", "activity")

        self.assertIn("from pg_stat_activity", body)
        self.assertIn("pid <> pg_backend_pid()", body)
        self.assertIn("backend_type = 'client backend'", body)
        self.assertIn("backend_xmin is not null", body)
        self.assertIn("usename <> current_user", body)

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
        panel = dashboard_panel("Autovacuum workers blocked on lock")
        targets = panel.get("targets", [])

        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].get("instant"))
        self.assertIn(
            "last_over_time(pgwatch_pg_autovacuum_blocked_wait_seconds",
            targets[0]["expr"],
        )
        self.assertIn("[1m]", targets[0]["expr"])

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


class TableStatsMxidFreezeAgeTest(unittest.TestCase):
    def test_table_stats_exports_mxid_freeze_age(self) -> None:
        # The new wraparound top-N panels query
        # `pgwatch_table_stats_mxid_freeze_age`. Pin both the column alias and
        # the corrected `mxid_age(...)` function (was `age(...)` previously,
        # which wraps to ~2^31-1 on a fresh DB with relminmxid=1).
        sql_text = normalized(metric_sql("table_stats"))
        self.assertIn("mxid_age(c.relminmxid)", sql_text)
        self.assertIn("as mxid_freeze_age", sql_text)
        # Guard the regression: only `mxid_age(c.relminmxid)` is correct.
        # `then age(c.relminmxid)` (without the `mxid_` prefix) was the bug
        # that wrapped the freeze-age past 2^31 on fresh databases.
        self.assertNotRegex(sql_text, r"\bthen\s+age\(c\.relminmxid\)")


class PgSettingsWraparoundTest(unittest.TestCase):
    def test_metric_definition_contract(self) -> None:
        definition = metric_def("pg_settings_wraparound")
        self.assertIn(11, definition["sqls"])
        self.assertEqual(definition.get("gauges"), ["*"])
        # pg_settings is cluster-global; must be is_instance_level so pgwatch
        # scrapes it once per cluster instead of once per database.
        self.assertTrue(definition.get("is_instance_level"))

    def test_sql_does_not_tag_with_datname(self) -> None:
        # Cluster-global values must not carry a per-database tag — that
        # creates duplicate Prometheus series and misleads dashboards.
        sql_text = normalized(metric_sql("pg_settings_wraparound"))
        self.assertNotIn("as tag_datname", sql_text)
        self.assertNotIn("current_database()", sql_text)

    def test_sql_exposes_threshold_columns(self) -> None:
        sql_text = normalized(metric_sql("pg_settings_wraparound"))
        # Soft-freeze GUCs (always present back to PG 9.x).
        self.assertIn("'autovacuum_freeze_max_age'", sql_text)
        self.assertIn("'autovacuum_multixact_freeze_max_age'", sql_text)
        self.assertIn("as autovacuum_freeze_max_age", sql_text)
        self.assertIn("as autovacuum_multixact_freeze_max_age", sql_text)
        # Failsafe GUCs are PG14+ only — must be coalesced to 0 so the
        # query does not blow up on PG 11–13 where the setting is missing.
        self.assertIn("'vacuum_failsafe_age'", sql_text)
        self.assertIn("'vacuum_multixact_failsafe_age'", sql_text)
        self.assertRegex(
            sql_text,
            r"coalesce\(\s*\(\s*select setting::int8 from pg_settings "
            r"where name = 'vacuum_failsafe_age'\s*\)\s*,\s*0\s*\)",
        )
        self.assertRegex(
            sql_text,
            r"coalesce\(\s*\(\s*select setting::int8 from pg_settings "
            r"where name = 'vacuum_multixact_failsafe_age'\s*\)\s*,\s*0\s*\)",
        )


class PgAutovacuumWorkersTest(unittest.TestCase):
    def test_metric_definition_contract(self) -> None:
        definition = metric_def("pg_autovacuum_workers")
        self.assertIn(11, definition["sqls"])
        self.assertEqual(definition.get("gauges"), ["*"])
        # pg_stat_activity is cluster-wide (active_workers includes backends
        # in any database), so this metric must be is_instance_level.
        self.assertTrue(definition.get("is_instance_level"))

    def test_sql_does_not_tag_with_datname(self) -> None:
        sql_text = normalized(metric_sql("pg_autovacuum_workers"))
        self.assertNotIn("as tag_datname", sql_text)
        self.assertNotIn("current_database()", sql_text)

    def test_sql_derives_active_and_max_in_a_single_cte(self) -> None:
        # active_workers and max_workers must be evaluated once (in a CTE) and
        # reused, otherwise free_slots = max - active drifts when a worker
        # starts/exits between two correlated subqueries.
        sql_text = normalized(metric_sql("pg_autovacuum_workers"))
        self.assertEqual(
            sql_text.count("from pg_stat_activity where backend_type = 'autovacuum worker'"),
            1,
        )
        self.assertEqual(
            sql_text.count("from pg_settings where name = 'autovacuum_max_workers'"),
            1,
        )
        self.assertIn("as active_workers", sql_text)
        self.assertIn("as max_workers", sql_text)
        self.assertIn("max_workers - active_workers", sql_text)
        self.assertIn("as free_slots", sql_text)


class PgAutovacuumQueueTest(unittest.TestCase):
    def test_metric_definition_contract(self) -> None:
        definition = metric_def("pg_autovacuum_queue")
        self.assertIn(11, definition["sqls"])
        self.assertEqual(definition.get("gauges"), ["*"])
        # statement_timeout must match peer autovacuum metrics (15s) — a
        # higher value risks holding scrape connections on large catalogs
        # and masks pathologically-slow per-table catalog scans.
        self.assertEqual(definition.get("statement_timeout_seconds"), 15)

    def test_sql_exposes_threshold_and_overdue_factor(self) -> None:
        sql_text = normalized(metric_sql("pg_autovacuum_queue"))
        # Per-relation overrides via pg_class.reloptions must be applied.
        self.assertIn("pg_options_to_table(c.reloptions)", sql_text)
        self.assertIn("'autovacuum_vacuum_threshold'", sql_text)
        self.assertIn("'autovacuum_vacuum_scale_factor'", sql_text)
        # Threshold and the derived overdue factor are the columns Dashboard 7
        # actually plots — pin both.
        self.assertIn("as autovacuum_threshold", sql_text)
        self.assertIn("as autovacuum_overdue_factor", sql_text)
        # Per-table tags consumed by the dashboard.
        for tag in ("tag_schemaname", "tag_relname"):
            self.assertIn(f"as {tag}", sql_text)
        # Dead/live tuple counts come from pg_stat_all_tables.
        self.assertIn("as n_dead_tup", sql_text)
        self.assertIn("as n_live_tup", sql_text)
        self.assertIn("pg_stat_all_tables", sql_text)

    def test_sql_clamps_reltuples_to_zero(self) -> None:
        # pg_class.reltuples is -1 for tables that have never been ANALYZE'd
        # (PG14+); without `greatest(c.reltuples, 0)` the threshold expression
        # `av_threshold + av_scale_factor * reltuples` evaluates to a value
        # just below av_threshold, producing a meaningless overdue factor.
        sql_text = normalized(metric_sql("pg_autovacuum_queue"))
        self.assertIn("greatest(c.reltuples, 0)", sql_text)

    def test_sql_uses_e_string_escape_for_underscore_glob(self) -> None:
        # SQL-style backslash escapes inside plain string literals depend on
        # standard_conforming_strings; existing metrics in this file use the
        # E'' prefix consistently for `\_` in like patterns. Inside the YAML
        # block scalar the literal backslash survives as `\\`.
        sql_text = normalized(metric_sql("pg_autovacuum_queue"))
        self.assertIn(r"e'pg\\_%'", sql_text)

    def test_sql_does_not_redundantly_tag_table_full_name(self) -> None:
        # tag_schemaname + tag_relname already carry the same information as
        # the previous tag_table_full_name; the redundant tag tripled label
        # cardinality with no added value (dashboards can build the full
        # name in legendFormat as `{{schemaname}}.{{relname}}`).
        sql_text = normalized(metric_sql("pg_autovacuum_queue"))
        self.assertNotIn("as tag_table_full_name", sql_text)


class PgAutovacuumBlockedTest(unittest.TestCase):
    def test_metric_definition_contract(self) -> None:
        definition = metric_def("pg_autovacuum_blocked")
        self.assertIn(11, definition["sqls"])
        self.assertEqual(definition.get("gauges"), ["*"])

    def test_sql_filters_to_autovacuum_workers_before_locks_join(self) -> None:
        # Filtering pg_stat_activity down to autovacuum workers waiting on a
        # lock must happen before joining pg_locks; otherwise the locks join
        # cost is unbounded on busy clusters.
        sql_text = normalized(metric_sql("pg_autovacuum_blocked"))
        self.assertIn("backend_type = 'autovacuum worker'", sql_text)
        self.assertIn("wait_event_type = 'lock'", sql_text)
        worker_filter_pos = sql_text.find("backend_type = 'autovacuum worker'")
        locks_join_pos = sql_text.find("from pg_locks")
        self.assertGreater(locks_join_pos, worker_filter_pos)

    def test_sql_exposes_expected_columns(self) -> None:
        sql_text = normalized(metric_sql("pg_autovacuum_blocked"))
        self.assertIn("as wait_seconds", sql_text)
        for tag in ("tag_worker_pid", "tag_blocker_pid", "tag_blocker_queryid"):
            self.assertIn(f"as {tag}", sql_text)
        # PG13 doesn't expose pg_stat_activity.query_id; using
        # to_jsonb()->>'query_id' keeps the SQL parse-clean across versions.
        self.assertIn("to_jsonb(blocker)->>'query_id'", sql_text)

    def test_sql_does_not_expose_user_controlled_labels(self) -> None:
        # tag_blocker_appname / tag_blocker_user were intentionally dropped:
        # application_name is user-controlled (any client can SET it) and
        # would create unbounded label cardinality plus leak usernames into
        # metric scrape payloads. Look the values up in pg_stat_activity by
        # blocker_pid instead of tagging them onto every series.
        sql_text = normalized(metric_sql("pg_autovacuum_blocked"))
        for forbidden in (
            "as tag_blocker_appname",
            "as tag_blocker_user",
            "as blocker_appname",
            "as blocker_user",
        ):
            self.assertNotIn(forbidden, sql_text)

    def test_inner_cte_projects_only_columns_consumed_downstream(self) -> None:
        # Keep the inner CTE projection minimal — anything not referenced in
        # the outer select is dead weight that masks future refactors.
        body = cte_body("pg_autovacuum_blocked", "av_workers_blocked")
        for unused in (
            "datname",
            "wait_event,",
            "wait_event_type,",
            "query,",
            "query_start)",
        ):
            # `query_start` is consumed; the trailing `)` guard above keeps
            # this assertion from accidentally matching the kept column.
            pass
        # Only `pid` and `query_start` are referenced downstream.
        self.assertIn("select pid, query_start", body)
        for forbidden in (" datname,", " wait_event,", " wait_event_type,", " query,"):
            self.assertNotIn(forbidden, body)


class PgVacuumProgressTest(unittest.TestCase):
    def test_sql_exposes_is_anti_wraparound_across_versions(self) -> None:
        # The MR replaces a brittle query-text regex with
        # `(backend_xid is not null)::int as is_anti_wraparound` — pin it
        # for every supported PG version branch (11 and 17) so a future
        # refactor cannot silently drop the column or revert to the regex.
        definition = metric_def("pg_vacuum_progress")
        for version in (11, 17):
            self.assertIn(version, definition["sqls"])
            sql_text = normalized(sql_for_version(definition["sqls"], version))
            self.assertIn("as is_anti_wraparound", sql_text)
            self.assertRegex(
                sql_text,
                r"\(\s*[a-z]+\.backend_xid\s+is\s+not\s+null\s*\)::int",
            )


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
