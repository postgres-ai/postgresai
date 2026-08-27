"""Regression coverage for MR !219 monitoring guardrails."""
import json
import re
from pathlib import Path

import yaml

from tests.grafana_dashboards.query_info_join import (
    QUERY_INFO_JOIN_OPERAND,
    join_operand_problems,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _duration_seconds(value):
    match = re.fullmatch(r"(\d+)([smh])", value)
    assert match is not None
    amount = int(match.group(1))
    multiplier = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
    return amount * multiplier


def test_pgwatch_sample_limit_allows_capped_full_preset():
    prometheus = yaml.safe_load(
        (PROJECT_ROOT / "config/prometheus/prometheus.yml").read_text()
    )
    pgwatch_job = next(
        job for job in prometheus["scrape_configs"]
        if job["job_name"] == "pgwatch-prometheus"
    )

    assert pgwatch_job["sample_limit"] >= 10000
    assert pgwatch_job["sample_limit"] < 50000

    query_info_job = next(
        job for job in prometheus["scrape_configs"]
        if job.get("metrics_path") == "/query_info_metrics"
    )
    assert query_info_job["job_name"] == "query-info"
    assert 1000 <= query_info_job["sample_limit"] <= 10000
    assert _duration_seconds(query_info_job["scrape_timeout"]) < _duration_seconds(
        query_info_job["scrape_interval"]
    )


def test_multixact_size_checks_pg_stat_file_execute_privilege():
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    multixact_sql = next(iter(metrics["metrics"]["multixact_size"]["sqls"].values()))

    assert "has_function_privilege" in multixact_sql
    assert "to_regprocedure('pg_stat_file(text,boolean)')" in multixact_sql
    assert "'execute'" in multixact_sql.lower()


def _compact_sql(sql):
    return re.sub(r"\s+", " ", sql.lower())


def test_pgwatch_metrics_yml_pg_stat_statements_has_top_n_filter():
    metric_checks = [
        (
            PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml",
            "pg_stat_statements",
            "exec_time_total",
        ),
        (
            PROJECT_ROOT / "config/pgwatch-postgres/metrics.yml",
            "pgss_queryid_queries",
            "total_exec_time",
        ),
    ]

    for metrics_path, metric_name, exec_time_column in metric_checks:
        metrics = yaml.safe_load(metrics_path.read_text())
        sqls = metrics["metrics"][metric_name]["sqls"]
        assert sqls
        for sql in sqls.values():
            compact_sql = _compact_sql(sql)
            assert "calls >= 3" in compact_sql
            assert f"{exec_time_column} >= 1000" in compact_sql
            # The 100-row cap may be expressed either as a global LIMIT (single-database
            # collection) or as a per-database window rank (cluster-wide collection).
            # Both bound cardinality; only the partitioning differs.
            assert (
                "limit 100" in compact_sql or "rn <= 100" in compact_sql
            ), f"{metric_name} lost its top-100 cardinality bound"


def test_pgwatch_prometheus_pg_stat_statements_is_cluster_wide():
    """pg_stat_statements is backed by shared memory and holds entries for every
    database in the cluster. Collecting it per-connected-database means a fleet is
    only visible where pgwatch happens to hold a connection, which on a multi-tenant
    instance silently hides almost all query activity.

    Collect cluster-wide instead, and bound cardinality per database so a busy
    tenant cannot crowd the others out of the top-N.
    """
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    metric = metrics["metrics"]["pg_stat_statements"]

    # Collected once per instance, not once per monitored database, so that several
    # sources pointing at one cluster do not each emit the same series.
    assert metric.get("is_instance_level") is True

    for version, sql in metric["sqls"].items():
        compact_sql = _compact_sql(sql)
        assert (
            "datname = current_database()" not in compact_sql
        ), f"v{version} restricts pg_stat_statements to the connected database"
        assert (
            "partition by datname" in compact_sql
        ), f"v{version} must rank statements within each database, not globally"
        assert "tag_datname" in compact_sql


def test_pgwatch_stat_views_use_topn_and_other_bucket():
    """High-cardinality per-relation metrics must bound cardinality by
    RANKING, not by IDENTITY. Read pg_stat_all_*/pg_statio_all_* directly
    (NOT the pg_stat_user_*/pg_statio_user_* views, which silently exclude
    pg_catalog/pg_toast and would hide bloat or hot scans in those
    relations), keep the top 100 by relevance, and aggregate the tail into
    a single `'$other$'` tag row so dashboard totals stay correct.

    The principle: a bloated pg_toast or a heavy _timescaledb_internal
    chunk should appear in the top-N when its activity/size warrants it.
    Schema-name filtering (`pg_stat_user_*` views, `NOT LIKE 'pg_toast%'`,
    `NOT LIKE '_timescaledb%'`) makes those issues invisible. Hand-rolled
    nspname LIKE filters or LIMIT-only truncation likewise silently drop
    the tail and break sums on extension-heavy or schema-heavy databases.

    The `'$other$'` sentinel uses `$` so it can never collide with a real
    schema/table/index identifier (a literal `other` schema or relation
    is a legal Postgres name and would otherwise produce duplicate
    Prometheus series with the synthetic tail bucket).
    """
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    # Per-metric: (base view, ranking expression that must appear inside
    # the row_number() window). Pinning the ORDER BY column guards
    # against a silent revert to the n_live_tup+n_dead_tup heuristic
    # (which starved big-but-static tables) or to a column that ignores
    # the metric's purpose.
    expectations = {
        "pg_stat_all_indexes": (
            "pg_stat_all_indexes",
            "order by idx_scan desc",
        ),
        "pg_stat_all_tables": (
            "pg_stat_all_tables",
            # Catalog-cached page count, not pg_total_relation_size() per row.
            "order by coalesce(c.relpages, 0) desc",
        ),
        "pg_statio_all_tables": (
            "pg_statio_all_tables",
            "order by heap_blks_read desc",
        ),
        "pg_statio_all_indexes": (
            "pg_statio_all_indexes",
            "order by idx_blks_read desc",
        ),
    }
    for metric_name, (base_view, order_by_expr) in expectations.items():
        for sql in metrics["metrics"][metric_name]["sqls"].values():
            compact_sql = _compact_sql(sql)
            # Reads the _all_ view, not the _user_ view — keeps catalog/toast/timescale visible.
            assert f"from {base_view}" in compact_sql, metric_name
            user_view = base_view.replace("_all_", "_user_")
            assert user_view not in compact_sql, metric_name
            # Top-N window + tail aggregation
            assert "row_number() over" in compact_sql, metric_name
            assert order_by_expr in compact_sql, (metric_name, order_by_expr)
            assert "rownum <= 100" in compact_sql, metric_name
            assert "rownum > 100" in compact_sql, metric_name
            # `'$other$'` sentinel cannot collide with a real identifier.
            # The plain `'other'` literal would collide with any schema or
            # relation literally named `other` (a legal Postgres name).
            assert "'$other$'" in compact_sql, metric_name
            assert "'other'::text" not in compact_sql, metric_name
            # Bare-aggregate guard: suppress the tail row when nothing was
            # truncated, so small DBs do not see a spurious all-zero
            # `'$other$'` row in dashboards.
            assert "having count(*) > 0" in compact_sql, metric_name
            # pg_stat_all_tables must not call pg_total_relation_size() per
            # row — the per-row catalog lookup blew past statement_timeout
            # on extension-heavy clusters and could raise on a
            # concurrently-dropped relation. Use the cached relpages join.
            if metric_name == "pg_stat_all_tables":
                assert "pg_total_relation_size(" not in compact_sql, metric_name
                assert "left join pg_class" in compact_sql, metric_name
            # No unfiltered LIMIT-only truncation left in place
            assert "limit 5000" not in compact_sql, metric_name
            # No identity-based schema exclusions sneaking back in.
            assert "schemaname like" not in compact_sql, metric_name
            assert "nspname like" not in compact_sql, metric_name
            assert "'pg_toast'" not in compact_sql, metric_name
            assert "'pg_catalog'" not in compact_sql, metric_name
            assert "_timescaledb" not in compact_sql, metric_name


def test_pgwatch_statio_skips_zero_activity_rows():
    """pg_statio tail is mostly zero-I/O rows on schema-heavy DBs. Skipping
    them cuts cardinality before the top-N cap is even reached and keeps
    the `'$other$'` bucket meaningful. This is NOT identity-based filtering:
    a row with every counter zero literally carries no information and
    cannot mask any issue.

    The OR-chain pins ALL counter fields, not just one — a future edit
    that accidentally collapses the chain to a single field (e.g. only
    heap_blks_read) would silently hide index-only or TOAST-only I/O
    activity.
    """
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    statio_tables_fields = (
        "heap_blks_read > 0",
        "heap_blks_hit > 0",
        "idx_blks_read > 0",
        "idx_blks_hit > 0",
        "toast_blks_read > 0",
        "toast_blks_hit > 0",
        "tidx_blks_read > 0",
        "tidx_blks_hit > 0",
    )
    for sql in metrics["metrics"]["pg_statio_all_tables"]["sqls"].values():
        compact_sql = _compact_sql(sql)
        for field in statio_tables_fields:
            assert field in compact_sql, field
    statio_indexes_fields = (
        "idx_blks_read > 0",
        "idx_blks_hit > 0",
    )
    for sql in metrics["metrics"]["pg_statio_all_indexes"]["sqls"].values():
        compact_sql = _compact_sql(sql)
        for field in statio_indexes_fields:
            assert field in compact_sql, field


def test_pgwatch_dockerfile_sha_pin_and_patch_present():
    dockerfile = (PROJECT_ROOT / "pgwatch/Dockerfile").read_text()

    assert "ARG PGWATCH_SHA=2995dbec0486dea5c5e7dcd502b94fbafbbe2fa5" in dockerfile
    assert 'grep -q \'return fmt.Errorf("unexpected extension %s version input: %s", ext, ver)\'' in dockerfile
    assert "return nil /* skip unparseable extension version */" in dockerfile


def test_docker_compose_pgwatch_services_use_patched_image():
    class ComposeLoader(yaml.SafeLoader):
        pass

    ComposeLoader.add_constructor(
        "!override",
        lambda loader, node: loader.construct_sequence(node),
    )
    compose = yaml.load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(),
        Loader=ComposeLoader,
    )

    for service_name in ("pgwatch-postgres", "pgwatch-prometheus"):
        service = compose["services"][service_name]
        assert service["image"] == "${PGAI_REGISTRY:-postgresai}/pgwatch:${PGAI_TAG:?PGAI_TAG is required}"
        assert service["build"]["context"] == "./pgwatch"


def test_queryid_dedup_trigger_is_partition_safe():
    init_sql = (PROJECT_ROOT / "config/sink-postgres/init.sql").read_text()
    assert init_sql.lower().count(
        "create or replace function enforce_queryid_uniqueness"
    ) == 1
    function_match = re.search(
        r"create or replace function enforce_queryid_uniqueness\(\).*?\$func\$\s*language plpgsql;",
        init_sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert function_match is not None
    function_sql = function_match.group(0).lower()

    assert "drop index if exists public.pgss_queryid_queries_upsert_idx" in init_sql.lower()
    assert "pg_advisory_xact_lock" in function_sql
    assert "hashtext(new.dbname)" in function_sql
    assert "hashtext(queryid_value)" in function_sql
    assert "md5(" not in function_sql
    assert "delete from public.pgss_queryid_queries" in function_sql
    assert "public.pgss_queryid_queries.time <= new.time" in function_sql
    assert "public.pgss_queryid_queries.time > new.time" in function_sql
    assert "update public.pgss_queryid_queries" not in function_sql
    assert "greatest(" not in function_sql
    assert "data->>'queryid'" in function_sql
    assert "new.dbname is null" in function_sql
    assert "on conflict" not in function_sql
    assert "create unique index" not in init_sql.lower()


def test_dashboard_2_pgss_query_info_expressions_have_or_fallbacks():
    # pgwatch_query_info arrives in sparse bursts, so an instant-vector join
    # misses its ~5 min lookback and the legend degrades to raw labels (#344).
    # Both branches need the same carry-forward operand: widening one alone
    # makes `or` emit two series per queryid. This test owns the branch shape;
    # the operand itself is shared with tests/grafana_dashboards/.
    dashboard_paths = [
        PROJECT_ROOT / "config/grafana/dashboards/Dashboard_2_Aggregated_query_analysis.json",
        PROJECT_ROOT / "postgres_ai_helm/config/grafana/dashboards/Dashboard_2_Aggregated_query_analysis.json",
    ]
    missing = []
    checked = 0
    operand = QUERY_INFO_JOIN_OPERAND.pattern
    group_left_pattern = re.compile(
        r"\*\s*on\(queryid\)\s*group_left\([^)]*\)\s*" + operand
    )
    # Deliberately not anchored to end-of-string: this guard owns the branch
    # shape and the operand, not whatever a panel wraps the pair in. An
    # end anchor blocked legitimate reshapes (an outer sum by(), a trailing
    # comparison) for no gain here.
    fallback_pattern = re.compile(
        r"\)\s+or\s+\(.*\s+unless\s+on\(queryid\)\s*" + operand,
        flags=re.DOTALL,
    )

    for dashboard_path in dashboard_paths:
        dashboard = json.loads(dashboard_path.read_text())
        for panel in dashboard.get("panels", []):
            nested_panels = panel.get("panels") if panel.get("collapsed") else None
            for dashboard_panel in nested_panels or [panel]:
                for target in dashboard_panel.get("targets", []) or []:
                    expr = target.get("expr") or ""
                    if "pgwatch_pg_stat_statements_" not in expr:
                        continue
                    if "pgwatch_query_info" not in expr:
                        continue
                    checked += 1
                    group_left_match = group_left_pattern.search(expr)
                    fallback_match = fallback_pattern.search(expr)
                    problems = join_operand_problems(expr)
                    if not group_left_match:
                        problems.append(
                            "group_left branch does not use the carry-forward operand"
                        )
                    if not fallback_match:
                        problems.append(
                            "`or ... unless on(queryid)` fallback does not use "
                            "the carry-forward operand"
                        )
                    if (
                        group_left_match
                        and fallback_match
                        and group_left_match.group(1) != fallback_match.group(1)
                    ):
                        problems.append(
                            f"branch lookbacks differ: {group_left_match.group(1)} "
                            f"vs {fallback_match.group(1)}"
                        )
                    if problems:
                        missing.append(
                            (
                                dashboard_path,
                                dashboard_panel.get("id"),
                                dashboard_panel.get("title"),
                                problems,
                            )
                        )

    assert checked >= 40
    assert missing == []
