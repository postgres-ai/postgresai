"""Regression coverage for MR !219 monitoring guardrails."""
import json
import re
from pathlib import Path

import yaml


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
            assert "limit 100" in compact_sql


def test_pgwatch_stat_views_use_topn_and_other_bucket():
    """High-cardinality per-relation metrics must bound cardinality by
    RANKING, not by IDENTITY. Read pg_stat_all_*/pg_statio_all_* directly
    (NOT the pg_stat_user_*/pg_statio_user_* views, which silently exclude
    pg_catalog/pg_toast and would hide bloat or hot scans in those
    relations), keep the top 100 by relevance, and aggregate the tail into
    a single `'other'` tag row so dashboard totals stay correct.

    The principle: a bloated pg_toast or a heavy _timescaledb_internal
    chunk should appear in the top-N when its activity/size warrants it.
    Schema-name filtering (`pg_stat_user_*` views, `NOT LIKE 'pg_toast%'`,
    `NOT LIKE '_timescaledb%'`) makes those issues invisible. Hand-rolled
    nspname LIKE filters or LIMIT-only truncation likewise silently drop
    the tail and break sums on extension-heavy or schema-heavy databases.
    """
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    expectations = {
        "pg_stat_all_indexes": "pg_stat_all_indexes",
        "pg_stat_all_tables": "pg_stat_all_tables",
        "pg_statio_all_tables": "pg_statio_all_tables",
        "pg_statio_all_indexes": "pg_statio_all_indexes",
    }
    for metric_name, base_view in expectations.items():
        for sql in metrics["metrics"][metric_name]["sqls"].values():
            compact_sql = _compact_sql(sql)
            # Reads the _all_ view, not the _user_ view — keeps catalog/toast/timescale visible.
            assert f"from {base_view}" in compact_sql, metric_name
            user_view = base_view.replace("_all_", "_user_")
            assert user_view not in compact_sql, metric_name
            # Top-N window + tail aggregation
            assert "row_number() over" in compact_sql, metric_name
            assert "rownum <= 100" in compact_sql, metric_name
            assert "rownum > 100" in compact_sql, metric_name
            assert "'other'" in compact_sql, metric_name
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
    the `'other'` bucket meaningful. This is NOT identity-based filtering:
    a row with every counter zero literally carries no information and
    cannot mask any issue.
    """
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml").read_text()
    )
    for sql in metrics["metrics"]["pg_statio_all_tables"]["sqls"].values():
        assert "heap_blks_read > 0" in _compact_sql(sql)
    for sql in metrics["metrics"]["pg_statio_all_indexes"]["sqls"].values():
        assert "idx_blks_read > 0" in _compact_sql(sql)


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
    dashboard_paths = [
        PROJECT_ROOT / "config/grafana/dashboards/Dashboard_2_Aggregated_query_analysis.json",
        PROJECT_ROOT / "postgres_ai_helm/config/grafana/dashboards/Dashboard_2_Aggregated_query_analysis.json",
    ]
    missing = []
    checked = 0
    fallback_pattern = re.compile(
        r"\)\s+or\s+\(.*\s+unless\s+on\(queryid\)\s+pgwatch_query_info\)\s*$",
        flags=re.DOTALL,
    )

    for dashboard_path in dashboard_paths:
        dashboard = json.loads(dashboard_path.read_text())
        for panel in dashboard.get("panels", []):
            nested_panels = panel.get("panels") if panel.get("collapsed") else None
            for dashboard_panel in nested_panels or [panel]:
                for target in dashboard_panel.get("targets", []) or []:
                    expr = target.get("expr") or ""
                    if "pgwatch_pg_stat_statements_" in expr and "pgwatch_query_info" in expr:
                        checked += 1
                        if not fallback_pattern.search(expr):
                            missing.append((dashboard_path, dashboard_panel.get("id"), dashboard_panel.get("title")))

    assert checked >= 40
    assert missing == []
