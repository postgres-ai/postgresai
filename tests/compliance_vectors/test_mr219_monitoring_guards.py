"""Regression coverage for MR !219 guardrails on the collector SQL in mon/metrics.yml.

The guards on the old stack (scrape limits, the pgwatch image, the sink's
query-id trigger, Dashboard 2's PromQL) are in old/tests/compliance_vectors/.
"""
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_multixact_size_checks_pg_stat_file_execute_privilege():
    metrics = yaml.safe_load(
        (PROJECT_ROOT / "mon/metrics.yml").read_text()
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
            PROJECT_ROOT / "mon/metrics.yml",
            "pg_stat_statements",
            "exec_time_total",
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
        (PROJECT_ROOT / "mon/metrics.yml").read_text()
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
        (PROJECT_ROOT / "mon/metrics.yml").read_text()
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
        (PROJECT_ROOT / "mon/metrics.yml").read_text()
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
