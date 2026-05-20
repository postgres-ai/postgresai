"""Integration coverage for MRs !262 and !267: execute the rewritten pgwatch
per-relation SQLs against a real PostgreSQL.

The compliance-vector tests in `test_mr219_monitoring_guards.py` are
substring assertions on the YAML — they cannot catch a typo in the
window function, a UNION ALL column-count mismatch, a misplaced HAVING,
or a stray cast that breaks at scrape time. This file actually runs each
SQL against a live cluster and asserts:

  (a) the query parses and executes without error,
  (b) the result set is bounded to <= 101 rows per database (100 top-N
      rows plus at most one `'$other$'` aggregate row),
  (c) when there are more than 100 source relations the `'$other$'` row
      is present, occurs exactly once, and its aggregated counters equal
      the sum of the excluded tail (only checked for metrics where (c) is
      cleanly testable on a seeded PG cluster — see `FULL_CROSS_CHECK`
      vs `SYNTAX_ONLY` below).

The HAVING-count(*)>0 guard (no `'$other$'` row when the tail is empty)
is exercised separately via a temporary table that exposes <=100 rows
of the same shape to the top-N + UNION ALL block.

Marked `integration` + `requires_postgres` so the default
`--disable-socket` unit-test run skips the file; the CI reporter:tests
job (already provisions a Postgres cluster) picks it up under
`--run-integration`.

Scope by MR:
  - !262: pg_stat_all_indexes, pg_stat_all_tables,
          pg_statio_all_tables, pg_statio_all_indexes (4 metrics)
  - !267: pg_total_relation_size, pg_class, table_size_detailed,
          table_stats (PG11 + PG16+ variants) get the full cross-check;
          pg_table_bloat, pg_btree_bloat, unused_indexes,
          redundant_indexes, rarely_used_indexes, pg_invalid_indexes
          get a SQL-parses-and-runs check only (forcing > 100 rows
          for those would require multi-MiB seed tables, the
          postgres_ai.pg_statistic helper view, or
          deliberately-redundant index pairs — the substring tests in
          test_mr219_monitoring_guards.py pin the structural shape;
          the value here is catching syntax/cast/column errors).
"""
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"


def _load_sql(metric_name: str, version: int | None = None) -> str:
    """Load the SQL for a metric. If `version` is None, assert there's
    only one variant (catches future divergence on single-variant metrics).
    If specified, return the SQL for that PG major version.

    `table_stats` ships PG11 and PG16+ variants (the PG16+ adds
    `last_seq_scan_s` and an inner relpages-based pre-filter for big
    catalogs); both must get integration coverage.
    """
    metrics = yaml.safe_load(METRICS_YML.read_text())
    sqls = metrics["metrics"][metric_name]["sqls"]
    if version is None:
        assert len(sqls) == 1, (
            f"{metric_name}: expected exactly one SQL version, got "
            f"{list(sqls)}; update this test to iterate every variant"
        )
        return next(iter(sqls.values()))
    assert version in sqls, (
        f"{metric_name}: requested SQL for PG {version}, but only have "
        f"variants for: {list(sqls)}"
    )
    return sqls[version]


# Metrics from !262 — fully cross-checked (tail sum == view sum).
METRIC_KEYS = (
    "pg_stat_all_indexes",
    "pg_stat_all_tables",
    "pg_statio_all_tables",
    "pg_statio_all_indexes",
)

# !267: metrics that get the full cross-check (parse + cap + '$other$'
# tail sum). All read from pg_class / pg_stat_all_tables on relations we
# can seed in bulk; no helper views needed; the >100-row threshold is
# trivially reachable.
#
# `table_stats` is NOT in this tuple — it is added separately by
# `_full_cross_check_params()`, one entry per PG-major SQL variant, via
# `TABLE_STATS_VARIANTS`. Keep this tuple for single-variant metrics only.
NEW_METRICS_FULL_CROSS_CHECK = (
    "pg_total_relation_size",
    "pg_class",
    "table_size_detailed",
)

# `table_stats` parametrized over its SQL variants. Tested separately so
# each variant gets its own (metric, version) pytest id.
TABLE_STATS_VARIANTS = (11, 16)

# !267: metrics that get only the "parses and runs against a real PG"
# check. These have either:
#   - helper-view dependencies (postgres_ai.pg_statistic, only created on
#     monitored DBs by pgwatch's --create-helpers flow; we mirror it in
#     the fixture, but the >1 MiB filter inside the bloat SQL would
#     require ~110 multi-MiB seed tables to force the '$other$' bucket —
#     too expensive for CI runtime), or
#   - filter patterns that make >100 rows artificial to construct
#     (idx_scan=0 indexes; redundant index pairs; invalid indexes).
# The substring tests in test_mr219_monitoring_guards.py pin the
# structural shape of every '$other$' bucket; the value here is making
# sure the SQL itself parses, casts cleanly, and runs without error on
# whichever rows the seed happens to produce.
NEW_METRICS_SYNTAX_ONLY = (
    "pg_table_bloat",
    "pg_btree_bloat",
    "unused_indexes",
    "redundant_indexes",
    "rarely_used_indexes",
    "pg_invalid_indexes",
)

# pgwatch's `--create-helpers` flow creates `postgres_ai.pg_statistic`
# on every monitored DB so the role can read column-level stats without
# the `pg_read_all_stats` privilege. On a stock test cluster the schema
# doesn't exist, so the bloat SQLs error out at parse time without it.
# A view over `pg_stats` works, but we can only project the concrete-typed
# columns: `pg_stats.most_common_vals` and a couple of others have type
# `anyarray` (a pseudo-type), and PostgreSQL refuses to create a view
# that exposes any column with a pseudo-type. The bloat SQLs only read
# `schemaname`, `tablename`, `attname`, `null_frac`, `avg_width`,
# `inherited` — project exactly those.
POSTGRES_AI_HELPER_VIEW_SQL = """
create schema if not exists postgres_ai;
create or replace view postgres_ai.pg_statistic as
  select schemaname, tablename, attname, inherited, null_frac, avg_width
    from pg_stats;
"""

# Output column name that holds the '$other$' literal in each metric.
# Most metrics use `tag_schemaname`, but `table_stats` and
# `table_size_detailed` use `tag_schema`, and the index metrics use
# `tag_schema_name` (snake-cased). Used to distinguish the aggregate row
# from the top-100.
SCHEMA_TAG_COL = {
    "pg_stat_all_indexes": "tag_schemaname",
    "pg_stat_all_tables": "tag_schemaname",
    "pg_statio_all_tables": "tag_schemaname",
    "pg_statio_all_indexes": "tag_schemaname",
    "pg_total_relation_size": "tag_schemaname",
    "pg_class": "tag_schemaname",
    "table_size_detailed": "tag_schema",
    "table_stats": "tag_schema",
    "pg_table_bloat": "tag_schemaname",
    "pg_btree_bloat": "tag_schemaname",
    "unused_indexes": "tag_schema_name",
    "redundant_indexes": "tag_schema_name",
    "rarely_used_indexes": "tag_schema_name",
    "pg_invalid_indexes": "tag_schema_name",
}

# Per metric: one numeric column that must appear in the output. Used by
# the SYNTAX_ONLY column-name assertion so a UNION ALL alias swap between
# the top-N arm and the '$other$' arm is caught at test time (psycopg2
# returns tuples, so without checking cur.description a column-name
# mismatch between arms would silently pass).
EXPECTED_NUMERIC_COL = {
    "pg_table_bloat": "bloat_size",
    "pg_btree_bloat": "bloat_size",
    "unused_indexes": "index_size_bytes",
    "redundant_indexes": "index_size_bytes",
    "rarely_used_indexes": "index_size_bytes",
    "pg_invalid_indexes": "index_size_bytes",
}

# Per metric: the base view it reads from, and a SQL expression that
# (after WHERE filters baked into the metric) gives the count of source
# rows that should be ranked. This is the denominator that decides
# whether the `'$other$'` bucket fires.
SOURCE_ROW_COUNT_SQL = {
    "pg_stat_all_indexes":
        "select count(*) from pg_stat_all_indexes",
    "pg_stat_all_tables":
        "select count(*) from pg_stat_all_tables",
    "pg_statio_all_tables": (
        "select count(*) from pg_statio_all_tables where "
        "heap_blks_read > 0 or heap_blks_hit > 0 "
        "or idx_blks_read > 0 or idx_blks_hit > 0 "
        "or toast_blks_read > 0 or toast_blks_hit > 0 "
        "or tidx_blks_read > 0 or tidx_blks_hit > 0"
    ),
    "pg_statio_all_indexes": (
        "select count(*) from pg_statio_all_indexes where "
        "idx_blks_read > 0 or idx_blks_hit > 0"
    ),
}

# Per-metric mapping of output column name -> source-view expression used
# to cross-check the top-N + `'$other$'` total against the underlying
# view. Most columns simply project the like-named view column, but the
# metric's top-N arm projects `(vacuum_count + autovacuum_count) as
# vacuum_count` (and the same for analyze), so the cross-check must sum
# the combined expression — using `sum(vacuum_count)` alone would miss
# every autovacuum and fail on any real cluster. Timestamp columns
# (last_vacuum, last_analyze) are excluded because they aggregate via
# `max()` of the tail, not `sum()`; the substring tests in
# test_mr219_monitoring_guards.py already pin that aggregation shape.
SUM_COLUMNS = {
    "pg_stat_all_indexes": {
        "idx_scan": "idx_scan",
        "idx_tup_read": "idx_tup_read",
        "idx_tup_fetch": "idx_tup_fetch",
    },
    "pg_stat_all_tables": {
        "seq_scan": "seq_scan",
        "seq_tup_read": "seq_tup_read",
        "idx_scan": "idx_scan",
        "idx_tup_fetch": "idx_tup_fetch",
        "n_tup_ins": "n_tup_ins",
        "n_tup_upd": "n_tup_upd",
        "n_tup_del": "n_tup_del",
        "n_tup_hot_upd": "n_tup_hot_upd",
        "n_live_tup": "n_live_tup",
        "n_dead_tup": "n_dead_tup",
        "vacuum_count": "vacuum_count + autovacuum_count",
        "analyze_count": "analyze_count + autoanalyze_count",
    },
    "pg_statio_all_tables": {
        c: c for c in (
            "heap_blks_read", "heap_blks_hit",
            "idx_blks_read", "idx_blks_hit",
            "toast_blks_read", "toast_blks_hit",
            "tidx_blks_read", "tidx_blks_hit",
        )
    },
    "pg_statio_all_indexes": {
        "idx_blks_read": "idx_blks_read",
        "idx_blks_hit": "idx_blks_hit",
    },
}
# Note: SUM_COLUMNS keys are limited to the !262 four metrics. The new
# !267 metrics use the same structural pattern but their source views
# require per-metric WHERE replay (the recursive CTE in `table_stats`,
# the `>0 bytes` / `relkind IN (...)` filters in `table_size_detailed`,
# the join + `n.nspname NOT IN` in `pg_class`, etc.). The cross-check
# logic in `test_pgwatch_topn_other_bucket_aggregates_tail` only runs
# for METRIC_KEYS; the new metrics get a less strict "parses + caps +
# '$other$' present" check via NEW_METRICS_FULL_CROSS_CHECK and
# NEW_METRICS_SYNTAX_ONLY below. The substring assertions in
# test_mr219_monitoring_guards.py pin the structural aggregate shape on
# every new metric independently of this file.


def _seed_relations(cur, count: int) -> None:
    """Create enough user tables + indexes to push every source view
    well past the 100-row cap, then touch them so the statio counters
    are non-zero (otherwise the statio queries' WHERE filters reject the
    rows before ranking)."""
    cur.execute("create schema if not exists mr262")
    for i in range(count):
        cur.execute(
            f"create table if not exists mr262.t_{i:04d} (id int primary key, payload text)"
        )
        # Enough rows + payload so each table materialises at least one
        # heap page and one index page — required for pg_statio_all_*
        # to report non-zero counters under its WHERE filter.
        cur.execute(
            f"insert into mr262.t_{i:04d} "
            f"select g, repeat('x', 200) from generate_series(1, 30) g "
            f"on conflict do nothing"
        )
    # Drive heap + index reads through a seq scan and an index lookup
    # on each seeded table so heap_blks_*/idx_blks_* counters move.
    # Force enable_seqscan=off for the id-lookup so the planner can't
    # short-circuit a 30-row table with a seq scan — without this, the
    # PK indexes never register idx_blks_read/idx_blks_hit > 0 and
    # pg_statio_all_indexes' WHERE filter drops every seeded index.
    cur.execute("set enable_seqscan = off")
    try:
        for i in range(count):
            cur.execute(f"select count(*) from mr262.t_{i:04d}")
            cur.execute(f"select payload from mr262.t_{i:04d} where id = 1")
    finally:
        cur.execute("reset enable_seqscan")
    # ANALYZE reads heap pages, guaranteeing heap_blks_hit > 0 even if
    # the previous SELECTs landed entirely on already-cached pages.
    # Plain `analyze` (no table list) records last_analyze on every
    # table; we rely on this for the `'$other$'` timestamp assertion.
    cur.execute("analyze")
    # VACUUM the user tables so last_vacuum is populated on the tail —
    # the `'$other$'` row aggregates last_vacuum via max() and the test
    # asserts it is non-zero (not the 1970-01-01 epoch sentinel). VACUUM
    # cannot run inside a function or transaction block, so loop in
    # Python (autocommit ensures each statement is its own transaction).
    for i in range(count):
        cur.execute(f"vacuum mr262.t_{i:04d}")
    # pg_statio counters live in backend-local memory until the backend
    # flushes them — usually every PGSTAT_STAT_INTERVAL (~500ms) or on
    # idle. Force an immediate flush on PG 15+; on older PG (13/14 in
    # Debian bullseye CI) the stats collector picks up the UDP updates
    # within ~500ms so pg_sleep(1) is the conservative wait. Without
    # this, only ~25 tables show up in pg_statio_all_tables and the
    # `'$other$'` bucket never fires for the statio metrics.
    cur.execute(
        "select count(*) from pg_proc where proname = 'pg_stat_force_next_flush'"
    )
    if cur.fetchone()[0] > 0:
        cur.execute("select pg_stat_force_next_flush()")
    cur.execute("select pg_sleep(1)")
    # Drop any backend-local stats snapshot so the next read pulls
    # fresh cumulative stats from shared memory / the stats collector.
    cur.execute("select pg_stat_clear_snapshot()")


@pytest.fixture(scope="function")
def seeded_cur(postgresql):
    conn = postgresql
    conn.autocommit = True
    cur = conn.cursor()
    _seed_relations(cur, count=110)
    yield cur
    cur.execute("drop schema if exists mr262 cascade")
    cur.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name", METRIC_KEYS)
def test_pgwatch_topn_sql_executes_and_caps_rows(seeded_cur, metric_name):
    """(a) SQL parses and runs; (b) result set <= 101 rows per DB."""
    sql = _load_sql(metric_name)
    seeded_cur.execute(sql.rstrip().rstrip(";"))
    rows = seeded_cur.fetchall()
    assert len(rows) <= 101, (
        f"{metric_name}: expected at most 101 rows (100 top-N + 1 "
        f"'$other$'), got {len(rows)}"
    )


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name", METRIC_KEYS)
def test_pgwatch_topn_other_bucket_aggregates_tail(seeded_cur, metric_name):
    """(c) When source rows > 100, exactly one `'$other$'` row exists
    and its counters equal the sum of the rows beyond the top-100."""
    seeded_cur.execute(SOURCE_ROW_COUNT_SQL[metric_name])
    source_count = seeded_cur.fetchone()[0]
    assert source_count > 100, (
        f"{metric_name}: pre-condition failed — only {source_count} source "
        "rows; seed should have pushed past 100"
    )

    sql = _load_sql(metric_name)
    seeded_cur.execute(sql.rstrip().rstrip(";"))
    colnames = [c.name for c in seeded_cur.description]
    rows = [dict(zip(colnames, r)) for r in seeded_cur.fetchall()]

    other_rows = [r for r in rows if r["tag_schemaname"] == "$other$"]
    assert len(other_rows) == 1, (
        f"{metric_name}: expected exactly one '$other$' row when source "
        f"count {source_count} > 100, got {len(other_rows)}"
    )
    top_n = [r for r in rows if r["tag_schemaname"] != "$other$"]
    assert len(top_n) == 100, (
        f"{metric_name}: expected exactly 100 top-N rows, got {len(top_n)}"
    )

    other = other_rows[0]
    for col in SUM_COLUMNS[metric_name]:
        assert other[col] >= 0, f"{metric_name}.{col}: must be non-negative"

    # The strongest cross-check: top-N counters + '$other$' counters
    # should equal the totals computed directly from the source view.
    # Compute every view-side sum in a SINGLE query so an autocommit
    # statement between samples can't drift catalog I/O counters across
    # the per-column comparison (pg_statio views read catalog tables to
    # answer the query, bumping the very counters we're measuring).
    # Apply the metric's WHERE filter — otherwise rows the metric
    # excluded (e.g. zero-counter rows in the statio views) would
    # inflate the view total and the comparison would fail.
    where_clause = ""
    if metric_name == "pg_statio_all_tables":
        where_clause = (
            " where heap_blks_read > 0 or heap_blks_hit > 0"
            " or idx_blks_read > 0 or idx_blks_hit > 0"
            " or toast_blks_read > 0 or toast_blks_hit > 0"
            " or tidx_blks_read > 0 or tidx_blks_hit > 0"
        )
    elif metric_name == "pg_statio_all_indexes":
        where_clause = " where idx_blks_read > 0 or idx_blks_hit > 0"

    col_map = SUM_COLUMNS[metric_name]
    select_list = ", ".join(
        f"coalesce(sum({expr}), 0)::int8" for expr in col_map.values()
    )
    seeded_cur.execute(
        f"select {select_list} from {metric_name}{where_clause}"
    )
    view_totals = dict(zip(col_map.keys(), seeded_cur.fetchone()))

    for col, view_total in view_totals.items():
        # Some pg_stat_* counters are NULL when the table has had no
        # activity of that kind yet; the production '$other$' arm uses
        # `coalesce(sum(col), 0)` so do the same on the top-N side here.
        union_total = sum(int(r[col] or 0) for r in rows)
        assert union_total == view_total, (
            f"{metric_name}.{col}: top-N + '$other$' total = {union_total}, "
            f"view sum = {view_total}; tail aggregation lost data"
        )

    # The `'$other$'` row for pg_stat_all_tables aggregates timestamps
    # via max() of the tail (not `0::int8`), so dashboards plotting
    # "last vacuum" don't render 1970-01-01 when the tail has been
    # vacuumed/analyzed. The seed vacuums + analyzes every user table,
    # so the tail's last_vacuum / last_analyze must be > epoch.
    if metric_name == "pg_stat_all_tables":
        assert other["last_vacuum"] > 0, (
            "pg_stat_all_tables.'$other$'.last_vacuum: expected the "
            "tail's max(last_vacuum) epoch > 0 (seed vacuums every user "
            "table), got 0 — the production arm fell back to the "
            "1970-01-01 sentinel instead of aggregating the tail"
        )
        assert other["last_analyze"] > 0, (
            "pg_stat_all_tables.'$other$'.last_analyze: expected > 0 "
            "after the seed's ANALYZE, got 0"
        )


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_pgwatch_topn_having_guard_suppresses_other_when_tail_empty(
    postgresql,
):
    """(d) HAVING count(*) > 0 must suppress the synthetic '$other$' row
    when there is no tail to aggregate.

    The four production SQLs are wired to fixed `pg_(stat|statio)_*` view
    names, so we can't shrink the source on a real cluster (the
    catalog already contributes well over 100 rows in pg_stat_all_indexes
    on any non-trivial DB). Instead we re-shape the same UNION ALL
    pattern around a tiny temporary table and assert that the HAVING
    guard cleanly suppresses the empty-tail bucket. The structural
    pinning of `having count(*) > 0` against the production YAML lives
    in test_mr219_monitoring_guards.py.
    """
    conn = postgresql
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("create temp table mr262_src (rownum int, n int)")
    cur.execute(
        "insert into mr262_src select g, g * 10 from generate_series(1, 5) g"
    )

    cur.execute(
        """
        with ranked as (select rownum, n from mr262_src)
        select '$other$'::text as tag, coalesce(sum(n), 0)::int8 as n
        from ranked
        where rownum > 100
        group by ()
        having count(*) > 0
        """
    )
    assert cur.fetchall() == [], (
        "HAVING guard should suppress '$other$' row when the tail is empty"
    )

    # And: when the tail is non-empty, the row appears exactly once.
    cur.execute("insert into mr262_src select g, g from generate_series(101, 105) g")
    cur.execute(
        """
        with ranked as (select rownum, n from mr262_src)
        select '$other$'::text as tag, coalesce(sum(n), 0)::int8 as n
        from ranked
        where rownum > 100
        group by ()
        having count(*) > 0
        """
    )
    out = cur.fetchall()
    assert len(out) == 1, (
        f"expected one '$other$' row once the tail is non-empty, got {len(out)}"
    )
    assert int(out[0][1]) == 101 + 102 + 103 + 104 + 105
    cur.close()


# ---------------------------------------------------------------------------
# !267 coverage: the remaining per-relation metrics that got the same
# top-N + `'$other$'` pattern. See module docstring for the rationale
# behind the split between FULL_CROSS_CHECK and SYNTAX_ONLY.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def seeded_cur_mr267(postgresql):
    """Like `seeded_cur` (the MR-262 fixture) but also creates the
    `postgres_ai.pg_statistic` helper view that the bloat metrics
    require. pgwatch's `--create-helpers` flow normally puts this view
    on every monitored DB on first connect; a stock test cluster doesn't
    have it, so the bloat metrics' SQL would fail to parse without it.
    """
    conn = postgresql
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(POSTGRES_AI_HELPER_VIEW_SQL)
    _seed_relations(cur, count=110)
    yield cur
    cur.execute("drop schema if exists mr262 cascade")
    cur.execute("drop view if exists postgres_ai.pg_statistic")
    cur.execute("drop schema if exists postgres_ai cascade")
    cur.close()


# Build (metric, version) tuples for parametrization. Most !267 metrics
# have a single PG-major variant; `table_stats` has two (PG11 + PG16+).
# Use `pytest.param` with explicit `id=` so each pytest line names the
# metric (and PG-version, where applicable) instead of getting an
# auto-generated index — keeps `pytest -k` filtering useful.
def _full_cross_check_params():
    params = [pytest.param(m, None, id=m) for m in NEW_METRICS_FULL_CROSS_CHECK]
    params.extend(
        pytest.param("table_stats", v, id=f"table_stats-pg{v}")
        for v in TABLE_STATS_VARIANTS
    )
    return params


def _skip_if_pg_version_below(cur, required_major: int, sql_version: int) -> None:
    """`table_stats` ships per-PG-major SQL variants. pgwatch's source
    picker selects the highest matching variant at scrape time, so the
    PG16+ SQL never runs on PG <16 in production. But the test
    parametrize fans out both variants unconditionally — gate each on
    the live cluster's PG major so a PG13 CI image (Debian bullseye
    ships PG13/14) doesn't fail with "column last_seq_scan does not
    exist" when the PG16+ SQL hits a missing column.

    Production parity: the version picker in pgwatch's metric runner
    enforces the same lower bound — this skip mirrors that gate at the
    test layer.
    """
    cur.execute("show server_version_num")
    server_version_num = int(cur.fetchone()[0])
    server_major = server_version_num // 10000
    if server_major < required_major:
        pytest.skip(
            f"table_stats SQL variant for PG {sql_version} requires server "
            f"PG >= {required_major}, but live cluster is PG {server_major}; "
            f"in production the version picker would route this cluster to "
            f"the PG{TABLE_STATS_VARIANTS[0]} variant instead."
        )


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name,pg_version", _full_cross_check_params())
def test_pgwatch_topn_new_metrics_parse_and_cap(
    seeded_cur_mr267, metric_name, pg_version
):
    """(a) + (b) for !267 metrics: SQL parses + runs, result <= 101 rows.

    Catches: typos in window function, UNION ALL column-count mismatches,
    misplaced HAVING, stray casts, column-name divergence between the
    top-100 arm and the `'$other$'` arm.
    """
    if metric_name == "table_stats" and pg_version == 16:
        _skip_if_pg_version_below(seeded_cur_mr267, required_major=16, sql_version=16)
    sql = _load_sql(metric_name, pg_version)
    seeded_cur_mr267.execute(sql.rstrip().rstrip(";"))
    rows = seeded_cur_mr267.fetchall()
    assert len(rows) <= 101, (
        f"{metric_name} (pg_version={pg_version}): expected at most 101 "
        f"rows (100 top-N + 1 '$other$'), got {len(rows)}"
    )


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name,pg_version", _full_cross_check_params())
def test_pgwatch_topn_new_metrics_other_bucket_present(
    seeded_cur_mr267, metric_name, pg_version
):
    """(c) for !267 metrics: when the seeded source > 100 user tables,
    each metric's output must contain exactly one `'$other$'` row.

    The full column-by-column cross-check (top-N sum + '$other$' sum ==
    view sum) is not performed here — that would require per-metric
    replay of the metric's WHERE clause to avoid inflating the view
    total with rows the metric excluded, which is bespoke and brittle
    enough per metric that the MR-219 substring tests pin the structural
    aggregate shape independently. The strongest assertion here is
    "the cap engaged and the synthetic row appeared as designed."
    """
    if metric_name == "table_stats" and pg_version == 16:
        _skip_if_pg_version_below(seeded_cur_mr267, required_major=16, sql_version=16)
    sql = _load_sql(metric_name, pg_version)
    seeded_cur_mr267.execute(sql.rstrip().rstrip(";"))
    colnames = [c.name for c in seeded_cur_mr267.description]
    rows = [dict(zip(colnames, r)) for r in seeded_cur_mr267.fetchall()]

    schema_col = SCHEMA_TAG_COL[metric_name]
    assert schema_col in colnames, (
        f"{metric_name} (pg_version={pg_version}): expected output column "
        f"{schema_col!r} in {colnames!r}"
    )

    other_rows = [r for r in rows if r[schema_col] == "$other$"]
    assert len(other_rows) == 1, (
        f"{metric_name} (pg_version={pg_version}): expected exactly one "
        f"'$other$' row (seed creates 110 user tables, all metrics here "
        f"include user tables in their source set); got {len(other_rows)}"
    )

    top_n = [r for r in rows if r[schema_col] != "$other$"]
    # `table_stats` includes pg_catalog/pg_toast tables in the rankable
    # set (the cap is the cardinality control, not a schema filter), so
    # the top-100 can include system tables alongside the 110 seeded
    # user tables — the count is still exactly 100.
    assert len(top_n) == 100, (
        f"{metric_name} (pg_version={pg_version}): expected exactly 100 "
        f"top-N rows, got {len(top_n)}"
    )


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name", NEW_METRICS_SYNTAX_ONLY)
def test_pgwatch_topn_new_metrics_syntax_only(seeded_cur_mr267, metric_name):
    """(a) for the bloat / index-list metrics: SQL parses and runs.

    These metrics can't easily be pushed past their `'$other$'`
    threshold in a fast CI test:
      - pg_table_bloat / pg_btree_bloat: filter rows with `bs * tblpages
        > 1 MiB`. Forcing 110 relations >1 MiB each would mean ~110 MB
        of seed data — too expensive for the reporter:tests CI runtime.
      - unused_indexes / rarely_used_indexes: filter on
        idx_scan = 0 / specific scans_per_write ratios that are awkward
        to construct in bulk.
      - redundant_indexes: requires actual pairs of overlapping btree
        indexes, ~220 indexes to push past 100 redundant pairs.
      - pg_invalid_indexes: requires invalid indexes (usually from
        failed CREATE INDEX CONCURRENTLY, hard to construct artificially).

    The MR-219 substring tests pin the structural shape of `$other$`,
    `having count(*) > 0`, the `row_number() over` clause, and the cap
    on every one of these metrics. The value here is: run the actual
    SQL, catch any cast / column-name / UNION-arity error at test time
    rather than scrape time.
    """
    sql = _load_sql(metric_name)
    seeded_cur_mr267.execute(sql.rstrip().rstrip(";"))
    colnames = [c.name for c in seeded_cur_mr267.description]
    rows = seeded_cur_mr267.fetchall()
    # Result must still be bounded by the cap even when the source set
    # is small — `top_n <= 100` strictly holds, and at most one '$other$'.
    assert len(rows) <= 101, (
        f"{metric_name}: result exceeded 101-row cap on a seeded cluster "
        f"(got {len(rows)})"
    )
    # Column-name check: a UNION ALL arm alias swap (e.g. `tag_schema_name`
    # in the top-N arm vs `tag_schema` in the '$other$' arm) would return
    # a shape-equivalent tuple and silently pass an only-row-count check.
    # Assert both the schema tag column AND one numeric column are present.
    schema_col = SCHEMA_TAG_COL[metric_name]
    assert schema_col in colnames, (
        f"{metric_name}: expected schema tag column {schema_col!r} in output "
        f"columns {colnames!r} — a UNION ALL arm alias swap would land here"
    )
    numeric_col = EXPECTED_NUMERIC_COL[metric_name]
    assert numeric_col in colnames, (
        f"{metric_name}: expected numeric column {numeric_col!r} in output "
        f"columns {colnames!r}"
    )


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name", NEW_METRICS_SYNTAX_ONLY)
def test_pgwatch_topn_syntax_only_no_other_when_below_cap(
    seeded_cur_mr267, metric_name
):
    """(d) for the bloat / index-list metrics: HAVING count(*) > 0 must
    suppress the synthetic `'$other$'` row when the source set has ≤100
    qualifying rows after the metric's WHERE filters.

    The MR-262 metrics have an independent guard test that uses a
    temporary table to construct the empty-tail scenario; the SYNTAX_ONLY
    metrics here exercise the same guard against the production SQL
    directly. On the seeded cluster (110 small user tables, all PKs, no
    invalid/redundant/unused/rarely-used indexes that match the filters)
    every one of these metrics returns well under 100 qualifying rows,
    so the `'$other$'` row should be absent. A regression that drops the
    `HAVING count(*) > 0` guard (e.g. a future refactor) would emit a
    synthetic aggregate row even when there is no tail to aggregate, and
    this test would catch it.
    """
    sql = _load_sql(metric_name)
    seeded_cur_mr267.execute(sql.rstrip().rstrip(";"))
    colnames = [c.name for c in seeded_cur_mr267.description]
    schema_col = SCHEMA_TAG_COL[metric_name]
    # `redundant_indexes` deliberately exposes `tag_schema_name` twice
    # (raw + formated_*); dict(zip(...)) would only keep the last column.
    # Walk the tuples positionally and look at every position whose name
    # matches `schema_col` — any one of them being '$other$' means the
    # aggregate row leaked.
    schema_positions = [i for i, n in enumerate(colnames) if n == schema_col]
    assert schema_positions, (
        f"{metric_name}: expected schema tag column {schema_col!r} in output "
        f"columns {colnames!r}"
    )
    rows = seeded_cur_mr267.fetchall()
    other_rows = [
        r for r in rows
        if any(r[pos] == "$other$" for pos in schema_positions)
    ]
    assert other_rows == [], (
        f"{metric_name}: HAVING count(*) > 0 should suppress the '$other$' "
        f"row when the source set has ≤100 qualifying rows after the "
        f"metric's WHERE filters; got {len(other_rows)} '$other$' row(s) "
        f"out of {len(rows)} total"
    )


# Per-metric SQL to compute the per-column source-view totals that the
# top-N + `'$other$'` UNION ALL must reproduce. Each entry returns one
# row whose columns correspond (by position) to TAIL_SUM_COLUMNS_NEW_267
# for that metric. The WHERE clause MUST match what the metric's CTE
# applies before ranking — otherwise rows the metric excluded would
# inflate the view total and the comparison would fail.
#
# For pg_total_relation_size: filters relkind = 'r' only.
# For pg_class: filters non-system schemas + relkind in ('r','i','m','v').
# For table_size_detailed: filters relkind in ('r','p','m') +
#   non-system schemas + no AccessExclusiveLock + total_relation_size_b > 0.
# For table_stats: filters partition-leaves + non-temp + non-timescaledb
#   schemas + no AccessExclusiveLock. On a test cluster with no partition
#   roots the rows_pre_rank set is exactly q_tstats minus partition roots,
#   which on this seed is all of q_tstats.
SOURCE_VIEW_SUM_SQL_267 = {
    "pg_total_relation_size": """
        select coalesce(sum(pg_total_relation_size(c.oid)), 0)::int8 as bytes
        from pg_class c
        where c.relkind = 'r'
    """,
    "pg_class": """
        select
          coalesce(sum(c.reltuples), 0)::float4 as reltuples,
          coalesce(sum(c.relpages), 0)::int4 as relpages,
          coalesce(sum(pg_relation_size(c.oid)), 0)::int8 as relation_size_bytes,
          coalesce(sum(pg_total_relation_size(c.oid)), 0)::int8 as total_relation_size_bytes
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname not in ('information_schema', 'pg_catalog')
          and c.relkind in ('r', 'i', 'm', 'v')
    """,
    "table_size_detailed": """
        with src as (
          select
            pg_relation_size(c.oid, 'main') as table_main_size_b,
            pg_relation_size(c.oid, 'fsm') as table_fsm_size_b,
            pg_relation_size(c.oid, 'vm') as table_vm_size_b,
            pg_indexes_size(c.oid) as table_indexes_size_b,
            pg_relation_size(c.reltoastrelid, 'main') as toast_main_size_b,
            pg_relation_size(c.reltoastrelid, 'fsm') as toast_fsm_size_b,
            pg_relation_size(c.reltoastrelid, 'vm') as toast_vm_size_b,
            pg_indexes_size(c.reltoastrelid) as toast_indexes_size_b,
            pg_total_relation_size(c.oid) as total_relation_size_b
          from pg_class c
          join pg_namespace n on n.oid = c.relnamespace
          where c.relkind in ('r', 'p', 'm')
            and n.nspname not in ('information_schema', 'pg_toast')
            and not exists (
              select 1 from pg_locks
              where relation = c.oid and mode = 'AccessExclusiveLock'
            )
        )
        select
          coalesce(sum(table_main_size_b), 0)::int8 as table_main_size_b,
          coalesce(sum(table_fsm_size_b), 0)::int8 as table_fsm_size_b,
          coalesce(sum(table_vm_size_b), 0)::int8 as table_vm_size_b,
          coalesce(sum(table_indexes_size_b), 0)::int8 as table_indexes_size_b,
          coalesce(sum(toast_main_size_b), 0)::int8 as toast_main_size_b,
          coalesce(sum(toast_fsm_size_b), 0)::int8 as toast_fsm_size_b,
          coalesce(sum(toast_vm_size_b), 0)::int8 as toast_vm_size_b,
          coalesce(sum(toast_indexes_size_b), 0)::int8 as toast_indexes_size_b,
          coalesce(sum(total_relation_size_b), 0)::int8 as total_relation_size_b
        from src
        where total_relation_size_b > 0
    """,
    # table_stats's source CTE has a partition-roots UNION arm; the test
    # cluster has no partition roots (the seeded tables are plain
    # CREATE TABLE, not PARTITION OF), so rows_pre_rank == q_tstats minus
    # the empty q_root_part set. Replay just the q_tstats path, summing
    # only the columns whose aggregate semantics on `'$other$'` is sum()
    # (size + counter columns). Timestamp `seconds_since_last_*` columns
    # use min() on the tail; freeze-age columns use max(); cross-checking
    # those would require excluding the top-100 from the source — skip
    # them here, the substring tests in test_mr219_monitoring_guards.py
    # pin those aggregation shapes structurally.
    "table_stats": """
        select
          coalesce(sum(pg_table_size(ut.relid)), 0)::int8 as table_size_b,
          coalesce(sum(pg_total_relation_size(ut.relid)), 0)::int8 as total_relation_size_b,
          coalesce(sum(case when c.reltoastrelid <> 0
                  then pg_total_relation_size(c.reltoastrelid)
                  else 0::int8 end), 0)::int8 as toast_size_b,
          coalesce(sum(case when 'autovacuum_enabled=off' = ANY (c.reloptions)
                  then 1 else 0 end), 0)::int8 as no_autovacuum,
          coalesce(sum(ut.seq_scan), 0)::int8 as seq_scan,
          coalesce(sum(ut.seq_tup_read), 0)::int8 as seq_tup_read,
          coalesce(sum(coalesce(ut.idx_scan, 0)), 0)::int8 as idx_scan,
          coalesce(sum(coalesce(ut.idx_tup_fetch, 0)), 0)::int8 as idx_tup_fetch,
          coalesce(sum(ut.n_tup_ins), 0)::int8 as n_tup_ins,
          coalesce(sum(ut.n_tup_upd), 0)::int8 as n_tup_upd,
          coalesce(sum(ut.n_tup_del), 0)::int8 as n_tup_del,
          coalesce(sum(ut.n_tup_hot_upd), 0)::int8 as n_tup_hot_upd,
          coalesce(sum(ut.n_live_tup), 0)::int8 as n_live_tup,
          coalesce(sum(ut.n_dead_tup), 0)::int8 as n_dead_tup,
          coalesce(sum(ut.vacuum_count), 0)::int8 as vacuum_count,
          coalesce(sum(ut.autovacuum_count), 0)::int8 as autovacuum_count,
          coalesce(sum(ut.analyze_count), 0)::int8 as analyze_count,
          coalesce(sum(ut.autoanalyze_count), 0)::int8 as autoanalyze_count
        from pg_stat_all_tables ut
        join pg_class c on c.oid = ut.relid
        where not exists (
                select 1 from pg_locks
                where relation = ut.relid and mode = 'AccessExclusiveLock'
              )
          and c.relpersistence <> 't'
          and not quote_ident(ut.schemaname) like E'\\_timescaledb%'
    """,
}

# Per-metric: output columns to cross-check (the production SQL outputs
# both top-N and `'$other$'` arms in the same column order, so summing
# `union_total = sum(top-N[col]) + other[col]` and comparing to the
# source-view total cleanly validates the tail aggregation). Order MUST
# match the SELECT list of SOURCE_VIEW_SUM_SQL_267[metric] above so the
# zip() below pairs the right columns.
TAIL_SUM_COLUMNS_NEW_267 = {
    "pg_total_relation_size": ("bytes",),
    "pg_class": (
        "reltuples",
        "relpages",
        "relation_size_bytes",
        "total_relation_size_bytes",
    ),
    "table_size_detailed": (
        "table_main_size_b",
        "table_fsm_size_b",
        "table_vm_size_b",
        "table_indexes_size_b",
        "toast_main_size_b",
        "toast_fsm_size_b",
        "toast_vm_size_b",
        "toast_indexes_size_b",
        "total_relation_size_b",
    ),
    "table_stats": (
        "table_size_b",
        "total_relation_size_b",
        "toast_size_b",
        "no_autovacuum",
        "seq_scan",
        "seq_tup_read",
        "idx_scan",
        "idx_tup_fetch",
        "n_tup_ins",
        "n_tup_upd",
        "n_tup_del",
        "n_tup_hot_upd",
        "n_live_tup",
        "n_dead_tup",
        "vacuum_count",
        "autovacuum_count",
        "analyze_count",
        "autoanalyze_count",
    ),
}


@pytest.mark.integration
@pytest.mark.requires_postgres
@pytest.mark.parametrize("metric_name,pg_version", _full_cross_check_params())
def test_pgwatch_topn_new_metrics_other_bucket_aggregates_tail(
    seeded_cur_mr267, metric_name, pg_version
):
    """(c) column-by-column tail-sum cross-check for !267 FULL_CROSS_CHECK
    metrics: for each metric whose `'$other$'` row uses `sum()` semantics
    on its numeric columns, the top-100 rows' counter sum plus the
    `'$other$'` row's counter must equal the source-view sum (computed
    directly with the metric's WHERE filters re-applied).

    A regression that lost data in the tail aggregation (wrong window
    function, missing UNION arm column, dropped `coalesce`) would show
    up as a per-column mismatch. The MR-262 test
    `test_pgwatch_topn_other_bucket_aggregates_tail` validates the same
    invariant for the 4 !262 metrics; this is the !267 equivalent.

    Timestamp / min / max / avg columns are not cross-checked here — they
    don't decompose into top-N + tail sums, and the substring tests in
    test_mr219_monitoring_guards.py pin those aggregation shapes
    structurally.
    """
    if metric_name == "table_stats" and pg_version == 16:
        _skip_if_pg_version_below(seeded_cur_mr267, required_major=16, sql_version=16)
    sql = _load_sql(metric_name, pg_version)
    seeded_cur_mr267.execute(sql.rstrip().rstrip(";"))
    colnames = [c.name for c in seeded_cur_mr267.description]
    rows = [dict(zip(colnames, r)) for r in seeded_cur_mr267.fetchall()]

    schema_col = SCHEMA_TAG_COL[metric_name]
    other_rows = [r for r in rows if r[schema_col] == "$other$"]
    assert len(other_rows) == 1, (
        f"{metric_name} (pg_version={pg_version}): cross-check requires "
        f"the `'$other$'` row to be present (seed is 110 user tables); "
        f"got {len(other_rows)}"
    )
    other = other_rows[0]

    seeded_cur_mr267.execute(SOURCE_VIEW_SUM_SQL_267[metric_name])
    view_totals = dict(
        zip(TAIL_SUM_COLUMNS_NEW_267[metric_name], seeded_cur_mr267.fetchone())
    )

    for col, view_total in view_totals.items():
        # The production '$other$' arm uses coalesce(sum(col), 0); mirror
        # that on the union side here so a NULL top-N value (e.g. an
        # idx_scan that was NULL before pgwatch saw any activity) does
        # not poison the union total. Use int() comparison to avoid
        # float-vs-int type mismatches on the reltuples column.
        union_total = sum(int(r[col] or 0) for r in rows)
        assert union_total == int(view_total), (
            f"{metric_name}.{col} (pg_version={pg_version}): top-N + "
            f"'$other$' total = {union_total}, view sum = {int(view_total)}; "
            f"tail aggregation lost data"
        )
