"""Integration coverage for MR !262: execute the four rewritten pgwatch
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
      the sum of the excluded tail.

The HAVING-count(*)>0 guard (no `'$other$'` row when the tail is empty)
is exercised separately via a temporary table that exposes <=100 rows
of the same shape to the top-N + UNION ALL block.

Marked `integration` + `requires_postgres` so the default
`--disable-socket` unit-test run skips the file; the CI reporter:tests
job (already provisions a Postgres cluster) picks it up under
`--run-integration`.
"""
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"


def _load_sql(metric_name: str) -> str:
    metrics = yaml.safe_load(METRICS_YML.read_text())
    sqls = metrics["metrics"][metric_name]["sqls"]
    # The four metrics ship a single PG11+ SQL today. Fail loudly if a
    # version-specific variant is added in the future without updating
    # this test to exercise every variant — otherwise only the first one
    # would get integration coverage.
    assert len(sqls) == 1, (
        f"{metric_name}: expected exactly one SQL version, got "
        f"{list(sqls)}; update this test to iterate every variant"
    )
    return next(iter(sqls.values()))


METRIC_KEYS = (
    "pg_stat_all_indexes",
    "pg_stat_all_tables",
    "pg_statio_all_tables",
    "pg_statio_all_indexes",
)

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
