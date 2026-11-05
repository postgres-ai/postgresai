"""Live-PostgreSQL coverage for the pairs the redundant-index SQL finds (#482).

The SQL pairs a candidate index with a covering index when the candidate's key
columns and opclasses are a leading prefix of the other's. It compared
`indkey::text` with LIKE, so column 1 matched column 12 (`'1'` is a text prefix
of `'12 3'`), and two indexes on different columns of any table with 10 or more
columns were reported as redundant.

The same SQL lives in two files:
  - config/pgwatch-prometheus/metrics.yml, metric `redundant_indexes`;
  - config/pgwatch-postgres/metrics.yml, metric `index_definitions`
    (its `redundant_index_pairs` CTE).

Each fixture index is built on enough rows to pass the SQL's `relpages > 5`
filter. Marked `integration` + `requires_postgres`; the CI reporter:tests job
runs it under `--run-integration`.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS_METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"
POSTGRES_METRICS_YML = PROJECT_ROOT / "config/pgwatch-postgres/metrics.yml"

ROWS = 20000

FIXTURE_SQL = f"""
create table wide (
  c1 int, c2 int, c3 int, c4 int, c5 int, c6 int,
  c7 int, c8 int, c9 int, c10 int, c11 int, c12 int
);
insert into wide
  select g, g % 7, g % 11, g, g, g, g, g, g, g, g % 13, g % 17
  from generate_series(1, {ROWS}) g;
-- Different columns: (c1) vs (c12, c3) and (c2, c1) vs (c2, c11).
create index wide_c1 on wide (c1);
create index wide_c12_c3 on wide (c12, c3);
create index wide_c2_c1 on wide (c2, c1);
create index wide_c2_c11 on wide (c2, c11);

create table prefix (a int, b int, c int);
insert into prefix select g, g, g % 100 from generate_series(1, {ROWS}) g;
-- True redundancy: (a) is a leading prefix of (a, b).
create index prefix_a on prefix (a);
create index prefix_a_b on prefix (a, b);
-- Exact duplicates.
create index dup_c_1 on prefix (c);
create index dup_c_2 on prefix (c);
-- A unique index is never the candidate.
create unique index prefix_b_uniq on prefix (b);
create index prefix_b_a on prefix (b, a);

create table opc (s text, x int);
insert into opc select md5(g::text), g from generate_series(1, {ROWS}) g;
-- Same column, different opclass: not paired.
create index opc_s on opc (s);
create index opc_s_pattern_x on opc (s text_pattern_ops, x);
-- Same opclass: paired.
create index opc_s_x on opc (s, x);

create table expr (s text, x int);
insert into expr select md5(g::text), g from generate_series(1, {ROWS}) g;
-- Expression columns (attnum 0): same expression is paired, another is not.
create index expr_lower on expr (lower(s));
create index expr_lower_x on expr (lower(s), x);
create index expr_upper_x on expr (upper(s), x);

create table part (a int, b int);
insert into part select g, g from generate_series(1, {ROWS}) g;
-- Partial indexes: paired only with the same predicate.
create index part_a on part (a) where b > 0;
create index part_a_b on part (a, b) where b > 0;
create index full_a_b on part (a, b);

analyze;
"""

# (candidate, covering index) pairs both SQLs must find.
EXPECTED_PAIRS = {
    ("prefix_a", "prefix_a_b"),
    ("opc_s", "opc_s_x"),
    ("expr_lower", "expr_lower_x"),
    ("part_a", "part_a_b"),
}
DUPLICATES = {("dup_c_1", "dup_c_2"), ("dup_c_2", "dup_c_1")}

# What the text-prefix comparison wrongly reported.
FALSE_PAIRS = {
    ("wide_c1", "wide_c12_c3"),
    ("wide_c2_c1", "wide_c2_c11"),
}


def _load_sqls(path: Path, metric_name: str) -> dict:
    metrics = yaml.safe_load(path.read_text())
    sqls = metrics["metrics"][metric_name]["sqls"]
    return {int(v): s.rstrip().rstrip(";") for v, s in sqls.items()}


@pytest.fixture(scope="function")
def h004_cur():
    """A throwaway database on the system PostgreSQL, holding the fixture."""
    import psycopg2
    from psycopg2 import sql

    connection_kwargs = {}
    for env, key in (("PGHOST", "host"), ("PGPORT", "port"), ("PGUSER", "user")):
        if os.environ.get(env):
            connection_kwargs[key] = os.environ[env]
    bootstrap_db = os.environ.get("PGDATABASE", "postgres")
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    test_db = f"test_h004_{uuid.uuid4().hex[:12]}"
    test_db_ident = sql.Identifier(test_db)

    try:
        bootstrap = psycopg2.connect(dbname=bootstrap_db, **connection_kwargs)
        bootstrap.autocommit = True
        with bootstrap.cursor() as cur:
            cur.execute(sql.SQL("create database {}").format(test_db_ident))
        bootstrap.close()
        conn = psycopg2.connect(dbname=test_db, **connection_kwargs)
        conn.autocommit = True
    except psycopg2.OperationalError as exc:
        if is_ci:
            raise RuntimeError(f"PostgreSQL connection failed in CI: {exc}") from exc
        pytest.skip(f"PostgreSQL not available: {exc}")

    cur = conn.cursor()
    try:
        cur.execute(FIXTURE_SQL)
        yield cur
    finally:
        cur.close()
        conn.close()
        cleanup = psycopg2.connect(dbname=bootstrap_db, **connection_kwargs)
        cleanup.autocommit = True
        with cleanup.cursor() as c:
            c.execute(sql.SQL("drop database if exists {}").format(test_db_ident))
        cleanup.close()


def _prometheus_pairs(cur, sql_text: str) -> set:
    # The metric returns duplicate column names (tag_schema_name, tag_index_name),
    # so read only unique ones through a subquery. The `$other$` row has index_id 0.
    cur.execute(
        "select index_id::regclass::text, tag_reason "
        f"from ({sql_text}) m where index_id <> 0"
    )
    return {
        (candidate, reason)
        for candidate, reasons in cur.fetchall()
        for reason in reasons.split(", ")
    }


def _postgres_sink_pairs(cur, sql_text: str) -> set:
    # index_definitions mixes unused and redundant indexes in its output; every
    # fixture index is unused, so read the pairs from its own CTE instead.
    marker = "select /* pgwatch_generated */"
    assert sql_text.count(marker) == 1
    head = sql_text[: sql_text.index(marker)]
    cur.execute(
        head
        + "select redundant_index_id::regclass::text, reason_index_id::regclass::text "
        "from redundant_index_pairs where schema_name = 'public'"
    )
    return set(cur.fetchall())


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_redundant_indexes_metric_pairs(h004_cur):
    for version, sql_text in _load_sqls(PROMETHEUS_METRICS_YML, "redundant_indexes").items():
        pairs = _prometheus_pairs(h004_cur, sql_text)
        assert not pairs & FALSE_PAIRS, f"PG{version}+ SQL pairs different columns: {pairs & FALSE_PAIRS}"
        # Mutual duplicates are reported once, in either direction.
        dups = pairs & DUPLICATES
        assert len(dups) == 1, f"PG{version}+ SQL: duplicates reported as {dups}"
        assert pairs - dups == EXPECTED_PAIRS, f"PG{version}+ SQL pairs: {sorted(pairs)}"


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_index_definitions_redundant_pairs(h004_cur):
    for version, sql_text in _load_sqls(POSTGRES_METRICS_YML, "index_definitions").items():
        pairs = _postgres_sink_pairs(h004_cur, sql_text)
        assert not pairs & FALSE_PAIRS, f"PG{version}+ SQL pairs different columns: {pairs & FALSE_PAIRS}"
        # This CTE still holds both directions of a mutual duplicate.
        assert pairs == EXPECTED_PAIRS | DUPLICATES, f"PG{version}+ SQL pairs: {sorted(pairs)}"
