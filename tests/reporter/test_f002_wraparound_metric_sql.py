"""Live-PostgreSQL execution coverage for the F002 wraparound metric SQL.

The static assertions in tests/xmin_horizon/test_metrics_sql_static.py pin the
shape of these SQLs against the YAML, but they cannot catch a cast that fails at
runtime, a column that only exists on newer PG, or an `mxid_age`/`age` mixup that
parses fine but errors on execution. Failure-mode FM-2 requires diagnostic SQL
to actually execute across versions, so this file runs each new F002 metric
against a real cluster and asserts it parses, runs, and returns the columns the
reporter consumes.

Marked `integration` + `requires_postgres` so the default `--disable-socket`
unit run skips it; the CI reporter:tests job (which provisions Postgres) runs it
under `--run-integration`.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"


def _load_sql(metric_name: str) -> str:
    metrics = yaml.safe_load(METRICS_YML.read_text())
    sqls = metrics["metrics"][metric_name]["sqls"]
    # Every F002 metric ships a single (PG11+) SQL variant today; assert that so
    # a future per-version split forces this test to iterate every variant.
    assert len(sqls) == 1, f"{metric_name}: expected one SQL variant, got {list(sqls)}"
    return next(iter(sqls.values())).rstrip().rstrip(";")


@pytest.fixture(scope="function")
def f002_postgresql():
    """Connect to CI's system PostgreSQL over its peer-auth Unix socket."""
    import psycopg2
    from psycopg2 import sql

    connection_kwargs = {}
    if os.environ.get("PGHOST"):
        connection_kwargs["host"] = os.environ["PGHOST"]
    if os.environ.get("PGPORT"):
        connection_kwargs["port"] = int(os.environ["PGPORT"])
    if os.environ.get("PGUSER"):
        connection_kwargs["user"] = os.environ["PGUSER"]
    bootstrap_db = os.environ.get("PGDATABASE", "postgres")
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    test_db = f"test_f002_{uuid.uuid4().hex[:12]}"
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

    try:
        yield conn
    finally:
        conn.close()
        cleanup = psycopg2.connect(dbname=bootstrap_db, **connection_kwargs)
        cleanup.autocommit = True
        with cleanup.cursor() as cur:
            cur.execute(sql.SQL("drop database if exists {}").format(test_db_ident))
        cleanup.close()


@pytest.fixture(scope="function")
def wraparound_cur(f002_postgresql):
    conn = f002_postgresql
    conn.autocommit = True
    cur = conn.cursor()
    # A plain heap plus a table with a lowered per-table freeze override so the
    # override projection exercises the reloptions path, not just the global GUC.
    cur.execute("create schema if not exists f002")
    cur.execute("create table if not exists f002.plain (id int primary key, payload text)")
    cur.execute(
        "create table if not exists f002.overridden (id int primary key) "
        "with (autovacuum_freeze_max_age = 100000, "
        "autovacuum_multixact_freeze_max_age = 100000)"
    )
    cur.execute("insert into f002.plain select g, repeat('x', 100) from generate_series(1, 50) g")
    yield cur
    cur.execute("drop schema if exists f002 cascade")
    cur.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_pg_settings_wraparound_executes_and_exposes_all_columns(wraparound_cur):
    wraparound_cur.execute(_load_sql("pg_settings_wraparound"))
    rows = wraparound_cur.fetchall()
    assert len(rows) == 1
    colnames = {c.name for c in wraparound_cur.description}
    for column in (
        "autovacuum_freeze_max_age",
        "vacuum_freeze_min_age",
        "vacuum_freeze_table_age",
        "autovacuum_multixact_freeze_max_age",
        "vacuum_multixact_freeze_min_age",
        "vacuum_multixact_freeze_table_age",
        "vacuum_failsafe_age",
        "vacuum_multixact_failsafe_age",
    ):
        assert column in colnames, f"pg_settings_wraparound missing column {column!r}"


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_pg_database_wraparound_executes(wraparound_cur):
    wraparound_cur.execute(_load_sql("pg_database_wraparound"))
    colnames = [c.name for c in wraparound_cur.description]
    assert {"tag_datname", "age_datfrozenxid", "age_datminmxid"} <= set(colnames)
    rows = wraparound_cur.fetchall()
    assert len(rows) >= 1
    # Ages must be non-negative int8s (the mxid_age fix guards against the
    # ~2^31 wrap that plain age(relminmxid) produced on fresh clusters).
    xid_i = colnames.index("age_datfrozenxid")
    mxid_i = colnames.index("age_datminmxid")
    for row in rows:
        assert row[xid_i] is not None and row[xid_i] >= 0
        assert row[mxid_i] is not None and row[mxid_i] >= 0


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_pg_table_wraparound_executes_caps_and_exposes_overrides(wraparound_cur):
    wraparound_cur.execute(_load_sql("pg_table_wraparound"))
    colnames = [c.name for c in wraparound_cur.description]
    rows = [dict(zip(colnames, r)) for r in wraparound_cur.fetchall()]
    for column in (
        "tag_schema_name",
        "tag_table_name",
        "tag_ranked_by",
        "xid_age",
        "multixact_age",
        "effective_freeze_max_age",
        "effective_multixact_freeze_max_age",
        "table_size_bytes",
    ):
        assert column in colnames, f"pg_table_wraparound missing column {column!r}"
    # Bounded to 50 per age, de-duplicated across the two sets, so at most 100.
    assert len(rows) <= 100, f"expected <= 100 rows (50 per age), got {len(rows)}"
    # The per-table override must be surfaced, not silently replaced by the GUC.
    overridden = [r for r in rows if r["tag_table_name"] == "overridden"]
    if overridden:  # present unless it fell outside both top-50 sets on a busy catalog
        assert overridden[0]["effective_freeze_max_age"] == 100000
        assert overridden[0]["effective_multixact_freeze_max_age"] == 100000


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_multixact_size_executes(wraparound_cur):
    # Reads the data directory via pg_stat_file/pg_ls_dir (or the RDS/Aurora
    # probes); on a stock superuser CI cluster the local branch runs. Assert it
    # parses/executes and returns exactly the member/offset/status columns the
    # reporter sums — never a single `bytes` column.
    wraparound_cur.execute(_load_sql("multixact_size"))
    colnames = [c.name for c in wraparound_cur.description]
    assert colnames == ["members_bytes", "offsets_bytes", "status_code"]
    rows = wraparound_cur.fetchall()
    assert len(rows) == 1
