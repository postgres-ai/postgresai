"""table_stats.no_autovacuum must see every spelling of autovacuum_enabled = false.

Postgres stores a reloption as the text it was given: `alter table t set
(autovacuum_enabled = false)` leaves `autovacuum_enabled=false` in
pg_class.reloptions, not `=off`. The F001 and F004 reports read this flag to say
why autovacuum never ran on a table; matching only `=off` made it 0 on such a
table (`pg_dead_tuples.autovacuum_disabled` already matches every spelling).
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"
SPELLINGS = ("false", "off", "0", "no", "f")


def _variants() -> dict:
    sqls = yaml.safe_load(METRICS_YML.read_text())["metrics"]["table_stats"]["sqls"]
    return {int(v): s.rstrip().rstrip(";") for v, s in sqls.items()}


@pytest.mark.unit
@pytest.mark.parametrize("version", sorted(_variants()))
def test_no_autovacuum_does_not_match_only_the_off_spelling(version: int) -> None:
    sql = _variants()[version]
    assert "'autovacuum_enabled=off' = ANY" not in sql
    assert "autovacuum_enabled=(false|fals|fal|fa|f|no|n|off|of|0)$" in sql


@pytest.fixture(scope="function")
def stats_cur():
    """A throwaway database on CI's PostgreSQL (skipped locally without one)."""
    import psycopg2
    from psycopg2 import sql

    kwargs = {k: os.environ[e] for k, e in (("host", "PGHOST"), ("user", "PGUSER")) if os.environ.get(e)}
    if os.environ.get("PGPORT"):
        kwargs["port"] = int(os.environ["PGPORT"])
    bootstrap_db = os.environ.get("PGDATABASE", "postgres")
    test_db = sql.Identifier(f"test_tstats_{uuid.uuid4().hex[:12]}")
    try:
        boot = psycopg2.connect(dbname=bootstrap_db, **kwargs)
        boot.autocommit = True
        with boot.cursor() as cur:
            cur.execute(sql.SQL("create database {}").format(test_db))
        boot.close()
        conn = psycopg2.connect(dbname=test_db.strings[0], **kwargs)
        conn.autocommit = True
    except psycopg2.OperationalError as exc:
        if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
            raise RuntimeError(f"PostgreSQL connection failed in CI: {exc}") from exc
        pytest.skip(f"PostgreSQL not available: {exc}")
    cur = conn.cursor()
    try:  # setup inside: a failed statement still drops the throwaway database
        cur.execute("create table on_by_default (id int)")
        for n, spelling in enumerate(SPELLINGS):
            cur.execute(f"create table off_{n} (id int) with (autovacuum_enabled = {spelling})")
        cur.execute("create table on_explicit (id int) with (autovacuum_enabled = true)")
        for table in ["on_by_default", "on_explicit"] + [f"off_{n}" for n in range(len(SPELLINGS))]:
            cur.execute(f"insert into {table} select generate_series(1, 1000)")  # the metric skips empty files
        yield cur
    finally:
        conn.close()
        boot = psycopg2.connect(dbname=bootstrap_db, **kwargs)
        boot.autocommit = True
        with boot.cursor() as c:
            c.execute(sql.SQL("drop database if exists {}").format(test_db))
        boot.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_no_autovacuum_is_1_for_every_off_spelling(stats_cur) -> None:
    stats_cur.execute("show server_version_num")
    major = int(stats_cur.fetchone()[0]) // 10000
    sql = _variants()[max(v for v in _variants() if v <= major)]
    stats_cur.execute(sql)
    cols = [c.name for c in stats_cur.description]
    flags = {row[cols.index("tag_table_name")]: row[cols.index("no_autovacuum")] for row in stats_cur.fetchall()}
    assert {f"off_{n}": flags.get(f"off_{n}") for n in range(len(SPELLINGS))} == {
        f"off_{n}": 1 for n in range(len(SPELLINGS))}
    assert flags.get("on_by_default") == 0
    assert flags.get("on_explicit") == 0
