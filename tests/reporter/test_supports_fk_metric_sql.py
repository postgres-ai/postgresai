"""Live-PostgreSQL coverage for the `supports_fk` value of four pgwatch metrics (#476).

An index supports a foreign key when the key is on the index's own table and the
key's columns are the index's leading columns. The `fk_indexes` CTE recorded the
table the key points to (`confrelid`) and matched it by name and by opclass prefix,
so an index backing a key read false unless its table was also referenced, an index
on the referenced table read true, and mixed-case or other-schema tables never
matched.

Metrics (config/pgwatch-prometheus/metrics.yml): `unused_indexes`,
`rarely_used_indexes`, `redundant_indexes`, `pg_invalid_indexes`. Marked
`integration` + `requires_postgres`; the CI reporter:tests job runs it under
`--run-integration`.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS_METRICS_YML = PROJECT_ROOT / "config/pgwatch-prometheus/metrics.yml"

ROWS = 20000

FIXTURE_SQL = f"""
create table parent (id int primary key, v int, w int);
insert into parent select g, g % 97, g % 89 from generate_series(1, {ROWS}) g;
create table pair (a int, b int, primary key (a, b));
insert into pair select g, g from generate_series(1, {ROWS}) g;

create table child (id int, parent_id int references parent (id), x int);
insert into child select g, g, g % 83 from generate_series(1, {ROWS}) g;
create table "Child2" ("Pid" int references parent (id), y int);
insert into "Child2" select g, g from generate_series(1, {ROWS}) g;
create schema s2;
create table s2.child3 (pid int references public.parent (id), z int);
insert into s2.child3 select g, g from generate_series(1, {ROWS}) g;
create table node (id int primary key, parent_id int references node (id), v int);
insert into node select g, nullif(g - 1, 0), g % 79 from generate_series(1, {ROWS}) g;
create table multi (a int, b int, c int, foreign key (a, b) references pair (a, b));
insert into multi select g, g, g from generate_series(1, {ROWS}) g;

-- Back a foreign key: the key on the index's own table, its columns leading.
create index child_pid on child (parent_id);
create index child_pid_x on child (parent_id, x);
create index "Child2_pid" on "Child2" ("Pid");
create index child3_pid on s2.child3 (pid);
create index node_pid on node (parent_id);
create index multi_ab on multi (a, b);
create index multi_ba on multi (b, a);
-- Do not: the key column not leading, the referenced table, a partial key.
create index child_x_pid on child (x, parent_id);
create index parent_v on parent (v);
create index parent_v_w on parent (v, w);
create index node_v on node (v);
create index multi_a on multi (a);

analyze;
"""

BACKS_FK = {"child_pid", "child_pid_x", '"Child2_pid"', "s2.child3_pid", "node_pid", "multi_ab", "multi_ba"}
NO_FK = {"child_x_pid", "parent_v", "parent_v_w", "node_v", "multi_a"}
TABLES = ["parent", "child", '"Child2"', "s2.child3", "node", "multi"]


def _sqls(metric: str) -> dict:
    metrics = yaml.safe_load(PROMETHEUS_METRICS_YML.read_text())
    return {int(v): s.rstrip().rstrip(";") for v, s in metrics["metrics"][metric]["sqls"].items()}


def _by_index(cur, metric: str, version: int, sql_text: str) -> dict:
    if metric == "pg_invalid_indexes":  # no index_id column: the schema and name tags
        cur.execute("select tag_schema_name, tag_index_name, supports_fk::int "
                    f"from ({sql_text}) m where tag_index_name <> '$other$'")
        rows = {(f"{s}.{n}" if s != "public" else (f'"{n}"' if n != n.lower() else n)): v
                for s, n, v in cur.fetchall()}
    else:  # the `$other$` row has index_id 0
        cur.execute(f"select index_id::regclass::text, supports_fk::int from ({sql_text}) m where index_id <> 0")
        rows = dict(cur.fetchall())
    assert rows, f"{metric} PG{version}+: no fixture index in the output"
    return rows


def _check(cur, metric: str, expected: set) -> None:
    for version, sql_text in _sqls(metric).items():
        rows = _by_index(cur, metric, version, sql_text)
        wrong = {name: value for name, value in rows.items()
                 if (name in BACKS_FK and value != 1) or (name in NO_FK and value != 0)}
        assert not wrong, f"{metric} PG{version}+: supports_fk wrong for {wrong}"
        assert expected <= set(rows), f"{metric} PG{version}+: missing {expected - set(rows)}"


@pytest.fixture(scope="function")
def fk_cur():
    """A throwaway database on the system PostgreSQL, holding the fixture."""
    import psycopg2
    from psycopg2 import sql

    connection_kwargs = {}
    for env, key in (("PGHOST", "host"), ("PGPORT", "port"), ("PGUSER", "user")):
        if os.environ.get(env):
            connection_kwargs[key] = os.environ[env]
    bootstrap_db = os.environ.get("PGDATABASE", "postgres")
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    test_db = f"test_fk_{uuid.uuid4().hex[:12]}"
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


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_unused_indexes_supports_fk(fk_cur):
    _check(fk_cur, "unused_indexes", BACKS_FK | NO_FK)


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_redundant_indexes_supports_fk(fk_cur):
    # Candidates: child_pid (in child_pid_x), parent_v (in parent_v_w), multi_a (in multi_ab).
    _check(fk_cur, "redundant_indexes", {"child_pid", "parent_v", "multi_a"})


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_rarely_used_indexes_supports_fk(fk_cur):
    # Rarely used: some index scans, under 10% of the table's scans, at most one per write.
    fk_cur.execute("set enable_seqscan = off; set enable_bitmapscan = off")
    for name in BACKS_FK | NO_FK:
        fk_cur.execute(f"select pg_catalog.pg_get_indexdef('{name}'::regclass)")
        table = {"child_pid": "child", "child_pid_x": "child", '"Child2_pid"': '"Child2"',
                 "s2.child3_pid": "s2.child3", "node_pid": "node", "multi_ab": "multi", "multi_ba": "multi",
                 "child_x_pid": "child", "parent_v": "parent", "parent_v_w": "parent", "node_v": "node",
                 "multi_a": "multi"}[name]
        column = fk_cur.fetchone()[0].split("(", 1)[1].split(")")[0].split(",")[0].strip()
        fk_cur.execute(f"select count(*) from {table} where {column} = 1")
    fk_cur.execute("reset enable_seqscan; reset enable_bitmapscan")
    for table in TABLES:
        for _ in range(40):
            fk_cur.execute(f"select count(*) from {table}")
    fk_cur.execute("select pg_stat_force_next_flush()")
    _check(fk_cur, "rarely_used_indexes", set())


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_pg_invalid_indexes_supports_fk(fk_cur):
    names = sorted(BACKS_FK | NO_FK)
    fk_cur.execute("update pg_index set indisvalid = false where indexrelid = any(%s::regclass[])", (names,))
    _check(fk_cur, "pg_invalid_indexes", BACKS_FK | NO_FK)
