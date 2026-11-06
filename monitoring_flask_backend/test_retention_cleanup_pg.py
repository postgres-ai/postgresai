"""Query-text retention cleanup against real Postgres partitions (#495).

Mocks cannot catch this: ctid is unique only within one partition, so the bug
is visible only when the DELETE runs across a real partition hierarchy.
"""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import psycopg2
import pytest

import app as app_module

ADMIN_URL = os.environ.get('PGSS_SINK_TEST_URL')
RETENTION_HOURS = 24
ROWS_PER_PARTITION = 25
SOURCES = ('source_a', 'source_b')


@pytest.fixture
def sink():
    """A throwaway database with the sink's list-then-range partition layout."""
    if not ADMIN_URL:
        if os.environ.get('CI'):
            pytest.fail('PGSS_SINK_TEST_URL is required in CI')
        pytest.skip('PGSS_SINK_TEST_URL is not set')

    name = f'retention_{uuid.uuid4().hex[:12]}'
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute(f'create database {name}')
        url = psycopg2.extensions.parse_dsn(ADMIN_URL)
        url['dbname'] = name
        conn = psycopg2.connect(**url)
        conn.autocommit = True
        now = datetime.now(timezone.utc)
        old, recent = now - timedelta(days=30), now - timedelta(hours=1)
        with conn.cursor() as cur:
            cur.execute(
                'create table public.pgss_queryid_queries ('
                ' time timestamptz not null, dbname text not null,'
                ' data jsonb not null, tag_data jsonb'
                ') partition by list (dbname)'
            )
            for s, source in enumerate(SOURCES):
                cur.execute(
                    f'create table part_{s} partition of public.pgss_queryid_queries'
                    f" for values in ('{source}') partition by range (time)"
                )
                cur.execute(
                    f'create table part_{s}_old partition of part_{s}'
                    ' for values from (%s) to (%s)',
                    (old - timedelta(days=1), old + timedelta(days=1)),
                )
                cur.execute(
                    f'create table part_{s}_recent partition of part_{s}'
                    ' for values from (%s) to (%s)',
                    (recent - timedelta(days=1), recent + timedelta(days=1)),
                )
                for t in (old, recent):
                    for i in range(ROWS_PER_PARTITION):
                        cur.execute(
                            'insert into public.pgss_queryid_queries (time, dbname, data)'
                            ' values (%s, %s, %s)',
                            (t, source, json.dumps({'queryid': f'{t.timestamp()}-{i}'})),
                        )
        yield conn, psycopg2.extensions.make_dsn(**url)
    finally:
        if conn:
            conn.close()
        with admin.cursor() as cur:
            cur.execute(f'drop database if exists {name} with (force)')
        admin.close()


def _counts(conn):
    with conn.cursor() as cur:
        cur.execute(
            'select tableoid::regclass::text, count(*)'
            ' from public.pgss_queryid_queries group by 1'
        )
        return dict(cur.fetchall())


@pytest.mark.parametrize('batch_size', [10, 10000])
def test_cleanup_deletes_only_rows_past_retention(sink, batch_size):
    conn, dsn = sink
    expected_before = {
        f'part_{s}_{age}': ROWS_PER_PARTITION
        for s in range(len(SOURCES)) for age in ('old', 'recent')
    }
    assert _counts(conn) == expected_before

    with patch.multiple(
        app_module,
        POSTGRES_SINK_URL=dsn,
        QUERYID_RETENTION_HOURS=RETENTION_HOURS,
        QUERYID_RETENTION_BATCH_SIZE=batch_size,
        QUERYID_RETENTION_MAX_ITERATIONS=10,
    ):
        app_module._run_retention_cleanup()

    assert _counts(conn) == {
        f'part_{s}_recent': ROWS_PER_PARTITION for s in range(len(SOURCES))
    }
