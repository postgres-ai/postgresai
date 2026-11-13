"""Shared fixtures for reporter tests."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytest
from unittest.mock import MagicMock

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture(name="prom_result")
def fixture_prom_result() -> Callable[[Optional[List[Dict]], str], Dict]:
    """Build a Prometheus-like payload for the happy-path tests."""

    def _builder(rows: Optional[List[Dict]] = None, status: str = "success") -> Dict:
        return {
            "status": status,
            "data": {
                "result": rows or [],
            },
        }

    return _builder


@pytest.fixture(name="series_sample")
def fixture_series_sample() -> Callable[[str, Optional[Dict], Optional[List[Tuple[Union[float, int], Union[float, int, str]]]]], Dict]:
    """Create metric entries (metric metadata + values array) for query_range tests."""

    def _builder(
        metric_name: str,
        labels: Optional[Dict] = None,
        values: Optional[List[Tuple[Union[float, int], Union[float, int, str]]]] = None,
    ) -> Dict:
        labels = labels or {}
        values = values or []
        return {
            "metric": {"__name__": metric_name, **labels},
            "values": [[ts, str(val)] for ts, val in values],
        }

    return _builder


@pytest.fixture
def generator():
    """Create a PostgresReportGenerator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


@pytest.fixture
def mock_prometheus_success():
    """Mock successful Prometheus response."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {
                        "cluster": "test-cluster",
                        "node_name": "node-01"
                    },
                    "value": [1234567890, "100"]
                }
            ]
        }
    }


@pytest.fixture
def mock_prometheus_empty():
    """Mock empty Prometheus response."""
    return {
        "status": "success",
        "data": {
            "result": []
        }
    }


@pytest.fixture
def sample_a003_report():
    """Sample A003 report for testing derived reports."""
    return {
        "checkId": "A003",
        "results": {
            "node-01": {
                "data": {
                    # D004 settings
                    "pg_stat_statements.max": {"setting": "10000"},
                    "pg_stat_statements.track": {"setting": "all"},

                    # F001 settings
                    "autovacuum": {"setting": "on"},
                    "autovacuum_max_workers": {"setting": "3"},
                    "autovacuum_naptime": {"setting": "60s"},

                    # G001 settings
                    "shared_buffers": {"setting": "128MB"},
                    "work_mem": {"setting": "4MB"},
                    "max_connections": {"setting": "100"},
                    "effective_cache_size": {"setting": "4GB"},
                }
            }
        }
    }


@pytest.fixture
def mock_pg_conn():
    """Mock PostgreSQL connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


# Override pytest-postgresql's postgresql fixture to use system PostgreSQL directly
# This avoids hanging when pytest-postgresql tries to start its own PostgreSQL instance
@pytest.fixture
def postgresql(request):
    """
    PostgreSQL fixture that connects to system PostgreSQL.

    In CI, PostgreSQL is started via 'service postgresql start' before tests run.
    This fixture connects to it directly instead of using pytest-postgresql's
    process management (which can hang trying to find/start pg_ctl).

    Note: Default credentials (postgres user, no password) are used intentionally
    for CI test environments only. This is NOT suitable for production-like testing.
    """
    import psycopg2
    from psycopg2 import sql
    import os

    # Try to connect to system PostgreSQL
    # In CI (Debian), PostgreSQL runs on localhost:5432 as user 'postgres'
    # Locally, use environment variables or defaults
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5432"))
    user = os.environ.get("PGUSER", "postgres")
    dbname = os.environ.get("PGDATABASE", "postgres")
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")

    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            host=host,
            port=port,
        )
        conn.autocommit = True

        # Create a test database for isolation
        # Using PID ensures uniqueness per test process
        test_db = f"test_reporter_{os.getpid()}"
        test_db_ident = sql.Identifier(test_db)

        with conn.cursor() as cur:
            # Use sql.SQL for safe identifier handling (prevents SQL injection)
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(test_db_ident))
            cur.execute(sql.SQL("CREATE DATABASE {}").format(test_db_ident))
        conn.close()

        # Connect to the test database
        conn = psycopg2.connect(
            dbname=test_db,
            user=user,
            host=host,
            port=port,
        )
        conn.autocommit = True
        # A test that fails before it closes its own connection leaves that
        # connection in a transaction, holding locks. Fail the fixtures'
        # cleanup then; without this it waits for the lock until the job
        # times out.
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '10s'")

        yield conn

        # Cleanup
        conn.close()
        cleanup_conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            host=host,
            port=port,
        )
        cleanup_conn.autocommit = True
        with cleanup_conn.cursor() as cur:
            # FORCE ends connections a failed test left open.
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(test_db_ident))
        cleanup_conn.close()

    except psycopg2.OperationalError as e:
        # In CI, connection failure is a real error - don't silently skip
        if is_ci:
            raise RuntimeError(f"PostgreSQL connection failed in CI: {e}") from e
        # Locally, skip if PostgreSQL is not available
        pytest.skip(f"PostgreSQL not available: {e}")
