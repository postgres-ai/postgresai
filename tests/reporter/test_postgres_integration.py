import json
from datetime import datetime, timezone
from typing import Callable, Tuple

import pytest

from reporter.postgres_reports import PostgresReportGenerator

Seeder = Callable[[str, str, str], None]


@pytest.fixture(scope="function")
def sink_index_data(postgresql) -> Tuple[str, Seeder]:
    conn = postgresql
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        create table if not exists public.index_definitions (
            time timestamptz not null,
            dbname text not null,
            data jsonb not null,
            tag_data jsonb
        )
        """
    )

    def seed(dbname: str, index_name: str, index_def: str) -> None:
        payload = {
            "indexrelname": index_name,
            "index_definition": index_def,
            "schemaname": "public",
            "relname": "tbl",
        }
        with conn.cursor() as seed_cur:
            seed_cur.execute(
                (
                    "insert into public.index_definitions "
                    "(time, dbname, data) values (%s, %s, %s::jsonb)"
                ),
                (datetime.now(timezone.utc), dbname, json.dumps(payload)),
            )

    host = conn.info.host or conn.info.hostaddr or "localhost"
    port = conn.info.port
    user = conn.info.user
    dbname = conn.info.dbname
    dsn = f"postgresql://{user}@{host}:{port}/{dbname}"

    yield dsn, seed

    cur.execute("truncate table public.index_definitions")
    cur.close()


QuerySeeder = Callable[[str, str, str], None]


@pytest.fixture(scope="function")
def sink_query_data(postgresql) -> Tuple[str, QuerySeeder]:
    """Fixture for testing query text retrieval from sink."""
    conn = postgresql
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        create table if not exists public.pgss_queryid_queries (
            time timestamptz not null,
            dbname text not null,
            data jsonb not null,
            tag_data jsonb
        )
        """
    )

    def seed(dbname: str, queryid: str, query_text: str) -> None:
        payload = {
            "queryid": queryid,
            "query": query_text,
        }
        with conn.cursor() as seed_cur:
            seed_cur.execute(
                (
                    "insert into public.pgss_queryid_queries "
                    "(time, dbname, data) values (%s, %s, %s::jsonb)"
                ),
                (datetime.now(timezone.utc), dbname, json.dumps(payload)),
            )

    host = conn.info.host or conn.info.hostaddr or "localhost"
    port = conn.info.port
    user = conn.info.user
    dbname = conn.info.dbname
    dsn = f"postgresql://{user}@{host}:{port}/{dbname}"

    yield dsn, seed

    cur.execute("truncate table public.pgss_queryid_queries")
    cur.close()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_index_definitions_from_sink(sink_index_data) -> None:
    dsn, seed = sink_index_data
    seed("db1", "idx_users", "CREATE INDEX idx_users ON users(id)")
    seed("db2", "idx_orders", "CREATE INDEX idx_orders ON orders(id)")

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    assert generator.connect_postgres_sink()

    definitions = generator.get_index_definitions_from_sink()

    assert definitions["db1.idx_users"] == "CREATE INDEX idx_users ON users(id)"
    assert definitions["db2.idx_orders"] == "CREATE INDEX idx_orders ON orders(id)"

    generator.close_postgres_sink()
    assert generator.pg_conn is None


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_index_definitions_from_sink_with_db_filter(sink_index_data) -> None:
    """Test filtering index definitions by database name."""
    dsn, seed = sink_index_data
    seed("db1", "idx_users", "CREATE INDEX idx_users ON users(id)")
    seed("db2", "idx_orders", "CREATE INDEX idx_orders ON orders(id)")
    seed("db1", "idx_posts", "CREATE INDEX idx_posts ON posts(user_id)")

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    assert generator.connect_postgres_sink()

    # Get only db1 indexes
    definitions = generator.get_index_definitions_from_sink(db_name="db1")

    assert "idx_users" in definitions
    assert "idx_posts" in definitions
    assert "idx_orders" not in definitions
    assert definitions["idx_users"] == "CREATE INDEX idx_users ON users(id)"
    assert definitions["idx_posts"] == "CREATE INDEX idx_posts ON posts(user_id)"

    generator.close_postgres_sink()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_index_definitions_returns_empty_when_no_connection(sink_index_data) -> None:
    """Test that get_index_definitions_from_sink returns empty dict when no connection."""
    dsn, seed = sink_index_data

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    # Don't connect - should return empty dict
    definitions = generator.get_index_definitions_from_sink()

    assert definitions == {}


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_connect_postgres_sink_with_invalid_url() -> None:
    """Test connection failure with invalid PostgreSQL URL."""
    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url="postgresql://baduser:badpass@invalid-host:5432/baddb",
    )

    result = generator.connect_postgres_sink()
    assert result is False
    assert generator.pg_conn is None


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_connect_postgres_sink_with_empty_url() -> None:
    """Test that empty sink URL returns False."""
    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url="",
    )

    result = generator.connect_postgres_sink()
    assert result is False


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_close_postgres_sink_when_not_connected() -> None:
    """Test that close_postgres_sink doesn't crash when not connected."""
    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url="",
    )

    # Should not raise
    generator.close_postgres_sink()
    assert generator.pg_conn is None


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_queryid_queries_from_sink(sink_query_data) -> None:
    """Test retrieving query texts from sink."""
    dsn, seed = sink_query_data
    seed("db1", "12345", "SELECT * FROM users WHERE id = $1")
    seed("db1", "67890", "SELECT COUNT(*) FROM orders")
    seed("db2", "11111", "UPDATE products SET price = $1")

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    assert generator.connect_postgres_sink()

    queries = generator.get_queryid_queries_from_sink()

    assert "db1" in queries
    assert "db2" in queries
    assert queries["db1"]["12345"] == "SELECT * FROM users WHERE id = $1"
    assert queries["db1"]["67890"] == "SELECT COUNT(*) FROM orders"
    assert queries["db2"]["11111"] == "UPDATE products SET price = $1"

    generator.close_postgres_sink()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_queryid_queries_from_sink_with_db_filter(sink_query_data) -> None:
    """Test filtering query texts by database names."""
    dsn, seed = sink_query_data
    seed("db1", "12345", "SELECT * FROM users")
    seed("db2", "67890", "SELECT * FROM orders")
    seed("db3", "11111", "SELECT * FROM products")

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    assert generator.connect_postgres_sink()

    queries = generator.get_queryid_queries_from_sink(db_names=["db1", "db3"])

    assert "db1" in queries
    assert "db3" in queries
    assert "db2" not in queries
    assert queries["db1"]["12345"] == "SELECT * FROM users"
    assert queries["db3"]["11111"] == "SELECT * FROM products"

    generator.close_postgres_sink()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_queryid_queries_with_text_limit(sink_query_data) -> None:
    """Test that query_text_limit truncates long queries."""
    dsn, seed = sink_query_data
    long_query = "SELECT * FROM users WHERE " + " AND ".join([f"col{i} = {i}" for i in range(1000)])
    seed("db1", "12345", long_query)

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    assert generator.connect_postgres_sink()

    queries = generator.get_queryid_queries_from_sink(query_text_limit=100)

    assert "db1" in queries
    assert "12345" in queries["db1"]
    # Cut at 100 characters, with a mark that it was cut.
    assert queries["db1"]["12345"] == long_query[:100] + "..."

    generator.close_postgres_sink()


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_get_queryid_queries_returns_empty_when_no_connection(sink_query_data) -> None:
    """Test that get_queryid_queries_from_sink returns empty dict when no connection."""
    dsn, seed = sink_query_data

    generator = PostgresReportGenerator(
        prometheus_url="http://unused",
        postgres_sink_url=dsn,
    )
    # Don't connect - should return empty dict
    queries = generator.get_queryid_queries_from_sink()

    assert queries == {}
