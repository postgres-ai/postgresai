"""Tests for the Flask monitoring backend."""
import importlib
import os
import re
import threading
import pytest
import json
import psycopg2
from unittest.mock import patch, mock_open, Mock, MagicMock, call

import app as app_module
from app import (
    app,
    read_version_file,
    smart_truncate_query,
    _escape_prometheus_label,
    escape_promql_label,
    escape_promql_regex_literal,
)


@pytest.fixture
def client():
    """Create test client."""
    original_trigger_migration_applied = app_module._trigger_migration_applied
    original_active_minutes = app_module.QUERYID_ACTIVE_MINUTES
    original_retention_hours = app_module.QUERYID_RETENTION_HOURS
    original_retention_max_iterations = app_module.QUERYID_RETENTION_MAX_ITERATIONS
    cleanup_was_running = app_module._cleanup_running.is_set()

    app.config['TESTING'] = True
    # Reset the lazy migration flag so tests are independent
    app_module._trigger_migration_applied = True  # Skip migration in tests
    app_module._cleanup_running.clear()

    try:
        with app.test_client() as client:
            yield client
    finally:
        app_module._trigger_migration_applied = original_trigger_migration_applied
        app_module.QUERYID_ACTIVE_MINUTES = original_active_minutes
        app_module.QUERYID_RETENTION_HOURS = original_retention_hours
        app_module.QUERYID_RETENTION_MAX_ITERATIONS = original_retention_max_iterations
        app_module._cleanup_running.clear()
        if cleanup_was_running:
            app_module._cleanup_running.set()


@pytest.fixture
def debug_mode():
    """Enable the /execute-query debug endpoint with a known secret key.

    Yields the app module so tests can assert on module-level state if needed.
    Proper fixture teardown ensures env/module state is always restored.
    """
    import os
    import app as app_module
    os.environ['ENABLE_DEBUG'] = 'true'
    app_module._DEBUG_SECRET_KEY = 'test-secret'
    yield app_module
    os.environ.pop('ENABLE_DEBUG', None)
    app_module._DEBUG_SECRET_KEY = ''


@pytest.fixture
def debug_mode_no_key():
    """Enable debug mode but with no secret key configured (tests the 403 path)."""
    import os
    import app as app_module
    os.environ['ENABLE_DEBUG'] = 'true'
    app_module._DEBUG_SECRET_KEY = ''
    yield app_module
    os.environ.pop('ENABLE_DEBUG', None)
    app_module._DEBUG_SECRET_KEY = ''


class TestVersionEndpoint:
    """Tests for the /version endpoint."""

    def test_version_endpoint_returns_json(self, client):
        """Test that /version returns valid JSON."""
        response = client.get('/version')
        assert response.status_code == 200
        assert response.content_type == 'application/json'

    def test_version_endpoint_returns_array(self, client):
        """Test that /version returns array for Grafana Infinity datasource."""
        response = client.get('/version')
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_version_endpoint_contains_version_field(self, client):
        """Test that /version response contains version field."""
        response = client.get('/version')
        data = json.loads(response.data)
        assert 'version' in data[0]

    def test_version_endpoint_contains_build_ts_field(self, client):
        """Test that /version response contains build_ts field."""
        response = client.get('/version')
        data = json.loads(response.data)
        assert 'build_ts' in data[0]

    def test_version_endpoint_contains_display_field(self, client):
        """Test that /version response contains pre-formatted display field."""
        response = client.get('/version')
        data = json.loads(response.data)
        assert 'display' in data[0]
        assert 'PostgresAI v' in data[0]['display']


class TestReadVersionFile:
    """Tests for the read_version_file function."""

    def test_read_version_file_success(self):
        """Test reading version file successfully."""
        mock_content = "1.2.3"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            result = read_version_file("/VERSION")
            assert result == "1.2.3"

    def test_read_version_file_strips_whitespace(self):
        """Test that version file content is stripped."""
        mock_content = "  1.2.3\n  "
        with patch("builtins.open", mock_open(read_data=mock_content)):
            result = read_version_file("/VERSION")
            assert result == "1.2.3"

    def test_read_version_file_not_found_returns_default(self):
        """Test that missing file returns default value."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            result = read_version_file("/VERSION")
            assert result == "unknown"

    def test_read_version_file_custom_default(self):
        """Test custom default value when file not found."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            result = read_version_file("/VERSION", default="0.0.0")
            assert result == "0.0.0"


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @patch('app.get_prometheus_client')
    def test_health_endpoint_healthy(self, mock_prom, client):
        """Test /health returns healthy when Prometheus is reachable."""
        mock_prom.return_value.get_current_metric_value.return_value = [{'value': 1}]
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'

    @patch('app.get_prometheus_client')
    def test_health_endpoint_unhealthy(self, mock_prom, client):
        """Test /health returns unhealthy when Prometheus is unreachable."""
        mock_prom.return_value.get_current_metric_value.side_effect = Exception("Connection failed")
        response = client.get('/health')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'


class TestSmartTruncateQuery:
    """Tests for the smart_truncate_query function."""

    def test_empty_query(self):
        """Test that empty queries return empty string."""
        assert smart_truncate_query('') == ''
        assert smart_truncate_query(None) == ''

    def test_short_query_unchanged(self):
        """Test that short queries are returned unchanged."""
        short_query = "SELECT * FROM users"
        assert smart_truncate_query(short_query, max_length=50) == short_query

    def test_strip_leading_block_comment(self):
        """Test that leading block comments are stripped."""
        query = "/* comment */ SELECT * FROM users"
        result = smart_truncate_query(query, max_length=50)
        assert result == "SELECT * FROM users"
        assert "/*" not in result

    def test_strip_multiple_block_comments(self):
        """Test that multiple leading block comments are stripped."""
        query = "/* c1 */ /* c2 */ SELECT id FROM orders"
        result = smart_truncate_query(query, max_length=50)
        assert result == "SELECT id FROM orders"

    def test_strip_leading_line_comment(self):
        """Test that leading single-line comments are stripped."""
        query = "-- this is a comment\nSELECT * FROM users"
        result = smart_truncate_query(query, max_length=50)
        assert "SELECT" in result
        assert "--" not in result

    def test_strip_mixed_comments(self):
        """Test that mixed comments are stripped."""
        query = "/* block */ -- line\nSELECT * FROM users"
        result = smart_truncate_query(query, max_length=50)
        assert "SELECT" in result
        assert "/*" not in result
        assert "--" not in result

    def test_select_with_from_extraction(self):
        """Test that SELECT queries show FROM clause tables."""
        query = "SELECT id, name, email, created_at, updated_at, status FROM users WHERE active = true"
        result = smart_truncate_query(query, max_length=40)
        # Smart truncation outputs lowercase
        assert "select" in result
        assert "from" in result
        assert "users" in result
        assert "..." in result
        assert len(result) <= 40

    def test_select_multiple_tables(self):
        """Test extraction of multiple tables from FROM clause."""
        query = "SELECT a.id, b.name FROM users a, orders b WHERE a.id = b.user_id"
        result = smart_truncate_query(query, max_length=60)
        # Smart truncation outputs lowercase
        assert "from" in result
        assert "users" in result

    def test_cte_extraction(self):
        """Test that CTEs are shown with their names."""
        query = "WITH active_users AS (SELECT * FROM users WHERE active) SELECT * FROM active_users"
        result = smart_truncate_query(query, max_length=70)
        # Smart truncation outputs lowercase
        assert "with" in result
        assert "active_users" in result
        # Should extract CTE name and the main FROM clause
        assert "select ... from" in result

    def test_multiple_ctes(self):
        """Test extraction of multiple CTE names."""
        query = "WITH cte1 AS (SELECT * FROM a), cte2 AS (SELECT * FROM b) SELECT * FROM cte1, cte2 WHERE id = 1"
        result = smart_truncate_query(query, max_length=50)
        # Smart truncation outputs lowercase
        assert "with" in result
        assert "cte1" in result
        assert "select ... from" in result

    def test_insert_query(self):
        """Test that INSERT queries show target table."""
        query = "INSERT INTO users (id, name, email) VALUES (1, 'John', 'john@example.com')"
        result = smart_truncate_query(query, max_length=40)
        # Smart truncation outputs lowercase
        assert "insert into" in result
        assert "users" in result

    def test_update_query(self):
        """Test that UPDATE queries show target table."""
        query = "UPDATE users SET email = 'new@example.com', status = 'active' WHERE id = 123"
        result = smart_truncate_query(query, max_length=40)
        # Smart truncation outputs lowercase
        assert "update" in result
        assert "users" in result

    def test_delete_query(self):
        """Test that DELETE queries show target table."""
        query = "DELETE FROM audit_logs WHERE created_at < '2024-01-01'"
        result = smart_truncate_query(query, max_length=40)
        # Smart truncation outputs lowercase
        assert "delete from" in result
        assert "audit_logs" in result

    def test_fallback_on_unknown_query(self):
        """Test fallback to simple truncation for unknown query types."""
        query = "VACUUM ANALYZE users"
        result = smart_truncate_query(query, max_length=15)
        assert len(result) <= 15
        assert result.endswith('...')

    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        query = "SELECT   *   FROM    users   WHERE   id = 1"
        result = smart_truncate_query(query, max_length=50)
        assert "  " not in result  # No double spaces

    def test_respects_max_length(self):
        """Test that result never exceeds max_length."""
        query = "SELECT very_long_column_name_1, very_long_column_name_2 FROM extremely_long_table_name WHERE condition"
        for max_len in [20, 30, 40, 50]:
            result = smart_truncate_query(query, max_length=max_len)
            assert len(result) <= max_len, f"Result '{result}' exceeds max_length {max_len}"

    def test_pgss_comment_stripping(self):
        """Test stripping of pg_stat_statements style comments."""
        query = "/* pgwatch_monitor_user */ SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
        result = smart_truncate_query(query, max_length=50)
        assert "pgwatch" not in result
        # Smart truncation outputs lowercase
        assert "select" in result
        assert "pg_stat_activity" in result

    def test_complex_query_with_joins(self):
        """Test handling of queries with JOINs."""
        query = "SELECT u.id, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed'"
        result = smart_truncate_query(query, max_length=40)
        # Smart truncation outputs lowercase
        assert "select" in result
        assert "from" in result
        assert "users" in result

    def test_select_without_from_clause(self):
        """Test SELECT queries that call functions without FROM clause."""
        query = "select current_database() as tag_datname, case when pg_is_in_recovery() then (pg_last_wal_replay_lsn() - $1) % ($2^$3)::bigint else (pg_current_wal_lsn() - $1) % ($2^$3)::bigint end"
        result = smart_truncate_query(query, max_length=100)
        # Should fall back to simple truncation since no FROM clause
        assert len(result) <= 100
        assert result.endswith('...')
        # Should start with select, not a comment
        assert result.lower().startswith('select')

    def test_select_without_from_short_truncation(self):
        """Test SELECT without FROM with shorter max_length."""
        query = "SELECT current_database(), pg_is_in_recovery(), now()"
        result = smart_truncate_query(query, max_length=30)
        assert len(result) <= 30
        # Should either fit or be truncated with ...
        if len(query) > 30:
            assert result.endswith('...')

    def test_inline_block_comment_stripped(self):
        """Test that inline block comments (not just leading) are stripped."""
        query = "SELECT /* inline comment */ id FROM users"
        result = smart_truncate_query(query, max_length=50)
        assert "/*" not in result
        assert "inline" not in result
        assert "SELECT" in result
        assert "users" in result

    def test_block_comment_with_newlines(self):
        """Test block comments containing newlines are stripped."""
        query = """/* This is a
        multi-line
        comment */ SELECT * FROM users"""
        result = smart_truncate_query(query, max_length=50)
        assert "/*" not in result
        assert "multi-line" not in result
        assert "SELECT" in result

    def test_pgwatch_comment_at_start(self):
        """Test pgwatch-style comments at query start are stripped."""
        query = "/* First we have to remove them from the extension */ ALTER EXTENSION pg_stat_statements DROP VIEW pg_stat_statements"
        result = smart_truncate_query(query, max_length=100)
        # Comment should be stripped
        assert "/*" not in result
        assert "First we have to" not in result
        # ALTER should be at the start
        assert result.upper().startswith("ALTER")

    def test_comment_followed_by_alter(self):
        """Test ALTER statement after comment is properly shown."""
        query = "/* comment */ ALTER TABLE users ADD COLUMN email VARCHAR(255)"
        result = smart_truncate_query(query, max_length=60)
        assert "/*" not in result
        assert "ALTER" in result

    def test_comment_followed_by_create(self):
        """Test CREATE statement after comment is properly shown."""
        query = "/* setup */ CREATE INDEX idx_users_email ON users(email)"
        result = smart_truncate_query(query, max_length=60)
        assert "/*" not in result
        assert "CREATE" in result

    def test_lowercase_output_for_select(self):
        """Test that smart truncation outputs lowercase for SELECT queries."""
        query = "SELECT id, name FROM users WHERE active = true"
        result = smart_truncate_query(query, max_length=40)
        # Should output lowercase "select ... from"
        assert "select" in result
        assert "from" in result

    def test_lowercase_output_for_insert(self):
        """Test that smart truncation outputs lowercase for INSERT queries."""
        query = "INSERT INTO users (id, name) VALUES (1, 'test')"
        result = smart_truncate_query(query, max_length=30)
        assert "insert into" in result

    def test_lowercase_output_for_update(self):
        """Test that smart truncation outputs lowercase for UPDATE queries."""
        query = "UPDATE users SET name = 'new' WHERE id = 1"
        result = smart_truncate_query(query, max_length=30)
        assert "update" in result

    def test_lowercase_output_for_delete(self):
        """Test that smart truncation outputs lowercase for DELETE queries."""
        query = "DELETE FROM users WHERE id = 1 AND status = 'inactive' AND created_at < now()"
        result = smart_truncate_query(query, max_length=30)
        assert "delete from" in result

    def test_lowercase_output_for_cte(self):
        """Test that smart truncation outputs lowercase for CTE queries."""
        query = "WITH active_users AS (SELECT * FROM users) SELECT * FROM active_users"
        result = smart_truncate_query(query, max_length=60)
        assert "with" in result
        assert "select" in result

    def test_multiple_inline_comments(self):
        """Test multiple inline comments are all stripped."""
        query = "SELECT /* c1 */ id, /* c2 */ name FROM /* c3 */ users"
        result = smart_truncate_query(query, max_length=50)
        assert "/*" not in result
        assert "c1" not in result
        assert "c2" not in result
        assert "c3" not in result

    def test_line_comment_at_end(self):
        """Test line comment at end of query is stripped."""
        query = "SELECT * FROM users -- get all users"
        result = smart_truncate_query(query, max_length=50)
        assert "--" not in result
        assert "get all users" not in result

    def test_very_long_query_respects_limit(self):
        """Test that very long queries respect the max_length limit."""
        long_query = "SELECT " + ", ".join([f"column_{i}" for i in range(100)]) + " FROM very_long_table_name"
        for max_len in [30, 60, 100]:
            result = smart_truncate_query(long_query, max_length=max_len)
            assert len(result) <= max_len, f"Result length {len(result)} exceeds max {max_len}"


class TestEscapePrometheusLabel:
    """Tests for the _escape_prometheus_label function."""

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _escape_prometheus_label('') == ''

    def test_none_returns_empty(self):
        """Test None input returns empty string."""
        assert _escape_prometheus_label(None) == ''

    def test_simple_string_unchanged(self):
        """Test simple strings without special chars are unchanged."""
        assert _escape_prometheus_label('hello world') == 'hello world'

    def test_escapes_backslash(self):
        """Test backslashes are escaped."""
        assert _escape_prometheus_label('path\\to\\file') == 'path\\\\to\\\\file'

    def test_escapes_double_quote(self):
        """Test double quotes are escaped."""
        assert _escape_prometheus_label('say "hello"') == 'say \\"hello\\"'

    def test_replaces_newline_with_space(self):
        """Test newlines are replaced with spaces."""
        assert _escape_prometheus_label('line1\nline2') == 'line1 line2'

    def test_combined_special_chars(self):
        """Test string with multiple special characters."""
        input_str = 'path\\file "name"\nend'
        result = _escape_prometheus_label(input_str)
        assert '\\\\' in result  # Escaped backslash
        assert '\\"' in result   # Escaped quote
        assert '\n' not in result  # No raw newlines

    def test_sql_query_escaping(self):
        """Test escaping of SQL query text."""
        query = 'SELECT * FROM "users" WHERE name = \'test\''
        result = _escape_prometheus_label(query)
        assert '\\"' in result  # Double quotes escaped
        assert "'" in result    # Single quotes unchanged (not escaped in Prometheus)

    def test_multiline_query(self):
        """Test multiline SQL query has newlines replaced."""
        query = "SELECT *\nFROM users\nWHERE id = 1"
        result = _escape_prometheus_label(query)
        assert '\n' not in result
        assert 'SELECT * FROM users WHERE id = 1' == result

    def test_backslash_before_quote(self):
        """Test backslash before quote is escaped correctly."""
        # Input: \" should become \\\"
        assert _escape_prometheus_label('\\"') == '\\\\\\"'

    def test_unicode_preserved(self):
        """Test unicode characters are preserved."""
        assert _escape_prometheus_label('héllo wörld') == 'héllo wörld'
        assert _escape_prometheus_label('日本語') == '日本語'

    def test_carriage_return(self):
        """Test carriage returns are handled."""
        result = _escape_prometheus_label('line1\r\nline2')
        # Should not contain raw \r or \n
        assert '\r' not in result
        assert '\n' not in result

    def test_tab_character(self):
        """Test tab characters are handled."""
        result = _escape_prometheus_label('col1\tcol2')
        # Tabs should be replaced with space
        assert '\t' not in result


class TestQueryInfoMetricsEndpoint:
    """Tests for the /query_info_metrics endpoint."""

    @patch('app.psycopg2.connect')
    def test_get_query_texts_uses_explicit_max_age_with_db_filter(self, mock_connect):
        """Direct sink lookup passes explicit max_age_hours after db_name."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        result = app_module.get_query_texts_from_sink(db_name='mydb', max_age_hours=48)

        assert result == {}
        query, params = mock_cursor.execute.call_args[0]
        # Filter must be against the JSONB ``real_dbname`` payload (the actual
        # Postgres database name), not the partition ``dbname`` column (which
        # stores the pgwatch source name and never matches the dashboard's
        # datname).
        assert "data->>'real_dbname' = %s" in query
        # Locking the regression: the partition ``dbname = %s`` filter must
        # NOT be present — a word-boundary check so we do not accidentally
        # match ``real_dbname``.
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query
        assert 'make_interval(hours => %s)' in query
        assert params == ('mydb', 48)

    @patch('app.psycopg2.connect')
    def test_get_query_texts_defaults_max_age_to_retention_window(self, mock_connect):
        """Direct sink lookup defaults max_age_hours to QUERYID_RETENTION_HOURS."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        result = app_module.get_query_texts_from_sink(max_age_hours=None)

        assert result == {}
        query, params = mock_cursor.execute.call_args[0]
        assert "data->>'real_dbname'" not in query
        # Even on the no-filter branch the buggy partition predicate must stay
        # gone — a partial revert that left ``dbname = %s`` in either branch
        # would still match all the substring checks otherwise.
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query
        assert 'make_interval(hours => %s)' in query
        assert params == (app_module.QUERYID_RETENTION_HOURS,)

    @patch('app.psycopg2.connect')
    def test_endpoint_returns_prometheus_format(self, mock_connect, client):
        """Test endpoint returns Prometheus exposition format."""
        # Mock database connection
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([
            {'queryid': '123', 'query': 'SELECT * FROM users'}
        ])

        response = client.get('/query_info_metrics')
        assert response.status_code == 200
        assert response.content_type == 'text/plain; charset=utf-8'

    @patch('app.psycopg2.connect')
    def test_endpoint_includes_help_and_type(self, mock_connect, client):
        """Test endpoint includes HELP and TYPE comments."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_info_metrics')
        data = response.data.decode('utf-8')
        assert '# HELP pgwatch_query_info' in data
        assert '# TYPE pgwatch_query_info gauge' in data

    @patch('app.psycopg2.connect')
    def test_displayname_has_queryid_prefix(self, mock_connect, client):
        """Test that displayname labels include queryid prefix."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([
            {'queryid': '12345', 'query': 'SELECT * FROM users WHERE id = 1'}
        ])

        response = client.get('/query_info_metrics')
        data = response.data.decode('utf-8')

        # All displayname variants should have "queryid | " prefix
        assert '12345 | ' in data
        # The queryid should appear in displayname labels
        assert 'displayname="12345 |' in data

    @patch('app.psycopg2.connect')
    def test_all_truncation_levels_present(self, mock_connect, client):
        """Test that all truncation level labels are present."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([
            {'queryid': '999', 'query': 'SELECT very_long_column_name FROM very_long_table_name WHERE condition = true'}
        ])

        response = client.get('/query_info_metrics')
        data = response.data.decode('utf-8')

        # Check all 8 label types are present
        assert 'displayname="' in data
        assert 'displayname_medium="' in data
        assert 'displayname_long="' in data
        assert 'displayname_raw_short="' in data
        assert 'displayname_raw_medium="' in data
        assert 'displayname_raw_long="' in data
        assert 'displayname_full="' in data
        assert 'displayname_queryid="' in data

    @patch('app.psycopg2.connect')
    def test_smart_truncation_strips_comments(self, mock_connect, client):
        """Test smart truncation removes comments from displayname."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([
            {'queryid': '111', 'query': '/* comment */ SELECT * FROM users'}
        ])

        response = client.get('/query_info_metrics')
        data = response.data.decode('utf-8')

        # Comment should not appear in smart truncation displayname
        # Note: We check for the escaped version since Prometheus labels are escaped
        assert '/* comment */' not in data or 'displayname_raw' in data.split('/* comment */')[0]

    @patch('app.psycopg2.connect')
    def test_db_name_filter(self, mock_connect, client):
        """Test db_name parameter filters results on the JSONB real_dbname payload."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_info_metrics?db_name=mydb')
        assert response.status_code == 200

        # Verify the main query (last execute call) filters by the JSONB
        # ``real_dbname`` payload — the partition ``dbname`` column holds
        # the pgwatch source name and would never match the dashboard's
        # datname value.
        call_args = mock_cursor.execute.call_args
        assert call_args is not None
        query = call_args[0][0]
        assert "data->>'real_dbname' = %s" in query
        # Lock in the regression: word-boundary check rejects a partial
        # revert that left the buggy ``dbname = %s`` filter in place.
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query

    @patch('app.psycopg2.connect')
    def test_db_name_all_skips_filter(self, mock_connect, client):
        """Test db_name='all' skips the filter."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_info_metrics?db_name=all')
        assert response.status_code == 200

        # Verify the main query (last execute call) was called without db_name filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        assert "data->>'real_dbname'" not in query
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query

    @patch('app.psycopg2.connect')
    def test_db_name_variable_skips_filter(self, mock_connect, client):
        """Test db_name starting with $ skips the filter."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_info_metrics?db_name=$db_name')
        assert response.status_code == 200

        # Verify the main query (last execute call) was called without db_name filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        assert "data->>'real_dbname'" not in query
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query

    @patch('app.psycopg2.connect')
    def test_time_filter_applied(self, mock_connect, client):
        """Test that the time filter is applied and QUERYID_ACTIVE_MINUTES flows through."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        # Temporarily override the active minutes constant
        original_minutes = app_module.QUERYID_ACTIVE_MINUTES
        app_module.QUERYID_ACTIVE_MINUTES = 42
        try:
            response = client.get('/query_info_metrics')
            assert response.status_code == 200

            # Verify the main query includes a time filter
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert "make_interval" in query
            assert "mins" in query
            # Verify the parameter value is passed correctly
            params = call_args[0][1]
            assert params == (42,)
        finally:
            app_module.QUERYID_ACTIVE_MINUTES = original_minutes

    @patch('app.psycopg2.connect')
    def test_time_filter_applied_with_db_filter(self, mock_connect, client):
        """Time filter + db_name filter path: both dbname and make_interval land in the query, params pair correctly."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        original_minutes = app_module.QUERYID_ACTIVE_MINUTES
        app_module.QUERYID_ACTIVE_MINUTES = 42
        try:
            response = client.get('/query_info_metrics?db_name=mydb')
            assert response.status_code == 200

            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            params = call_args[0][1]

            assert "data->>'real_dbname' = %s" in query
            assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, query
            assert 'make_interval(mins => %s)' in query
            # The db_filter branch passes (db_name, minutes) in that order.
            assert params == ('mydb', 42)
        finally:
            app_module.QUERYID_ACTIVE_MINUTES = original_minutes

    @patch('app._run_retention_cleanup')
    @patch('app.psycopg2.connect')
    def test_retention_cleanup_runs_in_background(self, mock_connect, mock_cleanup, client):
        """Test that retention cleanup is triggered and the running flag is cleared."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])
        mock_cursor.rowcount = 0
        started = threading.Event()
        release = threading.Event()
        threads = []
        original_thread = threading.Thread

        def cleanup_side_effect():
            started.set()
            release.wait(timeout=5)

        def make_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            threads.append(thread)
            return thread

        mock_cleanup.side_effect = cleanup_side_effect

        with patch('app.threading.Thread', side_effect=make_thread):
            response = client.get('/query_info_metrics')
            assert response.status_code == 200
            assert started.wait(timeout=2)
            assert app_module._cleanup_running.is_set() is True
            release.set()

        for thread in threads:
            thread.join(timeout=2)
            assert thread.is_alive() is False

        mock_cleanup.assert_called_once()
        assert app_module._cleanup_running.is_set() is False

    def test_start_retention_cleanup_thread_coalesces_concurrent_callers(self):
        """Concurrent scrapes should start at most one in-process cleanup thread."""
        app_module._cleanup_running.clear()
        start_barrier = threading.Barrier(3)
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        cleanup_finished = threading.Event()
        errors = []

        def guarded_cleanup():
            cleanup_started.set()
            try:
                release_cleanup.wait(timeout=5)
            finally:
                app_module._cleanup_running.clear()
                cleanup_finished.set()

        def caller():
            try:
                start_barrier.wait(timeout=2)
                app_module._start_retention_cleanup_thread()
            except Exception as exc:
                errors.append(exc)

        with patch('app._run_retention_cleanup_guarded', side_effect=guarded_cleanup) as mock_guarded:
            callers = [threading.Thread(target=caller) for _ in range(2)]
            for thread in callers:
                thread.start()
            start_barrier.wait(timeout=2)
            assert cleanup_started.wait(timeout=2)
            for thread in callers:
                thread.join(timeout=2)
                assert thread.is_alive() is False
            assert errors == []
            assert mock_guarded.call_count == 1
            release_cleanup.set()
            assert cleanup_finished.wait(timeout=2)

        assert app_module._cleanup_running.is_set() is False

    @patch('app.psycopg2.connect')
    def test_retention_cleanup_thread_start_failure_clears_flag(self, mock_connect, client):
        """Thread.start failure must not leave cleanup disabled forever."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])
        broken_thread = Mock()
        broken_thread.start.side_effect = RuntimeError("thread quota exhausted")

        with patch('app.threading.Thread', return_value=broken_thread), \
             patch('app.logger.warning') as mock_warning:
            response = client.get('/query_info_metrics')

        assert response.status_code == 200
        assert app_module._cleanup_running.is_set() is False
        warning_text = ' '.join(str(c) for c in mock_warning.call_args_list)
        assert 'Failed to start retention cleanup thread' in warning_text
        assert 'thread quota exhausted' in warning_text

    @patch('app.psycopg2.connect')
    def test_handles_db_connection_error(self, mock_connect, client):
        """Test endpoint handles database connection errors gracefully."""
        mock_connect.side_effect = Exception("Connection refused")

        response = client.get('/query_info_metrics')
        # Should still return 200 with empty metrics (graceful degradation)
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        assert '# HELP pgwatch_query_info' in data

    @patch('app.psycopg2.connect')
    def test_special_chars_in_query_escaped(self, mock_connect, client):
        """Test special characters in query are escaped in Prometheus labels."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([
            {'queryid': '222', 'query': 'SELECT "column" FROM users\nWHERE x = 1'}
        ])

        response = client.get('/query_info_metrics')
        data = response.data.decode('utf-8')

        # Raw newlines should be replaced
        assert 'users\nWHERE' not in data
        # Double quotes should be escaped
        assert '\\"column\\"' in data or 'column' in data


_GUARD_PROSRC = """
declare queryid_value text;
begin
  queryid_value := new.data->>'queryid';
  perform pg_advisory_xact_lock(lock_key);
  delete from public.pgss_queryid_queries
  where dbname = new.dbname
    and data->>'queryid' = queryid_value
    and time <= new.time;
  ...
end;
"""

_UNPATCHED_PROSRC = """
declare queryid_value text;
begin
  queryid_value := new.data->>'queryid';
  insert into pgss_queryid_queries values (...);
  return null;
end;
"""


class TestTriggerMigration:
    """Tests for the verify-only _apply_trigger_migration function.

    Migration no longer does DDL — the bootstrap role's init.sql owns
    that responsibility. This function verifies the deployed function
    body contains the advisory-lock dedup path and the trigger exists.
    """

    def _make_mock_conn(self, fetchone_sequence):
        """Build a mock connection whose cursor().fetchone() returns successive tuples."""
        app_module._trigger_migration_warning_state['last_at'] = 0.0
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = list(fetchone_sequence)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        return mock_conn, mock_cursor

    def test_verifies_function_body_and_trigger_when_both_present(self):
        """Guard present in function body + trigger exists → flag set, no DDL."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,),               # pg_try_advisory_lock
            (_GUARD_PROSRC,),      # pg_proc.prosrc with guard
            (1,),                  # trigger exists
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._apply_trigger_migration()

        calls = [c[0][0] for c in mock_cursor.execute.call_args_list]
        # Purely read-only: no CREATE / ALTER / DROP.
        assert not any('create or replace function' in c.lower() for c in calls)
        assert not any('create trigger' in c.lower() for c in calls)
        assert not any('alter function' in c.lower() for c in calls)
        assert app_module._trigger_migration_applied is True

    def test_missing_guard_logs_warning_and_sets_flag(self):
        """Deployed function body missing advisory-lock dedup → warn, don't retry-spam."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,),                 # advisory lock
            (_UNPATCHED_PROSRC,),    # prosrc without guard
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn), \
             patch('app.logger.warning') as mock_warning:
            app_module._apply_trigger_migration()

        # Verification failures are retryable so a later init.sql repair is detected.
        assert app_module._trigger_migration_applied is False
        # A clear remediation hint in the log.
        warning_text = ' '.join(str(c) for c in mock_warning.call_args_list)
        assert 'advisory-lock dedup' in warning_text
        assert 'init.sql' in warning_text

    def test_missing_function_logs_warning_and_sets_flag(self):
        """Function doesn't exist at all → warn, then retry on a later request."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,),   # advisory lock
            None,      # pg_proc lookup returns no row
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn), \
             patch('app.logger.warning') as mock_warning:
            app_module._apply_trigger_migration()

        assert app_module._trigger_migration_applied is False
        warning_text = ' '.join(str(c) for c in mock_warning.call_args_list)
        assert 'public.enforce_queryid_uniqueness' in warning_text
        assert 'init.sql' in warning_text

    def test_missing_trigger_logs_warning_and_sets_flag(self):
        """Function exists and is guarded but trigger is absent → warn, then retry."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,),                # advisory lock
            (_GUARD_PROSRC,),       # prosrc with guard
            None,                   # trigger lookup returns no row
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn), \
             patch('app.logger.warning') as mock_warning:
            app_module._apply_trigger_migration()

        assert app_module._trigger_migration_applied is False
        warning_text = ' '.join(str(c) for c in mock_warning.call_args_list)
        assert 'enforce_queryid_uniqueness_trigger' in warning_text
        assert 'init.sql' in warning_text

    def test_does_not_transfer_function_ownership(self):
        """Security: verification path must never issue ALTER FUNCTION ... OWNER."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,), (_GUARD_PROSRC,), (1,),
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._apply_trigger_migration()

        calls = [c[0][0].lower() for c in mock_cursor.execute.call_args_list]
        assert not any('owner to' in c for c in calls)

    def test_advisory_lock_taken_and_released(self):
        """Migration grabs pg_try_advisory_lock and releases it on completion."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,), (_GUARD_PROSRC,), (1,),
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._apply_trigger_migration()

        calls = [c[0][0] for c in mock_cursor.execute.call_args_list]
        assert any('pg_try_advisory_lock' in c for c in calls)
        assert any('pg_advisory_unlock' in c for c in calls)

    def test_advisory_lock_denied_skips_verification(self):
        """When another worker holds the advisory lock, skip without error; flag stays False."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (False,),   # lock denied
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._apply_trigger_migration()

        calls = [c[0][0] for c in mock_cursor.execute.call_args_list]
        assert any('pg_try_advisory_lock' in c for c in calls)
        assert not any('pg_proc' in c for c in calls)
        assert not any('pg_trigger' in c for c in calls)
        # Flag stays False so the next request retries.
        assert app_module._trigger_migration_applied is False

    def test_db_error_leaves_flag_false(self):
        """On DB error, flag remains False so retry is possible."""
        app_module._trigger_migration_applied = False

        with patch('app.psycopg2.connect', side_effect=Exception("connection refused")):
            app_module._apply_trigger_migration()

        assert app_module._trigger_migration_applied is False

    def test_successful_run_sets_flag_so_subsequent_calls_are_noop(self):
        """Once verification succeeds, subsequent calls must not touch the DB."""
        app_module._trigger_migration_applied = False
        mock_conn, mock_cursor = self._make_mock_conn([
            (True,), (_GUARD_PROSRC,), (1,),
        ])

        with patch('app.psycopg2.connect', return_value=mock_conn) as mock_connect:
            app_module._apply_trigger_migration()
            first_call_count = mock_connect.call_count
            # Second invocation should early-return without opening a new connection.
            app_module._apply_trigger_migration()
            assert mock_connect.call_count == first_call_count


class TestRetentionCleanup:
    """Tests for the _run_retention_cleanup function."""

    def _make_mock_conn(self, advisory_lock_granted=True):
        """Build a mock connection; pg_try_advisory_lock returns TRUE by default."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (advisory_lock_granted,)
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        return mock_conn, mock_cursor

    def _delete_calls(self, mock_cursor):
        """Return only the DELETE execute calls (advisory-lock calls excluded)."""
        return [
            c for c in mock_cursor.execute.call_args_list
            if 'delete' in c[0][0].lower()
        ]

    def test_single_batch_exits_loop(self):
        """When the DELETE removes fewer rows than the batch size, the loop exits after one iteration."""
        mock_conn, mock_cursor = self._make_mock_conn()
        mock_cursor.rowcount = 500  # less than batch size (10 000)

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._run_retention_cleanup()

        assert len(self._delete_calls(mock_cursor)) == 1

    def test_full_batch_then_partial_drains_backlog(self):
        """Full batch on iteration 1 keeps looping; partial batch on iteration 2 exits the loop."""
        mock_conn, mock_cursor = self._make_mock_conn()
        # psycopg2 sets cursor.rowcount after each execute. Only the DELETEs
        # care about rowcount; bump it only on DELETE calls so the
        # advisory-lock SELECTs don't consume a value.
        rowcounts = [app_module.QUERYID_RETENTION_BATCH_SIZE, 500]

        def update_rowcount(sql, *args, **kwargs):
            if 'delete' in sql.lower() and rowcounts:
                mock_cursor.rowcount = rowcounts.pop(0)

        mock_cursor.execute.side_effect = update_rowcount

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._run_retention_cleanup()

        assert len(self._delete_calls(mock_cursor)) == 2

    def test_max_iterations_cap_respected(self):
        """The loop is capped so a perpetually-full batch can't monopolize the connection."""
        mock_conn, mock_cursor = self._make_mock_conn()
        mock_cursor.rowcount = app_module.QUERYID_RETENTION_BATCH_SIZE  # always full

        original_max = app_module.QUERYID_RETENTION_MAX_ITERATIONS
        app_module.QUERYID_RETENTION_MAX_ITERATIONS = 3
        try:
            with patch('app.psycopg2.connect', return_value=mock_conn):
                app_module._run_retention_cleanup()
            assert len(self._delete_calls(mock_cursor)) == 3
        finally:
            app_module.QUERYID_RETENTION_MAX_ITERATIONS = original_max

    @pytest.mark.parametrize(
        ('env_name', 'setting_name'),
        [
            ('QUERYID_RETENTION_HOURS', 'QUERYID_RETENTION_HOURS'),
            ('QUERYID_ACTIVE_MINUTES', 'QUERYID_ACTIVE_MINUTES'),
            ('QUERYID_RETENTION_BATCH_SIZE', 'QUERYID_RETENTION_BATCH_SIZE'),
            ('QUERYID_RETENTION_MAX_ITERATIONS', 'QUERYID_RETENTION_MAX_ITERATIONS'),
        ],
    )
    def test_queryid_env_lower_bound_is_one(self, env_name, setting_name):
        """Query-info integer env vars use >=1 lower bounds."""
        try:
            with patch.dict(os.environ, {env_name: '0'}):
                importlib.reload(app_module)
                assert getattr(app_module, setting_name) == 1
        finally:
            importlib.reload(app_module)

    def test_connection_uses_statement_and_lock_timeouts(self):
        """psycopg2.connect is called with server-side timeouts so the thread can't hang."""
        mock_conn, mock_cursor = self._make_mock_conn()

        with patch('app.psycopg2.connect', return_value=mock_conn) as mock_connect:
            app_module._run_retention_cleanup()

        kwargs = mock_connect.call_args.kwargs
        assert 'options' in kwargs
        assert 'statement_timeout=60000' in kwargs['options']
        assert 'lock_timeout=5000' in kwargs['options']

    def test_advisory_lock_taken_and_released(self):
        """Cleanup serializes across workers via pg_try_advisory_lock."""
        mock_conn, mock_cursor = self._make_mock_conn()

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._run_retention_cleanup()

        calls = [c[0][0] for c in mock_cursor.execute.call_args_list]
        assert any('pg_try_advisory_lock' in c for c in calls)
        assert any('pg_advisory_unlock' in c for c in calls)

    def test_advisory_lock_denied_skips_delete(self):
        """When another worker already holds the cleanup lock, skip without running DELETE."""
        mock_conn, mock_cursor = self._make_mock_conn(advisory_lock_granted=False)

        with patch('app.psycopg2.connect', return_value=mock_conn):
            app_module._run_retention_cleanup()

        assert self._delete_calls(mock_cursor) == []

    def test_delete_error_after_lock_attempts_unlock_and_is_swallowed(self):
        """DELETE errors are caught after attempting to release the advisory lock."""
        mock_conn, mock_cursor = self._make_mock_conn()

        def execute_side_effect(sql, *args, **kwargs):
            if 'delete from public.pgss_queryid_queries' in sql.lower():
                raise psycopg2.OperationalError("delete failed")

        mock_cursor.execute.side_effect = execute_side_effect

        with patch('app.psycopg2.connect', return_value=mock_conn), \
             patch('app.logger.warning') as mock_warning:
            app_module._run_retention_cleanup()

        calls = [c[0][0].lower() for c in mock_cursor.execute.call_args_list]
        assert any('pg_try_advisory_lock' in c for c in calls)
        assert any('delete from public.pgss_queryid_queries' in c for c in calls)
        assert any('pg_advisory_unlock' in c for c in calls)
        warning_text = ' '.join(str(c) for c in mock_warning.call_args_list)
        assert 'Retention cleanup failed' in warning_text
        assert 'delete failed' in warning_text

    def test_db_exception_caught_not_propagated(self):
        """DB errors are caught and logged, not raised."""
        with patch('app.psycopg2.connect', side_effect=Exception("connection refused")):
            # Should not raise
            app_module._run_retention_cleanup()

    def test_retention_hours_parameter_flows(self):
        """QUERYID_RETENTION_HOURS is passed as a parameter to the DELETE."""
        original_hours = app_module.QUERYID_RETENTION_HOURS
        app_module.QUERYID_RETENTION_HOURS = 48
        mock_conn, mock_cursor = self._make_mock_conn()

        try:
            with patch('app.psycopg2.connect', return_value=mock_conn):
                app_module._run_retention_cleanup()

            delete_call = self._delete_calls(mock_cursor)[0]
            params = delete_call[0][1]
            assert params == (48, app_module.QUERYID_RETENTION_BATCH_SIZE)
        finally:
            app_module.QUERYID_RETENTION_HOURS = original_hours


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint with mocked psycopg2."""

    @patch('app.get_prometheus_client')
    def test_metrics_returns_json(self, mock_prom, client):
        """Test /metrics returns JSON with pg_stat_statements_metrics key."""
        mock_prom.return_value.all_metrics.return_value = [
            'pgwatch_pg_stat_statements_calls',
            'pgwatch_pg_stat_statements_rows',
            'other_metric',
        ]
        response = client.get('/metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'pg_stat_statements_metrics' in data

    @patch('app.get_prometheus_client')
    def test_metrics_filters_pgss_metrics(self, mock_prom, client):
        """Test /metrics only returns pg_stat_statements metrics."""
        mock_prom.return_value.all_metrics.return_value = [
            'pgwatch_pg_stat_statements_calls',
            'pgwatch_pg_stat_statements_rows',
            'node_cpu_seconds_total',
            'up',
        ]
        response = client.get('/metrics')
        data = json.loads(response.data)
        pgss = data['pg_stat_statements_metrics']
        assert all('pg_stat_statements' in m for m in pgss)
        assert 'node_cpu_seconds_total' not in pgss
        assert 'up' not in pgss

    @patch('app.get_prometheus_client')
    def test_metrics_empty_when_no_pgss(self, mock_prom, client):
        """Test /metrics returns empty list when no pg_stat_statements metrics exist."""
        mock_prom.return_value.all_metrics.return_value = ['node_cpu_seconds_total', 'up']
        response = client.get('/metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['pg_stat_statements_metrics'] == []

    @patch('app.get_prometheus_client')
    def test_metrics_returns_500_on_prometheus_error(self, mock_prom, client):
        """Test /metrics returns 500 when Prometheus is unreachable."""
        mock_prom.return_value.all_metrics.side_effect = Exception("Prometheus unreachable")
        response = client.get('/metrics')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestExecuteQueryEndpoint:
    """Tests for the gated /execute-query debug endpoint."""

    def test_returns_404_when_debug_disabled(self, client):
        """Test endpoint returns 404 when ENABLE_DEBUG is not set."""
        import os
        os.environ.pop('ENABLE_DEBUG', None)
        response = client.post('/execute-query',
                               json={'query': 'SELECT 1'},
                               content_type='application/json')
        assert response.status_code == 404

    def test_returns_403_when_no_secret_key_configured(self, client, debug_mode_no_key):
        """Test endpoint returns 403 when ENABLE_DEBUG=true but DEBUG_SECRET_KEY not set."""
        response = client.post('/execute-query',
                               json={'query': 'SELECT 1'},
                               content_type='application/json')
        assert response.status_code == 403

    def test_returns_400_on_missing_json_body(self, client, debug_mode):
        """Test endpoint returns 400 when Content-Type is not application/json."""
        response = client.post('/execute-query',
                               data='not json',
                               content_type='text/plain',
                               headers={'Authorization': 'Bearer test-secret'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data


class TestEscapePromqlLabel:
    """Unit tests for the escape_promql_label function."""

    def test_simple_value_unchanged(self):
        assert escape_promql_label("hello") == "hello"

    def test_empty_string(self):
        assert escape_promql_label("") == ""

    def test_escapes_backslash(self):
        assert escape_promql_label("path\\to") == "path\\\\to"

    def test_escapes_double_quote(self):
        assert escape_promql_label('say "hi"') == 'say \\"hi\\"'

    def test_escapes_newline(self):
        assert escape_promql_label("line1\nline2") == "line1\\nline2"

    def test_backslash_before_quote_double_escaped(self):
        # Input \", output \\\"
        assert escape_promql_label('\\"') == '\\\\\\"'

    def test_combined_special_chars(self):
        result = escape_promql_label('a\\b"c\nd')
        assert '\\\\' in result    # escaped backslash
        assert '\\"' in result     # escaped quote
        assert '\\n' in result     # escaped newline


class TestEscapePromqlRegexLiteral:
    """Unit tests for the escape_promql_regex_literal function."""

    def test_simple_value_unchanged(self):
        assert escape_promql_regex_literal("hello") == "hello"

    def test_empty_string(self):
        assert escape_promql_regex_literal("") == ""

    def test_escapes_dot(self):
        # Schema names like "public.myschema" must match literally
        assert escape_promql_regex_literal("public.myschema") == "public\\.myschema"

    def test_escapes_plus(self):
        assert escape_promql_regex_literal("a+b") == "a\\+b"

    def test_escapes_star(self):
        assert escape_promql_regex_literal("a*b") == "a\\*b"

    def test_escapes_pipe(self):
        assert escape_promql_regex_literal("a|b") == "a\\|b"

    def test_escapes_parens(self):
        assert escape_promql_regex_literal("(group)") == "\\(group\\)"

    def test_escapes_brackets(self):
        # '-' is not a RE2 metacharacter outside a character class; only '[' and ']' are escaped
        assert escape_promql_regex_literal("[0-9]") == "\\[0-9\\]"

    def test_escapes_caret_and_dollar(self):
        result = escape_promql_regex_literal("^start$end")
        assert "\\^" in result
        assert "\\$" in result

    def test_also_escapes_double_quote(self):
        result = escape_promql_regex_literal('"quoted"')
        assert '\\"' in result

    def test_also_escapes_backslash(self):
        result = escape_promql_regex_literal("path\\file")
        assert '\\\\' in result

    def test_schema_with_dot_and_special_chars(self):
        # Real-world: "my_schema.table+extra" must be a literal match
        result = escape_promql_regex_literal("my_schema.table+extra")
        assert "\\." in result
        assert "\\+" in result
        assert "my_schema" in result
        assert "table" in result


class TestExecuteQuerySQLAllowlist:
    """Tests for the SQL allowlist on the /execute-query debug endpoint."""

    def _setup_debug(self, app_module):
        import os
        os.environ['ENABLE_DEBUG'] = 'true'
        app_module._DEBUG_SECRET_KEY = 'test-secret'

    def _teardown_debug(self, app_module):
        import os
        os.environ.pop('ENABLE_DEBUG', None)
        app_module._DEBUG_SECRET_KEY = ''

    def test_auth_required_without_header(self, client):
        """Endpoint returns 401/403 when Authorization header is absent."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'SELECT 1'},
                                   content_type='application/json')
            assert response.status_code in (401, 403)
        finally:
            self._teardown_debug(app_module)

    def test_auth_required_with_wrong_token(self, client):
        """Endpoint returns 401/403 for a wrong Bearer token."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'SELECT 1'},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer wrong-token'})
            assert response.status_code in (401, 403)
        finally:
            self._teardown_debug(app_module)

    def test_select_allowlisted(self, client):
        """Valid SELECT query is accepted (reaches the DB layer, not rejected by allowlist)."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            with patch('app.psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [(1,)]
                mock_cursor.description = [('?column?',)]
                mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = client.post('/execute-query',
                                       json={'query': 'SELECT 1'},
                                       content_type='application/json',
                                       headers={'Authorization': 'Bearer test-secret'})
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'error' not in data or 'permitted' not in data.get('error', '')
        finally:
            self._teardown_debug(app_module)

    def test_explain_allowlisted(self, client):
        """EXPLAIN query is accepted by the allowlist."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            with patch('app.psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [('Seq Scan on t',)]
                mock_cursor.description = [('QUERY PLAN',)]
                mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = client.post('/execute-query',
                                       json={'query': 'EXPLAIN SELECT 1'},
                                       content_type='application/json',
                                       headers={'Authorization': 'Bearer test-secret'})
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'error' not in data or 'permitted' not in data.get('error', '')
        finally:
            self._teardown_debug(app_module)

    def test_drop_blocked(self, client):
        """DROP statement is rejected by the SQL allowlist."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'DROP TABLE users'},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'permitted' in data['error'].lower() or 'only' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_delete_blocked(self, client):
        """DELETE statement is rejected by the SQL allowlist."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'DELETE FROM users WHERE id = 1'},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'permitted' in data['error'].lower() or 'only' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_insert_blocked(self, client):
        """INSERT statement is rejected by the SQL allowlist."""
        import os
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "INSERT INTO users VALUES (1, 'x')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'permitted' in data['error'].lower() or 'only' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)


class TestExecuteQuerySQLBlocklist:
    """Tests for the secondary SQL blocklist on the /execute-query debug endpoint."""

    def _setup_debug(self, app_module):
        import os
        os.environ['ENABLE_DEBUG'] = 'true'
        app_module._DEBUG_SECRET_KEY = 'test-secret'

    def _teardown_debug(self, app_module):
        import os
        os.environ.pop('ENABLE_DEBUG', None)
        app_module._DEBUG_SECRET_KEY = ''

    def test_pg_read_file_blocked(self, client):
        """SELECT containing pg_read_file() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT pg_read_file('/etc/passwd')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'disallowed' in data['error'].lower() or 'pg_read_file' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_pg_shadow_blocked(self, client):
        """SELECT from pg_shadow is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'SELECT usename, passwd FROM pg_shadow'},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_pg_authid_blocked(self, client):
        """SELECT from pg_authid is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': 'SELECT rolname, rolpassword FROM pg_authid'},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_pg_ls_dir_blocked(self, client):
        """pg_ls_dir() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT * FROM pg_ls_dir('.')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_pg_stat_file_blocked(self, client):
        """pg_stat_file() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT * FROM pg_stat_file('pg_hba.conf')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_pg_read_binary_file_blocked(self, client):
        """pg_read_binary_file() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT pg_read_binary_file('pg_hba.conf')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_copy_blocked(self, client):
        """COPY statement is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "COPY users TO '/tmp/dump.csv'"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_do_blocked(self, client):
        """DO anonymous block is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "DO $$ BEGIN RAISE NOTICE 'x'; END $$"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_call_blocked(self, client):
        """CALL statement is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "CALL my_procedure()"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_dblink_blocked(self, client):
        """dblink() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'disallowed' in data['error'].lower() or 'dblink' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_lo_export_blocked(self, client):
        """lo_export() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT lo_export(1234, '/tmp/out')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'disallowed' in data['error'].lower() or 'lo_export' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_lo_import_blocked(self, client):
        """lo_import() is rejected by the secondary blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post('/execute-query',
                                   json={'query': "SELECT lo_import('/etc/passwd')"},
                                   content_type='application/json',
                                   headers={'Authorization': 'Bearer test-secret'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'disallowed' in data['error'].lower() or 'lo_import' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_writable_cte_blocked(self, client):
        """WITH clause is rejected to prevent writable CTEs."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': "WITH d AS (DELETE FROM users RETURNING id) SELECT id FROM d"},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_select_cte_also_blocked(self, client):
        """Even read-only CTEs are blocked (conservative policy: no WITH at all)."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': "WITH cte AS (SELECT 1) SELECT * FROM cte"},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
        finally:
            self._teardown_debug(app_module)

    # --- Block comment bypass tests ---

    def test_block_comment_before_drop_still_blocked(self, client):
        """Leading block comment does not let DROP bypass the allowlist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': '/* hello */ DROP TABLE users'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_block_comment_before_select_allowed(self, client):
        """Leading block comment before SELECT is accepted (comment stripped first)."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            with patch('app.psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [(1,)]
                mock_cursor.description = [('?column?',)]
                mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = client.post(
                    '/execute-query',
                    json={'query': '/* comment */ SELECT 1'},
                    content_type='application/json',
                    headers={'Authorization': 'Bearer test-secret'},
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'permitted' not in data.get('error', '')
        finally:
            self._teardown_debug(app_module)

    def test_inline_comment_containing_with_allowed(self, client):
        """Inline block comment containing WITH keyword is not treated as a real CTE."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            with patch('app.psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [(1,)]
                mock_cursor.description = [('?column?',)]
                mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = client.post(
                    '/execute-query',
                    json={'query': 'SELECT /* WITH DELETE */ 1'},
                    content_type='application/json',
                    headers={'Authorization': 'Bearer test-secret'},
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'disallowed' not in data.get('error', '')
        finally:
            self._teardown_debug(app_module)

    # --- Allowlist bypass tests ---

    def test_explain_with_pg_shadow_blocked(self, client):
        """EXPLAIN does not bypass the blocklist — pg_shadow is still rejected."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': 'EXPLAIN SELECT * FROM pg_shadow'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'disallowed' in data['error'].lower() or 'pg_shadow' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_pg_shadow_mixed_case_blocked(self, client):
        """pg_shadow in mixed/upper case is still caught by the case-insensitive blocklist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': 'SELECT * FROM PG_SHADOW'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
        finally:
            self._teardown_debug(app_module)

    def test_multi_statement_blocked(self, client):
        """Multi-statement SQL (semicolon-separated) is rejected."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            response = client.post(
                '/execute-query',
                json={'query': 'SELECT 1; DELETE FROM users'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'multi' in data['error'].lower() or 'statement' in data['error'].lower()
        finally:
            self._teardown_debug(app_module)

    def test_explain_format_json_allowed(self, client):
        """EXPLAIN (FORMAT JSON) with option clause is accepted by the allowlist."""
        import app as app_module
        self._setup_debug(app_module)
        try:
            with patch('app.psycopg2.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [('[{"Plan":{}}]',)]
                mock_cursor.description = [('QUERY PLAN',)]
                mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = client.post(
                    '/execute-query',
                    json={'query': 'EXPLAIN (FORMAT JSON) SELECT 1'},
                    content_type='application/json',
                    headers={'Authorization': 'Bearer test-secret'},
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'permitted' not in data.get('error', '')
        finally:
            self._teardown_debug(app_module)


class TestQueryTextsTimeout:
    """Tests for /query_texts timeout error path."""

    @patch('app.psycopg2.connect')
    def test_operationalerror_timeout_returns_empty_list(self, mock_connect, client):
        """OperationalError (e.g. connection timeout) is handled gracefully."""
        mock_connect.side_effect = psycopg2.OperationalError("connection timed out")

        response = client.get('/query_texts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert data == []

    @patch('app.psycopg2.connect')
    def test_statement_timeout_returns_empty_list(self, mock_connect, client):
        """Statement timeout OperationalError returns empty list, not 500."""
        mock_connect.side_effect = psycopg2.OperationalError(
            "ERROR:  canceling statement due to statement timeout"
        )

        response = client.get('/query_texts')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestQueryInfoMetricsTimeout:
    """Tests for /query_info_metrics timeout error path."""

    @patch('app.psycopg2.connect')
    def test_operationalerror_returns_200_with_header(self, mock_connect, client):
        """OperationalError from psycopg2 still returns Prometheus header lines."""
        mock_connect.side_effect = psycopg2.OperationalError("connection timed out")

        response = client.get('/query_info_metrics')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        assert '# HELP pgwatch_query_info' in data
        assert '# TYPE pgwatch_query_info gauge' in data

    @patch('app.psycopg2.connect')
    def test_statement_timeout_returns_empty_metrics(self, mock_connect, client):
        """Statement timeout still returns a valid (empty) Prometheus response."""
        mock_connect.side_effect = psycopg2.OperationalError(
            "ERROR:  canceling statement due to statement timeout"
        )

        response = client.get('/query_info_metrics')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        # Should have header but no metric lines (no rows fetched)
        assert '# HELP pgwatch_query_info' in data
        assert 'pgwatch_query_info{' not in data


# ---------------------------------------------------------------------------
# Additional coverage for /execute-query (issues 4, 6, 7, 8)
# ---------------------------------------------------------------------------

class TestExecuteQuerySuccessPath:
    """Issue 4 — success-path test: valid SELECT returning real results."""

    def test_returns_columns_and_rows(self, client, debug_mode):
        """A valid SELECT query returns 200 with populated columns and rows."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('alice', 1), ('bob', 2)]
            mock_cursor.description = [('name',), ('score',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': 'SELECT name, score FROM leaderboard'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['columns'] == ['name', 'score']
            assert data['rows'] == [['alice', 1], ['bob', 2]]


class TestExecuteQueryAuthSchemes:
    """Issue 6 — non-Bearer auth schemes (Basic, Token) must be rejected."""

    def test_basic_auth_rejected(self, client, debug_mode):
        """Authorization: Basic <token> is not accepted (only Bearer is valid)."""
        import base64
        creds = base64.b64encode(b'test-secret:').decode()
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
            headers={'Authorization': f'Basic {creds}'},
        )
        assert response.status_code in (401, 403)

    def test_token_auth_rejected(self, client, debug_mode):
        """Authorization: Token <token> is not accepted (only Bearer is valid)."""
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
            headers={'Authorization': 'Token test-secret'},
        )
        assert response.status_code in (401, 403)

    def test_apikey_scheme_rejected(self, client, debug_mode):
        """Authorization: ApiKey <token> is not accepted."""
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
            headers={'Authorization': 'ApiKey test-secret'},
        )
        assert response.status_code in (401, 403)


class TestExecuteQueryMultipleBlockComments:
    """Issue 7 — multiple sequential leading block comments before SELECT."""

    def test_two_sequential_block_comments_accepted(self, client, debug_mode):
        """/* a */ /* b */ SELECT 1 passes the allowlist (both comments stripped)."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [('?column?',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': '/* a */ /* b */ SELECT 1'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'permitted' not in data.get('error', '')

    def test_three_sequential_block_comments_accepted(self, client, debug_mode):
        """Three leading block comments before SELECT are all stripped correctly."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(42,)]
            mock_cursor.description = [('?column?',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': '/* x */ /* y */ /* z */ SELECT 42'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200


class TestExecuteQueryStatementTimeout:
    """Issue 8 — statement timeout during query execution returns 500."""

    def test_statement_timeout_returns_500(self, client, debug_mode):
        """OperationalError from statement_timeout inside execute-query returns 500."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = psycopg2.OperationalError(
                "ERROR:  canceling statement due to statement timeout"
            )
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': 'SELECT pg_sleep(999)'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data

    def test_line_comment_containing_blocked_keyword_not_blocked(self, client, debug_mode):
        """Issue 2 regression: '-- pg_read_file' in a line comment must not trigger blocklist."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [('?column?',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': "SELECT 1 -- pg_read_file('/etc/passwd')"},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200, (
                "A blocked keyword inside a line comment should NOT be rejected by the blocklist"
            )

    def test_query_with_double_dash_in_string_literal(self, client, debug_mode):
        """SELECT with -- inside a string literal must not be rejected by the blocklist."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('foo--bar',)]
            mock_cursor.description = [('?column?',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': "SELECT 'foo--bar'"},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200, (
                "A -- inside a string literal should NOT be treated as a comment by the blocklist"
            )


class TestDebugAuth:
    """Unit tests for _check_debug_auth() via the /execute-query endpoint."""

    def test_valid_token_allows_request(self, client, debug_mode):
        """Valid Bearer token returns None (auth passes, request proceeds)."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [('?column?',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': 'SELECT 1'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200

    def test_wrong_token_returns_401(self, client, debug_mode):
        """Wrong Bearer token is rejected with 401."""
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
            headers={'Authorization': 'Bearer wrong-secret'},
        )
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data

    def test_missing_header_returns_401(self, client, debug_mode):
        """Missing Authorization header is rejected with 401."""
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
        )
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data

    def test_non_bearer_scheme_returns_401(self, client, debug_mode):
        """Non-Bearer auth scheme (e.g. Basic) is rejected with 401."""
        response = client.post(
            '/execute-query',
            json={'query': 'SELECT 1'},
            content_type='application/json',
            headers={'Authorization': 'Basic dXNlcjpwYXNz'},
        )
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data


class TestExecuteQuerySessionSetup:
    """Verify connection session configuration in /execute-query."""

    def test_set_session_readonly_called(self, client, debug_mode):
        """set_session(autocommit=True, readonly=True) must be called on the connection."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [('?column?',)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            response = client.post(
                '/execute-query',
                json={'query': 'SELECT 1'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200
            mock_conn.set_session.assert_called_once_with(autocommit=True, readonly=True)

    def test_set_statement_timeout_called(self, client, debug_mode):
        """cursor.execute must be called with SET statement_timeout before the user query."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [('?column?',)]
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            response = client.post(
                '/execute-query',
                json={'query': 'SELECT 1'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200
            calls = mock_cursor.execute.call_args_list
            timeout_call = calls[0]
            assert timeout_call == call('SET statement_timeout = %s', ('10s',)), (
                "First cursor.execute call must set statement_timeout to 10s"
            )


class TestExecuteQueryShowAllowed:
    """SHOW commands must be accepted by the SQL allowlist."""

    def test_show_command_allowed(self, client, debug_mode):
        """SHOW statement_timeout passes the allowlist and executes successfully."""
        with patch('app.psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('10s',)]
            mock_cursor.description = [('statement_timeout',)]
            mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            response = client.post(
                '/execute-query',
                json={'query': 'SHOW statement_timeout'},
                content_type='application/json',
                headers={'Authorization': 'Bearer test-secret'},
            )
            assert response.status_code == 200, (
                "SHOW command should be allowed by the SQL allowlist"
            )
            data = json.loads(response.data)
            assert data['columns'] == ['statement_timeout']
            assert data['rows'] == [['10s']]


class _SinkFake:
    """
    Behavioral in-memory fake for ``public.pgss_queryid_queries``.

    Stores rows tagged with both ``partition_dbname`` (the pgwatch source
    name held by the partition column) and ``real_dbname`` (the actual
    Postgres database name embedded in the JSONB payload). When the
    backend issues a SELECT, the fake parses the SQL's WHERE clause for
    the predicates this MR exercises:

      - ``dbname = %s`` (the buggy partition-column filter — matched with
        a word boundary so it does not collide with ``real_dbname``)
      - ``data->>'real_dbname' = %s`` (the corrected JSONB filter)
      - ``data ? 'real_dbname'`` (explicit absent-key guard)

    Rows that satisfy ALL applied predicates are sorted by recency, then
    deduped by queryid (DISTINCT ON). When the SQL surfaces the
    ``source_dbnames`` column, each result row also carries the set of
    partition_dbnames that hold a row for the same queryid — used by
    callers to detect cross-source ambiguity.

    This fake is intentionally *not* a SQL string sniffer. Reintroducing
    the buggy ``dbname = %s`` filter would still execute against the
    fake; it would simply return zero rows because no real
    ``partition_dbname`` matches the dashboard's ``datname`` value. The
    test then fails — as it should.
    """

    def __init__(self):
        self._rows = []

    def add(self, *, partition_dbname, real_dbname, queryid, query,
            time_age_sec=0, real_dbname_present=True):
        self._rows.append({
            'partition_dbname': partition_dbname,
            'real_dbname': real_dbname if real_dbname_present else None,
            'real_dbname_present': real_dbname_present,
            'queryid': queryid,
            'query': query,
            'time_age_sec': time_age_sec,
        })

    def _bind_predicates(self, sql_lower, params):
        """
        Walk through %s placeholders in SQL order, classifying each by the
        preceding text and pulling the corresponding value from params.
        """
        bindings = {}
        param_iter = iter(params)
        for m in re.finditer(r'%s', sql_lower):
            start = max(0, m.start() - 80)
            preceding = sql_lower[start:m.start()]
            try:
                val = next(param_iter)
            except StopIteration:
                break
            if "data->>'real_dbname'" in preceding:
                bindings['real_dbname'] = val
            elif re.search(r"(?<!\w)dbname\s*=\s*$", preceding):
                bindings['partition_dbname'] = val
        return bindings

    def run(self, sql, params):
        sql_lower = (sql or '').lower()
        if 'pgss_queryid_queries' not in sql_lower:
            return []
        params = tuple(params or ())
        bindings = self._bind_predicates(sql_lower, params)
        require_real_dbname_present = "data ? 'real_dbname'" in sql_lower

        matched = []
        for row in self._rows:
            if require_real_dbname_present and not row['real_dbname_present']:
                continue
            if 'real_dbname' in bindings and row['real_dbname'] != bindings['real_dbname']:
                continue
            if 'partition_dbname' in bindings and row['partition_dbname'] != bindings['partition_dbname']:
                continue
            matched.append(row)

        # DISTINCT ON (queryid) ORDER BY time DESC. ``time_age_sec`` is the
        # age-since-now in seconds — smaller means more recent.
        matched.sort(key=lambda r: r['time_age_sec'])
        want_sources = 'source_dbnames' in sql_lower
        sources_by_queryid = {}
        if want_sources:
            for row in matched:
                sources_by_queryid.setdefault(row['queryid'], set()).add(row['partition_dbname'])

        seen = set()
        results = []
        for row in matched:
            if row['queryid'] in seen:
                continue
            seen.add(row['queryid'])
            out = {'queryid': row['queryid'], 'query': row['query']}
            if want_sources:
                out['source_dbnames'] = sorted(sources_by_queryid.get(row['queryid'], set()))
            results.append(out)
        return results


class _SinkFakeCursor:
    """Cursor-shaped wrapper that delegates execute() to a _SinkFake."""

    def __init__(self, fake):
        self._fake = fake
        self._rows = []
        self.execute = MagicMock(side_effect=self._execute)
        self.rowcount = 0

    def _execute(self, sql, params=()):
        self._rows = self._fake.run(sql, params)
        return None

    def __iter__(self):
        return iter(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wire_sink_connect(mock_connect, cursor):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.autocommit = True
    mock_connect.return_value = conn
    return conn


class TestPgssMetricsCsvQueryTextLookup:
    """
    Regression coverage for the ``query_text`` column on Dashboard 02's
    "Detailed table view (pg_stat_statements)" panel.

    The panel calls ``/pgss_metrics/csv?db_name=<datname>`` (Grafana's
    ``$db_name`` variable resolves to the Prometheus ``datname`` label —
    the actual Postgres database name). The Flask backend joins the
    Prometheus pgss metrics with query texts read from the sink Postgres
    table ``public.pgss_queryid_queries``.

    The sink table is partitioned by ``dbname`` — that column holds the
    pgwatch *source name* (e.g. ``my-host-cone-demo``), NOT the actual
    Postgres database name. The actual database name is stored inside
    the JSONB payload as ``data->>'real_dbname'``. Filtering by the
    partition ``dbname`` column with the dashboard's ``datname`` value
    never matches, so the resulting CSV has empty ``query_text`` for
    every row.
    """

    def _build_prom_sample(self, datname, queryid, metric_name, value, ts):
        return {
            'metric': {
                '__name__': metric_name,
                'datname': datname,
                'queryid': queryid,
                'user': 'postgres',
                'instance': 'pgwatch-prometheus:9091',
            },
            'values': [[ts, str(value)]],
        }

    @patch('app.PrometheusConnect')
    @patch('app.psycopg2.connect')
    def test_query_text_populated_when_dashboard_passes_datname(
        self, mock_pg_connect, mock_prom_cls, client
    ):
        """
        With ``db_name=<datname>`` (what Dashboard 02 sends), every CSV row
        must carry the seeded ``query_text`` for its ``queryid``.
        """
        import csv as csv_module
        import io as io_module

        datname = 'cone_demo'
        partition_dbname = 'my-host-cone-demo'  # pgwatch source name
        queryid = '-364883030418038032'
        query_text = 'WITH all_ledger AS (SELECT account_id FROM demo.ledger) SELECT 1'

        ts_end = 1779458600.0
        ts_start = ts_end - 60

        prom_instance = mock_prom_cls.return_value

        def range_data(metric_name, start_time, end_time):
            base_name = metric_name.split('{', 1)[0]
            if base_name != 'pgwatch_pg_stat_statements_calls':
                return []
            sample_ts = start_time.timestamp()
            value = 100 if sample_ts < ts_end - 30 else 250
            return [
                self._build_prom_sample(datname, queryid, base_name, value, sample_ts)
            ]
        prom_instance.get_metric_range_data.side_effect = range_data

        fake = _SinkFake()
        fake.add(
            partition_dbname=partition_dbname,
            real_dbname=datname,
            queryid=queryid,
            query=query_text,
        )
        sink_cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_pg_connect, sink_cursor)

        response = client.get(
            f'/pgss_metrics/csv?time_start={ts_start}&time_end={ts_end}&db_name={datname}'
        )

        assert response.status_code == 200, response.data
        rows = list(csv_module.DictReader(io_module.StringIO(response.data.decode())))
        assert rows, 'expected at least one CSV row for the seeded queryid'
        target = next((r for r in rows if r['queryid'] == queryid), None)
        assert target is not None, (
            f"queryid {queryid!r} missing from CSV; got: {[r['queryid'] for r in rows]}"
        )
        assert target['query_text'], (
            "query_text column was empty even though the sink has the "
            "queryid->text mapping under data->>'real_dbname' = datname"
        )
        # The truncator may shorten the text but the table identifier must survive.
        assert 'demo.ledger' in target['query_text'], (
            f"query_text does not reflect the seeded query: {target['query_text']!r}"
        )

        # Behavioral cross-check: the production SQL must have filtered on
        # the JSONB real_dbname payload, not the partition dbname column.
        sink_sqls = [c.args[0] for c in sink_cursor.execute.call_args_list]
        assert any("data->>'real_dbname' = %s" in sql for sql in sink_sqls), sink_sqls
        for sql in sink_sqls:
            # Word-boundary check so this does not match ``real_dbname``.
            assert re.search(r"(?<!\w)dbname\s*=\s*%s", sql) is None, sql

    @patch('app.psycopg2.connect')
    def test_get_query_texts_from_sink_filters_by_real_dbname(self, mock_connect):
        """
        Direct sink lookup must filter by ``data->>'real_dbname'`` (the
        actual Postgres database name from pgwatch) — NOT by the partition
        ``dbname`` column (which holds the pgwatch source name).
        """
        fake = _SinkFake()
        cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_connect, cursor)

        app_module.get_query_texts_from_sink(db_name='cone_demo', max_age_hours=48)

        query, params = cursor.execute.call_args[0]
        assert "data->>'real_dbname' = %s" in query, (
            "sink lookup must filter on data->>'real_dbname' so the dashboard's "
            "datname matches the JSONB payload, not the pgwatch source name. "
            f"Got SQL: {query}"
        )
        # Word-boundary check: the buggy ``dbname = %s`` filter (matching the
        # partition column) must be gone. A simple ``'dbname = %s' not in``
        # would slip past ``real_dbname = %s``; this assertion catches both.
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", query) is None, (
            "partition `dbname = %s` filter must not be present; got: " + query
        )
        assert params[0] == 'cone_demo'

    @patch('app.psycopg2.connect')
    def test_get_query_texts_from_sink_returns_text_for_matching_datname(self, mock_connect):
        """
        End-to-end behavioral check on the sink helper: the seeded query
        text must be returned when ``db_name`` matches the row's
        ``real_dbname`` even though the partition ``dbname`` is the
        distinct pgwatch source name.
        """
        fake = _SinkFake()
        fake.add(
            partition_dbname='my-host-cone-demo',
            real_dbname='cone_demo',
            queryid='qid-1',
            query='select 1 from demo.accounts',
        )
        cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_connect, cursor)

        result = app_module.get_query_texts_from_sink(db_name='cone_demo', max_age_hours=48)
        assert 'qid-1' in result, result
        assert 'demo.accounts' in result['qid-1']

    @patch('app.psycopg2.connect')
    def test_get_query_texts_from_sink_drops_rows_with_absent_real_dbname(self, mock_connect):
        """
        Rows missing ``real_dbname`` in their JSONB payload (older pgwatch
        versions, or non-pgss measurements) must be skipped when a
        ``db_name`` filter is applied. The SQL guards this with
        ``data ? 'real_dbname'`` so the behavior is explicit, not a
        side effect of ``data->>'real_dbname' = %s`` returning NULL.
        """
        fake = _SinkFake()
        fake.add(
            partition_dbname='legacy-source',
            real_dbname='cone_demo',
            queryid='qid-1',
            query='select 1',
            real_dbname_present=False,
        )
        cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_connect, cursor)

        result = app_module.get_query_texts_from_sink(db_name='cone_demo', max_age_hours=48)
        assert result == {}, result

    @patch('app.psycopg2.connect')
    def test_get_query_texts_from_sink_warns_on_cross_source_duplicates(self, mock_connect, caplog):
        """
        Two distinct pgwatch sources reporting the same ``real_dbname``
        produce a non-deterministic ``DISTINCT ON`` winner. The helper
        must log a single warning so operators have a signal that query
        texts are aggregating across sources (potential tenant leak).
        """
        app_module._logged_cross_source_warnings.clear()
        fake = _SinkFake()
        # Same queryid, same real_dbname, but two distinct source partitions.
        fake.add(
            partition_dbname='source-a',
            real_dbname='shared_db',
            queryid='qid-7',
            query='select 1 -- from a',
            time_age_sec=10,
        )
        fake.add(
            partition_dbname='source-b',
            real_dbname='shared_db',
            queryid='qid-7',
            query='select 2 -- from b',
            time_age_sec=5,  # newer, will win DISTINCT ON
        )
        cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_connect, cursor)

        with caplog.at_level('WARNING', logger='app'):
            result = app_module.get_query_texts_from_sink(db_name='shared_db', max_age_hours=48)

        assert 'qid-7' in result
        # Either "from a" or "from b" wins by time; we don't care which here.
        assert any('cross-source ambiguity' in r.message.lower() for r in caplog.records), [
            r.message for r in caplog.records
        ]

    @patch('app.psycopg2.connect')
    def test_get_query_texts_from_sink_skips_filter_when_db_name_is_all(self, mock_connect):
        """
        ``db_name='All'`` must skip the JSONB filter entirely — rows from
        every datname should flow through.
        """
        fake = _SinkFake()
        fake.add(
            partition_dbname='source-a',
            real_dbname='db_one',
            queryid='qid-A',
            query='select 1',
        )
        fake.add(
            partition_dbname='source-b',
            real_dbname='db_two',
            queryid='qid-B',
            query='select 2',
        )
        cursor = _SinkFakeCursor(fake)
        _wire_sink_connect(mock_connect, cursor)

        result = app_module.get_query_texts_from_sink(db_name='All', max_age_hours=48)
        assert set(result.keys()) == {'qid-A', 'qid-B'}, result

        sql = cursor.execute.call_args[0][0]
        assert "data->>'real_dbname'" not in sql
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", sql) is None, sql


class TestQueryTextLookupSqlShape:
    """
    SQL-shape regression coverage for ``_build_query_text_lookup_sql``.

    The behavioral tests above use ``_SinkFake``, which parses predicates
    but never executes SQL. That means a syntactically valid but
    semantically unsupported construct (e.g. ``array_agg(DISTINCT ...) OVER (...)``,
    which PostgreSQL rejects with ``FeatureNotSupported``) would slip
    past every fake-backed test and still ship to production — which is
    exactly the regression QA caught on iteration 2 of MR !271.

    These tests pin known-bad shapes that must not reappear in the
    emitted SQL.
    """

    def test_emitted_sql_does_not_use_distinct_in_window_aggregate(self):
        """
        PostgreSQL does not support ``DISTINCT`` inside window aggregates
        (``array_agg(DISTINCT x) OVER (...)`` raises ``FeatureNotSupported``).
        The MR !271 fix originally emitted exactly this pattern, which
        meant the filtered code path always errored out and the
        dashboard's query_text column stayed empty.
        """
        for with_db_filter in (True, False):
            sql = app_module._build_query_text_lookup_sql(
                with_db_filter=with_db_filter,
                time_clause='time > now() - make_interval(hours => %s)',
            )
            # Match ``array_agg ( distinct ... ) over`` allowing whitespace
            # and any column expression between the parens. Multiline because
            # the SQL is line-wrapped.
            assert re.search(
                r"array_agg\s*\(\s*distinct\b.*?\)\s*over\b",
                sql,
                flags=re.IGNORECASE | re.DOTALL,
            ) is None, (
                "PostgreSQL rejects DISTINCT inside window aggregates. "
                "Use a CTE + GROUP BY instead. SQL was: " + sql
            )

    def test_filtered_sql_still_surfaces_source_dbnames(self):
        """
        The cross-source ambiguity warning (an MR !271 review request)
        relies on the filtered path emitting a ``source_dbnames`` column.
        Pin its presence so a future refactor that drops it (and thereby
        silently disables the warning) fails loudly here.
        """
        sql = app_module._build_query_text_lookup_sql(
            with_db_filter=True,
            time_clause='time > now() - make_interval(hours => %s)',
        )
        assert 'source_dbnames' in sql, sql


@pytest.mark.skipif(
    not os.environ.get('PGSS_SINK_TEST_URL'),
    reason='Requires PGSS_SINK_TEST_URL pointing at a disposable Postgres for syntax checks',
)
class TestQueryTextLookupSqlExecutesOnRealPostgres:
    """
    Integration guard: parse-on-Postgres each emitted SQL via ``PREPARE``.

    Behavioral tests use a fake, so a syntactically-valid-but-semantically-
    unsupported construct (the MR !271 iteration-1 regression where
    ``array_agg(DISTINCT ...) OVER (...)`` raised ``FeatureNotSupported``
    at runtime) was invisible. This test runs each emitted SQL through
    ``PREPARE`` on a real Postgres so the planner validates it without
    needing any data.

    Gated on ``PGSS_SINK_TEST_URL`` so contributors without a local
    Postgres can still run the suite; CI can opt-in by setting the env.
    """

    @pytest.fixture
    def sink_conn(self):
        url = os.environ['PGSS_SINK_TEST_URL']
        conn = psycopg2.connect(url, connect_timeout=10)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                create table if not exists public.pgss_queryid_queries (
                    dbname text not null,
                    time timestamptz not null default now(),
                    data jsonb not null
                ) partition by list (dbname);
                create table if not exists public.pgss_queryid_queries_default
                    partition of public.pgss_queryid_queries default;
            """)
        try:
            yield conn
        finally:
            conn.close()

    @pytest.mark.parametrize('with_db_filter', [True, False])
    def test_emitted_sql_prepares_on_real_postgres(self, sink_conn, with_db_filter):
        sql = app_module._build_query_text_lookup_sql(
            with_db_filter=with_db_filter,
            time_clause='time > now() - make_interval(hours => %s)',
        )
        # PREPARE wants $N placeholders; psycopg2's %s won't work here. We
        # exercise the same SQL via a no-op execution that the planner
        # still parses end-to-end: wrap as a subquery in EXPLAIN.
        if with_db_filter:
            params = ('cone_demo', 24)
        else:
            params = (24,)
        with sink_conn.cursor() as cur:
            cur.execute('explain ' + sql, params)
            cur.fetchall()


class TestQueryTextsRouteFilterShape:
    """
    Mirror coverage for the ``/query_texts`` route: the SQL change in
    ``get_query_texts()`` (separate code path from
    ``get_query_texts_from_sink``) also needs the three filter-shape
    cases pinned and the buggy partition predicate locked out.
    """

    @patch('app.psycopg2.connect')
    def test_db_name_filter_uses_real_dbname_payload(self, mock_connect, client):
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_texts?db_name=mydb')
        assert response.status_code == 200

        sql, params = mock_cursor.execute.call_args[0]
        assert "data->>'real_dbname' = %s" in sql, sql
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", sql) is None, sql
        # Unbounded scans bring the lookup down at scale — the route must
        # carry a time predicate too (one of the HIGH findings on MR !271).
        assert 'make_interval' in sql, sql
        # First %s binding is db_name; the second is the retention window.
        assert params[0] == 'mydb'
        assert params[1] == app_module.QUERYID_RETENTION_HOURS

    @patch('app.psycopg2.connect')
    def test_db_name_all_skips_filter(self, mock_connect, client):
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_texts?db_name=All')
        assert response.status_code == 200

        sql = mock_cursor.execute.call_args[0][0]
        assert "data->>'real_dbname'" not in sql, sql
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", sql) is None, sql
        assert 'make_interval' in sql, sql

    @patch('app.psycopg2.connect')
    def test_db_name_unresolved_variable_skips_filter(self, mock_connect, client):
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_texts?db_name=$db_name')
        assert response.status_code == 200

        sql = mock_cursor.execute.call_args[0][0]
        assert "data->>'real_dbname'" not in sql, sql
        assert re.search(r"(?<!\w)dbname\s*=\s*%s", sql) is None, sql
        assert 'make_interval' in sql, sql
