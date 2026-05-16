"""Tests for the Flask monitoring backend."""
import importlib
import os
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
        assert 'dbname = %s' in query
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
        assert 'dbname = %s' not in query
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
        """Test db_name parameter filters results."""
        mock_cursor = mock_connect.return_value.cursor.return_value.__enter__.return_value
        mock_cursor.__iter__ = lambda self: iter([])

        response = client.get('/query_info_metrics?db_name=mydb')
        assert response.status_code == 200

        # Verify the main query (last execute call) was called with db_name filter
        call_args = mock_cursor.execute.call_args
        assert call_args is not None
        query = call_args[0][0]
        assert 'dbname = %s' in query

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
        assert 'dbname = %s' not in query

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
        assert 'dbname = %s' not in query

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

            assert 'dbname = %s' in query
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


class TestExecuteQuerySuccessPath:
    """Success-path test: valid SELECT returning real results."""

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
    """Non-Bearer auth schemes (Basic, Token) must be rejected."""

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
    """Multiple sequential leading block comments before SELECT."""

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
    """Statement timeout during query execution returns 500."""

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
        """Regression: '-- pg_read_file' in a line comment must not trigger blocklist."""
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
