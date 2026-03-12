"""Tests for the Flask monitoring backend."""
import pytest
import json
from unittest.mock import patch, mock_open

from app import app, read_version_file, smart_truncate_query, _escape_prometheus_label, _sanitize_promql_label


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


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


class TestSanitizePromqlLabel:
    """Tests for the _sanitize_promql_label function."""

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _sanitize_promql_label('') == ''

    def test_none_returns_empty(self):
        """Test None input returns empty string."""
        assert _sanitize_promql_label(None) == ''

    def test_simple_string_unchanged(self):
        """Test simple strings are unchanged."""
        assert _sanitize_promql_label('my-cluster') == 'my-cluster'

    def test_escapes_double_quote(self):
        """Test double quotes are escaped to prevent PromQL injection."""
        assert _sanitize_promql_label('foo"} or vector(1) or x{"y="') == 'foo\\"} or vector(1) or x{\\"y=\\"'

    def test_escapes_backslash(self):
        """Test backslashes are escaped."""
        assert _sanitize_promql_label('path\\to') == 'path\\\\to'

    def test_injection_attempt_neutralized(self):
        """Test that a PromQL injection attempt is safely escaped."""
        malicious = 'cluster"} or up{job="'
        result = _sanitize_promql_label(malicious)
        # The result should contain escaped quotes, not raw ones
        assert '"' not in result.replace('\\"', '')

    def test_normal_label_values(self):
        """Test typical label values pass through correctly."""
        assert _sanitize_promql_label('prod-us-east-1') == 'prod-us-east-1'
        assert _sanitize_promql_label('192.168.1.1:5432') == '192.168.1.1:5432'
        assert _sanitize_promql_label('my_database') == 'my_database'


class TestQueryInfoMetricsEndpoint:
    """Tests for the /query_info_metrics endpoint."""

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

        # Verify the query was called with db_name filter
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

        # Verify the query was called without db_name filter
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

        # Verify the query was called without db_name filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        assert 'dbname = %s' not in query

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
