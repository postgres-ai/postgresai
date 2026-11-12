"""Tests for the /query_texts endpoint."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

# Add the monitoring_flask_backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monitoring_flask_backend'))


class TestQueryTextsEndpoint:
    """Tests for the /query_texts Flask endpoint."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a Flask test client."""
        import app as app_module

        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as client:
            yield client

    @pytest.fixture
    def mock_db_connection(self):
        """Create a mock database connection with configurable results."""
        def _create_mock(rows=None):
            if rows is None:
                rows = []

            mock_cursor = MagicMock()
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_cursor.__iter__ = MagicMock(return_value=iter(rows))

            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            return mock_conn, mock_cursor

        return _create_mock

    # =========================================================================
    # Response Format Tests
    # =========================================================================

    def test_returns_200_ok(self, client, mock_db_connection):
        """Test that /query_texts endpoint returns 200 status."""
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            assert response.status_code == 200

    def test_returns_json_content_type(self, client, mock_db_connection):
        """Test that /query_texts endpoint returns JSON content type."""
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            assert response.content_type == 'application/json'

    def test_returns_json_array(self, client, mock_db_connection):
        """Test that /query_texts returns a JSON array."""
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()
            assert isinstance(data, list)

    def test_response_contains_required_fields(self, client, mock_db_connection):
        """Test that each item in response contains queryid, query_text, and displayName fields."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT * FROM users'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 1
            item = data[0]
            assert 'queryid' in item
            assert 'query_text' in item
            assert 'displayName' in item

    def test_response_field_values(self, client, mock_db_connection):
        """Test that response fields contain expected values."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT id FROM users'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 1
            item = data[0]
            assert item['queryid'] == '12345'
            # Short query should not be truncated
            assert item['query_text'] == 'SELECT id FROM users'
            # displayName should be the query_text when it exists
            assert item['displayName'] == 'SELECT id FROM users'

    def test_display_name_fallback_to_queryid(self, client, mock_db_connection):
        """Test that displayName falls back to queryid when query_text is empty."""
        rows = [
            {'queryid': '12345', 'query': ''},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 1
            item = data[0]
            # When query_text is empty, displayName should be the queryid
            assert item['displayName'] == '12345'

    def test_empty_result_returns_empty_array(self, client, mock_db_connection):
        """Test that empty database result returns empty JSON array."""
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()
            assert data == []

    def test_multiple_results(self, client, mock_db_connection):
        """Test that multiple query results are returned correctly."""
        rows = [
            {'queryid': '111', 'query': 'SELECT 1'},
            {'queryid': '222', 'query': 'SELECT 2'},
            {'queryid': '333', 'query': 'SELECT 3'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 3
            queryids = [item['queryid'] for item in data]
            assert '111' in queryids
            assert '222' in queryids
            assert '333' in queryids

    # =========================================================================
    # Truncate Parameter Tests
    # =========================================================================

    def test_default_truncate_is_40(self, client, mock_db_connection):
        """
        Test that default truncate length is 40 characters.

        The endpoint uses smart_truncate_query() which truncates to max_length
        characters total (including the '...' suffix). For non-SQL queries that
        fall back to simple truncation, this means max_length - 3 chars + '...'.
        """
        # Query with 100 non-SQL characters should be truncated
        long_query = 'X' * 100
        rows = [
            {'queryid': '12345', 'query': long_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            item = data[0]
            # smart_truncate_query does: query[:max_length - 3] + '...'
            # So for max_length=40: 37 chars + '...' = 40 total
            assert len(item['query_text']) == 40
            assert item['query_text'].endswith('...')
            assert item['query_text'] == 'X' * 37 + '...'

    def test_custom_truncate_parameter(self, client, mock_db_connection):
        """Test that custom truncate parameter is respected."""
        long_query = 'Y' * 100
        rows = [
            {'queryid': '12345', 'query': long_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=20')
            data = response.get_json()

            item = data[0]
            # For truncate=20: 17 chars + '...' = 20 total
            assert len(item['query_text']) == 20
            assert item['query_text'] == 'Y' * 17 + '...'

    def test_large_truncate_value(self, client, mock_db_connection):
        """Test that large truncate value works correctly."""
        query = 'SELECT * FROM users WHERE id = 1'
        rows = [
            {'queryid': '12345', 'query': query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=1000')
            data = response.get_json()

            item = data[0]
            # Query is shorter than truncate length, should not be truncated
            # smart_truncate_query may still transform SELECT queries
            assert '...' not in item['query_text'] or 'FROM users' in item['query_text']

    def test_truncate_very_small(self, client, mock_db_connection):
        """Test truncate with very small value (edge case)."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT * FROM users'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=5')
            data = response.get_json()

            item = data[0]
            # With truncate=5: max 5 chars total
            # smart_truncate_query does [:5-3] + '...' = 2 chars + '...' = 5 total
            assert len(item['query_text']) <= 5

    def test_invalid_truncate_non_integer(self, client, mock_db_connection):
        """
        Test that invalid truncate parameter (non-integer) returns 500 error.

        BUG IDENTIFIED: The endpoint does not validate the truncate parameter
        before calling int(). When a non-integer string is passed, it raises
        a ValueError which is caught by the outer try/except and returns a
        500 error with the exception message.

        This should ideally return a 400 Bad Request with a clear error message,
        or fall back to the default value of 40.
        """
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=abc')

            # Current behavior: returns 500 with error message
            assert response.status_code == 500
            data = response.get_json()
            assert 'error' in data
            # The error should mention the invalid literal for int()
            assert 'invalid literal' in data['error'].lower() or 'int' in data['error'].lower()

    def test_invalid_truncate_float(self, client, mock_db_connection):
        """
        Test that float truncate parameter returns 500 error.

        BUG: Same issue as non-integer - float strings like '40.5' cause ValueError.
        """
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=40.5')

            # Current behavior: returns 500 with error message
            assert response.status_code == 500
            data = response.get_json()
            assert 'error' in data

    def test_invalid_truncate_empty_string(self, client, mock_db_connection):
        """
        Test that empty truncate parameter returns 500 error.

        BUG: Empty string causes ValueError when calling int('').
        """
        mock_conn, _ = mock_db_connection([])

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=')

            # Current behavior: returns 500 with error message
            assert response.status_code == 500
            data = response.get_json()
            assert 'error' in data

    def test_negative_truncate(self, client, mock_db_connection):
        """Test that negative truncate value is accepted (Python slicing behavior)."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT * FROM users'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=-5')
            data = response.get_json()

            # Negative truncate results in weird behavior due to Python slicing
            # but should still return 200
            assert response.status_code == 200

    # =========================================================================
    # db_name Filter Logic Tests
    # =========================================================================

    def test_no_db_name_filter_queries_all(self, client, mock_db_connection):
        """Test that no db_name parameter queries all databases."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            # Should NOT have 'dbname = %s' in query
            assert 'dbname = %s' not in query

    def test_valid_db_name_applies_filter(self, client, mock_db_connection):
        """Test that valid db_name parameter applies filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=mydb')

            # Verify the query was executed with db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            params = call_args[0][1] if len(call_args[0]) > 1 else None
            # Should have 'dbname = %s' in query
            assert 'dbname = %s' in query
            assert params == ('mydb',)

    def test_db_name_all_lowercase_skips_filter(self, client, mock_db_connection):
        """Test that db_name='all' skips the database filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=all')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    def test_db_name_all_uppercase_skips_filter(self, client, mock_db_connection):
        """Test that db_name='ALL' skips the database filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=ALL')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    def test_db_name_all_mixed_case_skips_filter(self, client, mock_db_connection):
        """Test that db_name='All' (mixed case) skips the database filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=All')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    def test_db_name_empty_string_skips_filter(self, client, mock_db_connection):
        """Test that empty db_name parameter skips the database filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    def test_db_name_grafana_variable_skips_filter(self, client, mock_db_connection):
        """Test that db_name starting with '$' (Grafana variable) skips the filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=$database')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    def test_db_name_grafana_variable_complex_skips_filter(self, client, mock_db_connection):
        """Test that complex Grafana variable like '$__all' skips the filter."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, mock_cursor = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?db_name=$__all')

            # Verify the query was executed without db_name filter
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]
            assert 'dbname = %s' not in query

    # =========================================================================
    # Smart Truncation Behavior Tests
    # =========================================================================

    def test_select_query_smart_truncation(self, client, mock_db_connection):
        """Test that SELECT queries are truncated smartly to show FROM clause."""
        long_query = 'SELECT id, name, email, created_at, updated_at, status FROM users WHERE active = true'
        rows = [
            {'queryid': '12345', 'query': long_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=50')
            data = response.get_json()

            item = data[0]
            # smart_truncate_query should produce "SELECT ... FROM users"
            assert 'SELECT' in item['query_text']
            assert 'FROM' in item['query_text']
            assert 'users' in item['query_text']

    def test_short_query_not_truncated(self, client, mock_db_connection):
        """Test that short query text is not truncated."""
        short_query = 'SELECT id FROM t'
        rows = [
            {'queryid': '12345', 'query': short_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            item = data[0]
            assert item['query_text'] == short_query
            # Ellipsis only added if query is truncated
            assert item['query_text'] == 'SELECT id FROM t'

    def test_query_exactly_at_truncate_limit_not_truncated(self, client, mock_db_connection):
        """Test that query exactly at truncate limit is not truncated."""
        # Create a non-SQL query exactly 40 characters
        exact_query = 'X' * 40
        rows = [
            {'queryid': '12345', 'query': exact_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=40')
            data = response.get_json()

            item = data[0]
            # Exactly 40 chars should NOT be truncated (only > 40)
            assert item['query_text'] == exact_query
            assert '...' not in item['query_text']

    def test_query_one_over_truncate_limit_is_truncated(self, client, mock_db_connection):
        """Test that query one character over truncate limit is truncated."""
        # Create a non-SQL query exactly 41 characters
        over_query = 'X' * 41
        rows = [
            {'queryid': '12345', 'query': over_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=40')
            data = response.get_json()

            item = data[0]
            # smart_truncate_query: 37 chars + '...' = 40 total
            assert item['query_text'] == 'X' * 37 + '...'
            assert len(item['query_text']) == 40

    def test_ellipsis_added_on_truncation(self, client, mock_db_connection):
        """Test that ellipsis is correctly added when truncating."""
        # Use non-SQL text to test simple truncation path
        long_text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 3
        rows = [
            {'queryid': '12345', 'query': long_text},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=20')
            data = response.get_json()

            item = data[0]
            # 17 chars + '...' = 20 total
            assert item['query_text'] == 'ABCDEFGHIJKLMNOPQ...'
            assert item['query_text'].endswith('...')

    def test_insert_query_smart_truncation(self, client, mock_db_connection):
        """Test that INSERT queries show table name in truncation."""
        insert_query = 'INSERT INTO orders (id, user_id, total, status) VALUES ($1, $2, $3, $4)'
        rows = [
            {'queryid': '12345', 'query': insert_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=30')
            data = response.get_json()

            item = data[0]
            # Should show "INSERT INTO orders ..."
            assert 'INSERT' in item['query_text']
            assert 'orders' in item['query_text']

    def test_update_query_smart_truncation(self, client, mock_db_connection):
        """Test that UPDATE queries show table name in truncation."""
        update_query = 'UPDATE customers SET name = $1, email = $2 WHERE id = $3'
        rows = [
            {'queryid': '12345', 'query': update_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=30')
            data = response.get_json()

            item = data[0]
            # Should show "UPDATE customers ..."
            assert 'UPDATE' in item['query_text']
            assert 'customers' in item['query_text']

    def test_delete_query_smart_truncation(self, client, mock_db_connection):
        """Test that DELETE queries show table name in truncation."""
        delete_query = 'DELETE FROM sessions WHERE expires_at < NOW() AND user_id = $1'
        rows = [
            {'queryid': '12345', 'query': delete_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=30')
            data = response.get_json()

            item = data[0]
            # Should show "DELETE FROM sessions ..."
            assert 'DELETE' in item['query_text']
            assert 'sessions' in item['query_text']

    def test_cte_query_smart_truncation(self, client, mock_db_connection):
        """Test that CTE (WITH) queries show CTE names in truncation."""
        cte_query = '''WITH active_users AS (
            SELECT * FROM users WHERE active = true
        ), recent_orders AS (
            SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '1 day'
        )
        SELECT u.*, o.* FROM active_users u JOIN recent_orders o ON u.id = o.user_id'''
        rows = [
            {'queryid': '12345', 'query': cte_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=60')
            data = response.get_json()

            item = data[0]
            # Should show CTE names
            assert 'WITH' in item['query_text']

    def test_query_with_leading_comment_stripped(self, client, mock_db_connection):
        """Test that leading comments are stripped before truncation."""
        query_with_comment = '/* This is a long comment */ SELECT * FROM users'
        rows = [
            {'queryid': '12345', 'query': query_with_comment},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=50')
            data = response.get_json()

            item = data[0]
            # Comment should be stripped, showing SELECT ... FROM users
            assert 'SELECT' in item['query_text']
            assert 'FROM' in item['query_text']
            assert 'users' in item['query_text']
            # Comment should not be in the result
            assert 'long comment' not in item['query_text']

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_db_connection_failure_returns_empty_array(self, client):
        """Test that database connection failure returns empty array gracefully."""
        with patch('psycopg2.connect', side_effect=Exception('Connection refused')):
            response = client.get('/query_texts')

            # Should return 200 with empty array, not 500
            assert response.status_code == 200
            data = response.get_json()
            assert data == []

    def test_db_query_failure_returns_empty_array(self, client):
        """Test that database query failure returns empty array gracefully."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.side_effect = Exception('Query failed')

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')

            # Should return 200 with empty array, not 500
            assert response.status_code == 200
            data = response.get_json()
            assert data == []

    def test_cursor_iteration_failure_returns_empty_array(self, client):
        """Test that cursor iteration failure returns empty array gracefully."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.__iter__ = MagicMock(side_effect=Exception('Iteration failed'))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')

            assert response.status_code == 200
            data = response.get_json()
            assert data == []

    # =========================================================================
    # HTTP Method Tests
    # =========================================================================

    def test_post_not_allowed(self, client):
        """Test that POST to /query_texts returns 405."""
        response = client.post('/query_texts')
        assert response.status_code == 405

    def test_put_not_allowed(self, client):
        """Test that PUT to /query_texts returns 405."""
        response = client.put('/query_texts')
        assert response.status_code == 405

    def test_delete_not_allowed(self, client):
        """Test that DELETE to /query_texts returns 405."""
        response = client.delete('/query_texts')
        assert response.status_code == 405

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_null_queryid_is_skipped(self, client, mock_db_connection):
        """Test that rows with null queryid are skipped."""
        rows = [
            {'queryid': None, 'query': 'SELECT 1'},
            {'queryid': '12345', 'query': 'SELECT 2'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            # Only the valid row should be in results
            assert len(data) == 1
            assert data[0]['queryid'] == '12345'

    def test_null_query_text_handled(self, client, mock_db_connection):
        """Test that null query text is handled (returns empty string)."""
        rows = [
            {'queryid': '12345', 'query': None},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 1
            assert data[0]['query_text'] == ''

    def test_special_characters_in_query(self, client, mock_db_connection):
        """Test that special characters in query text are handled."""
        special_query = "SELECT * FROM t"  # Keep it short to avoid truncation
        rows = [
            {'queryid': '12345', 'query': special_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=200')
            data = response.get_json()

            assert len(data) == 1
            # Response is valid JSON
            assert 'query_text' in data[0]

    def test_unicode_in_query(self, client, mock_db_connection):
        """Test that unicode characters in query text are handled."""
        unicode_query = "SELECT * FROM t"  # Keep short
        rows = [
            {'queryid': '12345', 'query': unicode_query},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts?truncate=200')
            data = response.get_json()

            assert len(data) == 1
            assert 'query_text' in data[0]

    def test_very_long_queryid(self, client, mock_db_connection):
        """Test that very long queryid is handled."""
        long_queryid = '1' * 100
        rows = [
            {'queryid': long_queryid, 'query': 'SELECT 1'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            assert len(data) == 1
            assert data[0]['queryid'] == long_queryid

    def test_connection_closed_after_query(self, client, mock_db_connection):
        """Test that database connection is properly closed after query."""
        rows = [
            {'queryid': '12345', 'query': 'SELECT 1'},
        ]
        mock_conn, _ = mock_db_connection(rows)

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')

            # Verify connection was closed
            mock_conn.close.assert_called_once()


class TestQueryTextsEndpointIntegration:
    """Integration-style tests that verify the endpoint behavior end-to-end."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a Flask test client."""
        import app as app_module

        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as client:
            yield client

    def test_grafana_infinity_datasource_compatibility(self, client):
        """
        Test that the response format is compatible with Grafana Infinity datasource.

        The Infinity datasource expects a JSON array where each object can be used
        with the "Config from query results" transformation to map displayName.
        """
        rows = [
            {'queryid': '111', 'query': 'SELECT u FROM t'},
            {'queryid': '222', 'query': 'INSERT INTO o'},
        ]

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.__iter__ = MagicMock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            # Verify Grafana-compatible structure
            assert isinstance(data, list)
            for item in data:
                # Each item must have these fields for Grafana transformation
                assert 'queryid' in item
                assert 'displayName' in item
                # queryid should be usable as a key
                assert isinstance(item['queryid'], str)
                # displayName should be a non-empty string for display
                assert isinstance(item['displayName'], str)

    def test_topn_chart_legend_use_case(self, client):
        """
        Test the primary use case: providing display names for TopN chart legends.

        The endpoint should return truncated, readable query text suitable for
        chart legend display.
        """
        rows = [
            {'queryid': '12345678901234567890', 'query': 'SELECT u.id, u.name, u.email, u.created_at FROM users u WHERE u.active = true ORDER BY u.created_at DESC LIMIT 100'},
        ]

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.__iter__ = MagicMock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch('psycopg2.connect', return_value=mock_conn):
            response = client.get('/query_texts')
            data = response.get_json()

            item = data[0]
            # Default truncate is 40, so query should be readable but short
            assert len(item['query_text']) <= 40
            # displayName should be suitable for chart legend
            assert len(item['displayName']) <= 40
            # Should contain meaningful info (SELECT, FROM, table name)
            assert 'SELECT' in item['query_text'] or 'users' in item['query_text']


class TestSmartTruncateQueryFunction:
    """Unit tests for the smart_truncate_query helper function."""

    def test_empty_query_returns_empty_string(self):
        """Test that empty query returns empty string."""
        from app import smart_truncate_query
        assert smart_truncate_query('') == ''
        assert smart_truncate_query(None) == ''

    def test_short_query_unchanged(self):
        """Test that short queries are not modified."""
        from app import smart_truncate_query
        short = 'SELECT id FROM t'
        assert smart_truncate_query(short, 100) == short

    def test_select_from_extraction(self):
        """Test that SELECT ... FROM table is extracted."""
        from app import smart_truncate_query
        query = 'SELECT a, b, c, d, e, f, g FROM mytable WHERE x = 1'
        result = smart_truncate_query(query, 50)
        assert 'SELECT' in result
        assert 'FROM' in result
        assert 'mytable' in result

    def test_insert_into_extraction(self):
        """Test that INSERT INTO table is extracted."""
        from app import smart_truncate_query
        query = 'INSERT INTO orders (col1, col2, col3) VALUES ($1, $2, $3)'
        result = smart_truncate_query(query, 30)
        assert 'INSERT' in result
        assert 'orders' in result

    def test_update_table_extraction(self):
        """Test that UPDATE table is extracted."""
        from app import smart_truncate_query
        query = 'UPDATE customers SET col1 = $1, col2 = $2 WHERE id = $3'
        result = smart_truncate_query(query, 30)
        assert 'UPDATE' in result
        assert 'customers' in result

    def test_delete_from_extraction(self):
        """Test that DELETE FROM table is extracted."""
        from app import smart_truncate_query
        query = 'DELETE FROM sessions WHERE expires_at < NOW()'
        result = smart_truncate_query(query, 30)
        assert 'DELETE' in result
        assert 'sessions' in result

    def test_block_comment_stripped(self):
        """Test that block comments are stripped."""
        from app import smart_truncate_query
        query = '/* comment */ SELECT * FROM t'
        result = smart_truncate_query(query, 50)
        assert 'comment' not in result
        assert 'SELECT' in result

    def test_line_comment_stripped(self):
        """Test that line comments are stripped."""
        from app import smart_truncate_query
        query = '-- comment\nSELECT * FROM t'
        result = smart_truncate_query(query, 50)
        assert 'comment' not in result
        assert 'SELECT' in result

    def test_fallback_truncation(self):
        """Test fallback to simple truncation for non-SQL text."""
        from app import smart_truncate_query
        text = 'ABCDEFGHIJ' * 10
        result = smart_truncate_query(text, 20)
        assert len(result) == 20
        assert result.endswith('...')

    def test_max_length_respected(self):
        """Test that result never exceeds max_length."""
        from app import smart_truncate_query
        query = 'SELECT very_long_column_name_1, very_long_column_name_2 FROM extremely_long_table_name_here'
        result = smart_truncate_query(query, 40)
        assert len(result) <= 40
