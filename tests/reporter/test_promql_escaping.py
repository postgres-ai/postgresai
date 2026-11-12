"""Tests for PromQL label escaping and query filter building.

These tests verify that user-controlled values (cluster names, database names,
index names, etc.) are properly escaped when interpolated into PromQL queries,
preventing PromQL injection attacks.
"""
import sys
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Mock optional dependencies so the reporter imports without them installed.
# (The Flask backend's escape helpers are tested in old/tests/reporter/.)
# ---------------------------------------------------------------------------
for _mod in [
    "prometheus_api_client",
    "boto3",
    "botocore",
    "requests_aws4auth",
    "psycopg2",
    "psycopg2.extras",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


# ---------------------------------------------------------------------------
# reporter.postgres_reports._escape_promql_label
# ---------------------------------------------------------------------------

class TestEscapePromqlLabel:
    """Tests for PostgresReportGenerator._escape_promql_label static method."""

    @pytest.mark.unit
    def test_plain_string_unchanged(self):
        assert PostgresReportGenerator._escape_promql_label("my-cluster") == "my-cluster"

    @pytest.mark.unit
    def test_escapes_double_quotes(self):
        assert PostgresReportGenerator._escape_promql_label('db"name') == 'db\\"name'

    @pytest.mark.unit
    def test_escapes_backslashes(self):
        assert PostgresReportGenerator._escape_promql_label("path\\to") == "path\\\\to"

    @pytest.mark.unit
    def test_escapes_backslash_before_quote(self):
        """Backslash must be escaped first, then quote."""
        result = PostgresReportGenerator._escape_promql_label('a\\"b')
        assert result == 'a\\\\\\"b'

    @pytest.mark.unit
    def test_escapes_newline(self):
        """Newline character must be rendered as the two-character sequence \\n."""
        assert PostgresReportGenerator._escape_promql_label("line1\nline2") == "line1\\nline2"

    @pytest.mark.unit
    def test_empty_string(self):
        assert PostgresReportGenerator._escape_promql_label("") == ""

    @pytest.mark.unit
    def test_injection_attempt_closing_brace(self):
        """A value like: db"}} OR vector(1) should not break out of the label."""
        result = PostgresReportGenerator._escape_promql_label('db"}} OR vector(1)')
        assert result == 'db\\"}} OR vector(1)'

    @pytest.mark.unit
    def test_normal_postgres_identifiers(self):
        """Common PostgreSQL identifiers should pass through unchanged."""
        identifiers = [
            "public",
            "my_table",
            "idx_users_email",
            "pg_stat_statements",
            "node-01",
            "cluster.local",
        ]
        for ident in identifiers:
            assert PostgresReportGenerator._escape_promql_label(ident) == ident

    @pytest.mark.unit
    @pytest.mark.parametrize("payload,expected", [
        # Injection attempts
        ('x"} OR 1==1{cluster="bad', 'x\\"} OR 1==1{cluster=\\"bad'),
        ('name\\"}}', 'name\\\\\\"}}'),
        ('a\nb', 'a\\nb'),
        ('a\n"b', 'a\\n\\"b'),
    ])
    def test_parameterized_injection(self, payload, expected):
        assert PostgresReportGenerator._escape_promql_label(payload) == expected



# ---------------------------------------------------------------------------
# reporter.postgres_reports.PostgresReportGenerator — table bloat queries
# ---------------------------------------------------------------------------

class TestGetTableBloatEscaping:
    """Tests that generate_f004_heap_bloat_report properly escapes user-supplied
    parameters (cluster, node_name) in the PromQL queries it builds.

    The function is the canonical 'get_table_bloat' path in the reporter; these
    tests guard against PromQL injection via the cluster or node_name arguments.
    """

    @pytest.mark.unit
    def test_cluster_injection_escaped_in_queries(self, generator):
        """Malicious cluster name must be escaped in every PromQL query built."""
        malicious_cluster = 'evil"cluster}} OR vector(1) #'
        empty_result = {'status': 'success', 'data': {'result': []}}

        with patch.object(generator, 'get_all_databases', return_value=['testdb']):
            with patch.object(generator, 'query_instant', return_value=empty_result) as mock_qi:
                generator.generate_f004_heap_bloat_report(
                    cluster=malicious_cluster,
                    node_name='node-01',
                )

        for call_args in mock_qi.call_args_list:
            query = call_args[0][0]
            # The raw injection string must NOT appear verbatim in any query
            assert malicious_cluster not in query, (
                f"Unescaped injection payload found in query: {query!r}"
            )
            # The double-quote in the cluster name must be escaped as \"
            assert '\\"cluster' in query or 'evil\\"cluster' in query or 'evil\\\\' in query, (
                f"Cluster name not properly escaped in query: {query!r}"
            )

    @pytest.mark.unit
    def test_node_name_injection_escaped_in_queries(self, generator):
        """Malicious node_name must be escaped in every PromQL query built."""
        malicious_node = 'node"}} OR 1==1{'
        empty_result = {'status': 'success', 'data': {'result': []}}

        with patch.object(generator, 'get_all_databases', return_value=['mydb']):
            with patch.object(generator, 'query_instant', return_value=empty_result) as mock_qi:
                generator.generate_f004_heap_bloat_report(
                    cluster='local',
                    node_name=malicious_node,
                )

        for call_args in mock_qi.call_args_list:
            query = call_args[0][0]
            assert malicious_node not in query, (
                f"Unescaped node_name injection found in query: {query!r}"
            )

    @pytest.mark.unit
    def test_no_databases_returns_valid_structure(self, generator):
        """generate_f004_heap_bloat_report returns a dict even when no databases exist."""
        empty_result = {'status': 'success', 'data': {'result': []}}

        with patch.object(generator, 'get_all_databases', return_value=[]):
            with patch.object(generator, 'query_instant', return_value=empty_result):
                result = generator.generate_f004_heap_bloat_report(
                    cluster='local', node_name='node-01'
                )

        assert isinstance(result, dict)
