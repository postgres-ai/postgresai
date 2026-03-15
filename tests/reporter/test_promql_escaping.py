"""Tests for PromQL label escaping and query filter building.

These tests verify that user-controlled values (cluster names, database names,
index names, etc.) are properly escaped when interpolated into PromQL queries,
preventing PromQL injection attacks.
"""
import pytest

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


class TestEscapePromqlLabel:
    """Tests for _escape_promql_label static method."""

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
    def test_empty_string(self):
        assert PostgresReportGenerator._escape_promql_label("") == ""

    @pytest.mark.unit
    def test_injection_attempt_closing_brace(self):
        """A value like: db"}} OR vector(1) should not break out of the label."""
        result = PostgresReportGenerator._escape_promql_label('db"}} OR vector(1)')
        assert '"' not in result or result.count('\\"') == result.count('"')
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


class TestPromqlFilter:
    """Tests for _promql_filter method."""

    @pytest.mark.unit
    def test_single_label(self, generator):
        result = generator._promql_filter(cluster="local")
        assert result == '{cluster="local"}'

    @pytest.mark.unit
    def test_multiple_labels(self, generator):
        result = generator._promql_filter(cluster="local", node_name="node-01")
        assert 'cluster="local"' in result
        assert 'node_name="node-01"' in result
        assert result.startswith("{")
        assert result.endswith("}")

    @pytest.mark.unit
    def test_escapes_values(self, generator):
        result = generator._promql_filter(cluster='my"cluster')
        assert result == '{cluster="my\\"cluster"}'

    @pytest.mark.unit
    def test_empty_labels(self, generator):
        result = generator._promql_filter()
        assert result == "{}"
