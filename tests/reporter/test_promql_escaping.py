"""Tests for PromQL label escaping and query filter building.

These tests verify that user-controlled values (cluster names, database names,
index names, etc.) are properly escaped when interpolated into PromQL queries,
preventing PromQL injection attacks.
"""
import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Mock unavailable Flask-stack dependencies so the pure escape helpers can be
# imported without a running Flask application or installed packages.
for _mod in [
    "flask",
    "prometheus_api_client",
    "boto3",
    "botocore",
    "requests_aws4auth",
    "psycopg2",
    "psycopg2.extras",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from monitoring_flask_backend.app import (  # noqa: E402
    escape_promql_label as flask_escape_promql_label,
    escape_promql_regex_literal as flask_escape_promql_regex_literal,
)
from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def escape_function():
    """Return the flask escape_promql_label function; fail (not skip) if unavailable."""
    try:
        from monitoring_flask_backend.app import escape_promql_label
        return escape_promql_label
    except ImportError as e:
        pytest.fail(f"Could not import escape_promql_label from monitoring_flask_backend.app: {e}")


@pytest.fixture
def safe_label_value():
    """A label value that contains no PromQL special characters."""
    return "simple-cluster-name"


@pytest.fixture
def malicious_inputs():
    """Injection payloads that must be neutralised by the escape function."""
    return [
        '"}}OR 1=1{cluster="bad',
        'node\n01',
        'db\\name',
        '.*injected.*',
        'a"b',
        'path\\to\\"file',
    ]


@pytest.mark.unit
def test_backslash_escaped(escape_function):
    """Backslash must be doubled; the result must equal the expected literal, not merely differ."""
    result = escape_function("path\\to")
    assert result == "path\\\\to"


@pytest.mark.unit
def test_idempotent_escaping(escape_function, safe_label_value):
    """Safe values (no PromQL metacharacters) must pass through escaping unchanged."""
    assert escape_function(safe_label_value) == safe_label_value


@pytest.mark.unit
def test_cve_style_injection(escape_function, malicious_inputs):
    """All malicious payloads must have their double-quotes escaped so they cannot
    break out of a PromQL label value and inject arbitrary selectors."""
    for payload in malicious_inputs:
        escaped = escape_function(payload)
        # A raw unescaped double-quote in the output would allow label-boundary escape
        # We verify that every '"' in the output is preceded by a backslash.
        idx = 0
        while idx < len(escaped):
            ch = escaped[idx]
            if ch == '"':
                assert idx > 0 and escaped[idx - 1] == '\\', (
                    f"Unescaped '\"' at position {idx} in escaped output {escaped!r} "
                    f"(input: {payload!r})"
                )
            idx += 1


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


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


class TestFlaskEscapePromqlLabel:
    """Tests for the Flask app's module-level escape_promql_label function."""

    @pytest.mark.unit
    def test_plain_string_unchanged(self):
        assert flask_escape_promql_label("my-cluster") == "my-cluster"

    @pytest.mark.unit
    def test_escapes_double_quotes(self):
        assert flask_escape_promql_label('db"name') == 'db\\"name'

    @pytest.mark.unit
    def test_escapes_backslashes(self):
        assert flask_escape_promql_label("path\\to") == "path\\\\to"

    @pytest.mark.unit
    def test_escapes_backslash_before_quote(self):
        result = flask_escape_promql_label('a\\"b')
        assert result == 'a\\\\\\"b'

    @pytest.mark.unit
    def test_escapes_newline(self):
        """Newline character must be rendered as the two-character sequence \\n."""
        assert flask_escape_promql_label("line1\nline2") == "line1\\nline2"

    @pytest.mark.unit
    def test_empty_string(self):
        assert flask_escape_promql_label("") == ""

    @pytest.mark.unit
    def test_injection_attempt_closing_brace(self):
        result = flask_escape_promql_label('db"}} OR vector(1)')
        assert result == 'db\\"}} OR vector(1)'

    @pytest.mark.unit
    def test_normal_postgres_identifiers(self):
        identifiers = [
            "public",
            "my_table",
            "idx_users_email",
            "pg_stat_statements",
            "node-01",
            "cluster.local",
        ]
        for ident in identifiers:
            assert flask_escape_promql_label(ident) == ident

    @pytest.mark.unit
    @pytest.mark.parametrize("payload,expected", [
        ('x"} OR 1==1{cluster="bad', 'x\\"} OR 1==1{cluster=\\"bad'),
        ('name\\"}}', 'name\\\\\\"}}'),
        ('a\nb', 'a\\nb'),
        ('a\n"b', 'a\\n\\"b'),
    ])
    def test_parameterized_injection(self, payload, expected):
        assert flask_escape_promql_label(payload) == expected


class TestFlaskEscapePromqlRegexLiteral:
    """Tests for the Flask app's escape_promql_regex_literal function.

    This function is used when a user-supplied value is embedded inside a
    PromQL regex matcher (=~) and must be treated as a literal, e.g.:
        instance=~".*<node_name>.*"
    """

    @pytest.mark.unit
    def test_plain_string_unchanged(self):
        assert flask_escape_promql_regex_literal("node-01") == "node-01"

    @pytest.mark.unit
    def test_escapes_dot(self):
        """Dot is a regex wildcard and must be escaped."""
        assert flask_escape_promql_regex_literal("node.local") == "node\\.local"

    @pytest.mark.unit
    def test_escapes_star(self):
        assert flask_escape_promql_regex_literal("v1.*") == "v1\\.\\*"

    @pytest.mark.unit
    def test_escapes_plus(self):
        assert flask_escape_promql_regex_literal("a+b") == "a\\+b"

    @pytest.mark.unit
    def test_escapes_question_mark(self):
        assert flask_escape_promql_regex_literal("node?") == "node\\?"

    @pytest.mark.unit
    def test_escapes_pipe(self):
        assert flask_escape_promql_regex_literal("a|b") == "a\\|b"

    @pytest.mark.unit
    def test_escapes_parens(self):
        assert flask_escape_promql_regex_literal("(group)") == "\\(group\\)"

    @pytest.mark.unit
    def test_escapes_brackets(self):
        # '-' outside a character class is not a metacharacter; only '[' and ']' need escaping
        assert flask_escape_promql_regex_literal("[0-9]") == "\\[0-9\\]"

    @pytest.mark.unit
    def test_escapes_braces(self):
        assert flask_escape_promql_regex_literal("x{3}") == "x\\{3\\}"

    @pytest.mark.unit
    def test_escapes_caret_and_dollar(self):
        assert flask_escape_promql_regex_literal("^start$") == "\\^start\\$"

    @pytest.mark.unit
    def test_escapes_backslash(self):
        """Backslash must be doubled for PromQL string quoting."""
        assert flask_escape_promql_regex_literal("path\\to") == "path\\\\to"

    @pytest.mark.unit
    def test_escapes_double_quote(self):
        assert flask_escape_promql_regex_literal('say"hi"') == 'say\\"hi\\"'

    @pytest.mark.unit
    def test_escapes_newline(self):
        assert flask_escape_promql_regex_literal("line1\nline2") == "line1\\nline2"

    @pytest.mark.unit
    def test_hostname_with_dots(self):
        """Real-world hostnames with dots are treated as literals; hyphens are not metacharacters."""
        result = flask_escape_promql_regex_literal("db-01.us-east.example.com")
        assert result == "db-01\\.us-east\\.example\\.com"

    @pytest.mark.unit
    @pytest.mark.parametrize("payload,expected", [
        # Injection: try to break out of ".*<value>.*" anchor
        ('.*injected.*', '\\.\\*injected\\.\\*'),
        # Double-quote escape injection
        ('"}}', '\\"\\}\\}'),
        # Newline injection
        ('node\n01', 'node\\n01'),
        # Combined: dot + quote + newline
        ('a.b"c\nd', 'a\\.b\\"c\\nd'),
        # Pipe alternation
        ('prod|staging', 'prod\\|staging'),
    ])
    def test_parameterized_injection(self, payload, expected):
        assert flask_escape_promql_regex_literal(payload) == expected


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


class TestSchemaFilterEscaping:
    """Tests that schema_filter values are escaped with escape_promql_regex_literal.

    The /btree_bloat/csv and /table_info/csv Flask endpoints accept a 'schemaname'
    (schema_filter) query parameter and use escape_promql_regex_literal to embed it
    in a PromQL regex matcher.  These tests verify the escaping contract in isolation.
    """

    @pytest.mark.unit
    def test_schema_with_dot_escaped_for_regex(self):
        """Schema names containing dots must have dots escaped (dot = any char in RE2)."""
        schema = "my.schema"
        escaped = flask_escape_promql_regex_literal(schema)
        assert "\\." in escaped
        # Building the filter string should be safe
        filter_fragment = f'schemaname=~"{escaped}"'
        assert '"my.schema"' not in filter_fragment  # raw dot must not appear

    @pytest.mark.unit
    def test_schema_injection_escaped_for_regex(self):
        """Schema name with injection chars is neutralised by regex escaping."""
        schema = 'public"}} OR vector(1) #'
        escaped = flask_escape_promql_regex_literal(schema)
        filter_fragment = f'schemaname=~"{escaped}"'
        # No unescaped closing quote that would break out of the label value
        # Every '"' in the output must be preceded by '\'
        idx = 0
        while idx < len(escaped):
            if escaped[idx] == '"':
                assert idx > 0 and escaped[idx - 1] == '\\', (
                    f"Unescaped '\"' at position {idx} in escaped schema: {escaped!r}"
                )
            idx += 1

    @pytest.mark.unit
    @pytest.mark.parametrize("schema,expected_fragment", [
        ("public", "public"),
        ("my_schema", "my_schema"),
        ("schema.with.dots", "schema\\.with\\.dots"),
        ("schema+extra", "schema\\+extra"),
        ("schema*wild", "schema\\*wild"),
    ])
    def test_common_schema_names_escaped(self, schema, expected_fragment):
        """Common schema name patterns are correctly handled by regex escaping."""
        escaped = flask_escape_promql_regex_literal(schema)
        assert expected_fragment in escaped
