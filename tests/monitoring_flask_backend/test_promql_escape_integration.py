"""Integration tests: PromQL escape application in report generator endpoints.

These tests verify that get_pgss_metrics_csv, get_btree_bloat_csv, and
get_table_info_csv correctly apply escape_promql_label / escape_promql_regex_literal
when building PromQL filter strings.  They use injection payloads that would
produce incorrect (or exploitable) queries if escaping were removed or bypassed.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call

# Add the monitoring_flask_backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monitoring_flask_backend'))


def _make_prom_mock(query_results=None):
    """Return a mock PrometheusConnect with empty results by default."""
    mock_prom = MagicMock()
    mock_prom.custom_query.return_value = query_results or []
    mock_prom.get_current_metric_value.return_value = []
    mock_prom.get_metric_range_data.return_value = []
    return mock_prom


def _captured_queries(mock_prom):
    """Collect all query strings passed to custom_query / get_current_metric_value."""
    queries = []
    for c in mock_prom.custom_query.call_args_list:
        qs = c[1].get('query') or (c[0][0] if c[0] else '')
        queries.append(qs)
    for c in mock_prom.get_metric_range_data.call_args_list:
        # first positional arg or metric_name kwarg
        qs = c[1].get('metric_name') or (c[0][0] if c[0] else '')
        queries.append(qs)
    return queries


@pytest.fixture(autouse=True)
def isolate_app_module():
    """Force a fresh import of app for each test to avoid cross-test state."""
    if 'app' in sys.modules:
        del sys.modules['app']
    yield
    if 'app' in sys.modules:
        del sys.modules['app']


@pytest.fixture
def client():
    import app as app_module
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


class TestPgssMetricsCsvPromqlEscape:
    """Verify that /pgss_metrics/csv correctly escapes PromQL parameters."""

    # Required params so the endpoint doesn't 400 before building any query.
    BASE_PARAMS = 'time_start=1700000000&time_end=1700003600'

    def _get(self, client, extra_qs=''):
        url = f'/pgss_metrics/csv?{self.BASE_PARAMS}'
        if extra_qs:
            url += f'&{extra_qs}'
        return url

    def test_cluster_name_quotes_escaped(self, client):
        """Double-quote in cluster_name must be backslash-escaped in PromQL."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(self._get(client, 'cluster_name=my%22cluster'))

        queries = _captured_queries(mock_prom)
        assert queries, "Expected at least one PromQL query to be issued"
        # The raw " must appear as \" inside the label matcher, not as a bare "
        for q in queries:
            if 'cluster=' in q:
                assert 'my\\"cluster' in q or r'my\"cluster' in q, (
                    f"Unescaped quote in PromQL query: {q!r}")
                # Must NOT contain an unescaped bare double-quote after cluster=
                # i.e. cluster="my"cluster" (which would end the matcher early)
                assert 'cluster="my"cluster"' not in q

    def test_cluster_name_backslash_escaped(self, client):
        """Backslash in cluster_name must be doubled in PromQL."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(self._get(client, r'cluster_name=back%5Cslash'))

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'cluster=' in q:
                # A single backslash in input → must appear as \\ in PromQL
                assert '\\\\' in q, (
                    f"Backslash not doubled in PromQL query: {q!r}")

    def test_db_name_injection_payload_escaped(self, client):
        """Injection payload in db_name must not break out of the label matcher."""
        # Payload: closing the string early then injecting extra selector
        payload = 'mydb",cluster="evil'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(self._get(client, f'db_name={encoded}'))

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'datname=' in q:
                # The injected cluster="evil" must not appear as a bare label
                assert ',cluster="evil"' not in q, (
                    f"PromQL injection not escaped: {q!r}")

    def test_node_name_regex_metachar_escaped(self, client):
        """Regex metacharacters in node_name must be treated as literals."""
        # Payload: .* would match any node if unescaped
        payload = 'node.1.*'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(self._get(client, f'node_name={encoded}'))

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'instance=~' in q:
                # Dots and * must be escaped: \. and \*
                assert r'\.' in q, (
                    f"Dot not escaped as regex literal in PromQL: {q!r}")
                assert r'\*' in q, (
                    f"'*' not escaped as regex literal in PromQL: {q!r}")
                # Must not contain bare .* (unescaped regex wildcard)
                assert '.*node' not in q or '\\.*' in q or r'\.' in q

    def test_node_name_newline_escaped(self, client):
        """Newline in node_name must be escaped as \\n, not embedded."""
        import urllib.parse
        payload = 'node\ninjected'
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(self._get(client, f'node_name={encoded}'))

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'instance=~' in q:
                assert '\n' not in q, (
                    f"Raw newline found in PromQL query: {q!r}")


class TestBtreeBloatCsvPromqlEscape:
    """Verify that /btree_bloat/csv correctly escapes PromQL parameters."""

    def test_cluster_name_injection_escaped(self, client):
        """Injection payload in cluster_name must be escaped."""
        payload = 'prod",idxname="evil_idx'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/btree_bloat/csv?cluster_name={encoded}')

        queries = _captured_queries(mock_prom)
        assert queries, "Expected at least one PromQL query to be issued"
        for q in queries:
            if 'cluster=' in q:
                # Injected label must not appear unescaped
                assert ',idxname="evil_idx"' not in q, (
                    f"PromQL injection not escaped: {q!r}")

    def test_db_name_quote_escaped(self, client):
        """Double-quote in db_name must be backslash-escaped."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get('/btree_bloat/csv?db_name=my%22db')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'datname=' in q:
                assert 'datname="my"db"' not in q, (
                    f"Unescaped quote breaks label matcher: {q!r}")

    def test_schemaname_quote_escaped(self, client):
        """Double-quote in schemaname must be backslash-escaped."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get('/btree_bloat/csv?schemaname=pub%22lic')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'schemaname=' in q:
                assert 'schemaname="pub"lic"' not in q, (
                    f"Unescaped quote breaks label matcher: {q!r}")

    def test_tblname_injection_escaped(self, client):
        """Injection payload in tblname must not inject a new label."""
        payload = 'orders",cluster="attacker'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/btree_bloat/csv?tblname={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'tblname=' in q:
                assert ',cluster="attacker"' not in q, (
                    f"PromQL injection not escaped: {q!r}")

    def test_idxname_backslash_escaped(self, client):
        """Backslash in idxname must be doubled."""
        import urllib.parse
        payload = r'idx\name'
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/btree_bloat/csv?idxname={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'idxname=' in q:
                assert '\\\\' in q, (
                    f"Backslash not doubled in PromQL query: {q!r}")

    def test_node_name_quote_escaped(self, client):
        """Double-quote in node_name must be backslash-escaped."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get('/btree_bloat/csv?node_name=my%22node')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'node_name=' in q:
                assert 'node_name="my"node"' not in q, (
                    f"Unescaped quote breaks label matcher: {q!r}")


class TestTableInfoCsvPromqlEscape:
    """Verify that /table_info/csv correctly escapes PromQL parameters."""

    def test_cluster_name_injection_escaped(self, client):
        """Injection payload in cluster_name must not inject extra labels."""
        payload = 'mycluster",datname="attacker_db'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?cluster_name={encoded}')

        queries = _captured_queries(mock_prom)
        assert queries, "Expected at least one PromQL query to be issued"
        for q in queries:
            if 'cluster=' in q:
                assert ',datname="attacker_db"' not in q, (
                    f"PromQL injection not escaped: {q!r}")

    def test_db_name_quote_escaped(self, client):
        """Double-quote in db_name must be backslash-escaped."""
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get('/table_info/csv?db_name=my%22db')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'datname=' in q:
                assert 'datname="my"db"' not in q, (
                    f"Unescaped quote breaks label matcher: {q!r}")

    def test_schemaname_regex_metachar_escaped(self, client):
        """Regex metacharacters in schemaname must be escaped as literals.

        schemaname uses =~ (regex matcher), so dots/stars/etc. must be
        RE2-escaped to prevent wildcard expansion.
        """
        payload = 'public.schema.*'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?schemaname={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'schemaname=~' in q:
                assert r'\.' in q, (
                    f"Dot not escaped as regex literal in PromQL: {q!r}")
                assert r'\*' in q, (
                    f"'*' not escaped as regex literal in PromQL: {q!r}")
                # A bare .* must not appear in the middle of the value
                assert 'public.schema.*"' not in q, (
                    f"Unescaped regex metachar in PromQL query: {q!r}")

    def test_tblname_injection_escaped(self, client):
        """Injection payload in tblname must not inject a new label."""
        payload = 'orders",cluster="evil'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?tblname={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'tblname=' in q:
                assert ',cluster="evil"' not in q, (
                    f"PromQL injection not escaped: {q!r}")

    def test_node_name_injection_escaped(self, client):
        """Injection payload in node_name must not inject extra labels.

        node_name uses an exact label match (=), so escape_promql_label applies:
        double-quotes and backslashes are escaped, preventing label injection.
        """
        payload = 'node1",cluster="evil'
        import urllib.parse
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?node_name={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'node_name=' in q:
                # Injected label must not appear as an independent matcher
                assert ',cluster="evil"' not in q, (
                    f"PromQL injection not escaped: {q!r}")
                # The double-quote must be backslash-escaped
                assert 'node_name="node1"' not in q or r'\"' in q, (
                    f"Unescaped quote breaks label matcher: {q!r}")

    def test_schemaname_newline_escaped(self, client):
        """Newline in schemaname must not appear raw in the PromQL query."""
        import urllib.parse
        payload = 'myschema\ninjected'
        encoded = urllib.parse.quote(payload)
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?schemaname={encoded}')

        queries = _captured_queries(mock_prom)
        for q in queries:
            if 'schemaname=~' in q:
                assert '\n' not in q, (
                    f"Raw newline found in PromQL query: {q!r}")

    def test_all_params_combined_no_injection(self, client):
        """Multiple injection payloads simultaneously must all be escaped."""
        import urllib.parse
        cluster_payload = 'prod"cluster'
        db_payload = 'mydb"extra'
        schema_payload = 'pub.lic.*'
        tbl_payload = 'tbl"name'
        params = '&'.join([
            f'cluster_name={urllib.parse.quote(cluster_payload)}',
            f'db_name={urllib.parse.quote(db_payload)}',
            f'schemaname={urllib.parse.quote(schema_payload)}',
            f'tblname={urllib.parse.quote(tbl_payload)}',
        ])
        mock_prom = _make_prom_mock()
        with patch('app.get_prometheus_client', return_value=mock_prom):
            client.get(f'/table_info/csv?{params}')

        queries = _captured_queries(mock_prom)
        assert queries, "Expected at least one PromQL query to be issued"
        for q in queries:
            # No bare (unescaped) double-quotes should appear inside label values.
            # A valid escaped quote looks like \" — check by ensuring we never
            # see the pattern: ="...unescaped-quote...
            # The simplest invariant: the filter substring must not contain
            # a literal closing pattern that would terminate the matcher early.
            assert 'cluster="prod"' not in q, f"Injection in cluster label: {q!r}"
            assert 'datname="mydb"' not in q, f"Injection in datname label: {q!r}"
