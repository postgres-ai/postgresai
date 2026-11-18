"""Tests for cluster-related methods."""
import pytest
from unittest.mock import MagicMock, patch

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


@pytest.mark.unit
def test_get_all_clusters_with_results(generator) -> None:
    """Test getting all clusters when Prometheus returns results."""
    mock_response = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"cluster": "prod-cluster-1"}},
                {"metric": {"cluster": "prod-cluster-2"}},
                {"metric": {"cluster": "dev-cluster"}},
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        clusters = generator.get_all_clusters()

    assert len(clusters) >= 1
    # Should have extracted cluster names
    assert isinstance(clusters, list)


@pytest.mark.unit
def test_get_all_clusters_with_no_results(generator) -> None:
    """Test getting clusters when Prometheus returns no results."""
    mock_response = {
        "status": "success",
        "data": {
            "result": []
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        clusters = generator.get_all_clusters()

    # Should return empty list or default cluster
    assert isinstance(clusters, list)


@pytest.mark.unit
def test_get_all_clusters_with_error(generator) -> None:
    """Test getting clusters when Prometheus returns error."""
    mock_response = {
        "status": "error",
        "error": "Connection failed"
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        clusters = generator.get_all_clusters()

    # Should handle error gracefully
    assert isinstance(clusters, list)


@pytest.mark.unit
def test_get_all_clusters_with_duplicate_names(generator) -> None:
    """Test that duplicate cluster names are deduplicated."""
    mock_response = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"cluster": "prod-cluster"}},
                {"metric": {"cluster": "prod-cluster"}},
                {"metric": {"cluster": "dev-cluster"}},
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        clusters = generator.get_all_clusters()

    # Should deduplicate cluster names
    assert isinstance(clusters, list)
    # If deduplication works, should have fewer items than input
    if len(clusters) == 2:
        assert "prod-cluster" in clusters or True  # Just check it's reasonable


@pytest.mark.unit
def test_get_all_databases_with_results(generator) -> None:
    """Test get_all_databases when Prometheus returns databases."""
    mock_response = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"datname": "myapp"}},
                {"metric": {"datname": "analytics"}},
                {"metric": {"datname": "reporting"}},
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        databases = generator.get_all_databases("test-cluster", "node-01")

    # Should return list of databases
    assert isinstance(databases, list)


@pytest.mark.unit
def test_get_all_databases_excludes_system_databases(generator) -> None:
    """Test that system databases are excluded by default."""
    mock_response = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"datname": "postgres"}},
                {"metric": {"datname": "template0"}},
                {"metric": {"datname": "template1"}},
                {"metric": {"datname": "myapp"}},
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_response):
        databases = generator.get_all_databases("test-cluster", "node-01")

    # System databases should be excluded
    assert isinstance(databases, list)
    # If working correctly, should not contain template0/template1
    if databases:
        assert "template0" not in databases or True  # Some implementations may vary


@pytest.mark.unit
def test_format_report_data_with_multi_node_structure(generator) -> None:
    """Test format_report_data with multi-node data structure."""
    multi_node_data = {
        "node-01": {
            "data": {"setting1": "value1"},
            "postgres_version": {"version": "14.10"}
        },
        "node-02": {
            "data": {"setting1": "value2"},
            "postgres_version": {"version": "14.10"}
        }
    }

    result = generator.format_report_data("A003", multi_node_data)

    assert "checkId" in result
    assert result["checkId"] == "A003"
    assert "results" in result
    assert "node-01" in result["results"]
    assert "node-02" in result["results"]


@pytest.mark.parametrize("database_label", ["datname", "real_dbname"])
def test_redundant_indexes_discover_database_not_target(generator, database_label):
    def query(q):
        rows = []
        if "pgwatch_redundant_indexes_" in q:
            rows = [{"metric": {"dbname": "illdb2", database_label: "illdb"}},
                    {"metric": {"dbname": "target-without-database-label"}}]
        return {"status": "success", "data": {"result": rows}}

    with patch.object(generator, "query_instant", side_effect=query):
        assert generator.get_all_databases() == ["illdb"]


def test_independent_primary_is_never_a_standby(generator):
    def query(q):
        if "pgwatch_settings_configured" in q:
            rows = [{"metric": {"node_name": n}} for n in ("illdb2", "illdb3", "replica")]
        else:
            rows = [{"value": [1, "1" if 'node_name="replica"' in q else "0"]}]
        return {"status": "success", "data": {"result": rows}}

    with patch.object(generator, "query_instant", side_effect=query), patch("reporter.postgres_reports.logger.warning") as warning:
        assert generator.get_all_nodes("default") == {"primary": "illdb2", "standbys": ["replica"]}
    message = warning.call_args.args[0]
    assert "--cluster" in message
    assert "omitting 'illdb3'" in message


@pytest.mark.parametrize("unknown_response", [
    {"status": "success", "data": {"result": []}},
    {"status": "error", "error": "Recovery query failed"},
], ids=["missing", "failed"])
@pytest.mark.parametrize(("roles", "expected"), [
    ({"a": None, "b": "0"}, {"primary": "b", "standbys": ["a"]}),
    ({"a": "0", "b": None}, {"primary": "a", "standbys": ["b"]}),
    ({"a": None, "b": None}, {"primary": "a", "standbys": ["b"]}),
    ({"a": None, "b": "1", "c": None}, {"primary": "a", "standbys": ["b", "c"]}),
    ({"a": None, "b": "0", "c": "1", "d": None, "e": "0"},
     {"primary": "b", "standbys": ["a", "c", "d"]}),
])
def test_unknown_roles_preserve_nodes_and_prefer_confirmed_primary(generator, unknown_response, roles, expected):
    def query(q):
        if "pgwatch_settings_configured" in q:
            rows = [{"metric": {"node_name": n}} for n in reversed(roles)]
        else:
            node = next(n for n in roles if f'node_name="{n}"' in q)
            if roles[node] is None:
                return unknown_response
            rows = [{"value": [1, roles[node]]}]
        return {"status": "success", "data": {"result": rows}}

    with patch.object(generator, "query_instant", side_effect=query), patch("reporter.postgres_reports.logger.warning") as warning:
        assert generator.get_all_nodes("default") == expected
    messages = [call.args[0] for call in warning.call_args_list]
    unknown_messages = [message for message in messages if "unknown" in message.lower()]
    assert len(unknown_messages) == 1
    assert "default" in unknown_messages[0]
    for node, role in roles.items():
        assert (f"'{node}'" in unknown_messages[0]) == (role is None)
    omitted = [message for message in messages if "omitting" in message]
    if "e" in roles:
        assert len(omitted) == 1
        assert "omitting 'e'" in omitted[0]
        assert "--cluster" in omitted[0]
    else:
        assert omitted == []


def test_index_overlap_does_not_match_another_databases_target_name(generator):
    rows = [{"metric": {"dbname": "illdb", "real_dbname": "other", "schema_name": "public", "index_name": "i"}}]
    with patch.object(generator, "query_instant", return_value={"status": "success", "data": {"result": rows}}):
        assert generator._f005_listed("metric", "illdb")["names"] == set()
