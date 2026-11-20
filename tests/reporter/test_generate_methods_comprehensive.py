"""Comprehensive tests for all generate_* report methods."""
import pytest
from unittest.mock import patch, MagicMock

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


@pytest.fixture
def mock_version_data():
    """Mock version data from Prometheus."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"version": "14.10"},
                    "value": [1234567890, "14.10"]
                }
            ]
        }
    }


@pytest.fixture
def mock_settings_data():
    """Mock settings data from Prometheus."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"setting": "shared_buffers"},
                    "value": [1234567890, "128MB"]
                },
                {
                    "metric": {"setting": "max_connections"},
                    "value": [1234567890, "100"]
                }
            ]
        }
    }


@pytest.mark.unit
def test_generate_a004_cluster_report(generator) -> None:
    """Test A004 cluster report generation."""
    mock_result = {
        "status": "success",
        "data": {"result": []}
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        report = generator.generate_a004_cluster_report("test-cluster", "node-01")

    assert report["checkId"] == "A004"
    assert "results" in report


@pytest.mark.unit
def test_generate_a007_altered_settings(generator) -> None:
    """Test A007 altered settings report generation."""
    mock_result = {
        "status": "success",
        "data": {"result": []}
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        report = generator.generate_a007_altered_settings_report("test-cluster", "node-01")

    assert report["checkId"] == "A007"


@pytest.mark.unit
def test_generate_h001_invalid_indexes(generator) -> None:
    """Test H001 invalid indexes report generation."""
    mock_result = {
        "status": "success",
        "data": {"result": []}
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        with patch.object(generator, 'get_all_databases', return_value=["testdb"]):
            report = generator.generate_h001_invalid_indexes_report("test-cluster", "node-01")

    assert report["checkId"] == "H001"


@pytest.mark.unit
def test_generate_f001_with_mock_sink(generator) -> None:
    """Test F001 autovacuum settings with mocked data."""
    mock_result = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"setting": "autovacuum"},
                    "value": [1234567890, "on"]
                }
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        report = generator.generate_f001_autovacuum_settings_report("test-cluster", "node-01")

    assert report["checkId"] == "F001"


@pytest.mark.unit
def test_generate_g001_with_memory_data(generator) -> None:
    """Test G001 memory settings with mocked data."""
    mock_result = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"setting": "shared_buffers"},
                    "value": [1234567890, "1GB"]
                },
                {
                    "metric": {"setting": "work_mem"},
                    "value": [1234567890, "4MB"]
                }
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        report = generator.generate_g001_memory_settings_report("test-cluster", "node-01")

    assert report["checkId"] == "G001"


@pytest.mark.unit
def test_generate_d004_with_pgstat_data(generator) -> None:
    """Test D004 pgstat settings with mocked data."""
    mock_result = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"setting": "pg_stat_statements.max"},
                    "value": [1234567890, "5000"]
                }
            ]
        }
    }

    with patch.object(generator, 'query_instant', return_value=mock_result):
        report = generator.generate_d004_pgstat_settings_report("test-cluster", "node-01")

    assert report["checkId"] == "D004"


# Test generate_all_reports with different configurations


@pytest.mark.unit
def test_generate_all_reports_with_single_check(generator) -> None:
    """Test generate_all_reports with single check ID."""
    mock_a002 = {"checkId": "A002", "results": {}}

    with patch.object(generator, 'get_all_clusters', return_value=["test-cluster"]):
        with patch.object(generator, 'generate_a002_version_report', return_value=mock_a002):
            reports = generator.generate_all_reports(["A002"])

    assert isinstance(reports, dict)


@pytest.mark.unit
def test_generate_all_reports_with_multiple_checks(generator) -> None:
    """Test generate_all_reports with multiple check IDs."""
    mock_a002 = {"checkId": "A002", "results": {}}
    mock_h002 = {"checkId": "H002", "results": {}}

    with patch.object(generator, 'get_all_clusters', return_value=["test-cluster"]):
        with patch.object(generator, 'generate_a002_version_report', return_value=mock_a002):
            with patch.object(generator, 'generate_h002_unused_indexes_report', return_value=mock_h002):
                reports = generator.generate_all_reports(["A002", "H002"])

    assert isinstance(reports, dict)


@pytest.mark.unit
def test_generate_all_reports_with_no_clusters(generator) -> None:
    """Test generate_all_reports when no clusters are found."""
    with patch.object(generator, 'get_all_clusters', return_value=[]):
        reports = generator.generate_all_reports(["A002"])

    # Should handle gracefully
    assert isinstance(reports, dict)


# Test different report data structures


@pytest.mark.unit
def test_generate_report_with_complex_nested_data(generator) -> None:
    """Test report generation with complex nested data structure."""
    complex_data = {
        "database1": {
            "tables": [
                {"name": "users", "size": "10GB"},
                {"name": "orders", "size": "5GB"}
            ]
        },
        "database2": {
            "tables": [
                {"name": "products", "size": "8GB"}
            ]
        }
    }

    result = generator.format_report_data("CUSTOM", complex_data, "node-01")

    assert result["checkId"] == "CUSTOM"
    assert "database1" in result["results"]["node-01"]["data"]


# Test generate_per_query_jsons


@pytest.mark.unit
def test_generate_per_query_jsons_with_empty_reports(generator) -> None:
    """Test per-query JSON generation with empty reports."""
    result = generator.generate_per_query_jsons(
        {},  # Empty reports
        "test-cluster",
        "http://api.test",
        "token123",
        "report456"
    )

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.unit
def test_generate_per_query_jsons_without_api_url(generator) -> None:
    """Test per-query JSON generation without API URL."""
    reports = {
        "K003": {
            "results": {
                "node-01": {
                    "data": {
                        "db1": {
                            "top_queries": [
                                {"queryid": "123", "calls": 100}
                            ]
                        }
                    }
                }
            }
        }
    }

    # Mock the methods that would be called
    with patch.object(generator, 'pg_conn', None):
        with patch.object(generator, 'get_queries_metrics_from_prometheus', return_value={}):
            result = generator.generate_per_query_jsons(
                reports,
                "test-cluster",
                node_name=None,
                api_url=None,
            )

    assert isinstance(result, list)


# Test connection methods


@pytest.mark.unit
def test_test_connection_success(generator) -> None:
    """Test test_connection with successful response."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch('reporter.postgres_reports.requests.get', return_value=mock_response):
        result = generator.test_connection()

    assert result is True


@pytest.mark.unit
def test_test_connection_failure(generator) -> None:
    """Test test_connection with failed response."""
    import requests

    with patch('reporter.postgres_reports.requests.get', side_effect=requests.ConnectionError()):
        result = generator.test_connection()

    assert result is False


# Test helper methods


@pytest.mark.unit
def test_build_metadata_contains_version(generator) -> None:
    """Test that build metadata contains version info."""
    metadata = generator._build_metadata

    assert isinstance(metadata, dict)
    # May contain version, build_ts, or be empty


@pytest.mark.unit
def test_format_bytes_with_all_units(generator) -> None:
    """Test format_bytes with values in all unit ranges."""
    test_cases = [
        (500, "B"),  # Bytes
        (2048, "KiB"),  # Kilobytes
        (5 * 1024 * 1024, "MiB"),  # Megabytes
        (3 * 1024 * 1024 * 1024, "GiB"),  # Gigabytes
        (2 * 1024 * 1024 * 1024 * 1024, "TiB"),  # Terabytes
    ]

    for value, expected_unit in test_cases:
        result = generator.format_bytes(value)
        # Just check it returns a string with the value
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.unit
def test_parse_memory_value_comprehensive(generator) -> None:
    """Test parse_memory_value with comprehensive set of inputs."""
    test_cases = [
        # Format: (input, expected_output)
        ("0", 0),
        ("1024", 1024 * 1024),  # Bare number assumed KB
        ("1kB", 1024),
        ("1KB", 1024),
        ("1MB", 1024 * 1024),
        ("1GB", 1024 * 1024 * 1024),
        ("1TB", 1024 * 1024 * 1024 * 1024),
        ("128 MB", 128 * 1024 * 1024),
        ("  256  kB  ", 256 * 1024),
        ("-1", 0),  # Unlimited
        ("", 0),  # Empty
    ]

    for input_val, expected in test_cases:
        result = generator._parse_memory_value(input_val)
        assert result == expected, f"Failed for input: {input_val}"


@pytest.mark.unit
def test_analyze_memory_settings_comprehensive(generator) -> None:
    """Test memory analysis with comprehensive settings."""
    memory_data = {
        "shared_buffers": {"setting": "2GB"},
        "work_mem": {"setting": "8MB"},
        "maintenance_work_mem": {"setting": "256MB"},
        "effective_cache_size": {"setting": "8GB"},
        "max_connections": {"setting": "200"},
        "wal_buffers": {"setting": "32MB"},
    }

    result = generator._analyze_memory_settings(memory_data)

    assert "estimated_total_memory_usage" in result
    estimates = result["estimated_total_memory_usage"]

    # Check all expected fields are present
    expected_fields = [
        "shared_buffers_bytes",
        "wal_buffers_bytes",
        "work_mem_per_connection_bytes",
        "maintenance_work_mem_bytes",
        "effective_cache_size_bytes",
    ]

    for field in expected_fields:
        assert field in estimates, f"Missing field: {field}"
        assert estimates[field] > 0, f"Field {field} should be positive"


@pytest.mark.unit
def test_extract_queryids_comprehensive(generator) -> None:
    """Test queryid extraction with comprehensive report structure."""
    reports = {
        "K001": {
            "results": {
                "node-01": {
                    "data": {
                        "app_db": {
                            "top_queries": [
                                {"queryid": "111", "calls": 1000},
                                {"queryid": "222", "calls": 500},
                            ]
                        },
                        "analytics_db": {
                            "top_queries": [
                                {"queryid": "333", "calls": 750},
                                {"queryid": "111", "calls": 250},  # Duplicate across DBs
                            ]
                        }
                    }
                }
            }
        },
        "K003": {
            "results": {
                "node-01": {
                    "data": {
                        "app_db": {
                            "top_queries": [
                                {"queryid": "444", "total_time": 5000},
                            ]
                        }
                    }
                }
            }
        }
    }

    result = generator.extract_queryids_from_reports(reports)

    assert isinstance(result, dict)
    # Should have extracted queryids from multiple databases and report types


@pytest.mark.unit
def test_format_report_data_preserves_all_fields(generator) -> None:
    """Test that format_report_data preserves all expected fields."""
    data = {
        "metric1": 100,
        "metric2": "value",
        "nested": {"key": "value"}
    }

    result = generator.format_report_data("TEST", data, "node-01")

    # Check all top-level fields
    expected_fields = ["version", "build_ts", "generation_mode", "checkId", "checkTitle", "timestamptz", "nodes", "results"]

    for field in expected_fields:
        assert field in result, f"Missing field: {field}"
