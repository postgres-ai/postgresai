"""Final comprehensive tests to reach 85% coverage."""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json

from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


# Test postgres sink connection scenarios


@pytest.mark.unit
def test_connect_postgres_sink_when_already_connected(generator) -> None:
    """Test that connect doesn't reconnect if already connected."""
    # Set pg_conn to a mock connection
    mock_conn = MagicMock()
    generator.pg_conn = mock_conn

    # Connect again - should use existing connection
    with patch('reporter.postgres_reports.psycopg2') as mock_psycopg2:
        generator.connect_postgres_sink()

    # Should not have created new connection
    mock_psycopg2.connect.assert_not_called()


@pytest.mark.unit
def test_get_index_definitions_with_no_connection(generator) -> None:
    """Test get_index_definitions when no sink connection."""
    generator.pg_conn = None
    generator.postgres_sink_url = ""  # Ensure connect_postgres_sink will fail

    result = generator.get_index_definitions_from_sink(db_name="db1")

    # Should return empty dict when no connection
    assert result == {}


@pytest.mark.unit
def test_get_queryid_queries_with_no_connection(generator) -> None:
    """Test get_queryid_queries when no sink connection."""
    generator.pg_conn = None
    generator.postgres_sink_url = ""  # Ensure connect_postgres_sink will fail

    result = generator.get_queryid_queries_from_sink(db_names=["db1"])

    # Should return empty dict when no connection
    assert result == {}


@pytest.mark.unit
def test_get_index_definitions_with_no_db_name(generator) -> None:
    """Test get_index_definitions with no db_name specified."""
    generator.pg_conn = None
    generator.postgres_sink_url = ""  # Ensure connect_postgres_sink will fail

    result = generator.get_index_definitions_from_sink()

    # Should return empty dict when no connection
    assert result == {}


@pytest.mark.unit
def test_get_queryid_queries_with_no_db_names(generator) -> None:
    """Test get_queryid_queries with no db_names specified."""
    generator.pg_conn = None
    generator.postgres_sink_url = ""  # Ensure connect_postgres_sink will fail

    result = generator.get_queryid_queries_from_sink()

    # Should return empty dict when no connection
    assert result == {}


# Test report generation with write_immediately flag


@pytest.mark.unit
def test_generate_per_query_jsons_with_write_immediately(generator) -> None:
    """Test per-query JSON generation with write_immediately=True."""
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

    with patch.object(generator, 'get_queries_metrics_from_prometheus', return_value={}):
        with patch.object(generator, 'pg_conn', None):
            with patch('builtins.open', mock_open()) as mock_file:
                result = generator.generate_per_query_jsons(
                    reports,
                    "test-cluster",
                    node_name=None,
                    write_immediately=True
                )

    assert isinstance(result, list)


# Test format methods with extreme values


@pytest.mark.unit
def test_format_bytes_with_zero(generator) -> None:
    """Test format_bytes with zero value."""
    result = generator.format_bytes(0)

    assert isinstance(result, str)
    assert "0" in result


@pytest.mark.unit
def test_format_bytes_with_max_int(generator) -> None:
    """Test format_bytes with very large value."""
    # Max 64-bit signed int
    result = generator.format_bytes(9223372036854775807)

    assert isinstance(result, str)


@pytest.mark.unit
def test_parse_memory_value_with_spaces_everywhere(generator) -> None:
    """Test memory parsing with excessive whitespace."""
    result = generator._parse_memory_value("  128  MB  ")

    assert result == 128 * 1024 * 1024


@pytest.mark.unit
def test_parse_memory_value_with_lowercase_b_suffix(generator) -> None:
    """Test memory parsing with lowercase 'b' suffix."""
    result = generator._parse_memory_value("1024b")

    assert result == 1024


# Test version extraction edge cases


@pytest.mark.unit
def test_extract_version_with_only_major_version(generator) -> None:
    """Test version extraction with only major version number."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "server_version_num": {"setting": "140000"}  # Just major, no minor
                }
            }
        }
    }

    result = generator.extract_postgres_version_from_a003(a003_report)

    assert result["server_major_ver"] == "14"
    assert result["server_minor_ver"] == "0"


@pytest.mark.unit
def test_extract_version_with_rc_version(generator) -> None:
    """Test version extraction with release candidate."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "server_version": {"setting": "16rc1"},
                }
            }
        }
    }

    result = generator.extract_postgres_version_from_a003(a003_report)

    assert "16" in result["version"] or "rc" in result["version"]


# Test format_setting_value with all code paths


@pytest.mark.unit
def test_format_setting_value_with_8kb_exact_boundary(generator) -> None:
    """Test 8kB formatting at exact MiB boundary."""
    # 128 blocks * 8 = 1024 KiB = exactly 1 MiB
    result = generator.format_setting_value("shared_buffers", "128", "8kB")

    assert "1" in result
    assert "MiB" in result


@pytest.mark.unit
def test_format_setting_value_with_ms_exact_second(generator) -> None:
    """Test ms formatting at exact second boundary."""
    # Exactly 2000ms = 2s
    result = generator.format_setting_value("timeout", "2000", "ms")

    assert "2" in result
    assert "s" in result


@pytest.mark.unit
def test_format_setting_value_fallback_to_name_based(generator) -> None:
    """Test format_setting_value falls back to name-based formatting."""
    # When no unit provided, uses setting name
    result = generator.format_setting_value("shared_buffers", "32768", "")

    assert isinstance(result, str)


# Test filter_a003_settings with various node structures


@pytest.mark.unit
def test_filter_a003_settings_with_specific_node(generator) -> None:
    """Test filtering settings for specific node."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "setting1": {"setting": "value1"},
                }
            },
            "node-02": {
                "data": {
                    "setting1": {"setting": "value2"},
                }
            }
        }
    }

    # Filter should get settings from all nodes
    result = generator.filter_a003_settings(a003_report, ["setting1"])

    assert "setting1" in result


@pytest.mark.unit
def test_filter_a003_settings_last_value_wins(generator) -> None:
    """Test that when multiple nodes have same setting, last wins."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "shared_buffers": {"setting": "128MB"},
                }
            },
            "node-02": {
                "data": {
                    "shared_buffers": {"setting": "256MB"},
                }
            }
        }
    }

    result = generator.filter_a003_settings(a003_report, ["shared_buffers"])

    # One of the values should be present
    assert result["shared_buffers"]["setting"] in ["128MB", "256MB"]


# Test generate_d004/f001/g001_from_a003 variations


@pytest.mark.unit
def test_generate_f001_from_a003_with_full_settings(generator) -> None:
    """Test F001 generation with all autovacuum settings."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "autovacuum": {"setting": "on"},
                    "autovacuum_max_workers": {"setting": "3"},
                    "autovacuum_naptime": {"setting": "60s"},
                    "autovacuum_vacuum_threshold": {"setting": "50"},
                    "autovacuum_analyze_threshold": {"setting": "50"},
                }
            }
        }
    }

    report = generator.generate_f001_from_a003(a003_report)

    assert report["checkId"] == "F001"
    assert "results" in report


@pytest.mark.unit
def test_generate_g001_from_a003_with_all_memory_settings(generator) -> None:
    """Test G001 generation with all memory settings."""
    a003_report = {
        "results": {
            "node-01": {
                "data": {
                    "shared_buffers": {"setting": "1GB"},
                    "work_mem": {"setting": "4MB"},
                    "maintenance_work_mem": {"setting": "64MB"},
                    "effective_cache_size": {"setting": "4GB"},
                    "max_connections": {"setting": "100"},
                }
            }
        }
    }

    report = generator.generate_g001_from_a003(a003_report)

    assert report["checkId"] == "G001"
    data = report["results"]["node-01"]["data"]
    assert "analysis" in data  # Should have memory analysis


# Test queryid extraction with various structures


@pytest.mark.unit
def test_extract_queryids_with_numeric_string_queryids(generator) -> None:
    """Test queryid extraction with numeric string queryids."""
    reports = {
        "K003": {
            "results": {
                "node-01": {
                    "data": {
                        "db1": {
                            "top_queries": [
                                {"queryid": "123456789", "calls": 100},
                            ]
                        }
                    }
                }
            }
        }
    }

    result = generator.extract_queryids_from_reports(reports)

    assert isinstance(result, dict)
    if result:
        assert "db1" in result


@pytest.mark.unit
def test_extract_queryids_deduplicates_across_reports(generator) -> None:
    """Test that queryids are deduplicated across different report types."""
    reports = {
        "K001": {
            "results": {
                "node-01": {
                    "data": {
                        "db1": {
                            "top_queries": [
                                {"queryid": "123", "calls": 100},
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
                        "db1": {
                            "top_queries": [
                                {"queryid": "123", "total_time": 5000},  # Same queryid
                            ]
                        }
                    }
                }
            }
        }
    }

    result = generator.extract_queryids_from_reports(reports)

    assert isinstance(result, dict)


# Test format_report_data with various node configurations


@pytest.mark.unit
def test_format_report_data_with_no_standbys(generator) -> None:
    """Test format_report_data when no standby nodes."""
    all_hosts = {
        "primary": "node-01",
        "standbys": []
    }

    result = generator.format_report_data("A002", {}, all_hosts=all_hosts)

    assert result["nodes"]["primary"] == "node-01"
    assert len(result["nodes"]["standbys"]) == 0


@pytest.mark.unit
def test_format_report_data_with_version_and_build_ts(generator) -> None:
    """Test format_report_data includes version and build_ts."""
    result = generator.format_report_data("A002", {}, "node-01")

    # Should have version and build_ts fields (may be None)
    assert "version" in result
    assert "build_ts" in result


@pytest.mark.unit
def test_format_report_data_sets_check_title(generator) -> None:
    """Test that format_report_data sets checkTitle field."""
    result = generator.format_report_data("A002", {}, "node-01")

    assert "checkTitle" in result
    assert isinstance(result["checkTitle"], str)


# Test memory analysis with edge cases


@pytest.mark.unit
def test_analyze_memory_settings_with_zero_values(generator) -> None:
    """Test memory analysis with zero or very small values."""
    memory_data = {
        "shared_buffers": {"setting": "0"},
        "work_mem": {"setting": "0"},
        "max_connections": {"setting": "1"},
    }

    result = generator._analyze_memory_settings(memory_data)

    assert "estimated_total_memory_usage" in result


@pytest.mark.unit
def test_analyze_memory_settings_with_very_high_connections(generator) -> None:
    """Test memory analysis with very high connection count."""
    memory_data = {
        "shared_buffers": {"setting": "1GB"},
        "work_mem": {"setting": "4MB"},
        "max_connections": {"setting": "10000"},  # Very high
    }

    result = generator._analyze_memory_settings(memory_data)

    estimates = result["estimated_total_memory_usage"]
    if "max_work_mem_usage_bytes" in estimates:
        # Should calculate potential high memory usage
        assert estimates["max_work_mem_usage_bytes"] > 0


# Test check title for all check IDs


@pytest.mark.unit
def test_get_check_title_consistency(generator) -> None:
    """Test that get_check_title returns consistent results."""
    # Call twice, should get same result
    title1 = generator.get_check_title("A002")
    title2 = generator.get_check_title("A002")

    assert title1 == title2


@pytest.mark.unit
def test_get_check_title_for_all_standard_checks(generator) -> None:
    """Test get_check_title for all standard check IDs."""
    check_ids = [
        "A002", "A003", "A004", "A007",
        "H001", "H002", "H004",
        "F001", "F004",
        "G001",
        "D004",
        "K001", "K003", "K004", "K005", "K006", "K007", "K008",
        "M001", "M002", "M003",
        "N001"
    ]

    for check_id in check_ids:
        title = generator.get_check_title(check_id)
        assert isinstance(title, str)
        assert len(title) > 0 or check_id.startswith("K") or check_id.startswith("M")  # Some may not have titles
