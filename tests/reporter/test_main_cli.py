"""Tests for main function and CLI."""
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from reporter.postgres_reports import PostgresReportGenerator


@pytest.mark.unit
def test_main_exits_when_connection_fails(monkeypatch) -> None:
    """Test that main exits with code 1 when Prometheus connection fails."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
    ]

    with patch.object(sys, 'argv', test_args):
        with patch.object(PostgresReportGenerator, 'test_connection', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                from reporter import postgres_reports
                postgres_reports.main()

            assert exc_info.value.code == 1


@pytest.mark.unit
def test_main_auto_discover_clusters(monkeypatch) -> None:
    """Test that main auto-discovers all clusters when not specified."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--no-upload',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['cluster1', 'cluster2', 'cluster3']
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []
    mock_generator.pg_conn = None  # Add pg_conn attribute

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass  # Ignore normal exit

            # Should have called generate_all_reports for each cluster
            assert mock_generator.generate_all_reports.call_count == 3


@pytest.mark.unit
def test_main_uses_default_cluster_when_none_found(monkeypatch) -> None:
    """Test that main uses 'local' cluster when no clusters are discovered."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--no-upload',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = []  # No clusters found
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []
    mock_generator.pg_conn = None  # Add pg_conn attribute

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Should have generated reports for 'local' cluster
            mock_generator.generate_all_reports.assert_called()
            call_args = mock_generator.generate_all_reports.call_args
            assert call_args[0][0] == 'local'  # First arg should be 'local'


@pytest.mark.unit
def test_main_with_exclude_databases_parameter() -> None:
    """Test that --exclude-databases parameter is passed to generator."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--exclude-databases', 'test_db,staging_db,dev_db',
    ]

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator') as MockGenerator:
            mock_instance = MagicMock()
            mock_instance.test_connection.return_value = False  # Fail fast
            MockGenerator.return_value = mock_instance

            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Check that generator was created with excluded_databases
            call_args = MockGenerator.call_args
            excluded_dbs = call_args[0][2]  # Third positional arg
            assert excluded_dbs == ['test_db', 'staging_db', 'dev_db']


@pytest.mark.unit
def test_main_with_use_current_time_parameter() -> None:
    """Test that --use-current-time parameter is passed to generator."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--use-current-time',
    ]

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator') as MockGenerator:
            mock_instance = MagicMock()
            mock_instance.test_connection.return_value = False  # Fail fast
            MockGenerator.return_value = mock_instance

            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Check that generator was created with use_current_time=True
            call_kwargs = MockGenerator.call_args[1]
            assert call_kwargs['use_current_time'] is True


@pytest.mark.unit
def test_main_with_specific_check_id() -> None:
    """Test running main with a specific check ID."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--check-id', 'H002',
        '--no-upload',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']
    mock_generator.generate_h002_unused_indexes_report.return_value = {
        'check_id': 'H002',
        'results': {}
    }

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Should have called specific check generator
            mock_generator.generate_h002_unused_indexes_report.assert_called()


@pytest.mark.unit
def test_main_with_no_combine_nodes_flag() -> None:
    """Test that --no-combine-nodes sets combine_nodes=False."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--no-combine-nodes',
        '--no-upload',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Should have called generate_all_reports with combine_nodes=False
            # combine_nodes is 3rd positional argument (cluster, node_name, combine_nodes)
            call_args = mock_generator.generate_all_reports.call_args[0]
            assert call_args[2] is False  # combine_nodes parameter


@pytest.mark.unit
def test_main_skips_upload_when_report_creation_fails() -> None:
    """Test that main skips uploads when create_report returns None."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--api-url', 'https://api.test',
        '--token', 'test-token',
        '--project-name', 'some-project',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['cluster1']
    mock_generator.create_report.return_value = None  # Report creation fails
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Should have tried to create report
            mock_generator.create_report.assert_called()

            # generate_per_query_jsons should be called without api_url/token/report_id
            call_kwargs = mock_generator.generate_per_query_jsons.call_args[1]
            assert call_kwargs['api_url'] is None
            assert call_kwargs['report_id'] is None


@pytest.mark.unit
def test_main_with_specific_cluster() -> None:
    """Test running main with --cluster parameter."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--cluster', 'production',
        '--no-upload',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # Should NOT call get_all_clusters
            mock_generator.get_all_clusters.assert_not_called()

            # Should have called generate_all_reports for 'production'
            call_args = mock_generator.generate_all_reports.call_args
            assert call_args[0][0] == 'production'


@pytest.mark.unit
def test_main_skips_upload_when_project_name_missing() -> None:
    """Without --project-name, upload is skipped (no cluster-name fallback).

    The hardcoded/cluster-name project default was removed: a project name is
    required to upload, so create_report must NOT be called.
    """
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--cluster', 'my-cluster',
        '--api-url', 'https://api.test',
        '--token', 'test-token',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.create_report.return_value = 'report-123'
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # No project name -> no upload attempt.
            mock_generator.create_report.assert_not_called()


def test_main_uses_explicit_project_name() -> None:
    """An explicit --project-name is passed verbatim to create_report (no fallback)."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--cluster', 'my-cluster',
        '--api-url', 'https://api.test',
        '--token', 'test-token',
        '--project-name', 'explicit-project',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.create_report.return_value = 'report-123'
    mock_generator.generate_all_reports.return_value = {}
    mock_generator.generate_per_query_jsons.return_value = []

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            try:
                from reporter import postgres_reports
                postgres_reports.main()
            except SystemExit:
                pass

            # create_report should be called with the explicit project name
            # (NOT the cluster name).
            call_args = mock_generator.create_report.call_args[0]
            project_name = call_args[2]  # Third arg
            assert project_name == 'explicit-project'


@pytest.mark.unit
@pytest.mark.parametrize("check_id,method_name", [
    ("A002", "generate_a002_version_report"),
    ("A003", "generate_a003_settings_report"),
    ("A004", "generate_a004_cluster_report"),
    ("A007", "generate_a007_altered_settings_report"),
    ("H001", "generate_h001_invalid_indexes_report"),
    ("H002", "generate_h002_unused_indexes_report"),
    ("H004", "generate_h004_redundant_indexes_report"),
    ("K001", "generate_k001_query_calls_report"),
    ("K003", "generate_k003_top_queries_report"),
    ("K004", "generate_k004_temp_bytes_report"),
    ("K005", "generate_k005_wal_bytes_report"),
    ("K006", "generate_k006_shared_read_report"),
    ("K007", "generate_k007_shared_hit_report"),
    ("K008", "generate_k008_shared_hit_read_report"),
    ("M001", "generate_m001_mean_time_report"),
    ("M002", "generate_m002_rows_report"),
    ("M003", "generate_m003_io_time_report"),
    ("N001", "generate_n001_wait_events_report"),
])
def test_main_generates_specific_check_types(check_id: str, method_name: str) -> None:
    """Test that main correctly calls generator for specific check types."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--check-id', check_id,
        '--no-upload',
        '--output', '/tmp/test.json',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']

    # Mock the specific generate method
    mock_method = MagicMock(return_value={'check_id': check_id, 'results': {}})
    setattr(mock_generator, method_name, mock_method)

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            with patch('builtins.open', create=True) as mock_open:
                try:
                    from reporter import postgres_reports
                    postgres_reports.main()
                except SystemExit:
                    pass

                # Should have called the specific method
                mock_method.assert_called_once()


@pytest.mark.unit
def test_main_with_check_id_d004_generates_from_a003() -> None:
    """Test that D004 is generated from A003 when using specific check."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--check-id', 'D004',
        '--no-upload',
        '--output', '/tmp/test.json',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']
    mock_generator.generate_a003_settings_report.return_value = {
        'check_id': 'A003',
        'results': {}
    }
    mock_generator.generate_d004_from_a003.return_value = {
        'check_id': 'D004',
        'results': {}
    }

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            with patch('builtins.open', create=True):
                try:
                    from reporter import postgres_reports
                    postgres_reports.main()
                except SystemExit:
                    pass

                # Should generate A003 first, then D004 from it
                mock_generator.generate_a003_settings_report.assert_called_once()
                mock_generator.generate_d004_from_a003.assert_called_once()


@pytest.mark.unit
def test_main_with_check_id_f001_generates_from_a003() -> None:
    """Test that F001 is generated from A003 when using specific check."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--check-id', 'F001',
        '--no-upload',
        '--output', '/tmp/test.json',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']
    mock_generator.generate_a003_settings_report.return_value = {
        'check_id': 'A003',
        'results': {}
    }
    mock_generator.generate_f001_from_a003.return_value = {
        'check_id': 'F001',
        'results': {}
    }

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            with patch('builtins.open', create=True):
                try:
                    from reporter import postgres_reports
                    postgres_reports.main()
                except SystemExit:
                    pass

                # Should generate A003 first, then F001 from it
                mock_generator.generate_a003_settings_report.assert_called_once()
                mock_generator.generate_f001_from_a003.assert_called_once()


@pytest.mark.unit
def test_main_with_check_id_g001_generates_from_a003() -> None:
    """Test that G001 is generated from A003 when using specific check."""
    test_args = [
        'postgres_reports.py',
        '--prometheus-url', 'http://prom.test',
        '--postgres-sink-url', 'postgresql://user@host:5432/db',
        '--check-id', 'G001',
        '--no-upload',
        '--output', '/tmp/test.json',
    ]

    mock_generator = MagicMock(spec=PostgresReportGenerator)
    mock_generator.pg_conn = None
    mock_generator.test_connection.return_value = True
    mock_generator.get_all_clusters.return_value = ['local']
    mock_generator.generate_a003_settings_report.return_value = {
        'check_id': 'A003',
        'results': {}
    }
    mock_generator.generate_g001_from_a003.return_value = {
        'check_id': 'G001',
        'results': {}
    }

    with patch.object(sys, 'argv', test_args):
        with patch('reporter.postgres_reports.PostgresReportGenerator', return_value=mock_generator):
            with patch('builtins.open', create=True):
                try:
                    from reporter import postgres_reports
                    postgres_reports.main()
                except SystemExit:
                    pass

                # Should generate A003 first, then G001 from it
                mock_generator.generate_a003_settings_report.assert_called_once()
                mock_generator.generate_g001_from_a003.assert_called_once()
