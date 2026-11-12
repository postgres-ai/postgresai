"""Tests for AWS Managed Prometheus (AMP) authentication in get_prometheus_client()."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the monitoring_flask_backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monitoring_flask_backend'))


class TestAMPAuthentication:
    """Tests for AMP authentication configuration."""

    def test_amp_disabled_by_default(self, monkeypatch):
        """Test that AMP is disabled when ENABLE_AMP is not set."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)

        # Need to patch at module level before importing
        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {'boto3': MagicMock(), 'requests_aws4auth': MagicMock()}):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                # Force reimport
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                client = app.get_prometheus_client()

                # PrometheusConnect called with disable_ssl=True and auth=None
                mock_prom_connect.assert_called_once()
                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['disable_ssl'] is True
                assert call_kwargs['auth'] is None

    def test_amp_enabled_creates_aws4auth(self, monkeypatch):
        """Test that enabling AMP creates AWS4Auth with refreshable credentials."""
        monkeypatch.setenv('ENABLE_AMP', 'true')
        monkeypatch.setenv('AWS_REGION', 'us-west-2')

        mock_credentials = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.get_credentials.return_value = mock_credentials

        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value = mock_session_instance

        mock_aws4auth_class = MagicMock()
        mock_aws4auth_instance = MagicMock()
        mock_aws4auth_class.return_value = mock_aws4auth_instance

        mock_requests_aws4auth = MagicMock()
        mock_requests_aws4auth.AWS4Auth = mock_aws4auth_class

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {
            'boto3': mock_boto3,
            'requests_aws4auth': mock_requests_aws4auth
        }):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                # Force reimport
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                client = app.get_prometheus_client()

                # Verify boto3 session was created and credentials retrieved
                mock_boto3.Session.assert_called_once()
                mock_session_instance.get_credentials.assert_called_once()

                # Verify AWS4Auth was created with correct params
                mock_aws4auth_class.assert_called_once_with(
                    region='us-west-2',
                    service='aps',
                    refreshable_credentials=mock_credentials,
                )

                # Verify PrometheusConnect was called with SSL enabled and auth
                mock_prom_connect.assert_called_once()
                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['disable_ssl'] is False
                assert call_kwargs['auth'] is mock_aws4auth_instance

    def test_amp_enabled_no_credentials_graceful(self, monkeypatch):
        """Test that AMP handles missing credentials gracefully."""
        monkeypatch.setenv('ENABLE_AMP', 'true')

        mock_session_instance = MagicMock()
        mock_session_instance.get_credentials.return_value = None  # No credentials

        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value = mock_session_instance

        mock_aws4auth_class = MagicMock()
        mock_requests_aws4auth = MagicMock()
        mock_requests_aws4auth.AWS4Auth = mock_aws4auth_class

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {
            'boto3': mock_boto3,
            'requests_aws4auth': mock_requests_aws4auth
        }):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                client = app.get_prometheus_client()

                # AWS4Auth should not be created when credentials are None
                mock_aws4auth_class.assert_not_called()

                # PrometheusConnect still called but with auth=None
                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] is None

    def test_amp_default_region(self, monkeypatch):
        """Test that default region is us-east-1 when AWS_REGION not set."""
        monkeypatch.setenv('ENABLE_AMP', 'true')
        monkeypatch.delenv('AWS_REGION', raising=False)

        mock_credentials = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.get_credentials.return_value = mock_credentials

        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value = mock_session_instance

        mock_aws4auth_class = MagicMock()
        mock_requests_aws4auth = MagicMock()
        mock_requests_aws4auth.AWS4Auth = mock_aws4auth_class

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {
            'boto3': mock_boto3,
            'requests_aws4auth': mock_requests_aws4auth
        }):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                client = app.get_prometheus_client()

                # Verify default region is us-east-1
                call_kwargs = mock_aws4auth_class.call_args[1]
                assert call_kwargs['region'] == 'us-east-1'
