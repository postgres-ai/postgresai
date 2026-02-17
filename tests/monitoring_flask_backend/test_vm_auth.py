"""Tests for VictoriaMetrics Basic Auth in get_prometheus_client()."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch


# Add the monitoring_flask_backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monitoring_flask_backend'))


class TestVMBasicAuth:
    """Tests for VictoriaMetrics Basic Auth configuration."""

    def test_vm_auth_used_when_env_vars_set(self, monkeypatch):
        """Test that VM_AUTH_USERNAME/PASSWORD produce a Basic Auth tuple."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.setenv('VM_AUTH_PASSWORD', 'secret123')

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {'boto3': MagicMock(), 'requests_aws4auth': MagicMock()}):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                app.get_prometheus_client()

                mock_prom_connect.assert_called_once()
                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] == ('vmauth', 'secret123')
                assert call_kwargs['disable_ssl'] is True

    def test_vm_auth_not_used_when_env_vars_empty(self, monkeypatch):
        """Test that empty VM_AUTH vars result in auth=None."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', '')
        monkeypatch.setenv('VM_AUTH_PASSWORD', '')

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {'boto3': MagicMock(), 'requests_aws4auth': MagicMock()}):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                app.get_prometheus_client()

                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] is None

    def test_vm_auth_not_used_when_env_vars_missing(self, monkeypatch):
        """Test that missing VM_AUTH vars result in auth=None."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.delenv('VM_AUTH_USERNAME', raising=False)
        monkeypatch.delenv('VM_AUTH_PASSWORD', raising=False)

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {'boto3': MagicMock(), 'requests_aws4auth': MagicMock()}):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                app.get_prometheus_client()

                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] is None

    def test_vm_auth_not_used_when_only_username_set(self, monkeypatch):
        """Test that VM auth requires both username and password."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.delenv('VM_AUTH_PASSWORD', raising=False)

        mock_prom_connect = MagicMock()

        with patch.dict('sys.modules', {'boto3': MagicMock(), 'requests_aws4auth': MagicMock()}):
            with patch('prometheus_api_client.PrometheusConnect', mock_prom_connect):
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                app.get_prometheus_client()

                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] is None

    def test_amp_takes_precedence_over_vm_auth(self, monkeypatch):
        """Test that AMP auth takes precedence over VM Basic Auth."""
        monkeypatch.setenv('ENABLE_AMP', 'true')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.setenv('VM_AUTH_PASSWORD', 'secret123')

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
                if 'app' in sys.modules:
                    del sys.modules['app']

                import app
                app.get_prometheus_client()

                # AMP auth should win, not VM basic auth tuple
                call_kwargs = mock_prom_connect.call_args[1]
                assert call_kwargs['auth'] is mock_aws4auth_instance
                assert call_kwargs['auth'] != ('vmauth', 'secret123')
