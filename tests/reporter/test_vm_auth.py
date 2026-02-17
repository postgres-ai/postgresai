"""Tests for VictoriaMetrics Basic Auth in PostgresReportGenerator."""

import pytest
from unittest.mock import MagicMock, patch

from reporter import postgres_reports as postgres_reports_module


class TestReporterVMBasicAuth:
    """Tests for VM Basic Auth in PostgresReportGenerator."""

    def test_vm_auth_used_when_env_vars_set(self, monkeypatch):
        """Test that VM_AUTH_USERNAME/PASSWORD produce a Basic Auth tuple."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.setenv('VM_AUTH_PASSWORD', 'secret123')

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3:
            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            assert generator.auth == ('vmauth', 'secret123')

    def test_vm_auth_not_used_when_env_vars_empty(self, monkeypatch):
        """Test that empty VM_AUTH vars result in auth=None."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', '')
        monkeypatch.setenv('VM_AUTH_PASSWORD', '')

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3:
            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            assert generator.auth is None

    def test_vm_auth_not_used_when_env_vars_missing(self, monkeypatch):
        """Test that missing VM_AUTH vars result in auth=None."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.delenv('VM_AUTH_USERNAME', raising=False)
        monkeypatch.delenv('VM_AUTH_PASSWORD', raising=False)

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3:
            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            assert generator.auth is None

    def test_vm_auth_not_used_when_only_username_set(self, monkeypatch):
        """Test that VM auth requires both username and password."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.delenv('VM_AUTH_PASSWORD', raising=False)

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3:
            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            assert generator.auth is None

    def test_amp_takes_precedence_over_vm_auth(self, monkeypatch):
        """Test that AMP auth takes precedence over VM Basic Auth."""
        monkeypatch.setenv('ENABLE_AMP', 'true')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.setenv('VM_AUTH_PASSWORD', 'secret123')

        mock_credentials = MagicMock()
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_auth = MagicMock()

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3, \
             patch.object(postgres_reports_module, 'AWS4Auth', return_value=mock_auth):
            mock_boto3.Session.return_value = mock_session

            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            # AMP auth should win, not VM basic auth tuple
            assert generator.auth is mock_auth
            assert generator.auth != ('vmauth', 'secret123')

    def test_vm_auth_passed_to_requests(self, monkeypatch):
        """Test that VM basic auth tuple is passed to requests.get calls."""
        monkeypatch.delenv('ENABLE_AMP', raising=False)
        monkeypatch.setenv('VM_AUTH_USERNAME', 'vmauth')
        monkeypatch.setenv('VM_AUTH_PASSWORD', 'secret123')

        captured_kwargs = {}

        class DummyResponse:
            status_code = 200

        def fake_get(url, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyResponse()

        with patch.object(postgres_reports_module, 'boto3') as mock_boto3, \
             patch.object(postgres_reports_module.requests, 'get', fake_get):

            generator = postgres_reports_module.PostgresReportGenerator(
                prometheus_url="http://prom.test",
                postgres_sink_url="",
            )

            generator.test_connection()

            assert captured_kwargs.get('auth') == ('vmauth', 'secret123')
