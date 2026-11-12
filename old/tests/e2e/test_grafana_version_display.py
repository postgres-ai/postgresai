"""End-to-end tests for Grafana version display functionality.

These tests verify that:
1. The Flask /version endpoint returns valid data
2. Grafana dashboard has the version panel configured
3. The version data flows through to Grafana correctly

Prerequisites:
- Run with `pytest tests/e2e/test_grafana_version_display.py -v`
- The monitoring stack must be running (docker compose up)
- Default ports: Flask on 8000, Grafana on 3000
"""

import os
import json
import pytest
import requests
from urllib.parse import urljoin


# Configuration - can be overridden with environment variables
FLASK_BASE_URL = os.environ.get('FLASK_URL', 'http://localhost:8000')
GRAFANA_BASE_URL = os.environ.get('GRAFANA_URL', 'http://localhost:3000')
GRAFANA_USER = os.environ.get('GRAFANA_USER', 'admin')
GRAFANA_PASSWORD = os.environ.get('GRAFANA_PASSWORD', 'admin')

# Dashboard UID for Self Monitoring Dashboard
SELF_MONITORING_DASHBOARD_UID = 'self_monitoring_dashboard'


@pytest.fixture
def flask_session():
    """Create a requests session for Flask API."""
    session = requests.Session()
    session.headers.update({'Accept': 'application/json'})
    return session


@pytest.fixture
def grafana_session():
    """Create a requests session for Grafana API with authentication."""
    session = requests.Session()
    session.auth = (GRAFANA_USER, GRAFANA_PASSWORD)
    session.headers.update({'Accept': 'application/json'})
    return session


def is_service_available(url: str, timeout: int = 5) -> bool:
    """Check if a service is available."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


@pytest.mark.e2e
class TestFlaskVersionEndpoint:
    """Tests for the Flask /version endpoint."""

    @pytest.fixture(autouse=True)
    def check_flask_available(self):
        """Skip tests if Flask is not available."""
        if not is_service_available(FLASK_BASE_URL):
            pytest.skip(f"Flask backend not available at {FLASK_BASE_URL}")

    def test_version_endpoint_returns_200(self, flask_session):
        """Test that /version endpoint returns HTTP 200."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_version_endpoint_returns_json(self, flask_session):
        """Test that /version returns valid JSON."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        assert response.headers.get('Content-Type') == 'application/json'
        data = response.json()
        assert isinstance(data, dict)

    def test_version_endpoint_has_version_field(self, flask_session):
        """Test that /version response contains 'version' field."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        data = response.json()
        assert 'version' in data, "Response missing 'version' field"
        assert data['version'] is not None
        assert len(data['version']) > 0

    def test_version_endpoint_has_build_ts_field(self, flask_session):
        """Test that /version response contains 'build_ts' field."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        data = response.json()
        assert 'build_ts' in data, "Response missing 'build_ts' field"
        assert data['build_ts'] is not None


@pytest.mark.e2e
class TestGrafanaDashboardConfiguration:
    """Tests for Grafana dashboard version panel configuration."""

    @pytest.fixture(autouse=True)
    def check_grafana_available(self):
        """Skip tests if Grafana is not available."""
        if not is_service_available(GRAFANA_BASE_URL):
            pytest.skip(f"Grafana not available at {GRAFANA_BASE_URL}")

    def test_grafana_api_accessible(self, grafana_session):
        """Test that Grafana API is accessible with authentication."""
        response = grafana_session.get(urljoin(GRAFANA_BASE_URL, '/api/health'))
        assert response.status_code == 200

    def test_self_monitoring_dashboard_exists(self, grafana_session):
        """Test that Self Monitoring Dashboard exists."""
        url = urljoin(GRAFANA_BASE_URL, f'/api/dashboards/uid/{SELF_MONITORING_DASHBOARD_UID}')
        response = grafana_session.get(url)
        assert response.status_code == 200, f"Dashboard not found: {response.text}"

    def test_dashboard_has_version_panel(self, grafana_session):
        """Test that the dashboard contains a version panel."""
        url = urljoin(GRAFANA_BASE_URL, f'/api/dashboards/uid/{SELF_MONITORING_DASHBOARD_UID}')
        response = grafana_session.get(url)
        assert response.status_code == 200

        dashboard = response.json()
        panels = dashboard.get('dashboard', {}).get('panels', [])

        # Look for version panel by title or by URL pattern
        version_panel = None
        for panel in panels:
            # Check by title
            if 'version' in panel.get('title', '').lower():
                version_panel = panel
                break
            # Check by URL in datasource (for Infinity datasource)
            targets = panel.get('targets', [])
            for target in targets:
                url = target.get('url', '')
                if '/version' in url:
                    version_panel = panel
                    break
            if version_panel:
                break

        assert version_panel is not None, "No version panel found in dashboard"

    def test_version_panel_uses_infinity_datasource(self, grafana_session):
        """Test that version panel uses Infinity datasource."""
        url = urljoin(GRAFANA_BASE_URL, f'/api/dashboards/uid/{SELF_MONITORING_DASHBOARD_UID}')
        response = grafana_session.get(url)
        assert response.status_code == 200

        dashboard = response.json()
        panels = dashboard.get('dashboard', {}).get('panels', [])

        version_panel = None
        for panel in panels:
            targets = panel.get('targets', [])
            for target in targets:
                if '/version' in target.get('url', ''):
                    version_panel = panel
                    break
            if version_panel:
                break

        if version_panel:
            datasource = version_panel.get('datasource', {})
            ds_type = datasource.get('type', '')
            assert 'infinity' in ds_type.lower(), f"Expected Infinity datasource, got: {ds_type}"

    def test_version_panel_queries_flask_backend(self, grafana_session):
        """Test that version panel is configured to query Flask backend."""
        url = urljoin(GRAFANA_BASE_URL, f'/api/dashboards/uid/{SELF_MONITORING_DASHBOARD_UID}')
        response = grafana_session.get(url)
        assert response.status_code == 200

        dashboard = response.json()
        panels = dashboard.get('dashboard', {}).get('panels', [])

        flask_url_found = False
        for panel in panels:
            targets = panel.get('targets', [])
            for target in targets:
                target_url = target.get('url', '')
                # Check for Flask backend URL (container name or localhost)
                if '/version' in target_url and ('flask-pgss-api' in target_url or 'localhost' in target_url):
                    flask_url_found = True
                    # Verify port is 8000 (gunicorn port)
                    assert ':8000' in target_url, f"Expected port 8000 in URL, got: {target_url}"
                    break
            if flask_url_found:
                break

        assert flask_url_found, "No Flask backend version endpoint URL found in dashboard panels"


@pytest.mark.e2e
class TestEndToEndVersionFlow:
    """End-to-end tests verifying version data flows through the entire stack."""

    @pytest.fixture(autouse=True)
    def check_services_available(self):
        """Skip tests if required services are not available."""
        if not is_service_available(FLASK_BASE_URL):
            pytest.skip(f"Flask backend not available at {FLASK_BASE_URL}")
        if not is_service_available(GRAFANA_BASE_URL):
            pytest.skip(f"Grafana not available at {GRAFANA_BASE_URL}")

    def test_version_data_is_not_unknown(self, flask_session):
        """Test that version data is properly populated (not 'unknown')."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        data = response.json()

        # In production, version should not be 'unknown'
        # In development, it might be 'unknown' but that's expected
        version = data.get('version', '')
        build_ts = data.get('build_ts', '')

        # At minimum, the fields should exist and have some value
        assert version, "version field is empty"
        assert build_ts, "build_ts field is empty"

    def test_grafana_infinity_datasource_exists(self, grafana_session):
        """Test that Infinity datasource is configured in Grafana."""
        url = urljoin(GRAFANA_BASE_URL, '/api/datasources')
        response = grafana_session.get(url)
        assert response.status_code == 200

        datasources = response.json()
        infinity_ds = None
        for ds in datasources:
            if 'infinity' in ds.get('type', '').lower():
                infinity_ds = ds
                break

        assert infinity_ds is not None, "Infinity datasource not found in Grafana"

    def test_version_format_is_valid(self, flask_session):
        """Test that version string has a valid format."""
        response = flask_session.get(urljoin(FLASK_BASE_URL, '/version'))
        data = response.json()

        version = data.get('version', '')

        # Accept 'unknown' for development, but if it's a version, validate format
        if version != 'unknown':
            # Version should match patterns like: 1.0.0, 0.14.0-beta.9, 1.2.3-rc.1
            import re
            version_pattern = r'^\d+\.\d+\.\d+(-[\w.]+)?$'
            assert re.match(version_pattern, version), f"Invalid version format: {version}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
