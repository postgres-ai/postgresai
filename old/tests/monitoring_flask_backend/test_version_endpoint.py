"""Tests for the /version endpoint and version file reading functionality."""

import pytest
import os
import sys

# Add the monitoring_flask_backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monitoring_flask_backend'))


class TestReadVersionFile:
    """Tests for the read_version_file helper function."""

    def test_read_existing_file(self, tmp_path):
        """Test reading an existing file returns stripped content."""
        from app import read_version_file

        version_file = tmp_path / "VERSION"
        version_file.write_text("1.2.3\n")

        result = read_version_file(str(version_file))
        assert result == "1.2.3"

    def test_read_file_strips_whitespace(self, tmp_path):
        """Test that whitespace is properly stripped from file content."""
        from app import read_version_file

        version_file = tmp_path / "VERSION"
        version_file.write_text("  v2.0.0  \n\n")

        result = read_version_file(str(version_file))
        assert result == "v2.0.0"

    def test_read_nonexistent_file_returns_default(self):
        """Test reading a nonexistent file returns the default value."""
        from app import read_version_file

        result = read_version_file("/nonexistent/path/VERSION")
        assert result == "unknown"

    def test_read_nonexistent_file_custom_default(self):
        """Test reading a nonexistent file with custom default."""
        from app import read_version_file

        result = read_version_file("/nonexistent/path/VERSION", default="N/A")
        assert result == "N/A"

    def test_read_empty_file(self, tmp_path):
        """Test reading an empty file returns empty string."""
        from app import read_version_file

        version_file = tmp_path / "VERSION"
        version_file.write_text("")

        result = read_version_file(str(version_file))
        assert result == ""


class TestVersionEndpoint:
    """Tests for the /version Flask endpoint."""

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        """Create a Flask test client with mocked version files."""
        # Create mock version files
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.0.0-test")

        build_ts_file = tmp_path / "BUILD_TS"
        build_ts_file.write_text("2025-01-01T00:00:00Z")

        # Need to reload the module with mocked file paths
        import app as app_module

        # Patch the module-level constants
        monkeypatch.setattr(app_module, 'APP_VERSION', '1.0.0-test')
        monkeypatch.setattr(app_module, 'APP_BUILD_TS', '2025-01-01T00:00:00Z')

        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as client:
            yield client

    def test_version_endpoint_returns_200(self, client):
        """Test that /version endpoint returns 200 status."""
        response = client.get('/version')
        assert response.status_code == 200

    def test_version_endpoint_returns_json(self, client):
        """Test that /version endpoint returns JSON content type."""
        response = client.get('/version')
        assert response.content_type == 'application/json'

    def test_version_endpoint_response_structure(self, client):
        """Test that /version response contains expected keys."""
        response = client.get('/version')
        data = response.get_json()
        
        # Expecting a list with one item
        assert isinstance(data, list)
        assert len(data) == 1
        item = data[0]

        assert 'version' in item
        assert 'build_ts' in item

    def test_version_endpoint_response_values(self, client):
        """Test that /version response contains expected values."""
        response = client.get('/version')
        data = response.get_json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        item = data[0]

        assert item['version'] == '1.0.0-test'
        assert item['build_ts'] == '2025-01-01T00:00:00Z'

    def test_version_endpoint_post_not_allowed(self, client):
        """Test that POST to /version returns 405."""
        response = client.post('/version')
        assert response.status_code == 405

    def test_version_endpoint_put_not_allowed(self, client):
        """Test that PUT to /version returns 405."""
        response = client.put('/version')
        assert response.status_code == 405

    def test_version_endpoint_delete_not_allowed(self, client):
        """Test that DELETE to /version returns 405."""
        response = client.delete('/version')
        assert response.status_code == 405


class TestVersionEndpointWithMissingFiles:
    """Tests for /version endpoint when version files are missing."""

    @pytest.fixture
    def client_with_unknown_version(self, monkeypatch):
        """Create a Flask test client with unknown version (missing files)."""
        import app as app_module

        monkeypatch.setattr(app_module, 'APP_VERSION', 'unknown')
        monkeypatch.setattr(app_module, 'APP_BUILD_TS', 'unknown')

        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as client:
            yield client

    def test_version_endpoint_with_missing_files(self, client_with_unknown_version):
        """Test /version returns 'unknown' when files don't exist."""
        response = client_with_unknown_version.get('/version')
        data = response.get_json()

        assert response.status_code == 200
        assert isinstance(data, list)
        assert len(data) > 0
        item = data[0]
        
        assert item['version'] == 'unknown'
        assert item['build_ts'] == 'unknown'
