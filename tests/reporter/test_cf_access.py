"""Cloudflare Access service-token headers on the box's platform calls (internal#354).

A preview platform sits behind Cloudflare Access; a box there gets the
service token in CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET. With both set,
every call to the platform API carries them; otherwise nothing changes.
"""
import io
from unittest.mock import MagicMock, patch

import pytest
import requests

from reporter import postgres_reports, token_renew

API_URL = "https://api.example.test/api/general"
HEADERS = {"CF-Access-Client-Id": "id-123.access", "CF-Access-Client-Secret": "secret-456"}


def _ok(body):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    return response


@pytest.fixture
def cf_env(monkeypatch):
    monkeypatch.setenv("REPORTER_API_URL", API_URL)
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id-123.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "secret-456")


@pytest.mark.unit
def test_report_upload_calls_carry_the_access_headers(cf_env):
    with patch.object(postgres_reports.requests, "post", return_value=_ok({"id": 1})) as post:
        postgres_reports.make_request(API_URL, "/rpc/checkup_report_status_update", {"report_id": 1})
    assert post.call_args.kwargs["headers"] == HEADERS
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.unit
def test_token_renewal_carries_the_access_headers(cf_env, monkeypatch):
    monkeypatch.setattr("sys.argv", ["token_renew", API_URL])
    monkeypatch.setattr("sys.stdin", io.StringIO("tok\n"))
    with patch.object(token_renew.requests, "post", return_value=_ok({"expires_at": "x"})) as post:
        assert token_renew.main() == 0
    assert post.call_args.kwargs["headers"] == HEADERS
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.unit
@pytest.mark.parametrize("only", ["CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET", None])
@pytest.mark.parametrize("value", ["x", "", " \t "])
def test_without_both_values_no_access_headers(monkeypatch, only, value):
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    if only:
        monkeypatch.setenv(only, value)
    with patch.object(postgres_reports.requests, "post", return_value=_ok({})) as post:
        postgres_reports.make_request(API_URL, "/rpc/checkup_report_create", {})
    assert not post.call_args.kwargs.get("headers")


@pytest.mark.unit
@pytest.mark.parametrize("url", [
    "http://api.example.test/api/general",
    "https://other.example.test/api/general",
    "https://api.example.test:444/api/general",
    "https://api.example.test@evil.test/api/general",
])
@pytest.mark.parametrize("caller", ["report", "renew"])
def test_access_headers_are_refused_for_other_origins(cf_env, monkeypatch, url, caller):
    with patch.object(postgres_reports.requests, "post", return_value=_ok({"expires_at": "x"})) as post:
        if caller == "report":
            postgres_reports.make_request(url, "/rpc/checkup_report_create", {})
        else:
            monkeypatch.setattr("sys.argv", ["token_renew", url])
            monkeypatch.setattr("sys.stdin", io.StringIO("tok\n"))
            assert token_renew.main() == 0
    assert post.call_args.kwargs["headers"] == {}


@pytest.mark.unit
def test_access_origin_ignores_host_case_path_and_default_port(cf_env):
    url = "https://API.EXAMPLE.TEST:443/another/path"
    with patch.object(postgres_reports.requests, "post", return_value=_ok({})) as post:
        postgres_reports.make_request(url, "/rpc/checkup_report_create", {})
    assert post.call_args.kwargs["headers"] == HEADERS


@pytest.mark.unit
@pytest.mark.parametrize("configured, url, expected", [
    ("https://API.EXAMPLE.TEST:8443/base", "https://api.example.test:8443/path", HEADERS),
    ("https://api.example.test:8443/base", API_URL, {}),
    ("http://api.example.test/base", API_URL, {}),
    ("https://api.example.test:invalid/base", API_URL, {}),
])
def test_access_headers_use_the_configured_origin(cf_env, monkeypatch, configured, url, expected):
    monkeypatch.setenv("REPORTER_API_URL", configured)
    with patch.object(postgres_reports.requests, "post", return_value=_ok({})) as post:
        postgres_reports.make_request(url, "/rpc/checkup_report_create", {})
    assert post.call_args.kwargs["headers"] == expected


@pytest.mark.unit
@pytest.mark.parametrize("configured", [None, ""])
def test_unset_api_url_trusts_only_the_production_default(cf_env, monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("REPORTER_API_URL")
    else:
        monkeypatch.setenv("REPORTER_API_URL", configured)
    with patch.object(postgres_reports.requests, "post", return_value=_ok({})) as post:
        postgres_reports.make_request(API_URL, "/rpc/checkup_report_create", {})
        assert post.call_args.kwargs["headers"] == {}
        postgres_reports.make_request("https://postgres.ai/api/general", "/rpc/checkup_report_create", {})
        assert post.call_args.kwargs["headers"] == HEADERS


@pytest.mark.unit
@pytest.mark.parametrize("caller", ["report", "renew"])
@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_platform_redirect_is_a_failure(cf_env, monkeypatch, caller, status):
    response = _ok({"expires_at": "x"})
    response.status_code = status
    with patch.object(postgres_reports.requests, "post", return_value=response) as post:
        if caller == "report":
            with pytest.raises(requests.HTTPError):
                postgres_reports.make_request(API_URL, "/rpc/checkup_report_create", {})
        else:
            monkeypatch.setattr("sys.argv", ["token_renew", API_URL])
            monkeypatch.setattr("sys.stdin", io.StringIO("tok\n"))
            assert token_renew.main() == 1
    assert post.call_args.kwargs["allow_redirects"] is False
