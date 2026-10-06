"""Cloudflare Access service-token headers on the box's platform calls (internal#354).

A preview platform sits behind Cloudflare Access; a box there gets the
service token in CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET. With both set,
every call to the platform API carries them; otherwise nothing changes.
"""
import io
from unittest.mock import MagicMock, patch

import pytest

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
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id-123.access")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "secret-456")


@pytest.mark.unit
def test_report_upload_calls_carry_the_access_headers(cf_env):
    with patch.object(postgres_reports.requests, "post", return_value=_ok({"id": 1})) as post:
        postgres_reports.make_request(API_URL, "/rpc/checkup_report_status_update", {"report_id": 1})
    assert post.call_args.kwargs["headers"] == HEADERS


@pytest.mark.unit
def test_token_renewal_carries_the_access_headers(cf_env, monkeypatch):
    monkeypatch.setattr("sys.argv", ["token_renew", API_URL])
    monkeypatch.setattr("sys.stdin", io.StringIO("tok\n"))
    with patch.object(token_renew.requests, "post", return_value=_ok({"expires_at": "x"})) as post:
        assert token_renew.main() == 0
    assert post.call_args.kwargs["headers"] == HEADERS


@pytest.mark.unit
@pytest.mark.parametrize("only", ["CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET", None])
def test_without_both_values_no_access_headers(monkeypatch, only):
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    if only:
        monkeypatch.setenv(only, "x")
    with patch.object(postgres_reports.requests, "post", return_value=_ok({})) as post:
        postgres_reports.make_request(API_URL, "/rpc/checkup_report_create", {})
    assert not post.call_args.kwargs.get("headers")
