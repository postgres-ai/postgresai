"""Tests for reporter.token_renew: the box renews its 90-day token (internal#354)."""
import io
from unittest.mock import MagicMock, patch

import pytest
import requests

from reporter import token_renew

API_URL = "https://api.example.test/api/general"


def _response(status: int, body=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} Client Error", response=response)
    return response


def _run(monkeypatch, response):
    monkeypatch.setattr("sys.argv", ["token_renew", API_URL])
    monkeypatch.setattr("sys.stdin", io.StringIO("secret-token-123\n"))
    with patch.object(token_renew.requests, "post", return_value=response) as post:
        code = token_renew.main()
    return code, post


@pytest.mark.unit
def test_renews_with_the_token_in_the_body(monkeypatch, capsys):
    code, post = _run(monkeypatch, _response(200, {"expires_at": "2027-01-03T00:00:00+00:00"}))

    assert code == 0
    post.assert_called_once_with(
        API_URL + "/rpc/monitoring_instance_token_renew",
        json={"access_token": "secret-token-123"},
        headers={},
        timeout=30,
        allow_redirects=False,
    )
    out = capsys.readouterr()
    assert "2027-01-03T00:00:00+00:00" in out.out
    assert "secret-token-123" not in out.out + out.err


@pytest.mark.unit
def test_a_token_that_is_not_a_box_token_is_not_an_error(monkeypatch, capsys):
    code, _ = _run(monkeypatch, _response(403))

    assert code == 0
    assert "not a monitoring box token" in capsys.readouterr().out


@pytest.mark.unit
def test_a_refused_renewal_warns(monkeypatch, capsys):
    code, _ = _run(monkeypatch, _response(401))

    assert code == 1
    out = capsys.readouterr()
    assert "token renewal failed" in out.err
    assert "secret-token-123" not in out.out + out.err
