"""Cloudflare Access service-token headers for the box's platform calls (internal#354).

A preview platform sits behind Cloudflare Access. A box there gets the service
token in CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET; with both set, HTTPS
calls to the configured platform origin send them.
"""
import os
from urllib.parse import urlsplit


def _https_origin(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return None
        port = parsed.port if parsed.port is not None else 443
        return parsed.scheme, parsed.hostname.lower(), port
    except ValueError:
        return None


def cf_access_headers(url) -> dict:
    client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return {}
    trusted_url = os.environ.get("REPORTER_API_URL") or "https://postgres.ai/api/general"
    origin = _https_origin(url)
    if origin is None or origin != _https_origin(trusted_url):
        return {}
    return {"CF-Access-Client-Id": client_id, "CF-Access-Client-Secret": client_secret}
