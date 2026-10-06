"""Cloudflare Access service-token headers for the box's platform calls (internal#354).

A preview platform sits behind Cloudflare Access. A box there gets the service
token in CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET; with both set, every
platform call sends them. Otherwise (production) no header is added.
"""
import os


def cf_access_headers() -> dict:
    client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return {}
    return {"CF-Access-Client-Id": client_id, "CF-Access-Client-Secret": client_secret}
