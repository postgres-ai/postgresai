"""Renew this monitoring box's API token (postgres-ai/internal#354).

A box token lives 90 days. The daemon loop in config/scripts/postgres-reports.sh
runs this once a cycle, and each renewal moves the expiry to 90 days from now.
The token comes on stdin, never argv. A failed renewal is logged and the cycle
goes on: the token stays valid until it expires.

Usage: printf '%s\\n' "$token" | python -m reporter.token_renew <api_url>
"""
import sys

import requests

from reporter.cf_access import cf_access_headers


def main() -> int:
    api_url = sys.argv[1]
    token = sys.stdin.readline().strip()
    url = api_url + "/rpc/monitoring_instance_token_renew"
    try:
        response = requests.post(
            url,
            json={"access_token": token},
            headers=cf_access_headers(url),
            timeout=30,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.HTTPError("Platform API redirected the request", response=response)
        if response.status_code == 403:
            # A key set by hand (a member's or the org's), not one the platform
            # minted for this box: nothing to renew.
            print("postgres-reports: api_key is not a monitoring box token; not renewed")
            return 0
        response.raise_for_status()
        expires_at = response.json()["expires_at"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        print(f"postgres-reports: WARNING token renewal failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"postgres-reports: token renewed, expires at {expires_at}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
