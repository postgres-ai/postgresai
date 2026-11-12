"""The provisioned pgai-mon datasource is the one the SQL dashboards name, with bearer auth.

config/grafana/provisioning/datasources/pgai-mon.yml is loaded by Grafana together with every
other file in that directory, so its name and uid must not collide with datasources.yml; the
dashboards' targets must name its uid; Infinity must send the bearer token, taken from the
environment or a file (never a literal), and only to the host the targets call.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import yaml

from tests.grafana_dashboards.conftest import (
    INFINITY_TYPE,
    INFINITY_UID,
    PGAI_MON_SQL_URL,
    REPO_ROOT,
    dashboard_paths,
    iter_panels,
)

DATASOURCES_DIR = REPO_ROOT / "config" / "grafana" / "provisioning" / "datasources"
PGAI_MON_FILE = DATASOURCES_DIR / "pgai-mon.yml"
# Grafana's provisioning expansions: $VAR, ${VAR}, $__env{VAR}, $__file{path}
SECRET_REFERENCE = re.compile(r"^\$(\{\w+\}|\w+|__env\{\w+\}|__file\{[^}]+\})$")


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _datasource():
    sources = _load(PGAI_MON_FILE)["datasources"]
    assert len(sources) == 1
    return sources[0]


def test_it_is_the_datasource_the_dashboards_name():
    ds = _datasource()
    assert (ds["type"], ds["uid"], ds["access"]) == (INFINITY_TYPE, INFINITY_UID, "proxy")
    used = {json.dumps(t.get("datasource"), sort_keys=True)
            for path in dashboard_paths() for p in iter_panels(json.loads(path.read_text()))
            for t in p.get("targets") or []}
    assert used == {json.dumps({"type": INFINITY_TYPE, "uid": INFINITY_UID}, sort_keys=True)}


def test_bearer_token_from_the_environment_or_a_file():
    ds = _datasource()
    assert ds["jsonData"]["auth_method"] == "bearerToken"
    token = ds["secureJsonData"]["bearerToken"]
    assert SECRET_REFERENCE.match(token), f"bearerToken must reference an env var or file, not {token!r}"


def test_token_is_sent_only_to_the_host_the_targets_call():
    url = urlsplit(PGAI_MON_SQL_URL)
    assert _datasource()["jsonData"]["allowedHosts"] == [f"{url.scheme}://{url.netloc}"]


def test_reprovisioned_and_read_only_like_the_others():
    doc = _load(PGAI_MON_FILE)
    ds = _datasource()
    assert {"name": ds["name"], "orgId": ds["orgId"]} in doc["deleteDatasources"]
    assert ds["editable"] is False and ds["isDefault"] is False


def test_no_collision_with_the_other_provisioning_files():
    seen_names, seen_uids = {}, {}
    for path in sorted(DATASOURCES_DIR.glob("*.y*ml")):
        for ds in _load(path).get("datasources") or []:
            assert ds["name"] not in seen_names, (path.name, seen_names.get(ds["name"]))
            assert ds["uid"] not in seen_uids, (path.name, seen_uids.get(ds["uid"]))
            seen_names[ds["name"]], seen_uids[ds["uid"]] = path.name, path.name
    assert {"pgai-mon.yml", "datasources.yml"} <= set(seen_uids.values())


def test_secret_reference_pattern():
    for ok in ("${PGAI_MON_TOKEN}", "$PGAI_MON_TOKEN", "$__env{PGAI_MON_TOKEN}", "$__file{/run/secrets/t}"):
        assert SECRET_REFERENCE.match(ok), ok
    for bad in ("demo-token", "", "Bearer ${PGAI_MON_TOKEN}"):
        assert not SECRET_REFERENCE.match(bad), bad
