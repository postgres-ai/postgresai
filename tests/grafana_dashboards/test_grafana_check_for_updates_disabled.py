"""Bug 6 — Grafana must not show "New version available" banner on demo.

By default the embedded Grafana phones home and renders a "New version
available" banner at the top of every page. On the rc.6 demo this is
visible at the top of dashboard 1 and is visually distracting.

Suppress it via Grafana's [analytics] check_for_updates = false. There
are two deployment paths and both must be configured:

  - docker-compose: config/grafana/provisioning/grafana.ini
  - Helm: postgres_ai_helm/values.yaml under `grafana.grafana.ini`
"""
from __future__ import annotations

import configparser
import io
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

GRAFANA_INI_PATH = (
    REPO_ROOT / "config" / "grafana" / "provisioning" / "grafana.ini"
)
HELM_VALUES_PATH = REPO_ROOT / "postgres_ai_helm" / "values.yaml"


def _parse_ini(text: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False)
    parser.read_file(io.StringIO(text))
    return parser


def test_compose_grafana_ini_disables_update_check():
    """The docker-compose grafana.ini must have:

        [analytics]
        check_for_updates = false
    """
    assert GRAFANA_INI_PATH.exists(), (
        f"missing {GRAFANA_INI_PATH.relative_to(REPO_ROOT)}"
    )
    parser = _parse_ini(GRAFANA_INI_PATH.read_text())
    assert parser.has_section("analytics"), (
        f"{GRAFANA_INI_PATH.relative_to(REPO_ROOT)}: missing [analytics] "
        f"section — the 'New version available' banner will show on demo."
    )
    val = parser.get("analytics", "check_for_updates", fallback=None)
    assert val is not None, (
        f"{GRAFANA_INI_PATH.relative_to(REPO_ROOT)}: [analytics] section "
        f"missing check_for_updates key."
    )
    assert val.strip().lower() == "false", (
        f"{GRAFANA_INI_PATH.relative_to(REPO_ROOT)}: "
        f"[analytics] check_for_updates = {val!r}, expected 'false'."
    )


def test_helm_values_disables_update_check():
    """The Helm chart must pass an equivalent setting to its Grafana subchart
    via grafana.grafana.ini.analytics.check_for_updates = false.

    The grafana subchart renders any keys under `grafana.grafana.ini` as
    sections in the generated grafana.ini ConfigMap.
    """
    assert HELM_VALUES_PATH.exists()
    values = yaml.safe_load(HELM_VALUES_PATH.read_text()) or {}
    grafana_block = values.get("grafana") or {}
    ini_block = grafana_block.get("grafana.ini") or {}
    analytics = ini_block.get("analytics") or {}
    cfu = analytics.get("check_for_updates")
    assert cfu is False, (
        f"postgres_ai_helm/values.yaml: "
        f"grafana.grafana.ini.analytics.check_for_updates is {cfu!r}, "
        f"expected literal `false`. Without this the Helm-deployed Grafana "
        f"shows the 'New version available' banner."
    )
