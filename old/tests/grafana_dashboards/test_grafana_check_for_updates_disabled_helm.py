"""The Helm chart's Grafana must not show the "New version available" banner
(the compose half is tests/grafana_dashboards/test_grafana_check_for_updates_disabled.py)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
HELM_VALUES_PATH = REPO_ROOT / "old" / "postgres_ai_helm" / "values.yaml"


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
