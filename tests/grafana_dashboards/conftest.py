"""Shared fixtures for Grafana dashboard JSON lint tests.

Each test parametrizes over every dashboard JSON file found in:
  - config/grafana/dashboards/
  - postgres_ai_helm/config/grafana/dashboards/

The two trees must be kept in sync — every fix that lands in one MUST land in
the other. The lint tests therefore run against BOTH and fail if either
diverges.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DASHBOARD_DIRS = (
    REPO_ROOT / "config" / "grafana" / "dashboards",
    REPO_ROOT / "postgres_ai_helm" / "config" / "grafana" / "dashboards",
)


def _dashboard_paths() -> list[Path]:
    paths: list[Path] = []
    for d in DASHBOARD_DIRS:
        if not d.exists():
            continue
        paths.extend(sorted(Path(p) for p in glob.glob(str(d / "*.json"))))
    return paths


def dashboard_paths() -> list[Path]:
    return _dashboard_paths()


def iter_panels(dashboard: dict) -> Iterable[dict]:
    """Yield every panel in a dashboard, descending into collapsed row panels."""
    for p in dashboard.get("panels", []) or []:
        yield p
        # Row panels can contain nested panels (collapsed rows)
        for nested in p.get("panels", []) or []:
            yield nested


@pytest.fixture(scope="session")
def dashboards() -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for path in _dashboard_paths():
        with open(path) as f:
            out.append((path, json.load(f)))
    return out


def pytest_generate_tests(metafunc):
    """Parametrize tests that take a `dashboard_path` arg over every dashboard."""
    if "dashboard_path" in metafunc.fixturenames:
        paths = _dashboard_paths()
        ids = [os.path.relpath(p, REPO_ROOT) for p in paths]
        metafunc.parametrize("dashboard_path", paths, ids=ids)
