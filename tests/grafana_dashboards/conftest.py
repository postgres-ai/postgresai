"""Shared fixtures for Grafana dashboard JSON lint tests.

Each test that takes `dashboard_path` (or the `dashboards` fixture) runs over every dashboard
JSON file in DASHBOARD_DIR: the SQL dashboards that query pgai-mon through the Infinity
datasource. They are compiled (mon/grafana/compile.py) from the PromQL dashboards in
old/config/grafana/dashboards, which keep titles, units, legends, overrides and variables, so a
convention broken in a source dashboard fails here once it is recompiled.

A few files still read a source dashboard by its own path, because what they check is not in the
SQL dashboards: the host row and Dashboard 99 against the host-metric providers
(old/tests/grafana_dashboards/test_host_row.py, test_host_stats_dashboard.py), and the PromQL
helpers in query_info_join.py, which old/tests/compliance_vectors applies to the PromQL Dashboard 2.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The dashboards the lint tests run against: the one place to change when they move.
DASHBOARD_DIR = REPO_ROOT / "mon" / "grafana" / "dashboards"

# Where the SQL dashboards send their queries (mon/grafana/compile.py INFINITY and --target-url).
INFINITY_TYPE = "yesoreyeram-infinity-datasource"
INFINITY_UID = "pgai-mon"
PGAI_MON_SQL_URL = "http://pgai-mon:8765/sql?format=objects"

# Matches a reference to the db_name template variable in any Grafana syntax,
# with or without a format spec ($db_name, ${db_name}, ${db_name:regex}),
# while rejecting an unrelated name that merely starts the same
# (${db_name_source}).
DB_NAME_VAR_PATTERN = re.compile(r"\$(?:db_name\b|\{db_name(?::\w+)?\})")


def _dashboard_paths() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(DASHBOARD_DIR / "*.json")))


def dashboard_paths() -> list[Path]:
    return _dashboard_paths()


def dashboard_file(number: int) -> Path:
    """The dashboard numbered `number` (Dashboard_<number>_...json) in DASHBOARD_DIR."""
    hits = [p for p in _dashboard_paths() if is_dashboard(p, number)]
    assert len(hits) == 1, f"expected one Dashboard_{number}_*.json in {DASHBOARD_DIR}, found {hits}"
    return hits[0]


def is_dashboard(path: Path, number: int) -> bool:
    return path.name.startswith(f"Dashboard_{number}_")


def target_query(target: dict) -> str:
    """The query text a target sends: the SQL body of an Infinity POST, else PromQL or raw SQL."""
    body = (target.get("url_options") or {}).get("data")
    return body or target.get("expr") or target.get("rawSql") or ""


def call_args(sql: str, call: str) -> list[str]:
    """The top-level arguments of the first `call(` in `sql`."""
    i = sql.index(call + "(") + len(call) + 1
    args, depth, start, quote = [], 0, i, False
    for j in range(i, len(sql)):
        c = sql[j]
        if c == "'":
            quote = not quote
        elif quote:
            continue
        elif c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                args.append(sql[start:j].strip())
                return args
            depth -= 1
        elif c == "," and depth == 0:
            args.append(sql[start:j].strip())
            start = j + 1
    raise ValueError(f"unbalanced {call}(")


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
