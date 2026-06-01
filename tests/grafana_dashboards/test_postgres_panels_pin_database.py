"""Bug 1 — Postgres data-source panels must pin the database via ${db_name}.

The Grafana Postgres data source `PGWatch-PostgreSQL` does not declare a
default database in every deployment (the provisioned default is editable
and historically has been blanked out in some environments — including the
rc.6 demo). When a panel issues a raw SQL query without explicitly
selecting the database, Grafana renders a silent "No data" panel with a
small pink/magenta warning triangle in the header (no red border, no title
in the error).

Hover tooltip on the failing panel reads verbatim:

    You do not currently have a default database configured for this data
    source. Postgres requires a default database with which to connect.
    Please configure one through the Data Sources Configuration page, or
    if you are using a provisioning file, update that configuration file
    with a default database.

The robust fix: every Postgres-source panel must reference the dashboard's
`db_name` template variable in its `rawSql` (or a future `database`
target field). The other panels in these dashboards already follow this
pattern via Prometheus `datname="$db_name"` labels.

This test enumerates every panel across every dashboard JSON and asserts
the rule. Originally introduced as a RED test against
Dashboard_3_Single_query_analysis.json panel id=18 (the query-text panel).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import iter_panels

# The Postgres data source UID provisioned in
# config/grafana/provisioning/datasources/datasources.yml
POSTGRES_DS_UID = "P031DD592934B2F1F"
POSTGRES_DS_TYPE = "postgres"

# Accept either a templated ${db_name} substitution OR an explicit database
# pinned in the target via a `database` field (older Grafana schemas).
DB_NAME_TOKENS = ("${db_name}", "$db_name")


def _is_postgres_target(panel: dict, target: dict) -> bool:
    # Target-level datasource wins; otherwise inherit from panel
    ds = target.get("datasource") or panel.get("datasource") or {}
    if not isinstance(ds, dict):
        return False
    return (
        ds.get("type") == POSTGRES_DS_TYPE
        or ds.get("uid") == POSTGRES_DS_UID
    )


def _target_pins_database(target: dict) -> bool:
    raw_sql = target.get("rawSql") or ""
    if any(tok in raw_sql for tok in DB_NAME_TOKENS):
        return True
    # Some Grafana schemas pin via an explicit `database` target field.
    db = target.get("database")
    if isinstance(db, str) and db.strip():
        return True
    return False


def test_postgres_panels_pin_database_via_db_name(dashboard_path: Path):
    """Every panel that targets the Postgres data source must explicitly
    reference the ${db_name} template variable in its rawSql (or pin a
    `database` on the target). Otherwise Grafana renders "No data"
    silently when the provisioned data source has no default database.
    """
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    offenders: list[str] = []
    for panel in iter_panels(dashboard):
        for ti, target in enumerate(panel.get("targets", []) or []):
            if not _is_postgres_target(panel, target):
                continue
            if _target_pins_database(target):
                continue
            offenders.append(
                f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                f"target[{ti}] refId={target.get('refId')!r} "
                f"rawSql={(target.get('rawSql') or '')[:120]!r}"
            )

    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} Postgres-source panel "
        f"target(s) do not pin the database via ${{db_name}}:\n  "
        + "\n  ".join(offenders)
    )
