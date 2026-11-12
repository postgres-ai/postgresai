"""Old stack: config guards carried over from the CLI tests when the old stack moved to old/.

The live versions (mon/metrics.yml, config/grafana/provisioning/datasources/pgai-mon.yml,
mon/grafana/dashboards) are in cli/test/config-consistency.test.ts,
cli/test/system-schema-filter.test.ts and cli/test/checkup.integration.test.ts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "old/config"

DATASOURCE_UIDS = {
    "PGWatch-PostgreSQL": "P031DD592934B2F1F",
    "PGWatch-Prometheus": "P7A0D6631BB10B34F",
    "Infinity": "aerffb0z8rjlsc",
}

NOT_IN_SYSTEM_SCHEMAS = re.compile(r"\bnot in \('pg_catalog', 'information_schema', 'pg_toast'\)")
NOT_MATCH_TEMP_SCHEMAS = re.compile(r"!~ '\^pg_\(toast_\)\?temp_'")
NO_LIKE_ESCAPE = re.compile(r"not like 'pg\\_")


def _datasources() -> dict:
    return yaml.safe_load((OLD / "grafana/provisioning/datasources/datasources.yml").read_text())


def _org(org) -> int:
    return 1 if org is None else int(org)


@pytest.mark.unit
def test_datasources_delete_entries_match_stable_non_editable_datasources() -> None:
    cfg = _datasources()
    deleted = {(d["name"], _org(d.get("orgId"))) for d in cfg["deleteDatasources"]}
    sources = {(d["name"], _org(d.get("orgId"))): d for d in cfg["datasources"]}
    assert deleted == set(sources)
    for (name, _), ds in sources.items():
        assert ds["uid"] == DATASOURCE_UIDS[name], name
        assert ds["editable"] is False, name


@pytest.mark.unit
def test_postgres_datasource_pins_connection_database_in_json_data() -> None:
    # The Postgres plugin's SQL editor reads jsonData.database (postgresai#314, F-1).
    ds = next(d for d in _datasources()["datasources"] if d["name"] == "PGWatch-PostgreSQL")
    assert ds["jsonData"]["database"] == "measurements"
    assert ds["database"] == "measurements"


@pytest.mark.unit
def test_dashboard_pgwatch_metric_references_exist_in_metrics_yml() -> None:
    names = sorted(yaml.safe_load((ROOT / "mon/metrics.yml").read_text())["metrics"], key=len, reverse=True)
    generated = {"query_info"}  # exported by old/monitoring_flask_backend/app.py
    unknown, seen = set(), 0

    def strings(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("expr", "definition", "query") and isinstance(v, str):
                    yield v
                yield from strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from strings(v)

    for path in sorted((OLD / "grafana/dashboards").glob("*.json")):
        for text in strings(json.loads(path.read_text())):
            for ref in re.findall(r"\bpgwatch_([a-zA-Z_:][a-zA-Z0-9_:]*)\b", text):
                seen += 1
                if ref in generated:
                    continue
                if not any(ref == n or ref.startswith(n + "_") for n in names):
                    unknown.add(f"{path.name}: pgwatch_{ref}")
    assert seen > 0
    assert sorted(unknown) == []


@pytest.mark.unit
def test_index_definitions_excludes_system_schemas() -> None:
    # #345: two row-emitting CTEs (`indexes`, `redundant_index_pairs`) need the exclusion.
    metric = yaml.safe_load((OLD / "pgwatch-postgres/metrics.yml").read_text())["metrics"]["index_definitions"]
    assert metric["sqls"]
    for version, sql in metric["sqls"].items():
        assert len(NOT_IN_SYSTEM_SCHEMAS.findall(sql)) == 2, version
        assert len(NOT_MATCH_TEMP_SCHEMAS.findall(sql)) == 2, version
        assert not NO_LIKE_ESCAPE.search(sql), version
