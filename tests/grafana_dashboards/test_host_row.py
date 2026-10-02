"""Dashboard 1 has one Host row, read from the host_* vocabulary only.

RDS (rds-host-stats), Supabase and ClickHouse (config/prometheus/host_rules.yml)
all write host_* series labelled {cluster, node_name}, so one row serves every
provider. Each panel must show data for every provider: a panel only one
provider can fill is an empty panel on every other box. The vocabulary table in
docs/host-metrics.md must list exactly the names the providers write.

RDS DB load (Performance Insights) has no Supabase or ClickHouse equivalent,
so it gets its own collapsed row after Host, named for RDS. Grafana 12.3 has
no way to hide a row without data: a row repeated over an empty variable is
still rendered once, so on other providers this row stays closed and empty.
"""
from __future__ import annotations

import json
import re

import pytest
import yaml

from tests.grafana_dashboards.conftest import REPO_ROOT

DASHBOARDS = [
    REPO_ROOT / "config/grafana/dashboards/Dashboard_1_Node_performance_overview.json",
    REPO_ROOT / "postgres_ai_helm/config/grafana/dashboards/Dashboard_1_Node_performance_overview.json",
]

# title -> the host_* families charted on that panel, in order.
PANEL_FAMILIES = {
    "CPU utilization": [
        "host_cpu_utilization_percent",
        "host_os_cpu_percent",
        "host_os_cpu_iowait_percent",
    ],
    "Memory": [
        "host_memory_available_bytes",
        "host_os_memory_total_bytes",
        "host_os_memory_free_bytes",
        "host_os_memory_cached_bytes",
        "host_os_swap_used_bytes",
        "host_os_process_max_rss_bytes",
    ],
    "Storage": [
        "host_disk_free_bytes",
        "host_local_storage_free_bytes",
        "host_volume_used_bytes",
    ],
    "Disk IOPS /s": [
        "host_disk_read_iops",
        "host_disk_write_iops",
        "host_volume_read_iops",
        "host_volume_write_iops",
    ],
    "Network /s": [
        "host_network_receive_bytes_per_second",
        "host_network_transmit_bytes_per_second",
    ],
}

# title -> (unit, legend sortBy), per the dashboards README: rates sort by
# Mean, latency/saturation by Max, levels by the last value.
PANELS = {
    "CPU utilization": ("percent", "Max"),
    "Memory": ("bytes", "Last *"),
    "Storage": ("bytes", "Last *"),
    "Disk IOPS /s": ("iops", "Mean"),
    "Network /s": ("binBps", "Mean"),
}


def _written_by_rds() -> set[str]:
    source = (REPO_ROOT / "rds-host-stats/lib/poll.ts").read_text()
    return set(re.findall(r"'(host_[a-z0-9_]+)'", source))


def _written_by_rules() -> dict[str, set[str]]:
    rules = yaml.safe_load((REPO_ROOT / "config/prometheus/host_rules.yml").read_text())
    return {g["name"]: {r["record"] for r in g["rules"]} for g in rules["groups"]}


def _providers() -> dict[str, set[str]]:
    rules = _written_by_rules()
    return {"rds": _written_by_rds(), "supabase": rules["host-supabase"], "clickhouse": rules["host-clickhouse"]}


def _panels(path):
    return json.loads(path.read_text())["panels"]


def _row(path):
    rows = [p for p in _panels(path) if p.get("type") == "row" and p["title"].startswith("Host")]
    assert [r["title"] for r in rows] == ["Host"], "one Host row replaces the per-provider rows"
    return rows[0]


DB_LOAD_ROW = "RDS DB load (Performance Insights)"
SELECTOR = '{cluster="$cluster_name", node_name="$node_name"}'


def _db_load_row(path):
    rows = [p for p in _panels(path) if p.get("type") == "row" and p["title"] == DB_LOAD_ROW]
    assert len(rows) == 1
    return rows[0]


def _host_panels(path):
    return {p["title"]: p for p in _row(path)["panels"]}


def _family(expr):
    return expr.split("{", 1)[0]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_row_is_collapsed_last_and_reads_only_host_series(path):
    # Collapsed: a self-managed box writes no host_* and sees one closed row.
    # Last: collapsing a row from the UI folds every later top-level panel in.
    panels = _panels(path)
    row = _row(path)
    assert row["collapsed"] is True
    assert panels[-2:] == [row, _db_load_row(path)]
    # Grafana lays panels out by gridPos, not list order.
    bottom = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels[:-2])
    assert row["gridPos"]["y"] >= bottom
    assert panels[-1]["gridPos"]["y"] > row["gridPos"]["y"]
    for p in panels:
        for t in p.get("targets", []):
            assert "host_" not in t.get("expr", ""), p.get("title")
    everywhere = json.dumps(panels)
    assert "node_cpu_seconds_total" not in everywhere and "PostgresServer_" not in everywhere
    assert 'job=\\"supabase-host-metrics\\"' not in everywhere


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_row_charts_each_family_on_its_panel(path):
    host = _host_panels(path)
    assert list(host) == list(PANEL_FAMILIES)
    for title, families in PANEL_FAMILIES.items():
        exprs = [t["expr"] for t in host[title]["targets"]]
        assert [_family(e) for e in exprs] == families, title
        for expr in exprs:
            assert expr == f'{_family(expr)}{{cluster="$cluster_name", node_name="$node_name"}}', expr


def test_every_host_panel_has_data_from_every_provider():
    providers = _providers()
    for title, families in PANEL_FAMILIES.items():
        for provider, written in providers.items():
            assert set(families) & written, f"{title} is empty for {provider}"
    charted = {f for families in PANEL_FAMILIES.values() for f in families}
    written = set().union(*providers.values())
    assert charted <= written, charted - written


def test_vocabulary_doc_lists_exactly_what_providers_write():
    doc = (REPO_ROOT / "docs/host-metrics.md").read_text()
    documented = set(re.findall(r"^\| `(host_[a-z0-9_]+)` \|", doc, re.M))
    assert documented == set().union(*_providers().values())


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_panels_do_not_bridge_collector_outages(path):
    for title, panel in _host_panels(path).items():
        span = panel["fieldConfig"]["defaults"]["custom"]["spanNulls"]
        assert isinstance(span, int) and span <= 180000, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_table_legends_fit_their_rows(path):
    # Rendered in Grafana 12.3: a table legend under an 8-unit panel shows up
    # to four rows, a 15-unit one six. Memory has six series, so it and its
    # neighbour on the row need the taller panel.
    for title, panel in _host_panels(path).items():
        needed = 15 if len(panel["targets"]) >= 5 else 8
        assert panel["gridPos"]["h"] >= needed, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_panels_units_and_legend_sort(path):
    host = _host_panels(path)
    for title, (unit, sort_by) in PANELS.items():
        assert host[title]["fieldConfig"]["defaults"]["unit"] == unit, title
        assert host[title]["options"]["legend"]["sortBy"] == sort_by, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_percent_panels_use_soft_axis_limits(path):
    # A soft 0..100 axis keeps the scale readable without clipping a value a
    # provider reports out of range.
    for title, panel in _host_panels(path).items():
        defaults = panel["fieldConfig"]["defaults"]
        if defaults["unit"] != "percent":
            continue
        assert "min" not in defaults and "max" not in defaults, title
        assert defaults["custom"]["axisSoftMin"] == 0, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_rds_db_load_row_is_collapsed_last_and_charts_pi_db_load(path):
    # Last and collapsed, like Host: a node without PI data sees one closed row.
    row = _db_load_row(path)
    assert _panels(path)[-1] == row
    assert row["collapsed"] is True
    assert "repeat" not in row
    [panel] = row["panels"]
    assert panel["title"] == "DB load (average active sessions)"
    assert [t["expr"] for t in panel["targets"]] == [f"host_db_load{SELECTOR}"]
    assert panel["fieldConfig"]["defaults"]["unit"] == "short"
    assert panel["options"]["legend"]["sortBy"] == "Max"
    span = panel["fieldConfig"]["defaults"]["custom"]["spanNulls"]
    assert isinstance(span, int) and span <= 180000
    assert "host_db_load" in _written_by_rds()
