"""Dashboard 1 shows the RDS/Aurora host stats that rds-host-stats writes.

Each family must be charted on a panel scoped to the dashboard's cluster and
node, so an operator sees CPU, memory, storage, I/O and network beside the
Postgres stats of the same node. Both dashboard trees must carry the panels.
"""
from __future__ import annotations

import json

import pytest

from tests.grafana_dashboards.conftest import REPO_ROOT

DASHBOARDS = [
    REPO_ROOT / "config/grafana/dashboards/Dashboard_1_Node_performance_overview.json",
    REPO_ROOT / "postgres_ai_helm/config/grafana/dashboards/Dashboard_1_Node_performance_overview.json",
]

# title -> the host_* families charted on that panel, so a family wired onto
# the wrong panel fails here rather than passing a dashboard-wide search.
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
    "Disk latency": [
        "host_disk_read_latency_seconds",
        "host_disk_write_latency_seconds",
    ],
    "Disk queue depth": ["host_disk_queue_depth"],
    "Network /s": [
        "host_network_receive_bytes_per_second",
        "host_network_transmit_bytes_per_second",
    ],
    "DB load (Performance Insights)": ["host_db_load"],
    "Replica lag": ["host_replica_lag_seconds"],
    "Burst and EBS I/O balance": [
        "host_burst_balance_percent",
        "host_ebs_io_balance_percent",
    ],
}

# Every series the poller writes, except host_os_load1, which Dashboard 1
# leaves to the OS CPU panel's percentages.
COLLECTED = [
    "host_cpu_utilization_percent",
    "host_memory_available_bytes",
    "host_disk_read_iops",
    "host_disk_write_iops",
    "host_disk_read_latency_seconds",
    "host_disk_write_latency_seconds",
    "host_disk_queue_depth",
    "host_network_receive_bytes_per_second",
    "host_network_transmit_bytes_per_second",
    "host_disk_free_bytes",
    "host_burst_balance_percent",
    "host_ebs_io_balance_percent",
    "host_replica_lag_seconds",
    "host_local_storage_free_bytes",
    "host_volume_used_bytes",
    "host_volume_read_iops",
    "host_volume_write_iops",
    "host_db_load",
    "host_os_cpu_percent",
    "host_os_cpu_iowait_percent",
    "host_os_memory_total_bytes",
    "host_os_memory_free_bytes",
    "host_os_memory_cached_bytes",
    "host_os_swap_used_bytes",
    "host_os_process_max_rss_bytes",
]


def _panels(path):
    return json.loads(path.read_text())["panels"]


def _row(path):
    return next(p for p in _panels(path) if p.get("type") == "row" and p["title"] == "Host stats")


def _host_panels(path):
    return {
        p["title"]: p
        for p in _row(path)["panels"]
        if any("host_" in t.get("expr", "") for t in p.get("targets", []))
    }


def _family(expr):
    return expr.split("{", 1)[0]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_row_charts_every_family_on_its_panel(path):
    host = _host_panels(path)
    assert sorted(host) == sorted(PANEL_FAMILIES)
    for title, families in PANEL_FAMILIES.items():
        exprs = [t["expr"] for t in host[title]["targets"]]
        assert [_family(e) for e in exprs] == families, title
        for expr in exprs:
            assert 'cluster="$cluster_name"' in expr and 'node_name="$node_name"' in expr, expr
    charted = sorted(f for families in PANEL_FAMILIES.values() for f in families)
    assert charted == sorted(COLLECTED)


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_row_is_collapsed_and_last(path):
    # Only RDS/Aurora boxes write host_* today; an expanded row would show
    # every other box a screen of "No data". A collapsed row keeps its panels
    # inside it, and must be the last top-level panel: collapsing a row from
    # the UI folds every following top-level panel into it.
    panels = _panels(path)
    row = _row(path)
    assert row["collapsed"] is True
    assert panels[-1] is row
    assert not any("host_" in t.get("expr", "") for p in panels for t in p.get("targets", []))


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_panels_do_not_bridge_collector_outages(path):
    for title, panel in _host_panels(path).items():
        span = panel["fieldConfig"]["defaults"]["custom"]["spanNulls"]
        assert isinstance(span, int) and span <= 180000, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_table_legends_fit_their_rows(path):
    # Rendered in Grafana 12.3: a table legend under an 8-unit panel shows up
    # to four rows, an 11-unit one four, a 15-unit one six. Memory has six
    # series, so it and its neighbour on the row need the taller panel.
    for title, panel in _host_panels(path).items():
        needed = 15 if len(panel["targets"]) >= 5 else 8
        assert panel["gridPos"]["h"] >= needed, title


# title -> (unit, legend sortBy), per the README's unit and legend-sorting rules:
# rates sort by Mean, latency/saturation by Max, levels by the last value.
PANELS = {
    "CPU utilization": ("percent", "Max"),
    "Memory": ("bytes", "Last *"),
    "Storage": ("bytes", "Last *"),
    "Disk IOPS /s": ("iops", "Mean"),
    "Disk latency": ("s", "Max"),
    "Disk queue depth": ("short", "Max"),
    "Network /s": ("binBps", "Mean"),
    "DB load (Performance Insights)": ("short", "Mean"),
    "Replica lag": ("s", "Max"),
    "Burst and EBS I/O balance": ("percent", "Last *"),
}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_panels_units_and_legend_sort(path):
    host = _host_panels(path)
    assert sorted(host) == sorted(PANELS)
    for title, (unit, sort_by) in PANELS.items():
        assert host[title]["fieldConfig"]["defaults"]["unit"] == unit, title
        assert host[title]["options"]["legend"]["sortBy"] == sort_by, title


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_percent_panels_use_soft_axis_limits(path):
    # Like the rest of the dashboard: a soft 0..100 axis keeps the scale
    # readable without clipping a value that CloudWatch reports out of range.
    for title, panel in _host_panels(path).items():
        defaults = panel["fieldConfig"]["defaults"]
        if defaults["unit"] != "percent":
            continue
        assert "min" not in defaults and "max" not in defaults, title
        assert defaults["custom"]["axisSoftMin"] == 0, title
