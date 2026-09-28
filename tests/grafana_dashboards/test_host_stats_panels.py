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

FAMILIES = [
    "host_cpu_utilization_percent",
    "host_os_cpu_percent",
    "host_os_cpu_iowait_percent",
    "host_memory_available_bytes",
    "host_os_memory_total_bytes",
    "host_os_memory_free_bytes",
    "host_os_memory_cached_bytes",
    "host_os_swap_used_bytes",
    "host_os_process_max_rss_bytes",
    "host_disk_free_bytes",
    "host_local_storage_free_bytes",
    "host_volume_used_bytes",
    "host_disk_read_iops",
    "host_disk_write_iops",
    "host_volume_read_iops",
    "host_volume_write_iops",
    "host_disk_read_latency_seconds",
    "host_disk_write_latency_seconds",
    "host_disk_queue_depth",
    "host_network_receive_bytes_per_second",
    "host_network_transmit_bytes_per_second",
    "host_db_load",
]


def _panels(path):
    return json.loads(path.read_text())["panels"]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_row_charts_every_family(path):
    panels = _panels(path)
    titles = [p.get("title") for p in panels if p.get("type") == "row"]
    assert "Host stats" in titles
    exprs = [t["expr"] for p in panels for t in p.get("targets", []) if "host_" in t.get("expr", "")]
    for family in FAMILIES:
        assert any(family in e for e in exprs), family
    for expr in exprs:
        assert 'cluster="$cluster_name"' in expr and 'node_name="$node_name"' in expr, expr


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
}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.parts[-4])
def test_host_stats_panels_units_and_legend_sort(path):
    host = {
        p["title"]: p
        for p in _panels(path)
        if any("host_" in t.get("expr", "") for t in p.get("targets", []))
    }
    assert sorted(host) == sorted(PANELS)
    for title, (unit, sort_by) in PANELS.items():
        assert host[title]["fieldConfig"]["defaults"]["unit"] == unit, title
        assert host[title]["options"]["legend"]["sortBy"] == sort_by, title
