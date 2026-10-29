"""Dashboard 99 is the detailed host page.

Review of MR !466 (Nik, 2026-10-09): read and write must not share a chart;
whatever adds up to a whole is stacked (CPU modes to 100 %, memory to total
RAM, storage to the volume size); limits are drawn where the source reports
them. The RDS DB load row is the Dashboard 1 row, unchanged.
The selectors list clusters and nodes that have host_* series, so a node that
pgwatch does not monitor still shows. The Helm file is a symlink to the
Compose file, like the other Helm dashboards, so the two cannot drift apart.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tests.grafana_dashboards.conftest import REPO_ROOT
from tests.grafana_dashboards.test_host_row import _providers

D1 = REPO_ROOT / "config/grafana/dashboards/Dashboard_1_Node_performance_overview.json"
D99 = REPO_ROOT / "config/grafana/dashboards/Dashboard_99_Host_stats.json"
D99_HELM = REPO_ROOT / "postgres_ai_helm/config/grafana/dashboards/Dashboard_99_Host_stats.json"

DB_LOAD_ROW = "RDS DB load (Performance Insights)"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _without_grid_pos(panel: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in panel.items() if k != "gridPos"}
    if "panels" in out:
        out["panels"] = [_without_grid_pos(p) for p in out["panels"]]
    return out


def _d1_rows() -> dict[str, dict[str, Any]]:
    return {p["title"]: p for p in _load(D1)["panels"] if p.get("type") == "row"}


def test_helm_copy_is_a_symlink_to_the_compose_file():
    assert D99_HELM.is_symlink()
    assert D99_HELM.resolve() == D99.resolve()


def test_title_and_uid():
    dashboard = _load(D99)
    assert dashboard["title"] == "99. Host stats"
    assert dashboard["uid"] == "host-stats"


TITLES = [
    "CPU utilization",
    "Memory",
    "Storage",
    "Network /s",
    "Disk read IOPS /s",
    "Disk read throughput /s",
    "Disk read latency",
    "Disk write IOPS /s",
    "Disk write throughput /s",
    "Disk write latency",
]
STACKED = {"CPU utilization", "Memory", "Storage"}
# Series drawn as reference lines on a stacked panel: never part of the stack.
REFERENCE = {
    "total RAM",
    "swap used",
    "largest process RSS",
    "limit: size",
    "local free (Aurora)",
    "volume used (Aurora)",
}
LIMITS = {
    "Storage": "host_disk_size_bytes",
    "Disk read IOPS /s": "host_disk_provisioned_iops",
    "Disk write IOPS /s": "host_disk_provisioned_iops",
    "Disk read throughput /s": "host_disk_provisioned_throughput_bytes_per_second",
    "Disk write throughput /s": "host_disk_provisioned_throughput_bytes_per_second",
}


def _host_panels() -> dict[str, dict[str, Any]]:
    return {p["title"]: p for p in _load(D99)["panels"][:-1]}


def _families(panel: dict[str, Any]) -> set[str]:
    return {f for t in panel["targets"] for f in re.findall(r"\b(host_[a-z0-9_]+)\{", t["expr"])}


def _overrides(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        o["matcher"]["options"]: {p["id"]: p["value"] for p in o["properties"]}
        for o in panel["fieldConfig"]["overrides"]
    }


def test_panel_titles_and_order():
    assert list(_host_panels()) == TITLES


def test_db_load_row_equals_dashboard_1():
    d1_row = _d1_rows()[DB_LOAD_ROW]
    assert _without_grid_pos(_load(D99)["panels"][-1]) == _without_grid_pos(d1_row)


def test_stacked_panels_equal_dashboard_1():
    # Dashboard 1's Host row shows the same three panels, so a host reads the
    # same on both pages: queries, legends, stacking, colours, units and
    # descriptions. Only the panel id and the position differ.
    d1 = {p["title"]: p for p in _d1_rows()["Host"]["panels"]}

    def comparable(panel: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in panel.items() if k not in ("gridPos", "id")}

    for title in sorted(STACKED):
        assert comparable(d1[title]) == comparable(_host_panels()[title]), title


def test_every_series_on_a_stacked_panel_has_a_pinned_colour():
    # Palette colours follow series order, and which series a provider returns
    # differs, so an unpinned series changes colour from box to box.
    for title in sorted(STACKED):
        panel = _host_panels()[title]
        overrides = _overrides(panel)
        for target in panel["targets"]:
            colour = overrides[target["legendFormat"]]["color"]
            assert colour["mode"] == "fixed" and colour["fixedColor"], (title, target["legendFormat"])


def test_largest_process_rss_is_a_dashed_line_over_the_memory_stack():
    panel = _host_panels()["Memory"]
    [rss] = [t for t in panel["targets"] if t["legendFormat"] == "largest process RSS"]
    assert rss["expr"] == 'host_os_process_max_rss_bytes{cluster="$cluster_name", node_name="$node_name"}'
    props = _overrides(panel)["largest process RSS"]
    assert props["custom.stacking"]["mode"] == "none"
    assert props["custom.lineStyle"]["fill"] == "dash"
    assert props["custom.fillOpacity"] == 0


def test_read_and_write_are_never_on_one_panel():
    for title, panel in _host_panels().items():
        names = " ".join(_families(panel) - set(LIMITS.values()))
        assert not ("read" in names and "write" in names), title


def test_every_charted_family_is_written_by_a_provider():
    written = set().union(*_providers().values())
    for title, panel in _host_panels().items():
        assert _families(panel) <= written, (title, _families(panel) - written)
        for target in panel["targets"]:
            for expr in re.findall(r"host_[a-z0-9_]+(\{[^}]*\})", target["expr"]):
                assert expr == '{cluster="$cluster_name", node_name="$node_name"}', title


def test_wholes_are_stacked_and_reference_lines_are_not():
    for title, panel in _host_panels().items():
        mode = panel["fieldConfig"]["defaults"]["custom"]["stacking"]["mode"]
        assert mode == ("normal" if title in STACKED else "none"), title
        for name, props in _overrides(panel).items():
            if "custom.stacking" in props:
                assert name in REFERENCE or name.startswith("limit: "), (title, name)
                assert props["custom.stacking"]["mode"] == "none", (title, name)
                assert props["custom.fillOpacity"] == 0, (title, name)


def test_cpu_stack_adds_up_to_100_percent():
    legends = [t["legendFormat"] for t in _host_panels()["CPU utilization"]["targets"]]
    assert legends == ["user", "system", "iowait", "steal", "other (nice, irq, softirq)", "busy (no mode split)", "idle"]
    assert _host_panels()["CPU utilization"]["fieldConfig"]["defaults"]["custom"]["axisSoftMax"] == 100


def test_limit_lines_are_dashed_unstacked_and_where_the_source_has_them():
    for title, panel in _host_panels().items():
        limits = [t for t in panel["targets"] if t["legendFormat"].startswith("limit: ")]
        if title not in LIMITS:
            assert not limits, title
            continue
        [limit] = limits
        assert limit["expr"].startswith(LIMITS[title] + "{"), title
        props = _overrides(panel)[limit["legendFormat"]]
        assert props["custom.lineStyle"]["fill"] == "dash", title
        assert props["custom.stacking"]["mode"] == "none", title
        assert "limit unknown" in panel["description"], title


def test_units_and_legend_sort():
    expected = {
        "CPU utilization": ("percent", "Max"),
        "Memory": ("bytes", "Last *"),
        "Storage": ("bytes", "Last *"),
        "Network /s": ("binBps", "Mean"),
    }
    for direction in ("read", "write"):
        expected[f"Disk {direction} IOPS /s"] = ("iops", "Mean")
        expected[f"Disk {direction} throughput /s"] = ("binBps", "Mean")
        expected[f"Disk {direction} latency"] = ("s", "Max")
    for title, panel in _host_panels().items():
        defaults = panel["fieldConfig"]["defaults"]
        assert (defaults["unit"], panel["options"]["legend"]["sortBy"]) == expected[title], title
        assert defaults["custom"]["axisSoftMin"] == 0, title
        assert "min" not in defaults and "max" not in defaults, title
        span = defaults["custom"]["spanNulls"]
        assert isinstance(span, int) and span <= 180000, title


def test_panel_ids_are_unique():
    ids = [p["id"] for p in _load(D99)["panels"]] + [p["id"] for p in _load(D99)["panels"][-1]["panels"]]
    assert len(ids) == len(set(ids))


def test_selectors_read_host_series_not_pgwatch():
    # Panels filter on node_name="$node_name", so each selector holds one
    # value: "All" or a multi-value selection would match no series.
    expected = {
        "cluster_name": 'label_values({__name__=~"host_.+"},cluster)',
        "node_name": 'label_values({__name__=~"host_.+",cluster="$cluster_name"},node_name)',
    }
    variables = {v["name"]: v for v in _load(D99)["templating"]["list"]}
    assert list(variables) == list(expected)
    for name, variable in variables.items():
        assert variable["definition"] == expected[name], name
        assert variable["query"]["query"] == expected[name], name
        assert variable["type"] == "query", name
        assert variable["refresh"] == 1, name
        assert variable["regex"] == "", name
        assert not variable.get("multi"), name
        assert not variable.get("includeAll"), name
        assert not variable.get("hide"), name
        # Provisioning must discover the current host rather than carry a
        # selection saved from the author's environment.
        assert not variable.get("current", {}).get("value"), name
        assert not variable.get("current", {}).get("text"), name


def test_links_and_time_settings_equal_dashboard_1():
    d1, d99 = _load(D1), _load(D99)
    for key in ("links", "timezone", "time", "timepicker", "refresh", "graphTooltip"):
        assert d99.get(key) == d1.get(key), key
    assert d99["tags"] == ["postgres-ai", "node"]


def test_db_load_row_is_collapsed_last_and_below_the_host_panels():
    panels = _load(D99)["panels"]
    row = panels[-1]
    assert row["title"] == DB_LOAD_ROW
    assert row["collapsed"] is True
    bottom = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels[:-1])
    assert row["gridPos"]["y"] >= bottom
    # The DB load panel opens directly under its row, full width.
    assert [p["gridPos"] for p in row["panels"]] == [{"x": 0, "y": row["gridPos"]["y"] + 1, "w": 24, "h": 8}]


def test_host_panels_fit_their_legends_and_do_not_overlap():
    # Same rule as the Dashboard 1 Host row: a table legend under an 8-unit
    # panel shows up to four rows, a 15-unit one six, a 16-unit one seven.
    panels = _load(D99)["panels"][:-1]
    cells: set[tuple[int, int]] = set()
    for panel in panels:
        pos = panel["gridPos"]
        needed = 16 if len(panel["targets"]) >= 7 else 15 if len(panel["targets"]) >= 5 else 8
        assert pos["h"] >= needed, panel["title"]
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        mine = {(x, y) for x in range(pos["x"], pos["x"] + pos["w"]) for y in range(pos["y"], pos["y"] + pos["h"])}
        assert not cells & mine, panel["title"]
        cells |= mine


def test_memory_evaluation_fixture_uses_shipped_queries():
    fixture = _load(REPO_ROOT / "tests/host_rules/dashboard_memory.test.json")
    targets = _host_panels()["Memory"]["targets"][:5]
    for case in fixture["tests"]:
        assert [check["expr"] for check in case["metricsql_expr_test"]] == [
            "(" + target["expr"].replace("$cluster_name", "test").replace("$node_name", "node") + ") + 0"
            for target in targets
        ]


def test_cpu_evaluation_fixture_uses_shipped_queries():
    fixture = _load(REPO_ROOT / "tests/host_rules/dashboard_cpu.test.json")
    targets = _host_panels()["CPU utilization"]["targets"]
    for case in fixture["tests"]:
        assert [check["expr"] for check in case["metricsql_expr_test"]] == [
            "(" + target["expr"].replace("$cluster_name", "test").replace("$node_name", "node") + ") + 0"
            for target in targets
        ]
