"""Field labels used by links and display settings must reach Infinity's rows."""
import json
import re

import pytest

from mon.grafana import compile as C
from .conftest import iter_panels


# These index metrics never had queryid, including on the old Prometheus dashboards.
MISSING_LABELS = {
    ("Dashboard_10_Index_health_pgai_mon.json", 23, "A"): {"queryid"},
    ("Dashboard_10_Index_health_pgai_mon.json", 2, "A"): {"queryid"},
    ("Dashboard_10_Index_health_pgai_mon.json", 22, "A"): {"queryid"},
}


def test_referenced_field_labels_are_columns(dashboards):
    missing = {}
    for path, dashboard in dashboards:
        for panel in iter_panels(dashboard):
            settings = "".join(json.dumps(panel.get(k)) for k in ("fieldConfig", "options", "links"))
            labels = set(re.findall(r"__field\.labels\.(\w+)", settings))
            for target in panel.get("targets", []):
                if target.get("format") != "timeseries":
                    continue
                absent = labels - {c["selector"] for c in target.get("columns", [])}
                if absent:
                    missing[path.name, panel["id"], target["refId"]] = absent
    assert missing == MISSING_LABELS


@pytest.mark.parametrize("cols,params,expected", [
    (["t", "value", "queryid"], {}, ["datname", "queryid"]),
    (["t", "value", "labels"], {"metric": "m"}, ["datname", "queryid", "schema"]),
    (["t", "value", "labels"], {"metric": "m", "by": ["schema"]}, ["datname", "schema"]),
    (["t", "value", "labels"], {"metric": "m", "by": []}, ["datname"]),
    (["t", "value"], {"metric": "m"}, ["datname"]),
])
def test_only_available_panel_labels_are_added(monkeypatch, cols, params, expected):
    monkeypatch.setattr(C, "probe_label_keys", lambda url, metric: ["queryid", "schema", "unused"])
    panel = {
        "fieldConfig": {"defaults": {"displayName": "${__field.labels.series}"}},
        "options": {"label": "${__field.labels.schema}"},
        "links": [{"url": "${__field.labels.queryid}/${__field.labels.absent}/${__field.labels.datname}"}],
    }
    extra = ["datname"]
    assert C.timeseries_labels(panel, cols, params, extra, "unused") == expected
    assert extra == ["datname"]
