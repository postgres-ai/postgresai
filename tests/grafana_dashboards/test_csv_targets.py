"""Every chart uses the compact transport without losing series identity."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mon/grafana"))
import compile as C
import self_monitoring as S


def test_every_timeseries_is_csv_with_proven_identity():
    cases = {c["id"]: c for c in C.load_cases()}
    for spec in C.DASHBOARDS:
        source = json.loads((Path(C.DASH_DIR) / spec["src"]).read_text())
        compiled = json.loads((Path(C.HERE) / "dashboards" / spec["out"]).read_text())
        sources = {p["id"]: p for p in C.walk(source["panels"])}
        for panel in C.walk(compiled["panels"]):
            for target in panel.get("targets", []):
                if target["format"] == "table":
                    assert target["type"] == "json" and target["root_selector"] == "rows"
                    continue
                original = next(t for t in sources[panel["id"]]["targets"] if t["refId"] == target["refId"])
                case = C.match_case(spec["census"], sources[panel["id"]], original, cases)
                by = case.get("params", {}).get("by")
                # An empty top-k grouping ranks the source series without aggregating them.
                ungrouped = by is None or (by == [] and case["sql_file"].endswith("/topk_fn.sql"))
                named = set(re.findall(r"\{\{\s*(.*?)\s*\}\}", original.get("legendFormat", "")))
                keep = ungrouped or not set(by) <= named
                assert_csv(target, panel, keep)
    selfmon = json.loads((Path(C.HERE) / "dashboards" / S.OUT).read_text())
    for panel in C.walk(selfmon["panels"]):
        for target in panel.get("targets", []):
            # Self-monitoring has no compiler grouping metadata.
            assert_csv(target, panel, True)


def assert_csv(target, panel, keep):
    assert (target["type"], target["parser"], target["root_selector"]) == ("csv", "backend", "")
    assert target["url"].endswith("/sql?format=csv")
    columns = [c["selector"] for c in target["columns"]]
    assert columns[:3] == ["time", "value", "__series"]
    assert min(columns[2:]) == "__series"
    assert "label_set" not in columns
    assert "label_set" not in json.dumps({k: v for k, v in panel.items() if k not in ("targets", "panels")})
    assert ("__sid" in columns) == keep
    if keep:
        assert sorted(columns[2:])[:2] == ["__series", "__sid"]
        assert next(c for c in target["columns"] if c["selector"] == "__sid")["type"] == "string"
    assert {"matcher": {"id": "byFrameRefID", "options": target["refId"]},
            "properties": [{"id": "displayName", "value": "${__field.labels.__series}"}]} in panel["fieldConfig"]["overrides"]


def test_grouping_proof():
    assert C.legend_covers("total", [])
    assert C.legend_covers("{{ a }} - {{b}}", ["a", "b"])
    assert not C.legend_covers("{{a}}", ["a", "b"])
    assert not C.legend_covers("{{a}}", None)
    assert C.grouping({"sql_file": "sql/reports/dash/topk_fn.sql", "params": {"by": []}}) is None
    assert C.grouping({"sql_file": "sql/reports/dash/agg_fn.sql", "params": {"by": []}}) == []
