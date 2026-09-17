"""Keep shipped ASH visualizations and the published palette aligned."""
import json
import re
from pathlib import Path

from tests.grafana_dashboards.conftest import iter_panels

ROOT = Path(__file__).resolve().parents[2]
PALETTE = json.loads((ROOT / "docs/wait-event-colors.json").read_text())


def color_for(panel, label):
    color = panel["fieldConfig"]["defaults"]["color"]
    matches = []
    for override in panel["fieldConfig"].get("overrides", []):
        matcher = override["matcher"]
        if matcher["id"] == "byRegexp":
            # Grafana anchors bare matcher expressions to the whole field name.
            matched = re.fullmatch(matcher["options"], label)
        elif matcher["id"] == "byName":
            matched = matcher["options"] == label
        else:
            continue
        if matched:
            for prop in override["properties"]:
                if prop["id"] == "color":
                    matches.append(prop["value"])
                    color = prop["value"]
    assert len(matches) <= 1, f"Ambiguous color match for {label}: {matches}"
    return color


def test_ash_colors(dashboard_path):
    dashboard = json.loads(dashboard_path.read_text())
    for panel in iter_panels(dashboard):
        if not any("pgwatch_wait_events_total" in t.get("expr", "")
                   for t in panel.get("targets", [])):
            continue
        for category, expected in PALETTE.items():
            names = [category] if category != "CPU*" else ["CPU", "CPU*"]
            if category == "Unknown/Other":
                names = ["Unknown", "Other", "InjectionPoint", "FutureWait"]
            for name in names:
                for prefix in ("", "Postgres - ", "Idle Internal - "):
                    for suffix in ("", ":SomeEvent", " - SomeEvent - 123"):
                        label = prefix + name + suffix
                        assert color_for(panel, label) == {
                            "mode": "fixed", "fixedColor": expected
                        }, f"{dashboard_path.name} panel {panel['id']}: {label}"
        # Type matching must not leak into substrings or event names.
        for label in ("NotLock", "LockManager", "IOther", "FutureWait:Lock",
                      "FutureWait - LWLock - 123"):
            assert color_for(panel, label) == {
                "mode": "fixed", "fixedColor": PALETTE["Unknown/Other"]
            }, f"{dashboard_path.name} panel {panel['id']}: {label}"


def test_palette_documentation_matches_json():
    doc = (ROOT / "docs/COLOR_SCHEME.md").read_text()
    rows = dict(re.findall(r"^\| ([^|]+) \| `(#[0-9A-F]{6})` \|", doc, re.M))
    assert rows == PALETTE
    assert len(set(PALETTE.values())) == len(PALETTE)
    for name, color in PALETTE.items():
        rgb = ", ".join(str(int(color[i:i + 2], 16)) for i in (1, 3, 5))
        assert f"| {name} | `{color}` | {rgb} |" in doc


def test_ash_panels_are_present(dashboards):
    # Avoid a vacuous pass if metric names or provisioning paths change.
    found = {str(path.resolve()) + ":" + str(panel["id"])
             for path, dashboard in dashboards for panel in iter_panels(dashboard)
             if any("pgwatch_wait_events_total" in t.get("expr", "")
                    for t in panel.get("targets", []))}
    assert len(found) >= 5
