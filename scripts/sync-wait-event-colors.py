#!/usr/bin/env python3
"""Apply the documented palette to ASH charts, or check it without writing."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def panels(dashboard):
    for panel in dashboard.get("panels", []):
        yield panel
        yield from panels(panel)


def is_ash(panel):
    return any("pgwatch_wait_events_total" in target.get("expr", "")
               for target in panel.get("targets", []))


def overrides(palette):
    result = []
    for name, color in palette.items():
        if name == "Unknown/Other":
            continue  # The default color covers every unrecognized type.
        event = r"CPU\*?" if name == "CPU*" else re.escape(name)
        result.append({
            "matcher": {"id": "byRegexp", "options":
                        rf"^(?:(?:Postgres|Idle Internal) - )?{event}(?:$|[: ].*)"},
            "properties": [{"id": "color", "value":
                            {"fixedColor": color, "mode": "fixed"}}],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on drift; do not write")
    args = parser.parse_args()
    palette = json.loads((ROOT / "docs/wait-event-colors.json").read_text())
    drift = []
    count = 0
    for path in sorted((ROOT / "config/grafana/dashboards").glob("*.json")):
        original_text = path.read_text()
        dashboard = json.loads(original_text)
        replacements = []
        changed = False
        for panel in panels(dashboard):
            if not is_ash(panel):
                continue
            count += 1
            config = panel["fieldConfig"]
            # Preserve non-color overrides (units, line styles, visibility, etc.).
            retained = []
            for override in config.get("overrides", []):
                properties = [p for p in override["properties"] if p["id"] != "color"]
                if properties:
                    retained.append({**override, "properties": properties})
            expected = retained + overrides(palette)
            default = {"fixedColor": palette["Unknown/Other"], "mode": "fixed"}
            if config.get("overrides") != expected or config["defaults"].get("color") != default:
                drift.append(f"{path.name}: panel {panel['id']}")
                previous = json.loads(json.dumps(config))
                config["overrides"] = expected
                config["defaults"]["color"] = default
                replacements.append((previous, config))
                changed = True
        if changed and not args.check:
            # Keep unrelated hand-formatted dashboard sections untouched.
            edits = []
            decoder = json.JSONDecoder()
            for match in re.finditer(r'(?m)^(\s*)"fieldConfig":\s*', original_text):
                start = match.end()
                value, length = decoder.raw_decode(original_text[start:])
                for previous, updated in replacements:
                    if value == previous:
                        indent = match.group(1)
                        rendered = json.dumps(updated, indent=2, ensure_ascii=False)
                        rendered = rendered.replace("\n", "\n" + indent)
                        edits.append((start, start + length, rendered))
                        break
            if len(edits) != len(replacements):
                raise SystemExit(f"Could not locate ASH field configurations in {path}")
            for start, end, rendered in reversed(edits):
                original_text = original_text[:start] + rendered + original_text[end:]
            path.write_text(original_text)
    if not count:
        raise SystemExit("No ASH panels found")
    if args.check and drift:
        raise SystemExit("Wait-event color drift; run scripts/sync-wait-event-colors.py:\n"
                         + "\n".join(drift))
    print(f"{'Checked' if args.check else 'Synchronized'} {count} ASH panels")


if __name__ == "__main__":
    main()
