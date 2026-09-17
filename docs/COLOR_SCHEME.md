# PostgresAI wait-event color scheme

Use this palette for wait-event and Active Session History (ASH) visualizations
in PostgresAI monitoring **and all materials**: documentation, screenshots,
presentations, diagrams, demos, and reports. A type keeps its color regardless
of series rank, query ID, event name, or whether it represents a server process.
This is a data-visualization standard, not a new logo, UI theme, or brand palette.
Unrelated metrics and severity/status colors are outside its scope.

## Source and canonical values

Adopted from [pg_ash's documented palette](https://github.com/NikolayS/pg_ash/blob/2c3286ac2551a89ab6bf6e2c982b6988578d5640/docs/COLOR_SCHEME.md).
The previous Grafana dashboards shared some color families with pg_ash but not
its exact RGB values; IPC, BufferPin, Timeout, and LWLock also differed in hue.
The table below replaces those dashboard-specific assignments.

[wait-event-colors.json](wait-event-colors.json) is the machine-readable source
of truth. Use exact hex/RGB values, not Grafana named colors or positional
palettes. Named colors may change with the theme; positional palettes change
with series order.

| Type | Hex | RGB |
| --- | --- | --- |
| CPU* | `#50FA7B` | 80, 250, 123 |
| IdleTx | `#F1FA8C` | 241, 250, 140 |
| IO | `#1E64FF` | 30, 100, 255 |
| Lock | `#FF5555` | 255, 85, 85 |
| LWLock | `#FF79C6` | 255, 121, 198 |
| IPC | `#00C8FF` | 0, 200, 255 |
| Client | `#FFDC64` | 255, 220, 100 |
| Timeout | `#FFA500` | 255, 165, 0 |
| BufferPin | `#00D2B4` | 0, 210, 180 |
| Activity | `#9664FF` | 150, 100, 255 |
| Extension | `#BE96FF` | 190, 150, 255 |
| Unknown/Other | `#B4B4B4` | 180, 180, 180 |

`CPU*` is sampled active work without a reported wait, **not a measurement of
CPU utilization**. The asterisk is intentional. `IdleTx` means idle in a
transaction; it is not an ordinary idle session or a PostgreSQL wait-event type.
The colors describe categories, not a health verdict: green does not prove a
healthy workload and red does not by itself establish an incident.

Unknown types and “Other” use gray. `InjectionPoint`, which has no dedicated
entry in the source palette, also uses gray; retain its label. Do not invent a
new color for an unknown category without updating this standard.

## Implementation

- **Grafana:** all charts querying `pgwatch_wait_events_total` use explicit
  fixed-color overrides and a gray fallback. Match complete type names, including
  `CPU`/`CPU*`, `Type:event`, `Type - event - query`, and the `Postgres - ` and
  `Idle Internal - ` prefixes. `Lock` must not match `LWLock` or an event-name
  substring. Individual events inherit their type color.
- **Terminal output:** where truecolor is supported, use
  `\033[38;2;R;G;Bm`, then reset with `\033[0m`. Offer a no-color mode and keep
  labels/characters meaningful without ANSI color.
- **Docs, slides, and other assets:** take swatches from the JSON/table, keep
  category labels visible, and preserve the same mapping when redrawing charts.
  Do not reuse an old screenshot to illustrate the new palette. Historical
  screenshots may remain when clearly identified as historical.
- **Accessibility:** never make hue the only identifier. Keep legends, text,
  tooltips, and/or patterns; use neutral high-contrast text rather than these
  bright swatches for body copy. Review both light and dark backgrounds; add
  outlines or suitable chart surfaces where needed without silently changing
  semantic colors. This palette is not a claim of color-vision or text-contrast
  compliance.

## Keeping implementations aligned

After an intentional palette change, update the JSON and this table together,
then synchronize and validate:

```sh
python3 scripts/sync-wait-event-colors.py
python3 scripts/sync-wait-event-colors.py --check
python3 -m pytest tests/grafana_dashboards -q
```

The existing `quality:grafana-dashboards-lint` CI job checks palette drift,
complete type coverage, matcher collisions, and documentation parity.
Helm dashboard files symlink to the same Compose dashboard sources.
Contributors must link this standard when introducing wait-event visualizations
in other repositories; their review must check the mapping against this file.
Cross-repository documents and slide decks are subject to review, not this
repository's automated CI gate.

## Adoption scope

The initial dashboard alignment covers Dashboard 1 (node overview, panel 38),
Dashboard 3 (single query, panel 19), and Dashboard 4 (wait sampling, panels 1–3).
It changes color configuration only, not queries, sample data, or units.
External sites, existing published decks, and other repositories are not
retroactively rewritten by the synchronization script.
