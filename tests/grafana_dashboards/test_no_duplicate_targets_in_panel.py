"""Bug 3 — No duplicate / overlapping targets within a single Grafana panel.

The rc.6 demo surfaced Dashboard 1 ("Active session history") with the
legend showing duplicate `CPU*` and `IO` rows — same colour swatch,
identical min/max/mean. The root cause was four `targets[]` entries on
the same panel where the last one used `pgwatch_wait_events_total` with
NO label selector and the others used the same metric WITH filters. The
unfiltered target produced a superset that overlapped every filtered
one, generating duplicate series in the legend.

Three rules enforced here:

1. Within a single panel, no two targets may share the same non-empty
   `refId`. Grafana relies on `refId` to deduplicate; duplicate refIds
   silently merge.

2. Within a single panel, no two non-hidden targets may share the same
   normalized `expr` (Prometheus) or `rawSql` (Postgres). A textual
   duplicate is always a bug.

3. Within a single panel, if two non-hidden Prometheus targets reference
   the same metric name, both must include at least one label selector
   (a `{...}` block with content). An unfiltered selector on a metric
   that is filtered elsewhere produces a superset and thus duplicate
   series in the legend. This is the rule that catches the Dashboard 1
   ASH bug — target D (`sum by (wait_event_type)
   (pgwatch_wait_events_total)>0`) overlaps targets A/B/C which filter
   the same metric.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import iter_panels


def _normalize(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# Heuristic: extract Prometheus metric names that appear in an expr.
# Matches identifiers followed by `{` (with selector) or by `)`/whitespace
# (no selector). pgwatch metrics all start with `pgwatch_` which keeps the
# match precise; this lint is opt-in to that prefix to avoid false positives
# on aggregation functions like `sum`, `rate`, etc.
_PROM_METRIC_RE = re.compile(r"\b(pgwatch_[a-zA-Z0-9_]+)(\s*\{([^}]*)\})?")


def _prom_metric_uses(expr: str) -> list[tuple[str, bool]]:
    """Return [(metric, has_label_selector), ...] for every pgwatch metric
    referenced in the expression. has_label_selector is True iff the metric
    is followed by a `{...}` block whose contents are non-empty.
    """
    out: list[tuple[str, bool]] = []
    for m in _PROM_METRIC_RE.finditer(expr or ""):
        selector_contents = (m.group(3) or "").strip()
        out.append((m.group(1), bool(selector_contents)))
    return out


def test_no_duplicate_refids_in_panel(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    offenders: list[str] = []
    for panel in iter_panels(dashboard):
        tgts = panel.get("targets", []) or []
        refs = [t.get("refId") for t in tgts if t.get("refId")]
        dups = [k for k, v in Counter(refs).items() if v > 1]
        if dups:
            offenders.append(
                f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                f"duplicate refIds: {dups}"
            )
    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} panel(s) with duplicate "
        f"target refIds:\n  " + "\n  ".join(offenders)
    )


def test_no_duplicate_expressions_in_panel(dashboard_path: Path):
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    offenders: list[str] = []
    for panel in iter_panels(dashboard):
        seen: dict[str, str] = {}
        for ti, t in enumerate(panel.get("targets", []) or []):
            if t.get("hide"):
                continue
            body = _normalize(t.get("expr") or t.get("rawSql") or "")
            if not body:
                continue
            if body in seen:
                offenders.append(
                    f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                    f"targets {seen[body]} and refId={t.get('refId')!r} share "
                    f"the same expression: {body[:140]}"
                )
            else:
                seen[body] = f"refId={t.get('refId')!r}"
    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} duplicate-expression "
        f"target(s):\n  " + "\n  ".join(offenders)
    )


def test_no_unfiltered_metric_overlapping_filtered_in_panel(dashboard_path: Path):
    """If a Prometheus metric appears in multiple targets within a single
    panel, every target referencing that metric must include a non-empty
    `{label="..."}` selector. Otherwise the unfiltered target produces a
    superset and the legend shows duplicates.
    """
    with open(dashboard_path) as f:
        dashboard = json.load(f)

    offenders: list[str] = []
    for panel in iter_panels(dashboard):
        # metric_name -> list[(refId, has_selector, expr_snippet)]
        usage: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
        for t in panel.get("targets", []) or []:
            if t.get("hide"):
                continue
            expr = t.get("expr") or ""
            for metric, has_sel in _prom_metric_uses(expr):
                usage[metric].append(
                    (str(t.get("refId")), has_sel, _normalize(expr)[:140])
                )

        for metric, uses in usage.items():
            if len(uses) < 2:
                continue
            # Bug: the metric is used in multiple targets AND at least one of
            # them is unfiltered. The unfiltered target double-counts the
            # filtered ones.
            unfiltered = [u for u in uses if not u[1]]
            if unfiltered and any(u[1] for u in uses):
                snippets = "; ".join(f"refId={r} expr={s!r}" for r, _, s in uses)
                offenders.append(
                    f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                    f"metric={metric!r} appears in {len(uses)} targets, "
                    f"{len(unfiltered)} without a label selector; "
                    f"unfiltered target overlaps the filtered ones — {snippets}"
                )

    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} panel(s) where an "
        f"unfiltered Prometheus metric overlaps a filtered one in the "
        f"same panel:\n  " + "\n  ".join(offenders)
    )
