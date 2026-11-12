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
   normalized query (the SQL body POSTed to pgai-mon). A textual
   duplicate is always a bug.

3. Within a single panel, if two non-hidden targets read the same metric
   (`prom_*('<metric>', <matchers>, ...)` or `pgai_samples`), every read
   must pass a non-empty matcher list. An unfiltered read of a metric
   that is filtered elsewhere produces a superset and thus duplicate
   series in the legend. This is the rule that catches the Dashboard 1
   ASH bug — target D (`sum by (wait_event_type)
   (pgwatch_wait_events_total)>0`) overlapped targets A/B/C which filter
   the same metric.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import call_args, iter_panels, target_query


def _normalize(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# A metric read: an evaluator table function or pgai_samples, whose first two arguments are the
# metric name and the matcher list.
_METRIC_CALL_RE = re.compile(r"\b(prom_[a-z_]+|pgai_samples)\(\s*\(?'(pgwatch_\w+|pg_\w+)'")
_EMPTY_MATCHERS = {"[]", "[]::VARCHAR[][]", "CAST([] AS VARCHAR[][])"}


def _metric_uses(sql: str) -> list[tuple[str, bool]]:
    """Return [(metric, has_matchers), ...] for every metric the query reads. has_matchers is
    False when the matcher list is empty."""
    out: list[tuple[str, bool]] = []
    for m in _METRIC_CALL_RE.finditer(sql or ""):
        args = call_args(sql[m.start():], m.group(1))
        matchers = args[1] if len(args) > 1 else ""
        while matchers.startswith("(") and matchers.endswith(")"):
            matchers = matchers[1:-1].strip()
        out.append((m.group(2), matchers not in _EMPTY_MATCHERS))
    return out


def test_metric_uses_parser():
    assert _metric_uses("FROM prom_last(('pgwatch_a'), ([['cluster', '=', 'x']]), 1, 2, 3)") == [("pgwatch_a", True)]
    assert _metric_uses("FROM prom_irate('pgwatch_a', []::VARCHAR[][], 1, 2, 3)") == [("pgwatch_a", False)]
    assert _metric_uses("FROM prom_sum_over_time('pgwatch_a', [['a','=','b']] || CASE ('a') WHEN 'a' "
                        "THEN [] END, 1)") == [("pgwatch_a", True)]


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
            body = _normalize(target_query(t))
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
    """If a metric is read by several targets of one panel, every read must
    pass matchers. Otherwise the unfiltered target produces a superset and
    the legend shows duplicates.
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
            sql = target_query(t)
            for metric, has_sel in _metric_uses(sql):
                usage[metric].append(
                    (str(t.get("refId")), has_sel, _normalize(sql)[:140])
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
                    f"{len(unfiltered)} without matchers; "
                    f"unfiltered target overlaps the filtered ones — {snippets}"
                )

    assert not offenders, (
        f"{dashboard_path.name}: {len(offenders)} panel(s) where an "
        f"unfiltered metric read overlaps a filtered one in the "
        f"same panel:\n  " + "\n  ".join(offenders)
    )
