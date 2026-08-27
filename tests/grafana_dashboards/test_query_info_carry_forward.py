"""Guard: ``pgwatch_query_info`` must be carried forward, newest series wins.

The metric is emitted in sparse bursts (per-series staleness of hours), so an
instant-vector join misses its ~5 min lookback and pgss legends fall back to
the raw label set. Carrying it forward over days then admits a second failure:
one queryid can hold several ``displayname*`` series, which duplicates the
joined result — so the operand also has to reduce to one series. And the
operand can be perfect while ``group_left()`` copies no labels at all, which
renders the same raw-label failure by another route. See #344.

The canonical operand and its floor live in ``query_info_join`` so this file
and ``tests/compliance_vectors/test_mr219_monitoring_guards.py`` cannot drift
apart on what "carried forward" means.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tests.grafana_dashboards.conftest import (
    QUERY_INFO_JOIN_OPERAND,
    group_left_label_problems,
    iter_panels,
    join_operand_problems,
    strip_label_values_calls,
    unique_dashboard_paths,
)

# The 46 join sites on Dashboard 02, counted once per underlying file.
MIN_JOIN_OPERANDS = 46


def _promql_expressions(dashboard: dict) -> Iterator[tuple[str, str]]:
    """Yield (location, PromQL) for every query expression in a dashboard."""
    for panel in iter_panels(dashboard):
        for target in panel.get("targets", []) or []:
            expr = target.get("expr")
            if expr:
                location = (
                    f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                    f"target={target.get('refId')!r}"
                )
                yield location, expr

    for variable in dashboard.get("templating", {}).get("list", []) or []:
        query = variable.get("query")
        if isinstance(query, dict):
            query = query.get("query")
        definition = variable.get("definition")
        for field, expr in (("query", query), ("definition", definition)):
            if isinstance(expr, str) and expr:
                yield f"variable {variable.get('name')!r} {field}", expr


def _legend_labels(dashboard: dict, panel: dict) -> set[str]:
    """Labels a panel's legendFormat needs the join to copy across.

    ``{{displayname_long}}`` names a label directly; ``{{$legend_label}}``
    names a template variable, in which case *every* value it can take has to
    survive the join or one of the choices renders blank.
    """
    labels: set[str] = set()
    declared = dashboard.get("templating", {}).get("list", []) or []
    variables = {v.get("name"): v for v in declared}

    for target in panel.get("targets", []) or []:
        legend = target.get("legendFormat") or ""
        for token in _legend_tokens(legend):
            if token.startswith("$"):
                variable = variables.get(token[1:], {})
                for option in variable.get("options", []) or []:
                    value = option.get("value")
                    if isinstance(value, str) and value:
                        labels.add(value)
            else:
                labels.add(token)
    return labels


def _legend_tokens(legend: str) -> list[str]:
    tokens = []
    rest = legend
    while "{{" in rest and "}}" in rest:
        _, _, rest = rest.partition("{{")
        token, _, rest = rest.partition("}}")
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


def test_every_query_info_reference_is_a_join_operand(dashboard_path: Path) -> None:
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

    offenders = []
    for location, expr in _promql_expressions(dashboard):
        for problem in join_operand_problems(strip_label_values_calls(expr)):
            offenders.append(f"{location}: {problem}\n    {expr}")

    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def test_query_info_joins_copy_the_legend_labels(dashboard_path: Path) -> None:
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

    offenders = []
    for panel in iter_panels(dashboard):
        required = _legend_labels(dashboard, panel)
        if not required:
            continue
        for target in panel.get("targets", []) or []:
            expr = target.get("expr") or ""
            for problem in group_left_label_problems(expr, required):
                offenders.append(
                    f"panel id={panel.get('id')!r} title={panel.get('title')!r} "
                    f"target={target.get('refId')!r}: {problem}"
                )

    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def test_query_info_joins_are_actually_present() -> None:
    """The guards above must not pass because the joins were deleted."""
    operands = 0
    for path in unique_dashboard_paths():
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        for _, expr in _promql_expressions(dashboard):
            operands += len(QUERY_INFO_JOIN_OPERAND.findall(expr))

    assert operands >= MIN_JOIN_OPERANDS, (
        f"only {operands} pgwatch_query_info join operands found across the "
        f"dashboards, expected at least {MIN_JOIN_OPERANDS}"
    )
