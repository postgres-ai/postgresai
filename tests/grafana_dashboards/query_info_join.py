"""The canonical pgwatch_query_info join operand, shared by both guards.

pgwatch_query_info arrives in sparse bursts, so joining it as an instant vector
misses its ~5 min lookback and pgss legends degrade to raw labels. Carrying it
forward over days then admits a second risk: one queryid can hold several
displayname* series, which duplicates the joined result. Hence: carry forward,
newest series wins, value normalised back to 1. See #344.

A plain module rather than conftest, so importing it from another test package
does not load conftest a second time alongside pytest's own plugin instance.
"""
from __future__ import annotations

import re

QUERY_INFO_METRIC = "pgwatch_query_info"
PROMQL_DURATION = r"(?:\d+[smhdwy])+"
# Optional label matcher, so a future MR may scope the selector without the
# guards rejecting it out of hand.
QUERY_INFO_SELECTOR = QUERY_INFO_METRIC + r"(?!\w)(?:\{[^}]*\})?"

QUERY_INFO_JOIN_OPERAND = re.compile(
    r"\(\s*topk\s+by\s*\(\s*queryid\s*\)\s*\(\s*1\s*,\s*tlast_over_time\("
    + QUERY_INFO_SELECTOR
    + r"\[(" + PROMQL_DURATION + r")\]\)\)\s*>\s*bool\s+0\)"
)
# Any mention of the metric, so the guards can require that every one of them
# is part of a full join operand rather than only rejecting the bare name.
QUERY_INFO_ANY_REFERENCE = re.compile(QUERY_INFO_METRIC + r"(?!\w)")

# Grafana's label_values() takes a selector, not an expression, so a queryid
# picker built on it cannot use the operand and is exempt.
LABEL_VALUES_CALL = re.compile(r"\blabel_values\s*\([^)]*\)")

# Label matcher blocks, blanked out before scanning for join clauses so an
# `on(...)` inside a label *value* cannot be mistaken for a real one.
LABEL_MATCHER_BLOCK = re.compile(r"\{[^{}]*\}")

# What may immediately precede a join operand. The `group_left` form copies
# labels across and is checked; the `unless` form is set exclusion and needs
# none. Anything else — `group_right`, or no modifier at all — means the
# legend's labels never cross the join.
MATCH_CLAUSE = r"\b(on|ignoring)\b\s*\(([^)]*)\)\s*"
GROUP_LEFT_BEFORE_OPERAND = re.compile(MATCH_CLAUSE + r"group_left\s*\(([^)]*)\)\s*$")
_KEYWORD, _MATCH_LABELS, _COPIED_LABELS = 1, 2, 3
UNLESS_BEFORE_OPERAND = re.compile(r"\bunless\s+" + MATCH_CLAUSE + r"$")

_DURATION_UNIT_SECONDS = {
    "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000,
}

# The observed per-series staleness is hours, so anything shorter than an hour
# would satisfy the "is it wrapped?" guards while reinstating the bug.
MIN_QUERY_INFO_LOOKBACK_SECONDS = 3600


def promql_duration_seconds(duration: str) -> int:
    """Convert a PromQL duration ('7d', '1d12h') to seconds."""
    parts = re.findall(r"(\d+)([smhdwy])", duration)
    assert parts, f"unparseable PromQL duration: {duration!r}"
    return sum(int(amount) * _DURATION_UNIT_SECONDS[unit] for amount, unit in parts)


def join_operand_problems(expr: str) -> list[str]:
    """Everything wrong with how one expression references pgwatch_query_info."""
    problems: list[str] = []
    references = len(QUERY_INFO_ANY_REFERENCE.findall(expr))
    if not references:
        return problems

    operands = QUERY_INFO_JOIN_OPERAND.findall(expr)
    if len(operands) != references:
        problems.append(
            f"{references} reference(s) but {len(operands)} full join operand(s); "
            "every reference must be the canonical carry-forward operand"
        )
    for duration in operands:
        seconds = promql_duration_seconds(duration)
        if seconds < MIN_QUERY_INFO_LOOKBACK_SECONDS:
            problems.append(
                f"lookback [{duration}] = {seconds}s is below the "
                f"{MIN_QUERY_INFO_LOOKBACK_SECONDS}s staleness floor"
            )
    return problems


def strip_label_values_calls(expr: str) -> str:
    """Blank out label_values(...) spans, keeping offsets intact.

    Grafana's label_values() takes a selector, not an expression, so a queryid
    picker built on it cannot carry the metric forward and is exempt — but only
    the call itself, not everything else in the same expression.
    """
    return LABEL_VALUES_CALL.sub(lambda m: " " * len(m.group(0)), expr)


def group_left_label_problems(expr: str, required_labels: set[str]) -> list[str]:
    """Labels each join must copy across for the legend to resolve.

    The operand can be perfect while `group_left()` copies nothing — or is
    absent entirely — which renders exactly the raw-label failure the operand
    exists to prevent. Checked per join, not per expression, so one correct
    join cannot vouch for another.
    """
    problems: list[str] = []
    if not required_labels or not QUERY_INFO_JOIN_OPERAND.search(expr):
        return problems

    # Label values can contain anything, including text that looks like a join
    # clause, so blank the matcher blocks before reading the clauses.
    clauses = LABEL_MATCHER_BLOCK.sub(lambda m: " " * len(m.group(0)), expr)

    # Iterate the original expression: blanking is length-preserving, so the
    # offsets still line up, but a blanked label matcher would stop the
    # operand matching at all and silently switch this guard off.
    for operand in QUERY_INFO_JOIN_OPERAND.finditer(expr):
        prefix = clauses[: operand.start()].rstrip()

        if UNLESS_BEFORE_OPERAND.search(prefix):
            # Set exclusion: the operand only decides membership, so it copies
            # nothing and needs nothing.
            continue

        match = GROUP_LEFT_BEFORE_OPERAND.search(prefix)
        if match is None:
            problems.append(
                "a pgwatch_query_info join has neither group_left(...) nor "
                "unless on(...) before it, so the label(s) the legend renders "
                "never cross it: " + ", ".join(sorted(required_labels))
            )
            continue

        # Labels matched on are already present on the left-hand side.
        # `ignoring(...)` names the opposite set, so it exempts nothing.
        matched_on: set[str] = set()
        if match.group(_KEYWORD) == "on":
            matched_on = {
                label.strip()
                for label in match.group(_MATCH_LABELS).split(",")
                if label.strip()
            }
        copied = {
            label.strip()
            for label in match.group(_COPIED_LABELS).split(",")
            if label.strip()
        }
        missing = sorted(required_labels - copied - matched_on)
        if missing:
            problems.append(
                "group_left does not copy the label(s) the legend renders: "
                + ", ".join(missing)
            )

    return problems
