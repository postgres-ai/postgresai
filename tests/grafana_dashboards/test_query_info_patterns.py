"""Unit coverage for the pgwatch_query_info join patterns themselves.

Without this, the only proof that the dashboard guards are not vacuous is
hand-mutating the real dashboard JSON. These cases pin the exact regressions
the guards exist to catch, against synthetic expressions. See #344.
"""
from __future__ import annotations

import re

import pytest

from tests.grafana_dashboards.query_info_join import (
    MIN_QUERY_INFO_LOOKBACK_SECONDS,
    QUERY_INFO_JOIN_OPERAND,
    group_left_label_problems,
    join_operand_problems,
    promql_duration_seconds,
    strip_label_values_calls,
)

def operand(inner: str, dedup: str = "queryid", tail: str = " > bool 0") -> str:
    return f"(topk by ({dedup}) (1, {inner}){tail})"


CANONICAL = operand("tlast_over_time(pgwatch_query_info[7d])")

ACCEPTED = {
    "canonical": CANONICAL,
    "scoped by a label matcher": operand(
        'tlast_over_time(pgwatch_query_info{job="query-info"}[7d])'
    ),
    "compound duration": operand("tlast_over_time(pgwatch_query_info[1d12h])"),
    "loose whitespace": (
        "( topk by ( queryid ) ( 1 , "
        "tlast_over_time(pgwatch_query_info[7d])) >  bool 0)"
    ),
}

REJECTED = {
    "bare instant vector": "pgwatch_query_info",
    "carried forward but not de-duplicated": "last_over_time(pgwatch_query_info[7d])",
    # The wrapper-less form is the one that slipped past an earlier revision of
    # the guard: it looks carried-forward but neither de-duplicates nor
    # normalises the value.
    "right rollup, no wrapper": "tlast_over_time(pgwatch_query_info[7d])",
    "sibling metric": "tlast_over_time(pgwatch_query_info_total[7d])",
    # avg/count/sum_over_time would scale the joined value by the sample count;
    # the metric is a gauge whose value must stay 1 for the `*` join.
    "averaging rollup": "avg_over_time(pgwatch_query_info[7d])",
    "counting rollup": "count_over_time(pgwatch_query_info[7d])",
    # Without `> bool 0` the operand's value is a unix timestamp, which would
    # multiply every plotted rate by ~1.8e9.
    "value not normalised to 1": operand(
        "tlast_over_time(pgwatch_query_info[7d])", tail=""
    ),
    "de-duplicated across the wrong label": operand(
        "tlast_over_time(pgwatch_query_info[7d])", dedup="datname"
    ),
}


@pytest.mark.parametrize("expr", ACCEPTED.values(), ids=list(ACCEPTED))
def test_accepted_operands_match(expr: str) -> None:
    assert QUERY_INFO_JOIN_OPERAND.search(expr) is not None
    assert join_operand_problems(expr) == []


@pytest.mark.parametrize("expr", REJECTED.values(), ids=list(REJECTED))
def test_rejected_operands_are_reported(expr: str) -> None:
    assert QUERY_INFO_JOIN_OPERAND.search(expr) is None
    if re.search(r"pgwatch_query_info(?!\w)", expr):
        assert join_operand_problems(expr) != []


@pytest.mark.parametrize(
    "duration,seconds",
    [
        ("30s", 30), ("10m", 600), ("2h", 7200),
        ("7d", 604800), ("1w", 604800), ("1d12h", 129600),
    ],
)
def test_promql_duration_seconds(duration: str, seconds: int) -> None:
    assert promql_duration_seconds(duration) == seconds


@pytest.mark.parametrize("duration", ["10s", "5m", "59m"])
def test_short_lookbacks_are_below_the_staleness_floor(duration: str) -> None:
    """A wrapped-but-tiny window would reinstate the raw-label fallback."""
    match = QUERY_INFO_JOIN_OPERAND.search(
        operand(f"tlast_over_time(pgwatch_query_info[{duration}])")
    )
    assert match is not None
    assert promql_duration_seconds(match.group(1)) < MIN_QUERY_INFO_LOOKBACK_SECONDS


@pytest.mark.parametrize("duration", ["1h", "2d", "7d"])
def test_long_lookbacks_clear_the_staleness_floor(duration: str) -> None:
    assert promql_duration_seconds(duration) >= MIN_QUERY_INFO_LOOKBACK_SECONDS


LEGEND = {"displayname_long", "displayname_raw_long"}
GROUP_LEFT = "group_left(displayname_long, displayname_raw_long)"

# The shipped shape: a group_left branch that copies the labels, or'd with an
# `unless` branch that copies nothing because it only decides membership.
SHIPPED_JOIN = (
    f"(topk(10, M) * on(queryid) {GROUP_LEFT} {CANONICAL})"
    f" or (topk(10, M) unless on(queryid) {CANONICAL})"
)

# The scoped form the reviewer notes recommend adopting next. It must not
# switch the label check off: blanking label matchers once stopped the operand
# matching at all, which silently disarmed this guard.
SCOPED = (
    '(topk by (queryid) (1, '
    'tlast_over_time(pgwatch_query_info{job="query-info"}[7d])) > bool 0)'
)

LABEL_TRANSFER_OK = {
    "shipped two-branch shape": SHIPPED_JOIN,
    "single join copying the labels": f"M * on(queryid) {GROUP_LEFT} {CANONICAL}",
    "legend label supplied by on()": (
        "M * on(queryid, displayname_long, displayname_raw_long) "
        f"group_left() {CANONICAL}"
    ),
    "unless branch alone copies nothing": f"M unless on(queryid) {CANONICAL}",
    "scoped selector, labels copied": f"M * on(queryid) {GROUP_LEFT} {SCOPED}",
    "scoped selector on the unless branch": f"M unless on(queryid) {SCOPED}",
    "ignoring an unrelated label": f"M * ignoring(datname) {GROUP_LEFT} {CANONICAL}",
}

LABEL_TRANSFER_BROKEN = {
    "no grouping modifier at all": f"M * on(queryid) {CANONICAL}",
    "group_left with an empty list": f"M * on(queryid) group_left() {CANONICAL}",
    "group_left spaced before the paren": f"M * on(queryid) group_left () {CANONICAL}",
    "group_right instead": (
        "M * on(queryid) "
        f"group_right(displayname_long, displayname_raw_long) {CANONICAL}"
    ),
    "only one of the two legend labels": (
        f"M * on(queryid) group_left(displayname_long) {CANONICAL}"
    ),
    # A join clause hiding inside a label value must not exempt anything.
    "on() inside a label value": (
        'M{note="on(displayname_long,displayname_raw_long)"} '
        f"* on(queryid) group_left() {CANONICAL}"
    ),
    # An unbalanced paren in a label value would let the on(...) scan run past
    # the matcher and swallow the legend labels — this is what pins the
    # matcher-blanking step.
    "unbalanced paren in a label value": (
        'M{note="on(displayname_raw_long, displayname_long, x"} '
        f"* on(queryid) group_left() {CANONICAL}"
    ),
    # A token merely ending in "on" is not an on(...) clause.
    "comparison() must not read as on()": (
        "M * comparison(displayname_long, displayname_raw_long) "
        f"group_left() {CANONICAL}"
    ),
    # A label matcher must not switch the check off.
    "scoped selector, empty group_left": f"M * on(queryid) group_left() {SCOPED}",
    "scoped selector, no group_left": f"M * on(queryid) {SCOPED}",
    # ignoring(X) excludes X from matching, so it exempts nothing.
    "ignoring the legend labels": (
        "M * ignoring(displayname_long, displayname_raw_long) "
        f"group_left() {CANONICAL}"
    ),
    # One correct join must not vouch for a second, broken one.
    "second join copies nothing": (
        f"(A * on(queryid) {GROUP_LEFT} {CANONICAL})"
        f" + (B * on(queryid) {CANONICAL})"
    ),
}


@pytest.mark.parametrize(
    "expr", LABEL_TRANSFER_OK.values(), ids=list(LABEL_TRANSFER_OK)
)
def test_label_transfer_accepted(expr: str) -> None:
    assert group_left_label_problems(expr, LEGEND) == []


@pytest.mark.parametrize(
    "expr", LABEL_TRANSFER_BROKEN.values(), ids=list(LABEL_TRANSFER_BROKEN)
)
def test_label_transfer_rejected(expr: str) -> None:
    assert group_left_label_problems(expr, LEGEND) != []


def test_label_transfer_ignores_expressions_without_the_operand() -> None:
    """Other dashboards' group_left joins are none of this guard's business."""
    other = "M * on(queryid) group_left() other_metric"
    assert group_left_label_problems(other, LEGEND) == []


@pytest.mark.parametrize(
    "expr,exempt",
    [
        ("label_values(pgwatch_query_info, queryid)", True),
        ('label_values(pgwatch_query_info{cluster="c"}, queryid)', True),
        (
            "label_values(pgwatch_query_info, queryid) or pgwatch_query_info",
            False,
        ),
    ],
)
def test_label_values_exemption_covers_the_call_only(expr: str, exempt: bool) -> None:
    problems = join_operand_problems(strip_label_values_calls(expr))
    assert (problems == []) is exempt
