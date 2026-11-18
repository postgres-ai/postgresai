"""Guard: statement texts come from pgai-mon's ``query_texts``, one row per queryid.

The PromQL dashboards labelled pg_stat_statements series by joining ``pgwatch_query_info``
(produced by the Flask backend), and had to carry it forward and de-duplicate it (#344). The SQL
dashboards read ``query_texts`` instead. That view keeps one row per stored text of a
(queryid, dbname), so a queryid whose text changed has two rows: a plain join would split its
series in two. Every read of ``query_texts`` must therefore reduce to one row per queryid and
database (an aggregate, or a LIMIT 1 lookup), and every series still has a name when no text
was collected (it falls back to the queryid).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.grafana_dashboards.conftest import dashboard_file, iter_panels, target_query

# Dashboard 2: 23 time-series panels plus the pg_stat_statements table; Dashboard 3: the text panel.
MIN_QUERY_TEXT_READS = 25


def _queries(dashboard: dict):
    for panel in iter_panels(dashboard):
        for target in panel.get("targets", []) or []:
            sql = target_query(target)
            if sql:
                yield f"panel id={panel.get('id')!r} target={target.get('refId')!r}", panel, target, sql
    for v in dashboard.get("templating", {}).get("list", []) or []:
        if isinstance(v.get("definition"), str) and v["definition"]:
            yield f"variable {v.get('name')!r}", None, v, v["definition"]


def _enclosing(sql: str, at: int) -> str:
    """From `at` to the end of the enclosing parenthesised subquery (or of the statement)."""
    depth = 0
    for j in range(at, len(sql)):
        if sql[j] == "(":
            depth += 1
        elif sql[j] == ")":
            if depth == 0:
                return sql[at:j]
            depth -= 1
    return sql[at:]


def query_text_problems(sql: str) -> list[str]:
    problems = []
    if re.search(r"pgwatch_query_info(?!\w)", sql):
        problems.append("reads pgwatch_query_info (the Flask backend's metric); use query_texts")
    for m in re.finditer(r"\b(FROM|JOIN)\s+query_texts\b", sql):
        if m.group(1) == "JOIN":
            problems.append("joins query_texts directly: one row per stored text duplicates series")
            continue
        scope = _enclosing(sql, m.end())
        if not re.search(r"\bGROUP BY\b", scope) and not re.search(r"\bLIMIT 1\b", scope):
            problems.append("reads query_texts without reducing it to one row per queryid")
    return problems


def test_every_query_texts_read_is_one_row_per_queryid(dashboard_path: Path) -> None:
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    offenders = [f"{where}: {p}" for where, _, _, sql in _queries(dashboard) for p in query_text_problems(sql)]
    assert not offenders, f"{dashboard_path.name}:\n  " + "\n  ".join(offenders)


def test_query_texts_reads_are_actually_present() -> None:
    """The guard above must not pass because the reads were removed."""
    reads = 0
    for number in (2, 3):
        dashboard = json.loads(dashboard_file(number).read_text(encoding="utf-8"))
        reads += sum(len(re.findall(r"\bquery_texts\b", sql)) for _, _, _, sql in _queries(dashboard))
    assert reads >= MIN_QUERY_TEXT_READS, f"only {reads} query_texts reads in Dashboards 2 and 3"


def test_series_named_by_text_fall_back_to_the_queryid() -> None:
    """A time series whose name is the statement text keeps a name while no text is stored."""
    dashboard = json.loads(dashboard_file(2).read_text(encoding="utf-8"))
    named = 0
    for where, panel, target, sql in _queries(dashboard):
        if panel is None or target.get("format") != "timeseries" or "query_texts" not in sql:
            continue
        m = re.search(r"AS __key, (.*?) AS __series,", sql)
        assert m, where
        assert re.fullmatch(r"coalesce\(q\.\"_displayname\w*\", q\.labels\['queryid'\], ''\)", m.group(1)), \
            f"{where}: series is {m.group(1)}"
        named += 1
    assert named >= 23


ACCEPTED = {
    "aggregated subquery": "SELECT * FROM t LEFT JOIN (SELECT queryid, dbname, arg_max(query, last_seen) AS query "
                           "FROM query_texts GROUP BY ALL) qt ON qt.queryid = t.queryid",
    "aggregated CTE": "WITH x AS (SELECT queryid, any_value(query) FROM query_texts WHERE true GROUP BY ALL) "
                      "SELECT * FROM x",
    "one-row lookup": "WITH q AS (SELECT query FROM query_texts WHERE queryid = 1 ORDER BY last_seen DESC LIMIT 1) "
                      "SELECT * FROM q",
}
REJECTED = {
    "plain join": "SELECT * FROM t LEFT JOIN query_texts qt ON qt.queryid = t.queryid",
    "plain read in a CTE": "WITH x AS (SELECT queryid, query FROM query_texts) SELECT * FROM t JOIN x USING (queryid)",
    "GROUP BY outside the read": "SELECT * FROM (SELECT queryid, query FROM query_texts) x GROUP BY ALL",
    "the Flask metric": "SELECT * FROM prom_last('pgwatch_query_info', [], 0, 0, 1)",
}


@pytest.mark.parametrize("sql", ACCEPTED.values(), ids=list(ACCEPTED))
def test_accepted_reads(sql: str) -> None:
    assert query_text_problems(sql) == []


@pytest.mark.parametrize("sql", REJECTED.values(), ids=list(REJECTED))
def test_rejected_reads(sql: str) -> None:
    assert query_text_problems(sql) != []
