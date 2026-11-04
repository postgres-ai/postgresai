"""H004: one row per candidate, from its latest series (#483).

The redundant_indexes metric's redundant_to_json label holds the covering
index's size, so a candidate gets a new series each time that index grows. Two
collections in the 3-hour lookback return two series of one index: the report
lists the candidate once, with the values and covering indexes of its latest
series, and its table size, scans and foreign-key flag from that same series.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report
from tests.reporter.test_h004_readings_and_overlap import COVER, NOW, _entry, _gen, _labels, _responses

_esc = PostgresReportGenerator._escape_promql_label

OLD_COVER = dict(COVER, index_size_bytes=999424)
EARLIER = _labels("accounts_email_dup_idx", "demo.accounts_email_status_idx", [OLD_COVER])
LATEST = _labels("accounts_email_dup_idx", "demo.accounts_email_status_idx", [COVER])


def _matching(rows: List[Dict[str, Any]]):
    """Rows whose labels agree with every label matcher the query names (as a metric store selects)."""
    def pick(q: str) -> List[Dict[str, Any]]:
        return [r for r in rows
                if all(f'{k}="{_esc(v)}"' in q for k, v in r["metric"].items() if f'{k}="' in q)]
    return pick


def _series(earlier: str, latest: str) -> List[Dict[str, Any]]:
    return [{"metric": EARLIER, "value": [NOW, earlier]}, {"metric": LATEST, "value": [NOW, latest]}]


def _two_series(stamps: Any) -> Dict[str, Any]:
    base = _responses(**{
        "pgwatch_redundant_indexes_index_size_bytes": _matching(_series("819200", "827392")),
        "pgwatch_redundant_indexes_table_size_bytes": _matching(_series("5000000", "5242880")),
        "pgwatch_redundant_indexes_index_usage{": _matching(_series("4", "9")),
        "pgwatch_redundant_indexes_supports_fk": _matching(_series("1", "0")),
    })
    # the latest-sample time of each series first: its needle contains the size metric's
    return {"tlast_over_time(pgwatch_redundant_indexes_index_size_bytes": stamps, **base}


def test_candidate_listed_once_from_its_latest_series(monkeypatch: pytest.MonkeyPatch) -> None:
    stamps = [{"metric": EARLIER, "value": [NOW, str(NOW - 9000)]},
              {"metric": LATEST, "value": [NOW, str(NOW - 1800)]}]
    report = _gen(monkeypatch, _two_series(stamps)).generate_h004_redundant_indexes_report("c", "n")
    validate_report(report)
    entry = _entry(report)
    assert entry["total_count"] == 1
    (dup,) = entry["redundant_indexes"]
    assert dup["index_size_bytes"] == 827392
    assert entry["total_size_bytes"] == 827392
    assert dup["redundant_to"][0]["index_size_bytes"] == COVER["index_size_bytes"]
    # the lookups read the same series, not whichever the store returned first
    assert dup["table_size_bytes"] == 5242880
    assert dup["index_usage"] == 9
    assert dup["supports_fk"] is False


def test_latest_series_wins_whatever_order_the_store_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    stamps = [{"metric": LATEST, "value": [NOW, str(NOW - 1800)]},
              {"metric": EARLIER, "value": [NOW, str(NOW - 9000)]}]
    responses = _two_series(stamps)
    responses["pgwatch_redundant_indexes_index_size_bytes"] = _matching(list(reversed(_series("819200", "827392"))))
    (dup,) = _entry(_gen(monkeypatch, responses).generate_h004_redundant_indexes_report("c", "n"))["redundant_indexes"]
    assert dup["index_size_bytes"] == 827392 and dup["index_usage"] == 9


def test_no_sample_times_still_one_row_per_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    # a store without the latest-sample function: one row per index, its values from one series
    report = _gen(monkeypatch, _two_series([])).generate_h004_redundant_indexes_report("c", "n")
    entry = _entry(report)
    assert entry["total_count"] == 1
    (dup,) = entry["redundant_indexes"]
    sizes = {dup["index_size_bytes"], dup["table_size_bytes"], dup["index_usage"]}
    assert sizes in ({827392, 5242880, 9}, {819200, 5000000, 4})
    cover_size = json.loads(LATEST["redundant_to_json"] if dup["index_usage"] == 9 else EARLIER["redundant_to_json"])
    assert dup["redundant_to"][0]["index_size_bytes"] == cover_size[0]["index_size_bytes"]
