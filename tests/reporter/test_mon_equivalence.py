"""The reporter on pgai-mon gives the same answers and the same reports as on VictoriaMetrics.

Needs a pgai-mon binary ($PGAI_MON_BIN, or mon/target/*/pgai-mon); skipped without one (failed when CI is set). Builds a
store from tests/reporter/mon_fixture/ (real pgwatch data, 8 days, 3,705 series, plus 70 synthetic
series for what the demo box lacks: host metrics, lock waits, invalid indexes, pg_stat_kcache,
stats_reset, the older btree bloat gauges), serves it on 127.0.0.1 and compares, for the same
samples:

- every query in the cassette (what VictoriaMetrics v1.140.0 answered when the reporter ran at
  FIXTURE_AT) with the reporter's answer from pgai-mon (reporter/mon_client.py);
- every report (all checks, the per-query files, and the paths generate_all_reports leaves out:
  D004/F001/G001 read directly, K001 over 30 minutes) generated from the cassette with the same
  report generated from pgai-mon, and the latter against reporter/schemas.

Tolerances: mon_equiv.TOLERANCES. Re-record the cassette with mon_fixture/record.py after a change
to any query string (the test names the missing queries).
"""
import logging
import os
from pathlib import Path

import pytest

from tests.reporter import mon_equiv as eq
from tests.reporter.mon_harness import FIXTURE_AT, MonServer, build_store, find_binary
from reporter.mon_client import MonClient
from reporter.report_schemas import validate_query_file, validate_report

pytestmark = pytest.mark.enable_socket

DIRECT = ["D004_direct", "F001_direct", "G001_direct", "K001_30min"]
CHECKS = ["A002", "A003", "A004", "A005", "A007", "A014", "D001", "D004", "F001", "F002", "F004", "F005",
          "G001", "G003", "H001", "H002", "H004", "I001", "K001", "K003", "K004", "K005", "K006", "K007",
          "K008", "M001", "M002", "M003", "N001"]


@pytest.fixture(scope="module")
def mon(tmp_path_factory):
    binary = find_binary()
    if not binary:
        # CI builds the binary (mon:build) for this test: a missing one is an error there.
        (pytest.fail if os.environ.get("CI") else pytest.skip)(
            "no pgai-mon binary: set PGAI_MON_BIN or build mon/")
    store = build_store(binary, tmp_path_factory.mktemp("mon") / "store")
    log = tmp_path_factory.mktemp("monlog") / "serve.log"
    with MonServer(binary, store, log=log) as srv:
        yield srv


@pytest.fixture(scope="module")
def cassette():
    return eq.load_cassette()


@pytest.fixture(scope="module")
def reports(mon, cassette, tmp_path_factory):
    logging.disable(logging.WARNING)
    try:
        misses = []
        old_gen = eq.make_generator()
        eq.attach_cassette(old_gen, cassette, misses)
        tmp = tmp_path_factory.mktemp("reports")
        old = eq.run_reports(old_gen, tmp / "old")
        new = eq.run_reports(eq.make_generator(mon_url=mon.url), tmp / "new")
    finally:
        logging.disable(logging.NOTSET)
    return old, new, misses


def test_every_cassette_query_matches_victoriametrics(mon, cassette):
    client = MonClient(mon.url, clock=lambda: FIXTURE_AT)
    diffs, nonempty = {}, 0
    for key, e in sorted(cassette.items()):
        if e["kind"] == "instant":
            new = client.query_instant(e["query"])
            assert new.get("status") == "success", (e["query"], new)
            nonempty += bool((e["response"].get("data") or {}).get("result"))
        else:
            new = client.query_range(e["query"], e["start"], e["end"], e["step"])
            nonempty += bool(e["response"])
        d = eq.diff_prom(e["kind"], e["response"], new)
        if d:
            diffs[e["query"]] = d[:3]
    assert not diffs, f"{len(diffs)} of {len(cassette)} queries differ: {list(diffs.items())[:5]}"
    # the fixture must keep exercising the queries (an empty store would match trivially)
    assert len(cassette) >= 1450 and nonempty >= 700, (len(cassette), nonempty)


def test_range_errors_are_not_silent(mon):
    """A failed range query is [] for the reporter (as before), but logged as an error."""
    client = MonClient(mon.url, clock=lambda: FIXTURE_AT)
    assert client.query_range("histogram_quantile(0.9, x)", FIXTURE_AT - 3600, FIXTURE_AT, "60s") == []
    bad = client.query_instant("absent(x)")
    assert bad["status"] == "error" and "absent" in bad["error"]


def test_cassette_covers_every_report_query(reports):
    _, _, misses = reports
    assert not misses, f"queries not in the cassette (re-record with mon_fixture/record.py): {misses[:5]}"


@pytest.mark.parametrize("check", CHECKS + DIRECT + ["_wait_ready_targets"])
def test_report_matches_victoriametrics(reports, check):
    old, new, _ = reports
    assert check in old and check in new
    assert eq.diff_json(old[check], new[check]) == []


def test_per_query_files_match_victoriametrics(reports):
    old, new, _ = reports
    files = sorted(k for k in old if k.endswith(".json"))
    assert len(files) >= 40, len(files)
    assert files == sorted(k for k in new if k.endswith(".json"))
    diffs = {f: eq.diff_json(old[f], new[f])[:3] for f in files if eq.diff_json(old[f], new[f])}
    assert not diffs, diffs


def test_reports_have_content(reports):
    """Guards the fixture: the compared reports are not empty shells."""
    _, new, _ = reports
    data = lambda c: new[c]["results"]["node-01"]["data"]
    assert data("H002")["target_database"]["unused_indexes"]
    assert data("H004")["target_database"]["redundant_indexes"]
    assert data("F004")["target_database"]["bloated_tables"]
    assert data("F005")["target_database"]["bloated_indexes"]
    assert data("K003")["target_database"]["top_queries"]
    assert data("N001")
    # the synthetic series reach their reports
    assert data("G003")["lock_waits"]["pairs"] and data("G003")["deadlock_stats"]["stats_reset"]
    invalid = data("H001")["target_database"]["invalid_indexes"]
    assert {i["index_name"] for i in invalid} == {"t10_c_idx", "pgbench_accounts_abalance_idx_ccnew"}
    assert all(i["listed_hours"] and i["reading_hours"] for i in invalid)
    assert data("A014")["host_metrics"] and data("A014")["memory"]["available_min_pct"] == 8.0
    assert data("G001")["activity"]["host_memory"]["available_min_at"]
    assert data("D004")["pg_stat_kcache_status"]["extension_available"]
    assert data("F005")["postgres"]["bloated_indexes"]
    assert data("H002")["target_database"]["stats_reset"]["stats_reset_epoch"]
    assert new["K001_30min"]["results"]["node-01"]["data"]
    assert new["_wait_ready_targets"] == [["local", "node-01"]]


# Reports the old path already produced outside their schema (not changed here; reporter/schemas
# stays as it is): the new path must fail the same way, and nothing else may fail.
KNOWN_SCHEMA_GAPS = {
    "K001": "Additional properties are not allowed ('other_calls_hourly' was unexpected)",
}


def _schema_error(report):
    try:
        validate_report(report)
    except Exception as e:  # jsonschema.ValidationError
        return str(e).splitlines()[0]
    return None


@pytest.mark.parametrize("check", CHECKS)
def test_new_reports_match_schemas(reports, check):
    old, new, _ = reports
    assert _schema_error(new[check]) == _schema_error(old[check]) == KNOWN_SCHEMA_GAPS.get(check)


def test_new_query_files_match_schemas(reports):
    _, new, _ = reports
    for f in (k for k in new if k.endswith(".json")):
        validate_query_file(new[f])
