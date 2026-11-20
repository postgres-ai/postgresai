"""The reporter on pgai-mon gives the same answers and the same reports as on VictoriaMetrics.

Needs a pgai-mon binary ($PGAI_MON_BIN, or mon/target/*/pgai-mon); skipped without one (failed when CI is set). Builds a
store from tests/reporter/mon_fixture/ (real pgwatch data, 8 days, 3,705 series, plus 95 synthetic
series for what the demo box lacks: host metrics, lock waits, invalid indexes, pg_stat_kcache,
stats_reset, the older btree bloat gauges), serves it on 127.0.0.1 and compares, for the same
samples:

- every query in the cassette (what VictoriaMetrics v1.140.0 answered when the reporter ran at
  FIXTURE_AT) with the reporter's answer from pgai-mon (reporter/mon_client.py);
- every report (all checks, the per-query files, and the paths generate_all_reports leaves out:
  D004/F001/G001 read directly, K001 over 30 minutes) generated from the cassette with the same
  report generated from pgai-mon, and the latter against reporter/schemas.
- on pgai-mon only, the per-query files' batched metrics (#524) with the per-statement query,
  for every statement, node and database.

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

DIRECT = ["D004_direct", "F001_direct", "G001_direct", "K001_30min", "A014_clickhouse", "A014_clickhouse_recent"]
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
    # the fixture must keep exercising the queries (an empty store would match trivially); 499
    # recorded, 427 non-empty since the per-query files batch their statements (#524)
    assert len(cassette) >= 480 and nonempty >= 400, (len(cassette), nonempty)


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


@pytest.mark.parametrize("chunk", [100, 7])
def test_batched_per_query_metrics_equal_per_statement_queries(mon, reports, chunk):
    """#524: the per-query files fetch each metric for many statements in one query. On pgai-mon,
    for every statement, node and database of the fixture, that equals the per-statement query.
    A chunk of 7 (not a divisor of the fixture's statement count) merges several chunks."""
    _, new, _ = reports
    gen = eq.make_generator(mon_url=mon.url)
    gen.QUERYID_CHUNK = chunk
    checks = {c: new[c] for c in CHECKS}
    with eq.frozen():
        queryids_by_db = gen.extract_queryids_from_reports(checks)
        nodes = gen.get_all_nodes("local")
        node_names = [nodes["primary"], *nodes.get("standbys", [])]
        ids = sorted(set().union(*queryids_by_db.values()))
        assert len(ids) > 2 * chunk or chunk == 100, len(ids)
        compared, diffs = 0, []
        for node in node_names:
            batch = gen.get_queries_metrics_from_prometheus("local", node, ids, hours=24)
            for db, qids in queryids_by_db.items():
                for qid in qids:
                    one = gen.get_query_metrics_from_prometheus("local", node, db, qid, hours=24)
                    one.pop("time_range")
                    got = batch.get((db, qid), {})
                    compared += bool(one)
                    if list(got) != list(one) or not all(eq.close(got[k], one[k]) for k in one):
                        diffs.append((node, db, qid, one, got))
    assert not diffs, diffs[:3]
    # the fixture must keep exercising it: 24 statement x node x database combinations with
    # metrics in the fixture
    assert compared >= 20, compared


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
    sizing = data("A014")
    assert sizing["observed_size"]["size"] == "db.t4g.large"
    assert sizing["observed_size"]["region"] == "eu-central-1"
    assert sizing["observed_size"]["multi_az"] is True
    assert sizing["observed_size"]["storage"]["max_size_gb"] == 600
    assert sizing["observed_size"]["since_epoch"] is not None
    for name in ("cpu_credit_balance", "cpu_surplus_balance", "burst_balance_pct",
                 "ebs_io_balance_pct", "ebs_byte_balance_pct"):
        assert any(v is not None for v in sizing["series"][name])
    for check in ("A014_clickhouse", "A014_clickhouse_recent"):
        validate_report(new[check])
        ch = new[check]["results"]["clickhouse"]["data"]["observed_size"]
        assert ch["vcpus"] == 4 and ch["memory_bytes"] == 16 * 2 ** 30
        assert ch["size_source"] == "metrics"
        assert (ch["since_epoch"] is not None) == (check == "A014_clickhouse")
    spills = data("K004")["target_database"]
    q = next(q for q in spills["top_queries"] if q["queryid"] == "900000000001")
    assert q["calls"] > 0 and q["temp_bytes_read"] > 0 and q["exec_time_ms"] > 0
    assert spills["summary"]["database_temp"]["bytes"]["total"] > 0
    assert spills["summary"]["database_temp"]["files"]["total"] > 0
    assert "work_mem" in new["K004"]["results"]["node-01"]["settings"]
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


def test_query_texts_continued_in_pages_are_the_same(mon):
    """The continuation after a cut answer, on pgai-mon itself: pages of 7 rows (pgai-mon's own
    caps are far above the fixture) give what one answer gives."""
    whole = MonClient(mon.url).query_texts()
    assert sum(len(v) for v in whole.values()) > 14

    class Paged(MonClient):
        def _page(self, text):
            rows, _ = super()._page(text + " LIMIT 7")
            return rows, len(rows) == 7

    assert Paged(mon.url).query_texts() == whole


@pytest.mark.parametrize("window,range_query,keeps_name", [
    ("10799s", False, True), ("3h", False, False), ("3h", True, True),
])
def test_average_metric_name_matches_victoriametrics(mon, window, range_query, keeps_name):
    # VM v1.140.0 with its default cache rewrites instant averages at 3 h to sum/count.
    name = "host_cpu_utilization_percent"
    query = f'avg_over_time({name}{{cluster="local",node_name="node-01"}}[{window}])'
    client = MonClient(mon.url, clock=lambda: FIXTURE_AT)
    if range_query:
        rows = client.query_range(query, FIXTURE_AT - 600, FIXTURE_AT, "300s")
    else:
        result = client.query_instant(query)
        assert result["status"] == "success"
        rows = result["data"]["result"]
    expected_name = name if keeps_name else None
    assert rows and all(r["metric"].get("__name__") == expected_name for r in rows)
