"""The per-query files fetch every statement's metrics in a few batched queries (#524)."""
import re

import pytest

from reporter.postgres_reports import PostgresReportGenerator

METRICS = PostgresReportGenerator.PGSS_QUERY_METRICS


@pytest.fixture
def generator():
    return PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")


def _ok(*items):
    return {"status": "success", "data": {"resultType": "vector", "result": [
        {"metric": {"datname": db, "queryid": qid}, "value": [0, str(v)]} for db, qid, v in items]}}


def _warnings(capsys):
    """The reporter logs to stdout (reporter/logger.py), not through caplog."""
    return [line.split(" - WARNING - ", 1)[1] for line in capsys.readouterr().out.splitlines() if " - WARNING - " in line]


def _queryids(query):
    return re.search(r'queryid=~"([^"]*)"', query).group(1).split("|")


@pytest.mark.unit
def test_one_query_per_metric_and_chunk(generator, monkeypatch):
    sent = []
    monkeypatch.setattr(generator, "query_instant", lambda q: sent.append(q) or _ok())
    ids = [str(i) for i in range(250)]
    generator.get_queries_metrics_from_prometheus("c", "n1", ids, hours=24)

    assert len(sent) == 3 * len(METRICS)
    chunks = [sorted(_queryids(q)) for q in sent[::len(METRICS)]]
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert sorted(sum(chunks, [])) == sorted(ids)
    for q in sent:
        assert q.startswith("sum by (datname, queryid) (increase(pgwatch_pg_stat_statements_")
        assert '{cluster="c",node_name="n1",queryid=~"' in q
        assert q.endswith("[24h]))")
    assert {re.search(r"increase\((\w+)\{", q).group(1) for q in sent} == set(METRICS.values())


@pytest.mark.unit
def test_rows_of_every_chunk_are_merged(generator, monkeypatch):
    """Nodes with more than QUERYID_CHUNK statements: every chunk's rows reach the result."""
    def answer(q):
        if "_calls{" not in q:
            return _ok()
        return _ok(*[("db1", qid, int(qid) + 1) for qid in _queryids(q)])
    monkeypatch.setattr(generator, "query_instant", answer)
    ids = [str(i) for i in range(250)]
    out = generator.get_queries_metrics_from_prometheus("c", "n1", ids)
    assert out == {("db1", qid): {"calls": float(int(qid) + 1)} for qid in ids}


@pytest.mark.unit
def test_rows_map_to_statement_and_database_nonzero_only(generator, monkeypatch):
    def answer(q):
        if "_calls{" in q:
            return _ok(("db1", "1", 5), ("db2", "1", 7), ("db1", "2", 0))
        if "_rows{" in q:
            return _ok(("db1", "1", 3), ("db1", "2", "NaN"))
        return _ok()
    monkeypatch.setattr(generator, "query_instant", answer)
    out = generator.get_queries_metrics_from_prometheus("c", "n1", ["1", "2"])
    assert out == {("db1", "1"): {"calls": 5.0, "rows": 3.0}, ("db2", "1"): {"calls": 7.0}}
    # metric order as in the per-statement result
    assert list(out[("db1", "1")]) == ["calls", "rows"]


@pytest.mark.unit
@pytest.mark.parametrize("failure", [{"status": "error", "error": "413"}, {}, None, RuntimeError("boom")])
def test_failed_query_leaves_only_that_metric_out(generator, monkeypatch, capsys, failure):
    def answer(q):
        if "_rows{" in q:
            if isinstance(failure, Exception):
                raise failure
            return failure
        if "_calls{" in q:
            return _ok(("db1", "1", 5))
        return _ok()
    monkeypatch.setattr(generator, "query_instant", answer)
    out = generator.get_queries_metrics_from_prometheus("c", "n1", ["1"])
    assert out == {("db1", "1"): {"calls": 5.0}}
    assert _warnings(capsys) == ["Per-query files: rows on node n1 failed for 1 statements; left out of their files"]


@pytest.mark.unit
def test_empty_result_is_not_a_failure(generator, monkeypatch, capsys):
    monkeypatch.setattr(generator, "query_instant", lambda q: _ok())
    assert generator.get_queries_metrics_from_prometheus("c", "n1", ["1"]) == {}
    out = capsys.readouterr().out
    assert " - WARNING - " not in out and " - ERROR - " not in out


@pytest.mark.unit
def test_odd_ids_and_labels_are_escaped(generator, monkeypatch):
    sent = []
    monkeypatch.setattr(generator, "query_instant", lambda q: sent.append(q) or _ok())
    generator.get_queries_metrics_from_prometheus('cl"x', "n1", ["-12", "a.b|c"])
    q = sent[0]
    assert 'cluster="cl\\"x"' in q
    # the regex is an alternation of literals: '.' and '|' escaped, then backslashes escaped for PromQL
    assert 'queryid=~"-12|a\\\\.b\\\\|c"' in q


@pytest.mark.unit
def test_per_query_files_query_per_node_not_per_statement(generator, monkeypatch):
    """S1: 19 queries per node for up to 100 statements, whatever the number of databases."""
    ids = {str(i) for i in range(84)}
    monkeypatch.setattr(generator, "extract_queryids_from_reports", lambda r: {"db1": ids, "db2": {"3", "4"}})
    monkeypatch.setattr(generator, "get_queryid_queries_from_sink", lambda *a, **k: {})
    monkeypatch.setattr(generator, "get_all_nodes", lambda c: {"primary": "p", "standbys": ["s"]})
    sent = []
    monkeypatch.setattr(generator, "query_instant", lambda q: sent.append(q) or _ok(("db1", "3", 1)))
    out = generator.generate_per_query_jsons({}, "c", node_name=None, hours=24)
    assert len(out) == 84
    assert len(sent) == 2 * len(METRICS)
    assert sum('node_name="p"' in q for q in sent) == len(METRICS)
    f3 = next(f["data"] for f in out if f["data"]["query_id"] == "3")
    assert f3["results"]["s"]["db1"]["metrics"] == {k: 1.0 for k in METRICS}
    # no row for (db2, 3) in the batch: an empty block, not another database's metrics
    assert f3["results"]["s"]["db2"]["metrics"] == {}
    assert f3["results"]["p"]["db2"]["metrics"] == {}
    f4 = next(f["data"] for f in out if f["data"]["query_id"] == "4")
    assert f4["results"]["p"]["db1"]["metrics"] == {}
