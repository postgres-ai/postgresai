"""reporter/promsql.py and reporter/mon_client.py without a pgai-mon (no binary, no network)."""
import json

import pytest
import requests

from reporter import promsql
from reporter.mon_client import MonClient, MonError, format_value
from tests.reporter import mon_equiv as eq


@pytest.fixture(scope="module")
def cassette():
    return eq.load_cassette()


def test_every_reporter_query_compiles(cassette):
    """Each query the reporter sent in the recorded run compiles (the CI test without pgai-mon)."""
    for e in cassette.values():
        if e["kind"] == "instant":
            query, at = eq.mon_form(e["query"])
            sql = promsql.compile_instant(query, (e["time"] if at is None else at) * 1000)
        else:
            sql = promsql.compile_range(e["query"], int(e["start"] * 1000), int(e["end"] * 1000),
                                        promsql.duration_ms(e["step"]))
        assert sql.startswith("WITH ") and "SELECT l.name" in sql


@pytest.mark.parametrize("query", [
    "histogram_quantile(0.9, x)",
    "absent(x)",
    "x + ignoring(a) y",
    "x * on(a) group_left(b) y",
    "x > bool 1",
    "sum without (a) (x)",
    'x{__name__="y"}',
    '{job="a"}',
    "x[5m]",
    "rate(x[5m] offset 1h offset 1h)",
    "x % 2",
    "1 + 1",
    "label_replace(x, 'a', 'b', 'c', 'd')",
    "topk by (a) (3, x)",
    "x ==",
])
def test_outside_the_subset_is_unsupported(query):
    with pytest.raises(promsql.Unsupported):
        promsql.compile_instant(query, 1_794_236_820_000)


def test_parse_shapes():
    n = promsql.parse('topk(3, (sum by (queryid) (increase(m{a="1",b=~"x|y"}[1h]))))')
    assert isinstance(n, promsql.Agg) and n.op == "topk"
    inner = n.expr
    assert isinstance(inner, promsql.Agg) and inner.op == "sum" and inner.by == ["queryid"]
    sel = inner.expr.args[0]
    assert sel == promsql.Sel("m", [("a", "=", "1"), ("b", "=~", "x|y")], 3_600_000)

    n = promsql.parse("min_over_time(timestamp(m{a=\"1\"})[35d:1h])")
    assert isinstance(n.args[0], promsql.Sub) and n.args[0].window_ms == 35 * 86_400_000
    assert isinstance(n.args[0].expr, promsql.Call) and n.args[0].expr.fn == "timestamp"

    n = promsql.parse("a unless on (x, y) (b == 1)")
    assert n.op == "unless" and n.on == ["x", "y"] and n.rhs.op == "=="

    # precedence: * before -, and/unless before or
    n = promsql.parse("a - b * 2 or c and d")
    assert n.op == "or" and n.lhs.op == "-" and n.lhs.rhs.op == "*" and n.rhs.op == "and"
    assert promsql.parse("sum(m) by (a)").by == ["a"]
    assert promsql.parse("last_over_time(m[7d] offset 5m)").args[0].offset_ms == 300_000


def test_label_values_are_quoted_in_sql():
    sql = promsql.compile_instant('last_over_time(m{datname="it\'s \\"x\\""}[3h])', 1_000_000)
    assert "'it''s \"x\"'" in sql


def test_subquery_grid_is_victoriametrics_aligned():
    sql = promsql.compile_instant("max_over_time(sum(m)[7d:1m])", 1_794_236_820_000)
    # inner range: align_down(T - 7d - 1m - 5m, 1m) .. align_up(T + 1m, 1m)
    start = 1_794_236_820_000 - 7 * 86_400_000 - 60_000 - 300_000
    assert f"prom_last('m', CAST([] AS VARCHAR[][]), {start - start % 60_000}, {1_794_236_880_000}, 60000)" in sql


@pytest.mark.parametrize("start,end,step,expected", [
    (1000_000, 1000_000 + 10 * 3_600_000 + 123, 3_600_000, (1000_000, 1000_000 + 10 * 3_600_000 + 123)),
    (1_794_150_420_123, 1_794_236_820_123, 300_000, (1_794_150_300_000, 1_794_236_700_000)),
])
def test_range_bounds(start, end, step, expected):
    assert promsql.range_bounds(start, end, step) == expected


@pytest.mark.parametrize("text,ms", [("3h", 10_800_000), ("35d", 3_024_000_000), ("1h30m", 5_400_000),
                                     ("300", 300_000), ("30s", 30_000), ("2w", 1_209_600_000)])
def test_duration_ms(text, ms):
    assert promsql.duration_ms(text) == ms


@pytest.mark.parametrize("v,text", [
    # VictoriaMetrics v1.140.0 printed these (Go strconv 'f', -1)
    (7689907.0, "7689907"), (1e30, "1000000000000000000000000000000"), (1e-10, "0.0000000001"),
    (1234567890123456789012.0, "1234567890123456800000"), (123456789012345678.0, "123456789012345680"),
    (1 / 3, "0.3333333333333333"), (-0.000123, "-0.000123"), (0.0, "0"), (-0.0, "0"),
    (float("inf"), "+Inf"), (float("-inf"), "-Inf"), (float("nan"), "NaN"), (1794236833.448, "1794236833.448"),
])
def test_format_value(v, text):
    assert format_value(v) == text
    if v == v:
        assert float(text.replace("Inf", "inf")) == v


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self.body


class _Session:
    def __init__(self, body, status=200):
        self.body, self.status, self.sent = body, status, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.sent.append((url, json["sql"], headers))
        return _Resp(self.body, self.status)


def _rows(columns, rows):
    return {"columns": columns, "rows": rows}


def test_instant_result_shape_and_order():
    s = _Session(_rows(["name", "labels", "t", "value"], [
        ["m", '{"datname":"b"}', [1_794_236_820_000], [2.0]],
        [None, '{"datname":"a"}', [1_794_236_820_000], ["+Inf"]],
    ]))
    c = MonClient("http://mon:8765/", token="tok", clock=lambda: 1_794_236_820, session=s)
    r = c.query_instant('last_over_time(m[3h])')
    # VictoriaMetrics' order: by metric name, then by the label set
    assert r == {"status": "success", "data": {"resultType": "vector", "result": [
        {"metric": {"__name__": "m", "datname": "b"}, "value": [1_794_236_820, "2"]},
        {"metric": {"datname": "a"}, "value": [1_794_236_820, "+Inf"]}]}}
    url, sql, headers = s.sent[0]
    assert url == "http://mon:8765/sql" and headers["authorization"] == "Bearer tok"
    assert "prom_last_over_time('m'" in sql


def test_topk_instant_sorted_by_value_then_labels():
    s = _Session(_rows(["name", "labels", "t", "value"], [
        [None, '{"a":"z"}', [1000], [5.0]], [None, '{"a":"y"}', [1000], [5.0]], [None, '{"a":"x"}', [1000], [9.0]]]))
    r = MonClient("http://m", clock=lambda: 1, session=s).query_instant("topk(3, sum by (a) (x))")
    assert [x["metric"]["a"] for x in r["data"]["result"]] == ["x", "y", "z"]


def test_range_result_shape():
    s = _Session(_rows(["name", "labels", "t", "value"], [
        [None, "{}", [3_600_000, 7_200_000], [1.0, 3.5]]]))
    r = MonClient("http://m", session=s).query_range("sum(rate(x[5m]))", 3600, 7200, "1h")
    assert r == [{"metric": {}, "values": [[3600, "1"], [7200, "3.5"]]}]
    # one row per series: the answer's size does not grow with the labels times the steps
    assert "list(value ORDER BY t)" in s.sent[0][1] and "GROUP BY k" in s.sent[0][1]


def test_truncated_answer_is_an_error(capsys):
    """pgai-mon answers 200 with truncated=true past its caps: never a short result."""
    body = dict(_rows(["name", "labels", "t", "value"], [[None, "{}", [3_600_000], [1.0]]]), truncated=True)
    c = MonClient("http://m", clock=lambda: 1, session=_Session(body))
    with pytest.raises(MonError, match="truncated the result of the index definitions"):
        c.index_definitions()
    assert c.query_range("sum(x)", 0, 3600, "60s") == []
    assert c.query_instant("sum(x)")["status"] == "error"
    out = capsys.readouterr().out
    assert "truncated" in out and "(query: sum(x))" in out


class _Flaky(_Session):
    """Answers each status (or raises each exception) in turn, then the body."""

    def __init__(self, body, failures):
        super().__init__(body)
        self.failures = list(failures)

    def post(self, url, json=None, headers=None, timeout=None):
        self.sent.append((url, json["sql"], headers))
        if self.failures:
            f = self.failures.pop(0)
            if isinstance(f, Exception):
                raise f
            return _Resp({"error": {"type": "busy", "message": "busy"}}, f)
        return _Resp(self.body, 200)


@pytest.mark.parametrize("failures", [[429, 503], [408], [requests.ConnectionError("refused")]])
def test_transient_failures_are_retried(failures):
    s, slept = _Flaky(_rows(["a"], [[1]]), failures), []
    c = MonClient("http://m", session=s, sleep=slept.append)
    assert c.sql("select 1") == [{"a": 1}]
    assert len(s.sent) == len(failures) + 1 and slept == [1.0, 2.0][:len(failures)]


@pytest.mark.parametrize("failure", [400, 403, 500, requests.ReadTimeout("slow")])
def test_other_failures_are_not_retried(failure):
    s, slept = _Flaky(_rows(["a"], [[1]]), [failure]), []
    with pytest.raises(MonError):
        MonClient("http://m", session=s, sleep=slept.append).sql("select 1")
    assert len(s.sent) == 1 and slept == []


def test_retries_end_and_the_log_names_query_and_status(capsys):
    s, slept = _Flaky(_rows(["a"], [[1]]), [503, 503, 503]), []
    c = MonClient("http://m", clock=lambda: 1, session=s, sleep=slept.append)
    assert c.query_instant("sum(x)")["status"] == "error"
    assert len(s.sent) == 3 and len(slept) == 2
    out = capsys.readouterr().out
    assert "HTTP 503" in out and "(query: sum(x))" in out


def test_errors_become_status_error_or_empty():
    s = _Session({"error": {"type": "timeout", "message": "too slow"}}, status=504)
    c = MonClient("http://m", session=s)
    r = c.query_instant("sum(x)")
    assert r["status"] == "error" and "too slow" in r["error"]
    assert c.query_range("sum(x)", 0, 3600, "60s") == []
    with pytest.raises(MonError):
        c.sql("select 1")


def test_query_texts_and_index_definitions():
    s = _Session(_rows(["dbname", "queryid", "query"], [["db1", "-5", "select 1234567"], ["db2", "7", "select 2"]]))
    c = MonClient("http://m", session=s)
    assert c.query_texts(query_text_limit=8) == {"db1": {"-5": "select 1..."}, "db2": {"7": "select 2"}}
    assert "cluster" not in s.sent[-1][1] and "node_name" not in s.sent[-1][1]
    s.body = _rows(["dbname", "name", "definition"], [["db1", "i1", "CREATE INDEX i1 ON t (a)"]])
    assert c.index_definitions("db1") == {"i1": "CREATE INDEX i1 ON t (a)"}
    assert c.index_definitions() == {"db1.i1": "CREATE INDEX i1 ON t (a)"}
    assert "dbname = 'db1'" in s.sent[-2][1] and "valid_to IS NULL" in s.sent[-1][1]


def test_query_texts_of_one_cluster_and_node():
    """Two clusters with the same database name and queryid: each report gets its own text."""
    stored = [  # (cluster, node_name, dbname, queryid, query) as pgai-mon holds them
        ("c1", "n1", "app", 7, "select from c1"), ("c2", "n1", "app", 7, "select from c2"),
        (None, None, "app", 9, "select before the source columns")]

    class Store(_Session):  # answers the WHERE conditions query_texts writes
        def post(self, url, json=None, headers=None, timeout=None):
            sql = json["sql"]
            self.sent.append((url, sql, headers))
            keep = [r for r in stored if all(
                v is None or f"{col} = '{v}'" in sql or f"{col} =" not in sql
                for col, v in (("cluster", r[0]), ("node_name", r[1])))]
            return _Resp(_rows(["dbname", "queryid", "query"], [[r[2], str(r[3]), r[4]] for r in keep]), 200)

    c = MonClient("http://m", session=Store(None))
    assert c.query_texts(cluster="c1", node_name="n1") == {
        "app": {"7": "select from c1", "9": "select before the source columns"}}
    assert c.query_texts(cluster="c2", node_name="n1")["app"]["7"] == "select from c2"
    assert "(cluster = 'c2' OR cluster IS NULL) AND (node_name = 'n1' OR node_name IS NULL)" in c.session.sent[-1][1]
    pages = c.query_texts(cluster="it's")  # a cluster alone, quoted
    assert pages == {"app": {"9": "select before the source columns"}}
    assert "cluster = 'it''s'" in c.session.sent[-1][1] and "node_name" not in c.session.sent[-1][1]


def test_query_texts_continue_after_a_truncated_answer():
    class Pages(_Session):
        def post(self, url, json=None, headers=None, timeout=None):
            self.sent.append((url, json["sql"], headers))
            return _Resp(self.body.pop(0), 200)

    cols = ["dbname", "queryid", "query"]
    s = Pages([dict(_rows(cols, [["a", "-5", "q1"], ["it's", "7", "q2"]]), truncated=True),
               dict(_rows(cols, [["it's", "9", "q3"], [None, "1", "q4"]]), truncated=True),
               _rows(cols, [[None, "2", "q5"]])])
    c = MonClient("http://m", session=s)
    assert c.query_texts() == {"a": {"-5": "q1"}, "it's": {"7": "q2", "9": "q3"}, None: {"1": "q4", "2": "q5"}}
    first, second, third = (sent[1] for sent in s.sent)
    assert "ORDER BY dbname NULLS LAST, id" in first and "dbname >" not in first
    assert "dbname > 'it''s' OR dbname IS NULL OR (dbname = 'it''s' AND queryid > 7)" in second
    assert "dbname IS NULL AND queryid > 1" in third
    # an answer cut before its first row cannot be continued
    with pytest.raises(MonError, match="truncated"):
        MonClient("http://m", session=_Session(dict(_rows(cols, []), truncated=True))).query_texts()


def test_generator_routes_to_mon(monkeypatch):
    from reporter.postgres_reports import PostgresReportGenerator
    monkeypatch.setenv("PGAI_MON_URL", "http://pgai-mon:8765")
    gen = PostgresReportGenerator()
    assert gen.mon is not None and gen.mon.url == "http://pgai-mon:8765"
    calls = []
    monkeypatch.setattr(gen.mon, "query_instant", lambda q: calls.append(q) or {"status": "success"})
    monkeypatch.setattr(gen.mon, "query_texts", lambda limit, dbs, cluster=None, node_name=None: {"db": {"1": "x"}})
    monkeypatch.setattr(gen.mon, "index_definitions", lambda db: {"i": "d"})
    assert gen.query_instant("up") == {"status": "success"} and calls == ["up"]
    assert gen.get_queryid_queries_from_sink(10, ["db"]) == {"db": {"1": "x"}}
    assert gen.get_index_definitions_from_sink("db") == {"i": "d"}
    assert gen.pg_conn is None
    monkeypatch.delenv("PGAI_MON_URL")
    assert PostgresReportGenerator().mon is None
