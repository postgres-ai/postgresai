"""K003 per-statement window facts: calls, rows, buffers, WAL, temp, text and the monitoring flag."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from reporter import postgres_reports as pr
from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_report

TIMELINE = [3600 * h for h in range(470_000, 470_003)]  # three hours
PG = {"version": "17.6", "server_version_num": "170006", "server_major_ver": "17", "server_minor_ver": "6"}

# Per metric, per queryid: hourly increases (summed by the generator).
WINDOW = {
    "pgwatch_pg_stat_statements_calls": {"111": [10, 10, 10], "-222": [1, 0, 2]},
    "pgwatch_pg_stat_statements_rows": {"111": [10, 10, 10], "-222": [5, 0, 5]},
    "pgwatch_pg_stat_statements_shared_bytes_hit_total": {"111": [8192, 8192, 8192]},
    "pgwatch_pg_stat_statements_shared_bytes_read_total": {"111": [0, 0, 0]},
    "pgwatch_pg_stat_statements_wal_bytes": {"-222": [100, 0, 50]},
    # temp_bytes_written: no series at all -> the key stays absent (not known)
}
TEXTS = {"postgres": {"111": "select   *\n  from t where id = $1", "-222": "with x as ( /* pgwatch_generated */ select 1)"}}


def _series(qid_values: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    return [{"metric": {"queryid": qid}, "values": [[ts, str(v)] for ts, v in zip(TIMELINE, vals)]}
            for qid, vals in qid_values.items()]


@pytest.fixture(name="generator")
def fixture_generator(monkeypatch: pytest.MonkeyPatch) -> PostgresReportGenerator:
    gen = PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")
    monkeypatch.setattr(gen, "_get_postgres_version_info", lambda *a, **k: PG)
    monkeypatch.setattr(gen, "get_all_databases", lambda *a, **k: ["postgres"])

    def hourly(cluster, node_name, db_name, metric_name="", hours=24, step_s=3600, k=3):
        if metric_name.endswith("exec_time_total"):
            # "333" ranked only in idle hours: zero time over the window.
            return {"111": [100.0, 100.0, 100.0], "-222": [50.0, 0.0, 50.0], "333": [0.0, 0.0, 0.0]}, [5.0, 5.0, 5.0], TIMELINE
        return {}, [0.0, 0.0, 0.0], TIMELINE

    monkeypatch.setattr(gen, "_get_hourly_topk_pgss_data", hourly)
    gen.sink_reads = 0

    def texts(limit=655360, db_names=None):
        gen.sink_reads += 1
        return TEXTS

    monkeypatch.setattr(gen, "get_queryid_queries_from_sink", texts)

    def query_range(query, start, end, step="30s"):
        assert step == "3600s"
        if query.startswith("count(count by (queryid)"):
            return [{"metric": {}, "values": [[TIMELINE[0], "42"], [TIMELINE[1], "57"], [TIMELINE[2], "50"]]}]
        if query.startswith("count(resets("):  # a reset of 55 statements in the second hour
            return gen.resets
        if query.startswith("sum(increase(pgwatch_pg_stat_statements_calls"):
            return [{"metric": {}, "values": [[ts, "100"] for ts in TIMELINE]}]
        metric = re.search(r"increase\((pgwatch_[a-z_]+)\{", query).group(1)
        assert 'queryid=~"' in query and "333" not in query
        return _series(WINDOW.get(metric, {}))

    gen.resets = [{"metric": {}, "values": [[TIMELINE[1], "55"], [TIMELINE[2], "1"]]}]
    monkeypatch.setattr(gen, "query_range", query_range)
    return gen


def _db(report: Dict[str, Any]) -> Dict[str, Any]:
    return next(iter(report["results"].values()))["data"]["postgres"]


@pytest.mark.unit
def test_k003_rows_carry_window_facts_text_and_monitoring_flag(generator) -> None:
    report = generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180)
    validate_report(report)
    db = _db(report)
    rows = {q["queryid"]: q for q in db["top_queries"]}
    assert set(rows) == {"111", "-222"}  # the zero-time row is not a top statement
    assert rows["111"]["calls"] == 30 and rows["111"]["rows"] == 30
    assert rows["111"]["shared_blks_hit_bytes"] == 3 * 8192 and rows["111"]["shared_blks_read_bytes"] == 0
    assert "wal_bytes" not in rows["111"] and "temp_bytes_written" not in rows["111"]  # absent = not known
    assert rows["-222"]["wal_bytes"] == 150 and rows["-222"]["calls"] == 3
    assert rows["111"]["query_text"] == "select * from t where id = $1"
    assert rows["111"]["is_monitoring"] is False and rows["-222"]["is_monitoring"] is True
    assert rows["111"]["relations"] == ["t"] and "relations" not in rows["-222"]
    s = db["summary"]
    assert s["total_calls"] == 300 and s["statements_exported_max"] == 57
    assert s["window_seconds"] == 3 * 3600 and s["export_rule"] == pr.K003_EXPORT_RULE
    assert s["window_start"].endswith("+00:00") and s["window_end"].endswith("+00:00")
    assert generator.sink_reads == 1
    assert s["statements_reset_max"] == 55 and s["statements_reset_at"].startswith("2023-")


@pytest.mark.unit
def test_k003_no_counter_reset_is_zero(generator) -> None:
    # `or vector(0)`: an hour with no reset answers 0
    generator.resets = [{"metric": {}, "values": [[ts, "0"] for ts in TIMELINE]}]
    s = _db(generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180))["summary"]
    assert s["statements_reset_max"] == 0 and "statements_reset_at" not in s


@pytest.mark.unit
def test_k003_a_failed_query_is_absent_never_zero(generator, monkeypatch: pytest.MonkeyPatch) -> None:
    """query_range returns [] on an HTTP error: one failed metric loses only its own key."""
    real = generator.query_range

    def failing(query, start, end, step="30s"):
        if "wal_bytes" in query or query.startswith("count(resets("):
            return []
        return real(query, start, end, step)

    monkeypatch.setattr(generator, "query_range", failing)
    db = _db(generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180))
    rows = {q["queryid"]: q for q in db["top_queries"]}
    assert "wal_bytes" not in rows["-222"] and rows["-222"]["calls"] == 3
    assert "statements_reset_max" not in db["summary"] and db["summary"]["total_calls"] == 300


@pytest.mark.unit
def test_k003_each_database_keeps_its_own_facts(generator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *a, **k: ["postgres", "app"])
    real = generator.query_range

    def by_db(query, start, end, step="30s"):
        if 'datname="app"' in query and query.startswith("sum(increase(pgwatch_pg_stat_statements_calls"):
            return [{"metric": {}, "values": [[ts, "7"] for ts in TIMELINE]}]
        return real(query, start, end, step)

    monkeypatch.setattr(generator, "query_range", by_db)
    report = generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180)
    validate_report(report)
    data = next(iter(report["results"].values()))["data"]
    assert data["postgres"]["summary"]["total_calls"] == 300 and data["app"]["summary"]["total_calls"] == 21
    assert generator.sink_reads == 1  # one sink read per report
    generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180)
    assert generator.sink_reads == 2  # a new report reads it again


@pytest.mark.unit
def test_k003_text_is_one_line_and_capped() -> None:
    text = pr._one_line("select\n" + "x, " * 500, pr.K003_TEXT_CHARS)
    assert "\n" not in text and len(text) == pr.K003_TEXT_CHARS and text.endswith("…")


@pytest.mark.unit
def test_k003_survives_failed_window_queries_and_sink(generator, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(generator, "query_range", broken)
    monkeypatch.setattr(generator, "get_queryid_queries_from_sink", broken)
    report = generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=180)
    validate_report(report)
    db = _db(report)
    assert [q["queryid"] for q in db["top_queries"]] == ["111", "-222"]
    assert all("calls" not in q and "query_text" not in q for q in db["top_queries"])
    assert "total_calls" not in db["summary"] and db["summary"]["export_rule"] == pr.K003_EXPORT_RULE
    assert "statements_reset_max" not in db["summary"]  # not read: absent, never 0


@pytest.mark.unit
def test_k003_export_rule_matches_the_collector_config() -> None:
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/pgwatch-prometheus/metrics.yml").read_text())
    metric = cfg["metrics"]["pg_stat_statements"]
    rule = pr.K003_EXPORT_RULE
    for version, sql in metric["sqls"].items():
        flat = " ".join(sql.split())
        assert f"rn <= {rule['top_n_per_database']}" in flat, version
        assert f"calls >= {rule['min_calls']} and exec_time_total >= {rule['min_exec_time_ms']}" in flat, version
        assert "order by exec_time_total desc" in flat, version
        assert MONITOR in sql, version
    assert cfg["presets"]["full"]["metrics"]["pg_stat_statements"] == rule["interval_seconds"]


MONITOR = pr.MONITORING_MARKER


@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("with q as (select c.oid from pg_class c join pg_namespace n on n.oid = c.relnamespace) select * from q",
     ["pg_class", "pg_namespace"]),
    ("insert into public.audit_events (a) values ($1)", ["public.audit_events"]),
    ("update app.jobs set state = $1 where id in (select id from app.queue)", ["app.jobs", "app.queue"]),
    ('select * from "Weird Name" w, lateral generate_series(1, 2)', ['"Weird Name"']),
    ("select app.tick($1)", []),
    ("insert into t (id, v) values ($1, $2) on conflict (id) do update set v = excluded.v", ["t"]),
    ("select id from jobs where state = $1 for update skip locked", ["jobs"]),
    ("select id from jobs for update of jobs nowait", ["jobs"]),
    ("select extract(epoch from created_at), trim(both $1 from name) from people", ["people"]),
    ("select * from t where a is distinct from b", ["t"]),
    ("with x as materialized (select * from t) select * from x", ["t"]),
    ("with x as not materialized (select 1) select * from x join y on true", ["y"]),
    ("select * from (select * from inner_t) s join outer_t o on true", ["inner_t", "outer_t"]),
    ("insert into a(x, y) select x, y from b", ["a", "b"]),
    ("select array(select x from b) from a where n > (select 1 from c) limit (select 2 from d)", ["b", "a", "c"]),
    ("with a as (select 1), b(x) as (select 2) select * from a join b on true join t1 on true join t2 on true "
     "join t3 on true join t4 on true", ["t1", "t2", "t3"]),
])
def test_k003_relations_name_a_cut_statement(text, expected) -> None:
    assert pr.k003_relations(text) == expected


@pytest.mark.unit
def test_k003_reads_enough_text_to_see_every_collector_statements_marker() -> None:
    """is_monitoring reads K003_TEXT_FETCH_CHARS of the text: the marker of every collector SQL must fall in it."""
    root = Path(__file__).resolve().parents[2] / "config"
    for path in (root / "pgwatch-prometheus/metrics.yml", root / "pgwatch-postgres/metrics.yml"):
        for name, metric in (yaml.safe_load(path.read_text()).get("metrics") or {}).items():
            for version, sql in ((metric or {}).get("sqls") or {}).items():
                at = str(sql).find(MONITOR)
                assert at < pr.K003_TEXT_FETCH_CHARS - len(MONITOR), (path.name, name, version, at)


@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("select * from t join generate_series(1, 3) g on true join unnest($1) u on true", ["t"]),
    ('select * from public."order items" o join "a b"."c d" x on true', ['public."order items"', '"a b"."c d"']),
    ("select 'from fake' as s, x from t -- from also_fake\n join u on true /* join nope */", ["t", "u"]),
    ("select $f$ from inside $f$ from t", ["t"]),
    ("delete from only q where id = $1", ["q"]),
    ("select x as from, count(*) as update from t join u on true", ["t", "u"]),
    ("select * from q for update limit 1", ["q"]),
    ("copy t (a, b) from stdin with (format csv)", []),
    ("copy t from program 'cat x'", []),
    ('/* c */ fetch forward 2000 from "c_7f_1"', []),
    ("select E'it\\'s from fake' from t", ["t"]),
    ("select * into temporary table t2 from t", ["t2", "t"]),
    ("select * into unlogged t2 from t", ["t2", "t"]),
    ("select x update from t", ["t"]),
    ("select a as update order by 1", []),
    ("insert into temp as x (a) values (1)", ["temp"]),
    ("create trigger g before update on t for each row execute function f()", []),
    ("prepare p(int) as update t set a = $1", ["t"]),
    ("select * from program join stdin s on true", ["program", "stdin"]),
])
def test_k003_relations_read_tokens_not_text(text, expected) -> None:
    assert pr.k003_relations(text) == expected


@pytest.mark.unit
def test_k003_a_cut_text_never_names_its_split_last_word() -> None:
    """The sink cuts a text at K003_TEXT_FETCH_CHARS and appends "...": its last word may be split."""
    pad = "select * from t where x = " + "1 + " * 6000
    cut = pad[:pr.K003_TEXT_FETCH_CHARS - 13] + " from orders_history o"
    sink = cut[:pr.K003_TEXT_FETCH_CHARS] + "..."
    assert pr.k003_relations(sink, truncated=True) == ["t"]
    for tail in ("select * from public.ord", "select * from public.", 'select * from public."ord it'):
        assert pr.k003_relations(tail + "...", truncated=True) == [], tail
    assert pr.k003_relations("select 1 from t join orders o on true", truncated=False) == ["t", "orders"]


@pytest.mark.unit
def test_k003_relations_are_one_linear_pass() -> None:
    import time
    text = " union all ".join(["select id from events where k = $1"] * 450)  # about 20,000 characters
    chain = "select * from a where " + " or ".join(f"x in (select y from items where z = {i})" for i in range(400))
    start = time.perf_counter()
    assert pr.k003_relations(text) == ["events"] and pr.k003_relations(chain) == ["a", "items"]
    assert time.perf_counter() - start < 0.5
