"""Old (VictoriaMetrics) vs new (pgai-mon) reporter answers on the same fixture samples.

- A cassette (mon_fixture/vm_cassette.jsonl.gz) holds every query the reporter sent to
  VictoriaMetrics v1.140.0 while it generated all reports at FIXTURE_AT, with VM's answer.
  It is recorded by mon_fixture/record.py against a scratch VM loaded with the same samples.
- `run_reports(backend)` runs the reporter (all checks plus the per-query files) with its clock
  frozen at FIXTURE_AT and its queries answered by a backend: the cassette (the old stack's
  answers) or pgai-mon (the new path, through reporter/mon_client.py).
- `diff_prom` / `diff_json` compare with the tolerances listed in TOLERANCES.
"""
from __future__ import annotations

import contextlib
import gzip
import json
import math
import os
import time as _time
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from tests.reporter.mon_harness import FIXTURE, FIXTURE_AT

CASSETTE = FIXTURE / "vm_cassette.jsonl.gz"

# Everything not listed here must be identical.
TOLERANCES = {
    "float": "relative 1e-9 plus absolute 1e-12, compared as numbers, not as printed: Rust and Go "
             "may add in a different order (sums over series, reset corrections), and "
             "VictoriaMetrics stores a sample as a decimal of about 12 significant digits "
             "(5.999554672230515 reads back as 5.99955467223)",
    "topk_ties": "VictoriaMetrics breaks topk ties arbitrarily (Go map order and an unstable sort), "
                 "pgai-mon by the label set. The cassette stores VictoriaMetrics' topk answers with "
                 "the ties re-broken pgai-mon's way; record.py checks that this is the only change "
                 "(same values per step, same series above the k-th value)",
}
REL, ABS = 1e-9, 1e-12


# ----------------------------------------------------------------------------------------------
# Frozen clock
# ----------------------------------------------------------------------------------------------
class _FrozenTime:
    def __init__(self, at: float):
        self._at = at

    def time(self) -> float:
        return self._at

    def __getattr__(self, name):
        return getattr(_time, name)


def _frozen_datetime(at: float):
    class FrozenDatetime(_datetime):
        @classmethod
        def now(cls, tz=None):
            return _datetime.fromtimestamp(at, tz)

        @classmethod
        def utcnow(cls):
            return _datetime.utcfromtimestamp(at)
    return FrozenDatetime


@contextlib.contextmanager
def frozen(at: float = FIXTURE_AT) -> Iterator[None]:
    """The reporter module's time.time() and datetime.now() return `at`; local time is UTC."""
    import reporter.postgres_reports as pr
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    _time.tzset()
    saved = pr.time, pr.datetime
    pr.time, pr.datetime = _FrozenTime(at), _frozen_datetime(at)
    try:
        yield
    finally:
        pr.time, pr.datetime = saved
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        _time.tzset()


# ----------------------------------------------------------------------------------------------
# What the pgwatch Postgres sink returned, from the fixture's texts and definitions
# ----------------------------------------------------------------------------------------------
def _jsonl(name: str) -> List[dict]:
    return [json.loads(x) for x in (FIXTURE / name).read_text().splitlines() if x.strip()]


def sink_query_texts(query_text_limit: int = 655360, db_names: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    """The old sink query: distinct on (dbname, queryid) ... order by time desc."""
    best: Dict[Tuple[str, str], dict] = {}
    for r in _jsonl("query_texts.jsonl"):
        if db_names and r["dbname"] not in db_names:
            continue
        k = (r["dbname"], r["queryid"])
        if k not in best or (r["last_seen_ms"], r["query"]) > (best[k]["last_seen_ms"], best[k]["query"]):
            best[k] = r
    out: Dict[str, Dict[str, str]] = {}
    for (db, qid), r in sorted(best.items()):
        text = r["query"] or ""
        if len(text) > query_text_limit:
            text = text[:query_text_limit] + "..."
        out.setdefault(db, {})[qid] = text
    return out


def sink_index_definitions(db_name: Optional[str] = None) -> Dict[str, str]:
    latest: Dict[Tuple[str, str, str], dict] = {}
    for r in _jsonl("objects.jsonl"):
        if r["kind"] != "index" or (db_name and r["dbname"] != db_name):
            continue
        k = (r["dbname"], r["schema"], r["name"])
        if k not in latest or r["ts_ms"] >= latest[k]["ts_ms"]:
            latest[k] = r
    out: Dict[str, str] = {}
    for (db, _schema, name), r in sorted(latest.items()):
        if r["definition"] is not None:
            out.setdefault(name if db_name else f"{db}.{name}", r["definition"])
    return out


# ----------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------
def instant_key(query: str) -> str:
    return json.dumps(["instant", query, FIXTURE_AT])


def range_key(query: str, start: float, end: float, step: str) -> str:
    return json.dumps(["range", query, repr(float(start)), repr(float(end)), str(step)])


def load_cassette(path: Path = CASSETTE) -> Dict[str, Any]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            e = json.loads(line)
            out[e["key"]] = e
    return out


class Missing(KeyError):
    pass


def attach_cassette(gen, cassette: Dict[str, Any], misses: List[str]) -> None:
    """Answer gen's queries from the cassette, as VictoriaMetrics did; sink reads from the fixture."""
    def instant(query: str):
        e = cassette.get(instant_key(query))
        if e is None:
            misses.append(instant_key(query))
            return {}
        return e["response"]

    def rng(query, start_time, end_time, step="30s"):
        k = range_key(query, start_time.timestamp(), end_time.timestamp(), step)
        e = cassette.get(k)
        if e is None:
            misses.append(k)
            return []
        return e["response"]
    gen.query_instant = instant
    gen.query_range = rng
    gen.get_queryid_queries_from_sink = sink_query_texts
    gen.get_index_definitions_from_sink = sink_index_definitions


def make_generator(mon_url: Optional[str] = None):
    from reporter.postgres_reports import PostgresReportGenerator
    return PostgresReportGenerator(prometheus_url="http://127.0.0.1:9", postgres_sink_url="",
                                   sources_path=str(FIXTURE / "no-sources.yml"), mon_url=mon_url)


def run_reports(gen, out_dir: Path) -> Dict[str, Any]:
    """Every report main() writes for one cluster (all checks, then the per-query files), at the
    frozen time. Returns {check_id or per-query file name: JSON}."""
    with frozen():
        cwd = os.getcwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(out_dir)
        try:
            from reporter.wait_ready import targets_with_data
            ready = sorted(targets_with_data(gen.query_instant))
            clusters = gen.get_all_clusters()
            assert clusters == ["local"], clusters
            reports = gen.generate_all_reports("local", None, True)
            files = gen.generate_per_query_jsons(reports, "local", node_name=None, query_text_limit=66560,
                                                 hours=24, write_immediately=False, include_cluster_prefix=False)
            # Paths generate_all_reports leaves out: the settings reports read directly (the CLI's
            # --check-id, and the fallback when A003 fails) and K001's sub-hourly window.
            extra = {"D004_direct": gen.generate_d004_pgstat_settings_report("local", "node-01"),
                     "F001_direct": gen.generate_f001_autovacuum_settings_report("local", "node-01"),
                     "G001_direct": gen.generate_g001_memory_settings_report("local", "node-01"),
                     "K001_30min": gen.generate_k001_query_calls_report("local", "node-01", 30, use_hourly=False)}
        finally:
            os.chdir(cwd)
    out = {k: json.loads(json.dumps(v)) for k, v in {**reports, **extra}.items()}
    out["_wait_ready_targets"] = [list(t) for t in ready]
    for f in files or []:
        out[f["filename"]] = json.loads(json.dumps(f["data"]))
    return out


# ----------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------
def close(a: float, b: float) -> bool:
    if a == b or (a != a and b != b):
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= REL * max(abs(a), abs(b)) + ABS


def _num(s: Any) -> float:
    return float({"+Inf": "inf", "-Inf": "-inf"}.get(s, s))


def diff_prom(kind: str, old: Any, new: Any) -> List[str]:
    """Differences between two answers of the same query ([] = equivalent)."""
    out: List[str] = []
    if kind == "instant":
        if (old or {}).get("status") != (new or {}).get("status"):
            return [f"status {(old or {}).get('status')} vs {(new or {}).get('status')}"]
        old, new = (old.get("data") or {}).get("result") or [], (new.get("data") or {}).get("result") or []
    key = lambda r: tuple(sorted(r["metric"].items()))
    ko, kn = [key(r) for r in old], [key(r) for r in new]
    if set(ko) != set(kn):
        miss, extra = set(ko) - set(kn), set(kn) - set(ko)
        out.append(f"series: {len(miss)} only in VM {sorted(miss)[:2]}, {len(extra)} only in pgai-mon {sorted(extra)[:2]}")
    nmap = {key(r): r for r in new}
    for r in old:
        n = nmap.get(key(r))
        if n is None:
            continue
        if kind == "instant":
            if r["value"][0] != n["value"][0] or not close(_num(r["value"][1]), _num(n["value"][1])):
                out.append(f"{dict(key(r))}: {r['value']} vs {n['value']}")
        else:
            ts_o, ts_n = [p[0] for p in r["values"]], [p[0] for p in n["values"]]
            if ts_o != ts_n:
                out.append(f"{dict(key(r))}: timestamps {len(ts_o)} vs {len(ts_n)}, "
                           f"only VM {sorted(set(ts_o) - set(ts_n))[:3]}, only pgai-mon {sorted(set(ts_n) - set(ts_o))[:3]}")
                continue
            for (t, a), (_, b) in zip(r["values"], n["values"]):
                if not close(_num(a), _num(b)):
                    out.append(f"{dict(key(r))} at {t}: {a} vs {b}")
                    break
    if not out and ko != kn:
        # order: allowed only as a permutation of equal values (tie_order)
        vals = lambda rs: [_num(r["value"][1]) for r in rs] if kind == "instant" else None
        if kind != "instant" or vals(old) != vals(new):
            out.append(f"order differs: {ko[:3]} vs {kn[:3]}")
    return out


def diff_json(old: Any, new: Any, path: str = "") -> List[str]:
    """Structural JSON difference with the float tolerance; [] = equivalent."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            if k not in new:
                out.append(f"{path}/{k}: only in old")
            elif k not in old:
                out.append(f"{path}/{k}: only in new")
            else:
                out += diff_json(old[k], new[k], f"{path}/{k}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: list length {len(old)} vs {len(new)}"]
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out += diff_json(a, b, f"{path}[{i}]")
        return out
    if isinstance(old, bool) or isinstance(new, bool) or not (
            isinstance(old, (int, float)) and isinstance(new, (int, float))):
        return [] if old == new else [f"{path}: {old!r} vs {new!r}"]
    return [] if close(float(old), float(new)) else [f"{path}: {old!r} vs {new!r}"]
