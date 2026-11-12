#!/usr/bin/env python3
"""Record vm_cassette.jsonl.gz: every query the reporter sends while it generates all reports at
FIXTURE_AT, answered by VictoriaMetrics (dev tool, not run in CI).

Start a scratch VictoriaMetrics v1.140.0 with `-search.latencyOffset=0s` (the fixture is in the
past; instant queries carry time=FIXTURE_AT), load it with vm_load.py, then run from the repo root:

    python tests/reporter/mon_fixture/record.py --vm http://127.0.0.1:PORT

Instant queries go to /api/v1/query with time=FIXTURE_AT (the reporter itself sends none, so VM
uses "now"); range queries exactly as the reporter sends them. The reporter's other inputs (sink
reads) come from the fixture's JSONL files, as in the tests.

topk: VictoriaMetrics breaks ties arbitrarily (Go map order, unstable sort), so a run can list
any of several equal queries. The recorder asks VM for the inner expression too, re-breaks the
ties by (metric name, label set) as pgai-mon does, checks that VM's own answer has the same
values at every step and the same series above the k-th value, and stores that answer.
"""
import argparse
import gzip
import json
import re
import sys
import tempfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tests.reporter import mon_equiv as eq  # noqa: E402
from tests.reporter.mon_harness import FIXTURE_AT  # noqa: E402


def _sites():
    """The reporter source lines on the stack (file name, line): which check sent the query."""
    out, f = [], sys._getframe(2)
    while f:
        fn = f.f_code.co_filename
        if "/reporter/" in fn and "/tests/" not in fn:
            out.append([Path(fn).name, f.f_lineno])
        f = f.f_back
    return out


TOPK = re.compile(r"^\s*topk\(\s*(\d+)\s*,(.*)\)\s*$", re.S)


def _tie_key(metric):
    return metric.get("__name__", ""), sorted((k, v) for k, v in metric.items() if k != "__name__")


def canonical_topk(k, inner, vm_topk, kind):
    """topk over VM's own answer for the inner expression, ties broken by (name, label set) as
    pgai-mon does; asserts VM's topk differs from it only in the choice among ties."""
    if kind == "instant":
        inner = [{"metric": r["metric"], "values": [r["value"]]} for r in inner]
        vm_topk = [{"metric": r["metric"], "values": [r["value"]]} for r in vm_topk]
    steps = {}
    for r in inner:
        for t, v in r["values"]:
            if v not in ("NaN",):
                steps.setdefault(t, []).append((float(v), r["metric"], v))
    chosen = {}
    for t, cands in steps.items():
        cands.sort(key=lambda c: (-c[0], _tie_key(c[1])))
        for v, m, text in cands[:k]:
            key = json.dumps(sorted(m.items()))
            chosen.setdefault(key, {"metric": m, "values": []})["values"].append([t, text])
    vm_steps = {}
    for r in vm_topk:
        for t, v in r["values"]:
            vm_steps.setdefault(t, []).append((float(v), json.dumps(sorted(r["metric"].items()))))
    ours = {}
    for key, s in chosen.items():
        for t, v in s["values"]:
            ours.setdefault(t, []).append((float(v), key))
    assert set(vm_steps) == set(ours), "topk steps differ"
    for t in ours:
        a, b = sorted(vm_steps[t], reverse=True), sorted(ours[t], reverse=True)
        assert [x[0] for x in a] == [x[0] for x in b], f"topk values differ at {t}"
        kth = a[-1][0]
        assert {x[1] for x in a if x[0] > kth} == {x[1] for x in b if x[0] > kth}, f"topk members differ at {t}"
    out = [{"metric": s["metric"], "values": sorted(s["values"])} for s in chosen.values()]
    if kind == "instant":
        out = [{"metric": s["metric"], "value": s["values"][0]} for s in out]
        out.sort(key=lambda r: (-float(r["value"][1]), _tie_key(r["metric"])))
    else:
        out.sort(key=lambda r: sorted(r["metric"].items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", required=True)
    ap.add_argument("--out", default=str(eq.CASSETTE))
    a = ap.parse_args()
    base = a.vm.rstrip("/") + "/api/v1"
    entries = {}

    def vm_instant(query):
        r = requests.get(f"{base}/query", params={"query": query, "time": FIXTURE_AT}, timeout=120)
        resp = r.json() if r.status_code in (200, 400, 422) else {}
        resp.pop("stats", None)
        return resp

    def vm_range(query, params):
        r = requests.get(f"{base}/query_range", params=dict(params, query=query), timeout=120)
        body = r.json() if r.status_code == 200 else {}
        return body.get("data", {}).get("result", []) if body.get("status") == "success" else []

    def instant(query):
        resp = vm_instant(query)
        m = TOPK.match(query)
        if m and resp.get("status") == "success":
            inner = vm_instant(m.group(2))
            resp["data"]["result"] = canonical_topk(int(m.group(1)), inner["data"]["result"],
                                                    resp["data"]["result"], "instant")
        entries[eq.instant_key(query)] = {"key": eq.instant_key(query), "kind": "instant", "query": query,
                                          "time": FIXTURE_AT, "response": resp,
                                          "sites": entries.get(eq.instant_key(query), {}).get("sites", []) + [_sites()]}
        return resp

    def rng(query, start_time, end_time, step="30s"):
        params = {"start": start_time.timestamp(), "end": end_time.timestamp(), "step": step}
        resp = vm_range(query, params)
        m = TOPK.match(query)
        if m:
            resp = canonical_topk(int(m.group(1)), vm_range(m.group(2), params), resp, "range")
        k = eq.range_key(query, start_time.timestamp(), end_time.timestamp(), step)
        entries[k] = {"key": k, "kind": "range", "query": query, "start": start_time.timestamp(),
                      "end": end_time.timestamp(), "step": step, "response": resp,
                      "sites": entries.get(k, {}).get("sites", []) + [_sites()]}
        return resp

    gen = eq.make_generator()
    gen.query_instant = instant
    gen.query_range = rng
    gen.get_queryid_queries_from_sink = eq.sink_query_texts
    gen.get_index_definitions_from_sink = eq.sink_index_definitions
    with tempfile.TemporaryDirectory() as tmp:
        reports = eq.run_reports(gen, Path(tmp))
    with gzip.open(a.out, "wt", compresslevel=9) as f:
        for k in sorted(entries):
            f.write(json.dumps(entries[k], sort_keys=True, separators=(",", ":")) + "\n")
    n_inst = sum(1 for e in entries.values() if e["kind"] == "instant")
    print(f"{len(entries)} queries ({n_inst} instant, {len(entries) - n_inst} range), {len(reports)} reports -> {a.out}")


if __name__ == "__main__":
    main()
