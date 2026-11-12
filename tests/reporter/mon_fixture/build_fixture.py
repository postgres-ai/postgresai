#!/usr/bin/env python3
"""Build the reporter's pgai-mon test fixture from a larger pgai-mon store (dev tool, not run in CI).

Reads a running `pgai-mon serve` over a source store (real pgwatch data) and writes, next to this
file:
  samples.jsonl.gz     VictoriaMetrics export JSON lines (`pgai-mon load` input; null = staleness marker)
  query_texts.jsonl    query_texts rows for the kept queryids
  objects.jsonl        objects rows for the kept tables and their indexes

The slice: one cluster/node, every metric except pg_stat_statements for a fixed set of tables (so
unused, redundant and invalid index rows, bloat and table stats stay consistent), pg_stat_statements
for the busiest queryids plus a few quiet ones (only queryids with a stored text), DAYS days ending at the source's last sample. The
last NATIVE_HOURS keep one sample per NATIVE_S; earlier samples are thinned to one per THIN_S per series (both
engines get the same samples, so thinning changes no comparison). Real NaN values (not staleness
markers) are dropped: VictoriaMetrics cannot store them, so the old stack never saw them.

Usage: build_fixture.py --url http://127.0.0.1:PORT [--days 8]
"""
import argparse
import gzip
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

TABLES = {
    # redundant / unused / rarely used indexes
    "t10", "t30", "t60", "t100", "t110", "t112", "t117", "t118", "t280",
    # bloat, vacuum progress
    "t15", "t47", "t137", "t252", "t79",
    # pgbench
    "pgbench_accounts", "pgbench_branches", "pgbench_tellers", "pgbench_history",
    # catalog tables (system-schema filters) and pgwatch's aggregate rows
    "pg_class", "pg_attribute", "pg_statistic", "$other$", "$total$",
}
TOP_QUERYIDS = 20
QUIET_QUERYIDS = 5
NATIVE_HOURS = 26
NATIVE_S = 60      # at most one sample per minute in the native window
THIN_S = 1800      # earlier: one per 30 min
SETTINGS_S = 900   # settings (constant for days): one per 15 min throughout


def sql(url, text):
    req = urllib.request.Request(url.rstrip("/") + "/sql", data=text.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(d["error"])
    return [dict(zip(d["columns"], row)) for row in d["rows"]]


def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--cluster", default="local")
    ap.add_argument("--node", default="node-01")
    a = ap.parse_args()

    end_ms = sql(a.url, "select epoch_ms(max(ts)) e from samples")[0]["e"]
    start_ms = end_ms - a.days * 86400000
    native_ms = end_ms - NATIVE_HOURS * 3600000
    tables = "[" + ",".join(q(t) for t in sorted(TABLES)) + "]"
    table_key = ("regexp_replace(coalesce(labels['relname'], labels['tblname'], labels['table_name'], "
                 "labels['table_full_name']), '^.*\\.', '')")
    node = f"labels['cluster'] = {q(a.cluster)} and labels['node_name'] = {q(a.node)}"

    qids = sql(a.url, f"""
        with inc as (
          select labels['queryid'] qid, max(value) - min(value) d
          from metrics
          where metric = 'pgwatch_pg_stat_statements_exec_time_total' and {node}
            and ts > make_timestamptz({end_ms - 86400000}::BIGINT * 1000) and not isnan(value)
            and labels['queryid'] in (select CAST(queryid AS VARCHAR) from query_texts)
          group by 1)
        (select qid from inc order by d desc, qid limit {TOP_QUERYIDS})
        union
        (select qid from inc where d = 0 order by hash(qid) limit {QUIET_QUERYIDS})""")
    qid_list = "[" + ",".join(q(r["qid"]) for r in qids) + "]"

    keep = f"""
        select series_id, metric, labels from series
        where {node}
          and (metric not like 'pgwatch_pg_stat_statements%' or list_contains({qid_list}, labels['queryid']))
          and ({table_key} is null or list_contains({tables}, {table_key}))"""
    n = sql(a.url, f"select count(*) n from ({keep})")[0]["n"]
    print(f"series: {n}, queryids: {len(qids)}, window {start_ms}..{end_ms}")

    out = gzip.open(os.path.join(HERE, "samples.jsonl.gz"), "wt", compresslevel=9)
    total = 0
    series = sql(a.url, f"select series_id, metric, CAST(labels AS JSON) labels from ({keep}) order by metric, CAST(labels AS VARCHAR)")
    ids = [r["series_id"] for r in series]
    meta = {r["series_id"]: r for r in series}
    for i in range(0, len(ids), 400):
        chunk = ",".join(str(x) for x in ids[i:i + 400])
        settings_ids = ",".join(str(x) for x in ids[i:i + 400] if meta[x]["metric"].startswith("pgwatch_settings_"))
        rows = sql(a.url, f"""
            select series_id, list(epoch_ms(ts) order by ts) t, list(CASE WHEN stale THEN NULL ELSE value END order by ts) v
            from (
              select series_id, ts, value, stale,
                     row_number() over (partition by series_id, epoch_ms(ts) // {THIN_S * 1000} order by ts) rn,
                     row_number() over (partition by series_id, epoch_ms(ts) // {NATIVE_S * 1000} order by ts) rn_native,
                     row_number() over (partition by series_id, epoch_ms(ts) // {SETTINGS_S * 1000} order by ts) rn_settings
              from samples
              where series_id in ({chunk})
                and ts >= make_timestamptz({start_ms}::BIGINT * 1000) and ts <= make_timestamptz({end_ms}::BIGINT * 1000)
                and (stale or not isnan(value))
            )
            where stale
               or (series_id in ({settings_ids or 0}) and rn_settings = 1)
               or (series_id not in ({settings_ids or 0}) and (rn = 1 or (epoch_ms(ts) >= {native_ms} and rn_native = 1)))
            group by series_id""")
        for r in sorted(rows, key=lambda r: ids.index(r["series_id"])):
            m = meta[r["series_id"]]
            labels = m["labels"] if isinstance(m["labels"], dict) else json.loads(m["labels"])
            out.write(json.dumps({"metric": {"__name__": m["metric"], **labels},
                                  "values": r["v"], "timestamps": r["t"]}, separators=(",", ":")) + "\n")
            total += len(r["t"])
    out.close()
    print(f"samples: {total}")

    texts = sql(a.url, f"""
        select CAST(queryid AS VARCHAR) queryid, dbname, query, epoch_ms(first_seen) first_seen_ms, epoch_ms(last_seen) last_seen_ms
        from query_texts where list_contains({qid_list}, CAST(queryid AS VARCHAR)) order by 1, 2""")
    with open(os.path.join(HERE, "query_texts.jsonl"), "w") as f:
        for r in texts:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    objs = sql(a.url, f"""
        select kind, schema, name, definition, epoch_ms(valid_from) ts_ms, dbname from objects
        where list_contains({tables}, name)
           or (kind = 'index' and list_contains({tables}, regexp_extract(definition, ' ON (ONLY )?([^ ]+\\.)?([^ .]+) USING', 3)))
           or (kind = 'constraint' and list_contains({tables}, split_part(name, '.', 1)))
        order by dbname, kind, schema, name""")
    with open(os.path.join(HERE, "objects.jsonl"), "w") as f:
        for r in objs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"query_texts: {len(texts)}, objects: {len(objs)}")


if __name__ == "__main__":
    main()
