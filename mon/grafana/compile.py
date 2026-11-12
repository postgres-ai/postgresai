#!/usr/bin/env python3
"""Dashboard compiler: the ported dashboards (DASHBOARDS: 1-14, all
but 99) on pgai-mon via the Infinity datasource. Standard library only.

  compile.py --probe-url http://127.0.0.1:8765 [--out mon/grafana/dashboards]
  compile.py --check-url http://127.0.0.1:8765 [--out ...]     # run every emitted query

Reads config/grafana/dashboards/Dashboard_<n>_*.json (never written), mon/sql/reports/* and
mon/grafana/cases.py, and writes copies with new uids and titles ending in " (pgai-mon)". The
copies keep the original panels, layout, units and legends. Every PromQL target becomes an
Infinity target that POSTs `/sql?format=objects` with the ported SQL as a text/plain body:
- getvariable() parameters become Grafana expressions (GRAFANA_PARAMS). start and end are
  aligned down to the step (decision 35), and $__rate_interval becomes
  max(interval + scrape, 4 x scrape) with scrape = 5 s, the Prometheus datasource's
  scrapeInterval. Range targets of 7 d or more pin the step to the rollup tier (STEP), and
  there $__rate_interval is the step itself (README.md: rates differ slightly from Grafana's
  Prometheus datasource there).
- The per-target constants (metric, divisor, variant, ...) come from the matching case in
  cases.py, as SQL literals: the same inlining as diff.py --engine http.
- The query is wrapped to return (time epoch ms, value, series[, label columns]); `series` is
  the target's legendFormat rendered in SQL. A per-refId displayName override shows it, and the
  Infinity backend parser's timeseries format turns string columns into series (verify).
--probe-url is needed to compile: each target's SQL runs once with LIMIT 0 to learn whether
its labels come as a `labels` MAP or as plain columns. Things that cannot be checked without
Grafana are marked (verify) here and in README.md.
"""
import argparse
import copy
import json
import os
import re
import sys
import time
import types
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MON = os.path.dirname(HERE)
REPO = os.path.dirname(MON)
DASH_DIR = os.path.join(REPO, "config", "grafana", "dashboards")

INFINITY = {"type": "yesoreyeram-infinity-datasource", "uid": "aerffb0z8rjlsc"}
PGAI_URL = "http://pgai-mon:8765/sql?format=objects"
SCRAPE_MS = 5000  # provisioning/datasources: scrapeInterval '5s'

# Range panels pin the step to the rollup tier the range can use, rounded up to a whole bucket:
# >= 7 d -> a multiple of 1 h, >= 90 d -> a multiple of 1 d (UTC). Grafana's auto interval
# (30 d at ~1,000 px is ~40 min) is not on a bucket boundary, and an unaligned grid is evaluated
# from raw samples (CONSULT_S3.md section 2). Shorter ranges keep $__interval.
PIN_TIERS = ((90 * 86_400_000, 86_400_000), (7 * 86_400_000, 3_600_000))  # (range >=, bucket)
STEP = "CAST(CASE {} ELSE ${{__interval_ms}} END AS BIGINT)".format(" ".join(
    f"WHEN ${{__to}} - ${{__from}} >= {r} THEN ceil(${{__interval_ms}} / {b}.0) * {b}" for r, b in PIN_TIERS))


def pin_step(span_ms, iv):
    """STEP in Python (--check's reference)."""
    for r, b in PIN_TIERS:
        if span_ms >= r:
            return -(-iv // b) * b
    return iv


def grafana_params(step):
    al = lambda x: f"CAST(({x} - {x} % {step}) AS BIGINT)"  # noqa: E731
    return {
        "start_ms": al("${__from}"),
        "end_ms": al("${__to}"),
        "step_ms": f"CAST({step} AS BIGINT)",
        "interval_ms": f"CAST({step} AS BIGINT)",
        # Grafana's $__rate_interval: max($__interval + scrape, 4 x scrape). Where the step is
        # pinned it is the step itself: a window of step + 5 s is not on a bucket boundary, so
        # increase/rate would read raw (VM semantics take the sample before the window, so
        # windows equal to the step tile the range with no gap)
        "rate_interval_ms": f"CAST(greatest(${{__interval_ms}} + {SCRAPE_MS}, {4 * SCRAPE_MS}) AS BIGINT)"
        if step == "${__interval_ms}" else
        f"CAST(CASE WHEN ${{__to}} - ${{__from}} >= {PIN_TIERS[-1][0]} THEN {step} "
        f"ELSE greatest(${{__interval_ms}} + {SCRAPE_MS}, {4 * SCRAPE_MS}) END AS BIGINT)",
        "cluster": "'${cluster_name}'",
        "node": "'${node_name}'",
        "db_re": "'${db_name:regex}'",
        "db": "'${db_name}'",
        "top_n": "CAST(${top_n} AS BIGINT)",
        # $__range: the Prometheus datasource's range, (to - from) rounded to whole seconds
        "range_ms": "CAST(round((${__to} - ${__from}) / 1000.0) * 1000 AS BIGINT)",
        # the dashboards' own filter variables (Dashboard_4, 3, 7)
        "wet_re": "'${wait_event_type:regex}'",
        "we_re": "'${wait_event:regex}'",
        "query_id": "'${query_id}'",
        "schema_re": "'${schema_name:regex}'",
        "table_re": "'${table_name:regex}'",
        "index_re": "'${index_name:regex}'",       # Dashboard_11
        "datname_re": "'${datname:regex}'",
        "lag_alert_s": "CAST(${lag_alert_s} AS DOUBLE)",  # Dashboard_6's custom threshold (seconds)
    }


# instant targets (stat/table panels) keep $__interval: their one point needs no grid, and the
# step is a bare selector's lookback there
GRAFANA_PARAMS = grafana_params("${__interval_ms}")
GRAFANA_PARAMS_RANGE = grafana_params(STEP)
# the filter parameters and the dashboard variable each needs: a dashboard without that variable
# gets the case's literal instead (the parameter then sits in a branch the query does not take)
FILTER_VARS = {"wet_re": "wait_event_type", "we_re": "wait_event", "query_id": "query_id",
               "schema_re": "schema_name", "table_re": "table_name", "index_re": "index_name", "datname_re": "datname",
               "lag_alert_s": "lag_alert_s"}


def dashboard_params(src):
    names = {v.get("name") for v in src.get("templating", {}).get("list", [])}
    keep = lambda ps: {k: v for k, v in ps.items() if k not in FILTER_VARS or FILTER_VARS[k] in names}  # noqa: E731
    gp = keep(GRAFANA_PARAMS)
    # an instant target (table/stat panel): one point at the range end
    return keep(GRAFANA_PARAMS_RANGE), dict(gp, start_ms=gp["end_ms"])

DASHBOARDS = [
    dict(src="Dashboard_1_Node_performance_overview.json", census="D2", uid="pgai-mon-node-overview",
         out="Dashboard_1_Node_performance_overview_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_2_Aggregated_query_analysis.json", census="D1", uid="pgai-mon-query-topn",
         out="Dashboard_2_Aggregated_query_analysis_pgai_mon.json", extra_labels=["queryid", "datname"]),
    dict(src="Dashboard_4_Wait_Sampling_Dashboard.json", census="D4", uid="pgai-mon-wait-sampling",
         out="Dashboard_4_Wait_Sampling_Dashboard_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_3_Single_query_analysis.json", census="D3", uid="pgai-mon-single-query",
         out="Dashboard_3_Single_query_analysis_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_7_Autovacuum_and_xmin_horizon.json", census="D7", uid="pgai-mon-autovacuum-xmin",
         out="Dashboard_7_Autovacuum_and_xmin_horizon_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_13_Lock_waits.json", census="D13", uid="pgai-mon-lock-waits",
         out="Dashboard_13_Lock_waits_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_12_SLRU.json", census="D12", uid="pgai-mon-slru",
         out="Dashboard_12_SLRU_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_14_IO_Statistics.json", census="D14", uid="pgai-mon-io-stats",
         out="Dashboard_14_IO_Statistics_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_5_Backup_stats.json", census="D5", uid="pgai-mon-backup-stats",
         out="Dashboard_5_Backup_stats_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_9_Single_table_analysis.json", census="D9", uid="pgai-mon-single-table",
         out="Dashboard_9_Single_table_analysis_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_11_Single_index_analysis.json", census="D11", uid="pgai-mon-single-index",
         out="Dashboard_11_Single_index_analysis_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_6_Replication_and_HA.json", census="D6", uid="pgai-mon-replication",
         out="Dashboard_6_Replication_and_HA_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_8_Table_Stats.json", census="D8", uid="pgai-mon-table-stats",
         out="Dashboard_8_Table_Stats_pgai_mon.json", extra_labels=[]),
    dict(src="Dashboard_10_Index health.json", census="D10", uid="pgai-mon-index-health",
         out="Dashboard_10_Index_health_pgai_mon.json", extra_labels=[]),
    # a spec may set partial=True: the targets with no case then stay PromQL on the original
    # Prometheus datasource, and the panel becomes "-- Mixed --" (none does now)
]
# censuses whose case ids are "<census>/<panel id><refId>_..." (D2 also has "D2/host/<...>")
PANEL_KEYED = ("D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14")

# label_values(metric{matchers}, label) template variables -> SQL over the `series` view
LABEL_VALUES = re.compile(r"^\s*label_values\(\s*(\w+)\s*(?:\{(.*)\})?\s*,\s*(\w+)\s*\)\s*$")
MATCHER = re.compile(r'(\w+)\s*(=~|!~|!=|=)\s*"((?:[^"\\]|\\.)*)"')


def load_cases():
    """cases.py imports diff.dur (and diff imports duckdb/requests): stub it."""
    stub = types.ModuleType("diff")

    def dur(s):
        return int(s[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[s[-1]]
    stub.dur = dur
    saved = sys.modules.get("diff")
    sys.modules["diff"] = stub
    sys.path.insert(0, HERE)
    try:
        import cases
    finally:
        sys.path.pop(0)
        if saved is None:
            del sys.modules["diff"]
        else:
            sys.modules["diff"] = saved
    return cases.CASES


def sql_literal(v):
    """A Python parameter as a DuckDB literal (diff.py's sql_literal)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}::BIGINT"
    if isinstance(v, float):
        return f"{v!r}::DOUBLE"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, list):
        if not v:
            return "[]::VARCHAR[][]"
        return "[" + ", ".join(sql_literal(x) for x in v) + "]"
    raise TypeError(f"cannot render {v!r}")


def case_sql(case):
    """The case's SQL without whole-line comments: they quote PromQL ($var, $__interval), which
    Grafana would interpolate, and they would ride along in every request body."""
    if "sql" in case:
        sql = case["sql"]
    else:
        with open(os.path.join(MON, case["sql_file"])) as f:
            sql = f.read()
    return "\n".join(x for x in sql.splitlines() if not x.lstrip().startswith("--"))


# cases.py writes some selectors as constant matcher lists with the reference instance's values
# (cluster local, node node-01, db target_database): those become the dashboard's variables.
MATCHER_VARS = {("cluster", "=", "local"): "'${cluster_name}'", ("node_name", "=", "node-01"): "'${node_name}'",
                ("datname", "=~", "target_database"): "'${db_name:regex}'"}


EXPR_MATCHER = re.compile(r"""(\w+)\s*(=~|!~|!=|=)\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""")


def expr_matcher_vars(expr):
    """{(label, op): Grafana expression} for the target's matchers whose value is a dashboard
    variable (`schema="$schema_name"` -> '${schema_name}', `=~` -> '${schema_name:regex}')."""
    out = {}
    for label, op, dq, sq in EXPR_MATCHER.findall(expr or ""):
        v = dq if dq or not sq else sq
        if "$" not in v:
            continue
        g = "'" + gvar(v, op in ("=~", "!~")) + "'"
        if out.get((label, op), g) != g:
            raise SystemExit(f"matcher {label}{op} takes two variables in {expr!r}")
        out[(label, op)] = g
    return out


def matchers_literal(v, mvars=None):
    """A case's matcher list as SQL. A matcher whose (label, op) the target's expression gives a
    variable takes that variable (expr_matcher_vars); else the reference values in MATCHER_VARS
    become the dashboard variables; else the case's literal."""
    out = []
    mvars = mvars or {}
    for m in v:
        if isinstance(m, list) and len(m) == 3 and (m[0], m[1]) in mvars:
            out.append(f"[{sql_literal(m[0])}, {sql_literal(m[1])}, {mvars[(m[0], m[1])]}]")
        elif isinstance(m, list) and len(m) == 3 and tuple(m) in MATCHER_VARS:
            out.append(f"[{sql_literal(m[0])}, {sql_literal(m[1])}, {MATCHER_VARS[tuple(m)]}]")
        else:
            out.append(sql_literal(m))
    return "[" + ", ".join(out) + "]" if out else "[]::VARCHAR[][]"


# the matcher-list parameters (smatchers: the scalar side of irate_times_scalar.sql)
MATCHER_PARAMS = ("matchers", "smatchers")


def inline(sql, params, gparams, mvars=None):
    """getvariable('x') -> Grafana expression (gparams) or the case's literal (params). With
    gparams, matcher lists (MATCHER_PARAMS) get the dashboard variables in place of the reference values."""
    def sub(m):
        k = m.group(1)
        if gparams is not None and k in gparams:
            return "(" + gparams[k] + ")"
        if k not in params:
            raise KeyError(f"no parameter {k!r}")
        if gparams is not None and k in MATCHER_PARAMS:
            return "(" + matchers_literal(params[k], mvars) + ")"
        return "(" + sql_literal(params[k]) + ")"
    return re.sub(r"getvariable\('(\w+)'\)", sub, sql).strip().rstrip(";").strip()


def walk(panels):
    for p in panels:
        yield p
        yield from walk(p.get("panels", []))


def match_case(census, panel, target, by_id, partial=False):
    """The cases.py entry of a PromQL target, or None (non-Prometheus targets)."""
    expr = target.get("expr")
    if not expr:
        return None
    if census in PANEL_KEYED:  # case ids end in <panel id><refId>_...
        key = f"{panel['id']}{target['refId']}_"
        hits = [c for i, c in by_id.items() if i.startswith(census + "/") and i.split("/")[-1].startswith(key)]
    else:  # D1: one shape; metric and per call from the expression
        m = re.search(r"irate\(pgwatch_pg_stat_statements_(\w+)\{", expr)
        per_call = "/irate(pgwatch_pg_stat_statements_calls" in expr.replace(" ", "")
        hits = [by_id.get(f"D1/{m.group(1)}_{'per_call' if per_call else 'per_s'}")] if m else []
        hits = [h for h in hits if h]
    if not hits and partial:
        return None
    if len(hits) != 1:
        raise SystemExit(f"{census} panel {panel['id']} ref {target['refId']}: {len(hits)} matching cases")
    return hits[0]


# --------------------------------------------------------------------------------------------
# The output wrapper: (time, value, series[, labels])
# --------------------------------------------------------------------------------------------
def label_expr(cols, name):
    """A label of the inner query's row: from its `labels` MAP or a plain column."""
    if name in cols:
        return f'CAST(q."{name}" AS VARCHAR)'
    if "labels" in cols:
        return f"q.labels['{name}']"
    return "CAST(NULL AS VARCHAR)"


def auto_legend(cols):
    """legendFormat __auto (or none): the label set as text; one label shows as its value."""
    plain = [c for c in cols if c not in ("t", "value", "labels") and not c.startswith("_")]
    if "labels" in cols:
        return ("CASE WHEN cardinality(q.labels) = 1 THEN map_values(q.labels)[1] ELSE '{' || "
                "array_to_string(list_transform(list_sort(map_keys(q.labels)), "
                "lambda k: k || '=\"' || q.labels[k] || '\"'), ', ') || '}' END")
    if not plain:
        return "''"
    if len(plain) == 1:
        return f'coalesce(CAST(q."{plain[0]}" AS VARCHAR), \'\')'
    parts = " || ', ' || ".join(f"'{c}=\"' || coalesce(CAST(q.\"{c}\" AS VARCHAR), '') || '\"'" for c in plain)
    return "'{' || " + parts + " || '}'"


def legend_expr(fmt, cols):
    if fmt in (None, "", "__auto"):
        return auto_legend(cols)
    parts = []
    for i, piece in enumerate(re.split(r"\{\{\s*(.*?)\s*\}\}", fmt)):
        if i % 2 == 0:
            if piece:
                parts.append("'" + piece.replace("'", "''") + "'")
        elif piece == "$legend_label":
            # D1: the display column the ported SQL has (it has one: _displayname_long, used for
            # both "Smart truncation" and "Raw texts"), else the queryid
            disp = [c for c in cols if c.startswith("_displayname")]
            src = f'q."{disp[0]}"' if disp else "NULL"
            parts.append(f"coalesce({src}, {label_expr(cols, 'queryid')}, '')")
        else:
            parts.append(f"coalesce({label_expr(cols, piece)}, '')")
    return " || ".join(parts) or "''"


def wrap(sql, cols, legend, extra):
    """The Infinity-facing query. NaN/Inf points are dropped (Prometheus never returns NaN on
    a range query except staleness, and the strings "NaN"/"+Inf" would not parse as numbers)."""
    sel = ["CAST(q.t AS BIGINT) AS time", "CAST(q.value AS DOUBLE) AS value", f"{legend_expr(legend, cols)} AS series"]
    sel += [f"{label_expr(cols, x)} AS {x}" for x in extra]
    # the whole label set: series with equal legends stay apart (Prometheus shows both)
    sel.append(f"{auto_legend(cols)} AS label_set")
    return (f"WITH q AS (\n{sql}\n)\nSELECT {', '.join(sel)}\nFROM q\n"
            "WHERE q.value IS NOT NULL AND isfinite(CAST(q.value AS DOUBLE))\nORDER BY time")


def wrap_table(sql, cols, keys):
    """An instant table target (Prometheus format "table"): one row per series with the columns
    Prometheus gives, `Time`, one column per label, `Value`, so the panel's own organize /
    filterFieldsByName transformations (which name those columns) keep working. The label
    columns are the metric's label keys in the probe store at compile time (probe_label_keys)."""
    sel = ['CAST(q.t AS BIGINT) AS "Time"']
    sel += [f'{label_expr(cols, k)} AS "{k}"' for k in keys]
    sel.append('CAST(q.value AS DOUBLE) AS "Value"')
    return (f"WITH q AS (\n{sql}\n)\nSELECT {', '.join(sel)}\nFROM q\n"
            "WHERE q.value IS NOT NULL AND isfinite(CAST(q.value AS DOUBLE))\n"
            f"ORDER BY {auto_legend(cols)}")


def infinity_target(ref, body, hide, extra, table_keys=None):
    if table_keys is not None:
        cols = [{"selector": "Time", "text": "Time", "type": "timestamp_epoch"}]
        cols += [{"selector": k, "text": k, "type": "string"} for k in table_keys]
        cols.append({"selector": "Value", "text": "Value", "type": "number"})
    else:
        cols = [{"selector": "time", "text": "time", "type": "timestamp_epoch"},
                {"selector": "value", "text": "value", "type": "number"},
                {"selector": "series", "text": "series", "type": "string"}]
        cols += [{"selector": x, "text": x, "type": "string"} for x in extra + ["label_set"]]
    t = {
        "refId": ref,
        "datasource": dict(INFINITY),
        "type": "json",
        "source": "url",
        "parser": "backend",
        "format": "table" if table_keys is not None else "timeseries",
        "url": PGAI_URL,
        "url_options": {"method": "POST", "data": body, "body_type": "raw", "body_content_type": "text/plain",
                        "params": [], "headers": []},
        "root_selector": "rows",
        "columns": cols,
        "computed_columns": [],
        "filters": [],
        "global_query_id": "",
    }
    if hide:
        t["hide"] = True
    return t


# --------------------------------------------------------------------------------------------
# Template variables
# --------------------------------------------------------------------------------------------
def gvar(v, regex=False):
    """$var / ${var} inside a PromQL matcher value -> ${var} for Infinity (${var:regex} in a
    regex matcher: a multi-value or All selection becomes (a|b))."""
    return re.sub(r"\$\{?(\w+)\}?", r"${\1:regex}" if regex else r"${\1}", v)


def label_values_sql(query):
    m = LABEL_VALUES.match(query)
    if not m:
        raise SystemExit(f"unsupported variable query {query!r}")
    metric, matchers, label = m.group(1), m.group(2) or "", m.group(3)
    where = [f"metric = '{metric}'", f"labels['{label}'] IS NOT NULL"]
    for k, op, v in MATCHER.findall(matchers):
        lit = "'" + gvar(v, op in ("=~", "!~")).replace("'", "''") + "'"
        lv = f"coalesce(labels['{k}'], '')"
        where.append({"=": f"{lv} = {lit}", "!=": f"{lv} <> {lit}",
                      "=~": f"regexp_full_match({lv}, {lit})", "!~": f"NOT regexp_full_match({lv}, {lit})"}[op])
    # Prometheus scopes label_values to the time range; the series view has no time bounds, so
    # values of series that stopped long ago are listed too (GRAFANA.md decision 4, on branch spike-monitoring-v2-duckdb)
    return f"SELECT DISTINCT labels['{label}'] AS value FROM series WHERE {' AND '.join(where)} ORDER BY 1"


def compile_variable(v):
    if v.get("type") != "query":
        return copy.deepcopy(v)  # custom / textbox: unchanged
    q = v.get("query")
    q = q.get("query") if isinstance(q, dict) else q
    sql = label_values_sql(q)
    out = {k: copy.deepcopy(x) for k, x in v.items() if k not in ("query", "definition", "datasource", "options")}
    out["datasource"] = dict(INFINITY)
    out["definition"] = sql
    out["options"] = []
    out["query"] = {
        "queryType": "infinity",
        "refId": "variable",
        "infinityQuery": {
            "refId": "variable",
            "type": "json",
            "source": "url",
            "parser": "backend",
            "format": "table",
            "url": PGAI_URL,
            "url_options": {"method": "POST", "data": sql, "body_type": "raw", "body_content_type": "text/plain",
                            "params": [], "headers": []},
            "root_selector": "rows",
            "columns": [{"selector": "value", "text": "value", "type": "string"}],
            "computed_columns": [],
            "filters": [],
        },
    }
    return out


# --------------------------------------------------------------------------------------------
# Probe, compile, check
# --------------------------------------------------------------------------------------------
def post(url, sql, fmt=None):
    u = url.rstrip("/") + "/sql" + (f"?format={fmt}" if fmt else "")
    headers = {"Content-Type": "text/plain"}
    # pgai-mon needs credentials on every route but /livez
    if os.environ.get("PGAI_MON_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["PGAI_MON_TOKEN"]
    req = urllib.request.Request(u, data=sql.encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


def probe_columns(url, case):
    """Output columns of the case's SQL (LIMIT 0, with the case's own sample parameters)."""
    p = dict(case.get("params") or {})
    p.update({"start_ms": 1_000_000_000_000, "end_ms": 1_000_000_600_000, "step_ms": 60_000,
              "interval_ms": 60_000, "rate_interval_ms": 65_000, "range_ms": 600_000,
              "cluster": "x", "node": "x", "db": "x"})
    p.setdefault("db_re", "x")
    p.setdefault("top_n", 10)
    j = post(url, f"SELECT * FROM (\n{inline(case_sql(case), p, None)}\n) LIMIT 0")
    if "error" in j:
        raise SystemExit(f"probe {case['id']}: {j['error']}")
    return j["columns"]


def probe_label_keys(url, metric):
    """The label keys of a metric's series in the probe store (sorted, as Prometheus' table)."""
    j = post(url, f"SELECT DISTINCT unnest(map_keys(labels)) AS k FROM series WHERE metric = '{metric}' ORDER BY 1")
    if "error" in j:
        raise SystemExit(f"probe keys {metric}: {j['error']}")
    return [r[0] for r in j["rows"] if r[0] != "__name__"]


def table_label_keys(panel, probe_url, metric):
    """The label columns of an instant table: the metric's label keys in the probe store, plus
    every field the panel's transformations name (so they exist even when the probe store has
    no such series); Time/Value are the fixed columns."""
    keys = set(probe_label_keys(probe_url, metric)) if metric else set()
    for tr in panel.get("transformations", []):
        o = tr.get("options", {})
        for k in ("indexByName", "renameByName", "excludeByName"):
            keys |= set(o.get(k) or {})
        keys |= set((o.get("include") or {}).get("names") or [])
    return sorted(keys - {"Time", "Value", "__name__"})


def compile_dashboard(spec, by_id, probe_url):
    src = json.load(open(os.path.join(DASH_DIR, spec["src"])))
    d = copy.deepcopy(src)
    d["uid"] = spec["uid"]
    d["title"] = src["title"] + " (pgai-mon)"
    d["id"] = None
    d["version"] = 1
    n = 0
    queries = []
    d_range, d_inst = dashboard_params(src)
    for p in walk(d["panels"]):
        targets = p.get("targets") or []
        if not any(t.get("expr") for t in targets):
            continue
        new = []
        for t in targets:
            case = match_case(spec["census"], p, t, by_id, spec.get("partial", False))
            if case is None:
                new.append(t)  # e.g. Dashboard_2's Flask table (Infinity already): unchanged
                continue
            cols = probe_columns(probe_url, case)
            params = case.get("params") or {}
            instant = bool(t.get("instant") and not t.get("range"))
            gp = d_inst if instant else d_range
            mvars = expr_matcher_vars(t["expr"])
            table = instant and t.get("format") == "table" and p.get("type") == "table"
            keys = table_label_keys(p, probe_url, params.get("metric")) if table else None
            inner = inline(case_sql(case), params, gp, mvars)
            body = wrap_table(inner, cols, keys) if table else \
                wrap(inner, cols, t.get("legendFormat"), spec["extra_labels"])
            new.append(infinity_target(t["refId"], body, t.get("hide"), spec["extra_labels"], keys))
            queries.append({"panel": p["id"], "ref": t["refId"], "case": case["id"], "sql": body,
                            "instant": instant, "table": table, "qvars": query_sample_vars(params, gp, mvars)})
            n += 1
            # the legend: `series`, shown through a per-refId display name (verify); a table's
            # columns are named by their labels and keep their names
            if not table:
                fc = p.setdefault("fieldConfig", {"defaults": {}, "overrides": []})
                fc.setdefault("overrides", []).insert(0, {
                    "matcher": {"id": "byFrameRefID", "options": t["refId"]},
                    "properties": [{"id": "displayName", "value": "${__field.labels.series}"}]})
            # Prometheus targets' own min interval ("interval": "20" = 20 s) -> the panel's (verify)
            iv = str(t.get("interval") or "").strip()
            if iv and not p.get("interval"):
                p["interval"] = iv + "s" if iv.isdigit() else iv
        p["targets"] = new
        p["datasource"] = dict(INFINITY) if all("infinity" in json.dumps(x.get("datasource")) for x in new) \
            else {"type": "datasource", "uid": "-- Mixed --"}
    d["templating"]["list"] = [compile_variable(v) for v in src["templating"]["list"]]
    return d, n, queries


# --check: the instance the variables select (--instance); defaults are the reference store's
SAMPLE = {"cluster": "local", "node": "node-01", "db": "target_database", "db2": "postgres"}
STANDARD_TOKENS = {"${cluster_name}", "${node_name}", "${db_name:regex}", "${db_name}"}


def query_sample_vars(params, gp, mvars):
    """--check: the value each of the query's other variables takes, so that the emitted query
    and the case SQL select the same series: the case's own literal (a matcher's value, or a
    filter parameter's), or SAMPLE's for cluster / node / datname."""
    out = {}
    std = {"cluster": "cluster", "node_name": "node", "datname": "db"}
    for m in [m for k in MATCHER_PARAMS for m in params.get(k) or []]:
        if isinstance(m, list) and len(m) == 3 and (m[0], m[1]) in mvars:
            tok = mvars[(m[0], m[1])].strip("'")
            if tok not in STANDARD_TOKENS:
                out[tok] = SAMPLE[std[m[0]]] if m[0] in std else m[2]
    for k in FILTER_VARS:
        if k in gp and k in params:
            out[re.search(r"\$\{[^}]+\}", gp[k]).group(0)] = str(params[k])
    return out


def sample_vars(to_ms, span_ms, interval_ms, db_regex):
    return {"${__from}": str(to_ms - span_ms), "${__to}": str(to_ms), "${__interval_ms}": str(interval_ms),
            "${cluster_name}": SAMPLE["cluster"], "${node_name}": SAMPLE["node"], "${db_name:regex}": db_regex,
            "${db_name}": SAMPLE["db"], "${top_n}": "10", "${legend_label}": "displayname_long"}


def sample_matchers(v):
    """A case's constant matchers with the reference values replaced by SAMPLE's."""
    ref = {("cluster", "=", "local"): SAMPLE["cluster"], ("node_name", "=", "node-01"): SAMPLE["node"],
           ("datname", "=~", "target_database"): SAMPLE["db"], ("datname", "=", "target_database"): SAMPLE["db"]}
    return [[m[0], m[1], ref[tuple(m)]] if isinstance(m, list) and tuple(m) in ref else m for m in v]


def reference(case, to_ms, span_ms, iv, instant):
    """The case's SQL inlined the way diff.py --engine http does it, with the values Grafana would
    send (sample_vars, one db): the emitted query must give the same points."""
    if not instant:
        iv = pin_step(span_ms, iv)
    end = to_ms - to_ms % iv
    p = dict(case.get("params") or {})
    p.update({"start_ms": end if instant else (to_ms - span_ms) - (to_ms - span_ms) % iv, "end_ms": end,
              "step_ms": iv, "interval_ms": iv,
              "rate_interval_ms": iv if not instant and span_ms >= PIN_TIERS[-1][0] else max(iv + SCRAPE_MS, 4 * SCRAPE_MS),
              "cluster": SAMPLE["cluster"], "node": SAMPLE["node"], "db": SAMPLE["db"], "db_re": SAMPLE["db"],
              "top_n": 10, "range_ms": round(span_ms / 1000) * 1000})
    for k in MATCHER_PARAMS:
        if k in p:
            p[k] = sample_matchers(p[k])
    return (f"SELECT count(*) AS n, sum(CAST(value AS DOUBLE)) AS s FROM (\n{inline(case_sql(case), p, None)}\n) "
            "WHERE value IS NOT NULL AND isfinite(CAST(value AS DOUBLE))")


def interpolate(sql, vs):
    for k, v in vs.items():
        sql = sql.replace(k, v)
    left = re.findall(r"\$\{[^}]+\}", sql)
    if left:
        raise SystemExit(f"uninterpolated {left}")
    return sql


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe-url", required=True, help="pgai-mon to learn each SQL's output columns from")
    ap.add_argument("--out", default=os.path.join(HERE, "dashboards"))
    ap.add_argument("--target-url", default="http://pgai-mon:8765/sql?format=objects",
                    help="pgai-mon URL Grafana's Infinity calls (default: %(default)s)")
    ap.add_argument("--check", action="store_true",
                    help="also run every emitted query (and variable) through /sql?format=objects of --probe-url")
    ap.add_argument("--to-ms", type=int, help="--check: range end (default: newest `up` sample, else now)")
    ap.add_argument("--spans", default="6h:30s,24h:1m,7d:10m",
                    help="--check: range:interval pairs, the interval being Grafana's $__interval before "
                         "pinning (default: %(default)s; 7d:10m is pinned to a 1 h step)")
    ap.add_argument("--dashboards", help="--check: only these censuses, e.g. D3,D8 (default: all)")
    ap.add_argument("--instance", default="local,node-01,target_database,postgres",
                    help="--check: cluster,node,db,second db the variables select (default: %(default)s)")
    args = ap.parse_args()
    SAMPLE.update(zip(("cluster", "node", "db", "db2"), args.instance.split(",")))
    globals()["PGAI_URL"] = args.target_url
    by_id = {c["id"]: c for c in load_cases()}
    os.makedirs(args.out, exist_ok=True)
    report = []
    for spec in DASHBOARDS:
        d, n, queries = compile_dashboard(spec, by_id, args.probe_url)
        path = os.path.join(args.out, spec["out"])
        with open(path, "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"{path}: {n} targets, {len(d['templating']['list'])} variables")
        report.append((spec, d, queries))
    if not args.check:
        return
    # default end: the newest `up` sample (other metrics may run later than the dashboards' data)
    to = args.to_ms or post(args.probe_url, "SELECT CAST(epoch_ms(max(s.ts)) AS BIGINT) FROM samples s "
                            "JOIN series USING (series_id) WHERE series.metric = 'up'")["rows"][0][0] \
        or int(time.time() * 1000)
    print(f"range end {to}")
    unit = {"s": 1000, "m": 60_000, "h": 3600_000, "d": 86_400_000}
    ms = lambda x: int(x[:-1]) * unit[x[-1]]  # noqa: E731
    spans = [(ms(a), ms(b)) for a, b in (x.split(":") for x in args.spans.split(","))]
    bad = 0
    for spec, d, queries in report:
        if args.dashboards and spec["census"] not in args.dashboards.split(","):
            continue
        for span, iv in spans:
            for dbre in (SAMPLE["db"], f"({SAMPLE['db2']}|{SAMPLE['db']})"):
                vs = sample_vars(to, span, iv, dbre)
                for q in queries:
                    j = post(args.probe_url, interpolate(q["sql"], {**q["qvars"], **vs}), "objects")
                    vcol = "Value" if q["table"] else "value"
                    ok = "error" not in j and j.get("format") == "objects" and \
                        (j["columns"][0] == "Time" and j["columns"][-1] == "Value" if q["table"]
                         else j["columns"][:3] == ["time", "value", "series"])
                    parity = ""
                    if ok and dbre == SAMPLE["db"]:
                        rj = post(args.probe_url, reference(by_id[q["case"]], to, span, iv, q["instant"]))
                        if "rows" not in rj:  # the reference itself failed (e.g. the reader's memory cap)
                            j = {"error": f"reference: {rj.get('error')}"}
                            ok = False
                            bad += 1
                            print(f"BAD {spec['census']} {span // 3600_000}h db={dbre} panel {q['panel']}{q['ref']} "
                                  f"({q['case']}): {j['error']}")
                            continue
                        n, sm = rj["rows"][0]
                        got = sum(r[vcol] for r in j["rows"])
                        same = n == j["row_count"] and (sm or 0) == 0 == got or \
                            (sm is not None and abs(got - sm) <= 1e-9 * max(abs(sm), 1))
                        ok = same
                        parity = f" parity {'ok' if same else f'DIFF ref {n} rows sum {sm} got sum {got}'}"
                    if not ok:
                        bad += 1
                    print(f"{'ok ' if ok else 'BAD'} {spec['census']} {span // 3600_000}h db={dbre} panel {q['panel']}{q['ref']} "
                          f"({q['case']}): {j.get('row_count', j.get('error'))} rows, "
                          f"{len({r.get('series', i) for i, r in enumerate(j.get('rows', []))})} series, "
                          f"{j.get('elapsed_ms')} ms{parity}")
        for v in d["templating"]["list"]:
            if v.get("type") != "query":
                continue
            dv = {}
            for q in queries:  # the variables the panels use, with the values their cases have
                for k, x in q["qvars"].items():
                    dv.setdefault(k, x)
            sql = interpolate(v["definition"], {**dv, **sample_vars(to, 3600_000, 30_000, SAMPLE["db"])})
            j = post(args.probe_url, sql, "objects")
            ok = "error" not in j
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {spec['census']} variable {v['name']}: "
                  f"{[r['value'] for r in j.get('rows', [])] if ok else j['error']}")
    print(f"{bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
