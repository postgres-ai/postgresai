#!/usr/bin/env python3
"""Run every panel of the SQL dashboards against a pgai-mon, the way Grafana would, and classify
each panel: data, empty or error. Standard library only.

  check_panels.py --url http://127.0.0.1:8765 [--dashboards DIR] [--now-ms MS] [--range 6h]
                  [--var query_id=123 ...] [--json OUT] [--expect FILE]

Grafana's part is reproduced here: the dashboard's default time range ending at --now-ms
(default: the newest sample in the store), $__interval from the range and the panel's
maxDataPoints / min interval (kbn.roundInterval), and the template variables in order, each
query variable run through /sql and set to its current value when that is among its options,
else to its first option (All when the current value is All and the variable has includeAll).
Then each target's SQL body is interpolated (${var}, $var, [[var]], ${var:regex|sqlstring|csv|
pipe|raw|queryparam|date|...}, ${__from}, ${__to}, ${__interval_ms}, ${__range}, ...) and POSTed
to /sql using the target's response format. An Infinity filterExpression (`a < 1 && b > 0`) is applied to the rows.

A panel is `error` when any target fails, `data` when a target returns a row, `empty` when the
queries ran and returned nothing, `external` when its targets do not call pgai-mon, and `none`
when it has no targets (rows, text panels).

--expect FILE (JSON {"<dashboard uid>/<panel id>": "data" | "empty"}) makes the exit status 1 when
a listed panel is classified differently; without it the exit status is 1 on any error.
"""
import csv
import io
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAX_POINTS = 1000  # a panel with no maxDataPoints: about a full-width panel's pixels


# --------------------------------------------------------------------------------------------
# Grafana time and interval
# --------------------------------------------------------------------------------------------
UNIT_MS = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000,
           "M": 2_592_000_000, "y": 31_536_000_000}


def parse_duration_ms(s):
    m = re.fullmatch(r"\s*(\d+)\s*(ms|s|m|h|d|w|M|y)\s*", s or "")
    if not m:
        if s and s.strip().isdigit():
            return int(s.strip()) * 1000
        raise ValueError(f"bad duration {s!r}")
    return int(m.group(1)) * UNIT_MS[m.group(2)]


def parse_relative(expr, now_ms):
    """now, now-6h, now-7d/d (rounding is ignored), or epoch ms."""
    expr = str(expr).strip()
    if expr.isdigit():
        return int(expr)
    m = re.fullmatch(r"now(?:-(\d+)(ms|s|m|h|d|w|M|y))?(?:/\w)?", expr)
    if not m:
        raise ValueError(f"unsupported time {expr!r}")
    return now_ms - (int(m.group(1)) * UNIT_MS[m.group(2)] if m.group(1) else 0)


# kbn.roundInterval: (upper bound exclusive, interval), ms
ROUND = [(15, 10), (35, 20), (75, 50), (150, 100), (350, 200), (750, 500), (1500, 1000), (3500, 2000),
         (7500, 5000), (12500, 10000), (17500, 15000), (25000, 20000), (45000, 30000), (90000, 60000),
         (210000, 120000), (450000, 300000), (750000, 600000), (1050000, 900000), (1500000, 1200000),
         (2700000, 1800000), (5400000, 3600000), (9000000, 7200000), (16200000, 10800000),
         (32400000, 21600000), (86400000, 43200000), (604800000, 86400000), (1814400000, 604800000),
         (3628800000, 2592000000)]


def round_interval(ms):
    for bound, iv in ROUND:
        if ms < bound:
            return iv
    return 31536000000


def interval_ms(range_ms, panel):
    """Grafana's $__interval for a panel: range / maxDataPoints, rounded, at least the panel's
    min interval."""
    pts = panel.get("maxDataPoints") or DEFAULT_MAX_POINTS
    iv = round_interval(max(range_ms // pts, 1))
    if panel.get("interval"):
        iv = max(iv, parse_duration_ms(str(panel["interval"])))
    return iv


def interval_text(ms):
    for u, f in (("d", 86_400_000), ("h", 3_600_000), ("m", 60_000), ("s", 1000)):
        if ms % f == 0:
            return f"{ms // f}{u}"
    return f"{ms}ms"


def iso(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{ms % 1000:03d}Z"


# --------------------------------------------------------------------------------------------
# Variable formats (Grafana's templateSrv.formatValue)
# --------------------------------------------------------------------------------------------
REGEX_SPECIAL = re.compile(r"([\-\[\]/{}()*+?.\\^$|])")


def regex_escape(v):
    return REGEX_SPECIAL.sub(r"\\\1", v)


class CustomAll(str):
    """A variable's custom allValue: Grafana interpolates it unformatted (except :text, :queryparam)."""


def fmt_value(values, fmt, name):
    """values: list of strings (one for a single value)."""
    if len(values) == 1 and isinstance(values[0], CustomAll) and fmt not in ("text", "queryparam"):
        return str(values[0])
    single = len(values) == 1
    if fmt in (None, "", "glob"):
        return values[0] if single else "{" + ",".join(values) + "}"
    if fmt == "regex":
        e = [regex_escape(v) for v in values]
        return e[0] if single else "(" + "|".join(e) + ")"
    if fmt == "raw":
        return ",".join(values)
    if fmt in ("csv", "text"):
        return ",".join(values)
    if fmt == "pipe":
        return "|".join(values)
    if fmt == "sqlstring":
        return ",".join("'" + v.replace("'", "''") + "'" for v in values)
    if fmt == "singlequote":
        return ",".join("'" + v.replace("'", "\\'") + "'" for v in values)
    if fmt == "doublequote":
        return ",".join('"' + v.replace('"', '\\"') + '"' for v in values)
    if fmt == "json":
        return json.dumps(values[0] if single else values)
    if fmt == "queryparam":
        return "&".join(f"var-{name}=" + urllib.parse.quote(v, safe="") for v in values)
    if fmt == "percentencode":
        return urllib.parse.quote(values[0] if single else "{" + ",".join(values) + "}", safe="")
    raise ValueError(f"unsupported format {fmt!r} of ${name}")


VAR_RE = re.compile(r"\$\{(\w+)(?:\.(\w+))?(?::([\w:]+))?\}|\[\[(\w+)(?::(\w+))?\]\]|\$(\w+)")


def interpolate(text, vars_, builtins):
    """vars_: name -> list of values; builtins: name -> callable(fmt) -> str."""
    unresolved = []

    def sub(m):
        name = m.group(1) or m.group(4) or m.group(6)
        fmt = m.group(3) or m.group(5)
        if m.group(2):  # ${__field.labels.x}, ${__value.raw}: not a query variable
            return m.group(0)
        if name in builtins:
            return builtins[name](fmt)
        if name in vars_:
            return fmt_value(vars_[name], fmt, name)
        if not m.group(6):  # a bare $word Grafana does not know stays as it is (e.g. '$other$')
            unresolved.append(m.group(0))
        return m.group(0)
    out = VAR_RE.sub(sub, text)
    return out, unresolved


def builtin_vars(from_ms, to_ms, iv_ms):
    rng = to_ms - from_ms

    def tm(ms):
        def f(fmt):
            if fmt and fmt.startswith("date"):
                sub = fmt[5:] if fmt.startswith("date:") else ""
                if sub == "seconds":
                    return str(ms // 1000)
                return iso(ms)
            return str(ms)
        return f
    return {
        "__from": tm(from_ms), "__to": tm(to_ms),
        "__interval_ms": lambda f: str(iv_ms), "__interval": lambda f: interval_text(iv_ms),
        "__rate_interval": lambda f: interval_text(max(iv_ms + 5000, 20000)),
        "__range": lambda f: f"{round(rng / 1000)}s", "__range_s": lambda f: str(round(rng / 1000)),
        "__range_ms": lambda f: str(rng),
    }


# --------------------------------------------------------------------------------------------
# pgai-mon
# --------------------------------------------------------------------------------------------
def post(url, sql, timeout=300, fmt="objects", columns=()):
    req = urllib.request.Request(url.rstrip("/") + "/sql?format=" + fmt, data=sql.encode(),
                                 headers={"Content-Type": "text/plain"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if fmt == "csv":
                reader = csv.DictReader(io.StringIO(r.read().decode()))
                rows = list(reader)
                for row in rows:
                    for col in columns:
                        key = col["selector"]
                        if key in row and col["type"] in ("number", "timestamp_epoch"):
                            row[key] = (int(row[key]) if col["type"] == "timestamp_epoch" else float(row[key])) if row[key] else None
                j = {"columns": reader.fieldnames, "rows": rows, "row_count": len(rows)}
            else:
                j = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            j = json.load(e)
        except ValueError:
            j = {"error": {"message": f"HTTP {e.code}"}}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        j = {"error": {"message": f"transport: {e}"}}
    j.setdefault("elapsed_ms", round((time.time() - t0) * 1000))
    return j


def error_text(j):
    e = j.get("error")
    if isinstance(e, dict):
        return f"{e.get('type', 'error')}: {e.get('message', '')}"
    return str(e)


# --------------------------------------------------------------------------------------------
# Infinity filterExpression (the subset the dashboards use: comparisons joined by && / ||)
# --------------------------------------------------------------------------------------------
FILTER_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|(\w+)|(&&|\|\||==|!=|<=|>=|<|>|\(|\)))")


def row_filter(expr):
    if not expr:
        return None
    toks, pos = [], 0
    while pos < len(expr):
        m = FILTER_TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise ValueError(f"unsupported filterExpression {expr!r}")
        pos = m.end()
        num, ident, op = m.groups()
        if num:
            toks.append(num)
        elif ident:
            toks.append(f"_v(r, {ident!r})")
        else:
            toks.append({"&&": " and ", "||": " or "}.get(op, op))
    code = compile("".join(toks), "<filterExpression>", "eval")

    def _v(r, k):
        v = r.get(k)
        return float("nan") if v is None else v
    return lambda r: bool(eval(code, {"__builtins__": {}}, {"_v": _v, "r": r}))  # noqa: S307


# --------------------------------------------------------------------------------------------
# Dashboards
# --------------------------------------------------------------------------------------------
def walk(panels):
    for p in panels:
        yield p
        yield from walk(p.get("panels", []))


def is_pgai_target(t):
    return "infinity" in json.dumps(t.get("datasource") or {}) and "/sql" in (t.get("url") or "")


def custom_options(v):
    q = v.get("query")
    q = q.get("query") if isinstance(q, dict) else q
    out = []
    for part in (q or "").split(","):
        part = part.strip()
        if not part:
            continue
        if " : " in part:  # "Text : value"
            part = part.split(" : ", 1)[1].strip()
        out.append(part)
    return out or [o.get("value") for o in v.get("options", []) if o.get("value") is not None]


def current_values(v):
    c = (v.get("current") or {}).get("value")
    if c is None:
        return []
    return c if isinstance(c, list) else [c]


def resolve_variables(d, url, from_ms, to_ms, overrides):
    """name -> list of values, plus a report per query variable."""
    vars_, report = {}, []
    iv = interval_ms(to_ms - from_ms, {})
    for v in d.get("templating", {}).get("list", []):
        name, typ = v["name"], v.get("type")
        if name in overrides:
            vars_[name] = overrides[name]
            continue
        if typ == "query":
            sql, left = interpolate(v.get("definition") or "", vars_, builtin_vars(from_ms, to_ms, iv))
            if left:
                report.append({"variable": name, "class": "error", "error": f"uninterpolated {left}"})
                vars_[name] = [""]
                continue
            j = post(url, sql)
            if "error" in j:
                report.append({"variable": name, "class": "error", "error": error_text(j)})
                vars_[name] = [""]
                continue
            opts = [str(r["value"]) for r in j["rows"] if r.get("value") is not None]
            report.append({"variable": name, "class": "data" if opts else "empty", "options": len(opts)})
        elif typ == "custom":
            opts = custom_options(v)
        elif typ in ("textbox", "constant"):
            q = v.get("query")
            vars_[name] = current_values(v) or [q if isinstance(q, str) else ""]
            continue
        else:
            report.append({"variable": name, "class": "error", "error": f"unsupported variable type {typ}"})
            vars_[name] = [""]
            continue
        cur = current_values(v)
        if v.get("includeAll") and (cur == ["$__all"] or cur == ["All"]):
            vals = [CustomAll(v["allValue"])] if v.get("allValue") else opts
        else:
            vals = [x for x in cur if x in opts]
            if not vals:
                vals = opts[:1]
        vars_[name] = vals or [""]
    return vars_, report


SHOW_SQL = []


def run_target(url, t, vars_, builtins):
    body, left = interpolate((t.get("url_options") or {}).get("data") or "", vars_, builtins)
    if SHOW_SQL:
        print(f"-- refId {t.get('refId')}\n{body}\n")
    if left:
        return {"class": "error", "error": f"uninterpolated {sorted(set(left))}", "rows": 0, "ms": 0}
    j = post(url, body, fmt="csv" if t.get("type") == "csv" else "objects", columns=t.get("columns", []))
    if "error" in j:
        return {"class": "error", "error": error_text(j), "rows": 0, "ms": j.get("elapsed_ms")}
    rows = j.get("rows") or []
    flt = row_filter(t.get("filterExpression"))
    if flt:
        rows = [r for r in rows if flt(r)]
    sel = [c.get("selector") for c in t.get("columns") or []]
    missing = [c for c in sel if c and c not in (j.get("columns") or [])]
    if missing:
        return {"class": "error", "error": f"columns {missing} missing from the result", "rows": len(rows),
                "ms": j.get("elapsed_ms")}
    return {"class": "data" if rows else "empty", "rows": len(rows), "ms": j.get("elapsed_ms")}


def check_dashboard(path, url, now_ms, range_ms, overrides, only=None):
    d = json.load(open(path))
    t = d.get("time") or {"from": "now-6h", "to": "now"}
    to_ms = parse_relative(t.get("to", "now"), now_ms)
    from_ms = to_ms - range_ms if range_ms else parse_relative(t.get("from", "now-6h"), now_ms)
    vars_, vreport = resolve_variables(d, url, from_ms, to_ms, overrides)
    panels = []
    for p in walk(d.get("panels", [])):
        if p.get("type") == "row" or (only is not None and p.get("id") != only):
            continue
        targets = [x for x in p.get("targets") or [] if not x.get("hide")]
        rec = {"dashboard": d.get("uid"), "panel": p.get("id"), "title": p.get("title") or "", "type": p.get("type"),
               "targets": []}
        if not targets:
            rec["class"] = "none"
            panels.append(rec)
            continue
        builtins = builtin_vars(from_ms, to_ms, interval_ms(to_ms - from_ms, p))
        for x in targets:
            if not is_pgai_target(x):
                rec["targets"].append({"ref": x.get("refId"), "class": "external", "rows": 0, "ms": 0,
                                       "error": x.get("url") or json.dumps(x.get("datasource"))})
                continue
            r = run_target(url, x, vars_, builtins)
            r["ref"] = x.get("refId")
            rec["targets"].append(r)
        cls = [x["class"] for x in rec["targets"]]
        rec["class"] = "error" if "error" in cls else "data" if "data" in cls else \
            "empty" if "empty" in cls else "external"
        rec["rows"] = sum(x["rows"] for x in rec["targets"])
        rec["ms"] = sum(x.get("ms") or 0 for x in rec["targets"])
        panels.append(rec)
    return d, {"from_ms": from_ms, "to_ms": to_ms, "variables": {k: v for k, v in vars_.items()}}, vreport, panels


def newest_sample_ms(url):
    j = post(url, "SELECT CAST(epoch_ms(max(s.ts)) AS BIGINT) AS t FROM samples s JOIN series USING (series_id) "
                  "WHERE series.metric = 'pgwatch_db_size_size_b'")
    if "error" in j or not j["rows"] or j["rows"][0]["t"] is None:
        return int(time.time() * 1000)
    return j["rows"][0]["t"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="pgai-mon base URL")
    ap.add_argument("--dashboards", default=os.path.join(HERE, "dashboards"))
    ap.add_argument("--now-ms", type=int, help="Grafana's now (default: the newest pgwatch_db_size_size_b sample)")
    ap.add_argument("--range", help="override every dashboard's range, e.g. 6h (default: the dashboard's own)")
    ap.add_argument("--var", action="append", default=[], help="name=value[,value...]: set a variable")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--expect", help="JSON {\"<uid>/<panel id>\": class}: exit 1 on any mismatch")
    ap.add_argument("--quiet", action="store_true", help="print only problems and the summary")
    ap.add_argument("--only", help="only this panel, <file prefix>:<panel id> (e.g. Dashboard_1_:7); prints its SQL")
    args = ap.parse_args(argv)
    now = args.now_ms or newest_sample_ms(args.url)
    overrides = {}
    for kv in args.var:
        k, _, v = kv.partition("=")
        overrides[k] = v.split(",")
    rng = parse_duration_ms(args.range) if args.range else None
    results, bad = [], 0
    only_file, only_panel = (args.only.rsplit(":", 1) + [None])[:2] if args.only else (None, None)
    if args.only:
        SHOW_SQL.append(True)
    for f in sorted(os.listdir(args.dashboards)):
        if not f.endswith(".json") or (only_file and not f.startswith(only_file)):
            continue
        d, ctx, vreport, panels = check_dashboard(os.path.join(args.dashboards, f), args.url, now, rng, overrides,
                                                  int(only_panel) if only_panel else None)
        results.append({"file": f, "uid": d.get("uid"), "title": d.get("title"), **ctx,
                        "variable_queries": vreport, "panels": panels})
        print(f"== {f} {iso(ctx['from_ms'])} .. {iso(ctx['to_ms'])} "
              + " ".join(f"{k}={','.join(v)[:40]}" for k, v in ctx["variables"].items()))
        for v in vreport:
            if v["class"] != "data" or not args.quiet:
                print(f"   variable {v['variable']}: {v['class']} {v.get('options', '')}{v.get('error', '')}")
        for p in panels:
            if args.quiet and p["class"] in ("data", "none"):
                continue
            errs = "; ".join(f"{x['ref']}: {x['error'][:300]}" for x in p["targets"] if x.get("error"))
            print(f"   {p['class']:8} panel {p['panel']:>4} {p['type'] or '':12} {p['title'][:50]!r:52} "
                  f"rows {p.get('rows', 0):>6} {p.get('ms', 0):>6} ms {errs}")
    allp = [p for r in results for p in r["panels"]]
    counts = {c: sum(1 for p in allp if p["class"] == c) for c in ("data", "empty", "error", "external", "none")}
    verr = sum(1 for r in results for v in r["variable_queries"] if v["class"] == "error")
    print(f"panels: {counts}; variable query errors: {verr}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    if args.expect:
        want = json.load(open(args.expect))
        got = {f"{p['dashboard']}/{p['panel']}": p["class"] for p in allp}
        for k, cls in sorted(want.items()):
            if got.get(k) != cls:
                bad += 1
                print(f"MISMATCH {k}: expected {cls}, got {got.get(k)}")
    bad += counts["error"] + verr
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
