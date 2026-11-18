#!/usr/bin/env python3
"""Run every panel query of every provisioned dashboard through Grafana's own /api/ds/query.

  GF_PW=... grafana_panels.py EXPECTED_DASHBOARDS

GRAFANA_URL (default http://127.0.0.1:3000), GF_USER (default monitor, the compose admin) and
GF_PW (required) say where and as whom. Variables are resolved in order the way Grafana does
(All when the current value is All, else the first option); the last hour is queried. Prints one
line per dashboard and the detail of each panel in error. Exits 1 when Grafana cannot be reached
or refuses the login, fewer than EXPECTED_DASHBOARDS dashboards exist, any variable or panel query
fails, or no panel returns data at all (empty panels are fine on a demo database). Stdlib only.
"""
import base64, copy, json, os, re, sys, time, urllib.error, urllib.request

G = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000").rstrip("/")
AUTH = "Basic " + base64.b64encode(
    f"{os.environ.get('GF_USER', 'monitor')}:{os.environ.get('GF_PW', '')}".encode()).decode()
RANGE_MS = 3600_000
NOW = int(time.time() * 1000); FROM = NOW - RANGE_MS; TO = NOW
INTERVAL_MS = 15000


def req(path, body=None):
    r = urllib.request.Request(G + path, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": AUTH, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read() or b"null")
    except urllib.error.HTTPError as e:
        t = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(t)
        except ValueError:
            return e.code, {"message": t[:300]}


def sqlstring(v):
    if isinstance(v, list):
        return ",".join(sqlstring(x) for x in v)
    return "'" + str(v).replace("'", "''") + "'"


def regex(v):
    if isinstance(v, list):
        return "(" + "|".join(re.escape(str(x)) for x in v) + ")" if len(v) > 1 else regex(v[0])
    return re.escape(str(v))


def fmt(v, f):
    if v is None:
        v = ""
    if isinstance(v, dict):  # a custom allValue goes in raw, whatever the format
        return v["__raw"]
    if f == "sqlstring":
        return sqlstring(v)
    if f == "regex":
        return regex(v)
    if f in ("csv", None) and isinstance(v, list):
        return ",".join(map(str, v))
    if f == "pipe" and isinstance(v, list):
        return "|".join(map(str, v))
    return str(v) if not isinstance(v, list) else "{" + ",".join(map(str, v)) + "}"


def interp(s, vars_):
    builtins = {"__from": str(FROM), "__to": str(TO), "__interval_ms": str(INTERVAL_MS),
                "__interval": f"{INTERVAL_MS // 1000}s", "__range_s": str(RANGE_MS // 1000),
                "__range_ms": str(RANGE_MS), "__rate_interval": "60s", "__range": f"{RANGE_MS // 1000}s"}
    allv = {**builtins, **vars_}

    def rep(m):
        name = m.group(1) or m.group(3) or m.group(4)
        if name not in allv:
            return m.group(0)
        return fmt(allv[name], m.group(2))
    return re.sub(r"\$\{(\w+)(?::(\w+))?\}|\$(?:(__\w+)|(\w+))", rep, s)


def interp_obj(o, vars_):
    if isinstance(o, str):
        return interp(o, vars_)
    if isinstance(o, list):
        return [interp_obj(x, vars_) for x in o]
    if isinstance(o, dict):
        return {k: interp_obj(v, vars_) for k, v in o.items()}
    return o


def ds_query(target, ds):
    """-> (data|empty|error, info, first-column values)"""
    # Keep the target's parser, columns and root selector, including CSV chart targets.
    t = copy.deepcopy(target)
    t.setdefault("refId", "A")
    t["datasource"] = t.get("datasource") or ds  # the target's own datasource wins over the panel's
    t["intervalMs"] = INTERVAL_MS
    t["maxDataPoints"] = 1000
    code, res = req("/api/ds/query", {"queries": [t], "from": str(FROM), "to": str(TO)})
    if not (isinstance(res, dict) and "results" in res):
        msg = res.get("message", res) if isinstance(res, dict) else res
        return "error", f"HTTP {code}: {str(msg)[:300]}", []
    r = res["results"].get(t["refId"], {})
    if r.get("error"):
        return "error", str(r["error"])[:300], []
    rows = 0; vals = []
    for fr in r.get("frames", []):
        vs = fr.get("data", {}).get("values", [])
        if vs:
            rows += len(vs[0])
            vals.extend(vs[0][:200])
    return ("data" if rows else "empty"), f"{rows} rows", vals


def resolve_vars(dash, errors):
    vars_ = {}
    for v in dash.get("templating", {}).get("list", []):
        n = v["name"]; ty = v.get("type")
        cur = (v.get("current") or {}).get("value")
        if ty == "query":
            q = v.get("query")
            target = q.get("infinityQuery", q) if isinstance(q, dict) else {"refId": "variable"}
            st, info, vals = ds_query(interp_obj(target, vars_), v.get("datasource"))
            vals = [x for x in vals if x is not None]
            if st == "error":
                errors.append(f"variable ${n}: {info}")
            if v.get("includeAll") and (cur in ("$__all", ["$__all"]) or not vals):
                vars_[n] = {"__raw": v["allValue"]} if v.get("allValue") else (vals or [""])
            elif v.get("multi") and vals:
                vars_[n] = vals[:1]
            else:
                vars_[n] = vals[0] if vals else (cur if cur not in (None, "$__all") else "")
        elif ty in ("custom", "constant", "textbox", "interval"):
            if cur in ("$__all", ["$__all"]) and v.get("allValue"):
                vars_[n] = {"__raw": v["allValue"]}
            else:
                vars_[n] = cur if cur is not None else v.get("query", "")
        elif ty == "datasource":
            vars_[n] = cur
    return vars_


def panels(ps):
    for p in ps:
        if p.get("type") == "row":
            yield from panels(p.get("panels", []))
        else:
            yield p


def check_dashboard(dash, tot):
    """Runs every panel of one dashboard; adds to tot; returns the error lines."""
    errors = []
    vars_ = resolve_vars(dash, errors)
    for p in panels(dash.get("panels", [])):
        ts = [t for t in p.get("targets", []) if not t.get("hide")]
        if not ts:
            tot["noquery"] += 1; continue
        worst = "empty"
        for t in ts:
            st, info, _ = ds_query(interp_obj(t, vars_), p.get("datasource"))
            if st == "error":
                worst = "error"
                errors.append(f"panel {p.get('title')!r} ({t.get('refId')}): {info}")
            elif st == "data" and worst != "error":
                worst = "data"
        tot[worst] += 1
    return errors


def main():
    expected = int(sys.argv[1])
    try:
        code, found = req("/api/search?type=dash-db&limit=500")
    except OSError as e:
        sys.exit(f"panels: FAIL: cannot reach Grafana at {G}: {e}")
    if code != 200 or not isinstance(found, list):
        sys.exit(f"panels: FAIL: dashboard search -> HTTP {code} (login refused?)")
    tot = {"error": 0, "data": 0, "empty": 0, "noquery": 0}
    failed = False
    for d in found:
        c, full = req(f"/api/dashboards/uid/{d['uid']}")
        if c != 200:
            print(f"panels: {d['title']}: FAIL: load -> HTTP {c}"); failed = True; continue
        before = dict(tot)
        errors = check_dashboard(full["dashboard"], tot)
        counts = " ".join(f"{k} {tot[k] - before[k]}" for k in tot)
        print(f"panels: {d['title']}: {counts}")
        for e in errors:
            print(f"panels:   ERROR {e}")
        failed |= bool(errors)
    print("panels: total", json.dumps(tot))
    if len(found) < expected:
        print(f"panels: FAIL: {len(found)} dashboards in Grafana, want {expected}"); failed = True
    if tot["data"] == 0:
        print("panels: FAIL: no panel returned data"); failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
