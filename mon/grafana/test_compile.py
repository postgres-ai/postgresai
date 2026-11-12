#!/usr/bin/env python3
"""Offline checks of compile.py and of the dashboards it emitted (no pgai-mon, no Grafana).

  python3 mon/grafana/test_compile.py

The emitted SQL itself is checked against a store by `compile.py --check` (every target, two
ranges, two db selections, parity with diff.py-style inlining).
"""
import json
import os
import re
import unittest

import compile as C

ALLOWED_VARS = {"${__from}", "${__to}", "${__interval_ms}", "${cluster_name:sqlstring}", "${node_name:sqlstring}",
                "${db_name:regex}", "${db_name:sqlstring}", "${top_n:sqlstring}",
                # Dashboard_4, 3, 7 filter variables
                "${wait_event_type:regex}", "${wait_event:regex}", "${query_id:sqlstring}", "${schema_name:regex}",
                "${table_name:regex}",
                # Dashboard_9, 11, 6 (matchers on a variable, e.g. index_name=~"$index_name")
                "${schema_name:sqlstring}", "${table_name:sqlstring}", "${index_name:sqlstring}",
                "${index_name:regex}", "${datname:sqlstring}", "${datname:regex}", "${replica:regex}",
                # Dashboard_2's query text mode (its pg_stat_statements table)
                "${legend_label:sqlstring}",
                # Dashboard_6's custom threshold
                "${lag_alert_s:sqlstring}"}


def load(name):
    with open(name) as f:
        return json.load(f)


def panels(d):
    return {p["id"]: p for p in C.walk(d["panels"])}


class Emitted(unittest.TestCase):
    def pairs(self):
        for spec in C.DASHBOARDS:
            src = load(os.path.join(C.DASH_DIR, spec["src"]))
            out = load(os.path.join(C.HERE, "dashboards", spec["out"]))
            yield spec, src, out

    def test_identity_new_uid_and_title_same_tags(self):
        uids = set()
        for spec, src, out in self.pairs():
            self.assertNotEqual(out["uid"], src["uid"])
            self.assertEqual(out["uid"], spec["uid"])
            self.assertLessEqual(len(out["uid"]), 40)
            self.assertEqual(out["title"], src["title"] + " (pgai-mon)")
            self.assertEqual(out["tags"], src["tags"])
            self.assertEqual(out["tags"][0], "postgres-ai")
            uids.add(out["uid"])
        self.assertEqual(len(uids), len(C.DASHBOARDS))

    def test_same_panels_layout_units_legends(self):
        for spec, src, out in self.pairs():
            ps, po = panels(src), panels(out)
            self.assertEqual(sorted(ps), sorted(po))
            for i, a in ps.items():
                b = po[i]
                for k in ("type", "title", "gridPos", "options", "description", "maxDataPoints", "transformations"):
                    self.assertEqual(a.get(k), b.get(k), f"panel {i} {k}")
                da, db = a.get("fieldConfig", {}).get("defaults"), b.get("fieldConfig", {}).get("defaults")
                self.assertEqual(da, db, f"panel {i} field defaults (units, thresholds)")
                # the original overrides are kept, after the compiler's display-name ones
                oa = a.get("fieldConfig", {}).get("overrides", [])
                ob = b.get("fieldConfig", {}).get("overrides", [])
                self.assertEqual(ob[len(ob) - len(oa):], oa, f"panel {i} overrides")

    def test_every_promql_target_is_an_infinity_target(self):
        counts = {}
        for spec, src, out in self.pairs():
            ps, po = panels(src), panels(out)
            n = 0
            for i, a in ps.items():
                ta, tb = a.get("targets") or [], po[i].get("targets") or []
                self.assertEqual([t["refId"] for t in ta], [t["refId"] for t in tb], f"panel {i}")
                if (spec["census"], i) in C.TABLE_PANELS:
                    continue  # test_table_panels
                for x, y in zip(ta, tb):
                    if not x.get("expr") or (spec.get("partial") and y.get("expr")):
                        # non-Prometheus targets, and a partial port's unported ones, are copied
                        self.assertEqual(x, y)
                        continue
                    n += 1
                    self.assertNotIn("expr", y)
                    self.assertEqual(y["datasource"], C.INFINITY)
                    # an instant target of a table panel in Prometheus' table format: Time, a
                    # column per label, Value (the panel's transformations name those)
                    table = bool(x.get("instant")) and x.get("format") == "table" and a.get("type") == "table"
                    self.assertEqual((y["type"], y["source"], y["parser"], y["format"]),
                                     ("json", "url", "backend", "table" if table else "timeseries"))
                    sel = [c["selector"] for c in y["columns"]]
                    if table:
                        self.assertEqual((sel[0], sel[-1]), ("Time", "Value"))
                        named = set()
                        for tr in a.get("transformations") or []:
                            o = tr.get("options", {})
                            named |= set(o.get("indexByName") or {}) | set((o.get("include") or {}).get("names") or [])
                        self.assertTrue(named <= set(sel), f"panel {i}: {named - set(sel)}")
                    else:
                        self.assertEqual(sel[:3], ["time", "value", "series"])
                    self.assertTrue(y["url"].endswith(":8765/sql?format=objects"), y["url"])
                    self.assertEqual(y["root_selector"], "rows")
                    self.assertEqual(y["url_options"]["method"], "POST")
                    self.assertEqual(y.get("hide", False), x.get("hide", False))
                    body = y["url_options"]["data"]
                    self.assertNotIn("getvariable(", body)
                    # no reference-store literal leaks into a matcher (D9 44B-E once kept smatchers')
                    for lit in ("'local'", "'node-01'", "'target_database'", "'t135'", "'pgbench_accounts_pkey'"):
                        self.assertNotIn(lit, body, f"{spec['census']} panel {i}{y['refId']}")
                    self.assertFalse(body.lstrip().startswith("{"), "a JSON-looking body is parsed as JSON")
                    self.assertTrue(set(re.findall(r"\$\{[^}]+\}", body)) <= ALLOWED_VARS, body)
                    if not x.get("instant"):  # range: aligned to the pinned step
                        self.assertIn(f"${{__from}} - ${{__from}} % {C.STEP}", body)
                        self.assertIn(f"${{__to}} - ${{__to}} % {C.STEP}", body)
                    else:
                        self.assertIn("${__to} - ${__to} % ${__interval_ms}", body)
                    legend = {"matcher": {"id": "byFrameRefID", "options": y["refId"]},
                              "properties": [{"id": "displayName", "value": "${__field.labels.series}"}]}
                    if table:
                        self.assertNotIn(legend, po[i].get("fieldConfig", {}).get("overrides", []))
                    else:
                        self.assertIn(legend, po[i]["fieldConfig"]["overrides"])
                if any(t.get("expr") for t in tb):  # a partial port's panel mixes both datasources
                    self.assertEqual(po[i]["datasource"], {"type": "datasource", "uid": "-- Mixed --"})
                    self.assertTrue(spec.get("partial"))
                elif any(t.get("expr") for t in ta):
                    self.assertEqual(po[i]["datasource"], C.INFINITY)
            counts[spec["census"]] = n
        self.assertEqual(counts, {"D2": 79, "D1": 23, "D4": 5, "D3": 16, "D7": 28, "D13": 8, "D12": 7, "D14": 17,
                                  "D5": 13, "D9": 32, "D11": 9, "D6": 24, "D8": 26, "D10": 7})

    def test_table_panels(self):
        """The panels that called the Flask backend or the Postgres sink run pgai-mon SQL instead, with
        the original column selectors (the overrides and transformations name them)."""
        seen = set()
        for spec, src, out in self.pairs():
            ps, po = panels(src), panels(out)
            for i, a in ps.items():
                tp = C.TABLE_PANELS.get((spec["census"], i))
                if tp is None:
                    continue
                seen.add((spec["census"], i))
                b = po[i]
                self.assertEqual(b["datasource"], C.INFINITY)
                ta, tb = a["targets"], b["targets"]
                self.assertEqual([t["refId"] for t in ta], [t["refId"] for t in tb])
                for x, y in zip(ta, tb):
                    self.assertEqual(y["datasource"], C.INFINITY)
                    self.assertEqual((y["type"], y["source"], y["parser"], y["format"], y["root_selector"]),
                                     ("json", "url", "backend", "table", "rows"))
                    self.assertTrue(y["url"].endswith(":8765/sql?format=objects"), y["url"])
                    self.assertEqual(y["url_options"]["method"], "POST")
                    if tp["columns"] is None:
                        self.assertEqual(y["columns"], x["columns"])
                    self.assertEqual(y.get("filterExpression"), x.get("filterExpression"))
                    body = y["url_options"]["data"]
                    self.assertNotIn("getvariable(", body)
                    self.assertNotIn("flask", body)
                    self.assertTrue(set(re.findall(r"\$\{[^}]+\}", body)) <= ALLOWED_VARS, body)
                    # every column the panel reads is selected by the SQL (as a quoted name or an alias)
                    for c in y["columns"]:
                        sel = re.escape(c["selector"])
                        self.assertRegex(body, r'(?i)\bAS\s+"%s"|\bAS\s+%s\b|[\s.]%s,' % (sel, sel, sel),
                                         f"{spec['census']} panel {i}: {c['selector']}")
        self.assertEqual(seen, set(C.TABLE_PANELS))
        # nothing calls the Flask backend or the pgwatch Postgres sink any more
        for spec, src, out in self.pairs():
            text = json.dumps(out)
            self.assertNotIn("flask-pgss-api", text)
            self.assertNotIn("P031DD592934B2F1F", text)

    def test_variables(self):
        for spec, src, out in self.pairs():
            vs, vo = src["templating"]["list"], out["templating"]["list"]
            self.assertEqual([v["name"] for v in vs], [v["name"] for v in vo])
            for a, b in zip(vs, vo):
                if a["type"] != "query":
                    self.assertEqual(a, b)
                    continue
                for k in ("multi", "includeAll", "label", "current", "refresh", "allValue"):
                    self.assertEqual(a.get(k), b.get(k), f"{a['name']} {k}")
                self.assertEqual(b["datasource"], C.INFINITY)
                self.assertEqual(b["query"]["infinityQuery"]["url_options"]["data"], b["definition"])

    def test_plain_variables_are_sql_strings(self):
        """A one-value variable reaches the SQL only as ${x:sqlstring} (Grafana quotes it), never
        inside quotes of the compiler's own, where a ' in the value would end the literal."""
        for spec, src, out in self.pairs():
            bodies = [v["definition"] for v in out["templating"]["list"] if v["type"] == "query"]
            bodies += [(t.get("url_options") or {}).get("data", "")
                       for p in C.walk(out["panels"]) for t in p.get("targets") or []]
            for body in bodies:
                self.assertIsNone(re.search(r"'\$\{(?!__)\w+\}'|CAST\(\$\{(?!__)\w+\} ", body), body)

    def test_filter_variables_only_where_the_dashboard_has_them(self):
        for spec, src, out in self.pairs():
            names = {v["name"] for v in src["templating"]["list"]}
            for p in C.walk(out["panels"]):
                for t in p.get("targets") or []:
                    body = (t.get("url_options") or {}).get("data", "")
                    for var in re.findall(r"\$\{(\w+)(?::\w+)?\}", body):
                        if not var.startswith("__"):
                            self.assertIn(var, names, f"{spec['census']} panel {p['id']}{t['refId']}")


class Pieces(unittest.TestCase):
    def test_step_is_pinned_to_the_rollup_tier(self):
        h, d = 3_600_000, 86_400_000
        cases = [(6 * h, 30_000, 30_000), (7 * d - 1, 600_000, 600_000), (7 * d, 600_000, h),
                 (30 * d, 2_400_000, h), (30 * d, 2 * h + 1, 3 * h), (90 * d, 2 * h, d), (365 * d, 3 * d, 3 * d)]
        for span, iv, want in cases:
            self.assertEqual(C.pin_step(span, iv), want, (span, iv))
        self.assertIn("ELSE ${__interval_ms} END", C.STEP)
        # range targets use the pinned step; instants keep $__interval
        self.assertIn(C.STEP, C.GRAFANA_PARAMS_RANGE["step_ms"])
        self.assertNotIn("CASE", C.GRAFANA_PARAMS["step_ms"])

    def test_rate_interval_is_grafanas(self):
        e = C.GRAFANA_PARAMS["rate_interval_ms"]
        self.assertEqual(e, "CAST(greatest(${__interval_ms} + 5000, 20000) AS BIGINT)")
        # max(interval + scrape, 4 x scrape), scrape 5 s: 1 s -> 20 s, 30 s -> 35 s
        f = lambda iv: max(iv + C.SCRAPE_MS, 4 * C.SCRAPE_MS)  # noqa: E731
        self.assertEqual((f(1000), f(30000)), (20000, 35000))

    def test_label_values(self):
        self.assertEqual(
            C.label_values_sql('label_values(pgwatch_db_size_size_b{cluster="$cluster_name", node_name="$node_name", '
                               'datname!="template1"},datname)'),
            "SELECT DISTINCT labels['datname'] AS value FROM series WHERE metric = 'pgwatch_db_size_size_b' AND "
            "labels['datname'] IS NOT NULL AND coalesce(labels['cluster'], '') = '${cluster_name}' AND "
            "coalesce(labels['node_name'], '') = '${node_name}' AND coalesce(labels['datname'], '') <> 'template1' "
            "ORDER BY 1")
        self.assertIn("regexp_full_match(coalesce(labels['a'], ''), 'x.*')",
                      C.label_values_sql('label_values(m{a=~"x.*"}, b)'))
        # a variable in a regex matcher is interpolated as a regex (multi-value / All -> (a|b))
        self.assertIn("regexp_full_match(coalesce(labels['wait_event_type'], ''), '${wait_event_type:regex}')",
                      C.label_values_sql('label_values(m{wait_event_type=~"$wait_event_type"}, wait_event)'))
        self.assertIn("coalesce(labels['datname'], '') = '${db_name}'",
                      C.label_values_sql('label_values(m{datname="$db_name"}, schema)'))

    def test_sqlstring_vars(self):
        self.assertEqual(C.sqlstring_vars("a = '${query_id}' AND n = CAST(${top_n} AS BIGINT)"),
                         "a = ${query_id:sqlstring} AND n = CAST(${top_n:sqlstring} AS BIGINT)")
        # regex-formatted variables and built-ins are left alone
        s = "regexp_full_match(x, '${db_name:regex}') AND t < CAST(${__to} AS BIGINT)"
        self.assertEqual(C.sqlstring_vars(s), s)
        # --check renders ${x:sqlstring} the way Grafana does
        self.assertEqual(C.interpolate("${query_id:sqlstring}", {"${query_id}": "o'k"}), "'o''k'")

    def test_legend(self):
        cols = ["labels", "t", "value"]
        self.assertEqual(C.legend_expr("{{datname}}", cols), "coalesce(q.labels['datname'], '')")
        self.assertEqual(C.legend_expr("Max connections", cols), "'Max connections'")
        self.assertEqual(C.legend_expr("Postgres - {{wait_event_type}}", ["wait_event_type", "t", "value"]),
                         "'Postgres - ' || coalesce(CAST(q.\"wait_event_type\" AS VARCHAR), '')")
        self.assertIn("_displayname_long", C.legend_expr("{{$legend_label}}", ["labels", "t", "value", "_displayname_long"]))
        self.assertEqual(C.legend_expr("__auto", ["t", "value"]), "''")

    def test_matchers_become_variables(self):
        self.assertEqual(
            C.matchers_literal([["cluster", "=", "local"], ["node_name", "=", "node-01"],
                                ["datname", "=~", "target_database"], ["setting_name", "=", "block_size"]]),
            "[['cluster', '=', '${cluster_name}'], ['node_name', '=', '${node_name}'], "
            "['datname', '=~', '${db_name:regex}'], ['setting_name', '=', 'block_size']]")
        self.assertEqual(C.matchers_literal([]), "[]::VARCHAR[][]")


if __name__ == "__main__":
    unittest.main()
