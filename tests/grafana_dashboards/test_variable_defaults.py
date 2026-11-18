"""Fresh installs must select their own objects, never a saved developer's database."""
import json
import re

import pytest

from .conftest import dashboard_file


def test_no_deployment_specific_defaults(dashboard_path):
    dashboard = json.loads(dashboard_path.read_text())
    names = {"query_id", "schema_name", "table_name", "index_name", "db_name",
             "datname", "node_name", "cluster_name"}
    for variable in dashboard["templating"]["list"]:
        if variable.get("type") != "query" and variable["name"] not in names:
            continue
        current = variable.get("current") or {}
        allowed = (None, "", [], "$__all", ["$__all"]) if variable.get("includeAll") else (None, "", [])
        assert current.get("value") in allowed, variable["name"]
        assert current.get("text") in (None, "", [], "All", ["All"]), variable["name"]
        for option in variable.get("options") or []:
            assert option.get("value") in allowed, variable["name"]
            assert option.get("text") in (None, "", "All"), variable["name"]
        if variable.get("type") == "textbox":
            assert not variable.get("query"), variable["name"]


@pytest.mark.parametrize("number,name,label", [
    (9, "schema_name", "schema"), (9, "table_name", "table_name"),
    (11, "schema_name", "schemaname"), (11, "index_name", "indexrelname"),
])
def test_single_object_excludes_aggregate_bucket(number, name, label):
    dashboard = json.loads(dashboard_file(number).read_text())
    variable = next(v for v in dashboard["templating"]["list"] if v["name"] == name)
    sql = variable["query"]["infinityQuery"]["url_options"]["data"]
    assert sql == variable["definition"]
    assert f"labels['{label}'] <> '$other$'" in sql

    # Execute the shipped selector over a tiny series view, including legitimate dollar names.
    duckdb = pytest.importorskip("duckdb")
    sql = re.sub(r"\$\{\w+:sqlstring\}", "'selected'", sql)
    metric = re.search(r"metric = '([^']+)'", sql)[1]
    with duckdb.connect() as db:
        db.execute("CREATE TABLE series (metric VARCHAR, labels MAP(VARCHAR, VARCHAR))")
        for value in ("$other$", "$other$table", "a", "z"):
            labels = {key: "selected" for key in re.findall(r"labels\['([^']+)'\]", sql)}
            labels[label] = value
            db.execute("INSERT INTO series VALUES (?, ?)", [metric, labels])
        assert db.execute(sql).fetchall() == [("$other$table",), ("a",), ("z",)]
        db.execute("DELETE FROM series WHERE labels[?] <> '$other$'", [label])
        assert db.execute(sql).fetchall() == []


@pytest.mark.parametrize("number", [8, 10])
def test_overviews_keep_aggregate_bucket(number):
    dashboard = json.loads(dashboard_file(number).read_text())
    assert "<> '$other$'" not in json.dumps(dashboard)
