"""Managed targets upload reports from their own pgwatch sources (#878)."""
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reporter import postgres_reports
from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture
def run_projects(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    projects_file = tmp_path / ".pgai-report-projects.json"
    monkeypatch.setenv("REPORTER_PROJECTS_PATH", str(projects_file))
    generator = PostgresReportGenerator("http://prom.test", "")
    factory = MagicMock(return_value=generator)
    factory.DEFAULT_EXCLUDED_DATABASES = PostgresReportGenerator.DEFAULT_EXCLUDED_DATABASES
    monkeypatch.setattr(postgres_reports, "PostgresReportGenerator", factory)
    monkeypatch.setattr(generator, "test_connection", lambda: True)
    monkeypatch.setattr(generator, "get_all_databases", lambda *a: ["shared_db"])
    monkeypatch.setattr(generator, "get_all_clusters", lambda: ["default"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda *a: {})
    monkeypatch.setattr(generator, "get_queryid_queries_from_sink", lambda *a, **k: {})
    monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data",
                        lambda cluster, node, *a, **k: ({"42": [1]}, [0], [0]))
    monkeypatch.setattr(generator, "get_query_metrics_from_prometheus",
                        lambda cluster, node, *a, **k: {"object": f"{node}_table"})
    calls, uploads, finishes = [], [], []

    def query(q):
        match = re.search(r'node_name="([^\"]+)"', q)
        sources = [match[1]] if match else ["app", "app2"]
        rows = []
        for source in sources:
            labels = {"cluster": "default", "node_name": source, "datname": "shared_db",
                      "dbname": "shared_db", "schema_name": "public", "schemaname": "public",
                      "table_name": f"{source}_table", "tblname": f"{source}_table",
                      "index_name": f"{source}_index"}
            if any(metric in q for metric in ["pgwatch_settings_configured", "pgwatch_db_stats_in_recovery_int",
                                             "pgwatch_unused_indexes_", "pgwatch_redundant_indexes_",
                                             "pgwatch_pg_table_bloat_"]):
                rows.append({"metric": labels, "value": [0, "0" if "in_recovery" in q else "123"]})
        return {"status": "success", "data": {"result": rows}}

    monkeypatch.setattr(generator, "query_instant", query)
    def create(api, token, project, epoch):
        calls.append(project)
        return project
    monkeypatch.setattr(generator, "create_report", create)
    monkeypatch.setattr(generator, "upload_report_file",
                        lambda api, token, report, file: uploads.append((report, Path(file).name, json.loads(Path(file).read_text()))))
    monkeypatch.setattr(generator, "finish_report", lambda *a, **k: finishes.append((a, k)))

    def run(projects, fail=None):
        if projects is not None:
            projects_file.write_text(json.dumps({"projects": projects}))
        if fail:
            original = generator.generate_all_reports
            def generate(cluster, node, combine):
                if node == fail:
                    raise RuntimeError("synthetic project failure")
                return original(cluster, node, combine)
            monkeypatch.setattr(generator, "generate_all_reports", generate)
        monkeypatch.setattr(sys, "argv", ["postgres_reports.py", "--project-name", "legacy",
                                          "--api-url", "http://api.test", "--token", "synthetic-token"])
        postgres_reports.main()
        return calls, uploads, finishes
    return run, generator


def test_each_project_uploads_only_its_source(run_projects):
    run, generator = run_projects
    # The legacy classifier labels the second primary a standby; the new path bypasses it.
    assert generator.get_all_nodes("default") == {"primary": "app", "standbys": ["app2"]}
    calls, uploads, _ = run(["app", "app2"])
    assert calls == ["app", "app2"]
    app2 = {name: data for project, name, data in uploads if project == "app2"}
    for check in ["H002", "H004", "F004"]:
        data = app2[f"{check}.json"]
        text = json.dumps(data)
        assert "app2_table" in text
        assert "app_table" not in text
        assert "app_index" not in text
        assert list(data["results"]) == ["app2"]
    query = app2["query_42.json"]
    assert query["nodes"] == {"primary": "app2", "standbys": []}
    assert list(query["results"]) == ["app2"]
    assert "app_table" not in json.dumps(query)


def test_absent_file_preserves_combined_legacy_uploads(run_projects):
    run, _ = run_projects
    calls, uploads, _ = run(None)
    assert calls == ["legacy"]
    h002 = next(data for _, name, data in uploads if name == "H002.json")
    assert list(h002["results"]) == ["app", "app2"]


def test_failed_project_does_not_stop_next_project(run_projects):
    run, _ = run_projects
    calls, uploads, finishes = run(["app", "app2"], fail="app")
    assert calls == ["app", "app2"]
    assert uploads and all(project == "app2" for project, _, _ in uploads)
    assert any(args[2] == "app" and "error" in kwargs for args, kwargs in finishes)
    assert any(args[2] == "app2" and not kwargs for args, kwargs in finishes)


def test_empty_projects_does_not_upload_detached_host(run_projects):
    run, _ = run_projects
    calls, uploads, _ = run([])
    assert calls == []
    assert uploads == []


@pytest.mark.parametrize("method", ["get_index_definitions_from_sink", "get_queryid_queries_from_sink"])
def test_sink_metadata_is_scoped_to_pgwatch_source(method):
    generator = PostgresReportGenerator("http://prom.test", "")
    generator.report_source = "app2"
    generator.pg_conn = MagicMock()
    cursor = generator.pg_conn.cursor.return_value.__enter__.return_value
    cursor.__iter__.return_value = iter([])
    if method == "get_index_definitions_from_sink":
        generator.get_index_definitions_from_sink("shared_db")
        assert cursor.execute.call_args.args[1] == ("app2",)
    else:
        generator.get_queryid_queries_from_sink()
        assert cursor.execute.call_args.args[1] == (["app2"],)


def test_compose_mount_sees_atomic_projects_file_replacements():
    import yaml
    compose = yaml.safe_load((Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text())
    reporter = compose["services"]["postgres-reports"]
    assert "./:/app/pgai-stack:ro" in reporter["volumes"]
    assert "REPORTER_PROJECTS_PATH=/app/pgai-stack/.pgai-report-projects.json" in reporter["environment"]
