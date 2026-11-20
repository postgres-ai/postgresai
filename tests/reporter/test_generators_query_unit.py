from __future__ import annotations

import json
from typing import Any

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_query_file


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


class _FakeBatch(dict):
    """get_queries_metrics_from_prometheus's reply for one node: metrics for any (datname, queryid)."""

    def __init__(self, node_name: str, hours: int):
        super().__init__()
        self.node_name, self.hours = node_name, hours

    def get(self, key, default=None):
        db_name, queryid = key
        return {
            "calls": float(len(self.node_name) + len(db_name)),
            "total_time": float(len(queryid)),
            "rows": float(self.hours),
        }


def _fake_batch(cluster: str, node_name: str, queryids, hours: int) -> _FakeBatch:
    return _FakeBatch(node_name, hours)


@pytest.mark.unit
def test_generate_per_query_jsons_groups_by_queryid_and_is_node_first(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {
            "airflow_db_prod": {"qid_1"},
            "db_b": {"qid_1", "qid_2"},
        },
    )
    monkeypatch.setattr(
        generator,
        "get_queryid_queries_from_sink",
        lambda *args, **kwargs: {
            "airflow_db_prod": {"qid_1": "SELECT 1"},
            "db_b": {"qid_1": "SELECT 1", "qid_2": "SELECT 2"},
        },
    )
    monkeypatch.setattr(
        generator,
        "get_all_nodes",
        lambda cluster: {"primary": "main", "standbys": ["replica-1", "replica-2"]},
    )
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=False,
    )

    assert {item["filename"] for item in out} == {"prod_query_qid_1.json", "prod_query_qid_2.json"}

    q1 = next(item["data"] for item in out if item["filename"] == "prod_query_qid_1.json")
    validate_query_file(q1)

    assert q1["cluster_id"] == "prod"
    assert q1["query_id"] == "qid_1"
    assert q1["query_text"] == "SELECT 1"
    assert q1["nodes"]["primary"] == "main"
    assert q1["nodes"]["standbys"] == ["replica-1", "replica-2"]

    # Node is the primary dimension.
    assert set(q1["results"].keys()) == {"main", "replica-1", "replica-2"}
    assert set(q1["results"]["main"].keys()) == {"airflow_db_prod", "db_b"}

    # Each node's batch reply lands in that node's blocks.
    assert q1["results"]["replica-1"]["db_b"]["metrics"] == {"calls": 13.0, "total_time": 5.0, "rows": 24.0}

    # time_range is top-level, not in per-db metrics.
    assert q1["time_range"]["hours"] == 24
    assert "time_range" not in q1["results"]["main"]["airflow_db_prod"]["metrics"]


@pytest.mark.unit
def test_generate_per_query_jsons_write_immediately_prefixes_cluster_and_writes_timestamptz_last(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generator, "extract_queryids_from_reports", lambda reports: {"db1": {"qid_1"}})
    monkeypatch.setattr(
        generator, "get_queryid_queries_from_sink", lambda *args, **kwargs: {"db1": {"qid_1": "SELECT 1"}}
    )
    monkeypatch.setattr(generator, "get_all_nodes", lambda cluster: {"primary": "main", "standbys": ["replica-1"]})
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=True,
    )

    assert out == [{"filename": "prod_query_qid_1.json"}]
    p = tmp_path / "prod_query_qid_1.json"
    assert p.exists()

    payload = json.loads(p.read_text(encoding="utf-8"))
    validate_query_file(payload)

    # Ensure timestamptz is last key in the emitted JSON text (ordering requirement).
    raw = p.read_text(encoding="utf-8").rstrip()
    last_key_line = [ln for ln in raw.splitlines() if ln.lstrip().startswith('"')][-1]
    assert last_key_line.lstrip().startswith('"timestamptz"')


@pytest.mark.unit
def test_generate_per_query_jsons_returns_empty_when_no_queryids(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    """Test that generate_per_query_jsons returns empty list when no queryids found."""
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {},  # No queryids
    )

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=False,
    )

    assert out == []


@pytest.mark.unit
def test_generate_per_query_jsons_handles_missing_query_text(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    """Test that missing query text doesn't crash generation."""
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {"db1": {"qid_1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_queryid_queries_from_sink",
        lambda *args, **kwargs: {},  # No query texts found
    )
    monkeypatch.setattr(
        generator,
        "get_all_nodes",
        lambda cluster: {"primary": "main", "standbys": []},
    )
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=False,
    )

    # Should still generate the file, just with None query_text
    assert len(out) == 1
    assert out[0]["data"]["query_text"] is None


@pytest.mark.unit
def test_generate_per_query_jsons_without_cluster_prefix(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    """Test that include_cluster_prefix=False removes cluster prefix from filenames."""
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {"db1": {"qid_1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_queryid_queries_from_sink",
        lambda *args, **kwargs: {"db1": {"qid_1": "SELECT 1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_all_nodes",
        lambda cluster: {"primary": "main", "standbys": []},
    )
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=False,
        include_cluster_prefix=False,
    )

    # Filename should not have cluster prefix
    assert out[0]["filename"] == "query_qid_1.json"


@pytest.mark.unit
def test_generate_per_query_jsons_with_single_node(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    """Test generate_per_query_jsons with explicit node_name (backward compatibility)."""
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {"db1": {"qid_1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_queryid_queries_from_sink",
        lambda *args, **kwargs: {"db1": {"qid_1": "SELECT 1"}},
    )
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name="node-01",  # Explicit node
        hours=24,
        write_immediately=False,
    )

    assert len(out) == 1
    data = out[0]["data"]

    # Should have only the specified node
    assert set(data["results"].keys()) == {"node-01"}
    assert data["nodes"]["primary"] == "node-01"
    assert data["nodes"]["standbys"] == []


@pytest.mark.unit
def test_generate_per_query_jsons_uses_default_node_when_none_found(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    """Test that default node-01 is used when no nodes are found."""
    monkeypatch.setattr(
        generator,
        "extract_queryids_from_reports",
        lambda reports: {"db1": {"qid_1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_queryid_queries_from_sink",
        lambda *args, **kwargs: {"db1": {"qid_1": "SELECT 1"}},
    )
    monkeypatch.setattr(
        generator,
        "get_all_nodes",
        lambda cluster: {"primary": None, "standbys": []},  # No nodes found
    )
    monkeypatch.setattr(generator, "get_queries_metrics_from_prometheus", _fake_batch)

    out = generator.generate_per_query_jsons(
        reports={"K001": {}},
        cluster="prod",
        node_name=None,
        hours=24,
        write_immediately=False,
    )

    assert len(out) == 1
    data = out[0]["data"]

    # Should fall back to node-01
    assert set(data["results"].keys()) == {"node-01"}
    assert data["nodes"]["primary"] == "node-01"
