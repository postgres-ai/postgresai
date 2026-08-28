"""System-schema exclusion in the monitoring-mode reporter (#345).

A customer checkup listed `pg_catalog.pg_class_tblspc_relfilenode_index` as an
unused index to drop. The metric SQL now excludes system schemas, but a fleet
still scraping with an older metrics.yml keeps shipping pg_catalog rows into
Prometheus, so H001/H002/H004 filter them out again on the way into the report.
"""
from typing import Any, Callable, Dict

import pytest

from reporter.postgres_reports import PostgresReportGenerator, is_system_schema


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(prometheus_url="http://prom.test", postgres_sink_url="")


def _query_stub(prom_result, mapping: Dict[str, Any]) -> Callable[[str], Dict[str, Any]]:
    """query_instant stub matching the substrings in `mapping`."""

    def _fake(query: str) -> Dict[str, Any]:
        for needle, payload in mapping.items():
            if needle in query:
                return payload(query) if callable(payload) else payload
        return prom_result()

    return _fake


SYSTEM_SCHEMAS = [
    "pg_catalog",
    "information_schema",
    "pg_toast",
    "pg_temp_1",
    "pg_temp_374",
    "pg_toast_temp_1",
    "pg_toast_temp_374",
]

USER_SCHEMAS = ["public", "app", "postgres_ai", "pg_partman", "pg_temp", "$other$"]


@pytest.mark.unit
@pytest.mark.parametrize("schema", SYSTEM_SCHEMAS)
def test_is_system_schema_true(schema: str) -> None:
    assert is_system_schema(schema) is True


@pytest.mark.unit
@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_is_system_schema_false(schema: str) -> None:
    assert is_system_schema(schema) is False


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "", 0])
def test_is_system_schema_handles_empty_input(value: Any) -> None:
    assert is_system_schema(value) is False


@pytest.mark.unit
def test_h001_drops_system_schema_rows(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *a, **k: ["maindb"])
    monkeypatch.setattr(
        generator,
        "get_index_definitions_from_sink",
        lambda db: {"idx_invalid": "CREATE INDEX idx_invalid ON public.tbl (col)"},
    )

    catalog_metric = {
        "schema_name": "pg_catalog",
        "table_name": "pg_class",
        "index_name": "pg_class_tblspc_relfilenode_index",
        "relation_name": "pg_catalog.pg_class",
    }
    user_metric = {
        "schema_name": "public",
        "table_name": "tbl",
        "index_name": "idx_invalid",
        "relation_name": "public.tbl",
    }

    responses = {
        "pgwatch_pg_invalid_indexes_index_size_bytes": prom_result(
            [
                {"metric": catalog_metric, "value": [0, "1000000"]},
                {"metric": user_metric, "value": [0, "2048"]},
            ]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub(prom_result, responses))

    payload = generator.generate_h001_invalid_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["maindb"]

    assert [i["schema_name"] for i in db_data["invalid_indexes"]] == ["public"]
    # Totals are recomputed from the surviving rows, not from the raw scrape.
    assert db_data["total_count"] == 1
    assert db_data["total_size_bytes"] == 2048.0
    assert "pg_class_tblspc_relfilenode_index" not in str(payload)


@pytest.mark.unit
def test_h002_drops_system_schema_rows(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *a, **k: ["app"])
    monkeypatch.setattr(
        generator,
        "get_index_definitions_from_sink",
        lambda db: {"idx_unused": "CREATE INDEX idx_unused ON public.tbl (col)"},
    )

    def _row(schema: str, table: str, index: str) -> Dict[str, Any]:
        return {
            "metric": {
                "schema_name": schema,
                "table_name": table,
                "index_name": index,
                "reason": "Never Used Indexes",
                "idx_is_btree": "true",
                "supports_fk": "0",
            },
            "value": [0, "1024"],
        }

    responses = {
        "pgwatch_db_stats_postmaster_uptime_s": prom_result([{"value": [0, "3600"]}]),
        "pgwatch_stats_reset_stats_reset_epoch": prom_result([{"value": [0, "1700000000"]}]),
        "pgwatch_unused_indexes_index_size_bytes": prom_result(
            [
                _row("pg_catalog", "pg_class", "pg_class_tblspc_relfilenode_index"),
                _row("public", "tbl", "idx_unused"),
            ]
        ),
        "pgwatch_unused_indexes_idx_scan": prom_result([{"value": [0, "0"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub(prom_result, responses))

    payload = generator.generate_h002_unused_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["app"]

    assert [i["schema_name"] for i in db_data["unused_indexes"]] == ["public"]
    assert db_data["total_count"] == 1
    assert db_data["total_size_bytes"] == 1024.0
    assert "pg_class_tblspc_relfilenode_index" not in str(payload)


@pytest.mark.unit
def test_h004_drops_system_schema_rows(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *a, **k: ["app"])
    monkeypatch.setattr(
        generator,
        "get_index_definitions_from_sink",
        lambda db: {"idx_dup": "CREATE INDEX idx_dup ON public.tbl (col)"},
    )

    def _row(schema: str, table: str, index: str) -> Dict[str, Any]:
        return {
            "metric": {
                "schema_name": schema,
                "table_name": table,
                "index_name": index,
                "relation_name": f"{schema}.{table}",
                "access_method": "btree",
                "reason": "covers columns",
            },
            "value": [0, "4096"],
        }

    responses = {
        "pgwatch_redundant_indexes_index_size_bytes": prom_result(
            [
                _row("pg_catalog", "pg_depend", "pg_depend_reference_index"),
                _row("public", "tbl", "idx_dup"),
            ]
        ),
        "pgwatch_redundant_indexes_table_size_bytes": prom_result([{"value": [0, "8192"]}]),
        "pgwatch_redundant_indexes_index_usage": prom_result([{"value": [0, "2"]}]),
        "pgwatch_redundant_indexes_supports_fk": prom_result([{"value": [0, "1"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub(prom_result, responses))

    payload = generator.generate_h004_redundant_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["app"]

    assert [i["schema_name"] for i in db_data["redundant_indexes"]] == ["public"]
    assert db_data["total_count"] == 1
    assert db_data["total_size_bytes"] == 4096.0
    assert "pg_depend_reference_index" not in str(payload)


@pytest.mark.unit
@pytest.mark.parametrize("schema", SYSTEM_SCHEMAS)
def test_h002_drops_every_system_schema_variant(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
    schema: str,
) -> None:
    """A database whose only unused index is in a system schema is skipped entirely."""
    monkeypatch.setattr(generator, "get_all_databases", lambda *a, **k: ["app"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {})

    responses = {
        "pgwatch_db_stats_postmaster_uptime_s": prom_result([{"value": [0, "3600"]}]),
        "pgwatch_stats_reset_stats_reset_epoch": prom_result([{"value": [0, "1700000000"]}]),
        "pgwatch_unused_indexes_index_size_bytes": prom_result(
            [
                {
                    "metric": {
                        "schema_name": schema,
                        "table_name": "t",
                        "index_name": "some_idx",
                        "reason": "Never Used Indexes",
                        "idx_is_btree": "true",
                        "supports_fk": "0",
                    },
                    "value": [0, "1024"],
                }
            ]
        ),
        "pgwatch_unused_indexes_idx_scan": prom_result([{"value": [0, "0"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub(prom_result, responses))

    payload = generator.generate_h002_unused_indexes_report("local", "node-1")
    assert payload["results"]["node-1"]["data"] == {}
