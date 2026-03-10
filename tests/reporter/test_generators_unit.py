import json
import sys
from datetime import datetime, timedelta
from typing import Any, Callable

import pytest

from reporter import postgres_reports as postgres_reports_module
from reporter.postgres_reports import PostgresReportGenerator


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


def _success_metric(value: str) -> dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "value": [datetime.now().timestamp(), value],
                }
            ]
        },
    }


def _query_stub_factory(prom_result, mapping: dict[str, Any]) -> Callable[[str], dict[str, Any]]:
    """Return a query_instant stub that matches substrings defined in mapping keys.
    
    Args:
        prom_result: Fallback callable that returns a default Prometheus response
        mapping: Dict mapping query substrings to responses (either dict or callable)
        
    Returns:
        A callable that takes a query string and returns a Prometheus-like response
    """

    def _fake(query: str) -> dict[str, Any]:
        for needle, payload in mapping.items():
            if needle in query:
                return payload(query) if callable(payload) else payload
        return prom_result()

    return _fake


@pytest.mark.unit
def test_query_instant_hits_prometheus(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    captured: dict[str, Any] = {}

    class DummyResponse:
        status_code = 200
        text = "{}"

        @staticmethod
        def json() -> dict[str, Any]:
            return {"status": "success", "data": {"result": []}}

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ):
        captured["url"] = url
        captured["params"] = params
        return DummyResponse()

    monkeypatch.setattr(postgres_reports_module.requests, "get", fake_get)

    payload = generator.query_instant("up")

    assert payload["status"] == "success"
    assert captured["url"].endswith("/api/v1/query")
    assert captured["params"] == {"query": "up"}


@pytest.mark.unit
def test_query_range_hits_prometheus(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=5)
    captured: dict[str, Any] = {}

    class DummyResponse:
        status_code = 200
        text = "{}"

        @staticmethod
        def json() -> dict[str, Any]:
            return {"status": "success", "data": {"result": []}}

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ):
        captured["url"] = url
        captured["params"] = params
        return DummyResponse()

    monkeypatch.setattr(postgres_reports_module.requests, "get", fake_get)

    payload = generator.query_range("up", start, end, step="60s")

    assert payload == []
    assert captured["url"].endswith("/api/v1/query_range")
    assert captured["params"]["query"] == "up"
    assert captured["params"]["start"] == start.timestamp()


@pytest.mark.unit
def test_generate_a002_version_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    values = {
        "server_version": "15.3",
        "server_version_num": "150003",
        "max_connections": "200",
        "shared_buffers": "1024",
        "effective_cache_size": "2048",
    }

    def fake_query(query: str) -> dict[str, Any]:
        # A002 uses a helper that queries both settings via a single regex selector.
        if 'setting_name=~"server_version|server_version_num"' in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {
                            "metric": {
                                "setting_name": "server_version",
                                "setting_value": values["server_version"],
                            }
                        },
                        {
                            "metric": {
                                "setting_name": "server_version_num",
                                "setting_value": values["server_version_num"],
                            }
                        },
                    ]
                },
            }
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(generator, "query_instant", fake_query)

    report = generator.generate_a002_version_report("local", "node-1")
    version = report["results"]["node-1"]["data"]["version"]

    assert version["version"] == "15.3"
    assert version["server_major_ver"] == "15"
    assert version["server_minor_ver"] == "3"


@pytest.mark.unit
def test_generate_a004_cluster_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
) -> None:
    def fake_query(query: str) -> dict[str, Any]:
        if "pgwatch_db_size_size_b" in query and "sum(" not in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"datname": "db1"}, "value": [0, "1024"]},
                        {"metric": {"datname": "db2"}, "value": [0, "2048"]},
                    ]
                },
            }
        return _success_metric("42")

    monkeypatch.setattr(generator, "query_instant", fake_query)

    report = generator.generate_a004_cluster_report("local", "node-1")
    data = report["results"]["node-1"]["data"]

    assert "general_info" in data and "database_sizes" in data
    assert data["general_info"]["active_connections"]["value"] == "42"
    assert data["database_sizes"] == {"db1": 1024.0, "db2": 2048.0}


@pytest.mark.unit
def test_prometheus_to_dict_and_process_pgss(generator: PostgresReportGenerator) -> None:
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    later_time = base_time + timedelta(seconds=60)

    def make_metric(name: str, value: float, ts: datetime) -> dict[str, Any]:
        return {
            "metric": {
                "__name__": name,
                "datname": "db1",
                "queryid": "123",
                "user": "postgres",
                "instance": "inst1",
            },
            "values": [[ts.timestamp(), str(value)]],
        }

    start_metrics = [
        make_metric("pgwatch_pg_stat_statements_calls", 10, base_time),
        make_metric("pgwatch_pg_stat_statements_exec_time_total", 1000, base_time),
        make_metric("pgwatch_pg_stat_statements_rows", 200, base_time),
    ]
    end_metrics = [
        make_metric("pgwatch_pg_stat_statements_calls", 40, later_time),
        make_metric("pgwatch_pg_stat_statements_exec_time_total", 4000, later_time),
        make_metric("pgwatch_pg_stat_statements_rows", 260, later_time),
    ]

    mapping = {
        "calls": "calls",
        "exec_time_total": "total_time",
        "rows": "rows",
    }

    rows = generator._process_pgss_data(
        start_metrics,
        end_metrics,
        base_time,
        later_time,
        mapping,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["calls"] == 30
    assert row["total_time"] == 3000
    assert pytest.approx(row["total_time_per_sec"], 0.01) == 50
    assert row["rows_per_call"] == pytest.approx(2.0)


@pytest.mark.unit
def test_prometheus_to_dict_closest_value(generator: PostgresReportGenerator) -> None:
    reference_time = datetime(2024, 1, 1, 12, 0, 0)

    prom_data: list[dict[str, Any]] = [
        {
            "metric": {
                "__name__": "pgwatch_pg_stat_statements_calls",
                "datname": "db1",
                "queryid": "q1",
                "user": "postgres",
                "instance": "inst1",
            },
            "values": [
                [reference_time.timestamp() - 10, "10"],
                [reference_time.timestamp() + 5, "20"],
            ],
        }
    ]

    converted = generator._prometheus_to_dict(prom_data, reference_time)

    key = ("db1", "q1", "postgres", "inst1")
    assert key in converted
    assert converted[key]["calls"] == 20


@pytest.mark.unit
def test_generate_a003_settings_report(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    def fake_query(query: str) -> dict[str, Any]:
        assert "pgwatch_settings_configured" in query
        return {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {
                            "setting_name": "shared_buffers",
                            "setting_value": "128",
                            "category": "Memory",
                            "unit": "8kB",
                            "context": "postmaster",
                            "vartype": "integer",
                        }
                    },
                    {
                        "metric": {
                            "setting_name": "work_mem",
                            "setting_value": "512",
                            "category": "Memory",
                            "unit": "",
                            "context": "user",
                            "vartype": "integer",
                        }
                    },
                ]
            },
        }

    monkeypatch.setattr(generator, "query_instant", fake_query)

    report = generator.generate_a003_settings_report("local", "node-1")
    data = report["results"]["node-1"]["data"]

    assert data["shared_buffers"]["pretty_value"] == "1 MiB"
    assert data["work_mem"]["unit"] == ""
    assert data["work_mem"]["category"] == "Memory"


@pytest.mark.unit
def test_generate_a007_altered_settings_report(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    def fake_query(query: str) -> dict[str, Any]:
        # Handle version info query from _get_postgres_version_info
        if 'setting_name=~"server_version|server_version_num"' in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"setting_name": "server_version", "setting_value": "15.0"}},
                        {"metric": {"setting_name": "server_version_num", "setting_value": "150000"}},
                    ]
                },
            }
        # Handle altered settings query
        assert "pgwatch_settings_is_default" in query
        return {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {
                            "setting_name": "work_mem",
                            "setting_value": "1024",
                            "unit": "",
                            "category": "Memory",
                        }
                    },
                    {
                        "metric": {
                            "setting_name": "autovacuum",
                            "setting_value": "off",
                            "unit": "",
                            "category": "Autovacuum",
                        }
                    },
                ]
            },
        }

    monkeypatch.setattr(generator, "query_instant", fake_query)

    payload = generator.generate_a007_altered_settings_report("local", "node-1")
    data = payload["results"]["node-1"]["data"]

    assert set(data.keys()) == {"work_mem", "autovacuum"}
    assert "postgres_version" in payload["results"]["node-1"]  # postgres_version is at node level
    assert data["work_mem"]["pretty_value"] == "1 MiB"
    assert data["autovacuum"]["pretty_value"] == "off"


@pytest.mark.unit
def test_get_all_databases_merges_sources(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    def fake_query(query: str) -> dict[str, Any]:
        if "wraparound" in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"datname": "appdb"}, "value": [0, "1"]},
                        {"metric": {"datname": "template0"}, "value": [0, "1"]},  # excluded
                    ]
                },
            }
        if "unused_indexes" in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        # Reporter expects `datname` label for unused indexes metric.
                        {"metric": {"datname": "analytics"}, "value": [0, "1"]},
                        {"metric": {"datname": "appdb"}, "value": [0, "1"]},  # duplicate
                    ]
                },
            }
        if "redundant_indexes" in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"dbname": "warehouse"}, "value": [0, "1"]},
                    ]
                },
            }
        if "pg_btree_bloat_bloat_pct" in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"datname": "inventory"}, "value": [0, "1"]},
                    ]
                },
            }
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(generator, "query_instant", fake_query)

    databases = generator.get_all_databases("local", "node-1")

    assert databases == ["appdb", "analytics", "warehouse", "inventory"]


@pytest.mark.unit
def test_check_pg_stat_kcache_status(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, prom_result) -> None:
    responses = {
        "pgwatch_pg_stat_kcache_exec_total_time": prom_result(
            [
                {
                    "metric": {"queryid": "1", "tag_user": "postgres"},
                    "value": [0, "10"],
                }
            ]
        ),
        "pgwatch_pg_stat_kcache_exec_user_time": prom_result([{"metric": {}, "value": [0, "4"]}]),
        "pgwatch_pg_stat_kcache_exec_system_time": prom_result([{"metric": {}, "value": [0, "6"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    status = generator._check_pg_stat_kcache_status("local", "node-1")

    assert status["extension_available"] is True
    assert status["metrics_count"] == 1
    assert status["total_exec_time"] == 10.0
    assert status["total_user_time"] == 4.0
    assert status["sample_queries"][0]["queryid"] == "1"


@pytest.mark.unit
def test_check_pg_stat_statements_status(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, prom_result) -> None:
    response = prom_result(
        [
            {
                "metric": {"queryid": "1", "tag_user": "postgres", "datname": "db1"},
                "value": [0, "5"],
            }
        ]
    )
    monkeypatch.setattr(generator, "query_instant", lambda query: response)

    status = generator._check_pg_stat_statements_status("local", "node-1")

    assert status["extension_available"] is True
    assert status["metrics_count"] == 1
    assert status["total_calls"] == 5.0
    assert status["sample_queries"][0]["database"] == "db1"


@pytest.mark.unit
def test_generate_h001_invalid_indexes_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["maindb"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_invalid": "CREATE INDEX idx_invalid ON public.tbl USING btree (col)"})

    # H001 now queries multiple metrics and merges them by (schema_name, table_name, index_name)
    base_metric = {
        "schema_name": "public",
        "table_name": "tbl",
        "index_name": "idx_invalid",
        "relation_name": "public.tbl",
        "valid_index_name": "idx_valid_dup",
        "valid_index_definition": "CREATE INDEX idx_valid_dup ON public.tbl USING btree (col)",
    }
    responses = {
        "pgwatch_pg_invalid_indexes_index_size_bytes": prom_result(
            [{"metric": base_metric, "value": [0, "2048"]}]
        ),
        "pgwatch_pg_invalid_indexes_supports_fk": prom_result(
            [{"metric": base_metric, "value": [0, "1"]}]
        ),
        "pgwatch_pg_invalid_indexes_is_pk": prom_result(
            [{"metric": base_metric, "value": [0, "0"]}]
        ),
        "pgwatch_pg_invalid_indexes_is_unique": prom_result(
            [{"metric": base_metric, "value": [0, "0"]}]
        ),
        "pgwatch_pg_invalid_indexes_has_valid_duplicate": prom_result(
            [{"metric": base_metric, "value": [0, "1"]}]
        ),
        "pgwatch_pg_invalid_indexes_table_row_estimate": prom_result(
            [{"metric": base_metric, "value": [0, "1000"]}]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_h001_invalid_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["maindb"]

    assert db_data["total_count"] == 1
    assert db_data["total_size_bytes"] == 2048.0
    entry = db_data["invalid_indexes"][0]
    assert entry["index_name"] == "idx_invalid"
    assert entry["index_size_pretty"].endswith("KiB")
    assert entry["index_definition"].startswith("CREATE INDEX")
    # Decision tree fields - verify all are present and correctly parsed
    assert entry["supports_fk"] is True
    assert entry["is_pk"] is False
    assert entry["is_unique"] is False
    assert entry["has_valid_duplicate"] is True
    assert entry["valid_duplicate_name"] == "idx_valid_dup"
    assert entry["valid_duplicate_definition"] == "CREATE INDEX idx_valid_dup ON public.tbl USING btree (col)"
    assert entry["table_row_estimate"] == 1000
    # constraint_name should be null when not backing a constraint
    assert entry["constraint_name"] is None


@pytest.mark.unit
def test_generate_h002_unused_indexes_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["app"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_unused": "CREATE INDEX idx_unused ON t(c)"})

    responses = {
        "pgwatch_db_stats_postmaster_uptime_s": prom_result([{"value": [0, "3600"]}]),
        "pgwatch_stats_reset_stats_reset_epoch": prom_result([{"value": [0, "1700000000"]}]),
        "pgwatch_unused_indexes_index_size_bytes": prom_result(
            [
                {
                    "metric": {
                        "schema_name": "public",
                        "table_name": "tbl",
                        "index_name": "idx_unused",
                        "reason": "never scanned",
                        "idx_is_btree": "true",
                        "supports_fk": "0",
                    },
                    "value": [0, "1024"],
                }
            ]
        ),
        "pgwatch_unused_indexes_idx_scan": prom_result([{"value": [0, "0"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_h002_unused_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["app"]

    assert db_data["total_count"] == 1
    unused = db_data["unused_indexes"][0]
    assert unused["index_definition"].startswith("CREATE INDEX")
    assert unused["idx_scan"] == 0
    assert unused["index_size_pretty"].endswith("KiB")
    stats_reset = db_data["stats_reset"]
    assert stats_reset["stats_reset_epoch"] == 1700000000.0
    assert stats_reset["postmaster_startup_epoch"] is not None


@pytest.mark.unit
def test_generate_h004_redundant_indexes_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["app"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_dup": "CREATE INDEX idx_dup ON t(c)"})

    responses = {
        "pgwatch_redundant_indexes_index_size_bytes": prom_result(
            [
                {
                    "metric": {
                        "schema_name": "public",
                        "table_name": "tbl",
                        "index_name": "idx_dup",
                        "relation_name": "public.tbl",
                        "access_method": "btree",
                        "reason": "covers columns",
                    },
                    "value": [0, "4096"],
                }
            ]
        ),
        "pgwatch_redundant_indexes_table_size_bytes": prom_result([{"value": [0, "8192"]}]),
        "pgwatch_redundant_indexes_index_usage": prom_result([{"value": [0, "2"]}]),
        "pgwatch_redundant_indexes_supports_fk": prom_result([{"value": [0, "1"]}]),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_h004_redundant_indexes_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["app"]

    assert db_data["total_count"] == 1
    redundant = db_data["redundant_indexes"][0]
    assert redundant["index_definition"].startswith("CREATE INDEX")
    assert redundant["index_usage"] == 2.0
    assert redundant["index_size_pretty"].endswith("KiB")
    assert redundant["supports_fk"] is True


@pytest.mark.unit
def test_generate_d004_pgstat_settings_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    responses = {
        "pgwatch_settings_configured": prom_result(
            [
                {
                    "metric": {
                        "setting_name": "pg_stat_statements.max",
                        "setting_value": "1000",
                        "category": "Stats",
                        "unit": "",
                        "context": "postmaster",
                        "vartype": "integer",
                    }
                }
            ]
        )
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))
    monkeypatch.setattr(generator, "_check_pg_stat_kcache_status", lambda *args, **kwargs: {"extension_available": True})
    monkeypatch.setattr(generator, "_check_pg_stat_statements_status", lambda *args, **kwargs: {"extension_available": False})

    payload = generator.generate_d004_pgstat_settings_report("local", "node-1")
    data = payload["results"]["node-1"]["data"]

    assert "pg_stat_statements.max" in data["settings"]
    assert data["pg_stat_kcache_status"]["extension_available"] is True


@pytest.mark.unit
def test_generate_f001_autovacuum_settings_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    responses = {
        "pgwatch_settings_configured": prom_result(
            [
                {
                    "metric": {
                        "setting_name": "autovacuum_naptime",
                        "setting_value": "60",
                        "category": "Autovacuum",
                        "unit": "",
                        "context": "sighup",
                        "vartype": "integer",
                    }
                }
            ]
        )
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_f001_autovacuum_settings_report("local", "node-1")
    data = payload["results"]["node-1"]["data"]

    assert data["autovacuum_naptime"]["setting"] == "60"
    assert data["autovacuum_naptime"]["pretty_value"] == "1 min"


@pytest.mark.unit
def test_generate_f005_btree_bloat_report(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    responses = {
        "pgwatch_pg_stat_all_tables_last_vacuum": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "relname": "t"},
                    "value": [0, "1700000000"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_real_size_mib": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "2"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_table_size_mib": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "10"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_extra_size": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "1024"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_extra_pct": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "20"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_fillfactor": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "90"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_bloat_size": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "2048"],
                }
            ]
        ),
        "pgwatch_pg_btree_bloat_bloat_pct": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"},
                    "value": [0, "50"],
                }
            ]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_f005_btree_bloat_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["db1"]
    entry = db_data["bloated_indexes"][0]

    assert entry["real_size"] == 2 * 1024 * 1024
    assert entry["real_size_pretty"] == "2.00 MiB"
    assert entry["table_size"] == 10 * 1024 * 1024
    assert entry["table_size_pretty"] == "10.0 MiB"
    # Prometheus provides *_mib metrics, but the report output should expose bytes-only fields.
    assert "real_size_mib" not in entry
    assert "table_size_mib" not in entry
    assert entry["extra_size"] == 1024.0
    assert entry["bloat_pct"] == 50.0
    assert entry["fillfactor"] == 90.0
    assert entry["last_vacuum_epoch"] == 1700000000.0
    assert entry["last_vacuum"] == "2023-11-14T22:13:20+00:00"
    assert entry["bloat_size_pretty"].endswith("KiB")


@pytest.mark.unit
def test_generate_f004_heap_bloat_report_real_size_uses_real_size_mib(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    responses = {
        "pgwatch_db_size_size_b": prom_result(
            [
                {
                    "metric": {"datname": "db1"},
                    "value": [0, "1048576"],
                }
            ]
        ),
        "pgwatch_pg_stat_all_tables_last_vacuum": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "relname": "t"},
                    "value": [0, "1700000000"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_real_size_mib": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "128"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_extra_size": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "1024"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_extra_pct": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "10"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_fillfactor": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "100"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_bloat_size": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "2048"],
                }
            ]
        ),
        "pgwatch_pg_table_bloat_bloat_pct": prom_result(
            [
                {
                    "metric": {"schemaname": "public", "tblname": "t"},
                    "value": [0, "20"],
                }
            ]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    payload = generator.generate_f004_heap_bloat_report("local", "node-1")
    db_data = payload["results"]["node-1"]["data"]["db1"]
    entry = db_data["bloated_tables"][0]

    # Prometheus provides real_size_mib, but the report should expose real_size in bytes.
    assert entry["real_size"] == 128 * 1024 * 1024
    assert entry["real_size_pretty"] == "128 MiB"
    assert entry["fillfactor"] == 100.0
    assert entry["last_vacuum_epoch"] == 1700000000.0
    assert entry["last_vacuum"] == "2023-11-14T22:13:20+00:00"
    assert "real_size_mib" not in entry
    assert "real_size_bytes" not in entry


@pytest.mark.unit
def test_get_pgss_metrics_data_by_db_invokes_all_metrics(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    captured: list[str] = []

    def fake_query_range(query: str, start, end, step: str = "30s") -> list[dict]:
        captured.append(query)
        return []

    monkeypatch.setattr(generator, "query_range", fake_query_range)
    sentinel = [{"result": "ok"}]
    monkeypatch.setattr(generator, "_process_pgss_data", lambda *args, **kwargs: sentinel)

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=1)
    result = generator._get_pgss_metrics_data_by_db("local", "node-1", "db1", start, end)

    assert result == sentinel
    # Ensure at least one representative metric was queried with filters
    assert any("pgwatch_pg_stat_statements_calls" in q for q in captured)


@pytest.mark.unit
def test_generate_all_reports_invokes_every_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = PostgresReportGenerator()
    called: list[str] = []

    def stub(name: str):
        def _(*args, **kwargs):
            called.append(name)
            return {name: True}

        return _

    # Independent builders (not derived from A003)
    independent_builders = [
        "generate_a002_version_report",
        "generate_a003_settings_report",
        "generate_a004_cluster_report",
        "generate_a007_altered_settings_report",
        "generate_f004_heap_bloat_report",
        "generate_f005_btree_bloat_report",
        "generate_h001_invalid_indexes_report",
        "generate_h002_unused_indexes_report",
        "generate_h004_redundant_indexes_report",
        "generate_k001_query_calls_report",
        "generate_k003_top_queries_report",
        "generate_k004_temp_bytes_report",
        "generate_k005_wal_bytes_report",
        "generate_k006_shared_read_report",
        "generate_k007_shared_hit_report",
        "generate_k008_shared_hit_read_report",
        "generate_m001_mean_time_report",
        "generate_m002_rows_report",
        "generate_m003_io_time_report",
        "generate_n001_wait_events_report",
    ]

    # Builders derived from A003
    a003_derived_builders = [
        "generate_d004_from_a003",
        "generate_f001_from_a003",
        "generate_g001_from_a003",
        # S001 is not implemented yet
    ]

    for name in independent_builders:
        monkeypatch.setattr(generator, name, stub(name))

    for name in a003_derived_builders:
        monkeypatch.setattr(generator, name, stub(name))

    reports = generator.generate_all_reports("local", "node-1")

    # All report types should be generated
    expected_report_codes = {
        'A002', 'A003', 'A004', 'A007',
        'D004', 'F001', 'F004', 'F005', 'G001',
        'H001', 'H002', 'H004',
        'K001', 'K003', 'K004', 'K005', 'K006', 'K007', 'K008',
        'M001', 'M002', 'M003',
        'N001',
        # S001 is not implemented yet
    }
    assert set(reports.keys()) == expected_report_codes

    # All builders should be called
    all_builders = independent_builders + a003_derived_builders
    assert set(called) == set(all_builders)


@pytest.mark.unit
def test_create_report_uses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = PostgresReportGenerator()
    payloads: list[dict] = []

    def fake_make_request(api_url, endpoint, request_data):
        payloads.append({"endpoint": endpoint, "data": request_data})
        return {"report_id": 42}

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    report_id = generator.create_report("https://api", "tok", "proj", "123")

    assert report_id == 42
    assert payloads[0]["endpoint"] == "/rpc/checkup_report_create"
    assert payloads[0]["data"]["project"] == "proj"


@pytest.mark.unit
def test_upload_report_file_sends_contents(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = PostgresReportGenerator()
    captured: dict = {}

    def fake_make_request(api_url, endpoint, request_data):
        captured["endpoint"] = endpoint
        captured["data"] = request_data
        return {}

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    report_file = tmp_path / "A002_report.json"
    # check_id is derived from JSON payload (not filename).
    report_file.write_text('{"checkId": "A002", "foo": "bar"}', encoding="utf-8")

    generator.upload_report_file("https://api", "tok", 100, str(report_file))

    assert captured["endpoint"] == "/rpc/checkup_report_file_post"
    assert captured["data"]["check_id"] == "A002"
    assert captured["data"]["filename"] == report_file.name


@pytest.mark.unit
def test_upload_report_file_handles_404_gracefully(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    generator = PostgresReportGenerator()
    
    def fake_make_request(api_url, endpoint, request_data):
        import requests
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError(response=response)

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    report_file = tmp_path / "A002_report.json"
    report_file.write_text('{"foo": "bar"}', encoding="utf-8")

    # Should not raise exception
    generator.upload_report_file("https://api", "tok", 100, str(report_file))
    
    captured = capsys.readouterr()
    assert "Upload endpoint not available (404)" in captured.out
    assert "--no-upload" in captured.out


@pytest.mark.unit
def test_create_report_handles_404_gracefully(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    generator = PostgresReportGenerator()
    
    def fake_make_request(api_url, endpoint, request_data):
        import requests
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError(response=response)

    monkeypatch.setattr(postgres_reports_module, "make_request", fake_make_request)

    # Should not raise exception, should return None
    report_id = generator.create_report("https://api", "tok", "proj", "123")
    
    assert report_id is None
    captured = capsys.readouterr()
    assert "API endpoint not available (404)" in captured.out


@pytest.mark.unit
def test_main_runs_specific_check_without_upload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class DummyGenerator:
        DEFAULT_EXCLUDED_DATABASES = {'template0', 'template1', 'rdsadmin', 'azure_maintenance', 'cloudsqladmin'}

        def __init__(self, *args, **kwargs):
            self.closed = False
            self.pg_conn = None  # Add pg_conn attribute for memory cleanup check

        def get_all_clusters(self):
            # Match current reporter.main() behavior which always calls
            # get_all_clusters() when cluster is not explicitly provided.
            return ["local"]

        def test_connection(self) -> bool:
            return True

        def generate_a002_version_report(self, cluster, node_name):
            return {"checkId": "A002", "results": {node_name: {"data": {"ok": True}}}}

        def close_postgres_sink(self):
            self.closed = True
            self.pg_conn = None

    monkeypatch.setattr(postgres_reports_module, "PostgresReportGenerator", DummyGenerator)
    monkeypatch.setattr(sys, "argv", ["postgres_reports.py", "--check-id", "A002", "--output", "-", "--no-upload"])

    postgres_reports_module.main()

    captured = capsys.readouterr().out

    # main() prints progress banners along with the JSON payload.
    # Extract the JSON object from the captured stdout by finding the
    # first line that looks like JSON and ending before any trailing messages.
    lines = captured.splitlines()
    start_idx = 0
    end_idx = len(lines)
    
    # Find start of JSON
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            start_idx = i
            break
    
    # Find end of JSON (stop at first non-JSON line after JSON starts)
    brace_count = 0
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        brace_count += line.count("{") - line.count("}")
        if brace_count == 0 and line.endswith("}"):
            end_idx = i + 1
            break
    
    json_str = "\n".join(lines[start_idx:end_idx])

    output = json.loads(json_str)
    assert output["checkId"] == "A002"
    assert "results" in output


@pytest.mark.unit
def test_main_all_reports_does_not_crash_when_output_is_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression test for reporter/postgres_reports.py around the `del reports` block
    (~4308-4324).

    In ALL-reports mode, providing a normal file path via --output should NOT
    cause the process to crash. Current code crashes because it does `del reports`
    and then later references `reports` when handling args.output.
    """
    class DummyGenerator:
        DEFAULT_EXCLUDED_DATABASES = {'template0', 'template1', 'rdsadmin', 'azure_maintenance', 'cloudsqladmin'}

        def __init__(self, *args, **kwargs):
            self.pg_conn = None

        def test_connection(self) -> bool:
            return True

        def get_all_clusters(self):
            return ["local"]

        def generate_all_reports(self, cluster, node_name, combine_nodes=True):
            # Minimal plausible payload
            return {
                "A002": {"checkId": "A002", "results": {"node-1": {"data": {"ok": True}}}},
                "A003": {"checkId": "A003", "results": {"node-1": {"data": {"ok": True}}}},
            }

        def generate_per_query_jsons(self, *args, **kwargs):
            return []

        def close_postgres_sink(self):
            self.pg_conn = None

    monkeypatch.setattr(postgres_reports_module, "PostgresReportGenerator", DummyGenerator)
    monkeypatch.chdir(tmp_path)

    out_path = tmp_path / "all_reports.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postgres_reports.py",
            "--check-id",
            "ALL",
            "--cluster",
            "local",
            "--output",
            str(out_path),
            "--no-upload",
        ],
    )

    postgres_reports_module.main()


@pytest.mark.unit
def test_main_exits_when_connection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingGenerator:
        DEFAULT_EXCLUDED_DATABASES = {'template0', 'template1', 'rdsadmin', 'azure_maintenance', 'cloudsqladmin'}

        def __init__(self, *args, **kwargs):
            pass

        def test_connection(self) -> bool:
            return False

    monkeypatch.setattr(postgres_reports_module, "PostgresReportGenerator", FailingGenerator)
    monkeypatch.setattr(sys, "argv", ["postgres_reports.py", "--check-id", "A002"])

    with pytest.raises(SystemExit):
        postgres_reports_module.main()


# ============================================================================
# Negative test cases - Error handling
# ============================================================================


@pytest.mark.unit
def test_query_instant_handles_http_404_error(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_instant returns empty dict on HTTP 404 error."""
    class MockResponse:
        status_code = 404
        text = "Not Found"

        def json(self):
            return {"error": "not found"}

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    result = generator.query_instant("test_query")

    assert result == {}


@pytest.mark.unit
def test_query_instant_handles_http_500_error(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_instant returns empty dict on HTTP 500 error."""
    class MockResponse:
        status_code = 500
        text = "Internal Server Error"

        def json(self):
            raise ValueError("Invalid JSON")

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    result = generator.query_instant("test_query")

    assert result == {}


@pytest.mark.unit
def test_query_instant_handles_timeout(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_instant returns empty dict on request timeout."""
    import requests

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr("requests.get", fake_get)

    result = generator.query_instant("test_query")

    assert result == {}


@pytest.mark.unit
def test_query_instant_handles_connection_error(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_instant returns empty dict on connection error."""
    import requests

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        raise requests.ConnectionError("Failed to establish connection")

    monkeypatch.setattr("requests.get", fake_get)

    result = generator.query_instant("test_query")

    assert result == {}


@pytest.mark.unit
def test_query_instant_handles_malformed_json(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_instant returns empty dict when response has invalid JSON."""
    class MockResponse:
        status_code = 200

        def json(self):
            raise ValueError("Invalid JSON")

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    result = generator.query_instant("test_query")

    assert result == {}


@pytest.mark.unit
def test_query_range_handles_http_error(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_range returns empty list on HTTP error."""
    class MockResponse:
        status_code = 503
        text = "Service Unavailable"

        def json(self):
            return {"error": "service unavailable"}

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    start = datetime.now()
    end = start + timedelta(hours=1)
    result = generator.query_range("test_query", start, end)

    assert result == []


@pytest.mark.unit
def test_query_range_handles_timeout(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_range returns empty list on timeout."""
    import requests

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr("requests.get", fake_get)

    start = datetime.now()
    end = start + timedelta(hours=1)
    result = generator.query_range("test_query", start, end)

    assert result == []


@pytest.mark.unit
def test_query_range_handles_malformed_response(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_range handles response with missing expected fields."""
    class MockResponse:
        status_code = 200

        def json(self):
            # Missing 'data' or 'result' fields
            return {"status": "success"}

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    start = datetime.now()
    end = start + timedelta(hours=1)
    result = generator.query_range("test_query", start, end)

    assert result == []


@pytest.mark.unit
def test_query_range_handles_failed_status(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator) -> None:
    """Test that query_range handles Prometheus error status."""
    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "status": "error",
                "errorType": "bad_data",
                "error": "invalid query"
            }

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int | None = None):
        return MockResponse()

    monkeypatch.setattr("requests.get", fake_get)

    start = datetime.now()
    end = start + timedelta(hours=1)
    result = generator.query_range("test_query", start, end)

    assert result == []


@pytest.mark.unit
def test_make_request_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that make_request raises exception on HTTP error."""
    class MockResponse:
        status_code = 400

        def raise_for_status(self):
            import requests
            raise requests.HTTPError("400 Client Error")

        def json(self):
            return {}

    def fake_post(url: str, json: dict[str, Any] | None = None, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.post", fake_post)

    import requests
    with pytest.raises(requests.HTTPError):
        postgres_reports_module.make_request("http://api.test", "/endpoint", {"data": "test"})


@pytest.mark.unit
def test_make_request_raises_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that make_request raises exception on connection error."""
    import requests

    def fake_post(url: str, json: dict[str, Any] | None = None, **kwargs):
        raise requests.ConnectionError("Connection failed")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(requests.ConnectionError):
        postgres_reports_module.make_request("http://api.test", "/endpoint", {"data": "test"})
