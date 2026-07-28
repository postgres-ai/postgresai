from __future__ import annotations

from typing import Any, Callable

import pytest

from reporter.postgres_reports import PostgresReportGenerator
from reporter.report_schemas import validate_query_file, validate_report


@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    return PostgresReportGenerator(
        prometheus_url="http://prom.test",
        postgres_sink_url="",
    )


@pytest.fixture(name="fixed_pg_version")
def fixture_fixed_pg_version() -> dict[str, str]:
    return {
        "version": "15.3",
        "server_version_num": "150003",
        "server_major_ver": "15",
        "server_minor_ver": "3",
    }


def _query_stub_factory(
    prom_result: Callable[[list[dict] | None, str], dict],
    mapping: dict[str, Any],
) -> Callable[[str], dict[str, Any]]:
    def _fake(query: str) -> dict[str, Any]:
        for needle, payload in mapping.items():
            if needle in query:
                return payload(query) if callable(payload) else payload
        return prom_result([])

    return _fake


@pytest.mark.unit
def test_schema_a002(monkeypatch: pytest.MonkeyPatch, generator: PostgresReportGenerator, fixed_pg_version) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    report = generator.generate_a002_version_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_a003(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    resp = prom_result(
        [
            {
                "metric": {
                    "setting_name": "shared_buffers",
                    "setting_value": "128",
                    "category": "Memory",
                    "unit": "8kB",
                    "context": "postmaster",
                    "vartype": "integer",
                }
            }
        ]
    )
    monkeypatch.setattr(generator, "query_instant", lambda query: resp)

    report = generator.generate_a003_settings_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_a004(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    def fake_query(query: str) -> dict[str, Any]:
        if "pgwatch_db_size_size_b" in query and "sum(" not in query:
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"datname": "db1"}, "value": [0, "1024"]},
                    ]
                },
            }
        return {"status": "success", "data": {"result": [{"value": [0, "42"]}]}}

    monkeypatch.setattr(generator, "query_instant", fake_query)
    report = generator.generate_a004_cluster_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_a007(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    resp = prom_result(
        [
            {
                "metric": {
                    "setting_name": "work_mem",
                    "setting_value": "1024",
                    "unit": "",
                    "category": "Memory",
                }
            }
        ]
    )
    monkeypatch.setattr(generator, "query_instant", lambda query: resp)

    report = generator.generate_a007_altered_settings_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_d004(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    settings_resp = prom_result(
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
    monkeypatch.setattr(generator, "query_instant", lambda query: settings_resp)
    monkeypatch.setattr(
        generator,
        "_check_pg_stat_statements_status",
        lambda *args, **kwargs: {
            "extension_available": True,
            "metrics_count": 1,
            "total_calls": 5.0,
            "sample_queries": [{"queryid": "1", "user": "postgres", "database": "db1", "calls": 5.0}],
        },
    )
    monkeypatch.setattr(
        generator,
        "_check_pg_stat_kcache_status",
        lambda *args, **kwargs: {
            "extension_available": True,
            "metrics_count": 1,
            "total_exec_time": 10.0,
            "total_user_time": 4.0,
            "total_system_time": 6.0,
            "sample_queries": [{"queryid": "1", "user": "postgres", "exec_total_time": 10.0}],
        },
    )

    report = generator.generate_d004_pgstat_settings_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_f001(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    resp = prom_result(
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
    monkeypatch.setattr(generator, "query_instant", lambda query: resp)

    report = generator.generate_f001_autovacuum_settings_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_f002(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    settings = {
        "autovacuum_freeze_max_age": 200_000_000,
        "vacuum_freeze_min_age": 50_000_000,
        "vacuum_freeze_table_age": 150_000_000,
        "autovacuum_multixact_freeze_max_age": 400_000_000,
        "vacuum_multixact_freeze_min_age": 5_000_000,
        "vacuum_multixact_freeze_table_age": 150_000_000,
        "vacuum_failsafe_age": 1_600_000_000,
        "vacuum_multixact_failsafe_age": 1_600_000_000,
    }

    def fake_query(query: str) -> dict[str, Any]:
        for name, value in settings.items():
            if f"pgwatch_pg_settings_wraparound_{name}" in query:
                return prom_result([{"value": [0, str(value)]}])
        if "pgwatch_pg_database_wraparound_age_datfrozenxid" in query:
            return prom_result([{"metric": {"datname": "db1"}, "value": [0, "250000000"]}])
        if "pgwatch_pg_database_wraparound_age_datminmxid" in query:
            return prom_result([{"metric": {"datname": "db1"}, "value": [0, "1000"]}])
        table_values = {
            "xid_age": 250_000_000, "multixact_age": 1000,
            "effective_freeze_max_age": 200_000_000,
            "effective_multixact_freeze_max_age": 400_000_000,
            "table_size_bytes": 1_048_576,
        }
        for name, value in table_values.items():
            if f"pgwatch_pg_table_wraparound_{name}" in query:
                return prom_result([{
                    "metric": {"datname": "db1", "schema_name": "public", "table_name": "events", "ranked_by": "xid"},
                    "value": [0, str(value)],
                }])
        if "pgwatch_multixact_size_members_bytes" in query:
            return prom_result([{"value": [0, "1048576"]}])
        if "pgwatch_multixact_size_offsets_bytes" in query:
            return prom_result([{"value": [0, "524288"]}])
        if "pgwatch_multixact_size_status_code" in query:
            return prom_result([{"value": [0, "0"]}])
        return prom_result([])

    monkeypatch.setattr(generator, "query_instant", fake_query)
    report = generator.generate_f002_wraparound_report("local", "node-1")
    validate_report(report)
    assert report["generation_mode"] == "full"
    assert report["results"]["node-1"]["data"]["severity"] == "warning"
    assert report["results"]["node-1"]["data"]["multixact_size"]["bytes"] == 1_572_864


@pytest.mark.unit
def test_f002_threshold_boundaries(generator: PostgresReportGenerator) -> None:
    assert generator._wraparound_risk(199_999_999, 200_000_000, 1_600_000_000)["severity"] == "info"
    assert generator._wraparound_risk(200_000_000, 200_000_000, 1_600_000_000)["severity"] == "warning"
    assert generator._wraparound_risk(400_000_000, 200_000_000, 1_600_000_000)["severity"] == "high"
    assert generator._wraparound_risk(720_000_000, 500_000_000, 900_000_000)["severity"] == "high"
    assert generator._wraparound_risk(1_000_000_000, 800_000_000, 1_600_000_000)["severity"] == "critical"
    assert generator._wraparound_risk(500_000_000, 0, None)["severity"] == "info"


@pytest.mark.unit
def test_f002_handles_missing_monitoring_data(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    prom_result,
) -> None:
    monkeypatch.setattr(
        generator,
        "_get_postgres_version_info",
        lambda *args, **kwargs: {
            "version": "Unknown", "server_version_num": "Unknown",
            "server_major_ver": "Unknown", "server_minor_ver": "Unknown",
        },
    )
    monkeypatch.setattr(generator, "query_instant", lambda query: prom_result([]))
    report = generator.generate_f002_wraparound_report("local", "node-1")
    validate_report(report)
    data = report["results"]["node-1"]["data"]
    assert data["settings_available"] is False
    assert data["severity"] == "info"
    assert "settings are unavailable" in data["conclusions"][0]
    assert data["multixact_size"] == {"bytes": None, "size_pretty": None, "status_code": 2}


@pytest.mark.unit
def test_f002_deduplicates_stale_rank_labels(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)

    def fake_query(query: str) -> dict[str, Any]:
        if "pgwatch_pg_settings_wraparound_autovacuum_freeze_max_age" in query:
            return prom_result([{"value": [0, "200000000"]}])
        if "pgwatch_pg_settings_wraparound_autovacuum_multixact_freeze_max_age" in query:
            return prom_result([{"value": [0, "400000000"]}])
        if "pgwatch_pg_table_wraparound_xid_age" in query:
            return prom_result([
                {"metric": {"datname": "db1", "schema_name": "public", "table_name": "t", "ranked_by": "xid"}, "value": [0, "100"]},
                {"metric": {"datname": "db1", "schema_name": "public", "table_name": "t", "ranked_by": "multixact"}, "value": [0, "200"]},
            ])
        return prom_result([])

    monkeypatch.setattr(generator, "query_instant", fake_query)
    data = generator.generate_f002_wraparound_report("local", "node-1")["results"]["node-1"]["data"]
    assert len(data["tables"]) == 1
    assert data["tables"][0]["ranked_by"] == ["multixact", "xid"]
    assert data["tables"][0]["xid"]["age"] == 200


@pytest.mark.unit
def test_schema_f004(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    responses = {
        "pgwatch_db_size_size_b": prom_result([{"metric": {"datname": "db1"}, "value": [0, "2048"]}]),
        "pgwatch_pg_table_bloat_real_size": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t"}, "value": [0, "4096"]}]
        ),
        "pgwatch_pg_table_bloat_extra_size": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t"}, "value": [0, "1024"]}]
        ),
        "pgwatch_pg_table_bloat_extra_pct": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t"}, "value": [0, "25"]}]
        ),
        "pgwatch_pg_table_bloat_bloat_size": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t"}, "value": [0, "512"]}]
        ),
        "pgwatch_pg_table_bloat_bloat_pct": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t"}, "value": [0, "12.5"]}]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    report = generator.generate_f004_heap_bloat_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_f005(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])

    responses = {
        "pgwatch_db_size_size_b": prom_result([{"metric": {"datname": "db1"}, "value": [0, "2048"]}]),
        "pgwatch_pg_btree_bloat_extra_size": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"}, "value": [0, "1024"]}]
        ),
        "pgwatch_pg_btree_bloat_extra_pct": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"}, "value": [0, "20"]}]
        ),
        "pgwatch_pg_btree_bloat_bloat_size": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"}, "value": [0, "2048"]}]
        ),
        "pgwatch_pg_btree_bloat_bloat_pct": prom_result(
            [{"metric": {"schemaname": "public", "tblname": "t", "idxname": "idx"}, "value": [0, "50"]}]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))

    report = generator.generate_f005_btree_bloat_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_g001(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    resp = prom_result(
        [
            {
                "metric": {
                    "setting_name": "shared_buffers",
                    "setting_value": "128MB",
                    "category": "Memory",
                    "unit": "",
                    "context": "postmaster",
                    "vartype": "integer",
                }
            },
            {
                "metric": {
                    "setting_name": "work_mem",
                    "setting_value": "4MB",
                    "category": "Memory",
                    "unit": "",
                    "context": "user",
                    "vartype": "integer",
                }
            },
            {
                "metric": {
                    "setting_name": "maintenance_work_mem",
                    "setting_value": "64MB",
                    "category": "Memory",
                    "unit": "",
                    "context": "user",
                    "vartype": "integer",
                }
            },
            {
                "metric": {
                    "setting_name": "effective_cache_size",
                    "setting_value": "4GB",
                    "category": "Memory",
                    "unit": "",
                    "context": "user",
                    "vartype": "integer",
                }
            },
            {
                "metric": {
                    "setting_name": "max_connections",
                    "setting_value": "100",
                    "category": "Connections",
                    "unit": "",
                    "context": "postmaster",
                    "vartype": "integer",
                }
            },
            {
                "metric": {
                    "setting_name": "wal_buffers",
                    "setting_value": "16MB",
                    "category": "WAL",
                    "unit": "",
                    "context": "postmaster",
                    "vartype": "integer",
                }
            }
        ]
    )
    monkeypatch.setattr(generator, "query_instant", lambda query: resp)
    report = generator.generate_g001_memory_settings_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_h001(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["maindb"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_invalid": "CREATE INDEX idx_invalid ON public.tbl USING btree (col)"})
    responses = {
        "pgwatch_db_size_size_b": prom_result([{"metric": {"datname": "maindb"}, "value": [0, "8192"]}]),
        "pgwatch_pg_invalid_indexes": prom_result(
            [
                {
                    "metric": {
                        "schema_name": "public",
                        "table_name": "tbl",
                        "index_name": "idx_invalid",
                        "relation_name": "public.tbl",
                        "supports_fk": "1",
                    },
                    "value": [0, "2048"],
                }
            ]
        ),
    }
    monkeypatch.setattr(generator, "query_instant", _query_stub_factory(prom_result, responses))
    report = generator.generate_h001_invalid_indexes_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_h002(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["app"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_unused": "CREATE INDEX idx_unused ON t(c)"})

    responses = {
        "pgwatch_db_size_size_b": prom_result([{"metric": {"datname": "app"}, "value": [0, "8192"]}]),
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

    report = generator.generate_h002_unused_indexes_report("local", "node-1")
    validate_report(report)


@pytest.mark.unit
def test_schema_h004(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
    prom_result,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["app"])
    monkeypatch.setattr(generator, "get_index_definitions_from_sink", lambda db: {"idx_dup": "CREATE INDEX idx_dup ON t(c)"})

    responses = {
        "pgwatch_db_size_size_b": prom_result([{"metric": {"datname": "app"}, "value": [0, "8192"]}]),
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

    report = generator.generate_h004_redundant_indexes_report("local", "node-1")
    validate_report(report)


def _sample_query_metric_row() -> dict[str, Any]:
    # Must match _process_pgss_data() output keys for the current mapping used in _get_pgss_metrics_data_by_db().
    return {
        "queryid": "123",
        "database": "db1",
        "user": "postgres",
        "duration_seconds": 60.0,
        "calls": 30.0,
        "calls_per_sec": 0.5,
        "calls_per_call": 1.0,
        "total_time": 3000.0,
        "total_time_per_sec": 50.0,
        "total_time_per_call": 100.0,
        "rows": 60.0,
        "rows_per_sec": 1.0,
        "rows_per_call": 2.0,
        "shared_blks_hit": 10.0,
        "shared_blks_hit_per_sec": 0.166,
        "shared_blks_hit_per_call": 0.333,
        "shared_blks_read": 0.0,
        "shared_blks_read_per_sec": 0.0,
        "shared_blks_read_per_call": 0.0,
        "shared_blks_dirtied": 0.0,
        "shared_blks_dirtied_per_sec": 0.0,
        "shared_blks_dirtied_per_call": 0.0,
        "shared_blks_written": 0.0,
        "shared_blks_written_per_sec": 0.0,
        "shared_blks_written_per_call": 0.0,
        "blk_read_time": 0.0,
        "blk_read_time_per_sec": 0.0,
        "blk_read_time_per_call": 0.0,
        "blk_write_time": 0.0,
        "blk_write_time_per_sec": 0.0,
        "blk_write_time_per_call": 0.0,
    }


@pytest.mark.unit
def test_schema_k001(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])
    monkeypatch.setattr(generator, "_get_pgss_metrics_data_by_db", lambda *args, **kwargs: [_sample_query_metric_row()])

    report = generator.generate_k001_query_calls_report("local", "node-1", time_range_minutes=60)
    validate_report(report)


@pytest.mark.unit
def test_schema_k003(
    monkeypatch: pytest.MonkeyPatch,
    generator: PostgresReportGenerator,
    fixed_pg_version,
) -> None:
    monkeypatch.setattr(generator, "_get_postgres_version_info", lambda *args, **kwargs: fixed_pg_version)
    monkeypatch.setattr(generator, "get_all_databases", lambda *args, **kwargs: ["db1"])
    monkeypatch.setattr(generator, "_get_pgss_metrics_data_by_db", lambda *args, **kwargs: [_sample_query_metric_row()])

    report = generator.generate_k003_top_queries_report("local", "node-1", time_range_minutes=60, limit=50)
    validate_report(report)


@pytest.mark.unit
def test_schema_query_file() -> None:
    payload = {
        "cluster_id": "prod",
        "query_id": "qid_1",
        "query_text": "SELECT 1",
        "nodes": {"primary": "main", "standbys": ["replica-1", "replica-2"]},
        "results": {
            "main": {
                "db1": {"metrics": {"calls": 1, "total_time": 2.5}},
            },
            "replica-1": {
                "db1": {"metrics": {"calls": 0, "total_time": 0}},
            },
        },
        "time_range": {"hours": 24, "start_time": "2025-01-01T00:00:00+00:00", "end_time": "2025-01-02T00:00:00+00:00"},
        "timestamptz": "2025-01-02T00:00:00+00:00",
    }
    validate_query_file(payload)
