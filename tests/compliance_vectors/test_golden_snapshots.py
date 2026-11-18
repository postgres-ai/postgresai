"""
Golden Snapshot Tests - Phase 2

Tests report generators using syrupy snapshots to catch unexpected output changes.
Each report has 4 test cases: happy_path, empty_metrics, partial_data, error_handling.

Run with: pytest tests/compliance_vectors/test_golden_snapshots.py -v
Update snapshots: pytest tests/compliance_vectors/test_golden_snapshots.py --snapshot-update
"""
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from syrupy import SnapshotAssertion
from syrupy.filters import props

from reporter.postgres_reports import PostgresReportGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(name="generator")
def fixture_generator() -> PostgresReportGenerator:
    """Create a generator whose queries answer nothing unless a test stubs them.

    The reports ask for more than each test stubs (the Postgres version, the
    per-query series): those get the "no data" answers, not a network call.
    """
    gen = PostgresReportGenerator(
        prometheus_url="http://prom.test:9090",
        postgres_sink_url=None,
    )
    gen.query_instant = lambda query: {}
    gen.query_range = lambda query, start_time, end_time, step="30s": []
    return gen


@pytest.fixture(name="prom_result")
def fixture_prom_result() -> Callable[[Optional[List[Dict]], str], Dict]:
    """Build a Prometheus-like payload."""
    def _builder(rows: Optional[List[Dict]] = None, status: str = "success") -> Dict:
        return {
            "status": status,
            "data": {
                "result": rows or [],
            },
        }
    return _builder


@pytest.fixture(name="prom_range_result")
def fixture_prom_range_result() -> Callable[[Optional[List[Dict]], str], Dict]:
    """Build a Prometheus range query result."""
    def _builder(rows: Optional[List[Dict]] = None, status: str = "success") -> Dict:
        return {
            "status": status,
            "data": {
                "resultType": "matrix",
                "result": rows or [],
            },
        }
    return _builder


# ============================================================================
# Sanitizer - normalize volatile fields for stable snapshots
# ============================================================================

class SnapshotSanitizer:
    """Normalize volatile identity fields only - preserve logic outputs."""

    # Fields that are truly volatile (change between runs)
    VOLATILE_IDENTITY = {
        "created_at", "generated_at", "timestamp", "request_id", "run_id",
        "report_generated_at", "data_collected_at", "timestamptz"
    }

    @staticmethod
    def sanitize(data: Any) -> Any:
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k in SnapshotSanitizer.VOLATILE_IDENTITY:
                    if "id" in k:
                        result[k] = "00000000-0000-0000-0000-000000000000"
                    else:
                        result[k] = "2026-01-01T00:00:00Z"
                else:
                    result[k] = SnapshotSanitizer.sanitize(v)
            return result
        elif isinstance(data, list):
            return [SnapshotSanitizer.sanitize(item) for item in data]
        return data


# ============================================================================
# G001 Memory Settings Report Tests
# ============================================================================

class TestG001MemorySettingsReport:
    """Golden snapshot tests for generate_g001_memory_settings_report()

    4 test cases as specified in TMP_TESTCOV.md:
    - happy_path: Normal operation with valid metrics
    - empty_metrics: Prometheus returns empty data
    - partial_data: Some settings missing
    - error_handling: Malformed response (documents legacy behavior)
    """

    def test_happy_path(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Normal operation with valid memory settings."""
        # As pg_settings has them: a bare number in the setting's unit.
        metrics = [
            {"metric": {"setting_name": "shared_buffers", "setting_value": "16384", "category": "Memory", "unit": "8kB", "context": "postmaster", "vartype": "integer"}},
            {"metric": {"setting_name": "work_mem", "setting_value": "4096", "category": "Memory", "unit": "kB", "context": "user", "vartype": "integer"}},
            {"metric": {"setting_name": "maintenance_work_mem", "setting_value": "65536", "category": "Memory", "unit": "kB", "context": "user", "vartype": "integer"}},
            {"metric": {"setting_name": "effective_cache_size", "setting_value": "524288", "category": "Memory", "unit": "8kB", "context": "user", "vartype": "integer"}},
            {"metric": {"setting_name": "max_connections", "setting_value": "100", "category": "Connections", "unit": "", "context": "postmaster", "vartype": "integer"}},
        ]
        monkeypatch.setattr(generator, "query_instant", lambda q: prom_result(metrics))

        result = generator.generate_g001_memory_settings_report("test-cluster", "node-01")
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_empty_metrics(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Prometheus returns empty data."""
        monkeypatch.setattr(generator, "query_instant", lambda q: prom_result([]))

        result = generator.generate_g001_memory_settings_report("test-cluster", "node-01")
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_partial_data(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Only some memory settings available."""
        metrics = [
            {"metric": {"setting_name": "shared_buffers", "setting_value": "32768", "category": "Memory", "unit": "8kB", "context": "postmaster", "vartype": "integer"}},
        ]
        monkeypatch.setattr(generator, "query_instant", lambda q: prom_result(metrics))

        result = generator.generate_g001_memory_settings_report("test-cluster", "node-01")
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_error_response(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Prometheus returns error status - documents legacy behavior."""
        monkeypatch.setattr(generator, "query_instant", lambda q: prom_result([], status="error"))

        # Legacy behavior: returns report structure with empty results (no exception)
        result = generator.generate_g001_memory_settings_report("test-cluster", "node-01")
        assert result is not None
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot


# ============================================================================
# K001 Query Calls Report Tests
# ============================================================================

class TestK001QueryCallsReport:
    """Golden snapshot tests for generate_k001_query_calls_report()

    4 test cases as specified in TMP_TESTCOV.md.
    """

    def _mock_k001_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        databases: List[str],
        per_query_data: Dict[str, Dict[str, List[float]]],
        other_data: Dict[str, List[float]],
        timeline: List[int],
    ) -> None:
        """Helper to mock K001 dependencies."""
        monkeypatch.setattr(generator, "get_all_databases", lambda c, n: databases)

        def mock_hourly_topk(cluster, node, db, metric, hours):
            return (
                per_query_data.get(db, {}),
                other_data.get(db, [0.0] * hours),
                timeline,
            )
        monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data", mock_hourly_topk)

    def test_happy_path(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Normal operation with query metrics across multiple databases."""
        databases = ["postgres", "app_db"]
        timeline = [1704067200, 1704070800, 1704074400]  # 3 hours
        per_query = {
            "postgres": {
                "123456789": [100.0, 150.0, 200.0],
                "987654321": [50.0, 75.0, 100.0],
            },
            "app_db": {
                "111222333": [500.0, 600.0, 700.0],
            },
        }
        other = {
            "postgres": [10.0, 15.0, 20.0],
            "app_db": [5.0, 10.0, 15.0],
        }

        self._mock_k001_dependencies(monkeypatch, generator, databases, per_query, other, timeline)

        result = generator.generate_k001_query_calls_report("test-cluster", "node-01", time_range_minutes=180)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_empty_metrics(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """No databases or no query data."""
        self._mock_k001_dependencies(monkeypatch, generator, [], {}, {}, [])

        result = generator.generate_k001_query_calls_report("test-cluster", "node-01")
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_partial_data(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Some databases have data, others don't."""
        databases = ["postgres", "empty_db"]
        timeline = [1704067200, 1704070800]
        per_query = {
            "postgres": {"123456789": [100.0, 200.0]},
        }
        other = {"postgres": [5.0, 10.0]}

        self._mock_k001_dependencies(monkeypatch, generator, databases, per_query, other, timeline)

        result = generator.generate_k001_query_calls_report("test-cluster", "node-01", time_range_minutes=120)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_single_database(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Single database with multiple queries."""
        databases = ["main"]
        timeline = [1704067200]
        per_query = {
            "main": {
                "111": [1000.0],
                "222": [500.0],
                "333": [250.0],
            },
        }
        other = {"main": [50.0]}

        self._mock_k001_dependencies(monkeypatch, generator, databases, per_query, other, timeline)

        result = generator.generate_k001_query_calls_report("test-cluster", "node-01", time_range_minutes=60)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot


# ============================================================================
# K003 Top Queries Report Tests
# ============================================================================

class TestK003TopQueriesReport:
    """Golden snapshot tests for generate_k003_top_queries_report()

    4 test cases as specified in TMP_TESTCOV.md.
    """

    def _mock_k003_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        databases: List[str],
        exec_data: Dict[str, Dict[str, List[float]]],
        plan_data: Dict[str, Dict[str, List[float]]],
        other_exec: Dict[str, List[float]],
        other_plan: Dict[str, List[float]],
        timeline: List[int],
    ) -> None:
        """Helper to mock K003 dependencies."""
        monkeypatch.setattr(generator, "get_all_databases", lambda c, n: databases)

        def mock_hourly_topk(cluster, node, db, metric, hours):
            if "exec_time" in metric:
                return (
                    exec_data.get(db, {}),
                    other_exec.get(db, [0.0] * hours),
                    timeline,
                )
            elif "plan_time" in metric:
                return (
                    plan_data.get(db, {}),
                    other_plan.get(db, [0.0] * hours),
                    timeline,
                )
            return ({}, [0.0] * hours, timeline)

        monkeypatch.setattr(generator, "_get_hourly_topk_pgss_data", mock_hourly_topk)

    def test_happy_path(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Normal operation with exec and plan time data."""
        databases = ["postgres"]
        timeline = [1704067200, 1704070800, 1704074400]
        exec_data = {
            "postgres": {
                "123456789": [1000.0, 1500.0, 2000.0],  # ms
                "987654321": [500.0, 750.0, 1000.0],
            },
        }
        plan_data = {
            "postgres": {
                "123456789": [100.0, 150.0, 200.0],
                "987654321": [50.0, 75.0, 100.0],
            },
        }
        other_exec = {"postgres": [100.0, 150.0, 200.0]}
        other_plan = {"postgres": [10.0, 15.0, 20.0]}

        self._mock_k003_dependencies(
            monkeypatch, generator, databases,
            exec_data, plan_data, other_exec, other_plan, timeline
        )

        result = generator.generate_k003_top_queries_report("test-cluster", "node-01", time_range_minutes=180)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_empty_metrics(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """No databases or no query data."""
        self._mock_k003_dependencies(
            monkeypatch, generator, [], {}, {}, {}, {}, []
        )

        result = generator.generate_k003_top_queries_report("test-cluster", "node-01")
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_exec_only_no_plan(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Only exec time available (older PG without plan time tracking)."""
        databases = ["legacy_db"]
        timeline = [1704067200, 1704070800]
        exec_data = {
            "legacy_db": {
                "111": [5000.0, 6000.0],
                "222": [2000.0, 2500.0],
            },
        }
        # Plan time is zero (not available)
        plan_data = {"legacy_db": {}}
        other_exec = {"legacy_db": [500.0, 600.0]}
        other_plan = {"legacy_db": [0.0, 0.0]}

        self._mock_k003_dependencies(
            monkeypatch, generator, databases,
            exec_data, plan_data, other_exec, other_plan, timeline
        )

        result = generator.generate_k003_top_queries_report("test-cluster", "node-01", time_range_minutes=120)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot

    def test_multiple_databases(
        self,
        snapshot: SnapshotAssertion,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
    ) -> None:
        """Multiple databases with varying query counts."""
        databases = ["db1", "db2"]
        timeline = [1704067200]
        exec_data = {
            "db1": {"1001": [10000.0], "1002": [5000.0]},
            "db2": {"1003": [20000.0]},
        }
        plan_data = {
            "db1": {"1001": [1000.0], "1002": [500.0]},
            "db2": {"1003": [2000.0]},
        }
        other_exec = {"db1": [100.0], "db2": [200.0]}
        other_plan = {"db1": [10.0], "db2": [20.0]}

        self._mock_k003_dependencies(
            monkeypatch, generator, databases,
            exec_data, plan_data, other_exec, other_plan, timeline
        )

        result = generator.generate_k003_top_queries_report("test-cluster", "node-01", time_range_minutes=60)
        sanitized = SnapshotSanitizer.sanitize(result)
        assert sanitized == snapshot


# ============================================================================
# _analyze_memory_settings Tests (P0 requirement)
# ============================================================================

class TestAnalyzeMemorySettings:
    """Tests for _analyze_memory_settings() - P0 function.

    Tests memory analysis calculations with various input configurations.
    """

    def test_standard_config(
        self,
        snapshot: SnapshotAssertion,
        generator: PostgresReportGenerator,
    ) -> None:
        """Standard PostgreSQL memory configuration."""
        memory_data = {
            "shared_buffers": {"setting": "128MB"},
            "work_mem": {"setting": "4MB"},
            "maintenance_work_mem": {"setting": "64MB"},
            "effective_cache_size": {"setting": "4GB"},
            "max_connections": {"setting": "100"},
            "wal_buffers": {"setting": "16MB"},
        }
        result = generator._analyze_memory_settings(memory_data)
        assert result == snapshot

    def test_high_memory_config(
        self,
        snapshot: SnapshotAssertion,
        generator: PostgresReportGenerator,
    ) -> None:
        """High-memory production configuration."""
        memory_data = {
            "shared_buffers": {"setting": "8GB"},
            "work_mem": {"setting": "256MB"},
            "maintenance_work_mem": {"setting": "2GB"},
            "effective_cache_size": {"setting": "24GB"},
            "max_connections": {"setting": "500"},
            "wal_buffers": {"setting": "64MB"},
        }
        result = generator._analyze_memory_settings(memory_data)
        assert result == snapshot

    def test_minimal_config(
        self,
        snapshot: SnapshotAssertion,
        generator: PostgresReportGenerator,
    ) -> None:
        """Minimal/default configuration with missing values."""
        memory_data = {
            "shared_buffers": {"setting": "32MB"},
        }
        result = generator._analyze_memory_settings(memory_data)
        assert result == snapshot

    def test_empty_config(
        self,
        snapshot: SnapshotAssertion,
        generator: PostgresReportGenerator,
    ) -> None:
        """Empty configuration - uses defaults."""
        result = generator._analyze_memory_settings({})
        assert result == snapshot


# ============================================================================
# get_all_nodes Contract Tests (P0 requirement)
# ============================================================================

class TestGetAllNodesContract:
    """Contract tests for get_all_nodes() - P0 function.

    Verifies return shape and edge case handling rather than exact values.
    """

    def test_single_primary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Single primary node, no replicas."""
        # Mock query_instant to return one node
        def mock_query(query: str) -> Dict:
            if "pgwatch_settings_configured" in query:
                return prom_result([{"metric": {"node_name": "node-01"}}])
            if "in_recovery_int" in query:
                return prom_result([{"value": [0, "0"]}])  # Not in recovery = primary
            return prom_result([])

        monkeypatch.setattr(generator, "query_instant", mock_query)

        result = generator.get_all_nodes("test-cluster")

        # Contract: returns dict with 'primary' and 'standbys' keys
        assert "primary" in result
        assert "standbys" in result
        assert result["primary"] == "node-01"
        assert result["standbys"] == []

    def test_primary_with_replicas(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Primary with multiple standbys."""
        def mock_query(query: str) -> Dict:
            if "pgwatch_settings_configured" in query:
                return prom_result([
                    {"metric": {"node_name": "primary-01"}},
                    {"metric": {"node_name": "replica-01"}},
                    {"metric": {"node_name": "replica-02"}},
                ])
            if "in_recovery_int" in query:
                if "primary-01" in query:
                    return prom_result([{"value": [0, "0"]}])  # Primary
                else:
                    return prom_result([{"value": [0, "1"]}])  # Standby
            return prom_result([])

        monkeypatch.setattr(generator, "query_instant", mock_query)

        result = generator.get_all_nodes("test-cluster")

        assert result["primary"] == "primary-01"
        assert len(result["standbys"]) == 2
        assert "replica-01" in result["standbys"]
        assert "replica-02" in result["standbys"]

    def test_empty_cluster(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """No nodes found in cluster."""
        monkeypatch.setattr(generator, "query_instant", lambda q: prom_result([]))

        result = generator.get_all_nodes("empty-cluster")

        # Contract: returns valid structure even when empty
        assert result["primary"] is None
        assert result["standbys"] == []

    def test_multiple_primaries_handled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generator: PostgresReportGenerator,
        prom_result: Callable,
    ) -> None:
        """Edge case: several primaries under one cluster label (unrelated servers, or a split brain)."""
        def mock_query(query: str) -> Dict:
            if "pgwatch_settings_configured" in query:
                return prom_result([
                    {"metric": {"node_name": "node-01"}},
                    {"metric": {"node_name": "node-02"}},
                ])
            if "in_recovery_int" in query:
                return prom_result([{"value": [0, "0"]}])  # Both report as primary
            return prom_result([])

        monkeypatch.setattr(generator, "query_instant", mock_query)

        result = generator.get_all_nodes("test-cluster")

        # Contract: the first becomes the primary; another primary is never a standby (#406)
        assert result["primary"] == "node-01"
        assert result["standbys"] == []
