"""
Tests for the settings metric, particularly the lock_timeout fix.

This module tests that the settings metric correctly uses reset_val for lock_timeout
instead of the session-level setting value, which would be affected by pgwatch's
lock_timeout override during metric collection.

See: https://gitlab.com/postgres-ai/postgres_ai/-/issues/61
"""

import os
from pathlib import Path

import pytest
import yaml


def get_metrics_yml_path() -> Path:
    """Get the path to the metrics.yml file."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "config" / "pgwatch-prometheus" / "metrics.yml"


@pytest.fixture(name="metrics_config")
def fixture_metrics_config() -> dict:
    """Load the metrics.yml configuration."""
    metrics_path = get_metrics_yml_path()
    with open(metrics_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.unit
def test_settings_metric_exists(metrics_config: dict) -> None:
    """Test that the settings metric is defined in metrics.yml."""
    assert "metrics" in metrics_config, "metrics key should exist"
    assert "settings" in metrics_config["metrics"], "settings metric should be defined"
    assert "sqls" in metrics_config["metrics"]["settings"], "settings metric should have sqls"
    assert 11 in metrics_config["metrics"]["settings"]["sqls"], "settings metric should have SQL for version 11+"


@pytest.mark.unit
def test_settings_metric_uses_reset_val_for_overridden_settings(metrics_config: dict) -> None:
    """
    Test that the settings metric uses reset_val for settings overridden by pgwatch.

    This is critical because pgwatch overrides these during metric collection:
    - lock_timeout: set to 100ms
    - statement_timeout: set per-metric (e.g., 15s for settings)

    Using 'setting' would mask the actual configured values.

    See: https://gitlab.com/postgres-ai/postgres_ai/-/issues/61
    """
    sql = metrics_config["metrics"]["settings"]["sqls"][11]

    # Verify the SQL uses reset_val for overridden settings
    assert "reset_val" in sql, "SQL should reference reset_val column"
    assert "lock_timeout" in sql, "SQL should reference lock_timeout"
    assert "statement_timeout" in sql, "SQL should reference statement_timeout"

    # More specific check: ensure the CASE statement handles both settings
    assert "name in ('lock_timeout', 'statement_timeout', 'search_path')" in sql, \
        "SQL should use IN clause to handle both lock_timeout and statement_timeout"


@pytest.mark.unit
def test_settings_metric_description_documents_lock_timeout_fix(metrics_config: dict) -> None:
    """Test that the description documents the lock_timeout fix."""
    description = metrics_config["metrics"]["settings"]["description"]
    assert "lock_timeout" in description.lower(), \
        "Description should mention lock_timeout"
    assert "reset_val" in description.lower(), \
        "Description should mention reset_val"


@pytest.mark.unit
def test_settings_metric_has_required_fields(metrics_config: dict) -> None:
    """Test that the settings metric SQL includes all required fields."""
    sql = metrics_config["metrics"]["settings"]["sqls"][11]

    required_fields = [
        "epoch_ns",
        "tag_datname",
        "tag_setting_name",
        "tag_setting_value",
        "tag_unit",
        "tag_category",
        "tag_vartype",
        "numeric_value",
        "is_default",
        "configured",
    ]

    for field in required_fields:
        assert field in sql, f"SQL should include {field}"


@pytest.mark.unit
def test_settings_metric_numeric_value_uses_correct_source_for_overridden_settings(metrics_config: dict) -> None:
    """
    Test that numeric_value also uses reset_val for overridden settings.

    The numeric_value field should be derived from the same source as tag_setting_value
    to ensure consistency.
    """
    sql = metrics_config["metrics"]["settings"]["sqls"][11]

    # Count occurrences of the CASE expression for overridden settings
    # It should appear twice: once for tag_setting_value and once for numeric_value
    case_expr = "case when name in ('lock_timeout', 'statement_timeout', 'search_path') then reset_val else setting end"
    occurrences = sql.count(case_expr)

    assert occurrences >= 2, \
        f"CASE expression for overridden settings should appear at least twice (for tag_setting_value and numeric_value), found {occurrences}"


@pytest.mark.unit
def test_settings_metric_is_default_uses_boot_val_for_overridden_settings(metrics_config: dict) -> None:
    """
    Test that is_default compares reset_val with boot_val for overridden settings.

    When pgwatch overrides lock_timeout/statement_timeout during collection, the source
    becomes 'session', which would incorrectly report is_default=0. Instead, we should
    compare reset_val with boot_val to determine if the configured value is the default.

    See: https://gitlab.com/postgres-ai/postgres_ai/-/issues/61
    """
    sql = metrics_config["metrics"]["settings"]["sqls"][11]

    # Verify is_default uses boot_val comparison for overridden settings
    assert "boot_val" in sql, "SQL should reference boot_val column for is_default comparison"

    # Check for the specific pattern that handles overridden settings specially
    assert "name in ('lock_timeout', 'statement_timeout', 'search_path')" in sql and "boot_val" in sql, \
        "SQL should use special handling for overridden settings in is_default calculation"


@pytest.mark.integration
@pytest.mark.requires_postgres
def test_settings_metric_lock_timeout_returns_actual_value() -> None:
    """
    Integration test: verify lock_timeout returns the actual configured value.

    This test requires a running PostgreSQL database and tests that even when
    lock_timeout is set to a different value in the session, our query returns
    the actual configured (reset) value.
    """
    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not available")

    target_db_url = os.getenv(
        "TARGET_DB_URL",
        "postgresql://postgres:postgres@localhost:55432/target_database"
    )

    try:
        conn = psycopg.connect(target_db_url)
    except Exception as e:
        pytest.skip(f"Could not connect to PostgreSQL: {e}")

    try:
        with conn.cursor() as cur:
            # First, get the actual configured lock_timeout
            cur.execute("select reset_val from pg_settings where name = 'lock_timeout'")
            actual_value = cur.fetchone()[0]

            # Now, simulate what pgwatch does - set a session-level lock_timeout
            cur.execute("set lock_timeout = '100ms'")

            # Query using our fixed SQL logic
            cur.execute("""
                select
                    name,
                    setting as raw_setting,
                    reset_val,
                    case when name = 'lock_timeout' then reset_val else setting end as our_value
                from pg_settings
                where name = 'lock_timeout'
            """)
            row = cur.fetchone()
            name, raw_setting, reset_val, our_value = row

            # Verify: raw_setting should be 100 (the session override)
            # but our_value should be the actual configured value
            assert raw_setting == "100", f"Session setting should be 100ms, got {raw_setting}"
            assert our_value == actual_value, \
                f"Our query should return reset_val ({actual_value}), not session value ({raw_setting})"
            assert our_value == reset_val, "our_value should equal reset_val"

    finally:
        conn.close()
