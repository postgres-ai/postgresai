"""
Tests for the settings metric, particularly the lock_timeout fix.

This module tests that the settings metric correctly uses reset_val for lock_timeout
instead of the session-level setting value, which would be affected by pgwatch's
lock_timeout override during metric collection.

See: https://gitlab.com/postgres-ai/postgres_ai/-/issues/61
"""

import re
from pathlib import Path

import pytest
import yaml


def get_metrics_yml_path() -> Path:
    """Get the path to the metrics.yml file."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "mon" / "metrics.yml"


@pytest.fixture(name="metrics_config")
def fixture_metrics_config() -> dict:
    """Load the metrics.yml configuration."""
    metrics_path = get_metrics_yml_path()
    with open(metrics_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# The settings the collector's session overrides; the SQL must list at least these.
OVERRIDDEN_SETTINGS = {"lock_timeout", "statement_timeout", "search_path"}


def normalized(sql: str) -> str:
    """Collapse whitespace: the tests do not depend on how the SQL is wrapped."""
    return re.sub(r"\s+", " ", sql)


def overridden_setting_lists(sql: str) -> list[tuple[set[str], str]]:
    """Every `when name in (...) then <expr>` branch: (names, what it yields)."""
    return [
        (set(re.findall(r"'([^']+)'", names)), expr.strip())
        for names, expr in re.findall(
            r"when name in \(([^)]*)\) then (.*?) else", normalized(sql)
        )
    ]


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

    # The value branch lists every overridden setting and yields reset_val.
    value_lists = [names for names, expr in overridden_setting_lists(sql) if expr == "reset_val"]
    assert len(value_lists) == 1, "one CASE should replace setting with reset_val"
    assert OVERRIDDEN_SETTINGS <= value_lists[0]


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

    tag_setting_value and numeric_value must both come from the one expression
    that picks reset_val (effective_setting), never from the raw setting column.
    """
    sql = normalized(metrics_config["metrics"]["settings"]["sqls"][11])

    assert "then reset_val else setting end as effective_setting" in sql
    assert "effective_setting as tag_setting_value" in sql
    assert re.search(r"then effective_setting::bigint else null end as numeric_value", sql), \
        "numeric_value should be cast from effective_setting"
    assert not re.search(r"(?<![a-z_])setting::", sql), \
        "nothing should be cast from the raw setting column"


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

    # The is_default branch covers every setting the value branch covers: a
    # setting in one list only would report a wrong is_default (or a wrong value).
    lists = overridden_setting_lists(sql)
    assert len(lists) == 2, f"expected the value and the is_default branches, found {len(lists)}"
    (value_names, value_expr), (default_names, default_expr) = lists
    assert value_expr == "reset_val"
    assert default_expr == "(reset_val = boot_val)"
    assert OVERRIDDEN_SETTINGS <= value_names
    assert value_names <= default_names
