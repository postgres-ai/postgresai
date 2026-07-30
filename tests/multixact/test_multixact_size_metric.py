"""Static regression coverage for PostgreSQL 19 multixact monitoring."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(name="metric")
def fixture_metric() -> dict:
    metrics_path = (
        Path(__file__).parents[2]
        / "config"
        / "pgwatch-prometheus"
        / "metrics.yml"
    )
    config = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    return config["metrics"]["multixact_size"]


def sql_for_version(sqls: dict[int, str], version: int) -> str:
    """Match pgwatch's nearest-compatible-version selection."""
    compatible = [key for key in sqls if key <= version]
    if not compatible:
        raise ValueError(f"No multixact SQL supports PostgreSQL {version}")
    return sqls[max(compatible)]


@pytest.mark.unit
def test_pg19_selects_native_multixact_sql(metric: dict) -> None:
    sql = sql_for_version(metric["sqls"], 19)

    assert sql == metric["sqls"][19]
    assert "pg_catalog.pg_get_multixact_stats()" in sql
    assert "pg_ls_dir" not in sql
    assert "query_to_xml" not in sql


@pytest.mark.unit
def test_future_versions_keep_native_multixact_sql(metric: dict) -> None:
    assert sql_for_version(metric["sqls"], 20) == metric["sqls"][19]


@pytest.mark.unit
def test_pre_pg19_keeps_legacy_provider_probes(metric: dict) -> None:
    sql = sql_for_version(metric["sqls"], 18)

    assert sql == metric["sqls"][11]
    assert "aurora_stat_file" in sql
    assert "rds_tools.pg_ls_multixactdir()" in sql
    assert "pg_ls_dir" in sql


@pytest.mark.unit
def test_pg19_preserves_metric_contract(metric: dict) -> None:
    sql = metric["sqls"][19]

    assert "members_bytes" in sql
    assert "offsets_bytes" in sql
    assert "status_code" in sql
    assert metric["gauges"] == [
        "members_bytes",
        "offsets_bytes",
        "status_code",
    ]


@pytest.mark.unit
def test_pg19_offset_estimate_uses_slru_geometry(metric: dict) -> None:
    sql = metric["sqls"][19]

    assert "ceiling(num_mxids::numeric / 32768) * 262144" in sql


@pytest.mark.unit
def test_pg19_nulls_report_unavailable(metric: dict) -> None:
    sql = metric["sqls"][19]

    assert "num_mxids is not null and members_size is not null then 0" in sql
    assert "else 2" in sql
