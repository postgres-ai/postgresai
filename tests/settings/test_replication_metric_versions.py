"""Static regression tests for versioned replication metric SQL."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


METRICS_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "pgwatch-prometheus"
    / "metrics.yml"
)
REPLICATION_METRICS = ("replication", "pg_stat_replication")


def metric_sql(metric_name: str, version: int) -> str:
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = yaml.safe_load(metrics_file)
    return metrics["metrics"][metric_name]["sqls"][version]


def normalized(sql_text: str) -> str:
    return re.sub(r"\s+", " ", sql_text).lower()


@pytest.mark.unit
@pytest.mark.parametrize("metric_name", REPLICATION_METRICS)
def test_replication_11_sql_has_no_reply_time(metric_name: str) -> None:
    assert "reply_time" not in normalized(metric_sql(metric_name, 11))


@pytest.mark.unit
@pytest.mark.parametrize("metric_name", REPLICATION_METRICS)
def test_replication_12_sql_clamps_reply_time_lag(metric_name: str) -> None:
    sql = normalized(metric_sql(metric_name, 12))

    assert "reply_time" in sql
    assert "greatest(0," in sql
    assert "extract(epoch from" in sql
    assert "extract(seconds from" not in sql
    assert re.search(r" as reply_time_lag(?:_ms)?(?:,| from)", sql)
