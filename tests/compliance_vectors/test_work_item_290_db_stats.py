"""Regression coverage for work item 290's Aurora-safe db_stats metric."""

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "config" / "pgwatch-prometheus" / "metrics.yml"


def test_backup_start_time_is_lazily_guarded_on_supported_versions() -> None:
    metrics = yaml.safe_load(METRICS_FILE.read_text(encoding="utf-8"))
    db_stats_sqls = metrics["metrics"]["db_stats"]["sqls"]
    guarded_expression = re.compile(
        r"case\s+when\s+to_regproc\(\s*'aurora_version'\s*\)\s+is\s+null\s+"
        r"then\s+extract\(\s*epoch\s+from\s+\(\s*now\(\s*\)\s*-\s*"
        r"pg_backup_start_time\(\s*\)\s*\)\s*\)\s*::\s*int8\s+end\s+as\s+backup_duration_s",
        re.IGNORECASE,
    )

    assert guarded_expression.search(db_stats_sqls[12])
    assert guarded_expression.search(db_stats_sqls[14])
    assert "pg_backup_start_time()" not in db_stats_sqls[11]
    assert "pg_backup_start_time()" not in db_stats_sqls[15]
