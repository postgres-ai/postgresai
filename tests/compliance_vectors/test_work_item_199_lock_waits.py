"""Regression coverage for work item 199 lock_waits metric labels."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "config" / "pgwatch-prometheus" / "metrics.yml"


def test_lock_waits_session_pids_are_labels_not_metric_values() -> None:
    metrics = METRICS_FILE.read_text()
    lock_waits_sql = metrics.split("  lock_waits:", 1)[1].split("  xmin_horizon:", 1)[0]

    assert "blocked.pid::text as tag_blocked_pid" in lock_waits_sql
    assert "blocker.pid::text as tag_blocker_pid" in lock_waits_sql
    assert "blocked.pid as blocked_pid" not in lock_waits_sql
    assert "blocker.pid as blocker_pid" not in lock_waits_sql

    gauges = lock_waits_sql.split("    gauges:", 1)[1].split("    statement_timeout_seconds:", 1)[0]
    assert "blocked_pid" not in gauges
    assert "blocker_pid" not in gauges
    assert "blocked_ms" in gauges
    assert "blocker_tx_ms" in gauges
