"""Old stack: the reporter's K003 monitoring-query filter must see the marker in
every old/config/pgwatch-postgres/metrics.yml SQL. The live check over
mon/metrics.yml is in tests/reporter/test_k003_window_facts.py."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reporter import postgres_reports as pr

MONITOR = pr.MONITORING_MARKER


@pytest.mark.unit
def test_k003_reads_enough_text_to_see_every_pgwatch_postgres_marker() -> None:
    path = Path(__file__).resolve().parents[3] / "old/config/pgwatch-postgres/metrics.yml"
    for name, metric in (yaml.safe_load(path.read_text()).get("metrics") or {}).items():
        for version, sql in ((metric or {}).get("sqls") or {}).items():
            at = str(sql).find(MONITOR)
            assert at < pr.K003_TEXT_FETCH_CHARS - len(MONITOR), (name, version, at)
