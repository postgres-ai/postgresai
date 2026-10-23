"""Tests for reporter.wait_ready: the first run waits for every target's data."""
import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest

from reporter import wait_ready as wait_ready_module
from reporter.postgres_reports import PostgresReportGenerator
from reporter.wait_ready import read_targets, targets_with_data, wait_ready

CLI_SOURCES = """# PGWatch Sources Configuration - prometheus Instance
- name: prod-db
  conn_str: postgresql://mon:p#w@db:5432/app
  preset_metrics: full
  custom_metrics: null
  is_enabled: true
  group: default
  custom_tags:
    env: production
    cluster: default
    node_name: prod-db
    sink_type: prometheus
- name: 'prod.main'
  conn_str: postgresql://mon:pw@db2:5432/app
  is_enabled: true
  custom_tags:
    cluster: "c2"
    node_name: node-01  # primary
- name: off
  is_enabled: false
  custom_tags:
    cluster: c3
    node_name: off
- name: bare
  is_enabled: true
"""


def prom(state):
    """A fake Prometheus: state maps series name -> set of (cluster, node_name)."""
    def query(q):
        for series, targets in state.items():
            if series in q:
                return {"status": "success", "data": {"result": [
                    {"metric": {"cluster": c, "node_name": n}, "value": [0, "1"]} for c, n in targets
                ]}}
        return {"status": "success", "data": {"result": []}}
    return query


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def clock(self):
        return self.now


@pytest.mark.unit
def test_read_targets_takes_enabled_cluster_and_node(tmp_path):
    path = tmp_path / "sources.yml"
    path.write_text(CLI_SOURCES)
    assert read_targets(str(path)) == [("default", "prod-db"), ("c2", "node-01"), ("default", "bare")]


@pytest.mark.unit
def test_read_targets_missing_file_is_empty(tmp_path):
    assert read_targets(str(tmp_path / "none.yml")) == []


@pytest.mark.unit
def test_a_target_needs_both_settings_and_db_stats():
    query = prom({
        "pgwatch_settings_configured": {("c1", "n1"), ("c2", "n2")},
        "pgwatch_db_stats_in_recovery_int": {("c1", "n1")},
    })
    assert targets_with_data(query) == {("c1", "n1")}
    assert targets_with_data(lambda q: {}) == set()


@pytest.mark.unit
def test_waits_until_every_target_has_data():
    clock = FakeClock()
    state = {"pgwatch_settings_configured": set(), "pgwatch_db_stats_in_recovery_int": set()}
    query = prom(state)
    polls = []

    def counting(q):
        polls.append(q)
        # The first target's data lands after 2 checks, the second's after 4.
        checks = len(polls) // 2
        if checks >= 2:
            state["pgwatch_settings_configured"] = {("c1", "n1")} | ({("c2", "n2")} if checks >= 4 else set())
            state["pgwatch_db_stats_in_recovery_int"] = set(state["pgwatch_settings_configured"])
        return query(q)

    ready = wait_ready(counting, [("c1", "n1"), ("c2", "n2")], 1800, 15, clock.sleep, clock.clock)
    assert ready is True
    assert clock.sleeps == [15, 15, 15, 15]


@pytest.mark.unit
def test_gives_up_at_the_time_limit(capsys):
    clock = FakeClock()
    query = prom({"pgwatch_settings_configured": {("c1", "n1")},
                  "pgwatch_db_stats_in_recovery_int": {("c1", "n1")}})
    ready = wait_ready(query, [("c1", "n1"), ("c2", "n2")], 60, 15, clock.sleep, clock.clock)
    assert ready is False
    assert clock.sleeps == [15, 15, 15, 15]
    assert "no data after 60s for: c2/n2; running anyway" in capsys.readouterr().out


@pytest.mark.unit
def test_no_targets_does_not_wait():
    clock = FakeClock()
    assert wait_ready(lambda q: {}, [], 1800, 15, clock.sleep, clock.clock) is True
    assert clock.sleeps == []


@pytest.mark.unit
def test_read_targets_unquotes_before_it_strips_comments(tmp_path):
    # js-yaml writes a name with " #" in quotes.
    path = tmp_path / "sources.yml"
    path.write_text("- name: 'a #b'\n  is_enabled: true\n- name: c  # note\n  is_enabled: true\n")
    assert read_targets(str(path)) == [("default", "a #b"), ("default", "c")]


@pytest.mark.unit
@pytest.mark.parametrize("entry", ["main", "python -m"])
def test_main_passes_url_timeout_and_poll(entry, tmp_path, monkeypatch, capsys):
    """`python -m reporter.wait_ready URL SOURCES TIMEOUT POLL`, as
    postgres-reports.sh runs it: the URL goes to the generator, the wait
    uses query_instant, gives up after TIMEOUT and sleeps POLL between checks."""
    sources = tmp_path / "sources.yml"
    sources.write_text("- name: n1\n  is_enabled: true\n  custom_tags:\n    cluster: c1\n")
    clock = FakeClock()
    monkeypatch.setattr(wait_ready_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(wait_ready_module.time, "monotonic", clock.clock)
    generator = MagicMock(spec=PostgresReportGenerator)
    generator.query_instant.side_effect = prom({})
    generator.query_range.side_effect = AssertionError("query_range is not the readiness query")
    argv = ["x", "http://prom.test:9090", str(sources), "60", "5"]

    with patch("reporter.postgres_reports.PostgresReportGenerator", return_value=generator) as cls:
        if entry == "main":
            assert wait_ready_module.main(argv) == 0
        else:
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_module("reporter.wait_ready", run_name="__main__")
            assert exit_info.value.code == 0

    cls.assert_called_once_with("http://prom.test:9090", postgres_sink_url=None)
    assert clock.sleeps == [5.0] * 12
    assert generator.query_instant.call_count == 13 * 2
    assert generator.query_range.call_count == 0
    out = capsys.readouterr().out
    assert "waiting for data: c1/n1; next check in 5s" in out
    assert "no data after 60s for: c1/n1; running anyway" in out


LONG = "denis-clickhouse-test-1-4224e50c-pg9qxanm35t63qsngnpdvpynnd-c0-eu-central-1-aws-pg-clickhouse-cloud-postgres"


@pytest.mark.unit
def test_read_targets_reads_folded_block_scalars(tmp_path):
    """js-yaml writes a long name as a folded block scalar (`name: >-` with the
    value on the next, deeper line). Reading `>-` as the name made the first
    run wait the whole readiness timeout for a target that never matches."""
    path = tmp_path / "sources.yml"
    path.write_text(
        "# PGWatch Sources Configuration - prometheus Instance\n"
        "- name: >-\n"
        f"    {LONG}\n"
        "  conn_str: >-\n"
        "    postgresql://u:p@host:5432/postgres?sslmode=require\n"
        "  is_enabled: true\n"
        "  custom_tags:\n"
        "    cluster: default\n"
        "    node_name: >-\n"
        f"      {LONG}\n"
        "- name: >-\n"
        f"    {LONG}-2\n"
        "  is_enabled: true\n"
    )
    assert read_targets(str(path)) == [("default", LONG), ("default", LONG + "-2")]
