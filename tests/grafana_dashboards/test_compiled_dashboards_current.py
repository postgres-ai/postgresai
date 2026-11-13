"""The committed dashboards are what mon/grafana/compile.py emits today.

The JSON under mon/grafana/dashboards is generated; a hand edit there, or a change to the
compiler, its cases or the source dashboards without recompiling, fails here. The compiler
needs a pgai-mon to learn each query's output columns, so this runs one over a recorded store.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.reporter import mon_harness

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "mon" / "grafana" / "dashboards"


def test_committed_dashboards_match_the_compiler(tmp_path):
    binary = mon_harness.find_binary()
    if binary is None:
        if os.environ.get("CI"):
            pytest.fail("no pgai-mon binary (PGAI_MON_BIN)")
        pytest.skip("no pgai-mon binary: build mon/ or set PGAI_MON_BIN")
    store = mon_harness.build_store(binary, tmp_path / "store")
    out = tmp_path / "out"
    out.mkdir()
    with mon_harness.MonServer(binary, store) as mon:
        run = subprocess.run(
            [sys.executable, str(REPO_ROOT / "mon" / "grafana" / "compile.py"), "--probe-url", mon.url, "--out", str(out)],
            capture_output=True, text=True, timeout=300,
        )
    assert run.returncode == 0, run.stderr[-2000:]
    emitted = {p.name: p.read_text() for p in out.glob("*.json")}
    committed = {p.name: p.read_text() for p in COMMITTED.glob("*.json")}
    assert sorted(emitted) == sorted(committed)
    stale = [name for name in sorted(committed) if emitted[name] != committed[name]]
    assert not stale, f"recompile (mon/grafana/README.md): {stale}"
