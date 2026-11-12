"""Test helpers: build the reporter's pgai-mon fixture store and serve it on 127.0.0.1.

The fixture (tests/reporter/mon_fixture/) is a slice of real pgwatch data; see build_fixture.py.
`pgai-mon load` turns samples.jsonl.gz and synthetic.jsonl.gz, query_texts.jsonl and objects.jsonl into a store.

The binary comes from $PGAI_MON_BIN, else mon/target/{release,debug}/pgai-mon, else
$CARGO_TARGET_DIR/{release,debug}/pgai-mon. Without one the tests that need it are skipped.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "mon_fixture"
# The fixture's evaluation time: its last sample (2026-11-09 15:07:28.444 UTC) rounded down to a
# whole minute, so every check sees data and the instant time is whole seconds.
FIXTURE_AT = 1794236820
# the real slice, then the series it lacks (synthetic.py)
SAMPLE_FILES = ("samples.jsonl.gz", "synthetic.jsonl.gz")


def find_binary() -> Optional[str]:
    cands = [os.environ.get("PGAI_MON_BIN")]
    for base in (REPO / "mon" / "target", Path(os.environ["CARGO_TARGET_DIR"]) if os.environ.get("CARGO_TARGET_DIR") else None):
        if base:
            cands += [str(base / "release" / "pgai-mon"), str(base / "debug" / "pgai-mon")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def build_store(binary: str, store: Path) -> Path:
    """Load the fixture samples, query texts and objects into `store`."""
    if store.exists():
        shutil.rmtree(store)
    data = b"".join(gzip.open(FIXTURE / name, "rb").read() for name in SAMPLE_FILES)
    # The fixture keeps queryid as a string (pgwatch's label); `load` takes the integer.
    query_texts = store.parent / "query_texts.jsonl"
    with open(FIXTURE / "query_texts.jsonl") as src, open(query_texts, "w") as dst:
        for line in src:
            if line.strip():
                row = json.loads(line)
                row["queryid"] = int(row["queryid"])
                dst.write(json.dumps(row) + "\n")
    proc = subprocess.run([binary, "load", "--store", str(store), "--input", "-",
                           "--query-texts", str(query_texts),
                           "--objects", str(FIXTURE / "objects.jsonl")], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"pgai-mon load failed: {proc.stderr.decode()[-2000:]}")
    return store


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MonServer:
    """`pgai-mon serve --no-compact` over a store, on a random 127.0.0.1 port."""

    def __init__(self, binary: str, store: Path, log: Optional[Path] = None):
        self.binary, self.store = binary, store
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "MonServer":
        out = open(self.log, "wb") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            # 127.0.0.1 only, so no token (as tests/grafana_dashboards/test_sql_panels_run.py).
            [self.binary, "serve", "--store", str(self.store), "--listen", f"127.0.0.1:{self.port}", "--no-compact",
             "--no-auth",
             "--metrics-yml", str(REPO / "mon" / "metrics.yml"),
             "--macros-sql", str(REPO / "mon" / "sql" / "macros.sql"), "--timeout-ms", "120000"],
            stdout=out, stderr=subprocess.STDOUT, cwd=str(REPO))
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"pgai-mon serve exited with {self.proc.returncode}")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2).read()
                return self
            except OSError:
                time.sleep(0.2)
        self.__exit__(None, None, None)
        raise RuntimeError("pgai-mon serve did not become ready")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
