"""Test helpers: build the reporter's pgai-mon fixture store and serve it on 127.0.0.1.

The fixture (tests/reporter/mon_fixture/) is a slice of real pgwatch data; see build_fixture.py.
`pgai-mon load` turns samples.jsonl.gz into a store; the query_texts and objects Parquet files are
then registered in the store's catalog.json (pgai-mon has no loader for them yet).

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
    """Load the fixture samples into `store` and register query_texts / objects."""
    if store.exists():
        shutil.rmtree(store)
    with gzip.open(FIXTURE / "samples.jsonl.gz", "rb") as f:
        proc = subprocess.run([binary, "load", "--store", str(store), "--input", "-"], input=f.read(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"pgai-mon load failed: {proc.stderr.decode()[-2000:]}")
    _register(store, "query_texts", "query_texts.parquet", "last_seen_ms", FIXTURE / "query_texts.jsonl")
    _register(store, "objects", "objects.parquet", "ts_ms", FIXTURE / "objects.jsonl")
    return store


def _register(store: Path, kind: str, parquet: str, ts_key: str, jsonl: Path) -> None:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    ts = [int(r[ts_key]) for r in rows]
    day = max(ts) // 86_400_000
    date = time.strftime("%Y-%m-%d", time.gmtime(day * 86400))
    cat_path = store / "catalog.json"
    cat = json.loads(cat_path.read_text())
    file_id = cat["next_file_id"]
    cat["next_file_id"] = file_id + 1
    rel = f"{kind}/date={date}/L2-{file_id:08d}.parquet"
    (store / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE / parquet, store / rel)
    cat["files"].append({"kind": kind, "path": rel, "day": day, "level": 2, "rows": len(rows),
                         "bytes": (store / rel).stat().st_size, "min_ts": min(ts), "max_ts": max(ts),
                         "created_ms": max(ts)})
    tmp = cat_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat))
    tmp.replace(cat_path)


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
            [self.binary, "serve", "--store", str(self.store), "--listen", f"127.0.0.1:{self.port}", "--no-compact",
             "--metrics-yml", str(REPO / "config" / "pgwatch-prometheus" / "metrics.yml"),
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
