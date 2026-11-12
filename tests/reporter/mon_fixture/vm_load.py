#!/usr/bin/env python3
"""Push the fixture samples into a scratch VictoriaMetrics (dev tool, not run in CI).

Prometheus remote write (hand-encoded protobuf + snappy; needs the `cramjam` package), so
staleness markers (null in samples.jsonl.gz) reach VictoriaMetrics as the StaleNaN bit pattern
0x7ff0000000000002, which /api/v1/import would drop.

Usage: vm_load.py --vm http://127.0.0.1:PORT
"""
import argparse
import gzip
import json
import os
import struct
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STALE = struct.pack("<Q", 0x7FF0000000000002)


def _varint(n):
    out = bytearray()
    n &= (1 << 64) - 1
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num, wire, payload):
    key = _varint((num << 3) | wire)
    return key + _varint(len(payload)) + payload if wire == 2 else key + payload


def _series(labels, timestamps, values):
    body = b"".join(_field(1, 2, _field(1, 2, k.encode()) + _field(2, 2, str(v).encode()))
                    for k, v in sorted(labels.items()))
    for t, v in zip(timestamps, values):
        raw = STALE if v is None else struct.pack("<d", float(v))
        body += _field(2, 2, _field(1, 1, raw) + _field(2, 0, _varint(int(t))))
    return _field(1, 2, body)


def _post(url, parts):
    import cramjam
    body = bytes(cramjam.snappy.compress_raw(b"".join(parts)))
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Encoding": "snappy", "Content-Type": "application/x-protobuf",
        "X-Prometheus-Remote-Write-Version": "0.1.0"})
    urllib.request.urlopen(req, timeout=120).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", required=True)
    a = ap.parse_args()
    buf, n, samples = [], 0, 0
    with gzip.open(os.path.join(HERE, "samples.jsonl.gz"), "rt") as f:
        for line in f:
            d = json.loads(line)
            buf.append(_series(d["metric"], d["timestamps"], d["values"]))
            n += 1
            samples += len(d["timestamps"])
            if len(buf) >= 200:
                _post(a.vm.rstrip("/") + "/api/v1/write", buf)
                buf = []
    if buf:
        _post(a.vm.rstrip("/") + "/api/v1/write", buf)
    urllib.request.urlopen(a.vm.rstrip("/") + "/internal/force_flush", timeout=60).read()
    print(f"remote-wrote {n} series, {samples} samples")


if __name__ == "__main__":
    main()
