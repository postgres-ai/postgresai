"""Wait until Prometheus has data for every monitoring target (first run only).

postgres-reports.sh runs this once, before the first report run. It reads the
targets from the pgwatch sources file (generated from instances.yml) and polls
Prometheus until each enabled target's cluster/node_name has `settings` and
`db_stats` samples, or until the time limit. It always exits 0: on timeout the
run goes ahead, and the reporter does not upload a report without data.

Usage: python -m reporter.wait_ready PROMETHEUS_URL SOURCES_PATH TIMEOUT_S POLL_S
"""
import re
import sys
import time
from typing import Callable, List, Optional, Set, Tuple

# Both series every report reads first: cluster/node discovery uses
# `settings`, primary/standby detection uses `db_stats`.
READY_SERIES = ("pgwatch_settings_configured", "pgwatch_db_stats_in_recovery_int")

Target = Tuple[str, str]


def _scalar(raw: str) -> str:
    value = raw.strip()
    # A quoted scalar may hold " #" (js-yaml quotes such a name): unquote first,
    # strip a trailing comment only from a plain scalar.
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        if end > 0:
            return value[1:end]
    return value.split(" #", 1)[0].strip()


def read_targets(path: str) -> List[Target]:
    """Return (cluster, node_name) of each enabled target in a sources file.

    The file is the list instances.yml holds (one `- name:` item per target,
    `cluster` and `node_name` under `custom_tags`). Defaults follow the CLI:
    cluster "default", node_name = the target name.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    def value(i: int, raw: str) -> str:
        # A block scalar (js-yaml writes long names as `>-`): the value is the
        # following lines indented deeper than this key (a `- name:` key sits
        # after the dash).
        if not re.match(r"^[>|][+-]?$", raw.strip()):
            return _scalar(raw)
        indent = len(lines[i]) - len(lines[i].lstrip(" -"))
        parts = []
        for nxt in lines[i + 1:]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            parts.append(nxt.strip())
        return (" " if raw.strip()[0] == ">" else "\n").join(p for p in parts if p)

    items = []
    for i, line in enumerate(lines):
        m = re.match(r"^-\s+name:\s*(.*)$", line)
        if m:
            items.append({"name": value(i, m.group(1)), "is_enabled": "true"})
            continue
        if not items:
            continue
        m = re.match(r"^\s+(is_enabled|cluster|node_name):\s*(.*)$", line)
        if m:
            items[-1][m.group(1)] = value(i, m.group(2))
    targets = []
    for item in items:
        if item["is_enabled"].lower() != "true":
            continue
        target = (item.get("cluster") or "default", item.get("node_name") or item["name"])
        if target not in targets:
            targets.append(target)
    return targets


def targets_with_data(query: Callable[[str], dict]) -> Set[Target]:
    """Targets that have samples of every READY_SERIES in Prometheus."""
    ready: Optional[Set[Target]] = None
    for series in READY_SERIES:
        result = query(f"count by (cluster, node_name) (last_over_time({series}[3h]))")
        found = set()
        if result.get("status") == "success":
            for item in result.get("data", {}).get("result", []):
                labels = item.get("metric", {})
                found.add((labels.get("cluster", ""), labels.get("node_name", "")))
        ready = found if ready is None else ready & found
    return ready or set()


def wait_ready(query: Callable[[str], dict], targets: List[Target], timeout_s: float,
               poll_s: float, sleep=None, clock=None) -> bool:
    """Poll until every target has data. True when ready, False on timeout."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    if not targets:
        print("postgres-reports: no enabled targets in the sources file; not waiting", flush=True)
        return True
    deadline = clock() + timeout_s
    while True:
        ready = targets_with_data(query)
        missing = [t for t in targets if t not in ready]
        if not missing:
            print(f"postgres-reports: data ready for {len(targets)} target(s)", flush=True)
            return True
        names = ", ".join(f"{c}/{n}" for c, n in missing)
        if clock() >= deadline:
            print(f"postgres-reports: no data after {int(timeout_s)}s for: {names}; "
                  "running anyway", flush=True)
            return False
        print(f"postgres-reports: waiting for data: {names}; next check in {int(poll_s)}s",
              flush=True)
        sleep(poll_s)


def main(argv: List[str]) -> int:
    prometheus_url, sources_path, timeout_s, poll_s = argv[1], argv[2], float(argv[3]), float(argv[4])
    from reporter.postgres_reports import PostgresReportGenerator

    generator = PostgresReportGenerator(prometheus_url, postgres_sink_url=None)
    wait_ready(generator.query_instant, read_targets(sources_path), timeout_s, poll_s)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
