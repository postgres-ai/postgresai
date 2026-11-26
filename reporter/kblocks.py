"""K006–K008 (top statements by shared blocks) and the chart data of A005 and K006–K008.

What CheckupLoop asked the monitoring box for and now reads from the file (postgresai#534):

- per statement: K003's window facts plus its time (and, in K006, its read time), and its calls
  per hour (`hourly_values`);
- per database: the block reads and hits of pg_stat_database over the window, its size and
  when its statistics were last reset;
- per node: `block_size` and `track_io_timing` (`settings_facts`), and the deadlocks of each
  database per hour over the window (`deadlocks_by_database`);
- per report: the scrape health of the monitoring stack (`scrape_targets`).

All of it comes from monitoring; nothing here opens a connection to the monitored database.
A fact that cannot be read is never written as zero: the facts with a reason key say why, the
others are left out (CHANGELOG.md, #534). A counter's `hourly` counts an hour with no readings
as 0: compare its `start`/`end`.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reporter import k004

#: The settings these reports read (K006-O1 block size, K006-O3 read time).
SETTINGS = ("block_size", "track_io_timing")
SETTING_PATTERN = r"^(block_size|track_io_timing)$"
#: Readings of one collection (one value per database, read a few seconds apart) count as read at once.
STEP_TOLERANCE_S = 600
#: An hourly reading belongs to the timeline stamp it is within half an hour of.
HOUR_MATCH_TOLERANCE_S = 1800
#: Why a counter has no window: an empty range reply is both "no readings" and a failed query.
NO_COUNTER = "no two readings of the counter in the window (none collected, or the monitoring query failed)"


def hourly_values(points: Optional[Dict[int, float]], timeline: Sequence[int]) -> List[Optional[float]]:
    """A statement's value per hour of `timeline` (each stamp the end of its hour), from an hourly
    range reply: the reading nearest each stamp within half an hour (a metrics store may align its
    reply to the hour when the timeline is not: --use-current-time), None for an hour with no
    reading (the statement was not exported then; 0 = measured zero)."""
    points = points or {}
    out: List[Optional[float]] = []
    for stamp in timeline:
        near = [(abs(ts - stamp), ts) for ts in points if abs(ts - stamp) < HOUR_MATCH_TOLERANCE_S]
        out.append(points[min(near)[1]] if near else None)
    return out


def _setting_entries(result: Dict[str, Any], newest: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
    """setting name -> value -> {setting, unit} from a `tlast_over_time(pgwatch_settings_configured…)`
    reply (either label form; each series' value is the time of its last reading). `newest`: only the
    values read last for each setting (a value it replaced in the lookback is not the value now)."""
    rows = []
    for item in ((result or {}).get("data") or {}).get("result") or []:
        metric = item.get("metric") or {}
        name = metric.get("setting_name") or metric.get("tag_setting_name") or ""
        if name not in SETTINGS:
            continue
        try:
            at = float((item.get("value") or [None, 0])[1])
        except (TypeError, ValueError):
            at = 0.0
        rows.append((name, at, metric))
    last = {}
    for name, at, _ in rows:
        last[name] = max(last.get(name, at), at)
    seen: Dict[str, Dict[str, Dict[str, str]]] = {}
    for name, at, metric in rows:
        if newest and at < last[name] - STEP_TOLERANCE_S:
            continue
        value = str(metric.get("setting_value") or metric.get("tag_setting_value") or "")
        unit = str(metric.get("unit") or metric.get("tag_unit") or "")
        entry = seen.setdefault(name, {}).setdefault(value, {"setting": value, "unit": ""})
        entry["unit"] = entry["unit"] or unit
    return seen


def settings_facts(now: Dict[str, Any], window: Dict[str, Any]) -> Dict[str, Any]:
    """`block_size` and `track_io_timing` as monitoring has them at the window's end (`now`: the
    last value per series), and whether `track_io_timing` had another value within the window
    (`window`: every value seen over it). A setting whose series disagree at the end is left out
    (a database-level value differs). Returns {"settings": {...}} or {"settings_error": reason}."""
    if not isinstance(now, dict) or now.get("status") != "success":
        return {"settings_error": "the monitoring query failed"}
    current = _setting_entries(now, newest=True)
    settings = {name: next(iter(values.values())) for name, values in current.items() if len(values) == 1}
    if not settings:
        return {"settings_error": "monitoring has no block_size or track_io_timing for this node"}
    timing = settings.get("track_io_timing")
    if timing is not None and isinstance(window, dict) and window.get("status") == "success":
        seen = _setting_entries(window).get("track_io_timing") or {}
        timing["changed_in_window"] = len(seen) > 1
    return {"settings": settings}


def timing_on(settings: Dict[str, Any]) -> Optional[bool]:
    """Whether track_io_timing is on at the window's end; None when not known."""
    value = ((settings or {}).get("track_io_timing") or {}).get("setting")
    if value in ("on", "1", "true"):
        return True
    if value in ("off", "0", "false"):
        return False
    return None


def scrape_targets(result: Dict[str, Any], cluster: Optional[str] = None
                   ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """The monitoring stack's scrape targets with `up` (1 = scraped, 0 = not), sorted by job and
    instance, from an instant `up` reply; a target labelled with another `cluster` is left out.
    `up` of the pgwatch job means its exporter was scraped, not that the database was reached.
    (targets, None) or (None, reason)."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return None, "the monitoring query failed"
    targets = []
    for item in (result.get("data") or {}).get("result") or []:
        metric = item.get("metric") or {}
        if cluster is not None and metric.get("cluster") not in (None, "", cluster):
            continue
        try:
            up = int(float((item.get("value") or [None, None])[1]))
        except (TypeError, ValueError):
            continue
        targets.append({"job": str(metric.get("job") or ""), "instance": str(metric.get("instance") or ""),
                        "up": up})
    if not targets:
        return None, "monitoring has no scrape targets"
    return sorted(targets, key=lambda t: (t["job"], t["instance"])), None


def deadlocks_by_database(series: Sequence[Dict[str, Any]], timeline: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    """Each database's deadlocks over the window (k004.counter_hours: total, per hour, busiest five
    minutes), from five-minute readings of pgwatch_db_stats_deadlocks. A database with fewer than two
    readings is left out."""
    by_db: Dict[str, List[Dict[str, Any]]] = {}
    for item in series or []:
        db = str((item.get("metric") or {}).get("datname") or "")
        if db:
            by_db.setdefault(db, []).append(item)
    out: Dict[str, Dict[str, Any]] = {}
    for db, items in sorted(by_db.items()):
        window = k004.counter_hours(items, timeline)
        if window is not None:
            out[db] = window
    return out
