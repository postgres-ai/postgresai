"""K004 "Top queries by temp bytes written" on a monitoring box: what a spill decision needs.

Beside the statements that write the most temporary-file bytes (with K003's window facts),
the K004 file carries the database-wide temporary-file counters over the same window (the
total, per hour, and the busiest five minutes), and the settings that bound or log a spill
with the values stored per role and per database. The per-statement and the database-wide
counters measure different things (pg_stat_statements vs pg_stat_database): they are kept
apart, never subtracted.

The platform's K004 report reads this file (platform-all). A fact that could not be read is
left out with its reason, never written as zero or "none".
"""
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from reporter import a005, d001
from reporter.g003 import _rows

#: The settings a spill decision reads (criteria K004-O5) and the role's statement limit (K004-O6:
#: beside the longest call). From monitoring (pg_settings: value, unit, context); the stored
#: role and database values come from the node.
SETTINGS = ("work_mem", "hash_mem_multiplier", "log_temp_files", "temp_file_limit", "pg_stat_statements.track",
            "statement_timeout")
SETTING_PATTERN = r"^(work_mem|hash_mem_multiplier|log_temp_files|temp_file_limit|pg_stat_statements[.]track|statement_timeout)$"
OVERRIDE_PATTERN = r"^(work_mem|hash_mem_multiplier|log_temp_files|temp_file_limit|statement_timeout)$"
#: Readings of the database counters every five minutes: the busiest five minutes (K004-O1).
STEP_S = 300


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def counter_hours(series: Sequence[Dict[str, Any]], timeline: Sequence[int]) -> Optional[Dict[str, Any]]:
    """A counter's rise over the hourly window `timeline` (each stamp the end of its hour), from
    five-minute readings: the total, the rise in each hour, and the largest rise per second between
    two readings STEP_S apart with the time it ended. A series' first reading never counts; after a fall
    (a reset) the next reading is the rise; series are merged by their highest reading per stamp (a label change goes on with
    the same counter). None without two readings."""
    points: Dict[float, float] = {}
    for item in series or []:
        for epoch, value in item.get("values") or []:
            try:
                reading = float(value)
            except (TypeError, ValueError):
                continue
            points[float(epoch)] = max(points.get(float(epoch), reading), reading)
    ordered = sorted(points.items())
    if len(ordered) < 2 or not timeline:
        return None
    hourly = [0.0] * len(timeline)
    total, peak, peak_at = 0.0, None, None
    for (t0, before), (t1, after) in zip(ordered, ordered[1:]):
        # A fall to under half is a reset: the counter started again from zero, so the reading after it is its
        # rise (as PromQL increase() counts it). A smaller dip is two merged series of one counter (one missed a
        # stamp): no rise.
        rise = (after if after < before / 2 else 0.0) if after < before else after - before
        total += rise
        for i, stamp in enumerate(timeline):  # the hour (stamp - 3600, stamp] the reading ends in
            if stamp - 3600 < t1 <= stamp:
                hourly[i] += rise
                break
        if 0 < t1 - t0 <= 2 * STEP_S:  # a gap is no five-minute rate
            rate = rise / (t1 - t0)
            if peak is None or rate > peak:
                peak, peak_at = rate, t1
    out: Dict[str, Any] = {"start": _iso(ordered[0][0]), "end": _iso(ordered[-1][0]), "total": total,
                           "hourly": hourly}
    if peak is not None:
        out["peak_per_s"] = peak
        out["peak_at"] = _iso(peak_at)
    return out


def settings_from_monitoring(result: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """The SETTINGS from an instant pgwatch_settings_configured reply (value, unit, context), and the names
    left out because their series disagree: the metric has one series per monitored database (a database's
    own value) and keeps a value changed within the range, so there the node's value is not known."""
    seen: Dict[str, Dict[str, Dict[str, str]]] = {}
    for item in ((result or {}).get("data") or {}).get("result") or []:
        metric = item.get("metric") or {}
        name = metric.get("setting_name") or metric.get("tag_setting_name") or ""
        if name in SETTINGS:
            value = str(metric.get("setting_value") or metric.get("tag_setting_value") or "")
            unit = str(metric.get("unit") or metric.get("tag_unit") or "")  # the older form keeps the tag_ prefix
            entry = seen.setdefault(name, {}).setdefault(value, {"setting": value, "unit": "", "context": ""})
            entry["unit"] = entry["unit"] or unit
            entry["context"] = entry["context"] or str(metric.get("context") or metric.get("tag_context") or "")
    out = {name: next(iter(values.values())) for name, values in seen.items() if len(values) == 1}
    return out, sorted(name for name, values in seen.items() if len(values) > 1)


def node_facts(conn_str: Optional[str], *, opener: Callable = a005.open_database) -> Dict[str, Any]:
    """From the node: the SETTINGS as a session starts with them, with pg_settings.context (what a change
    reaches: K004-R5), and the stored role and database values of OVERRIDE_PATTERN ({"values": [...],
    "rows_read": n}). A fact that cannot be read is "<key>_error" with its reason."""
    if not conn_str:
        return {"node_error": "no connection for this node in the sources file"}
    secrets = a005._secrets(conn_str)
    try:
        conn = opener(conn_str)
    except Exception as err:
        return {"node_error": a005.error_text(err, secrets)}
    out: Dict[str, Any] = {}
    try:
        for key, read in (("settings", lambda: d001.read_settings(conn, SETTING_PATTERN)),
                          ("overrides", lambda: d001.read_overrides(conn, OVERRIDE_PATTERN)),
                          ("platform_allowed", lambda: platform_allowed(conn))):
            try:
                out[key] = read()
            except Exception as err:  # one fact: recorded, the other goes on
                out[f"{key}_error"] = a005.error_text(err, secrets)
    finally:
        conn.close()
    if out.get("platform_allowed") is None:
        out.pop("platform_allowed", None)  # not a managed platform with such a list
    if "settings" in out:
        out["settings"] = {name: {"setting": str(info["setting"]), "unit": str(info["unit"] or ""),
                                  "context": str(info["context"] or "")}
                           for name, info in out["settings"].items()}
    return out


#: A managed platform's list of superuser settings its customer's role may set (Supabase: supautils).
ALLOWED_SQL = "select current_setting('supautils.privileged_role_allowed_configs', true) as allowed"


def platform_allowed(conn) -> Optional[List[str]]:
    """Which superuser-level SETTINGS the platform lets its customer's role set (supautils, Supabase:
    entries are names or a "prefix.*"), or None when the node has no such list."""
    rows = _rows(conn, ALLOWED_SQL)
    raw = rows[0].get("allowed") if rows else None
    if not raw:
        return None
    entries = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [n for n in SETTINGS
            if any(n == e or (e.endswith(".*") and n.startswith(e[:-1])) for e in entries)]
