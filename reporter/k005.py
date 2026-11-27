"""K005 "Top queries by WAL generation" on a monitoring box: what a WAL decision needs.

Beside the statements that write the most WAL (with K003's window facts and their full-page
images and records), the K005 file carries, over the same hourly window: the server's WAL (the
rise of the WAL position: total, per hour, busiest five minutes), each database's size at the
start and the end, checkpoints timed and requested, the archiver (segments archived, failures,
the longest time since the last archived segment), the largest `pg_wal` directory, the tables
with the most updates and their HOT share, and the samples waiting on WAL writes; from the node,
the WAL and checkpoint settings with their context, the replication slots and the indexes of the
most-updated tables.

The WAL position also moves when a segment is switched before it is full (`archive_timeout`): a
quiet server then shows one segment per `archive_timeout` seconds. The file keeps the archiver's
segment count and the setting so a reader can tell that floor from written WAL.

The platform's K005 report reads this file (platform-all). A fact that could not be read is left
out with its reason ("<key>_error"), never written as zero or "none".
"""
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from reporter import a005, d001
from reporter.g003 import _rows
from reporter.k004 import STEP_S, _iso, counter_hours

#: The settings a WAL decision reads (criteria K005-O6, R2) and the switch floor (archive_timeout).
SETTINGS = ("checkpoint_timeout", "max_wal_size", "min_wal_size", "checkpoint_completion_target",
            "wal_compression", "wal_level", "full_page_writes", "wal_buffers", "wal_keep_size",
            "max_slot_wal_keep_size", "archive_mode", "archive_timeout", "log_checkpoints", "wal_segment_size")
SETTING_PATTERN = "^(" + "|".join(SETTINGS) + ")$"
#: Tables ranked by updates in the window (K005-O4): the top ones, with their indexes from the node.
TOP_TABLES = 3
#: Candidates asked for exactly (the instant ranking can miss a table whose series was renamed).
TABLE_CANDIDATES = 6
#: Wait events of a session waiting for WAL to be written or flushed (K005-O7).
#: PostgreSQL 17 renamed the IO events (WALSync -> WalSync, WALWrite -> WalWrite; the LWLock is WALWrite);
#: 12 and older name the LWLock WALWriteLock.
WAL_WAIT_EVENTS = ("WalSync", "WalWrite", "WALSync", "WALWrite", "WALWriteLock")
#: Not session work: the same filter as the weekly load chart (platform-all checkup_loop_weekly).
IDLE_WAIT_TYPES = "Activity|Client|Timeout"
INDEXES_PER_TABLE = 40


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def merged_points(series: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """One series from several (a label change goes on with the same counter): the highest reading per stamp."""
    points: Dict[float, float] = {}
    for item in series or []:
        for epoch, value in item.get("values") or []:
            reading = _num(value)
            if reading is not None:
                points[float(epoch)] = max(points.get(float(epoch), reading), reading)
    return [[t, v] for t, v in sorted(points.items())]


def gauge_ends(series: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A gauge's first and last reading in the window, and its highest (database size, pg_wal size)."""
    points = merged_points(series)
    if not points:
        return None
    high = max(points, key=lambda p: p[1])
    return {"start": _iso(points[0][0]), "end": _iso(points[-1][0]), "first": points[0][1],
            "last": points[-1][1], "max": high[1], "max_at": _iso(high[0])}


def gauge_max(series: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The highest reading of a gauge in the window and when (the longest time since an archived segment)."""
    out = gauge_ends(series)
    return {"max": out["max"], "max_at": out["max_at"]} if out else None


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(B|kB|KB|MB|GB)?\s*$")
_UNITS = {None: 1, "B": 1, "kB": 1024, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}


def segment_bytes(series: Sequence[Dict[str, Any]]) -> Optional[int]:
    """The WAL segment size from the archiver metric's label ("16 MB"), or None."""
    for item in series or []:
        label = ((item.get("metric") or {}).get("wal_segment_size") or "")
        m = _SIZE.match(str(label))
        if m:
            return int(float(m.group(1)) * _UNITS[m.group(2)])
    return None


def window_counter(series: Sequence[Dict[str, Any]], timeline: Sequence[int]) -> Optional[Dict[str, Any]]:
    """counter_hours over merged series (one counter under several label sets)."""
    return counter_hours([{"values": merged_points(series)}], timeline)


#: The span the most WAL written in it is reported for (K005-R2: WAL between two 15-minute checkpoints).
PEAK_SPAN_S = 900


def peak_span(series: Sequence[Dict[str, Any]], span_s: int = PEAK_SPAN_S) -> Optional[Dict[str, Any]]:
    """The largest rise of a never-falling counter (the WAL position) over `span_s` seconds of readings, and when
    that span ended; None without readings that far apart. A fall (a reset) ends the comparison there."""
    points = merged_points(series)
    best, best_at = None, None
    j = 0
    for i, (t1, v1) in enumerate(points):
        while j < i and t1 - points[j][0] > span_s:
            j += 1
        t0, v0 = points[j]
        if t1 - t0 < span_s * 0.9 or v1 < v0:
            continue
        if any(points[k + 1][1] < points[k][1] for k in range(j, i)):
            continue
        if best is None or v1 - v0 > best:
            best, best_at = v1 - v0, t1
    return {"bytes": best, "seconds": span_s, "end": _iso(best_at)} if best is not None else None


def wait_share(samples: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """WAL-write wait samples against all session samples, from an hourly `sum by (wait_event)
    (sum_over_time(pgwatch_wait_events_total[1h]))` range reply. None with no samples at all."""
    wal, total, events = 0.0, 0.0, {}
    for item in samples or []:
        event = str((item.get("metric") or {}).get("wait_event") or "")
        value = sum(v for v in (_num(x[1]) for x in item.get("values") or []) if v is not None)
        total += value
        if event in WAL_WAIT_EVENTS:
            wal += value
            events[event] = events.get(event, 0.0) + value
    if total <= 0:
        return None
    return {"wal_samples": wal, "all_samples": total, "wal_events": events}


SLOTS_SQL = """
select slot_name, slot_type, active, wal_status,
       pg_wal_lsn_diff(case when pg_is_in_recovery() then pg_last_wal_receive_lsn() else pg_current_wal_lsn() end,
                       restart_lsn)::bigint as retained_bytes
from pg_replication_slots order by slot_name
"""

INDEXES_SQL = """
select quote_ident(n.nspname) as schema, quote_ident(c.relname) as "table", i.relname as index, pg_relation_size(i.oid)::bigint as bytes,
       x.indisunique as is_unique, x.indisprimary as is_primary, pg_get_expr(x.indpred, x.indrelid) as predicate,
       s.idx_scan as scans
from pg_index x
join pg_class c on c.oid = x.indrelid
join pg_namespace n on n.oid = c.relnamespace
join pg_class i on i.oid = x.indexrelid
left join pg_stat_all_indexes s on s.indexrelid = x.indexrelid
where quote_ident(n.nspname) = any(%s) and quote_ident(c.relname) = any(%s)
order by n.nspname, c.relname, i.relname
"""

#: Which of the wanted tables the node has as a plain table (indexes on the table itself): a partition root keeps its
#: indexes on the partitions (pgwatch sums partitions under the root), and a dropped table is not found.
TABLES_SQL = """
select quote_ident(n.nspname) as schema, quote_ident(c.relname) as "table",
       c.relkind = 'r' and not exists (select 1 from pg_inherits h where h.inhparent = c.oid) as plain
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where quote_ident(n.nspname) = any(%s) and quote_ident(c.relname) = any(%s)
"""


def read_slots(conn) -> List[Dict[str, Any]]:
    return [{"slot_name": str(r["slot_name"]), "slot_type": str(r["slot_type"]), "active": bool(r["active"]),
             "wal_status": str(r["wal_status"] or ""),
             **({"retained_bytes": float(r["retained_bytes"])} if r["retained_bytes"] is not None else {})}
            for r in _rows(conn, SLOTS_SQL)]


def read_indexes(conn, tables: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """The indexes of each `tables` entry ({"schema", "table"}, as pgwatch labels them: quote_ident()ed) as
    "schema.table" -> [index, ...]. A plain table with no index is an empty list; a partition root or a table the
    node does not have is left out (REV r2: [] would claim "no index" for what was not read)."""
    if not tables:
        return {}
    wanted = {(t["schema"], t["table"]) for t in tables}
    names = (sorted({s for s, _ in wanted}), sorted({t for _, t in wanted}))
    plain = {(r["schema"], r["table"]) for r in _rows(conn, TABLES_SQL, names) if r["plain"]} & wanted
    rows = _rows(conn, INDEXES_SQL, names)
    out: Dict[str, List[Dict[str, Any]]] = {f"{s}.{t}": [] for s, t in plain}
    for r in rows:
        if (r["schema"], r["table"]) not in plain:
            continue
        entry = {"name": str(r["index"]), "bytes": float(r["bytes"] or 0), "unique": bool(r["is_unique"]),
                 "primary": bool(r["is_primary"])}
        if r["predicate"]:
            entry["predicate"] = str(r["predicate"])[:200]
        if r["scans"] is not None:
            entry["scans"] = float(r["scans"])
        out.setdefault(f'{r["schema"]}.{r["table"]}', []).append(entry)
    return {k: v[:INDEXES_PER_TABLE] for k, v in out.items()}


def node_facts(conn_str: Optional[str], tables: Sequence[Dict[str, Any]] = (), *,
               opener: Callable = a005.open_database) -> Dict[str, Any]:
    """From the node: the SETTINGS with pg_settings.context (a reload or a restart: K005-R2), the replication
    slots with the WAL each keeps, and the indexes of `tables` that live in the node's database (each entry's
    "database" must be the connection's). A fact that cannot be read is "<key>_error" with its reason."""
    if not conn_str:
        return {"node_error": "no connection for this node in the sources file"}
    secrets = a005._secrets(conn_str)
    try:
        conn = opener(conn_str)
    except Exception as err:
        return {"node_error": a005.error_text(err, secrets)}
    out: Dict[str, Any] = {}
    try:
        try:
            current = _rows(conn, "select current_database() as db")[0]["db"]
        except Exception as err:
            current = None
            out["indexes_error"] = a005.error_text(err, secrets)
        here = [t for t in tables if current is not None and t.get("database") == current]
        reads = [("settings", lambda: d001.read_settings(conn, SETTING_PATTERN)),
                 ("slots", lambda: read_slots(conn))]
        if here:
            reads.append(("indexes", lambda: read_indexes(conn, here)))
        for key, read in reads:
            try:
                out[key] = read()
            except Exception as err:  # one fact: recorded, the other goes on
                out[f"{key}_error"] = a005.error_text(err, secrets)
    finally:
        conn.close()
    if "settings" in out:
        out["settings"] = {name: {"setting": str(info["setting"]), "unit": str(info["unit"] or ""),
                                  "context": str(info["context"] or "")}
                           for name, info in out["settings"].items()}
    if tables and current is not None and not here:
        out.setdefault("indexes_error", "the most-updated tables are in another database than the node's connection")
    elif here and "indexes" in out and len(out["indexes"]) < len(tables):
        out["indexes_error"] = ("indexes read only for plain tables in the node's database: a partition root keeps "
                                "them on its partitions, and a table dropped since the window is not found")
    return out


def settings_from_monitoring(result: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """The SETTINGS from an instant pgwatch_settings_configured reply when the node cannot be read; a name
    whose series disagree is left out (a database's own value, or a change within the range)."""
    seen: Dict[str, Dict[str, Dict[str, str]]] = {}
    for item in ((result or {}).get("data") or {}).get("result") or []:
        metric = item.get("metric") or {}
        name = metric.get("setting_name") or metric.get("tag_setting_name") or ""
        if name in SETTINGS:
            value = str(metric.get("setting_value") or metric.get("tag_setting_value") or "")
            unit = str(metric.get("unit") or metric.get("tag_unit") or "")
            entry = seen.setdefault(name, {}).setdefault(value, {"setting": value, "unit": "", "context": ""})
            entry["unit"] = entry["unit"] or unit
            entry["context"] = entry["context"] or str(metric.get("context") or metric.get("tag_context") or "")
    return {name: next(iter(values.values())) for name, values in seen.items() if len(values) == 1}


__all__ = ["SETTINGS", "STEP_S", "TOP_TABLES", "TABLE_CANDIDATES", "WAL_WAIT_EVENTS", "IDLE_WAIT_TYPES",
           "counter_hours", "window_counter", "gauge_ends", "gauge_max", "segment_bytes", "wait_share", "node_facts",
           "settings_from_monitoring", "merged_points"]
