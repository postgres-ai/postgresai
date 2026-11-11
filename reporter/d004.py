"""D004 "Observability extensions" on a monitoring box: what an extension decision needs.

Monitoring keeps the pg_stat_statements settings and the calls of the heaviest statements
(the pgwatch metric keeps the top 100 per database), not the extension's own state. Whether
an extension is available, which version is installed in which database against its
default version, how many entries pg_stat_statements holds against its maximum, how many
deallocation rounds it ran since its reset, and the values stored per role and database
live only in the catalog, so the reporter reads them straight from the node, as for A005
(reporter/a005.py: the same connection and read-only transactions). Whether the statistics
grow (D004-O2) comes from monitoring: the rise of the monitored statements' calls over a
window.

The platform's D004 report reads this file (platform-all). A fact that could not be read is
left out with its reason, never written as "none".
"""
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from psycopg2 import sql as pgsql

from reporter import a005
from reporter.d001 import _iso, read_overrides, read_settings
from reporter.g003 import _rows

#: The observability extensions D004 covers (criteria D004 §1): statement, wait and plan statistics.
#: auto_explain is D001's and is a library, not an extension.
EXTENSIONS = ("pg_stat_statements", "pg_stat_kcache", "pg_wait_sampling", "pg_stat_monitor",
              "pg_qualstats", "pg_store_plans")
#: Their settings and the two preload lists (D004-S1). track_* are D005's.
SETTING_PATTERN = (r"^(pg_stat_statements|pg_stat_kcache|pg_wait_sampling|pg_stat_monitor|pg_qualstats|pg_store_plans)\..*$"
                   r"|^(shared_preload_libraries|session_preload_libraries)$")
OVERRIDE_PATTERN = r"^(pg_stat_statements|pg_stat_kcache|pg_wait_sampling|pg_stat_monitor|pg_qualstats|pg_store_plans)\..*$"

AVAILABLE_SQL = """
select name, default_version, installed_version
from pg_available_extensions
where name = any(%s)
order by name
"""

INSTALLED_SQL = """
select e.extname as name, e.extversion as version, n.nspname as schema
from pg_extension as e join pg_namespace as n on n.oid = e.extnamespace
where e.extname = any(%s)
order by e.extname
"""

# The databases to read installed extensions in: no per-role privilege scan (A005's list has one), no
# template (an open connection to template1 makes a concurrent CREATE DATABASE fail).
DATABASES_SQL = """
select d.datname, d.datname = current_database() as is_current,
  has_database_privilege(d.oid, 'CONNECT') as can_connect
from pg_database as d
where d.datallowconn and not d.datistemplate
order by d.datname
"""

# Entries per database (the view covers every database of the node). Churn shapes: an entry
# called once, and a statement on a temporary table, which gets a new query id per table
# (docs: "pg_stat_statements" > queryid). The query text needs pg_read_all_stats; without it
# the text is "<insufficient privilege>" and the temporary-table count is left out.
ENTRIES_SQL = pgsql.SQL("""
select d.datname as database, count(*)::bigint as entries, coalesce(sum(p.calls), 0)::bigint as calls,
  (count(*) filter (where p.calls = 1))::bigint as called_once,
  -- a heuristic: the text mentions temp, temporary or pg_temp[_N]
  (count(*) filter (where p.query ~* '\\m(pg_temp(_[0-9]+)?|temp|temporary)\\M'))::bigint as temp_table_text,
  bool_and(p.query <> '<insufficient privilege>') as texts_visible
from {pgss} as p left join pg_database as d on d.oid = p.dbid
group by d.datname
order by count(*) desc
""")

INFO_SQL = pgsql.SQL("select dealloc, stats_reset, now() as read_at from {info}")

def read_available(conn) -> Dict[str, Dict[str, Any]]:
    """{name: {default_version}} of the D004 extensions the instance offers; absent = not offered."""
    return {row["name"]: {"default_version": row["default_version"]}
            for row in _rows(conn, AVAILABLE_SQL, (list(EXTENSIONS),))}


def read_installed(conn) -> Dict[str, Dict[str, str]]:
    return {row["name"]: {"version": row["version"], "schema": row["schema"]}
            for row in _rows(conn, INSTALLED_SQL, (list(EXTENSIONS),))}


def read_statements(conn, schema: str) -> Dict[str, Any]:
    """Entries against the maximum, per database, and the deallocation rounds since the reset."""
    out: Dict[str, Any] = {"schema": schema}
    rows = _rows(conn, ENTRIES_SQL.format(pgss=pgsql.Identifier(schema, "pg_stat_statements")))
    visible = all(row["texts_visible"] is not False for row in rows)
    out["entries"] = sum(int(row["entries"] or 0) for row in rows)
    out["by_database"] = [
        {
            "database": row["database"],
            "entries": int(row["entries"] or 0),
            "calls": int(row["calls"] or 0),
            "called_once": int(row["called_once"] or 0),
            **({"temp_table_text": int(row["temp_table_text"] or 0)} if visible else {}),
        }
        for row in rows
    ]
    if not visible:
        out["temp_table_text_error"] = "the reading role has no pg_read_all_stats: statement texts are hidden"
    try:
        info = _rows(conn, INFO_SQL.format(info=pgsql.Identifier(schema, "pg_stat_statements_info")))
    except Exception as err:  # pg_stat_statements_info is from 14 (extension 1.9); the rest stands
        out["dealloc_error"] = a005.error_text(err)
    else:
        if info:
            out["dealloc"] = int(info[0]["dealloc"] or 0)
            out["stats_reset"] = _iso(info[0]["stats_reset"])
            out["read_at"] = _iso(info[0]["read_at"])
            if isinstance(info[0]["stats_reset"], datetime) and isinstance(info[0]["read_at"], datetime):
                # The span the rounds cover, so a rate a day is derived from figures the file holds.
                span = info[0]["read_at"] - info[0]["stats_reset"]
                out["days_since_reset"] = round(span.total_seconds() / 86400, 2)
    return out


def catalog_facts(conn_str: str, *, opener: Callable = a005.open_database,
                  clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """The catalog part of the D004 data of one node. Raises only when the node cannot be
    opened or its databases listed; a fact that cannot be read is left out with its reason."""
    secrets = a005._secrets(conn_str)
    started = clock()
    conn = opener(conn_str)
    out: Dict[str, Any] = {}
    try:
        def fact(key: str, read: Callable[[], Any], into: Optional[Dict[str, Any]] = None) -> Any:
            target = out if into is None else into
            try:
                value = read()
            except Exception as err:  # one fact: recorded, the rest go on
                target[f"{key}_error"] = a005.error_text(err, secrets)
                return None
            target[key] = value
            return value

        fact("read_as", lambda: _rows(conn, "select current_user as role, current_database() as database")[0])
        fact("settings", lambda: read_settings(conn, SETTING_PATTERN))
        fact("available", lambda: read_available(conn))
        fact("overrides", lambda: read_overrides(conn, OVERRIDE_PATTERN))

        # Installed is per database: an extension is created in each database that uses it.
        databases: Dict[str, Dict[str, Any]] = {}
        pgss_db: Optional[str] = None
        pgss_schema: Optional[str] = None
        opened = 0
        listed = fact("databases_listed", lambda: a005._all(conn, DATABASES_SQL))
        out.pop("databases_listed", None)
        if listed is None:  # the facts read so far stand; the list's reason is recorded
            out["databases_error"] = out.pop("databases_listed_error")
            listed = []
        for row in listed:
            entry: Dict[str, Any] = {}
            databases[row["datname"]] = entry
            try:
                if row["is_current"]:
                    entry["installed"] = read_installed(conn)
                else:
                    if not row["can_connect"]:
                        raise RuntimeError("the checkup role has no CONNECT privilege on this database")
                    if opened >= a005.MAX_DATABASES:
                        raise RuntimeError(f"not read: more than {a005.MAX_DATABASES} databases")
                    if clock() - started > a005.TIME_BUDGET_S:
                        raise RuntimeError("not read: the time budget ran out")
                    opened += 1
                    other = opener(conn_str, row["datname"])
                    try:
                        entry["installed"] = read_installed(other)
                    finally:
                        other.close()
            except Exception as err:  # one database: recorded, the rest go on
                entry["error"] = a005.error_text(err, secrets)
                continue
            found = entry["installed"].get("pg_stat_statements")
            if found and (pgss_db is None or row["is_current"]):
                pgss_db, pgss_schema = row["datname"], found["schema"]
        if "databases_error" not in out:
            out["databases"] = databases

        # The view is the same in every database it is created in: read it once, in the
        # connected database when it is there.
        if pgss_db is None and "databases_error" not in out:
            out["statements_error"] = "pg_stat_statements is not created in any database read"
        elif pgss_db is not None:
            if pgss_db == (out.get("read_as") or {}).get("database"):
                fact("statements", lambda: read_statements(conn, pgss_schema))
            else:
                try:
                    other = opener(conn_str, pgss_db)
                except Exception as err:
                    out["statements_error"] = a005.error_text(err, secrets)
                else:
                    try:
                        fact("statements", lambda: read_statements(other, pgss_schema))
                    finally:
                        other.close()
            if "statements" in out:
                out["statements"]["database"] = pgss_db
    finally:
        conn.close()
    return out


def per_round(entries: int) -> int:
    """Entries one deallocation round removes when it runs at `entries` (pg_stat_statements.c
    entry_dealloc, PostgreSQL 13-18): max(10, entries * 5 / 100), at most all of them. A round
    runs when the entries reach `pg_stat_statements.max`, so pass the maximum."""
    return min(entries, max(10, entries * 5 // 100))


def monitored_calls_window(series: List[Dict[str, Any]], counter_window: Callable) -> List[Dict[str, Any]]:
    """Per database: the rise of the monitored statements' calls over the window (D004-O2).
    The metric keeps the top 100 statements per database by execution time, so a rise
    confirms collection for that scope only, never that no statement was lost."""
    by_db: Dict[str, List[Dict[str, Any]]] = {}
    for item in series or []:
        by_db.setdefault(str((item.get("metric") or {}).get("datname") or ""), []).append(item)
    out = []
    for database, items in sorted(by_db.items()):
        window = counter_window(items, labels=("datname", "queryid"))
        if window:
            window["database"] = database
            window["statements"] = len({str((i.get("metric") or {}).get("queryid")) for i in items})
            out.append(window)
    return out
