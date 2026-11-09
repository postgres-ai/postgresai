"""D001 "Logging settings" on a monitoring box: what a logging decision needs.

The settings file (A003) comes from monitoring, which has no `pg_settings.context`
and keeps only some of the logging settings. What a change reaches (context), the
values stored per role and per database, which roles log in and how long their
sessions stay open, and how long the longest calls run live only in the catalog,
so the reporter reads them straight from the node, as for A005 (reporter/a005.py,
the same connection and read-only transactions). How many temporary files and
autovacuum runs a window has comes from monitoring: a delta over the window, not
the counter since its reset.

The platform's D001 report reads this file (platform-all). A fact that could not be
read is left out with its reason, never written as "none".
"""
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from psycopg2 import sql as pgsql

from reporter import a005
from reporter.g003 import PGSS_SCHEMA_SQL, _rows

#: The settings D001 reads (criteria D001-O1): every log_* and auto_explain.* setting,
#: both preload lists, and deadlock_timeout (the wait after which log_lock_waits logs).
SETTING_PATTERN = r"^(log_.*|logging_collector|auto_explain\..*|shared_preload_libraries|session_preload_libraries|deadlock_timeout)$"
#: The settings a role or database row can hold that D001 reports, plus the role's statement
#: limit: a statement cancelled by its limit is logged as an error, never by duration.
OVERRIDE_PATTERN = r"^(log_.*|auto_explain\..*|session_preload_libraries|statement_timeout)$"
#: Thresholds the longest calls are counted against, in ms: the starting values of an
#: automated report (1 s) and the current auto_explain default on managed platforms (10 s).
#: The node's own thresholds are added (log_min_duration_statement, auto_explain.log_min_duration),
#: and those stored per role and per database: a role's gap is counted against its own value.
BASE_THRESHOLDS_MS = (1000, 10000)
MAX_ROLES = 60

# reset_val: what a session starts with, so a reader's own `set` never reads back as the value.
SETTINGS_SQL = """
select name, reset_val as setting, coalesce(unit, '') as unit, category, context, vartype, source,
  boot_val, pending_restart
from pg_settings
where name ~ %s
order by name
"""

OVERRIDES_SQL = """
select r.rolname as role, d.datname as database, s.setconfig as config
from pg_db_role_setting as s
left join pg_roles as r on r.oid = s.setrole
left join pg_database as d on d.oid = s.setdatabase
order by r.rolname nulls first, d.datname nulls first
"""

# Client sessions per login role, and when the oldest began: a role setting reaches a
# session only at its next login, so a long-lived pooled session keeps the old value.
LOGIN_ROLES_SQL = """
select r.rolname as role, r.rolsuper as superuser,
  -- Monitoring logins are members of pg_monitor or a monitoring role, not application traffic.
  coalesce((select array_agg(g.rolname::text order by g.rolname)
            from pg_auth_members as m join pg_roles as g on g.oid = m.roleid
            where m.member = r.oid), '{}') as member_of,
  count(a.pid) filter (where a.backend_type = 'client backend') as sessions,
  min(a.backend_start) filter (where a.backend_type = 'client backend') as oldest_session_start,
  (array_agg(distinct a.application_name) filter (where a.backend_type = 'client backend'
                                                 and a.application_name <> ''))[1:5] as applications
from pg_roles as r
left join pg_stat_activity as a on a.usesysid = r.oid
where r.rolcanlogin and r.rolname !~ '^pg_'
group by r.oid, r.rolname, r.rolsuper
order by count(a.pid) filter (where a.backend_type = 'client backend') desc, r.rolname
limit 60
"""

# Without pg_read_all_stats, other roles' sessions read back with a null backend_type.
SEES_SESSIONS_SQL = "select pg_has_role(current_user, 'pg_read_all_stats', 'usage') as visible"

# to_jsonb: the column names change between major versions (max_time up to 12,
# stats_since and minmax_stats_since from 17). Groups, not calls: max_exec_time over a
# threshold proves at least one call over it; it never counts the calls (D001-O4).
STATEMENTS_SQL = pgsql.SQL("""
with s as (
  select p.userid, p.calls,
    coalesce((to_jsonb(p) ->> 'max_exec_time'), (to_jsonb(p) ->> 'max_time'))::float8 as max_ms,
    coalesce(to_jsonb(p) ->> 'minmax_stats_since', to_jsonb(p) ->> 'stats_since') as since
  from {pgss} as p
)
select r.rolname as role, count(*) as groups, sum(s.calls)::bigint as calls, max(s.max_ms) as longest_ms,
  min(s.since) as stats_since,
  (select jsonb_object_agg(t::text, (select count(*) from s as x where x.userid = s.userid and x.max_ms >= t)
                           order by t)
   from unnest({thresholds}::float8[]) as t) as groups_at_or_over,
  -- An upper bound of the calls at or over a threshold: every call of those groups.
  (select jsonb_object_agg(t::text, (select coalesce(sum(x.calls), 0) from s as x
                                     where x.userid = s.userid and x.max_ms >= t)
                           order by t)
   from unnest({thresholds}::float8[]) as t) as calls_of_groups_at_or_over
from s join pg_roles as r on r.oid = s.userid
group by r.rolname, s.userid
order by 4 desc nulls last
limit {limit}
""")

PGSS_INFO_SQL = pgsql.SQL("select dealloc, stats_reset from {info}")


def _in_scope(name: str, pattern: str) -> bool:
    return re.match(pattern, name) is not None


def _config(config: Optional[Sequence[str]], pattern: str) -> Dict[str, str]:
    """{name: value} of a setconfig array, the names D001 reads only."""
    out: Dict[str, str] = {}
    for item in config or []:
        name, _, value = str(item).partition("=")
        if _in_scope(name, pattern):
            out[name] = value
    return out


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def read_settings(conn) -> Dict[str, Dict[str, Any]]:
    rows = _rows(conn, SETTINGS_SQL, (SETTING_PATTERN,))
    return {
        row["name"]: {
            "setting": row["setting"],
            "unit": row["unit"],
            "category": row["category"],
            "context": row["context"],
            "vartype": row["vartype"],
            "source": row["source"],
            "is_default": row["setting"] == row["boot_val"],
            "pending_restart": bool(row["pending_restart"]),
        }
        for row in rows
    }


def read_overrides(conn) -> Dict[str, Any]:
    """Every stored role, database and role-in-database value of a D001 setting, and how
    many rows the catalog holds (so "none of them sets logging" is a read, not a guess)."""
    rows = _rows(conn, OVERRIDES_SQL)
    found = []
    for row in rows:
        for name, value in _config(row["config"], OVERRIDE_PATTERN).items():
            scope = ("role in database" if row["role"] and row["database"]
                     else "role" if row["role"] else "database" if row["database"] else "all")
            found.append({"scope": scope, "role": row["role"], "database": row["database"],
                          "setting": name, "value": value})
    return {"rows_read": len(rows), "values": found}


def read_login_roles(conn) -> List[Dict[str, Any]]:
    return [
        {
            "role": row["role"],
            "superuser": bool(row["superuser"]),
            "member_of": list(row["member_of"] or []),
            "sessions": int(row["sessions"] or 0),
            "oldest_session_start": _iso(row["oldest_session_start"]),
            "applications": list(row["applications"] or []),
        }
        for row in _rows(conn, LOGIN_ROLES_SQL)
    ]


def hide_sessions(roles: List[Dict[str, Any]]) -> None:
    """Unknown, never 0: this reader cannot see other roles' sessions."""
    for role in roles:
        role.update(sessions=None, oldest_session_start=None, applications=[])


_DURATION_SETTINGS = ("log_min_duration_statement", "auto_explain.log_min_duration", "log_min_duration_sample")
_MS_PER = {"us": 0.001, "ms": 1, "s": 1000, "min": 60_000, "h": 3_600_000, "d": 86_400_000}
#: Postgres takes no duration above this; a value past it can only be a placeholder.
_MAX_MS = 2147483647
#: Each threshold is a subquery per role: dozens of stored values are not worth a slow read. A value
#: past the cap gives the platform an upper bound, not a wrong count.
MAX_STORED_THRESHOLDS = 20


def _stored_ms(value: Any) -> Optional[int]:
    """Whole ms of a stored duration, rounded as Postgres rounds it: `5s`, `500ms`, `250` (a bare
    number is ms); None if it is no duration Postgres would hold."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*", str(value), re.ASCII)
    factor = _MS_PER.get(match.group(2) or "ms") if match else None
    if factor is None or len(match.group(1)) > 20:
        return None
    ms = round(round(float(match.group(1)) * factor, 6))
    return ms if ms <= _MAX_MS else None


def thresholds_of(settings: Dict[str, Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> List[int]:
    """The base thresholds plus the node's own positive duration thresholds, on the server and
    stored per role or database, in ms."""
    out = set(BASE_THRESHOLDS_MS)
    for name in _DURATION_SETTINGS:
        info = settings.get(name) or {}
        try:
            value = int(str(info.get("setting")))
        except ValueError:
            continue
        if value > 0 and (info.get("unit") or "ms") == "ms":
            out.add(value)
    stored = {_stored_ms(item.get("value")) for item in (overrides or {}).get("values") or []
              if item.get("setting") in _DURATION_SETTINGS}
    return sorted(out | set(sorted(v for v in stored if v)[:MAX_STORED_THRESHOLDS]))


def read_statements(conn, thresholds: Sequence[int]) -> Dict[str, Any]:
    """Per role: query groups, calls, the longest call and the groups whose longest call
    reached each threshold, since the statistics began. Raises when pg_stat_statements
    is not there."""
    schema = _rows(conn, PGSS_SCHEMA_SQL)
    if not schema:
        raise RuntimeError("pg_stat_statements is not installed")
    name = schema[0]["schema"]
    query = STATEMENTS_SQL.format(pgss=pgsql.Identifier(name, "pg_stat_statements"),
                                  thresholds=pgsql.Literal(list(thresholds)), limit=pgsql.Literal(MAX_ROLES))
    roles = [
        {
            "role": row["role"],
            "groups": int(row["groups"] or 0),
            "calls": int(row["calls"] or 0),
            "longest_call_ms": round(float(row["longest_ms"]), 1) if row["longest_ms"] is not None else None,
            "stats_since": _iso(row["stats_since"]),
            "groups_at_or_over_ms": {k: int(v) for k, v in (row["groups_at_or_over"] or {}).items()},
            "calls_of_groups_at_or_over_ms": {k: int(v) for k, v in (row["calls_of_groups_at_or_over"] or {}).items()},
        }
        for row in _rows(conn, query)
    ]
    out: Dict[str, Any] = {"thresholds_ms": list(thresholds), "by_role": roles}
    try:
        info = _rows(conn, PGSS_INFO_SQL.format(info=pgsql.Identifier(name, "pg_stat_statements_info")))
        if info:
            out["dealloc"] = int(info[0]["dealloc"] or 0)
            out["stats_reset"] = _iso(info[0]["stats_reset"])
    except Exception:  # pg_stat_statements_info is from 14; the rest stands
        pass
    return out


def catalog_facts(conn_str: str, *, opener: Callable = a005.open_database) -> Dict[str, Any]:
    """The catalog part of the D001 data of one node. Raises only when the node cannot
    be opened; a fact that cannot be read is left out with its reason."""
    secrets = a005._secrets(conn_str)
    conn = opener(conn_str)
    out: Dict[str, Any] = {}
    try:
        def fact(key: str, read: Callable[[], Any]) -> Any:
            try:
                value = read()
            except Exception as err:  # one fact: recorded, the rest go on
                out[f"{key}_error"] = a005.error_text(err, secrets)
                return None
            out[key] = value
            return value

        fact("read_as", lambda: _rows(conn, "select current_user as role, current_database() as database")[0])
        settings = fact("settings", lambda: read_settings(conn))
        overrides = fact("overrides", lambda: read_overrides(conn))
        roles = fact("login_roles", lambda: read_login_roles(conn))
        if roles is not None:
            try:
                sees = _rows(conn, SEES_SESSIONS_SQL)
                hidden = bool(sees) and sees[0].get("visible") is False
                reason = "the reading role has no pg_read_all_stats: other roles' sessions are not visible"
            except Exception as err:  # unknown, so never read as "no session"
                hidden, reason = True, a005.error_text(err, secrets)
            if hidden:
                hide_sessions(roles)
                out["sessions_error"] = reason
        fact("statements", lambda: read_statements(conn, thresholds_of(settings or {}, overrides)))
    finally:
        conn.close()
    return out


def _epoch_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def counter_window(series: Sequence[Dict[str, Any]], *, labels: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """The rise of a counter over a range query's readings: the sum of the rises between
    readings of each series. A reset (a fall) and a series' first value never count
    (PromQL's increase() can count the first value from 0). Series with the same `labels`
    values are merged first: a series that changes another label goes on with the same
    counter. None without two readings."""
    merged: Dict[tuple, Dict[float, float]] = {}
    for item in series or []:
        metric = item.get("metric") or {}
        key = tuple(str(metric.get(name) or "") for name in labels) if labels else \
            tuple(sorted((k, str(v)) for k, v in metric.items() if k != "__name__"))
        points = merged.setdefault(key, {})
        for epoch, value in item.get("values") or []:
            try:
                reading = float(value)
            except (TypeError, ValueError):
                continue
            points[float(epoch)] = max(points.get(float(epoch), reading), reading)
    total, start, end = 0.0, None, None
    for points in merged.values():
        ordered = sorted(points.items())
        if len(ordered) < 2:
            continue
        start = ordered[0][0] if start is None else min(start, ordered[0][0])
        end = ordered[-1][0] if end is None else max(end, ordered[-1][0])
        total += sum(after - before for (_, before), (_, after) in zip(ordered, ordered[1:]) if after > before)
    if start is None or end <= start:
        return None
    return {"start": _epoch_iso(start), "end": _epoch_iso(end), "days": round((end - start) / 86400, 2),
            "total": int(total)}
