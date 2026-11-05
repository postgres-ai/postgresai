"""G003 "Timeouts, locks, deadlocks" on a monitoring box: what a timeout decision needs.

Which roles have which limit lives only in the catalog, so the reporter reads it
straight from the node, as for A005 (reporter/a005.py, the same connection and
read-only transactions): the values stored per role and per database, the longest
completed call per role, and the procedures and functions that commit or set a
timeout. How often deadlocks abort work and who blocks whom come from monitoring:
a delta over a window (the counter since the last reset says nothing about now)
and the lock-wait samples.

The platform's G003 report reads this file (platform-all). A fact that could not
be read is left out with its reason, never written as "none".
"""
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from psycopg2 import sql as pgsql

from reporter import a005

#: Settings G003 reports. A role or database row keeps only these, not its search_path.
TIMEOUT_SETTINGS = (
    "statement_timeout",
    "transaction_timeout",
    "lock_timeout",
    "idle_in_transaction_session_timeout",
    "idle_session_timeout",
    "deadlock_timeout",
    "log_lock_waits",
)
MAX_ROLES = 60
MAX_ROUTINES = 30
MAX_LOCK_PAIRS = 15
QUERY_CHARS = 120

ROLES_SQL = """
select r.rolname as role, r.rolcanlogin as can_login, r.rolsuper as superuser, r.rolbypassrls as bypass_rls,
  coalesce((select array_agg(g.rolname::text order by g.rolname)
            from pg_auth_members as m join pg_roles as g on g.oid = m.roleid
            where m.member = r.oid), '{}') as member_of,
  (select s.setconfig from pg_db_role_setting as s
   where s.setrole = r.oid and s.setdatabase = 0) as config
from pg_roles as r
where r.rolname !~ '^pg_'
order by r.rolname
"""

ROLE_DATABASE_SQL = """
select r.rolname as role, d.datname as database, s.setconfig as config
from pg_db_role_setting as s
join pg_roles as r on r.oid = s.setrole
join pg_database as d on d.oid = s.setdatabase
order by r.rolname, d.datname
"""

DATABASE_SQL = """
select d.datname as database, s.setconfig as config
from pg_db_role_setting as s
join pg_database as d on d.oid = s.setdatabase
where s.setrole = 0
order by d.datname
"""

# reset_val: what a session starts with, so a reader's own `set` never reads back as the value.
SERVER_SQL = """
select name, reset_val as setting, coalesce(unit, '') as unit, source
from pg_settings
where name = any(%s)
order by name
"""

PGSS_SCHEMA_SQL = """
select n.nspname as schema
from pg_extension as e join pg_namespace as n on n.oid = e.extnamespace
where e.extname = 'pg_stat_statements'
"""

# to_jsonb: the column names change between major versions (max_time up to 12,
# stats_since and minmax_stats_since from 17).
CALLS_SQL = pgsql.SQL("""
with s as (
  select p.userid, p.queryid, p.calls, p.query,
    coalesce((to_jsonb(p) ->> 'max_exec_time'), (to_jsonb(p) ->> 'max_time'))::float8 as max_ms,
    coalesce(to_jsonb(p) ->> 'minmax_stats_since', to_jsonb(p) ->> 'stats_since') as since
  from {pgss} as p
)
select r.rolname as role, sum(s.calls)::bigint as calls,
  (array_agg(s.max_ms order by s.max_ms desc nulls last))[1] as longest_ms,
  (array_agg(s.queryid::text order by s.max_ms desc nulls last))[1] as longest_queryid,
  -- Utility text stays verbatim in pg_stat_statements, passwords included: only plain queries.
  (array_agg(case when s.query ~* '^\\s*(insert|update|delete|merge|values)\\M'
                       or (s.query ~* '^\\s*(select|with)\\M' and s.query !~* '\\minto\\M')
                  then left(s.query, {chars}) end order by s.max_ms desc nulls last))[1] as longest_query,
  (array_agg(s.since order by s.max_ms desc nulls last))[1] as longest_since,
  min(s.since) as stats_since
from s join pg_roles as r on r.oid = s.userid
group by r.rolname
order by 3 desc nulls last
limit {limit}
""")

# Routines that decide which timeout bounds a long job (G003-O4): procedures, and
# any routine that sets a timeout of its own. An extension's routines are not the
# customer's. Reads the connected database.
ROUTINES_SQL = """
select n.nspname as schema, p.proname as name, p.prokind as kind, pg_get_userbyid(p.proowner) as owner,
  b.src ~* '\\mcommit\\M' as commits,
  b.src ~* '\\m(query_canceled|57014)\\M' as catches_cancel,
  (select array_agg(c) from unnest(p.proconfig) as c
   where split_part(c, '=', 1) = any(%(names)s)) as config
from pg_proc as p
join pg_namespace as n on n.oid = p.pronamespace
-- The code without its comments and `on commit drop`, which are not a commit.
cross join lateral (select regexp_replace(regexp_replace(regexp_replace(p.prosrc, '/\\*.*?\\*/', '', 'g'),
                                                         '--[^\\n]*', '', 'g'),
                                          'on\\s+commit\\s+\\w+(\\s+\\w+)?', '', 'gi') as src) as b
where n.nspname not in ('pg_catalog', 'information_schema') and n.nspname !~ '^pg_(toast|temp)'
  and (p.prokind = 'p' or exists (select 1 from unnest(p.proconfig) as c
                                  where split_part(c, '=', 1) = any(%(names)s)))
  and not exists (select 1 from pg_depend as d
                  where d.classid = 'pg_proc'::regclass and d.objid = p.oid and d.deptype = 'e')
order by n.nspname, p.proname
limit %(limit)s
"""


def _settings(config: Optional[Sequence[str]]) -> Dict[str, str]:
    """{name: value} of a setconfig array, G003's settings only."""
    out: Dict[str, str] = {}
    for item in config or []:
        name, _, value = str(item).partition("=")
        if name in TIMEOUT_SETTINGS:
            out[name] = value
    return out


def _rows(conn, sql: Any, params: Any = None) -> List[Dict[str, Any]]:
    def fetch(cur):
        names = [c.name for c in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    # a005._run takes the query as given; the parameters go into one prepared text.
    text = sql
    if params is not None:
        with conn.cursor() as cur:
            text = cur.mogrify(sql, params).decode()
    elif isinstance(sql, pgsql.Composable):
        text = sql.as_string(conn)
    return a005._run(conn, text, fetch)


def read_roles(conn) -> List[Dict[str, Any]]:
    per_database: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in _rows(conn, ROLE_DATABASE_SQL):
        values = _settings(row["config"])
        if values:
            per_database.setdefault(row["role"], {})[row["database"]] = values
    return [
        {
            "role": row["role"],
            "can_login": bool(row["can_login"]),
            "superuser": bool(row["superuser"]),
            "bypass_rls": bool(row["bypass_rls"]),
            "member_of": list(row["member_of"] or []),
            "settings": _settings(row["config"]),
            "database_settings": per_database.get(row["role"], {}),
        }
        for row in _rows(conn, ROLES_SQL)
    ]


def read_calls(conn) -> Dict[str, Dict[str, Any]]:
    """{role: calls, longest completed call}; raises when pg_stat_statements is not there."""
    schema = _rows(conn, PGSS_SCHEMA_SQL)
    if not schema:
        raise RuntimeError("pg_stat_statements is not installed")
    query = CALLS_SQL.format(pgss=pgsql.Identifier(schema[0]["schema"], "pg_stat_statements"),
                             chars=pgsql.Literal(QUERY_CHARS), limit=pgsql.Literal(MAX_ROLES))
    return {
        row["role"]: {
            "calls": int(row["calls"] or 0),
            "longest_call_ms": round(float(row["longest_ms"]), 1) if row["longest_ms"] is not None else None,
            "longest_call_queryid": row["longest_queryid"],
            "longest_call_query": row["longest_query"],
            "longest_call_since": row["longest_since"],
            "stats_since": row["stats_since"],
        }
        for row in _rows(conn, query)
    }


def read_routines(conn) -> List[Dict[str, Any]]:
    params = {"names": list(TIMEOUT_SETTINGS), "limit": MAX_ROUTINES}
    return [
        {
            "schema": row["schema"],
            "name": row["name"],
            "kind": "procedure" if row["kind"] == "p" else "function",
            "owner": row["owner"],
            "commits": bool(row["commits"]),
            "catches_cancel": bool(row["catches_cancel"]),
            "settings": _settings(row["config"]),
        }
        for row in _rows(conn, ROUTINES_SQL, params)
    ]


def read_server(conn) -> Dict[str, Dict[str, str]]:
    """The values a session starts with, and where each comes from (`user`: stored on the
    reader's own role). Not through a005._run: its `set local` would hide the source."""
    try:
        with conn.cursor() as cur:
            cur.execute(SERVER_SQL, (list(TIMEOUT_SETTINGS),))
            return {name: {"setting": setting, "unit": unit, "source": source}
                    for name, setting, unit, source in cur.fetchall()}
    finally:
        conn.rollback()


def _role_order(role: Dict[str, Any]) -> Tuple[int, str]:
    """The rows a report can use come first when a server has more roles than MAX_ROLES."""
    useful = bool(role.get("settings") or role.get("database_settings") or role.get("calls"))
    return (0 if useful else 1, str(role.get("role")))


def catalog_facts(conn_str: str, *, opener: Callable = a005.open_database) -> Dict[str, Any]:
    """The catalog part of the G003 data of one node. Raises only when the node
    cannot be opened; a fact that cannot be read is left out with its reason."""
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
        fact("server_settings", lambda: read_server(conn))
        fact("database_settings", lambda: {
            row["database"]: values for row in _rows(conn, DATABASE_SQL)
            for values in [_settings(row["config"])] if values})
        roles = fact("roles", lambda: read_roles(conn))
        calls = fact("calls", lambda: read_calls(conn))
        out.pop("calls", None)
        if roles is not None and calls:
            for role in roles:
                role.update(calls.get(role["role"], {}))
        if roles is not None and len(roles) > MAX_ROLES:
            out["roles_total"] = len(roles)
            out["roles"] = sorted(roles, key=_role_order)[:MAX_ROLES]
        fact("routines", lambda: read_routines(conn))
    finally:
        conn.close()
    return out


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def deadlock_window(series: Sequence[Dict[str, Any]], *, step_seconds: int = 3600) -> Optional[Dict[str, Any]]:
    """Deadlocks per step from the counter's readings (a range query's result).

    Sums the rise between readings, per database: a reset never counts as
    deadlocks, and neither does a series' first value (PromQL's increase() can
    count it from 0). A database's readings are merged first: a series that
    changes labels goes on with the same counter. None without two readings.
    """
    by_database: Dict[str, Dict[float, float]] = {}
    for item in series or []:
        metric = item.get("metric") or {}
        name = str(metric.get("real_dbname") or metric.get("datname") or metric.get("dbname") or "")
        points = by_database.setdefault(name, {})
        for epoch, value in item.get("values") or []:
            try:
                reading = float(value)
            except (TypeError, ValueError):
                continue
            points[float(epoch)] = max(points.get(float(epoch), reading), reading)

    per_day: Dict[str, int] = {}
    per_hour: Dict[int, int] = {}
    per_step: Dict[float, int] = {}
    totals: Dict[str, int] = {}
    start, end = None, None
    for name, points in by_database.items():
        ordered = sorted(points.items())
        if len(ordered) < 2:
            continue
        start = min(start, ordered[0][0]) if start is not None else ordered[0][0]
        end = max(end, ordered[-1][0]) if end is not None else ordered[-1][0]
        totals.setdefault(name, 0)
        for (_, before), (at, after) in zip(ordered, ordered[1:]):
            rise = int(after - before)
            if rise <= 0:  # no change, or a statistics reset
                continue
            totals[name] += rise
            when = datetime.fromtimestamp(at, timezone.utc)
            day = when.strftime("%Y-%m-%d")
            per_day[day] = per_day.get(day, 0) + rise
            per_hour[when.hour] = per_hour.get(when.hour, 0) + rise
            per_step[at] = per_step.get(at, 0) + rise
    if start is None:
        return None

    total = sum(totals.values())
    days = round((end - start) / 86400, 1)
    out: Dict[str, Any] = {
        "start": _iso(start),
        "end": _iso(end),
        "days": days,
        "step_seconds": step_seconds,
        "deadlocks": total,
        "days_with_deadlocks": len(per_day),
        "per_day": [{"date": day, "deadlocks": per_day[day]} for day in sorted(per_day)],
        "by_database": {name: count for name, count in totals.items() if count},
    }
    if per_hour:
        hour = max(sorted(per_hour), key=lambda h: per_hour[h])
        worst = max(sorted(per_step), key=lambda s: per_step[s])
        out["peak_hour_utc"] = hour
        out["peak_hour_deadlocks"] = per_hour[hour]
        out["worst_step_end"] = _iso(worst)
        out["worst_step_deadlocks"] = per_step[worst]
    return out


def _epoch(item: Dict[str, Any]) -> Optional[float]:
    try:
        return float(item["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


#: The labels that say which statement waited on which; pids and modes are not part of it.
_PAIR_LABELS = ("blocked_query_id", "blocked_user", "blocked_table", "blocker_query_id", "blocker_user", "blocker_table")


def lock_waits(longest: Sequence[Dict[str, Any]], at: Sequence[Dict[str, Any]], *, window: Dict[str, Any],
               query_text: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """Who blocked whom over the window, from one max_over_time and one tmax_over_time
    per series of the lock-wait metric. A series is one pair of sessions and statements;
    its value is the blocked statement's age at the sample, not the length of the wait."""
    labels = lambda item: tuple(sorted((k, v) for k, v in (item.get("metric") or {}).items() if k != "__name__"))
    times = {labels(item): _epoch(item) for item in at or []}
    pairs: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for item in longest or []:
        metric = item.get("metric") or {}
        age = _epoch(item)
        if age is None:
            continue
        key = tuple(str(metric.get(name) or "") for name in _PAIR_LABELS)
        pair = pairs.setdefault(key, {"sessions": 0, "longest_ms": -1.0, "longest_at": None})
        pair["sessions"] += 1
        if age > pair["longest_ms"]:
            pair["longest_ms"], pair["longest_at"] = age, times.get(labels(item))
    rows = []
    for key, pair in sorted(pairs.items(), key=lambda kv: (-kv[1]["longest_ms"], kv[0]))[:MAX_LOCK_PAIRS]:
        fields = dict(zip(_PAIR_LABELS, key))
        fields.update(sessions=pair["sessions"], longest_blocked_statement_age_ms=int(pair["longest_ms"]),
                      longest_at=_iso(pair["longest_at"]) if pair["longest_at"] else None)
        if query_text:
            fields["blocked_query"] = _short(query_text(fields["blocked_query_id"]))
            fields["blocker_query"] = _short(query_text(fields["blocker_query_id"]))
        rows.append(fields)
    return {"start": window.get("start"), "end": window.get("end"), "days": window.get("days"),
            "pairs_total": len(pairs), "session_pairs_total": sum(p["sessions"] for p in pairs.values()),
            "pairs": rows}


_PLAIN_QUERY = re.compile(r"\s*(select|insert|update|delete|merge|with|values)\b", re.IGNORECASE)
#: `select ... into` is a utility statement too: below 16 it keeps its literals. `insert into` is not.
_INTO = re.compile(r"\binto\b", re.IGNORECASE)


def _short(text: Optional[str]) -> Optional[str]:
    """One line of a plain query. Utility text stays verbatim in pg_stat_statements,
    passwords included, so it is left out, as in CALLS_SQL."""
    plain = _PLAIN_QUERY.match(str(text or ""))
    if not plain or (plain.group(1).lower() in ("select", "with") and _INTO.search(str(text))):
        return None
    one_line = " ".join(str(text).split())
    return one_line[:QUERY_CHARS]
