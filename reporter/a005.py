"""A005 "Extensions" on a monitoring box: catalog facts of every database.

The Python twin of the CLI's express A005 (`generateA005` in cli/lib/checkup.ts):
the same queries, limits, conclusions and JSON (reporter/schemas/A005.schema.json).
The platform's A013 report rules CVEs out from these facts and marks its
post-upgrade items (postgresai#474, platform-all#1108).

The reporter reads it straight from each monitored node, once per run, with the
connection pgwatch uses (the node's `conn_str` in the sources file). Every
database that accepts connections is read over its own short connection; a
database that is not read is listed with the reason. A connection string or a
password never reaches a log line or the file.
"""
import os
import re
import ssl
import time
from urllib.parse import quote
from typing import Any, Callable, Dict, List, Optional

import psycopg2
import psycopg2.extensions

# The same limits as the CLI (A005_MAX_DATABASES, A005_TIME_BUDGET_MS).
MAX_DATABASES = 100
TIME_BUDGET_S = 60
ERROR_CHARS = 200
CONNECT_TIMEOUT_S = 10
STATEMENT_TIMEOUT = "30s"

# Per database: extensions, index access methods with the extension of their
# operator classes, and three counts (A005_DATABASE_SQL in the CLI).
DATABASE_SQL = """
select
  coalesce((select json_object_agg(e.extname, e.extversion order by e.extname)
            from pg_extension as e), '{}'::json) as extensions,
  coalesce((select json_agg(json_build_object('am', t.am, 'extension', t.extension, 'indexes', t.indexes)
                            order by t.am, t.extension nulls first)
            from (
              select am.amname as am, x.extname as extension, count(*)::int as indexes
              from pg_index as i
              join pg_class as c on c.oid = i.indexrelid
              join pg_namespace as n on n.oid = c.relnamespace
              join pg_am as am on am.oid = c.relam
              left join lateral (
                select distinct e.extname
                from unnest(i.indclass::oid[]) as opc(oid)
                join pg_depend as d on d.classid = 'pg_opclass'::regclass and d.objid = opc.oid
                  and d.refclassid = 'pg_extension'::regclass and d.deptype = 'e'
                join pg_extension as e on e.oid = d.refobjid
              ) as x on true
              where n.nspname not in ('pg_catalog', 'information_schema', 'pg_toast')
                and n.nspname !~ '^pg_(toast_)?temp_'
              group by am.amname, x.extname
            ) as t), '[]'::json) as index_access_methods,
  (select count(*) from pg_policy)::int as row_security_policies,
  (select count(*) from pg_statistic_ext)::int as extended_statistics,
  (select count(*) from pg_constraint
   where contype in ('u', 'p') and condeferrable)::int as deferrable_uniques
"""

DATABASES_SQL = """
select d.datname, d.datname = current_database() as is_current,
  has_database_privilege(d.oid, 'CONNECT') as can_connect,
  (select count(*) from pg_roles as r
   where not r.rolsuper and has_database_privilege(r.oid, d.oid, 'CREATE'))::int as roles_with_create
from pg_database as d
where d.datallowconn
order by d.datname
"""

SLOTS_SQL = """
select database, plugin, count(*)::int as slots, (count(*) filter (where active))::int as active
from pg_replication_slots
where slot_type = 'logical'
group by database, plugin
order by database, plugin
"""

REPLICATION_ROLES_SQL = """
select count(*)::int as replication
from pg_roles as r
where not r.rolsuper
  and (r.rolreplication or exists (
    select 1 from pg_roles as p
    where p.rolname = 'rds_replication' and p.oid <> r.oid and pg_has_role(r.oid, p.oid, 'member')))
"""

PASSWORD_HASHES_SQL = """
select (count(*) filter (where rolpassword like 'md5%'))::int as md5_passwords,
  (count(*) filter (where rolpassword like 'SCRAM-SHA-256$%'
                      and split_part(split_part(rolpassword, '$', 2), ':', 1) <> '4096'))::int
    as nondefault_scram_iterations
from pg_authid
"""

HBA_SQL = """
select (count(*) filter (where auth_method = 'cert'))::int as cert,
  (count(*) filter (where auth_method = 'trust'))::int as trust,
  (count(*) filter (where type = 'hostgssenc'))::int as hostgssenc,
  (count(*) filter (where error is not null))::int as errors
from pg_hba_file_rules
"""

VERSION_SQL = "select name, setting from pg_settings where name in ('server_version', 'server_version_num')"

_PASSWORD_IN_DSN = re.compile(r"(://[^:/@\s]*:)[^@\s]*@|(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)


def _secrets(conn_str: str) -> List[str]:
    """The conn_str's password, as written and URL-encoded (a message may quote either)."""
    try:
        password = psycopg2.extensions.parse_dsn(conn_str).get("password", "")
    except Exception:
        return []
    return [p for p in {password, quote(password, safe="")} if p]


def clip(text: str) -> str:
    """At most ERROR_CHARS characters, keeping the end: libpq names the cause last."""
    if len(text) <= ERROR_CHARS:
        return text
    head = (ERROR_CHARS - 3) // 3
    return text[:head] + "..." + text[len(text) - (ERROR_CHARS - 3 - head):]


def _redacted(err: BaseException, secrets: Optional[List[str]] = None) -> str:
    """The whole message on one line, without any password."""
    message = " ".join(str(err).split()) or type(err).__name__
    message = _PASSWORD_IN_DSN.sub(lambda m: (m.group(1) + "***@") if m.group(1) else m.group(2) + "***", message)
    for secret in secrets or []:
        if secret:
            message = message.replace(secret, "***")
    return message


def error_text(err: BaseException, secrets: Optional[List[str]] = None) -> str:
    """One line of at most ERROR_CHARS characters, without any password."""
    return clip(_redacted(err, secrets)) or "unknown error"


def system_ca_file() -> Optional[str]:
    """The system CA bundle. libpq's sslrootcert=system does not work with the
    libpq psycopg2-binary bundles (its OpenSSL looks in a build path), and
    verify-ca refuses it."""
    for path in (ssl.get_default_verify_paths().cafile, "/etc/ssl/certs/ca-certificates.crt"):
        if path and os.path.isfile(path):
            return path
    return None


class InvalidConnection(ValueError):
    """A conn_str libpq cannot parse: its own message may quote a password."""


def open_database(conn_str: str, database: Optional[str] = None):
    """A connection to `database` (default: the one in conn_str) of the node.

    The conn_str is used as pgwatch uses it: its own options stay. A verify-*
    sslmode without sslrootcert trusts the system CA bundle, as pgwatch and the
    CLI do. Read-only and the timeouts are set per transaction (see _run), so
    nothing is left on a pooled backend.
    """
    try:
        params = psycopg2.extensions.parse_dsn(conn_str)
    except Exception:
        raise InvalidConnection("the connection string is not valid") from None
    kwargs: Dict[str, Any] = {}
    if "connect_timeout" not in params:
        kwargs["connect_timeout"] = CONNECT_TIMEOUT_S
    if "application_name" not in params:
        kwargs["application_name"] = "postgresai-reporter-a005"
    if str(params.get("sslmode", "")).startswith("verify") and "sslrootcert" not in params:
        ca = system_ca_file()
        if ca:
            kwargs["sslrootcert"] = ca
    if database is not None:
        kwargs["dbname"] = database
    conn = psycopg2.connect(conn_str, **kwargs)
    try:
        conn.set_session(readonly=True, autocommit=False)
    except Exception:
        conn.close()
        raise
    return conn


def _run(conn, sql: str, fetch: Callable):
    """One read-only transaction per query: a failed query aborts only itself."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"set local statement_timeout = '{STATEMENT_TIMEOUT}'")
            cur.execute("set local search_path = pg_catalog, pg_temp")
            cur.execute(sql)
            return fetch(cur)
    finally:
        conn.rollback()


def _one(conn, sql: str) -> Dict[str, Any]:
    def fetch(cur):
        row = cur.fetchone()
        return dict(zip([c.name for c in cur.description], row)) if row else {}
    return _run(conn, sql, fetch)


def _all(conn, sql: str) -> List[Dict[str, Any]]:
    def fetch(cur):
        names = [c.name for c in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    return _run(conn, sql, fetch)


def read_database(conn) -> Dict[str, Any]:
    row = _one(conn, DATABASE_SQL)
    return {
        "extensions": row.get("extensions") or {},
        "index_access_methods": row.get("index_access_methods") or [],
        "row_security_policies": int(row.get("row_security_policies") or 0),
        "extended_statistics": int(row.get("extended_statistics") or 0),
        "deferrable_uniques": int(row.get("deferrable_uniques") or 0),
    }


def postgres_version(conn) -> Dict[str, str]:
    settings = {r["name"]: r["setting"] for r in _all(conn, VERSION_SQL)}
    num = settings.get("server_version_num", "")
    known = num.isdigit() and len(num) >= 6  # the CLI's parseVersionNum: Postgres 10 and later
    major = str(int(num) // 10000) if known else ""
    minor = str(int(num) % 10000) if known else ""
    return {"version": settings.get("server_version", ""), "server_version_num": num,
            "server_major_ver": major, "server_minor_ver": minor}


def conclusions(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """The CLI's a005Conclusions, word for word."""
    out: List[str] = []
    recs: List[str] = []
    dbs = list(data["databases"].items())
    unread = [(name, db) for name, db in dbs if not db["read"]]
    installed = set()
    for _, db in dbs:
        installed.update((db.get("extensions") or {}).keys())

    def n(count: int, noun: str) -> str:
        return f"{count} {noun}{'' if count == 1 else 's'}"

    out.append(f"{len(dbs) - len(unread)} of {n(len(dbs), 'database')} read; "
               f"{n(len(installed), 'distinct extension')} installed.")
    if unread:
        out.append(f"Not read: {', '.join(name for name, _ in unread)}. "
                   "A fact about extensions or indexes is not known for the server.")
        recs.append("Let the checkup role connect to every database (CONNECT privilege) so that A005 covers the whole server.")
    with_create = sum(1 for _, db in dbs if db["roles_with_create"] > 0)
    if with_create:
        out.append(f"{with_create} of {n(len(dbs), 'database')} {'has' if with_create == 1 else 'have'} "
                   "a non-superuser role with CREATE: it can install trusted extensions itself.")
    roles = data.get("roles") or {}
    if roles.get("md5_passwords") is None:
        out.append(f"Stored password hashes are not known: {roles.get('password_hashes_error') or 'pg_authid is not readable'}.")
    elif roles["md5_passwords"] > 0:
        out.append(f"{n(roles['md5_passwords'], 'role')} with an MD5 password hash.")
        recs.append("Set password_encryption = 'scram-sha-256' and reset the passwords of the roles with MD5 hashes.")
    slots = sum(s["slots"] for s in data["logical_slots"])
    if slots:
        out.append(f"{n(slots, 'logical replication slot')}.")
    return {"conclusions": out, "recommendations": recs}


def node_facts(conn_str: str, *, opener: Callable = open_database,
               clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """The A005 result of one node: {"data": ..., "postgres_version": ...}.

    Raises only when the node itself cannot be read (the first connection or
    the database list); every later failure is recorded in the data.
    """
    secrets = _secrets(conn_str)
    started = clock()
    conn = opener(conn_str)
    try:
        version = postgres_version(conn)
        listed = _all(conn, DATABASES_SQL)
        databases: Dict[str, Dict[str, Any]] = {}
        opened = 0
        for row in listed:
            entry: Dict[str, Any] = {"read": False, "roles_with_create": int(row["roles_with_create"] or 0)}
            databases[row["datname"]] = entry
            try:
                if row["is_current"]:
                    entry.update(read_database(conn), read=True)
                    continue
                if not row["can_connect"]:
                    raise RuntimeError("the checkup role has no CONNECT privilege on this database")
                if opened >= MAX_DATABASES:
                    raise RuntimeError(f"not read: more than {MAX_DATABASES} databases")
                if clock() - started > TIME_BUDGET_S:
                    raise RuntimeError("not read: the time budget ran out")
                opened += 1
                other = opener(conn_str, row["datname"])
                try:
                    entry.update(read_database(other), read=True)
                finally:
                    other.close()
            except Exception as err:  # one database: recorded, the rest go on
                entry["error"] = error_text(err, secrets)

        def optional(fn: Callable[[], Any]):
            try:
                return fn(), None
            except Exception as err:  # an optional fact: unknown, with the reason
                return None, error_text(err, secrets)

        slots, slots_error = optional(lambda: [
            {"database": r["database"], "plugin": r["plugin"], "slots": int(r["slots"]), "active": int(r["active"])}
            for r in _all(conn, SLOTS_SQL)])
        replication, replication_error = optional(lambda: int(_one(conn, REPLICATION_ROLES_SQL)["replication"]))
        hashes, hashes_error = optional(lambda: _one(conn, PASSWORD_HASHES_SQL))
        scram, _ = optional(lambda: _one(conn, "select current_setting('scram_iterations', true) as v").get("v"))
        subscriptions, subscriptions_error = optional(
            lambda: int(_one(conn, "select count(*)::int as n from pg_subscription")["n"]))

        def hba():
            r = _one(conn, HBA_SQL)
            return {k: int(r[k]) for k in ("cert", "trust", "hostgssenc", "errors")}

        hba_rules, hba_error = optional(hba)
    finally:
        conn.close()

    data: Dict[str, Any] = {
        "databases": databases,
        "logical_slots": slots or [],
        "logical_slots_error": slots_error,
        "roles": {
            "replication": replication,
            "replication_error": replication_error,
            "md5_passwords": int(hashes["md5_passwords"]) if hashes else None,
            "nondefault_scram_iterations": int(hashes["nondefault_scram_iterations"]) if hashes else None,
            "password_hashes_error": hashes_error,
            "scram_iterations": scram,
        },
        "subscriptions": subscriptions,
        "subscriptions_error": subscriptions_error,
        "hba_rules": hba_rules,
        "hba_rules_error": hba_error,
    }
    data.update(conclusions(data))
    return {"data": data, "postgres_version": version}


def unreachable_node(err: BaseException, conn_str: str = "") -> Dict[str, Any]:
    """The A005 result of a node that could not be read: no database is listed,
    every fact is unknown with the reason (the platform then reads no fact from it)."""
    secrets = _secrets(conn_str) if conn_str else []
    reason = clip(f"the node could not be read: {_redacted(err, secrets) or 'unknown error'}")
    data: Dict[str, Any] = {
        "databases": {}, "logical_slots": [], "logical_slots_error": reason,
        "roles": {"replication": None, "replication_error": reason, "md5_passwords": None,
                  "nondefault_scram_iterations": None, "password_hashes_error": reason, "scram_iterations": None},
        "subscriptions": None, "subscriptions_error": reason, "hba_rules": None, "hba_rules_error": reason,
        "conclusions": [f"Not read: {reason}."], "recommendations": [],
    }
    return {"data": data}
