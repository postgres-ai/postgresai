"""Fixtures for compliance_vectors integration tests.

CI environments (the reporter:tests job runs as the `postgres` superuser
with PG started by `service postgresql start`) cannot use
pytest-postgresql's native `postgresql` fixture — its pg_ctl-managed
per-test cluster hangs there, the same hang `tests/reporter/conftest.py`
documents. Set `PGAI_USE_SYSTEM_PG=1` in the CI job and this conftest
overrides the fixture to connect directly to the running cluster.

Local development (no env var set) keeps pytest-postgresql's default,
so a developer with PG binaries on PATH but no running system cluster
can still run the tests.

The reporter override yields a psycopg2 connection; this conftest uses
psycopg3 to match the API the MR 262 tests use (`cursor.description[i]
.name`). Both drivers ship in `reporter/requirements-dev.txt`.
"""
import os
import uuid

import pytest


if os.environ.get("PGAI_USE_SYSTEM_PG"):

    @pytest.fixture
    def postgresql():
        """psycopg3 connection to a throwaway DB on the system Postgres
        cluster. Each test gets its own database so seed schemas don't
        bleed between tests.

        Honors PGHOST/PGPORT/PGUSER/PGDATABASE the same way `psql` does.
        Crucially, if PGHOST is unset (the default `su - postgres`
        environment in the reporter:tests CI image), psycopg falls back
        to the Unix socket — which the Debian default pg_hba.conf maps
        to `peer` auth for the `postgres` superuser, so no password is
        required. Hardcoding `host=localhost` would force TCP and hit
        the default `md5`/`scram-sha-256` auth, failing with
        `fe_sendauth: no password supplied`.
        """
        import psycopg
        from psycopg import sql

        # Build kwargs lazily so an unset PGHOST yields a true
        # Unix-socket connection instead of `host='localhost'` (TCP).
        common_kwargs = {"autocommit": True}
        if os.environ.get("PGHOST"):
            common_kwargs["host"] = os.environ["PGHOST"]
        if os.environ.get("PGPORT"):
            common_kwargs["port"] = int(os.environ["PGPORT"])
        if os.environ.get("PGUSER"):
            common_kwargs["user"] = os.environ["PGUSER"]
        bootstrap_db = os.environ.get("PGDATABASE", "postgres")

        # uuid (not just os.getpid()) so multiple tests in one pytest
        # process can each get a fresh DB even if a previous test left
        # a backend lingering — `DROP DATABASE` would otherwise raise
        # ObjectInUse on the second test.
        test_db = f"test_mr262_{uuid.uuid4().hex[:12]}"
        test_db_ident = sql.Identifier(test_db)

        bootstrap = psycopg.connect(dbname=bootstrap_db, **common_kwargs)
        with bootstrap.cursor() as cur:
            cur.execute(sql.SQL("drop database if exists {}").format(test_db_ident))
            cur.execute(sql.SQL("create database {}").format(test_db_ident))
        bootstrap.close()

        conn = psycopg.connect(dbname=test_db, **common_kwargs)
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
            # Don't let a flaky cleanup mask the real test failure —
            # the throwaway DB will be garbage-collected by the next
            # test invocation that gets the same uuid (effectively
            # never) or by an orphan sweep.
            try:
                cleanup = psycopg.connect(dbname=bootstrap_db, **common_kwargs)
                with cleanup.cursor() as cur:
                    cur.execute(
                        sql.SQL("drop database if exists {}").format(test_db_ident)
                    )
                cleanup.close()
            except Exception:
                pass
