"""
Integration checks for xmin_horizon metric collection.

The test creates real PostgreSQL blockers where the target environment supports
that blocker type, waits for pgwatch collection, and validates the exported
Prometheus series. Optional paths can be made mandatory with REQUIRE_* flags so
CI can fail instead of silently skipping coverage.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import psycopg
from psycopg import IsolationLevel, sql
import requests

POLL_INTERVAL_SECONDS = 5
SNAPSHOT_XMIN_TIMEOUT_SECONDS = 30
SNAPSHOT_XMIN_POLL_SECONDS = 0.2
MIN_BLOCKER_AGE_TX = 5
HORIZON_MAX_TOLERANCE_TX = 1.0
ACTIVITY_AGE_TOLERANCE_TX = 100.0
DATA_HORIZON_METRIC = "pgwatch_xmin_horizon_data_horizon_age_tx"
CATALOG_HORIZON_METRIC = "pgwatch_xmin_horizon_catalog_horizon_age_tx"
SNAPSHOT_XMIN_METRIC = "pgwatch_xmin_horizon_snapshot_xmin"
DATA_COMPONENTS = {
    "pg_stat_activity",
    "pg_replication_slots",
    "pg_stat_replication",
    "pg_prepared_xacts",
}

def env_with_fallback(primary: str, fallback: str) -> str | None:
    """Return primary env value when set, otherwise fall back to another key."""
    if primary in os.environ:
        return os.environ[primary]
    return os.environ.get(fallback)


COMPONENT_SUMMARY_METRICS = {
    "pg_stat_activity": "pgwatch_xmin_horizon_pg_stat_activity_age_tx",
    "pg_replication_slots": "pgwatch_xmin_horizon_pg_replication_slots_age_tx",
    "pg_replication_slots_catalog": (
        "pgwatch_xmin_horizon_pg_replication_slots_catalog_age_tx"
    ),
    "pg_stat_replication": "pgwatch_xmin_horizon_pg_stat_replication_age_tx",
    "pg_prepared_xacts": "pgwatch_xmin_horizon_pg_prepared_xacts_age_tx",
}

COMPONENT_COUNT_METRICS = {
    "pg_stat_activity": "pgwatch_xmin_horizon_pg_stat_activity_count",
    "pg_replication_slots": "pgwatch_xmin_horizon_pg_replication_slots_count",
    "pg_replication_slots_catalog": (
        "pgwatch_xmin_horizon_pg_replication_slots_catalog_count"
    ),
    "pg_stat_replication": "pgwatch_xmin_horizon_pg_stat_replication_count",
    "pg_prepared_xacts": "pgwatch_xmin_horizon_pg_prepared_xacts_count",
}


@dataclass
class ComponentScenario:
    component: str
    summary_metric: str
    detail_labels: dict[str, str]
    expected_labels: dict[str, str] | None = None
    active: bool = False


class xminHorizonTest:
    """Owns one xmin-horizon integration run and cleans up created blockers.

    A run opens primary and optional standby connections, creates blocker state,
    validates direct PostgreSQL and Prometheus observations, and then releases
    all owned transactions, prepared xacts, slots, and tables in ``cleanup()``.
    Instances are single-use; call ``run()`` once and let its ``finally`` block
    perform cleanup.
    """
    def __init__(
        self,
        target_db_url: str,
        prometheus_url: str,
        standby_db_url: str | None = None,
        collection_wait_seconds: int = 60,
        prometheus_username: str | None = None,
        prometheus_password: str | None = None,
    ):
        self.target_db_url = target_db_url
        self.standby_db_url = standby_db_url
        self.prometheus_url = prometheus_url.rstrip("/")
        if bool(prometheus_username) != bool(prometheus_password):
            raise ValueError(
                "Both prometheus_username and prometheus_password must be set together"
            )
        self.prometheus_auth = (
            (prometheus_username, prometheus_password)
            if prometheus_username and prometheus_password
            else None
        )
        self.collection_wait_seconds = collection_wait_seconds
        self.started_at_seconds = time.time()
        self.target_conn: psycopg.Connection | None = None
        self.snapshot_conn: psycopg.Connection | None = None
        self.standby_snapshot_conn: psycopg.Connection | None = None
        self.snapshot_pid: int | None = None
        self.snapshot_appname = f"xmin_horizon_test_{int(time.time())}_{os.getpid()}"
        self.standby_appname = (
            f"xmin_horizon_standby_{int(time.time())}_{os.getpid()}"
        )
        self.target_datname: str | None = None
        self.target_user: str | None = None
        run_id = f"{int(time.time())}_{os.getpid()}"
        self.prepared_gid = f"xmin_horizon_prepared_{run_id}"
        self.logical_slot_name = f"xmin_horizon_logical_{run_id}"
        self.physical_slot_name = f"xmin_horizon_physical_{run_id}"
        self.standby_physical_slot_name = f"xmin_horizon_standby_{run_id}"
        self.prepared_xact_created = False
        self.replication_feedback_seen = False
        self.replication_feedback_labels: dict[str, str] = {}
        self.slot_xmin_labels: dict[str, str] = {}
        self.logical_slot_components: set[str] = set()
        self.standby_slot_configured = False
        self.cleanup_errors: list[str] = []

    @staticmethod
    def prometheus_label_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    def prometheus_selector(self, **labels: str) -> str:
        return ",".join(
            f'{key}="{self.prometheus_label_value(value)}"'
            for key, value in labels.items()
        )

    def setup(self) -> None:
        """Open the target connection and create a clean test table."""
        print("Setting up test environment...")
        self.target_conn = psycopg.connect(self.target_db_url)
        self.target_conn.autocommit = True

        response = requests.get(
            f"{self.prometheus_url}/api/v1/status/config",
            auth=self.prometheus_auth,
            timeout=5,
        )
        response.raise_for_status()
        print("[ok] Prometheus connection verified")

        with self.target_conn.cursor() as cur:
            cur.execute("select current_database(), current_user")
            self.target_datname, self.target_user = cur.fetchone()
            cur.execute(
                """
                drop table if exists xmin_horizon_test_table cascade;
                create table xmin_horizon_test_table (
                    id int generated always as identity primary key,
                    payload text not null
                );
                insert into xmin_horizon_test_table (payload)
                values ('seed row');
                """
            )
        print(f"[ok] Test table ready in database {self.target_datname}")

    def advance_primary_xids(self, reason: str, count: int = 10) -> None:
        """Commit separate writes after a blocker so age metrics turn positive."""
        with self.target_conn.cursor() as cur:
            for _ in range(count):
                cur.execute(
                    "insert into xmin_horizon_test_table (payload) values (%s)",
                    (reason,),
                )
        print(f"[ok] Advanced primary xids for {reason} ({count} writes)")

    def create_repeatable_read_snapshot(self) -> None:
        """Create the required pg_stat_activity xmin blocker."""
        print("\nCreating repeatable-read snapshot blocker...")
        self.snapshot_conn = psycopg.connect(
            self.target_db_url,
            application_name=self.snapshot_appname,
        )
        self.snapshot_conn.isolation_level = IsolationLevel.REPEATABLE_READ
        self.snapshot_conn.autocommit = False

        with self.snapshot_conn.cursor() as cur:
            cur.execute("select pg_backend_pid()")
            self.snapshot_pid = cur.fetchone()[0]
            cur.execute("select * from xmin_horizon_test_table limit 1")
            cur.fetchone()

        print(f"[ok] Snapshot transaction started (pid={self.snapshot_pid})")
        self.wait_for_snapshot_xmin()
        self.advance_primary_xids("activity snapshot blocker")

    def wait_for_snapshot_xmin(self) -> None:
        """Poll pg_stat_activity until the snapshot exposes backend_xmin."""
        deadline = time.monotonic() + SNAPSHOT_XMIN_TIMEOUT_SECONDS
        last_state = None
        last_appname = None

        while time.monotonic() <= deadline:
            with self.target_conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        backend_xmin is not null,
                        state,
                        application_name
                    from pg_stat_activity
                    where pid = %s
                    """,
                    (self.snapshot_pid,),
                )
                row = cur.fetchone()
            if not row:
                raise RuntimeError("Snapshot backend not found in pg_stat_activity")

            has_backend_xmin, last_state, last_appname = row
            if has_backend_xmin:
                print(
                    f"[ok] backend_xmin populated (state={last_state}, "
                    f"app={last_appname})"
                )
                return
            time.sleep(SNAPSHOT_XMIN_POLL_SECONDS)

        raise RuntimeError(
            "backend_xmin is null for the snapshot transaction "
            f"after {SNAPSHOT_XMIN_TIMEOUT_SECONDS}s "
            f"(state={last_state}, app={last_appname})"
        )

    @staticmethod
    def require_env(name: str) -> bool:
        return os.getenv(name) == "1"

    def create_prepared_transaction(self) -> bool:
        """Create a pg_prepared_xacts blocker when prepared xacts are enabled."""
        print("\nCreating prepared transaction blocker...")
        require = self.require_env("REQUIRE_PREPARED_XACTS_TEST")
        with self.target_conn.cursor() as cur:
            cur.execute("show max_prepared_transactions")
            max_prepared = int(cur.fetchone()[0])
        if max_prepared <= 0:
            message = "max_prepared_transactions=0; pg_prepared_xacts path skipped"
            if require:
                raise RuntimeError(message)
            print(f"[skip] {message}")
            return False

        conn = psycopg.connect(self.target_db_url)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into xmin_horizon_test_table (payload) values (%s)",
                    ("prepared blocker",),
                )
                cur.execute(
                    sql.SQL("prepare transaction {}").format(
                        sql.Literal(self.prepared_gid)
                    )
                )
            self.prepared_xact_created = True
        finally:
            conn.close()

        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select count(*)::int, max(age(transaction))::int8
                from pg_prepared_xacts
                where gid = %s
                """,
                (self.prepared_gid,),
            )
            count, age_tx = cur.fetchone()
        if count != 1 or age_tx is None:
            raise RuntimeError("Prepared transaction blocker was not created")
        self.advance_primary_xids("prepared transaction blocker")
        print(f"[ok] Prepared transaction created (gid={self.prepared_gid})")
        return True

    def create_logical_replication_slot(self) -> bool:
        """Create a logical slot with catalog_xmin to cover slot blocker paths."""
        print("\nCreating logical replication slot blocker...")
        require = self.require_env("REQUIRE_REPLICATION_SLOT_TEST")
        with self.target_conn.cursor() as cur:
            cur.execute("show wal_level")
            wal_level = cur.fetchone()[0]
        if wal_level != "logical":
            message = f"wal_level={wal_level}; logical slot path skipped"
            if require:
                raise RuntimeError(message)
            print(f"[skip] {message}")
            return False

        try:
            with self.target_conn.cursor() as cur:
                cur.execute(
                    "select * from "
                    "pg_create_logical_replication_slot(%s, 'test_decoding')",
                    (self.logical_slot_name,),
                )
                cur.fetchone()
        except Exception as exc:
            if require:
                raise RuntimeError("Could not create logical replication slot") from exc
            print(f"[skip] Could not create logical replication slot: {exc}")
            return False

        with self.target_conn.cursor() as cur:
            cur.execute(
                "insert into xmin_horizon_test_table (payload) "
                "values ('slot blocker')"
            )
            cur.execute(
                "select * from pg_logical_slot_peek_changes(%s, null, 1)",
                (self.logical_slot_name,),
            )
            cur.fetchall()

        self.verify_slot_age_expression(self.logical_slot_name, expect_populated=True)
        # Advance well past MIN_BLOCKER_AGE_TX so the first post-advance pgwatch
        # scrape clears the threshold even when its 30 s cadence happens to land
        # just after the slot creation on slow runners.
        self.advance_primary_xids("logical replication slot blocker", count=30)
        print(f"[ok] Logical replication slot created ({self.logical_slot_name})")
        return True

    def create_physical_null_xmin_slot(self) -> bool:
        """Create a slot whose xmin/catalog_xmin are null for regression coverage."""
        print("\nCreating physical replication slot for null-xmin regression...")
        require = self.require_env("REQUIRE_REPLICATION_SLOT_TEST")
        try:
            with self.target_conn.cursor() as cur:
                cur.execute(
                    "select * from pg_create_physical_replication_slot(%s)",
                    (self.physical_slot_name,),
                )
                cur.fetchone()
        except Exception as exc:
            if require:
                raise RuntimeError(
                    "Could not create physical replication slot"
                ) from exc
            print(f"[skip] Could not create physical replication slot: {exc}")
            return False

        self.verify_slot_age_expression(self.physical_slot_name, expect_populated=False)
        print(f"[ok] Physical null-xmin slot checked ({self.physical_slot_name})")
        return True

    def verify_slot_age_expression(
        self,
        slot_name: str,
        expect_populated: bool,
    ) -> None:
        """Validate slot-age expressions against real slot state.

        For all-null xmin/catalog_xmin slots this asserts the legacy
        replication_slots expression returns NULL. For populated logical slots,
        the populated xid source records which split xmin_horizon component
        should be visible in Prometheus.
        """
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select
                    age(xmin)::int8,
                    age(catalog_xmin)::int8,
                    greatest(age(xmin), age(catalog_xmin))::int8
                        as replication_slots_age_tx
                from pg_replication_slots
                where slot_name = %s
                """,
                (slot_name,),
            )
            row = cur.fetchone()

        if not row:
            raise RuntimeError(f"Replication slot {slot_name} was not found")

        xmin_age, catalog_xmin_age, replication_slots_age = row
        populated_ages = [
            age for age in (xmin_age, catalog_xmin_age) if age is not None
        ]

        if expect_populated:
            if not populated_ages:
                raise RuntimeError(f"Slot {slot_name} has no xmin/catalog_xmin blocker")
            expected_age = max(populated_ages)
            if replication_slots_age != expected_age:
                raise RuntimeError(
                    f"Slot {slot_name} replication_slots age {replication_slots_age} "
                    f"did not propagate populated age {expected_age}"
                )
            if xmin_age is not None:
                self.logical_slot_components.add("pg_replication_slots")
            if catalog_xmin_age is not None:
                self.logical_slot_components.add("pg_replication_slots_catalog")
        elif populated_ages:
            raise RuntimeError(
                f"Expected null xmin/catalog_xmin for {slot_name}, got {populated_ages}"
            )
        elif replication_slots_age is not None:
            raise RuntimeError(
                f"Null xmin/catalog_xmin slot {slot_name} returned "
                f"replication_slots xmin_age_tx {replication_slots_age}"
            )

    def capture_replication_slot_xmin_component(self) -> None:
        """Capture the current top pg_replication_slots.xmin blocker labels."""
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select
                    coalesce(database, '')::text as blocker_database,
                    slot_name::text,
                    slot_type::text,
                    coalesce(plugin, '')::text as slot_plugin,
                    case
                      when coalesce(to_jsonb(s)->>'invalidation_reason', '') <> '' then 'invalidated'
                      when coalesce(to_jsonb(s)->>'conflicting', 'false') = 'true' then 'conflicting'
                      when not active and coalesce(to_jsonb(s)->>'inactive_since', '') <> '' then 'inactive'
                      when not active then 'unused'
                      else 'active'
                    end::text as slot_status,
                    age(xmin)::int8 as age_tx
                from pg_replication_slots as s
                where xmin is not null
                order by age(xmin) desc, slot_name asc
                limit 1
                """
            )
            row = cur.fetchone()

        if not row:
            self.slot_xmin_labels = {}
            return

        blocker_database, slot_name, slot_type, slot_plugin, slot_status, age_tx = row
        self.logical_slot_components.add("pg_replication_slots")
        self.slot_xmin_labels = {
            "horizon_type": "data",
            "blocker_database": blocker_database,
            "slot_name": slot_name,
            "slot_type": slot_type,
            "slot_plugin": slot_plugin,
            "slot_xmin_source": "xmin",
            "slot_status": slot_status,
        }
        print(
            "[ok] pg_replication_slots xmin blocker captured: "
            f"slot={slot_name}, type={slot_type}, age={age_tx}"
        )

    def configure_standby_replication_slot(self) -> bool:
        """Route the standby through a test physical slot after idle baseline."""
        if self.standby_slot_configured:
            return True
        if not self.standby_db_url:
            print("[skip] No standby DB URL; skipping standby physical slot setup")
            return False

        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select 1
                from pg_replication_slots
                where slot_name = %s
                """,
                (self.standby_physical_slot_name,),
            )
            if not cur.fetchone():
                cur.execute(
                    "select pg_create_physical_replication_slot(%s)",
                    (self.standby_physical_slot_name,),
                )

        with psycopg.connect(self.standby_db_url, connect_timeout=5) as standby_conn:
            standby_conn.autocommit = True
            with standby_conn.cursor() as cur:
                cur.execute(
                    sql.SQL("alter system set primary_slot_name = {}").format(
                        sql.Literal(self.standby_physical_slot_name)
                    )
                )
                cur.execute("select pg_reload_conf()")

        deadline = time.monotonic() + SNAPSHOT_XMIN_TIMEOUT_SECONDS
        last_row = None
        while time.monotonic() <= deadline:
            with self.target_conn.cursor() as cur:
                cur.execute(
                    """
                    select active, xmin is not null, active_pid
                    from pg_replication_slots
                    where slot_name = %s
                    """,
                    (self.standby_physical_slot_name,),
                )
                last_row = cur.fetchone()
            if last_row and last_row[0] and last_row[1]:
                self.standby_slot_configured = True
                print(
                    "[ok] Standby physical slot is active with xmin: "
                    f"{self.standby_physical_slot_name}"
                )
                return True
            time.sleep(SNAPSHOT_XMIN_POLL_SECONDS)

        raise RuntimeError(
            "Standby physical replication slot did not become active with xmin: "
            f"slot={self.standby_physical_slot_name}, last={last_row}"
        )

    def create_standby_feedback_snapshot(self) -> bool:
        """Open a standby snapshot so hot_standby_feedback reports backend_xmin."""
        require = self.require_env("REQUIRE_STANDBY_FEEDBACK_TEST")
        if not self.standby_db_url:
            message = "STANDBY_DB_URL is required for standby-feedback coverage"
            if require:
                raise RuntimeError(message)
            print(f"[skip] {message}; checking for existing feedback rows")
            return False

        print("\nCreating standby feedback snapshot blocker...")
        deadline = time.monotonic() + self.collection_wait_seconds
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            conn: psycopg.Connection | None = None
            try:
                conn = psycopg.connect(
                    self.standby_db_url,
                    application_name=self.standby_appname,
                    connect_timeout=5,
                )
                conn.isolation_level = IsolationLevel.REPEATABLE_READ
                conn.autocommit = False
                with conn.cursor() as cur:
                    cur.execute("select pg_is_in_recovery(), current_database()")
                    in_recovery, standby_datname = cur.fetchone()
                    if not in_recovery:
                        message = "STANDBY_DB_URL is not connected to a standby"
                        if require:
                            raise RuntimeError(message)
                        print(f"[skip] {message}")
                        conn.close()
                        return False
                    if standby_datname != self.target_datname:
                        raise RuntimeError(
                            "standby database mismatch: "
                            f"{standby_datname} != {self.target_datname}"
                        )
                    cur.execute("select count(*) from pg_database")
                    cur.fetchone()

                self.standby_snapshot_conn = conn
                self.advance_primary_xids("standby feedback blocker")
                print(
                    "[ok] Standby snapshot started "
                    f"(app={self.standby_appname})"
                )
                return True
            except Exception as exc:
                last_error = exc
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(POLL_INTERVAL_SECONDS)

        message = f"Could not create standby feedback snapshot: {last_error}"
        if require:
            raise RuntimeError(message)
        print(f"[skip] {message}")
        return False

    def observe_standby_feedback(self) -> bool:
        """Detect hot_standby_feedback in pg_stat_replication."""
        print("\nChecking pg_stat_replication standby-feedback path...")
        require = self.require_env("REQUIRE_STANDBY_FEEDBACK_TEST")
        if not self.create_standby_feedback_snapshot():
            return False

        deadline = time.monotonic() + self.collection_wait_seconds
        last_total = 0
        last_count = 0
        last_age = None

        while True:
            with self.target_conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        count(*)::int,
                        count(*) filter (where backend_xmin is not null)::int,
                        max(age(backend_xmin))::int8
                    from pg_stat_replication
                    """
                )
                last_total, last_count, last_age = cur.fetchone()

            if last_count > 0 and last_age is not None:
                with self.target_conn.cursor() as cur:
                    cur.execute(
                        """
                        select
                            coalesce(state, '')::text,
                            coalesce(sync_state, '')::text,
                            coalesce(application_name, '')::text
                        from pg_stat_replication
                        where backend_xmin is not null
                        order by age(backend_xmin) desc, pid asc
                        limit 1
                        """
                    )
                    state, sync_state, standby_name = cur.fetchone()
                self.replication_feedback_labels = {
                    "replication_state": state,
                    "replication_sync_state": sync_state,
                    "standby_name": standby_name,
                }
                self.replication_feedback_seen = True
                print(
                    "[ok] Found standby-feedback row(s), "
                    f"total={last_total}, with_xmin={last_count}, max age={last_age}"
                )
                return True

            if time.monotonic() >= deadline:
                break
            time.sleep(POLL_INTERVAL_SECONDS)

        message = "No pg_stat_replication.backend_xmin row found"
        if require:
            raise RuntimeError(
                f"{message}; total_replication_rows={last_total}, "
                f"last_count={last_count}"
            )
        print(f"[skip] {message}; set REQUIRE_STANDBY_FEEDBACK_TEST=1 to require it")
        return False

    def query_prometheus(self, query: str) -> list[dict]:
        """Run an instant query against Prometheus or VictoriaMetrics."""
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            auth=self.prometheus_auth,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload}")
        return payload.get("data", {}).get("result", [])

    def wait_for_collection(
        self,
        description: str,
        predicate: Callable[[], bool],
    ) -> None:
        """Poll Prometheus until predicate succeeds or the deadline expires."""
        print(
            f"\nPolling up to {self.collection_wait_seconds} seconds "
            f"for {description}..."
        )
        deadline = time.monotonic() + self.collection_wait_seconds
        attempts = 0
        while True:
            attempts += 1
            if predicate():
                print(f"[ok] {description} observed after {attempts} attempt(s)")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {description} within "
                    f"{self.collection_wait_seconds}s"
                )
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    def query_summary(self, metric_name: str) -> list[dict]:
        if not self.target_datname:
            raise RuntimeError("target database name was not captured")
        database_selector = self.prometheus_selector(datname=self.target_datname)
        return self.query_prometheus(f"{metric_name}{{{database_selector}}}")

    def query_summary_snapshot(self) -> list[dict]:
        """Fetch all xmin_horizon summary/count series in one instant query."""
        if not self.target_datname:
            raise RuntimeError("target database name was not captured")
        database_selector = self.prometheus_selector(datname=self.target_datname)
        name_pattern = (
            "pgwatch_xmin_horizon_"
            "((pg_stat_activity|pg_replication_slots|pg_replication_slots_catalog|"
            "pg_stat_replication|pg_prepared_xacts)_(age_tx|count)|"
            "data_horizon_age_tx|catalog_horizon_age_tx|snapshot_xmin)"
        )
        return self.query_prometheus(
            f'{{__name__=~"{name_pattern}",{database_selector}}}'
        )

    def query_detail(self, **labels: str) -> list[dict]:
        if not self.target_datname:
            raise RuntimeError("target database name was not captured")
        selector = self.prometheus_selector(datname=self.target_datname, **labels)
        return self.query_prometheus(
            f"pgwatch_xmin_horizon_blockers_age_tx{{{selector}}}"
        )

    @staticmethod
    def comparable_labels(reference_metric: dict) -> list[str]:
        match_labels = [
            "cluster",
            "datname",
            "env",
            "instance",
            "job",
            "node_name",
            "sink_type",
            "sys_id",
        ]
        return [
            label
            for label in match_labels
            if label in reference_metric and reference_metric.get(label) != ""
        ]

    @classmethod
    def require_comparable_labels(cls, reference_metric: dict) -> list[str]:
        comparable_labels = cls.comparable_labels(reference_metric)
        if not comparable_labels:
            raise RuntimeError(
                "reference metric has no comparable source labels: "
                f"{reference_metric}"
            )
        return comparable_labels

    @classmethod
    def matching_result(
        cls,
        results: list[dict],
        reference_metric: dict,
    ) -> dict | None:
        """Find one series from the same monitored source as reference_metric."""
        comparable_labels = cls.require_comparable_labels(reference_metric)

        for result in results:
            metric = result.get("metric", {})
            if all(
                metric.get(label) == reference_metric.get(label)
                for label in comparable_labels
            ):
                return result
        return None

    @classmethod
    def same_source_results(
        cls,
        results: list[dict],
        reference_metric: dict,
    ) -> list[dict]:
        """Return every series from the same monitored source as reference_metric."""
        comparable_labels = cls.require_comparable_labels(reference_metric)
        return [
            result
            for result in results
            if all(
                result.get("metric", {}).get(label) == reference_metric.get(label)
                for label in comparable_labels
            )
        ]

    @staticmethod
    def sample_value(result: dict) -> float:
        return float(result["value"][1])

    def expected_blocker_counts_from_database(self) -> dict[str, int]:
        """Return test-owned PostgreSQL blocker counts by component."""
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select
                  (select count(*)::int
                   from pg_stat_activity
                   where application_name = %s
                     and backend_type = 'client backend'
                     and backend_xmin is not null) as activity_count,
                  (select count(*)::int
                   from pg_replication_slots
                   where slot_name in (%s, %s)
                     and xmin is not null) as slot_count,
                  (select count(*)::int
                   from pg_replication_slots
                   where slot_name = %s
                     and catalog_xmin is not null) as slot_catalog_count,
                  (select count(*)::int
                   from pg_stat_replication
                   where backend_xmin is not null) as replication_count,
                  (select count(*)::int
                   from pg_prepared_xacts
                   where gid = %s) as prepared_count
                """,
                (
                    self.snapshot_appname,
                    self.logical_slot_name,
                    self.standby_physical_slot_name,
                    self.logical_slot_name,
                    self.prepared_gid,
                ),
            )
            (
                activity_count,
                slot_count,
                slot_catalog_count,
                replication_count,
                prepared_count,
            ) = cur.fetchone()

        return {
            "pg_stat_activity": activity_count,
            "pg_replication_slots": slot_count,
            "pg_replication_slots_catalog": slot_catalog_count,
            "pg_stat_replication": replication_count,
            "pg_prepared_xacts": prepared_count,
        }

    def expected_blocker_components_from_database(self) -> set[str]:
        """Return components whose live PostgreSQL blocker count is non-zero."""
        return {
            component
            for component, count in self.expected_blocker_counts_from_database().items()
            if count > 0
        }

    def activity_snapshot_age_from_database(self) -> float:
        """Read the live age of this run's pg_stat_activity blocker."""
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select age(backend_xmin)::int8
                from pg_stat_activity
                where application_name = %s
                  and backend_xmin is not null
                """,
                (self.snapshot_appname,),
            )
            row = cur.fetchone()
        if not row or row[0] is None:
            raise RuntimeError("Activity snapshot blocker has no backend_xmin")
        return float(row[0])

    def metric_matches_database_age(
        self,
        metric_name: str,
        metric_value: float,
        database_age: float,
    ) -> bool:
        """Validate a scrape value against current DB age without stale blockers."""
        drift = abs(database_age - metric_value)
        if drift > ACTIVITY_AGE_TOLERANCE_TX:
            print(
                f"[fail] {metric_name} age {metric_value} is not within "
                f"{ACTIVITY_AGE_TOLERANCE_TX} tx of live DB age {database_age}"
            )
            return False
        print(
            f"[ok] {metric_name} age matches live DB age: "
            f"metric={metric_value}, db={database_age}, drift={drift}"
        )
        return True

    def verify_idle_metrics_zero(self) -> bool:
        """Validate summary metrics exist and owned-idle components are zero."""
        print("\nVerifying idle xmin_horizon baseline...")
        zero_components = {
            "pg_stat_activity",
            "pg_replication_slots",
            "pg_replication_slots_catalog",
            "pg_prepared_xacts",
        }
        if not self.standby_db_url:
            zero_components.add("pg_stat_replication")

        def baseline_ready() -> bool:
            snapshot_results = self.query_summary_snapshot()
            if not snapshot_results:
                return False

            results_by_name: dict[str, list[dict]] = {}
            for result in snapshot_results:
                metric_name = result.get("metric", {}).get("__name__")
                if metric_name:
                    results_by_name.setdefault(metric_name, []).append(result)

            required_names = {
                *COMPONENT_SUMMARY_METRICS.values(),
                *COMPONENT_COUNT_METRICS.values(),
                DATA_HORIZON_METRIC,
                CATALOG_HORIZON_METRIC,
                SNAPSHOT_XMIN_METRIC,
            }
            if any(name not in results_by_name for name in required_names):
                return False

            data_horizon_results = results_by_name[DATA_HORIZON_METRIC]
            catalog_horizon_results = results_by_name[CATALOG_HORIZON_METRIC]
            snapshot_xmin_results = results_by_name[SNAPSHOT_XMIN_METRIC]
            for data_horizon_result in data_horizon_results:
                if float(data_horizon_result["value"][0]) < self.started_at_seconds:
                    return False
                reference_metric = data_horizon_result["metric"]
                catalog_horizon_result = self.matching_result(
                    catalog_horizon_results,
                    reference_metric,
                )
                snapshot_xmin_result = self.matching_result(
                    snapshot_xmin_results,
                    reference_metric,
                )
                if not catalog_horizon_result or not snapshot_xmin_result:
                    return False
                if self.sample_value(snapshot_xmin_result) <= 0:
                    return False
                component_values = {}
                count_values = {}
                for component, metric_name in COMPONENT_SUMMARY_METRICS.items():
                    result = self.matching_result(
                        results_by_name[metric_name],
                        reference_metric,
                    )
                    if not result:
                        return False
                    component_values[component] = self.sample_value(result)
                for component, metric_name in COMPONENT_COUNT_METRICS.items():
                    result = self.matching_result(
                        results_by_name[metric_name],
                        reference_metric,
                    )
                    if not result:
                        return False
                    count_values[component] = self.sample_value(result)
                if any(component_values[component] != 0.0 for component in zero_components):
                    return False
                if any(count_values[component] != 0.0 for component in zero_components):
                    return False
                expected_data_horizon = max(
                    component_values[component] for component in DATA_COMPONENTS
                )
                expected_catalog_horizon = max(component_values.values())
                if self.sample_value(data_horizon_result) != expected_data_horizon:
                    return False
                if self.sample_value(catalog_horizon_result) != expected_catalog_horizon:
                    return False

            observed_components = {
                result.get("metric", {}).get("component")
                for result in self.query_detail()
                if result.get("metric", {}).get("component")
            }
            return not (observed_components & zero_components)

        self.wait_for_collection("idle xmin_horizon zero baseline", baseline_ready)
        print(
            "[ok] idle xmin_horizon baseline is present; owned components are zero"
        )
        return True

    def verify_mock_non_client_activity_xmin_excluded(self) -> bool:
        """Exercise the activity filter against mocked non-client rows."""
        with self.target_conn.transaction():
            with self.target_conn.cursor() as cur:
                cur.execute(
                    """
                    create temp table pg_stat_activity (
                        pid int not null,
                        backend_type text not null,
                        backend_xmin xid
                    ) on commit drop
                    """
                )
                cur.execute(
                    """
                    insert into pg_stat_activity (pid, backend_type, backend_xmin)
                    values
                      (1, 'client backend', txid_current()::text::xid),
                      (2, 'walsender', txid_current()::text::xid),
                      (pg_backend_pid(), 'client backend', txid_current()::text::xid)
                    """
                )
                cur.execute(
                    """
                    select count(*)::int, max(age(backend_xmin))::int8
                    from pg_stat_activity
                    where pid <> pg_backend_pid()
                      and backend_type = 'client backend'
                      and backend_xmin is not null
                    """
                )
                count, age_tx = cur.fetchone()

        if count != 1 or age_tx is None:
            print(
                "[fail] mocked non-client backend_xmin was not excluded: "
                f"count={count}, age={age_tx}"
            )
            return False
        print("[ok] mocked non-client backend_xmin is excluded by activity filter")
        return True

    def verify_non_client_activity_xmin_excluded(self) -> bool:
        """Assert live non-client backend_xmin rows do not inflate activity counts."""
        last_observation: dict[str, object] = {}

        def exclusion_ready() -> bool:
            detail_results = self.query_detail(component="pg_stat_activity")
            if not detail_results:
                last_observation["reason"] = "missing activity detail metric"
                return False

            reference_metric = detail_results[0]["metric"]
            count_result = self.matching_result(
                self.query_summary(COMPONENT_COUNT_METRICS["pg_stat_activity"]),
                reference_metric,
            )
            if not count_result:
                last_observation["reason"] = "missing activity count metric"
                return False

            with self.target_conn.cursor() as cur:
                cur.execute(
                    """
                    select
                      count(*) filter (where backend_type = 'client backend')::int,
                      count(*) filter (where backend_type <> 'client backend')::int,
                      coalesce(
                        array_to_string(
                          array_agg(distinct backend_type)
                            filter (where backend_type <> 'client backend'),
                          ','
                        ),
                        ''
                      )::text
                    from pg_stat_activity
                    where pid <> pg_backend_pid()
                      and backend_xmin is not null
                    """
                )
                client_count, non_client_count, non_client_types = cur.fetchone()

            metric_count = self.sample_value(count_result)
            last_observation.update(
                client_count=client_count,
                non_client_count=non_client_count,
                non_client_types=non_client_types,
                metric_count=metric_count,
            )
            return metric_count == float(client_count)

        try:
            self.wait_for_collection(
                "pg_stat_activity non-client backend_xmin exclusion",
                exclusion_ready,
            )
        except TimeoutError:
            print(
                "[fail] pg_stat_activity non-client exclusion failed: "
                f"{last_observation}"
            )
            return False

        print(
            "[ok] pg_stat_activity count excludes non-client backend_xmin rows: "
            f"{last_observation}"
        )
        return True

    def validate_blocker_cardinality_contract(self, reference_metric: dict) -> bool:
        """Assert blocker details emit one row per active component, none for empty."""
        last_observed: set[str] = set()
        expected_components = self.expected_blocker_components_from_database()
        deadline = time.monotonic() + self.collection_wait_seconds
        attempt = 0
        while True:
            attempt += 1
            detail_results = self.same_source_results(
                self.query_detail(),
                reference_metric,
            )
            last_observed = {
                result.get("metric", {}).get("component")
                for result in detail_results
                if result.get("metric", {}).get("component")
            }
            if last_observed == expected_components:
                print(
                    "[ok] xmin_horizon_blockers variable-cardinality contract "
                    f"holds: {sorted(last_observed)}"
                )
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            print(
                "[warn] xmin_horizon_blockers component set mismatch "
                f"on attempt {attempt}: observed={sorted(last_observed)}, "
                f"expected={sorted(expected_components)}"
            )
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

        print(
            "[fail] xmin_horizon_blockers component set mismatch: "
            f"observed={sorted(last_observed)}, expected={sorted(expected_components)}"
        )
        return False

    def component_metric_values(
        self,
        results_by_component: dict[str, list[dict]],
        reference_metric: dict,
    ) -> dict[str, float]:
        values = {}
        for component, results in results_by_component.items():
            result = self.matching_result(results, reference_metric)
            values[component] = self.sample_value(result) if result else 0.0
        return values

    def validate_metrics(
        self,
        summary_results_by_component: dict[str, list[dict]],
        count_results_by_component: dict[str, list[dict]],
        data_horizon_results: list[dict],
        catalog_horizon_results: list[dict],
        detail_results: list[dict],
    ) -> bool:
        """Validate summary/detail metrics and split horizon max semantics."""
        print("\nValidating xmin_horizon metric structure...")

        activity_results = summary_results_by_component["pg_stat_activity"]
        if not activity_results:
            print("[fail] Missing pg_stat_activity summary metric")
            return False
        if not data_horizon_results:
            print("[fail] Missing xmin data horizon metric")
            return False
        if not catalog_horizon_results:
            print("[fail] Missing xmin catalog horizon metric")
            return False
        if not detail_results:
            print("[fail] Missing xmin_horizon_blockers activity detail metric")
            return False

        detail_metric = detail_results[0]["metric"]
        summary_result = self.matching_result(activity_results, detail_metric)
        data_horizon_result = self.matching_result(data_horizon_results, detail_metric)
        catalog_horizon_result = self.matching_result(
            catalog_horizon_results,
            detail_metric,
        )

        if not summary_result:
            print(
                "[fail] Missing pg_stat_activity summary metric "
                "for test blocker source"
            )
            return False
        if not data_horizon_result:
            print("[fail] Missing xmin data horizon metric for test blocker source")
            return False
        if not catalog_horizon_result:
            print("[fail] Missing xmin catalog horizon metric for test blocker source")
            return False

        component_values = self.component_metric_values(
            summary_results_by_component,
            detail_metric,
        )
        count_values = self.component_metric_values(
            count_results_by_component,
            detail_metric,
        )
        expected_data_horizon = max(
            component_values[component] for component in DATA_COMPONENTS
        )
        expected_catalog_horizon = max(component_values.values())
        summary_value = component_values["pg_stat_activity"]
        data_horizon_value = self.sample_value(data_horizon_result)
        catalog_horizon_value = self.sample_value(catalog_horizon_result)
        detail_value = self.sample_value(detail_results[0])
        expected_counts = self.expected_blocker_counts_from_database()
        expected_components = {
            component for component, count in expected_counts.items() if count > 0
        }
        ok = self.validate_blocker_cardinality_contract(detail_metric)

        for component, expected_count in expected_counts.items():
            actual_count = count_values.get(component, 0.0)
            if expected_count == 0 and actual_count != 0.0:
                print(
                    f"[fail] inactive {component} count metric is {actual_count}, "
                    "expected 0"
                )
                ok = False
            elif expected_count > 0 and actual_count < float(expected_count):
                print(
                    f"[fail] active {component} count metric {actual_count} "
                    f"is below database lower bound {expected_count}"
                )
                ok = False
            else:
                print(
                    f"[ok] {component} count metric is valid: "
                    f"metric={actual_count}, db_lower_bound={expected_count}"
                )

        activity_database_age = self.activity_snapshot_age_from_database()
        if not self.metric_matches_database_age(
            "pg_stat_activity summary",
            summary_value,
            activity_database_age,
        ):
            ok = False

        activity_only = expected_components == {"pg_stat_activity"}
        if activity_only:
            if data_horizon_value != summary_value:
                print(
                    f"[fail] activity-only data horizon age {data_horizon_value} "
                    f"does not exactly match summary {summary_value}"
                )
                ok = False
            else:
                print(
                    f"[ok] activity-only data horizon exactly matches: "
                    f"{data_horizon_value}"
                )
        elif abs(data_horizon_value - expected_data_horizon) > HORIZON_MAX_TOLERANCE_TX:
            print(
                f"[fail] data horizon age {data_horizon_value} is not within "
                f"{HORIZON_MAX_TOLERANCE_TX} tx of expected "
                f"{expected_data_horizon} from {component_values}"
            )
            ok = False
        else:
            print(
                f"[ok] data horizon age matches max semantics: {data_horizon_value} "
                f"(expected {expected_data_horizon})"
            )

        if abs(catalog_horizon_value - expected_catalog_horizon) > HORIZON_MAX_TOLERANCE_TX:
            print(
                f"[fail] catalog horizon age {catalog_horizon_value} is not within "
                f"{HORIZON_MAX_TOLERANCE_TX} tx of expected "
                f"{expected_catalog_horizon} from {component_values}"
            )
            ok = False
        else:
            print(
                f"[ok] catalog horizon age matches max semantics: "
                f"{catalog_horizon_value} (expected {expected_catalog_horizon})"
            )

        expected_activity_labels = {
            "blocker_appname": self.snapshot_appname,
            "blocker_database": self.target_datname,
            "blocker_user": self.target_user,
        }
        for label, expected_value in expected_activity_labels.items():
            actual_value = detail_metric.get(label)
            label_matches = (
                actual_value == expected_value
                or (expected_value == "" and actual_value is None)
            )
            if not label_matches:
                print(
                    f"[fail] detail metric {label} mismatch: "
                    f"{actual_value} != {expected_value}"
                )
                ok = False
            else:
                print(f"[ok] detail metric {label} matches: {expected_value}")

        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select count(*)::int
                from pg_stat_activity
                where application_name = %s
                  and backend_xmin is not null
                """,
                (self.snapshot_appname,),
            )
            snapshot_blocker_count = cur.fetchone()[0]
        if snapshot_blocker_count != 1:
            print(
                "[fail] expected exactly one test snapshot blocker, "
                f"got {snapshot_blocker_count}"
            )
            ok = False
        else:
            print("[ok] test snapshot blocker is present in pg_stat_activity")

        valid_blocker_states = {"active", "idle", "idle in transaction"}
        if detail_metric.get("blocker_state") not in valid_blocker_states:
            print(
                "[fail] unexpected blocker_state: "
                f"{detail_metric.get('blocker_state')}"
            )
            ok = False
        else:
            print(f"[ok] blocker_state is valid: {detail_metric.get('blocker_state')}")

        if detail_metric.get("datname") != self.target_datname:
            print(
                "[fail] detail metric database mismatch: "
                f"{detail_metric.get('datname')} != {self.target_datname}"
            )
            ok = False
        else:
            print(f"[ok] detail metric database matches: {self.target_datname}")

        if not self.metric_matches_database_age(
            "pg_stat_activity detail",
            detail_value,
            activity_database_age,
        ):
            ok = False

        return ok

    def validate_component_metric(self, scenario: ComponentScenario) -> bool:
        """Validate summary and blocker-detail rows for an optional component."""
        if not scenario.active:
            print(f"[skip] {scenario.component} Prometheus check skipped")
            return True

        print(f"\nValidating {scenario.component} Prometheus metrics...")
        last_errors: list[str] = []
        last_detail_metric: dict = {}

        def component_metric_ready() -> bool:
            nonlocal last_errors, last_detail_metric
            errors: list[str] = []
            summary_results = self.query_summary(scenario.summary_metric)
            detail_results = self.query_detail(**scenario.detail_labels)

            if not summary_results:
                errors.append(f"Missing summary metric for {scenario.component}")
            elif max(self.sample_value(result) for result in summary_results) < MIN_BLOCKER_AGE_TX:
                errors.append(
                    f"{scenario.component} summary metric is below "
                    f"{MIN_BLOCKER_AGE_TX}"
                )

            if not detail_results:
                errors.append(f"Missing blocker detail metric for {scenario.component}")
            else:
                wrong_components = [
                    result.get("metric", {}).get("component")
                    for result in detail_results
                    if result.get("metric", {}).get("component") != scenario.component
                ]
                if wrong_components:
                    errors.append(f"Wrong detail component labels: {wrong_components}")
                elif max(self.sample_value(result) for result in detail_results) < MIN_BLOCKER_AGE_TX:
                    errors.append(
                        f"{scenario.component} detail metric is below "
                        f"{MIN_BLOCKER_AGE_TX}"
                    )
                else:
                    last_detail_metric = detail_results[0].get("metric", {})
                    for label, expected_value in (scenario.expected_labels or {}).items():
                        actual_value = last_detail_metric.get(label)
                        label_matches = (
                            actual_value == expected_value
                            or (expected_value == "" and actual_value is None)
                        )
                        if not label_matches:
                            errors.append(
                                f"{scenario.component} label {label} mismatch: "
                                f"{actual_value} != {expected_value}"
                            )

            last_errors = errors
            return not errors

        try:
            self.wait_for_collection(
                f"{scenario.component} valid summary/detail metrics",
                component_metric_ready,
            )
        except TimeoutError:
            for error in last_errors or ["metric validation did not become ready"]:
                print(f"[fail] {error}")
            return False

        print(f"[ok] {scenario.component} summary metric meets lower bound")
        print(f"[ok] {scenario.component} blocker detail metric is present")
        for label, expected_value in (scenario.expected_labels or {}).items():
            print(
                f"[ok] {scenario.component} label {label} matches: "
                f"{expected_value}"
            )
        return True

    def verify_activity_metric_collected(self) -> bool:
        """Check activity detail and exact split-horizon max semantics."""
        print("\nVerifying activity metric collection...")
        activity_labels = {"component": "pg_stat_activity"}

        def activity_metrics_ready() -> bool:
            detail_rows = self.query_detail(**activity_labels)
            positive_detail = next(
                (
                    result
                    for result in detail_rows
                    if self.sample_value(result) > 0
                ),
                None,
            )
            if not positive_detail:
                return False
            reference_metric = positive_detail["metric"]
            summary_result = self.matching_result(
                self.query_summary(COMPONENT_SUMMARY_METRICS["pg_stat_activity"]),
                reference_metric,
            )
            data_horizon_result = self.matching_result(
                self.query_summary(DATA_HORIZON_METRIC),
                reference_metric,
            )
            catalog_horizon_result = self.matching_result(
                self.query_summary(CATALOG_HORIZON_METRIC),
                reference_metric,
            )
            return bool(
                summary_result
                and data_horizon_result
                and catalog_horizon_result
                and self.sample_value(summary_result) > 0
                and self.sample_value(data_horizon_result) > 0
                and self.sample_value(catalog_horizon_result) > 0
            )

        self.wait_for_collection(
            "pg_stat_activity blocker metrics with positive age",
            activity_metrics_ready,
        )

        detail_results = self.query_detail(**activity_labels)
        summary_results_by_component = {
            component: self.query_summary(metric_name)
            for component, metric_name in COMPONENT_SUMMARY_METRICS.items()
        }
        count_results_by_component = {
            component: self.query_summary(metric_name)
            for component, metric_name in COMPONENT_COUNT_METRICS.items()
        }
        data_horizon_results = self.query_summary(DATA_HORIZON_METRIC)
        catalog_horizon_results = self.query_summary(CATALOG_HORIZON_METRIC)

        sample_counts = {
            key: len(value)
            for key, value in summary_results_by_component.items()
        }
        print(f"[ok] Summary samples: {sample_counts}")
        print(f"[ok] Data horizon samples: {len(data_horizon_results)}")
        print(f"[ok] Catalog horizon samples: {len(catalog_horizon_results)}")
        print(f"[ok] Activity detail samples: {len(detail_results)}")
        return self.validate_metrics(
            summary_results_by_component,
            count_results_by_component,
            data_horizon_results,
            catalog_horizon_results,
            detail_results,
        )

    def active_optional_scenarios(
        self,
        scenarios: list[ComponentScenario],
    ) -> list[ComponentScenario]:
        """Return optional blocker scenarios that are active in this run."""
        return [scenario for scenario in scenarios if scenario.active]

    def validate_required_optional_coverage(
        self,
        scenarios: list[ComponentScenario],
        physical_slot_active: bool,
        replication_active: bool,
    ) -> bool:
        """Fail CI when requested optional coverage did not execute."""
        scenario_active = {scenario.component: scenario.active for scenario in scenarios}
        requirements = {
            "REQUIRE_REPLICATION_SLOT_TEST": [
                (
                    "pg_replication_slots",
                    scenario_active.get("pg_replication_slots", False),
                ),
                (
                    "pg_replication_slots_catalog",
                    scenario_active.get("pg_replication_slots_catalog", False),
                ),
                ("replication_slots_null_xmin", physical_slot_active),
            ],
            "REQUIRE_PREPARED_XACTS_TEST": [
                ("pg_prepared_xacts", scenario_active.get("pg_prepared_xacts", False)),
            ],
            "REQUIRE_STANDBY_FEEDBACK_TEST": [
                ("pg_stat_replication", replication_active),
            ],
        }
        missing = [
            name
            for env_name, checks in requirements.items()
            if self.require_env(env_name)
            for name, active in checks
            if not active
        ]
        if missing:
            print(
                "[fail] Required optional xmin_horizon coverage did not run: "
                f"{', '.join(sorted(missing))}"
            )
            return False
        return True

    def warn_optional_skips(
        self,
        scenarios: list[ComponentScenario],
        physical_slot_active: bool,
    ) -> None:
        """Print an explicit warning when optional local coverage is skipped."""
        skipped = [scenario.component for scenario in scenarios if not scenario.active]
        if not physical_slot_active:
            skipped.append("replication_slots_null_xmin")
        if skipped:
            print(
                "\n[warn] Optional xmin_horizon coverage skipped: "
                f"{', '.join(sorted(skipped))}. Set REQUIRE_* flags in CI "
                "to make these paths mandatory."
            )

    def verify_replication_slots_null_xmin_prometheus(
        self,
        active: bool,
    ) -> bool:
        """Verify replication_slots leaves xmin_age_tx absent for null-xmin slots."""
        if not active:
            message = "replication_slots null-xmin Prometheus check skipped"
            if self.require_env("REQUIRE_REPLICATION_SLOT_TEST"):
                raise RuntimeError(message)
            print(f"\n[skip] {message}")
            return True

        selector = self.prometheus_selector(
            datname=self.target_datname,
            slot_name=self.physical_slot_name,
        )
        readiness_query = f"pgwatch_replication_slots_non_active_int{{{selector}}}"
        xmin_age_query = f"pgwatch_replication_slots_xmin_age_tx{{{selector}}}"
        self.wait_for_collection(
            "replication_slots null-xmin slot scrape",
            lambda: bool(self.query_prometheus(readiness_query)),
        )
        readiness_results = self.query_prometheus(readiness_query)
        physical_slot_metric = readiness_results[0]["metric"]
        xmin_age_results = self.query_prometheus(xmin_age_query)
        if xmin_age_results:
            value = self.sample_value(xmin_age_results[0])
            print(
                "[fail] replication_slots null-xmin metric is present "
                f"with value {value}; expected it to be absent"
            )
            return False
        print("[ok] replication_slots null-xmin slot leaves xmin_age_tx absent")

        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select count(*)::int
                from pg_replication_slots
                where xmin is not null
                """
            )
            expected_slot_count = cur.fetchone()[0]

        def slot_summary_excludes_null_slot() -> bool:
            count_result = self.matching_result(
                self.query_summary(COMPONENT_COUNT_METRICS["pg_replication_slots"]),
                physical_slot_metric,
            )
            age_result = self.matching_result(
                self.query_summary(COMPONENT_SUMMARY_METRICS["pg_replication_slots"]),
                physical_slot_metric,
            )
            if not count_result or not age_result:
                return False
            count_value = self.sample_value(count_result)
            age_value = self.sample_value(age_result)
            if count_value != float(expected_slot_count):
                return False
            return age_value > 0 if expected_slot_count > 0 else age_value == 0

        self.wait_for_collection(
            "xmin_horizon slot summary excluding null-xmin slot",
            slot_summary_excludes_null_slot,
        )
        count_result = self.matching_result(
            self.query_summary(COMPONENT_COUNT_METRICS["pg_replication_slots"]),
            physical_slot_metric,
        )
        age_result = self.matching_result(
            self.query_summary(COMPONENT_SUMMARY_METRICS["pg_replication_slots"]),
            physical_slot_metric,
        )
        count_value = self.sample_value(count_result)
        age_value = self.sample_value(age_result)
        if count_value != float(expected_slot_count):
            print(
                "[fail] xmin_horizon pg_replication_slots_count "
                f"is {count_value}, expected {expected_slot_count}"
            )
            return False
        print(
            "[ok] xmin_horizon pg_replication_slots_count excludes null slot: "
            f"{count_value}"
        )
        if expected_slot_count > 0 and age_value <= 0:
            print(
                "[fail] xmin_horizon pg_replication_slots_age_tx "
                f"is {age_value}, expected > 0"
            )
            return False
        if expected_slot_count == 0 and age_value != 0:
            print(
                "[fail] xmin_horizon pg_replication_slots_age_tx "
                f"is {age_value}, expected 0"
            )
            return False
        print(
            "[ok] xmin_horizon pg_replication_slots_age_tx is valid: "
            f"{age_value}"
        )
        return True

    def verify_optional_blocker_metrics(
        self,
        scenarios: list[ComponentScenario],
    ) -> bool:
        """Verify every active optional blocker component."""
        active = self.active_optional_scenarios(scenarios)
        if not active:
            print(
                "\n[skip] No optional xmin blocker scenarios active "
                "in this environment"
            )
            return True

        return all(self.validate_component_metric(scenario) for scenario in active)

    def verify_cross_component_horizon_metric(
        self,
        scenarios: list[ComponentScenario],
    ) -> bool:
        """Validate horizon=max(component ages) after optional blockers exist."""
        active = self.active_optional_scenarios(scenarios)
        if not active:
            print(
                "\n[skip] Cross-component horizon check skipped; "
                "no optional blockers"
            )
            return not any(
                self.require_env(name)
                for name in (
                    "REQUIRE_REPLICATION_SLOT_TEST",
                    "REQUIRE_PREPARED_XACTS_TEST",
                    "REQUIRE_STANDBY_FEEDBACK_TEST",
                )
            )

        print("\nValidating cross-component split-horizon max semantics...")
        required_components = {scenario.component for scenario in active}
        live_components = self.expected_blocker_components_from_database()
        missing_components = required_components - live_components
        if missing_components:
            print(
                "[fail] Cross-component blockers are no longer live: "
                f"missing={sorted(missing_components)}, live={sorted(live_components)}"
            )
            return False

        detail_results = self.query_detail(component="pg_stat_activity")
        summary_results_by_component = {
            component: self.query_summary(metric_name)
            for component, metric_name in COMPONENT_SUMMARY_METRICS.items()
        }
        count_results_by_component = {
            component: self.query_summary(metric_name)
            for component, metric_name in COMPONENT_COUNT_METRICS.items()
        }
        data_horizon_results = self.query_summary(DATA_HORIZON_METRIC)
        catalog_horizon_results = self.query_summary(CATALOG_HORIZON_METRIC)

        if not detail_results:
            print("[fail] Missing activity detail reference for cross-component check")
            return False

        detail_metric = detail_results[0]["metric"]
        component_values = self.component_metric_values(
            summary_results_by_component,
            detail_metric,
        )
        non_zero_components = {
            component: value
            for component, value in component_values.items()
            if value > 0
        }
        if len(non_zero_components) < 2:
            print(
                "[fail] Cross-component check needs at least two non-zero "
                f"components, got {non_zero_components}"
            )
            return False

        print(f"[ok] Cross-component inputs: {non_zero_components}")
        return self.validate_metrics(
            summary_results_by_component,
            count_results_by_component,
            data_horizon_results,
            catalog_horizon_results,
            detail_results,
        )

    def drop_replication_slot_if_exists(self, slot_name: str) -> None:
        """Terminate active users and drop a test replication slot if it exists."""
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                select active, active_pid
                from pg_replication_slots
                where slot_name = %s
                """,
                (slot_name,),
            )
            row = cur.fetchone()
            if not row:
                return

            active, active_pid = row
            if active_pid is not None:
                print(
                    f"[warn] Terminating active replication slot backend "
                    f"{active_pid} for {slot_name}"
                )
                cur.execute("select pg_terminate_backend(%s)", (active_pid,))
                deadline = time.monotonic() + 10
                while time.monotonic() <= deadline:
                    cur.execute(
                        """
                        select active
                        from pg_replication_slots
                        where slot_name = %s
                        """,
                        (slot_name,),
                    )
                    active_row = cur.fetchone()
                    if not active_row:
                        return
                    if not active_row[0]:
                        active = False
                        break
                    time.sleep(0.5)

            if active:
                raise RuntimeError(f"Replication slot {slot_name} is still active")

            cur.execute("select pg_drop_replication_slot(%s)", (slot_name,))
            cur.execute(
                "select 1 from pg_replication_slots where slot_name = %s",
                (slot_name,),
            )
            if cur.fetchone():
                raise RuntimeError(f"Replication slot {slot_name} still exists")

    def cleanup(self) -> None:
        """Release blockers and remove test artifacts."""
        print("\nCleaning up...")

        if self.snapshot_conn:
            try:
                self.snapshot_conn.rollback()
            except Exception as exc:
                print(f"[warn] Could not roll back snapshot connection: {exc}")
            try:
                self.snapshot_conn.close()
            except Exception as exc:
                print(f"[warn] Could not close snapshot connection: {exc}")

        if self.standby_snapshot_conn:
            try:
                self.standby_snapshot_conn.rollback()
            except Exception as exc:
                print(f"[warn] Could not roll back standby snapshot: {exc}")
            try:
                self.standby_snapshot_conn.close()
            except Exception as exc:
                print(f"[warn] Could not close standby snapshot: {exc}")

        if self.target_conn:
            if self.standby_slot_configured and self.standby_db_url:
                try:
                    with psycopg.connect(
                        self.standby_db_url,
                        connect_timeout=5,
                    ) as standby_conn:
                        standby_conn.autocommit = True
                        with standby_conn.cursor() as cur:
                            cur.execute("alter system reset primary_slot_name")
                            cur.execute("select pg_reload_conf()")
                except Exception as exc:
                    print(f"[warn] Could not reset standby slot config: {exc}")

            with self.target_conn.cursor() as cur:
                if self.prepared_xact_created:
                    try:
                        cur.execute(
                            sql.SQL("rollback prepared {}").format(
                                sql.Literal(self.prepared_gid)
                            )
                        )
                    except Exception as exc:
                        print(f"[warn] Could not rollback prepared xact: {exc}")
            for slot_name in (
                self.logical_slot_name,
                self.physical_slot_name,
                self.standby_physical_slot_name,
            ):
                try:
                    self.drop_replication_slot_if_exists(slot_name)
                except Exception as exc:
                    message = f"Could not drop replication slot {slot_name}: {exc}"
                    self.cleanup_errors.append(message)
                    print(f"[fail] {message}")
            with self.target_conn.cursor() as cur:
                try:
                    cur.execute("drop table if exists xmin_horizon_test_table cascade")
                except Exception as exc:
                    print(f"[warn] Could not drop test table: {exc}")
            try:
                self.target_conn.close()
            except Exception:
                pass

        if self.cleanup_errors:
            raise RuntimeError("; ".join(self.cleanup_errors))

        print("[ok] Cleanup complete")

    def run(self) -> bool:
        try:
            self.setup()
            if not self.verify_idle_metrics_zero():
                print(
                    "\nTest FAILED: dirty xmin_horizon baseline; "
                    "aborting before creating blockers"
                )
                return False
            mock_non_client_filter_valid = (
                self.verify_mock_non_client_activity_xmin_excluded()
            )
            self.create_repeatable_read_snapshot()
            activity_valid = self.verify_activity_metric_collected()

            slot_active = self.create_logical_replication_slot()
            physical_slot_active = self.create_physical_null_xmin_slot()
            prepared_active = self.create_prepared_transaction()
            replication_active = self.observe_standby_feedback()
            replication_scenario = ComponentScenario(
                component="pg_stat_replication",
                summary_metric=(
                    "pgwatch_xmin_horizon_pg_stat_replication_age_tx"
                ),
                detail_labels={"component": "pg_stat_replication"},
                expected_labels=self.replication_feedback_labels,
                active=replication_active,
            )
            replication_valid = self.validate_component_metric(replication_scenario)
            standby_slot_active = self.configure_standby_replication_slot()
            if standby_slot_active:
                self.capture_replication_slot_xmin_component()
            non_client_activity_exclusion_valid = (
                self.verify_non_client_activity_xmin_excluded()
            )

            optional_scenarios = [
                ComponentScenario(
                    component="pg_replication_slots",
                    summary_metric=COMPONENT_SUMMARY_METRICS["pg_replication_slots"],
                    detail_labels={
                        "component": "pg_replication_slots",
                        "slot_xmin_source": "xmin",
                    },
                    expected_labels=self.slot_xmin_labels,
                    active=bool(self.slot_xmin_labels),
                ),
                ComponentScenario(
                    component="pg_replication_slots_catalog",
                    summary_metric=(
                        COMPONENT_SUMMARY_METRICS["pg_replication_slots_catalog"]
                    ),
                    detail_labels={
                        "component": "pg_replication_slots_catalog",
                        "slot_xmin_source": "catalog_xmin",
                    },
                    expected_labels={
                        "horizon_type": "catalog",
                        "blocker_database": self.target_datname,
                        "slot_name": self.logical_slot_name,
                        "slot_type": "logical",
                        "slot_plugin": "test_decoding",
                        "slot_xmin_source": "catalog_xmin",
                    },
                    active=(
                        slot_active
                        and "pg_replication_slots_catalog"
                        in self.logical_slot_components
                    ),
                ),
                ComponentScenario(
                    component="pg_prepared_xacts",
                    summary_metric="pgwatch_xmin_horizon_pg_prepared_xacts_age_tx",
                    detail_labels={"component": "pg_prepared_xacts"},
                    expected_labels={
                        "blocker_database": self.target_datname,
                        "prepared_gid": self.prepared_gid,
                        "owner": self.target_user,
                    },
                    active=prepared_active,
                ),
            ]
            coverage_valid = self.validate_required_optional_coverage(
                optional_scenarios,
                physical_slot_active,
                replication_active,
            )
            self.warn_optional_skips(optional_scenarios, physical_slot_active)
            optional_valid = self.verify_optional_blocker_metrics(optional_scenarios)
            null_slot_valid = self.verify_replication_slots_null_xmin_prometheus(
                physical_slot_active
            )
            cross_component_valid = self.verify_cross_component_horizon_metric(
                optional_scenarios
            )

            if (
                coverage_valid
                and mock_non_client_filter_valid
                and activity_valid
                and replication_valid
                and non_client_activity_exclusion_valid
                and optional_valid
                and null_slot_valid
                and cross_component_valid
            ):
                print("\nTest PASSED: xmin_horizon metric is working correctly")
                return True

            print("\nTest FAILED: xmin_horizon metric validation failed")
            return False
        except Exception as exc:
            print(f"\nTest ERROR: {exc}")
            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test xmin_horizon metric collection")
    parser.add_argument(
        "--target-db-url",
        default=os.getenv(
            "TARGET_DB_URL",
            "postgresql://postgres:postgres@localhost:55432/target_database",
        ),
        help="Target database connection URL",
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:59090"),
        help="Prometheus/VictoriaMetrics API URL",
    )
    parser.add_argument(
        "--standby-db-url",
        default=os.getenv("STANDBY_DB_URL"),
        help="Optional standby database URL for hot_standby_feedback coverage",
    )
    parser.add_argument(
        "--collection-wait",
        type=int,
        default=int(os.getenv("COLLECTION_WAIT_SECONDS", "480")),
        help="Seconds to wait for pgwatch to collect metrics",
    )
    parser.add_argument(
        "--prometheus-username",
        default=env_with_fallback("PROMETHEUS_USERNAME", "VM_AUTH_USERNAME"),
        help="Optional Prometheus/VictoriaMetrics basic auth username",
    )
    parser.add_argument(
        "--prometheus-password",
        default=env_with_fallback("PROMETHEUS_PASSWORD", "VM_AUTH_PASSWORD"),
        help="Optional Prometheus/VictoriaMetrics basic auth password",
    )

    args = parser.parse_args()

    test = xminHorizonTest(
        target_db_url=args.target_db_url,
        prometheus_url=args.prometheus_url,
        standby_db_url=args.standby_db_url,
        collection_wait_seconds=args.collection_wait,
        prometheus_username=args.prometheus_username,
        prometheus_password=args.prometheus_password,
    )

    success = test.run()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
