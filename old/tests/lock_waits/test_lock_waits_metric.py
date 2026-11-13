"""
Test script to verify lock_waits metric collection.

This script:
1. Creates lock contention scenarios in the target database
2. Waits for pgwatch to collect metrics
3. Verifies the lock_waits metric is collected in Prometheus
4. Validates the data structure and content
"""

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import psycopg
import requests


class LockWaitsTest:
    def __init__(
        self,
        target_db_url: str,
        prometheus_url: str,
        test_dbname: str = "target_database",
        collection_wait_seconds: int = 60,
    ):
        """
        Initialize the test.

        Args:
            target_db_url: Connection string for the target database being monitored
            prometheus_url: URL for Prometheus/VictoriaMetrics API
            test_dbname: Name of the database being monitored
            collection_wait_seconds: How long to wait for pgwatch to collect metrics
        """
        self.target_db_url = target_db_url
        self.prometheus_url = prometheus_url.rstrip("/")
        self.test_dbname = test_dbname
        self.collection_wait_seconds = collection_wait_seconds
        self.target_conn: Optional[psycopg.Connection] = None
        self.blocker_conn: Optional[psycopg.Connection] = None

    def setup(self):
        """Set up database connections and test table."""
        print("Setting up test environment...")

        # Connect to target database
        self.target_conn = psycopg.connect(self.target_db_url)
        self.target_conn.autocommit = True

        # Verify Prometheus is accessible
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/status/config", timeout=5)
            response.raise_for_status()
            print("✓ Prometheus connection verified")
        except Exception as e:
            print(f"⚠ Warning: Could not verify Prometheus connection: {e}")

        # Create test table
        with self.target_conn.cursor() as cur:
            cur.execute(
                """
                drop table if exists lock_test_table cascade;
                create table lock_test_table (
                    id int8 generated always as identity primary key,
                    name text not null,
                    value numeric(10, 2),
                    created_at timestamptz default now()
                );
                insert into lock_test_table (name, value)
                values
                    ('Item 1', 100.50),
                    ('Item 2', 200.75),
                    ('Item 3', 300.25);
                """
            )
        print("✓ Test table created")

    def create_lock_contention(self, duration_seconds: int = 30):
        """
        Create lock contention by:
        1. Starting a transaction that locks a row
        2. Starting another transaction that tries to lock the same row (will wait)
        3. Keeping both transactions open for the specified duration
        """
        print(f"\nCreating lock contention for {duration_seconds} seconds...")

        # Connection 1: Blocker - acquires lock and holds it
        self.blocker_conn = psycopg.connect(self.target_db_url)
        self.blocker_conn.autocommit = False
        blocker_cur = self.blocker_conn.cursor()
        blocker_cur.execute("begin")
        blocker_cur.execute(
            "select * from lock_test_table where id = 1 for update"
        )
        blocker_cur.fetchone()
        print("✓ Blocker transaction started (holding lock on row id=1)")

        # Small delay to ensure blocker has the lock
        time.sleep(1)

        # Connection 2: Waiter - tries to acquire same lock (will wait)
        waiter_conn = psycopg.connect(self.target_db_url)
        waiter_conn.autocommit = False
        waiter_cur = waiter_conn.cursor()
        waiter_cur.execute("begin")
        print("✓ Waiter transaction started (waiting for lock on row id=1)")

        # Execute the waiting query in a separate thread so it can block
        waiter_error = []
        waiter_done = threading.Event()

        def run_waiter():
            try:
                # This will block until blocker releases the lock
                waiter_cur.execute(
                    "select * from lock_test_table where id = 1 for update"
                )
                waiter_cur.fetchone()
                print("  ✓ Waiter acquired lock (blocker released)")
            except Exception as e:
                waiter_error.append(str(e))
                print(f"  Waiter error: {e}")
            finally:
                waiter_done.set()

        waiter_thread = threading.Thread(target=run_waiter, daemon=True)
        waiter_thread.start()

        # Give waiter time to start waiting
        time.sleep(2)

        # Verify waiter is actually waiting
        with self.target_conn.cursor() as check_cur:
            check_cur.execute(
                """
                select pid, state, wait_event_type, wait_event
                from pg_stat_activity
                where datname = current_database()
                and pid <> pg_backend_pid()
                and wait_event_type = 'Lock'
                """
            )
            waiting_pids = check_cur.fetchall()
            if waiting_pids:
                print(f"  ✓ Confirmed {len(waiting_pids)} process(es) waiting for locks")
                for pid, state, wait_type, wait_event in waiting_pids:
                    print(f"    PID {pid}: state={state}, wait_event={wait_event}")
            else:
                print("  ⚠ No processes found waiting for locks")

        # Keep locks held for the duration
        print(f"  Holding locks for {duration_seconds} seconds...")
        time.sleep(duration_seconds)

        # Cleanup: commit blocker first, then waiter
        print("  Releasing blocker lock...")
        blocker_cur.execute("commit")
        blocker_cur.close()
        self.blocker_conn.close()
        self.blocker_conn = None

        # Wait for waiter to complete
        waiter_done.wait(timeout=5)
        try:
            waiter_cur.execute("commit")
        except Exception:
            pass
        waiter_cur.close()
        waiter_conn.close()

        print("✓ Lock contention ended")

    def verify_metric_collected(self) -> List[Dict]:
        """
        Verify that lock_waits metric was collected in Prometheus.

        Returns:
            List of lock_waits metric samples found
        """
        print("\nVerifying metric collection...")

        # Wait for pgwatch to collect metrics
        print(f"  Waiting {self.collection_wait_seconds} seconds for pgwatch to collect metrics...")
        time.sleep(self.collection_wait_seconds)

        # Query Prometheus for lock_waits metrics
        # pgwatch exports metrics with prefix pgwatch_<metric_name>_<field>
        metrics_to_check = [
            "pgwatch_lock_waits_waiting_ms",
            "pgwatch_lock_waits_blocker_tx_ms",
        ]

        records = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        for metric_name in metrics_to_check:
            try:
                # Query for recent samples
                query = f'{metric_name}{{datname="{self.test_dbname}"}}'
                response = requests.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={
                        "query": query,
                        "time": datetime.now(timezone.utc).timestamp(),
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    for result in data["data"]["result"]:
                        metric = result.get("metric", {})
                        value = result.get("value", [None, None])
                        
                        # Convert timestamp
                        timestamp = float(value[0]) if value[0] else None
                        if timestamp:
                            metric_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                            if metric_time >= cutoff_time:
                                records.append(
                                    {
                                        "time": metric_time,
                                        "metric": metric_name,
                                        "labels": metric,
                                        "value": float(value[1]) if value[1] else None,
                                    }
                                )
            except Exception as e:
                print(f"  ⚠ Error querying {metric_name}: {e}")

        print(f"  ✓ Found {len(records)} lock_waits metric samples")

        return records

    def validate_metric_structure(self, records: List[Dict]) -> bool:
        """
        Validate that the metric records have the expected structure.

        Args:
            records: List of metric samples to validate

        Returns:
            True if validation passes, False otherwise
        """
        if not records:
            print("  ⚠ No records to validate")
            return False

        print("\nValidating metric structure...")

        # Expected labels in Prometheus metrics
        expected_labels = [
            "datname",
            "waiting_user",
            "waiting_appname",
            "waiting_table",
            "waiting_query_id",
            "waiting_mode",
            "waiting_locktype",
            "waiting_pid",
            "blocker_user",
            "blocker_appname",
            "blocker_table",
            "blocker_query_id",
            "blocker_mode",
            "blocker_locktype",
            "blocker_pid",
        ]

        all_valid = True
        unique_samples = {}
        
        # Group samples by their label combination
        for record in records:
            labels = record.get("labels", {})
            # Create a key from relevant labels
            key = (
                labels.get("waiting_pid"),
                labels.get("blocker_pid"),
                labels.get("waiting_table"),
            )
            if key not in unique_samples:
                unique_samples[key] = record

        print(f"  Found {len(unique_samples)} unique lock wait samples")

        for i, (key, record) in enumerate(list(unique_samples.items())[:5]):  # Validate first 5
            print(f"\n  Sample {i+1}:")
            labels = record.get("labels", {})
            metric_name = record.get("metric", "")
            value = record.get("value")

            # Check datname matches
            if labels.get("datname") != self.test_dbname:
                print(f"    ⚠ datname mismatch: {labels.get('datname')} != {self.test_dbname}")
            else:
                print(f"    ✓ datname matches: {labels.get('datname')}")

            # Check key labels are present
            key_labels = ["waiting_pid", "blocker_pid", "waiting_mode", "blocker_mode"]
            missing_labels = [label for label in key_labels if not labels.get(label)]
            if missing_labels:
                print(f"    ⚠ Missing key labels: {missing_labels}")
            else:
                print(f"    ✓ Key labels present")

            # Validate metric value
            if value is not None:
                try:
                    float(value)
                    print(f"    ✓ Metric value is numeric: {value}")
                    if "waiting_ms" in metric_name or "blocker_tx_ms" in metric_name:
                        print(f"      Value: {value} ms")
                except (ValueError, TypeError):
                    print(f"    ✗ Metric value is not numeric: {value}")
                    all_valid = False
            else:
                print(f"    ⚠ Metric value is None")

        return all_valid

    def cleanup(self):
        """Clean up test resources."""
        print("\nCleaning up...")

        if self.blocker_conn:
            try:
                self.blocker_conn.close()
            except Exception:
                pass

        if self.target_conn:
            try:
                with self.target_conn.cursor() as cur:
                    cur.execute("drop table if exists lock_test_table cascade")
                self.target_conn.close()
            except Exception:
                pass

        print("✓ Cleanup complete")

    def run(self) -> bool:
        """
        Run the complete test.

        Returns:
            True if test passes, False otherwise
        """
        try:
            self.setup()
            self.create_lock_contention(duration_seconds=30)
            records = self.verify_metric_collected()
            is_valid = self.validate_metric_structure(records)

            if is_valid and records:
                print("\n✅ Test PASSED: lock_waits metric is working correctly")
                return True
            else:
                print("\n❌ Test FAILED: lock_waits metric validation failed")
                return False

        except Exception as e:
            print(f"\n❌ Test ERROR: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def main():
    """Main entry point for the test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test lock_waits metric collection"
    )
    parser.add_argument(
        "--target-db-url",
        default=os.getenv(
            "TARGET_DB_URL", "postgresql://postgres:postgres@localhost:55432/target_database"
        ),
        help="Target database connection URL",
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv(
            "PROMETHEUS_URL",
            "http://localhost:59090",
        ),
        help="Prometheus/VictoriaMetrics API URL",
    )
    parser.add_argument(
        "--test-dbname",
        default=os.getenv("TEST_DBNAME", "target_database"),
        help="Name of the database being monitored",
    )
    parser.add_argument(
        "--collection-wait",
        type=int,
        default=int(os.getenv("COLLECTION_WAIT_SECONDS", "60")),
        help="Seconds to wait for pgwatch to collect metrics",
    )

    args = parser.parse_args()

    test = LockWaitsTest(
        target_db_url=args.target_db_url,
        prometheus_url=args.prometheus_url,
        test_dbname=args.test_dbname,
        collection_wait_seconds=args.collection_wait,
    )

    success = test.run()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

