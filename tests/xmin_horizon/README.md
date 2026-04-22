# xmin horizon metric testing

This directory contains an integration-style test for the `xmin_horizon` monitoring feature.

## What it covers

The test always validates the `pg_stat_activity` path by:

- opening a repeatable-read transaction to create a stable snapshot
- confirming `backend_xmin` is populated for that backend
- polling Prometheus until pgwatch exposes the new detail metric
- checking the summary metric `pgwatch_xmin_horizon_pg_stat_activity_age_tx`
- checking the split horizon metrics `pgwatch_xmin_horizon_data_horizon_age_tx` and `pgwatch_xmin_horizon_catalog_horizon_age_tx`
- checking the detail metric scoped to the target database and per-run `application_name`
- asserting split horizon max semantics; in an activity-only environment, data and catalog horizons equal the activity summary value exactly

The test also creates and validates additional real blockers when the target PostgreSQL instance supports them:

- `pg_replication_slots`: creates a logical replication slot and verifies `xmin` and `catalog_xmin` summary/detail component rows separately when each source is populated
- `pg_prepared_xacts`: creates a prepared transaction when `max_prepared_transactions > 0` and verifies summary/detail component rows
- `pg_stat_replication`: when `STANDBY_DB_URL` is set, opens a repeatable-read transaction on that standby, advances the primary, and verifies hot-standby-feedback rows with `backend_xmin`.

Static SQL regression checks run before the integration flow to catch unsafe changes in `metrics.yml` without waiting for Prometheus collection.

Replication-slot regression checks run directly against PostgreSQL, independent of Prometheus collection timing:

- a physical slot with null `xmin` and `catalog_xmin` must leave the existing `pgwatch_replication_slots_xmin_age_tx` metric absent
- the `xmin_horizon` slot summary must count only slots with populated `xmin`, while `pg_replication_slots_catalog` counts only slots with populated `catalog_xmin`
- a logical slot with a populated `catalog_xmin` must propagate through the catalog horizon path instead of being folded into the data slot path

The `xmin_horizon_blockers` metric intentionally has variable cardinality: each scrape emits one row per active component and no row for an empty source, so each monitored source can expose 0-5 component rows. Detail labels identify the selected top blocker, including horizon type, activity `queryid`, slot name/type/status/xmin-source fields, standby name, and prepared transaction `gid`; query text stays out of Prometheus and should be resolved from query storage by `queryid`. The test compares the emitted component set with direct PostgreSQL source checks to catch stale or missing rows.

Unsupported optional paths are reported as skipped by default. The GitLab `integration:xmin-horizon` job runs the replication-slot and prepared-xact paths with:

- `REQUIRE_REPLICATION_SLOT_TEST=1`
- `REQUIRE_PREPARED_XACTS_TEST=1`

The GitLab `integration:xmin-horizon` job also starts the local `target-standby` service and sets:

- `STANDBY_DB_URL=postgresql://postgres:postgres@target-standby:5432/target_database`
- `REQUIRE_STANDBY_FEEDBACK_TEST=1`

The `integration:xmin-horizon:standby-feedback` CI job is available for external multi-host coverage when `XMIN_HORIZON_STANDBY_TARGET_DB_URL`, `XMIN_HORIZON_STANDBY_DB_URL`, and `XMIN_HORIZON_STANDBY_PROMETHEUS_URL` are configured.

## Prerequisites

1. The monitoring stack is running.
2. Python dependencies are installed from `tests/xmin_horizon/requirements.txt`.
3. The target database is a primary and is being monitored by pgwatch.
4. Optional blocker paths require matching PostgreSQL configuration:
   - logical slots require `wal_level=logical` and the `test_decoding` output plugin
   - prepared transactions require `max_prepared_transactions > 0`
   - standby-feedback coverage requires `STANDBY_DB_URL` for an attached standby with `hot_standby_feedback=on`, or an environment that already reports `backend_xmin`

## Run

```bash
./tests/xmin_horizon/run_test.sh
```

Or run the Python script directly:

```bash
python3 tests/xmin_horizon/test_xmin_horizon_metric.py \
  --target-db-url "postgresql://postgres:postgres@localhost:55432/target_database" \
  --standby-db-url "postgresql://postgres:postgres@localhost:55433/target_database" \
  --prometheus-url "http://localhost:59090" \
  --collection-wait 240
```

You can also use environment variables:

```bash
export TARGET_DB_URL="postgresql://postgres:postgres@localhost:55432/target_database"
export PROMETHEUS_URL="http://localhost:59090"
export STANDBY_DB_URL="postgresql://postgres:postgres@localhost:55433/target_database"
export COLLECTION_WAIT_SECONDS=240
export REQUIRE_REPLICATION_SLOT_TEST=1
export REQUIRE_PREPARED_XACTS_TEST=1
export REQUIRE_STANDBY_FEEDBACK_TEST=1

./tests/xmin_horizon/run_test.sh
```
