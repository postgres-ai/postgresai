# Lock waits metric testing

This directory contains tests and scripts to verify that the `lock_waits` metric is working correctly.

## Overview

The `lock_waits` metric collects detailed information about lock waits in Postgres, including:
- Waiting and blocking process IDs
- User names and application names
- Lock modes and types
- Affected tables
- Query IDs (PostgreSQL 14+)
- Wait durations and blocker transaction durations

## Test components

### 1. Python test script (`test_lock_waits_metric.py`)

Automated test that:
- Creates lock contention scenarios in the target database
- Waits for pgwatch to collect metrics
- Verifies the metric is collected in Prometheus/VictoriaMetrics
- Validates the metric structure and labels

### 2. SQL script (`create_lock_contention.sql`)

Manual SQL script to create lock contention for testing. Can be run in multiple psql sessions.

## Prerequisites

1. Docker Compose stack running:
   ```bash
   docker-compose up -d
   ```

2. Python dependencies:
   ```bash
   pip install psycopg requests
   ```

3. Ensure `lock_waits` metric is enabled in pgwatch configuration:
   - Check `config/pgwatch-prometheus/metrics.yml` includes `lock_waits`
   - Verify pgwatch is collecting metrics from the target database

## Running the automated test

### Basic usage

```bash
# From the project root
python tests/lock_waits/test_lock_waits_metric.py
```

### With custom configuration

```bash
python tests/lock_waits/test_lock_waits_metric.py \
  --target-db-url "postgresql://postgres:postgres@localhost:55432/target_database" \
  --prometheus-url "http://localhost:59090" \
  --test-dbname "target_database" \
  --collection-wait 90
```

### Environment variables

You can also set these via environment variables:

```bash
export TARGET_DB_URL="postgresql://postgres:postgres@localhost:55432/target_database"
export PROMETHEUS_URL="http://localhost:59090"
export TEST_DBNAME="target_database"
export COLLECTION_WAIT_SECONDS=90

python tests/lock_waits/test_lock_waits_metric.py
```

## Manual testing

### Step 1: create lock contention

Open two psql sessions to the target database:

**Session 1 (Blocker):**
```sql
BEGIN;
SELECT * FROM lock_test_table WHERE id = 1 FOR UPDATE;
-- Keep this transaction open
```

**Session 2 (Waiter):**
```sql
BEGIN;
SELECT * FROM lock_test_table WHERE id = 1 FOR UPDATE;
-- This will wait for Session 1 to release the lock
```

### Step 2: verify metric collection

Wait for pgwatch to collect metrics (check collection interval in pgwatch config, typically 15-30 seconds), then query Prometheus:

```bash
# Query Prometheus API for lock_waits metrics
curl "http://localhost:59090/api/v1/query?query=pgwatch_lock_waits_waiting_ms{datname=\"target_database\"}"

# Or use PromQL in Grafana Explore
pgwatch_lock_waits_waiting_ms{datname="target_database"}
pgwatch_lock_waits_blocker_tx_ms{datname="target_database"}
```

### Step 3: check Grafana dashboard

1. Open Grafana: http://localhost:3000
2. Navigate to "Lock waits details" dashboard
3. Select the database from the dropdown
4. Verify that lock wait events appear in the panels

## Expected results

### Successful test output

```
Setting up test environment...
✓ Test table created

Creating lock contention for 30 seconds...
✓ Blocker transaction started (holding lock on row id=1)
✓ Waiter transaction started (waiting for lock on row id=1)
  Holding locks for 30 seconds...
✓ Lock contention ended

Verifying metric collection...
  Waiting 60 seconds for pgwatch to collect metrics...
  ✓ Found 5 lock_waits records

Validating metric structure...

  Record 1:
    ✓ All required data fields present
    ✓ waiting_ms is numeric: 25000 ms
    ✓ blocker_tx_ms is numeric: 30000 ms

✅ Test PASSED: lock_waits metric is working correctly
```

## Troubleshooting

### No records found

- **Check pgwatch is running**: `docker ps | grep pgwatch-prometheus`
- **Check pgwatch logs**: `docker logs pgwatch-prometheus`
- **Verify metric is enabled**: Check `config/pgwatch-prometheus/metrics.yml`
- **Check Prometheus is accessible**: `curl http://localhost:59090/api/v1/status/config`
- **Increase wait time**: Use `--collection-wait 120` to wait longer
- **Check database name**: Ensure `--test-dbname` matches the monitored database
- **Verify metrics exist**: `curl "http://localhost:59090/api/v1/label/__name__/values" | grep lock_waits`

### Invalid data structure

- **Check PostgreSQL version**: Metric requires PostgreSQL 14+ for query_id support
- **Verify metric SQL**: Check the SQL query in `metrics.yml` is correct
- **Check pgwatch version**: Ensure pgwatch version supports the metric format
- **Check Prometheus labels**: Verify metrics have expected labels (datname, waiting_pid, blocker_pid, etc.)

### Connection errors

- **Verify Docker containers**: `docker-compose ps`
- **Check connection strings**: Verify URLs match your docker-compose configuration
- **Check Prometheus URL**: Ensure Prometheus/VictoriaMetrics is accessible at the specified URL
- **Check network**: Ensure containers can communicate (same Docker network)

## Integration with CI/CD

The test can be integrated into CI/CD pipelines:

```yaml
# Example GitLab CI
test_lock_waits:
  stage: test
  script:
    - docker-compose up -d
    - sleep 30  # Wait for services to start
    - pip install psycopg
    - python tests/lock_waits/test_lock_waits_metric.py
      --target-db-url "$TARGET_DB_URL"
      --sink-db-url "$SINK_DB_URL"
      --collection-wait 90
  only:
    - merge_requests
    - main
```

## Additional test scenarios

### Test different lock types

Modify the test to create different types of locks:

```sql
-- Table-level lock
LOCK TABLE lock_test_table IN EXCLUSIVE MODE;

-- Advisory lock
SELECT pg_advisory_lock(12345);
```

### Test multiple concurrent waits

Create multiple waiting transactions to test the LIMIT clause:

```sql
-- Session 1: Blocker
BEGIN;
SELECT * FROM lock_test_table WHERE id = 1 FOR UPDATE;

-- Sessions 2-10: Multiple waiters
-- Each in separate psql session
BEGIN;
SELECT * FROM lock_test_table WHERE id = 1 FOR UPDATE;
```

## Related files

- `config/pgwatch-prometheus/metrics.yml` - Metric definition
- `config/grafana/dashboards/Dashboard_13_Lock_waits.json` - Grafana dashboard
- `workload_examples/lock_wait_test.sql` - Basic lock test SQL

