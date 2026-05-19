# pg_index_pilot Test Suite

## Overview

This directory contains the automated test suite for pg_index_pilot. The tests are designed to run in CI/CD pipelines (GitLab CI) and can also be executed locally for development.

## Test Structure

### Core Tests
- `01_basic_installation.sql` - Verifies schema, tables, and functions are properly installed
- `02_functionality.sql` - Tests core functionality including index detection, bloat estimation, and reindexing
- `03_security.sql` - Security and permission tests, SQL injection protection

### Test Runner
- `run_tests.sh` - Shell script that orchestrates test execution
  - Handles PostgreSQL connection
  - Runs tests in sequence
  - Generates JUnit XML output for CI
  - Supports multiple PostgreSQL versions (13+)

## Running Tests Locally

### Prerequisites
- PostgreSQL 13 or higher
- psql client
- Bash shell

### Quick Start
```bash
# Run all tests with default settings (localhost)
./run_tests.sh

# Run tests against specific database
./run_tests.sh -h myhost -p 5432 -d testdb -u myuser -w mypass

# Install only (useful for manual testing)
./run_tests.sh -i

# Skip installation (if already installed)
./run_tests.sh -s
```

### Options
- `-h HOST` - Database host (default: localhost)
- `-p PORT` - Database port (default: 5432)
- `-d DATABASE` - Database name (default: test_index_pilot)
- `-u USER` - Database user (default: postgres)
- `-w PASSWORD` - Database password
- `-i` - Install only, don't run tests
- `-s` - Skip installation, run tests only

### Environment Variables
You can also use environment variables:
```bash
export DB_HOST=myhost
export DB_PORT=5432
export DB_NAME=testdb
export DB_USER=myuser
export DB_PASS=mypass
./run_tests.sh
```

## CI/CD Integration

### GitLab CI
The project includes `.gitlab-ci.yml` which:
- Tests against PostgreSQL versions 13, 14, 15, 16, 17, and 18
- Runs security scans for hardcoded passwords and SQL injection
- Tests non-superuser mode (simulating managed services)
- Generates JUnit reports for test visualization

### Running in Docker
```bash
# Test with PostgreSQL 16
docker run --rm -d --name pg16 -e POSTGRES_PASSWORD=postgres postgres:16-alpine

# Run tests
docker exec pg16 bash -c "cd /path/to/tests && ./run_tests.sh"

# Cleanup
docker stop pg16
```

## Test Development

### Adding New Tests
1. Create a new SQL file with numeric prefix (e.g., `04_performance.sql`)
2. Use `\set ON_ERROR_STOP on` to fail fast
3. Use `DO $$ ... END $$;` blocks for test logic
4. Use `RAISE NOTICE 'PASS: ...'` for success
5. Use `RAISE EXCEPTION 'FAIL: ...'` for failures

### Test Template
```sql
-- Test XX: Description
\set ON_ERROR_STOP on
\set QUIET on

\echo '======================================'
\echo 'TEST XX: Test Name'
\echo '======================================'

DO $$
BEGIN
    -- Test logic here
    IF condition THEN
        RAISE NOTICE 'PASS: Test description';
    ELSE
        RAISE EXCEPTION 'FAIL: Test description';
    END IF;
END $$;

\echo 'TEST XX: PASSED'
\echo ''
```

## Test Coverage

Current test coverage includes:
- ✅ Schema installation verification
- ✅ Function existence checks
- ✅ Permission validation
- ✅ Index detection and monitoring
- ✅ Bloat estimation accuracy
- ✅ Baseline establishment
- ✅ Reindex threshold detection
- ✅ SQL injection protection
- ✅ Non-superuser mode compatibility
- ✅ FDW/dblink security

## Troubleshooting

### Common Issues

1. **Connection refused**
   - Check PostgreSQL is running
   - Verify host/port settings
   - Check pg_hba.conf allows connections

2. **Permission denied**
   - Ensure user has CREATE DATABASE privilege
   - For non-superuser tests, check GRANT statements

3. **Test failures**
   - Check `test-results.xml` for detailed error messages
   - Review `/tmp/test_output.log` for full output
   - Ensure PostgreSQL version is 13+

### Debug Mode
```bash
# Run with verbose output
PGPASSWORD=mypass psql -h localhost -U postgres -d test_index_pilot -f 01_basic_installation.sql
```

## Contributing

When adding new features to pg_index_pilot:
1. Add corresponding tests
2. Ensure tests pass locally
3. Verify CI pipeline passes
4. Update this README if test structure changes