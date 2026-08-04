# Health check

Run health checks on a Postgres database.

$ARGUMENTS should be a connection string like `postgresql://user@host:5432/db`

If password is needed, use one of these methods (in order of preference):
1. `.pgpass` file (~/.pgpass) - most secure, not visible in process listings
2. `PGPASSFILE` environment variable pointing to a password file
3. `PGPASSWORD` environment variable (visible in process listings on some systems)

```bash
postgresai checkup $ARGUMENTS
```

Present the report output to the user. Add a brief summary at the end highlighting the most critical issues and recommended next steps.
