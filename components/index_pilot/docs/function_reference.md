## Function reference

### Table of contents

- [Core Functions](#core-functions)
- [Bloat Analysis](#bloat-analysis)
- [Non-Superuser Mode Functions](#non-superuser-mode-functions)
- [Configuration](#configuration)
- [FDW and connection setup](#fdw-and-connection-setup)
- [Maintenance helpers and meta](#maintenance-helpers-and-meta)

### Core Functions

#### `index_pilot.do_reindex()`
Manually triggers reindexing for specific objects.
```sql
procedure index_pilot.do_reindex(
    _datname name, 
    _schemaname name, 
    _relname name, 
    _indexrelname name, 
    _force boolean default false  -- Force reindex regardless of bloat
)
```

#### `index_pilot.periodic()`
Main procedure for automated bloat detection and reindexing.
```sql
procedure index_pilot.periodic(
    real_run boolean default false,  -- Execute actual reindexing
    force boolean default false      -- Force all eligible indexes
)
```

### Bloat Analysis

#### `index_pilot.get_index_bloat_estimates()`
Returns current bloat estimates for all indexes in a database.
```sql
function index_pilot.get_index_bloat_estimates(_datname name) 
returns table(
    datname name, 
    schemaname name, 
    relname name, 
    indexrelname name, 
    indexsize bigint, 
    estimated_bloat real
)
```

Notes:
- `estimated_bloat` is computed as `indexsize / (best_ratio * estimated_tuples)` using cached state in `index_pilot.index_latest_state`.
- Immediately after baseline initialization (see `do_force_populate_index_stats`) `estimated_bloat` will be ~1.0 by definition; it grows as indexes bloat further.

### Non-Superuser Mode Functions

#### `index_pilot.check_permissions()`
Verifies permissions for non-superuser mode operation.
```sql
function index_pilot.check_permissions() 
returns table(
    permission text, 
    status boolean
)
```

### Configuration

#### `index_pilot.get_setting()`
Reads effective setting with precedence (index → table → schema → db → global).
```sql
function index_pilot.get_setting(
  _datname text,
  _schemaname text,
  _relname text,
  _indexrelname text,
  _key text
) returns text
```

#### `index_pilot.set_or_replace_setting()`
Sets/overrides a setting at a specific scope.
```sql
function index_pilot.set_or_replace_setting(
  _datname text,
  _schemaname text,
  _relname text,
  _indexrelname text,
  _key text,
  _value text,
  _comment text
) returns void
```

### FDW and connection setup

#### `index_pilot.setup_fdw_self_connection()`
Creates or ensures the foreign server for self-connection.
```sql
function index_pilot.setup_fdw_self_connection(
  _host text default 'localhost',
  _port integer default null,
  _dbname text default null
) returns text
```

#### `index_pilot.setup_user_mapping()`
Creates or updates user mapping with password.
```sql
function index_pilot.setup_user_mapping(
  _username text default null,
  _password text default null
) returns text
```

#### `index_pilot.setup_connection()`
Configures secure connection via postgres_fdw user mapping.
```sql
function index_pilot.setup_connection(
  _host text,
  _port integer default 5432,
  _username text default 'index_pilot',
  _password text default null
) returns text
```

#### `index_pilot.setup_fdw_complete()`
One-shot helper: server, user mapping, connection test.
```sql
function index_pilot.setup_fdw_complete(
  _password text,
  _host text default 'localhost',
  _port integer default null,
  _username text default null
) returns table(step text, result text)
```

#### `index_pilot.check_fdw_security_status()`
Checks security-related FDW components.
```sql
function index_pilot.check_fdw_security_status()
returns table(component text, status text, details text)
```

### Maintenance helpers and meta

#### `index_pilot.do_force_populate_index_stats()`
Initializes baseline using current sizes/tuples without reindex.
```sql
function index_pilot.do_force_populate_index_stats(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name
) returns void
```
Examples:
```sql
-- Initialize baseline for a target DB
select index_pilot.do_force_populate_index_stats('your_database', null, null, null);

-- Initialize baseline for one schema
select index_pilot.do_force_populate_index_stats('your_database', 'bot', null, null);
```

When to use:
- After initial registration, to establish best_ratio without reindexing.
- After major data reshaping, to reset baseline for specific schemas/tables.

#### `index_pilot.check_environment()`
Aggregated environment and installation self-check (PostgreSQL version, extensions, schema/tables, core routines presence).
```sql
function index_pilot.check_environment()
returns table(
  component text,
  is_ok boolean,
  details text
)
```

#### `index_pilot.check_update_structure_version()`
Migrates internal tables to the required version if needed.
```sql
function index_pilot.check_update_structure_version() returns void
```

#### `index_pilot.version()`
Returns current code version.
```sql
function index_pilot.version() returns text
```

