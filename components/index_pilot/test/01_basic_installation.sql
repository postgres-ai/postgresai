-- Test 01: Basic Installation Verification
-- Exit on first error for CI
\set ON_ERROR_STOP on
\set QUIET on

-- Test output formatting
\echo '======================================'
\echo 'TEST 01: Basic Installation'
\echo '======================================'

-- 1. Verify schema exists
do $$
begin
  if not exists (select 1 from pg_namespace where nspname = 'index_pilot') then
    raise exception 'FAIL: index_pilot schema not found';
  end if;
  raise notice 'PASS: Schema index_pilot exists';
end $$;

-- 2. Verify version function
do $$
declare
  _version text;
begin
  select index_pilot.version() into _version;
  if _version is null or _version = '' then
    raise exception 'FAIL: Version function returned empty';
  end if;
  raise notice 'PASS: Version function works (%)', _version;
end $$;

-- 3. Verify required tables exist
do $$
declare
  _table_count integer;
  _expected_tables text[] := ARRAY[
    'config',
    'index_latest_state', 
    'reindex_history',
    'current_processed_index',
    'tables_version'
  ];
  _table text;
begin
  foreach _table in ARRAY _expected_tables loop
    if not exists (
      select 1 from information_schema.tables 
      where table_schema = 'index_pilot' 
      and table_name = _table
    ) then
      raise exception 'FAIL: Required table index_pilot.% not found', _table;
    end if;
  end loop;
  raise notice 'PASS: All required tables exist';
end $$;

-- 4. Verify core functions exist
do $$
declare
  _functions text[] := ARRAY[
    'periodic',
    'do_reindex',
    'get_index_bloat_estimates',
    'check_permissions'
  ];
  _func text;
begin
  foreach _func in ARRAY _functions loop
    if not exists (
      select 1 from pg_proc p
      join pg_namespace n on p.pronamespace = n.oid
      where n.nspname = 'index_pilot' 
      and p.proname = _func
    ) then
      raise exception 'FAIL: Required function index_pilot.% not found', _func;
    end if;
  end loop;
  raise notice 'PASS: All core functions exist';
end $$;

-- 5. Verify permissions check runs
do $$
declare
  _count integer;
begin
  select count(*) into _count from index_pilot.check_permissions();
  if _count < 1 then
    raise exception 'FAIL: check_permissions returned no results';
  end if;
  raise notice 'PASS: Permissions check returns % items', _count;
end $$;

-- 6. Verify default configuration
do $$
declare
  _config_count integer;
begin
  select count(*) into _config_count from index_pilot.config;
  if _config_count < 4 then
    raise exception 'FAIL: Missing default configuration (found % entries)', _config_count;
  end if;
  raise notice 'PASS: Default configuration present (% entries)', _config_count;
end $$;

\echo 'TEST 01: PASSED'
\echo ''