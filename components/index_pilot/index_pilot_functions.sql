begin;

-- Turn off useless (in this particular case) NOTICE noise
set client_min_messages to warning;

/*
 * Get current version of pg_index_pilot
 * Returns version string for compatibility checks and diagnostics
 */
create function index_pilot.version() returns text as $body$
  select '0.1.beta1';
$body$ language sql immutable;


/*
 * Check if PostgreSQL version has critical REINDEX CONCURRENTLY bugs fixed
 * Returns true for PG versions safe for concurrent reindexing (12.10+, 13.6+, 14.4+)
 */
create function index_pilot._check_pg_version_bugfixed() returns boolean as
$body$
  /* Fixes not covered here:
       - 17.6, 16.10: fix in BRIN (rebuild required) -- we don't support BRIN yet
       - 16.2 (and backpatches): rare GIN corruption fixed in f76b975d5 (TODO: evaluate if it's worth including)
       - 15.8 (and backpatches): a fix related to SP-GiST (TODO: evaluate if it's worth including)
       - pre-PG12 fixes – we now have PG13+ as a requirement
  */
  select (
    (
      /* PG12.10 fix: Enforce standard locking protocol for TOAST table updates,
         to prevent problems with REINDEX CONCURRENTLY
         https://gitlab.com/postgres/postgres/-/commit/5ed74d874
        ("Fix corruption of toast indexes with REINDEX CONCURRENTLY") */
      current_setting('server_version_num')::integer >= 120010
      and current_setting('server_version_num')::integer < 130000
    ) or (
      /* PG13.6 fix: Enforce standard locking protocol for TOAST table updates,
         to prevent problems with REINDEX CONCURRENTLY (Michael Paquier)
         https://gitlab.com/postgres/postgres/-/commit/9acea52ea
         ("Fix corruption of toast indexes with REINDEX CONCURRENTLY") */
      current_setting('server_version_num')::integer >= 130006
      and current_setting('server_version_num')::integer < 140000
    ) or (
      /* PG14.4 fix: Prevent possible corruption of indexes created or rebuilt
         with the CONCURRENTLY option (Álvaro Herrera)
         https://gitlab.com/postgres/postgres/-/commit/042b584c7
         ("Revert changes to CONCURRENTLY that "sped up" Xmin advance") */
      current_setting('server_version_num')::integer >= 140004
    )
  );
$body$
language sql;


/*
 * Check if PostgreSQL 14 version has critical REINDEX CONCURRENTLY bug fixed
 * Returns false for dangerous PG 14.0-14.3 versions (bug #17485)
 */
create function index_pilot._check_pg14_version_bugfixed() returns boolean as
$body$
  select
    current_setting('server_version_num')::integer < 140000
    or current_setting('server_version_num')::integer >= 140004;
$body$
language sql;

/*
 * Validate PostgreSQL version safety and raise appropriate warnings/errors
 * Raises EXCEPTION for PG 14.0-14.3, WARNING for other affected versions
 */
create function index_pilot._validate_pg_version() returns void as
$body$
begin
  if not index_pilot._check_pg14_version_bugfixed() then
    raise exception using
      message = format(
        'The database version %s is affected by PostgreSQL BUG #17485 which makes using pg_index_pilot unsafe, please update to latest minor release.',
        current_setting('server_version')
      ),
      detail = 'See https://www.postgresql.org/message-id/202205251144.6t4urostzc3s@alvherre.pgsql';
  end if;

  if not index_pilot._check_pg_version_bugfixed() then
    raise warning using
      message = format(
        'The database version %s is affected by PostgreSQL bugs which make using pg_index_pilot potentially unsafe, please update to latest minor release.',
        current_setting('server_version')
      ),
      detail = 
        'See https://www.postgresql.org/message-id/E1mumI4-0001Zp-PB@gemulon.postgresql.org '
        'and https://www.postgresql.org/message-id/E1n8C7O-00066j-Q5@gemulon.postgresql.org';
  end if;
end;
$body$
language plpgsql;


/*
 * Installation-time safety validation
 * Blocks unsafe deployments: enforces PG13+ requirement and detects known bugs
 */
do $$
begin
  if current_setting('server_version_num')::int < 130000 then
    raise exception 'pg_index_pilot requires PostgreSQL 13 or higher; version in use: %.',
    current_setting('server_version');
  end if;

  -- Validate PostgreSQL version safety
  perform index_pilot._validate_pg_version();
end;
$$;

/*
 * Comprehensive environment validation for pg_index_pilot setup
 * Complete preflight check: version, extensions, schema, permissions, FDW connectivity
 */
create function index_pilot.check_environment()
returns table(
  component text,
  is_ok boolean,
  details text
) as
$body$
declare
  _missing_permissions_count integer;
  _res record;
  _fdw_self_ok boolean := false;
begin
  -- PostgreSQL version
  return query select 
    'PostgreSQL version (>=13)'::text,
    (current_setting('server_version_num')::int >= 130000),
    current_setting('server_version');

  -- Known bugfix statuses
  return query select 
    'Known bugs fixed'::text,
    index_pilot._check_pg_version_bugfixed(),
    case when index_pilot._check_pg_version_bugfixed() then 'Minor version is safe' else 'Upgrade to latest minor recommended' end;

  return query select 
    'PG14 bug #17485 fixed'::text,
    index_pilot._check_pg14_version_bugfixed(),
    case when index_pilot._check_pg14_version_bugfixed() then 'Not affected' else 'Update to 14.4 or newer' end;

  -- Extensions
  return query select 
    'Extension: dblink'::text,
    exists (select 1 from pg_extension where extname = 'dblink'),
    'Run: create extension dblink;';

  return query select 
    'Extension: postgres_fdw'::text,
    exists (select 1 from pg_extension where extname = 'postgres_fdw'),
    'Run: create extension postgres_fdw;';

  -- Schema presence
  return query select 
    'Schema: index_pilot'::text,
    exists (select 1 from pg_namespace where nspname = 'index_pilot'),
    '';

  -- Required tables
  for _res in
    select unnest(array[
      'config',
      'index_latest_state',
      'reindex_history',
      'current_processed_index',
      'tables_version'
    ]) as tbl
  loop
    return query select 
      format('Table: %I.%I', 'index_pilot', _res.tbl),
      exists (
        select
        from information_schema.tables 
        where table_schema = 'index_pilot' and table_name = _res.tbl
      ),
      '';
  end loop;

  -- Core routines presence
  for _res in
    select unnest(array[
      'version',
      'periodic',
      'do_reindex',
      'get_index_bloat_estimates',
      'check_permissions'
    ]) as func
  loop
    return query select 
      format('Function: %I.%I(..)', 'index_pilot', _res.func),
      exists (
        select
        from pg_proc as p
        join pg_namespace as n on p.pronamespace = n.oid
        where n.nspname = 'index_pilot' and p.proname = _res.func
      ),
      '';
  end loop;

  -- Permissions summary
  select count(*) into _missing_permissions_count 
  from index_pilot.check_permissions() as p 
  where p.status = false;

  return query select 
    'Permissions summary'::text,
    (_missing_permissions_count = 0),
    format('Missing: %s', _missing_permissions_count);

  -- FDW security status (detailed lines)
  for _res in select * from index_pilot.check_fdw_security_status() loop
    return query 
    select 
      format('FDW: %s', _res.component)::text,
      (lower(_res.status) in ('ok','installed','granted','exists','secure','configured')),
      _res.details::text;
  end loop;

  -- Control DB architecture checks
  return query select 
    'Control DB: table index_pilot.target_databases'::text,
    exists (
      select 1 from information_schema.tables where table_schema = 'index_pilot' and table_name = 'target_databases'
    ),
    'Required for multi-database control mode';

  if exists (
    select 1 
    from information_schema.tables
    where 
      table_schema = 'index_pilot' 
      and table_name = 'target_databases'
  ) then
    return query 
    select 
      'Control DB: registered targets'::text,
      ((select count(*) from index_pilot.target_databases where enabled) > 0),
      (select string_agg(database_name, ', ') from index_pilot.target_databases);

    return query 
    select 
      'Safety: current DB not listed as target'::text,
      not exists (
        select 1 
        from index_pilot.target_databases 
        where database_name = current_database()
        ),
      'Do not register the control database as a target';
  end if;

  -- Best-effort FDW connectivity test (use any enabled target's fdw_server_name)
  begin
    perform dblink_connect(
      'env_test',
      coalesce(
        (select fdw_server_name from index_pilot.target_databases where enabled limit 1),
        null
      )
    );
    perform dblink_disconnect('env_test');
    _fdw_self_ok := true;
  exception when others then
    _fdw_self_ok := false;
  end;

  return query select 
    'FDW self-connection test'::text,
    _fdw_self_ok,
    case when _fdw_self_ok then 'Connected via user mapping' else 'Ensure at least one enabled target with valid user mapping' end;

  return;
end;
$body$
language plpgsql;


-- Install dblink extension for remote database operations
-- Note: postgres_fdw is NOT installed here - it should already be configured
-- with proper servers and user mappings to register target databases
create extension if not exists dblink;

/*
 * Validate table structure version meets minimum requirements
 * Throws exception if schema is outdated and needs upgrade
 */
create function index_pilot._check_structure_version() returns void as
$body$
declare
  _tables_version integer;
  _required_version integer := 1;
begin
  select version into strict _tables_version from index_pilot.tables_version;

  if (_tables_version < _required_version) then
    raise exception using
      message = format(
        'Current tables version %s is less than minimally required %s for %s code version.',
        _tables_version,
        _required_version,
        index_pilot.version()
      ),
      hint = 'Update tables structure.';
  end if;
end;
$body$
language plpgsql;


/*
 * Automatically upgrade table structure to required version
 * Performs incremental schema migrations using version-specific upgrade functions
 */
create function index_pilot.check_update_structure_version() returns void as
$body$
declare
   _tables_version integer;
   _required_version integer := 1;
begin
  select version into strict _tables_version from index_pilot.tables_version;

  while (_tables_version < _required_version) loop
    execute format(
      'select index_pilot._structure_version_%s_%s()',
      _tables_version,
      _tables_version + 1
    );

    _tables_version := _tables_version + 1;
  end loop;

  return;
end;
$body$
language plpgsql;


-- FDW and connection management functions have been moved to index_pilot_fdw.sql


/*
 * Get reindexable indexes from remote database
 * Filters for safe indexes: excludes system schemas, BRIN, exclusion constraints
 */
create function index_pilot._remote_get_indexes_indexrelid(_datname name)
returns table(
  datname name, 
  schemaname name, 
  relname name, 
  indexrelname name, 
  indexrelid oid
) as
$body$
declare
  _use_toast_tables text;
begin
  if index_pilot._check_pg_version_bugfixed() then 
    _use_toast_tables := 'True';
  else 
    _use_toast_tables := 'False';
  end if;
    
  -- Secure FDW connection for querying indexes
  perform index_pilot._connect_securely(_datname);
    
  return query select
    _datname, 
    _res.schemaname,
    _res.relname,
    _res.indexrelname,
    _res.indexrelid
  from
    dblink(
      _datname,
      format(
        $sql$
          select
            n.nspname as schemaname,
            c.relname,
            i.relname as indexrelname,
            x.indexrelid
          from pg_index as x
          join pg_catalog.pg_class as c on c.oid = x.indrelid
          join pg_catalog.pg_class as i on i.oid = x.indexrelid
          join pg_catalog.pg_namespace as n on n.oid = c.relnamespace
          join pg_catalog.pg_am as a on a.oid = i.relam
          -- TOAST indexes info
          left join pg_catalog.pg_class as c1 on c1.reltoastrelid = c.oid and n.nspname = 'pg_toast'
          left join pg_catalog.pg_namespace as n1 on c1.relnamespace = n1.oid
          where 
            true
            -- limit reindex for indexes on tables/mviews/TOAST
            -- and c.relkind = any (array['r'::"char", 't'::"char", 'm'::"char"])
            -- limit reindex for indexes on tables/mviews (skip TOAST until bugfix of BUG #17268)
            and ((c.relkind = any (array['r'::"char", 'm'::"char"])) or ((c.relkind = 't'::"char") and %s))
            -- ignore exclusion constraints
            and not exists (select from pg_constraint where pg_constraint.conindid = i.oid and pg_constraint.contype = 'x')
            -- ignore indexes for system tables and index_pilot own tables
            and n.nspname not in ('pg_catalog', 'information_schema', 'index_pilot')
            -- ignore indexes on TOAST tables of system tables and index_pilot own tables
            and (n1.nspname is null or n1.nspname not in ('pg_catalog', 'information_schema', 'index_pilot'))
            -- skip BRIN indexes... please see BUG #17205 https://www.postgresql.org/message-id/flat/17205-42b1d8f131f0cf97%%40postgresql.org
            and a.amname not in ('brin') and x.indislive
            -- skip indexes on temp relations
            and c.relpersistence <> 't' -- t = temporary table/sequence
            -- debug only
            -- order by 1, 2, 3
        $sql$, 
        _use_toast_tables
      )
    )
    as _res(
      schemaname name,
      relname name,
      indexrelname name,
      indexrelid oid
    );
end;
$body$
language plpgsql;


/*
 * Convert shell-style wildcard patterns to PostgreSQL regex format
 * Transforms * to .* and ? to . with anchors for exact matching
 */
create function index_pilot._pattern_convert(
  _var text
) returns text as
$body$
  select '^(' || replace(replace(_var, '*', '.*'), '?', '.') || ')$';
$body$
language sql strict immutable;


/*
 * Get configuration setting value using hierarchical priority lookup
 * Searches: index → table → schema → database → global priority order
 */
create function index_pilot.get_setting(
  _datname text,
  _schemaname text,
  _relname text,
  _indexrelname text,
  _key text
) returns text as
$body$
declare
  _value text;
begin
  perform index_pilot._check_structure_version();

  -- raise notice 'debug: |%|%|%|%|', _datname, _schemaname, _relname, _indexrelname;

  select _t.value into _value from (
    -- per index setting
    select 
      1 as priority,
      value from index_pilot.config 
    where
      _key = config.key
	    and (_datname operator(pg_catalog.~) index_pilot._pattern_convert(config.datname))
	    and (_schemaname operator(pg_catalog.~) index_pilot._pattern_convert(config.schemaname))
	    and (_relname operator(pg_catalog.~) index_pilot._pattern_convert(config.relname))
	    and (_indexrelname operator(pg_catalog.~) index_pilot._pattern_convert(config.indexrelname))
	    and config.indexrelname is not null
	    and true
    union all
    -- per table setting
    select 
      2 as priority,
      value from index_pilot.config 
    where
      _key = config.key
      and (_datname operator(pg_catalog.~) index_pilot._pattern_convert(config.datname))
      and (_schemaname operator(pg_catalog.~) index_pilot._pattern_convert(config.schemaname))
      and (_relname operator(pg_catalog.~) index_pilot._pattern_convert(config.relname))
      and config.relname is not null
      and config.indexrelname is null
    union all
    -- per schema setting
    select 
      3 as priority,
      value from index_pilot.config 
    where
      _key = config.key
      and (_datname operator(pg_catalog.~) index_pilot._pattern_convert(config.datname))
      and (_schemaname operator(pg_catalog.~) index_pilot._pattern_convert(config.schemaname))
      and config.schemaname is not null
      and config.relname is null
    union all
    -- per database setting
    select 
      4 as priority,
      value from index_pilot.config 
    where
      _key = config.key
      and (_datname      operator(pg_catalog.~) index_pilot._pattern_convert(config.datname))
      and config.datname is not null
      and config.schemaname is null
    union all
    -- global setting
    select 
      5 as priority,
      value from index_pilot.config 
    where
      _key = config.key
      and config.datname is null
    ) as _t
    where value is not null
    order by priority
    limit 1;
  
  return _value;
end;
$body$
language plpgsql stable;


/*
 * Set or update configuration setting at appropriate hierarchy level
 * Auto-detects specificity level based on null parameters, handles conflicts
 */
create function index_pilot.set_or_replace_setting(
  _datname text,
  _schemaname text,
  _relname text,
  _indexrelname text,
  _key text,
  _value text,
  _comment text
) returns void as
$body$
begin
    perform index_pilot._check_structure_version();

    if _datname is null then
      insert into index_pilot.config (datname, schemaname, relname, indexrelname, key, value, comment)
      values (_datname, _schemaname, _relname, _indexrelname, _key, _value, _comment)
      on conflict (key) 
      where datname is null 
      do update set 
        value = excluded.value, 
        comment = excluded.comment;
    elsif _schemaname is null then
      insert into index_pilot.config (datname, schemaname, relname, indexrelname, key, value, comment)
      values (_datname, _schemaname, _relname, _indexrelname, _key, _value, _comment)
      on conflict (key, datname) 
      where schemaname is null 
      do update set 
        value = excluded.value, 
        comment = excluded.comment;
    elsif _relname is null    then
      insert into index_pilot.config (datname, schemaname, relname, indexrelname, key, value, comment)
      values (_datname, _schemaname, _relname, _indexrelname, _key, _value, _comment)
      on conflict (key, datname, schemaname)
      where relname is null 
      do update set 
        value = excluded.value, 
        comment = excluded.comment;
    elsif _indexrelname is null then
      insert into index_pilot.config (datname, schemaname, relname, indexrelname, key, value, comment)
      values (_datname, _schemaname, _relname, _indexrelname, _key, _value, _comment)
      on conflict (key, datname, schemaname, relname) 
      where indexrelname is null 
      do update set 
        value = excluded.value, 
        comment = excluded.comment;
    else
      insert into index_pilot.config (datname, schemaname, relname, indexrelname, key, value, comment)
      values (_datname, _schemaname, _relname, _indexrelname, _key, _value, _comment)
      on conflict (key, datname, schemaname, relname, indexrelname) 
      do update set 
        value = excluded.value, 
        comment = excluded.comment;
    end if;
    return;
end;
$body$
language plpgsql;


/*
 * Get detailed index information from remote database with filtering
 * Returns comprehensive metrics, clamps zero tuples, supports wildcard filtering
 */
create function index_pilot._remote_get_indexes_info(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name
) returns table(
  datid oid,
  indexrelid oid,
  datname name,
  schemaname name,
  relname name,
  indexrelname name,
  indisvalid boolean,
  indexsize bigint,
  estimated_tuples bigint
) as
$body$
declare
  _use_toast_tables text;
begin
  if index_pilot._check_pg_version_bugfixed() then 
    _use_toast_tables := 'True';
  else 
    _use_toast_tables := 'False';
  end if;
    
  -- Secure FDW connection for querying index info
  perform index_pilot._connect_securely(_datname);

  return query select
    d.oid as datid,
    _res.indexrelid,
    _datname,
    _res.schemaname,
    _res.relname,
    _res.indexrelname,
    _res.indisvalid,
    _res.indexsize,
    -- Clamp zero tuples to 1 to avoid division by zero and infinite bloat estimates in calculations
    greatest(1, indexreltuples)
    -- do not apply relsize/relpage correction; that approach was found to be unnecessarily complex and unreliable
    -- greatest(1, (case when relpages=0 then indexreltuples else relsize*indexreltuples/(relpages*current_setting('block_size')) end as estimated_tuples))
  from
    dblink(_datname,
      format(
        $sql$
          select
            x.indexrelid,
            n.nspname as schemaname,
            c.relname,
            i.relname as indexrelname,
            x.indisvalid,
            i.reltuples::bigint as indexreltuples,
            pg_catalog.pg_relation_size(i.oid)::bigint as indexsize
            -- debug only
            -- , pg_namespace.nspname
            -- , c3.relname,
            -- , am.amname
          from pg_index as x
          join pg_catalog.pg_class as c           on c.oid = x.indrelid
          join pg_catalog.pg_class as i           on i.oid = x.indexrelid
          join pg_catalog.pg_namespace as n       on n.oid = c.relnamespace
          join pg_catalog.pg_am as a              on a.oid = i.relam
          -- TOAST indexes info
          left join pg_catalog.pg_class as c1     on c1.reltoastrelid = c.oid and n.nspname = 'pg_toast'
          left join pg_catalog.pg_namespace as n1 on c1.relnamespace = n1.oid

          where true
          -- limit reindex for indexes on tables/mviews/TOAST
          -- and c.relkind = any (array['r'::"char", 't'::"char", 'm'::"char"])
          -- limit reindex for indexes on tables/mviews (skip TOAST until bugfix of BUG #17268)
          and ((c.relkind = any (array['r'::"char", 'm'::"char"])) or ((c.relkind = 't'::"char") and %s))
          -- ignore exclusion constraints
          and not exists (select from pg_constraint where pg_constraint.conindid = i.oid and pg_constraint.contype = 'x')
          -- ignore indexes for system tables and index_pilot own tables
          and n.nspname not in ('pg_catalog', 'information_schema', 'index_pilot')
          -- ignore indexes on TOAST tables of system tables and index_pilot own tables
          and (n1.nspname is null or n1.nspname not in ('pg_catalog', 'information_schema', 'index_pilot'))
          -- skip BRIN indexes... please see BUG #17205 https://www.postgresql.org/message-id/flat/17205-42b1d8f131f0cf97%%40postgresql.org
          and a.amname not in ('brin') and x.indislive
          -- skip indexes on temp relations
          and c.relpersistence <> 't' -- t = temporary table/sequence
          -- debug only
          -- order by 1,2,3
        $sql$,
        _use_toast_tables
      )
    )
    as _res(
      indexrelid oid,
      schemaname name,
      relname name,
      indexrelname name,
      indisvalid boolean,
      indexreltuples bigint,
      indexsize bigint
    ),
    pg_database as d
    where
      d.datname = _datname
      and (_schemaname is null or _res.schemaname = _schemaname)
      and (_relname is null or _res.relname = _relname)
      and (_indexrelname is null or _res.indexrelname = _indexrelname);
end;
$body$
language plpgsql;


/*
 * Record and maintain index information in the tracking table
 * Updates metadata, manages bloat ratios, cleans removed indexes, supports filtering
 */
create function index_pilot._record_indexes_info(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name,
  _force_populate boolean default false
) returns void as
$body$
declare
  index_info record;
  _connection_created boolean := false;
begin
  -- Establish dblink connection for managed services mode with cleanup guarantee
  if dblink_get_connections() is null or not (_datname = any(dblink_get_connections())) then
    perform index_pilot._dblink_connect_if_not(_datname);
    _connection_created := true;
  end if;
  
  -- merge index data fetched from the database and index_latest_state
  -- now keep info about all potentially interesting indexes (even small ones)
  -- we can do it now because we keep exactly one entry in index_latest_state per index (without history)
  with _actual_indexes as (
    select datid, indexrelid, datname, schemaname, relname, indexrelname, indisvalid, indexsize, estimated_tuples
    from index_pilot._remote_get_indexes_info(_datname, _schemaname, _relname, _indexrelname)
  ),
  _old_indexes as (
    delete from index_pilot.index_latest_state as i
    where not exists (
      select from _actual_indexes
      where
        i.datid = _actual_indexes.datid
	      and i.indexrelid = _actual_indexes.indexrelid
    )
    and i.datname = _datname
    and (_schemaname is null or i.schemaname = _schemaname)
    and (_relname is null or i.relname = _relname)
    and (_indexrelname is null or i.indexrelname = _indexrelname)
  )
  -- todo: do something with ugly code duplication in index_pilot._reindex_index and index_pilot._record_indexes_info
  insert into index_pilot.index_latest_state as i
  (datid, datname, schemaname, relname, indexrelid, indexrelname, indexsize, indisvalid, estimated_tuples, best_ratio)
  select 
    datid, 
    datname, 
    schemaname, 
    relname, 
    indexrelid, 
    indexrelname, 
    indexsize,
    indisvalid, 
    estimated_tuples,
    case
      when (indexsize > pg_size_bytes(index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'minimum_reliable_index_size'))) then
        -- initialize baseline from the current ratio on first sighting (insert),
        -- including after REINDEX/OID change; _force_populate is not needed on insert
        indexsize::real / estimated_tuples::real
      else
        -- too small for reliable baseline
          null
      end
    as best_ratio
  from _actual_indexes
  on conflict (datid, indexrelid)
  do update set
    mtime = now(),
    datname = excluded.datname,
    schemaname = excluded.schemaname,
    relname = excluded.relname,
    indexrelname = excluded.indexrelname,
    indisvalid = excluded.indisvalid,
    indexsize = excluded.indexsize,
    estimated_tuples = excluded.estimated_tuples,
    best_ratio =
      case
        -- _force_populate=true set (or write) best ratio to current ratio (except the case when index too small to be reliably estimated)
        when (_force_populate and excluded.indexsize > pg_size_bytes(index_pilot.get_setting(excluded.datname, excluded.schemaname, excluded.relname, excluded.indexrelname, 'minimum_reliable_index_size')))
          then excluded.indexsize::real / excluded.estimated_tuples::real
        -- if index is too small, keep previous
        when (excluded.indexsize < pg_size_bytes(index_pilot.get_setting(excluded.datname, excluded.schemaname, excluded.relname, excluded.indexrelname, 'minimum_reliable_index_size')))
          then i.best_ratio
        -- fill only if missing
        when (i.best_ratio is null)
          then excluded.indexsize::real / excluded.estimated_tuples::real
        -- otherwise keep baseline unchanged (non-increasing, unaffected by reltuples drift)
        else least(i.best_ratio, excluded.indexsize::real / excluded.estimated_tuples::real)
      end;

  -- tell about not valid indexes
  for index_info in
    select indexrelname, relname, schemaname, datname from index_pilot.index_latest_state
      where not indisvalid
      and datname = _datname
      and (_schemaname is null or schemaname = _schemaname)
      and (_relname is null or relname = _relname)
      and (_indexrelname is null or indexrelname = _indexrelname)
    loop
      raise warning 'Not valid index % on %.% found in %.',
      index_info.indexrelname, index_info.schemaname, index_info.relname, index_info.datname;
    end loop;

exception when others then
  -- Guaranteed connection cleanup on any exception
  if _connection_created and dblink_get_connections() is not null 
     and _datname = any(dblink_get_connections()) then
    perform dblink_disconnect(_datname);
  end if;
  raise; -- Re-raise the original exception
end;
$body$
language plpgsql;


/*
 * Clean up old and stale records from tracking tables
 * Removes old history records and stale database state records based on retention settings
 */
create function index_pilot._cleanup_old_records() returns void as
$body$
begin
  -- TODO replace with fast distinct implementation
  with rels as materialized (
    select distinct datname, schemaname, relname, indexrelname
    from index_pilot.reindex_history
  ), age_limit as materialized (
    select *, now() - index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'reindex_history_retention_period')::interval as max_age
    from rels
  )
  delete from index_pilot.reindex_history
  using age_limit
  where
    reindex_history.datname = age_limit.datname
    and reindex_history.schemaname = age_limit.schemaname
    and reindex_history.relname = age_limit.relname
    and reindex_history.indexrelname = age_limit.indexrelname
    and reindex_history.entry_timestamp < age_limit.max_age;
    
  -- clean index_latest_state for not existing databases
  delete from index_pilot.index_latest_state
  where datid not in (
    select oid from pg_database
    where
      not datistemplate
      and datallowconn
      and index_pilot.get_setting(datname, null, null, null, 'skip')::boolean is distinct from true
  );

  return;
end;
$body$
language plpgsql;


/*
 * Calculate and return bloat estimates for all indexes in a database
 * Compares current size ratios with historical best, ordered by bloat level
 */
create function index_pilot.get_index_bloat_estimates(
  _datname name
) returns table(
  datname name, 
  schemaname name, 
  relname name, 
  indexrelname name, 
  indexsize bigint, 
  estimated_bloat real
) as
$body$
declare
  _datid oid;
begin
  perform index_pilot._check_structure_version();

  select oid from pg_database as d where d.datname = _datname into _datid;

  -- calculate estimated bloat by comparing the current size-to-tuple ratio with the best observed ratio
  return query select 
    _datname, 
    i.schemaname, 
    i.relname, 
    i.indexrelname, 
    i.indexsize,
    (i.indexsize::real / (i.best_ratio * estimated_tuples::real)) as estimated_bloat
  from index_pilot.index_latest_state as i
  where
    i.datid = _datid
    -- and indisvalid is true
  -- use NULLS FIRST so that indexes with null estimated bloat
  -- (which will be reindexed in the next scheduled run) appear first;
  -- order by most bloated indexes first
  order by estimated_bloat desc nulls first;
end;
$body$
language plpgsql strict;


/*
 * Perform concurrent reindexing of a specific index
 * Executes REINDEX INDEX CONCURRENTLY via secure dblink with logging and error handling
 */
create function index_pilot._reindex_index(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name
) returns void as
$body$
declare
  _indexsize_before bigint;
  _indexsize_after  bigint;
  _timestamp        timestamp;
  _reindex_duration interval;
  _analyze_duration interval :='0s';
  _estimated_tuples bigint;
  _indexrelid oid;
  _datid oid;
  _indisvalid boolean;
begin
  -- Ensure a dblink connection exists for the target database using FDW for secure password handling.
  -- The connection name is set to the database name (note: not unique per index).
  if dblink_get_connections() is null or not (_datname = any(dblink_get_connections())) then
    -- Establish a secure connection to the target database, handling control database mode if needed
    perform index_pilot._connect_securely(_datname);

    raise notice 'Created dblink connection: %', _datname;
  end if;

  -- raise notice 'working with %.%.% %', _datname, _schemaname, _relname, _indexrelname;

  -- Retrieve the current index size and confirm the index exists in the target database
  select indexsize, estimated_tuples into _indexsize_before, _estimated_tuples
  from index_pilot._remote_get_indexes_info(_datname, _schemaname, _relname, _indexrelname)
  where indisvalid;

  -- If the index doesn't exist anymore, exit the function
  if not found then
    return;
  end if;

  -- Perform the reindex operation using synchronous dblink
  _timestamp := pg_catalog.clock_timestamp ();
  
  -- Execute REINDEX INDEX CONCURRENTLY in a synchronous manner (similar to the original pg_index_watch)
  -- This operation blocks until the reindexing process is fully completed
  begin
    perform dblink(
      _datname,
      format('reindex index concurrently %I.%I', _schemaname, _indexrelname)
    );

    raise notice 'reindex index concurrently %.% completed successfully', _schemaname, _indexrelname;
  exception when others then
    raise notice 'reindex failed for %.%: %', _schemaname, _indexrelname, SQLERRM;
    -- Continue anyway, the index might have issues
    -- This allows the function to complete successfully even if the reindex fails
  end;
  
  -- Don't disconnect - keep connection for reuse (like original pg_index_watch)

  _reindex_duration := pg_catalog.clock_timestamp() - _timestamp;

  -- Retrieve the index size after reindexing
  select indexsize into _indexsize_after
  from index_pilot._remote_get_indexes_info(_datname, _schemaname, _relname, _indexrelname)
  where indisvalid;
  
  -- If the index doesn't exist anymore or is invalid, use the original size
  if _indexsize_after is null then
    _indexsize_after := _indexsize_before;
  end if;

  -- Log the completed reindex operation to the reindex_history table
  insert into index_pilot.reindex_history (
    datname, 
    schemaname,
    relname,
    indexrelname,
    indexsize_before,
    indexsize_after, 
    estimated_tuples, 
    reindex_duration, 
    analyze_duration,
    entry_timestamp
  ) values (
    _datname, 
    _schemaname, 
    _relname, 
    _indexrelname,
    _indexsize_before,
    _indexsize_after, 
    _estimated_tuples, 
    _reindex_duration, 
    '0'::interval,
    now()
  );
  
  raise notice 'reindex COMPLETED: %.% - size before: %, size after: %, duration: %', 
    _schemaname, _indexrelname, 
    pg_size_pretty(_indexsize_before), 
    pg_size_pretty(_indexsize_after),
    _reindex_duration;
end;
$body$
language plpgsql strict;


/*
 * Main reindexing orchestrator procedure
 * Identifies and reindexes bloated indexes based on thresholds and estimates
 */
create procedure index_pilot.do_reindex(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name,
  _force boolean default false
) as
$body$
declare
  _index record;
  _connection_created boolean := false;
begin
  perform index_pilot._check_structure_version();

  -- IMPORTANT: Do not run in the current database to prevent deadlocks
  if _datname = current_database() then
    raise exception using
      message = format(
        'Cannot REINDEX in current database %s - this causes deadlocks.',
        _datname
      ),
      hint = 'Use separate control database.';
  end if;

  -- Ensure dblink connection is established before starting any transaction with cleanup guarantee
  if dblink_get_connections() is null or not (_datname = any(dblink_get_connections())) then
    perform index_pilot._dblink_connect_if_not(_datname);
    _connection_created := true;
    commit; -- Commit after connection to minimize risk of locking issues
  end if;
  for _index in
    select datname, schemaname, relname, indexrelname, indexsize, estimated_bloat
    -- index_size_threshold check logic is now handled inside get_index_bloat_estimates
    -- The force option causes index_rebuild_scale_factor to be ignored, reindexing all eligible indexes
    -- Indexes that are too small (below index_size_threshold) or explicitly set to skip in the config are always ignored, even with force enabled
    -- todo: consider revisiting this logic in the future
    from index_pilot.get_index_bloat_estimates(_datname)
    where
      (_schemaname is null or schemaname = _schemaname)
      and (_relname is null or relname = _relname)
      and (_indexrelname is null or indexrelname = _indexrelname)
      and (_force or
        (
          -- ignore indexes that are too small to be relevant
          indexsize >= pg_size_bytes(index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'index_size_threshold'))
          -- ignore indexes explicitly marked to be skipped
          and index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'skip')::boolean is distinct from true
          -- placeholder for future configurability using get_setting
          and (
            estimated_bloat is null
            or estimated_bloat >= index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'index_rebuild_scale_factor')::float
          )
        )
      )
    loop
      -- Record what we're working on
      insert into index_pilot.current_processed_index(
        datname,
          schemaname,
          relname,
          indexrelname
      )
      values (
        _index.datname,
        _index.schemaname,
        _index.relname,
        _index.indexrelname
      );

      -- Record the start of the reindex operation with status 'in_progress'
      -- Use cached information from index_latest_state rather than querying the remote database
      insert into index_pilot.reindex_history (
        datname, schemaname, relname, indexrelname,
        indexsize_before, indexsize_after, estimated_tuples, 
        reindex_duration, analyze_duration, entry_timestamp, status
      ) 
      select 
        datname, 
        schemaname, 
        relname, 
        indexrelname,
        indexsize, 
        null, 
        estimated_tuples,  -- null until completion
        null, 
        null, 
        now(), 
        'in_progress'
      from (
        select distinct on (datid, indexrelid)
          datname, schemaname, relname, indexrelname, indexsize, estimated_tuples
        from index_pilot.index_latest_state
        where datname = _index.datname
          and schemaname = _index.schemaname
          and relname = _index.relname
          and indexrelname = _index.indexrelname
          and indisvalid
        order by datid, indexrelid, mtime desc
      ) latest;
      
      -- Commit to release all locks before starting synchronous reindex
      commit;
      
      -- Perform REINDEX synchronously for robust and predictable operation
      -- Synchronous execution enables immediate status updates and simplifies process management
      begin
        -- Run REINDEX INDEX CONCURRENTLY synchronously
        perform dblink_exec(
          _index.datname,
          format('reindex index concurrently %I.%I', _index.schemaname, _index.indexrelname)
        );
           
        raise notice 'REINDEX INDEX CONCURRENTLY completed for %.%', _index.schemaname, _index.indexrelname;
           
        -- Retrieve the index size after reindexing is complete
        declare
          _final_size bigint;
        begin
          select indexsize into _final_size
          from index_pilot._remote_get_indexes_info(_index.datname, _index.schemaname, _index.relname, _index.indexrelname)
          where indisvalid;

          -- Update reindex_history with completion timestamp, status, and final index size
          update index_pilot.reindex_history
          set
            reindex_duration = clock_timestamp() - entry_timestamp,
            status = 'completed',
            indexsize_after = _final_size
          where
            datname = _index.datname
            and schemaname = _index.schemaname
            and relname = _index.relname
            and indexrelname = _index.indexrelname
            and status = 'in_progress';
        end;
             
      exception when others then
        raise warning 'REINDEX failed for %.%: %', _index.schemaname, _index.indexrelname, sqlerrm;
           
        -- Record failure in reindex_history with error details
        update index_pilot.reindex_history
        set
          status = 'failed',
          error_message = sqlerrm,
          reindex_duration = clock_timestamp() - entry_timestamp
        where
          datname = _index.datname
          and schemaname = _index.schemaname
          and relname = _index.relname
          and indexrelname = _index.indexrelname
          and status = 'in_progress';
      end;
       
      -- Clean up tracking record after successful reindex
      delete from index_pilot.current_processed_index
      where
        datname = _index.datname 
        and schemaname = _index.schemaname 
        and relname = _index.relname 
        and indexrelname = _index.indexrelname;
      
      -- Commit the cleanup
      commit;

      -- Completion status and tracking are updated synchronously in the preceding steps
    end loop;
  return;

-- It prevents commits from working and needs to be done differently
-- exception when others then
--   -- Guaranteed connection cleanup on any exception
--   if _connection_created and dblink_get_connections() is not null 
--      and _datname = any(dblink_get_connections()) then
--     perform dblink_disconnect(_datname);
--   end if;
--   raise; -- Re-raise the original exception
end;
$body$
language plpgsql;


/*
 * Force-populate index statistics and bloat baselines without reindexing
 * Records current size-to-tuple ratios as optimal baselines, supports filtering
 */
create function index_pilot.do_force_populate_index_stats(
  _datname name,
  _schemaname name,
  _relname name,
  _indexrelname name
) returns void as
$body$
declare
  _connection_created boolean := false;
begin
  -- Ensure table structure is at required version
  perform index_pilot._check_structure_version();

  -- Ensure dblink connection is established before starting any transaction with cleanup guarantee
  if dblink_get_connections() is null or not (_datname = any(dblink_get_connections())) then
    perform index_pilot._dblink_connect_if_not(_datname);
    _connection_created := true;
  end if;

  -- Force-populate best_ratio from current state without reindexing
  perform index_pilot._record_indexes_info(_datname, _schemaname, _relname, _indexrelname, _force_populate=>true);
  return;

exception when others then
  -- Guaranteed connection cleanup on any exception
  if _connection_created and dblink_get_connections() is not null 
     and _datname = any(dblink_get_connections()) then
    perform dblink_disconnect(_datname);
  end if;
  raise; -- Re-raise the original exception
end;
$body$
language plpgsql;


/*
 * Acquire advisory lock to prevent concurrent periodic executions
 * Prevents resource conflicts and duplicate processing, returns lock ID or raises exception
 */
create function index_pilot._check_lock() returns bigint as
$body$
declare
  _id bigint;
  _is_not_running boolean;
begin
  -- Get the lock id for the index_pilot namespace
  select oid from pg_namespace where nspname = 'index_pilot' into _id;

  -- Check if the lock is already held by another instance
  select pg_try_advisory_lock(_id) into _is_not_running;

  -- If the lock is already held by another instance, raise an error
  if not _is_not_running then
    raise 'Previous launch of index_pilot.periodic is still running.';
  end if;

  return _id;
end;
$body$
language plpgsql;


/*
 * Clean up orphaned invalid indexes from failed REINDEX INDEX CONCURRENTLY operations
 * Drops leftover "_ccnew" indexes and cleans tracking records to prevent storage waste
 */
create procedure index_pilot._cleanup_our_not_valid_indexes() as
$body$
declare
  _index record;
begin
  for _index in
    select datname, schemaname, relname, indexrelname
    from index_pilot.current_processed_index
  loop
    -- Ensure we have a connection to the target database
    if dblink_get_connections() is null or not (_index.datname = any(dblink_get_connections())) then
      perform index_pilot._connect_securely(_index.datname);
    end if;
    
    -- Check if the invalid _ccnew index exists
    if exists (
      select from dblink(_index.datname,
        format(
          $sql$
            select x.indexrelid
            from pg_index x
            join pg_catalog.pg_class as c on c.oid = x.indrelid
            join pg_catalog.pg_class as i on i.oid = x.indexrelid
            join pg_catalog.pg_namespace as n on n.oid = c.relnamespace
            where
              n.nspname = %1$L
              and c.relname = %2$L
              and i.relname = %3$L || '_ccnew'
              and not x.indisvalid
          $sql$,
          _index.schemaname,
          _index.relname,
          _index.indexrelname
        )
      ) as _res(indexrelid oid))
    then
      if not exists (
        select from dblink(
          _index.datname,
          format(
            $sql$
              select x.indexrelid
              from pg_index x
              join pg_catalog.pg_class as c on c.oid = x.indrelid
              join pg_catalog.pg_class as i on i.oid = x.indexrelid
              join pg_catalog.pg_namespace as n on n.oid = c.relnamespace
              where
                n.nspname = %1$L
                and c.relname = %2$L
                and i.relname = %3$L
            $sql$,
            _index.schemaname,
            _index.relname,
            _index.indexrelname
          )
        ) as _res(indexrelid oid))
      then
        -- Log the missing original index
        raise warning 'The invalid index %.%_ccnew exists, but no original index %.% was found in database %',
          _index.schemaname, _index.indexrelname, _index.schemaname, _index.indexrelname, _index.datname;
      end if;

      -- Drop the invalid _ccnew index
      perform dblink(_index.datname, format('drop index concurrently %I.%I',
        _index.schemaname, _index.indexrelname || '_ccnew'));

      -- Log the drop
      raise warning 'The invalid index %.%_ccnew was dropped in database %',
        _index.schemaname, _index.indexrelname, _index.datname;
    end if;

    -- Clean up the current_processed_index record
    delete from index_pilot.current_processed_index
    where
      datname = _index.datname and
      schemaname = _index.schemaname and
      relname = _index.relname and
      indexrelname = _index.indexrelname;
  end loop;
end;
$body$
language plpgsql;


/*
 * Main periodic execution procedure for automated index maintenance
 * Primary entry point for scheduled operations: validates, migrates, processes databases
 */
create or replace procedure index_pilot.periodic(
  real_run boolean default false,
  force boolean default false
) as
$body$
declare
  _datname name;
  _schemaname name;
  _relname name;
  _indexrelname name;
  _id bigint;
begin
  -- Validate PostgreSQL version safety
  perform index_pilot._validate_pg_version();

  -- Acquire advisory lock to prevent concurrent executions
  select index_pilot._check_lock() into _id;

  -- Check if the table structure is up to date
  perform index_pilot.check_update_structure_version();

  -- Check if we're in control database mode
  if exists (select from pg_tables where schemaname = 'index_pilot' and tablename = 'target_databases') then
    -- Control database mode: process all enabled target databases
    for _datname in 
      select database_name
      from index_pilot.target_databases
      where enabled
    loop
      -- Clean old history for this database
      delete from index_pilot.reindex_history
      where datname = _datname
        and entry_timestamp < now() - coalesce(
          index_pilot.get_setting(datname, schemaname, relname, indexrelname, 'reindex_history_retention_period')::interval,
          '10 years'::interval
        );
          
      -- Record indexes for this database
      perform index_pilot._record_indexes_info(_datname, null, null, null);
          
      if real_run then
        call index_pilot.do_reindex(_datname, null, null, null, force);
        -- refresh snapshot right after reindex to clamp baseline with current ratio
        perform index_pilot._record_indexes_info(_datname, null, null, null);
      end if;
    end loop;
        
    -- Note: No need to update completed reindexes - all tracking is synchronous now
        
    -- Clean up any invalid _ccnew indexes from failed reindexes
    call index_pilot._cleanup_our_not_valid_indexes();
  else
    -- Standalone mode (shouldn't happen with our fixes, but keep for safety)
    raise exception 'Control database architecture required. Cannot run periodic in standalone mode.';
  end if;

  -- Note: best_ratio updates are now handled during snapshot insertion
  -- Historical snapshots preserve the best_ratio calculation at the time of observation
  -- No need to update old snapshots - new snapshots will have updated best_ratio values

  perform pg_advisory_unlock(_id);
end;
$body$
language plpgsql;


/*
 * Comprehensive permission and setup validation for pg_index_pilot
 * Validates required permissions, extensions, and FDW configuration for managed services
 */
create function index_pilot.check_permissions() returns table(
  permission text, 
  status boolean
) as
$body$
begin
  return query select 
    'Can create indexes'::text, 
    has_database_privilege(current_database(), 'create');

  return query select 
    'Can read pg_stat_user_indexes'::text,
    has_table_privilege('pg_stat_user_indexes', 'select');

  return query select 
    'Has dblink extension'::text,
    exists (select from pg_extension where extname = 'dblink');

  return query select 
    'Has postgres_fdw extension'::text,
    exists (select from pg_extension where extname = 'postgres_fdw');

  return query select 
    'Has target servers registered'::text,
    exists (select 1 from index_pilot.target_databases);

  return query select 
    'Has user mapping for dblink'::text,
    exists (
      select 1 from pg_user_mappings as um
      where um.usename = current_user
        and um.srvname in (select fdw_server_name from index_pilot.target_databases where enabled)
    );

  -- Verify reindex capability by checking ownership of at least one index
  return query select 
    'Can reindex (owns indexes)'::text,
    exists (
      select from pg_index as i
      join pg_class as c on i.indexrelid = c.oid
      join pg_namespace as n on c.relnamespace = n.oid
      where
        n.nspname not in ('pg_catalog', 'information_schema')
        and pg_has_role(c.relowner, 'usage')
      limit 1
    );
end;
$body$
language plpgsql;


/*
 * Installation-time permission validation and user guidance
 * Shows setup status and provides clear feedback on missing requirements
 */
do $$
declare
  _perm record;
  _all_ok boolean := true;
begin
  raise notice 'pg_index_pilot - monitoring current database only';
  raise notice 'Database: %', current_database();
  raise notice '';
  raise notice 'Checking permissions...';

  for _perm in select * from index_pilot.check_permissions() loop
    raise notice '  %: %',
      rpad(_perm.permission, 30),
      case when _perm.status then 'OK' else 'MISSING' end;
      if not _perm.status then
        _all_ok := false;
      end if;
  end loop;

  raise notice '';

  if _all_ok then
    raise notice 'All permissions OK. You can use pg_index_pilot.';
  else
    raise warning 'Some permissions are missing. pg_index_pilot may not work correctly.';
  end if;

  raise notice '';
  raise notice 'Usage: call index_pilot.periodic(true);  -- true = perform actual reindexing';
end $$;

commit;