-- Required permissions for postgres_ai monitoring user (template-filled by cli/lib/init.ts)

-- Allow connect
grant connect on database {{DB_IDENT}} to {{ROLE_IDENT}};

-- Standard monitoring privileges
grant pg_monitor to {{ROLE_IDENT}};
grant select on pg_catalog.pg_index to {{ROLE_IDENT}};

-- Cluster-wide pg_stat_statements collection needs pg_read_all_stats (without it, other
-- users' rows have queryid and query text redacted to NULL). pg_monitor includes it, so it
-- is not granted directly: on PostgreSQL 16+ that needs ADMIN OPTION on pg_read_all_stats,
-- which Supabase's `postgres` role does not have (it has ADMIN OPTION on pg_monitor only).

-- Create postgres_ai schema for our objects
-- Using IF NOT EXISTS for idempotency - prepare-db can be run multiple times
create schema if not exists postgres_ai;
grant usage on schema postgres_ai to {{ROLE_IDENT}};

-- For bloat analysis: expose pg_statistic via a view
create or replace view postgres_ai.pg_statistic as
select
    n.nspname as schemaname,
    c.relname as tablename,
    a.attname,
    s.stanullfrac as null_frac,
    s.stawidth as avg_width,
    false as inherited
from pg_catalog.pg_statistic s
join pg_catalog.pg_class c on c.oid = s.starelid
join pg_catalog.pg_namespace n on n.oid = c.relnamespace
join pg_catalog.pg_attribute a on a.attrelid = s.starelid and a.attnum = s.staattnum
where a.attnum > 0 and not a.attisdropped;

grant select on postgres_ai.pg_statistic to {{ROLE_IDENT}};

-- Hardened clusters sometimes revoke PUBLIC on schema public
grant usage on schema public to {{ROLE_IDENT}};

-- Grant access to the schema where pg_stat_statements is installed.
-- Some providers (e.g., Supabase) install extensions in a separate 'extensions' schema
-- rather than pg_catalog. This DO block detects the schema and grants USAGE if needed.
do {{DO_QUOTE}}
declare
  ext_schema text;
begin
  select n.nspname into ext_schema
  from pg_catalog.pg_extension e
  join pg_catalog.pg_namespace n on e.extnamespace = n.oid
  where e.extname = 'pg_stat_statements';

  -- Only grant if extension exists and is in a non-standard schema
  if ext_schema is not null and ext_schema not in ('pg_catalog', 'public') then
    execute pg_catalog.format('grant usage on schema %I to %I', ext_schema, {{ROLE_LITERAL}});
  end if;
end {{DO_QUOTE}};

-- [SEARCH_PATH_BLOCK_START] Resolve built-ins from pg_catalog first; keep pg_temp last.
-- Do not include the role-named schema: it may be writable by that role.
-- public stays for pg_stat_statements. Residual risk: a function in public with
-- an exact-type match (e.g. public.quote_ident(name)) still beats pg_catalog.
-- The path for this database includes its pg_stat_statements schema if it is in a non-standard location.
do {{DO_QUOTE}}
declare
  ext_schema text;
  role_path constant text := 'pg_catalog, postgres_ai, public, pg_temp';
  db_path text := role_path;
begin
  -- This block runs as the admin: resolve its operators and functions from pg_catalog only
  -- (CVE-2018-1058). The pin is inside the block: a DO block always runs in a transaction,
  -- also in psql autocommit.
  set local search_path = pg_catalog, pg_temp;
  -- Detect pg_stat_statements extension schema
  select n.nspname into ext_schema
  from pg_catalog.pg_extension e
  join pg_catalog.pg_namespace n on e.extnamespace = n.oid
  where e.extname = 'pg_stat_statements';

  if ext_schema is not null and ext_schema not in ('pg_catalog', 'public') then
    db_path := 'pg_catalog, postgres_ai, ' || pg_catalog.quote_ident(ext_schema) || ', public, pg_temp';
  end if;

  -- Role-wide value: for databases that prepare-db did not run in. All databases share
  -- this one row, so write it only when it holds another value.
  if not exists (
    select from pg_catalog.pg_db_role_setting
    where setrole = (select oid from pg_catalog.pg_roles where rolname = {{ROLE_LITERAL}})
      and setdatabase = 0
      and ('search_path=' || role_path) = any (setconfig)
  ) then
    execute pg_catalog.format('alter user %I set search_path = %s', {{ROLE_LITERAL}}, role_path);
  end if;
  -- This database's value: other databases keep their own pg_stat_statements schema.
  execute pg_catalog.format('alter user %I in database %I set search_path = %s', {{ROLE_LITERAL}}, pg_catalog.current_database(), db_path);
end {{DO_QUOTE}};
-- [SEARCH_PATH_BLOCK_END]
