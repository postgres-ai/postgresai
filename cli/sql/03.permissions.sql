-- Required permissions for postgres_ai monitoring user (template-filled by cli/lib/init.ts)

-- Allow connect
grant connect on database {{DB_IDENT}} to {{ROLE_IDENT}};

-- Standard monitoring privileges
grant pg_monitor to {{ROLE_IDENT}};
grant select on pg_catalog.pg_index to {{ROLE_IDENT}};

-- Required for cluster-wide pg_stat_statements collection.
-- pg_stat_statements is backed by shared memory and holds entries for every database
-- in the cluster, but a role without pg_read_all_stats sees other users' rows with
-- queryid and query text redacted to NULL -- which collapses the per-queryid grouping
-- in the pg_stat_statements metric. pg_monitor already implies pg_read_all_stats;
-- granting it explicitly documents the dependency and keeps the metric working if the
-- pg_monitor grant above is ever narrowed. Idempotent.
grant pg_read_all_stats to {{ROLE_IDENT}};

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
do $$
declare
  ext_schema text;
begin
  select n.nspname into ext_schema
  from pg_extension e
  join pg_namespace n on e.extnamespace = n.oid
  where e.extname = 'pg_stat_statements';

  -- Only grant if extension exists and is in a non-standard schema
  if ext_schema is not null and ext_schema not in ('pg_catalog', 'public') then
    execute format('grant usage on schema %I to {{ROLE_IDENT}}', ext_schema);
  end if;
end $$;

-- [SEARCH_PATH_BLOCK_START] Keep search_path predictable; postgres_ai first so our objects are found.
-- Dynamically include the pg_stat_statements extension schema if it's in a non-standard location.
do $$
declare
  ext_schema text;
  sp text;
begin
  -- Detect pg_stat_statements extension schema
  select n.nspname into ext_schema
  from pg_extension e
  join pg_namespace n on e.extnamespace = n.oid
  where e.extname = 'pg_stat_statements';

  -- Build search_path: include extension schema if in non-standard location
  if ext_schema is not null and ext_schema not in ('pg_catalog', 'public') then
    sp := 'postgres_ai, ' || quote_ident(ext_schema) || ', "$user", public, pg_catalog';
  else
    sp := 'postgres_ai, "$user", public, pg_catalog';
  end if;

  execute format('alter user {{ROLE_IDENT}} set search_path = %s', sp);
end $$;
-- [SEARCH_PATH_BLOCK_END]


