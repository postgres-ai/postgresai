-- Required permissions for postgres_ai monitoring user (template-filled by cli/lib/init.ts)
-- The first DO blocks also run as the admin; pin before any name resolution.
set search_path = pg_catalog, pg_temp;

-- Allow connect
grant connect on database {{DB_IDENT}} to {{ROLE_IDENT}};

-- Standard monitoring privileges
grant pg_monitor to {{ROLE_IDENT}};
grant select on pg_catalog.pg_index to {{ROLE_IDENT}};

-- Cluster-wide pg_stat_statements collection needs pg_read_all_stats (without it, other
-- users' rows have queryid and query text redacted to NULL). pg_monitor includes it, so it
-- is not granted directly: on PostgreSQL 16+ that needs ADMIN OPTION on pg_read_all_stats,
-- which Supabase's `postgres` role does not have (it has ADMIN OPTION on pg_monitor only).

-- Create postgres_ai schema for our objects; an existing one is kept (prepare-db
-- can be run multiple times). Default privileges can grant CREATE on every new
-- schema to another role (ClickHouse Cloud grants it to chpg_console_admin).
-- A schema created here is still empty, so revoke those grants.
do {{DO_QUOTE}}
declare
  create_grantee text;
begin
  set local search_path = pg_catalog, pg_temp;
  if exists (select from pg_catalog.pg_namespace where nspname = 'postgres_ai') then
    return;
  end if;
  begin
    create schema postgres_ai;
  exception when duplicate_schema or unique_violation then
    -- Created concurrently (unique_violation when its CREATE raced this one
    -- past the exists check): checked below like any existing schema.
    return;
  end;
  for create_grantee in
    select distinct case when a.grantee = 0 then 'public'
                         else pg_catalog.quote_ident(pg_catalog.pg_get_userbyid(a.grantee)) end
    from pg_catalog.pg_namespace n, pg_catalog.aclexplode(n.nspacl) a
    where n.nspname = 'postgres_ai' and a.privilege_type = 'CREATE' and a.grantee <> n.nspowner
  loop
    execute pg_catalog.format('revoke create on schema postgres_ai from %s', create_grantee);
  end loop;
end {{DO_QUOTE}};
-- An existing schema keeps its owner, and CREATE OR REPLACE keeps the owner of
-- existing functions/views. Refuse a schema that another role can still write:
-- the monitoring role searches postgres_ai and calls our helper from it.
do {{DO_QUOTE}}
declare
  schema_oid oid;
  admin_oid oid;
  schema_owner text;
  create_grantee text;
  schema_extension text;
  foreign_object text;
begin
  set local search_path = pg_catalog, pg_temp;
  select oid into admin_oid from pg_catalog.pg_roles where rolname = current_user;
  select oid into schema_oid from pg_catalog.pg_namespace where nspname = 'postgres_ai';
  -- Under repeatable read, a schema another session committed after this
  -- snapshot is invisible here; it cannot be checked, so refuse.
  if schema_oid is null then
    raise exception 'postgres_ai schema was created by another session during prepare-db'
      using errcode = '40001',
        hint = 'Run prepare-db again.';
  end if;

  select pg_catalog.pg_get_userbyid(n.nspowner) into schema_owner
  from pg_catalog.pg_namespace n where n.oid = schema_oid;
  if schema_owner <> current_user then
    raise exception 'postgres_ai schema must be owned by the prepare-db administrator'
      using errcode = '42501',
        detail = pg_catalog.format('Schema owner: %s; prepare-db user: %s', schema_owner, current_user),
        hint = 'Have a DBA inspect postgres_ai, then transfer its ownership to the prepare-db user if safe.';
  end if;

  select case when a.grantee = 0 then 'PUBLIC' else pg_catalog.pg_get_userbyid(a.grantee) end
  into create_grantee
    from pg_catalog.pg_namespace n,
      pg_catalog.aclexplode(coalesce(n.nspacl, pg_catalog.acldefault('n', n.nspowner))) a
    where n.oid = schema_oid and a.privilege_type = 'CREATE' and a.grantee <> admin_oid
    limit 1;
  if create_grantee is not null then
    raise exception 'postgres_ai schema grants CREATE to another role'
      using errcode = '42501',
        detail = pg_catalog.format('CREATE grantee: %s', create_grantee),
        hint = 'Have a DBA inspect the grant, then revoke CREATE on postgres_ai from that role or PUBLIC if safe.';
  end if;

  -- Trusted extensions can install bootstrap-owned members without owner
  -- records in pg_shdepend (bootstrap superuser oid 10). Keep extensions out
  -- of the monitoring search path regardless of their extension owner.
  select e.extname into schema_extension
  from pg_catalog.pg_extension e
  where e.extnamespace = schema_oid
  limit 1;
  if schema_extension is not null then
    raise exception 'postgres_ai schema contains an extension'
      using errcode = '42501',
        detail = pg_catalog.format('Extension: %s', schema_extension),
        hint = 'Have a DBA inspect and move or remove this extension if safe before retrying prepare-db.';
  end if;

  -- Pinned/predefined roles have no pg_shdepend owner rows. Read catalog
  -- owners directly before the dependency scan so those objects cannot hide.
  with owned as (
    select 'pg_class'::pg_catalog.regclass::oid as classid, c.oid as objid, c.relowner as owner_oid
      from pg_catalog.pg_class c where c.relnamespace = schema_oid
    union all select 'pg_proc'::pg_catalog.regclass::oid, p.oid, p.proowner
      from pg_catalog.pg_proc p where p.pronamespace = schema_oid
    union all select 'pg_type'::pg_catalog.regclass::oid, t.oid, t.typowner
      from pg_catalog.pg_type t where t.typnamespace = schema_oid
    union all select 'pg_operator'::pg_catalog.regclass::oid, o.oid, o.oprowner
      from pg_catalog.pg_operator o where o.oprnamespace = schema_oid
    union all select 'pg_opclass'::pg_catalog.regclass::oid, o.oid, o.opcowner
      from pg_catalog.pg_opclass o where o.opcnamespace = schema_oid
    union all select 'pg_opfamily'::pg_catalog.regclass::oid, o.oid, o.opfowner
      from pg_catalog.pg_opfamily o where o.opfnamespace = schema_oid
    union all select 'pg_collation'::pg_catalog.regclass::oid, c.oid, c.collowner
      from pg_catalog.pg_collation c where c.collnamespace = schema_oid
    union all select 'pg_conversion'::pg_catalog.regclass::oid, c.oid, c.conowner
      from pg_catalog.pg_conversion c where c.connamespace = schema_oid
    union all select 'pg_ts_dict'::pg_catalog.regclass::oid, d.oid, d.dictowner
      from pg_catalog.pg_ts_dict d where d.dictnamespace = schema_oid
    union all select 'pg_ts_config'::pg_catalog.regclass::oid, c.oid, c.cfgowner
      from pg_catalog.pg_ts_config c where c.cfgnamespace = schema_oid
    union all select 'pg_statistic_ext'::pg_catalog.regclass::oid, s.oid, s.stxowner
      from pg_catalog.pg_statistic_ext s where s.stxnamespace = schema_oid
  )
  select pg_catalog.format('%s %s (owner %s)', o.type, o.identity,
                           pg_catalog.pg_get_userbyid(owned.owner_oid)) into foreign_object
  from owned
  cross join lateral pg_catalog.pg_identify_object(owned.classid, owned.objid, 0) o
  where owned.owner_oid <> admin_oid
  limit 1;

  -- Retain a dependency scan for object kinds not represented above.
  if foreign_object is null then
  select o.identity into foreign_object
  from pg_catalog.pg_shdepend s
  cross join lateral pg_catalog.pg_identify_object(s.classid, s.objid, s.objsubid) o
  where s.dbid = (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database())
    and s.deptype = 'o' and s.refobjid <> admin_oid
    and o.schema = 'postgres_ai'
  limit 1;
  end if;
  if foreign_object is not null then
    raise exception 'postgres_ai schema contains objects owned by another role'
      using errcode = '42501',
        detail = pg_catalog.format('Foreign-owned object: %s', foreign_object),
        hint = 'Have a DBA inspect this object, then reassign or remove it if safe before retrying prepare-db.';
  end if;
end {{DO_QUOTE}};
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
