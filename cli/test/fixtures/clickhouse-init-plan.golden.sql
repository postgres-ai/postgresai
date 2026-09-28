-- step: 01.role
-- Role creation / password update (template-filled by cli/lib/init.ts)
--
-- Always uses a race-safe pattern (create if missing, then always alter to set the password):
--   do $$ begin
--     if not exists (select 1 from pg_catalog.pg_roles where rolname = '...') then
--       begin
--         create user "..." with password '...';
--       exception when duplicate_object then
--         null;
--       end;
--     end if;
--     alter user "..." with password '...';
--   end $$;
do $$ begin
  if not exists (select 1 from pg_catalog.pg_roles where rolname = 'postgres_ai_mon') then
    begin
      create user "postgres_ai_mon" with password 'CLICKHOUSE_MONITORING_PASSWORD';
    exception when duplicate_object then
      null;
    end;
  end if;
  alter user "postgres_ai_mon" with password 'CLICKHOUSE_MONITORING_PASSWORD';
end $$;

-- step: 02.extensions
-- Extensions required for postgres_ai monitoring

-- Enable pg_stat_statements for query performance monitoring
-- Note: Uses IF NOT EXISTS because extension may already be installed.
-- We do NOT drop this extension in unprepare-db since it may have been pre-existing.
create extension if not exists pg_stat_statements;

-- step: 03.permissions
-- Required permissions for postgres_ai monitoring user (template-filled by cli/lib/init.ts)

-- Allow connect
grant connect on database "postgres" to "postgres_ai_mon";

-- Standard monitoring privileges
grant pg_monitor to "postgres_ai_mon";
grant select on pg_catalog.pg_index to "postgres_ai_mon";

-- Required for cluster-wide pg_stat_statements collection.
-- pg_stat_statements is backed by shared memory and holds entries for every database
-- in the cluster, but a role without pg_read_all_stats sees other users' rows with
-- queryid and query text redacted to NULL -- which collapses the per-queryid grouping
-- in the pg_stat_statements metric. pg_monitor already implies pg_read_all_stats;
-- granting it explicitly documents the dependency and keeps the metric working if the
-- pg_monitor grant above is ever narrowed. Idempotent.
grant pg_read_all_stats to "postgres_ai_mon";

-- Create postgres_ai schema for our objects
-- Using IF NOT EXISTS for idempotency - prepare-db can be run multiple times
create schema if not exists postgres_ai;
grant usage on schema postgres_ai to "postgres_ai_mon";

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

grant select on postgres_ai.pg_statistic to "postgres_ai_mon";

-- Hardened clusters sometimes revoke PUBLIC on schema public
grant usage on schema public to "postgres_ai_mon";

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
    execute format('grant usage on schema %I to "postgres_ai_mon"', ext_schema);
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

  execute format('alter user "postgres_ai_mon" set search_path = %s', sp);
end $$;
-- [SEARCH_PATH_BLOCK_END]

-- step: 06.helpers
-- Helper functions for postgres_ai monitoring user (template-filled by cli/lib/init.ts)
-- These functions use SECURITY DEFINER to allow the monitoring user to perform
-- operations they don't have direct permissions for.

/*
 * table_describe
 *
 * Collects comprehensive information about a table for LLM analysis.
 * Returns a compact text format with:
 * - Table metadata (type, size estimates)
 * - Columns (name, type, nullable, default)
 * - Indexes
 * - Constraints (PK, FK, unique, check)
 * - Maintenance stats (vacuum/analyze times)
 *
 * Usage:
 *   select postgres_ai.table_describe('public.users');
 *   select postgres_ai.table_describe('my_table');  -- uses search_path
 */
create or replace function postgres_ai.table_describe(
  in table_name text,
  out result text
)
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_oid oid;
  v_schema text;
  v_table text;
  v_relkind char;
  v_relpages int;
  v_reltuples float;
  v_lines text[] := '{}';
  v_line text;
  v_rec record;
  v_constraint_count int := 0;
begin
  -- Resolve table name to OID (handles schema-qualified and search_path)
  v_oid := table_name::regclass::oid;

  -- Get basic table info
  select
    n.nspname,
    c.relname,
    c.relkind,
    c.relpages,
    c.reltuples
  into v_schema, v_table, v_relkind, v_relpages, v_reltuples
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  where c.oid = v_oid;

  -- Validate object type - only tables, views, and materialized views are supported
  if v_relkind not in ('r', 'p', 'v', 'm', 'f') then
    raise exception 'table_describe does not support % (relkind=%)',
      case v_relkind
        when 'i' then 'indexes'
        when 'I' then 'partitioned indexes'
        when 'S' then 'sequences'
        when 't' then 'TOAST tables'
        when 'c' then 'composite types'
        else format('objects of type "%s"', v_relkind)
      end,
      v_relkind;
  end if;

  -- Header
  v_lines := array_append(v_lines, format('Table: %I.%I', v_schema, v_table));
  v_lines := array_append(v_lines, format('Type: %s | relpages: %s | reltuples: %s',
    case v_relkind
      when 'r' then 'table'
      when 'p' then 'partitioned table'
      when 'v' then 'view'
      when 'm' then 'materialized view'
      when 'f' then 'foreign table'
    end,
    v_relpages,
    case when v_reltuples < 0 then '-1' else v_reltuples::bigint::text end
  ));

  -- Vacuum/analyze stats (only for tables and materialized views, not views)
  if v_relkind in ('r', 'p', 'm', 'f') then
    select
      format('Vacuum: %s (auto: %s) | Analyze: %s (auto: %s)',
        coalesce(to_char(last_vacuum at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS UTC'), 'never'),
        coalesce(to_char(last_autovacuum at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS UTC'), 'never'),
        coalesce(to_char(last_analyze at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS UTC'), 'never'),
        coalesce(to_char(last_autoanalyze at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS UTC'), 'never')
      )
    into v_line
    from pg_stat_all_tables
    where relid = v_oid;

    if v_line is not null then
      v_lines := array_append(v_lines, v_line);
    end if;
  end if;

  v_lines := array_append(v_lines, '');

  -- Columns
  v_lines := array_append(v_lines, 'Columns:');
  for v_rec in
    select
      a.attname,
      format_type(a.atttypid, a.atttypmod) as data_type,
      a.attnotnull,
      (select pg_get_expr(d.adbin, d.adrelid, true)
       from pg_attrdef d
       where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef) as default_val,
      a.attidentity,
      a.attgenerated
    from pg_attribute a
    where a.attrelid = v_oid
      and a.attnum > 0
      and not a.attisdropped
    order by a.attnum
  loop
    v_line := format('  %s %s', v_rec.attname, v_rec.data_type);

    if v_rec.attnotnull then
      v_line := v_line || ' NOT NULL';
    end if;

    if v_rec.attidentity = 'a' then
      v_line := v_line || ' GENERATED ALWAYS AS IDENTITY';
    elsif v_rec.attidentity = 'd' then
      v_line := v_line || ' GENERATED BY DEFAULT AS IDENTITY';
    elsif v_rec.attgenerated = 's' then
      v_line := v_line || format(' GENERATED ALWAYS AS (%s) STORED', v_rec.default_val);
    elsif v_rec.default_val is not null then
      v_line := v_line || format(' DEFAULT %s', v_rec.default_val);
    end if;

    v_lines := array_append(v_lines, v_line);
  end loop;

  -- View definition (for views and materialized views)
  if v_relkind in ('v', 'm') then
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Definition:');
    v_line := pg_get_viewdef(v_oid, true);
    if v_line is not null then
      -- Indent the view definition
      v_line := '  ' || replace(v_line, e'\n', e'\n  ');
      v_lines := array_append(v_lines, v_line);
    end if;
  end if;

  -- Indexes (tables, partitioned tables, and materialized views can have indexes)
  if v_relkind in ('r', 'p', 'm') then
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Indexes:');
    for v_rec in
      select
        i.relname as index_name,
        pg_get_indexdef(i.oid) as index_def,
        ix.indisprimary,
        ix.indisunique
      from pg_index ix
      join pg_class i on i.oid = ix.indexrelid
      where ix.indrelid = v_oid
      order by ix.indisprimary desc, ix.indisunique desc, i.relname
    loop
      v_line := '  ';
      if v_rec.indisprimary then
        v_line := v_line || 'PRIMARY KEY: ';
      elsif v_rec.indisunique then
        v_line := v_line || 'UNIQUE: ';
      else
        v_line := v_line || 'INDEX: ';
      end if;
      -- Extract just the column part from index definition
      v_line := v_line || v_rec.index_name || ' ' ||
        regexp_replace(v_rec.index_def, '^CREATE.*INDEX.*ON.*USING\s+\w+\s*', '');
      v_lines := array_append(v_lines, v_line);
    end loop;

    if not exists (select 1 from pg_index where indrelid = v_oid) then
      v_lines := array_append(v_lines, '  (none)');
    end if;
  end if;

  -- Constraints (only tables can have constraints)
  if v_relkind in ('r', 'p', 'f') then
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Constraints:');
    v_constraint_count := 0;

    for v_rec in
      select
        conname,
        contype,
        pg_get_constraintdef(oid, true) as condef
      from pg_constraint
      where conrelid = v_oid
        and contype != 'p'  -- skip primary key (shown with indexes)
      order by
        case contype when 'f' then 1 when 'u' then 2 when 'c' then 3 else 4 end,
        conname
    loop
      v_constraint_count := v_constraint_count + 1;
      v_line := '  ';
      case v_rec.contype
        when 'f' then v_line := v_line || 'FK: ';
        when 'u' then v_line := v_line || 'UNIQUE: ';
        when 'c' then v_line := v_line || 'CHECK: ';
        else v_line := v_line || v_rec.contype || ': ';
      end case;
      v_line := v_line || v_rec.conname || ' ' || v_rec.condef;
      v_lines := array_append(v_lines, v_line);
    end loop;

    if v_constraint_count = 0 then
      v_lines := array_append(v_lines, '  (none)');
    end if;

    -- Foreign keys referencing this table
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Referenced by:');
    v_constraint_count := 0;

    for v_rec in
      select
        conname,
        conrelid::regclass::text as from_table,
        pg_get_constraintdef(oid, true) as condef
      from pg_constraint
      where confrelid = v_oid
        and contype = 'f'
      order by conrelid::regclass::text, conname
    loop
      v_constraint_count := v_constraint_count + 1;
      v_lines := array_append(v_lines, format('  %s.%s %s',
        v_rec.from_table, v_rec.conname, v_rec.condef));
    end loop;

    if v_constraint_count = 0 then
      v_lines := array_append(v_lines, '  (none)');
    end if;
  end if;

  -- Partition info (if partitioned table or partition)
  if v_relkind = 'p' then
    -- This is a partitioned table - show partition key and partitions
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Partitioning:');

    select format('  %s BY %s',
      case partstrat
        when 'r' then 'RANGE'
        when 'l' then 'LIST'
        when 'h' then 'HASH'
        else partstrat
      end,
      pg_get_partkeydef(v_oid)
    )
    into v_line
    from pg_partitioned_table
    where partrelid = v_oid;

    if v_line is not null then
      v_lines := array_append(v_lines, v_line);
    end if;

    -- List partitions
    v_constraint_count := 0;
    for v_rec in
      select
        c.oid::regclass::text as partition_name,
        pg_get_expr(c.relpartbound, c.oid) as partition_bound,
        c.relpages,
        c.reltuples
      from pg_inherits i
      join pg_class c on c.oid = i.inhrelid
      where i.inhparent = v_oid
      order by c.oid::regclass::text
    loop
      v_constraint_count := v_constraint_count + 1;
      v_lines := array_append(v_lines, format('  %s: %s (relpages: %s, reltuples: %s)',
        v_rec.partition_name, v_rec.partition_bound,
        v_rec.relpages,
        case when v_rec.reltuples < 0 then '-1' else v_rec.reltuples::bigint::text end
      ));
    end loop;

    v_lines := array_append(v_lines, format('  Total partitions: %s', v_constraint_count));

  elsif exists (select 1 from pg_inherits where inhrelid = v_oid) then
    -- This is a partition - show parent and bound
    v_lines := array_append(v_lines, '');
    v_lines := array_append(v_lines, 'Partition of:');

    select format('  %s FOR VALUES %s',
      i.inhparent::regclass::text,
      pg_get_expr(c.relpartbound, c.oid)
    )
    into v_line
    from pg_inherits i
    join pg_class c on c.oid = i.inhrelid
    where i.inhrelid = v_oid;

    if v_line is not null then
      v_lines := array_append(v_lines, v_line);
    end if;
  end if;

  result := array_to_string(v_lines, e'\n');
end;
$$;

comment on function postgres_ai.table_describe(text) is
  'Returns comprehensive table information in compact text format for LLM analysis';

grant execute on function postgres_ai.table_describe(text) to "postgres_ai_mon";

-- step: 04.optional_rds
-- Optional permissions for RDS Postgres / Aurora (best effort)

create extension if not exists rds_tools;
grant execute on function rds_tools.pg_ls_multixactdir() to "postgres_ai_mon";

-- step: 05.optional_self_managed
-- Optional permissions for self-managed Postgres (best effort)

grant execute on function pg_catalog.pg_stat_file(text) to "postgres_ai_mon";
grant execute on function pg_catalog.pg_stat_file(text, boolean) to "postgres_ai_mon";
grant execute on function pg_catalog.pg_ls_dir(text) to "postgres_ai_mon";
grant execute on function pg_catalog.pg_ls_dir(text, boolean, boolean) to "postgres_ai_mon";
