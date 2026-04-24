-- Initialize PostgreSQL sink database for storing pgwatch measurements
-- This database will store all the monitoring metrics collected by PGWatch
-- Based on https://pgwat.ch/latest/howto/metrics_db_bootstrap.html

-- Create the pgwatch role for measurements database
create role pgwatch with login password 'pgwatchadmin';

-- Create the measurements database owned by pgwatch
create database measurements owner pgwatch;

-- Switch to the measurements database context
\c measurements;

-- Create extensions that might be useful for metrics storage
create extension if not exists btree_gist;
create extension if not exists pg_stat_statements;

-- Grant necessary permissions to pgwatch user
grant all privileges on database measurements to pgwatch;
grant all privileges on schema public to pgwatch;



-- Create a partitioned table for queryid-to-query mappings with LIST partitioning by dbname
create table if not exists public.pgss_queryid_queries (
  time timestamptz not null,
  dbname text not null,
  data jsonb not null,
  tag_data jsonb
) partition by list (dbname);

-- Create indexes for efficient lookups
create index if not exists pgss_queryid_queries_dbname_time_idx on public.pgss_queryid_queries (dbname, time);

-- Index for dedup trigger: without this, the BEFORE INSERT trigger does a full
-- sequential scan per row during COPY, making bulk inserts take 20+ minutes
-- and causing duplicate rows to pile up across overlapping COPY batches.
create index if not exists pgss_queryid_queries_queryid_idx on public.pgss_queryid_queries ((data->>'queryid'));

-- Use existing subpartitions schema


-- Set ownership and grant permissions to pgwatch
alter table public.pgss_queryid_queries owner to pgwatch;
grant all privileges on table public.pgss_queryid_queries to pgwatch;
-- Ensure pgwatch can use sequences (if any are created)
grant usage on schema public to pgwatch;
-- Grant permissions on all future tables in public schema
alter default privileges in schema public grant all on tables to pgwatch; 

-- Drop the old parent unique index if a previous deployment created it.
-- pgwatch creates a second partition level by time; PostgreSQL cannot
-- inherit a unique index that omits every partition key, so uniqueness is
-- enforced by the trigger-level advisory lock below instead.
drop index if exists public.pgss_queryid_queries_upsert_idx;

create or replace function enforce_queryid_uniqueness()
returns trigger as $func$
declare
  queryid_value text;
begin
  queryid_value := new.data->>'queryid';

  -- Allow incomplete rows through; pgwatch emits non-NULL dbnames/queryids
  -- for pg_stat_statements rows, and the fallback avoids advisory-lock NULLs.
  if queryid_value is null or new.dbname is null then
    return new;
  end if;

  -- Serialize by (dbname, queryid) so concurrent COPY batches cannot insert
  -- duplicate rows for the same queryid. This avoids a unique parent index,
  -- which is incompatible with pgwatch's dbname/time partition hierarchy.
  perform pg_advisory_xact_lock(hashtext(new.dbname), hashtext(queryid_value));

  -- Replace older/equal queryid rows without updating the time partition key.
  -- If a newer row already exists, skip the stale insert instead of rewinding
  -- the active-query window used by /query_info_metrics.
  if exists (
    select 1
    from public.pgss_queryid_queries
    where public.pgss_queryid_queries.dbname = new.dbname
      and public.pgss_queryid_queries.data->>'queryid' = queryid_value
      and public.pgss_queryid_queries.time > new.time
    limit 1
  ) then
    return null;
  end if;

  delete from public.pgss_queryid_queries
  where public.pgss_queryid_queries.dbname = new.dbname
    and public.pgss_queryid_queries.data->>'queryid' = queryid_value
    and public.pgss_queryid_queries.time <= new.time;

  return new;
end;
$func$ language plpgsql;


-- Function stays owned by the bootstrap superuser — not pgwatch — so a
-- compromised flask-pgss-api container (which connects as pgwatch) can't
-- rewrite the dedup/recursion-guard logic. Upgrading the function body
-- requires re-running this script as the bootstrap role. PUBLIC holds
-- EXECUTE on SQL functions by default, so the pgwatch role can still
-- fire the trigger.

create or replace trigger enforce_queryid_uniqueness_trigger
  before insert
  on pgss_queryid_queries
  for each row
  execute function enforce_queryid_uniqueness();

-- Create a partitioned table for index definitions with LIST partitioning by dbname
create table if not exists public.index_definitions (
  time timestamptz not null,
  dbname text not null,
  data jsonb not null,
  tag_data jsonb
) partition by list (dbname);

-- Create indexes for efficient lookups
create index if not exists index_definitions_dbname_time_idx on public.index_definitions (dbname, time);

-- Index for dedup trigger: same pattern as pgss_queryid_queries
create index if not exists index_definitions_dedup_idx on public.index_definitions (
  dbname, (data->>'indexrelname'), (data->>'schemaname'), (data->>'relname')
);

-- Set ownership and grant permissions to pgwatch
alter table public.index_definitions owner to pgwatch;
grant all privileges on table public.index_definitions to pgwatch;

-- Create function to enforce index definition uniqueness
create or replace function enforce_index_definition_uniqueness()
returns trigger as $$
declare
  index_name text;
  schema_name text;
  table_name text;
  index_definition text;
begin
  -- Extract index information from the data JSONB
  index_name := new.data->>'indexrelname';
  schema_name := new.data->>'schemaname';
  table_name := new.data->>'relname';
  index_definition := new.data->>'index_definition';
  
  -- Allow NULL index names through
  if index_name is null then
    return new;
  end if;
  
  -- Silently skip if duplicate exists
  if exists (
    select 1 
    from index_definitions
    where dbname = new.dbname
      and data->>'indexrelname' = index_name
      and data->>'schemaname' = schema_name
      and data->>'relname' = table_name
      and data->>'index_definition' = index_definition
    limit 1
  ) then
    return null;  -- Cancels INSERT silently
  end if;
  
  return new;
end;
$$ language plpgsql;

create or replace trigger enforce_index_definition_uniqueness_trigger
  before insert
  on index_definitions
  for each row
  execute function enforce_index_definition_uniqueness();


