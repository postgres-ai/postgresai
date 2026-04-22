-- Initialize target database for monitoring
-- Enable pg_stat_statements extension for query monitoring
create extension if not exists pg_stat_statements;

-- Create a sample table for demonstration
create table if not exists sample_data (
    id serial primary key,
    name varchar(100),
    created_at timestamp default current_timestamp
);

-- Insert some sample data
insert into sample_data (name) values
    ('Sample Record 1'),
    ('Sample Record 2'),
    ('Sample Record 3');

-- Create the 'monitor' user for pgwatch collection. The 'replicator' user
-- (required by target-standby) is created by 01-create-replicator.sh, which
-- passes the password as a psql variable so it never lands on the server
-- command line or as a queryable GUC.
create user monitor with password 'monitor_pass';
grant connect on database target_database to monitor;
grant usage on schema public to monitor;

-- Create postgres_ai schema and pg_statistic view (matches cli/sql/02.permissions.sql)
create schema if not exists postgres_ai;
grant usage on schema postgres_ai to pg_monitor;

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

-- Grant specific access instead of all tables
grant select on postgres_ai.pg_statistic to pg_monitor;

-- Grant access to monitoring views
grant select on pg_stat_statements to monitor;
grant select on pg_stat_database to monitor;
grant select on pg_stat_user_tables to monitor;
-- Grant pg_monitor role to monitor user for enhanced monitoring capabilities
grant pg_monitor to monitor;
grant execute on function pg_stat_file(text) to monitor;
grant execute on function pg_stat_file(text, boolean) to monitor;
grant execute on function pg_ls_dir(text) to monitor;
grant execute on function pg_ls_dir(text, boolean, boolean) to monitor;
-- Set search path for the monitor user
alter user monitor set search_path = "$user", public, pg_catalog;
