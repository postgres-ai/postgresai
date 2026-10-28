-- Revoke permissions and drop objects created by prepare-db (template-filled by cli/lib/init.ts)

-- Objects in other schemas (for example, a planted public.=) must not run as the admin.
-- Printed plans may run in psql autocommit, where SET LOCAL is ignored.
set search_path = pg_catalog, pg_temp;
-- The CLI reads the "unprepare-db:" WARNINGs. A higher setting would hide them.
set client_min_messages = warning;

-- Drop the postgres_ai.pg_statistic view
drop view if exists postgres_ai.pg_statistic;

-- Revoke only the grants that the role holds. The postgres_ai schema stays
-- because it may contain customer objects. The pg_read_all_stats membership
-- comes from older prepare-db versions. The functions come from optional
-- permissions (04, 05). A non-superuser admin cannot revoke a grant that a
-- superuser made. Pass 1 revokes. Pass 2 sends one "grant left" WARNING for
-- each grant that stays. The CLI shows these WARNINGs and exits 1.
do {{DO_QUOTE}}
declare
  role_oid oid;
  stmt text;
begin
  select oid into role_oid from pg_catalog.pg_roles where rolname = {{ROLE_LITERAL}};
  if role_oid is null then
    return; -- The monitoring role was already removed.
  end if;

  for pass in 1..2 loop
    for stmt in
      -- PG 16+ keeps one row for each grantor. A revoke without GRANTED BY
      -- removes only the row of the current user. PG < 16 can keep the oid of
      -- a dropped grantor; then the statement has no GRANTED BY.
      select pg_catalog.format('revoke %s from %I', m.roleid::regrole, {{ROLE_LITERAL}})
        || coalesce(' granted by ' || pg_catalog.quote_ident(g.rolname), '')
      from pg_catalog.pg_auth_members m
      left join pg_catalog.pg_roles g on g.oid = m.grantor
      where m.member = role_oid
        and m.roleid in ('pg_monitor'::regrole, 'pg_read_all_stats'::regrole)
      union
      select pg_catalog.format('revoke select on pg_catalog.pg_index from %I', {{ROLE_LITERAL}})
      from pg_catalog.pg_class c
      where c.oid operator(pg_catalog.=) pg_catalog.to_regclass('pg_catalog.pg_index')::oid
        and exists (
          select from pg_catalog.aclexplode(c.relacl) a
          where a.grantee = role_oid and a.privilege_type = 'SELECT')
      union
      select pg_catalog.format('revoke connect on database %I from %I', d.datname, {{ROLE_LITERAL}})
      from pg_catalog.pg_database d
      where d.datname = pg_catalog.current_database()
        and exists (
          select from pg_catalog.aclexplode(d.datacl) a
          where a.grantee = role_oid and a.privilege_type = 'CONNECT')
      union
      select pg_catalog.format('revoke usage on schema postgres_ai from %I', {{ROLE_LITERAL}})
      from pg_catalog.pg_namespace n
      where n.nspname = 'postgres_ai'
        and exists (
          select from pg_catalog.aclexplode(n.nspacl) a
          where a.grantee = role_oid and a.privilege_type = 'USAGE')
      union
      -- prepare-db grants USAGE when pg_stat_statements lives in a custom
      -- extension schema. Retaining the login role must remove that grant too.
      select pg_catalog.format('revoke usage on schema %I from %I', n.nspname, {{ROLE_LITERAL}})
      from pg_catalog.pg_extension e
      join pg_catalog.pg_namespace n on n.oid = e.extnamespace
      where e.extname = 'pg_stat_statements'
        and n.nspname not in ('pg_catalog', 'public')
        and exists (
          select from pg_catalog.aclexplode(n.nspacl) a
          where a.grantee = role_oid and a.privilege_type = 'USAGE')
      union
      select pg_catalog.format('revoke execute on function %s from %I', p.oid::regprocedure, {{ROLE_LITERAL}})
      from pg_catalog.pg_proc p
      where p.oid in (
          pg_catalog.to_regprocedure('rds_tools.pg_ls_multixactdir()'),
          pg_catalog.to_regprocedure('pg_catalog.pg_stat_file(text)'),
          pg_catalog.to_regprocedure('pg_catalog.pg_stat_file(text, boolean)'),
          pg_catalog.to_regprocedure('pg_catalog.pg_ls_dir(text)'),
          pg_catalog.to_regprocedure('pg_catalog.pg_ls_dir(text, boolean, boolean)'))
        and exists (
          select from pg_catalog.aclexplode(p.proacl) a
          where a.grantee = role_oid and a.privilege_type = 'EXECUTE')
    loop
      if pass = 2 then
        raise warning 'unprepare-db: grant left; a superuser can run: %', stmt;
      else
        begin
          execute stmt;
        exception when insufficient_privilege then
          raise warning 'unprepare-db: % failed: %', stmt, sqlerrm;
        end;
      end if;
    end loop;
  end loop;
end {{DO_QUOTE}};

-- Step 02 keeps a direct USAGE grant on public for --keep-role. Step 03
-- revokes every direct schema USAGE grant only inside the DROP ROLE transaction.
