-- Drop the monitoring role only when it has no remaining dependencies.
-- A reused role may own customer objects. Do not transfer or delete those
-- objects on the role's behalf; PostgreSQL will reject the drop instead.

-- Objects in other schemas (for example, a planted public.format) must not run as the admin.
-- Printed plans may run in psql autocommit, where SET LOCAL is ignored.
set search_path = pg_catalog, pg_temp;

-- prepare-db grants schema USAGE, but the extension can move or be dropped
-- before cleanup. Revoke every direct schema USAGE grant to the role in the
-- role-drop transaction; unrelated dependencies roll all revokes back.
-- prepare-db also grants CONNECT on the target database, which can differ
-- from the database of this session. Step 02 revokes it only on the current
-- one, so revoke direct CONNECT grants on every database here too.
do {{DO_QUOTE}}
declare
  role_oid oid;
  schema_name name;
  database_name name;
begin
  select oid into role_oid from pg_catalog.pg_roles where rolname = {{ROLE_LITERAL}};
  if role_oid is null then
    return;
  end if;

  for schema_name in
    select distinct n.nspname
    from pg_catalog.pg_namespace n,
      pg_catalog.aclexplode(n.nspacl) a
    where a.grantee = role_oid and a.privilege_type = 'USAGE'
  loop
    execute pg_catalog.format('revoke usage on schema %I from %I', schema_name, {{ROLE_LITERAL}});
  end loop;

  for database_name in
    select distinct d.datname
    from pg_catalog.pg_database d,
      pg_catalog.aclexplode(d.datacl) a
    where a.grantee = role_oid and a.privilege_type = 'CONNECT'
  loop
    execute pg_catalog.format('revoke connect on database %I from %I', database_name, {{ROLE_LITERAL}});
  end loop;
end {{DO_QUOTE}};

drop role if exists {{ROLE_IDENT}};
