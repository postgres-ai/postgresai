-- Optional permissions for self-managed Postgres (best effort)
set search_path = pg_catalog, pg_temp;

grant execute on function pg_catalog.pg_stat_file(text) to {{ROLE_IDENT}};
grant execute on function pg_catalog.pg_stat_file(text, boolean) to {{ROLE_IDENT}};
grant execute on function pg_catalog.pg_ls_dir(text) to {{ROLE_IDENT}};
grant execute on function pg_catalog.pg_ls_dir(text, boolean, boolean) to {{ROLE_IDENT}};


