-- Drop helper functions created by prepare-db (template-filled by cli/lib/init.ts)
-- Run before dropping the postgres_ai schema.

-- Objects in other schemas (for example, a planted public.format) must not run as the admin.
-- Use a session SET because --print-sql plans may be run by psql in autocommit.
-- SET LOCAL outside an explicit transaction is ignored there.
set search_path = pg_catalog, pg_temp;

drop function if exists postgres_ai.explain_generic(text, text, text);
drop function if exists postgres_ai.table_describe(text);
