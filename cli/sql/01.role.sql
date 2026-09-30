-- Role creation / password update (template-filled by cli/lib/init.ts)
--
-- Always uses a race-safe pattern (create if missing, then alter to set the password;
-- `pgai connect` omits the alter so an existing role keeps its password):
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
{{ROLE_STMT}}


