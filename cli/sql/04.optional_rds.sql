-- Optional permissions for RDS Postgres / Aurora (best effort)

create schema if not exists rds_tools;
create extension if not exists rds_tools with schema rds_tools;
grant execute on function rds_tools.pg_ls_multixactdir() to {{ROLE_IDENT}};


