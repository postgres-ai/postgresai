-- Extensions required for postgres_ai monitoring

-- Enable pg_stat_statements for query performance monitoring
-- Note: Uses IF NOT EXISTS because extension may already be installed.
-- We do NOT drop this extension in unprepare-db since it may have been pre-existing.
create extension if not exists pg_stat_statements with schema public;


