-- SQL script to manually create lock contention for testing lock_waits metric
-- 
-- Usage:
--   1. Run this script in Session 1 (blocker)
--   2. Run the same script in Session 2 (waiter) - it will wait
--   3. Check the sink database for lock_waits records
--   4. Commit or rollback Session 1 to release the lock

-- Create test table if it doesn't exist
drop table if exists lock_test_table cascade;
create table lock_test_table (
    id int8 generated always as identity primary key,
    name text not null,
    value numeric(10, 2),
    created_at timestamptz default now()
);

insert into lock_test_table (name, value)
values
    ('Item 1', 100.50),
    ('Item 2', 200.75),
    ('Item 3', 300.25);

-- ============================================
-- SESSION 1 (BLOCKER) - Run this first
-- ============================================
begin;

-- Acquire exclusive lock on row id=1
-- Keep this transaction open to hold the lock
select * from lock_test_table where id = 1 for update;

-- Transaction is now holding the lock
-- DO NOT COMMIT YET - keep this session open

-- ============================================
-- SESSION 2 (WAITER) - Run this in another psql session
-- ============================================
begin;

-- This will wait for Session 1 to release the lock
select * from lock_test_table where id = 1 for update;

-- This query will block until Session 1 commits or rolls back
-- You should see it waiting in pg_stat_activity

-- ============================================
-- To release the lock, commit or rollback Session 1:
-- ============================================
-- commit;  -- or rollback;

-- ============================================
-- Alternative: Test with different lock types
-- ============================================

-- Test with table-level lock
-- SESSION 1:
-- begin;
-- lock table lock_test_table in exclusive mode;

-- SESSION 2:
-- begin;
-- select * from lock_test_table;  -- Will wait

-- Test with advisory lock
-- SESSION 1:
-- begin;
-- select pg_advisory_lock(12345);

-- SESSION 2:
-- begin;
-- select pg_advisory_lock(12345);  -- Will wait

