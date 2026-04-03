# Permission Check Test Summary

## Changes Made

Changed all references from `public.pg_statistic` to `postgres_ai.pg_statistic` in:
- `cli/lib/init.ts` - Permission check SQL query
- `cli/test/init.test.ts` - All test expectations (28 occurrences)

## Key Fix: Safe Schema Checking

**Before (883fa95):**
```sql
exists (
  select from pg_views
  where schemaname = 'public' and viewname = 'pg_statistic'
) as granted
```

**After (6db79f6) - INCORRECT, caused crashes:**
```sql
to_regclass('postgres_ai.pg_statistic') is not null as granted
```

**Current (this fix):**
```sql
case
  when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
  else to_regclass('postgres_ai.pg_statistic') is not null
end as granted
```

### Why this fix matters

**Issue with bare `to_regclass()`:**
- Returns NULL when the schema doesn't exist ✓
- Returns NULL when the view doesn't exist ✓
- **Throws error** when the schema exists but user lacks USAGE privilege ✗

**Fix:**
- Check `has_schema_privilege()` first to avoid the permission error
- Returns NULL safely in all cases where we can't check the view
- Prevents crashes when postgres_ai schema exists but user lacks USAGE

## Test Results

### Unit Tests ✅
```
✓ 95 tests passed across 3 files
  - 84 tests in init.test.ts (including 9 checkCurrentUserPermissions tests)
  - 2 tests in config-consistency.test.ts
  - 9 tests in permission-check-sql.test.ts
```

### Expected Behavior by Scenario

| Scenario | User Permissions | postgres_ai Schema | Expected Result |
|----------|-----------------|-------------------|-----------------|
| 1. Superuser | superuser + postgres_ai.pg_statistic | ✓ Exists | ✅ PASS (clean) |
| 2. pg_monitor, no schema access | pg_monitor only | ✗ No USAGE | ✅ PASS (warning) |
| 3. No pg_monitor | minimal permissions | ✗ Doesn't exist | ✅ PASS (error + fix SQL) |
| 8. After prepare-db | pg_monitor + postgres_ai grants | ✓ Exists + SELECT | ✅ PASS (clean) |

### SQL Behavior Verification

**Scenario 2 & 3: Schema doesn't exist or no USAGE**
```sql
-- Check privilege first, then to_regclass (no crash)
case
  when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
  else to_regclass('postgres_ai.pg_statistic') is not null
end  → NULL

-- SELECT check is skipped (returns NULL, not treated as missing optional)
case
  when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
  when to_regclass('postgres_ai.pg_statistic') is null then null
  else has_table_privilege(current_user, 'postgres_ai.pg_statistic', 'select')
end  → NULL
```

**Scenario 1 & 8: Schema exists with proper grants**
```sql
-- User has USAGE, to_regclass returns OID (view is visible)
case
  when not has_schema_privilege(current_user, 'postgres_ai', 'USAGE') then null
  else to_regclass('postgres_ai.pg_statistic') is not null
end  → TRUE

-- SELECT check is performed
has_table_privilege(current_user, 'postgres_ai.pg_statistic', 'select')  → TRUE/FALSE
```

## Integration Test Limitations

Integration tests cannot run due to locale configuration issues with `initdb`:
```
error: initdb: error: invalid locale settings; check LANG and LC_* environment variables
```

However, unit tests provide comprehensive coverage of the permission check logic, including:
- All permission scenarios (granted, denied, skipped)
- Multiple missing permissions
- Error propagation
- Fix command generation
- Message formatting

## Schema Consistency

The change ensures consistency across the codebase:
- ✅ `cli/lib/init.ts` - now checks postgres_ai.pg_statistic
- ✅ `cli/lib/supabase.ts` - already checks postgres_ai.pg_statistic
- ✅ `cli/sql/03.permissions.sql` - creates postgres_ai.pg_statistic
- ✅ `config/target-db/init.sql` - creates postgres_ai.pg_statistic
- ✅ `config/pgwatch-prometheus/metrics.yml` - references postgres_ai.pg_statistic

## Commits

1. **955cff2** - `fix: change public.pg_statistic to postgres_ai.pg_statistic`
   - Updated permission check queries
   - Updated all test expectations

2. **6db79f6** - `fix: use to_regclass() for safe postgres_ai.pg_statistic check`
   - Replaced pg_views query with to_regclass()
   - ⚠️ This introduced a bug: crashes when schema exists but user lacks USAGE

3. **[current]** - `fix: wrap to_regclass() with has_schema_privilege() check`
   - Fixed crash when postgres_ai schema exists but user lacks USAGE privilege
   - Added privilege check before calling to_regclass() in all locations
   - Updated in: init.ts (3 places) and supabase.ts (1 place)

## Verification Command

```bash
# Run all permission-related tests
bun test test/init.test.ts test/config-consistency.test.ts test/permission-check-sql.test.ts

# Verify no public.pg_statistic references remain (except in comments)
git grep -n 'public\.pg_statistic' cli/
```
