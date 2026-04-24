/**
 * Shared test utilities for CLI tests.
 */

export interface MockClientOptions {
  /** Database name returned by current_database() queries (default: "testdb") */
  databaseName?: string;
  /** Version rows for pg_settings version query (default: PG 16.3) */
  versionRows?: any[];
  settingsRows?: any[];
  databaseSizesRows?: any[];
  dbStatsRows?: any[];
  connectionStatesRows?: any[];
  uptimeRows?: any[];
  invalidIndexesRows?: any[];
  unusedIndexesRows?: any[];
  redundantIndexesRows?: any[];
  tableBloatRows?: any[];
  indexBloatRows?: any[];
  vacuumStatsRows?: any[];
  deadlockStatsRows?: any[];
  pgStatStatementsExtensionRows?: any[];
  pgStatStatementsStatsRows?: any[];
  pgStatStatementsSampleRows?: any[];
  pgStatKcacheExtensionRows?: any[];
  pgStatKcacheStatsRows?: any[];
  pgStatKcacheSampleRows?: any[];
  sensitiveColumnsRows?: any[];
}

const DEFAULT_VERSION_ROWS = [
  { name: "server_version", setting: "16.3" },
  { name: "server_version_num", setting: "160003" },
];

const defaultSettingsRows = [
  { tag_setting_name: "shared_buffers", tag_setting_value: "128MB", tag_unit: "", tag_category: "Resource Usage / Memory", tag_vartype: "string", is_default: 1, setting_normalized: null, unit_normalized: null },
  { tag_setting_name: "work_mem", tag_setting_value: "4MB", tag_unit: "", tag_category: "Resource Usage / Memory", tag_vartype: "string", is_default: 1, setting_normalized: null, unit_normalized: null },
  { tag_setting_name: "autovacuum", tag_setting_value: "on", tag_unit: "", tag_category: "Autovacuum", tag_vartype: "bool", is_default: 1, setting_normalized: null, unit_normalized: null },
  { tag_setting_name: "pg_stat_statements.max", tag_setting_value: "5000", tag_unit: "", tag_category: "Custom", tag_vartype: "integer", is_default: 0, setting_normalized: null, unit_normalized: null },
];

/**
 * Create a mock PostgreSQL client for testing report generators.
 * Routes SQL queries to appropriate mock data based on query patterns.
 */
export function createMockClient(options: MockClientOptions = {}) {
  const {
    databaseName = "testdb",
    versionRows = DEFAULT_VERSION_ROWS,
    settingsRows = defaultSettingsRows,
    databaseSizesRows = [],
    dbStatsRows = [],
    connectionStatesRows = [],
    uptimeRows = [],
    invalidIndexesRows = [],
    unusedIndexesRows = [],
    redundantIndexesRows = [],
    tableBloatRows = [],
    indexBloatRows = [],
    vacuumStatsRows = [],
    deadlockStatsRows = [{ deadlocks: "0", conflicts: "0", stats_reset: null }],
    pgStatStatementsExtensionRows = [],
    pgStatStatementsStatsRows = [],
    pgStatStatementsSampleRows = [],
    pgStatKcacheExtensionRows = [],
    pgStatKcacheStatsRows = [],
    pgStatKcacheSampleRows = [],
    sensitiveColumnsRows = [],
  } = options;

  return {
    query: async (sql: string) => {
      // Version query (simple inline - used by getPostgresVersion)
      if (sql.includes("server_version") && sql.includes("server_version_num") && sql.includes("pg_settings") && !sql.includes("tag_setting_name")) {
        return { rows: versionRows };
      }
      // Settings metric query (from metrics.yml - has tag_setting_name, tag_setting_value)
      if (sql.includes("tag_setting_name") && sql.includes("tag_setting_value") && sql.includes("pg_settings")) {
        return { rows: settingsRows };
      }
      // Database sizes (simple inline - lists all databases)
      if (sql.includes("pg_database") && sql.includes("pg_database_size") && sql.includes("datistemplate")) {
        return { rows: databaseSizesRows };
      }
      // db_size metric (current database size from metrics.yml)
      if (sql.includes("pg_database_size(current_database())") && sql.includes("size_b")) {
        return { rows: [{ tag_datname: databaseName, size_b: "1073741824" }] };
      }
      // db_stats metric (from metrics.yml)
      if (sql.includes("pg_stat_database") && sql.includes("xact_commit") && sql.includes("pg_control_system")) {
        return { rows: dbStatsRows };
      }
      // Stats reset metric (from metrics.yml)
      if (sql.includes("stats_reset") && sql.includes("pg_stat_database") && sql.includes("seconds_since_reset")) {
        return { rows: [{ tag_database_name: databaseName, stats_reset_epoch: "1704067200", seconds_since_reset: "2592000" }] };
      }
      // Postmaster startup time (simple inline - used by getStatsReset)
      if (sql.includes("pg_postmaster_start_time") && sql.includes("postmaster_startup_epoch")) {
        return { rows: [{ postmaster_startup_epoch: "1704067200", postmaster_startup_time: "2024-01-01 00:00:00+00" }] };
      }
      // Connection states (simple inline)
      if (sql.includes("pg_stat_activity") && sql.includes("state") && sql.includes("group by")) {
        return { rows: connectionStatesRows };
      }
      // Uptime info (simple inline)
      if (sql.includes("pg_postmaster_start_time()") && sql.includes("uptime") && !sql.includes("postmaster_startup_epoch")) {
        return { rows: uptimeRows };
      }
      // Invalid indexes (H001) - from metrics.yml
      if (sql.includes("indisvalid = false") && sql.includes("fk_indexes")) {
        return { rows: invalidIndexesRows };
      }
      // Unused indexes (H002) - from metrics.yml
      if (sql.includes("Never Used Indexes") && sql.includes("idx_scan = 0")) {
        return { rows: unusedIndexesRows };
      }
      // Redundant indexes (H004) - from metrics.yml
      if (sql.includes("redundant_indexes_grouped") && sql.includes("columns like")) {
        return { rows: redundantIndexesRows };
      }
      // F004/F005: bloat metrics from metrics.yml
      if (sql.includes("tag_idxname") && sql.includes("bloat_size")) {
        return { rows: indexBloatRows };
      }
      if (sql.includes("tag_tblname") && sql.includes("bloat_size")) {
        return { rows: tableBloatRows };
      }
      // Vacuum stats used by F004/F005
      if (sql.includes("pg_stat_user_tables") && sql.includes("last_vacuum")) {
        return { rows: vacuumStatsRows };
      }
      // G003: Deadlock/conflict stats
      if (sql.includes("coalesce(sum(deadlocks)") && sql.includes("current_database()")) {
        return { rows: deadlockStatsRows };
      }
      // D004: pg_stat_statements extension check
      if (sql.includes("pg_extension") && sql.includes("pg_stat_statements")) {
        return { rows: pgStatStatementsExtensionRows };
      }
      // D004: pg_stat_statements aggregate and sample queries
      if (sql.includes("from pg_stat_statements") && sql.includes("count(*) as cnt")) {
        return { rows: pgStatStatementsStatsRows };
      }
      if (sql.includes("from pg_stat_statements s") && sql.includes("order by calls desc")) {
        return { rows: pgStatStatementsSampleRows };
      }
      // D004: pg_stat_kcache extension check
      if (sql.includes("pg_extension") && sql.includes("pg_stat_kcache")) {
        return { rows: pgStatKcacheExtensionRows };
      }
      // D004: pg_stat_kcache aggregate and sample queries
      if (sql.includes("from pg_stat_kcache") && sql.includes("count(*) as cnt")) {
        return { rows: pgStatKcacheStatsRows };
      }
      if (sql.includes("from pg_stat_kcache k") && sql.includes("order by")) {
        return { rows: pgStatKcacheSampleRows };
      }
      // G001: Memory settings query
      if (sql.includes("pg_size_bytes") && sql.includes("shared_buffers") && sql.includes("work_mem")) {
        return { rows: [{
          shared_buffers_bytes: "134217728",
          wal_buffers_bytes: "4194304",
          work_mem_bytes: "4194304",
          maintenance_work_mem_bytes: "67108864",
          effective_cache_size_bytes: "4294967296",
          max_connections: 100,
        }] };
      }
      // S001: Sensitive columns query (from metrics.yml)
      if (sql.includes("information_schema.columns") && sql.includes("tag_schema_name") && sql.includes("tag_column_name")) {
        return { rows: sensitiveColumnsRows };
      }
      throw new Error(`Unexpected query: ${sql}`);
    },
  };
}
