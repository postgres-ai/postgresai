-- H002 Unused indexes, the whole report as ONE SQL statement over the data-contract views.
-- Reproduces reporter/postgres_reports.py generate_h002_unused_indexes_report() (L1339-1496)
-- and validates against reporter/schemas/H002.schema.json. Output: one row, column `report` (JSON).
--
-- Params (getvariable): cluster, node, end_ms (report time T, epoch ms), contract_version.
-- Index definitions come from the `objects` view (kind = 'index'), which replaces the Postgres
-- sink lookup; the metric's index_definition label is the fallback, as in the reporter.
--
-- Intentional differences (documented in sql/README.md on branch spike-monitoring-v2-duckdb):
--  * "first result wins" picks in the reporter (uptime, stats_reset, idx_scan, server_version)
--    depend on VictoriaMetrics' result order; here: the lexicographically smallest label set.
--  * table_writes: the reporter keeps ONE series per (schema, table) when pgwatch's
--    value-dependent label table_size_cardinality_mb splits a table into several series (dict
--    overwrite, VM order); here they are summed (the increase over the whole window).
--  * table_writes calls prom_increase_sql directly: its window (the scan window in hours) is
--    computed from the data, and a Rust table function takes constant arguments only.
--  * the 35 x 1h readings of R4, the per-index N+1 idx_scan queries and the 3 increase() calls
--    are one pass each; no PromQL round trips.
WITH
p AS (SELECT getvariable('end_ms') AS t, getvariable('cluster') AS cluster, getvariable('node') AS node),
cn AS (SELECT [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]] AS m),
-- ---------------------------------------------------------------- databases (get_all_databases)
dbs AS (
  SELECT DISTINCT d AS datname FROM (
    SELECT labels['datname'] AS d FROM prom_last_over_time('pgwatch_pg_database_wraparound_age_datfrozenxid', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT labels['datname'] FROM prom_last_over_time('pgwatch_unused_indexes_index_size_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT labels['dbname'] FROM prom_last_over_time('pgwatch_redundant_indexes_index_size_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT labels['datname'] FROM prom_last_over_time('pgwatch_pg_btree_bloat_bloat_pct', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT labels['datname'] FROM prom_last_over_time('pgwatch_pg_stat_statements_calls', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT labels['datname'] FROM prom_last_over_time('pgwatch_wait_events_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]] || [['datname','=~','.+'],['datname','!=','server_process']], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000))
  WHERE coalesce(d, '') <> '' AND d NOT IN ('template0', 'template1', 'rdsadmin', 'azure_maintenance', 'cloudsqladmin')
),
db_size AS (
  SELECT labels['datname'] AS datname, arg_min(value, CAST(labels AS VARCHAR)) AS bytes
  FROM prom_last_over_time('pgwatch_db_size_size_b', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
  GROUP BY 1
),
uptime AS (
  SELECT arg_min(value, CAST(labels AS VARCHAR)) AS s
  FROM prom_last_over_time('pgwatch_db_stats_postmaster_uptime_s', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
),
startup AS (SELECT CASE WHEN s IS NOT NULL AND s <> 0 THEN getvariable('end_ms') / 1e3 - s END AS epoch FROM uptime),
-- stats_reset: labelled database_name by pgwatch; datname tried first, as the reporter does
stats_reset_all AS (
  SELECT coalesce(labels['datname'], '') AS by_datname, coalesce(labels['database_name'], '') AS by_dbname,
         CAST(labels AS VARCHAR) AS lk, value
  FROM prom_last_over_time('pgwatch_stats_reset_stats_reset_epoch', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
),
stats_reset AS (
  SELECT d.datname,
         coalesce((SELECT arg_min(value, lk) FROM stats_reset_all WHERE by_datname = d.datname),
                  (SELECT arg_min(value, lk) FROM stats_reset_all WHERE by_dbname = d.datname)) AS epoch
  FROM dbs d
),
-- ---------------------------------------------------------------- per-index series
idx_last AS (       -- every unused_indexes_* series at T, keyed by (db, schema, table, index)
  SELECT regexp_replace(metric_short, '^pgwatch_unused_indexes_', '') AS what,
         labels['datname'] AS datname,
         coalesce(labels['schema_name'], 'unknown') AS schema_name, coalesce(labels['table_name'], 'unknown') AS table_name,
         coalesce(labels['index_name'], 'unknown') AS index_name, labels, value
  FROM (
    SELECT 'pgwatch_unused_indexes_index_size_bytes' AS metric_short, * FROM prom_last_over_time('pgwatch_unused_indexes_index_size_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT 'pgwatch_unused_indexes_supports_fk', * FROM prom_last_over_time('pgwatch_unused_indexes_supports_fk', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT 'pgwatch_unused_indexes_idx_is_btree', * FROM prom_last_over_time('pgwatch_unused_indexes_idx_is_btree', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
    UNION ALL SELECT 'pgwatch_unused_indexes_idx_scan', * FROM prom_last_over_time('pgwatch_unused_indexes_idx_scan', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000))
),
-- min_over_time(timestamp(idx_scan)[35d:1h]): raw sample time of the last sample in each 1h step
first_zero AS (
  SELECT labels['datname'] AS datname,
         coalesce(labels['schema_name'], 'unknown') AS schema_name, coalesce(labels['table_name'], 'unknown') AS table_name,
         coalesce(labels['index_name'], 'unknown') AS index_name, min(value) AS epoch
  FROM prom_timestamp('pgwatch_unused_indexes_idx_scan', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         prom_sq_start(getvariable('end_ms'), 35 * 86400000::BIGINT, 3600000), prom_sq_end(getvariable('end_ms'), 3600000), 3600000)
  WHERE t > getvariable('end_ms') - 35 * 86400000::BIGINT AND t <= getvariable('end_ms')
  GROUP BY ALL
),
defs AS (SELECT dbname, name, any_value(definition) AS definition FROM objects WHERE kind = 'index' AND valid_to IS NULL GROUP BY ALL),
idx AS (
  SELECT s.datname, s.schema_name, s.table_name, s.index_name,
         coalesce(s.labels['reason'], 'Unknown') AS reason,
         s.value AS index_size_bytes,
         coalesce((SELECT arg_min(value, CAST(labels AS VARCHAR)) FROM idx_last x WHERE x.what = 'idx_scan'
                     AND x.datname = s.datname AND x.schema_name = s.schema_name AND x.table_name = s.table_name
                     AND x.index_name = s.index_name), 0) AS idx_scan,
         coalesce(d.definition, nullif(s.labels['index_definition'], ''), 'Definition not available') AS index_definition,
         (SELECT any_value(value) > 0 FROM idx_last x WHERE x.what = 'idx_is_btree' AND x.datname = s.datname
            AND x.schema_name = s.schema_name AND x.table_name = s.table_name AND x.index_name = s.index_name) AS idx_is_btree,
         (SELECT any_value(value) > 0 FROM idx_last x WHERE x.what = 'supports_fk' AND x.datname = s.datname
            AND x.schema_name = s.schema_name AND x.table_name = s.table_name AND x.index_name = s.index_name) AS supports_fk,
         fz.epoch AS zero_epoch
  FROM idx_last s
  LEFT JOIN defs d ON d.dbname = s.datname AND d.name = s.index_name
  LEFT JOIN first_zero fz USING (datname, schema_name, table_name, index_name)
  WHERE s.what = 'index_size_bytes'
    AND s.schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')          -- #345
    AND NOT regexp_matches(s.schema_name, '^pg_(toast_)?temp_')
),
-- ---------------------------------------------------------------- scan window and table writes
win AS (            -- min over ALL first-zero readings of the db (the reporter does not filter system schemas here)
  SELECT datname, min(epoch) AS start_epoch FROM first_zero GROUP BY datname
),
span AS (SELECT datname, start_epoch, greatest(1, CAST(floor((getvariable('end_ms') / 1e3 - start_epoch) / 3600) AS BIGINT)) AS hours FROM win),
writes_raw AS (
  SELECT 'upd' AS col, * FROM prom_increase_sql('pgwatch_table_stats_n_tup_upd', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000,
                                     (SELECT max(hours) FROM span) * 3600000)
  UNION ALL SELECT 'hot', * FROM prom_increase_sql('pgwatch_table_stats_n_tup_hot_upd', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000,
                                     (SELECT max(hours) FROM span) * 3600000)
  UNION ALL SELECT 'ins', * FROM prom_increase_sql('pgwatch_table_stats_n_tup_ins', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('end_ms'), getvariable('end_ms'), 300000,
                                     (SELECT max(hours) FROM span) * 3600000)
),
writes AS (         -- table_stats labels are quote_ident()ed, unused_indexes labels are raw
  SELECT col, labels['datname'] AS datname,
         CASE WHEN regexp_full_match(coalesce(labels['schema'], ''), '".*"') THEN replace(labels['schema'][2:-2], '""', '"') ELSE coalesce(labels['schema'], '') END AS schema_name,
         CASE WHEN regexp_full_match(coalesce(labels['table_name'], ''), '".*"') THEN replace(labels['table_name'][2:-2], '""', '"') ELSE coalesce(labels['table_name'], '') END AS table_name,
         sum(value) AS v
  FROM writes_raw GROUP BY ALL
),
tw AS (
  SELECT k.datname, k.schema_name, k.table_name,
         max(v) FILTER (WHERE col = 'upd') AS upd, max(v) FILTER (WHERE col = 'hot') AS hot,
         max(v) FILTER (WHERE col = 'ins') AS ins, bool_or(col = 'upd') AS has_upd, bool_or(col = 'hot') AS has_hot,
         bool_or(col = 'ins') AS has_ins
  FROM (SELECT DISTINCT datname, schema_name, table_name FROM idx) k
  JOIN writes w USING (datname, schema_name, table_name)
  GROUP BY ALL
),
tw_json AS (
  SELECT tw.datname, to_json(list(json_object(
           'schema_name', schema_name, 'table_name', table_name,
           'n_tup_upd', CASE WHEN has_upd THEN CAST(round_even(upd, 0) AS BIGINT) END,
           'n_tup_hot_upd', CASE WHEN has_hot THEN CAST(round_even(hot, 0) AS BIGINT) END,
           'n_tup_ins', CASE WHEN has_ins THEN CAST(round_even(ins, 0) AS BIGINT) END,
           'hot_share_pct', CASE WHEN has_upd AND upd <> 0 AND has_hot THEN round_even(100.0 * hot / upd, 1) END,
           'window', span.hours || 'h') ORDER BY schema_name, table_name)) AS j
  FROM tw JOIN span USING (datname)
  WHERE has_upd OR has_ins
  GROUP BY tw.datname
),
-- ---------------------------------------------------------------- JSON assembly
idx_json AS (
  SELECT datname, sum(index_size_bytes) AS total, count(*) AS n,
         to_json(list(json_merge_patch(json_object(
           'schema_name', schema_name, 'table_name', table_name, 'index_name', index_name,
           'index_definition', index_definition, 'reason', reason, 'idx_scan', idx_scan,
           'index_size_bytes', index_size_bytes, 'idx_is_btree', idx_is_btree, 'supports_fk', supports_fk,
           'index_size_pretty', fmt_bytes(index_size_bytes)),
           CASE WHEN zero_epoch IS NOT NULL THEN json_object('zero_scans_since_epoch', zero_epoch,
                     'zero_scans_since_time', py_utc_z(zero_epoch))
                ELSE json_object() END)
           ORDER BY index_size_bytes DESC, schema_name, table_name, index_name)) AS j
  FROM idx GROUP BY datname
),
db_json AS (   -- json_merge_patch drops null members of the patch, so every nullable member is
               -- built in the target object; the patch only removes the two keys when absent
  SELECT i.datname, CASE WHEN sp.datname IS NULL
                         THEN json_merge_patch(full_j, '{"scan_window": null, "table_writes": null}')
                         ELSE full_j END AS j
  FROM (
  SELECT i.*, sp.datname AS sp_datname, json_object(
           'unused_indexes', i.j, 'total_count', i.n, 'total_size_bytes', i.total,
           'total_size_pretty', fmt_bytes(i.total),
           'database_size_bytes', coalesce(ds.bytes, 0), 'database_size_pretty', fmt_bytes(coalesce(ds.bytes, 0)),
           'stats_reset', json_object(
              'stats_reset_epoch', sr.epoch,
              'stats_reset_time', CASE WHEN sr.epoch IS NOT NULL AND sr.epoch <> 0 THEN py_isoformat(sr.epoch) END,
              'days_since_reset', CASE WHEN sr.epoch IS NOT NULL AND sr.epoch <> 0
                                       THEN CAST(floor((getvariable('end_ms') / 1e3 - sr.epoch) / 86400) AS BIGINT) END,
              'postmaster_startup_epoch', (SELECT epoch FROM startup),
              'postmaster_startup_time', CASE WHEN (SELECT epoch FROM startup) IS NOT NULL THEN py_isoformat((SELECT epoch FROM startup)) END),
           'scan_window', json_object(
                 'start_epoch', sp.start_epoch, 'start_time', py_utc_z(sp.start_epoch),
                 'end_epoch', getvariable('end_ms') / 1e3,
                 'end_time', py_utc_z(getvariable('end_ms') / 1e3),
                 'days', round_even((getvariable('end_ms') / 1e3 - sp.start_epoch) / 86400, 1),
                 'lookback', '35d', 'reading_step', '1h',
                 'restart_in_window', (SELECT epoch FROM startup) > sp.start_epoch,
                 'reset_in_window', CASE WHEN sr.epoch IS NOT NULL THEN sr.epoch > sp.start_epoch END),
           'table_writes', coalesce(tw.j, json('[]'))) AS full_j
  FROM idx_json i
  JOIN dbs USING (datname)
  LEFT JOIN db_size ds USING (datname)
  LEFT JOIN stats_reset sr USING (datname)
  LEFT JOIN span sp USING (datname)
  LEFT JOIN tw_json tw USING (datname)
  ) i
  LEFT JOIN span sp USING (datname)
),
version AS (
  SELECT arg_min(labels['setting_value'], CAST(labels AS VARCHAR)) FILTER (WHERE labels['setting_name'] = 'server_version') AS v,
         arg_min(labels['setting_value'], CAST(labels AS VARCHAR)) FILTER (WHERE labels['setting_name'] = 'server_version_num') AS num
  FROM prom_last_over_time('pgwatch_settings_configured',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]] || [['setting_name','=~','server_version|server_version_num']],
         getvariable('end_ms'), getvariable('end_ms'), 300000, 10800000)
),
pv AS (
  SELECT json_object('version', coalesce(v, 'Unknown'), 'server_version_num', coalesce(num, 'Unknown'),
           'server_major_ver', CASE WHEN v IS NULL THEN 'Unknown' ELSE coalesce(nullif(split_part(split_part(v, ' ', 1), '.', 1), ''), 'Unknown') END,
           'server_minor_ver', CASE WHEN v IS NULL OR split_part(split_part(v, ' ', 1), '.', 1) = '' THEN 'Unknown'
                                    WHEN strpos(split_part(v, ' ', 1), '.') = 0 THEN '0'
                                    ELSE substr(split_part(v, ' ', 1), strpos(split_part(v, ' ', 1), '.') + 1) END) AS j
  FROM version
)
SELECT json_object(
  'contract_version', getvariable('contract_version'), 'version', NULL, 'build_ts', NULL,
  'generation_mode', 'full', 'checkId', 'H002', 'checkTitle', 'Unused indexes',
  'timestamptz', py_isoformat(getvariable('end_ms') / 1e3) || '+00:00',
  'nodes', json_object('primary', getvariable('node'), 'standbys', json('[]')),
  'results', json_object(getvariable('node'), json_object(
      'data', coalesce((SELECT json_group_object(datname, j) FROM db_json), json('{}')),
      'postgres_version', (SELECT j FROM pv)))) AS report;
