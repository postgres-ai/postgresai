-- Dashboard_8 panel 45 (table stats table), formerly the Flask backend's /table_info/csv
-- (instant mode: the panel passes no time range): per table, the newest value of each metric
-- in the day before the range end (Flask: last_over_time(...[1d]) at now). Counters are the
-- cumulative values, as there. Tables are keyed by (datname, schema, table); the schema label is
-- `schemaname` or `schema` and the table label `relname` or `table_name`, depending on the
-- metric. Differences from Flask: the schema filter applies to both schema labels (Flask
-- matched `schemaname` only, so the table_size_detailed and table_stats columns were empty
-- whenever a schema was selected), and index relations (relkind i) of
-- pg_class_total_relation_size_bytes are not listed as tables.
-- Params: cluster, node, db_re, schema_re, end_ms.
WITH v AS (
  SELECT 'total_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_class_total_relation_size_bytes',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'table_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_size_detailed_table_main_size_b',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'index_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_size_detailed_table_indexes_size_b',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'toast_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_size_detailed_total_toast_size_b',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'seq_scan' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_stat_all_tables_seq_scan',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'idx_scan' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_stat_all_tables_idx_scan',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'n_tup_ins' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_stats_n_tup_ins',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'n_tup_upd' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_stats_n_tup_upd',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'n_tup_del' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_stats_n_tup_del',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'n_tup_hot_upd' AS m, labels, value
  FROM prom_last_over_time('pgwatch_table_stats_n_tup_hot_upd',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'heap_blks_read' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_statio_all_tables_heap_blks_read',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'heap_blks_hit' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_statio_all_tables_heap_blks_hit',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'idx_blks_read' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_statio_all_tables_idx_blks_read',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'idx_blks_hit' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_statio_all_tables_idx_blks_hit',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
), k AS (
  SELECT m, value, coalesce(labels['datname'], '') AS database,
         coalesce(labels['schemaname'], labels['schema'], '') AS schema,
         coalesce(labels['relname'], labels['table_name'], labels['tblname'], '') AS table_name
  FROM v
  WHERE coalesce(labels['relkind'], '') <> 'i'
)
SELECT schema, table_name,
       max(value) FILTER (WHERE m = 'total_size') AS total_size,
       max(value) FILTER (WHERE m = 'table_size') AS table_size,
       max(value) FILTER (WHERE m = 'index_size') AS index_size,
       max(value) FILTER (WHERE m = 'toast_size') AS toast_size,
       max(value) FILTER (WHERE m = 'seq_scan') AS seq_scan,
       max(value) FILTER (WHERE m = 'idx_scan') AS idx_scan,
       max(value) FILTER (WHERE m = 'n_tup_ins') AS n_tup_ins,
       max(value) FILTER (WHERE m = 'n_tup_upd') AS n_tup_upd,
       max(value) FILTER (WHERE m = 'n_tup_del') AS n_tup_del,
       max(value) FILTER (WHERE m = 'n_tup_hot_upd') AS n_tup_hot_upd,
       max(value) FILTER (WHERE m = 'heap_blks_read') AS heap_blks_read,
       max(value) FILTER (WHERE m = 'heap_blks_hit') AS heap_blks_hit,
       max(value) FILTER (WHERE m = 'idx_blks_read') AS idx_blks_read,
       max(value) FILTER (WHERE m = 'idx_blks_hit') AS idx_blks_hit
FROM k
WHERE regexp_full_match(schema, getvariable('schema_re'))
GROUP BY database, schema, table_name
ORDER BY total_size DESC NULLS LAST, database, schema, table_name
