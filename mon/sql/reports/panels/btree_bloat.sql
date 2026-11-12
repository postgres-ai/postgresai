-- Dashboard_10 panel 24 (B-tree index bloat table), formerly the Flask backend's
-- /btree_bloat/csv: per index, the newest value of each pgwatch_pg_btree_bloat_* metric in the
-- day before the range end (Flask: last_over_time(...[1d]) at now). The panel's own
-- filterExpression keeps is_na < 1 and bloat_pct > 0.
-- Params: cluster, node, db_re, end_ms.
WITH v AS (
  SELECT 'real_size_mib' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_real_size_mib',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'extra_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_extra_size',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'extra_pct' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_extra_pct',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'fillfactor' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_fillfactor',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'bloat_size' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_bloat_size',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'bloat_pct' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_bloat_pct',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
  UNION ALL
  SELECT 'is_na' AS m, labels, value
  FROM prom_last_over_time('pgwatch_pg_btree_bloat_is_na',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, 86400000)
)
SELECT coalesce(labels['datname'], '') AS database, coalesce(labels['schemaname'], '') AS schemaname,
       coalesce(labels['tblname'], '') AS tblname, coalesce(labels['idxname'], '') AS idxname,
       max(value) FILTER (WHERE m = 'real_size_mib') AS real_size_mib,
       max(value) FILTER (WHERE m = 'extra_size') AS extra_size,
       max(value) FILTER (WHERE m = 'extra_pct') AS extra_pct,
       max(value) FILTER (WHERE m = 'fillfactor') AS fillfactor,
       max(value) FILTER (WHERE m = 'bloat_size') AS bloat_size,
       max(value) FILTER (WHERE m = 'bloat_pct') AS bloat_pct,
       CAST(max(value) FILTER (WHERE m = 'is_na') AS BIGINT) AS is_na
FROM v
GROUP BY ALL
ORDER BY bloat_size DESC NULLS LAST, database, schemaname, tblname, idxname
