-- Panel 4 A:  sum by (state) (pgwatch_pg_stat_activity_count{cluster, node_name, datname=~"$db_name"}) > 0
-- Panel 39 A: the same with state!="idle" (non_idle = true)
SELECT state, t, value FROM (
  SELECT nullif(coalesce(labels['state'], ''), '') AS state, t, sum(value) AS value
  FROM prom_last('pgwatch_pg_stat_activity_count',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]]
         || CASE WHEN getvariable('non_idle') THEN [['state','!=','idle']] ELSE [] END,
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
  GROUP BY ALL)
WHERE value > 0;
