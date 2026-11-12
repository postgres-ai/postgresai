-- Panel 32: sum by (lockmode) (rate(pgwatch_locks_mode_count{cluster, node_name, datname=~"$db_name"}[$__rate_interval]) * 60)
-- (sum by of x * 60 = 60 * sum by of x; the multiplication moves outside the group sink)
SELECT nullif(coalesce(labels['lockmode'], ''), '') AS lockmode, t, value * 60 AS value
FROM prom_agg('sum', 'rate', 'pgwatch_locks_mode_count',
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_by := ['lockmode']);
