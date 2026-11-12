-- Panels 12, 47, 14 A/B, 49, 29: irate(<metric>{cluster, node_name, datname=~"$db_name"}[$__rate_interval]) per series
SELECT labels, t, value
FROM prom_irate(getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'));
