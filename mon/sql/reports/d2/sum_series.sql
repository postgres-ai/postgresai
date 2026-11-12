-- Panel 44 A-E: sum(<metric>{datname=~"$db_name", cluster, node_name})
SELECT t, sum(value) AS value
FROM prom_last(getvariable('metric'),
       [['datname','=~',getvariable('db_re')],['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
GROUP BY t;
