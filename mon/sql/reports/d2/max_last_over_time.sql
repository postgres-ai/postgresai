-- Panels 34 B/C, 35 B/C: max(last_over_time(<metric>{cluster, node_name}[5m])) [> 0]
SELECT t, value FROM (
  SELECT t, max(value) AS value
  FROM prom_last_over_time(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 300000)
  GROUP BY t)
WHERE NOT getvariable('gt0') OR value > 0;
