-- Panels 54 A-D, 55 B-D (op = 'and'):   <metric>{c,n} and on(cluster, node_name) (o1{c,n} and on(cluster, node_name) o2 ...)
-- Panel 55 E, 54 F-like (op = 'unless'): <metric>{c,n} unless on(cluster, node_name) (o1 and on(...) o2 ...)
-- `and`/`unless` on(cluster, node_name): keep/drop left points where ALL others have a value at t.
-- For 54 F the left side is the CPU total (see host_cpu.sql variant 'total'); metric = '__cpu_total__'.
WITH lft AS (
  SELECT labels, t, value FROM prom_last(getvariable('metric'), [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
), oth AS (            -- the 4 gate metrics, as constant arguments (no LATERAL: table functions need constants)
  SELECT 1 AS om, labels['cluster'] AS cluster, labels['node_name'] AS node_name, t
  FROM prom_last(getvariable('others')[1], [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
  UNION ALL
  SELECT 2 AS om, labels['cluster'] AS cluster, labels['node_name'] AS node_name, t
  FROM prom_last(getvariable('others')[2], [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
  UNION ALL
  SELECT 3 AS om, labels['cluster'] AS cluster, labels['node_name'] AS node_name, t
  FROM prom_last(getvariable('others')[3], [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
  UNION ALL
  SELECT 4 AS om, labels['cluster'] AS cluster, labels['node_name'] AS node_name, t
  FROM prom_last(getvariable('others')[4], [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
), all_present AS (
  SELECT cluster, node_name, t FROM oth GROUP BY ALL
  HAVING count(DISTINCT om) = len(getvariable('others'))
)
SELECT l.labels, l.t, l.value FROM lft l
WHERE (getvariable('op') = 'and') = EXISTS (
  SELECT 1 FROM all_present a
  WHERE a.cluster IS NOT DISTINCT FROM l.labels['cluster'] AND a.node_name IS NOT DISTINCT FROM l.labels['node_name'] AND a.t = l.t);
