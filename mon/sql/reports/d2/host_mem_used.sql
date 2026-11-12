-- Panel 55 A: (total - free - cached - buffers) or on(cluster, node_name) (total - available)
-- Subtractions are one-to-one on all labels; `or on(cluster, node_name)` adds right points only
-- where the left has no series for that (cluster, node_name) at t.
WITH tot AS (SELECT * FROM prom_last('host_os_memory_total_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
fr AS (SELECT * FROM prom_last('host_os_memory_free_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
ca AS (SELECT * FROM prom_last('host_os_memory_cached_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
bu AS (SELECT * FROM prom_last('host_os_memory_buffers_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
av AS (SELECT * FROM prom_last('host_memory_available_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
l AS (SELECT labels, t, tot.value - fr.value - ca.value - bu.value AS value
      FROM tot JOIN fr USING (labels, t) JOIN ca USING (labels, t) JOIN bu USING (labels, t)),
r AS (SELECT labels, t, tot.value - av.value AS value FROM tot JOIN av USING (labels, t))
SELECT * FROM l
UNION ALL
SELECT * FROM r WHERE NOT EXISTS (SELECT 1 FROM l WHERE l.t = r.t
  AND l.labels['cluster'] IS NOT DISTINCT FROM r.labels['cluster'] AND l.labels['node_name'] IS NOT DISTINCT FROM r.labels['node_name']);
