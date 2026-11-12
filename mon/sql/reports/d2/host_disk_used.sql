-- Panel 56 A: host_disk_size_bytes{c,n} - host_disk_free_bytes{c,n} (one-to-one on all labels)
WITH a AS (SELECT * FROM prom_last('host_disk_size_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
b AS (SELECT * FROM prom_last('host_disk_free_bytes', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms')))
SELECT labels, t, a.value - b.value AS value FROM a JOIN b USING (labels, t);
