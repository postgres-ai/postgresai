-- Dashboard_7 panel 2 Vacuum timeline: ../d2/vacuum_timeline.sql plus the panel's
-- schema_name=~"$schema_name", table_name=~"$table_name" matchers. The `or` chain over phases
-- 7..1 keeps, per (schema_name, table_name, vacuum_mode) and step, the highest phase present.
-- Params: cluster, node, db, schema_re, table_re, start_ms, end_ms, step_ms.
SELECT nullif(labels['schema_name'], '') AS schema_name, nullif(labels['table_name'], '') AS table_name,
       nullif(labels['vacuum_mode'], '') AS vacuum_mode, t, max(CAST(labels['phase'] AS INTEGER)) AS value
FROM prom_last('pgwatch_pg_vacuum_progress_index_vacuum_count',
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')],
        ['phase','=~','[1-7]'],['schema_name','=~',getvariable('schema_re')],['table_name','=~',getvariable('table_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
GROUP BY ALL;
