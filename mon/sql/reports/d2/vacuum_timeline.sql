-- Panel 42: group by (schema_name, table_name, vacuum_mode) (x{..., phase="7"}) * 0 + 7 or ... phase="1") * 0 + 1
-- `or` keeps the left (higher phase) on a label-set collision: max(phase) per label set.
SELECT nullif(labels['schema_name'], '') AS schema_name, nullif(labels['table_name'], '') AS table_name,
       nullif(labels['vacuum_mode'], '') AS vacuum_mode, t, max(CAST(labels['phase'] AS INTEGER)) AS value
FROM prom_last('pgwatch_pg_vacuum_progress_index_vacuum_count',
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')],
        ['phase','=~','[1-7]']],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
GROUP BY ALL;
