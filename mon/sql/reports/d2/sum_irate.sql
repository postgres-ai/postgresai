-- Panels 7 A/B, 6 A, 10 A/B, 25 A:  sum(irate(<metric>{cluster, node_name, datname=~"$db_name"}[$__rate_interval])) / <divisor>
-- prom_agg: the sum is a per-step group sink (Rust) instead of GROUP BY over every (series, step) row.
SELECT t, value / getvariable('divisor') AS value
FROM prom_agg('sum', 'irate', getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'));
