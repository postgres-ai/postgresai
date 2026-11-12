-- One query's pg_stat_statements rate (Dashboard_3 panels 2-17):
--   irate(<metric>{cluster, node_name, datname=~db_re, queryid="<query_id>"}[$__rate_interval])
-- per_call: the same divided by irate(pgwatch_pg_stat_statements_calls{same}[...]) (one-to-one
-- on all labels; 0/0 = NaN dropped, as VictoriaMetrics drops NaN points).
-- Params: metric, per_call, cluster, node, db_re, query_id, start_ms, end_ms, step_ms, rate_interval_ms.
WITH a AS (
  SELECT labels, t, value
  FROM prom_irate(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],
          ['datname','=~',getvariable('db_re')],['queryid','=',getvariable('query_id')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  SELECT labels, t, value
  FROM prom_irate('pgwatch_pg_stat_statements_calls',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],
          ['datname','=~',getvariable('db_re')],['queryid','=',getvariable('query_id')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
)
SELECT labels, t, value FROM a WHERE NOT getvariable('per_call')
UNION ALL
SELECT labels, t, value FROM (
  SELECT a.labels, a.t, a.value / b.value AS value FROM a JOIN b ON a.labels = b.labels AND a.t = b.t
  WHERE getvariable('per_call'))
WHERE NOT isnan(value);
