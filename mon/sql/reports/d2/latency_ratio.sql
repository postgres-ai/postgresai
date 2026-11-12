-- Panel 48 A/B: sum(irate(<metric>{sel}[$__rate_interval]))/1000 / sum(irate(pgwatch_pg_stat_statements_calls{sel}[$__rate_interval]))
WITH a AS (
  SELECT t, value AS v
  FROM prom_agg('sum', 'irate', getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  SELECT t, value AS v
  FROM prom_agg('sum', 'irate', 'pgwatch_pg_stat_statements_calls',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
)
SELECT t, value FROM (SELECT t, a.v / 1000 / b.v AS value FROM a JOIN b USING (t)) WHERE NOT isnan(value);
