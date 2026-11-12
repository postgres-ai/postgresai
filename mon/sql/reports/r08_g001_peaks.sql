-- R8 G001 peaks (reporter L2819-2829, L2860-2861), all instant, subquery [7d:1m]
-- variant 'topk_app': topk(5, max_over_time((sum by (application_name) (pgwatch_pg_stat_activity_count{c,n}))[7d:1m]))
-- variant 'max':      max_over_time((sum(pgwatch_pg_stat_activity_count{c,n}))[7d:1m])
-- variant 'peak_at':  max_over_time(timestamp((sum(pgwatch_pg_stat_activity_count{c,n})) >= <top>)[7d:1m])
--                     timestamp() of an AGGREGATE is the step time g (not a raw sample time)
-- variant 'avg':      avg_over_time((sum(pgwatch_pg_stat_activity_count{c,n}))[7d:1m])
-- Params: variant, top (for peak_at: the exact float from 'max').
WITH raw AS (
  SELECT labels, t, value
  FROM prom_last('pgwatch_pg_stat_activity_count',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 60000), prom_sq_end(getvariable('end_ms'), 60000), 60000)
), win AS (SELECT * FROM raw WHERE t > getvariable('end_ms') - 7 * 86400000::BIGINT AND t <= getvariable('end_ms')),
by_app AS (SELECT coalesce(labels['application_name'], '') AS application_name, t, sum(value) AS v FROM win GROUP BY ALL),
total AS (SELECT t, sum(value) AS v FROM win GROUP BY t),
topk_app AS (
  SELECT nullif(app, '') AS application_name, max(v) AS value FROM (SELECT application_name AS app, v FROM by_app) GROUP BY app
  QUALIFY row_number() OVER (ORDER BY max(v) DESC, app) <= 5
)
SELECT application_name, value FROM topk_app WHERE getvariable('variant') = 'topk_app'
UNION ALL
SELECT NULL, max(v) FROM total WHERE getvariable('variant') = 'max' HAVING count(*) > 0
UNION ALL
SELECT NULL, max(t) / 1e3 FROM total WHERE getvariable('variant') = 'peak_at' AND v >= getvariable('top') HAVING count(*) > 0
UNION ALL
SELECT NULL, avg(v) FROM total WHERE getvariable('variant') = 'avg' HAVING count(*) > 0;
