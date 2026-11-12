-- R7 G001 p5 over a subquery (reporter L2893-2900)
-- PromQL (stand-in on present data):
--   quantile_over_time(0.05, (sum(pgwatch_pg_stat_activity_count{cluster,node_name}))[7d:1m])
-- PromQL (real, host_* absent in the reference; verified on the synthetic VM):
--   quantile_over_time(0.05, (100 * host_memory_available_bytes{..} / host_os_memory_total_bytes{..})[7d:1m])
-- Param: variant ('activity' | 'host_memory'). VM quantile = (n-1)*phi interpolation = quantile_cont.
WITH inner_activity AS (
  SELECT MAP {}::MAP(VARCHAR, VARCHAR) AS labels, t, sum(value) AS v
  FROM prom_last('pgwatch_pg_stat_activity_count',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 60000), prom_sq_end(getvariable('end_ms'), 60000), 60000)
  GROUP BY t
), inner_host AS (   -- one-to-one on all labels (__name__ dropped)
  SELECT a.labels, a.t, 100 * a.value / b.value AS v
  FROM prom_last('host_memory_available_bytes',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 60000), prom_sq_end(getvariable('end_ms'), 60000), 60000) a
  JOIN prom_last('host_os_memory_total_bytes',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 60000), prom_sq_end(getvariable('end_ms'), 60000), 60000) b
    ON a.labels = b.labels AND a.t = b.t
), inner_q AS (
  SELECT * FROM inner_activity WHERE getvariable('variant') = 'activity'
  UNION ALL SELECT * FROM inner_host WHERE getvariable('variant') = 'host_memory'
)
SELECT labels, quantile_cont(v, 0.05) AS value     -- the subquery's labels survive (unlike an aggregate)
FROM inner_q
WHERE t > getvariable('end_ms') - 7 * 86400000::BIGINT AND t <= getvariable('end_ms') AND NOT isnan(v)
GROUP BY labels;
