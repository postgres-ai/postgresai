-- R4 H002 first reading in the lookback (reporter L1410-1411)
-- PromQL: min_over_time(timestamp(pgwatch_unused_indexes_idx_scan{cluster,node_name,datname})[35d:1h])  (instant)
-- The subquery evaluates timestamp(x) as a range query on the 1h grid
-- [align_down(T-35d-1h-5m), align_up(T+1h)]: timestamp() of a raw selector is the RAW sample time
-- of the last sample in (g-1h, g]; then min over grid points g in (T-35d, T].
WITH g AS (
  SELECT series_id, labels, t AS g, value
  FROM prom_timestamp('pgwatch_unused_indexes_idx_scan',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
         prom_sq_start(getvariable('end_ms'), 35 * 86400000::BIGINT, 3600000),
         prom_sq_end(getvariable('end_ms'), 3600000), 3600000)
)
SELECT any_value(labels) AS labels, min(value) AS value
FROM g
WHERE g > getvariable('end_ms') - 35 * 86400000::BIGINT AND g <= getvariable('end_ms')
GROUP BY series_id;
