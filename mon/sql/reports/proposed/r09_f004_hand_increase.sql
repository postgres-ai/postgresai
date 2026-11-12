-- R9 F004 hand increase: the ported query with the grouping labels joined once from a small series
-- lookup instead of carrying the label MAP on every (series, step) row. Same result.
WITH sel AS (      -- the series' grouping labels, once (no MAP per row)
  SELECT series_id, labels['schema'] AS schema, labels['table_name'] AS table_name
  FROM series
  WHERE metric = getvariable('metric') AND prom_match(labels,
        [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]])
), a AS (
  SELECT series_id, t, value
  FROM prom_last(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 300000), prom_sq_end(getvariable('end_ms'), 300000), 300000)
), b AS (        -- offset 5m: evaluate on the grid shifted back 5m, then shift t forward
  SELECT series_id, t + 300000 AS t, value
  FROM prom_last_over_time(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
         prom_sq_start(getvariable('end_ms'), 7 * 86400000::BIGINT, 300000) - 300000,
         prom_sq_end(getvariable('end_ms'), 300000) - 300000, 300000, 7 * 86400000::BIGINT)
), amax AS (
  SELECT sel.schema, sel.table_name, a.t, max(a.value) AS v FROM a JOIN sel USING (series_id) GROUP BY ALL
), bmax AS (
  SELECT sel.schema, sel.table_name, b.t, max(b.value) AS v
  FROM b JOIN a USING (series_id, t) JOIN sel USING (series_id)   -- `and x`: keep b only where x has a value at g
  GROUP BY ALL
), d AS (
  SELECT schema, table_name, t, greatest(amax.v - bmax.v, 0) AS v
  FROM amax JOIN bmax USING (schema, table_name, t)
)
SELECT schema, table_name, sum(v) AS value
FROM d
WHERE t > getvariable('end_ms') - 7 * 86400000::BIGINT AND t <= getvariable('end_ms') AND NOT isnan(v)
GROUP BY ALL;
