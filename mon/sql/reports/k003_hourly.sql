-- K003 hourly series as the reporter builds them (R1 + R2 + Python "other"), in ONE SQL pass,
-- with query text joined from query_texts (replaces the Postgres sink lookup L430-487).
-- Not diffed against a single PromQL (it is 3 PromQL calls + Python); diffed piecewise by r01/r02.
WITH inc AS (
  SELECT labels['queryid'] AS queryid, t, value
  FROM prom_increase(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 3600000)
), by_q AS (
  SELECT queryid, t, sum(value) AS value FROM inc GROUP BY ALL
), top AS (
  SELECT DISTINCT queryid FROM by_q WHERE NOT isnan(value)
  QUALIFY row_number() OVER (PARTITION BY t ORDER BY value DESC, queryid) <= getvariable('k')
), total AS (SELECT t, sum(value) AS total FROM by_q GROUP BY t)
SELECT prom_ts(b.t) AS hour_end, b.queryid, qt.query, b.value,
       total.total, greatest(0, total.total - sum(b.value) OVER (PARTITION BY b.t)) AS other
FROM by_q b JOIN top USING (queryid) JOIN total USING (t)
LEFT JOIN (SELECT queryid, any_value(query) AS query FROM query_texts WHERE dbname = getvariable('db') GROUP BY 1) qt
  ON qt.queryid = TRY_CAST(b.queryid AS BIGINT)
ORDER BY hour_end, value DESC;
