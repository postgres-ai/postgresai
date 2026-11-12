-- R2 K003 hourly total and per-query series for the top-k union (reporter L4996-5020)
-- PromQL (mode=total):  sum(increase(<metric>{cluster,node_name,datname}[1h]))
-- PromQL (mode=union):  sum by (queryid) (increase(<metric>{..., queryid=~"^(?:q1|q2|...)$"}[1h]))
-- Params: cluster, node, db, metric, mode ('total'|'union'), qid_re (regex of the R1 union)
-- In v2 the union needs no regex round trip: see r01+r02 combined in k003_hourly.sql.
WITH inc AS (
  SELECT labels['queryid'] AS queryid, t, value
  FROM prom_increase(getvariable('metric'),
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]]
         || CASE WHEN getvariable('mode') = 'union' THEN [['queryid','=~',getvariable('qid_re')]] ELSE [] END,
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 3600000)
)
SELECT CASE WHEN getvariable('mode') = 'union' THEN queryid END AS queryid, t, sum(value) AS value
FROM inc
GROUP BY 1, t;
