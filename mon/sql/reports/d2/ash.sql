-- Panel 38 Active session history (A: db; B: server_process non-Activity; C: server_process Activity)
-- sum by (wait_event_type) (
--   (sum by (wait_event_type, instance) (sum_over_time(pgwatch_wait_events_total{sel}[$__interval])) default 0)
--   / on (instance) group_left() (sum by (instance) (sum_over_time(up{job="pgwatch-prometheus"}[$__interval])) > 0))
-- `default 0` (MetricsQL): every (wait_event_type, instance) series that has a value anywhere in
-- the range gets 0 at its empty steps. Division by the number of scrapes in the interval.
-- range-wide: `default 0` picks its series from the whole range; the oracle must not chunk this (diff.chunk_safe).
WITH w AS (
  SELECT coalesce(labels['wait_event_type'], '') AS wet, coalesce(labels['instance'], '') AS instance, t, sum(value) AS v
  FROM prom_sum_over_time('pgwatch_wait_events_total',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]]
         || CASE getvariable('variant')
              WHEN 'a' THEN [['datname','=~',getvariable('db_re')]]
              WHEN 'b' THEN [['datname','=','server_process'],['wait_event_type','!=','Activity']]
              ELSE [['datname','=','server_process'],['wait_event_type','=','Activity']] END,
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('interval_ms'))
  GROUP BY ALL
), filled AS (       -- default 0
  SELECT k.wet, k.instance, g.t, coalesce(w.v, 0) AS v
  FROM (SELECT DISTINCT wet, instance FROM w) k
  CROSS JOIN (SELECT unnest(range(getvariable('start_ms'), getvariable('end_ms') + 1, getvariable('step_ms'))) AS t) g
  LEFT JOIN w USING (wet, instance, t)
), up AS (
  SELECT coalesce(labels['instance'], '') AS instance, t, sum(value) AS n
  FROM prom_sum_over_time('up', [['job','=','pgwatch-prometheus']],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('interval_ms'))
  GROUP BY ALL HAVING sum(value) > 0
)
SELECT nullif(wet, '') AS wait_event_type, t, sum(v / n) AS value
FROM filled JOIN up USING (instance, t)
GROUP BY ALL;
