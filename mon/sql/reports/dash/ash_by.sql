-- Active session history grouped by `by` (Dashboard_4 panels 2/3, Dashboard_3 panel 19):
-- sum by (<by>) (
--   (sum by (<by>, instance) (sum_over_time(pgwatch_wait_events_total{sel}[$__interval])) default 0)
--   / on (instance) group_left() (sum by (instance) (sum_over_time(up{job="pgwatch-prometheus"}[$__interval])) > 0))
-- sel = cluster, node_name, datname=~db_re plus, by `filter`:
--   'wet'    wait_event_type=~wet_re                         (D4 panel 2)
--   'wet_we' wait_event_type=~wet_re, wait_event=~we_re      (D4 panel 3)
--   'query'  query_id="<query_id>", query_id!=""             (D3 panel 19)
-- Same `default 0` and `up` division as ../d2/ash.sql (which is this with by = [wait_event_type]).
-- Params: by (label list), filter, cluster, node, db_re, wet_re, we_re, query_id, start_ms,
-- end_ms, step_ms, interval_ms.
-- range-wide: `default 0` picks its series from the whole range; the oracle must not chunk this (diff.chunk_safe).
WITH w AS (
  SELECT list_transform(getvariable('by'), lambda k: coalesce(labels[k], '')) AS k,
         coalesce(labels['instance'], '') AS instance, t, sum(value) AS v
  FROM prom_sum_over_time('pgwatch_wait_events_total',
         [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]]
         || CASE getvariable('filter')
              WHEN 'wet' THEN [['wait_event_type','=~',getvariable('wet_re')]]
              WHEN 'wet_we' THEN [['wait_event_type','=~',getvariable('wet_re')],['wait_event','=~',getvariable('we_re')]]
              WHEN 'query' THEN [['query_id','=',getvariable('query_id')],['query_id','!=','']]
              ELSE CAST([] AS VARCHAR[][]) END,
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('interval_ms'))
  GROUP BY ALL
), filled AS (       -- default 0
  SELECT d.k, d.instance, g.t, coalesce(w.v, 0) AS v
  FROM (SELECT DISTINCT k, instance FROM w) d
  CROSS JOIN (SELECT unnest(range(getvariable('start_ms'), getvariable('end_ms') + 1, getvariable('step_ms'))) AS t) g
  LEFT JOIN w ON w.k = d.k AND w.instance = d.instance AND w.t = g.t
), up AS (
  SELECT coalesce(labels['instance'], '') AS instance, t, sum(value) AS n
  FROM prom_sum_over_time('up', [['job','=','pgwatch-prometheus']],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('interval_ms'))
  GROUP BY ALL HAVING sum(value) > 0
), r AS (
  SELECT k, t, sum(v / n) AS value FROM filled JOIN up USING (instance, t) GROUP BY ALL
)
SELECT map_from_entries(list_filter(
         list_transform(range(1, len(k) + 1), lambda i: {'k': getvariable('by')[i], 'v': k[i]}),
         lambda e: e.v <> '')) AS labels, t, value
FROM r;
