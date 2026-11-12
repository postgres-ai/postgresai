-- CPU panels over host_* (absent in the reference stack; verified on the synthetic VM).
-- total = max by (cluster, node_name) (host_os_cpu_percent{c,n} or on(cluster, node_name)
--                 (host_cpu_utilization_percent{c,n} unless on(cluster, node_name) host_os_cpu_percent{c,n}))
-- variant 'other' (54 E): clamp_min(total - host_os_cpu_user_percent - ..._system_ - ..._iowait_ - ..._steal_, 0)
--                         (one-to-one on ALL labels: matches only if host series carry just cluster, node_name)
-- variant 'idle'  (54 G): clamp_min(100 - total, 0)
-- variant 'total_unless_split' (54 F): total unless on(cluster, node_name) (user and on(..) system and on(..) iowait and on(..) steal)
WITH p AS (SELECT * FROM prom_last('host_os_cpu_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
u AS (SELECT * FROM prom_last('host_cpu_utilization_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
total AS (
  SELECT labels['cluster'] AS cluster, labels['node_name'] AS node_name, t, max(value) AS v FROM (
    SELECT labels, t, value FROM p
    UNION ALL
    SELECT u.labels, u.t, u.value FROM u WHERE NOT EXISTS (
      SELECT 1 FROM p WHERE p.t = u.t AND p.labels['cluster'] IS NOT DISTINCT FROM u.labels['cluster']
                         AND p.labels['node_name'] IS NOT DISTINCT FROM u.labels['node_name']))
  GROUP BY ALL
),
total_l AS (SELECT map_from_entries(list_filter([{'k':'cluster','v':cluster},{'k':'node_name','v':node_name}], lambda e: e.v IS NOT NULL)) AS labels, t, v FROM total),
us AS (SELECT * FROM prom_last('host_os_cpu_user_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
sy AS (SELECT * FROM prom_last('host_os_cpu_system_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
io AS (SELECT * FROM prom_last('host_os_cpu_iowait_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
st AS (SELECT * FROM prom_last('host_os_cpu_steal_percent', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')]], getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))),
split AS (   -- user and on(c,n) system and on(c,n) iowait and on(c,n) steal: (cluster, node_name, t) where all 4 exist
  SELECT us.labels['cluster'] AS cluster, us.labels['node_name'] AS node_name, us.t FROM us
  WHERE EXISTS (SELECT 1 FROM sy WHERE sy.t = us.t AND sy.labels['cluster'] IS NOT DISTINCT FROM us.labels['cluster'] AND sy.labels['node_name'] IS NOT DISTINCT FROM us.labels['node_name'])
    AND EXISTS (SELECT 1 FROM io WHERE io.t = us.t AND io.labels['cluster'] IS NOT DISTINCT FROM us.labels['cluster'] AND io.labels['node_name'] IS NOT DISTINCT FROM us.labels['node_name'])
    AND EXISTS (SELECT 1 FROM st WHERE st.t = us.t AND st.labels['cluster'] IS NOT DISTINCT FROM us.labels['cluster'] AND st.labels['node_name'] IS NOT DISTINCT FROM us.labels['node_name'])
)
SELECT labels, t, greatest(total_l.v - us.value - sy.value - io.value - st.value, 0) AS value
FROM total_l JOIN us USING (labels, t) JOIN sy USING (labels, t) JOIN io USING (labels, t) JOIN st USING (labels, t)
WHERE getvariable('variant') = 'other'
UNION ALL
SELECT labels, t, greatest(100 - v, 0) FROM total_l WHERE getvariable('variant') = 'idle'
UNION ALL
SELECT labels, t, v FROM total_l
WHERE getvariable('variant') = 'total_unless_split'
  AND NOT EXISTS (SELECT 1 FROM split s WHERE s.t = total_l.t AND s.cluster IS NOT DISTINCT FROM total_l.labels['cluster']
                                         AND s.node_name IS NOT DISTINCT FROM total_l.labels['node_name']);
