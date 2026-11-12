-- R10 N001 wait events per hour (reporter L4331-4339)
-- PromQL: sum by (wait_event_type, wait_event, query_id) (sum_over_time(pgwatch_wait_events_total{c,n,d}[1h]))
--         query_range 24h step 1h
SELECT labels['wait_event_type'] AS wait_event_type, labels['wait_event'] AS wait_event,
       labels['query_id'] AS query_id, t, sum(value) AS value
FROM prom_sum_over_time('pgwatch_wait_events_total',
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 3600000)
GROUP BY ALL;
