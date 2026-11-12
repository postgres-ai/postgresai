-- R1 K003 hourly top-k (reporter L4985-4988)
-- PromQL: topk(3, (sum by (queryid) (increase(<metric>{cluster,node_name,datname}[1h]))))
--         query_range start=floor_hour(now)-23h end=floor_hour(now) step=1h
-- Params: cluster, node, db, metric (exec_time_total | plan_time_total), k, start_ms, end_ms, step_ms
-- One generic prom_topk call with p_by := ['queryid'] (sum by queryid, then a per-step top-k): the
-- Rust evaluator keeps one k-heap per step instead of materialising every (series, hour) row.
-- Ties at rank k: VM keeps an arbitrary member; here value, then queryid (deterministic).
-- (Proposed by the collector agent for the 192 MB reader on 14 d data; oracle: prom_topk_sql.)
SELECT labels['queryid'] AS queryid, t, value
FROM prom_topk(getvariable('k'), 'increase', getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 3600000, p_by := ['queryid']);
