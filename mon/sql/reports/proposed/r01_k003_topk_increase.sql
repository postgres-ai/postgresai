-- R1 K003 top-k on the Rust evaluator: topk(k, sum by (queryid) (increase(...[1h]))) as one prom_topk call
-- with p_by (Rust sum by labels + per-step k-heap). Same output and tie order as r01_k003_topk_increase.sql.
SELECT labels['queryid'] AS queryid, t, value
FROM prom_topk(getvariable('k'), 'increase', getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 3600000, p_by := ['queryid']);
