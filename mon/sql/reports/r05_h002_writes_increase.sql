-- R5 H002 writes on candidate tables (reporter L1309)
-- PromQL: increase(pgwatch_table_stats_<column>{cluster,node_name,datname}[<N>h])   (instant)
-- Params: metric (n_tup_upd | n_tup_hot_upd | n_tup_ins), window_ms = N h.
SELECT labels, value
FROM prom_increase(getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=',getvariable('db')]],
       getvariable('end_ms'), getvariable('end_ms'), 300000, getvariable('window_ms'));
