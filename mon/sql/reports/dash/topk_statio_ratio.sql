-- topk($top_n, (Σ irate(<x>_blks_<num>)) / (Σ irate(<x>_blks_read) + Σ irate(<x>_blks_hit))),
-- x in heap, idx, toast, tidx, over pgwatch_pg_statio_all_tables_* (Dashboard_8 42A: num = hit,
-- 43A: num = read). statio_ratio.sql (Dashboard_9 46) over every matching table, then a per-step
-- rank: a table counts only when all eight series have a value; 0/0 = NaN dropped before ranking.
-- prom_topk_expr (Rust, one scan of the eight metrics; the oracle is prom_topk_expr_sql): the
-- numerator takes the <num> terms (n = 1), the denominator all eight (d = 1).
-- Params: num, matchers, top_n, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_topk_expr(getvariable('top_n'), 'irate',
       [{'m': 'pgwatch_pg_statio_all_tables_heap_blks_hit', 'n': CAST(getvariable('num') = 'hit' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_heap_blks_read', 'n': CAST(getvariable('num') = 'read' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_idx_blks_hit', 'n': CAST(getvariable('num') = 'hit' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_idx_blks_read', 'n': CAST(getvariable('num') = 'read' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_toast_blks_hit', 'n': CAST(getvariable('num') = 'hit' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_toast_blks_read', 'n': CAST(getvariable('num') = 'read' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_tidx_blks_hit', 'n': CAST(getvariable('num') = 'hit' AS DOUBLE), 'd': 1.0},
        {'m': 'pgwatch_pg_statio_all_tables_tidx_blks_read', 'n': CAST(getvariable('num') = 'read' AS DOUBLE), 'd': 1.0}],
       getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'),
       getvariable('rate_interval_ms'));
