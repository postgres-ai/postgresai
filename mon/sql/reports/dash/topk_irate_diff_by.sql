-- topk($top_n, sum by (<by>) (irate(<metric>{matchers}[$__rate_interval])
--                             - irate(<other>{matchers}[$__rate_interval])))   (Dashboard_8 10A:
-- non-HOT updates = n_tup_upd - n_tup_hot_upd). The `-` is one-to-one on the full label set (a
-- series counts at a step only when both sides have a value there); then the sum by, then a
-- per-step rank (NaN dropped; ties by the label set as text). prom_topk_expr (Rust; the oracle
-- is prom_topk_expr_sql).
-- Params: metric, other, matchers, by, top_n, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_topk_expr(getvariable('top_n'), 'irate',
       [{'m': getvariable('metric'), 'n': 1.0, 'd': 0.0}, {'m': getvariable('other'), 'n': -1.0, 'd': 0.0}],
       getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'),
       getvariable('rate_interval_ms'), p_by := CAST(getvariable('by') AS VARCHAR[]));
