-- topk($top_n, [sum by (<by>)] (<fn>(<metric>{matchers}[$__rate_interval]))) (Dashboard_8 7A, 24A,
-- 26A, 29A, 32A, 39A, 40A, 41A, and with by: 3A, 5A, 6A; Dashboard_10 27A, 28A): prom_topk
-- (Rust; the oracle is prom_topk_sql). by = [] ranks the series themselves.
-- Params: fn, metric, matchers, by, top_n, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_topk(getvariable('top_n'), getvariable('fn'), getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_by := CAST(getvariable('by') AS VARCHAR[]));
