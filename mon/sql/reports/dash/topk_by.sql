-- topk(<k>, sum by (<by>) (<fn>(<metric>{matchers}[$__rate_interval]))) (Dashboard_5 15A).
-- Params: k, fn, metric, matchers, by, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_topk(getvariable('k'), getvariable('fn'), getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_by := CAST(getvariable('by') AS VARCHAR[]));
