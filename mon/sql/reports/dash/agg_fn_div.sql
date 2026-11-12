-- <op> by (<by>) (<fn>(<metric>{matchers}[$__rate_interval])) / <divisor> per step (Dashboard_6
-- 103A, 212A, 243A: ms -> s). agg_fn.sql with a divisor; the labels are the group's.
-- Params: op, fn, metric, matchers, by, divisor, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value / getvariable('divisor') AS value
FROM prom_agg(getvariable('op'), getvariable('fn'), getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_by := CAST(getvariable('by') AS VARCHAR[]));
