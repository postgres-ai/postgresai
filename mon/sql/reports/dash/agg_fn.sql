-- <op> by (<by>) (<fn>(<metric>{matchers}[$__rate_interval])) per step (Dashboard_5 7A,
-- Dashboard_9 21A-D and 37A-D): prom_agg (Rust; the oracle is prom_agg_sql).
-- op: sum, avg, count, min, max; fn: rate, irate, ...; by: label list ([] = one group).
-- Params: op, fn, metric, matchers, by, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_agg(getvariable('op'), getvariable('fn'), getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_by := CAST(getvariable('by') AS VARCHAR[]));
