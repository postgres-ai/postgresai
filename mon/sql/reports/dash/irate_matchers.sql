-- irate(<metric>{matchers}[$__rate_interval]) per series (Dashboard_14 panels 2-32): the
-- ../d2/irate_series.sql shape with the matchers as a parameter.
-- Params: metric, matchers, start_ms, end_ms, step_ms, rate_interval_ms.
SELECT labels, t, value
FROM prom_irate(getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'));
