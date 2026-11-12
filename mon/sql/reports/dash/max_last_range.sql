-- max(last_over_time(<metric>{matchers}[$__range])) / <divisor> > 0 (Dashboard_6 243B).
-- Params: metric, matchers, divisor, range_ms, start_ms, end_ms, step_ms.
SELECT t, value FROM (
  SELECT t, max(value) / getvariable('divisor') AS value
  FROM prom_last_over_time(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('range_ms'))
  WHERE NOT isnan(value)
  GROUP BY t)
WHERE value > 0;
