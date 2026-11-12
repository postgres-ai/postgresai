-- Dashboard_6 221A and 222A-C (one standby selected):
--   <f>(max_over_time(<metric>{matchers}[$__rate_interval])) and on()
--     (count(count by (<cond_by>) (max_over_time(<cond_metric>{matchers}[$__range]))) == 1)
-- f: clamp_min(x, 0) (clamp0) and / divisor. The `and on()` keeps every left series at the steps
-- where exactly one <cond_by> value has samples in the dashboard range.
-- Params: metric, matchers, divisor, clamp0, cond_metric, cond_by, start_ms, end_ms, step_ms,
-- rate_interval_ms, range_ms.
WITH a AS (
  SELECT labels, t, value
  FROM prom_max_over_time(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), c AS (
  SELECT t
  FROM prom_max_over_time(getvariable('cond_metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('range_ms'))
  GROUP BY t HAVING count(DISTINCT coalesce(labels[getvariable('cond_by')], '')) = 1
)
SELECT labels, t,
       (CASE WHEN getvariable('clamp0') THEN greatest(value, 0) ELSE value END) / getvariable('divisor') AS value
FROM a JOIN c USING (t);
