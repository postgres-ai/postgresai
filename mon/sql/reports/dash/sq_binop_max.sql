-- Dashboard_6 221B-D (one standby selected), a rollup over an expression (MetricsQL's implicit
-- subquery: `(expr)[d]` with no step uses the query step as the inner step):
--   clamp_min(max_over_time((<metric>{matchers} - <other>{matchers})[$__rate_interval]), 0) and on()
--     (count(count by (<cond_by>) (max_over_time(<cond_metric>{matchers}[$__range]))) == 1)
-- 1. The inner `a - b` (one-to-one on the full label set) at every point g of the subquery grid:
--    absolute multiples of the step from prom_sq_start to prom_sq_end (bare selectors via prom_last
--    at that step, VM's default rollup).
-- 2. max over the inner points in (t - $__rate_interval, t] at each outer step t; NaN skipped.
-- 3. clamp_min 0, and the `and on()` condition of replica_stage.sql.
-- Params: metric, other, matchers, cond_metric, cond_by, start_ms, end_ms, step_ms,
-- rate_interval_ms, range_ms.
WITH a AS (
  SELECT labels, t, value
  FROM prom_last(getvariable('metric'), getvariable('matchers'),
         prom_sq_start(getvariable('start_ms'), getvariable('rate_interval_ms'), getvariable('step_ms')),
         prom_sq_end(getvariable('end_ms'), getvariable('step_ms')), getvariable('step_ms'))
), b AS (
  SELECT labels, t, value
  FROM prom_last(getvariable('other'), getvariable('matchers'),
         prom_sq_start(getvariable('start_ms'), getvariable('rate_interval_ms'), getvariable('step_ms')),
         prom_sq_end(getvariable('end_ms'), getvariable('step_ms')), getvariable('step_ms'))
), d AS (
  SELECT a.labels, a.t AS g, a.value - b.value AS v
  FROM a JOIN b ON a.labels = b.labels AND a.t = b.t
  WHERE NOT isnan(a.value - b.value)
), o AS (
  SELECT unnest(generate_series(getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))) AS t
), r AS (
  SELECT d.labels, o.t, max(d.v) AS value
  FROM o JOIN d ON d.g > o.t - getvariable('rate_interval_ms') AND d.g <= o.t
  GROUP BY 1, 2
), c AS (
  SELECT t
  FROM prom_max_over_time(getvariable('cond_metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('range_ms'))
  GROUP BY t HAVING count(DISTINCT coalesce(labels[getvariable('cond_by')], '')) = 1
)
SELECT labels, t, greatest(value, 0) AS value FROM r JOIN c USING (t);
