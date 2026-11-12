-- Dashboard_6 231B, a rollup over an aggregation (implicit subquery at the query step):
--   max_over_time((max(<metric>{matchers}))[$__rate_interval]) / <divisor>
-- The inner max over every series at each point g of the subquery grid (prom_last at the step,
-- NaN skipped), then the max over the inner points in (t - $__rate_interval, t] at each outer t.
-- The result has no labels.
-- Params: metric, matchers, divisor, start_ms, end_ms, step_ms, rate_interval_ms.
WITH d AS (
  SELECT t AS g, max(value) AS v
  FROM prom_last(getvariable('metric'), getvariable('matchers'),
         prom_sq_start(getvariable('start_ms'), getvariable('rate_interval_ms'), getvariable('step_ms')),
         prom_sq_end(getvariable('end_ms'), getvariable('step_ms')), getvariable('step_ms'))
  WHERE NOT isnan(value)
  GROUP BY t
), o AS (
  SELECT unnest(generate_series(getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))) AS t
)
SELECT MAP {}::MAP(VARCHAR, VARCHAR) AS labels, o.t, max(d.v) / getvariable('divisor') AS value
FROM o JOIN d ON d.g > o.t - getvariable('rate_interval_ms') AND d.g <= o.t
GROUP BY o.t;
