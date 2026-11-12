-- Dashboard_6 233A, an explicit subquery: minutes per hour above the replay-lag threshold,
--   sum_over_time(((max by (application_name) (<metric>{matchers})) > bool ($lag_alert_s * 1000))
--                 [<sq_window>:<sq_step>])
-- 1. At every point g of the [1h:1m] grid (absolute multiples of sq_step from prom_sq_start to
--    prom_sq_end): max by application_name of the bare selector (prom_last at sq_step), then
--    1 if above the threshold else 0 (`> bool` keeps the series).
-- 2. The sum of the inner points in (t - sq_window, t] at each outer step t.
-- Params: metric, matchers, lag_alert_s, sq_window_ms, sq_step_ms, start_ms, end_ms, step_ms.
WITH d AS (
  SELECT prom_by(labels, ['application_name']) AS labels, t AS g,
         CASE WHEN max(value) > getvariable('lag_alert_s') * 1000 THEN 1.0 ELSE 0.0 END AS v
  FROM prom_last(getvariable('metric'), getvariable('matchers'),
         prom_sq_start(getvariable('start_ms'), getvariable('sq_window_ms'), getvariable('sq_step_ms')),
         prom_sq_end(getvariable('end_ms'), getvariable('sq_step_ms')), getvariable('sq_step_ms'))
  WHERE NOT isnan(value)
  GROUP BY 1, 2
), o AS (
  SELECT unnest(generate_series(getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))) AS t
)
SELECT d.labels, o.t, sum(d.v) AS value
FROM o JOIN d ON d.g > o.t - getvariable('sq_window_ms') AND d.g <= o.t
GROUP BY 1, 2;
