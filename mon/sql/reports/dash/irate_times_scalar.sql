-- irate(<metric>{matchers}[$__rate_interval]) * on() group_left()
--   last_over_time(<smetric>{smatchers}[<swindow>])          (Dashboard_9 44B-E/45B-E, Dashboard_11 4A/B:
-- bytes = blocks x block_size). on(): the right side must be one series; at a step where it has
-- more, PromQL fails the query ("many-to-many"), and this returns no point for that step.
-- The result keeps the left side's labels.
-- Params: metric, matchers, smetric, smatchers, swindow_ms, start_ms, end_ms, step_ms, rate_interval_ms.
WITH a AS (
  SELECT labels, t, value
  FROM prom_irate(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  SELECT t, any_value(value) AS value
  FROM prom_last_over_time(getvariable('smetric'), getvariable('smatchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('swindow_ms'))
  GROUP BY t HAVING count(*) = 1
)
SELECT a.labels, a.t, a.value * b.value AS value FROM a JOIN b USING (t);
