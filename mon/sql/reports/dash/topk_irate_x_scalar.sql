-- topk($top_n, irate(<metric>{matchers}[$__rate_interval])) * on() group_left()
--   last_over_time(<smetric>{smatchers}[<swindow>])     (Dashboard_8 33A, 34A, 36A, 38A;
-- Dashboard_10 25A, 26A: blocks x block_size). The rank is on the rate (prom_topk), then each kept
-- series is multiplied by the one right-hand series; at a step where the right side has more than
-- one series PromQL fails the query, and this returns no point for that step.
-- Params: metric, matchers, smetric, smatchers, swindow_ms, top_n, start_ms, end_ms, step_ms,
-- rate_interval_ms.
WITH a AS (
  SELECT labels, t, value
  FROM prom_topk(getvariable('top_n'), 'irate', getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  -- the right-hand series when it is the only one at t: a top 2 (prom_topk evaluates a 30 d
  -- grid in time chunks; prom_last_over_time opened every day file at once). A NaN right-hand
  -- value is dropped by the rank (it gave NaN points before).
  SELECT t, any_value(value) AS value
  FROM prom_topk(2, 'last_over_time', getvariable('smetric'), getvariable('smatchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('swindow_ms'))
  GROUP BY t HAVING count(*) = 1
)
SELECT a.labels, a.t, a.value * b.value AS value FROM a JOIN b USING (t);
