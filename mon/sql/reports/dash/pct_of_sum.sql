-- 100 * <metric>{matchers} / (<metric>{matchers} + <other>{matchers}) (Dashboard_14 panel 4A,
-- buffer hit ratio): one-to-one on the full label set; 0/0 = NaN dropped (VictoriaMetrics drops
-- NaN points).
-- Params: metric, other, matchers, start_ms, end_ms, step_ms.
WITH a AS (
  SELECT labels, t, value FROM prom_last(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
), b AS (
  SELECT labels, t, value FROM prom_last(getvariable('other'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
)
SELECT labels, t, value FROM (
  SELECT a.labels, a.t, 100 * a.value / (a.value + b.value) AS value
  FROM a JOIN b ON a.labels = b.labels AND a.t = b.t)
WHERE NOT isnan(value);
