-- scale * <metric>{matchers} / clamp_min(<den_metric>{matchers}, den_min) (Dashboard_7 panel 92C):
-- one-to-one on the full label set; the result has the left side's labels.
-- Params: metric, den_metric, matchers, scale, den_min, start_ms, end_ms, step_ms.
WITH a AS (
  SELECT labels, t, value FROM prom_last(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
), b AS (
  SELECT labels, t, greatest(value, CAST(getvariable('den_min') AS DOUBLE)) AS value
  FROM prom_last(getvariable('den_metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
)
SELECT labels, t, value FROM (
  SELECT a.labels, a.t, CAST(getvariable('scale') AS DOUBLE) * a.value / b.value AS value
  FROM a JOIN b ON a.labels = b.labels AND a.t = b.t)
WHERE NOT isnan(value);
