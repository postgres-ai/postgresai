-- <metric>{matchers} <op> <other>{matchers}, op '-' or '/' (Dashboard_5 1A archive lag,
-- Dashboard_11 1A Boguk ratio): one-to-one on the full label set; the result has the label set.
-- x/0 = ±Inf is kept (as PromQL), 0/0 = NaN dropped (as VictoriaMetrics).
-- Params: op, metric, other, matchers, start_ms, end_ms, step_ms.
WITH a AS (
  SELECT labels, t, value FROM prom_last(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
), b AS (
  SELECT labels, t, value FROM prom_last(getvariable('other'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
)
SELECT labels, t, value FROM (
  SELECT a.labels, a.t, CASE getvariable('op') WHEN '-' THEN a.value - b.value WHEN '/' THEN a.value / b.value END AS value
  FROM a JOIN b ON a.labels = b.labels AND a.t = b.t)
WHERE NOT isnan(value);
