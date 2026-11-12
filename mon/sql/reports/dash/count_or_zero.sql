-- count(<metric>{matchers}) or vector(0) (Dashboard_13 panel 1): the number of series present at
-- each step, 0 at steps with none.
-- Params: metric, matchers, start_ms, end_ms, step_ms.
WITH c AS (
  SELECT t, CAST(count(value) AS DOUBLE) AS n
  FROM prom_last(getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
  GROUP BY t
)
SELECT g.t, coalesce(c.n, 0) AS value
FROM (SELECT unnest(range(getvariable('start_ms'), getvariable('end_ms') + 1, getvariable('step_ms'))) AS t) g
LEFT JOIN c USING (t);
