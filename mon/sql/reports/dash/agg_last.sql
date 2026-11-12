-- <op> by (<by>) (<metric>{matchers}): an aggregation over an instant-vector selector per step
-- (Dashboard_13 panels 2/3/5/6). op: sum, avg, count, min, max; by: label list ([] = one group).
-- NaN samples are skipped, as prom_agg does. (`by` goes through list_transform so an empty list,
-- which binds as VARCHAR[][], still types as VARCHAR[].)
-- Params: op, metric, matchers, by, start_ms, end_ms, step_ms.
SELECT prom_by(labels, list_transform(getvariable('by'), lambda k: CAST(k AS VARCHAR))) AS labels, t,
       CASE getvariable('op') WHEN 'sum' THEN sum(value) WHEN 'avg' THEN avg(value)
            WHEN 'count' THEN CAST(count(value) AS DOUBLE) WHEN 'min' THEN min(value) WHEN 'max' THEN max(value) END AS value
FROM prom_last(getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'))
WHERE NOT isnan(value)
GROUP BY 1, 2;
