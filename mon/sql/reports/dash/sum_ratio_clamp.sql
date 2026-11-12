-- clamp_max(sum(irate(<a>{m}[ri])) / (sum(irate(<a>{m}[ri])) + sum(irate(<b>{m}[ri]))), 1)
-- (Dashboard_5 3A, archive success rate). sum() of no series is no series, so a step needs both
-- metrics; 0/0 = NaN dropped. Params: metric, other, matchers, start_ms, end_ms, step_ms, rate_interval_ms.
WITH a AS (
  SELECT t, value FROM prom_agg('sum', 'irate', getvariable('metric'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), b AS (
  SELECT t, value FROM prom_agg('sum', 'irate', getvariable('other'), getvariable('matchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
)
-- (NaN first: DuckDB's least(NaN, 1) is 1, while clamp_max(NaN, 1) is NaN)
SELECT t, least(value, 1) AS value FROM (SELECT a.t, a.value / (a.value + b.value) AS value FROM a JOIN b USING (t))
WHERE NOT isnan(value);
