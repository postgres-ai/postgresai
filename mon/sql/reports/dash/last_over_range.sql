-- last_over_time(<metric>{matchers}[$__range]) (Dashboard_5 14A/B, Dashboard_6 251A/252A: instant
-- tables over the dashboard range). Params: metric, matchers, range_ms, start_ms, end_ms, step_ms.
SELECT labels, t, value
FROM prom_last_over_time(getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('range_ms'));
