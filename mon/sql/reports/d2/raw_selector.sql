-- A bare selector `<metric>{matchers}` (VM default_rollup: last sample in (t - max(step, ~1.25 x scrape), t]).
SELECT labels, t, value
FROM prom_last(getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'));
