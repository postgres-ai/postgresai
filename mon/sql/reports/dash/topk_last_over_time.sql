-- topk($top_n, last_over_time(<metric>{matchers}[window])) (Dashboard_7 panels 54A, 55A, 91A).
-- Params: metric, matchers, window_ms, top_n, start_ms, end_ms, step_ms.
SELECT labels, t, value
FROM prom_topk(getvariable('top_n'), 'last_over_time', getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('window_ms'));
