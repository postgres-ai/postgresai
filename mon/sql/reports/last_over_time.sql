-- Generic instant/range `last_over_time(<metric>{matchers}[window])` used by R3 (H002 inputs:
-- db_size_size_b, postmaster_uptime_s, stats_reset_epoch, unused_indexes_*), R6 (A003
-- settings), D1 panels 14C/46A/51A/52A. Params: metric, matchers (LIST of [label, op, value]),
-- window_ms, start_ms, end_ms, step_ms.
SELECT labels, t, value
FROM prom_last_over_time(getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('window_ms'));
