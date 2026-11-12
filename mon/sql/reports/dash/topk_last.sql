-- topk($top_n, <metric>{matchers}): per step, the top_n series by their bare-selector value
-- (Dashboard_8 1A, 23A, 25A, 28A, 2A, 8A; Dashboard_10 2A, 22A, 23A). prom_topk with fn 'last'
-- (Rust, a k-heap per step; the oracle is prom_topk_sql): NaN dropped, ties broken by the label
-- set as text. prom_last + a per-step rank in SQL held every (series, step) row with its labels
-- (155-178 MB of reader RSS at 24 h).
-- Params: metric, matchers, top_n, start_ms, end_ms, step_ms.
SELECT labels, t, value
FROM prom_topk(getvariable('top_n'), 'last', getvariable('metric'), getvariable('matchers'),
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 0);
