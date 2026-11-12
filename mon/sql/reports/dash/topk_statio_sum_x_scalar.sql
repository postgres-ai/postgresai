-- topk($top_n, irate(heap_blks_<num>) + irate(idx_blks_<num>) + irate(toast_blks_<num>)
--              + irate(tidx_blks_<num>)) * on() group_left() last_over_time(<smetric>{smatchers}[<swindow>])
-- over pgwatch_pg_statio_all_tables_* (Dashboard_8 31A: num = hit, 37A: num = read). Every `+` is
-- one-to-one on the full label set, so a table counts at a step only when all four series have a
-- value there (a table without TOAST never does). Rank first (prom_topk_expr: Rust, one scan of
-- the four metrics; the oracle is prom_topk_expr_sql), then the block-size multiply (one
-- right-hand series, as in irate_times_scalar.sql).
-- Params: num, matchers, smetric, smatchers, swindow_ms, top_n, start_ms, end_ms, step_ms,
-- rate_interval_ms.
WITH top AS (
  SELECT labels, t, value
  FROM prom_topk_expr(getvariable('top_n'), 'irate',
         [{'m': 'pgwatch_pg_statio_all_tables_heap_blks_' || getvariable('num'), 'n': 1.0, 'd': 0.0},
          {'m': 'pgwatch_pg_statio_all_tables_idx_blks_' || getvariable('num'), 'n': 1.0, 'd': 0.0},
          {'m': 'pgwatch_pg_statio_all_tables_toast_blks_' || getvariable('num'), 'n': 1.0, 'd': 0.0},
          {'m': 'pgwatch_pg_statio_all_tables_tidx_blks_' || getvariable('num'), 'n': 1.0, 'd': 0.0}],
         getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'),
         getvariable('rate_interval_ms'))
), b AS (
  -- the right-hand series when it is the only one at t: a top 2 (prom_topk evaluates a 30 d
  -- grid in time chunks; prom_last_over_time opened every day file at once). A NaN right-hand
  -- value is dropped by the rank (it gave NaN points before).
  SELECT t, any_value(value) AS value
  FROM prom_topk(2, 'last_over_time', getvariable('smetric'), getvariable('smatchers'),
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('swindow_ms'))
  GROUP BY t HAVING count(*) = 1
)
SELECT top.labels, top.t, top.value * b.value AS value FROM top JOIN b USING (t);
