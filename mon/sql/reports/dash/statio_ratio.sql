-- Dashboard_9 panel 46 (shared block hit ratio of one table), A: num = 'hit', B: num = 'read':
--   (Σ irate(<x>_blks_<num>)) / (Σ irate(<x>_blks_read) + Σ irate(<x>_blks_hit)),  x in heap, idx, toast, tidx
-- over pgwatch_pg_statio_all_tables_*. Every `+` and `/` is one-to-one on the full label set, so
-- a label set counts at a step only when all eight series have a value there. 0/0 = NaN dropped.
-- Params: num, matchers, start_ms, end_ms, step_ms, rate_interval_ms.
WITH x AS (
  SELECT 'heap_hit' AS k, labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_heap_blks_hit', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'idx_hit', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_idx_blks_hit', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'toast_hit', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_toast_blks_hit', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'tidx_hit', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_tidx_blks_hit', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'heap_read', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_heap_blks_read', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'idx_read', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_idx_blks_read', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'toast_read', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_toast_blks_read', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
  UNION ALL SELECT 'tidx_read', labels, t, value FROM prom_irate('pgwatch_pg_statio_all_tables_tidx_blks_read', getvariable('matchers'), getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'))
), g AS (
  SELECT labels, t,
         sum(value) FILTER (WHERE k LIKE '%_hit') AS h, sum(value) FILTER (WHERE k LIKE '%_read') AS r
  FROM x GROUP BY labels, t HAVING count(DISTINCT k) = 8
)
SELECT labels, t, value FROM (
  SELECT labels, t, (CASE getvariable('num') WHEN 'hit' THEN h ELSE r END) / (r + h) AS value FROM g)
WHERE NOT isnan(value);
