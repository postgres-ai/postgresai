-- Dashboard_11 1C: the index_rebuild_scale_factor that applies to one index, by priority:
--   max by () (lot(x{s,i,d}) * 8 or lot(x{s,i,d="*"}) * 7 or lot(x{s,i="*",d}) * 6 or lot(x{s="*",i,d}) * 5
--              or lot(x{s="*",i="*",d}) * 4 or lot(x{s,i="*",d="*"}) * 3 or lot(x{s="*",i,d="*"}) * 2
--              or lot(x{s="*",i="*",d="*"}) * 1)
--   / max by () (8 * (lot(...) * 0 + 1) or ... the same selectors)
-- lot = last_over_time(...[1w]), x = pgwatch_pg_index_pilot_config_config_value_normalized
-- {config_key="index_rebuild_scale_factor"}; s = schemaname=~schema_re, i = indexrelname=~index_re,
-- d = datname=~datname_re. `or` keeps the left side's series for a label set, so a series that
-- several selectors match counts once, at its highest priority.
-- Params: schema_re, index_re, datname_re, start_ms, end_ms, step_ms.
WITH x AS (
  SELECT labels, t, value
  FROM prom_last_over_time('pgwatch_pg_index_pilot_config_config_value_normalized',
         [['config_key','=','index_rebuild_scale_factor']],
         getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), 604800000)
), m AS (
  SELECT t, value,
         regexp_full_match(coalesce(labels['schemaname'], ''), getvariable('schema_re')) AS s, coalesce(labels['schemaname'], '') = '*' AS s_all,
         regexp_full_match(coalesce(labels['indexrelname'], ''), getvariable('index_re')) AS i, coalesce(labels['indexrelname'], '') = '*' AS i_all,
         regexp_full_match(coalesce(labels['datname'], ''), getvariable('datname_re')) AS d, coalesce(labels['datname'], '') = '*' AS d_all
  FROM x
), p AS (
  SELECT t, value,
         CASE WHEN s AND i AND d THEN 8 WHEN s AND i AND d_all THEN 7 WHEN s AND i_all AND d THEN 6
              WHEN s_all AND i AND d THEN 5 WHEN s_all AND i_all AND d THEN 4 WHEN s AND i_all AND d_all THEN 3
              WHEN s_all AND i AND d_all THEN 2 WHEN s_all AND i_all AND d_all THEN 1 END AS prio
  FROM m
)
SELECT t, value FROM (SELECT t, max(value * prio) / max(prio) AS value FROM p WHERE prio IS NOT NULL GROUP BY t)
WHERE NOT isnan(value);
