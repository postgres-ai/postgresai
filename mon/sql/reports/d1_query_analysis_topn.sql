-- Dashboard_2_Aggregated_query_analysis.json (census "D1"): all 23 PromQL panels, one shape.
--
-- PromQL (per second, per_call = false):
--   (topk($top_n, irate(<metric>{cluster,node_name,datname=~"$db_name"}[$__rate_interval]))
--      * on(queryid) group_left(displayname, ...) (topk by (queryid) (1, tlast_over_time(pgwatch_query_info[7d])) > bool 0))
--   or (topk($top_n, irate(...)) unless on(queryid) (topk by (queryid) (1, tlast_over_time(pgwatch_query_info[7d])) > bool 0))
-- PromQL (per call, per_call = true): the same with irate(<metric>) / irate(pgwatch_pg_stat_statements_calls)
--
-- v2: the group_left(displayname...) join against pgwatch_query_info (Flask-produced) becomes a
-- LEFT JOIN to query_texts; the `or ... unless` fallback disappears (a LEFT JOIN keeps rows
-- without text). query_texts holds one row per stored text, so the join reads the newest text of
-- each (queryid, dbname): a queryid whose text changed must not split into two series. Display columns are prefixed with '_' so the diff harness ignores them.
-- The full query text is not on the points (it cost 30-40 MiB of the 192 MB reader at 24h/30s):
-- the legend/hover fetches it once per queryid.
-- top-k per step (as in PromQL range queries) via pgai-mon's Rust prom_topk_rs (one k-heap per
-- step; the macro prom_topk and its SQL oracle prom_topk_sql return the same rows), same ties
-- (value, queryid, label set).
-- Keyed shape (as dash/ash_by.sql; compile.py wraps cases with keyed=True): the points r(k, t,
-- value) carry only the series id; names(k, labels, _displayname_long) is built once per series,
-- so the labels are cast and the text is cleaned once per series, not once per point (#516: the
-- per-point regexp_replace was 64 % of a cold 24 h load).
-- The text: query_text_names, the newest text of each (queryid, dbname) by first appearance; a
-- refresh of last_seen cannot change it, so a refresh-only text snapshot keeps cached panels.
-- Params: metric, per_call, top_n, db_re, rate_interval_ms, start_ms, end_ms, step_ms.
WITH r AS MATERIALIZED (
  SELECT series_id AS k, t, value, labels AS lj
  FROM prom_topk_rs(CAST(getvariable('top_n') AS BIGINT), 'irate', getvariable('metric'),
         CAST(to_json([['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]]) AS VARCHAR),
         CAST(getvariable('start_ms') AS BIGINT), CAST(getvariable('end_ms') AS BIGINT), CAST(getvariable('step_ms') AS BIGINT),
         CAST(getvariable('rate_interval_ms') AS BIGINT),
         CASE WHEN getvariable('per_call') THEN 'pgwatch_pg_stat_statements_calls' ELSE '' END, 'false', '[]')
), keys AS (
  SELECT k, CAST(CAST(any_value(lj) AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels FROM r GROUP BY k
), names AS MATERIALIZED (
  SELECT keys.k, keys.labels, left(regexp_replace(qt.query, '\s+', ' ', 'g'), 80) AS _displayname_long
  FROM keys
  LEFT JOIN query_text_names qt
    ON qt.queryid = TRY_CAST(keys.labels['queryid'] AS BIGINT) AND qt.dbname = keys.labels['datname']
)
SELECT labels, t, value, _displayname_long FROM r JOIN names USING (k)
