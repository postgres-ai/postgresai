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
-- the legend/hover fetches it once per queryid with d1_query_texts.sql.
-- top-k per step (as in PromQL range queries) via the generic prom_topk: the SQL macro
-- prom_topk_sql or pgai-mon's Rust prom_topk_rs (one k-heap per step), same ties (value,
-- queryid, label set).
-- Params: metric, per_call, top_n, db_re, rate_interval_ms, start_ms, end_ms, step_ms.
SELECT top.labels, top.t, top.value,
       left(regexp_replace(qt.query, '\s+', ' ', 'g'), 80) AS _displayname_long
FROM prom_topk(getvariable('top_n'), 'irate', getvariable('metric'),
       [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
       getvariable('start_ms'), getvariable('end_ms'), getvariable('step_ms'), getvariable('rate_interval_ms'),
       p_den_metric := CASE WHEN getvariable('per_call') THEN 'pgwatch_pg_stat_statements_calls' ELSE '' END) top
LEFT JOIN (SELECT queryid, dbname, arg_max(query, last_seen) AS query FROM query_texts GROUP BY ALL) qt
  ON qt.queryid = TRY_CAST(top.labels['queryid'] AS BIGINT) AND qt.dbname = top.labels['datname'];
