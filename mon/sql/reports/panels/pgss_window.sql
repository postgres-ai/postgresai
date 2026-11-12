-- Dashboard_2 panel 20 (pg_stat_statements over the dashboard range, one row per statement),
-- formerly the Flask backend's /pgss_metrics/csv. For each statement series, the counter value
-- nearest the range start and nearest the range end (the newest sample within 1 minute either
-- side of each, as Flask's +-1 minute lookups), and from them per column: the difference
-- (end - start; a statement with no sample near the start counts from 0, as there; one with
-- none near the end is left out, where Flask counted the end as 0 and showed negative values;
-- an end value below the start value is a reset (pg_stat_statements_reset, eviction) and counts
-- from 0, where Flask showed the negative difference), per second
-- (over the actual time between the two `calls` samples, else the range) and per call. Bytes
-- become 8 KiB blocks. Rows are sorted by exec_time descending.
-- query_text: the statement text from query_texts, whitespace-normalized; with legend_label
-- containing 'raw' cut at 150 characters (147 + '...'), else shortened the way Flask's
-- smart_truncate_query does (comments dropped; "select ... from <tables>", "with <ctes> select
-- ... from <tables>", "insert into <t> ...", "update <t> ...", "delete from <t> ..."). The main
-- statement after a WITH is the text after the last ') select', where Flask counts parentheses.
-- Both edges come from one two-point range evaluation per metric (step = end - start).
-- Params: cluster, node, db_re, start_ms, end_ms, legend_label.
WITH v AS (
  SELECT m, labels, CASE WHEN t = getvariable('start_ms') + 60000 THEN 'a' ELSE 'b' END AS edge, value FROM (
  SELECT 'calls' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_calls', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'exec_time' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_exec_time_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'plan_time' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_plan_time_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'rows' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_rows', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'shared_blks_hit' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_shared_bytes_hit_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'shared_blks_read' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_shared_bytes_read_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'shared_blks_dirtied' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_shared_bytes_dirtied_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'shared_blks_written' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_shared_bytes_written_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'blk_read_time' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_block_read_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'blk_write_time' AS m, labels, t, value
  FROM prom_last_over_time('pgwatch_pg_stat_statements_block_write_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  UNION ALL
  SELECT 'ts' AS m, labels, t, value
  FROM prom_tlast_over_time('pgwatch_pg_stat_statements_calls', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('start_ms') + 60000, getvariable('end_ms') + 60000,
         greatest(getvariable('end_ms') - getvariable('start_ms'), 1), 120000)
  )
), e AS (
  SELECT coalesce(labels['datname'], '') AS datname, coalesce(labels['queryid'], '') AS queryid,
         coalesce(labels['user'], '') AS usr, coalesce(labels['instance'], '') AS inst,
         count(*) FILTER (WHERE edge = 'b') AS n_end,
         max(value) FILTER (WHERE edge = 'a' AND m = 'calls') AS a_calls, max(value) FILTER (WHERE edge = 'b' AND m = 'calls') AS b_calls,
         max(value) FILTER (WHERE edge = 'a' AND m = 'exec_time') AS a_exec_time, max(value) FILTER (WHERE edge = 'b' AND m = 'exec_time') AS b_exec_time,
         max(value) FILTER (WHERE edge = 'a' AND m = 'plan_time') AS a_plan_time, max(value) FILTER (WHERE edge = 'b' AND m = 'plan_time') AS b_plan_time,
         max(value) FILTER (WHERE edge = 'a' AND m = 'rows') AS a_rows, max(value) FILTER (WHERE edge = 'b' AND m = 'rows') AS b_rows,
         max(value) FILTER (WHERE edge = 'a' AND m = 'shared_blks_hit') AS a_shared_blks_hit, max(value) FILTER (WHERE edge = 'b' AND m = 'shared_blks_hit') AS b_shared_blks_hit,
         max(value) FILTER (WHERE edge = 'a' AND m = 'shared_blks_read') AS a_shared_blks_read, max(value) FILTER (WHERE edge = 'b' AND m = 'shared_blks_read') AS b_shared_blks_read,
         max(value) FILTER (WHERE edge = 'a' AND m = 'shared_blks_dirtied') AS a_shared_blks_dirtied, max(value) FILTER (WHERE edge = 'b' AND m = 'shared_blks_dirtied') AS b_shared_blks_dirtied,
         max(value) FILTER (WHERE edge = 'a' AND m = 'shared_blks_written') AS a_shared_blks_written, max(value) FILTER (WHERE edge = 'b' AND m = 'shared_blks_written') AS b_shared_blks_written,
         max(value) FILTER (WHERE edge = 'a' AND m = 'blk_read_time') AS a_blk_read_time, max(value) FILTER (WHERE edge = 'b' AND m = 'blk_read_time') AS b_blk_read_time,
         max(value) FILTER (WHERE edge = 'a' AND m = 'blk_write_time') AS a_blk_write_time, max(value) FILTER (WHERE edge = 'b' AND m = 'blk_write_time') AS b_blk_write_time,
         max(value) FILTER (WHERE edge = 'a' AND m = 'ts') AS a_ts, max(value) FILTER (WHERE edge = 'b' AND m = 'ts') AS b_ts
  FROM v
  GROUP BY ALL
), d AS (
  SELECT datname, queryid, usr, inst,
         coalesce(b_ts - a_ts, (getvariable('end_ms') - getvariable('start_ms')) / 1000.0) AS duration_seconds,
         CASE WHEN coalesce(b_calls, 0) >= coalesce(a_calls, 0) THEN coalesce(b_calls, 0) - coalesce(a_calls, 0) ELSE b_calls END AS calls,
         CASE WHEN coalesce(b_exec_time, 0) >= coalesce(a_exec_time, 0) THEN coalesce(b_exec_time, 0) - coalesce(a_exec_time, 0) ELSE b_exec_time END AS exec_time,
         CASE WHEN coalesce(b_plan_time, 0) >= coalesce(a_plan_time, 0) THEN coalesce(b_plan_time, 0) - coalesce(a_plan_time, 0) ELSE b_plan_time END AS plan_time,
         CASE WHEN coalesce(b_rows, 0) >= coalesce(a_rows, 0) THEN coalesce(b_rows, 0) - coalesce(a_rows, 0) ELSE b_rows END AS rows,
         CASE WHEN coalesce(b_shared_blks_hit, 0) >= coalesce(a_shared_blks_hit, 0) THEN coalesce(b_shared_blks_hit, 0) - coalesce(a_shared_blks_hit, 0) ELSE b_shared_blks_hit END / 8192 AS shared_blks_hit,
         CASE WHEN coalesce(b_shared_blks_read, 0) >= coalesce(a_shared_blks_read, 0) THEN coalesce(b_shared_blks_read, 0) - coalesce(a_shared_blks_read, 0) ELSE b_shared_blks_read END / 8192 AS shared_blks_read,
         CASE WHEN coalesce(b_shared_blks_dirtied, 0) >= coalesce(a_shared_blks_dirtied, 0) THEN coalesce(b_shared_blks_dirtied, 0) - coalesce(a_shared_blks_dirtied, 0) ELSE b_shared_blks_dirtied END / 8192 AS shared_blks_dirtied,
         CASE WHEN coalesce(b_shared_blks_written, 0) >= coalesce(a_shared_blks_written, 0) THEN coalesce(b_shared_blks_written, 0) - coalesce(a_shared_blks_written, 0) ELSE b_shared_blks_written END / 8192 AS shared_blks_written,
         CASE WHEN coalesce(b_blk_read_time, 0) >= coalesce(a_blk_read_time, 0) THEN coalesce(b_blk_read_time, 0) - coalesce(a_blk_read_time, 0) ELSE b_blk_read_time END AS blk_read_time,
         CASE WHEN coalesce(b_blk_write_time, 0) >= coalesce(a_blk_write_time, 0) THEN coalesce(b_blk_write_time, 0) - coalesce(a_blk_write_time, 0) ELSE b_blk_write_time END AS blk_write_time
  FROM e
  WHERE n_end > 0
), t AS (
  SELECT CAST(queryid AS VARCHAR) AS queryid, dbname,
         trim(regexp_replace(any_value(query), '\s+', ' ', 'g')) AS norm,
         trim(regexp_replace(regexp_replace(regexp_replace(any_value(query), '/\*.*?\*/', ' ', 'gs'),
              '--[^\n]*', ' ', 'g'), '\s+', ' ', 'g')) AS bare
  FROM query_texts
  WHERE regexp_full_match(dbname, getvariable('db_re'))
  GROUP BY ALL
), smart AS (
  SELECT queryid, dbname, norm, bare,
         regexp_extract_all(bare, '(?i)(\w+)\s+AS\s*\(', 1) AS ctes,
         CASE WHEN regexp_matches(bare, '(?i)^WITH\s') THEN coalesce(nullif(regexp_extract(bare, '(?i).*\)\s*(SELECT\b.*)$', 1), ''), bare)
              ELSE bare END AS main
  FROM t
), smart2 AS (
  SELECT *, list_filter(list_transform(
              string_split(trim(regexp_replace(regexp_extract(main, '(?i)\bFROM\s+(.*)$', 1),
                '(?i)\b(WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL|ORDER|GROUP|HAVING|LIMIT|OFFSET|UNION|INTERSECT|EXCEPT|FOR|RETURNING)\b.*$', '')), ','),
              lambda p: CASE WHEN starts_with(trim(p), '(') THEN '' ELSE regexp_extract(trim(p), '^([\w.]+)', 1) END),
              lambda x: x <> '') AS tables
  FROM smart
), txt AS (
  SELECT queryid, dbname, norm,
         CASE WHEN length(bare) <= 150 THEN bare
              WHEN regexp_matches(bare, '(?i)^SELECT') OR len(ctes) > 0 THEN
                CASE WHEN len(ctes) > 0 AND len(tables) > 0 THEN 'with ' || array_to_string(ctes, ', ') || ' select ... from ' || array_to_string(tables, ', ')
                     WHEN len(ctes) > 0 THEN 'with ' || array_to_string(ctes, ', ') || ' select ...'
                     WHEN len(tables) > 0 THEN 'select ... from ' || array_to_string(tables, ', ')
                     ELSE left(bare, 147) || '...' END
              WHEN regexp_matches(bare, '(?i)^INSERT\s+INTO\s+\S') THEN 'insert into ' || regexp_extract(bare, '(?i)^INSERT\s+INTO\s+(\S+)', 1) || ' ...'
              WHEN regexp_matches(bare, '(?i)^UPDATE\s+\S') THEN 'update ' || regexp_extract(bare, '(?i)^UPDATE\s+(\S+)', 1) || ' ...'
              WHEN regexp_matches(bare, '(?i)^DELETE\s+FROM\s+\S') THEN 'delete from ' || regexp_extract(bare, '(?i)^DELETE\s+FROM\s+(\S+)', 1) || ' ...'
              ELSE left(bare, 147) || '...' END AS smart
  FROM smart2
)
SELECT d.queryid,
       coalesce(CASE WHEN contains(lower(getvariable('legend_label')), 'raw')
                     THEN CASE WHEN length(txt.norm) > 150 THEN left(txt.norm, 147) || '...' ELSE txt.norm END
                     ELSE CASE WHEN length(txt.smart) > 150 THEN left(txt.smart, 147) || '...' ELSE txt.smart END END, '') AS query_text,
       d.duration_seconds,
       d.calls, CASE WHEN d.duration_seconds > 0 THEN d.calls / d.duration_seconds ELSE 0 END AS calls_per_sec, CASE WHEN d.calls > 0 THEN d.calls / d.calls ELSE 0 END AS calls_per_call,
       d.exec_time, CASE WHEN d.duration_seconds > 0 THEN d.exec_time / d.duration_seconds ELSE 0 END AS exec_time_per_sec, CASE WHEN d.calls > 0 THEN d.exec_time / d.calls ELSE 0 END AS exec_time_per_call,
       d.plan_time, CASE WHEN d.duration_seconds > 0 THEN d.plan_time / d.duration_seconds ELSE 0 END AS plan_time_per_sec, CASE WHEN d.calls > 0 THEN d.plan_time / d.calls ELSE 0 END AS plan_time_per_call,
       d.rows, CASE WHEN d.duration_seconds > 0 THEN d.rows / d.duration_seconds ELSE 0 END AS rows_per_sec, CASE WHEN d.calls > 0 THEN d.rows / d.calls ELSE 0 END AS rows_per_call,
       d.shared_blks_hit, CASE WHEN d.duration_seconds > 0 THEN d.shared_blks_hit / d.duration_seconds ELSE 0 END AS shared_blks_hit_per_sec, CASE WHEN d.calls > 0 THEN d.shared_blks_hit / d.calls ELSE 0 END AS shared_blks_hit_per_call,
       d.shared_blks_read, CASE WHEN d.duration_seconds > 0 THEN d.shared_blks_read / d.duration_seconds ELSE 0 END AS shared_blks_read_per_sec, CASE WHEN d.calls > 0 THEN d.shared_blks_read / d.calls ELSE 0 END AS shared_blks_read_per_call,
       d.shared_blks_dirtied, CASE WHEN d.duration_seconds > 0 THEN d.shared_blks_dirtied / d.duration_seconds ELSE 0 END AS shared_blks_dirtied_per_sec, CASE WHEN d.calls > 0 THEN d.shared_blks_dirtied / d.calls ELSE 0 END AS shared_blks_dirtied_per_call,
       d.shared_blks_written, CASE WHEN d.duration_seconds > 0 THEN d.shared_blks_written / d.duration_seconds ELSE 0 END AS shared_blks_written_per_sec, CASE WHEN d.calls > 0 THEN d.shared_blks_written / d.calls ELSE 0 END AS shared_blks_written_per_call,
       d.blk_read_time, CASE WHEN d.duration_seconds > 0 THEN d.blk_read_time / d.duration_seconds ELSE 0 END AS blk_read_time_per_sec, CASE WHEN d.calls > 0 THEN d.blk_read_time / d.calls ELSE 0 END AS blk_read_time_per_call,
       d.blk_write_time, CASE WHEN d.duration_seconds > 0 THEN d.blk_write_time / d.duration_seconds ELSE 0 END AS blk_write_time_per_sec, CASE WHEN d.calls > 0 THEN d.blk_write_time / d.calls ELSE 0 END AS blk_write_time_per_call
FROM d LEFT JOIN txt ON txt.queryid = d.queryid AND txt.dbname = d.datname
ORDER BY d.exec_time DESC, d.queryid, d.datname, d.usr, d.inst
