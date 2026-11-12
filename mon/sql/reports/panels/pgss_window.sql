-- Dashboard_2 panel 20 (pg_stat_statements over the dashboard range, one row per statement),
-- formerly the Flask backend's /pgss_metrics/csv. Per statement series and column: the counter's
-- increase over the range (prom_increase_pure: VictoriaMetrics' increase_pure, no extrapolation,
-- a series that starts inside the range counts from 0, a counter reset counts from 0), per second
-- (over the range) and per call. Bytes become 8 KiB blocks. Rows are sorted by exec_time
-- descending. Flask subtracted the sample nearest the range start from the one nearest the end:
-- the same numbers on a range without resets, where Flask showed negative values across a
-- reset (pg_stat_statements_reset, eviction) and dropped statements not sampled at the end.
-- The evaluation reads rollups for long ranges (about 0.2 s per column at 30 days).
-- query_text: the statement text from query_texts, whitespace-normalized; with legend_label
-- containing 'raw' cut at 150 characters (147 + '...'), else shortened the way Flask's
-- smart_truncate_query does (comments dropped; "select ... from <tables>", "with <ctes> select
-- ... from <tables>", "insert into <t> ...", "update <t> ...", "delete from <t> ..."). The main
-- statement after a WITH is the text after the last ') select', where Flask counts parentheses.
-- Params: cluster, node, db_re, start_ms, end_ms, legend_label.
WITH v AS (
  SELECT 'calls' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_calls', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'exec_time' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_exec_time_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'plan_time' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_plan_time_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'rows' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_rows', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'shared_blks_hit' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_shared_bytes_hit_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'shared_blks_read' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_shared_bytes_read_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'shared_blks_dirtied' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_shared_bytes_dirtied_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'shared_blks_written' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_shared_bytes_written_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'blk_read_time' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_block_read_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
  UNION ALL
  SELECT 'blk_write_time' AS m, labels, value
  FROM prom_increase_pure('pgwatch_pg_stat_statements_block_write_total', [['cluster','=',getvariable('cluster')],['node_name','=',getvariable('node')],['datname','=~',getvariable('db_re')]],
         getvariable('end_ms'), getvariable('end_ms'), 60000, greatest(getvariable('end_ms') - getvariable('start_ms'), 1))
), d AS (
  SELECT coalesce(labels['datname'], '') AS datname, coalesce(labels['queryid'], '') AS queryid,
         coalesce(labels['user'], '') AS usr, coalesce(labels['instance'], '') AS inst,
         (getvariable('end_ms') - getvariable('start_ms')) / 1000.0 AS duration_seconds,
         coalesce(max(value) FILTER (WHERE m = 'calls'), 0) AS calls,
         coalesce(max(value) FILTER (WHERE m = 'exec_time'), 0) AS exec_time,
         coalesce(max(value) FILTER (WHERE m = 'plan_time'), 0) AS plan_time,
         coalesce(max(value) FILTER (WHERE m = 'rows'), 0) AS rows,
         coalesce(max(value) FILTER (WHERE m = 'shared_blks_hit'), 0) / 8192 AS shared_blks_hit,
         coalesce(max(value) FILTER (WHERE m = 'shared_blks_read'), 0) / 8192 AS shared_blks_read,
         coalesce(max(value) FILTER (WHERE m = 'shared_blks_dirtied'), 0) / 8192 AS shared_blks_dirtied,
         coalesce(max(value) FILTER (WHERE m = 'shared_blks_written'), 0) / 8192 AS shared_blks_written,
         coalesce(max(value) FILTER (WHERE m = 'blk_read_time'), 0) AS blk_read_time,
         coalesce(max(value) FILTER (WHERE m = 'blk_write_time'), 0) AS blk_write_time
  FROM v
  WHERE NOT isnan(value)
  GROUP BY ALL
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
