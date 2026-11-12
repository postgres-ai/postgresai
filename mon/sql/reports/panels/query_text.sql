-- Dashboard_3 panel 18 (the statement's text), formerly a raw SQL query on the pgwatch
-- Postgres sink (pgss_queryid_queries): the newest stored text of the queryid in the selected
-- databases, or a placeholder row while none has been collected.
-- Params: query_id, db_re.
WITH q AS (
  SELECT query AS qt
  FROM query_texts
  WHERE CAST(queryid AS VARCHAR) = getvariable('query_id') AND regexp_full_match(dbname, getvariable('db_re'))
  ORDER BY last_seen DESC, dbname
  LIMIT 1
)
SELECT qt AS "Query text" FROM q
UNION ALL
SELECT '(query text not yet collected for this queryid; pgai-mon stores it with its next pg_stat_statements snapshot)'
WHERE NOT EXISTS (SELECT 1 FROM q)
