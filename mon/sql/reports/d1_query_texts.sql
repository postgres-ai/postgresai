-- Dashboard_2 legend/hover: the full query text, once per queryid of the panel's result.
-- Param: queryids (VARCHAR[] of the panel's labels['queryid']), db_re.
SELECT CAST(queryid AS VARCHAR) AS queryid, dbname, any_value(query) AS query
FROM query_texts
WHERE list_contains(getvariable('queryids'), CAST(queryid AS VARCHAR))
  AND regexp_full_match(dbname, getvariable('db_re'))
GROUP BY ALL;
