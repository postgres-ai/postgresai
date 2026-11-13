-- Consumer contract.
-- Consumers query ONLY these views. The binary defines the internal _raw_<kind> views over
-- the catalog's current Parquet file list (no directory listing) and then runs this file
-- once; _raw_* are re-pointed after every flush/compaction, these views stay.
--
-- _raw_samples      (series_id UBIGINT, ts TIMESTAMPTZ, value DOUBLE, stale BOOLEAN)  -- layout-independent
-- _raw_series       (series_id UBIGINT, metric VARCHAR, labels MAP(VARCHAR,VARCHAR), first_seen TIMESTAMPTZ)
-- _raw_logs, _raw_plan_events, _raw_plan_defs, _raw_query_texts, _raw_objects: see README.

-- one row per unique (metric, labels). Series files are per day (a series has a row in every
-- day it has samples, so time-scoped lookups read only their window's days). This view does not
-- fold them (that is a dedupe of the whole retention): it reads the series registry, one row per
-- series in disjoint files built at compaction, plus the series of not-yet-registered day files
-- that are new to the registry (pgai_series_tail_ids: one file per id). Filters on metric push
-- down into the registry Parquet.
CREATE OR REPLACE VIEW series AS
SELECT series_id, metric, labels FROM _raw_series_reg
UNION ALL
SELECT t.series_id, t.metric, t.labels
FROM _raw_series_tail t JOIN pgai_series_tail_ids() n ON t.series_id = n.series_id AND t.filename = n.file;

-- every sample. stale (extra column) = a staleness marker (value NaN): the series ended at ts,
-- as VictoriaMetrics' StaleNaN; a real NaN value has stale = false.
CREATE OR REPLACE VIEW samples AS
SELECT ts, series_id, value, stale FROM _raw_samples;

-- samples join series, convenience
CREATE OR REPLACE VIEW metrics AS
SELECT s.ts, r.metric, r.labels, s.value
FROM _raw_samples s JOIN series r USING (series_id);

-- pg_stat_statements texts, stored once per (source, queryid, dbname, text); first/last seen.
-- source / cluster / node_name: the monitored node, as its series' dbname / cluster /
-- node_name labels (NULL in rows collected before they were recorded).
CREATE OR REPLACE VIEW query_texts AS
SELECT queryid, dbname, max(query) AS query, min(first_seen) AS first_seen, max(last_seen) AS last_seen,
       source, cluster, node_name
FROM _raw_query_texts
GROUP BY queryid, dbname, query_hash, source, cluster, node_name;

-- Postgres jsonlog lines; parameter values dropped; auto_explain lines carry plan_hash.
-- queryid (extra column) = auto_explain "Query Identifier" or jsonlog query_id.
CREATE OR REPLACE VIEW logs AS
SELECT ts, level, sqlstate, pid, dbname, username, application, message,
       query_fingerprint, plan_hash, raw::JSON AS raw, queryid
FROM _raw_logs;

-- one row per (plan_hash, queryid): normalized auto_explain plans (literals redacted)
CREATE OR REPLACE VIEW plans AS
SELECT e.plan_hash, e.queryid, min(e.ts) AS first_seen, max(e.ts) AS last_seen,
       count(*) AS count, sum(e.total_ms) AS total_ms, any_value(d.plan)::JSON AS plan
FROM _raw_plan_events e
LEFT JOIN (SELECT plan_hash, any_value(plan) AS plan FROM _raw_plan_defs GROUP BY plan_hash) d
  USING (plan_hash)
GROUP BY e.plan_hash, e.queryid;

-- extra: every auto_explain execution (time, buffers, top node) for "how did time change"
CREATE OR REPLACE VIEW plan_events AS
SELECT ts, plan_hash, queryid, dbname, total_ms, top_node, shared_hit, shared_read,
       shared_dirtied, shared_written, temp_read, temp_written
FROM _raw_plan_events;

-- table/index/constraint/view/function definitions with validity ranges; only changes are
-- stored. valid_to NULL = current. dbname is an extra column (names repeat across databases);
-- source / cluster / node_name as in query_texts (names repeat across nodes too).
CREATE OR REPLACE VIEW objects AS
SELECT kind, schema, name, definition, valid_from, valid_to, dbname, source, cluster, node_name
FROM (
  SELECT kind, schema, name, definition, dbname, source, cluster, node_name, ts AS valid_from,
         lead(ts) OVER (PARTITION BY source, dbname, kind, schema, name ORDER BY ts) AS valid_to
  FROM _raw_objects
)
WHERE definition IS NOT NULL;

-- extra: metric dictionary from metrics.yml (interval = collection period = lookback to use)
CREATE OR REPLACE VIEW metric_dict_prefix AS
SELECT metric_group, prefix, interval_s, instance_level, gauges, description FROM _metric_dict;

CREATE OR REPLACE VIEW metric_dict AS
SELECT m.metric, d.metric_group, d.interval_s, d.instance_level,
       CASE WHEN d.metric_group IS NULL THEN NULL
            WHEN list_contains(d.gauges, '*') OR list_contains(d.gauges, substr(m.metric, length(d.prefix) + 2))
            THEN 'gauge' ELSE 'counter' END AS type,
       d.description
FROM (SELECT DISTINCT metric FROM _raw_series) m
LEFT JOIN (
  -- longest matching prefix wins (pg_stat_all_tables vs pg_stat_all_tables_..)
  SELECT * FROM _metric_dict
) d ON m.metric = d.prefix OR starts_with(m.metric, d.prefix || '_')
QUALIFY row_number() OVER (PARTITION BY m.metric ORDER BY length(d.prefix) DESC NULLS LAST) = 1;
