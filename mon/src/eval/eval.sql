-- Rust-backed PromQL-shaped functions (src/eval). Same names, signatures and output columns as
-- sql/macros.sql: (series_id, labels, t, value). The SQL originals stay as prom_<fn>_sql
-- (the parity oracle). Matchers go to Rust as JSON. Labels come from prom_series_rs: the selected
-- series of the call's time window only (never a join with the whole `series` registry).

CREATE OR REPLACE MACRO prom_rate(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('rate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('rate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_irate(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('irate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('irate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_increase(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('increase', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('increase', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_delta(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('delta', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('delta', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_increase_pure(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('increase_pure', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('increase_pure', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_idelta(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('idelta', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('idelta', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_last_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('last_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('last_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_tlast_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('tlast_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('tlast_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_first_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('first_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('first_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_avg_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('avg_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('avg_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_sum_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('sum_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('sum_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_min_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('min_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('min_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_max_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('max_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('max_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_count_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('count_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('count_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_stddev_over_time(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('stddev_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('stddev_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_last(p_metric, p_matchers, p_start, p_end, p_step) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('last', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(0 AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('last', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(0 AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_timestamp(p_metric, p_matchers, p_start, p_end, p_step) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('timestamp', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(0 AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('timestamp', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(0 AS BIGINT))) s USING (series_id);

CREATE OR REPLACE MACRO prom_quantile_over_time(p_phi, p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT e.series_id, s.labels, e.t, e.value
FROM prom_eval_rs('quantile_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(p_phi AS DOUBLE)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs('quantile_over_time', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);


-- topk/bottomk per step of fn(metric{matchers}), or of fn(metric) / fn(den_metric) matched on
-- identical label sets, or of sum by (p_by) (fn(metric)); NaN dropped; ties: value,
-- labels['queryid'], labels as text (D1's and R1's order). Evaluated in Rust series by series
-- (per step the k best plus ties at the k-th; with p_by, one row per group), not by
-- materialising every point. prom_topk_rs returns the winners' labels itself (read for the
-- finalists only), so no join to every selected series' labels.
-- With p_by the labels are the group's: prom_by(labels, p_by).
CREATE OR REPLACE MACRO prom_by_rs(labels, keys) AS
  map_from_entries(list_filter(list_transform(keys, lambda k: {'k': k, 'v': labels[k]}),
                               lambda e: coalesce(e.v, '') <> ''));
CREATE OR REPLACE MACRO prom_topk(p_k, p_fn, p_metric, p_matchers, p_start, p_end, p_step, p_window, p_den_metric := '', p_bottom := false, p_by := CAST([] AS VARCHAR[])) AS TABLE
SELECT e.series_id,
       CASE WHEN len(p_by) > 0 THEN prom_by_rs(CAST(CAST(e.labels AS JSON) AS MAP(VARCHAR, VARCHAR)), p_by)
            ELSE CAST(CAST(e.labels AS JSON) AS MAP(VARCHAR, VARCHAR)) END AS labels,
       e.t, e.value
FROM prom_topk_rs(CAST(p_k AS BIGINT), p_fn, p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(coalesce(p_den_metric, '') AS VARCHAR), CAST(p_bottom AS VARCHAR), CAST(to_json(p_by) AS VARCHAR)) e;

-- <op> by (p_by) (fn(metric{matchers})) per step, op in sum/avg/count/min/max; p_by empty = one
-- group (labels {}). Accumulated in Rust one series at a time (one row per group and step), so
-- a sum over 2,000 series never materialises 2,000 grids. No SQL oracle: it equals
--   SELECT prom_by_rs(labels, p_by), t, <op>(value) FROM prom_<fn>(...) GROUP BY ALL
CREATE OR REPLACE MACRO prom_agg(p_op, p_fn, p_metric, p_matchers, p_start, p_end, p_step, p_window, p_by := CAST([] AS VARCHAR[])) AS TABLE
SELECT prom_by_rs(s.labels, p_by) AS labels, e.t, e.value
FROM prom_agg_rs(p_op, p_fn, p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(to_json(p_by) AS VARCHAR)) e
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM prom_series_rs(p_fn, p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s
  USING (series_id);

-- Label-set keyed irate, for expressions over several metrics that share label sets (sums and
-- ratios of rates: Dashboard_8 31A/37A/42A/43A): (lk, t, value), lk = a hash of the label set
-- (sorted entries; the metric name is not a label). Aggregate and rank on lk, then join each
-- label set's labels once with prom_labels_k. prom_irate carries the labels MAP on every point,
-- which is most of those queries' memory (~10 MB per call on 1 h at a 5 s step, more in the
-- GROUP BY), and what pushed them past the reader cap.
CREATE OR REPLACE MACRO prom_irate_k(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT s.lk, e.t, e.value
FROM prom_eval_rs('irate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), CAST(0.0 AS DOUBLE)) e
JOIN (SELECT series_id, hash(list_sort(map_entries(CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR))))) AS lk
      FROM prom_series_rs('irate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT))) s USING (series_id);

-- The label sets of a prom_irate_k call's selection: (lk, labels).
CREATE OR REPLACE MACRO prom_labels_k(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT DISTINCT ON (lk) hash(list_sort(map_entries(m))) AS lk, m AS labels
FROM (SELECT CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS m
      FROM prom_series_rs('irate', p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT)));

-- topk/bottomk per step of an expression over several metrics matched one-to-one on identical
-- label sets (Dashboard_8 10A, 31A/37A, 42A/43A): Σ n_i fn(m_i), or Σ n_i fn(m_i) / Σ d_i fn(m_i)
-- when any d is non-zero. p_terms: [{'m': metric, 'n': coef, 'd': coef}, ...], each metric once;
-- a label set has a value at a step only when every term's series has one there. Then, as
-- prom_topk: the optional sum by (p_by) (not over a ratio), NaN dropped, the same tie order;
-- labels = the label set (the first term's series). One prom_topk_rs call reads every term in
-- one scan and keeps per label set and step two sums and a count, never every (series, step)
-- row with its labels (the SQL form of 10A/31A/42A held 140-180 MB of reader RSS at 24 h).
-- Oracle: prom_topk_expr_sql (macros.sql, up to 8 terms).
CREATE OR REPLACE MACRO prom_topk_expr(p_k, p_fn, p_terms, p_matchers, p_start, p_end, p_step, p_window, p_bottom := false, p_by := CAST([] AS VARCHAR[])) AS TABLE
SELECT e.series_id,
       CASE WHEN len(p_by) > 0 THEN prom_by_rs(CAST(CAST(e.labels AS JSON) AS MAP(VARCHAR, VARCHAR)), p_by)
            ELSE CAST(CAST(e.labels AS JSON) AS MAP(VARCHAR, VARCHAR)) END AS labels,
       e.t, e.value
FROM prom_topk_rs(CAST(p_k AS BIGINT), p_fn, CAST(p_terms[1].m AS VARCHAR), CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT), CAST(p_step AS BIGINT), CAST(p_window AS BIGINT), '', CAST(p_bottom AS VARCHAR), CAST(to_json(p_by) AS VARCHAR), terms := CAST(to_json(p_terms) AS VARCHAR)) e;

-- Raw samples of one metric in [p_start, p_end] (ms, inclusive), read by the Rust scan: the
-- portable way to read raw data. It works on local and object stores alike; on an s3:// store
-- the `samples`/`metrics` views are empty (README, "SQL over an object store"). Same rows as
-- `samples` joined to `series` for that metric and window: NaN values and staleness markers
-- (stale = true) included. Unordered; add ORDER BY.
CREATE OR REPLACE MACRO pgai_samples(p_metric, p_matchers, p_start, p_end) AS TABLE
SELECT x.series_id, s.labels, make_timestamptz(x.ts_ms * 1000) AS ts, x.value, x.stale
FROM pgai_samples_rs(p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT)) x
JOIN (SELECT series_id, CAST(CAST(labels AS JSON) AS MAP(VARCHAR, VARCHAR)) AS labels
      FROM pgai_series_rs(p_metric, CAST(to_json(p_matchers) AS VARCHAR), CAST(p_start AS BIGINT), CAST(p_end AS BIGINT))) s USING (series_id);
