-- PromQL / MetricsQL semantics as DuckDB macros, matched to VictoriaMetrics v1.140.0
-- (app/vmselect/promql/rollup.go, eval.go; flags at their defaults:
--  -search.maxStalenessInterval=0, -search.maxLookback=0, -search.minStalenessInterval=0).
--
-- Consumers code only against the data-contract views (series, samples, ...). Load the views
-- file first, then this file. All times are epoch MILLISECONDS (BIGINT), like VictoriaMetrics.
--
-- Selector matchers are a list of [label, op, value] triples, op in '=', '!=', '=~', '!~':
--   {cluster="local", datname=~"a|b"}  ->  [['cluster','=','local'], ['datname','=~','a|b']]
-- A missing label matches as '' (PromQL). Regexes are fully anchored RE2 (Go and DuckDB both use RE2).
--
-- Names: the implementations here are prom_<fn>_sql, the parity oracle.
-- Callers use the generic prom_<fn>, bound to an implementation by sql/bind_sql.sql (these
-- macros) or sql/bind_rs.sql (pgai-mon's Rust table functions prom_<fn>_rs, same signature and
-- output columns). prom_select, prom_edges, prom_window and the scalars are SQL-only helpers.
-- Families:
--   prom_<fn>(metric, matchers, start, end, step, window)  -> (series_id, labels, t, value)
--       edge rollups via two ASOF joins: rate, increase, irate, delta, idelta, increase_pure,
--       last (bare selector = default_rollup), last_over_time, tlast_over_time, timestamp,
--       first_over_time, count_over_time, resets/changes are NOT here (see prom_window).
--       window = 0 means "no [d] given" (VM then uses step, widened to the scrape-based
--       lookback for rate/irate/default_rollup/timestamp).
--   prom_window(metric, matchers, start, end, step, window) -> (series_id, labels, t, ts, v)
--       every raw sample in (t - window, t]; GROUP BY series_id, t with any SQL aggregate gives
--       sum/avg/min/max/count/quantile/stddev_over_time, tmax_over_time (arg_max), etc.
--   prom_points(ts, start, step, window, end) -> LIST(BIGINT)
--       the same window explosion as a scalar, for subqueries over any relation:
--       f_over_time((expr)[W:S]) = UNNEST(prom_points(...)) over expr's step-grid output.
--
-- Instant query at time T: start = end = T, step = 300000 (VM default step = 5m; it is
-- also the instant lookback and the instant maxPrevInterval).

----------------------------------------------------------------------------------------------
-- Scalars
----------------------------------------------------------------------------------------------
CREATE OR REPLACE MACRO prom_match(labels, matchers) AS
  coalesce(list_bool_and(list_transform(matchers, lambda m:
    CASE m[2]
      WHEN '='  THEN coalesce(labels[m[1]], '') = m[3]
      WHEN '!=' THEN coalesce(labels[m[1]], '') <> m[3]
      WHEN '=~' THEN regexp_full_match(coalesce(labels[m[1]], ''), m[3])
      WHEN '!~' THEN NOT regexp_full_match(coalesce(labels[m[1]], ''), m[3])
    END)), true);

-- getMaxPrevInterval(): how far before the window start the "previous" sample may be.
CREATE OR REPLACE MACRO prom_max_prev(si) AS
  CASE WHEN si <= 2000  THEN si + 4 * si
       WHEN si <= 4000  THEN si + 2 * si
       WHEN si <= 8000  THEN si + si
       WHEN si <= 16000 THEN si + si // 2
       WHEN si <= 32000 THEN si + si // 4
       ELSE si + si // 8 END;

-- alignStartEnd() (subqueries always; range queries when the cache is on and >= 50 points).
CREATE OR REPLACE MACRO prom_align_down(x, step) AS x - x % step;
CREATE OR REPLACE MACRO prom_align_up(x, step) AS CASE WHEN x % step > 0 THEN x + step - x % step ELSE x END;
-- AdjustStartEnd(): what /api/v1/query_range does without nocache=1.
CREATE OR REPLACE MACRO prom_range_start(s, e, step) AS
  CASE WHEN (e - s) // step + 1 < 50 THEN s ELSE prom_align_down(s, step) END;
CREATE OR REPLACE MACRO prom_range_end(s, e, step) AS
  CASE WHEN (e - s) // step + 1 < 50 THEN e
       ELSE prom_align_up(e, step)
            - step * greatest(0, (prom_align_up(e, step) - prom_align_down(s, step)) // step - (e - s) // step) END;

-- removeCounterResets(): the correction a sample adds given the previous raw value.
-- A drop of less than 1/8 of the previous value is a "partial reset" (VM issue 2787) and is
-- flattened; a bigger drop is a real reset and the new value counts from 0.
CREATE OR REPLACE MACRO prom_reset_adj(v, pv) AS
  CASE WHEN pv IS NULL OR v >= pv THEN 0.0
       WHEN (pv - v) * 8 < pv THEN pv - v
       ELSE pv END;

-- Grid points t in [start, end] (step grid) whose window (t - window, t] contains ts.
CREATE OR REPLACE MACRO prom_points(ts, p_start, p_step, p_window, p_end) AS
  list_transform(
    range(greatest(0, CAST(ceil((ts - p_start) / CAST(p_step AS DOUBLE)) AS BIGINT)),
          least((p_end - p_start) // p_step,
                CAST(floor((ts + p_window - 1 - p_start) / CAST(p_step AS DOUBLE)) AS BIGINT)) + 1),
    lambda k: p_start + k * p_step);

CREATE OR REPLACE MACRO prom_ms(ts) AS epoch_ms(ts);
CREATE OR REPLACE MACRO prom_ts(ms) AS make_timestamptz(CAST(ms AS BIGINT) * 1000);

----------------------------------------------------------------------------------------------
-- Selection: the samples VM fetches for one rollup, with counter-reset-corrected values
----------------------------------------------------------------------------------------------
-- p_fetch_start..p_end inclusive, like storage.NewSearchQuery(minTimestamp, End).
-- samples.stale marks a staleness marker (the series vanished from a collection; VM StaleNaN).
-- VM drops markers before every rollup except default_rollup (a bare selector), where a trailing
-- marker hides the series at once instead of letting it linger for the lookback: p_keep_stale.
-- A real NaN value (stale = false) never hides a series; like VM, rollups skip NaN values.
-- Columns: series_id, labels, ts_ms, v (raw), c (reset-corrected, accumulated from the first
-- fetched sample exactly like removeCounterResets), rn (1-based), n_total, si (scrape interval
-- estimate: 0.6-quantile of the last 20 intervals, truncated to ms; NULL if < 2 samples).
-- prom_raw is the one place these macros read raw samples. On an object store the reader
-- rebinds it to the Rust scan (query.rs), since the `samples` view is not available there.
CREATE OR REPLACE MACRO prom_raw(p_metric, p_matchers, p_fetch_start, p_end) AS TABLE
SELECT x.series_id, epoch_ms(x.ts) AS ts_ms, x.value AS v, x.stale
FROM samples x
WHERE x.series_id IN (SELECT series_id FROM series WHERE metric = p_metric AND prom_match(labels, p_matchers))
  AND x.ts >= make_timestamptz(CAST(p_fetch_start AS BIGINT) * 1000)
  AND x.ts <= make_timestamptz(CAST(p_end AS BIGINT) * 1000);

CREATE OR REPLACE MACRO prom_select(p_metric, p_matchers, p_fetch_start, p_end, p_keep_stale := false) AS TABLE
WITH sel AS (
  SELECT series_id, labels FROM series
  WHERE metric = p_metric AND prom_match(labels, p_matchers)
), raw AS (
  SELECT series_id, ts_ms, v
  FROM prom_raw(p_metric, p_matchers, p_fetch_start, p_end)
  WHERE NOT isnan(v) OR (p_keep_stale AND stale)
), seq AS (
  SELECT series_id, ts_ms, v,
         lag(v) OVER w AS pv,
         ts_ms - lag(ts_ms) OVER w AS dts,
         row_number() OVER w AS rn,
         count(*) OVER (PARTITION BY series_id) AS n_total
  FROM raw WINDOW w AS (PARTITION BY series_id ORDER BY ts_ms)
), adj AS (
  SELECT *, prom_reset_adj(v, pv) AS a FROM seq
), resets AS (      -- running correction as a SEQUENTIAL left fold over the (few) resets, like
                   -- removeCounterResets; a windowed SUM adds in a thread-dependent order and
                   -- made the last bit, and so topk ties, vary between runs
  SELECT series_id, ts_ms,
         list_reduce(list(a) OVER (PARTITION BY series_id ORDER BY ts_ms
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                     lambda x, y: x + y) AS corr_after
  FROM adj WHERE a <> 0
), corr AS (
  SELECT adj.*, v + coalesce(last_value(r.corr_after IGNORE NULLS) OVER (PARTITION BY adj.series_id ORDER BY adj.ts_ms
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0.0) AS c
  FROM adj LEFT JOIN resets r USING (series_id, ts_ms)
), si AS (
  SELECT series_id, CAST(trunc(quantile_cont(dts, 0.6)) AS BIGINT) AS si
  FROM corr WHERE rn > n_total - 20 AND dts IS NOT NULL
  GROUP BY series_id
)
SELECT corr.series_id, sel.labels, ts_ms, v, c, rn, n_total, si.si
FROM corr JOIN sel USING (series_id) LEFT JOIN si USING (series_id);

----------------------------------------------------------------------------------------------
-- Edge rollups: everything rollupConfig.doInternal hands a rollup function, per (series, t)
----------------------------------------------------------------------------------------------
-- p_silence: needSilenceIntervalForRollupFunc (fetch 5m more history; rate, increase, irate,
--            delta, idelta, default_rollup, changes, resets, ...).
-- p_adjust:  rollupFuncsCanAdjustWindow (window = max(step, maxPrev) when no [d] is given).
CREATE OR REPLACE MACRO prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, p_silence, p_adjust, p_keep_stale := false) AS TABLE
WITH f AS (
  SELECT * FROM prom_select(p_metric, p_matchers,
           p_start - (CASE WHEN p_silence THEN 300000 ELSE 0 END) - greatest(p_window, p_step), p_end, p_keep_stale := p_keep_stale)
), s AS (
  SELECT series_id, any_value(labels) AS labels, any_value(n_total) AS n_total,
         CASE WHEN p_start < p_end THEN prom_max_prev(coalesce(any_value(si), p_step)) ELSE p_step END AS maxprev
  FROM f GROUP BY series_id
), s2 AS (
  SELECT *, CASE WHEN p_window > 0 THEN p_window
                 WHEN p_adjust AND p_step < maxprev THEN maxprev
                 ELSE p_step END AS w
  FROM s
), fx AS MATERIALIZED (            -- per sample: itself, its neighbours (for prev/first/second/last/penultimate/next)
  SELECT series_id, ts_ms, rn, v, c,
         lead(ts_ms) OVER w1 AS nx_ts, lead(v) OVER w1 AS nx_v, lead(c) OVER w1 AS nx_c,
         lead(v, 2) OVER w1 AS nx2_v, lead(c, 2) OVER w1 AS nx2_c,
         lag(ts_ms) OVER w1 AS pv_ts, lag(v) OVER w1 AS pv_v, lag(c) OVER w1 AS pv_c
  FROM (SELECT series_id, ts_ms, rn, v, c FROM f
        UNION ALL                      -- sentinel rn = 0 before every series: "no sample <= x"
        SELECT series_id, -4611686018427387904, 0, NULL, NULL FROM s)
  WINDOW w1 AS (PARTITION BY series_id ORDER BY rn)
), q AS MATERIALIZED (   -- no labels here: MAP payloads make the ASOF sorts 10x slower
  SELECT s2.series_id, s2.n_total, s2.maxprev, s2.w, g.t, g.t - s2.w AS ta
  FROM s2 CROSS JOIN (SELECT unnest(range(p_start, p_end + 1, p_step)) AS t) g
)
SELECT q.series_id, s2.labels, q.t, q.w, q.maxprev,
       b.rn - a.rn AS n,                                                    -- samples in (t-w, t]
       (a.rn > 0 AND a.rn < q.n_total AND a.ts_ms > q.ta - q.maxprev) AS has_prev,  -- rfa.prevValue
       (a.rn > 0) AS has_real_prev,                                         -- rfa.realPrevValue
       a.ts_ms AS prev_ts, a.v AS prev_v, a.c AS prev_c,
       a.nx_ts AS first_ts, a.nx_v AS first_v, a.nx_c AS first_c,
       a.nx2_v AS second_v, a.nx2_c AS second_c,
       b.ts_ms AS last_ts, b.v AS last_v, b.c AS last_c,
       b.pv_ts AS pen_ts, b.pv_v AS pen_v, b.pv_c AS pen_c,
       b.nx_v AS next_v, b.nx_c AS next_c                                    -- rfa.realNextValue
FROM q
ASOF JOIN fx a ON q.series_id = a.series_id AND q.ta >= a.ts_ms
ASOF JOIN fx b ON q.series_id = b.series_id AND q.t  >= b.ts_ms
JOIN s2 ON s2.series_id = q.series_id;

-- rollupDerivFast over counter-reset-corrected values. VM does NOT extrapolate to the window
-- edges (Prometheus does): it divides by the time between the previous sample (or the first
-- in-window sample) and the last in-window sample.
CREATE OR REPLACE MACRO prom_rate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
    CASE WHEN has_prev THEN CASE WHEN n = 0 THEN 0.0
                                 ELSE (last_c - prev_c) / ((last_ts - prev_ts) / 1e3) END
         WHEN n < 2 THEN NULL
         ELSE (last_c - first_c) / ((last_ts - first_ts) / 1e3) END AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, true))
WHERE value IS NOT NULL;

-- rollupDelta over corrected values (= increase). Without a usable previous sample it uses the
-- real previous sample (any distance inside the fetched range), else the "counter started at 0"
-- heuristic: prev = 0 if |first| < 10 * (|second - first| + 1), otherwise prev = first.
CREATE OR REPLACE MACRO prom_delta_core(has_prev, has_real_prev, n, prev_x, first_x, second_x, last_x, next_x) AS
  CASE WHEN has_prev THEN CASE WHEN n = 0 THEN 0.0 ELSE last_x - prev_x END
       WHEN n = 0 THEN NULL
       WHEN has_real_prev THEN last_x - prev_x
       WHEN abs(first_x) < 10 * (abs(CASE WHEN n > 1 THEN second_x - first_x
                                          WHEN next_x IS NOT NULL THEN next_x - first_x
                                          ELSE 0.0 END) + 1)
            THEN last_x
       ELSE last_x - first_x END;

CREATE OR REPLACE MACRO prom_increase_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
         prom_delta_core(has_prev, has_real_prev, n, prev_c, first_c, second_c, last_c, next_c) AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, false))
WHERE value IS NOT NULL;

-- delta(): rollupDelta on raw values (no reset handling), gauges.
CREATE OR REPLACE MACRO prom_delta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
         prom_delta_core(has_prev, has_real_prev, n, prev_v, first_v, second_v, last_v, next_v) AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, false))
WHERE value IS NOT NULL;

-- increase_pure(): like increase but a series without history counts from 0.
CREATE OR REPLACE MACRO prom_increase_pure_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
    CASE WHEN has_prev THEN CASE WHEN n = 0 THEN 0.0 ELSE last_c - prev_c END
         WHEN n = 0 THEN NULL
         WHEN has_real_prev THEN last_c - prev_c
         ELSE last_c END AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, false))
WHERE value IS NOT NULL;

-- rollupIderiv over corrected values: the last two samples in the window.
CREATE OR REPLACE MACRO prom_irate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
    CASE WHEN n = 0 THEN NULL
         WHEN n = 1 THEN CASE WHEN has_prev THEN (last_c - prev_c) / ((last_ts - prev_ts) / 1e3) END
         ELSE (last_c - pen_c) / ((last_ts - pen_ts) / 1e3) END AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, true))
WHERE value IS NOT NULL;

-- prom_irate_k / prom_labels_k (eval.sql) on the macros: the same lk from the same label set.
CREATE OR REPLACE MACRO prom_irate_k_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT hash(list_sort(map_entries(labels))) AS lk, t, value FROM prom_irate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window);

CREATE OR REPLACE MACRO prom_labels_k_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT DISTINCT ON (lk) hash(list_sort(map_entries(labels))) AS lk, labels
FROM series WHERE metric = p_metric AND prom_match(labels, p_matchers);

-- rollupIdelta on raw values.
CREATE OR REPLACE MACRO prom_idelta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, value FROM (
  SELECT series_id, labels, t,
    CASE WHEN n = 0 THEN CASE WHEN has_prev THEN 0.0 END
         WHEN n = 1 THEN CASE WHEN has_prev THEN last_v - prev_v ELSE last_v END
         ELSE last_v - pen_v END AS value
  FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, true, false))
WHERE value IS NOT NULL;

-- A bare selector `x` (default_rollup): the last raw sample in (t - w, t], w = step widened to
-- the scrape-based maxPrev (range) or 5m (instant). NOTE: on VM a range query with step=1h
-- looks back a whole hour for a bare selector, Prometheus would look back 5m.
CREATE OR REPLACE MACRO prom_last_sql(p_metric, p_matchers, p_start, p_end, p_step) AS TABLE
SELECT series_id, labels, t, last_v AS value
FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, 0, true, true, p_keep_stale := true)
WHERE n > 0 AND NOT isnan(last_v);

-- Same, plus the raw sample timestamp (seconds): `timestamp(x)` (rollupTlast).
CREATE OR REPLACE MACRO prom_timestamp_sql(p_metric, p_matchers, p_start, p_end, p_step) AS TABLE
SELECT series_id, labels, t, last_ts / 1e3 AS value
FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, 0, false, true)
WHERE n > 0;

CREATE OR REPLACE MACRO prom_last_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, last_v AS value
FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, false, false)
WHERE n > 0;

CREATE OR REPLACE MACRO prom_tlast_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, last_ts / 1e3 AS value
FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, false, false)
WHERE n > 0;

CREATE OR REPLACE MACRO prom_first_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, t, first_v AS value
FROM prom_edges(p_metric, p_matchers, p_start, p_end, p_step, p_window, false, false)
WHERE n > 0;

----------------------------------------------------------------------------------------------
-- Window explosion: all raw samples in (t - w, t] for every grid point t
----------------------------------------------------------------------------------------------
-- No [d] (window = 0) means window = step, as VM does for *_over_time.
-- Aggregate in SQL: avg(v) = avg_over_time, sum(v) = sum_over_time, count(*) = count_over_time,
-- quantile_cont(v, q) = quantile_over_time (same (n-1)*phi interpolation),
-- stddev_pop(v) = stddev_over_time (VM uses the population form), max/min, arg_max(ts, v) ~ tmax
-- (VM tmax returns the LAST timestamp of the max: use arg_max(ts, (v, ts))).
CREATE OR REPLACE MACRO prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, labels, unnest(prom_points(ts_ms, p_start, p_step,
                                 CASE WHEN p_window > 0 THEN p_window ELSE p_step END, p_end)) AS t,
       ts_ms AS ts, v
FROM prom_select(p_metric, p_matchers,
                 p_start - greatest(CASE WHEN p_window > 0 THEN p_window ELSE p_step END, p_step), p_end);

-- *_over_time as table functions (same output columns as the edge rollups), on prom_window.
CREATE OR REPLACE MACRO prom_avg_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, avg(v) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_sum_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, sum(v) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_min_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, min(v) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_max_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, max(v) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_count_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, CAST(count(*) AS DOUBLE) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_stddev_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, stddev_pop(v) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;
CREATE OR REPLACE MACRO prom_quantile_over_time_sql(p_phi, p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT series_id, any_value(labels) AS labels, t, quantile_cont(v, p_phi) AS value FROM prom_window(p_metric, p_matchers, p_start, p_end, p_step, p_window) GROUP BY series_id, t;


-- topk / bottomk per step of fn(metric{matchers}[window]) (fn 'last': the bare selector, window
-- unused), or of fn(metric) / fn(den_metric) matched on identical label sets (Dashboard 1's per-call panels); NaN dropped; ties broken by value, then
-- labels['queryid'], then the label set as text (REVIEW.md C1). Same signature and output as
-- pgai-mon's Rust prom_topk (prom_topk_rs). p_by: sum by those labels before ranking.
CREATE OR REPLACE MACRO prom_topk_sql(p_k, p_fn, p_metric, p_matchers, p_start, p_end, p_step, p_window,
                                      p_den_metric := '', p_bottom := false,
                                      p_by := CAST([] AS VARCHAR[])) AS TABLE
WITH num AS (
  SELECT * FROM prom_rate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'rate'
  UNION ALL SELECT * FROM prom_irate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'irate'
  UNION ALL SELECT * FROM prom_increase_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase'
  UNION ALL SELECT * FROM prom_delta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'delta'
  UNION ALL SELECT * FROM prom_idelta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'idelta'
  UNION ALL SELECT * FROM prom_increase_pure_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase_pure'
  UNION ALL SELECT * FROM prom_last_sql(p_metric, p_matchers, p_start, p_end, p_step) WHERE p_fn = 'last'
  UNION ALL SELECT * FROM prom_last_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'last_over_time'
  UNION ALL SELECT * FROM prom_avg_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'avg_over_time'
  UNION ALL SELECT * FROM prom_sum_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'sum_over_time'
  UNION ALL SELECT * FROM prom_max_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'max_over_time'
  UNION ALL SELECT * FROM prom_min_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'min_over_time'
), den AS (
  SELECT * FROM prom_rate_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'rate'
  UNION ALL SELECT * FROM prom_irate_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'irate'
  UNION ALL SELECT * FROM prom_increase_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase'
  UNION ALL SELECT * FROM prom_delta_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'delta'
  UNION ALL SELECT * FROM prom_idelta_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'idelta'
  UNION ALL SELECT * FROM prom_increase_pure_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase_pure'
  UNION ALL SELECT * FROM prom_last_sql(p_den_metric, p_matchers, p_start, p_end, p_step) WHERE p_fn = 'last'
  UNION ALL SELECT * FROM prom_last_over_time_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'last_over_time'
  UNION ALL SELECT * FROM prom_avg_over_time_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'avg_over_time'
  UNION ALL SELECT * FROM prom_sum_over_time_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'sum_over_time'
  UNION ALL SELECT * FROM prom_max_over_time_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'max_over_time'
  UNION ALL SELECT * FROM prom_min_over_time_sql(p_den_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'min_over_time'
), gn AS (      -- p_by: sum by those labels first (PromQL topk(k, sum by (...) (fn(x)))); the
               -- group's labels are prom_by(labels, p_by), its series_id any member's
  SELECT min(series_id) AS series_id, prom_by(labels, p_by) AS labels, t, sum(value) AS value
  FROM num WHERE len(p_by) > 0 GROUP BY 2, 3
  UNION ALL SELECT series_id, labels, t, value FROM num WHERE len(p_by) = 0
), gd AS (
  SELECT min(series_id) AS series_id, prom_by(labels, p_by) AS labels, t, sum(value) AS value
  FROM den WHERE len(p_by) > 0 GROUP BY 2, 3
  UNION ALL SELECT series_id, labels, t, value FROM den WHERE len(p_by) = 0
), v AS (
  SELECT series_id, labels, t, value FROM gn WHERE coalesce(p_den_metric, '') = ''
  UNION ALL
  SELECT gn.series_id, gn.labels, gn.t, gn.value / gd.value
  FROM gn JOIN gd ON gn.labels = gd.labels AND gn.t = gd.t
  WHERE coalesce(p_den_metric, '') <> ''
)
SELECT series_id, labels, t, value FROM v
WHERE NOT isnan(value)
QUALIFY row_number() OVER (PARTITION BY t ORDER BY CASE WHEN p_bottom THEN value ELSE -value END,
                                                  labels['queryid'], CAST(labels AS VARCHAR)) <= p_k;

-- fn(metric{matchers}[window]) for the fn named at run time (series_id, labels, t, value).
CREATE OR REPLACE MACRO prom_fn_sql(p_fn, p_metric, p_matchers, p_start, p_end, p_step, p_window) AS TABLE
SELECT * FROM prom_rate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'rate'
UNION ALL SELECT * FROM prom_irate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'irate'
UNION ALL SELECT * FROM prom_increase_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase'
UNION ALL SELECT * FROM prom_delta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'delta'
UNION ALL SELECT * FROM prom_idelta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'idelta'
UNION ALL SELECT * FROM prom_increase_pure_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase_pure'
UNION ALL SELECT * FROM prom_last_sql(p_metric, p_matchers, p_start, p_end, p_step) WHERE p_fn = 'last'
UNION ALL SELECT * FROM prom_last_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'last_over_time'
UNION ALL SELECT * FROM prom_avg_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'avg_over_time'
UNION ALL SELECT * FROM prom_sum_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'sum_over_time'
UNION ALL SELECT * FROM prom_max_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'max_over_time'
UNION ALL SELECT * FROM prom_min_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'min_over_time';

-- topk / bottomk per step of an expression over several metrics matched one-to-one on identical
-- label sets: Σ n_i fn(m_i), or Σ n_i fn(m_i) / Σ d_i fn(m_i) when any d is non-zero; p_terms
-- [{'m': metric, 'n': coef, 'd': coef}, ...] (at most 8 here), each metric once. A label set has
-- a value at a step only when every term has one; then the optional sum by (p_by), NaN dropped
-- and prom_topk_sql's order. series_id = the first term's series (a group's: the least over the
-- range of its label sets with a value). Same
-- signature and output as pgai-mon's Rust prom_topk_expr (prom_topk_rs with terms).
CREATE OR REPLACE MACRO prom_topk_expr_sql(p_k, p_fn, p_terms, p_matchers, p_start, p_end, p_step, p_window,
                                           p_bottom := false, p_by := CAST([] AS VARCHAR[])) AS TABLE
WITH x AS (
  SELECT 1 AS i, * FROM prom_fn_sql(p_fn, p_terms[1].m, p_matchers, p_start, p_end, p_step, p_window)
  UNION ALL SELECT 2, * FROM prom_fn_sql(p_fn, p_terms[2].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 2
  UNION ALL SELECT 3, * FROM prom_fn_sql(p_fn, p_terms[3].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 3
  UNION ALL SELECT 4, * FROM prom_fn_sql(p_fn, p_terms[4].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 4
  UNION ALL SELECT 5, * FROM prom_fn_sql(p_fn, p_terms[5].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 5
  UNION ALL SELECT 6, * FROM prom_fn_sql(p_fn, p_terms[6].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 6
  UNION ALL SELECT 7, * FROM prom_fn_sql(p_fn, p_terms[7].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 7
  UNION ALL SELECT 8, * FROM prom_fn_sql(p_fn, p_terms[8].m, p_matchers, p_start, p_end, p_step, p_window) WHERE len(p_terms) >= 8
), e AS (
  SELECT min(series_id) FILTER (WHERE i = 1) AS series_id, labels, t,
         sum(value * p_terms[i].n) AS nv, sum(value * p_terms[i].d) AS dv
  FROM x WHERE NOT isnan(value)
  GROUP BY labels, t HAVING count(*) = len(p_terms)
), v AS (
  SELECT series_id, labels, t,
         CASE WHEN len(list_filter(p_terms, lambda x: x.d <> 0)) > 0 THEN nv / dv ELSE nv END AS value
  FROM e
), g AS (
  SELECT min(series_id) AS series_id, prom_by(labels, p_by) AS labels, t, sum(value) AS value
  FROM v WHERE len(p_by) > 0 GROUP BY 2, 3
  UNION ALL SELECT series_id, labels, t, value FROM v WHERE len(p_by) = 0
)
SELECT min(series_id) OVER (PARTITION BY labels) AS series_id, labels, t, value FROM g
WHERE NOT isnan(value)
QUALIFY row_number() OVER (PARTITION BY t ORDER BY CASE WHEN p_bottom THEN value ELSE -value END,
                                                  labels['queryid'], CAST(labels AS VARCHAR)) <= p_k;

-- sum/avg/count/min/max by (p_by) of fn(metric{matchers}[window]) per step: PromQL
-- `<op> by (...) (fn(x[w]))`. Output (labels, t, value), labels = prom_by(labels, p_by). Same
-- signature and output as pgai-mon's Rust prom_agg (prom_agg_rs, a group sink).
CREATE OR REPLACE MACRO prom_agg_sql(p_op, p_fn, p_metric, p_matchers, p_start, p_end, p_step, p_window,
                                     p_by := CAST([] AS VARCHAR[])) AS TABLE
WITH x AS (
  SELECT * FROM prom_rate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'rate'
  UNION ALL SELECT * FROM prom_irate_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'irate'
  UNION ALL SELECT * FROM prom_increase_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase'
  UNION ALL SELECT * FROM prom_delta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'delta'
  UNION ALL SELECT * FROM prom_idelta_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'idelta'
  UNION ALL SELECT * FROM prom_increase_pure_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'increase_pure'
  UNION ALL SELECT * FROM prom_last_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'last_over_time'
  UNION ALL SELECT * FROM prom_avg_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'avg_over_time'
  UNION ALL SELECT * FROM prom_sum_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'sum_over_time'
  UNION ALL SELECT * FROM prom_max_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'max_over_time'
  UNION ALL SELECT * FROM prom_min_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'min_over_time'
  UNION ALL SELECT * FROM prom_count_over_time_sql(p_metric, p_matchers, p_start, p_end, p_step, p_window) WHERE p_fn = 'count_over_time'
)
SELECT prom_by(labels, p_by) AS labels, t,
       CASE p_op WHEN 'sum' THEN sum(value) WHEN 'avg' THEN avg(value) WHEN 'count' THEN CAST(count(value) AS DOUBLE)
                 WHEN 'min' THEN min(value) WHEN 'max' THEN max(value) END AS value
FROM x WHERE NOT isnan(value)
GROUP BY 1, 2;

----------------------------------------------------------------------------------------------
-- Helpers for the step grid of a subquery `(expr)[W:S]` evaluated at instant/range points
----------------------------------------------------------------------------------------------
-- VM evaluates expr as a range query on [align_down(start - W - S - 5m, S), align_up(end + S, S)].
CREATE OR REPLACE MACRO prom_sq_start(p_start, p_window, p_sq_step) AS
  prom_align_down(p_start - p_window - p_sq_step - 300000, p_sq_step);
CREATE OR REPLACE MACRO prom_sq_end(p_end, p_sq_step) AS prom_align_up(p_end + p_sq_step, p_sq_step);

-- Labels of an aggregation result: `sum by (a, b)` keeps only a, b; empty values are dropped.
CREATE OR REPLACE MACRO prom_by(labels, keys) AS
  map_from_entries(list_filter(list_transform(keys, lambda k: {'k': k, 'v': labels[k]}),
                               lambda e: coalesce(e.v, '') <> ''));

----------------------------------------------------------------------------------------------
-- Reporter formatting helpers (reporter/postgres_reports.py), so a check can be one SQL file
----------------------------------------------------------------------------------------------
-- format_bytes(): IEC units, '%.0f' >= 100, '%.1f' >= 10, else '%.2f'; exact powers of 1024.
CREATE OR REPLACE MACRO fmt_bytes_k(v) AS
  CASE WHEN v >= 1099511627776 THEN 4 WHEN v >= 1073741824 THEN 3 WHEN v >= 1048576 THEN 2
       WHEN v >= 1024 THEN 1 ELSE 0 END;
CREATE OR REPLACE MACRO fmt_bytes_x(x, k) AS
  CASE WHEN x >= 100 THEN printf('%.0f', x) WHEN x >= 10 THEN printf('%.1f', x) ELSE printf('%.2f', x) END
  || ' ' || ['B', 'KiB', 'MiB', 'GiB', 'TiB'][k + 1];
CREATE OR REPLACE MACRO fmt_bytes(v) AS
  CASE WHEN v = 0 THEN '0 B' ELSE fmt_bytes_x(v / pow(1024, fmt_bytes_k(v)), fmt_bytes_k(v)) END;
-- datetime.fromtimestamp(e).isoformat() with TZ=UTC (microseconds round half-even, omitted when 0)
CREATE OR REPLACE MACRO py_us(e) AS CAST(trunc(e) AS BIGINT) * 1000000 + CAST(round_even((e - trunc(e)) * 1e6, 0) AS BIGINT);
CREATE OR REPLACE MACRO py_isoformat(e) AS
  strftime(make_timestamp(py_us(e)), '%Y-%m-%dT%H:%M:%S')
  || CASE WHEN py_us(e) % 1000000 <> 0 THEN '.' || lpad(CAST(py_us(e) % 1000000 AS VARCHAR), 6, '0') ELSE '' END;
-- datetime.fromtimestamp(e, tz=utc).strftime('%Y-%m-%dT%H:%M:%SZ')
CREATE OR REPLACE MACRO py_utc_z(e) AS strftime(make_timestamp(py_us(e)), '%Y-%m-%dT%H:%M:%SZ');
