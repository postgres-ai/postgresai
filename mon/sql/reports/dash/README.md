# Dashboards 3-14 in SQL

Every PromQL target maps to one SQL file plus parameters: the `D3/` to `D14/` cases in mon/grafana/cases.py, ids `<census>/<panel id><refId>_<what>` (the key
mon/grafana/compile.py matches targets on). Shapes that Dashboard_1 already has reuse
`../d2/` and `../last_over_time.sql`. New shapes are here.

Common parameters: cluster, node, db_re (the `$db_name` regex), db, start_ms, end_ms, step_ms,
interval_ms (`$__interval`), rate_interval_ms (`$__rate_interval`). Dashboard filter variables:
wet_re (`$wait_event_type`), we_re (`$wait_event`), query_id (`$query_id`), schema_re
(`$schema_name`), table_re (`$table_name`), index_re (`$index_name`), datname_re (`$datname`).
range_ms is `$__range` (whole seconds, as Grafana's `${__range_s}`).

| Dashboard / panel and ref | File | Params |
|---|---|---|
| D4 1 A/B/C Active session history | ../d2/ash.sql | variant a/b/c |
| D4 2 A, D4 3 A, D3 19 A ASH by event (and query) | ash_by.sql | by, filter wet / wet_we / query |
| D3 2-17 one query's pg_stat_statements rates | irate_query.sql | metric, per_call, query_id |
| D7 54 A, 55 A, 91 A top-N by last_over_time | topk_last_over_time.sql | metric, matchers, window_ms, top_n |
| D7 54 B/C, 55 B/C wraparound settings | ../d2/max_last_over_time.sql | metric, gt0 |
| D7 54 D, 55 D, 91 B constants | ../d2/constant.sql | value |
| D7 11 A-G, 12 A-E, 92 A/B, 3 A; D13 4 A (instant) | ../d2/raw_selector.sql | metric, matchers |
| D7 92 C % of autovacuum workers | scaled_ratio.sql | metric, den_metric, matchers, scale, den_min |
| D7 93 A (instant) | ../last_over_time.sql | metric, matchers, window_ms |
| D7 2 A Vacuum timeline, filtered | vacuum_timeline_st.sql | schema_re, table_re |
| D13 1 A Lock conflicts | count_or_zero.sql | metric, matchers |
| D13 2 A/B, 3 A, 5 A/B, 6 A | agg_last.sql | op, metric, matchers, by |
| D12 46-52 A SLRU | ../d2/irate_series.sql | metric |
| D14 2-32 pg_stat_io rates | irate_matchers.sql | metric, matchers |
| D14 4 A Buffer hit ratio | pct_of_sum.sql | metric, other, matchers |
| D5 6A | ../last_over_time.sql | metric, window_ms |
| D5 7A; D9 21 A-D, 37 A-D | agg_fn.sql | op, fn, metric, matchers, by |
| D5 2 A/B; D9 39 A-E, 42 A/B; D11 5A, 6 A/B | irate_matchers.sql | metric, matchers |
| D5 3A archive success ratio, clamp_max(.., 1) | sum_ratio_clamp.sql | metric, other, matchers |
| D5 1A; D11 1A one-to-one `-` or `/` of two selectors | binop_last.sql | op, metric, other, matchers |
| D5 9 A/B; D9 32A, 33A, 38A; D11 3A, 1B | ../d2/raw_selector.sql | metric, matchers |
| D5 11A, 12A; D9 38 B-E | agg_last.sql | op, metric, matchers, by |
| D5 14 A/B, D6 251A, 252A (instant) `last_over_time(m[$__range])` | last_over_range.sql | metric, matchers |
| D5 15A topk by | topk_by.sql | k, fn, metric, matchers, by |
| D9 44 B-E, 45 B-E; D11 4 A/B rate times a per-object size | irate_times_scalar.sql | metric, matchers, smetric, smatchers, swindow_ms |
| D9 46 A/B hit (or read) share of 8 statio rates | statio_ratio.sql | num |
| D11 1C index_pilot rebuild scale factor (most specific config wins) | index_pilot_scale.sql | schema_re, index_re, datname_re |
| D6 221A, 222 A-C replica lag, while one replica is seen over `$__range` | replica_stage.sql | metric, cond_metric, cond_by, clamp0, divisor |
| D6 243B `max(last_over_time(m[$__range]))` | max_last_range.sql | metric, divisor |

| D6 102A, 105A, 106A, 107A | agg_last.sql | op, metric, matchers, by |
| D6 104A, 211A, 231A, 241A, 242A | agg_fn.sql | op, fn, metric, matchers, by |
| D6 103A, 212A, 243A (ms -> s) | agg_fn_div.sql | op, fn, metric, matchers, by, divisor |
| D6 221 B-D `clamp_min(max_over_time((a - b)[$__rate_interval]), 0) and on() (one standby)` | sq_binop_max.sql | metric, other, matchers, cond_metric, cond_by |
| D6 231B `max_over_time((max(m))[$__rate_interval]) / 1000` | sq_agg_max.sql | metric, matchers, divisor |
| D6 233A `sum_over_time((max by (application_name) (m) > bool $lag_alert_s*1000)[1h:1m])` | sq_threshold_sum.sql | metric, matchers, lag_alert_s, sq_window_ms, sq_step_ms |
| D8 1A, 2A, 8A, 23A, 25A, 28A; D10 2A, 22A, 23A `topk($top_n, m)` | topk_last.sql | metric, matchers, top_n |
| D8 7A, 24A, 26A, 29A, 32A, 39A, 40A, 41A; D8 3A, 5A, 6A (by); D10 27A, 28A | topk_fn.sql | fn, metric, matchers, by, top_n |
| D8 10A `topk(sum by (...) (irate(upd) - irate(hot_upd)))` | topk_irate_diff_by.sql | metric, other, matchers, by, top_n |
| D8 31A, 37A `topk(Σ4 irate) * block_size` | topk_statio_sum_x_scalar.sql | num, matchers, smetric, smatchers, swindow_ms, top_n |
| D8 42A, 43A `topk(hit or read share of 8 statio rates)` | topk_statio_ratio.sql | num, matchers, top_n |
| D8 33A, 34A, 36A, 38A; D10 25A, 26A `topk(irate) * block_size` | topk_irate_x_scalar.sql | metric, matchers, smetric, smatchers, swindow_ms, top_n |

The rollups over expressions evaluate the inner expression on the subquery grid (absolute
multiples of the inner step, prom_sq_start to prom_sq_end; MetricsQL's `(expr)[d]` with no
step uses the query step) and window it in SQL. topk over an expression ranks per step with
`QUALIFY row_number() OVER (PARTITION BY t ORDER BY value DESC, labels as text) <= top_n`,
after dropping NaN. See NOTE_subquery_topk.md.
