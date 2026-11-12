# Design note: subqueries, rollups over expressions, topk over expressions

Status: implemented SQL-side (2026-11-10): sq_binop_max.sql, sq_agg_max.sql,
sq_threshold_sum.sql, topk_*.sql. Parity: VM and macro oracle, all pass (D6 on the synthetic
store; D6, D8, D10 on a1). Scope: the Dashboard_6 subquery / rollup-over-expression targets
and the Dashboard_8 / Dashboard_10 topk targets.

Found while implementing: queries that add four or eight `prom_irate` calls (D8 31A/37A/42A/43A)
pass the 192 MB reader cap on live data, about 30 MB of RSS per table-function instance. The
SQL shape is not the cause, since matching on a key instead of a MAP saved about 10 %. This is
a collector/src item (grafana/README.md).

Update (2026-11-10, reader memory at the product caps): the SQL form held every (series, step)
row, most with a `labels` MAP, before ranking. At 24 h:60 s on a1 that was 137-178 MiB of reader
RSS for D8 23A/25A/28A (bare selector), 10A (irate difference) and 31A/37A/42A/43A (sums and
ratios of irates), against the 192 MiB cap. Now on the Rust path: topk_last.sql calls `prom_topk`
with fn 'last'; the expression targets call `prom_topk_expr` (`prom_topk_rs` with a `terms`
list: one scan over every term's metric, per label set and step two sums and a count, then the
k-heaps). Every D8/D10 target peaks at 46-63 MiB at 24 h:60 s and under 100 MiB at 7 d:1 h.
Oracles: `prom_topk_sql` (new 'last' branch) and `prom_topk_expr_sql` (macros.sql, up to 8
terms), so `diff.py --oracle-url` still checks both.

## The question

Two ways to port these:

- **SQL side.** Compose the existing prom_* table functions in SQL, then do the outer step
  (a window over a grid, or a per-step rank) with plain DuckDB SQL.
- **Rust side.** Extend `prom_topk` (and add a rollup-over-expression function) in
  collector/src, with matching `prom_*_sql` macros as the oracle.

Recommendation: **SQL side for both**. The Rust path stays open for later, for one case only:
topk over a ratio at pg_stat_statements scale, if it is ever measured to be too slow.

## 1. Rollup over an expression, and subqueries (D6)

Targets:

| Target | Shape |
|---|---|
| 221 B/C/D | `clamp_min(max_over_time((a - b)[$__rate_interval:]), 0)`: a rollup over a binary expression, with an implicit step-sized subquery |
| 231 B | `max_over_time((max(m))[$__rate_interval:])`: a rollup over an aggregation |
| 233 A | `sum_over_time(((max by (application_name) (m)) > bool X)[1h:1m])`: an explicit subquery with a 1 m inner step |

**SQL side.**

1. Evaluate the inner expression on the subquery's own grid. That grid runs from
   `prom_sq_start(end, range, step)` to `prom_sq_end(end, step)`, minus the window. Use the
   existing table functions plus a join or GROUP BY, as binop_last.sql and agg_fn.sql already
   do. The macros' `prom_sq_start` / `prom_sq_end` already give Prometheus' subquery alignment
   (absolute multiples of the inner step); r09_f004_hand_increase.sql uses them for R9.
2. Apply the outer rollup in SQL. For each outer step t, take the inner points in
   `(t - range, t]`, either with a range join on the outer grid or with
   `max/sum(...) OVER (PARTITION BY labels ORDER BY t RANGE BETWEEN range PRECEDING AND
   CURRENT ROW)`, then keep only the outer grid's t. `> bool` is a CASE. clamp_min is
   `greatest`, filtered for NaN first: `greatest(NaN, 0)` is NaN in DuckDB, as in PromQL, but
   `least(NaN, 1)` is 1, the D5 3A bug.
3. One shared file, e.g. `rollup_over_expr.sql`, parameterised by the rollup function and the
   inner step. The implicit `[$__rate_interval:]` subquery uses the outer step as its inner
   step, which is Grafana's `$__interval`.

**Rust side.** The table functions take only constant arguments. They can't consume a relation
(an inner expression's output), so a Rust rollup over an expression needs one of two things.
Either a function per shape (`prom_rollup_binop(fn, op, metric_a, metric_b, ...)`, which grows
with every new shape), or an expression evaluator inside pgai-mon. The evaluator is the
"PromQL engine" the spike chose not to build. Neither pays for 5 targets.

**Parity oracle.** The SQL composition gets both checks with no new work:
- `diff.py --oracle-url` rewrites each generic call to its `_sql` macro, so the Rust inner
  evaluation is checked against the macros;
- `diff.py --vm` checks the whole expression, subquery alignment included, against
  VictoriaMetrics. VM's subquery semantics match Prometheus for aligned steps. Check this on
  233A, where 1h:1m against a 15 s scrape is the case to watch.

The Rust path would need a new `_sql` macro twin per function before it had an oracle.

## 2. topk over an expression (D8, D10)

Census, from the source JSON:

- **Dashboard_8:** 26 targets, all topk/bottomk. 5 are over a plain selector, 8 over
  irate(selector), and 13 over expressions: sums of 2-4 irates, upd minus hot_upd, hit
  ratios, and an irate times a settings scalar (`* on() group_left() last_over_time(...)`).
- **Dashboard_10:** 7 targets. 3 are over a selector, 2 over irate, and 2 over irate times a
  scalar.

**SQL side.** Compute the inner expression as these files already do: irate_times_scalar.sql,
statio_ratio.sql and binop_last.sql, or a new `irate_sum.sql` for sums of irates. Then rank
per step:

```sql
SELECT ... FROM inner
QUALIFY row_number() OVER (PARTITION BY t ORDER BY value DESC) <= getvariable('top_n')
```

- bottomk is `ASC`.
- NaN is excluded first, since Prometheus ranks NaN last for topk.
- This is per-step selection, as Prometheus does on a range query, so the legend can show
  more than k series over the range. That matches Grafana on Prometheus today.
- Faithful except for which member of a tie is shown. diff.py's topk mode is tie-tolerant
  already.
- topk over a plain selector or irate(selector) uses the existing `prom_topk` (fn
  'last_over_time' / 'irate'), as topk_last_over_time.sql and topk_by.sql do. The macro
  `prom_topk_sql` has no plain 'last' branch. Add it (macro only, a few lines) or use the
  row_number form over raw_selector.sql. Either way it has an oracle.

**Rust side.** Extend `prom_topk` with a denominator metric (it already has `p_den_metric`),
a list of metrics to sum, a minus, and a scalar multiplier. Each addition is a new argument
in collector/src, in the macro twin and in the binding. The 13 + 2 expression targets need
four different extensions. The only gain is that `prom_topk` never materialises all series'
inner values. That matters for pg_stat_statements (tens of thousands of series), not for
tables and indexes (hundreds to low thousands per database), where the SQL form ranks a few
thousand rows per step.

**Parity oracle.**
- SQL side: the same two checks as in section 1, automatically. `--oracle-url` rewrites the
  inner prom_irate / prom_last_over_time calls to their macros. `--vm` runs the whole
  `topk(...)` in tie-tolerant mode.
- Rust side: an oracle only once each extension is mirrored in `prom_topk_sql`. Until then,
  VM only.

## Summary

| | SQL side | Rust side |
|---|---|---|
| D6 subquery / rollup over expression (5 targets) | One shared file. Oracle and VM checks exist today. | Needs per-shape functions or an evaluator. No oracle until macro twins exist. |
| D8/D10 topk over expression (15 targets) | Reuses the inner-expression files plus `QUALIFY row_number()`. Oracle and VM checks exist today. | Four `prom_topk` extensions in collector/src plus macros. Only worth it at pgss cardinality. |
| D8/D10 topk over selector / irate (18 targets) | Existing `prom_topk`. Add the 'last' branch to `prom_topk_sql`, or use row_number. | Not needed. |

Cleaner: the SQL side. It adds no collector surface, every target keeps two independent
checks (macro oracle plus VM), and the shapes are the ones the D5/D9/D11 port already
validated.

## Instant table panels (D5 14, D6 251/252, D7 93A, D13 4A)

Implemented in compile.py, as option A (`wrap_table`, `table_label_keys`):

- An instant target with Prometheus format "table", in a table panel, becomes an Infinity
  "table" format target with the columns `Time`, one string column per label, and `Value`.
  That is Prometheus' own column shape, so the panels' organize and filterFieldsByName
  transformations work unchanged.
- The label columns are the metric's label keys in the probe store, plus every field the
  panel's transformations name.
- Such targets get no `${__field.labels.series}` display-name override.
- Limit: a label that first appears after compile time, and that no transformation names,
  has no column.

Option B, for later and in collector/src: a `/sql?format=objects&expand=labels` option that
flattens the `labels` MAP into object keys per row. That removes the compile-time key list.
A dynamic PIVOT inside the SQL is not possible: the read-only guard rejects PIVOT.
