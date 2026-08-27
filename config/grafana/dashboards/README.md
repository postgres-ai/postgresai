# Grafana dashboards naming conventions

This document outlines the naming principles used in PostgresAI Grafana dashboards.

## Tag vocabulary

Every shipped dashboard JSON MUST include a top-level `"tags"` array whose
first entry is the project label `postgres-ai`, followed by 1-3 thematic tags
drawn from the controlled vocabulary below. The leading project label lets
operators filter the dashboard library to PostgresAI-shipped content even
when other Grafana provisioning providers are configured on the same
instance; the thematic tags give topical navigation.

**Project label (mandatory, always first):** `postgres-ai`

**Thematic vocabulary (pick 1-3):**

| Tag | Meaning |
|-----|---------|
| `overview` | High-level / landing-page dashboards |
| `node` | Per-node performance |
| `queries` | Query analysis (pg_stat_statements, etc.) |
| `waits` | Wait events |
| `ash` | Active Session History |
| `wal` | WAL generation and archiving |
| `backups` | Backup state and lag |
| `dr` | Disaster recovery |
| `replication` | Streaming/logical replication |
| `ha` | High availability state |
| `autovacuum` | Autovacuum workload |
| `xmin` | xmin horizon, bloat causes |
| `tables` | Table-level stats |
| `indexes` | Index-level stats |
| `slru` | SLRU caches |
| `locks` | Lock contention |
| `io` | I/O statistics (pg_stat_io) |
| `self-monitoring` | Monitoring of the monitoring stack itself |
| `monitoring-stack` | pgwatch/VictoriaMetrics/Grafana plumbing |

The mapping for shipped dashboards lives in the dashboards themselves; the
file `config/grafana/dashboards/<name>.json` is the source of truth.

When adding a new dashboard:

1. Add `postgres-ai` as the first tag.
2. Pick 1-3 thematic tags from the table above. Add to the table if a new
   theme is genuinely needed (and explain why in the PR).
3. Choose a stable top-level `uid` and never reuse one previously shipped:
   Grafana provisioning blocks the entire provider when two files share a
   top-level `uid` (see `config/init-configs.sh` for the upgrade-time
   cleanup of stale dashboards).

## Terminology rules

### Bloat metrics
Always use **"Estimated bloat"** when referring to bloat metrics. The bloat
values shown in these dashboards are based on estimation queries that use
pg_stat_user_tables statistics - they are not precise measurements like
pgstattuple would provide.

**Correct:**
- "Estimated bloat %"
- "Estimated bloat size"
- "Top $top_n tables by estimated bloat %"

**Incorrect:**
- "Bloat %"
- "Bloat size"

### Shared block I/O
Use **"Shared block reads"** and **"Shared block hits"** - these are the correct
Postgres terms.

- **Shared block hits**: Data was found in Postgres's shared buffer pool
- **Shared block reads**: Data was read into the shared buffer pool from the OS
  page cache. Note: This does NOT necessarily mean a disk read occurred - the data
  may have been served from the OS file system cache.

**Correct:**
- "Shared block reads"
- "Shared block hits"
- "Shared block hit ratio"

**Incorrect:**
- "Block disk reads" (we don't know if actual disk I/O occurred)
- "Block cache hits" (ambiguous - could mean OS cache or PG buffer pool)

### Rate metrics
For rate-based panels (showing per-second values), append `/s` to the title:

**Examples:**
- "Tuple operations /s"
- "Size growth /s"
- "Shared block hits /s"

### Section (row) naming
- **"Activity stats"**: For table dashboards showing tuple operations
- **"Index usage stats"**: For index dashboards showing scan/fetch metrics
- **"Estimated bloat stats"**: For bloat-related metrics (always include "Estimated")
- **"IO stats"**: For shared buffer pool I/O metrics
- **"Size stats"**: For size-related metrics

## Legend sorting

Every table-mode legend MUST declare a default sort (`sortBy` + `sortDesc: true`).
Without one the legend renders in datasource order, so a top-N panel does not put
the top offender first. Pick the key from what the panel measures:

| Panel measures | Examples | `sortBy` |
|----------------|----------|----------|
| A rate (per second) | calls/s, rows/s, bytes/s, ops/s, ASH, sessions, locks | `Mean` |
| Per call, latency, or saturation | per-call metrics, query latency, utilization % | `Max` |
| A level / state | sizes, estimated bloat, XID age, archive lag, retained WAL | `Last` (or `Last *`) |

Always descending.

Rate panels use `Mean`, not `Max`, because the top-N panels are built on
`topk(N, irate(...))`: a single scrape gap or counter reset yields a one-point
spike, and `Max` ranks that artifact above sustained load. Level panels use the
latest value because a table that was badly bloated *before* it was repacked
should not keep the top row.

`tests/grafana_dashboards/test_legend_sort_declared.py` enforces this on every
MR: it fails if a table legend has no `sortBy`, sorts ascending, or names a
column `calcs` does not display. The choice of key stays a judgement call.

**`sortBy` must name a column that `calcs` actually displays, or Grafana silently
ignores the sort.** Use the reducer display names:

| `calcs` entry | Column name |
|---------------|-------------|
| `min` / `max` / `mean` | `Min` / `Max` / `Mean` |
| `last` | `Last` |
| `lastNotNull` | `Last *` (note the asterisk) |

## Joining a sparsely-emitted metric

Some metrics are not emitted on every scrape. `pgwatch_query_info` — the
queryid-to-query-text mapping the pg_stat_statements legends join against — is
exported only for queryids active in the last `QUERYID_ACTIVE_MINUTES`, so a
queryid that goes quiet develops gaps of hours. Joined as a bare instant vector
it silently misses its ~5 min lookback, and the legend degrades to the raw label
set (`{cluster="…", datname="…", queryid="…"}`) instead of the query text.

Join such a metric through this operand, on **both** branches of the
`or … unless` pair:

```promql
(topk by (queryid) (1, tlast_over_time(pgwatch_query_info[7d])) > bool 0)
```

Each part earns its place:

| Part | Why |
|------|-----|
| `tlast_over_time(…[7d])` | Carries the last known mapping forward, so a stale queryid still resolves. Use the same window on both branches: widening one only would let a queryid match both, and `or` would emit it twice. |
| `topk by (queryid) (1, …)` | One queryid can hold several `displayname*` series inside a window that wide — the exporter's text pick is not stable across redeploys or sources. Without it the joined series is duplicated and the stacked total inflated. `tlast_over_time` returns the sample *time*, so the newest mapping wins; on an exact timestamp tie the pick is arbitrary but there is still exactly one. |
| `> bool 0` | Restores the value to `1`. The operand is multiplied into the metric being ranked, and without this the plotted rate would be scaled by a unix timestamp. |

`group_left(...)` must copy every label the panel's `legendFormat` renders. An
operand that is otherwise perfect still produces the raw-label failure if the
labels never cross the join, and when the legend is driven by a template
variable (`{{$legend_label}}`) that means *every* value the variable can take.

`tlast_over_time` is a MetricsQL extension, not PromQL — these dashboards ship
against VictoriaMetrics in both compose and Helm, alongside other MetricsQL
already in use here (`default 0`). Against a strict Prometheus there is no
equivalent: `last_over_time` carries the mapping forward but cannot de-duplicate
it, and the join then fails outright with "duplicate series for the match group".

The window is a judgement call — it must comfortably exceed the observed
per-series staleness, and it bounds how long a superseded query text can linger.
The mechanical parts are enforced on every MR by two tests sharing one
definition in `tests/grafana_dashboards/query_info_join.py`:
`tests/grafana_dashboards/test_query_info_carry_forward.py` (the operand, the
lookback floor, and the `group_left` label transfer, across every dashboard) and
`tests/compliance_vectors/test_mr219_monitoring_guards.py` (the `or … unless`
branch shape on Dashboard 02). They run in different CI jobs.

## Units

- **binBps**: Use binary bytes per second (KiB/s, MiB/s, GiB/s) for Postgres
  block I/O rates, as Postgres uses binary block sizes (typically 8 KiB)
- **bytes**: Use for absolute size measurements
- **percent**: Use for percentage values (0-100 scale)
- **ops**: Use for operations per second
