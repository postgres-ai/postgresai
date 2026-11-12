-- Host metrics vocabulary (docs/host-metrics.md): the 37 recording rules of the old
-- old/config/prometheus/host_rules.yml (groups host-supabase and host-clickhouse) as one query.
-- pgai-mon evaluates it every 60 s at t = getvariable('t_ms') (an instant, aligned to the minute)
-- and writes the rows back as host_* series {cluster, node_name[, provider]} (src/rules.rs),
-- so dashboards, reports and sizing rules read host_* like any other series.
--
-- Semantics follow the PromQL/MetricsQL they replace:
--   rate(x[3m])            prom_rate(x, matchers, t, t, 300000, 180000) (VictoriaMetrics rate)
--   bare selector x         prom_last(x, matchers, t, t, 300000)          (5 m lookback, as vmalert's step)
--   labels_equal(q, a, b)   coalesce(labels[a], '') = coalesce(labels[b], '')
--   sum/max by (c, n)       GROUP BY cluster, node_name; an empty group gives no row
--   a / (b > 0)             a row only where b > 0
-- Every selector requires cluster and node_name (cluster!="", node_name!=""): an unlabelled
-- provider series never feeds an unlabelled host_* series. Output: metric, cluster, node_name,
-- provider (host_provider_info only), value; NaN and NULL rows are dropped (VictoriaMetrics drops
-- NaN results too).
WITH
-- ---------------------------------------------------------------------------------------------
-- Supabase (job supabase-host-metrics, relayed by instance-jobs). Only the primary
-- (supabase_identifier = supabase_project_ref) carries the target's cluster/node_name meaning.
-- The exposition is sampled up to a minute before the scrape and counters step every 120 s, so
-- CPU shares are ratios of the CPU counters and per-second rates are divided by the rate of the
-- host's own node_time_seconds.
-- ---------------------------------------------------------------------------------------------
sb_cpu AS (
  SELECT labels['cluster'] AS cluster, labels['node_name'] AS node_name, labels['mode'] AS mode, value
  FROM prom_rate('node_cpu_seconds_total',
                 [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']],
                 getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  WHERE coalesce(labels['supabase_identifier'], '') = coalesce(labels['supabase_project_ref'], '')
),
sb_ctr AS (
  SELECT 'reads' AS k, labels, value FROM prom_rate('node_disk_reads_completed_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'writes', labels, value FROM prom_rate('node_disk_writes_completed_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'read_bytes', labels, value FROM prom_rate('node_disk_read_bytes_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'written_bytes', labels, value FROM prom_rate('node_disk_written_bytes_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'read_time', labels, value FROM prom_rate('node_disk_read_time_seconds_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'write_time', labels, value FROM prom_rate('node_disk_write_time_seconds_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'net_rx', labels, value FROM prom_rate('node_network_receive_bytes_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db'],['device','!=','lo']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'net_tx', labels, value FROM prom_rate('node_network_transmit_bytes_total',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db'],['device','!=','lo']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'clock', labels, value FROM prom_rate('node_time_seconds',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
),
-- sum by (cluster, node_name) of each counter's rate (max for the clock), primary only
sb_sum AS (
  SELECT k, labels['cluster'] AS cluster, labels['node_name'] AS node_name,
         CASE WHEN k = 'clock' THEN max(value) ELSE sum(value) END AS v
  FROM sb_ctr
  WHERE coalesce(labels['supabase_identifier'], '') = coalesce(labels['supabase_project_ref'], '')
  GROUP BY k, labels['cluster'], labels['node_name']
),
sb_clock AS (SELECT cluster, node_name, v FROM sb_sum WHERE k = 'clock' AND v > 0),
sb_gauge AS (
  SELECT 'avail' AS k, labels, value FROM prom_last('node_memory_MemAvailable_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'total', labels, value FROM prom_last('node_memory_MemTotal_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'free', labels, value FROM prom_last('node_memory_MemFree_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'cached', labels, value FROM prom_last('node_memory_Cached_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'buffers', labels, value FROM prom_last('node_memory_Buffers_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'swap_total', labels, value FROM prom_last('node_memory_SwapTotal_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'swap_free', labels, value FROM prom_last('node_memory_SwapFree_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  -- Postgres data lives on the /data volume.
  SELECT 'fs_avail', labels, value FROM prom_last('node_filesystem_avail_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db'],['mountpoint','=','/data']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'fs_size', labels, value FROM prom_last('node_filesystem_size_bytes',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db'],['mountpoint','=','/data']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'time', labels, value FROM prom_last('node_time_seconds',
    [['job','=','supabase-host-metrics'],['cluster','!=',''],['node_name','!=',''],['service_type','=','db']], getvariable('t_ms'), getvariable('t_ms'), 300000)
),
sb_g AS (
  SELECT k, labels, labels['cluster'] AS cluster, labels['node_name'] AS node_name, value,
         -- the series identity without the metric name, for one-to-one matching (SwapTotal - SwapFree)
         array_to_string(list_sort(list_transform(map_entries(labels), e -> e.key || '=' || e.value)), ',') AS lkey
  FROM sb_gauge
  WHERE coalesce(labels['supabase_identifier'], '') = coalesce(labels['supabase_project_ref'], '')
),
sb_gmax AS (SELECT k, cluster, node_name, max(value) AS v FROM sb_g GROUP BY ALL),
sb_swap AS (
  SELECT t.cluster, t.node_name, max(t.value - f.value) AS v
  FROM sb_g t JOIN sb_g f ON t.k = 'swap_total' AND f.k = 'swap_free' AND t.lkey = f.lkey
  GROUP BY ALL
),
supabase AS (
  SELECT 'host_provider_info' AS metric, cluster, node_name, 'Supabase' AS provider, v * 0 + 1 AS value
  FROM sb_gmax WHERE k = 'time'
  UNION ALL
  SELECT 'host_cpu_utilization_percent', cluster, node_name, NULL,
         100 * (1 - sum(value) FILTER (WHERE mode = 'idle') / sum(value))
  FROM sb_cpu GROUP BY ALL
  UNION ALL
  SELECT m.metric, c.cluster, c.node_name, NULL, 100 * sum(c.value) FILTER (WHERE c.mode = m.mode) / sum(c.value)
  FROM sb_cpu c, (VALUES ('host_os_cpu_iowait_percent', 'iowait'), ('host_os_cpu_user_percent', 'user'),
                         ('host_os_cpu_system_percent', 'system'), ('host_os_cpu_steal_percent', 'steal')) m(metric, mode)
  GROUP BY m.metric, c.cluster, c.node_name
  UNION ALL
  SELECT m.metric, g.cluster, g.node_name, NULL, g.v
  FROM sb_gmax g JOIN (VALUES ('avail', 'host_memory_available_bytes'), ('total', 'host_os_memory_total_bytes'),
                              ('free', 'host_os_memory_free_bytes'), ('cached', 'host_os_memory_cached_bytes'),
                              ('buffers', 'host_os_memory_buffers_bytes'), ('fs_avail', 'host_disk_free_bytes'),
                              ('fs_size', 'host_disk_size_bytes')) m(k, metric) USING (k)
  UNION ALL
  SELECT 'host_os_swap_used_bytes', cluster, node_name, NULL, v FROM sb_swap
  UNION ALL
  -- per-second rates: summed over every block device (root disk included: the compute size's
  -- IOPS and MB/s limits cap all attached volumes together) and network device but lo,
  -- divided by the rate of the host clock, absent without a moving clock of the primary
  SELECT m.metric, s.cluster, s.node_name, NULL, s.v / c.v
  FROM sb_sum s JOIN sb_clock c USING (cluster, node_name)
  JOIN (VALUES ('reads', 'host_disk_read_iops'), ('writes', 'host_disk_write_iops'),
               ('read_bytes', 'host_disk_read_bytes_per_second'), ('written_bytes', 'host_disk_write_bytes_per_second'),
               ('net_rx', 'host_network_receive_bytes_per_second'), ('net_tx', 'host_network_transmit_bytes_per_second')) m(k, metric)
    ON s.k = m.k
  UNION ALL
  -- latency: time spent per operation, where there were operations
  SELECT m.metric, t.cluster, t.node_name, NULL, t.v / n.v
  FROM sb_sum t JOIN sb_sum n USING (cluster, node_name)
  JOIN (VALUES ('read_time', 'reads', 'host_disk_read_latency_seconds'),
               ('write_time', 'writes', 'host_disk_write_latency_seconds')) m(tk, nk, metric)
    ON t.k = m.tk AND n.k = m.nk
  WHERE n.v > 0
),
-- ---------------------------------------------------------------------------------------------
-- ClickHouse Managed Postgres (jobs clickhouse-<name>). Two targets on one service give the
-- same service series twice: counters are deduplicated per (org, service) before summing.
-- ---------------------------------------------------------------------------------------------
ch_cpu AS (
  SELECT labels['cluster'] AS cluster, labels['node_name'] AS node_name, labels['mode'] AS mode, value
  FROM prom_rate('PostgresServer_CPUSeconds_Total', [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']],
                 getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
),
ch_ctr AS (
  SELECT 'host_disk_read_iops' AS metric, labels, value FROM prom_rate('PostgresServer_DiskReads_Total',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'host_disk_write_iops', labels, value FROM prom_rate('PostgresServer_DiskWrites_Total',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'host_network_receive_bytes_per_second', labels, value FROM prom_rate('PostgresServer_NetworkReceiveBytes_Total',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
  UNION ALL
  SELECT 'host_network_transmit_bytes_per_second', labels, value FROM prom_rate('PostgresServer_NetworkTransmitBytes_Total',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000, 180000)
),
ch_svc AS (   -- max by (cluster, node_name, clickhouse_org, postgres_service)
  SELECT metric, labels['cluster'] AS cluster, labels['node_name'] AS node_name, max(value) AS v
  FROM ch_ctr GROUP BY metric, labels['cluster'], labels['node_name'], labels['clickhouse_org'], labels['postgres_service']
),
ch_gauge AS (
  SELECT 'mem_limit' AS k, labels, value FROM prom_last('PostgresServer_MemoryLimitBytes',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'mem_used_pct', labels, value FROM prom_last('PostgresServer_MemoryUsedPercent',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'mem_cache_pct', labels, value FROM prom_last('PostgresServer_MemoryCachePercent',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'storage_limit', labels, value FROM prom_last('PostgresServer_StorageLimitBytes',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000)
  UNION ALL
  SELECT 'fs_used_pct', labels, value FROM prom_last('PostgresServer_FilesystemUsedPercent',
    [['job','=~','clickhouse-.+'],['cluster','!=',''],['node_name','!=','']], getvariable('t_ms'), getvariable('t_ms'), 300000)
),
ch_g AS (   -- max by (cluster, node_name) of each gauge, one row per node
  SELECT labels['cluster'] AS cluster, labels['node_name'] AS node_name,
         max(value) FILTER (WHERE k = 'mem_limit') AS mem_limit,
         max(value) FILTER (WHERE k = 'mem_used_pct') AS mem_used_pct,
         max(value) FILTER (WHERE k = 'mem_cache_pct') AS mem_cache_pct,
         max(value) FILTER (WHERE k = 'storage_limit') AS storage_limit,
         max(value) FILTER (WHERE k = 'fs_used_pct') AS fs_used_pct
  FROM ch_gauge GROUP BY ALL
),
clickhouse AS (
  SELECT 'host_provider_info' AS metric, cluster, node_name, 'ClickHouse' AS provider, mem_limit * 0 + 1 AS value FROM ch_g
  UNION ALL
  SELECT 'host_cpu_utilization_percent', cluster, node_name, NULL,
         100 * (1 - sum(value) FILTER (WHERE mode = 'idle') / sum(value))
  FROM ch_cpu GROUP BY ALL
  UNION ALL
  SELECT m.metric, c.cluster, c.node_name, NULL, 100 * sum(c.value) FILTER (WHERE c.mode = m.mode) / sum(c.value)
  FROM ch_cpu c, (VALUES ('host_os_cpu_iowait_percent', 'iowait'), ('host_os_cpu_user_percent', 'user'),
                         ('host_os_cpu_system_percent', 'system'), ('host_os_cpu_steal_percent', 'steal')) m(metric, mode)
  GROUP BY m.metric, c.cluster, c.node_name
  UNION ALL
  SELECT 'host_os_memory_total_bytes', cluster, node_name, NULL, mem_limit FROM ch_g
  UNION ALL
  SELECT 'host_memory_available_bytes', cluster, node_name, NULL, mem_limit * (1 - mem_used_pct / 100) FROM ch_g
  UNION ALL
  SELECT 'host_os_memory_cached_bytes', cluster, node_name, NULL, mem_limit * mem_cache_pct / 100 FROM ch_g
  UNION ALL
  SELECT 'host_disk_free_bytes', cluster, node_name, NULL, storage_limit * (1 - fs_used_pct / 100) FROM ch_g
  UNION ALL
  SELECT 'host_disk_size_bytes', cluster, node_name, NULL, storage_limit FROM ch_g
  UNION ALL
  SELECT metric, cluster, node_name, NULL, sum(v) FROM ch_svc GROUP BY ALL
)
SELECT metric, cluster, node_name, provider, CAST(value AS VARCHAR) AS value
FROM (SELECT * FROM supabase UNION ALL SELECT * FROM clickhouse)
WHERE value IS NOT NULL AND NOT isnan(value) AND cluster IS NOT NULL AND node_name IS NOT NULL
ORDER BY metric, cluster, node_name, provider
