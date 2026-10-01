# Host metrics vocabulary

Every managed-Postgres provider we collect host metrics from lands in VictoriaMetrics under the same `host_*` names, in the same units, with exactly two labels: `cluster` and `node_name`, equal to the pgwatch tags of the target in `instances.yml`. Dashboards, alerts and sizing rules read `host_*` only. They never read a provider's raw series.

| Provider | Raw source | How it becomes `host_*` | Labels from |
| --- | --- | --- | --- |
| RDS / Aurora | CloudWatch, Performance Insights, Enhanced Monitoring | `rds-host-stats` writes `host_*` directly ([README](../rds-host-stats/README.md)) | `PGAI_CLUSTER`, `PGAI_NODE_NAME` (must equal the target's tags) |
| Supabase | node_exporter families from `/customer/v1/privileged/metrics`, relayed by `instance-jobs` (job `supabase-host-metrics`) | recording rules, group `host-supabase` | `sources-generator` copies the tags of the one Supabase target in `instances.yml` into the scrape target |
| ClickHouse Managed Postgres | `PostgresServer_*` from the ClickHouse Cloud `/prometheus` endpoint (jobs `clickhouse-<name>`) | recording rules, group `host-clickhouse` | `mon targets add` writes the target's tags into the scrape file |

The recording rules live in `config/prometheus/host_rules.yml`. The `vmalert` service evaluates them every 60 s over 3-minute rate windows and writes the results back to `sink-prometheus`. Their golden evaluation is `tests/host_rules/host_rules.test.yml`, and CI runs it (`quality:host-rules`).

## Names

A dash means the provider does not expose the value. RDS sources are CloudWatch `AWS/RDS` metrics unless marked PI (Performance Insights) or EM (Enhanced Monitoring). Supabase sources are node_exporter metrics without the `node_` prefix.

| Name | Unit | RDS / Aurora | Supabase | ClickHouse |
| --- | --- | --- | --- | --- |
| `host_cpu_utilization_percent` | percent | CPUUtilization | 100 × (1 − idle share of `cpu_seconds_total`) | 100 × (1 − idle share of `CPUSeconds_Total`) |
| `host_os_cpu_percent` | percent | EM cpuUtilization.total | – | – |
| `host_os_cpu_iowait_percent` | percent | EM cpuUtilization.wait | iowait share of `cpu_seconds_total` | iowait share of `CPUSeconds_Total` |
| `host_os_load1` | load | EM loadAverageMinute.one | `load1` | – |
| `host_memory_available_bytes` | bytes | FreeableMemory | `memory_MemAvailable_bytes` | MemoryLimitBytes × (1 − MemoryUsedPercent / 100) |
| `host_os_memory_total_bytes` | bytes | EM memory.total | `memory_MemTotal_bytes` | MemoryLimitBytes |
| `host_os_memory_free_bytes` | bytes | EM memory.free | `memory_MemFree_bytes` | – |
| `host_os_memory_cached_bytes` | bytes | EM memory.cached | `memory_Cached_bytes` | MemoryLimitBytes × MemoryCachePercent / 100 |
| `host_os_swap_used_bytes` | bytes | EM swap.total − swap.free | `memory_SwapTotal_bytes` − `memory_SwapFree_bytes` | – |
| `host_os_process_max_rss_bytes` | bytes | EM max process RSS | – | – |
| `host_disk_free_bytes` | bytes | FreeStorageSpace (RDS) | `filesystem_avail_bytes` on `/data` | StorageLimitBytes × (1 − FilesystemUsedPercent / 100) |
| `host_local_storage_free_bytes` | bytes | FreeLocalStorage (Aurora) | – | – |
| `host_volume_used_bytes` | bytes | VolumeBytesUsed (Aurora cluster) | – | – |
| `host_disk_read_iops` | operations/s | ReadIOPS | rate of `disk_reads_completed_total`, all disks | rate of `DiskReads_Total` |
| `host_disk_write_iops` | operations/s | WriteIOPS | rate of `disk_writes_completed_total`, all disks | rate of `DiskWrites_Total` |
| `host_volume_read_iops` | operations/s | VolumeReadIOPs / 300 (Aurora cluster) | – | – |
| `host_volume_write_iops` | operations/s | VolumeWriteIOPs / 300 (Aurora cluster) | – | – |
| `host_disk_read_latency_seconds` | seconds | ReadLatency | read time / reads completed (0 when idle) | – |
| `host_disk_write_latency_seconds` | seconds | WriteLatency | write time / writes completed (0 when idle) | – |
| `host_disk_queue_depth` | requests | DiskQueueDepth | rate of `disk_io_time_weighted_seconds_total` | – |
| `host_network_receive_bytes_per_second` | bytes/s | NetworkReceiveThroughput | rate of `network_receive_bytes_total`, without `lo` | rate of `NetworkReceiveBytes_Total` |
| `host_network_transmit_bytes_per_second` | bytes/s | NetworkTransmitThroughput | rate of `network_transmit_bytes_total`, without `lo` | rate of `NetworkTransmitBytes_Total` |
| `host_db_load` | average active sessions | PI db.load.avg | – | – |
| `host_replica_lag_seconds` | seconds | ReplicaLag (RDS), AuroraReplicaLag × 0.001 (Aurora) | – | – |
| `host_burst_balance_percent` | percent | BurstBalance (RDS) | – | – |
| `host_ebs_io_balance_percent` | percent | EBSIOBalance% (RDS) | – | – |

`tests/grafana_dashboards/test_host_row.py` fails if this table and the names the providers write drift apart.

## Dashboard

Dashboard 1 has one collapsed **Host** row, the last on the page. It has five panels: CPU utilization, Memory, Storage, Disk IOPS /s and Network /s. Each panel reads `host_*` with the dashboard's cluster and node selectors and charts at least one family that every provider writes, so none of them is empty for RDS, Supabase or ClickHouse. A self-managed node writes no `host_*`, and its row stays closed and empty. The other families (latency, queue depth, load, PI DB load, replica lag, burst and EBS balance) are not every provider's, so they are stored and queryable but not charted.

## Limits

- Supabase: only the primary (`supabase_identifier` = `supabase_project_ref`) is mapped. Read replicas are relayed but get no `host_*`. With no Supabase target in `instances.yml`, or more than one, the series carry no `cluster`/`node_name` and produce no `host_*`.
- Supabase disk IOPS and latency are summed over all block devices, including the root disk.
- ClickHouse: the raw names and units come from a fixture built from ClickHouse's documentation, not from a recorded response. `DiskReads_Total` and `DiskWrites_Total` are taken to be operation counts.
- The Helm chart ships the same dashboard but has no `vmalert` and no host collectors, so its Host row is empty.
