# Host metrics vocabulary

Every managed-Postgres provider we collect host metrics from lands in VictoriaMetrics under the same `host_*` names, in the same units, with exactly two labels: `cluster` and `node_name`, equal to the pgwatch tags of the target in `instances.yml`. Dashboards, alerts and sizing rules read `host_*` only. They never read a provider's raw series.

| Provider | Raw source | How it becomes `host_*` | Labels from |
| --- | --- | --- | --- |
| RDS / Aurora | CloudWatch, Enhanced Monitoring | `rds-host-stats` writes `host_*` directly ([README](../rds-host-stats/README.md)) | `.env`: `PGAI_CLUSTER`, `PGAI_NODE_NAME` (with `RDS_DB_INSTANCE_IDENTIFIER`, `AWS_REGION`) |
| Supabase | node_exporter families from `/customer/v1/privileged/metrics`, relayed by `instance-jobs` (job `supabase-host-metrics`) | recording rules, group `host-supabase` | the scrape file `host-metrics/supabase-<name>.yml`, while `PGAI_SUPABASE_HOST_METRICS` is true |
| ClickHouse Managed Postgres | `PostgresServer_*` from the ClickHouse Cloud `/prometheus` endpoint (jobs `clickhouse-<name>`) | recording rules, group `host-clickhouse` | the scrape file `host-metrics/clickhouse-<name>.yml` |

`mon targets add` (and `mon local-install --db-url`, which goes through it) copies the target's tags into the last column, so they are set in one place. A scrape file is reloaded into `sink-prometheus`; `rds-host-stats` pushes past samples and is not scraped, so it reads `.env`; it starts with `docker compose --profile rds up -d rds-host-stats`, and `mon targets add`/`remove` recreate it when it is running. It polls one instance: the last RDS instance endpoint added. A cluster, reader or proxy endpoint names no instance and sets nothing. After editing a target's tags in `instances.yml`, re-run `mon targets add` for it.

The recording rules live in `config/prometheus/host_rules.yml`. The `vmalert` service evaluates them every 60 s over 3-minute rate windows and writes the results back to `sink-prometheus`. Their golden evaluation is `tests/host_rules/host_rules.test.yml`, and CI runs it (`quality:host-rules`).

## Names

A dash means the provider does not expose the value. RDS sources are CloudWatch `AWS/RDS` metrics unless marked EM (Enhanced Monitoring). Supabase sources are node_exporter metrics without the `node_` prefix.

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
| `host_disk_read_iops` | operations/s | ReadIOPS | rate of `disk_reads_completed_total`, all disks, per second of `time_seconds` | rate of `DiskReads_Total` |
| `host_disk_write_iops` | operations/s | WriteIOPS | rate of `disk_writes_completed_total`, all disks, per second of `time_seconds` | rate of `DiskWrites_Total` |
| `host_volume_read_iops` | operations/s | VolumeReadIOPs / 300 (Aurora cluster) | – | – |
| `host_volume_write_iops` | operations/s | VolumeWriteIOPs / 300 (Aurora cluster) | – | – |
| `host_network_receive_bytes_per_second` | bytes/s | NetworkReceiveThroughput | rate of `network_receive_bytes_total`, without `lo`, per second of `time_seconds` | rate of `NetworkReceiveBytes_Total` |
| `host_network_transmit_bytes_per_second` | bytes/s | NetworkTransmitThroughput | rate of `network_transmit_bytes_total`, without `lo`, per second of `time_seconds` | rate of `NetworkTransmitBytes_Total` |

`tests/grafana_dashboards/test_host_row.py` fails if this table and the names the providers write drift apart.

## Dashboard

Dashboard 1 has one collapsed **Host** row, the last on the page. It has five panels: CPU utilization, Memory, Storage, Disk IOPS /s and Network /s. Each panel reads `host_*` with the dashboard's cluster and node selectors and charts at least one family that every provider writes, so none of them is empty for RDS, Supabase or ClickHouse. A self-managed node writes no `host_*`, and its row stays closed and empty. The other families are not every provider's, so they are stored and queryable but not charted. Series no panel reads are not collected: GetMetricData bills per metric.

## Limits

- Supabase: only the primary (`supabase_identifier` = `supabase_project_ref`) is mapped. Read replicas are relayed but get no `host_*`. The relay serves one project, so a box has one Supabase scrape job: the last Supabase target added owns it, and it keeps only that project's series.
- Supabase disk IOPS are summed over all block devices, including the root disk.
- Supabase serves an exposition sampled up to a minute before the scrape, and with one 60 s scraper it refreshes only every other scrape. So the CPU shares are ratios of the CPU counters, and the per-second rates are divided by the rate of the host's own `node_time_seconds`. A plain `rate()` swings between 2/3 and 4/3 of the true value, and `1 − idle rate` went negative on a live project.
- ClickHouse: the raw names and units match a response recorded from a live service (`cli/test/fixtures/clickhouse-prometheus.txt`). Its HELP text says `DiskReads_Total` and `DiskWrites_Total` count operations. The endpoint refreshes once a minute. A new service serves no counters (CPU, disk, network) for its first few minutes, so those panels start a little later than memory and storage.
- ClickHouse network bytes include loopback. On a live r6gd.medium with light pgbench load they were about 2.5 times the NIC traffic in `/proc/net/dev`, and loopback made up the difference. Memory is relative to `MemoryLimitBytes` (8 GiB on r6gd.medium), not the guest's `MemTotal` (7.6 GiB), so available memory reads about 5 % higher than `MemAvailable`.
- The Helm chart ships the same dashboard but has no `vmalert` and no host collectors, so its Host row is empty.
