# Host metrics vocabulary

Every managed-Postgres provider we collect host metrics from lands in pgai-mon's store under the same `host_*` names, in the same units, with `cluster` and `node_name` equal to the tags of the target in `instances.yml`. `host_provider_info` also carries a fixed `provider` label (`Amazon RDS`, `Supabase`, or `ClickHouse`) for safe dashboard branding; it never contains an account, organization, service, instance, host, or credential identifier. Dashboards, alerts and sizing rules read `host_*` only. They never read a provider's raw series.

| Provider | Raw source | How it becomes `host_*` | Labels from |
| --- | --- | --- | --- |
| RDS / Aurora | CloudWatch, Enhanced Monitoring, Performance Insights | pgai-mon polls RDS and writes `host_*` directly | `.env`: `PGAI_CLUSTER`, `PGAI_NODE_NAME` (with `RDS_DB_INSTANCE_IDENTIFIER`, `AWS_REGION`) |
| Supabase | node_exporter families from `/customer/v1/privileged/metrics`, relayed by `instance-jobs` (job `supabase-host-metrics`) | recording rules in `mon/sql/host_rules.sql` | the scrape file `host-metrics/supabase-<name>.yml`, while `PGAI_SUPABASE_HOST_METRICS` is true |
| ClickHouse Managed Postgres | `PostgresServer_*` from the ClickHouse Cloud `/prometheus` endpoint (jobs `clickhouse-<name>`) | recording rules in `mon/sql/host_rules.sql` | the scrape file `host-metrics/clickhouse-<name>.yml` |

`mon targets add` (and `mon local-install --db-url`, which goes through it) copies the target's tags into the last column, so they are set in one place. pgai-mon re-reads the scrape files in `host-metrics/` by itself (every 10 s); nothing is restarted. For RDS, `mon targets add` writes `RDS_DB_INSTANCE_IDENTIFIER`, `AWS_REGION`, `PGAI_CLUSTER` and `PGAI_NODE_NAME` to `.env`, and `mon targets add`/`remove` recreate pgai-mon when it is running so it reads them. It polls one instance: the last RDS instance endpoint added. A cluster, reader or proxy endpoint names no instance and sets nothing. After editing a target's tags in `instances.yml`, re-run `mon targets add` for it.

The recording rules live in `mon/sql/host_rules.sql` (formerly vmalert's `old/config/prometheus/host_rules.yml`). pgai-mon evaluates them every 60 s over 3-minute rate windows and writes the results back to its store as `host_*` series; it does so when scrape files are configured. RDS needs no rules: pgai-mon writes its `host_*` directly. Their golden evaluation is `mon/tests/host_rules.rs`, run by CI's `mon:test` job.

## Names

A dash means the provider does not expose the value. RDS sources are CloudWatch `AWS/RDS` metrics unless marked EM (Enhanced Monitoring) or PI (Performance Insights). Supabase sources are node_exporter metrics without the `node_` prefix.

| Name | Unit | RDS / Aurora | Supabase | ClickHouse |
| --- | --- | --- | --- | --- |
| `host_provider_info` | identity | constant 1, provider `Amazon RDS` | constant 1, provider `Supabase` | constant 1, provider `ClickHouse` |
| `host_cpu_utilization_percent` | percent | CPUUtilization | 100 × (1 − idle share of `cpu_seconds_total`) | 100 × (1 − idle share of `CPUSeconds_Total`) |
| `host_os_cpu_percent` | percent | EM cpuUtilization.total | – | – |
| `host_os_cpu_iowait_percent` | percent | EM cpuUtilization.wait | iowait share of `cpu_seconds_total` | iowait share of `CPUSeconds_Total` |
| `host_os_cpu_user_percent` | percent | EM cpuUtilization.user | user share of `cpu_seconds_total` | user share of `CPUSeconds_Total` |
| `host_os_cpu_system_percent` | percent | EM cpuUtilization.system | system share of `cpu_seconds_total` | system share of `CPUSeconds_Total` |
| `host_os_cpu_steal_percent` | percent | EM cpuUtilization.steal | steal share of `cpu_seconds_total` | steal share of `CPUSeconds_Total` |
| `host_os_load1` | load | EM loadAverageMinute.one | – | – |
| `host_memory_available_bytes` | bytes | FreeableMemory | `memory_MemAvailable_bytes` | MemoryLimitBytes × (1 − MemoryUsedPercent / 100) |
| `host_os_memory_total_bytes` | bytes | EM memory.total | `memory_MemTotal_bytes` | MemoryLimitBytes |
| `host_os_memory_free_bytes` | bytes | EM memory.free | `memory_MemFree_bytes` | – |
| `host_os_memory_cached_bytes` | bytes | EM memory.cached | `memory_Cached_bytes` | MemoryLimitBytes × MemoryCachePercent / 100 |
| `host_os_memory_buffers_bytes` | bytes | EM memory.buffers | `memory_Buffers_bytes` | – |
| `host_os_swap_used_bytes` | bytes | EM swap.total − swap.free | `memory_SwapTotal_bytes` − `memory_SwapFree_bytes` | – |
| `host_os_process_max_rss_bytes` | bytes | EM max process RSS | – | – |
| `host_disk_free_bytes` | bytes | FreeStorageSpace (RDS) | `filesystem_avail_bytes` on `/data` | StorageLimitBytes × (1 − FilesystemUsedPercent / 100) |
| `host_disk_size_bytes` | bytes | DescribeDBInstances AllocatedStorage × 2^30 (RDS) | `filesystem_size_bytes` on `/data` | StorageLimitBytes |
| `host_disk_max_size_bytes` | bytes | DescribeDBInstances MaxAllocatedStorage × 2^30 (RDS, storage autoscaling on) | – | – |
| `host_local_storage_free_bytes` | bytes | FreeLocalStorage (Aurora) | – | – |
| `host_volume_used_bytes` | bytes | VolumeBytesUsed (Aurora cluster) | – | – |
| `host_disk_read_iops` | operations/s | ReadIOPS | rate of `disk_reads_completed_total`, all disks, per second of `time_seconds` | rate of `DiskReads_Total` |
| `host_disk_write_iops` | operations/s | WriteIOPS | rate of `disk_writes_completed_total`, all disks, per second of `time_seconds` | rate of `DiskWrites_Total` |
| `host_disk_provisioned_iops` | operations/s | DescribeDBInstances Iops (RDS gp3, io1, io2) | – | – |
| `host_disk_read_bytes_per_second` | bytes/s | ReadThroughput | rate of `disk_read_bytes_total`, all disks, per second of `time_seconds` | – |
| `host_disk_write_bytes_per_second` | bytes/s | WriteThroughput | rate of `disk_written_bytes_total`, all disks, per second of `time_seconds` | – |
| `host_disk_provisioned_throughput_bytes_per_second` | bytes/s | DescribeDBInstances StorageThroughput × 2^20 (RDS gp3) | – | – |
| `host_disk_read_latency_seconds` | seconds | ReadLatency | rate of `disk_read_time_seconds_total` / rate of `disk_reads_completed_total`, all disks | – |
| `host_disk_write_latency_seconds` | seconds | WriteLatency | rate of `disk_write_time_seconds_total` / rate of `disk_writes_completed_total`, all disks | – |
| `host_volume_read_iops` | operations/s | VolumeReadIOPs / 300 (Aurora cluster) | – | – |
| `host_volume_write_iops` | operations/s | VolumeWriteIOPs / 300 (Aurora cluster) | – | – |
| `host_network_receive_bytes_per_second` | bytes/s | NetworkReceiveThroughput | rate of `network_receive_bytes_total`, without `lo`, per second of `time_seconds` | rate of `NetworkReceiveBytes_Total` |
| `host_network_transmit_bytes_per_second` | bytes/s | NetworkTransmitThroughput | rate of `network_transmit_bytes_total`, without `lo`, per second of `time_seconds` | rate of `NetworkTransmitBytes_Total` |
| `host_db_load` | average active sessions | PI `db.load.avg`, 60 s | – | – |
| `host_instance_info` | identity | constant 1; labels `instance_class`, `engine`, `storage_type`, `multi_az`, `region` from DescribeDBInstances | – | – |
| `host_cpu_cores` | vCPUs | – | – | CPUCores |
| `host_cpu_credit_balance` | credits | CPUCreditBalance, 300 s (db.t*) | – | – |
| `host_cpu_surplus_credit_balance` | credits | CPUSurplusCreditBalance, 300 s (db.t*) | – | – |
| `host_disk_burst_balance_percent` | percent | BurstBalance, 300 s (gp2) | – | – |
| `host_ebs_io_balance_percent` | percent | EBSIOBalance%, 300 s (RDS) | – | – |
| `host_ebs_byte_balance_percent` | percent | EBSByteBalance%, 300 s (RDS) | – | – |

## Dashboard

Dashboard 1 has one collapsed **Host** row, second to last on the page. It starts with a provider identity panel, then CPU utilization, Memory and Storage. Disk read IOPS, disk write IOPS, network receive and network transmit each have their own full-width chart so series with different operational limits are never overlaid. Each panel reads `host_*` with the dashboard's cluster and node selectors and charts at least one family that every provider writes, so none is empty for RDS, Supabase or ClickHouse. A self-managed node writes no `host_*`, and its row stays closed and empty. The other families are not every provider's, so they are stored and queryable but not charted. pgai-mon's RDS poller requests a CloudWatch metric only when a panel or the A014 resource sizing report reads it: GetMetricData bills per metric. The instance info (`host_instance_info`), the autoscaling ceiling (`host_disk_max_size_bytes`), ClickHouse's `host_cpu_cores` and the CPU credit and EBS burst balances (`host_cpu_credit_balance`, `host_cpu_surplus_credit_balance`, `host_disk_burst_balance_percent`, `host_ebs_io_balance_percent`, `host_ebs_byte_balance_percent`) feed A014's observed size and series and are not charted.

DB load (`host_db_load`, RDS and Aurora with Performance Insights on) has no Supabase or ClickHouse equivalent, so it is not in the Host row. It has its own collapsed row after Host, **RDS DB load (Performance Insights)**. Grafana 12.3 cannot hide a row that has no data (a row repeated over an empty variable is still rendered once), so on Supabase, ClickHouse, self-managed nodes and RDS without Performance Insights this row stays closed and empty.

Dashboard **99. Host stats** (uid `host-stats`) is the detailed host page. Its panels are open; **RDS DB load (Performance Insights)** is a collapsed row below them. The cluster and node selectors read `host_*` series only, so a node does not need Postgres metrics to show there. A self-managed node writes no `host_*`, so the selectors do not list it and every panel is empty.

- Series that add up to a whole are stacked. CPU modes add up to 100 %; memory components add up to total RAM; used + free storage add up to the volume size.
- Read and write are never on one panel. IOPS, throughput and latency each have a read panel and a write panel.
- A limit is a dashed red line that is not stacked, drawn only where the source reports it. No line means the limit is unknown. RDS provisioned IOPS and throughput are shared by reads and writes, so the same line is on both panels.

What each panel shows per source (EM = Enhanced Monitoring):

| Panel | RDS | Aurora | Supabase | ClickHouse |
| --- | --- | --- | --- | --- |
| CPU utilization (stacked to 100 %) | with EM: user, system, iowait, steal, other, idle; without EM: busy (no mode split), idle | same as RDS | user, system, iowait, steal, other, idle | user, system, iowait, steal, other, idle |
| Memory (stacked to total RAM) | with EM: used, buffers, cached, free, total RAM, swap used, largest process RSS; without EM: available only, total unknown | same as RDS | used, buffers, cached, free, total RAM, swap used | used, available, total RAM |
| Storage (stacked to size) | used, free, limit: size (AllocatedStorage) | local free, volume used; size limit unknown | used, free, limit: size (`/data`) | used, free, limit: size (StorageLimitBytes) |
| Network /s | receive, transmit | receive, transmit | receive, transmit | receive, transmit (includes loopback) |
| Disk read / write IOPS /s | read / write; limit: provisioned IOPS for gp3, io1, io2 (unknown for gp2 and magnetic) | read / write, volume read / write; limit unknown | read / write; limit unknown | read / write; limit unknown |
| Disk read / write throughput /s | read / write; limit: provisioned throughput for gp3 (unknown otherwise) | read / write; limit unknown | read / write; limit unknown | not exposed |
| Disk read / write latency | read / write | read / write | read / write | not exposed |

Dashboard 1's Host row shows the same CPU utilization, Memory and Storage panels; only their position and size differ. Its other host panels are unchanged. Every series on these three panels has a fixed colour, the same on both dashboards, so a series keeps its colour whichever other series a provider reports.

## Limits

- Supabase: only the primary (`supabase_identifier` = `supabase_project_ref`) is mapped. Read replicas are relayed but get no `host_*`. The relay serves one project, so a box has one Supabase scrape job: the last Supabase target added owns it, and it keeps only that project's series.
- Supabase disk IOPS, throughput and latency are summed (latency: weighted) over all block devices, including the root disk. That is deliberate: a Supabase compute size's IOPS and MB/s limits are the instance's EBS limits, which cap all attached volumes together, so the root disk's I/O counts against them. The recorded Supabase fixture holds no `disk_read_bytes_total`, `disk_written_bytes_total`, `disk_read_time_seconds_total`, `disk_write_time_seconds_total` or `memory_Buffers_bytes`; the rules use node_exporter's standard names and are not yet checked against a live project.
- Supabase serves an exposition sampled up to a minute before the scrape, and with one 60 s scraper it refreshes only every other scrape. So the CPU shares are ratios of the CPU counters, and the per-second rates are divided by the rate of the host's own `node_time_seconds`. A plain `rate()` swings between 2/3 and 4/3 of the true value, and `1 − idle rate` went negative on a live project.
- ClickHouse: the raw names and units match a response recorded from a live service (`cli/test/fixtures/clickhouse-prometheus.txt`). Its HELP text says `DiskReads_Total` and `DiskWrites_Total` count operations. The endpoint refreshes once a minute. A new service serves no counters (CPU, disk, network) for its first few minutes, so those panels start a little later than memory and storage.
- ClickHouse network bytes include loopback. On a live r6gd.medium with light pgbench load they were about 2.5 times the NIC traffic in `/proc/net/dev`, and loopback made up the difference. Memory is relative to `MemoryLimitBytes` (8 GiB on r6gd.medium), not the guest's `MemTotal` (7.6 GiB), so available memory reads about 5 % higher than `MemAvailable`.
