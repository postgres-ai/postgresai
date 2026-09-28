# RDS host stats

We poll one RDS or Aurora PostgreSQL instance every 60 seconds (`RDS_POLL_INTERVAL_SECONDS`) from the monitoring VM. We read a 15-minute window, deduplicate timestamps, and send `host_*` series to VictoriaMetrics with the pgwatch `cluster` and `node_name` labels.

CloudWatch and Performance Insights are queried up to the last minute boundary, and a bucket is written only once a further period has passed after it closed (about 1–2 minutes behind the clock for 60 s metrics, 5–10 minutes for the 5-minute Aurora volume metrics). A bucket written while CloudWatch is still aggregating it would be frozen by the per-timestamp dedupe. Enhanced Monitoring events are individual samples and are written as soon as they are read.

## Metrics

CloudWatch sources below use `AWS/RDS`; OS sources use Enhanced Monitoring in `RDSOSMetrics`. We collect PI and OS metrics only when enabled on the instance. Each poll reads the newest GetLogEvents page (up to 1 MB); at Enhanced Monitoring intervals of 1–5 s, OS samples missed while the service was down are not backfilled.

| Series | Source | Unit |
| --- | --- | --- |
| host_cpu_utilization_percent | CPUUtilization | percent |
| host_memory_available_bytes | FreeableMemory | bytes |
| host_disk_read_iops | ReadIOPS | operations/s |
| host_disk_write_iops | WriteIOPS | operations/s |
| host_disk_read_latency_seconds | ReadLatency | seconds |
| host_disk_write_latency_seconds | WriteLatency | seconds |
| host_disk_queue_depth | DiskQueueDepth | requests |
| host_network_receive_bytes_per_second | NetworkReceiveThroughput | bytes/s |
| host_network_transmit_bytes_per_second | NetworkTransmitThroughput | bytes/s |
| host_disk_free_bytes | FreeStorageSpace (RDS) | bytes |
| host_burst_balance_percent | BurstBalance (RDS) | percent |
| host_ebs_io_balance_percent | EBSIOBalance% (RDS) | percent |
| host_replica_lag_seconds | ReplicaLag (RDS), AuroraReplicaLag × 0.001 (Aurora) | seconds |
| host_local_storage_free_bytes | FreeLocalStorage (Aurora) | bytes |
| host_volume_used_bytes | VolumeBytesUsed (Aurora cluster) | bytes |
| host_volume_read_iops | VolumeReadIOPs / 300 (Aurora cluster) | operations/s |
| host_volume_write_iops | VolumeWriteIOPs / 300 (Aurora cluster) | operations/s |
| host_db_load | PI db.load.avg | average active sessions |
| host_os_cpu_percent | OS cpuUtilization.total | percent |
| host_os_cpu_iowait_percent | OS cpuUtilization.wait | percent |
| host_os_load1 | OS loadAverageMinute.one | load |
| host_os_memory_total_bytes | OS memory.total × 1024 | bytes |
| host_os_memory_free_bytes | OS memory.free × 1024 | bytes |
| host_os_memory_cached_bytes | OS memory.cached × 1024 | bytes |
| host_os_swap_used_bytes | OS (swap.total − swap.free) × 1024 | bytes |
| host_os_process_max_rss_bytes | OS max(processList.rss), excluding id 0, × 1024 | bytes |

## Run

We use `docker compose --profile rds up -d rds-host-stats`. Set `PGAI_TAG`, `RDS_DB_INSTANCE_IDENTIFIER`, `AWS_REGION`, `PGAI_CLUSTER`, and `PGAI_NODE_NAME`; the last two must match the pgwatch target labels. Set both `RDS_ROLE_ARN` and `RDS_EXTERNAL_ID` to assume a customer role, or neither to use the default AWS credential chain. Base credentials come from the VM instance profile; containers reach IMDSv2 only when the instance's metadata hop limit is at least 2.

`PROMETHEUS_URL` defaults to `http://sink-prometheus:9090` (also fixed in compose). We use basic auth when both `VM_AUTH_USERNAME` and `VM_AUTH_PASSWORD` are set. Compose limits default to `RDS_HOST_STATS_CPUS=0.1` and `RDS_HOST_STATS_MEM=134217728` bytes. Failed polls are logged and retried on the next tick; samples of a failed VictoriaMetrics write are sent again with the next poll. A Performance Insights or Enhanced Monitoring failure, or a malformed Enhanced Monitoring event, is logged and skipped for that poll while the CloudWatch samples are still written. A poll that exceeds 30 s exits the service with status 1, and the compose restart policy starts it again without the stuck connection. SIGTERM/SIGINT exit cleanly.

## Customer IAM

The customer creates the role with the PostgresAI RDS provider template (platform-all `terraform/rds-privatelink-provider` or its CloudFormation twin). It allows exactly four read-only calls, scoped to the one instance where AWS supports it: `cloudwatch:GetMetricData`, `logs:GetLogEvents` on the instance's `RDSOSMetrics` stream, `pi:GetResourceMetrics` and `rds:DescribeDBInstances`. The trust policy admits the PostgresAI AWS account only with `sts:ExternalId` = `postgresai-org-<org id>`, and sessions last at most one hour. We pass that role as `RDS_ROLE_ARN` and the External ID as `RDS_EXTERNAL_ID`; the assumed credentials stay in this process's memory.

## Customer cost and Enhanced Monitoring

GetMetricData costs $0.01 per 1,000 metrics requested: 13 RDS metrics × 1,440 polls/day × 30 days ≈ $5.60/month per instance (Aurora requests 14 metrics). A longer `RDS_POLL_INTERVAL_SECONDS` lowers this in proportion; the 15-minute window still covers every bucket up to 13 minutes. Logs GetLogEvents and PI Standard API reads are free. AWS bills Enhanced Monitoring's own Logs ingestion when the customer enables it.

We enable Enhanced Monitoring with `aws rds modify-db-instance --db-instance-identifier <id> --monitoring-interval 15 --monitoring-role-arn <arn> --apply-immediately`. The monitoring role must allow RDS Enhanced Monitoring to publish OS metrics to CloudWatch Logs.

## Tests

From `rds-host-stats/`, run `bun install --frozen-lockfile`, `bun test`, and `bunx tsc --noEmit`. Fixture tests run offline; the live e2e is skipped unless its environment variables are set.

To re-record against AWS, run `bun test/record.ts test/fixtures/<case>` with authorized AWS credentials; the recorder redacts endpoint addresses, the account ID in ARNs, VPC/subnet/security-group IDs and KMS key IDs. For the live e2e, see the command and prerequisites in the header of `test/e2e.test.ts`; it needs a real instance with PI and Enhanced Monitoring, plus VictoriaMetrics.
