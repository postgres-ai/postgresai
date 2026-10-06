# RDS host stats

We poll one RDS or Aurora PostgreSQL instance every 60 seconds (`RDS_POLL_INTERVAL_SECONDS`) from the monitoring VM. We read a 15-minute window, deduplicate timestamps, and send `host_*` series to VictoriaMetrics with the pgwatch `cluster` and `node_name` labels.

CloudWatch and Performance Insights are queried up to the last minute boundary, and a bucket is written only once a further period has passed after it closed (about 1–2 minutes behind the clock for 60 s metrics, 5–10 minutes for the 5-minute Aurora volume metrics). A bucket written while CloudWatch is still aggregating it would be frozen by the per-timestamp dedupe. Enhanced Monitoring events are individual samples and are written as soon as they are read.

## Metrics

CloudWatch sources below use `AWS/RDS`; OS sources use Enhanced Monitoring in `RDSOSMetrics`; DB load (`host_db_load`) is Performance Insights `db.load.avg` at 60 s. We collect PI and OS metrics only when enabled on the instance. Each poll reads the newest GetLogEvents page (up to 1 MB); at Enhanced Monitoring intervals of 1–5 s, OS samples missed while the service was down are not backfilled.

The series, their sources and units are listed with the other providers' in [docs/host-metrics.md](../docs/host-metrics.md).

## Run

We use `docker compose --profile rds up -d rds-host-stats`. `postgresai mon targets add` with the instance endpoint writes `RDS_DB_INSTANCE_IDENTIFIER`, `AWS_REGION`, `PGAI_CLUSTER` and `PGAI_NODE_NAME` to `.env` from the target (see [docs/host-metrics.md](../docs/host-metrics.md)). Set both `RDS_ROLE_ARN` and `RDS_EXTERNAL_ID` to assume a read-only role in the customer's account (on PostgresAI-hosted collectors provisioning sets them; see Customer IAM), or neither to use the default AWS credential chain. Base credentials come from the VM instance profile; containers reach IMDSv2 only when the instance's metadata hop limit is at least 2.

`PROMETHEUS_URL` defaults to `http://sink-prometheus:9090` (also fixed in compose). We use basic auth when both `VM_AUTH_USERNAME` and `VM_AUTH_PASSWORD` are set. Compose limits default to `RDS_HOST_STATS_CPUS=0.5` and `RDS_HOST_STATS_MEM=134217728` bytes. At 0.1 CPU the cold first AWS call took 1.2 s instead of 0.28 s, and on a smaller VM every poll hit the 5 s connect timeout. A missing or invalid variable is logged once and the service idles instead of exiting, so the restart policy does not loop. An idling container still shows as `Up` and has no healthcheck, so when `host_*` series are missing, read `docker logs rds-host-stats`. Fix the variable in `.env` and recreate the container with `docker compose --profile rds up -d rds-host-stats`; `docker compose restart` reuses the environment recorded when the container was created, so it does not pick up the change. Failed polls are logged and retried on the next tick; samples of a failed VictoriaMetrics write are sent again with the next poll. A Performance Insights or Enhanced Monitoring failure is logged and skipped for that poll while the CloudWatch samples are still written. A malformed Enhanced Monitoring event is logged once and skipped. A poll that exceeds 30 s exits the service with status 1, and the compose restart policy starts it again without the stuck connection. SIGTERM/SIGINT exit cleanly.

## Customer IAM

This section describes the PostgresAI-hosted flow. The console field, the platform-issued External ID and the collector instance profile are provisioned outside this repository (platform-all!883, postgres_ai-ansible!50, infra!68).

The customer creates the role with the PostgresAI RDS provider template (platform-all `terraform/rds-privatelink-provider` or its CloudFormation twin). We make four read-only calls: `cloudwatch:GetMetricData`, `logs:GetLogEvents` on the instance's `RDSOSMetrics` stream, `pi:GetResourceMetrics` on the instance's `metrics/rds/<DbiResourceId>`, and `rds:DescribeDBInstances`. The trust policy admits only our collector role, `arn:aws:iam::005923036815:role/pgai-rds-host-stats-collector`, and only with `sts:ExternalId` equal to the organization's External ID: `postgresai-` plus 32 random hex characters, issued and stored by the platform and shown in the console setup. Sessions last at most one hour.

The customer pastes the stack's `HostStatsRoleArn` output into the console. Provisioning passes it as `RDS_ROLE_ARN`, and the platform reads the External ID for the organization itself and passes it as `RDS_EXTERNAL_ID`; the browser never sends it. The collector EC2 runs with the `pgai-rds-host-stats-collector` instance profile, which may only assume other accounts' roles with an External ID of that shape. Its credentials, read from IMDSv2 (hop limit 2, so the container can reach it), are the source credentials for `sts:AssumeRole`. The assumed credentials stay in this process's memory.

A self-hosted operator who wants role mode creates the role in the monitored account with those four permissions, writes a trust policy that names the principal the monitoring VM runs as and requires an External ID of their choosing, and sets `RDS_ROLE_ARN` and `RDS_EXTERNAL_ID` in `.env`. The VM's default credential chain provides the source credentials for `sts:AssumeRole`. Without a role, the VM's own credentials need those four permissions directly.

## Customer cost and Enhanced Monitoring

GetMetricData costs $0.01 per 1,000 metrics requested: 7 RDS metrics × 1,440 polls/day × 30 days ≈ $3.02/month per instance (Aurora requests 10 metrics, ≈ $4.32). A longer `RDS_POLL_INTERVAL_SECONDS` (at most 300) lowers this in proportion. A bucket is written by a poll that ends between two periods and 15 minutes after the bucket starts: a span of 13 minutes for the 60 s metrics, and of 5 minutes for the 300 s Aurora volume metrics, which is why the interval is capped. Logs GetLogEvents reads are free. Performance Insights adds one GetResourceMetrics call per poll, 1,440 × 30 = 43,200 calls/month per instance; AWS does not charge for the first 1 million Performance Insights API calls per account and month (CloudWatch pricing, Database Insights), so with nothing else in the account calling the PI API, 22 instances polled every 60 s stay inside it even in a 31-day month (44,640 calls each); calls beyond it cost $0.01 per 1,000 (≈ $0.43/month per instance). We read only the last 15 minutes, inside the free 7-day retention. AWS bills Enhanced Monitoring's own Logs ingestion when the customer enables it.

We enable Enhanced Monitoring with `aws rds modify-db-instance --db-instance-identifier <id> --monitoring-interval 15 --monitoring-role-arn <arn> --apply-immediately`. The monitoring role must allow RDS Enhanced Monitoring to publish OS metrics to CloudWatch Logs.

## Tests

From `rds-host-stats/`, run `bun install --frozen-lockfile`, `bun test`, and `bunx tsc --noEmit`. Fixture tests run offline; the live e2e is skipped unless its environment variables are set.

The recorder projects requests in order from each case's `recorded.json`; responses are replaced, so there is one source for the request envelopes. For a new case, seed `recorded.json` with an array of `{ "client": "rds", "command": "DescribeDBInstancesCommand", "input": { "DBInstanceIdentifier": "<instance>" } }` entries for the desired calls (outputs are optional before recording).

To re-record against AWS, run `bun test/record.ts test/fixtures/<case>` with authorized AWS credentials; the recorder redacts endpoint addresses, the account ID in ARNs, VPC/subnet/security-group IDs and KMS key IDs. For the live e2e, see the command and prerequisites in the header of `test/e2e.test.ts`; it needs a real instance with PI and Enhanced Monitoring, plus VictoriaMetrics.
