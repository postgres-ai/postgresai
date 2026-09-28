import { CloudWatchClient, GetMetricDataCommand, type GetMetricDataCommandOutput } from '@aws-sdk/client-cloudwatch'
import { CloudWatchLogsClient, GetLogEventsCommand, type GetLogEventsCommandOutput } from '@aws-sdk/client-cloudwatch-logs'
import { GetResourceMetricsCommand, PIClient, type GetResourceMetricsCommandOutput } from '@aws-sdk/client-pi'
import { DescribeDBInstancesCommand, RDSClient, type DescribeDBInstancesCommandOutput } from '@aws-sdk/client-rds'
import { fromTemporaryCredentials } from '@aws-sdk/credential-providers'

type Client = { send(command: any): Promise<any> }
export type Clients = { rds: Client; cloudwatch: Client; pi: Client; logs: Client }
export type Target = { instanceId: string; cluster: string; nodeName: string }
export type Role = { arn: string; externalId: string }
export type Auth = { username: string; password: string }

export function createClients(region: string, role?: Role): Clients {
  const credentials = role ? fromTemporaryCredentials({
    params: { RoleArn: role.arn, ExternalId: role.externalId, RoleSessionName: 'postgresai-rds-host-stats', DurationSeconds: 3600 },
    clientConfig: { region, requestHandler: { connectionTimeout: 5_000, requestTimeout: 10_000 } },
  }) : undefined
  const config = { region, credentials, requestHandler: { connectionTimeout: 5_000, requestTimeout: 10_000 } }
  return { rds: new RDSClient(config), cloudwatch: new CloudWatchClient(config), pi: new PIClient(config), logs: new CloudWatchLogsClient(config) }
}

type Metric = [name: string, source: string, scale?: number]
const common: Metric[] = [
  ['host_cpu_utilization_percent', 'CPUUtilization'],
  ['host_memory_available_bytes', 'FreeableMemory'],
  ['host_disk_read_iops', 'ReadIOPS'],
  ['host_disk_write_iops', 'WriteIOPS'],
  ['host_disk_read_latency_seconds', 'ReadLatency'],
  ['host_disk_write_latency_seconds', 'WriteLatency'],
  ['host_disk_queue_depth', 'DiskQueueDepth'],
  ['host_network_receive_bytes_per_second', 'NetworkReceiveThroughput'],
  ['host_network_transmit_bytes_per_second', 'NetworkTransmitThroughput'],
]
const rdsMetrics: Metric[] = [
  ['host_disk_free_bytes', 'FreeStorageSpace'],
  ['host_burst_balance_percent', 'BurstBalance'],
  ['host_ebs_io_balance_percent', 'EBSIOBalance%'],
  ['host_replica_lag_seconds', 'ReplicaLag'],
]
const auroraMetrics: Metric[] = [
  ['host_local_storage_free_bytes', 'FreeLocalStorage'],
  ['host_replica_lag_seconds', 'AuroraReplicaLag', 0.001],
]
const volumeMetrics: Metric[] = [
  ['host_volume_used_bytes', 'VolumeBytesUsed'],
  ['host_volume_read_iops', 'VolumeReadIOPs', 1 / 300],
  ['host_volume_write_iops', 'VolumeWriteIOPs', 1 / 300],
]
type OSMetric = {
  timestamp: string
  cpuUtilization: { total: number; wait: number }
  loadAverageMinute: { one: number }
  memory: { total: number; free: number; cached: number }
  swap: { total: number; free: number }
  processList?: { id: number; rss: number }[]
}
const osMetrics: [string, (m: OSMetric) => number][] = [
  ['host_os_cpu_percent', m => m.cpuUtilization.total],
  ['host_os_cpu_iowait_percent', m => m.cpuUtilization.wait],
  ['host_os_load1', m => m.loadAverageMinute.one],
  ['host_os_memory_total_bytes', m => m.memory.total * 1024],
  ['host_os_memory_free_bytes', m => m.memory.free * 1024],
  ['host_os_memory_cached_bytes', m => m.memory.cached * 1024],
  ['host_os_swap_used_bytes', m => (m.swap.total - m.swap.free) * 1024],
]
const escapeLabel = (value: string) => value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')

export async function pollOnce(clients: Clients, target: Target, now: Date, state: Map<string, number>): Promise<string> {
  const description: DescribeDBInstancesCommandOutput = await clients.rds.send(new DescribeDBInstancesCommand({ DBInstanceIdentifier: target.instanceId }))
  const db = description.DBInstances?.[0]
  if (!db) throw new Error('RDS instance not found')
  const aurora = db.Engine === 'aurora-postgresql'
  const start = new Date(now.getTime() - 15 * 60_000)
  const metrics = [...common, ...(aurora ? auroraMetrics : rdsMetrics)]
  const queries = (items: Metric[], dimension: string, value: string | undefined, period: number) => items.map(([Id, MetricName]) => ({
    Id, MetricStat: { Metric: { Namespace: 'AWS/RDS', MetricName, Dimensions: [{ Name: dimension, Value: value }] }, Period: period, Stat: 'Average' },
  }))
  const MetricDataQueries = queries(metrics, 'DBInstanceIdentifier', target.instanceId, 60)
  if (aurora) {
    MetricDataQueries.push(...queries(volumeMetrics, 'DBClusterIdentifier', db.DBClusterIdentifier, 300))
    metrics.push(...volumeMetrics)
  }
  let text = ''
  const labels = `{cluster="${escapeLabel(target.cluster)}",node_name="${escapeLabel(target.nodeName)}"}`
  for (const [key, time] of state) if (time < start.getTime() - 15 * 60_000) state.delete(key)
  const emit = (name: string, value: number, time: number) => {
    const key = `${name}${labels} ${time}`
    if (!state.has(key)) {
      text += `${name}${labels} ${value} ${time}\n`
      state.set(key, time)
    }
  }
  const result: GetMetricDataCommandOutput = await clients.cloudwatch.send(new GetMetricDataCommand({ StartTime: start, EndTime: now, MetricDataQueries }))
  for (const [name, , scale = 1] of metrics) {
    const data = result.MetricDataResults?.find(r => r.Id === name)
    const points = (data?.Timestamps ?? []).map((time, i) => ({ time: time.getTime(), value: data?.Values?.[i] }))
    for (const point of points.sort((a, b) => a.time - b.time)) {
      if (typeof point.value === 'number') emit(name, point.value * scale, point.time)
    }
  }
  if (db.PerformanceInsightsEnabled) {
    const result: GetResourceMetricsCommandOutput = await clients.pi.send(new GetResourceMetricsCommand({
      ServiceType: 'RDS', Identifier: db.DbiResourceId, StartTime: start, EndTime: now,
      PeriodInSeconds: 60, MetricQueries: [{ Metric: 'db.load.avg' }],
    }))
    for (const metric of result.MetricList ?? []) {
      for (const point of metric.DataPoints ?? []) {
        if (typeof point.Value === 'number' && point.Timestamp) emit('host_db_load', point.Value, point.Timestamp.getTime())
      }
    }
  }
  if ((db.MonitoringInterval ?? 0) > 0) {
    const result: GetLogEventsCommandOutput = await clients.logs.send(new GetLogEventsCommand({
      logGroupName: 'RDSOSMetrics', logStreamName: db.DbiResourceId, startTime: start.getTime(), endTime: now.getTime(),
    }))
    const events: OSMetric[] = (result.events ?? []).map(event => JSON.parse(event.message!))
    for (const [name, value] of osMetrics) {
      for (const event of events) emit(name, value(event), new Date(event.timestamp).getTime())
    }
    for (const event of events) {
      const processes = event.processList?.filter(p => p.id !== 0) ?? []
      if (processes.length) emit('host_os_process_max_rss_bytes', Math.max(...processes.map(p => p.rss)) * 1024, new Date(event.timestamp).getTime())
    }
  }
  return text
}

export async function writeSamples(url: string, text: string, auth?: Auth): Promise<void> {
  if (!text) return
  const headers: Record<string, string> = { 'Content-Type': 'text/plain' }
  if (auth) headers.Authorization = `Basic ${Buffer.from(`${auth.username}:${auth.password}`).toString('base64')}`
  const response = await fetch(`${url}/api/v1/import/prometheus`, { method: 'POST', headers, body: text, signal: AbortSignal.timeout(10_000) })
  if (!response.ok) throw new Error(`Prometheus import failed: HTTP ${response.status}`)
}
