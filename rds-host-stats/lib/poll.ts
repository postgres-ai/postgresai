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

export const requestHandler = { connectionTimeout: 5_000, requestTimeout: 10_000, throwOnRequestTimeout: true }

export function createClients(region: string, role?: Role): Clients {
  const credentials = role ? fromTemporaryCredentials({
    params: { RoleArn: role.arn, ExternalId: role.externalId, RoleSessionName: 'postgresai-rds-host-stats', DurationSeconds: 3600 },
    clientConfig: { region, requestHandler },
  }) : undefined
  const config = { region, credentials, requestHandler }
  return { rds: new RDSClient(config), cloudwatch: new CloudWatchClient(config), pi: new PIClient(config), logs: new CloudWatchLogsClient(config) }
}

type Metric = [name: string, source: string, scale?: number]
const common: Metric[] = [
  ['host_cpu_utilization_percent', 'CPUUtilization'],
  ['host_memory_available_bytes', 'FreeableMemory'],
  ['host_disk_read_iops', 'ReadIOPS'],
  ['host_disk_write_iops', 'WriteIOPS'],
  ['host_network_receive_bytes_per_second', 'NetworkReceiveThroughput'],
  ['host_network_transmit_bytes_per_second', 'NetworkTransmitThroughput'],
]
const rdsMetrics: Metric[] = [
  ['host_disk_free_bytes', 'FreeStorageSpace'],
]
const auroraMetrics: Metric[] = [
  ['host_local_storage_free_bytes', 'FreeLocalStorage'],
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

export class PollTimeout extends Error {
  name = 'PollTimeout'
}

// text: the import body of the samples not yet written. errors: sources that
// failed this poll (Performance Insights, Enhanced Monitoring); their samples
// are retried on the next poll, the rest are kept. A malformed Enhanced
// Monitoring event is reported once and never retried: it stays malformed.
export type Poll = { text: string; errors: Error[] }

// requestTimeout stops at the response headers; a stalled body would otherwise hang the loop.
export async function pollOnce(clients: Clients, target: Target, now: Date, state: Map<string, number>): Promise<Poll> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new PollTimeout('poll timed out after 30 s')), 30_000)
  })
  try {
    return await Promise.race([poll(clients, target, now, state), deadline])
  } finally {
    clearTimeout(timer)
  }
}

const toError = (e: unknown) => (e instanceof Error ? e : new Error(String(e)))

async function poll(clients: Clients, target: Target, now: Date, state: Map<string, number>): Promise<Poll> {
  const description: DescribeDBInstancesCommandOutput = await clients.rds.send(new DescribeDBInstancesCommand({ DBInstanceIdentifier: target.instanceId }))
  const db = description.DBInstances?.[0]
  if (!db) throw new Error('RDS instance not found')
  const aurora = db.Engine === 'aurora-postgresql'
  // CloudWatch and PI buckets are queried over the 15 minutes up to the last
  // minute boundary, so no bucket is cut by the wall clock. A bucket is written
  // only once a further period has passed after it closed: a partial Average
  // written early would be frozen by the per-timestamp dedupe, and the complete
  // one dropped. CloudWatch stamps a bucket at its start, PI at its end, so the
  // gate is 2 periods for CloudWatch and 1 for PI. The OS log window follows
  // the clock; its events are samples.
  const end = new Date(Math.floor(now.getTime() / 60_000) * 60_000)
  const start = new Date(end.getTime() - 15 * 60_000)
  const closed = (time: number, period: number) => time + 2 * period * 1000 <= end.getTime()
  const closedPI = (time: number) => time + 60_000 <= end.getTime()
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
  const errors: Error[] = []
  const labels = `{cluster="${escapeLabel(target.cluster)}",node_name="${escapeLabel(target.nodeName)}"}`
  for (const [key, time] of state) if (time < start.getTime() - 15 * 60_000) state.delete(key)
  // Only finite numbers reach the import body: a value from a log event is
  // untrusted text, and a string with a newline would add lines of its own.
  const emit = (name: string, value: unknown, time: number) => {
    if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(time)) return
    const key = `${name}${labels} ${time}`
    if (!state.has(key)) {
      text += `${name}${labels} ${value} ${time}\n`
      state.set(key, time)
    }
  }
  const result: GetMetricDataCommandOutput = await clients.cloudwatch.send(new GetMetricDataCommand({ StartTime: start, EndTime: end, MetricDataQueries }))
  for (const [name, , scale = 1] of metrics) {
    const data = result.MetricDataResults?.find(r => r.Id === name)
    const period = MetricDataQueries.find(q => q.Id === name)?.MetricStat.Period ?? 60
    const points = (data?.Timestamps ?? []).map((time, i) => ({ time: time.getTime(), value: data?.Values?.[i] }))
    for (const point of points.sort((a, b) => a.time - b.time)) {
      if (typeof point.value === 'number' && closed(point.time, period)) emit(name, point.value * scale, point.time)
    }
  }
  if (db.PerformanceInsightsEnabled) {
    try {
      const result: GetResourceMetricsCommandOutput = await clients.pi.send(new GetResourceMetricsCommand({
        ServiceType: 'RDS', Identifier: db.DbiResourceId, StartTime: start, EndTime: end,
        PeriodInSeconds: 60, MetricQueries: [{ Metric: 'db.load.avg' }],
      }))
      // PI stamps a point at the end of its minute and rounds EndTime up, so a
      // point after `end` covers a minute that is still open, and the point at
      // `end` a minute that closed moments ago and may still be ingesting.
      for (const metric of result.MetricList ?? []) {
        for (const point of metric.DataPoints ?? []) {
          if (point.Timestamp && closedPI(point.Timestamp.getTime())) emit('host_db_load', point.Value, point.Timestamp.getTime())
        }
      }
    } catch (error) {
      errors.push(toError(error))
    }
  }
  if ((db.MonitoringInterval ?? 0) > 0) {
    try {
      const result: GetLogEventsCommandOutput = await clients.logs.send(new GetLogEventsCommand({
        logGroupName: 'RDSOSMetrics', logStreamName: db.DbiResourceId, startTime: now.getTime() - 15 * 60_000, endTime: now.getTime(),
      }))
      const events: OSMetric[] = []
      for (const event of result.events ?? []) {
        try {
          const os = JSON.parse(event.message ?? '')
          if (typeof os?.timestamp !== 'string' || !Number.isFinite(Date.parse(os.timestamp))) throw new Error('no timestamp')
          events.push(os)
        } catch (error) {
          // The 15-minute log window returns the same event on every poll;
          // report it once. Remembered in state like a written sample, so
          // it is pruned with them and re-reported only if the write failed.
          const key = `skipped ${event.timestamp} ${event.message}`
          if (!state.has(key)) {
            state.set(key, event.timestamp ?? now.getTime())
            errors.push(new Error(`Enhanced Monitoring event at ${event.timestamp} skipped: ${toError(error).message}`))
          }
        }
      }
      const at = (os: OSMetric) => new Date(os.timestamp).getTime()
      for (const [name, value] of osMetrics) {
        for (const os of events) {
          try {
            emit(name, value(os), at(os))
          } catch {
            // a field missing from this event; the series stays empty for it
          }
        }
      }
      for (const os of events) {
        // id 0 rows are the "OS processes" and "RDS processes" aggregates.
        const rss = Array.isArray(os.processList) ? os.processList.filter(p => p && p.id !== 0 && typeof p.rss === 'number').map(p => p.rss) : []
        if (rss.length) emit('host_os_process_max_rss_bytes', Math.max(...rss) * 1024, at(os))
      }
    } catch (error) {
      errors.push(toError(error))
    }
  }
  return { text, errors }
}

export async function writeSamples(url: string, text: string, auth?: Auth): Promise<void> {
  if (!text) return
  const headers: Record<string, string> = { 'Content-Type': 'text/plain' }
  if (auth) headers.Authorization = `Basic ${Buffer.from(`${auth.username}:${auth.password}`).toString('base64')}`
  const response = await fetch(`${url}/api/v1/import/prometheus`, { method: 'POST', headers, body: text, signal: AbortSignal.timeout(10_000) })
  if (!response.ok) throw new Error(`Prometheus import failed: HTTP ${response.status}`)
}
