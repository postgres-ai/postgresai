import { CloudWatchClient, GetMetricDataCommand, type GetMetricDataCommandOutput } from '@aws-sdk/client-cloudwatch'
import { CloudWatchLogsClient, GetLogEventsCommand, type GetLogEventsCommandOutput } from '@aws-sdk/client-cloudwatch-logs'
import { GetResourceMetricsCommand, PIClient, type GetResourceMetricsCommandOutput } from '@aws-sdk/client-pi'
import { DescribeDBInstancesCommand, RDSClient, type DescribeDBInstancesCommandOutput } from '@aws-sdk/client-rds'
import { fromTemporaryCredentials } from '@aws-sdk/credential-providers'

type Client = { send(command: any): Promise<any> }
export type Clients = { rds: Client; cloudwatch: Client; pi: Client; logs: Client }
export type Target = { instanceId: string; cluster: string; nodeName: string; region?: string }
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
  ['host_disk_read_bytes_per_second', 'ReadThroughput'],
  ['host_disk_write_bytes_per_second', 'WriteThroughput'],
  ['host_disk_read_latency_seconds', 'ReadLatency'],
  ['host_disk_write_latency_seconds', 'WriteLatency'],
]
const rdsMetrics: Metric[] = [
  ['host_disk_free_bytes', 'FreeStorageSpace'],
]
const auroraMetrics: Metric[] = [
  ['host_local_storage_free_bytes', 'FreeLocalStorage'],
]
// Credit and burst balances that limit what the instance can sustain (resource
// sizing, platform-all#1101). CloudWatch publishes them every 5 minutes, so
// they are read at most once per 5 minutes (a failed read again on the next
// poll), over the 30 minutes up to the last 5-minute boundary: CloudWatch
// aligns the buckets to the start, so every read stamps the same buckets, each
// is seen by several reads, and a failed read does not lose it. Only where they apply:
// CPU credits on t-classes, BurstBalance on gp2; the EBS balances exist only on
// classes whose EBS baseline is below their burst, and cost little elsewhere.
const creditMetrics: Metric[] = [
  ['host_cpu_credit_balance', 'CPUCreditBalance'],
  ['host_cpu_surplus_credit_balance', 'CPUSurplusCreditBalance'],
]
const gp2Metrics: Metric[] = [
  ['host_disk_burst_balance_percent', 'BurstBalance'],
]
const ebsMetrics: Metric[] = [
  ['host_ebs_io_balance_percent', 'EBSIOBalance%'],
  ['host_ebs_byte_balance_percent', 'EBSByteBalance%'],
]
const balanceEvery = 5 * 60_000
const balanceKey = 'balances read at'
const volumeMetrics: Metric[] = [
  ['host_volume_used_bytes', 'VolumeBytesUsed'],
  ['host_volume_read_iops', 'VolumeReadIOPs', 1 / 300],
  ['host_volume_write_iops', 'VolumeWriteIOPs', 1 / 300],
]
type OSMetric = {
  timestamp: string
  cpuUtilization: { total: number; wait: number; user: number; system: number; steal: number }
  loadAverageMinute: { one: number }
  memory: { total: number; free: number; cached: number; buffers: number }
  swap: { total: number; free: number }
  processList?: { id: number; rss: number }[]
}
const osMetrics: [string, (m: OSMetric) => number][] = [
  ['host_os_cpu_percent', m => m.cpuUtilization.total],
  ['host_os_cpu_iowait_percent', m => m.cpuUtilization.wait],
  ['host_os_cpu_user_percent', m => m.cpuUtilization.user],
  ['host_os_cpu_system_percent', m => m.cpuUtilization.system],
  ['host_os_cpu_steal_percent', m => m.cpuUtilization.steal],
  ['host_os_load1', m => m.loadAverageMinute.one],
  ['host_os_memory_total_bytes', m => m.memory.total * 1024],
  ['host_os_memory_free_bytes', m => m.memory.free * 1024],
  ['host_os_memory_cached_bytes', m => m.memory.cached * 1024],
  ['host_os_memory_buffers_bytes', m => m.memory.buffers * 1024],
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
  const providerLabels = `{cluster="${escapeLabel(target.cluster)}",node_name="${escapeLabel(target.nodeName)}",provider="Amazon RDS"}`
  const providerMetric = 'host_provider_info'
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
  // A fixed, non-sensitive identity series lets the dashboard name the
  // provider without displaying account, region, instance or host IDs.
  const providerKey = `${providerMetric}${providerLabels} ${end.getTime()}`
  if (!state.has(providerKey)) {
    text += `${providerMetric}${providerLabels} 1 ${end.getTime()}\n`
    state.set(providerKey, end.getTime())
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
  // What the instance is (resource sizing, platform-all#1101): one sample per
  // poll, its labels from the instance description.
  const info = {
    instance_class: db.DBInstanceClass, engine: db.Engine, storage_type: db.StorageType,
    multi_az: db.MultiAZ === undefined ? undefined : String(db.MultiAZ), region: target.region,
  }
  const infoLabels = Object.entries(info).filter(([, v]) => typeof v === 'string' && v !== '')
    .map(([k, v]) => `,${k}="${escapeLabel(v!)}"`).join('')
  const infoMetric = 'host_instance_info'
  const infoKey = `${infoMetric}{cluster="${escapeLabel(target.cluster)}",node_name="${escapeLabel(target.nodeName)}"${infoLabels}} ${end.getTime()}`
  if (!state.has(infoKey)) {
    text += `${infoKey.replace(/ \d+$/, '')} 1 ${end.getTime()}\n`
    state.set(infoKey, end.getTime())
  }
  // Storage limits from the instance description, stamped at `end` on every
  // poll. Only RDS (not Aurora) reports them: Aurora storage grows on demand
  // and has no provisioned IOPS or throughput. gp2 and magnetic storage report
  // no Iops or StorageThroughput, so their limit series stay empty.
  if (!aurora) {
    const GiB = 1024 ** 3
    if ((db.AllocatedStorage ?? 0) > 0) emit('host_disk_size_bytes', db.AllocatedStorage! * GiB, end.getTime())
    if ((db.Iops ?? 0) > 0) emit('host_disk_provisioned_iops', db.Iops, end.getTime())
    if ((db.StorageThroughput ?? 0) > 0) emit('host_disk_provisioned_throughput_bytes_per_second', db.StorageThroughput! * 1024 ** 2, end.getTime())
    // Storage autoscaling's ceiling; absent when autoscaling is off.
    if ((db.MaxAllocatedStorage ?? 0) > 0) emit('host_disk_max_size_bytes', db.MaxAllocatedStorage! * GiB, end.getTime())
  }
  const lastBalances = state.get(balanceKey)
  if (!aurora && (lastBalances === undefined || end.getTime() - lastBalances >= balanceEvery)) {
    const balances = [
      ...(/^db\.t\d/.test(db.DBInstanceClass ?? '') ? creditMetrics : []),
      ...(db.StorageType === 'gp2' ? gp2Metrics : []),
      ...ebsMetrics,
    ]
    // A failure is reported and read again on the next poll; the samples above are kept.
    try {
      const until = Math.floor(end.getTime() / balanceEvery) * balanceEvery
      const result5: GetMetricDataCommandOutput = await clients.cloudwatch.send(new GetMetricDataCommand({
        StartTime: new Date(until - 30 * 60_000), EndTime: new Date(until),
        MetricDataQueries: queries(balances, 'DBInstanceIdentifier', target.instanceId, 300),
      }))
      for (const [name] of balances) {
        const data = result5.MetricDataResults?.find(r => r.Id === name)
        const points = (data?.Timestamps ?? []).map((time, i) => ({ time: time.getTime(), value: data?.Values?.[i] }))
        for (const point of points.sort((a, b) => a.time - b.time)) {
          if (typeof point.value === 'number' && closed(point.time, 300)) emit(name, point.value, point.time)
        }
      }
      state.set(balanceKey, end.getTime())
    } catch (error) {
      errors.push(toError(error))
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
