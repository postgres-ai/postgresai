import { expect, test } from 'bun:test'
import { readdirSync } from 'node:fs'
import { pollOnce } from '../lib/poll'

type Recorded = { client: string; command: string; input: unknown; output: unknown }

const iso = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/
const revive = (json: string) => JSON.parse(json, (_, v) => (typeof v === 'string' && iso.test(v) ? new Date(v) : v))

// Replays recorded AWS responses and fails on any request that differs from the recorded one.
function replay(recorded: Recorded[]) {
  const pending = [...recorded]
  const client = (name: string) => ({
    async send(command: { constructor: { name: string }; input: unknown }) {
      const i = pending.findIndex((r) => r.client === name && r.command === command.constructor.name)
      if (i < 0) throw new Error(`unexpected ${name} ${command.constructor.name}`)
      const [r] = pending.splice(i, 1)
      expect(JSON.parse(JSON.stringify(command.input))).toEqual(r.input)
      return revive(JSON.stringify(r.output))
    },
  })
  const clients = { rds: client('rds'), cloudwatch: client('cloudwatch'), pi: client('pi'), logs: client('logs') }
  return { clients, pending }
}

const root = `${import.meta.dir}/fixtures`

for (const name of readdirSync(root)) {
  const dir = `${root}/${name}`
  const recorded: Recorded[] = await Bun.file(`${dir}/recorded.json`).json()
  const { now, instanceId } = await Bun.file(`${dir}/case.json`).json()
  const target = { instanceId, cluster: 'ci', nodeName: 'node-01' }

  const metricData = (r: Recorded[]) =>
    r.find((c) => c.command === 'GetMetricDataCommand')!.output as { MetricDataResults: { Id: string; Timestamps: string[]; Values: number[] }[] }
  const cpuLine = (value: number, time: number) => `host_cpu_utilization_percent{cluster="ci",node_name="node-01"} ${value} ${time}\n`

  test(`${name}: writes the golden host_* samples`, async () => {
    const { clients, pending } = replay(recorded)
    const { text, errors } = await pollOnce(clients, target, new Date(now), new Map())
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text())
    expect(errors).toEqual([])
    expect(pending).toEqual([])
  })

  // CloudWatch returns the bucket that `now` falls into as a partial Average,
  // and the per-timestamp dedupe would freeze it. Buckets closed for less than
  // a period are left for the next poll.
  test(`${name}: the newest CloudWatch buckets are not written until they have closed for a period`, async () => {
    const early = structuredClone(recorded)
    const series = metricData(early).MetricDataResults.find((r) => r.Id === 'host_cpu_utilization_percent')!
    series.Timestamps.unshift(new Date(Date.parse(now) - 60_000).toISOString())
    series.Values.unshift(99)
    const { text } = await pollOnce(replay(early).clients, target, new Date(now), new Map())
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text())
    expect(text).not.toContain(cpuLine(99, Date.parse(now) - 60_000))
  })

  // The recorded requests carry an EndTime on the minute. A poll 17 s later
  // sends the same CloudWatch and PI requests and the same import text; only
  // the OS log window follows the clock.
  test(`${name}: an unaligned clock queries CloudWatch and PI up to the last minute boundary`, async () => {
    const later = new Date(Date.parse(now) + 17_000)
    const shifted = structuredClone(recorded)
    const logs = shifted.find((r) => r.command === 'GetLogEventsCommand')
    if (logs) Object.assign(logs.input as object, { startTime: later.getTime() - 15 * 60_000, endTime: later.getTime() })
    const { clients, pending } = replay(shifted)
    const { text } = await pollOnce(clients, target, later, new Map())
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text())
    expect(pending).toEqual([])
  })

  // PI stamps a point at the end of its minute. The recorded replies carry a
  // point at `end` itself, a minute that closed as the poll started; it is
  // written by the next poll, like a CloudWatch bucket.
  test(`${name}: Performance Insights points at or after the minute boundary are not written`, async () => {
    const pi = structuredClone(recorded).filter((r) => r.command === 'GetResourceMetricsCommand')
    if (!pi.length) return
    const future = structuredClone(recorded)
    const metric = (future.find((r) => r.command === 'GetResourceMetricsCommand')!.output as { MetricList: { DataPoints: { Timestamp: string; Value: number }[] }[] }).MetricList[0]
    expect(metric.DataPoints.at(-1)!.Timestamp).toBe(now)
    const before = metric.DataPoints.at(-2)!
    metric.DataPoints.at(-1)!.Value = 66
    metric.DataPoints.push({ Timestamp: new Date(Date.parse(now) + 60_000).toISOString(), Value: 77 })
    const { text } = await pollOnce(replay(future).clients, target, new Date(now), new Map())
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text())
    expect(text).not.toContain(' 66 ')
    expect(text).not.toContain(' 77 ')
    expect(text).toContain(`host_db_load{cluster="ci",node_name="node-01"} ${before.Value} ${Date.parse(now) - 60_000}\n`)
  })

  test(`${name}: an instance without Performance Insights gets no PI call and no DB load`, async () => {
    const off = structuredClone(recorded)
    const describe = off.find((r) => r.command === 'DescribeDBInstancesCommand')!.output as { DBInstances: { PerformanceInsightsEnabled?: boolean }[] }
    describe.DBInstances[0].PerformanceInsightsEnabled = false
    const { clients, pending } = replay(off)
    const { text, errors } = await pollOnce(clients, target, new Date(now), new Map())
    expect(errors).toEqual([])
    expect(text).toBe((await Bun.file(`${dir}/expected.prom`).text()).replace(/^host_db_load.*\n/gm, ''))
    expect(pending.map((r) => r.command)).toEqual(['GetResourceMetricsCommand'])
  })

  test(`${name}: a Performance Insights failure keeps the CloudWatch samples and reports the error`, async () => {
    const { clients } = replay(recorded)
    const failing = { ...clients, pi: { send: async () => { throw Object.assign(new Error('stream not found'), { name: 'ResourceNotFoundException' }) } } }
    const { text, errors } = await pollOnce(failing, target, new Date(now), new Map())
    const golden = await Bun.file(`${dir}/expected.prom`).text()
    expect(text).toContain(golden.split('\n')[0])
    expect(text).not.toContain('host_db_load')
    const hasPI = recorded.some((r) => r.command === 'GetResourceMetricsCommand')
    expect(errors.map((e) => e.name)).toEqual(hasPI ? ['ResourceNotFoundException'] : [])
  })

  test(`${name}: an Enhanced Monitoring failure keeps the CloudWatch samples and reports the error`, async () => {
    const { clients } = replay(recorded)
    const failing = { ...clients, logs: { send: async () => { throw Object.assign(new Error('rate exceeded'), { name: 'ThrottlingException' }) } } }
    const { text, errors } = await pollOnce(failing, target, new Date(now), new Map())
    const golden = await Bun.file(`${dir}/expected.prom`).text()
    expect(text).toContain(golden.split('\n')[0])
    expect(text).not.toContain('host_os_')
    const hasEM = recorded.some((r) => r.command === 'GetLogEventsCommand')
    expect(errors.map((e) => e.name)).toEqual(hasEM ? ['ThrottlingException'] : [])
  })

  // The same 15-minute log window comes back on every poll, so a malformed
  // event is reported once per state, not once per tick.
  test(`${name}: a malformed Enhanced Monitoring event is skipped, reported once, and not interpolated`, async () => {
    const logs = recorded.find((r) => r.command === 'GetLogEventsCommand')
    if (!logs) return
    const bad = structuredClone(recorded)
    const events = (bad.find((r) => r.command === 'GetLogEventsCommand')!.output as { events: { timestamp: number; message: string }[] }).events
    const time = Date.parse(now) - 30_000
    events.push(
      { timestamp: time, message: 'not json' },
      { timestamp: time, message: JSON.stringify({ timestamp: 'yesterday', loadAverageMinute: { one: 9.125 } }) },
      { timestamp: time, message: JSON.stringify({ timestamp: new Date(time).toISOString(), cpuUtilization: { total: `1 ${time}\nevil{cluster="x"} 1`, wait: 'NaN' }, memory: {}, swap: {}, loadAverageMinute: { one: 2.5 }, processList: 'x' }) },
    )
    const state = new Map<string, number>()
    const { text, errors } = await pollOnce(replay(bad).clients, target, new Date(now), state)
    expect(text).not.toContain('evil')
    expect(text).not.toContain('NaN')
    expect(text).not.toContain(' 9.125 ')
    expect(text).toContain(`host_os_load1{cluster="ci",node_name="node-01"} 2.5 ${time}\n`)
    expect(errors.map((e) => e.message)).toEqual([
      `Enhanced Monitoring event at ${time} skipped: JSON Parse error: Unexpected identifier "not"`,
      `Enhanced Monitoring event at ${time} skipped: no timestamp`,
    ])
    const again = await pollOnce(replay(bad).clients, target, new Date(now), state)
    expect(again.text).toBe('')
    expect(again.errors).toEqual([])
  })

  // The two id 0 rows are AWS's "OS processes" and "RDS processes" aggregates.
  // A null or rss-less row is an AWS quirk, not a reason to drop the series.
  test(`${name}: the largest process RSS skips the id 0 aggregates and malformed rows`, async () => {
    const logs = recorded.find((r) => r.command === 'GetLogEventsCommand')
    if (!logs) return
    const odd = structuredClone(recorded)
    const events = (odd.find((r) => r.command === 'GetLogEventsCommand')!.output as { events: { timestamp: number; message: string }[] }).events
    const time = Date.parse(now) - 30_000
    events.push({ timestamp: time, message: JSON.stringify({ timestamp: new Date(time).toISOString(), processList: [null, { id: 0, rss: 500_000 }, { id: 7 }, { id: 5, rss: 10 }, { id: 6, rss: 20 }] }) })
    const { text, errors } = await pollOnce(replay(odd).clients, target, new Date(now), new Map())
    expect(errors).toEqual([])
    expect(text).toContain(`host_os_process_max_rss_bytes{cluster="ci",node_name="node-01"} ${20 * 1024} ${time}\n`)
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text() + `host_os_process_max_rss_bytes{cluster="ci",node_name="node-01"} ${20 * 1024} ${time}\n`)
  })

  test(`${name}: a sample that shows up late behind a newer one is still written`, async () => {
    const late = structuredClone(recorded)
    const series = metricData(late).MetricDataResults.find((r) => r.Id === 'host_cpu_utilization_percent')!
    const [time, value] = [series.Timestamps.splice(1, 1)[0], series.Values.splice(1, 1)[0]]
    const state = new Map<string, number>()
    await pollOnce(replay(late).clients, target, new Date(now), state)
    const { text } = await pollOnce(replay(recorded).clients, target, new Date(now), state)
    expect(text).toBe(cpuLine(value, Date.parse(time)))
  })

  test(`${name}: a repeated poll writes nothing already written`, async () => {
    const state = new Map<string, number>()
    await pollOnce(replay(recorded).clients, target, new Date(now), state)
    expect((await pollOnce(replay(recorded).clients, target, new Date(now), state)).text).toBe('')
  })
}

// The recorded Aurora cluster was too young to report its cluster-level and
// Aurora-only metrics, so their scales are asserted on a synthetic reply.
test('aurora-postgresql: Aurora-only series are scaled to per-second rates', async () => {
  const dir = `${root}/aurora-postgresql`
  const recorded: Recorded[] = await Bun.file(`${dir}/recorded.json`).json()
  const { now, instanceId } = await Bun.file(`${dir}/case.json`).json()
  const synthetic = structuredClone(recorded)
  const results = (synthetic.find((r) => r.command === 'GetMetricDataCommand')!.output as { MetricDataResults: { Id: string; Timestamps: string[]; Values: number[] }[] }).MetricDataResults
  const time = new Date(Date.parse(now) - 10 * 60_000)
  const raw: Record<string, number> = {
    host_local_storage_free_bytes: 3e10,
    host_volume_used_bytes: 5e9,
    host_volume_read_iops: 30_000,
    host_volume_write_iops: 6_000,
  }
  for (const [id, value] of Object.entries(raw)) {
    const series = results.find((r) => r.Id === id)!
    series.Timestamps.push(time.toISOString())
    series.Values.push(value)
  }
  // A 300 s bucket 3 minutes old is closed for a 60 s metric, not for a volume one.
  const recent = new Date(Date.parse(now) - 3 * 60_000)
  for (const id of ['host_volume_used_bytes', 'host_local_storage_free_bytes']) {
    const series = results.find((r) => r.Id === id)!
    series.Timestamps.push(recent.toISOString())
    series.Values.push(1)
  }
  const { text } = await pollOnce(replay(synthetic).clients, { instanceId, cluster: 'ci', nodeName: 'node-01' }, new Date(now), new Map())
  const labels = '{cluster="ci",node_name="node-01"}'
  expect(text).toContain(`host_local_storage_free_bytes${labels} 30000000000 ${time.getTime()}\n`)
  expect(text).toContain(`host_volume_used_bytes${labels} 5000000000 ${time.getTime()}\n`)
  expect(text).toContain(`host_volume_read_iops${labels} 100 ${time.getTime()}\n`)
  expect(text).toContain(`host_volume_write_iops${labels} 20 ${time.getTime()}\n`)
  expect(text).toContain(`host_local_storage_free_bytes${labels} 1 ${recent.getTime()}\n`)
  expect(text).not.toContain(`host_volume_used_bytes${labels} 1 ${recent.getTime()}\n`)
})

// Recorded instances had no throughput/latency samples. Exercise actual values
// and DescribeDBInstances units without relying on a live AWS account.
test('rds-postgres: disk rates, latency and provisioned limits retain their units', async () => {
  const dir = `${root}/rds-postgres`
  const recorded: Recorded[] = await Bun.file(`${dir}/recorded.json`).json()
  const { now, instanceId } = await Bun.file(`${dir}/case.json`).json()
  const db = (recorded.find(r => r.command === 'DescribeDBInstancesCommand')!.output as { DBInstances: Record<string, unknown>[] }).DBInstances[0]
  Object.assign(db, { AllocatedStorage: 100, Iops: 3000, StorageThroughput: 125 })
  const results = (recorded.find(r => r.command === 'GetMetricDataCommand')!.output as { MetricDataResults: { Id: string; Timestamps: string[]; Values: number[] }[] }).MetricDataResults
  const time = Date.parse(now) - 3 * 60_000
  const raw = { host_disk_read_bytes_per_second: 1048576, host_disk_write_bytes_per_second: 2097152, host_disk_read_latency_seconds: 0.002, host_disk_write_latency_seconds: 0.004 }
  for (const [id, value] of Object.entries(raw)) {
    results.push({ Id: id, Timestamps: [new Date(time).toISOString()], Values: [value] })
  }
  const target = { instanceId, cluster: 'ci', nodeName: 'node-01' }
  const { text, errors } = await pollOnce(replay(recorded).clients, target, new Date(now), new Map())
  expect(errors).toEqual([])
  for (const [id, value] of Object.entries(raw)) expect(text).toContain(`${id}{cluster="ci",node_name="node-01"} ${value} ${time}\n`)
  expect(text).toContain(`host_disk_size_bytes{cluster="ci",node_name="node-01"} ${100 * 1024 ** 3} ${Date.parse(now)}\n`)
  expect(text).toContain(`host_disk_provisioned_iops{cluster="ci",node_name="node-01"} 3000 ${Date.parse(now)}\n`)
  expect(text).toContain(`host_disk_provisioned_throughput_bytes_per_second{cluster="ci",node_name="node-01"} ${125 * 1024 ** 2} ${Date.parse(now)}\n`)
  delete db.Iops
  delete db.StorageThroughput
  const unknown = await pollOnce(replay(recorded).clients, target, new Date(now), new Map())
  expect(unknown.text).not.toContain('host_disk_provisioned_')
  expect(unknown.text).toContain('host_disk_size_bytes')
})

// Resource sizing (platform-all#1101): what the instance is, its storage
// ceiling, and the credit and burst balances that limit what it can sustain.
test('rds-postgres: the instance info series carries the class, storage and region', async () => {
  const dir = `${root}/rds-postgres`
  const recorded: Recorded[] = await Bun.file(`${dir}/recorded.json`).json()
  const { now, instanceId } = await Bun.file(`${dir}/case.json`).json()
  const db = (recorded.find(r => r.command === 'DescribeDBInstancesCommand')!.output as { DBInstances: Record<string, unknown>[] }).DBInstances[0]
  Object.assign(db, { MultiAZ: true, MaxAllocatedStorage: 40, StorageType: 'gp3"x' })
  const { text, errors } = await pollOnce(replay(recorded).clients, { instanceId, cluster: 'ci', nodeName: 'node-01', region: 'eu-central-1' }, new Date(now), new Map())
  expect(errors).toEqual([])
  expect(text).toContain(`host_instance_info{cluster="ci",node_name="node-01",instance_class="db.t4g.micro",engine="postgres",storage_type="gp3\\"x",multi_az="true",region="eu-central-1"} 1 ${Date.parse(now)}\n`)
  expect(text).toContain(`host_disk_max_size_bytes{cluster="ci",node_name="node-01"} ${40 * 1024 ** 3} ${Date.parse(now)}\n`)
})

test('rds-postgres: balances are read every 5 minutes, only where they apply, and a failure is retried', async () => {
  const dir = `${root}/rds-postgres`
  const recorded: Recorded[] = await Bun.file(`${dir}/recorded.json`).json()
  const { now, instanceId } = await Bun.file(`${dir}/case.json`).json()
  const target = { instanceId, cluster: 'ci', nodeName: 'node-01' }
  const t0 = Date.parse(now)
  // Every GetMetricData request, with a reply that has no points.
  const calls: { StartTime: Date; EndTime: Date; MetricDataQueries: { Id: string; MetricStat: { Period: number } }[] }[] = []
  const describe = (patch: Record<string, unknown>) => ({
    async send(command: { constructor: { name: string } }) {
      const out = structuredClone(recorded.find(r => r.command === 'DescribeDBInstancesCommand')!.output) as { DBInstances: Record<string, unknown>[] }
      Object.assign(out.DBInstances[0], { PerformanceInsightsEnabled: false, MonitoringInterval: 0, ...patch })
      return out
    },
  })
  let fail = false
  const cloudwatch = {
    async send(command: { input: (typeof calls)[number] }) {
      if (fail && command.input.MetricDataQueries[0].MetricStat.Period === 300) throw Object.assign(new Error('throttled'), { name: 'Throttling' })
      calls.push(command.input)
      return { MetricDataResults: [] }
    },
  }
  const none = { send: async () => { throw new Error('not expected') } }
  const balanceIds = () => calls.filter(c => c.MetricDataQueries[0].MetricStat.Period === 300).map(c => c.MetricDataQueries.map(q => q.Id))
  const state = new Map<string, number>()

  await pollOnce({ rds: describe({}), cloudwatch, pi: none, logs: none }, target, new Date(t0), state)
  expect(balanceIds()).toEqual([['host_cpu_credit_balance', 'host_cpu_surplus_credit_balance', 'host_ebs_io_balance_percent', 'host_ebs_byte_balance_percent']])
  // On 5-minute boundaries (CloudWatch aligns the buckets to StartTime), so a
  // later read stamps the same buckets.
  const window = calls.filter(c => c.MetricDataQueries[0].MetricStat.Period === 300)[0]
  expect([window.StartTime.getTime() % 300_000, window.EndTime.getTime() % 300_000]).toEqual([0, 0])
  expect(window.EndTime.getTime() - window.StartTime.getTime()).toBe(30 * 60_000)
  // 4 minutes later: not again.
  await pollOnce({ rds: describe({}), cloudwatch, pi: none, logs: none }, target, new Date(t0 + 4 * 60_000), state)
  expect(balanceIds().length).toBe(1)
  // 5 minutes after the first: a non-burstable gp2 instance reads BurstBalance, not CPU credits.
  await pollOnce({ rds: describe({ DBInstanceClass: 'db.m7g.large', StorageType: 'gp2' }), cloudwatch, pi: none, logs: none }, target, new Date(t0 + 5 * 60_000), state)
  expect(balanceIds()[1]).toEqual(['host_disk_burst_balance_percent', 'host_ebs_io_balance_percent', 'host_ebs_byte_balance_percent'])
  // A failed read is reported, the minute's samples are still written, and it is read again on the next poll.
  fail = true
  const failed = await pollOnce({ rds: describe({}), cloudwatch, pi: none, logs: none }, target, new Date(t0 + 10 * 60_000), state)
  expect(failed.errors.map(e => e.name)).toEqual(['Throttling'])
  expect(failed.text).toContain('host_instance_info')
  fail = false
  await pollOnce({ rds: describe({}), cloudwatch, pi: none, logs: none }, target, new Date(t0 + 11 * 60_000), state)
  expect(balanceIds().length).toBe(3)
  // Aurora: never.
  const aurora = new Map<string, number>()
  await pollOnce({ rds: describe({ Engine: 'aurora-postgresql', DBClusterIdentifier: 'c' }), cloudwatch, pi: none, logs: none }, target, new Date(t0), aurora)
  expect(balanceIds().length).toBe(3)
})
