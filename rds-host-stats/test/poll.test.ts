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

  test(`${name}: writes the golden host_* samples`, async () => {
    const { clients, pending } = replay(recorded)
    const text = await pollOnce(clients, target, new Date(now), new Map())
    expect(text).toBe(await Bun.file(`${dir}/expected.prom`).text())
    expect(pending).toEqual([])
  })

  test(`${name}: a sample that shows up late behind a newer one is still written`, async () => {
    const late = structuredClone(recorded)
    const cpu = late
      .find((r) => r.command === 'GetMetricDataCommand')!
      .output as { MetricDataResults: { Id: string; Timestamps: string[]; Values: number[] }[] }
    const series = cpu.MetricDataResults.find((r) => r.Id === 'host_cpu_utilization_percent')!
    const [time, value] = [series.Timestamps.splice(1, 1)[0], series.Values.splice(1, 1)[0]]
    const state = new Map<string, number>()
    await pollOnce(replay(late).clients, target, new Date(now), state)
    const text = await pollOnce(replay(recorded).clients, target, new Date(now), state)
    expect(text).toBe(`host_cpu_utilization_percent{cluster="ci",node_name="node-01"} ${value} ${Date.parse(time)}\n`)
  })

  test(`${name}: a repeated poll writes nothing already written`, async () => {
    const state = new Map<string, number>()
    await pollOnce(replay(recorded).clients, target, new Date(now), state)
    expect(await pollOnce(replay(recorded).clients, target, new Date(now), state)).toBe('')
  })
}
