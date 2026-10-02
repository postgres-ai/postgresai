import { expect, test } from 'bun:test'

// Measured on the release image against real AWS endpoints: at 0.1 CPU the
// first TLS + SigV4 call of a cold poll took 1.2 s instead of 0.28 s, and QA
// saw every poll hit the 5 s connect timeout on a smaller VM. From 0.25 CPU
// up, call latency no longer depended on the cap; 0.5 keeps margin for a
// slower vCPU and a full Enhanced Monitoring page (up to 1 MB of JSON).
test('compose gives the poller at least half a CPU by default', () => {
  const compose = Bun.YAML.parse(require('node:fs').readFileSync(`${import.meta.dir}/../../docker-compose.yml`, 'utf8')) as any
  const service = compose.services['rds-host-stats']
  const match = /^\$\{RDS_HOST_STATS_CPUS:-([0-9.]+)\}$/.exec(service.cpus)
  expect(match).not.toBeNull()
  expect(Number(match![1])).toBeGreaterThanOrEqual(0.5)
})
