// Real end-to-end run against a live RDS instance and a real VictoriaMetrics:
//   RDS_HOST_STATS_E2E_INSTANCE=<db instance id> RDS_HOST_STATS_E2E_VM_URL=http://127.0.0.1:8428 bun test test/e2e.test.ts
// Needs AWS credentials for an account where the instance has Enhanced Monitoring and PI on.
import { expect, test } from 'bun:test'
import { createClients, pollOnce, writeSamples } from '../lib/poll'

const instanceId = process.env.RDS_HOST_STATS_E2E_INSTANCE
const vm = process.env.RDS_HOST_STATS_E2E_VM_URL

const families = [
  'host_cpu_utilization_percent',
  'host_memory_available_bytes',
  'host_disk_free_bytes',
  'host_disk_read_iops',
  'host_disk_write_iops',
  'host_disk_read_latency_seconds',
  'host_disk_write_latency_seconds',
  'host_disk_queue_depth',
  'host_network_receive_bytes_per_second',
  'host_network_transmit_bytes_per_second',
  'host_db_load',
  'host_os_cpu_percent',
  'host_os_cpu_iowait_percent',
  'host_os_load1',
  'host_os_memory_total_bytes',
  'host_os_memory_free_bytes',
  'host_os_memory_cached_bytes',
  'host_os_swap_used_bytes',
  'host_os_process_max_rss_bytes',
]

test.skipIf(!instanceId || !vm)('live RDS: every host_* family lands in VictoriaMetrics', async () => {
  const clients = createClients(process.env.AWS_REGION ?? 'us-east-1')
  const text = await pollOnce(clients, { instanceId: instanceId!, cluster: 'e2e', nodeName: 'node-01' }, new Date(), new Map())
  await writeSamples(vm!, text)
  await fetch(`${vm}/internal/force_flush`)
  const start = Math.floor(Date.now() / 1000) - 3600
  const res = await fetch(`${vm}/api/v1/label/__name__/values?match[]={cluster="e2e"}&start=${start}`)
  const { data } = (await res.json()) as { data: string[] }
  console.log(JSON.stringify({ written: text.split('\n').length - 1, families: data }))
  expect(data).toEqual(expect.arrayContaining(families))
}, 60_000)
