import { createServer, type AddressInfo } from 'node:net'
import { expect, test } from 'bun:test'

test('refuses a role ARN without an External ID, and the reverse', () => {
  const base = { PATH: process.env.PATH, RDS_DB_INSTANCE_IDENTIFIER: 'db', AWS_REGION: 'us-east-1', PGAI_CLUSTER: 'c', PGAI_NODE_NAME: 'n' }
  for (const extra of [{ RDS_ROLE_ARN: 'arn:aws:iam::123456789012:role/r' }, { RDS_EXTERNAL_ID: 'postgresai-org-1' }]) {
    const run = Bun.spawnSync(['bun', `${import.meta.dir}/../bin/rds-host-stats.ts`], { env: { ...base, ...extra } })
    expect(run.exitCode).toBe(2)
    expect(run.stderr.toString()).toContain('RDS_ROLE_ARN and RDS_EXTERNAL_ID')
  }
})

// A timed-out poll can leave a socket stuck mid-body that neither aborting nor
// destroying the SDK client closes under Bun, so the service exits and the
// container restart policy starts it clean.
test('exits after a poll times out on a stalled AWS response', async () => {
  const stalling = createServer((socket) => {
    socket.once('data', () => socket.write('HTTP/1.1 200 OK\r\ncontent-type: text/xml\r\ncontent-length: 1000\r\n\r\n<Describe'))
  })
  await new Promise<void>((resolve) => stalling.listen(0, '127.0.0.1', resolve))
  const run = Bun.spawn(['bun', `${import.meta.dir}/../bin/rds-host-stats.ts`], {
    env: {
      PATH: process.env.PATH,
      RDS_DB_INSTANCE_IDENTIFIER: 'db',
      AWS_REGION: 'us-east-1',
      PGAI_CLUSTER: 'c',
      PGAI_NODE_NAME: 'n',
      AWS_ENDPOINT_URL: `http://127.0.0.1:${(stalling.address() as AddressInfo).port}`,
      AWS_ACCESS_KEY_ID: 'test',
      AWS_SECRET_ACCESS_KEY: 'test',
    },
    stderr: 'pipe',
  })
  const killer = setTimeout(() => run.kill(), 45_000)
  const code = await run.exited
  clearTimeout(killer)
  stalling.close()
  expect(code).toBe(1)
  expect(await new Response(run.stderr).text()).toContain('timed out')
}, 50_000)

// A mock of the two AWS calls the loop needs for an instance without PI or
// Enhanced Monitoring, and of the VictoriaMetrics import endpoint. Both are
// served from one port; AWS requests carry `Action=` in a form body.
function mockStack(options: { instance: boolean; importStatuses: number[] }) {
  const imports: string[] = []
  const bucket = new Date(Math.floor(Date.now() / 60_000) * 60_000 - 5 * 60_000)
  const rds = options.instance
    ? '<DBInstance><DBInstanceIdentifier>db</DBInstanceIdentifier><Engine>postgres</Engine><DbiResourceId>db-X</DbiResourceId><PerformanceInsightsEnabled>false</PerformanceInsightsEnabled><MonitoringInterval>0</MonitoringInterval></DBInstance>'
    : ''
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(req) {
      const url = new URL(req.url)
      if (url.pathname === '/api/v1/import/prometheus') {
        imports.push(await req.text())
        return new Response('', { status: options.importStatuses[imports.length - 1] ?? 204 })
      }
      const body = await req.text()
      const xml = { 'content-type': 'text/xml' }
      if (body.includes('Action=DescribeDBInstances')) {
        return new Response(`<DescribeDBInstancesResponse xmlns="http://rds.amazonaws.com/doc/2014-10-31/"><DescribeDBInstancesResult><DBInstances>${rds}</DBInstances></DescribeDBInstancesResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></DescribeDBInstancesResponse>`, { headers: xml })
      }
      // CloudWatch speaks AWS JSON 1.0 in this SDK; timestamps are epoch seconds.
      if (req.headers.get('x-amz-target')?.endsWith('.GetMetricData')) {
        const result = { Id: 'host_cpu_utilization_percent', Label: 'CPUUtilization', Timestamps: [bucket.getTime() / 1000], Values: [12.5], StatusCode: 'Complete' }
        return Response.json({ MetricDataResults: [result], Messages: [] }, { headers: { 'content-type': 'application/x-amz-json-1.0' } })
      }
      return new Response(`unexpected request ${req.headers.get('x-amz-target')} ${body.slice(0, 80)}`, { status: 500 })
    },
  })
  const env = {
    PATH: process.env.PATH,
    RDS_DB_INSTANCE_IDENTIFIER: 'db',
    AWS_REGION: 'us-east-1',
    PGAI_CLUSTER: 'c',
    PGAI_NODE_NAME: 'n',
    RDS_POLL_INTERVAL_SECONDS: '2',
    AWS_ENDPOINT_URL: `http://127.0.0.1:${server.port}`,
    PROMETHEUS_URL: `http://127.0.0.1:${server.port}`,
    AWS_ACCESS_KEY_ID: 'test',
    AWS_SECRET_ACCESS_KEY: 'test',
  }
  return { server, env, imports, sample: `host_cpu_utilization_percent{cluster="c",node_name="n"} 12.5 ${bucket.getTime()}\n` }
}

async function runTicks(env: Record<string, string | undefined>, ticks: number) {
  const run = Bun.spawn(['bun', `${import.meta.dir}/../bin/rds-host-stats.ts`], { env, stdout: 'pipe', stderr: 'pipe' })
  let stdout = ''
  const reader = run.stdout.getReader()
  const decoder = new TextDecoder()
  while ((stdout.match(/samples written/g) ?? []).length < ticks) {
    const { value, done } = await reader.read()
    if (done) break
    stdout += decoder.decode(value)
  }
  const alive = run.exitCode === null
  run.kill()
  await run.exited
  return { stdout, stderr: await new Response(run.stderr).text(), alive }
}

// State is committed only after the write succeeds, so the samples of a tick
// whose import failed are sent again on the next one.
test('re-sends the samples of a failed VictoriaMetrics write on the next tick', async () => {
  const stack = mockStack({ instance: true, importStatuses: [500, 204] })
  try {
    const { stdout, stderr, alive } = await runTicks(stack.env, 2)
    expect(alive).toBe(true)
    expect(stderr).toContain('Prometheus import failed: HTTP 500')
    expect(stdout).toBe('samples written: 0\nsamples written: 1\n')
    expect(stack.imports).toEqual([stack.sample, stack.sample])
  } finally {
    void stack.server.stop(true)
  }
}, 30_000)

test('stays up and reports zero samples when the instance is not found', async () => {
  const stack = mockStack({ instance: false, importStatuses: [] })
  try {
    const { stdout, stderr, alive } = await runTicks(stack.env, 2)
    expect(alive).toBe(true)
    expect(stderr).toContain('Error: RDS instance not found')
    expect(stdout).toBe('samples written: 0\nsamples written: 0\n')
    expect(stack.imports).toEqual([])
  } finally {
    void stack.server.stop(true)
  }
}, 30_000)

// Above 300 s a poll could miss a 300 s Aurora volume bucket entirely.
test('refuses a poll interval that is not an integer from 1 to 300', () => {
  for (const value of ['0', '1.5', '60s', '301']) {
    const env = { PATH: process.env.PATH, RDS_DB_INSTANCE_IDENTIFIER: 'db', AWS_REGION: 'us-east-1', PGAI_CLUSTER: 'c', PGAI_NODE_NAME: 'n', RDS_POLL_INTERVAL_SECONDS: value }
    const run = Bun.spawnSync(['bun', `${import.meta.dir}/../bin/rds-host-stats.ts`], { env })
    expect(run.exitCode, value).toBe(2)
    expect(run.stderr.toString(), value).toContain('RDS_POLL_INTERVAL_SECONDS must be an integer from 1 to 300')
  }
})
