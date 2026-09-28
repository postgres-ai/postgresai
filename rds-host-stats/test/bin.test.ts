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
