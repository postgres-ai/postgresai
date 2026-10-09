import { expect, test } from 'bun:test'
import { mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// Run the actual recorder in a child so AWS module mocks never affect poll tests.
async function record(rows: unknown[], fail = false) {
  const dir = mkdtempSync(join(tmpdir(), 'rds-recorder-'))
  try {
    await Bun.write(`${dir}/recorded.json`, JSON.stringify(rows))
    await Bun.write(`${dir}/mock.ts`, `
      import { mock } from 'bun:test'
      class Client {
        async send(c) {
          if (${fail}) throw new Error('AWS unavailable')
          return { $metadata: { requestId: 'removed' }, command: c.constructor.name,
            dates: ['StartTime', 'EndTime'].filter(k => k in c.input).every(k => c.input[k] instanceof Date) }
        }
      }
      for (const [pkg, client, command] of ${JSON.stringify([
        ['cloudwatch', 'CloudWatchClient', 'GetMetricDataCommand'],
        ['cloudwatch-logs', 'CloudWatchLogsClient', 'GetLogEventsCommand'],
        ['pi', 'PIClient', 'GetResourceMetricsCommand'],
        ['rds', 'RDSClient', 'DescribeDBInstancesCommand'],
      ].map(([pkg, client, command]) => [import.meta.resolve('@aws-sdk/client-' + pkg), client, command]))}) {
        const Cmd = class { constructor(input) { this.input = input } }
        Object.defineProperty(Cmd, 'name', { value: command })
        mock.module(pkg, () => ({ [client]: Client, [command]: Cmd }))
      }
      await import(${JSON.stringify(`${import.meta.dir}/record.ts`)})
    `)
    const child = Bun.spawn([process.execPath, `${dir}/mock.ts`, dir], { stdout: 'pipe', stderr: 'pipe' })
    const stderr = await new Response(child.stderr).text()
    return { exitCode: await child.exited, stderr, rows: await Bun.file(`${dir}/recorded.json`).json() }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

for (const name of readdirSync(`${import.meta.dir}/fixtures`)) {
  test(`${name}: recorder preserves request order, revives dates and replaces responses`, async () => {
    const rows = await Bun.file(`${import.meta.dir}/fixtures/${name}/recorded.json`).json()
    const result = await record(rows)
    expect(result.stderr).toBe('')
    expect(result.exitCode).toBe(0)
    expect(result.rows).toEqual(rows.map(({ client, command, input }: any) => ({
      client, command, input, output: { command, dates: true },
    })))
  })
}

test('a new fixture needs only request envelopes; failed recording preserves the previous file', async () => {
  const rows = [{ client: 'rds', command: 'DescribeDBInstancesCommand', input: { DBInstanceIdentifier: 'demo' } }]
  expect((await record(rows)).rows[0].output.command).toBe('DescribeDBInstancesCommand')
  const result = await record(rows, true)
  expect(result.exitCode).not.toBe(0)
  expect(result.stderr).toContain('AWS unavailable')
  expect(result.rows).toEqual(rows)
})
