import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const roleArn = 'arn:aws:iam::123456789012:role/customer'
const externalId = 'postgresai-test-external-id'
type Request = { action: string; body: string; authorization: string; token?: string }

// Run the service with an offline AWS transport in a child, so the shared
// request handler and clock in the other tests are untouched.
async function runRole(fail = false) {
  const dir = mkdtempSync(join(tmpdir(), 'rds-role-'))
  try {
    await Bun.write(`${dir}/mock.ts`, `
      import { setSystemTime } from 'bun:test'
      import { requestHandler } from ${JSON.stringify(`${import.meta.dir}/../lib/poll.ts`)}
      setSystemTime(new Date('2026-10-01T12:00:00Z'))
      const requests = []
      const imports = []
      Object.assign(requestHandler, {
        async handle(req) {
          const body = typeof req.body === 'string' ? req.body : Buffer.from(req.body).toString()
          const action = new URLSearchParams(body).get('Action') || req.headers['x-amz-target'].split('.').at(-1)
          requests.push({ action, body, authorization: req.headers.authorization, token: req.headers['x-amz-security-token'] })
          let statusCode = 200
          let contentType = 'text/xml'
          let response
          if (action === 'AssumeRole') {
            statusCode = ${fail} ? 403 : 200
            response = ${fail}
              ? '<ErrorResponse><Error><Type>Sender</Type><Code>AccessDenied</Code><Message>offline STS denied</Message></Error><RequestId>r</RequestId></ErrorResponse>'
              : '<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><AssumeRoleResult><Credentials><AccessKeyId>assumed</AccessKeyId><SecretAccessKey>assumed-secret</SecretAccessKey><SessionToken>assumed-token</SessionToken><Expiration>2030-01-01T00:00:00Z</Expiration></Credentials></AssumeRoleResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></AssumeRoleResponse>'
          } else if (action === 'DescribeDBInstances') {
            response = '<DescribeDBInstancesResponse xmlns="http://rds.amazonaws.com/doc/2014-10-31/"><DescribeDBInstancesResult><DBInstances><DBInstance><DBInstanceIdentifier>db</DBInstanceIdentifier><Engine>postgres</Engine><DbiResourceId>db-X</DbiResourceId><PerformanceInsightsEnabled>false</PerformanceInsightsEnabled><MonitoringInterval>0</MonitoringInterval></DBInstance></DBInstances></DescribeDBInstancesResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></DescribeDBInstancesResponse>'
          } else if (action === 'GetMetricData') {
            contentType = 'application/x-amz-json-1.0'
            response = JSON.stringify({ MetricDataResults: [{ Id: 'host_cpu_utilization_percent', Timestamps: [Date.parse('2026-10-01T11:55:00Z') / 1000], Values: [12.5], StatusCode: 'Complete' }] })
          } else {
            throw new Error('Unexpected offline AWS request: ' + action)
          }
          return { response: { statusCode, headers: { 'content-type': contentType }, body: Buffer.from(response) } }
        },
        destroy() {},
      })
      globalThis.fetch = async (url, options) => {
        if (url !== 'http://offline.invalid/api/v1/import/prometheus') throw new Error('Unexpected offline import')
        imports.push(options.body)
        return new Response(null, { status: 204 })
      }
      Bun.sleep = async () => {
        console.log(JSON.stringify({ requests, imports }))
        process.exit(0)
      }
      await import(${JSON.stringify(`${import.meta.dir}/../bin/rds-host-stats.ts`)})
    `)
    const child = Bun.spawn([process.execPath, `${dir}/mock.ts`], {
      env: {
        PATH: process.env.PATH, AWS_REGION: 'us-east-1', AWS_ACCESS_KEY_ID: 'ambient', AWS_SECRET_ACCESS_KEY: 'ambient-secret', AWS_MAX_ATTEMPTS: '1',
        RDS_ROLE_ARN: roleArn, RDS_EXTERNAL_ID: externalId, RDS_DB_INSTANCE_IDENTIFIER: 'db', PGAI_CLUSTER: 'c', PGAI_NODE_NAME: 'n', PROMETHEUS_URL: 'http://offline.invalid',
      },
      stdout: 'pipe', stderr: 'pipe',
    })
    const killer = setTimeout(() => child.kill(), 10_000)
    try {
      const stdout = await new Response(child.stdout).text()
      const stderr = await new Response(child.stderr).text()
      expect(await child.exited).toBe(0)
      const result: { requests: Request[]; imports: string[] } = JSON.parse(stdout.trimEnd().split('\n').at(-1)!)
      return { ...result, stdout, stderr }
    } finally {
      clearTimeout(killer)
    }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

test('customer role uses the configured AssumeRole parameters and assumed credentials', async () => {
  const { requests, imports, stderr } = await runRole()
  expect(stderr).toBe('')
  expect(requests.map(r => r.action)).toEqual(['AssumeRole', 'DescribeDBInstances', 'AssumeRole', 'GetMetricData'])
  for (const request of requests.filter(r => r.action === 'AssumeRole')) {
    const params = new URLSearchParams(request.body)
    expect(params.get('RoleArn')).toBe(roleArn)
    expect(params.get('ExternalId')).toBe(externalId)
    expect(request.authorization).toContain('Credential=ambient/')
  }
  for (const request of requests.filter(r => r.action !== 'AssumeRole')) {
    expect(request.authorization).toContain('Credential=assumed/')
    expect(request.token).toBe('assumed-token')
  }
  expect(imports).toEqual(['host_cpu_utilization_percent{cluster="c",node_name="n"} 12.5 1790855700000\n'])
}, 15_000)

test('an STS failure writes no metrics and never queries RDS or CloudWatch', async () => {
  const { requests, imports, stdout, stderr } = await runRole(true)
  expect(stderr).toContain('AccessDenied: offline STS denied')
  expect(requests.map(r => r.action)).toEqual(['AssumeRole'])
  expect(imports).toEqual([])
  expect(stdout).toContain('samples written: 0')
}, 15_000)
