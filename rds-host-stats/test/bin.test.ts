import { expect, test } from 'bun:test'

test('refuses a role ARN without an External ID, and the reverse', () => {
  const base = { PATH: process.env.PATH, RDS_DB_INSTANCE_IDENTIFIER: 'db', AWS_REGION: 'us-east-1', PGAI_CLUSTER: 'c', PGAI_NODE_NAME: 'n' }
  for (const extra of [{ RDS_ROLE_ARN: 'arn:aws:iam::123456789012:role/r' }, { RDS_EXTERNAL_ID: 'postgresai-org-1' }]) {
    const run = Bun.spawnSync(['bun', `${import.meta.dir}/../bin/rds-host-stats.ts`], { env: { ...base, ...extra } })
    expect(run.exitCode).toBe(2)
    expect(run.stderr.toString()).toContain('RDS_ROLE_ARN and RDS_EXTERNAL_ID')
  }
})
