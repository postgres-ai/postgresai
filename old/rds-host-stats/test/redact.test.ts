import { expect, test } from 'bun:test'
import { redact } from './redact'

test('redact strips the recording account from every identifier the fixtures can carry', () => {
  const input = JSON.stringify({
    Address: 'db.abcdefghijkl.us-east-1.rds.amazonaws.com',
    MonitoringRoleArn: 'arn:aws:iam::783495713736:role/rds-monitoring-role',
    DBInstanceArn: 'arn:aws:rds:us-east-1:783495713736:db:pgai-test',
    VpcId: 'vpc-0123456789abcdef0',
    SubnetIdentifier: 'subnet-0a1b2c3d',
    VpcSecurityGroupId: 'sg-0123456789abcdef0',
    KmsKeyId: 'arn:aws:kms:us-east-1:783495713736:key/12345678-1234-1234-1234-123456789abc',
    MultiRegionKey: 'arn:aws:kms:us-east-1:783495713736:key/mrk-0123456789abcdef0123456789abcdef',
  }, null, 2)
  const output = redact(input)
  expect(output).not.toContain('783495713736')
  expect(output).not.toMatch(/(vpc|subnet|sg)-[0-9a-f]{8}/)
  expect(output).not.toMatch(/key\/(mrk-)?[0-9a-f]{8}/)
  expect(output).not.toContain('abcdefghijkl')
  expect(JSON.parse(output)).toEqual({
    Address: 'redacted',
    MonitoringRoleArn: 'arn:aws:iam::123456789012:role/rds-monitoring-role',
    DBInstanceArn: 'arn:aws:rds:us-east-1:123456789012:db:pgai-test',
    VpcId: 'vpc-redacted',
    SubnetIdentifier: 'subnet-redacted',
    VpcSecurityGroupId: 'sg-redacted',
    KmsKeyId: 'arn:aws:kms:us-east-1:123456789012:key/redacted',
    MultiRegionKey: 'arn:aws:kms:us-east-1:123456789012:key/redacted',
  })
})
