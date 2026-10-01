// Re-records a fixture case against the real AWS APIs:
//   bun test/record.ts test/fixtures/<case>
// Reads <case>/calls.json, sends each call with the default credential chain,
// and writes <case>/recorded.json. Endpoint addresses, the account ID in ARNs,
// VPC/subnet/security-group IDs and KMS key IDs are redacted (test/redact.ts).
import { CloudWatchClient, GetMetricDataCommand } from '@aws-sdk/client-cloudwatch'
import { CloudWatchLogsClient, GetLogEventsCommand } from '@aws-sdk/client-cloudwatch-logs'
import { DescribeDBInstancesCommand, RDSClient } from '@aws-sdk/client-rds'
import { redact } from './redact'

const dir = process.argv[2]
if (!dir) throw new Error('usage: bun test/record.ts <fixture dir>')
const region = process.env.AWS_REGION ?? 'us-east-1'
const clients = {
  rds: new RDSClient({ region }),
  cloudwatch: new CloudWatchClient({ region }),
  logs: new CloudWatchLogsClient({ region }),
}
const commands = { DescribeDBInstancesCommand, GetMetricDataCommand, GetLogEventsCommand }
type Call = { client: keyof typeof clients; command: keyof typeof commands; input: Record<string, unknown> }

const calls: Call[] = await Bun.file(`${dir}/calls.json`).json()
const recorded = []
for (const call of calls) {
  const input = { ...call.input }
  for (const k of ['StartTime', 'EndTime']) if (typeof input[k] === 'string') input[k] = new Date(input[k] as string)
  const client = clients[call.client] as { send(c: unknown): Promise<Record<string, unknown>> }
  const { $metadata, ...output } = await client.send(new (commands[call.command] as new (i: unknown) => unknown)(input))
  recorded.push({ ...call, output })
}
await Bun.write(`${dir}/recorded.json`, redact(JSON.stringify(recorded, null, 2)) + '\n')
