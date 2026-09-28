import { createClients, PollTimeout, pollOnce, writeSamples } from '../lib/poll'

const env = process.env
function required(name: string): string {
  const value = env[name]
  if (!value) {
    console.error(`${name} is required`)
    process.exit(2)
  }
  return value
}

if (Boolean(env.RDS_ROLE_ARN) !== Boolean(env.RDS_EXTERNAL_ID)) {
  console.error('RDS_ROLE_ARN and RDS_EXTERNAL_ID must both be set or both be unset')
  process.exit(2)
}
const target = {
  instanceId: required('RDS_DB_INSTANCE_IDENTIFIER'),
  cluster: required('PGAI_CLUSTER'),
  nodeName: required('PGAI_NODE_NAME'),
}
const region = required('AWS_REGION')
const interval = Number(env.RDS_POLL_INTERVAL_SECONDS || 60)
if (!Number.isInteger(interval) || interval < 1) {
  console.error('RDS_POLL_INTERVAL_SECONDS must be a positive integer')
  process.exit(2)
}
const role = env.RDS_ROLE_ARN ? { arn: env.RDS_ROLE_ARN, externalId: env.RDS_EXTERNAL_ID! } : undefined
const clients = createClients(region, role)
const url = env.PROMETHEUS_URL || 'http://sink-prometheus:9090'
const auth = env.VM_AUTH_USERNAME && env.VM_AUTH_PASSWORD ? { username: env.VM_AUTH_USERNAME, password: env.VM_AUTH_PASSWORD } : undefined
process.on('SIGTERM', () => process.exit(0))
process.on('SIGINT', () => process.exit(0))
let state = new Map<string, number>()
let nextTick = Date.now()
while (true) {
  let written = 0
  try {
    // state is committed only after the write succeeds, so the samples of a
    // failed write are sent again on the next tick.
    const pending = new Map(state)
    const { text, errors } = await pollOnce(clients, target, new Date(), pending)
    for (const error of errors) console.error(`${error.name}: ${error.message}`)
    await writeSamples(url, text, auth)
    state = pending
    written = text.split('\n').length - 1
  } catch (error) {
    console.error(error instanceof Error ? `${error.name}: ${error.message}` : 'poll failed')
    // A stalled socket survives the SDK client under Bun; a restart drops it.
    if (error instanceof PollTimeout) process.exit(1)
  }
  console.log(`samples written: ${written}`)
  do { nextTick += interval * 1000 } while (nextTick <= Date.now())
  await Bun.sleep(nextTick - Date.now())
}
