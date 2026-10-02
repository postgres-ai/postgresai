import { createClients, PollTimeout, pollOnce, writeSamples } from '../lib/poll'

const env = process.env
process.on('SIGTERM', () => process.exit(0))
process.on('SIGINT', () => process.exit(0))

// Returns the first configuration problem, or undefined.
function problem(): string | undefined {
  if (Boolean(env.RDS_ROLE_ARN) !== Boolean(env.RDS_EXTERNAL_ID)) {
    return 'RDS_ROLE_ARN and RDS_EXTERNAL_ID must both be set or both be unset'
  }
  for (const name of ['RDS_DB_INSTANCE_IDENTIFIER', 'PGAI_CLUSTER', 'PGAI_NODE_NAME', 'AWS_REGION']) {
    if (!env[name]) return `${name} is required`
  }
  // A 300 s bucket is written by a poll that ends 10 to 15 minutes after it
  // starts, so a longer interval would skip Aurora volume buckets.
  const interval = Number(env.RDS_POLL_INTERVAL_SECONDS || 60)
  if (!Number.isInteger(interval) || interval < 1 || interval > 300) {
    return 'RDS_POLL_INTERVAL_SECONDS must be an integer from 1 to 300'
  }
}

// Compose restarts the service on any exit and defaults the required
// variables to empty, so a configuration error idles with one log line instead of exiting
// into a restart loop.
const invalid = problem()
if (invalid) {
  console.error(`${invalid}; rds-host-stats will idle until the configuration is fixed and the container is recreated (docker compose --profile rds up -d rds-host-stats)`)
  await new Promise(() => setInterval(() => {}, 2 ** 31 - 1))
}
const target = { instanceId: env.RDS_DB_INSTANCE_IDENTIFIER!, cluster: env.PGAI_CLUSTER!, nodeName: env.PGAI_NODE_NAME! }
const region = env.AWS_REGION!
const interval = Number(env.RDS_POLL_INTERVAL_SECONDS || 60)
const role = env.RDS_ROLE_ARN ? { arn: env.RDS_ROLE_ARN, externalId: env.RDS_EXTERNAL_ID! } : undefined
const clients = createClients(region, role)
const url = env.PROMETHEUS_URL || 'http://sink-prometheus:9090'
const auth = env.VM_AUTH_USERNAME && env.VM_AUTH_PASSWORD ? { username: env.VM_AUTH_USERNAME, password: env.VM_AUTH_PASSWORD } : undefined
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
