type Client = { send(command: any): Promise<any> }
export type Clients = { rds: Client; cloudwatch: Client; pi: Client; logs: Client }
export type Target = { instanceId: string; cluster: string; nodeName: string }
export type Role = { arn: string; externalId: string }
export type Auth = { username: string; password: string }

export function createClients(region: string, role?: Role): Clients {
  throw new Error('not implemented')
}

export async function pollOnce(clients: Clients, target: Target, now: Date, state: Map<string, number>): Promise<string> {
  throw new Error('not implemented')
}

export async function writeSamples(url: string, text: string, auth?: Auth): Promise<void> {
  throw new Error('not implemented')
}
