import { afterAll, expect, test } from 'bun:test'
import { writeSamples } from '../lib/poll'

const requests: { path: string; auth: string | null; body: string }[] = []
let status = 204
const server = Bun.serve({
  port: 0,
  async fetch(req) {
    if (new URL(req.url).pathname === '/stall/api/v1/import/prometheus') return new Promise<Response>(() => {})
    requests.push({ path: new URL(req.url).pathname, auth: req.headers.get('authorization'), body: await req.text() })
    return new Response(null, { status })
  },
})
afterAll(() => {
  void server.stop(true)
})
const url = `http://127.0.0.1:${server.port}`

test('posts the import text with basic auth', async () => {
  await writeSamples(url, 'host_x{cluster="c"} 1 1000\n', { username: 'u', password: 'p' })
  expect(requests.at(-1)).toEqual({
    path: '/api/v1/import/prometheus',
    auth: `Basic ${btoa('u:p')}`,
    body: 'host_x{cluster="c"} 1 1000\n',
  })
})

test('sends nothing for an empty poll', async () => {
  const before = requests.length
  await writeSamples(url, '')
  expect(requests.length).toBe(before)
})

test('a rejected write throws without echoing credentials', async () => {
  status = 401
  const error = await writeSamples(url, 'host_x 1 1000\n', { username: 'u', password: 'secret-pw' }).catch((e: Error) => e)
  expect(error).toBeInstanceOf(Error)
  expect(String(error)).toContain('401')
  expect(String(error)).not.toContain('secret-pw')
})

test('a stalled VictoriaMetrics fails the write instead of hanging the poller', async () => {
  const error = await writeSamples(`${url}/stall`, 'host_x 1 1000\n').catch((e: Error) => e)
  expect(error).toBeInstanceOf(Error)
}, 15_000)
