import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// pgai-mon reloads the host-metrics files by itself: `targets add` and `remove`
// write and delete them and signal nothing.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const hostname = "reload.pg.clickhouse.cloud";
const conn = `postgresql://monitor:password@${hostname}:5432/postgres`;
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
let dir: string, projectDir: string, log: string, workerUrl: string, server: Worker;
let env: Record<string, string>;

beforeEach(async () => {
  dir = mkdtempSync(`${tmpdir()}/clickhouse-reload-`);
  projectDir = `${dir}/project`;
  const fakeBin = `${dir}/bin`;
  log = `${dir}/docker.log`;
  for (const path of [projectDir, fakeBin, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(log, "");
  writeFileSync(`${fakeBin}/docker`, `#!/bin/sh
if [ "$1" = info ]; then exit 0; fi
if [ "$1" = compose ] && [ "$2" = version ]; then exit 0; fi
shift 3
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  *) exit 0 ;;
esac
`);
  chmodSync(`${fakeBin}/docker`, 0o755);
  // Serve on a separate thread: spawnSync blocks this thread's event loop.
  workerUrl = URL.createObjectURL(new Blob([`
    const base = "/v1/organizations/${orgId}/postgres";
    const service = { id: "${serviceId}", name: "ch", state: "running", hostname: "${hostname}" };
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
      if (request.headers.get("authorization") !== "Basic " + btoa("key:good-secret")) {
        return new Response("Unauthorized", { status: 401 });
      }
      const path = new URL(request.url).pathname;
      if (path === base) return Response.json({ result: [service] });
      if (path === base + "/${serviceId}") return Response.json({ result: service });
      return new Response("Not found", { status: 404 });
    } });
    postMessage(server.url.origin);
  `], { type: "application/javascript" }));
  server = new Worker(workerUrl);
  const origin = await new Promise<string>((resolve, reject) => {
    server.onmessage = (event) => resolve(event.data);
    server.onerror = (event) => reject(new Error(event.message));
  });
  env = {
    PATH: `${fakeBin}:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`,
    PGAI_PROJECT_DIR: projectDir, CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: "key",
    CLICKHOUSE_KEY_SECRET: "good-secret", CLICKHOUSE_API_URL: origin, FAKE_DOCKER_LOG: log,
  };
});
afterEach(() => { server?.terminate(); URL.revokeObjectURL(workerUrl); rmSync(dir, { recursive: true, force: true }); });

function run(args: string[]) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], {
    cwd: dir, env, timeout: 20000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}
// Any docker call that would signal, exec into, or (re)start a container.
function signals() {
  return readFileSync(log, "utf8").split("\n").filter((line) => /^(kill|exec|run|up|start|restart)(?: |$)|vmalert|sink-prometheus/.test(line));
}

test("targets add writes the scrape job and key and signals nothing", () => {
  const result = run(["add", conn, "ch"]);
  expect(result.exitCode, result.stderr).toBe(0);
  expect(result.stdout).toContain("Host metrics: ClickHouse Cloud Prometheus endpoint for service");
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.yml`)).toBe(true);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.secret`)).toBe(true);
  expect(signals()).toEqual([]);
  expect(existsSync(`${projectDir}/.env`) ? readFileSync(`${projectDir}/.env`, "utf8") : "").not.toContain("COMPOSE_PROFILES");
});

test("targets remove deletes the scrape job and key and signals nothing", () => {
  expect(run(["add", conn, "ch"]).exitCode).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"]);
  expect(result.exitCode, result.stderr).toBe(0);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.yml`)).toBe(false);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.secret`)).toBe(false);
  expect(signals()).toEqual([]);
});

test("targets add output never carries the key secret", () => {
  const result = run(["add", conn, "ch"]);
  expect(result.stdout + result.stderr).not.toContain("good-secret");
  const denied = Bun.spawnSync([process.execPath, cli, "mon", "targets", "add", conn, "ch2"], {
    cwd: dir, env: { ...env, CLICKHOUSE_KEY_SECRET: "bad-secret-value" }, timeout: 20000,
  });
  expect(denied.exitCode).toBe(1);
  expect(denied.stdout.toString() + denied.stderr.toString()).not.toContain("bad-secret-value");
});

test("targets add help names the least-privilege key role and where the key is kept", () => {
  const help = Bun.spawnSync([process.execPath, cli, "mon", "targets", "add", "--help"], { cwd: dir, env, timeout: 20000 }).stdout.toString();
  expect(help).toContain("Basic Service API Reader");
  expect(help).toContain("host-metrics/clickhouse-my-db.secret (the key secret, mode 0600)");
});
