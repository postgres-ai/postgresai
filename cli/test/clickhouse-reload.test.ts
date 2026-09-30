import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { HOST_METRICS_VERIFY_SCRIPT } from "../lib/clickhouse";
import pkg from "../package.json";

const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const hostname = "reload.pg.clickhouse.cloud";
const conn = `postgresql://monitor:password@${hostname}:5432/postgres`;
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
const execLine = `exec -T sink-prometheus sh -c grep -q '^scrape_config_files:' /postgres_ai_configs/prometheus/prometheus.yml && test -f "$1" sh /etc/pgai/host-metrics/clickhouse-ch.yml`;
const killLine = "kill -s SIGHUP sink-prometheus";
const probeLine = "ps -a -q --status running --status paused sink-prometheus";
const verifyRemoveLine = `exec -T sink-prometheus sh -c ${HOST_METRICS_VERIFY_SCRIPT} sh "scrapePool":"clickhouse-ch" absent`;
const reloadError = "Reloading sink-prometheus failed. Run 'postgresai mon restart' to load the host metrics change.";
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
  ps) [ -n "$FAKE_PS_CODE" ] && exit "$FAKE_PS_CODE"; case " $* " in *" --status \${FAKE_PS_STATE:-running} "*) echo 0123abcd ;; esac; exit 0 ;;
  kill) exit \${FAKE_KILL_CODE:-0} ;;
  exec) case "$*" in *api/v1/targets*) exit \${FAKE_VERIFY_CODE:-0} ;; esac; exit \${FAKE_EXEC_CODE:-0} ;;
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

function run(args: string[], codes: { FAKE_KILL_CODE?: string; FAKE_EXEC_CODE?: string; FAKE_VERIFY_CODE?: string; FAKE_PS_STATE?: string; FAKE_PS_CODE?: string } = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], {
    cwd: dir, env: { ...env, ...codes }, timeout: 20000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}
function verifyAddLine() {
  const revision = readFileSync(`${projectDir}/host-metrics/clickhouse-ch.yml`, "utf8").match(/__pgai_rev: (r[0-9a-f]{16})\n/)![1];
  return `exec -T sink-prometheus sh -c ${HOST_METRICS_VERIFY_SCRIPT} sh "__pgai_rev":"${revision}" present`;
}
function reloadLog() {
  return readFileSync(log, "utf8").split("\n").filter((line) => /^(kill|exec|ps)(?: |$)/.test(line));
}

test("targets add fails when the sink-prometheus reload fails", () => {
  const result = run(["add", conn, "ch"], { FAKE_KILL_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  expect(result.stderr).toContain(reloadError);
  expect(result.stdout).not.toContain("Host metrics: ClickHouse Cloud");
  expect(reloadLog()).toEqual([execLine, killLine]);
});

test("targets add reports an older stack instead of success", () => {
  const result = run(["add", conn, "ch"], { FAKE_EXEC_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  // `mon update` keeps PGAI_TAG, so without this step the stack restarts on the old config image.
  expect(result.stderr).toContain(`sink-prometheus cannot load host metrics: it is not running, or this monitoring stack predates host metrics support. To upgrade the stack to ${pkg.version}, set PGAI_TAG=${pkg.version} in .env, then run 'postgresai mon update', 'postgresai mon stop' and 'postgresai mon start'. The scrape files are saved and will be picked up.`);
  expect(result.stdout).not.toContain("Host metrics: ClickHouse Cloud");
  expect(reloadLog()).toEqual([execLine]);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.yml`)).toBe(true);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.secret`)).toBe(true);
});

test("targets add reports host metrics after a verified reload", () => {
  const result = run(["add", conn, "ch"]);
  expect(result.exitCode, result.stderr).toBe(0);
  expect(result.stdout).toContain("Host metrics: ClickHouse Cloud Prometheus endpoint for service");
  expect(result.stderr).not.toContain("sink-prometheus");
  expect(reloadLog()).toEqual([execLine, killLine, verifyAddLine()]);
});

test("targets remove fails when the sink-prometheus reload fails", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_KILL_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  expect(result.stderr).toContain(reloadError);
  expect(reloadLog()).toEqual([probeLine, killLine]);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.yml`)).toBe(false);
});

test("targets add fails when sink-prometheus rejects the new scrape job", () => {
  const result = run(["add", conn, "ch"], { FAKE_VERIFY_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  expect(result.stderr).toContain("sink-prometheus did not load the scrape job 'clickhouse-ch' after the reload. Check 'docker logs sink-prometheus' for the error. The scrape files are saved.");
  expect(result.stdout).not.toContain("Host metrics: ClickHouse Cloud");
  expect(reloadLog()).toEqual([execLine, killLine, verifyAddLine()]);
});

test("targets remove fails when sink-prometheus still scrapes the removed job", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_VERIFY_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  expect(result.stderr).toContain("sink-prometheus still scrapes 'clickhouse-ch' after the reload. Check 'docker logs sink-prometheus' for the error.");
  expect(reloadLog()).toEqual([probeLine, killLine, verifyRemoveLine]);
});

test("targets remove succeeds after a verified reload", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"]);
  expect(result.exitCode, result.stderr).toBe(0);
  expect(result.stderr).not.toContain("sink-prometheus");
  expect(reloadLog()).toEqual([probeLine, killLine, verifyRemoveLine]);
});

test("targets remove on a stopped stack succeeds without a reload", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_PS_STATE: "exited", FAKE_EXEC_CODE: "1", FAKE_VERIFY_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(0);
  expect(result.stderr).not.toContain("still scrapes");
  expect(result.stdout).toContain("sink-prometheus is not running; it will not load 'clickhouse-ch' when it starts.");
  expect(reloadLog()).toEqual([probeLine]);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-ch.yml`)).toBe(false);
});

test("a failed targets add leaves the stack alone", () => {
  const result = run(["add", conn, "ch"], { FAKE_EXEC_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(1);
  const calls = readFileSync(log, "utf8").split("\n").filter(Boolean);
  // `up -d pgwatch-*` would start a stopped sink-prometheus again as a dependency.
  expect(calls.filter((line) => /^(up|run|start|restart)(?: |$)/.test(line))).toEqual([]);
  expect(result.stdout).not.toContain("Monitoring target configuration applied");
  expect(result.stderr).toContain("The Postgres target is saved but not applied. Fix the error above and re-run this command.");
});

test("targets add output never carries the key secret", () => {
  for (const codes of [{}, { FAKE_EXEC_CODE: "1" }, { FAKE_KILL_CODE: "1" }, { FAKE_VERIFY_CODE: "1" }]) {
    const result = run(["add", conn, "ch"], codes);
    expect(result.stdout + result.stderr).not.toContain("good-secret");
  }
  const denied = Bun.spawnSync([process.execPath, cli, "mon", "targets", "add", conn, "ch"], {
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

test.each([["paused", {}], ["running", { FAKE_EXEC_CODE: "1" }]])("targets remove fails when sink-prometheus is %s but cannot be checked", (state, codes) => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_PS_STATE: state, FAKE_VERIFY_CODE: "1", ...codes });
  expect(result.exitCode, result.stderr).toBe(1);
  expect(result.stdout).not.toContain("sink-prometheus is not running");
});

// docker-compose v1 has no `ps --status`: the state is unknown, so remove
// reloads and verifies as for a running sink-prometheus.
test("targets remove reloads when the sink-prometheus state is unknown", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_PS_CODE: "1" });
  expect(result.exitCode, result.stderr).toBe(0);
  expect(reloadLog()).toEqual([probeLine, killLine, verifyRemoveLine]);
});

test("targets remove without a sink-prometheus container succeeds without a reload", () => {
  const added = run(["add", conn, "ch"]);
  expect(added.exitCode, added.stderr).toBe(0);
  writeFileSync(log, "");
  const result = run(["remove", "ch"], { FAKE_PS_STATE: "none" });
  expect(result.exitCode, result.stderr).toBe(0);
  expect(result.stdout).toContain("sink-prometheus is not running; it will not load 'clickhouse-ch' when it starts.");
  expect(reloadLog()).toEqual([probeLine]);
});
