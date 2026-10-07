import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// `mon local-install --db-url` is the main onboarding path, so it must set up
// ClickHouse host metrics exactly like `mon targets add` does.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const hostname = "li.pg.clickhouse.cloud";
const orgId = "ca04a310-730d-4ce0-93dd-39f2cd2d5e6f";
const serviceId = "0c330583-6396-86d0-82cd-ed0f23b0d38c";
const name = "li-pg-clickhouse-cloud-postgres";
let dir: string, projectDir: string, workerUrl: string, server: Worker, env: Record<string, string>;

beforeEach(async () => {
  dir = mkdtempSync(`${tmpdir()}/clickhouse-local-install-`);
  projectDir = `${dir}/project`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  // A git checkout, so local-install does not fetch a compose file from GitLab.
  mkdirSync(`${projectDir}/.git`);
  // No Docker: local-install stops at its first compose call, after the target step.
  writeFileSync(`${dir}/bin/docker`, "#!/bin/sh\nexit 1\n");
  chmodSync(`${dir}/bin/docker`, 0o755);
  workerUrl = URL.createObjectURL(new Blob([`
    const base = "/v1/organizations/${orgId}/postgres";
    const service = { id: "${serviceId}", name: "li", state: "running", hostname: "${hostname}" };
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
      if (request.headers.get("authorization") !== "Basic " + btoa("key:good-secret")) return new Response("Unauthorized", { status: 401 });
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
    PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir,
    CLICKHOUSE_ORG_ID: orgId, CLICKHOUSE_KEY_ID: "key", CLICKHOUSE_KEY_SECRET: "good-secret", CLICKHOUSE_API_URL: origin, PGAI_TEST_TLS_VERIFY: "success",
  };
});
afterEach(() => { server?.terminate(); URL.revokeObjectURL(workerUrl); rmSync(dir, { recursive: true, force: true }); });

function localInstall(dbUrl: string, extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, "--preload", resolve(import.meta.dir, "cli-offline-preload.ts"), cli, "mon", "local-install", "--db-url", dbUrl, "-y"], {
    cwd: dir, env: { ...env, ...extra }, timeout: 60000,
  });
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}

test("local-install --db-url sets up ClickHouse host metrics", () => {
  const result = localInstall(`postgres://monitor:password@${hostname}:5432/postgres?sslmode=require&channel_binding=require`);
  expect(result.stdout).toContain(`Monitoring target '${name}' added`);
  expect(result.stdout).toContain("Host metrics: ClickHouse Cloud Prometheus endpoint for service li (scraped every 60s)");
  const [scrape] = Bun.YAML.parse(readFileSync(`${projectDir}/host-metrics/clickhouse-${name}.yml`, "utf8")) as any[];
  expect(scrape.job_name).toBe(`clickhouse-${name}`);
  expect(scrape.static_configs[0].labels).toMatchObject({ cluster: "default", node_name: name });
  expect(readFileSync(`${projectDir}/host-metrics/clickhouse-${name}.secret`, "utf8")).toBe("good-secret");
  expect(statSync(`${projectDir}/host-metrics/clickhouse-${name}.secret`).mode & 0o777).toBe(0o600);
  const saved = readFileSync(`${projectDir}/instances.yml`, "utf8");
  expect(saved).toContain(`postgres://monitor:password@${hostname}:5432/postgres?sslmode=verify-full`);
  expect(saved).not.toContain("channel_binding");
  expect(result.stderr).toContain("Warning: the collector can't do channel binding; it connects with TLS and full certificate verification; upgraded to sslmode=verify-full");
  expect(result.stdout + result.stderr).not.toContain("good-secret");
});

test("local-install --db-url keeps the Postgres target when host metrics fail", () => {
  const result = localInstall(`postgresql://monitor:password@${hostname}:5432/postgres`, { CLICKHOUSE_KEY_SECRET: "bad-secret" });
  expect(result.stderr).toContain("ClickHouse Cloud rejected the API key (401)");
  expect(readFileSync(`${projectDir}/instances.yml`, "utf8")).toContain(`name: ${name}`);
  expect(existsSync(`${projectDir}/host-metrics/clickhouse-${name}.yml`)).toBe(false);
  expect(result.stdout + result.stderr).not.toContain("bad-secret");
});

test("local-install stops when the target is not saved", () => {
  const result = localInstall("postgresql://monitor:password@[bad/postgres");
  expect(result.exitCode).toBe(1);
  expect(result.stderr).toContain("Invalid connection string format");
  expect(result.stdout).not.toContain("Step 3");
});

test("interactive local-install stops when the target is not saved", async () => {
  // Answer each prompt only once it is shown: piped stdin at EOF closes the prompt reader.
  const proc = Bun.spawn([process.execPath, cli, "mon", "local-install"], { cwd: dir, env, stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  const answers: [string, string][] = [
    ["API key? (Y/n)", "n\n"],
    ["add a PostgreSQL instance now? (Y/n)", "y\n"],
    ["Enter connection string", "postgresql://monitor:password@[bad/postgres\n"],
  ];
  let stdout = "";
  const decoder = new TextDecoder();
  const reader = proc.stdout.getReader();
  const timer = setTimeout(() => proc.kill(), 30000);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    stdout += decoder.decode(value);
    while (answers.length && stdout.includes(answers[0][0])) {
      proc.stdin.write(answers.shift()![1]);
      proc.stdin.flush();
    }
  }
  clearTimeout(timer);
  const exitCode = await proc.exited;
  const stderr = await new Response(proc.stderr).text();
  expect(answers).toEqual([]);
  expect(stderr).toContain("Invalid connection string format");
  expect(stdout).not.toContain("Step 3");
  expect(exitCode).toBe(1);
});

test("read-only commands work in a project directory they cannot write", () => {
  writeFileSync(`${projectDir}/instances.yml`, "- name: ro\n  conn_str: postgresql://u:p@h:5432/d\n  is_enabled: true\n");
  chmodSync(projectDir, 0o555);
  try {
    const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", "list"], { cwd: projectDir, env, timeout: 30000 });
    expect(result.stderr.toString()).not.toContain("EACCES");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.toString()).toContain("Target: ro");
  } finally {
    chmodSync(projectDir, 0o755);
  }
});

test("local-install refuses an unverified channel_binding=require URL before saving", () => {
  const result = localInstall(`postgres://monitor:password@${hostname}:5432/postgres?sslmode=require&channel_binding=require`, { PGAI_TEST_TLS_VERIFY: "DEPTH_ZERO_SELF_SIGNED_CERT" });
  expect(result.exitCode).toBe(1);
  expect(result.stderr).toContain("sslrootcert=<CA file>");
  expect(result.stderr).toContain("removing channel_binding=require from the URL");
  expect(existsSync(`${projectDir}/instances.yml`)).toBe(false);
  expect(existsSync(`${projectDir}/host-metrics`)).toBe(false);
  expect(result.stdout).not.toContain("Step 3");
});
