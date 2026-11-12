import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import { resolve } from "path";
import { composeDefinesService, composeProjectName, failedServices, oldStackVolumeNames, parseComposePs, redactStoreUrl, stackProblems, stackSwitchNotice, storeNotice } from "../lib/stack";

/**
 * The switch of an existing install from the old pgwatch + VictoriaMetrics
 * stack to pgai-mon, and the install/update path on the new compose file.
 *
 * Docker is a shim on PATH that records its argv (and COMPOSE_PROFILES), keeps
 * a copy of the compose file a `down` was given, and answers `ps` / `volume ls`
 * from environment variables. The compose files are the fixtures under
 * fixtures/compose/, fetched through the test-only PGAI_COMPOSE_SOURCE seam.
 */

const FIXTURES = resolve(import.meta.dir, "fixtures", "compose");
const NEW_COMPOSE = resolve(FIXTURES, "pgai-mon.yml");
const OLD_COMPOSE = resolve(FIXTURES, "old-stack.yml");
const OLD_ENV = [
  "PGAI_TAG=0.17.0",
  "REPLICATOR_PASSWORD=keep-replicator",
  "VM_AUTH_USERNAME=vmauth",
  "VM_AUTH_PASSWORD=old-vm-password",
  "VM_DELETE_AUTH_KEY=old-delete-key",
  "VM_RETENTION_PERIOD=720h",
  "",
].join("\n");

let tempDir: string;

beforeAll(() => {
  tempDir = fs.realpathSync(fs.mkdtempSync(resolve(os.tmpdir(), "pgai-stack-switch-")));
});

afterAll(() => {
  if (tempDir && fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true });
});

function makeProject(name: string, compose: string, env = OLD_ENV) {
  const dir = resolve(tempDir, name);
  const bin = resolve(tempDir, `${name}-bin`);
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(bin, { recursive: true });
  fs.copyFileSync(compose, resolve(dir, "docker-compose.yml"));
  fs.writeFileSync(resolve(dir, ".env"), env, { mode: 0o600 });
  fs.writeFileSync(resolve(dir, "instances.yml"), "- name: keep-me\n  conn_str: postgresql://m:p@h:5432/db\n");
  fs.writeFileSync(resolve(dir, ".pgwatch-config"), "grafana_password=pw\n", { mode: 0o600 });
  const log = resolve(bin, "docker-calls.log");
  fs.writeFileSync(
    resolve(bin, "docker"),
    [
      "#!/bin/sh",
      `printf '%s |PROFILES=%s\\n' "$*" "\${COMPOSE_PROFILES-}" >> ${JSON.stringify(log)}`,
      'prev=""; file=""',
      'for a in "$@"; do [ "$prev" = "-f" ] && file="$a"; prev="$a"; done',
      `case " $* " in *" down "*) cp "$file" ${JSON.stringify(resolve(bin, "down-compose.yml"))} ;; esac`,
      'case " $* " in',
      '  " info ") exit 0 ;;',
      '  *" compose version "*) exit 0 ;;',
      '  *" volume ls "*) for v in $VOLUMES; do echo "$v"; done; exit 0 ;;',
      '  *" ps -q sink-prometheus "*) [ -n "$OLD_RUNNING" ] && echo old123; exit 0 ;;',
      '  *" ps -q pgai-mon "*) [ -n "$NEW_RUNNING" ] && echo new123; exit 0 ;;',
      '  *" pull "*) exit ${PULL_EXIT:-0} ;;',
      '  " ps --filter"*) for c in $RUNNING; do echo "$c"; done; exit 0 ;;',
      "esac",
      "exit 0",
    ].join("\n"),
    { mode: 0o755 },
  );
  return {
    dir,
    calls: (): string[] => (fs.existsSync(log) ? fs.readFileSync(log, "utf8").trim().split("\n") : []),
    downCompose: (): string | null => {
      const f = resolve(bin, "down-compose.yml");
      return fs.existsSync(f) ? fs.readFileSync(f, "utf8") : null;
    },
    run(args: string[], env: Record<string, string> = {}) {
      const childEnv: Record<string, string | undefined> = { ...process.env };
      for (const key of Object.keys(childEnv)) if (key.startsWith("PGAI_") || key.startsWith("COMPOSE_")) delete childEnv[key];
      const result = Bun.spawnSync([process.execPath, resolve(import.meta.dir, "..", "bin", "postgres-ai.ts"), ...args], {
        cwd: dir,
        env: {
          ...childEnv,
          HOME: dir,
          XDG_CONFIG_HOME: resolve(dir, ".config"),
          PATH: `${bin}:${process.env.PATH}`,
          NODE_ENV: "test",
          PGAI_COMPOSE_SOURCE: NEW_COMPOSE,
          ...env,
        },
        timeout: 30000,
      });
      return {
        status: result.exitCode,
        stdout: new TextDecoder().decode(result.stdout),
        stderr: new TextDecoder().decode(result.stderr),
      };
    },
  };
}

const readEnv = (dir: string) => fs.readFileSync(resolve(dir, ".env"), "utf8");
const composeCalls = (calls: string[]) => calls.filter((c) => c.startsWith("compose "));

describe("mon update on an old-stack install", () => {
  test("switches to pgai-mon: new compose, token, old containers removed through the old file, stack started, volumes kept", () => {
    const p = makeProject("old-running", OLD_COMPOSE);
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", VOLUMES: "old-running_victoria_metrics_data other_volume" });
    expect(r.status).toBe(0);

    // The compose file is the new stack's; the old one is backed up.
    const compose = fs.readFileSync(resolve(p.dir, "docker-compose.yml"), "utf8");
    expect(composeDefinesService(compose, "pgai-mon")).toBe(true);
    const backups = fs.readdirSync(p.dir).filter((f) => f.startsWith("docker-compose.yml.bak-0.17.0-"));
    expect(backups.length).toBe(1);
    expect(fs.readFileSync(resolve(p.dir, backups[0]), "utf8")).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));

    // .env gains the token, keeps everything else (the additive path).
    const env = readEnv(p.dir);
    expect(env).toMatch(/^PGAI_MON_TOKEN=[0-9a-f]{64}$/m);
    expect(env).toContain("VM_RETENTION_PERIOD=720h");
    expect(env).toContain("REPLICATOR_PASSWORD=keep-replicator");
    expect(fs.statSync(resolve(p.dir, ".env")).mode & 0o777).toBe(0o600);
    expect(fs.readFileSync(resolve(p.dir, "instances.yml"), "utf8")).toContain("keep-me");

    // Running state is read before the file changes; then pull, old down, new up.
    const calls = composeCalls(p.calls());
    const step = (re: RegExp) => calls.findIndex((c) => re.test(c));
    const probe = step(/ ps -q sink-prometheus /);
    const pull = step(/ pull /);
    const down = step(/ down --remove-orphans /);
    const up = step(/ up -d --remove-orphans /);
    expect(probe).toBeGreaterThanOrEqual(0);
    expect(pull).toBeGreaterThan(probe);
    expect(down).toBeGreaterThan(pull);
    expect(up).toBeGreaterThan(down);
    // The down ran on the OLD file, with every old profile active; the up on the new file.
    expect(p.downCompose()).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));
    expect(calls[down]).toContain("PROFILES=demo,host-metrics,instance-jobs,rds");
    expect(calls[up]).toContain(`-f ${resolve(p.dir, "docker-compose.yml")} up`);
    // No volume is touched, and the old file's temp copy is gone.
    expect(p.calls().some((c) => /down .*-v\b|volume rm/.test(c))).toBe(false);
    expect(fs.readdirSync(p.dir).some((f) => f.startsWith(".docker-compose.old-stack"))).toBe(false);
    // The token never reaches argv.
    const token = env.match(/^PGAI_MON_TOKEN=(.*)$/m)![1];
    expect(p.calls().some((c) => c.includes(token))).toBe(false);

    // What happened, and how to remove the old volumes (only the ones that exist).
    expect(r.stdout).toContain("Switched this install from pgwatch + VictoriaMetrics to pgai-mon.");
    expect(r.stdout).toContain("Metric history starts fresh");
    expect(r.stdout).toContain("docker volume rm old-running_victoria_metrics_data\n");
    expect(r.stdout).not.toContain("old-running_sink_postgres_data");
    expect(r.stdout).toContain(storeNotice(undefined)!);
  });

  test("a stopped old stack is switched but not started", () => {
    const p = makeProject("old-stopped", OLD_COMPOSE);
    const r = p.run(["mon", "update"]);
    expect(r.status).toBe(0);
    const calls = composeCalls(p.calls());
    expect(calls.some((c) => / down --remove-orphans /.test(c))).toBe(true);
    expect(calls.some((c) => / up /.test(c))).toBe(false);
    expect(r.stdout).toContain("the old stack was not running, so the new one was not started");
  });

  test("no new compose file: the old stack is left as it is and the command fails", () => {
    const p = makeProject("old-no-fetch", OLD_COMPOSE);
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", PGAI_COMPOSE_SOURCE: resolve(tempDir, "missing.yml") });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("still the old pgwatch + VictoriaMetrics stack's");
    expect(fs.readFileSync(resolve(p.dir, "docker-compose.yml"), "utf8")).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));
    expect(composeCalls(p.calls()).some((c) => / (pull|down|up) /.test(c))).toBe(false);
  });

  test("a failed pull leaves the old stack running and puts its compose file back, so a re-run retries", () => {
    const p = makeProject("old-pull-fails", OLD_COMPOSE);
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", PULL_EXIT: "1" });
    expect(r.status).toBe(1);
    expect(fs.readFileSync(resolve(p.dir, "docker-compose.yml"), "utf8")).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));
    expect(composeCalls(p.calls()).some((c) => / (down|up) /.test(c))).toBe(false);
    expect(r.stderr).toContain("re-run 'postgresai mon update' to switch");
  });

  test("mon update-config switches the same way", () => {
    const p = makeProject("old-update-config", OLD_COMPOSE);
    const r = p.run(["mon", "update-config"], { OLD_RUNNING: "1" });
    expect(r.status).toBe(0);
    const calls = composeCalls(p.calls());
    expect(calls.some((c) => / down --remove-orphans /.test(c))).toBe(true);
    expect(calls.some((c) => / up -d --remove-orphans /.test(c))).toBe(true);
    expect(calls.some((c) => / run /.test(c))).toBe(false);
    expect(readEnv(p.dir)).toMatch(/^PGAI_MON_TOKEN=[0-9a-f]{64}$/m);
  });
});

describe("mon update on a pgai-mon install", () => {
  test("pulls and switches nothing", () => {
    const p = makeProject("new-update", NEW_COMPOSE, "PGAI_TAG=0.18.0\nPGAI_MON_TOKEN=" + "a".repeat(64) + "\nPGAI_MON_STORE=s3://bucket/prefix\n");
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1" });
    expect(r.status).toBe(0);
    const calls = composeCalls(p.calls());
    expect(calls.some((c) => / pull /.test(c))).toBe(true);
    expect(calls.some((c) => / (down|up) /.test(c))).toBe(false);
    expect(readEnv(p.dir)).toContain("PGAI_MON_TOKEN=" + "a".repeat(64));
    expect(r.stdout).not.toContain("Switched this install");
  });

});

describe("mon local-install", () => {
  test("on a stopped old-stack install: rewrites .env for pgai-mon, removes the old containers, starts the new stack", () => {
    const p = makeProject("li-old-stopped", OLD_COMPOSE);
    const r = p.run(["mon", "local-install", "-y"], { VOLUMES: "li-old-stopped_victoria_metrics_data li-old-stopped_sink_postgres_data" });
    expect(r.status).toBe(0);

    const env = readEnv(p.dir);
    expect(env).toMatch(/^PGAI_MON_TOKEN=[0-9a-f]{64}$/m);
    // The old stack's keys are dropped, other operator keys kept.
    expect(env).not.toMatch(/^VM_(AUTH_|DELETE_)/m);
    expect(env).toContain("VM_RETENTION_PERIOD=720h");
    expect(r.stdout).not.toMatch(/Preserved existing \.env settings:.*VM_AUTH/);

    const calls = composeCalls(p.calls());
    const down = calls.findIndex((c) => / down --remove-orphans /.test(c));
    const up = calls.findIndex((c) => / up -d --force-recreate --remove-orphans/.test(c));
    expect(down).toBeGreaterThanOrEqual(0);
    expect(up).toBeGreaterThan(down);
    expect(p.downCompose()).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));
    expect(calls.some((c) => /sources-generator/.test(c))).toBe(false);

    expect(r.stdout).toContain("pgai-mon: metrics collection, storage and SQL");
    expect(r.stdout).toContain("docker volume rm li-old-stopped_victoria_metrics_data li-old-stopped_sink_postgres_data");
    expect(r.stdout).toContain(storeNotice(undefined)!);
    expect(r.stdout).not.toContain("VictoriaMetrics Auth");
  });

  test("on a running old stack: refuses and changes nothing", () => {
    const p = makeProject("li-old-running", OLD_COMPOSE);
    const r = p.run(["mon", "local-install", "-y"], { OLD_RUNNING: "1" });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("'postgresai mon update'");
    expect(readEnv(p.dir)).toBe(OLD_ENV);
    expect(fs.readFileSync(resolve(p.dir, "docker-compose.yml"), "utf8")).toBe(fs.readFileSync(OLD_COMPOSE, "utf8"));
  });

  test("on the new stack: no switch, no notice about old volumes; a set store is named", () => {
    const p = makeProject("li-new", NEW_COMPOSE, "PGAI_TAG=0.18.0\nPGAI_MON_STORE=s3://bucket/prefix\n");
    const r = p.run(["mon", "local-install", "-y"]);
    expect(r.status).toBe(0);
    expect(composeCalls(p.calls()).some((c) => / down /.test(c))).toBe(false);
    expect(r.stdout).not.toContain("Switched this install");
    expect(r.stdout).not.toContain(storeNotice(undefined)!);
    expect(r.stdout).toContain("Metrics history goes to s3://bucket/prefix");
    expect(readEnv(p.dir)).toContain("PGAI_MON_STORE=s3://bucket/prefix");
  });
});

describe("stack helpers", () => {
  test("composeDefinesService reads the services map only", () => {
    expect(composeDefinesService(fs.readFileSync(OLD_COMPOSE, "utf8"), "sink-prometheus")).toBe(true);
    expect(composeDefinesService(fs.readFileSync(NEW_COMPOSE, "utf8"), "sink-prometheus")).toBe(false);
    expect(composeDefinesService("<!doctype html><html></html>", "pgai-mon")).toBe(false);
    expect(composeDefinesService("pgai-mon: {}\n", "pgai-mon")).toBe(false);
    expect(composeDefinesService(null, "pgai-mon")).toBe(false);
  });

  test("composeProjectName follows compose: explicit name first, else the directory, normalized", () => {
    expect(composeProjectName("monitoring")).toBe("monitoring");
    expect(composeProjectName("My.Box_1")).toBe("mybox_1");
    expect(composeProjectName("monitoring", "Postgres_AI")).toBe("postgres_ai");
    expect(oldStackVolumeNames("monitoring")).toEqual(["monitoring_victoria_metrics_data", "monitoring_sink_postgres_data"]);
  });

  test("the switch notice names no volume when none is left", () => {
    expect(stackSwitchNotice([])).not.toContain("docker volume rm");
  });

  test("the store notice: the local volume when unset or blank, else the store without credentials", () => {
    expect(storeNotice(undefined)).toContain("PGAI_MON_STORE");
    expect(storeNotice("  ")).toContain("PGAI_MON_STORE");
    expect(storeNotice("s3://b/p")).toBe("Metrics history goes to s3://b/p");
    expect(redactStoreUrl("s3://AKIA:secret@bucket/prefix?X-Amz-Security-Token=t")).toBe("s3://bucket/prefix");
    expect(storeNotice("s3://key:secret@b/p?token=t")).not.toMatch(/key|secret|token/);
  });

  test("parseComposePs reads both compose output shapes and skips noise", () => {
    const a = { Service: "pgai-mon", State: "running", Health: "healthy", ExitCode: 0 };
    const b = { Service: "config-init", State: "exited", Health: "", ExitCode: 0 };
    expect(parseComposePs(JSON.stringify([a, b]))).toEqual([a, b]);
    expect(parseComposePs(`${JSON.stringify(a)}\nWARN something\n${JSON.stringify(b)}\n`)).toEqual([a, b]);
    expect(parseComposePs("")).toEqual([]);
    expect(parseComposePs("[not json")).toEqual([]);
  });

  test("stackProblems: every expected service running and healthy, one-shots exited 0", () => {
    const ok = [
      { Service: "config-init", State: "exited", ExitCode: 0 },
      { Service: "pgai-mon", State: "running", Health: "healthy" },
      { Service: "postgres-reports", State: "running", Health: "" },
    ];
    const expected = ["config-init", "pgai-mon", "postgres-reports"];
    expect(stackProblems(expected, ok)).toEqual([]);
    expect(stackProblems([...expected, "grafana"], ok)).toEqual(["grafana: missing"]);
    expect(stackProblems(expected, ok.slice(1))).toEqual([]);
    const broken = [
      { Service: "config-init", State: "exited", ExitCode: 1 },
      { Service: "pgai-mon", State: "restarting" },
      { Service: "postgres-reports", State: "created" },
      { Service: "grafana", State: "running", Health: "unhealthy" },
      { Service: "target-db", State: "running", Health: "starting" },
    ];
    expect(stackProblems([...expected, "grafana", "target-db"], broken)).toEqual([
      "config-init: exited (1)", "pgai-mon: restarting", "postgres-reports: created", "grafana: unhealthy", "target-db: starting",
    ]);
    // Only the failed ones get their logs printed after a failed `up`.
    expect(failedServices(broken)).toEqual(["config-init", "pgai-mon", "grafana"]);
    expect(failedServices(ok)).toEqual([]);
  });
});
