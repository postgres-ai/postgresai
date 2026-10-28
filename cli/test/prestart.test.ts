import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { buildLocalInstallEnv, instancesUseDemoTarget } from "../bin/postgres-ai";

// Pre-baked monitoring images (postgres_ai-ansible#43): `--prestart` starts
// the infrastructure at first boot, `--continue-prestarted` finishes the
// install when the deploy arrives. Driven against a fake docker that logs
// every call and answers `ps` from the environment.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
let dir: string, projectDir: string, log: string;

const FAKE_DOCKER = `#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$*" in
  "info"|"compose version") exit 0 ;;
  "ps --filter name=grafana-with-datasources --filter name=pgwatch --format {{.Names}}")
    [ -n "$RUNNING" ] && printf '%s\\n' $RUNNING; exit 0 ;;
  "ps -aq --filter name=^target-db\\$ "*) case " $DEMO_CONTAINERS " in *" target-db "*) echo id-db ;; esac; exit 0 ;;
  "ps -aq --filter name=^target-standby\\$ "*) case " $DEMO_CONTAINERS " in *" target-standby "*) echo id-sb ;; esac; exit 0 ;;
  compose*" ps -q "*) svc="\${*##* }"; case " $READERS_RUNNING " in *" $svc "*) echo "id-$svc" ;; esac; exit 0 ;;
  inspect*) exit 1 ;;
  *) exit 0 ;;
esac
`;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/pgai-prestart-`);
  projectDir = `${dir}/project`;
  for (const p of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(p);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  // A git checkout, so local-install does not fetch a compose file.
  mkdirSync(`${projectDir}/.git`);
  log = `${dir}/docker.log`;
  writeFileSync(log, "");
  writeFileSync(`${dir}/bin/docker`, FAKE_DOCKER);
  chmodSync(`${dir}/bin/docker`, 0o755);
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync(
    [process.execPath, "--preload", resolve(import.meta.dir, "cli-offline-preload.ts"), cli, ...args],
    {
      cwd: dir,
      env: {
        PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`,
        PGAI_PROJECT_DIR: projectDir, DOCKER_LOG: log, ...extra,
      },
      timeout: 60000,
    },
  );
  return { exitCode: result.exitCode, stdout: result.stdout.toString(), stderr: result.stderr.toString() };
}

const calls = () => readFileSync(log, "utf8").split("\n").filter(Boolean);
const composeCalls = (verb: string) => calls().filter((c) => c.startsWith("compose ") && c.includes(` ${verb}`));
const env = () => readFileSync(`${projectDir}/.env`, "utf8");
const DB_URL = "postgresql://monitor:pw@127.0.0.1:1/appdb?sslmode=disable";

describe("mon local-install --prestart", () => {
  test("starts only the infrastructure, with fresh per-box secrets and local binds", () => {
    const r = run(["mon", "local-install", "--prestart"]);
    expect(r.exitCode).toBe(0);

    const ups = composeCalls("up");
    expect(ups.length).toBe(1);
    const services = ups[0].split(" up -d ")[1].split(" ");
    expect(services).toEqual([
      "sink-postgres", "sink-prometheus", "grafana", "monitoring_flask_backend",
      "self-cadvisor", "self-node-exporter", "self-postgres-exporter",
    ]);
    // Nothing that reads the target, and no demo database.
    for (const s of ["pgwatch-postgres", "pgwatch-prometheus", "postgres-reports", "target-db", "sources-generator"]) {
      expect(services).not.toContain(s);
    }

    expect(env()).toMatch(/^BIND_HOST=127\.0\.0\.1:$/m);
    expect(env()).toMatch(/^GRAFANA_BIND_HOST=127\.0\.0\.1:$/m);
    expect(env()).toMatch(/^VM_AUTH_PASSWORD=.+$/m);
    expect(env()).toMatch(/^REPLICATOR_PASSWORD=[0-9a-f]{64}$/m);

    // The bind-mount sources exist as files, so compose cannot make them directories.
    expect(statSync(`${projectDir}/instances.yml`).isFile()).toBe(true);
    const cfg = `${projectDir}/.pgwatch-config`;
    expect(statSync(cfg).isFile()).toBe(true);
    expect(statSync(cfg).mode & 0o777).toBe(0o600);
    expect(readFileSync(cfg, "utf8")).toMatch(/^grafana_password=.+$/m);
    expect(readFileSync(cfg, "utf8")).not.toMatch(/api_key/);

    expect(existsSync(`${projectDir}/.prestarted`)).toBe(true);
    expect(statSync(`${projectDir}/.prestarted`).mode & 0o777).toBe(0o600);
  });

  test("two boxes never share secrets", () => {
    run(["mon", "local-install", "--prestart"]);
    const first = { env: env(), cfg: readFileSync(`${projectDir}/.pgwatch-config`, "utf8") };
    rmSync(`${projectDir}/.env`);
    rmSync(`${projectDir}/.pgwatch-config`);
    rmSync(`${projectDir}/.prestarted`);
    run(["mon", "local-install", "--prestart"]);
    const pick = (s: string, k: string) => s.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1];
    expect(pick(env(), "VM_AUTH_PASSWORD")).not.toBe(pick(first.env, "VM_AUTH_PASSWORD"));
    expect(pick(readFileSync(`${projectDir}/.pgwatch-config`, "utf8"), "grafana_password"))
      .not.toBe(pick(first.cfg, "grafana_password"));
  });

  test("keeps bind hosts that are already set", () => {
    writeFileSync(`${projectDir}/.env`, "BIND_HOST=0.0.0.0:\n");
    expect(run(["mon", "local-install", "--prestart"]).exitCode).toBe(0);
    expect(env()).toMatch(/^BIND_HOST=0\.0\.0\.0:$/m);
    expect(env()).toMatch(/^GRAFANA_BIND_HOST=127\.0\.0\.1:$/m);
  });

  test("refuses on a stack that is already running, and changes nothing there", () => {
    const r = run(["mon", "local-install", "--prestart"], { RUNNING: "grafana-with-datasources" });
    expect(r.exitCode).toBe(1);
    expect(r.stderr).toContain("already running");
    expect(composeCalls("up")).toEqual([]);
    expect(existsSync(`${projectDir}/.prestarted`)).toBe(false);
  });

  test("refuses to take a target or a key", () => {
    for (const extra of [["--db-url", DB_URL], ["--api-key", "k"], ["--demo"], ["--continue-prestarted"]]) {
      const r = run(["mon", "local-install", "--prestart", ...extra]);
      expect(r.exitCode).toBe(1);
      expect(r.stderr).toContain("--prestart starts the stack without a target");
    }
    expect(composeCalls("up")).toEqual([]);
  });

  test("a failed start leaves no marker", () => {
    writeFileSync(`${dir}/bin/docker`, FAKE_DOCKER.replace('  *) exit 0 ;;', '  compose*" up "*) exit 3 ;;\n  *) exit 0 ;;'));
    const r = run(["mon", "local-install", "--prestart"]);
    expect(r.exitCode).not.toBe(0);
    expect(existsSync(`${projectDir}/.prestarted`)).toBe(false);
  });
});

describe("mon local-install --continue-prestarted", () => {
  const prestart = () => expect(run(["mon", "local-install", "--prestart"]).exitCode).toBe(0);

  test("takes the prestarted stack and recreates only what changed", () => {
    prestart();
    writeFileSync(log, "");
    const r = run(["mon", "local-install", "--continue-prestarted", "--db-url", DB_URL, "-y"],
      { RUNNING: "grafana-with-datasources" });
    expect(r.stderr).not.toContain("already running");
    expect(readFileSync(`${projectDir}/instances.yml`, "utf8")).toContain("127.0.0.1:1/appdb");
    const ups = composeCalls("up");
    expect(ups.some((c) => c.endsWith(" up -d"))).toBe(true);
    expect(ups.some((c) => c.includes("--force-recreate"))).toBe(false);
    expect(composeCalls("run").some((c) => c.includes("sources-generator"))).toBe(true);
  });

  test("recreates the target readers that already run (a re-run)", () => {
    prestart();
    writeFileSync(log, "");
    run(["mon", "local-install", "--continue-prestarted", "--db-url", DB_URL, "-y"], {
      RUNNING: "grafana-with-datasources pgwatch-postgres",
      READERS_RUNNING: "pgwatch-postgres postgres-reports",
    });
    const recreate = composeCalls("up").filter((c) => c.includes("--force-recreate"));
    expect(recreate.length).toBe(1);
    expect(recreate[0]).toMatch(/up -d --force-recreate --no-deps pgwatch-postgres postgres-reports$/);
  });

  test("reloads host metrics when the target brought scrape files", () => {
    prestart();
    mkdirSync(`${projectDir}/host-metrics`, { recursive: true });
    writeFileSync(`${projectDir}/host-metrics/clickhouse-x.yml`, "[]\n");
    writeFileSync(log, "");
    run(["mon", "local-install", "--continue-prestarted", "--db-url", DB_URL, "-y"], { RUNNING: "grafana-with-datasources" });
    expect(calls().some((c) => c.startsWith("compose ") && c.endsWith("kill -s SIGHUP sink-prometheus"))).toBe(true);
  });

  test("without the marker it is today's install, and a running stack still refuses", () => {
    const r = run(["mon", "local-install", "--continue-prestarted", "--db-url", DB_URL, "-y"],
      { RUNNING: "grafana-with-datasources" });
    expect(r.stdout).toContain("no .prestarted marker");
    expect(r.stderr).toContain("already running");
    expect(composeCalls("up").filter((c) => !c.includes("instance-jobs"))).toEqual([]);
  });

  test("without the marker on a stopped stack it installs as today (--force-recreate)", () => {
    run(["mon", "local-install", "--continue-prestarted", "--db-url", DB_URL, "-y"]);
    expect(composeCalls("up").some((c) => c.endsWith(" up -d --force-recreate"))).toBe(true);
  });

  test("the marker alone does not open a running stack to a plain re-run", () => {
    prestart();
    writeFileSync(log, "");
    const r = run(["mon", "local-install", "--db-url", DB_URL, "-y"], { RUNNING: "grafana-with-datasources" });
    expect(r.stderr).toContain("already running");
    expect(composeCalls("up").filter((c) => !c.includes("instance-jobs"))).toEqual([]);
  });
});

describe("the demo profile", () => {
  const parse = (content: string) =>
    Object.fromEntries(content.split("\n").map((l) => l.match(/^([A-Z_]+)=(.*)$/)).filter(Boolean).map((m) => [m![1], m![2]]));

  test("--demo turns it on, a non-demo install turns it off, other profiles stay", () => {
    expect(parse(buildLocalInstallEnv("", "1.0.0", null, null, undefined, true).content).COMPOSE_PROFILES).toBe("demo");
    const off = buildLocalInstallEnv("COMPOSE_PROFILES=instance-jobs,demo\n", "1.0.0", null, null, undefined, false).content;
    expect(parse(off).COMPOSE_PROFILES).toBe("instance-jobs");
    const none = buildLocalInstallEnv("COMPOSE_PROFILES=instance-jobs\n", "1.0.0").content;
    expect(parse(none).COMPOSE_PROFILES).toBe("instance-jobs");
  });

  test("instancesUseDemoTarget recognises the demo target only", () => {
    const f = `${dir}/i.yml`;
    writeFileSync(f, readFileSync(resolve(import.meta.dir, "../../instances.demo.yml"), "utf8"));
    expect(instancesUseDemoTarget(f)).toBe(true);
    writeFileSync(f, "- name: x\n  conn_str: postgresql://u:p@db.example.com:5432/target-db\n");
    expect(instancesUseDemoTarget(f)).toBe(false);
    expect(instancesUseDemoTarget(`${dir}/missing.yml`)).toBe(false);
  });

  test("local-install --demo runs with the profile on", () => {
    run(["mon", "local-install", "--demo", "-y"]);
    expect(env()).toMatch(/^COMPOSE_PROFILES=demo$/m);
  });

  test("mon start on a demo stack installed before the profile turns it on", () => {
    writeFileSync(`${projectDir}/instances.yml`, readFileSync(resolve(import.meta.dir, "../../instances.demo.yml"), "utf8"));
    writeFileSync(`${projectDir}/.env`, "PGAI_TAG=1.0.0\n");
    run(["mon", "start"]);
    expect(env()).toMatch(/^COMPOSE_PROFILES=demo$/m);
    expect(calls()).not.toContain("rm -f target-db");
  });

  test("mon update on a non-demo stack removes the old demo containers", () => {
    writeFileSync(`${projectDir}/instances.yml`, "- name: x\n  conn_str: postgresql://u:p@db.example.com:5432/app\n");
    writeFileSync(`${projectDir}/.env`, "PGAI_TAG=1.0.0\n");
    run(["mon", "update"], { DEMO_CONTAINERS: "target-db target-standby" });
    expect(calls()).toContain("rm -f target-db");
    expect(calls()).toContain("rm -f target-standby");
    expect(env()).not.toMatch(/COMPOSE_PROFILES=.*demo/);
  });

  test("an explicit demo profile is left alone", () => {
    writeFileSync(`${projectDir}/instances.yml`, "- name: x\n  conn_str: postgresql://u:p@db.example.com:5432/app\n");
    writeFileSync(`${projectDir}/.env`, "PGAI_TAG=1.0.0\nCOMPOSE_PROFILES=demo\n");
    run(["mon", "start"], { DEMO_CONTAINERS: "target-db" });
    expect(calls()).not.toContain("rm -f target-db");
  });
});
