import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { load } from "js-yaml";

// `mon targets add` copies the target's cluster/node_name into the file its
// host metrics collector reads, for Supabase and RDS as for ClickHouse.
const cli = resolve(import.meta.dir, "../bin/postgres-ai.ts");
const ref = "abcdefghijklmnopqrst";
// A platform-provisioned box connects through the pooler as postgres_ai_mon.<ref>.
const supabase = (user = `postgres_ai_mon.${ref}`) => `postgresql://${user}:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres`;
const rds = "postgresql://u:pw@mydb.c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres";
let dir: string, projectDir: string, log: string, env: Record<string, string>;

beforeEach(() => {
  dir = mkdtempSync(`${tmpdir()}/host-metrics-providers-`);
  projectDir = `${dir}/project`;
  log = `${dir}/docker.log`;
  for (const path of [projectDir, `${dir}/bin`, `${dir}/home`, `${dir}/xdg`]) mkdirSync(path);
  writeFileSync(`${projectDir}/docker-compose.yml`, "services: {}\n");
  writeFileSync(log, "");
  // Every compose call succeeds; sink-prometheus is running.
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh
if [ "$1" = info ]; then exit 0; fi
if [ "$1" = compose ] && [ "$2" = version ]; then exit 0; fi
shift 3
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
  *" ps "*rds-host-stats) [ -n "$FAKE_RDS_PS_FAILS" ] && exit 1; [ -n "$FAKE_RDS_RUNNING" ] && echo 4567ef ;;
  "up -d --no-deps vmalert") [ "$FAKE_VMALERT_FAILS" = signal ] && kill -TERM $$; [ -n "$FAKE_VMALERT_FAILS" ] && exit 1 ;;
  *" up "*rds-host-stats) echo "$RDS_DB_INSTANCE_IDENTIFIER $AWS_REGION $PGAI_CLUSTER $PGAI_NODE_NAME" >> "$FAKE_DOCKER_LOG_DIR/rds-env.log" ;;
  exec*) [ -n "$FAKE_EXEC_FAILS" ] && exit 1 ;;
  kill*) [ -n "$FAKE_KILL_FAILS" ] && exit 1 ;;
  ps*) echo 0123abcd ;;
esac
exit 0
`);
  chmodSync(`${dir}/bin/docker`, 0o755);
  env = { PATH: `${dir}/bin:/usr/bin:/bin`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/xdg`, PGAI_PROJECT_DIR: projectDir, FAKE_DOCKER_LOG: log, FAKE_DOCKER_LOG_DIR: dir };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function run(args: string[], extra: Record<string, string> = {}) {
  const result = Bun.spawnSync([process.execPath, cli, "mon", "targets", ...args], { cwd: dir, env: { ...env, ...extra }, timeout: 20000 });
  return { exitCode: result.exitCode, out: result.stdout.toString() + result.stderr.toString() };
}
const scrapeFiles = () => existsSync(`${projectDir}/host-metrics`) ? readdirSync(`${projectDir}/host-metrics`).sort() : [];
const reloads = () => readFileSync(log, "utf8").split("\n").filter((line) => line.startsWith("kill "));
const profiles = () => envFile().match(/^COMPOSE_PROFILES=.*$/m)?.[0];
const vmalertCalls = () => readFileSync(log, "utf8").split("\n").filter((line) => line.endsWith(" vmalert"));
const envFile = () => existsSync(`${projectDir}/.env`) ? readFileSync(`${projectDir}/.env`, "utf8") : "";

test("Supabase: the relay scrape job carries the target's cluster and node_name", () => {
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(added.exitCode, added.out).toBe(0);
  expect(readFileSync(`${projectDir}/host-metrics/supabase-sb.yml`, "utf8")).toBe(readFileSync(`${import.meta.dir}/fixtures/supabase-scrape.golden.yml`, "utf8"));
  expect(added.out).toContain("Host metrics: Supabase, relayed by instance-jobs (scraped every 60s)");
  expect(reloads()).toEqual(["kill -s SIGHUP sink-prometheus"]);
  // vmalert maps the relayed series to host_*, so it runs from now on.
  expect(profiles()).toBe("COMPOSE_PROFILES=host-metrics");
  expect(vmalertCalls()).toEqual(["up -d --no-deps vmalert"]);
});

test("vmalert: its profile is added to and removed from COMPOSE_PROFILES, other profiles kept", () => {
  writeFileSync(`${projectDir}/.env`, "COMPOSE_PROFILES=instance-jobs\n");
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  expect(profiles()).toBe("COMPOSE_PROFILES=instance-jobs,host-metrics");
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  // Every add makes sure vmalert runs; up -d is a no-op when it does.
  expect(vmalertCalls()).toEqual(["up -d --no-deps vmalert", "up -d --no-deps vmalert"]);
  const removed = run(["remove", "sb"]);
  expect(removed.exitCode, removed.out).toBe(0);
  expect(profiles()).toBe("COMPOSE_PROFILES=instance-jobs");
  expect(vmalertCalls()).toEqual(["up -d --no-deps vmalert", "up -d --no-deps vmalert", "rm -sf vmalert"]);
});

test.each([["1"], ["signal"]])("vmalert: a failed start (%p) is retried by the next targets add", (how) => {
  const failed = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true", FAKE_VMALERT_FAILS: how });
  expect(failed.exitCode).toBe(1);
  expect(failed.out).toContain("Host metrics: 'docker compose --profile host-metrics up -d --no-deps vmalert' failed.");
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  expect(vmalertCalls()).toEqual(["up -d --no-deps vmalert", "up -d --no-deps vmalert"]);
  expect(profiles()).toBe("COMPOSE_PROFILES=host-metrics");
});

test.each([["COMPOSE_PROFILES=instance-jobs # pinned"], ['COMPOSE_PROFILES="instance-jobs"# pinned'], ["COMPOSE_PROFILES='instance-jobs'"]])("vmalert: %p keeps its profile and gains host-metrics", (line) => {
  writeFileSync(`${projectDir}/.env`, `${line}\n`);
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  expect(profiles()).toBe("COMPOSE_PROFILES=instance-jobs,host-metrics");
});

// Compose prefers an exported COMPOSE_PROFILES over .env.
test("vmalert: an exported COMPOSE_PROFILES without the profile is reported", () => {
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true", COMPOSE_PROFILES: "instance-jobs" });
  expect(added.exitCode, added.out).toBe(0);
  expect(added.out).toContain("COMPOSE_PROFILES=instance-jobs is exported and has no host-metrics, so compose will not keep vmalert running: add host-metrics to it or unset it");
});

test("vmalert: an exported empty COMPOSE_PROFILES is reported too", () => {
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true", COMPOSE_PROFILES: "" });
  expect(added.out).toContain("COMPOSE_PROFILES= is exported and has no host-metrics");
});

// An older stack cannot load the scrape file; after the upgrade the user
// runs mon stop/start, which must bring vmalert up without another add.
test("vmalert: a failed add still records the profile, and starts nothing", () => {
  const failed = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true", FAKE_EXEC_FAILS: "1" });
  expect(failed.exitCode).toBe(1);
  expect(profiles()).toBe("COMPOSE_PROFILES=host-metrics");
  expect(vmalertCalls()).toEqual([]);
});

test("vmalert: a failed removal keeps the profile, so the retry removes vmalert", () => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  const failed = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "false", FAKE_KILL_FAILS: "1" });
  expect(failed.exitCode).toBe(1);
  expect(profiles()).toBe("COMPOSE_PROFILES=host-metrics");
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "false" }).exitCode).toBe(0);
  expect(vmalertCalls().at(-1)).toBe("rm -sf vmalert");
  expect(profiles()).toBeUndefined();
});

test("vmalert: compose starts it only under the host-metrics profile", () => {
  const compose = load(readFileSync(resolve(import.meta.dir, "../../docker-compose.yml"), "utf8")) as { services: Record<string, { profiles?: string[] }> };
  expect(compose.services.vmalert.profiles).toEqual(["host-metrics"]);
});

test("Supabase: the relay serves one project, so a second target takes the job over", () => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  const second = run(["add", supabase("postgres_ai_mon.bbbbbbbbbbbbbbbbbbbb"), "sb2"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(second.exitCode, second.out).toBe(0);
  expect(scrapeFiles()).toEqual(["supabase-sb2.yml"]);
});

test("Supabase: the flag is read from .env when the environment does not set it", () => {
  writeFileSync(`${projectDir}/.env`, "PGAI_SUPABASE_HOST_METRICS=true\n");
  expect(run(["add", supabase(), "sb"]).exitCode).toBe(0);
  expect(scrapeFiles()).toEqual(["supabase-sb.yml"]);
});

// Compose drops an inline comment from an unquoted value, so the relay is on.
test.each([["PGAI_SUPABASE_HOST_METRICS=true  # relay"], ['PGAI_SUPABASE_HOST_METRICS="true" # relay']])("Supabase: %p in .env turns the job on", (line) => {
  writeFileSync(`${projectDir}/.env`, `${line}\n`);
  const added = run(["add", supabase(), "sb"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(scrapeFiles()).toEqual(["supabase-sb.yml"]);
});

const relayOff = "Host metrics: the relay runs in instance-jobs, which is not enabled. Enable it with 'postgresai mon local-install --instance-jobs'.";
test("Supabase: a relay job without the instance-jobs profile is reported", () => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).out).toContain(relayOff);
  writeFileSync(`${projectDir}/.env`, "COMPOSE_PROFILES=instance-jobs # pinned\n");
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(added.exitCode, added.out).toBe(0);
  expect(added.out).not.toContain(relayOff);
});

test.each([["false"], [""]])("Supabase: with the flag %p the job is removed and nothing scrapes the relay", (flag) => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  writeFileSync(log, "");
  const off = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: flag });
  expect(off.exitCode, off.out).toBe(0);
  expect(scrapeFiles()).toEqual([]);
  expect(reloads()).toEqual(["kill -s SIGHUP sink-prometheus"]);
  expect(profiles()).toBeUndefined();
  expect(vmalertCalls()).toEqual(["rm -sf vmalert"]);
});

test("Supabase: with host metrics off, a dotted target name is still added and pgwatch reloaded", () => {
  const added = run(["add", supabase(), "prod.main"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(readFileSync(`${projectDir}/instances.yml`, "utf8")).toContain("- name: prod.main\n");
  expect(scrapeFiles()).toEqual([]);
  expect(readFileSync(log, "utf8")).toBe("run --rm sources-generator\nup -d --force-recreate pgwatch-prometheus pgwatch-postgres\n");
});

test("Supabase: host metrics turned off in .env, a dotted name is added and the other target's job removed", () => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  writeFileSync(`${projectDir}/.env`, `${envFile()}PGAI_SUPABASE_HOST_METRICS=false\n`);
  const added = run(["add", supabase(), "prod.main"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(readFileSync(`${projectDir}/instances.yml`, "utf8")).toContain("- name: prod.main\n");
  expect(scrapeFiles()).toEqual([]);
  expect(profiles()).toBeUndefined();
});

// Refused before anything is saved: a re-run cannot change the name.
test("Supabase: with host metrics on, a dotted target name is refused and nothing is saved", () => {
  const added = run(["add", supabase(), "prod.main"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(added.exitCode).toBe(1);
  expect(added.out).toContain("a Supabase target name may use only letters, digits, '_' and '-' while PGAI_SUPABASE_HOST_METRICS is true. Choose another name.");
  expect(added.out).not.toContain("re-run this command");
  expect(run(["list"]).out).toContain("No monitoring targets configured");
  expect(scrapeFiles()).toEqual([]);
  expect(readFileSync(log, "utf8")).toBe("");
});

test("Supabase: targets remove drops the relay job", () => {
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  const removed = run(["remove", "sb"]);
  expect(removed.exitCode, removed.out).toBe(0);
  expect(scrapeFiles()).toEqual([]);
});

test("RDS: rds-host-stats gets the instance, region and the target's labels in .env", () => {
  writeFileSync(`${projectDir}/.env`, "PGAI_TAG=0.17.0\nAWS_REGION=eu-west-1\n");
  const added = run(["add", rds, "rds1"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(envFile()).toBe("PGAI_TAG=0.17.0\nAWS_REGION=us-east-1\nRDS_DB_INSTANCE_IDENTIFIER=mydb\nPGAI_CLUSTER=default\nPGAI_NODE_NAME=rds1\n");
  expect(added.out).toContain("Host metrics: rds-host-stats polls RDS instance mydb (us-east-1). Start it with: docker compose --profile rds up -d rds-host-stats");
  const removed = run(["remove", "rds1"]);
  expect(removed.exitCode, removed.out).toBe(0);
  expect(envFile()).toBe("PGAI_TAG=0.17.0\n");
});

// The host is what follows the last "@", as `targets add` reads it.
test("RDS: targets remove clears .env when the password has an '@'", () => {
  const conn = rds.replace("u:pw@", "u:p@ss@");
  const added = run(["add", conn, "rds1"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(envFile()).toContain("RDS_DB_INSTANCE_IDENTIFIER=mydb\n");
  const removed = run(["remove", "rds1"]);
  expect(removed.exitCode, removed.out).toBe(0);
  expect(envFile()).not.toMatch(/RDS_DB_INSTANCE_IDENTIFIER|AWS_REGION|PGAI_CLUSTER|PGAI_NODE_NAME/);
});

// A running rds-host-stats keeps the environment it started with.
test("RDS: a running rds-host-stats is recreated on add and remove", () => {
  const recreate = "--profile rds up -d --no-deps rds-host-stats";
  const recreates = () => readFileSync(log, "utf8").split("\n").filter((line) => line === recreate);
  expect(run(["add", rds, "rds1"], { FAKE_RDS_RUNNING: "1" }).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate]);
  expect(run(["remove", "rds1"], { FAKE_RDS_RUNNING: "1" }).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate, recreate]);
  expect(run(["add", rds, "rds1"]).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate, recreate]);
});

// An exported variable would win over .env inside compose.
test("RDS: the recreated rds-host-stats gets the new values, not exported ones", () => {
  const add = run(["add", rds, "rds1"], { FAKE_RDS_RUNNING: "1", RDS_DB_INSTANCE_IDENTIFIER: "stale", AWS_REGION: "eu-west-1" });
  expect(add.exitCode, add.out).toBe(0);
  expect(readFileSync(`${dir}/rds-env.log`, "utf8")).toBe("mydb us-east-1 default rds1\n");
  const removed = run(["remove", "rds1"], { FAKE_RDS_RUNNING: "1", RDS_DB_INSTANCE_IDENTIFIER: "stale" });
  expect(removed.exitCode, removed.out).toBe(0);
  expect(readFileSync(`${dir}/rds-env.log`, "utf8")).toBe("mydb us-east-1 default rds1\n   \n");
});

test("RDS: when compose cannot tell whether rds-host-stats runs, the user is told to recreate it", () => {
  const add = run(["add", rds, "rds1"], { FAKE_RDS_PS_FAILS: "1" });
  expect(add.exitCode, add.out).toBe(0);
  expect(add.out).toContain("If rds-host-stats is running, recreate it: docker compose --profile rds up -d rds-host-stats");
});

test.each([
  ["writer", "postgresql://u:pw@db.cluster-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
  ["reader", "postgresql://u:pw@db.cluster-ro-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
  ["proxy", "postgresql://u:pw@px.proxy-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
])("RDS: a %s endpoint names no instance, so rds-host-stats is left alone", (_kind, conn) => {
  const added = run(["add", conn, "aurora"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(envFile()).not.toContain("RDS_DB_INSTANCE_IDENTIFIER");
  expect(added.out).toContain("Host metrics: add the RDS instance endpoint");
});
