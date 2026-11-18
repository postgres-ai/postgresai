import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { load } from "js-yaml";

// `mon targets add` copies the target's cluster/node_name into what pgai-mon
// reads for host metrics, for Supabase and RDS as for ClickHouse.
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
  // Every compose call succeeds; the reporter runs, pgai-mon runs when asked to.
  writeFileSync(`${dir}/bin/docker`, `#!/bin/sh
if [ "$1" = info ]; then exit 0; fi
if [ "$1" = compose ] && [ "$2" = version ]; then exit 0; fi
shift 3
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
  "ps -q pgai-mon") [ -n "$FAKE_MON_PS_FAILS" ] && exit 1; [ -n "$FAKE_MON_RUNNING" ] && echo 4567ef ;;
  "ps -q postgres-reports") [ -z "$FAKE_REPORTER_STOPPED" ] && echo 0123abcd ;;
  "up -d --no-deps pgai-mon") echo "$RDS_DB_INSTANCE_IDENTIFIER $AWS_REGION $PGAI_CLUSTER $PGAI_NODE_NAME" >> "$FAKE_DOCKER_LOG_DIR/rds-env.log" ;;
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
const relayOff = "Host metrics: the relay runs in instance-jobs, which is not enabled. Enable it with 'postgresai mon local-install --instance-jobs'.";
const scrapeFiles = () => existsSync(`${projectDir}/host-metrics`) ? readdirSync(`${projectDir}/host-metrics`).sort() : [];
const calls = () => readFileSync(log, "utf8").split("\n").filter(Boolean);
// pgai-mon reloads instances.yml and host-metrics/ by itself: a target change
// only asks whether the reporter runs and restarts it. Nothing is signalled,
// generated or (outside RDS settings) recreated.
const changes = () => calls().filter((line) => !line.startsWith("ps "));
const profiles = () => envFile().match(/^COMPOSE_PROFILES=.*$/m)?.[0];
const envFile = () => existsSync(`${projectDir}/.env`) ? readFileSync(`${projectDir}/.env`, "utf8") : "";

test("Supabase: the relay scrape job carries the target's cluster and node_name", () => {
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(added.exitCode, added.out).toBe(0);
  expect(readFileSync(`${projectDir}/host-metrics/supabase-sb.yml`, "utf8")).toBe(readFileSync(`${import.meta.dir}/fixtures/supabase-scrape.golden.yml`, "utf8"));
  expect(added.out).toContain("Host metrics: Supabase, relayed by instance-jobs (scraped every 60s)");
  // pgai-mon picks the file up and maps the series to host_* itself: no
  // reload, no vmalert, no profile.
  expect(changes()).toEqual(["restart postgres-reports"]);
  expect(profiles()).toBeUndefined();
});

test("a stack with the old host-metrics profile in COMPOSE_PROFILES keeps it untouched", () => {
  writeFileSync(`${projectDir}/.env`, "COMPOSE_PROFILES=instance-jobs,host-metrics\n");
  expect(run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" }).exitCode).toBe(0);
  expect(run(["remove", "sb"]).exitCode).toBe(0);
  expect(profiles()).toBe("COMPOSE_PROFILES=instance-jobs,host-metrics");
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

// The flag is read as compose 5.0.2 reads it (checked with `docker compose
// config`): an inline comment needs a space before "#" (a tab is part of the
// value), quotes are dropped, a double-quoted value may escape '"' and '\\',
// text after the closing quote is dropped, an unterminated quote is not true
// (compose refuses the file).
test.each([
  ["PGAI_SUPABASE_HOST_METRICS=true  # relay", true],
  ["PGAI_SUPABASE_HOST_METRICS=true\t# relay", false],
  ['PGAI_SUPABASE_HOST_METRICS="true" # relay', true],
  ["PGAI_SUPABASE_HOST_METRICS='true'", true],
  ['PGAI_SUPABASE_HOST_METRICS="tru\\"e"', false],
  ['PGAI_SUPABASE_HOST_METRICS="true\\\\"', false],
  ['PGAI_SUPABASE_HOST_METRICS="true"x', true],
  ['PGAI_SUPABASE_HOST_METRICS="true', false],
  ["export PGAI_SUPABASE_HOST_METRICS=true", true],
  ["PGAI_SUPABASE_HOST_METRICS=true#relay", false],
  ["PGAI_SUPABASE_HOST_METRICS=", false],
  ["PGAI_SUPABASE_HOST_METRICS=false\nPGAI_SUPABASE_HOST_METRICS=true", true],
])("Supabase: %p in .env → relay job on: %p", (line, on) => {
  writeFileSync(`${projectDir}/.env`, `${line}\n`);
  const added = run(["add", supabase(), "sb"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(scrapeFiles()).toEqual(on ? ["supabase-sb.yml"] : []);
});

test.each([
  ["COMPOSE_PROFILES='instance-jobs'", true],
  ['COMPOSE_PROFILES="a,instance-jobs"', true],
  ["COMPOSE_PROFILES=a#x,instance-jobs", true],
  ["COMPOSE_PROFILES=instance-jobs # pinned", true],
  ["COMPOSE_PROFILES='other'", false],
  ["COMPOSE_PROFILES=instance-jobs\t# pinned", false],
])("Supabase: %p in .env enables the relay profile: %p", (line, enabled) => {
  writeFileSync(`${projectDir}/.env`, `${line}\n`);
  const added = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "true" });
  expect(added.exitCode, added.out).toBe(0);
  expect(scrapeFiles()).toEqual(["supabase-sb.yml"]);
  expect(added.out.includes(relayOff)).toBe(!enabled);
});

test("Supabase: an exported empty flag overrides .env, as it does for compose", () => {
  writeFileSync(`${projectDir}/.env`, "PGAI_SUPABASE_HOST_METRICS=true\n");
  expect(run(["add", supabase(), "sb"]).exitCode).toBe(0);
  expect(scrapeFiles()).toEqual(["supabase-sb.yml"]);
  const off = run(["add", supabase(), "sb"], { PGAI_SUPABASE_HOST_METRICS: "" });
  expect(off.exitCode, off.out).toBe(0);
  expect(scrapeFiles()).toEqual([]);
});

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
  expect(changes()).toEqual(["restart postgres-reports"]);
});

test("Supabase: with host metrics off, a dotted target name is still added; nothing is regenerated", () => {
  const added = run(["add", supabase(), "prod.main"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(readFileSync(`${projectDir}/instances.yml`, "utf8")).toContain("- name: prod.main\n");
  expect(scrapeFiles()).toEqual([]);
  expect(readFileSync(log, "utf8")).toBe("ps -q postgres-reports\nrestart postgres-reports\n");
  expect(added.out).toContain("pgai-mon reloads instances.yml by itself");
});

// The reporter's daily timer would otherwise give a target added later its
// first report a day after it is added. Remove does not restart it: discovery
// sees a removed node for 3 h, so a run then would report it again.
test("targets add restarts a running postgres-reports; remove does not, and a stopped one is not started", () => {
  const restarts = () => calls().filter((line) => line.startsWith("restart "));
  expect(run(["add", rds, "rds1"]).exitCode).toBe(0);
  expect(restarts()).toEqual(["restart postgres-reports"]);
  expect(run(["remove", "rds1"]).exitCode).toBe(0);
  expect(restarts()).toEqual(["restart postgres-reports"]);
  expect(run(["add", rds, "rds2"], { FAKE_REPORTER_STOPPED: "1" }).exitCode).toBe(0);
  expect(restarts()).toEqual(["restart postgres-reports"]);
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

test("RDS: pgai-mon gets the instance, region and the target's labels in .env", () => {
  writeFileSync(`${projectDir}/.env`, "PGAI_TAG=0.17.0\nAWS_REGION=eu-west-1\n");
  const added = run(["add", rds, "rds1"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(envFile()).toBe("PGAI_TAG=0.17.0\nAWS_REGION=us-east-1\nRDS_DB_INSTANCE_IDENTIFIER=mydb\nPGAI_CLUSTER=default\nPGAI_NODE_NAME=rds1\n");
  expect(added.out).toContain("Host metrics: pgai-mon polls RDS instance mydb (us-east-1)");
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

// `targets add` writes the RDS keys only for a conn_str it can parse as a URL,
// so keys next to a target it cannot parse were put there by hand: remove keeps them.
test("RDS: targets remove keeps .env for a target whose conn_str is not a URL", () => {
  writeFileSync(`${projectDir}/instances.yml`, "- name: rds1\n  conn_str: host=mydb.c9akciq32xyz.us-east-1.rds.amazonaws.com user=u password=pw dbname=postgres\n");
  const keys = "AWS_REGION=us-east-1\nRDS_DB_INSTANCE_IDENTIFIER=mydb\nPGAI_CLUSTER=default\nPGAI_NODE_NAME=rds1\n";
  writeFileSync(`${projectDir}/.env`, keys);
  const removed = run(["remove", "rds1"]);
  expect(removed.exitCode, removed.out).toBe(0);
  expect(removed.out).toContain("Monitoring target 'rds1' removed");
  expect(removed.out).not.toContain("no longer polls RDS instance");
  expect(envFile()).toBe(keys);
});

// A running pgai-mon keeps the environment it started with; one whose RDS
// settings did not change is left alone.
test("RDS: a running pgai-mon is recreated when the RDS settings change, and only then", () => {
  const recreate = "up -d --no-deps pgai-mon";
  const recreates = () => calls().filter((line) => line === recreate);
  expect(run(["add", rds, "rds1"], { FAKE_MON_RUNNING: "1" }).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate]);
  expect(run(["add", rds, "rds1"], { FAKE_MON_RUNNING: "1" }).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate]);
  expect(run(["remove", "rds1"], { FAKE_MON_RUNNING: "1" }).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate, recreate]);
  // A stopped pgai-mon reads .env when it starts.
  expect(run(["add", rds, "rds1"]).exitCode).toBe(0);
  expect(recreates()).toEqual([recreate, recreate]);
});

// An exported variable would win over .env inside compose.
test("RDS: the recreated pgai-mon gets the new values, not exported ones", () => {
  const add = run(["add", rds, "rds1"], { FAKE_MON_RUNNING: "1", RDS_DB_INSTANCE_IDENTIFIER: "stale", AWS_REGION: "eu-west-1" });
  expect(add.exitCode, add.out).toBe(0);
  expect(readFileSync(`${dir}/rds-env.log`, "utf8")).toBe("mydb us-east-1 default rds1\n");
  const removed = run(["remove", "rds1"], { FAKE_MON_RUNNING: "1", RDS_DB_INSTANCE_IDENTIFIER: "stale" });
  expect(removed.exitCode, removed.out).toBe(0);
  expect(readFileSync(`${dir}/rds-env.log`, "utf8")).toBe("mydb us-east-1 default rds1\n   \n");
});

test("RDS: when compose cannot tell whether pgai-mon runs, the user is told to recreate it", () => {
  const add = run(["add", rds, "rds1"], { FAKE_MON_PS_FAILS: "1" });
  expect(add.exitCode, add.out).toBe(0);
  expect(add.out).toContain("If pgai-mon is running, recreate it so it reads the new RDS settings: docker compose up -d --no-deps pgai-mon");
});

test.each([
  ["writer", "postgresql://u:pw@db.cluster-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
  ["reader", "postgresql://u:pw@db.cluster-ro-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
  ["proxy", "postgresql://u:pw@px.proxy-c9akciq32xyz.us-east-1.rds.amazonaws.com:5432/postgres"],
])("RDS: a %s endpoint names no instance, so no RDS settings are written", (_kind, conn) => {
  const added = run(["add", conn, "aurora"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(envFile()).not.toContain("RDS_DB_INSTANCE_IDENTIFIER");
  expect(added.out).toContain("Host metrics: add the RDS instance endpoint");
});


test("target commands create the host-metrics bind source before starting services", () => {
  expect(existsSync(`${projectDir}/host-metrics`)).toBe(false);
  const added = run(["add", rds, "rds1"]);
  expect(added.exitCode, added.out).toBe(0);
  expect(statSync(`${projectDir}/host-metrics`).isDirectory()).toBe(true);
  expect(statSync(`${projectDir}/host-metrics`).mode & 0o777).toBe(0o700);
});
