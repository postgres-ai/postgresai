import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import { resolve } from "path";
import { composeDefinesService, composeProjectName, failedServices, grafanaBindNotice, oldStackVolumeNames, parseComposePs, redactStoreUrl, restartLoops, stackProblems, stackSwitchNotice, storeNotice } from "../lib/stack";

/**
 * The switch of an existing install from the old pgwatch + VictoriaMetrics
 * stack to pgai-mon, and the install/update path on the new compose file.
 *
 * Docker is a shim on PATH that records its argv (with COMPOSE_PROFILES and
 * GF_SECURITY_ADMIN_PASSWORD), keeps a copy of the compose file a `down` was
 * given, and answers `ps` / `volume ls` from environment variables. As compose
 * does, it fails every call on a file that requires the Grafana password
 * (`:?`) when neither the environment nor .env has it. The compose files are the fixtures under
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

function makeProject(name: string, compose: string, env = OLD_ENV, pgwatchConfig = "grafana_password=pw\n") {
  const dir = resolve(tempDir, name);
  const bin = resolve(tempDir, `${name}-bin`);
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(bin, { recursive: true });
  fs.copyFileSync(compose, resolve(dir, "docker-compose.yml"));
  fs.writeFileSync(resolve(dir, ".env"), env, { mode: 0o600 });
  fs.writeFileSync(resolve(dir, "instances.yml"), "- name: keep-me\n  conn_str: postgresql://m:p@h:5432/db\n");
  fs.writeFileSync(resolve(dir, ".pgwatch-config"), pgwatchConfig, { mode: 0o600 });
  // `mon generate-grafana-password` must not need it.
  fs.writeFileSync(resolve(bin, "openssl"), "#!/bin/sh\nexit 127\n", { mode: 0o755 });
  const log = resolve(bin, "docker-calls.log");
  fs.writeFileSync(
    resolve(bin, "docker"),
    [
      "#!/bin/sh",
      `printf '%s |PROFILES=%s |GF=%s\\n' "$*" "\${COMPOSE_PROFILES-}" "\${GF_SECURITY_ADMIN_PASSWORD-}" >> ${JSON.stringify(log)}`,
      'prev=""; file=""',
      'for a in "$@"; do [ "$prev" = "-f" ] && file="$a"; prev="$a"; done',
      'if [ -n "$file" ] && grep -q "GF_SECURITY_ADMIN_PASSWORD:?" "$file" && [ -z "${GF_SECURITY_ADMIN_PASSWORD-}" ] \\',
      '   && ! grep -q "^GF_SECURITY_ADMIN_PASSWORD=." .env 2>/dev/null; then',
      '  echo "required variable GF_SECURITY_ADMIN_PASSWORD is missing a value" >&2; exit 15',
      "fi",
      `case " $* " in *" down "*) cp "$file" ${JSON.stringify(resolve(bin, "down-compose.yml"))} ;; esac`,
      'case " $* " in',
      '  " info ") exit 0 ;;',
      '  " inspect -f {{.State.Running}} "*) echo "${GRAFANA_RUNNING:-false}"; exit 0 ;;',
      // The restart count of container abc123: 0 at the first call, 3 more at each later one.
      `  " inspect -f {{.Id}} {{.RestartCount}} "*) f=${JSON.stringify(resolve(bin, "restarts"))}; n=$(cat "$f" 2>/dev/null || echo 0); echo $((n + 3)) > "$f"; echo "abc123def $n"; exit 0 ;;`,
      `  *" reset-admin-password "*) cat > ${JSON.stringify(resolve(bin, "reset-stdin"))}; exit \${RESET_EXIT:-0} ;;`,
      '  *" compose version "*) exit 0 ;;',
      '  " ps -aq --filter label=com.docker.compose.project="*) echo own-container; exit 0 ;;',
      '  " network ls -q --filter label=com.docker.compose.project="*) echo own-network; exit 0 ;;',
      '  *" volume ls "*) for v in $VOLUMES; do echo "$v"; done; exit 0 ;;',
      '  *" ps -q sink-prometheus "*) [ -n "$OLD_RUNNING" ] && echo old123; exit 0 ;;',
      '  *" ps -q pgai-mon "*) [ -n "$NEW_RUNNING" ] && echo new123; exit 0 ;;',
      // As compose reads it: a PGAI_TAG in the environment wins over .env.
      '  *" pull "*) tag="${PGAI_TAG-$(sed -n "s/^PGAI_TAG=//p" .env)}"; [ -z "$PULL_NEEDS_TAG" ] || [ "$tag" = "$PULL_NEEDS_TAG" ] || exit 1; exit ${PULL_EXIT:-0} ;;',
      '  *" down "*) exit ${DOWN_EXIT:-0} ;;',
      // UP_HANGS: compose waiting for a container that never turns healthy; it exits 130 on SIGTERM.
      '  *" up "*) [ -z "$UP_HANGS" ] || { trap "exit 130" TERM; while :; do sleep 0.2; done; }; exit ${UP_EXIT:-0} ;;',
      '  *" config --services "*) for s in $CONFIG_SERVICES; do echo "$s"; done; exit 0 ;;',
      '  *" ps -a --format json "*) printf "%s\\n" "$PS_JSON"; exit 0 ;;',
      // This project's containers ("<service>\t<id>"); GRAFANA_RUNNING adds a running Grafana.
      '  " ps --filter"*) for c in $RUNNING; do echo "$c"; done; [ "$GRAFANA_RUNNING" = true ] && printf "grafana\\tgraf123\\n"; exit 0 ;;',
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
    run(args: string[], env: Record<string, string> = {}, stdin = "") {
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
        stdin: Buffer.from(stdin),
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

const CLI_VERSION: string = JSON.parse(fs.readFileSync(resolve(import.meta.dir, "..", "package.json"), "utf8")).version;
const NEW_ENV = `PGAI_TAG=0.18.0\nPGAI_MON_TOKEN=${"a".repeat(64)}\nPGAI_MON_READ_TOKEN=${"b".repeat(64)}\n`;
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
    // The way back: stop the new stack, put the old file and tag back, start.
    const backup = fs.readdirSync(p.dir).find((f) => f.startsWith("docker-compose.yml.bak-0.17.0-"))!;
    expect(r.stdout).toContain(
      "To go back to the old stack (while its volumes exist):\n  postgresai mon stop\n" +
        "  docker volume rm old-running_postgres_ai_configs   # the old stack fills it again\n" +
        `  cp ${resolve(p.dir, backup)} ${resolve(p.dir, "docker-compose.yml")}\n` +
        `  set PGAI_TAG=0.17.0 in ${resolve(p.dir, ".env")}\n  cd ${p.dir} && docker compose up -d\n`,
    );
    // `mon update` leaves the old stack's keys in .env, so it keeps no copy of the file.
    expect(readEnv(p.dir)).toContain("VM_AUTH_PASSWORD=old-vm-password");
    expect(fs.readdirSync(p.dir).some((f) => f.startsWith(".env.bak-"))).toBe(false);
    expect(r.stdout).toContain(storeNotice(undefined)!);
    // GRAFANA_BIND_HOST was unset: the old stack published Grafana on every
    // interface, this one on 127.0.0.1 only. Said once; .env is not changed.
    expect(r.stdout.split(grafanaBindNotice(undefined)!).length).toBe(2);
    expect(env).not.toContain("GRAFANA_BIND_HOST");
  });

  test("a GRAFANA_BIND_HOST the old install set, or Grafana left out, gets no bind notice", () => {
    const set = makeProject("old-bind-set", OLD_COMPOSE, `${OLD_ENV}GRAFANA_BIND_HOST=10.0.0.5:\n`);
    const r1 = set.run(["mon", "update"], { OLD_RUNNING: "1" });
    expect(r1.status).toBe(0);
    expect(r1.stdout).toContain("Switched this install");
    expect(r1.stdout).not.toContain("Grafana now listens");
    const empty = makeProject("old-bind-empty", OLD_COMPOSE, `${OLD_ENV}GRAFANA_BIND_HOST=\n`);
    expect(empty.run(["mon", "update"], { OLD_RUNNING: "1" }).stdout).not.toContain("Grafana now listens");
    const off = makeProject("old-no-grafana", OLD_COMPOSE, `${OLD_ENV}GRAFANA_SCALE=0\n`);
    const r3 = off.run(["mon", "update"], { OLD_RUNNING: "1" });
    expect(r3.stdout).toContain("Switched this install");
    expect(r3.stdout).not.toContain("Grafana now listens");
  }, 20000); // three CLI runs: 1 s on an idle machine, over the default 5 s on a loaded one

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
    expect(r.stderr).toContain("Re-run 'postgresai mon update' to switch");
    // The pull ran at the CLI's version, which the message names; the old tag is back in .env.
    expect(r.stderr).toContain(`PGAI_TAG=${CLI_VERSION}`);
    expect(readEnv(p.dir)).toMatch(/^PGAI_TAG=0\.17\.0$/m);
  });

  test("the switch moves PGAI_TAG to the CLI's version before the pull: the old tag names no pgai-mon image", () => {
    const p = makeProject("old-tag", OLD_COMPOSE);
    // The shim's pull fails unless .env already has the new tag when it runs.
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", PULL_NEEDS_TAG: CLI_VERSION });
    expect(r.status).toBe(0);
    expect(readEnv(p.dir)).toMatch(new RegExp(`^PGAI_TAG=${CLI_VERSION.replace(/\./g, "\\.")}$`, "m"));
    expect(readEnv(p.dir)).not.toContain("PGAI_TAG=0.17.0");
    expect(r.stdout).toContain(`PGAI_TAG in .env: 0.17.0 -> ${CLI_VERSION}`);
  });

  test("a PGAI_TAG the user exported, different from the one in .env, still wins in compose", () => {
    const p = makeProject("old-tag-exported", OLD_COMPOSE);
    // The shim's pull fails unless compose would use the exported tag.
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", PGAI_TAG: "9.9.9", PULL_NEEDS_TAG: "9.9.9" });
    expect(r.status).toBe(0);
  });

  test("the old stack's containers cannot be removed: the message names the backup file to run `down` on", () => {
    const p = makeProject("old-down-fails", OLD_COMPOSE);
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", DOWN_EXIT: "1" });
    expect(r.status).toBe(1);
    const backup = fs.readdirSync(p.dir).find((f) => f.startsWith("docker-compose.yml.bak-0.17.0-"))!;
    expect(r.stderr).toContain(`docker compose -f ${resolve(p.dir, backup)} down --remove-orphans`);
    expect(r.stderr).toContain("COMPOSE_PROFILES=demo,host-metrics,instance-jobs,rds VM_AUTH_USERNAME=unused VM_AUTH_PASSWORD=unused");
    expect(r.stderr).not.toContain("<the old compose file>");
    expect(composeCalls(p.calls()).some((c) => / up /.test(c))).toBe(false);
  });

  test("the new stack does not start after the old one is gone: the rollback steps are printed", () => {
    const p = makeProject("old-up-fails", OLD_COMPOSE);
    const r = p.run(["mon", "update"], { OLD_RUNNING: "1", UP_EXIT: "1" });
    expect(r.status).toBe(1);
    const backup = fs.readdirSync(p.dir).find((f) => f.startsWith("docker-compose.yml.bak-0.17.0-"))!;
    expect(r.stderr).toContain("    postgresai mon stop\n    docker volume rm old-up-fails_postgres_ai_configs");
    expect(r.stderr).toContain(`cp ${resolve(p.dir, backup)} ${resolve(p.dir, "docker-compose.yml")}`);
    expect(r.stderr).toContain(`set PGAI_TAG=0.17.0 in ${resolve(p.dir, ".env")}`);
    expect(r.stderr).toContain(`cd ${p.dir} && docker compose up -d`);
    expect(r.stderr).toContain("volumes are untouched");
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
    expect(calls.some((c) => / down /.test(c))).toBe(false);
    expect(readEnv(p.dir)).toContain("PGAI_MON_TOKEN=" + "a".repeat(64));
    expect(r.stdout).not.toContain("Switched this install");
  });

  test("--grafana on a stopped stack does not start Grafana alone; on a running one it applies", () => {
    const stopped = makeProject("grafana-stopped", NEW_COMPOSE, NEW_ENV);
    const r1 = stopped.run(["mon", "update", "--grafana"]);
    expect(r1.status).toBe(0);
    expect(composeCalls(stopped.calls()).some((c) => / up /.test(c))).toBe(false);
    expect(r1.stdout).toContain("the stack is not running, so Grafana was left as it is");
    // On a running stack the `up -d` that applies the update scales Grafana too.
    const running = makeProject("grafana-running", NEW_COMPOSE, NEW_ENV);
    expect(running.run(["mon", "update", "--grafana"], { NEW_RUNNING: "1" }).status).toBe(0);
    expect(composeCalls(running.calls()).some((c) => / up -d \|/.test(c))).toBe(true);
  });
});

describe("mon update moves the stack to this CLI's version and applies it (#536)", () => {
  const tagLine = (tag: string) => new RegExp(`^PGAI_TAG=${tag.replace(/\./g, "\\.")}$`, "m");
  // A pgai-mon compose file of an earlier version: the refresh replaces it.
  const staleCompose = (name: string) => {
    const f = resolve(tempDir, `${name}-stale.yml`);
    fs.writeFileSync(f, `# an earlier version\n${fs.readFileSync(NEW_COMPOSE, "utf8")}`);
    return f;
  };

  test("PGAI_TAG moves to the CLI's version before the pull, and the output says so", () => {
    const p = makeProject("tag-advances", NEW_COMPOSE, NEW_ENV);
    // The shim's pull fails unless .env has the CLI's version when it runs.
    const r = p.run(["mon", "update"], { PULL_NEEDS_TAG: CLI_VERSION });
    expect(r.status).toBe(0);
    expect(readEnv(p.dir)).toMatch(tagLine(CLI_VERSION));
    expect(r.stdout).toContain(`PGAI_TAG in .env: 0.18.0 -> ${CLI_VERSION}`);
  });

  test("--tag runs the tag given instead, so a pinned tag can be kept; --help shows it", () => {
    const keep = makeProject("tag-kept", NEW_COMPOSE, NEW_ENV);
    const r = keep.run(["mon", "update", "--tag", "0.18.0"], { PULL_NEEDS_TAG: "0.18.0" });
    expect(r.status).toBe(0);
    expect(readEnv(keep.dir)).toMatch(tagLine("0.18.0"));
    expect(r.stdout).not.toContain("PGAI_TAG in .env");
    const other = makeProject("tag-other", NEW_COMPOSE, NEW_ENV);
    expect(other.run(["mon", "update", "--tag", "9.9.9"], { PULL_NEEDS_TAG: "9.9.9" }).status).toBe(0);
    expect(readEnv(other.dir)).toMatch(tagLine("9.9.9"));
    const help = keep.run(["mon", "update", "--help"]);
    expect(help.stdout).toContain("--tag <tag>");
    expect(help.stdout).toContain(CLI_VERSION);
  });

  test("a --tag that is not a Docker tag is refused before anything changes", () => {
    const p = makeProject("tag-bad", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "update", "--tag", "1.0\nEVIL=1"]);
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("--tag");
    expect(readEnv(p.dir)).toBe(NEW_ENV);
    expect(composeCalls(p.calls()).some((c) => / pull /.test(c))).toBe(false);
  });

  test("a git checkout, or PGAI_SKIP_COMPOSE_REFRESH=1, keeps its tag unless --tag is given: its compose file is not the CLI's", () => {
    const git = makeProject("tag-git", NEW_COMPOSE, NEW_ENV);
    fs.mkdirSync(resolve(git.dir, ".git"));
    fs.writeFileSync(resolve(tempDir, "tag-git-bin", "git"), '#!/bin/sh\n[ "$1" = rev-parse ] && echo main\nexit 0\n', { mode: 0o755 });
    const r = git.run(["mon", "update"], { PULL_NEEDS_TAG: "0.18.0" });
    expect(r.status).toBe(0);
    expect(readEnv(git.dir)).toMatch(tagLine("0.18.0"));
    expect(r.stdout).toContain("PGAI_TAG stays 0.18.0: docker-compose.yml here is not refreshed by this CLI (a git checkout)");
    expect(git.run(["mon", "update", "--tag", "9.9.9"], { PULL_NEEDS_TAG: "9.9.9" }).status).toBe(0);
    expect(readEnv(git.dir)).toMatch(tagLine("9.9.9"));

    const baked = makeProject("tag-baked", NEW_COMPOSE, NEW_ENV);
    const b = baked.run(["mon", "update"], { PGAI_SKIP_COMPOSE_REFRESH: "1", PULL_NEEDS_TAG: "0.18.0" });
    expect(b.status).toBe(0);
    expect(readEnv(baked.dir)).toMatch(tagLine("0.18.0"));
    expect(b.stdout).toContain("(PGAI_SKIP_COMPOSE_REFRESH=1)");
  });

  test("a failed pull puts the old compose file and PGAI_TAG back and changes no container", () => {
    const stale = staleCompose("pull-fails");
    const p = makeProject("pull-fails", stale, NEW_ENV);
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1", PULL_EXIT: "1" });
    expect(r.status).toBe(1);
    expect(fs.readFileSync(resolve(p.dir, "docker-compose.yml"), "utf8")).toBe(fs.readFileSync(stale, "utf8"));
    expect(readEnv(p.dir)).toMatch(tagLine("0.18.0"));
    expect(composeCalls(p.calls()).some((c) => / (down|up) /.test(c))).toBe(false);
    expect(r.stderr).toContain(`The pull used PGAI_TAG=${CLI_VERSION}`);
    expect(r.stderr).toContain("docker-compose.yml and PGAI_TAG=0.18.0 were put back");
    expect(r.stdout).not.toContain("completed successfully");
  });

  test("the compose file for the new version cannot be fetched: the tag stays, nothing is pulled, the command fails", () => {
    const p = makeProject("no-compose", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1", PGAI_COMPOSE_SOURCE: resolve(tempDir, "does-not-exist.yml") });
    expect(r.status).toBe(1);
    expect(readEnv(p.dir)).toMatch(tagLine("0.18.0"));
    expect(composeCalls(p.calls()).some((c) => / (pull|up) /.test(c))).toBe(false);
    expect(r.stderr).toContain("PGAI_TAG stays 0.18.0");
  });

  test("a running stack is brought up on the new images with its profiles, after the pull", () => {
    const p = makeProject("applies-running", staleCompose("applies-running"), `${NEW_ENV}COMPOSE_PROFILES=instance-jobs\n`);
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1" });
    expect(r.status).toBe(0);
    const calls = composeCalls(p.calls());
    const pull = calls.findIndex((c) => / pull /.test(c));
    const up = calls.findIndex((c) => / up -d \|/.test(c));
    expect(pull).toBeGreaterThanOrEqual(0);
    expect(up).toBeGreaterThan(pull);
    // Compose reads the profiles from .env itself; the shim sees them here
    // because bun loads .env into the CLI's environment. What this catches is
    // an `up` given other profiles, or none of the Grafana password.
    expect(calls[up]).toContain("|PROFILES=instance-jobs |GF=pw");
    // One `up` for the whole stack, nothing stopped or removed.
    expect(calls.filter((c) => / up /.test(c)).length).toBe(1);
    expect(calls.some((c) => / (down|rm|restart) /.test(c))).toBe(false);
    expect(r.stdout).toContain(`The stack runs PGAI_TAG=${CLI_VERSION}`);
    expect(r.stdout).not.toContain("mon start");
  });

  test("a failed `up` fails the command and does not claim success", () => {
    const p = makeProject("applies-up-fails", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1", UP_EXIT: "1" });
    expect(r.status).toBe(1);
    expect(r.stdout).not.toContain("The stack runs");
    expect(r.stderr).toContain("did not start on the new images");
  });

  test("a stopped stack is left stopped, and the output says 'mon start' runs the new images", () => {
    const p = makeProject("applies-stopped", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "update"]);
    expect(r.status).toBe(0);
    expect(composeCalls(p.calls()).some((c) => / up /.test(c))).toBe(false);
    expect(r.stdout).toContain("The stack is not running: 'postgresai mon start' starts it on the new images");
    expect(r.stdout).not.toContain("The stack runs");
  });
});

describe("the Grafana password reaches every compose call", () => {
  test("calls that only ask (`ps`) get the password from .pgwatch-config, so their answers are real", () => {
    // instance-jobs is started only when `ps -q pgai-mon` says the stack runs.
    const p = makeProject("gf-direct", NEW_COMPOSE, `${NEW_ENV}COMPOSE_PROFILES=instance-jobs\n`);
    const r = p.run(["mon", "update"], { NEW_RUNNING: "1" });
    expect(r.status).toBe(0);
    const calls = composeCalls(p.calls());
    expect(calls.length).toBeGreaterThan(0);
    for (const c of calls.filter((c) => !/^compose version/.test(c))) expect(c).toEndWith("|GF=pw");
    expect(calls.some((c) => / ps -q pgai-mon /.test(c))).toBe(true);
    // The update's `up -d` covers instance-jobs through the profile.
    expect(calls.some((c) => / up -d \|PROFILES=instance-jobs /.test(c))).toBe(true);
    // Not on argv, and not written into .env.
    expect(p.calls().some((c) => c.split(" |")[0].includes("pw"))).toBe(false);
    expect(readEnv(p.dir)).not.toContain("GF_SECURITY_ADMIN_PASSWORD");
  });

  test("mon start sees a crashed pgai-mon instead of calling the stack already running", () => {
    const p = makeProject("gf-start-crashed", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "start"], {
      RUNNING: "grafana-with-datasources",
      CONFIG_SERVICES: "pgai-mon grafana",
      PS_JSON: JSON.stringify([{ Service: "pgai-mon", State: "restarting" }, { Service: "grafana", State: "running" }]),
    });
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("Not all services are running and healthy (pgai-mon: restarting)");
    expect(r.stdout).not.toContain("already running");
    expect(composeCalls(p.calls()).some((c) => / up -d /.test(c))).toBe(true);
  });

  for (const command of ["update", "update-config", "start"]) {
    test(`mon ${command} on an install without a saved password generates one before the first compose call`, () => {
      const p = makeProject(`gf-missing-${command}`, NEW_COMPOSE, "PGAI_TAG=0.18.0\n", "");
      const r = p.run(["mon", command]);
      expect(r.stderr).not.toContain("GF_SECURITY_ADMIN_PASSWORD");
      expect(r.status).toBe(0);
      const saved = fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8").match(/^grafana_password=(.+)$/m);
      expect(saved).not.toBeNull();
      expect(Buffer.from(saved![1], "base64").length).toBe(12);
      expect(fs.statSync(resolve(p.dir, ".pgwatch-config")).mode & 0o777).toBe(0o600);
      const calls = composeCalls(p.calls()).filter((c) => !/^compose version/.test(c));
      for (const c of calls) expect(c).toEndWith(`|GF=${saved![1]}`);
      // The tokens the compose file requires are there too.
      expect(readEnv(p.dir)).toMatch(/^PGAI_MON_TOKEN=[0-9a-f]{64}$/m);
      expect(readEnv(p.dir)).toMatch(/^PGAI_MON_READ_TOKEN=[0-9a-f]{64}$/m);
    });
  }

  test("a password already saved, or set in .env, is kept", () => {
    const saved = makeProject("gf-kept", NEW_COMPOSE, NEW_ENV);
    expect(saved.run(["mon", "start"]).status).toBe(0);
    expect(fs.readFileSync(resolve(saved.dir, ".pgwatch-config"), "utf8")).toBe("grafana_password=pw\n");
    const inEnv = makeProject("gf-in-env", NEW_COMPOSE, `${NEW_ENV}GF_SECURITY_ADMIN_PASSWORD=from-env-file\n`, "");
    expect(inEnv.run(["mon", "start"]).status).toBe(0);
    expect(fs.readFileSync(resolve(inEnv.dir, ".pgwatch-config"), "utf8")).toBe("");
  });

  test("mon stop works without a saved password (`down` uses no value) and saves none", () => {
    const p = makeProject("gf-stop", NEW_COMPOSE, NEW_ENV, "");
    const r = p.run(["mon", "stop"]);
    expect(r.status).toBe(0);
    const downs = composeCalls(p.calls()).filter((c) => / down /.test(c));
    expect(downs.length).toBe(1);
    expect(downs[0]).toEndWith("|GF=unused");
    expect(fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8")).toBe("");
  });

  test("`up` never gets the placeholder: without a password it fails as compose reports it", () => {
    const p = makeProject("gf-reset-up", NEW_COMPOSE, NEW_ENV, "");
    const r = p.run(["mon", "restart", "grafana"]);
    expect(r.status).not.toBe(0);
    expect(p.calls().some((c) => c.includes("GF=unused") && / up /.test(c))).toBe(false);
  });
});

describe("mon restart, reset, generate-grafana-password", () => {
  test("mon restart <service> re-creates that service alone; without a service it restarts the stack", () => {
    const p = makeProject("restart", NEW_COMPOSE, NEW_ENV);
    expect(p.run(["mon", "restart", "grafana"]).status).toBe(0);
    expect(p.run(["mon", "restart"]).status).toBe(0);
    const calls = composeCalls(p.calls()).filter((c) => !/^compose version/.test(c));
    expect(calls[0]).toContain(" up -d --force-recreate --no-deps grafana |");
    expect(calls[1]).toContain(" restart |");
    // What the bind notice tells the operator to run is this command.
    expect(grafanaBindNotice(undefined)).toContain("'postgresai mon restart grafana'");
  });

  test("mon reset <service> says what it does: the container is replaced, named volumes are kept", () => {
    const p = makeProject("reset", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "reset", "pgai-mon"], {}, "y\n");
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("remove its container and start a new one");
    expect(r.stdout).toContain("Data in named volumes (pgai_mon_data, grafana_data, ...) is kept");
    expect(r.stdout).not.toMatch(/remove its volume|data for this service will be lost/i);
    // Nothing here removes a named volume.
    expect(p.calls().some((c) => /volume rm| down /.test(c))).toBe(false);
    expect(composeCalls(p.calls()).some((c) => / rm -f -v pgai-mon /.test(c))).toBe(true);
  });

  test("mon generate-grafana-password needs no openssl", () => {
    const p = makeProject("gen-password", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "generate-grafana-password"]);
    expect(r.status).toBe(0);
    const saved = fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8").match(/^grafana_password=(.+)$/m)![1];
    expect(saved).not.toBe("pw");
    expect(Buffer.from(saved, "base64").length).toBe(12);
    expect(r.stdout).toContain(`Password: ${saved}`);
    // Grafana is down: nothing to tell, and the output says so.
    expect(p.calls().some((c) => c.includes("reset-admin-password"))).toBe(false);
    expect(r.stdout).toContain("Grafana is not running");
  });

  test("mon generate-grafana-password sets the password in a running Grafana, on stdin", () => {
    const p = makeProject("gen-password-running", NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "generate-grafana-password"], { GRAFANA_RUNNING: "true" });
    expect(r.status).toBe(0);
    const saved = fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8").match(/^grafana_password=(.+)$/m)![1];
    expect(fs.readFileSync(`${p.dir}-bin/reset-stdin`, "utf8").trim()).toBe(saved);
    expect(p.calls().some((c) => c.includes(saved))).toBe(false);
    expect(r.stdout).toContain("now uses this password");
  });

  test("mon generate-grafana-password keeps the saved password when Grafana refuses the new one", () => {
    const p = makeProject("gen-password-refused", NEW_COMPOSE, NEW_ENV);
    const before = fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8");
    const r = p.run(["mon", "generate-grafana-password"], { GRAFANA_RUNNING: "true", RESET_EXIT: "1" });
    expect(r.status).toBe(1);
    expect(fs.readFileSync(resolve(p.dir, ".pgwatch-config"), "utf8")).toBe(before);
    expect(r.stderr).toContain("saved password is unchanged");
  });

});

describe("a second install on the same host (#510)", () => {
  // The old stack, or another install, runs beside this one: every container
  // is found through this install's compose project, every port is its own.
  const SIDE_ENV = `${NEW_ENV}COMPOSE_PROJECT_NAME=pgai-v4\nGRAFANA_PORT=3001\nPGAI_MON_PORT=18765\n`;

  test("prints the Grafana URL with GRAFANA_PORT", () => {
    const p = makeProject("side-credentials", NEW_COMPOSE, SIDE_ENV);
    const r = p.run(["mon", "show-grafana-credentials"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("URL:      http://localhost:3001");
  });

  test("mon health and mon stop look only at this project's containers, never at a container name", () => {
    const p = makeProject("side-health", NEW_COMPOSE, SIDE_ENV);
    p.run(["mon", "health"]);
    p.run(["mon", "stop"], { DOWN_EXIT: "1" });
    const docker = p.calls().filter((c) => !c.startsWith("compose "));
    expect(docker.some((c) => c.startsWith("ps --filter label=com.docker.compose.project=pgai-v4 -a "))).toBe(true);
    expect(docker.some((c) => c.startsWith("ps -aq --filter label=com.docker.compose.project=pgai-v4 "))).toBe(true);
    expect(docker.some((c) => c.startsWith("network rm own-network "))).toBe(true);
    // The old stack's containers have these fixed names: never touched from here.
    for (const name of ["grafana-with-datasources", "postgres-reports", "self-cadvisor", "instance-jobs"]) {
      expect(docker.filter((c) => c.split(" |")[0].split(" ").includes(name))).toEqual([]);
    }
    expect(docker.some((c) => /name=/.test(c.split(" |")[0]))).toBe(false);
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
    // The way back restores the old .env from a copy: this install dropped keys the old compose file requires.
    const envBackup = resolve(p.dir, ".env.bak-0.17.0");
    expect(fs.readFileSync(envBackup, "utf8")).toBe(OLD_ENV);
    expect(fs.statSync(envBackup).mode & 0o777).toBe(0o600);
    expect(r.stdout).toContain(
      "To go back to the old stack (while its volumes exist):\n  postgresai mon stop\n" +
        "  docker volume rm li-old-stopped_postgres_ai_configs   # the old stack fills it again\n  cp ",
    );
    expect(r.stdout).toContain(`  cp ${envBackup} ${resolve(p.dir, ".env")}\n  cd ${p.dir} && docker compose up -d\n`);
    expect(r.stdout).toContain(grafanaBindNotice(undefined)!);
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

  test("a service that keeps restarting ends the install with exit code 1 and its name", () => {
    const p = makeProject("li-restart-loop", NEW_COMPOSE, "PGAI_TAG=0.18.0\n");
    const r = p.run(["mon", "local-install", "-y"], {
      UP_HANGS: "1",
      PS_JSON: JSON.stringify([{ ID: "abc123", Service: "pgai-mon", State: "restarting" }]),
    });
    // Not 130, which compose returns when the CLI stops it: nobody cancelled this run.
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/✗ Service pgai-mon failed \(keeps restarting \(3 restarts in \d+ s\)\)/);
  }, 30000);

  test("on the new stack: no switch, no notice about old volumes; a set store is named", () => {
    const p = makeProject("li-new", NEW_COMPOSE, "PGAI_TAG=0.18.0\nPGAI_MON_STORE=s3://bucket/prefix\n");
    const r = p.run(["mon", "local-install", "-y"]);
    expect(r.status).toBe(0);
    expect(composeCalls(p.calls()).some((c) => / down /.test(c))).toBe(false);
    expect(r.stdout).not.toContain("Switched this install");
    expect(r.stdout).not.toContain(storeNotice(undefined)!);
    expect(r.stdout).not.toContain("Grafana now listens");
    expect(r.stdout).toContain("Metrics history goes to s3://bucket/prefix");
    expect(readEnv(p.dir)).toContain("PGAI_MON_STORE=s3://bucket/prefix");
    // A first query to run, among the next steps.
    expect(r.stdout).toContain('postgresai mon sql --local "select metric, count(*) from series group by 1 order by 2 desc limit 10"');
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
    expect(composeProjectName("monitoring", "postgres_ai")).toBe("postgres_ai");
    expect(() => composeProjectName("monitoring", "Postgres_AI")).toThrow("Invalid COMPOSE_PROJECT_NAME");
    expect(oldStackVolumeNames("monitoring")).toEqual(["monitoring_victoria_metrics_data", "monitoring_sink_postgres_data"]);
  });

  test("the Grafana bind notice: only when GRAFANA_BIND_HOST is unset, one line, no reopening", () => {
    const n = grafanaBindNotice(undefined)!;
    expect(n).toContain("127.0.0.1:3000 only");
    expect(n).toContain("SSH tunnel");
    expect(n).toContain("GRAFANA_BIND_HOST");
    expect(n).not.toContain("\n");
    expect(grafanaBindNotice(null)).toBe(n);
    expect(grafanaBindNotice("")).toBeNull();
    expect(grafanaBindNotice("127.0.0.1:")).toBeNull();
    expect(grafanaBindNotice("0.0.0.0:")).toBeNull();
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

  test("restartLoops counts restarts since the baseline, per service, from the limit up", () => {
    const now = [
      { id: "a", service: "pgai-mon", restarts: 3 },
      { id: "b", service: "grafana", restarts: 7 },
      { id: "c", service: "postgres-reports", restarts: 2 },
    ];
    // New containers count from 0; an old install's past restarts do not count.
    expect(restartLoops({}, now)).toEqual([{ service: "pgai-mon", restarts: 3 }, { service: "grafana", restarts: 7 }]);
    expect(restartLoops({ a: 1, b: 5 }, now)).toEqual([]);
    expect(restartLoops({ a: 0, b: 4 }, now)).toEqual([{ service: "pgai-mon", restarts: 3 }, { service: "grafana", restarts: 3 }]);
    // A recreated container (lower count under a new id) is not blamed for the old one.
    expect(restartLoops({ a: 9 }, [{ id: "a2", service: "pgai-mon", restarts: 0 }])).toEqual([]);
    // Scaled replicas: one entry per service, with the worst count.
    expect(restartLoops({}, [
      { id: "x", service: "w", restarts: 4 }, { id: "y", service: "w", restarts: 6 },
    ])).toEqual([{ service: "w", restarts: 6 }]);
  });
});

// A stopped neighboring install still owns its historical volumes and networks.
describe("project-scoped cleanup", () => {
  const volumes = "monitoring_grafana_data postgres_ai_configs_pgai_mon_data other_grafana_data";

  for (const project of ["pgai-v4", "monitoring", "postgres_ai_configs"]) {
    test(`clean in ${project} respects historical volume ownership`, () => {
      const p = makeProject(`clean-${project}`, NEW_COMPOSE, `${NEW_ENV}COMPOSE_PROJECT_NAME=${project}\n`);
      const r = p.run(["mon", "clean"], { VOLUMES: volumes });
      expect(r.status).toBe(0);
      const calls = p.calls().map((c) => c.split(" |PROFILES=")[0]);
      expect(calls.some((c) => c.endsWith(" down -v"))).toBe(true);
      expect(calls).toContain(`ps -aq --filter label=com.docker.compose.project=${project}`);
      expect(calls).toContain("rm -f own-container");
      expect(calls).toContain(`network prune -f --filter label=com.docker.compose.project=${project}`);
      expect(calls).not.toContain("network prune -f");
      // dangling images only: no image of any install is removed by name
      expect(calls.filter((c) => c.startsWith("image "))).toEqual(["image prune -f"]);
      const removed = calls.filter((c) => c.startsWith("volume rm "));
      if (project === "pgai-v4") {
        expect(removed).toEqual([]);
        expect(r.stdout).toContain("Kept monitoring_grafana_data, postgres_ai_configs_pgai_mon_data");
        expect(r.stdout).toContain("docker volume rm monitoring_grafana_data postgres_ai_configs_pgai_mon_data");
      } else {
        expect(removed).toEqual(["volume rm monitoring_grafana_data", "volume rm postgres_ai_configs_pgai_mon_data"]);
      }
    });
  }

  test("clean --keep-volumes keeps all volumes, including on a default project", () => {
    const p = makeProject("clean-keep", NEW_COMPOSE, `${NEW_ENV}COMPOSE_PROJECT_NAME=monitoring\n`);
    const r = p.run(["mon", "clean", "--keep-volumes"], { VOLUMES: volumes });
    expect(r.status).toBe(0);
    const calls = p.calls().map((c) => c.split(" |PROFILES=")[0]);
    expect(calls.some((c) => c.endsWith(" down"))).toBe(true);
    expect(calls.some((c) => c.startsWith("volume "))).toBe(false);
    expect(calls.filter((c) => c.startsWith("image "))).toEqual(["image prune -f"]);
    expect(calls).not.toContain("network prune -f");
  });

  test("stop's fallback removes networks by project label, never by name", () => {
    const p = makeProject("stop-networks", NEW_COMPOSE, `${NEW_ENV}COMPOSE_PROJECT_NAME=pgai-v4\n`);
    const r = p.run(["mon", "stop"], { DOWN_EXIT: "1" });
    expect(r.status).toBe(0);
    const calls = p.calls().map((c) => c.split(" |PROFILES=")[0]);
    expect(calls).toContain("network ls -q --filter label=com.docker.compose.project=pgai-v4");
    expect(calls.filter((c) => c.startsWith("network rm "))).toEqual(["network rm own-network"]);
    expect(calls).toContain("rm -f own-container");
  });
});

for (const project of ["monitoring!", " monitoring "]) {
  test(`clean rejects invalid explicit project ${JSON.stringify(project)} before deleting resources`, () => {
    const p = makeProject(`clean-invalid-${project.length}`, NEW_COMPOSE, NEW_ENV);
    const r = p.run(["mon", "clean"], { COMPOSE_PROJECT_NAME: project, VOLUMES: "monitoring_grafana_data" });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain("Invalid COMPOSE_PROJECT_NAME");
    expect(p.calls()).toEqual([]);
  });
}
