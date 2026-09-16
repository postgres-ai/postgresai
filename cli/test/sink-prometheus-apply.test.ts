import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import { resolve } from "path";

/**
 * postgresai#359: writing the VictoriaMetrics admin keys to `.env` does nothing
 * on its own. The flags live on the sink-prometheus command line, and
 * `mon restart` is `docker compose restart`, which re-runs the container
 * exactly as recorded. So `mon update-config` has to bring the service in line
 * with the compose file, and these tests pin what it actually invokes.
 *
 * Docker is faked with a shim on PATH that records its argv, so this runs
 * anywhere and asserts the command rather than its effect.
 */
describe("applySinkPrometheusConfig", () => {
  let tempDir: string;

  const makeProject = (name: string, psOutput: string, upExit = 0) => {
    const dir = resolve(tempDir, name);
    const bin = resolve(dir, "bin");
    fs.mkdirSync(bin, { recursive: true });

    fs.writeFileSync(resolve(dir, ".env"), "PGAI_TAG=0.16.0\nVM_AUTH_USERNAME=vmauth\nVM_AUTH_PASSWORD=pw\n");
    fs.writeFileSync(resolve(dir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(dir, "instances.yml"), "# instances\n");

    const log = resolve(dir, "docker-calls.log");
    // `compose version` must succeed so getComposeCmd picks `docker compose`.
    fs.writeFileSync(
      resolve(bin, "docker"),
      [
        "#!/bin/sh",
        `echo "$@" >> ${JSON.stringify(log)}`,
        'case " $* " in',
        '  *" version "*) exit 0 ;;',
        `  *" ps "*) printf '%s' ${JSON.stringify(psOutput)}; exit 0 ;;`,
        `  *" up "*) exit ${upExit} ;;`,
        "esac",
        "exit 0",
      ].join("\n"),
      { mode: 0o755 },
    );
    return { dir, log };
  };

  const runCli = (dir: string, args: string[]) => {
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = process.execPath || "bun";
    const result = Bun.spawnSync([bunBin, cliPath, ...args], {
      env: { ...process.env, PATH: `${resolve(dir, "bin")}:${process.env.PATH}`, PGAI_TAG: undefined },
      cwd: dir,
    });
    return {
      status: result.exitCode,
      stdout: new TextDecoder().decode(result.stdout),
      stderr: new TextDecoder().decode(result.stderr),
    };
  };

  const runUpdateConfig = (dir: string) => runCli(dir, ["mon", "update-config"]);

  const dockerCalls = (log: string): string[] =>
    fs.existsSync(log) ? fs.readFileSync(log, "utf8").trim().split("\n") : [];

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-sink-apply-"));
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true });
  });

  test("recreates the running service, scoped and without touching its dependencies", () => {
    const { dir, log } = makeProject("running", "abc123containerid\n");
    const result = runUpdateConfig(dir);
    const calls = dockerCalls(log);

    // It must ask whether the service is running before doing anything.
    expect(calls.some((c) => c.includes("ps") && c.includes("sink-prometheus"))).toBe(true);

    const up = calls.find((c) => c.includes(" up "));
    expect(up).toBeDefined();
    expect(up).toContain("sink-prometheus");
    // Scoped: a bare `up -d` recreates grafana and strips the RDS CA mount.
    expect(up).toContain("--no-deps");
    // A running container proves config-init already populated the config
    // volume, so re-running it would only risk re-seeding a live-patched
    // prometheus.yml.
    expect(up).not.toContain("--force-recreate");
    expect(result.stdout).toContain("sink-prometheus is running the current config");
  });

  test("does not start a stack the operator stopped", () => {
    const { dir, log } = makeProject("stopped", "");
    const result = runUpdateConfig(dir);

    expect(dockerCalls(log).some((c) => c.includes(" up "))).toBe(false);
    expect(result.stdout).toContain("not running");
    expect(result.status).toBe(0);
  });

  test("a failed apply is reported and fails the command", () => {
    const { dir } = makeProject("upfails", "abc123containerid\n", 1);
    const result = runUpdateConfig(dir);

    expect(result.stderr).toContain("NOT in effect");
    // Must not exit 0: the keys are in .env but the container is still the old one.
    expect(result.status).not.toBe(0);
  });

  test("mon update does not claim success after a failed apply", () => {
    // It used to print the failure and then "✓ Update completed successfully",
    // so the last line an operator read was a green check over an exposed box.
    const { dir } = makeProject("update-upfails", "abc123containerid\n", 1);
    const result = runCli(dir, ["mon", "update"]);

    expect(result.stdout).not.toContain("Update completed successfully");
    expect(result.status).not.toBe(0);
  });
});
