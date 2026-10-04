import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

/** Offline CLI smoke fixture: never use the developer's config or services. */
export function createCliSandbox() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-cli-smoke-"));
  const sourceCli = path.resolve(import.meta.dir, "..");
  const home = path.join(root, "home");
  const configHome = path.join(root, "config");
  const projectDir = path.join(root, "project");
  const binDir = path.join(root, "bin");
  const cliDir = path.join(root, "cli");
  for (const dir of [home, configHome, projectDir, binDir, path.join(cliDir, "bin")]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  // Copy the entry point so its demo-template lookup also stays disposable.
  // Libraries and dependencies are only read through these links.
  const cliPath = path.join(cliDir, "bin", "postgres-ai.ts");
  fs.copyFileSync(path.join(sourceCli, "bin", "postgres-ai.ts"), cliPath);
  for (const name of ["lib", "node_modules", "package.json", "sql"]) {
    fs.symlinkSync(path.join(sourceCli, name), path.join(cliDir, name));
  }
  const demoPath = path.join(root, "instances.demo.yml");
  fs.copyFileSync(path.join(sourceCli, "..", "instances.demo.yml"), demoPath);
  fs.writeFileSync(path.join(projectDir, "docker-compose.yml"), "services: {}\n");
  // Support detection only; any compose work fails before starting services.
  fs.writeFileSync(path.join(binDir, "docker"),
    '#!/bin/sh\ncase "$*" in\n  "info"|"compose version") exit 0 ;;\n  ps\ *) exit 0 ;;\n  *) exit 1 ;;\nesac\n', { mode: 0o700 });
  fs.writeFileSync(path.join(binDir, "docker-compose"), "#!/bin/sh\nexit 1\n", { mode: 0o700 });

  return {
    root, home, configHome, projectDir, demoPath, cliPath, binDir,
    run(args: string[], env: NodeJS.ProcessEnv = {}, options: { cwd?: string } = {}) {
      const childEnv = { ...process.env };
      // These values can implicitly select credentials, adoption or host-metric setup.
      for (const name of Object.keys(childEnv)) {
        if (name.startsWith("PGAI_")) delete childEnv[name];
      }
      const result = Bun.spawnSync([
        process.execPath, "--preload", path.join(import.meta.dir, "cli-offline-preload.ts"), cliPath, ...args,
      ], {
        cwd: options.cwd || projectDir,
        env: {
          ...childEnv, ...env,
          HOME: home,
          XDG_CONFIG_HOME: configHome,
          PATH: `${binDir}${path.delimiter}${process.env.PATH || ""}`,
        },
        timeout: 10000,
      });
      if (result.signalCode) throw new Error(`CLI smoke subprocess ended with ${result.signalCode}`);
      return {
        status: result.exitCode,
        stdout: new TextDecoder().decode(result.stdout),
        stderr: new TextDecoder().decode(result.stderr),
      };
    },
    cleanup() { fs.rmSync(root, { recursive: true, force: true }); },
  };
}
