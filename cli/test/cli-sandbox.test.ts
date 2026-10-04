import { describe, test, expect } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { createCliSandbox } from "./cli-sandbox";

describe("CLI smoke isolation", () => {
  test("fake Docker parses, supports inspection, and refuses service work", () => {
    const sandbox = createCliSandbox();
    try {
      const docker = path.join(sandbox.binDir, "docker");
      const env = { ...process.env, HOME: sandbox.home, XDG_CONFIG_HOME: sandbox.configHome };
      const probe = (command: string[]) => Bun.spawnSync(command, { env, timeout: 5000 });
      expect(probe(["/bin/sh", "-n", docker]).exitCode).toBe(0);
      expect(probe([docker, "info"]).exitCode).toBe(0);
      expect(probe([docker, "compose", "version"]).exitCode).toBe(0);
      expect(probe([docker, "ps", "--format", "{{.Names}}"]).exitCode).toBe(0);
      expect(probe([docker, "compose", "up", "-d"]).exitCode).toBe(1);
    } finally {
      sandbox.cleanup();
    }
  });

  for (const useXdg of [true, false]) {
    test(`local-install preserves a dummy parent config (XDG=${useXdg})`, () => {
      const sandbox = createCliSandbox();
      try {
        const parentHome = path.join(sandbox.root, "dummy-parent-home");
        const parentConfigHome = path.join(sandbox.root, "dummy-parent-config");
        const parentConfig = path.join(useXdg ? parentConfigHome : path.join(parentHome, ".config"), "postgresai", "config.json");
        fs.mkdirSync(path.dirname(parentConfig), { recursive: true });
        const original = JSON.stringify({ apiKey: "dummy-original", orgId: 123, baseUrl: "https://example.invalid" });
        fs.writeFileSync(parentConfig, original);

        const result = sandbox.run([
          "mon", "local-install", "--api-key", "dummy-test-fixture",
          "--db-url", "postgresql://user:pass@localhost:5432/testdb",
        ], { HOME: parentHome, XDG_CONFIG_HOME: useXdg ? parentConfigHome : undefined });

        expect(result.status).toBe(1); // offline compose refuses any service work
        expect(result.stdout).toContain("Using API key provided via --api-key parameter");
        expect(result.stdout).toContain("Using database URL provided via --db-url parameter");
        expect(fs.readFileSync(parentConfig, "utf8")).toBe(original);
        const childConfig = JSON.parse(fs.readFileSync(path.join(sandbox.configHome, "postgresai", "config.json"), "utf8"));
        expect(childConfig.apiKey).toBe("dummy-test-fixture");
        expect(fs.existsSync(path.join(sandbox.projectDir, "instances.yml"))).toBe(true);
      } finally {
        sandbox.cleanup();
      }
    });
  }
});
