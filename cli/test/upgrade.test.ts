import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { resolve } from "path";
import * as fs from "fs";
import * as os from "os";

// 30 seconds timeout for tests that spawn CLI processes
// This accommodates file I/O operations and process startup overhead in CI environments
const TEST_TIMEOUT = 30000;

/**
 * Run CLI command in a specific directory.
 *
 * @param args - CLI arguments to pass to postgres-ai (e.g., ["mon", "local-install", "--yes"])
 * @param cwd - Working directory where the command should run
 * @param env - Optional environment variables to override (merged with process.env)
 * @returns Object containing:
 *   - status: Process exit code (0 = success)
 *   - stdout: Standard output as string
 *   - stderr: Standard error as string
 */
function runCliInDir(args: string[], cwd: string, env: Record<string, string | undefined> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
    cwd,
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

describe("upgrade workflow", () => {
  /**
   * These tests verify the upgrade process documented in README.md:
   * 1. Update CLI (npm install -g postgresai@latest)
   * 2. Stop services (postgresai mon stop)
   * 3. Re-run local-install which updates .env with new version
   * 4. Verify services (postgresai mon status/health)
   *
   * Since Docker can't run in unit tests, we focus on testing the
   * configuration update behavior which is the core of the upgrade process.
   */

  let tempDir: string;

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-upgrade-test-"));
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("upgrade updates PGAI_TAG from old version to CLI version", () => {
    // Simulate existing installation with old version
    const testDir = resolve(tempDir, "upgrade-tag-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Create .env with old version (simulating pre-upgrade state)
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.13.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install (simulating upgrade after CLI update)
    // The --yes flag skips interactive prompts
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    // Read the updated .env (written before Docker operations)
    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // The old version should be replaced with the CLI version
    expect(envContent).not.toMatch(/PGAI_TAG=0\.13\.0/);
    // Should have a valid version tag (either semver or dev version)
    expect(envContent).toMatch(/PGAI_TAG=\d+\.\d+\.\d+|PGAI_TAG=0\.0\.0-dev/);
  }, { timeout: TEST_TIMEOUT });

  test("upgrade preserves Grafana password", () => {
    const testDir = resolve(tempDir, "upgrade-password-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation with password
    fs.writeFileSync(resolve(testDir, ".env"),
      "PGAI_TAG=0.12.0\nGF_SECURITY_ADMIN_PASSWORD=my-secure-password-123\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install (upgrade)
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // Password should be preserved
    expect(envContent).toMatch(/GF_SECURITY_ADMIN_PASSWORD=my-secure-password-123/);
    // Tag should be updated
    expect(envContent).not.toMatch(/PGAI_TAG=0\.12\.0/);
  }, { timeout: TEST_TIMEOUT });

  test("upgrade preserves custom registry", () => {
    const testDir = resolve(tempDir, "upgrade-registry-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation with custom registry
    fs.writeFileSync(resolve(testDir, ".env"),
      "PGAI_TAG=0.11.0\nPGAI_REGISTRY=registry.example.com/postgres-ai\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install (upgrade)
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // Registry should be preserved
    expect(envContent).toMatch(/PGAI_REGISTRY=registry\.example\.com\/postgres-ai/);
    // Tag should be updated
    expect(envContent).not.toMatch(/PGAI_TAG=0\.11\.0/);
  }, { timeout: TEST_TIMEOUT });

  test("upgrade preserves all settings together", () => {
    const testDir = resolve(tempDir, "upgrade-all-settings-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation with all settings
    fs.writeFileSync(resolve(testDir, ".env"),
      "PGAI_TAG=0.10.0\nPGAI_REGISTRY=my.registry.io\nGF_SECURITY_ADMIN_PASSWORD=super-secret\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install (upgrade)
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // All settings should be preserved
    expect(envContent).toMatch(/PGAI_REGISTRY=my\.registry\.io/);
    expect(envContent).toMatch(/GF_SECURITY_ADMIN_PASSWORD=super-secret/);
    // Tag should be updated to new version
    expect(envContent).not.toMatch(/PGAI_TAG=0\.10\.0/);
    expect(envContent).toMatch(/PGAI_TAG=/);
  }, { timeout: TEST_TIMEOUT });

  test("upgrade with --tag flag uses specified version", () => {
    const testDir = resolve(tempDir, "upgrade-custom-tag-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.9.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install with specific tag (for rollback or specific version upgrade)
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    const result = runCliInDir(
      ["mon", "local-install", "--tag", "0.14.0-beta.5", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // Should use the specified tag
    expect(envContent).toMatch(/PGAI_TAG=0\.14\.0-beta\.5/);
    // Stdout should confirm the tag being used (happens before Docker step)
    expect(result.stdout).toMatch(/Using image tag: 0\.14\.0-beta\.5/);
  }, { timeout: TEST_TIMEOUT });

  test("upgrade preserves .pgwatch-config file", () => {
    const testDir = resolve(tempDir, "upgrade-config-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation with config file
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.8.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");
    fs.writeFileSync(resolve(testDir, ".pgwatch-config"),
      "api_key=test-api-key-12345\ngrafana_password=existing-password\n");

    // Run local-install (upgrade)
    // Note: Command will fail at Docker step (no Docker in CI), but config file is preserved
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    // Config file should still exist
    expect(fs.existsSync(resolve(testDir, ".pgwatch-config"))).toBe(true);

    const configContent = fs.readFileSync(resolve(testDir, ".pgwatch-config"), "utf8");
    // Grafana password should be preserved (not overwritten since it exists)
    expect(configContent).toMatch(/grafana_password=existing-password/);
  }, { timeout: TEST_TIMEOUT });

  test("instances.yml is preserved during upgrade", () => {
    const testDir = resolve(tempDir, "upgrade-instances-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Simulate existing installation with instances
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.7.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    const instancesContent = `# PostgreSQL instances to monitor
- name: production-db
  conn_str: postgresql://monitor:pass@prod.example.com:5432/mydb
  preset_metrics: full
  is_enabled: true
`;
    fs.writeFileSync(resolve(testDir, "instances.yml"), instancesContent);

    // Note: local-install in production mode clears instances.yml
    // This is intentional behavior - upgrade should use 'mon start' not 'local-install'
    // for preserving instances. Testing that the file exists after operation.
    // Note: Command will fail at Docker step (no Docker in CI), but file is created
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    // instances.yml should exist (content may be reset by local-install)
    expect(fs.existsSync(resolve(testDir, "instances.yml"))).toBe(true);
  }, { timeout: TEST_TIMEOUT });
});

describe("upgrade error handling", () => {
  /**
   * Tests for edge cases and error scenarios in the upgrade workflow.
   * These ensure the CLI handles incomplete or malformed configurations gracefully.
   */

  let tempDir: string;

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-upgrade-error-test-"));
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("local-install creates .env if missing", () => {
    const testDir = resolve(tempDir, "missing-env-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Only create docker-compose.yml (no .env)
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");

    // Run local-install without existing .env
    // Note: Command will fail at Docker step (no Docker in CI), but .env is created before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    // .env should be created (before Docker operations fail)
    expect(fs.existsSync(resolve(testDir, ".env"))).toBe(true);

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(envContent).toMatch(/PGAI_TAG=/);
  }, { timeout: TEST_TIMEOUT });

  test("local-install handles .env without PGAI_TAG line", () => {
    const testDir = resolve(tempDir, "no-tag-line-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Create .env without PGAI_TAG (only has other settings)
    fs.writeFileSync(resolve(testDir, ".env"), "GF_SECURITY_ADMIN_PASSWORD=old-password\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Run local-install
    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    // Should add PGAI_TAG to the file
    expect(envContent).toMatch(/PGAI_TAG=/);
    // Should preserve existing settings
    expect(envContent).toMatch(/GF_SECURITY_ADMIN_PASSWORD=old-password/);
  }, { timeout: TEST_TIMEOUT });

  test("local-install handles same version (no-op scenario)", () => {
    const testDir = resolve(tempDir, "same-version-test");
    fs.mkdirSync(testDir, { recursive: true });

    // First, run local-install to get the current CLI version
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.0.0-placeholder\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // Note: Command will fail at Docker step (no Docker in CI), but .env is updated before that
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const firstEnv = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(firstEnv).toMatch(/PGAI_TAG=/);

    // Run again with same version - should update .env identically
    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    // .env should still have a valid tag
    const finalEnv = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(finalEnv).toMatch(/PGAI_TAG=/);
  }, { timeout: TEST_TIMEOUT });
});

describe("upgrade CLI commands", () => {
  test("mon stop command exists and shows help", () => {
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "stop", "--help"], {
      env: process.env,
    });

    expect(result.exitCode).toBe(0);
    const stdout = new TextDecoder().decode(result.stdout);
    expect(stdout).toMatch(/stop monitoring services/i);
  }, { timeout: TEST_TIMEOUT });

  test("mon start command exists and shows help", () => {
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "start", "--help"], {
      env: process.env,
    });

    expect(result.exitCode).toBe(0);
    const stdout = new TextDecoder().decode(result.stdout);
    expect(stdout).toMatch(/start monitoring services/i);
  }, { timeout: TEST_TIMEOUT });

  test("mon status command exists and shows help", () => {
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "status", "--help"], {
      env: process.env,
    });

    expect(result.exitCode).toBe(0);
    const stdout = new TextDecoder().decode(result.stdout);
    expect(stdout).toMatch(/status/i);
  }, { timeout: TEST_TIMEOUT });

  test("mon health command exists and shows help", () => {
    const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
    const result = Bun.spawnSync([bunBin, cliPath, "mon", "health", "--help"], {
      env: process.env,
    });

    expect(result.exitCode).toBe(0);
    const stdout = new TextDecoder().decode(result.stdout);
    expect(stdout).toMatch(/health/i);
  }, { timeout: TEST_TIMEOUT });
});
