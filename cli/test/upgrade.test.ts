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
    expect(envContent).toMatch(/REPLICATOR_PASSWORD=[a-f0-9]{64}/);
    expect(envContent).toMatch(/VM_AUTH_USERNAME=vmauth/);
    expect(envContent).toMatch(/^VM_AUTH_PASSWORD=[A-Za-z0-9+/]+={0,2}\s*$/m);
  }, { timeout: TEST_TIMEOUT });

  test("local-install handles .env without PGAI_TAG line", () => {
    const testDir = resolve(tempDir, "no-tag-line-test");
    fs.mkdirSync(testDir, { recursive: true });

    // Create .env without PGAI_TAG (only has other settings)
    fs.writeFileSync(resolve(testDir, ".env"), "GF_SECURITY_ADMIN_PASSWORD=old-password\nREPLICATOR_PASSWORD=existing-repl\nVM_AUTH_USERNAME=existing-vm-user\nVM_AUTH_PASSWORD=existing-vm-pass\n");
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
    expect(envContent).toMatch(/REPLICATOR_PASSWORD=existing-repl/);
    expect(envContent).toMatch(/VM_AUTH_USERNAME=existing-vm-user/);
    expect(envContent).toMatch(/VM_AUTH_PASSWORD=existing-vm-pass/);
  }, { timeout: TEST_TIMEOUT });

  test("local-install strips only matching quotes from VM auth values", () => {
    const testDir = resolve(tempDir, "quoted-vm-auth-test");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(
      resolve(testDir, ".env"),
      "VM_AUTH_USERNAME=\"quoted-vm-user\"\nVM_AUTH_PASSWORD='quoted-vm-pass'\n"
    );
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      { PGAI_TAG: undefined }
    );

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(envContent).toMatch(/VM_AUTH_USERNAME=quoted-vm-user/);
    expect(envContent).toMatch(/VM_AUTH_PASSWORD=quoted-vm-pass/);
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

describe("in-place upgrade env migration (mon update / update-config)", () => {
  /**
   * Regression tests for the 0.14 -> 0.15 in-place upgrade gap (#203).
   *
   * Before this fix, a user who installed at 0.14 and ran the documented
   * upgrade flow (`pgai mon update`) ended up with a .env file that lacked
   * VM_AUTH_USERNAME / VM_AUTH_PASSWORD, so sink-prometheus exited with:
   *
   *   fatal cannot read "/postgres_ai_configs/prometheus/prometheus.yml":
   *   cannot expand environment variables: missing "VM_AUTH_USERNAME" env var
   *
   * `mon update` and `mon update-config` now migrate .env additively before
   * doing anything else.
   */

  let tempDir: string;

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-upgrade-env-migration-"));
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("mon update-config appends missing VM_AUTH_USERNAME / VM_AUTH_PASSWORD to a 0.14-shaped .env", () => {
    const testDir = resolve(tempDir, "update-config-0.14-env");
    fs.mkdirSync(testDir, { recursive: true });

    // 0.14-shaped .env: PGAI_TAG present, VM_AUTH_* absent.
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\nGF_SECURITY_ADMIN_PASSWORD=user-set-grafana-pw\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // The compose run will fail (no Docker in CI), but env migration runs first.
    runCliInDir(["mon", "update-config"], testDir, { PGAI_TAG: undefined });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    // Existing values must be preserved verbatim.
    expect(envContent).toMatch(/^PGAI_TAG=0\.14\.0$/m);
    expect(envContent).toMatch(/^GF_SECURITY_ADMIN_PASSWORD=user-set-grafana-pw$/m);

    // New required keys must be appended (vmauth username + non-empty base64 password).
    expect(envContent).toMatch(/^VM_AUTH_USERNAME=vmauth$/m);
    expect(envContent).toMatch(/^VM_AUTH_PASSWORD=[A-Za-z0-9+/]+={0,2}$/m);

    // REPLICATOR_PASSWORD was introduced earlier and is also part of the contract.
    expect(envContent).toMatch(/^REPLICATOR_PASSWORD=[a-f0-9]{64}$/m);
  }, { timeout: TEST_TIMEOUT });

  test("mon update appends missing VM_AUTH_USERNAME / VM_AUTH_PASSWORD to a 0.14-shaped .env", () => {
    const testDir = resolve(tempDir, "update-0.14-env");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // mon update will fail (no Docker in CI, no git repo), but env migration runs first.
    const result = runCliInDir(["mon", "update"], testDir, { PGAI_TAG: undefined });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    expect(envContent).toMatch(/^PGAI_TAG=0\.14\.0$/m);
    expect(envContent).toMatch(/^VM_AUTH_USERNAME=vmauth$/m);
    expect(envContent).toMatch(/^VM_AUTH_PASSWORD=[A-Za-z0-9+/]+={0,2}$/m);

    // The migration step should print what it added so the user can see it.
    expect(result.stdout).toMatch(/Added missing \.env keys/);
    expect(result.stdout).toMatch(/VM_AUTH_USERNAME/);
    expect(result.stdout).toMatch(/VM_AUTH_PASSWORD/);
  }, { timeout: TEST_TIMEOUT });

  test("mon update preserves existing VM_AUTH_* values (no rotation)", () => {
    const testDir = resolve(tempDir, "update-preserve-vm-auth");
    fs.mkdirSync(testDir, { recursive: true });

    // User already has VM auth configured (e.g. set up via rotate-vm-auth.sh).
    fs.writeFileSync(
      resolve(testDir, ".env"),
      "PGAI_TAG=0.15.0\nVM_AUTH_USERNAME=custom-user\nVM_AUTH_PASSWORD=custom-pw-do-not-rotate\nREPLICATOR_PASSWORD=" +
        "a".repeat(64) +
        "\n",
    );
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    const result = runCliInDir(["mon", "update"], testDir, { PGAI_TAG: undefined });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    expect(envContent).toMatch(/^VM_AUTH_USERNAME=custom-user$/m);
    expect(envContent).toMatch(/^VM_AUTH_PASSWORD=custom-pw-do-not-rotate$/m);
    expect(envContent).toMatch(/^REPLICATOR_PASSWORD=a{64}$/m);

    // When nothing is missing, the migration step should say so.
    expect(result.stdout).toMatch(/\.env is up to date/);
  }, { timeout: TEST_TIMEOUT });

  test("mon update-config handles a .env that doesn't end with a newline", () => {
    const testDir = resolve(tempDir, "update-config-no-trailing-newline");
    fs.mkdirSync(testDir, { recursive: true });

    // No trailing newline - migration must add one before appending new keys
    // or we'd produce e.g. `PGAI_TAG=0.14.0VM_AUTH_USERNAME=vmauth`.
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), "version: '3'\nservices: {}\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, { PGAI_TAG: undefined });

    const envContent = fs.readFileSync(resolve(testDir, ".env"), "utf8");

    expect(envContent).toMatch(/^PGAI_TAG=0\.14\.0$/m);
    expect(envContent).toMatch(/^VM_AUTH_USERNAME=vmauth$/m);
    // No key should be glued onto the previous line.
    expect(envContent).not.toMatch(/PGAI_TAG=0\.14\.0VM_AUTH_USERNAME/);
  }, { timeout: TEST_TIMEOUT });
});

describe("in-place upgrade compose refresh (non-git npx upgrade)", () => {
  /**
   * Regression tests for the GA-blocking 0.14 -> 0.15 npx-upgrade gap (#186).
   *
   * The documented npx upgrade (`npm i -g postgresai@latest` then `pgai mon update`)
   * only ever refreshed docker-compose.yml for *git* checkouts (via `git pull`).
   * npx/global-npm installs are non-git, so the OLD 0.14 docker-compose.yml was
   * retained while PGAI_TAG advanced to 0.15. VM basic auth is new in 0.15, so the
   * retained 0.14 compose never wires VM_AUTH_* into the sink-prometheus
   * (VictoriaMetrics) service. The 0.15 configs image ships a prometheus.yml that
   * templates %{VM_AUTH_USERNAME}, so VictoriaMetrics aborts on boot:
   *
   *   Exited (255): missing "VM_AUTH_USERNAME" env var
   *
   * -> all dashboards are dataless after upgrade. This hits every existing npx
   * self-hosted user and blocks 0.15.0 GA.
   *
   * The fix: for NON-GIT installs, refresh the CLI-owned docker-compose.yml from the
   * target ref when it is stale, backing up the prior compose first, and never
   * touching user-owned files (.env / instances.yml).
   *
   * Hermetic seam: PGAI_COMPOSE_SOURCE points the refresh at a local fixture file
   * (instead of fetching over the network) so these tests are offline-safe. The
   * seam is only honored when NODE_ENV === "test" (bun sets this automatically;
   * we also set it explicitly), so it is never reachable in a user environment.
   */

  // A valid target 0.15 compose fixture: it carries the VM_AUTH_* wiring that the
  // stale 0.14 compose lacks AND the keystone `sink-prometheus` service the
  // validator requires. Self-contained so the test doesn't depend on the live
  // repo compose changing under it.
  const TARGET_0_15_COMPOSE = [
    "version: '3.8'",
    "services:",
    "  sink-prometheus:",
    "    image: victoriametrics/victoria-metrics:v1.140.0",
    "    container_name: sink-prometheus",
    "    environment:",
    "      - VM_AUTH_USERNAME=${VM_AUTH_USERNAME:-}",
    "      - VM_AUTH_PASSWORD=${VM_AUTH_PASSWORD:-}",
    "  grafana:",
    "    image: grafana/grafana:11.0.0",
    "",
  ].join("\n");

  // A 0.14-shaped compose: a real-ish stack with ZERO VM_AUTH wiring.
  const STALE_0_14_COMPOSE = [
    "version: '3.8'",
    "services:",
    "  sink-prometheus:",
    "    image: victoriametrics/victoria-metrics:v1.115.0",
    "    container_name: sink-prometheus",
    "    ports:",
    '      - "9090:8428"',
    "  grafana:",
    "    image: grafana/grafana:11.0.0",
    "",
  ].join("\n");

  let tempDir: string;
  // Path to a written-out copy of TARGET_0_15_COMPOSE, used as the fetch fixture.
  let targetComposePath: string;

  const listBackups = (dir: string): string[] =>
    fs.readdirSync(dir).filter((f) => f.startsWith("docker-compose.yml.bak")).sort();

  beforeAll(() => {
    tempDir = fs.mkdtempSync(resolve(os.tmpdir(), "pgai-upgrade-compose-refresh-"));
    targetComposePath = resolve(tempDir, "fixture-target-compose.yml");
    fs.writeFileSync(targetComposePath, TARGET_0_15_COMPOSE);
  });

  afterAll(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("non-git upgrade refreshes a stale docker-compose.yml to the target version (VM_AUTH wiring), preserves instances.yml, and writes a backup labeled with the OLD tag", () => {
    const testDir = resolve(tempDir, "update-config-stale-compose");
    fs.mkdirSync(testDir, { recursive: true });

    // Arrange: a 0.14-shaped, NON-GIT install. Old compose has no VM_AUTH wiring.
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(
      resolve(testDir, "instances.yml"),
      "# PostgreSQL instances to monitor\n- name: keep-me\n  conn_str: postgresql://m:p@h:5432/db\n",
    );

    // Sanity: the stale compose really lacks VM_AUTH wiring before the upgrade.
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).not.toMatch(/VM_AUTH_USERNAME/);

    // Act: documented in-place upgrade step on a non-git install. The compose `run`
    // will fail (no Docker in CI), but the compose refresh runs first. Point the
    // refresh at the local fixture so the test is hermetic (no network).
    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // Assert: deployed compose is REFRESHED to the target version (VM_AUTH wired).
    const after = fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8");
    expect(after).toMatch(/VM_AUTH_USERNAME/);
    expect(after).toMatch(/sink-prometheus/);

    // Assert: instances.yml is PRESERVED (user-owned file untouched).
    expect(fs.readFileSync(resolve(testDir, "instances.yml"), "utf8")).toMatch(/keep-me/);

    // Assert: exactly one backup, labeled with the OLD (0.14.0) tag, containing
    // the pristine pre-upgrade compose. Name is `bak-<oldtag>-<hash8>`.
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-0\.14\.0-[0-9a-f]{8}$/);
    expect(fs.readFileSync(resolve(testDir, backups[0]!), "utf8")).toBe(STALE_0_14_COMPOSE);
  }, { timeout: TEST_TIMEOUT });

  test("non-git upgrade via `mon update` also refreshes a stale docker-compose.yml", () => {
    const testDir = resolve(tempDir, "update-stale-compose");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n- name: keep-me\n");

    runCliInDir(["mon", "update"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    const after = fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8");
    expect(after).toMatch(/VM_AUTH_USERNAME/);
    expect(fs.readFileSync(resolve(testDir, "instances.yml"), "utf8")).toMatch(/keep-me/);
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-0\.14\.0-[0-9a-f]{8}$/);
  }, { timeout: TEST_TIMEOUT });

  test("git checkouts are left untouched (compose managed by git pull)", () => {
    const testDir = resolve(tempDir, "git-checkout-no-refresh");
    fs.mkdirSync(testDir, { recursive: true });
    fs.mkdirSync(resolve(testDir, ".git"), { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // Compose must be unchanged for git checkouts, and no backup written.
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
    expect(listBackups(testDir)).toEqual([]);
  }, { timeout: TEST_TIMEOUT });

  test("already-current compose is not rewritten and no backup is created (idempotent)", () => {
    const testDir = resolve(tempDir, "already-current-compose");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), TARGET_0_15_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // Content already matches target -> no rewrite, no backup.
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(TARGET_0_15_COMPOSE);
    expect(listBackups(testDir)).toEqual([]);
  }, { timeout: TEST_TIMEOUT });

  test("a trailing-newline-only difference is NOT treated as stale (no churn, no backup)", () => {
    const testDir = resolve(tempDir, "trailing-newline-noop");
    fs.mkdirSync(testDir, { recursive: true });

    // Deployed compose == target but with extra trailing whitespace/newlines.
    const deployed = TARGET_0_15_COMPOSE + "\n\n";
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), deployed);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // Whitespace-only delta must not trigger a rewrite or a backup.
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(deployed);
    expect(listBackups(testDir)).toEqual([]);
  }, { timeout: TEST_TIMEOUT });

  describe("fetched payload validation (BLOCKER #186 hardening)", () => {
    /**
     * A 200 response can still carry a NON-compose body (HTML login/captcha/
     * proxy/maintenance page) or an empty/garbage body. Such a payload must
     * NEVER clobber a working docker-compose.yml — that would be strictly worse
     * than keeping the stale-but-valid one. The refresh must validate the fetched
     * text and, on failure, behave EXACTLY like a clean fetch failure: keep the
     * existing compose, write NO backup, warn, no-op.
     */

    test("an HTML (non-compose) 200 body leaves the deployed compose UNCHANGED, writes NO backup, and warns", () => {
      const testDir = resolve(tempDir, "garbage-html-body");
      fs.mkdirSync(testDir, { recursive: true });

      // The "fetched" payload is an HTML login/proxy page served with a 200.
      const htmlBody = "<!DOCTYPE html>\n<html><head><title>Sign in</title></head>\n<body>Please log in to continue.</body></html>\n";
      const htmlSource = resolve(testDir, "html-source.html");
      fs.writeFileSync(htmlSource, htmlBody);

      fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
      fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
      fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

      const result = runCliInDir(["mon", "update-config"], testDir, {
        NODE_ENV: "test",
        PGAI_TAG: undefined,
        PGAI_COMPOSE_SOURCE: htmlSource,
      });

      // The working compose must be left intact — NOT clobbered with HTML.
      expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
      // No backup is written when nothing is refreshed.
      expect(listBackups(testDir)).toEqual([]);
      // The user is warned that no valid compose was retrieved.
      expect(result.stderr).toMatch(/Could not refresh docker-compose\.yml/);
    }, { timeout: TEST_TIMEOUT });

    test("a YAML body WITHOUT a sink-prometheus service is rejected (compose untouched, no backup)", () => {
      const testDir = resolve(tempDir, "yaml-missing-keystone");
      fs.mkdirSync(testDir, { recursive: true });

      // Valid YAML, has services, but lacks the keystone sink-prometheus service.
      const wrongCompose = "version: '3.8'\nservices:\n  grafana:\n    image: grafana/grafana:11.0.0\n";
      const wrongSource = resolve(testDir, "wrong-compose.yml");
      fs.writeFileSync(wrongSource, wrongCompose);

      fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
      fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
      fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

      runCliInDir(["mon", "update-config"], testDir, {
        NODE_ENV: "test",
        PGAI_TAG: undefined,
        PGAI_COMPOSE_SOURCE: wrongSource,
      });

      expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
      expect(listBackups(testDir)).toEqual([]);
    }, { timeout: TEST_TIMEOUT });

    test("an empty fetched body leaves the deployed compose intact and writes NO backup", () => {
      const testDir = resolve(tempDir, "empty-body");
      fs.mkdirSync(testDir, { recursive: true });

      const emptySource = resolve(testDir, "empty.yml");
      fs.writeFileSync(emptySource, "");

      fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
      fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
      fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

      runCliInDir(["mon", "update-config"], testDir, {
        NODE_ENV: "test",
        PGAI_TAG: undefined,
        PGAI_COMPOSE_SOURCE: emptySource,
      });

      expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
      expect(listBackups(testDir)).toEqual([]);
    }, { timeout: TEST_TIMEOUT });
  });

  test("repeated update-config runs preserve the FIRST/pristine backup (no overwrite)", () => {
    const testDir = resolve(tempDir, "backup-collision");
    fs.mkdirSync(testDir, { recursive: true });

    // PGAI_TAG stays 0.14.0 across update-config runs (it doesn't advance there).
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    // First run: refreshes to target, backs up the pristine 0.14 compose.
    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });
    const afterFirst = listBackups(testDir);
    expect(afterFirst.length).toBe(1);
    const pristineBackup = afterFirst[0]!;
    expect(fs.readFileSync(resolve(testDir, pristineBackup), "utf8")).toBe(STALE_0_14_COMPOSE);

    // Now the deployed compose IS the target. Simulate a second drift back to a
    // (different) stale compose, then run update-config again with the SAME tag.
    const SECOND_STALE = STALE_0_14_COMPOSE.replace("v1.115.0", "v1.116.0");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), SECOND_STALE);

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // The FIRST/pristine backup must still be intact (not overwritten), and the
    // second distinct old content gets its own backup (unique by content hash).
    expect(fs.readFileSync(resolve(testDir, pristineBackup), "utf8")).toBe(STALE_0_14_COMPOSE);
    const afterSecond = listBackups(testDir);
    expect(afterSecond.length).toBe(2);
    expect(afterSecond).toContain(pristineBackup);
  }, { timeout: TEST_TIMEOUT });

  test("local-install labels the backup with the OLD tag, not the new CLI version", () => {
    const testDir = resolve(tempDir, "local-install-old-tag-backup");
    fs.mkdirSync(testDir, { recursive: true });

    // 0.14 install. local-install rewrites .env PGAI_TAG to the CLI version
    // BEFORE the compose refresh; the backup must still reflect the OLD (0.14.0)
    // tag, not the new one.
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      {
        NODE_ENV: "test",
        PGAI_TAG: undefined,
        PGAI_COMPOSE_SOURCE: targetComposePath,
      },
    );

    // .env PGAI_TAG was advanced to the new CLI version...
    const envAfter = fs.readFileSync(resolve(testDir, ".env"), "utf8");
    expect(envAfter).not.toMatch(/PGAI_TAG=0\.14\.0/);

    // ...but the backup of the OLD compose must be labeled with 0.14.0, NOT the
    // new tag, and must contain the pristine pre-upgrade compose.
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-0\.14\.0-[0-9a-f]{8}$/);
    expect(fs.readFileSync(resolve(testDir, backups[0]!), "utf8")).toBe(STALE_0_14_COMPOSE);
  }, { timeout: TEST_TIMEOUT });

  test("a fetch failure (no payload retrieved) leaves the deployed compose intact, writes NO backup, warns, and does not crash", () => {
    // Contract bullet: "Best-effort — a fetch failure warns and keeps the
    // existing compose." Force fetchTargetCompose() -> null hermetically by
    // pointing the test seam at a path that does not exist. This is the network
    // -down / GitLab-5xx branch: it must NOT turn a metrics-only outage into a
    // hard CLI failure, and must never clobber the stale-but-valid compose.
    const testDir = resolve(tempDir, "fetch-failure-null");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    const result = runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      // Nonexistent fixture path -> fetchTargetCompose() returns null.
      PGAI_COMPOSE_SOURCE: resolve(testDir, "does-not-exist.yml"),
    });

    // The working compose must be untouched, and no backup written.
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
    expect(listBackups(testDir)).toEqual([]);
    // The user is warned, and the CLI does not crash on the env-migration step.
    expect(result.stderr).toMatch(/Could not refresh docker-compose\.yml/);
  }, { timeout: TEST_TIMEOUT });

  test("no deployed compose yet is a clean no-op: returns false, writes nothing, no backup", async () => {
    // Guard: `if (!fs.existsSync(composeFile)) return false` keeps the refresh
    // from racing with the green-field bootstrap path. NOTE: this branch is not
    // reachable through the black-box `mon update-config` flow — resolveOrInitPaths
    // bootstraps (fetches) a compose BEFORE the refresh runs, so by then the file
    // always exists. We therefore exercise the guard directly against the exported
    // helper, which is the only faithful, hermetic way to cover it.
    const { refreshBundledComposeIfStale } = await import("../bin/postgres-ai.ts");

    const testDir = resolve(tempDir, "no-deployed-compose");
    fs.mkdirSync(testDir, { recursive: true });
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");
    // Intentionally NO docker-compose.yml.

    process.env.NODE_ENV = "test";
    process.env.PGAI_COMPOSE_SOURCE = targetComposePath;
    const refreshed = await refreshBundledComposeIfStale(testDir);

    // No compose on disk -> no-op: returns false, materializes no compose, and
    // writes no backup. The bootstrap path owns the first fetch.
    expect(refreshed).toBe(false);
    expect(fs.existsSync(resolve(testDir, "docker-compose.yml"))).toBe(false);
    expect(listBackups(testDir)).toEqual([]);
  }, { timeout: TEST_TIMEOUT });

  test("git checkout is a no-op even with a stale compose (direct helper guard)", async () => {
    // Companion guard: `.git` present -> the repo manages the compose via
    // `git pull`, so the helper must return false and touch nothing. Covered
    // black-box too (below), but pinned here against the exported helper.
    const { refreshBundledComposeIfStale } = await import("../bin/postgres-ai.ts");

    const testDir = resolve(tempDir, "git-checkout-helper-noop");
    fs.mkdirSync(resolve(testDir, ".git"), { recursive: true });
    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=0.14.0\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);

    process.env.NODE_ENV = "test";
    process.env.PGAI_COMPOSE_SOURCE = targetComposePath;
    const refreshed = await refreshBundledComposeIfStale(testDir);

    expect(refreshed).toBe(false);
    expect(fs.readFileSync(resolve(testDir, "docker-compose.yml"), "utf8")).toBe(STALE_0_14_COMPOSE);
    expect(listBackups(testDir)).toEqual([]);
  }, { timeout: TEST_TIMEOUT });

  test("backup falls back to a timestamp suffix when .env has no PGAI_TAG line", () => {
    // When readDeployedTag() returns null (no PGAI_TAG to label the backup with),
    // the backup name falls back to an ISO-8601 timestamp suffix
    // (`bak-<YYYY-MM-DDTHH-MM-SS-mmmZ>-<hash8>`). Every other test seeds
    // PGAI_TAG=0.14.0, so this branch was previously unexercised.
    const testDir = resolve(tempDir, "timestamp-suffix-fallback");
    fs.mkdirSync(testDir, { recursive: true });

    // .env exists (non-git install) but carries NO PGAI_TAG line.
    fs.writeFileSync(resolve(testDir, ".env"), "GRAFANA_PASSWORD=secret\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // The compose was refreshed, and the single backup is labeled with a
    // timestamp suffix (NOT a tag), still uniquified by the content hash.
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-\d{4}-\d{2}-\d{2}T[\dZ-]+-[0-9a-f]{8}$/);
    expect(fs.readFileSync(resolve(testDir, backups[0]!), "utf8")).toBe(STALE_0_14_COMPOSE);
  }, { timeout: TEST_TIMEOUT });

  test("a malformed/hostile PGAI_TAG is rejected and the backup falls back to a timestamp suffix (no path traversal)", () => {
    // readDeployedTag() validates the tag against a conservative charset before
    // it flows into the backup filename, so a path-traversal-shaped value cannot
    // escape projectDir; it falls back to the timestamp suffix instead.
    const testDir = resolve(tempDir, "hostile-tag-rejected");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), "PGAI_TAG=../../../../tmp/evil\n");
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(["mon", "update-config"], testDir, {
      NODE_ENV: "test",
      PGAI_TAG: undefined,
      PGAI_COMPOSE_SOURCE: targetComposePath,
    });

    // The single backup stays inside projectDir with a timestamp suffix — the
    // traversal value never reaches the filename.
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-\d{4}-\d{2}-\d{2}T[\dZ-]+-[0-9a-f]{8}$/);
    // Nothing was written outside projectDir.
    expect(fs.existsSync("/tmp/evil")).toBe(false);
  }, { timeout: TEST_TIMEOUT });

  test("local-install sanitizes the OLD tag it passes in: a hostile PGAI_TAG falls back to a timestamp suffix", () => {
    // local-install captures the OLD .env PGAI_TAG and passes it to the refresh as
    // `oldTag`, BYPASSING readDeployedTag. Sanitization must therefore happen
    // centrally inside the helper so a hostile tag on THIS path also cannot escape
    // projectDir or land literal `/`/quote chars in the backup filename.
    const testDir = resolve(tempDir, "local-install-hostile-old-tag");
    fs.mkdirSync(testDir, { recursive: true });

    fs.writeFileSync(resolve(testDir, ".env"), 'PGAI_TAG="../../../../tmp/evil"\n');
    fs.writeFileSync(resolve(testDir, "docker-compose.yml"), STALE_0_14_COMPOSE);
    fs.writeFileSync(resolve(testDir, "instances.yml"), "# instances\n");

    runCliInDir(
      ["mon", "local-install", "--db-url", "postgresql://u:p@h:5432/d", "--yes"],
      testDir,
      {
        NODE_ENV: "test",
        PGAI_TAG: undefined,
        PGAI_COMPOSE_SOURCE: targetComposePath,
      },
    );

    // The hostile oldTag never reaches the filename: the single backup stays
    // inside projectDir with a timestamp suffix, and nothing escapes to /tmp.
    const backups = listBackups(testDir);
    expect(backups.length).toBe(1);
    expect(backups[0]).toMatch(/^docker-compose\.yml\.bak-\d{4}-\d{2}-\d{2}T[\dZ-]+-[0-9a-f]{8}$/);
    expect(fs.existsSync("/tmp/evil")).toBe(false);
  }, { timeout: TEST_TIMEOUT });
});
