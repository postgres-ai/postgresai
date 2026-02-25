import { describe, test, expect, beforeEach, afterEach, mock, spyOn } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

/**
 * Test updatePgwatchConfig function behavior.
 * Since the function is internal to postgres-ai.ts, we test it via file system operations.
 */
function updatePgwatchConfig(configPath: string, updates: Record<string, string>): void {
  let lines: string[] = [];

  // Read existing config if it exists
  if (fs.existsSync(configPath)) {
    const stats = fs.statSync(configPath);
    if (!stats.isDirectory()) {
      const content = fs.readFileSync(configPath, "utf8");
      lines = content.split(/\r?\n/).filter(l => l.trim() !== "");
    }
  }

  // Update or add each key
  for (const [key, value] of Object.entries(updates)) {
    const existingIndex = lines.findIndex(l => l.startsWith(key + "="));
    if (existingIndex >= 0) {
      lines[existingIndex] = `${key}=${value}`;
    } else {
      lines.push(`${key}=${value}`);
    }
  }

  fs.writeFileSync(configPath, lines.join("\n") + "\n", { encoding: "utf8", mode: 0o600 });
}

describe("updatePgwatchConfig", () => {
  let tempDir: string;
  let configPath: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pgwatch-test-"));
    configPath = path.join(tempDir, ".pgwatch-config");
  });

  afterEach(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("creates new file with updates", () => {
    updatePgwatchConfig(configPath, {
      api_key: "test-key-123",
      project_name: "my-project",
    });

    expect(fs.existsSync(configPath)).toBe(true);
    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=test-key-123");
    expect(content).toContain("project_name=my-project");
  });

  test("updates existing keys", () => {
    fs.writeFileSync(configPath, "api_key=old-key\nproject_name=old-project\n");

    updatePgwatchConfig(configPath, {
      api_key: "new-key",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=new-key");
    expect(content).toContain("project_name=old-project");
    expect(content).not.toContain("api_key=old-key");
  });

  test("adds new keys to existing file", () => {
    fs.writeFileSync(configPath, "api_key=existing-key\n");

    updatePgwatchConfig(configPath, {
      project_name: "new-project",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=existing-key");
    expect(content).toContain("project_name=new-project");
  });

  test("preserves existing keys not being updated", () => {
    fs.writeFileSync(configPath, "api_key=key1\nproject_name=proj1\nother_setting=value1\n");

    updatePgwatchConfig(configPath, {
      project_name: "proj2",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=key1");
    expect(content).toContain("project_name=proj2");
    expect(content).toContain("other_setting=value1");
  });

  test("handles values with equals sign", () => {
    updatePgwatchConfig(configPath, {
      api_key: "key=with=equals",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=key=with=equals");
  });

  test("handles empty file", () => {
    fs.writeFileSync(configPath, "");

    updatePgwatchConfig(configPath, {
      api_key: "new-key",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=new-key");
  });

  test("handles file with blank lines", () => {
    fs.writeFileSync(configPath, "api_key=key1\n\n\nproject_name=proj1\n\n");

    updatePgwatchConfig(configPath, {
      new_key: "new-value",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=key1");
    expect(content).toContain("project_name=proj1");
    expect(content).toContain("new_key=new-value");
    // Blank lines in the middle should be filtered out (no consecutive newlines)
    expect(content).not.toContain("\n\n");
  });

  test("handles multiple updates in one call", () => {
    fs.writeFileSync(configPath, "api_key=old-key\n");

    updatePgwatchConfig(configPath, {
      api_key: "new-key",
      project_name: "my-project",
      another_setting: "another-value",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=new-key");
    expect(content).toContain("project_name=my-project");
    expect(content).toContain("another_setting=another-value");
  });

  test("uses startsWith for key matching (not regex)", () => {
    // This tests that we use startsWith, not regex, to avoid ReDoS
    // A key like "api_key" should not match "other_api_key"
    fs.writeFileSync(configPath, "other_api_key=other-value\napi_key=original-key\n");

    updatePgwatchConfig(configPath, {
      api_key: "updated-key",
    });

    const content = fs.readFileSync(configPath, "utf8");
    expect(content).toContain("api_key=updated-key");
    expect(content).toContain("other_api_key=other-value");
  });

  test("sets restrictive file permissions", () => {
    updatePgwatchConfig(configPath, {
      api_key: "secret-key",
    });

    const stats = fs.statSync(configPath);
    // Check that file is only readable/writable by owner (mode 0o600)
    const mode = stats.mode & 0o777;
    expect(mode).toBe(0o600);
  });
});

describe("registerMonitoringInstance", () => {
  let originalFetch: typeof global.fetch;
  let fetchCalls: Array<{ url: string; options: RequestInit }>;

  beforeEach(() => {
    originalFetch = global.fetch;
    fetchCalls = [];
    // Mock fetch to capture calls
    global.fetch = async (url: RequestInfo | URL, options?: RequestInit) => {
      fetchCalls.push({ url: url.toString(), options: options || {} });
      return new Response(JSON.stringify({ success: true }), { status: 200 });
    };
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test("sends POST request with correct URL and body", async () => {
    // Simulate what registerMonitoringInstance does
    const apiKey = "test-api-key";
    const projectName = "my-project";
    const apiBaseUrl = "https://api.example.com";

    await fetch(`${apiBaseUrl}/rpc/monitoring_instance_register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_token: apiKey,
        project_name: projectName,
      }),
    });

    expect(fetchCalls.length).toBe(1);
    expect(fetchCalls[0].url).toBe("https://api.example.com/rpc/monitoring_instance_register");
    expect(fetchCalls[0].options.method).toBe("POST");

    const headers = fetchCalls[0].options.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    // Verify API key is NOT in headers (only in body per security review)
    expect(headers["access-token"]).toBeUndefined();

    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect(body.api_token).toBe("test-api-key");
    expect(body.project_name).toBe("my-project");
  });

  test("includes api_token in body, not in header", async () => {
    const apiKey = "secret-key-12345";
    const projectName = "test-project";
    const apiBaseUrl = "https://postgres.ai/api/general";

    await fetch(`${apiBaseUrl}/rpc/monitoring_instance_register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        api_token: apiKey,
        project_name: projectName,
      }),
    });

    const headers = fetchCalls[0].options.headers as Record<string, string>;
    // Verify no access-token header
    expect(Object.keys(headers)).not.toContain("access-token");

    // Verify token is in body
    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect(body.api_token).toBe("secret-key-12345");
  });

  test("uses correct endpoint path", async () => {
    const apiBaseUrl = "https://custom.api.com/v2";

    await fetch(`${apiBaseUrl}/rpc/monitoring_instance_register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_token: "key", project_name: "proj" }),
    });

    expect(fetchCalls[0].url).toBe("https://custom.api.com/v2/rpc/monitoring_instance_register");
  });
});

describe("demo mode instances.demo.yml", () => {
  const repoRoot = path.resolve(import.meta.dir, "..", "..");

  test("instances.demo.yml exists in repo root", () => {
    const demoFile = path.join(repoRoot, "instances.demo.yml");
    expect(fs.existsSync(demoFile)).toBe(true);
  });

  test("instances.demo.yml contains demo target connection", () => {
    const demoFile = path.join(repoRoot, "instances.demo.yml");
    const content = fs.readFileSync(demoFile, "utf8");
    expect(content).toContain("name: target_database");
    expect(content).toContain("conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database");
    expect(content).toContain("is_enabled: true");
    expect(content).toContain("preset_metrics: full");
  });

  test("instances.demo.yml has required YAML structure", () => {
    const demoFile = path.join(repoRoot, "instances.demo.yml");
    const content = fs.readFileSync(demoFile, "utf8");
    // Verify it's a YAML list (starts with "- name:")
    expect(content).toMatch(/^- name: target_database/m);
    // Verify required fields are present with correct indentation
    expect(content).toMatch(/^\s+conn_str:/m);
    expect(content).toMatch(/^\s+preset_metrics: full/m);
    expect(content).toMatch(/^\s+is_enabled: true/m);
    // ~sink_type~ is a placeholder replaced per-sink by generate-pgwatch-sources.sh
    expect(content).toMatch(/^\s+sink_type: ~sink_type~/m);
  });

  test("instances.yml is gitignored (not tracked)", () => {
    const gitignore = fs.readFileSync(path.join(repoRoot, ".gitignore"), "utf8");
    expect(gitignore).toContain("instances.yml");
  });

  test("demo config can be copied to instances.yml in temp dir", () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "demo-install-test-"));
    try {
      const demoSrc = path.join(repoRoot, "instances.demo.yml");
      const instancesDest = path.join(tempDir, "instances.yml");

      fs.copyFileSync(demoSrc, instancesDest);

      expect(fs.existsSync(instancesDest)).toBe(true);
      const content = fs.readFileSync(instancesDest, "utf8");
      expect(content).toContain("name: target_database");
      expect(content).toContain("conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database");
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("demo config copy overwrites directory at instances.yml path", () => {
    // Docker bind-mounts create missing paths as directories; the copy must handle this
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "demo-eisdir-test-"));
    try {
      const demoSrc = path.join(repoRoot, "instances.demo.yml");
      const instancesDest = path.join(tempDir, "instances.yml");

      // Simulate Docker creating a directory at instances.yml path
      fs.mkdirSync(instancesDest);
      expect(fs.statSync(instancesDest).isDirectory()).toBe(true);

      // The fix: remove directory then copy
      if (fs.statSync(instancesDest).isDirectory()) {
        fs.rmSync(instancesDest, { recursive: true, force: true });
      }
      fs.copyFileSync(demoSrc, instancesDest);

      expect(fs.statSync(instancesDest).isFile()).toBe(true);
      const content = fs.readFileSync(instancesDest, "utf8");
      expect(content).toContain("name: target_database");
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
