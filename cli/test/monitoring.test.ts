import { describe, test, expect, beforeEach, afterEach, mock, spyOn } from "bun:test";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as yaml from "js-yaml";
import { Client } from "pg";
import {
  addInstanceToFile,
  removeInstanceFromFile,
  loadInstances,
  buildInstance,
  buildClientConfig,
  sslOptionFromConnString,
  warnIfLaxSslmode,
  isLaxSslmode,
  extractSslmode,
  InstancesParseError,
} from "../lib/instances";
import {
  registerMonitoringInstance,
  resolveAdoptedProject,
} from "../bin/postgres-ai";

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
  // Each test sets `respond(call) => Response`; defaults to a 200 with no body.
  let respond: (call: { url: string; options: RequestInit }) => Response;
  const apiBaseUrl = "https://api.example.com";
  const registerUrl = `${apiBaseUrl}/rpc/monitoring_instance_register`;

  beforeEach(() => {
    originalFetch = global.fetch;
    fetchCalls = [];
    respond = () => new Response(JSON.stringify({ project_id: 7 }), { status: 200 });
    // Mock fetch to capture calls and return the test-configured response.
    global.fetch = async (url: RequestInfo | URL, options?: RequestInit) => {
      const call = { url: url.toString(), options: options || {} };
      fetchCalls.push(call);
      return respond(call);
    };
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  // retryDelayMs: 0 keeps the retry path instant under test.
  const opts = (extra?: Record<string, unknown>) => ({ apiBaseUrl, retryDelayMs: 0, ...extra });

  test("posts api_token + project_name in the body (never in headers), legacy mode", async () => {
    await registerMonitoringInstance("secret-key-12345", "my-project", opts());

    expect(fetchCalls.length).toBe(1);
    expect(fetchCalls[0].url).toBe(registerUrl);
    expect(fetchCalls[0].options.method).toBe("POST");

    const headers = fetchCalls[0].options.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    // API key only in body, never in an access-token header (security review).
    expect(headers["access-token"]).toBeUndefined();

    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect(body.api_token).toBe("secret-key-12345");
    expect(body.project_name).toBe("my-project");
  });

  test("honors a custom apiBaseUrl for the endpoint path", async () => {
    await registerMonitoringInstance("key", "proj", opts({ apiBaseUrl: "https://custom.api.com/v2" }));
    expect(fetchCalls[0].url).toBe("https://custom.api.com/v2/rpc/monitoring_instance_register");
  });

  // Issue platform-all#311: console-provisioned installs pass instance_id so
  // the platform adopts the provisioned instance instead of self-registering
  // a duplicate under an auto-created "postgres-ai-monitoring" project.
  test("includes instance_id in body when adopting a provisioned instance", async () => {
    const instanceId = "019eb300-3f2a-7a75-b54d-4f10572b25b8";

    await registerMonitoringInstance("key", "postgres-ai-monitoring", opts({ instanceId }));

    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect(body.instance_id).toBe(instanceId);
    // instance_id rides in the body next to api_token — never in headers.
    const headers = fetchCalls[0].options.headers as Record<string, string>;
    expect(headers["instance-id"]).toBeUndefined();
  });

  test("omits instance_id from the body for legacy self-registration", async () => {
    await registerMonitoringInstance("key", "my-project", opts());

    const body = JSON.parse(fetchCalls[0].options.body as string);
    // PostgREST matches the 3-arg function via its default — the key must be
    // ABSENT (not null) so legacy CLIs and the new one hit the same overload.
    expect("instance_id" in body).toBe(false);
  });

  test("a 200 with {project_id, project_name} returns a populated result", async () => {
    respond = () => new Response(JSON.stringify({ project_id: 42, project_name: "prod-db", created: false }), { status: 200 });

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(reg).toEqual({ instanceId: undefined, projectId: 42, projectName: "prod-db", created: false });
  });

  test("a non-JSON 200 returns {} (success, but no fields) — not null", async () => {
    respond = () => new Response("<html>oops</html>", { status: 200 });

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(reg).toEqual({});
  });

  test("mistyped fields are dropped at runtime (project_id must be a number)", async () => {
    // A spoofed/older platform returning a string id must not poison the
    // persistence decision, which prefers a numeric project_id.
    respond = () => new Response(JSON.stringify({ project_id: "13", project_name: "ok-name", created: "yes" }), { status: 200 });

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(reg).toEqual({ instanceId: undefined, projectId: undefined, projectName: "ok-name", created: undefined });
  });

  test("when adopting, a first non-OK response triggers one retry that succeeds", async () => {
    respond = (call) => {
      // First attempt 503, second 200.
      return fetchCalls.length === 1
        ? new Response("upstream down", { status: 503 })
        : new Response(JSON.stringify({ project_id: 9 }), { status: 200 });
    };

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(fetchCalls.length).toBe(2);
    expect(reg).toEqual({ instanceId: undefined, projectId: 9, projectName: undefined, created: undefined });
  });

  test("when adopting, two failures exhaust the retry and return null", async () => {
    respond = () => new Response("upstream down", { status: 503 });

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(fetchCalls.length).toBe(2); // one initial + one retry
    expect(reg).toBeNull();
  });

  test("legacy mode does NOT retry — a single failure returns null after one attempt", async () => {
    respond = () => new Response("upstream down", { status: 503 });

    const reg = await registerMonitoringInstance("key", "p", opts());

    expect(fetchCalls.length).toBe(1);
    expect(reg).toBeNull();
  });
});

describe("resolveAdoptedProject — what gets persisted to .pgwatch-config", () => {
  test("prefers the numeric project_id over the name (survives renames)", () => {
    expect(resolveAdoptedProject({ projectId: 42, projectName: "prod-db" })).toBe("42");
  });

  test("project_id === 0 is a valid id and is honored", () => {
    expect(resolveAdoptedProject({ projectId: 0, projectName: "prod-db" })).toBe("0");
  });

  test("falls back to project_name when there is no id", () => {
    expect(resolveAdoptedProject({ projectName: "prod-db" })).toBe("prod-db");
  });

  test("returns null for a null response", () => {
    expect(resolveAdoptedProject(null)).toBeNull();
  });

  test("returns null for a fieldless (succeeded-but-empty) response", () => {
    expect(resolveAdoptedProject({})).toBeNull();
  });

  test("rejects a project_name with a newline (config-file injection, CWE-93)", () => {
    expect(resolveAdoptedProject({ projectName: "prod\ninjected=evil" })).toBeNull();
  });

  test("rejects a project_name containing '='", () => {
    expect(resolveAdoptedProject({ projectName: "a=b" })).toBeNull();
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
    // ~sink_type~ is a sed token substituted by generate-pgwatch-sources.sh; values: postgres, prometheus
    expect(content).toMatch(/^\s+sink_type: ~sink_type~/m);
  });

  test("instances.yml is gitignored (not tracked)", () => {
    const gitignore = fs.readFileSync(path.join(repoRoot, ".gitignore"), "utf8");
    expect(gitignore).toMatch(/^instances\.yml$/m);
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

describe("docker-compose: default network has IPv6 enabled", () => {
  const repoRoot = path.resolve(import.meta.dir, "..", "..");

  test("root docker-compose.yml declares networks.default with enable_ipv6", () => {
    const composePath = path.join(repoRoot, "docker-compose.yml");
    const content = fs.readFileSync(composePath, "utf8");

    // Use string match so we also catch the env-overridable form
    // `enable_ipv6: ${PGAI_ENABLE_IPV6:-true}`, which Compose interpolates
    // before the YAML strict-bool check.
    expect(content).toMatch(/^networks:/m);
    expect(content).toMatch(/^\s*default:/m);
    expect(content).toMatch(/enable_ipv6:\s*(true|\$\{PGAI_ENABLE_IPV6:-true\})/);

    // Also assert the YAML is parseable (catches indentation regressions).
    const parsed = yaml.load(content) as any;
    expect(parsed?.networks?.default).toBeDefined();
  });

  test("env override default resolves to 'true' when PGAI_ENABLE_IPV6 is unset", () => {
    // Mirror Compose's `${VAR:-default}` interpolation rule. Verifies that the
    // template's default value matches the documented one.
    const composePath = path.join(repoRoot, "docker-compose.yml");
    const content = fs.readFileSync(composePath, "utf8");
    const m = content.match(/enable_ipv6:\s*\$\{PGAI_ENABLE_IPV6:-(\w+)\}/);
    expect(m).not.toBeNull();
    expect(m![1]).toBe("true");
  });
});

describe("addInstanceToFile / removeInstanceFromFile round-trip", () => {
  let tempDir: string;
  let instancesFile: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "instances-test-"));
    instancesFile = path.join(tempDir, "instances.yml");
  });

  afterEach(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("add to empty file produces a valid YAML list", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    const list = loadInstances(instancesFile);
    expect(list.length).toBe(1);
    expect(list[0].name).toBe("t1");
  });

  test("add → remove → add cycle keeps file parseable (regression)", () => {
    // The previous bug: after `remove` left `[]` in the file, `add` appended a
    // list-item next to it, producing two YAML documents in one file.
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    removeInstanceFromFile(instancesFile, "t1");
    expect(loadInstances(instancesFile)).toEqual([]);

    addInstanceToFile(instancesFile, buildInstance("t2", "postgresql://u:p@h:5432/db2"));

    // Must NOT throw "end of the stream or a document separator is expected".
    const list = loadInstances(instancesFile);
    expect(list.length).toBe(1);
    expect(list[0].name).toBe("t2");
  });

  test("sink_type placeholder survives the round-trip", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    const content = fs.readFileSync(instancesFile, "utf8");
    // js-yaml emits ~sink_type~ unquoted (it's only special when standalone);
    // sed s/~sink_type~/.../g still hits it as raw text regardless.
    expect(content).toContain("~sink_type~");
  });

  test("add throws InstancesParseError on a corrupted file (no silent overwrite)", () => {
    // Silent overwrite would discard credentials in conn_str values. Refuse.
    fs.writeFileSync(instancesFile, "key: [unclosed\nfoo: bar\n", "utf8");

    expect(() =>
      addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db")),
    ).toThrow(InstancesParseError);

    // File contents are unchanged.
    expect(fs.readFileSync(instancesFile, "utf8")).toBe("key: [unclosed\nfoo: bar\n");
  });

  test("add rejects duplicate name", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    expect(() =>
      addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db")),
    ).toThrow(/already exists/);
  });

  test("add replaces a directory at the target path (Docker bind-mount artifact)", () => {
    fs.mkdirSync(instancesFile);
    expect(fs.statSync(instancesFile).isDirectory()).toBe(true);
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    expect(fs.statSync(instancesFile).isFile()).toBe(true);
    expect(loadInstances(instancesFile).length).toBe(1);
  });
});

describe("sslOptionFromConnString — libpq semantics", () => {
  test("sslmode=require → SSL without chain verification", () => {
    expect(sslOptionFromConnString("postgresql://u:p@h/db?sslmode=require"))
      .toEqual({ rejectUnauthorized: false });
  });

  test("sslmode unset → SSL without chain verification", () => {
    expect(sslOptionFromConnString("postgresql://u:p@h/db"))
      .toEqual({ rejectUnauthorized: false });
  });

  test("sslmode=disable → no SSL", () => {
    expect(sslOptionFromConnString("postgresql://u:p@h/db?sslmode=disable")).toBe(false);
  });

  test("sslmode=verify-ca → chain verification, no hostname check", () => {
    const opt = sslOptionFromConnString("postgresql://u:p@h/db?sslmode=verify-ca");
    expect(opt).toMatchObject({ rejectUnauthorized: true });
    expect(typeof (opt as any).checkServerIdentity).toBe("function");
  });

  test("sslmode=verify-full → chain + hostname verification", () => {
    expect(sslOptionFromConnString("postgresql://u:p@h/db?sslmode=verify-full"))
      .toEqual({ rejectUnauthorized: true });
  });

  test("sslmode=prefer → SSL without chain verification", () => {
    expect(sslOptionFromConnString("postgresql://u:p@h/db?sslmode=prefer"))
      .toEqual({ rejectUnauthorized: false });
  });

  test("malformed connection string → safe default (no chain verification)", () => {
    expect(sslOptionFromConnString("not-a-url")).toEqual({ rejectUnauthorized: false });
  });
});

describe("buildClientConfig — actual node-postgres Client gets the intended ssl (regression)", () => {
  // The previous code passed `{ connectionString, ssl }` to `new Client(...)`.
  // node-postgres' ConnectionParameters internally does
  // `Object.assign({}, config, parse(connectionString))`, so the parsed
  // `connectionString.ssl` REPLACES the explicit `ssl`. Net effect:
  //   `?sslmode=require` + `ssl: { rejectUnauthorized: false }` → `ssl: {}`
  //                                                              (chain verified)
  // — exactly the bug the MR claims to fix. This integration test asserts
  // against `client.connectionParameters.ssl`, so the bug cannot return.

  test("require: actual Client.connectionParameters.ssl has rejectUnauthorized:false", () => {
    const c = new Client(buildClientConfig("postgresql://u:p@h/db?sslmode=require"));
    expect(c.connectionParameters.ssl).toEqual({ rejectUnauthorized: false });
  });

  test("verify-full: actual Client.connectionParameters.ssl has rejectUnauthorized:true", () => {
    const c = new Client(buildClientConfig("postgresql://u:p@h/db?sslmode=verify-full"));
    expect(c.connectionParameters.ssl).toEqual({ rejectUnauthorized: true });
  });

  test("disable: actual Client.connectionParameters.ssl is false", () => {
    const c = new Client(buildClientConfig("postgresql://u:p@h/db?sslmode=disable"));
    expect(c.connectionParameters.ssl).toBe(false);
  });

  test("unset: actual Client.connectionParameters.ssl has rejectUnauthorized:false", () => {
    const c = new Client(buildClientConfig("postgresql://u:p@h/db"));
    expect(c.connectionParameters.ssl).toEqual({ rejectUnauthorized: false });
  });

  test("connectionTimeoutMillis is forwarded (exact value, not just truthy)", () => {
    const c = new Client(buildClientConfig("postgresql://u:p@h/db?sslmode=require", { connectionTimeoutMillis: 5000 }));
    // node-postgres v8 stores it on `_connectionTimeoutMillis`. Asserting the
    // exact value catches regressions that would silently swap in the default.
    expect((c as any)._connectionTimeoutMillis).toBe(5000);
  });
});

describe("warnIfLaxSslmode — UX warning for lax sslmode", () => {
  let stderrSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    stderrSpy = spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    stderrSpy.mockRestore();
  });

  for (const sslmode of ["require", "prefer", "allow"]) {
    test(`warns when sslmode=${sslmode}`, () => {
      warnIfLaxSslmode(`postgresql://u:p@h/db?sslmode=${sslmode}`);
      expect(stderrSpy).toHaveBeenCalledTimes(1);
      const msg = String(stderrSpy.mock.calls[0][0]);
      expect(msg).toContain(`sslmode=${sslmode}`);
      expect(msg).toContain("NOT verified");
      expect(msg).toContain("verify-full");
    });
  }

  test("warns when sslmode is unset (uses '(unset)' label)", () => {
    warnIfLaxSslmode("postgresql://u:p@h/db");
    expect(stderrSpy).toHaveBeenCalledTimes(1);
    expect(String(stderrSpy.mock.calls[0][0])).toContain("sslmode=(unset)");
  });

  test("does NOT warn when sslmode=verify-full or verify-ca or disable", () => {
    warnIfLaxSslmode("postgresql://u:p@h/db?sslmode=verify-full");
    warnIfLaxSslmode("postgresql://u:p@h/db?sslmode=verify-ca");
    warnIfLaxSslmode("postgresql://u:p@h/db?sslmode=disable");
    expect(stderrSpy).not.toHaveBeenCalled();
  });
});

describe("buildClientConfig — silences pg-connection-string deprecation warning", () => {
  // pg-connection-string v2.x prints `process.emitWarning("SECURITY WARNING:
  // ... 'prefer'/'require'/'verify-ca' ...")` whenever a recognised lax
  // sslmode appears. Without our `uselibpqcompat=true` shim, this would fire
  // on every CLI invocation against a Supabase-shaped URL. Assert it doesn't.

  let warnings: string[];
  let origEmitWarning: typeof process.emitWarning;

  beforeEach(() => {
    warnings = [];
    origEmitWarning = process.emitWarning;
    (process as any).emitWarning = (warning: any) => {
      warnings.push(typeof warning === "string" ? warning : String(warning));
    };
  });

  afterEach(() => {
    (process as any).emitWarning = origEmitWarning;
  });

  for (const sslmode of ["require", "prefer", "verify-ca"]) {
    test(`no SECURITY WARNING for sslmode=${sslmode}`, () => {
      buildClientConfig(`postgresql://u:p@h/db?sslmode=${sslmode}`);
      const security = warnings.filter((w) => w.includes("SECURITY"));
      expect(security).toEqual([]);
    });
  }
});

describe("extractSslmode / isLaxSslmode", () => {
  test("extractSslmode returns lowercase", () => {
    expect(extractSslmode("postgresql://u:p@h/db?sslmode=REQUIRE")).toBe("require");
  });

  test("extractSslmode returns '' for unparseable URLs", () => {
    expect(extractSslmode("not-a-url")).toBe("");
  });

  test("isLaxSslmode covers the full set", () => {
    expect(isLaxSslmode("")).toBe(true);
    expect(isLaxSslmode("require")).toBe(true);
    expect(isLaxSslmode("prefer")).toBe(true);
    expect(isLaxSslmode("allow")).toBe(true);
    expect(isLaxSslmode("verify-ca")).toBe(false);
    expect(isLaxSslmode("verify-full")).toBe(false);
    expect(isLaxSslmode("disable")).toBe(false);
  });
});
