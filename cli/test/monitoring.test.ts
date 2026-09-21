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
  warnIfTransactionPoolerPort,
  isTransactionPoolerUrl,
  isLaxSslmode,
  extractSslmode,
  InstancesParseError,
} from "../lib/instances";
import {
  registerMonitoringInstance,
  resolveAdoptedProject,
  planMonitoringRegistration,
  instanceIdToPersist,
  persistInstanceId,
  selfRegisterAndPersist,
  adoptAndPersist,
  statOwner,
  applyProjectName,
  applyApiKey,
  updatePgwatchConfig,
} from "../bin/postgres-ai";

/**
 * Test updatePgwatchConfig behaviour.
 *
 * The real function is imported rather than reimplemented here: a local copy
 * silently drifted from it once already (it never gained the chmod that keeps
 * the api_key-bearing file owner-only), so these tests passed while shipped
 * behaviour was untested.
 */

/**
 * No test may leave process.exitCode set.
 *
 * `bun test` exits with whatever process.exitCode was left behind, so a leak
 * here fails the job while every test passes -- which is exactly what happened:
 * 1661 pass, 0 fail, exit 1, and nothing red in the output to point at.
 * selfRegisterAndPersist sets it deliberately, so several tests below reach it.
 *
 * A root-level afterEach runs after each describe's own afterEach, so this
 * catches a leak from ANY test in the file, names the offender, and does not
 * depend on declaration order. Two earlier attempts at this did: a positional
 * check at the end of the file (blind to a later describe re-zeroing it) and a
 * count of `= 0` against `= exitBefore` in this file's own source (blind to
 * formatting, to the two being in different describes, and to anything
 * registered after it). Both stayed green against a deliberately leaking test.
 *
 * Note the restore must be `0`: in Bun, assigning `undefined` does NOT clear a
 * set exitCode, which is how the first fix leaked anyway.
 */
afterEach(() => {
  // Read, RESET, then assert. Without the reset one leak reddens every test
  // after it: removing the restores below gave 11 failures of which only 6
  // were genuine, the other 5 being innocent tests in later describes blamed
  // for someone else's leak. `bun test` shares one process, so across files it
  // would be worse still.
  const leaked = process.exitCode ?? 0;
  process.exitCode = 0;
  expect(leaked).toBe(0);
});

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
  // the platform adopts the provisioned instance and returns its real project
  // — the CLI sends NO project_name on the adopt path (the hardcoded
  // "postgres-ai-monitoring" default was removed).
  test("includes instance_id in body when adopting a provisioned instance", async () => {
    const instanceId = "019eb300-3f2a-7a75-b54d-4f10572b25b8";

    await registerMonitoringInstance("key", undefined, opts({ instanceId }));

    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect(body.instance_id).toBe(instanceId);
    // instance_id rides in the body next to api_token — never in headers.
    const headers = fetchCalls[0].options.headers as Record<string, string>;
    expect(headers["instance-id"]).toBeUndefined();
  });

  test("omits project_name from the body when undefined (adopt path)", async () => {
    await registerMonitoringInstance("key", undefined, opts({ instanceId: "i" }));

    const body = JSON.parse(fetchCalls[0].options.body as string);
    // The adopt path sends no name; the platform returns the real project.
    expect("project_name" in body).toBe(false);
    expect(body.api_token).toBe("key");
  });

  test("omits project_name from the body when empty/whitespace", async () => {
    await registerMonitoringInstance("key", "   ", opts({ instanceId: "i" }));

    const body = JSON.parse(fetchCalls[0].options.body as string);
    expect("project_name" in body).toBe(false);
  });

  test("omits instance_id from the body for legacy self-registration", async () => {
    await registerMonitoringInstance("key", "my-project", opts());

    const body = JSON.parse(fetchCalls[0].options.body as string);
    // PostgREST matches the 3-arg function via its default — the key must be
    // ABSENT (not null) so legacy CLIs and the new one hit the same overload.
    expect("instance_id" in body).toBe(false);
    // A real project name still rides in the body for legacy registration.
    expect(body.project_name).toBe("my-project");
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

  test("when adopting, a deterministic 4xx is NOT retried — returns null after one attempt", async () => {
    // Retry exists for transient 5xx/connection blips; a 4xx (bad
    // instance_id, conflict, auth) is deterministic and must fail fast.
    respond = () => new Response('{"message":"instance not found"}', { status: 404 });

    const reg = await registerMonitoringInstance("key", "p", opts({ instanceId: "i" }));

    expect(fetchCalls.length).toBe(1);
    expect(reg).toBeNull();
  });
});

describe("planMonitoringRegistration — mon local-install registration decision", () => {
  test("instance id present → adopt, no project name required", () => {
    const plan = planMonitoringRegistration({ instanceId: "i-123" });
    expect(plan.kind).toBe("adopt");
    expect(plan.projectName).toBeUndefined();
  });

  test("instance id present + project → adopt, carries the trimmed name", () => {
    const plan = planMonitoringRegistration({ instanceId: "i-123", project: "  prod-db  " });
    expect(plan.kind).toBe("adopt");
    expect(plan.projectName).toBe("prod-db");
  });

  test("no instance id + no project → error-missing-project (exit 1 path)", () => {
    const plan = planMonitoringRegistration({});
    expect(plan.kind).toBe("error-missing-project");
    expect(plan.projectName).toBeUndefined();
  });

  test("no instance id + empty/whitespace project → error-missing-project", () => {
    expect(planMonitoringRegistration({ project: "" }).kind).toBe("error-missing-project");
    expect(planMonitoringRegistration({ project: "   " }).kind).toBe("error-missing-project");
  });

  test("no instance id + real project → legacy self-register with the trimmed name", () => {
    const plan = planMonitoringRegistration({ project: "  my-project  " });
    expect(plan.kind).toBe("self-register");
    expect(plan.projectName).toBe("my-project");
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

// instances.yml carries password-bearing conn_strs, so every write must leave
// it owner-only (#353). Mode bits are meaningless on Windows, skip there.
describe.skipIf(process.platform === "win32")("instances.yml permissions (#353)", () => {
  let tempDir: string;
  let instancesFile: string;
  const modeOf = (p: string) => (fs.statSync(p).mode & 0o777).toString(8);

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "instances-mode-test-"));
    instancesFile = path.join(tempDir, "instances.yml");
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  test("add creates a missing file as 0600 regardless of umask", () => {
    const savedUmask = process.umask(0o000);
    try {
      addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    } finally {
      process.umask(savedUmask);
    }
    expect(modeOf(instancesFile)).toBe("600");
  });

  test("add tightens an existing world-readable file to 0600", () => {
    // writeFileSync's `mode` only applies on creation; a pre-existing loose
    // file must be chmod'ed explicitly.
    fs.writeFileSync(instancesFile, "[]\n", { mode: 0o644 });
    fs.chmodSync(instancesFile, 0o644);
    expect(modeOf(instancesFile)).toBe("644");

    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    expect(modeOf(instancesFile)).toBe("600");
    expect(loadInstances(instancesFile).map((i) => i.name)).toEqual(["t1"]);
  });

  test("remove of an unknown name leaves content and mode untouched", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    fs.chmodSync(instancesFile, 0o644);
    const before = fs.readFileSync(instancesFile, "utf8");

    expect(removeInstanceFromFile(instancesFile, "does-not-exist")).toBe(false);
    expect(fs.readFileSync(instancesFile, "utf8")).toBe(before);
    expect(modeOf(instancesFile)).toBe("644");
  });

  test("remove tightens an existing world-readable file to 0600", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    addInstanceToFile(instancesFile, buildInstance("t2", "postgresql://u:p@h:5432/db2"));
    fs.chmodSync(instancesFile, 0o644);
    expect(modeOf(instancesFile)).toBe("644");

    expect(removeInstanceFromFile(instancesFile, "t1")).toBe(true);
    expect(modeOf(instancesFile)).toBe("600");
    expect(loadInstances(instancesFile).map((i) => i.name)).toEqual(["t2"]);
  });

  test("add after replacing a directory at the target path is 0600", () => {
    fs.mkdirSync(instancesFile);
    const savedUmask = process.umask(0o000);
    try {
      addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    } finally {
      process.umask(savedUmask);
    }
    expect(modeOf(instancesFile)).toBe("600");
  });

  // A file owned by another UID but writable (root-created by a sudo install,
  // a bind mount, a CI image): the write succeeds and the chmod raises EPERM.
  // Tightening is best-effort hardening; it must not fail an add/remove that
  // already landed, or pgwatch's sources are never regenerated (#353).
  const eperm = () => {
    const err = new Error("EPERM: operation not permitted, chmod") as NodeJS.ErrnoException;
    err.code = "EPERM";
    throw err;
  };
  const failChmod = () => [
    spyOn(fs, "chmodSync").mockImplementation(eperm),
    spyOn(fs, "fchmodSync").mockImplementation(eperm),
  ];

  test("add survives a chmod it is not allowed to perform", () => {
    fs.writeFileSync(instancesFile, "[]\n", "utf8");
    const spies = failChmod();
    const errSpy = spyOn(console, "error").mockImplementation(() => {});
    try {
      expect(() =>
        addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db")),
      ).not.toThrow();
      expect(loadInstances(instancesFile).map((i) => i.name)).toEqual(["t1"]);
      expect(errSpy.mock.calls.flat().join("\n")).toMatch(/could not restrict permissions/);
    } finally {
      spies.forEach((s) => s.mockRestore());
      errSpy.mockRestore();
    }
  });

  test("remove survives it too", () => {
    addInstanceToFile(instancesFile, buildInstance("t1", "postgresql://u:p@h:5432/db"));
    const spies = failChmod();
    const errSpy = spyOn(console, "error").mockImplementation(() => {});
    try {
      expect(removeInstanceFromFile(instancesFile, "t1")).toBe(true);
      expect(loadInstances(instancesFile)).toEqual([]);
    } finally {
      spies.forEach((s) => s.mockRestore());
      errSpy.mockRestore();
    }
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

describe("isTransactionPoolerUrl — detects Supabase transaction-mode pooler", () => {
  test("true for a pooler host on the transaction port", () => {
    expect(
      isTransactionPoolerUrl(
        "postgresql://u.ref:p@aws-1-eu-west-1.pooler.supabase.com:6543/postgres",
      ),
    ).toBe(true);
  });

  test("false for the same pooler host on the session port", () => {
    expect(
      isTransactionPoolerUrl(
        "postgresql://u.ref:p@aws-1-eu-west-1.pooler.supabase.com:5432/postgres",
      ),
    ).toBe(false);
  });

  test("false for a direct host, even on 6543", () => {
    // A direct host runs no pooler; 6543 there is just an unusual port and
    // rewriting or warning about it would be wrong.
    expect(
      isTransactionPoolerUrl(
        "postgresql://postgres:p@db.abcdefghij.supabase.co:6543/postgres",
      ),
    ).toBe(false);
  });

  test("false for an unrelated host on 6543", () => {
    expect(
      isTransactionPoolerUrl("postgresql://u:p@db.example.com:6543/postgres"),
    ).toBe(false);
  });

  test("false for an unparseable connection string", () => {
    expect(isTransactionPoolerUrl("not-a-url")).toBe(false);
  });
});

describe("warnIfTransactionPoolerPort — UX warning for transaction pooling", () => {
  let stderrSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    stderrSpy = spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    stderrSpy.mockRestore();
  });

  test("warns, naming both ports and the failure it causes", () => {
    warnIfTransactionPoolerPort(
      "postgresql://u.ref:p@aws-1-eu-west-1.pooler.supabase.com:6543/postgres",
    );
    expect(stderrSpy).toHaveBeenCalledTimes(1);
    const msg = String(stderrSpy.mock.calls[0][0]);
    expect(msg).toContain("6543");
    expect(msg).toContain("5432");
    expect(msg).toContain("42P05");
  });

  test("does NOT warn on the session-mode port", () => {
    warnIfTransactionPoolerPort(
      "postgresql://u.ref:p@aws-1-eu-west-1.pooler.supabase.com:5432/postgres",
    );
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  test("does NOT warn on a direct host", () => {
    warnIfTransactionPoolerPort(
      "postgresql://postgres:p@db.abcdefghij.supabase.co:5432/postgres",
    );
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


describe("instanceIdToPersist — what reaches .pgwatch-config", () => {
  const ID = "11111111-1111-1111-1111-111111111111";
  const OTHER = "22222222-2222-2222-2222-222222222222";

  // The container bind-mounts .pgwatch-config and idles until instance_id is
  // there. Before this, a SELF-REGISTRATION discarded the id the platform
  // returned and only an --instance-id run ever persisted one, so the ordinary
  // path for a non-console-provisioned instance left the container with no
  // identity, permanently.
  test("a self-registration persists the id the platform just created", () => {
    expect(instanceIdToPersist({ instanceId: ID, projectId: 7, created: true })).toBe(ID);
  });

  test("the platform's id wins over the one this run asked to adopt", () => {
    // Authoritative: it is the row the platform actually adopted.
    expect(instanceIdToPersist({ instanceId: ID }, OTHER)).toBe(ID);
  });

  test("falls back to the requested id when the platform does not echo one", () => {
    expect(instanceIdToPersist({ projectId: 7 }, ID)).toBe(ID);
    expect(instanceIdToPersist(null, ID)).toBe(ID);
  });

  test("nothing usable means nothing is written", () => {
    expect(instanceIdToPersist(null, undefined)).toBeNull();
    expect(instanceIdToPersist({}, undefined)).toBeNull();
    expect(instanceIdToPersist({ instanceId: "" }, "")).toBeNull();
    expect(instanceIdToPersist({ instanceId: "   " }, undefined)).toBeNull();
  });

  test("a value that is not a uuid is refused, not written verbatim", () => {
    // It goes into a key=value file, so \r, \n or = could inject further
    // config keys — the same reason PROJECT_NAME_RE exists.
    for (const bad of [
      "not-a-uuid",
      `${ID}\napi_key=stolen`,
      `${ID}=x`,
      `${ID}\r\nproject_name=other`,
      "11111111-1111-1111-1111-11111111111",
    ]) {
      expect(instanceIdToPersist({ instanceId: bad }, undefined)).toBeNull();
    }
    // ...and a bad platform value does not shadow a good requested one.
    expect(instanceIdToPersist({ instanceId: "not-a-uuid" }, ID)).toBe(ID);
  });

  test("surrounding whitespace is trimmed rather than rejected", () => {
    expect(instanceIdToPersist({ instanceId: `  ${ID}  ` }, undefined)).toBe(ID);
  });
});

/**
 * The install writes the instance id where the container reads it.
 *
 * These drive `persistInstanceId` itself rather than its id-picking helper,
 * because the bug that shipped was not a wrong id -- it was no write at all.
 * A test of the helper alone stays green while the file is never touched.
 */
describe("persistInstanceId — what the install leaves on disk", () => {
  let dir: string;
  const configPath = () => path.join(dir, ".pgwatch-config");
  const read = () => fs.readFileSync(configPath(), "utf8");

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "persist-id-"));
    updatePgwatchConfig(configPath(), { api_key: "pai-token", project_name: "rig" });
  });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  test("self-registration persists the id the platform returned", () => {
    persistInstanceId(dir, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any, undefined);
    expect(read()).toContain("instance_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    // the keys the install already wrote survive the merge
    expect(read()).toContain("api_key=pai-token");
    expect(read()).toContain("project_name=rig");
  });

  test("an explicit --instance-id install still lands the id", () => {
    persistInstanceId(dir, null, "11111111-2222-3333-4444-555555555555");
    expect(read()).toContain("instance_id=11111111-2222-3333-4444-555555555555");
  });

  test("a re-run that registers nothing does not clobber a working id", () => {
    persistInstanceId(dir, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any, undefined);
    const before = read();

    // registration failed this time -- returns null, no id requested
    persistInstanceId(dir, null, undefined);
    expect(read()).toBe(before);
    expect(read()).toContain("instance_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");

    // and a reply carrying a blank id is just as inert
    persistInstanceId(dir, { instanceId: "" } as any, undefined);
    expect(read()).toBe(before);
  });

  test("the file stays owner-only", () => {
    persistInstanceId(dir, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any, undefined);
    expect(fs.statSync(configPath()).mode & 0o777).toBe(0o600);
  });

  test("a pre-existing world-readable config is narrowed, not left open", () => {
    // writeFileSync's `mode` applies only on creation, so on an install re-run
    // -- where the file already exists -- the chmod is the only thing keeping
    // the api_key out of other users' reach.
    fs.chmodSync(configPath(), 0o644);
    persistInstanceId(dir, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any, undefined);
    expect(fs.statSync(configPath()).mode & 0o777).toBe(0o600);
  });
});

/**
 * The self-registration step, driven directly.
 *
 * This is the step that carried the bug, and it used to be unreachable from a
 * test: it lived inline in a 600-line command action, so the only coverage
 * possible was grepping the source for the call. Grep coverage does not work
 * here -- an earlier version of these tests passed while BOTH call sites were
 * commented out, because `toContain` matches the text of a commented-out call.
 * `selfRegisterAndPersist` takes its registrar as a parameter so the real
 * sequence (register, persist, report) runs against a stub.
 */
describe("selfRegisterAndPersist", () => {
  let dir: string;
  // Stable arrays cleared in place, never reassigned: spyOn memoises the spy
  // per method, so a mockImplementation created in the first beforeEach keeps
  // closing over the array it captured then. Reassigning orphaned the captures
  // and every later assertion on `errored` saw an empty list.
  const logged: string[] = [];
  const errored: string[] = [];
  // Initialised at declaration, not in beforeEach: if beforeEach throws before
  // the save, afterEach would otherwise assign `undefined` over console.log and
  // poison every later test in the file.
  let logSpy: any = console.log;
  let errSpy: any = console.error;
  let exitBefore = 0;
  const configPath = () => path.join(dir, ".pgwatch-config");

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "self-reg-"));
    updatePgwatchConfig(configPath(), { api_key: "pai-token", project_name: "rig" });
    // Save the originals FIRST: if anything below throws, afterEach still has
    // real functions to put back rather than assigning undefined over console.
    logSpy = console.log;
    errSpy = console.error;
    logged.length = 0;
    errored.length = 0;
    // Plain assignment, not spyOn: spyOn memoises one spy per method, and this
    // file installs console.error spies in several describes. Sharing that spy
    // meant a later mockImplementation did not take effect and every assertion
    // on the captured output silently saw an empty list.
    console.log = (...a: any[]) => { logged.push(a.join(" ")); };
    console.error = (...a: any[]) => { errored.push(a.join(" ")); };
    // These helpers set process.exitCode on failure, which is process-wide and
    // would make `bun test` exit non-zero with every test passing. Note the
    // restore must be 0, not `undefined`: assigning undefined does NOT clear a
    // set exitCode, which is why the first attempt at this leaked anyway.
    exitBefore = process.exitCode ?? 0;
    process.exitCode = 0;
  });
  afterEach(() => {
    console.log = logSpy;
    console.error = errSpy;
    process.exitCode = exitBefore;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const stub = (reg: any) => (async () => reg) as any;

  test("persists the id the platform minted and says so", async () => {
    const reg = await selfRegisterAndPersist(dir, "pai-token", "rig", {},
      stub({ instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }));

    expect(fs.readFileSync(configPath(), "utf8"))
      .toContain("instance_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    expect(reg).not.toBeNull();
    expect(logged.join("\n")).toContain("Registered monitoring instance");
    expect(errored.join("\n")).toBe("");
  });

  test("passes the caller's options through to the registrar", async () => {
    let seen: any;
    await selfRegisterAndPersist(dir, "pai-token", "rig",
      { apiBaseUrl: "https://example.invalid", debug: true, orgScope: { kind: "alias", alias: "acme" } as any },
      (async (key: string, project: string, opts: any) => {
        seen = { key, project, opts };
        return { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" };
      }) as any);

    expect(seen.key).toBe("pai-token");
    expect(seen.project).toBe("rig");
    expect(seen.opts.apiBaseUrl).toBe("https://example.invalid");
    expect(seen.opts.debug).toBe(true);
    expect(seen.opts.orgScope).toEqual({ kind: "alias", alias: "acme" });
  });

  test("a failed registration is visible to the caller, not just the console", async () => {
    // The provisioning flow reads the exit status, not stdout. Exiting 0 here
    // reported a working box that has no identity.
    await selfRegisterAndPersist(dir, "pai-token", "rig", {}, stub(null));
    expect(process.exitCode).toBe(1);
  });

  test("a minted id is not thrown away when only the write failed", async () => {
    // Re-running with --project would register a SECOND instance and split the
    // health matrix, so the id and the safe flag must both be printed.
    fs.rmSync(configPath());
    fs.mkdirSync(configPath());
    await selfRegisterAndPersist(dir, "pai-token", "rig", {},
      stub({ instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }));

    const warning = errored.join("\n");
    expect(warning).toContain("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    expect(warning).toContain("--instance-id aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    // The advice that would create a duplicate must NOT be given here.
    expect(warning).not.toContain("--project rig");
  });

  test("a failed registration warns instead of passing silently", async () => {
    // The bug this replaces: registration returns null, nothing is written,
    // nothing is printed, and the install still reports success while the
    // container idles forever.
    await selfRegisterAndPersist(dir, "pai-token", "rig", {}, stub(null));

    expect(fs.readFileSync(configPath(), "utf8")).not.toContain("instance_id=");
    const warning = errored.join("\n");
    expect(warning).toContain("instance jobs will idle");
    // Nothing was minted, so re-registering by name is the right advice here.
    expect(warning).toContain("mon local-install --project rig");
    expect(logged.join("\n")).not.toContain("Registered monitoring instance");
  });

  test("a reply carrying no usable id warns too", async () => {
    // A 200 is not success for our purposes: what matters is whether an id
    // reached the file the container reads.
    for (const reply of [{}, { instanceId: "" }, { instanceId: "not-a-uuid" }]) {
      errored.length = 0;
      logged.length = 0;
      await selfRegisterAndPersist(dir, "pai-token", "rig", {}, stub(reply));
      expect(errored.join("\n")).toContain("instance jobs will idle");
      expect(logged.join("\n")).not.toContain("Registered monitoring instance");
    }
    expect(fs.readFileSync(configPath(), "utf8")).not.toContain("instance_id=");
  });

  test("nothing usable to write means no write at all", () => {
    const dir2 = fs.mkdtempSync(path.join(os.tmpdir(), "nowrite-"));
    try {
      const cfg = path.join(dir2, ".pgwatch-config");
      expect(persistInstanceId(dir2, null, undefined)).toBe(false);
      // An empty rewrite would create the file (and cost a chmod) for nothing.
      expect(fs.existsSync(cfg)).toBe(false);
    } finally {
      fs.rmSync(dir2, { recursive: true, force: true });
    }
  });

  test("the adopt path writes both keys in one guarded call", () => {
    // Two calls meant two read-modify-write cycles and, worse, the second was
    // unguarded: an unwritable config threw past the first call's catch and
    // killed the install with an unhandled rejection after the services were up.
    const dir2 = fs.mkdtempSync(path.join(os.tmpdir(), "adopt-"));
    try {
      const cfg = path.join(dir2, ".pgwatch-config");
      fs.mkdirSync(cfg); // the EISDIR case the guard exists for
      let threw = false;
      let ok: boolean | undefined;
      try {
        ok = persistInstanceId(dir2, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any,
          "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", { project_name: "adopted" });
      } catch { threw = true; }
      expect(threw).toBe(false);
      expect(ok).toBe(false);

      // And on a writable config, one call lands both keys.
      fs.rmdirSync(cfg);
      expect(persistInstanceId(dir2, { instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" } as any,
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", { project_name: "adopted" })).toBe(true);
      const written = fs.readFileSync(cfg, "utf8");
      expect(written).toContain("instance_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
      expect(written).toContain("project_name=adopted");
    } finally {
      fs.rmSync(dir2, { recursive: true, force: true });
    }
  });

  test("a duplicate key is not left behind for the other readers to find", () => {
    // The reporter greps `| head -n 1` and the Go loader now also takes the
    // first, but a stale line left below a rewritten one used to make them
    // disagree about which line was the credential.
    fs.writeFileSync(configPath(), "api_key=stale\napi_key=alsostale\n", { mode: 0o600 });
    updatePgwatchConfig(configPath(), { api_key: "fresh" });
    const lines = fs.readFileSync(configPath(), "utf8").split("\n").filter(Boolean);
    expect(lines.filter(l => l.startsWith("api_key="))).toEqual(["api_key=fresh"]);
  });

  test("a DOUBLED BOM does not turn a rewrite into a duplicate either", () => {
    // Not for the same reason as the single BOM, and the shipped comment said
    // otherwise until this was checked against both readers: at two or more
    // BOMs both the Go loader and the reporter's grep skip the BOM'd line and
    // AGREE, so there is no reader split -- the stale duplicate is merely
    // invisible to both. Stripping all of them is what lets the rewrite match,
    // so the dedupe filter drops the stale line and the file is normalised.
    fs.writeFileSync(configPath(), "\uFEFF\uFEFFapi_key=OLD\nproject_name=p\n", { mode: 0o600 });
    updatePgwatchConfig(configPath(), { api_key: "NEW" });

    const written = fs.readFileSync(configPath(), "utf8");
    expect(written.split("\n").filter(l => l.startsWith("api_key="))).toEqual(["api_key=NEW"]);
    expect(written).not.toContain("OLD");
    expect(written.charCodeAt(0)).not.toBe(0xfeff);
  });

  test("a path that exists but is not a regular file is refused, not written through", () => {
    // A FIFO blocks readFileSync forever, and narrowing the guard to skip the
    // READ only moves the hang to the write, which blocks on a FIFO just the
    // same. The install must not hang; persistInstanceId turns this into a
    // warning.
    const asDirectory = path.join(dir, "cfgdir");
    fs.mkdirSync(asDirectory);
    expect(() => updatePgwatchConfig(asDirectory, { api_key: "NEW" }))
      .toThrow(/not a regular file/);
  });

  test("a BOM does not turn a rewrite into a duplicate", () => {
    // The seam: the Go loader strips a leading BOM and takes the FIRST match;
    // the reporter's grep skips the BOM'd line and takes the next one. So if a
    // rewrite appends instead of replacing, the two readers pick different
    // lines as the credential and the box polls with a revoked token, PT401,
    // ten-minute backoff, forever. Reachable because appending a key by hand
    // is a documented step and a hand-edit can introduce a BOM.
    fs.writeFileSync(configPath(), "\uFEFFapi_key=OLD\nproject_name=p\n", { mode: 0o600 });
    updatePgwatchConfig(configPath(), { api_key: "NEW" });

    const written = fs.readFileSync(configPath(), "utf8");
    expect(written.split("\n").filter(l => l.startsWith("api_key="))).toEqual(["api_key=NEW"]);
    expect(written).not.toContain("OLD");
    // And the BOM is normalised away rather than left for the next writer.
    expect(written.charCodeAt(0)).not.toBe(0xfeff);
  });

  test("a re-run whose registration fails leaves a working id alone", async () => {
    await selfRegisterAndPersist(dir, "pai-token", "rig", {},
      stub({ instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }));
    const before = fs.readFileSync(configPath(), "utf8");

    await selfRegisterAndPersist(dir, "pai-token", "rig", {}, stub(null));
    expect(fs.readFileSync(configPath(), "utf8")).toBe(before);
  });

  test("an unwritable config warns and does not take the install down", async () => {
    // The services are already up by the time this runs, so a throw here would
    // become an unhandled rejection and skip the rest of the install.
    fs.rmSync(configPath());
    fs.mkdirSync(configPath()); // Docker creates a bind-mount target as a dir

    await selfRegisterAndPersist(dir, "pai-token", "rig", {},
      stub({ instanceId: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }));

    const warning = errored.join("\n");
    expect(warning).toContain("could not record the monitoring instance id");
    expect(warning).toContain("instance jobs will idle");
  });
});

/**
 * `mon update` must never self-register.
 *
 * A hand-run update on a box whose config has no instance_id must not create a
 * second monitoring instance -- that splits the health matrix between two rows
 * for one box. This is a structural assertion because the update action is not
 * callable from a test; comments are stripped first, so a commented-out call
 * cannot satisfy it (the failure mode that made the first version of these
 * guards useless).
 */
describe("mon update does not register", () => {
  const stripComments = (src: string) =>
    src
      .replace(/^\s*\/\/.*$/gm, "")
      .replace(/^[ \t]*\/\*[\s\S]*?\*\/[ \t]*$/gm, "");

  test("stripComments removes commented-out calls but not live code", () => {
    // Without this the helper could be replaced by the identity function and
    // every assertion that depends on it would still pass -- which was true
    // until this test existed. It also pins the bug the helper had: running
    // the block regex first, unanchored, let a `/*` inside a `//` comment open
    // a false comment that swallowed real code (measured: five live lines of
    // bin/postgres-ai.ts, because one comment mentions an `/api/v1/…/*` path).
    const fixture = [
      "  // await selfRegisterAndPersist(projectDir, apiKey);",
      "  /* persistInstanceId(projectDir, reg, instanceId); */",
      "  // see /api/v1/admin/tsdb/* for details",
      "  const LIVE_CODE = registerMonitoringInstance;",
      '  const url = "https://example.com/a";',
    ].join("\n");

    const out = stripComments(fixture);
    expect(out).not.toContain("await selfRegisterAndPersist");
    expect(out).not.toContain("persistInstanceId(projectDir");
    // ...while the live lines below the `/*`-bearing line survive.
    expect(out).toContain("const LIVE_CODE = registerMonitoringInstance;");
    expect(out).toContain('const url = "https://example.com/a";');
  });

  test("the update action body calls neither register nor persist", () => {
    const source = fs.readFileSync(
      path.resolve(import.meta.dir, "../bin/postgres-ai.ts"),
      "utf8",
    );
    const start = source.indexOf('.command("update")');
    expect(start).toBeGreaterThan(-1);
    const end = source.indexOf("\n  .command(", start + 1);
    expect(end).toBeGreaterThan(start);

    const updateAction = stripComments(source.slice(start, end));
    expect(updateAction.length).toBeGreaterThan(500); // not an empty slice
    for (const forbidden of [
      "registerMonitoringInstance",
      "selfRegisterAndPersist",
      "persistInstanceId",
      "monitoring_instance_register",
    ]) {
      expect(updateAction).not.toContain(forbidden);
    }
  });

  test("registration reaches the command only through the two persisting helpers", () => {
    const source = stripComments(
      fs.readFileSync(path.resolve(import.meta.dir, "../bin/postgres-ai.ts"), "utf8"),
    );
    const localInstall = source.indexOf('.command("local-install")');
    const localInstallEnd = source.indexOf("\n  .command(", localInstall + 1);
    expect(localInstall).toBeGreaterThan(-1);
    expect(localInstallEnd).toBeGreaterThan(localInstall);

    const region = source.slice(localInstall, localInstallEnd);

    // Exactly one call site each, and each REACHED by the branch it belongs
    // to. Pinning the condition alongside the call is what makes this more
    // than a text search: a mutation that left `await adoptAndPersist(` in
    // place but made the branch unreachable went green without it.
    expect(region).toMatch(/if \(instanceId\) \{\s*const reg = await adoptAndPersist\(/);
    expect(region).toMatch(/\} else \{\s*await selfRegisterAndPersist\(/);
    const adopts = [...source.matchAll(/await adoptAndPersist\(/g)].map(m => m.index!);
    const persists = [...source.matchAll(/await selfRegisterAndPersist\(/g)].map(m => m.index!);
    expect(adopts.length).toBe(1);
    expect(persists.length).toBe(1);

    // The command action must not call registerMonitoringInstance at all: the
    // helpers reach it through their injectable `register` parameter, and
    // going direct would skip the persist-then-report the helpers exist for.
    expect(region).not.toMatch(/registerMonitoringInstance\s*\(/);
    // Each helper still defaults to the real registrar. Scoped to its OWN
    // signature: an unbounded [\s\S]*? ran past adoptAndPersist and matched
    // selfRegisterAndPersist's parameter instead, so removing the first one's
    // default left the suite green.
    const signatureOf = (name: string) => {
      const at = source.indexOf(`async function ${name}(`);
      expect(at).toBeGreaterThan(-1);
      // Bounded by the NEXT function, not by "): Promise<": if only this
      // function's return annotation ever changed, that terminator would walk
      // forward into its sibling and the slice would span both -- the same
      // vacuity this assertion exists to fix, one step narrower.
      const nextFn = source.indexOf("\nasync function ", at + 1);
      const end = nextFn === -1 ? source.length : nextFn;
      const close = source.indexOf("): Promise<", at);
      expect(close).toBeGreaterThan(at);
      expect(close).toBeLessThan(end);
      return source.slice(at, close);
    };
    for (const name of ["adoptAndPersist", "selfRegisterAndPersist"]) {
      expect(signatureOf(name)).toContain(
        "register: typeof registerMonitoringInstance = registerMonitoringInstance",
      );
    }

    // Neither may be fire-and-forget. The regexes above require `await`, so a
    // `void`-ed call is invisible to them -- which is exactly the shape of the
    // original bug, and it went green here until this line was restored.
    expect(source).not.toMatch(/void\s+(registerMonitoringInstance|selfRegisterAndPersist)\(/);
    for (const at of [...adopts, ...persists]) {
      expect(at).toBeGreaterThan(localInstall);
      expect(at).toBeLessThan(localInstallEnd);
    }
  });


});

/**
 * The registration request is bounded.
 *
 * local-install now awaits this call, so an unanswered socket would block the
 * install indefinitely -- after the services are already up. Previously the
 * promise was `void`-ed, which hid the hang rather than preventing it.
 */
describe("registerMonitoringInstance is bounded", () => {
  let realFetch: typeof fetch;
  beforeEach(() => { realFetch = globalThis.fetch; });
  afterEach(() => { globalThis.fetch = realFetch; });

  test("the request carries an abort signal", async () => {
    let seen: RequestInit | undefined;
    globalThis.fetch = (async (_url: any, init: any) => {
      seen = init;
      return new Response(JSON.stringify({ instance_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }), {
        status: 200, headers: { "Content-Type": "application/json" },
      });
    }) as any;

    await registerMonitoringInstance("pai-token", "rig", { apiBaseUrl: "https://example.invalid" });
    expect(seen?.signal).toBeInstanceOf(AbortSignal);
    expect(seen?.signal?.aborted).toBe(false);
  });

  test("a server that never answers does not hang the caller", async () => {
    globalThis.fetch = ((_url: any, init: any) =>
      new Promise((_resolve, reject) => {
        // Behave like a real unanswered request: settle only when aborted.
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("timed out", "TimeoutError")));
      })) as any;

    // A small bound rather than the production default, so the suite does not
    // sleep for it. The mechanism under test is the same either way; that the
    // default is sane is asserted by the helper's own tests.
    const started = Date.now();
    const reg = await Promise.race([
      registerMonitoringInstance("pai-token", "rig", {
        apiBaseUrl: "https://example.invalid",
        timeoutMs: 100,
      }),
      new Promise((resolve) => setTimeout(() => resolve("HUNG"), 10_000)),
    ]);

    // Best-effort: a timeout is a failure like any other, so it returns null
    // rather than throwing -- the install continues and warns.
    expect(reg).toBeNull();
    expect(Date.now() - started).toBeLessThan(5_000);
  }, 20_000);
});

/**
 * The adopt step, driven directly.
 *
 * Extracted for the same reason as its self-registration sibling. A structural
 * assertion on the source could not tell a live call from a dead one: a
 * mutation that wrapped the call in `if (0)` left the suite green.
 */
describe("adoptAndPersist", () => {
  let dir: string;
  // Stable arrays cleared in place, never reassigned: spyOn memoises the spy
  // per method, so a mockImplementation created in the first beforeEach keeps
  // closing over the array it captured then. Reassigning orphaned the captures
  // and every later assertion on `errored` saw an empty list.
  const logged: string[] = [];
  const errored: string[] = [];
  // Initialised at declaration, not in beforeEach: if beforeEach throws before
  // the save, afterEach would otherwise assign `undefined` over console.log and
  // poison every later test in the file.
  let logSpy: any = console.log;
  let errSpy: any = console.error;
  let exitBefore = 0;
  const ID = "11111111-2222-3333-4444-555555555555";
  const configPath = () => path.join(dir, ".pgwatch-config");

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "adopt-p-"));
    updatePgwatchConfig(configPath(), { api_key: "pai-token" });
    // Save the originals FIRST: if anything below throws, afterEach still has
    // real functions to put back rather than assigning undefined over console.
    logSpy = console.log;
    errSpy = console.error;
    logged.length = 0;
    errored.length = 0;
    // Plain assignment, not spyOn: spyOn memoises one spy per method, and this
    // file installs console.error spies in several describes. Sharing that spy
    // meant a later mockImplementation did not take effect and every assertion
    // on the captured output silently saw an empty list.
    console.log = (...a: any[]) => { logged.push(a.join(" ")); };
    console.error = (...a: any[]) => { errored.push(a.join(" ")); };
    // These helpers set process.exitCode on failure, which is process-wide and
    // would make `bun test` exit non-zero with every test passing. Note the
    // restore must be 0, not `undefined`: assigning undefined does NOT clear a
    // set exitCode, which is why the first attempt at this leaked anyway.
    exitBefore = process.exitCode ?? 0;
    process.exitCode = 0;
  });
  afterEach(() => {
    console.log = logSpy;
    console.error = errSpy;
    process.exitCode = exitBefore;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const stub = (reg: any) => (async () => reg) as any;

  test("persists the id and the adopted project in one write", () => {
    return adoptAndPersist(dir, "pai-token", "fallback", ID, {},
      stub({ instanceId: ID, projectId: 7 })).then(() => {
      const written = fs.readFileSync(configPath(), "utf8");
      expect(written).toContain(`instance_id=${ID}`);
      expect(written).toContain("project_name=7");
      expect(logged.join("\n")).toContain("Adopted monitoring instance (project: 7)");
    });
  });

  test("passes the instance id to the registrar so the platform adopts", async () => {
    let seen: any;
    await adoptAndPersist(dir, "pai-token", "fallback", ID, { debug: true },
      (async (_k: string, _p: string, o: any) => { seen = o; return { instanceId: ID, projectId: 7 }; }) as any);
    expect(seen.instanceId).toBe(ID);
    expect(seen.debug).toBe(true);
  });

  test("a failed adoption warns, but still records the id the operator gave", async () => {
    await adoptAndPersist(dir, "pai-token", "fallback", ID, {}, stub(null));

    // The id came from --instance-id, so it names a row the console already
    // provisioned: it is worth recording even when this adoption call failed,
    // because the container can then poll as soon as the cause is cleared.
    expect(fs.readFileSync(configPath(), "utf8")).toContain(`instance_id=${ID}`);
    // But the operator must not be left thinking adoption succeeded.
    expect(errored.join("\n")).toContain(`Could not adopt provisioned instance ${ID}`);
    expect(logged.join("\n")).not.toContain("Adopted monitoring instance");
  });

  test("a reply with no project still persists the id", async () => {
    // Adoption succeeded; only the project is missing. The id is what the
    // container needs, so it must still land.
    await adoptAndPersist(dir, "pai-token", "fallback", ID, {}, stub({ instanceId: ID }));
    expect(fs.readFileSync(configPath(), "utf8")).toContain(`instance_id=${ID}`);
    expect(errored.join("\n")).toContain("returned no project");
  });

  test("an unwritable config warns instead of throwing past the guard", async () => {
    fs.rmSync(configPath());
    fs.mkdirSync(configPath());
    let threw = false;
    try {
      await adoptAndPersist(dir, "pai-token", "fallback", ID, {},
        stub({ instanceId: ID, projectId: 7 }));
    } catch { threw = true; }
    expect(threw).toBe(false);
    expect(errored.join("\n")).toContain("instance jobs will idle");
  });
});

/**
 * `.env` keys the installer derives from the box, rather than defaults.
 */
describe("statOwner", () => {
  let dir: string;
  beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), "owner-")); });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  test("reads the owner of a regular file", () => {
    const f = path.join(dir, ".pgwatch-config");
    fs.writeFileSync(f, "api_key=k\n", { mode: 0o600 });
    const st = fs.statSync(f);
    expect(statOwner(f)).toBe(`${st.uid}:${st.gid}`);
  });

  test("refuses a directory, so the container is never told to run as its owner", () => {
    // Docker creates a bind-mount target as a root-owned DIRECTORY when the
    // file is missing. Copying its owner produced INSTANCE_JOBS_USER=0:0 -- a
    // container asked to run as root to read a config that does not exist.
    const d = path.join(dir, ".pgwatch-config");
    fs.mkdirSync(d);
    expect(statOwner(d)).toBeNull();
  });

  test("distinguishes a missing file from one that is the wrong type", () => {
    // undefined = "no such file", where falling back to this process's owner is
    // right. null = "exists but is not a regular file", where there is no owner
    // worth copying AND no fallback: `?? processOwner()` is also 0:0 under
    // sudo, so collapsing the two would write INSTANCE_JOBS_USER=0:0 anyway.
    expect(statOwner(path.join(dir, "nope"))).toBeUndefined();
    expect(statOwner(null)).toBeUndefined();
    expect(statOwner(undefined)).toBeUndefined();

    const d = path.join(dir, "adir");
    fs.mkdirSync(d);
    expect(statOwner(d)).toBeNull();
  });

  test("follows a symlink to the target's owner, which is the owner that can read it", () => {
    const f = path.join(dir, "real");
    fs.writeFileSync(f, "api_key=k\n", { mode: 0o600 });
    const link = path.join(dir, "link");
    fs.symlinkSync(f, link);
    expect(statOwner(link)).toBe(statOwner(f));
  });
});

/**
 * The two step-1 writers of `.pgwatch-config`.
 *
 * Extracted from the local-install action for the same reason
 * persistInstanceId was: inline in a 600-line command action, the only
 * available coverage was grepping the source, which a dead-code mutation walks
 * straight through. Both run before anything is started, so both report and
 * stop rather than surfacing a stack trace.
 */
describe("applyProjectName / applyApiKey", () => {
  let dir: string;
  const logged: string[] = [];
  const errored: string[] = [];
  let logSpy!: typeof console.log;
  let errSpy!: typeof console.error;

  beforeEach(() => {
    logSpy = console.log;
    errSpy = console.error;
    logged.length = 0;
    errored.length = 0;
    console.log = (...a: any[]) => { logged.push(a.join(" ")); };
    console.error = (...a: any[]) => { errored.push(a.join(" ")); };
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "apply-"));
  });
  afterEach(() => {
    console.log = logSpy;
    console.error = errSpy;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const config = () => path.join(dir, ".pgwatch-config");

  test("the project name reaches the file the reporter reads", () => {
    expect(applyProjectName(dir, "acme")).toBe(true);
    expect(fs.readFileSync(config(), "utf8")).toContain("project_name=acme");
    expect(logged.join("\n")).toContain("Using project name: acme");
  });

  test("the api key reaches it too, at 0600, and is never echoed", () => {
    expect(applyApiKey(dir, "pai-token")).toBe(true);
    expect(fs.readFileSync(config(), "utf8")).toContain("api_key=pai-token");
    expect(fs.statSync(config()).mode & 0o777).toBe(0o600);
    expect(logged.join("\n")).toContain("API key saved");
    // The SUCCESS path prints too, and the leak test below only drives the
    // failure path -- appending the key to this log line stayed green.
    expect(logged.join("\n")).not.toContain("pai-token");
    expect(errored.join("\n")).not.toContain("pai-token");
  });

  // Separate cases rather than a loop, so a failure names which writer broke.
  test.each([
    ["applyProjectName", (d: string) => applyProjectName(d, "acme")],
    ["applyApiKey", (d: string) => applyApiKey(d, "pai-token")],
  ])("%s reports a stray directory in a sentence, not a stack trace", (_name, call) => {
    fs.mkdirSync(config());

    let threw = false;
    let ok: boolean | undefined;
    try { ok = call(dir); } catch { threw = true; }

    expect(threw).toBe(false);
    expect(ok).toBe(false);
    const said = errored.join("\n");
    expect(said).toContain("not a regular file");
    // The message names the path, so the operator knows what to remove.
    expect(said).toContain(config());
    expect(said).not.toContain("at <anonymous>"); // i.e. not a stack trace
  });

  test("the api key is never printed", () => {
    // It is the credential: a failure path must not echo it back.
    fs.mkdirSync(config());
    applyApiKey(dir, "pai-super-secret");
    expect(errored.join("\n")).not.toContain("pai-super-secret");
    expect(logged.join("\n")).not.toContain("pai-super-secret");
  });
});
