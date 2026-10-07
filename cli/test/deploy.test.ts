import { describe, test, expect } from "bun:test";
import {
  DBLAB_STEPS,
  StepView,
  estimateMonthlyCents,
  formatDuration,
  resolveSshKeys,
  splitDbUrl,
  watchDblabDeploy,
  type CloudInstance,
  type DeployOptions,
} from "../lib/deploy";

const options: DeployOptions = {
  create_options: ["cloud"],
  sizes: [
    { code: "S", vcpus: 4, ram_gib: 8, monthly_price_cents: 4900, available: true },
    { code: "M", vcpus: 8, ram_gib: 16, monthly_price_cents: 9900, available: true },
  ],
  disk_monthly_price_cents_per_gib: 15,
  min_disk_gib: 20,
  max_disk_gib: 1000,
  max_instances: 3,
  instances_used: 0,
  locations: ["fsn1"],
  ssh_keys: [
    { id: "k-1", name: "laptop" },
    { id: "k-2", name: "ci" },
    { id: "k-3", name: "ci" },
  ],
  is_admin: true,
};

describe("deploy helpers", () => {
  test("the estimate is the size price plus disk per GiB", () => {
    expect(estimateMonthlyCents(options, "s", 50)).toBe(4900 + 50 * 15);
    expect(estimateMonthlyCents(options, "XL", 50)).toBeNull();
  });

  test("ssh keys resolve by id or by a unique name", () => {
    expect(resolveSshKeys(options, ["laptop", "k-2"])).toEqual(["k-1", "k-2"]);
    expect(() => resolveSshKeys(options, ["ci"])).toThrow("More than one SSH key");
    expect(() => resolveSshKeys(options, ["nope"])).toThrow('No SSH key "nope"');
  });

  test("the db url is split like the Console sends it", () => {
    expect(splitDbUrl("postgresql://u:p%40ss@h:5432/d")).toEqual({ url: "postgresql://u@h:5432/d", password: "p@ss" });
    expect(() => splitDbUrl("postgresql://u@h/d")).toThrow("include the password");
    expect(() => splitDbUrl("mysql://u:p@h/d")).toThrow("postgresql://");
  });

  test("durations read naturally", () => {
    expect(formatDuration(4_000)).toBe("4s");
    expect(formatDuration(125_000)).toBe("2m 05s");
    expect(formatDuration(3_725_000)).toBe("1h 02m");
  });
});

describe("StepView", () => {
  test("plain mode prints one line per change, with durations, and no escape codes", () => {
    let t = 0;
    const out: string[] = [];
    const v = new StepView("Deploying", DBLAB_STEPS, false, (s) => out.push(s), () => t);
    v.start();
    v.advance("create_server");
    t = 30_000;
    v.advance("install_engine");
    t = 95_000;
    v.fail();
    v.stop();
    const text = out.join("");
    expect(text).toBe(
      "Deploying\nCreate server: started\nCreate server: done (30s)\nInstall DBLab Engine: started\nInstall DBLab Engine: FAILED (1m 05s)\n",
    );
    expect(text).not.toContain("\x1b");
  });

  test("a failure on a later step than the running one leaves no step spinning", () => {
    const v = new StepView("Deploying", DBLAB_STEPS, true, () => {}, () => 0);
    v.advance("connect_agents");
    v.fail("retrieve_data");
    const frame = v.render().join("\n");
    expect(frame).toMatch(/✓.*Connect DBLab and Joe to PostgresAI/);
    expect(frame).toMatch(/✗.*Copy data from the source/);
  });

  test("tty frame: done steps get a check, the running one a spinner, later ones stay pending", () => {
    let t = 0;
    const v = new StepView("Deploying", DBLAB_STEPS, true, () => {}, () => t);
    v.advance("create_server");
    t = 10_000;
    v.advance("install_joe");
    const frame = v.render().join("\n").replace(/\x1b\[[0-9;]*m/g, "");
    expect(frame).toContain("✓ Create server 10s");
    expect(frame).toContain("✓ Install DBLab Engine");
    expect(frame).toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏] Install Joe/);
    expect(frame).toContain("· Ready");
    expect(frame).toContain("elapsed 10s");
  });

  test("tty mode redraws in place instead of appending", () => {
    const out: string[] = [];
    const v = new StepView("Deploying", DBLAB_STEPS, true, (s) => out.push(s), () => 0);
    v.start();
    v.stop();
    const lines = DBLAB_STEPS.length + 2;
    expect(out.join("")).toContain(`\x1b[${lines}A`);
  });
});

describe("watchDblabDeploy", () => {
  const inst = (over: Partial<CloudInstance>): CloudInstance => ({
    id: 5, project_id: 1, project_name: "db", created_at: "", is_cloud: true,
    deploy_status: "installing", deploy_step: "install_engine", deploy_error: null, deploy_updated_at: null,
    size: "S", disk_gib: 50, location: "fsn1", server_ip: null, is_job_backed: false, joe_instance_id: null, ...over,
  });

  const withFetch = async (replies: CloudInstance[][], fn: () => Promise<void>) => {
    const orig = globalThis.fetch;
    let i = 0;
    globalThis.fetch = (async () =>
      new Response(JSON.stringify(replies[Math.min(i++, replies.length - 1)]), { status: 200 })) as unknown as typeof fetch;
    try {
      await fn();
    } finally {
      globalThis.fetch = orig;
    }
  };

  const api = { apiKey: "k", apiBaseUrl: "http://x/api/general" };
  const noSleep = async () => {};

  test("follows the steps to ready", async () => {
    await withFetch(
      [[inst({ deploy_step: "create_server", deploy_status: "launching" })], [inst({ deploy_step: "retrieve_data", deploy_status: "retrieving" })], [inst({ deploy_status: "ready", deploy_step: "ready" })]],
      async () => {
        const out: string[] = [];
        const v = new StepView("D", DBLAB_STEPS, false, (s) => out.push(s), () => 0);
        const r = await watchDblabDeploy(api, 5, v, { sleep: noSleep });
        expect(r.kind).toBe("ready");
        expect(out.join("")).toContain("Copy data from the source: done");
        expect(out.join("")).toContain("Ready: done");
      },
    );
  });

  test("a failure ends the watch with the platform's reason, on the failing step", async () => {
    await withFetch(
      [[inst({ deploy_step: "retrieve_data", deploy_status: "retrieving" })], [inst({ deploy_status: "destroying", deploy_step: "destroy_server", deploy_error: "pg_dump: refused" })]],
      async () => {
        const out: string[] = [];
        const v = new StepView("D", DBLAB_STEPS, false, (s) => out.push(s), () => 0);
        const r = await watchDblabDeploy(api, 5, v, { sleep: noSleep });
        expect(r.kind).toBe("failed");
        if (r.kind === "failed") expect(r.instance.deploy_error).toBe("pg_dump: refused");
        expect(out.join("")).toContain("Copy data from the source: FAILED");
      },
    );
  });

  test("transient errors are retried, then given up on", async () => {
    const orig = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response("down", { status: 503 });
    }) as unknown as typeof fetch;
    try {
      const v = new StepView("D", DBLAB_STEPS, false, () => {}, () => 0);
      await expect(watchDblabDeploy(api, 5, v, { sleep: noSleep, maxErrors: 3 })).rejects.toThrow();
      expect(calls).toBe(3);
    } finally {
      globalThis.fetch = orig;
    }
  });
});

describe("confirm", () => {
  test("asks on stderr, so stdout stays clean for --json and redirects", async () => {
    const { confirm } = await import("../lib/deploy-commands");
    const { Readable } = await import("stream");
    const out: string[] = [];
    const err: string[] = [];
    const o = process.stdout.write.bind(process.stdout);
    const e = process.stderr.write.bind(process.stderr);
    process.stdout.write = ((s: string) => (out.push(String(s)), true)) as typeof process.stdout.write;
    process.stderr.write = ((s: string) => (err.push(String(s)), true)) as typeof process.stderr.write;
    try {
      const yes = await confirm("Deploy? [y/N] ", { input: Readable.from(["y\n"]) });
      expect(yes).toBe(true);
    } finally {
      process.stdout.write = o;
      process.stderr.write = e;
    }
    expect(err.join("")).toContain("Deploy? [y/N]");
    expect(out.join("")).not.toContain("Deploy?");
  });
});
