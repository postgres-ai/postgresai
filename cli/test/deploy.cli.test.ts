import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

// Async spawn: the fake platform runs in this process (see dblab.cli.test.ts).
async function runCli(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin = typeof process.execPath === "string" && process.execPath.length > 0 ? process.execPath : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    env: { ...process.env, CI: "1", PGAI_DEPLOY_POLL_MS: "10", ...env },
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
  });
  const [status, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { status, stdout, stderr };
}

type Body = Record<string, unknown>;

// The platform functions' argument names (db/api/v1/functions/*.sql in platform-all).
const RPC_ARGS: Record<string, string[]> = {
  dblab_cloud_deploy_options: ["org_id"],
  dblab_instance_deploy: ["org_id", "name", "db_url", "size", "disk_gib", "ssh_key_ids", "location"],
  dblab_cloud_instances: ["org_id", "p_instance_id"],
  dblab_instance_destroy: ["instance_id"],
  monitoring_instance_deploy_status: ["p_instance_id"],
  monitoring_instance_create: [
    "db_url", "db_pass", "org_id", "grafana_access", "billing_mode", "provision", "server_location", "server_image",
    "server_type", "ssh_login_user", "prepare_monitoring_db", "project_name", "software_plan", "ssh_key_ids",
    "si_jwt_token", "cloud_hcloud_api_token", "cloud_aws_access_key_id", "cloud_aws_secret_access_key",
    "cloud_gcp_credentials", "postgresai_version", "in_promo_code", "in_vcpus",
  ],
};

/** Fake platform for the deploy rpcs. `progress` is served one entry per poll. */
async function startFakeApi(opts: {
  progress?: Body[];
  deployError?: { status: number; body: Body };
  monProgress?: Body[];
  monCreate?: Body;
  options?: Body;
} = {}) {
  const calls: { fn: string; body: Body; headers: Record<string, string> }[] = [];
  let poll = 0;
  let monPoll = 0;
  const progress = opts.progress ?? [
    { deploy_status: "launching", deploy_step: "create_server" },
    { deploy_status: "installing", deploy_step: "install_joe", server_ip: "203.0.113.9" },
    { deploy_status: "ready", deploy_step: "ready", server_ip: "203.0.113.9" },
  ];
  const json = (b: unknown, status = 200) =>
    new Response(JSON.stringify(b), { status, headers: { "Content-Type": "application/json" } });

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const fn = url.pathname.split("/rpc/")[1] ?? "";
      const body = (req.method === "POST" ? await req.json().catch(() => ({})) : {}) as Body;
      calls.push({ fn, body, headers: Object.fromEntries(req.headers.entries()) });
      // Like PostgREST: a body whose keys are not the function's arguments is a 404.
      const allowed = RPC_ARGS[fn];
      if (allowed && Object.keys(body).some((k) => !allowed.includes(k))) {
        return json({ code: "PGRST202", message: `Could not find the v1.${fn}(${Object.keys(body).sort().join(", ")}) function` }, 404);
      }
      switch (fn) {
        case "dblab_cloud_deploy_options":
          return json({
            create_options: ["cloud"],
            sizes: [{ code: "S", vcpus: 4, ram_gib: 8, monthly_price_cents: 4900, available: true }],
            disk_monthly_price_cents_per_gib: 15, min_disk_gib: 20, max_disk_gib: 1000,
            max_instances: 3, instances_used: 0, locations: ["fsn1"],
            ssh_keys: [{ id: "0199-key", name: "laptop" }], is_admin: true,
            has_payment_method: true, org_alias: "acme", ...opts.options,
          });
        case "dblab_instance_deploy":
          if (opts.deployError) return json(opts.deployError.body, opts.deployError.status);
          return json({ id: 41, project_id: 9, project_name: body.name, deploy_status: "launching", monthly_price_cents: 5650 });
        case "dblab_cloud_instances": {
          const p = progress[Math.min(poll++, progress.length - 1)];
          return json([{
            id: 41, project_id: 9, project_name: "app-db", created_at: "", is_cloud: true,
            deploy_error: null, deploy_updated_at: null, size: "S", disk_gib: 50, location: "fsn1",
            server_ip: null, is_job_backed: false, joe_instance_id: null, ...p,
          }]);
        }
        case "dblab_instance_destroy":
          return json({ result: body.instance_id, deploy_status: "destroying" });
        case "monitoring_instance_create":
          // SI's reply as raw_http_request wraps it.
          return json(opts.monCreate ?? { id: "0199-mon", response: { status_code: 200, content: { taskID: "t", otCode: "o" } } });
        case "monitoring_instance_deploy_status": {
          const seq = opts.monProgress ?? [{ status: "launch_requested" }, { status: "active", grafana_url: "https://g.example/" }];
          return json({ id: "0199-mon", project_name: "app", error: null, failed_task: null, grafana_url: null, ...seq[Math.min(monPoll++, seq.length - 1)] });
        }
      }
      return new Response("not found", { status: 404 });
    },
  });
  const env = {
    XDG_CONFIG_HOME: mkdtempSync(resolve(tmpdir(), "pgai-deploy-test-")),
    PGAI_API_KEY: "test-key",
    PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}/api/general`,
  };
  return { env, calls, fn: (name: string) => calls.filter((c) => c.fn === name), stop: () => server.stop(true) };
}

const DEPLOY = ["dblab", "deploy", "--name", "app-db", "--db-url", "postgresql://u:secret@db.example:5432/app", "--ssh-key", "laptop"];

describe("pgai dblab deploy", () => {
  // As `pgai connect` does: with no card, nothing is asked or created; the user gets the billing page, exit 3.
  test("no card on file: stops before the prompt with the billing page, exit 3", async () => {
    const api = await startFakeApi({ options: { has_payment_method: false } });
    try {
      const r = await runCli(DEPLOY, { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(r.stderr).toContain("Add a payment method at https://console.example/acme/billing, then re-run.");
      expect(r.stderr).not.toContain("Pass --yes");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("no card on file, --json: an action_required object on stdout, exit 3", async () => {
    const api = await startFakeApi({ options: { has_payment_method: false } });
    try {
      const r = await runCli([...DEPLOY, "--yes", "--json"], { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(JSON.parse(r.stdout)).toEqual({
        status: "action_required", next: "Add a payment method at https://console.example/acme/billing, then re-run.",
      });
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("the platform's own no-card refusal (PT402) reads the same, exit 3", async () => {
    const api = await startFakeApi({
      options: { has_payment_method: null },
      deployError: { status: 402, body: { code: "PT402", details: "Add a payment method to your organization first.",
        hint: "Open Billing in the Console and add a card." } },
    });
    try {
      const r = await runCli([...DEPLOY, "--yes"], { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(r.stderr).toContain("Add a payment method at https://console.example/acme/billing, then re-run.");
    } finally {
      api.stop();
    }
  });

  test("deploys, follows the steps to ready, and prints plain lines when piped", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.stderr).toBe("");
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("Estimated cost: $56.50/month (size S, 50 GiB), prorated.");
      expect(r.stdout).toContain("Create server: done");
      expect(r.stdout).toContain("Ready: done");
      expect(r.stdout).toContain('DBLab "app-db" (id 41) is ready.');
      // A clone needs a DB user and password; the hint is a command that works as printed.
      expect(r.stdout).toContain(
        "Create a clone: PGAI_CLONE_DB_PASSWORD=<password> pgai dblab clone create --project app-db --db-user <user>");
      expect(r.stdout).toContain("ssh -N -L 6000:127.0.0.1:6000 root@203.0.113.9");
      expect(r.stdout).not.toContain("\x1b");
      const body = api.fn("dblab_instance_deploy")[0].body;
      expect(body).toEqual({
        name: "app-db", db_url: "postgresql://u:secret@db.example:5432/app", size: "S", disk_gib: 50, ssh_key_ids: ["0199-key"],
      });
      expect(api.fn("dblab_instance_deploy")[0].headers["access-token"]).toBe("test-key");
    } finally {
      api.stop();
    }
  });

  test("a failed deploy exits 1 with the platform's reason", async () => {
    const api = await startFakeApi({
      progress: [
        { deploy_status: "retrieving", deploy_step: "retrieve_data" },
        { deploy_status: "destroying", deploy_step: "retrieve_data", deploy_error: "Copying data from the source database failed: connection refused" },
      ],
    });
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(1);
      expect(r.stdout).toContain("Copy data from the source: FAILED");
      expect(r.stderr).toContain('Deploy failed at "Copy data from the source": Copying data from the source database failed: connection refused');
      expect(r.stderr).toContain("you are not charged");
    } finally {
      api.stop();
    }
  });

  test("refuses without an SSH key, naming the org's keys", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "deploy", "--name", "x1", "--db-url", "postgresql://u:p@h/d", "--yes"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("--ssh-key is required");
      expect(r.stderr).toContain("Org keys: laptop.");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("asks for --yes when it cannot ask", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(DEPLOY, api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Pass --yes to confirm the monthly cost");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("--no-wait returns the id and how to follow it", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes", "--no-wait"], api.env);
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("Deploy started: id 41. Follow it: pgai dblab instances watch 41");
      expect(api.fn("dblab_cloud_instances")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab deploy: output contracts", () => {
  test("--json: stdout is exactly one JSON document", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes", "--json"], api.env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout).kind).toBe("ready");
    } finally {
      api.stop();
    }
  });

  test("a refusal shows the platform's words, not the HTTP status", async () => {
    const api = await startFakeApi({
      deployError: { status: 409, body: { code: "PT409", message: "Conflict", details: 'A DBLab named "app-db" already exists.', hint: "Choose another name." } },
    });
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain('A DBLab named "app-db" already exists.');
      expect(r.stderr).toContain("Hint: Choose another name.");
      expect(r.stderr).not.toContain("HTTP 409");
    } finally {
      api.stop();
    }
  });

  test("PGPASSWORD fills a URL without a password", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "deploy", "--name", "app-db", "--db-url", "postgresql://u@db.example:5432/app", "--ssh-key", "laptop", "--yes", "--no-wait"], { ...api.env, PGPASSWORD: "s3cr@t" });
      expect(r.status).toBe(0);
      expect(api.fn("dblab_instance_deploy")[0].body.db_url).toBe("postgresql://u:s3cr%40t@db.example:5432/app");
    } finally {
      api.stop();
    }
  });
});

describe("REV round 1 (!459)", () => {
  test("--debug never prints the source password (in the URL or from PGPASSWORD)", async () => {
    const api = await startFakeApi();
    try {
      const a = await runCli([...DEPLOY, "--yes", "--no-wait", "--debug"], api.env);
      expect(a.status).toBe(0);
      expect(a.stderr).toContain("Debug: body");
      expect(a.stderr).not.toContain("secret");
      const b = await runCli(["dblab", "deploy", "--name", "app-db", "--db-url", "postgresql://u@db.example:5432/app", "--ssh-key", "laptop", "--yes", "--no-wait", "--debug"], { ...api.env, PGPASSWORD: "Sup3rS3cretPW" });
      expect(b.status).toBe(0);
      expect(b.stderr).not.toContain("Sup3rS3cretPW");
      // Unencoded '/' or '@' in the password (generated passwords have them).
      for (const url of ["postgresql://u:ab/cdSECRET@db.example:5432/app", "postgresql://u:p@ssSECRET@db.example:5432/app"]) {
        const c = await runCli(["dblab", "deploy", "--name", "app-db", "--db-url", url, "--ssh-key", "laptop", "--yes", "--no-wait", "--debug"], api.env);
        expect(c.stderr).toContain("Debug: body");
        expect(c.stderr).not.toContain("SECRET");
      }
    } finally {
      api.stop();
    }
  });

  test("watch on a self-managed DBLab says there is no deploy to follow, and ends", async () => {
    const api = await startFakeApi({ progress: [{ is_cloud: false, deploy_status: null, deploy_step: null }] });
    try {
      const r = await runCli(["dblab", "instances", "watch", "41"], api.env);
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("not deployed in PostgresAI cloud");
      expect(r.stdout).not.toContain("Deploying");
      const j = await runCli(["dblab", "instances", "watch", "41", "--json"], api.env);
      expect(j.status).toBe(0);
      expect(JSON.parse(j.stdout).kind).toBe("not_cloud");
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab instances", () => {
  test("watch and delete refuse an id that is not a number", async () => {
    const api = await startFakeApi();
    try {
      for (const args of [["watch", "abc"], ["watch", "12x"], ["delete", "abc", "--yes"]]) {
        const r = await runCli(["dblab", "instances", ...args], api.env);
        expect(r.status).toBe(1);
        expect(r.stderr).toContain("is not a DBLab instance id");
      }
      expect(api.calls).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("list: a failed deploy reads Deploy failed, its reason on one line", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "Permission denied.\nDETAIL: more" }] });
    try {
      const r = await runCli(["dblab", "instances", "list"], api.env);
      expect(r.stdout).toMatch(/41\s+app-db\s+Deploy failed/);
      expect(r.stdout).toContain("Permission denied. (details: pgai dblab instances watch 41)");
      expect(r.stdout).not.toContain("DETAIL");
    } finally {
      api.stop();
    }
  });

  test("list: a removed server's IP is not shown for a failed or deleted deploy", async () => {
    for (const row of [
      { deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "permission denied", server_ip: "203.0.113.9" },
      { deploy_status: "destroyed", deploy_step: "ready", deploy_error: null, server_ip: "203.0.113.9" },
    ]) {
      const api = await startFakeApi({ progress: [row] });
      try {
        const r = await runCli(["dblab", "instances", "list"], api.env);
        expect(r.stdout).toMatch(/41\s+app-db\s+(Deploy failed|Deleted)\s+S 50GiB/);
        expect(r.stdout).not.toContain("203.0.113.9");
      } finally {
        api.stop();
      }
    }
  });

  test("watch on a deleted DBLab says it was deleted, not that it failed", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "ready", deploy_error: null }] });
    try {
      const r = await runCli(["dblab", "instances", "watch", "41"], api.env);
      expect(r.stderr).toContain('DBLab "app-db" (id 41) was deleted.');
      expect(r.stderr).not.toContain("failed");
      expect(r.stdout).not.toContain("Deploying");
    } finally {
      api.stop();
    }
  });

  test("watch on a deploy that already failed summarises it", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "permission denied" }] });
    try {
      const r = await runCli(["dblab", "instances", "watch", "41"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain('Deploy failed at "Copy data from the source": permission denied');
      expect(r.stderr).toContain("The server was removed; you were not charged for it.");
      expect(r.stdout).not.toContain("Deploying");
    } finally {
      api.stop();
    }
  });

  test("list shows the deploy status", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "retrieving", deploy_step: "retrieve_data", server_ip: "203.0.113.9" }] });
    try {
      const r = await runCli(["dblab", "instances", "list"], api.env);
      expect(r.status).toBe(0);
      expect(r.stdout).toMatch(/^ID\s+NAME\s+STATE/);
      expect(r.stdout).toMatch(/41\s+app-db\s+Copying data\s+S 50GiB 203\.0\.113\.9/);
    } finally {
      api.stop();
    }
  });

  test("delete --yes destroys the instance", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "instances", "delete", "41", "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_instance_destroy")[0].body).toEqual({ instance_id: 41 });
    } finally {
      api.stop();
    }
  });
});

describe("pgai mon deploy", () => {
  test("sends the Console's managed create and follows it to the Grafana URL", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["mon", "deploy", "--db-url", "postgresql://mon:p%40ss@db.example:5432/app", "--name", "app", "--vcpus", "8"], api.env);
      expect(r.stderr).toBe("");
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("Monitoring is ready: https://g.example/");
      const body = api.fn("monitoring_instance_create")[0].body;
      expect(body).toMatchObject({
        db_url: "postgresql://mon@db.example:5432/app", db_pass: "p@ss", billing_mode: "managed",
        software_plan: "scale", provision: "hetzner", server_type: "CCX23", server_location: "fsn1", project_name: "app",
        in_vcpus: 8,
      });
      expect(body).not.toHaveProperty("si_jwt_token");
      // No org in the config: an empty org_id lets the token name the org (0 would be another org).
      expect(body.org_id).toBe("");
    } finally {
      api.stop();
    }
  });

  test("the provisioning service refusing the launch fails at once, with its reason", async () => {
    const api = await startFakeApi({
      monCreate: { id: "0199-mon", response: { status_code: 400, message: "Bad Request", content: { Error: 'Image "x" is not in the allowlist' } } },
    });
    try {
      const r = await runCli(["mon", "deploy", "--db-url", "postgresql://mon:p@db.example/app"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain('The deploy did not start: Image "x" is not in the allowlist');
      expect(api.fn("monitoring_instance_deploy_status")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("a monitoring instance deleted while watched ends the watch", async () => {
    const api = await startFakeApi({ monProgress: [{ status: "launch_requested" }, { status: "deleted" }] });
    try {
      const r = await runCli(["mon", "deploy", "--db-url", "postgresql://mon:p@db.example/app"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Deploy failed");
    } finally {
      api.stop();
    }
  });

  test("a failed monitoring deploy exits 1 with the reason", async () => {
    const api = await startFakeApi({ monProgress: [{ status: "failed", error: "Cannot connect to the database" }] });
    try {
      const r = await runCli(["mon", "deploy", "--db-url", "postgresql://mon:p@db.example/app"], api.env);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain("Deploy failed: Cannot connect to the database");
    } finally {
      api.stop();
    }
  });
});
