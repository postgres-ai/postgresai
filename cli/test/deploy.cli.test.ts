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

/** At a terminal: what a person reads (stdout and stderr on one screen), control sequences dropped. */
async function runTty(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  let out = "";
  const proc = Bun.spawn([process.execPath, cliPath, ...args], {
    env: { ...process.env, PGAI_DEPLOY_POLL_MS: "10", PGAI_NO_FEEDBACK_TIP: "1", CI: "", ...env },
    terminal: { cols: 200, rows: 60, data(_t, bytes) { out += new TextDecoder().decode(bytes); } },
  });
  const status = await proc.exited;
  return { status, screen: out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "") };
}

type Body = Record<string, unknown>;

// The platform functions' argument names (db/api/v1/functions/*.sql in platform-all).
const RPC_ARGS: Record<string, string[]> = {
  dblab_cloud_deploy_options: ["org_id", "cloud"],
  dblab_instance_deploy: ["org_id", "name", "db_url", "size", "disk_gib", "ssh_key_ids", "location", "provider", "supabase_source_id",
    "cloud", "privatelink"],
  dblab_cloud_source_probe: ["org_id", "db_url", "provider"],
  dblab_cloud_instances: ["org_id", "p_instance_id"],
  dblab_instance_destroy: ["instance_id"],
  dblab_tunnel_access: ["org_id", "p_instance_id"],
  dblab_tunnel_keys_set: ["org_id", "p_instance_id", "ssh_key_ids"],
  dblab_tunnel_key_add: ["org_id", "p_instance_id", "public_key", "name"],
  dblab_tunnel_key_remove: ["org_id", "p_instance_id", "ssh_key_id"],
};

/** Fake platform for the deploy rpcs. `progress` is served one entry per poll. */
async function startFakeApi(opts: {
  progress?: Body[];
  deployError?: { status: number; body: Body };
  options?: Body;
  /** The org's instances as listed (instances watch / delete by name); default: the deploy being followed. */
  rows?: Body[];
  /** v1.dblab_cloud_source_probe's reply; absent: the platform has no probe (404). */
  probe?: Body;
  /** Extra keys of v1.dblab_instance_deploy's reply (e.g. the cloud and location it chose). */
  deployReply?: Body;
  /** The DBLab's clone-access keys (platform-all#953). */
  tunnelKeys?: { id: string; name: string }[];
  /** The caller is a member, not an admin: the access is read-only. */
  member?: boolean;
} = {}) {
  const calls: { fn: string; body: Body; headers: Record<string, string> }[] = [];
  let poll = 0;
  let tunnelKeys: { id: string; name: string }[] = opts.tunnelKeys ?? [];
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
          return json({ id: 41, project_id: 9, project_name: body.name, deploy_status: "launching", monthly_price_cents: 5650, ...opts.deployReply });
        case "dblab_cloud_instances": {
          if (opts.rows) return json(body.p_instance_id ? opts.rows.filter((r) => r.id === body.p_instance_id) : opts.rows);
          const p = progress[Math.min(poll++, progress.length - 1)];
          return json([{
            id: 41, project_id: 9, project_name: "app-db", created_at: "", is_cloud: true,
            deploy_error: null, deploy_updated_at: null, size: "S", disk_gib: 50, location: "fsn1",
            server_ip: null, is_job_backed: false, joe_instance_id: null, ...p,
          }]);
        }
        case "dblab_instance_destroy":
          return json({ result: body.instance_id, deploy_status: "destroying" });
        case "dblab_cloud_source_probe":
          if (opts.probe) return json(opts.probe);
          break;
        // Clone access (platform-all#953): the reply is the access after the change.
        case "dblab_tunnel_access":
        case "dblab_tunnel_keys_set":
        case "dblab_tunnel_key_add":
        case "dblab_tunnel_key_remove": {
          if (fn === "dblab_tunnel_keys_set") {
            const names = new Map([["0199-key", "laptop"], ...tunnelKeys.map((k) => [k.id, k.name] as [string, string])]);
            tunnelKeys = (body.ssh_key_ids as string[]).map((id) => ({ id, name: names.get(id) ?? id }));
          }
          if (fn === "dblab_tunnel_key_add") tunnelKeys = [...tunnelKeys, { id: "0199-new", name: (body.name as string) ?? "me@laptop" }];
          if (fn === "dblab_tunnel_key_remove") tunnelKeys = tunnelKeys.filter((k) => k.id !== body.ssh_key_id);
          return json({ instance_id: body.p_instance_id, ssh_user: "dblab-tunnel", host: "203.0.113.9",
            ports: { from: 6000, to: 6100 }, ready: true, can_edit: !opts.member,
            keys: tunnelKeys.map((k) => ({ ...k, added_at: null })) });
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
  // As pgai mon deploy does: with no card, nothing is asked or created; the user gets the billing page, exit 3.
  test("no card on file, at a terminal: the billing page before any prompt, exit 3", async () => {
    const api = await startFakeApi({ options: { has_payment_method: false } });
    try {
      const r = await runTty(DEPLOY, { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(r.screen).toContain("Add a payment method at https://console.example/acme/billing, then re-run.");
      expect(r.screen).not.toContain("Deploy? [y/N]");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("no card on file, piped: an action_required object on stdout, exit 3", async () => {
    const api = await startFakeApi({ options: { has_payment_method: false } });
    try {
      for (const extra of [[], ["--yes", "--json"]]) {
        const r = await runCli([...DEPLOY, ...extra], { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
        expect(r.status).toBe(3);
        expect(JSON.parse(r.stdout)).toEqual({
          status: "action_required", name: "app-db", next: "Add a payment method at https://console.example/acme/billing, then re-run.",
        });
      }
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
      expect(JSON.parse(r.stdout).next).toBe("Add a payment method at https://console.example/acme/billing, then re-run.");
    } finally {
      api.stop();
    }
  });

  test("at a terminal: the price, the steps, the ready text and the clone hint", async () => {
    const api = await startFakeApi();
    try {
      const r = await runTty([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain("Estimated cost: $56.50/month (size S, 50 GiB), prorated.");
      expect(r.screen).toContain("Create server");
      expect(r.screen).toContain('DBLab "app-db" (id 41) is ready.');
      // A clone needs a DB user and password; the hint is a command that works as printed.
      expect(r.screen).toContain(
        "Create a clone: PGAI_CLONE_DB_PASSWORD=<password> pgai dblab clone create --project app-db --db-user <user>");
      expect(r.screen).toContain("ssh -N -L 6000:127.0.0.1:6000 root@203.0.113.9");
      const body = api.fn("dblab_instance_deploy")[0].body;
      expect(body).toEqual({
        name: "app-db", db_url: "postgresql://u:secret@db.example:5432/app", size: "S", disk_gib: 50, ssh_key_ids: ["0199-key"],
      });
      expect(api.fn("dblab_instance_deploy")[0].headers["access-token"]).toBe("test-key");
    } finally {
      api.stop();
    }
  });

  test("piped: one JSON result on stdout; the price and each step a JSON event on stderr, no escape codes", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toMatchObject({ status: "ready", id: 41, name: "app-db" });
      const events = r.stderr.trim().split("\n").map((l) => JSON.parse(l));
      expect(events[0]).toEqual({ event: "billing", message: "Estimated cost: $56.50/month (size S, 50 GiB), prorated." });
      expect(events.some((e) => e.event === "step" && /^Ready: done/.test(e.message))).toBe(true);
      expect(r.stdout + r.stderr).not.toContain("\x1b");
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
      const out = JSON.parse(r.stdout);
      expect(out.status).toBe("failed");
      expect(out.next).toContain('Deploy failed at "Copy data from the source": Copying data from the source database failed: connection refused');
      expect(out.next).toContain("you are not charged");
      expect(r.stderr).toContain("Copy data from the source: FAILED");
    } finally {
      api.stop();
    }
  });

  test("refuses without an SSH key, naming the org's keys", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "deploy", "--name", "x1", "--db-url", "postgresql://u:p@h/d", "--yes"], api.env);
      expect(r.status).toBe(1);
      expect(JSON.parse(r.stdout).next).toContain("--ssh-key is required");
      expect(JSON.parse(r.stdout).next).toContain("Org keys: laptop.");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("--no-wait at a terminal: the id and how to follow it", async () => {
    const api = await startFakeApi();
    try {
      const r = await runTty([...DEPLOY, "--yes", "--no-wait"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain("Deploy started: id 41. Follow it: pgai dblab instances watch 41");
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
      expect(JSON.parse(r.stdout).status).toBe("ready");
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
      const next = JSON.parse(r.stdout).next;
      expect(next).toContain('A DBLab named "app-db" already exists.');
      expect(next).toContain("Hint: Choose another name.");
      expect(next).not.toContain("HTTP 409");
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
  });});

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
      const t = await runTty(["dblab", "instances", "watch", "41"], api.env);
      expect(t.status).toBe(0);
      expect(t.screen).toContain("not deployed in PostgresAI cloud");
      expect(t.screen).not.toContain("Deploying");
      const j = await runCli(["dblab", "instances", "watch", "41", "--json"], api.env);
      expect(j.status).toBe(0);
      expect(JSON.parse(j.stdout)).toMatchObject({ status: "ready", id: 41, next: "DBLab 41 is not deployed in PostgresAI cloud: there is no deploy to follow." });
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab instances", () => {
  test("watch and delete refuse an instance the org does not have, by id or name", async () => {
    const api = await startFakeApi();
    try {
      for (const args of [["watch", "abc"], ["watch", "12"], ["delete", "abc", "--yes"]]) {
        const r = await runCli(["dblab", "instances", ...args], api.env);
        expect(r.status).toBe(1);
        expect(JSON.parse(r.stdout).next).toMatch(/^No instance (abc|12)\. See: pgai dblab instances list$/);
      }
      expect(api.fn("dblab_instance_destroy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("list: a long database error is summed up by its cause, not cut before it", async () => {
    const deploy_error = "Copying data from the source database failed: failed to dump the database app: failed to dump a database: exit code: 1. Output: pg_dump: error: query failed: ERROR:  permission denied for table orders\r\npg_dump: detail: Query was: LOCK TABLE public.orders";
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error }] });
    try {
      const r = await runTty(["dblab", "instances", "list"], api.env);
      expect(r.screen).toContain("Copying data from the source database failed: permission denied for table orders (details: pgai dblab instances watch 41)");
      expect(r.screen).toContain("SIZE / SERVER / SOURCE");
    } finally {
      api.stop();
    }
  });

  test("list at a terminal: a failed deploy reads Deploy failed, its reason on one line", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "Permission denied.\nDETAIL: more" }] });
    try {
      const r = await runTty(["dblab", "instances", "list"], api.env);
      expect(r.screen).toMatch(/41\s+app-db\s+Deploy failed/);
      expect(r.screen).toContain("Permission denied. (details: pgai dblab instances watch 41)");
      expect(r.screen).not.toContain("DETAIL");
    } finally {
      api.stop();
    }
  });

  test("list at a terminal: a removed server's IP is not shown for a failed or deleted deploy", async () => {
    for (const row of [
      { deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "permission denied", server_ip: "203.0.113.9" },
      { deploy_status: "destroyed", deploy_step: "ready", deploy_error: null, server_ip: "203.0.113.9" },
    ]) {
      const api = await startFakeApi({ progress: [row] });
      try {
        const r = await runTty(["dblab", "instances", "list"], api.env);
        expect(r.screen).toMatch(/41\s+app-db\s+(Deploy failed|Deleted)\s+S 50GiB/);
        expect(r.screen).not.toContain("203.0.113.9");
      } finally {
        api.stop();
      }
    }
  });

  test("watch on a deleted DBLab says it was deleted, not that it failed", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "ready", deploy_error: null }] });
    try {
      const t = await runTty(["dblab", "instances", "watch", "41"], api.env);
      expect(t.screen).toContain('DBLab "app-db" (id 41) was deleted.');
      expect(t.screen).not.toContain("failed");
      const j = await runCli(["dblab", "instances", "watch", "41"], api.env);
      expect(JSON.parse(j.stdout)).toMatchObject({ status: "deleted", id: 41 });
    } finally {
      api.stop();
    }
  });

  test("watch on a deploy that already failed summarises it", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "destroyed", deploy_step: "retrieve_data", deploy_error: "permission denied" }] });
    try {
      const r = await runTty(["dblab", "instances", "watch", "41"], api.env);
      expect(r.status).toBe(1);
      expect(r.screen).toContain('Deploy failed at "Copy data from the source": permission denied');
      expect(r.screen).toContain("The server was removed; you were not charged for it.");
      expect(r.screen).not.toContain("Deploying");
    } finally {
      api.stop();
    }
  });

  test("list at a terminal shows the deploy status", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "retrieving", deploy_step: "retrieve_data", server_ip: "203.0.113.9" }] });
    try {
      const r = await runTty(["dblab", "instances", "list"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toMatch(/ID\s+NAME\s+STATE/);
      expect(r.screen).toMatch(/41\s+app-db\s+Copying data\s+S 50GiB 203\.0\.113\.9/);
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
  });});

// One surface with pgai mon deploy (postgresai#412): the URL as the argument,
// a default name, --wait/--no-wait, one result shape, instances by id or name.
const ROW = {
  project_id: 9, created_at: "", is_cloud: true, deploy_step: "ready", deploy_error: null, deploy_updated_at: null,
  size: "S", disk_gib: 50, location: "fsn1", server_ip: "203.0.113.9", is_job_backed: true, joe_instance_id: null,
};

describe("pgai dblab deploy: the surface it shares with pgai mon deploy", () => {
  test("the URL as the argument; the name from the database's; the result on stdout, the steps on stderr", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "deploy", "postgresql://u:secret@db.example:5432/Shop_DB", "--ssh-key", "laptop", "--yes", "--json"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_instance_deploy")[0].body.name).toBe("shop-db");
      expect(JSON.parse(r.stdout)).toEqual({
        id: 41, name: "app-db", server_ip: "203.0.113.9", status: "ready",
        next: "PGAI_CLONE_DB_PASSWORD=<password> pgai dblab clone create --project app-db --db-user <user>",
      });
      expect(r.stderr).toContain("Estimated cost: $56.50/month (size S, 50 GiB), prorated.");
    } finally {
      api.stop();
    }
  });

  test("the argument and --db-url naming different databases: refused, nothing sent", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli(["dblab", "deploy", "postgresql://u@a/x", "--db-url", "postgresql://u@b/y", "--ssh-key", "laptop", "--yes", "--json"], api.env);
      expect(r.status).toBe(1);
      expect(JSON.parse(r.stdout)).toEqual({ status: "failed", next: "Pass the database URL once: as the argument or as --db-url, not both." });
      expect(api.calls).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test.each([{ flag: ["--no-wait"] }, { flag: ["--wait", "0"] }])("$flag: in_progress at once, with how to follow it, exit 0", async ({ flag }) => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes", "--json", ...flag], api.env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toEqual({ status: "in_progress", id: 41, name: "app-db", next: "pgai dblab instances watch 41" });
      expect(api.fn("dblab_cloud_instances")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("--wait that runs out: in_progress, the deploy goes on, exit 0", async () => {
    const api = await startFakeApi({ progress: [{ deploy_status: "installing", deploy_step: "install_engine" }] });
    try {
      const r = await runCli([...DEPLOY, "--yes", "--json", "--wait", "0.001"], { ...api.env, PGAI_DEPLOY_POLL_MS: "50" });
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toEqual({ status: "in_progress", id: 41, name: "app-db", next: "pgai dblab instances watch 41" });
    } finally {
      api.stop();
    }
  });

  test("no --yes and nobody to ask: action required with the price, exit 3, nothing deployed", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--json"], api.env);
      expect(r.status).toBe(3);
      expect(JSON.parse(r.stdout)).toEqual({ status: "action_required", name: "app-db", next: "Re-run with --yes to accept $56.50/month (size S, 50 GiB), prorated." });
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("instances watch and delete take a name or an id; a name two instances have is refused with their ids", async () => {
    const rows = [
      { ...ROW, id: 41, project_name: "app-db", deploy_status: "ready" },
      { ...ROW, id: 42, project_name: "shop", deploy_status: "ready" },
      { ...ROW, id: 43, project_name: "shop", deploy_status: "ready" },
    ];
    const api = await startFakeApi({ rows });
    try {
      const watched = await runCli(["dblab", "instances", "watch", "app-db", "--json"], api.env);
      expect(watched.status).toBe(0);
      expect(JSON.parse(watched.stdout)).toMatchObject({ status: "ready", id: 41, name: "app-db" });
      const status = await runCli(["dblab", "instances", "status", "42", "--json"], api.env);
      expect(JSON.parse(status.stdout)).toMatchObject({ status: "ready", id: 42, name: "shop" });
      const ambiguous = await runCli(["dblab", "instances", "delete", "shop", "--yes", "--json"], api.env);
      expect(ambiguous.status).toBe(1);
      expect(JSON.parse(ambiguous.stdout)).toEqual({
        status: "failed", next: "More than one instance is named shop (ids 42, 43): use the id. See: pgai dblab instances list",
      });
      const deleted = await runCli(["dblab", "instances", "delete", "app-db", "--yes", "--json"], api.env);
      expect(deleted.status).toBe(0);
      expect(JSON.parse(deleted.stdout)).toEqual({ status: "deleting", id: 41, name: "app-db", next: "pgai dblab instances list" });
      expect(api.fn("dblab_instance_destroy").map((c) => c.body)).toEqual([{ instance_id: 41 }]);
      const asked = await runCli(["dblab", "instances", "delete", "42", "--json"], api.env);
      expect(asked.status).toBe(3);
      expect(JSON.parse(asked.stdout)).toEqual({ status: "action_required", id: 42, name: "shop", next: "pgai dblab instances delete 42 --yes" });
    } finally {
      api.stop();
    }
  });

  test("instances list: the same status words as pgai mon instances list", async () => {
    const rows = [
      { ...ROW, id: 41, project_name: "a", deploy_status: "retrieving" },
      { ...ROW, id: 42, project_name: "b", deploy_status: "destroy_failed" },
      { ...ROW, id: 43, project_name: "c", is_cloud: false, deploy_status: null },
    ];
    const api = await startFakeApi({ rows });
    try {
      const r = await runCli(["dblab", "instances", "list", "--json"], api.env);
      expect(JSON.parse(r.stdout).map((x: { id: number; status: string | null }) => [x.id, x.status])).toEqual([[41, "in_progress"], [42, "failed"], [43, null]]);
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab deploy: refusals read as pgai mon deploy's", () => {
  test("a token whose user is not an org admin: action required, exit 3, nothing deployed", async () => {
    for (const opts of [{ options: { is_admin: false } }, { deployError: { status: 403, body: { code: "PT403", message: "Forbidden", details: "Only an organization Admin can deploy." } } }]) {
      const api = await startFakeApi(opts);
      try {
        const r = await runCli([...DEPLOY, "--yes", "--json"], api.env);
        expect(r.status).toBe(3);
        expect(JSON.parse(r.stdout)).toEqual({ status: "action_required", name: "app-db",
          next: "Only an organization admin can deploy: ask an admin to run it, or use an admin's API key." });
      } finally {
        api.stop();
      }
    }
  });
});

describe("pgai dblab deploy: no card is told first, as pgai mon deploy does", () => {
  test("an org with no card and no SSH keys hears about the card, not the key", async () => {
    const api = await startFakeApi({ options: { has_payment_method: false, ssh_keys: [] } });
    try {
      const r = await runCli(["dblab", "deploy", "postgresql://u:p@h/app", "--yes", "--json"], { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(JSON.parse(r.stdout)).toEqual({ status: "action_required", name: "app",
        next: "Add a payment method at https://console.example/acme/billing, then re-run." });
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab instances: a deleted instance does not compete for its name", () => {
  test("a name redeployed after a deleted one picks the live one; by id the deleted one is still found", async () => {
    const rows = [
      { ...ROW, id: 50, project_name: "fx", deploy_status: "ready" },
      { ...ROW, id: 47, project_name: "fx", deploy_status: "destroyed" },
    ];
    const api = await startFakeApi({ rows });
    try {
      const byName = await runCli(["dblab", "instances", "watch", "fx", "--json"], api.env);
      expect(byName.status).toBe(0);
      expect(JSON.parse(byName.stdout)).toMatchObject({ status: "ready", id: 50 });
      const byId = await runCli(["dblab", "instances", "watch", "47", "--json"], api.env);
      expect(JSON.parse(byId.stdout)).toMatchObject({ status: "deleted", id: 47 });
    } finally {
      api.stop();
    }
  });
});

// platform-all#893
describe("pgai dblab deploy --provider", () => {
  const RDS_URL = "postgresql://u:secret@db1.abc.us-east-1.rds.amazonaws.com:5432/app";
  const probe = (over: Body = {}) => ({
    reachable: true, provider: "rds", detected_by: "host", postgres_major: 17, supported: true, ...over,
  });

  test("the source is named before the confirmation: detected by host, by extensions, or self-managed", async () => {
    for (const [p, line] of [
      [probe(), "Source: Amazon RDS (detected)"],
      [probe({ detected_by: "extensions" }), "Source: Amazon RDS (detected from extensions)"],
      [probe({ provider: "generic", detected_by: "default" }), "Source: PostgreSQL (self-managed)"],
    ] as [Body, string][]) {
      const api = await startFakeApi({ probe: p });
      try {
        // No --yes and nobody to ask: it stops at the confirmation, after naming the source.
        const r = await runCli(["dblab", "deploy", RDS_URL, "--name", "app-db", "--ssh-key", "laptop", "--json"], api.env);
        expect(r.status).toBe(3);
        const events = r.stderr.trim().split("\n").map((l) => JSON.parse(l));
        expect(events).toContainEqual({ event: "source", message: line });
        expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
      } finally {
        api.stop();
      }
    }
  });

  test("--provider is sent to the test and the deploy", async () => {
    const api = await startFakeApi({ probe: probe({ detected_by: "flag" }) });
    try {
      const r = await runCli([...DEPLOY, "--provider", "RDS", "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_cloud_source_probe")[0].body).toEqual({ db_url: "postgresql://u:secret@db.example:5432/app", provider: "rds" });
      expect(api.fn("dblab_instance_deploy")[0].body.provider).toBe("rds");
      expect(r.stderr).toContain('"Source: Amazon RDS (from --provider)"');
    } finally {
      api.stop();
    }
  });

  test("--provider generic names the direct path, as pgai mon deploy does", async () => {
    const api = await startFakeApi({ probe: probe({ provider: "generic", detected_by: "flag" }) });
    try {
      const r = await runCli([...DEPLOY, "--provider", " Generic ", "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_cloud_source_probe")[0].body).toEqual({ db_url: "postgresql://u:secret@db.example:5432/app", provider: "generic" });
      expect(api.fn("dblab_instance_deploy")[0].body.provider).toBe("generic");
      expect(r.stderr).toContain('"Source: PostgreSQL (from --provider)"');
    } finally {
      api.stop();
    }
  });

  test("an unknown provider is refused before anything is called, exit 1", async () => {
    const api = await startFakeApi();
    try {
      for (const bad of ["clickhouse", "foo"]) {
        const r = await runCli([...DEPLOY, "--provider", bad, "--yes"], api.env);
        expect(r.status).toBe(1);
        expect(JSON.parse(r.stdout).next).toBe("--provider must be one of: generic, rds, supabase, self-managed");
      }
      expect(api.calls).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("a Supabase URL without a password goes to the Console, exit 3, nothing tested or deployed", async () => {
    const api = await startFakeApi({ probe: probe() });
    try {
      const url = "postgresql://postgres.abcdefghij@aws-1-eu-central-1.pooler.supabase.com:5432/postgres";
      const r = await runCli(["dblab", "deploy", url, "--ssh-key", "laptop", "--yes", "--json"], {
        ...api.env, PGAI_UI_BASE_URL: "https://console.example", PGPASSWORD: "",
      });
      expect(r.status).toBe(3);
      expect(JSON.parse(r.stdout)).toMatchObject({
        status: "action_required",
        next: "This Supabase URL has no password. Add it to the URL or set PGPASSWORD, or finish in the Console, which needs no password: https://console.example/acme/instances/deploy?source=supabase",
      });
      expect(api.fn("dblab_cloud_source_probe")).toHaveLength(0);
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
      // With its password it deploys like any other source.
      const ok = await runCli(["dblab", "deploy", url.replace("@", ":pw@"), "--ssh-key", "laptop", "--yes"], api.env);
      expect(ok.status).toBe(0);
      expect(api.fn("dblab_instance_deploy")).toHaveLength(1);
    } finally {
      api.stop();
    }
  });

  test("an unreachable source fails before the confirmation with the reason and what to do", async () => {
    const api = await startFakeApi({
      probe: probe({ reachable: false, error: "connection timed out", hint: "The RDS instance must be publicly accessible." }),
    });
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(1);
      expect(JSON.parse(r.stdout).next).toBe(
        "Cannot connect to the source database: connection timed out\nHint: The RDS instance must be publicly accessible."
        + "\nA private RDS deploys over PrivateLink in the Console: https://console.postgres.ai/acme/instances/deploy?source=rds&network=private");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  // platform-all#925: a private RDS is deployed over PrivateLink, from the Console.
  test("an RDS whose name resolves to a private address goes to the PrivateLink wizard, exit 3", async () => {
    const api = await startFakeApi({ probe: probe({ reachable: false, error: "connection timed out" }) });
    try {
      const r = await runCli(["dblab", "deploy", "postgresql://u:secret@10.77.1.5:5432/app", "--name", "app-db", "--ssh-key", "laptop",
        "--provider", "rds", "--yes", "--json"], { ...api.env, PGAI_UI_BASE_URL: "https://console.example" });
      expect(r.status).toBe(3);
      expect(JSON.parse(r.stdout)).toMatchObject({
        status: "action_required",
        next: "This RDS is not reachable from the internet. If it is private, deploy it over PrivateLink in the Console: "
          + "https://console.example/acme/instances/deploy?source=rds&network=private",
      });
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
      // No price first: it would be this cloud's, and the private deploy runs on AWS (design review).
      expect(r.stderr).not.toContain("Estimated cost");
    } finally {
      api.stop();
    }
  });

  test("instances list names the source", async () => {
    const api = await startFakeApi({
      rows: [{ id: 41, project_id: 9, project_name: "app-db", created_at: "", is_cloud: true, deploy_status: "ready",
        deploy_step: "ready", deploy_error: null, deploy_updated_at: null, size: "S", disk_gib: 50, location: "fsn1",
        server_ip: "203.0.113.9", is_job_backed: true, joe_instance_id: null, source_provider: "supabase", source_name: "Shop" }],
    });
    try {
      const r = await runTty(["dblab", "instances", "list"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain("S 50GiB 203.0.113.9 from Supabase · Shop");
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab deploy --cloud (platform-all#925)", () => {
  const AWS_OPTIONS = {
    cloud: "aws",
    sizes: [{ code: "S", vcpus: 2, ram_gib: 8, monthly_price_cents: 9900, available: true }],
    locations: ["us-east-1", "us-east-2"],
    clouds: [
      { code: "hetzner", label: "Hetzner", available: true },
      { code: "aws", label: "AWS", available: true },
      { code: "gcp", label: "Google Cloud", available: false },
    ],
  };

  test("without --cloud nothing new is sent, so an older platform still answers", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_cloud_deploy_options")[0].body).toEqual({});
      expect(api.fn("dblab_instance_deploy")[0].body).not.toHaveProperty("cloud");
    } finally {
      api.stop();
    }
  });

  test("--cloud aws: that cloud's price before the confirmation, and the cloud sent to the deploy", async () => {
    const api = await startFakeApi({ options: AWS_OPTIONS });
    try {
      const r = await runCli([...DEPLOY, "--cloud", "AWS", "--location", "us-east-2", "--yes", "--json"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_cloud_deploy_options")[0].body).toEqual({ cloud: "aws" });
      expect(api.fn("dblab_instance_deploy")[0].body).toMatchObject({ cloud: "aws", location: "us-east-2" });
      const events = r.stderr.trim().split("\n").map((l) => JSON.parse(l));
      expect(events).toContainEqual({ event: "billing", message: "Estimated cost: $106.50/month (size S on AWS, 50 GiB), prorated." });
    } finally {
      api.stop();
    }
  });

  // REV r1: --no-wait --json (the agents' path) carries where it runs too.
  test("--no-wait --json: the result names the cloud and location", async () => {
    const api = await startFakeApi({ options: AWS_OPTIONS, deployReply: { cloud: "aws", location: "eu-central-1" } });
    try {
      const r = await runCli([...DEPLOY, "--cloud", "aws", "--yes", "--json", "--no-wait"], api.env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toMatchObject({ status: "in_progress", id: 41, cloud: "aws", location: "eu-central-1" });
    } finally {
      api.stop();
    }
  });

  // REV r1: an explicit --cloud hetzner is today's request, so an older platform still answers it.
  test("--cloud hetzner sends no cloud", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCli([...DEPLOY, "--cloud", "hetzner", "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_cloud_deploy_options")[0].body).toEqual({});
      expect(api.fn("dblab_instance_deploy")[0].body).not.toHaveProperty("cloud");
    } finally {
      api.stop();
    }
  });

  // AW7: with no --location the platform picks the region (the source's); the progress names it.
  test("the progress names the cloud and the location the platform chose", async () => {
    const api = await startFakeApi({ options: AWS_OPTIONS, deployReply: { cloud: "aws", location: "eu-central-1" } });
    try {
      const r = await runTty([...DEPLOY, "--cloud", "aws", "--yes"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain('Deploying DBLab "app-db" (id 41) on AWS, eu-central-1');
    } finally {
      api.stop();
    }
  });

  test("an unknown cloud, a cloud that is off, or another cloud's location: refused before any deploy, exit 1", async () => {
    const api = await startFakeApi({ options: AWS_OPTIONS });
    try {
      const bad = await runCli([...DEPLOY, "--cloud", "azure", "--yes"], api.env);
      expect(bad.status).toBe(1);
      expect(JSON.parse(bad.stdout).next).toBe("--cloud must be one of: hetzner, aws, gcp");
      expect(api.calls).toHaveLength(0);

      const off = await runCli([...DEPLOY, "--cloud", "gcp", "--yes"], api.env);
      expect(off.status).toBe(1);
      expect(JSON.parse(off.stdout).next).toBe("Google Cloud is not available.");

      const where = await runCli([...DEPLOY, "--cloud", "aws", "--location", "fsn1", "--yes"], api.env);
      expect(where.status).toBe(1);
      expect(JSON.parse(where.stdout).next).toBe('Unknown location "fsn1" for AWS. Choose one of: us-east-1, us-east-2.');
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("the result names the cloud and region the server runs in", async () => {
    const api = await startFakeApi({
      options: { ...AWS_OPTIONS, cloud: "gcp", locations: ["us-central1"], clouds: AWS_OPTIONS.clouds.map((c) => ({ ...c, available: true })) },
      progress: [{ deploy_status: "ready", deploy_step: "ready", server_ip: "203.0.113.11", cloud: "gcp", location: "us-central1" }],
    });
    try {
      const r = await runCli([...DEPLOY, "--cloud", "gcp", "--yes", "--json"], api.env);
      expect(r.status).toBe(0);
      expect(JSON.parse(r.stdout)).toMatchObject({ status: "ready", id: 41, cloud: "gcp", location: "us-central1", server_ip: "203.0.113.11" });
    } finally {
      api.stop();
    }
  });

  test("instances list names an AWS or GCP server's cloud and region", async () => {
    const api = await startFakeApi({
      rows: [{ id: 42, project_id: 9, project_name: "app-db", created_at: "", is_cloud: true, deploy_status: "ready",
        deploy_step: "ready", deploy_error: null, deploy_updated_at: null, size: "S", disk_gib: 50, location: "europe-west3",
        cloud: "gcp", ssh_user: "root", server_ip: "203.0.113.10", is_job_backed: true, joe_instance_id: null }],
    });
    try {
      const r = await runTty(["dblab", "instances", "list"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain("S 50GiB Google Cloud europe-west3 203.0.113.10");
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab instances watch: a failed deploy's server IP is not shown (it may be another instance's now)", () => {
  test("no server_ip in the result of a destroyed deploy", async () => {
    const api = await startFakeApi({ rows: [{ ...ROW, id: 47, project_name: "gone", deploy_status: "destroyed", deploy_error: "boom", server_ip: "203.0.113.9" }] });
    try {
      const r = await runCli(["dblab", "instances", "watch", "47", "--json"], api.env);
      expect(JSON.parse(r.stdout)).not.toHaveProperty("server_ip");
    } finally {
      api.stop();
    }
  });
});

// platform-all#953: servers in PostgresAI cloud get no customer keys; clone access is added afterwards.
describe("pgai dblab deploy without customer SSH keys", () => {
  const NO_KEY = ["dblab", "deploy", "--name", "app-db", "--db-url", "postgresql://u:secret@db.example:5432/app", "--yes", "--json"];
  const TUNNEL_PROGRESS = [
    { deploy_status: "launching", deploy_step: "create_server", ssh_user: "dblab-tunnel" },
    { deploy_status: "ready", deploy_step: "ready", server_ip: "203.0.113.9", ssh_user: "dblab-tunnel" },
  ];

  test("off: deploys with no key, sends none, and the ready hint names the tunnel user and pgai dblab access", async () => {
    const api = await startFakeApi({ options: { customer_ssh_keys: "off", ssh_keys: [] }, progress: TUNNEL_PROGRESS });
    try {
      const r = await runTty(NO_KEY.filter((a) => a !== "--json"), api.env);
      expect(r.status).toBe(0);
      const deploy = api.fn("dblab_instance_deploy");
      expect(deploy).toHaveLength(1);
      expect("ssh_key_ids" in deploy[0].body).toBe(false);
      expect(r.screen).toContain("pgai dblab access add 41 --public-key-file ~/.ssh/id_ed25519.pub");
      expect(r.screen).toContain("ssh -N -L 6000:127.0.0.1:6000 dblab-tunnel@203.0.113.9");
    } finally {
      api.stop();
    }
  });

  test("off: --ssh-key is refused before anything is asked, naming pgai dblab access", async () => {
    const api = await startFakeApi({
      options: { customer_ssh_keys: "off" },
      probe: { reachable: true, provider: "generic", detected_by: "default", postgres_major: 16, supported: true },
    });
    try {
      const r = await runCli([...NO_KEY, "--ssh-key", "laptop"], api.env);
      expect(r.status).toBe(1);
      expect(r.stdout + r.stderr).toContain("get no SSH keys");
      expect(r.stdout + r.stderr).toContain("pgai dblab access add");
      expect(api.fn("dblab_instance_deploy")).toHaveLength(0);
      // Refused before the source database is contacted.
      expect(api.fn("dblab_cloud_source_probe")).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("optional: no key is fine; a key is sent", async () => {
    const api = await startFakeApi({ options: { customer_ssh_keys: "optional" }, progress: TUNNEL_PROGRESS });
    try {
      expect((await runCli(NO_KEY, api.env)).status).toBe(0);
      expect("ssh_key_ids" in api.fn("dblab_instance_deploy")[0].body).toBe(false);
      expect((await runCli([...NO_KEY, "--name", "app-db2", "--ssh-key", "laptop"], api.env)).status).toBe(0);
      expect(api.fn("dblab_instance_deploy")[1].body.ssh_key_ids).toEqual(["0199-key"]);
    } finally {
      api.stop();
    }
  });

  test("required (or a platform that predates the switch): no key is refused", async () => {
    const api = await startFakeApi({ options: { customer_ssh_keys: "required" } });
    try {
      const r = await runCli(NO_KEY, api.env);
      expect(r.status).toBe(1);
      expect(r.stdout + r.stderr).toContain("--ssh-key is required. Org keys: laptop.");
    } finally {
      api.stop();
    }
  });
});

describe("pgai dblab access", () => {
  const ROWS = [{ id: 41, project_name: "app-db", is_cloud: true, deploy_status: "ready", server_ip: "203.0.113.9", ssh_user: "dblab-tunnel" }];

  test("list: the keys and the tunnel command; none: how to add one", async () => {
    const api = await startFakeApi({ rows: ROWS, tunnelKeys: [{ id: "0199-key", name: "laptop" }] });
    try {
      const r = await runTty(["dblab", "access", "list", "app-db"], api.env);
      expect(r.status).toBe(0);
      expect(r.screen).toContain("laptop  0199-key");
      expect(r.screen).toContain("ssh -N -L 6000:127.0.0.1:6000 dblab-tunnel@203.0.113.9");
      expect(api.fn("dblab_tunnel_access")[0].body).toEqual({ p_instance_id: 41 });
    } finally {
      api.stop();
    }
    const empty = await startFakeApi({ rows: ROWS });
    try {
      const r = await runTty(["dblab", "access", "list", "41"], empty.env);
      expect(r.screen).toContain("No SSH key can open a tunnel to the clones yet.");
      expect(r.screen).toContain("pgai dblab access add 41 --public-key-file ~/.ssh/id_ed25519.pub");
      const j = await runCli(["dblab", "access", "list", "41"], empty.env);
      expect(JSON.parse(j.stdout).keys).toEqual([]);
    } finally {
      empty.stop();
    }
  });

  test("list as a member: no command they can't run; told who can change it", async () => {
    const none = await startFakeApi({ rows: ROWS, member: true });
    try {
      const r = await runTty(["dblab", "access", "list", "41"], none.env);
      expect(r.screen).toContain("Ask an admin of this organization to add yours.");
      expect(r.screen).not.toContain("pgai dblab access add");
    } finally {
      none.stop();
    }
    const some = await startFakeApi({ rows: ROWS, member: true, tunnelKeys: [{ id: "0199-key", name: "laptop" }] });
    try {
      const r = await runTty(["dblab", "access", "list", "41"], some.env);
      expect(r.screen).toContain("laptop  0199-key");
      expect(r.screen).toContain("Only an organization admin can change them.");
      expect(r.screen).toContain('psql "host=127.0.0.1 port=6000 user=<clone user> dbname=<db>"');
    } finally {
      some.stop();
    }
  });

  test("add --public-key-file: sends the key text; a private key file is refused locally", async () => {
    const api = await startFakeApi({ rows: ROWS });
    const dir = mkdtempSync(resolve(tmpdir(), "pgai-access-"));
    const pub = resolve(dir, "id_ed25519.pub");
    const priv = resolve(dir, "id_ed25519");
    await Bun.write(pub, "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFExvmPf29KRq3Lm2ECFjIhlyRs9OI7f9h3X56nT6YIm me@laptop\n");
    // A private key file's shape, split so secret scanners do not read it as one.
    const marker = ["OPENSSH", "PRIVATE", "KEY"].join(" ");
    await Bun.write(priv, `-----BEGIN ${marker}-----\nxxx\n-----END ${marker}-----\n`);
    try {
      const r = await runCli(["dblab", "access", "add", "app-db", "--public-key-file", pub, "--json"], api.env);
      expect(r.status).toBe(0);
      const add = api.fn("dblab_tunnel_key_add");
      expect(add[0].body.public_key).toBe("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFExvmPf29KRq3Lm2ECFjIhlyRs9OI7f9h3X56nT6YIm me@laptop");
      expect(add[0].body.p_instance_id).toBe(41);
      expect(JSON.parse(r.stdout).keys).toHaveLength(1);

      const p = await runCli(["dblab", "access", "add", "app-db", "--public-key-file", priv], api.env);
      expect(p.status).toBe(1);
      expect(p.stdout + p.stderr).toContain("is a private key");
      expect(api.fn("dblab_tunnel_key_add")).toHaveLength(1);
    } finally {
      api.stop();
    }
  });

  test("add --ssh-key keeps the keys already there; remove takes one away", async () => {
    const api = await startFakeApi({ rows: ROWS, tunnelKeys: [{ id: "0199-other", name: "other" }] });
    try {
      expect((await runCli(["dblab", "access", "add", "app-db", "--ssh-key", "laptop"], api.env)).status).toBe(0);
      expect(api.fn("dblab_tunnel_keys_set")[0].body.ssh_key_ids).toEqual(["0199-other", "0199-key"]);
      const r = await runCli(["dblab", "access", "remove", "app-db", "--ssh-key", "laptop"], api.env);
      expect(r.status).toBe(0);
      expect(api.fn("dblab_tunnel_key_remove")[0].body).toEqual({ p_instance_id: 41, ssh_key_id: "0199-key" });
      expect(JSON.parse(r.stdout).keys.map((k: { name: string }) => k.name)).toEqual(["other"]);
    } finally {
      api.stop();
    }
  });

  test("add with neither a file nor a key: a usage error, nothing sent", async () => {
    const api = await startFakeApi({ rows: ROWS });
    try {
      const r = await runCli(["dblab", "access", "add", "app-db"], api.env);
      expect(r.status).toBe(1);
      expect(r.stdout + r.stderr).toContain("Give --public-key-file <path> or --ssh-key <name|id>.");
      expect(api.calls.filter((c) => c.fn.startsWith("dblab_tunnel"))).toHaveLength(0);
    } finally {
      api.stop();
    }
  });
});
