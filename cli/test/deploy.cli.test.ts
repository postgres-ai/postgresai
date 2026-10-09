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
  dblab_cloud_deploy_options: ["org_id"],
  dblab_instance_deploy: ["org_id", "name", "db_url", "size", "disk_gib", "ssh_key_ids", "location"],
  dblab_cloud_instances: ["org_id", "p_instance_id"],
  dblab_instance_destroy: ["instance_id"],
};

/** Fake platform for the deploy rpcs. `progress` is served one entry per poll. */
async function startFakeApi(opts: {
  progress?: Body[];
  deployError?: { status: number; body: Body };
  options?: Body;
  /** The org's instances as listed (instances watch / delete by name); default: the deploy being followed. */
  rows?: Body[];
} = {}) {
  const calls: { fn: string; body: Body; headers: Record<string, string> }[] = [];
  let poll = 0;
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

  test.each([["--no-wait"], ["--wait", "0"]])("%s: in_progress at once, with how to follow it, exit 0", async (...flag) => {
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
