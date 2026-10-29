import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";

// `pgai billing status` (postgresai#462). Async spawn: the fake platform runs in this process.
async function runCli(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const proc = Bun.spawn([process.execPath, cliPath, ...args], {
    env: { ...process.env, CI: "1", PGAI_NO_FEEDBACK_TIP: "1", ...env },
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

/** At a terminal: what a person reads, control sequences dropped. */
async function runTty(args: string[], env: Record<string, string>) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  let out = "";
  const proc = Bun.spawn([process.execPath, cliPath, ...args], {
    env: { ...process.env, PGAI_NO_FEEDBACK_TIP: "1", CI: "", ...env },
    terminal: { cols: 200, rows: 60, data(_t, bytes) { out += new TextDecoder().decode(bytes); } },
  });
  const status = await proc.exited;
  return { status, screen: out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "") };
}

type Body = Record<string, unknown>;

/** Fake platform: v1.dblab_cloud_deploy_options only, or a refusal. */
function startFakeApi(opts: { options?: Body; refusal?: { status: number; body: Body } } = {}) {
  const calls: { fn: string; body: Body; headers: Record<string, string> }[] = [];
  const json = (b: unknown, status = 200) =>
    new Response(JSON.stringify(b), { status, headers: { "Content-Type": "application/json" } });
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const fn = new URL(req.url).pathname.split("/rpc/")[1] ?? "";
      const body = (req.method === "POST" ? await req.json().catch(() => ({})) : {}) as Body;
      calls.push({ fn, body, headers: Object.fromEntries(req.headers.entries()) });
      if (fn !== "dblab_cloud_deploy_options") return new Response("not found", { status: 404 });
      if (opts.refusal) return json(opts.refusal.body, opts.refusal.status);
      return json({
        create_options: ["cloud"], sizes: [], ssh_keys: [], is_admin: false,
        has_payment_method: true, org_alias: "acme", ...opts.options,
      });
    },
  });
  const env = {
    XDG_CONFIG_HOME: mkdtempSync(resolve(tmpdir(), "pgai-billing-test-")),
    PGAI_API_KEY: "test-key",
    PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}/api/general`,
    PGAI_UI_BASE_URL: "https://console.example",
  };
  return { env, calls, stop: () => server.stop(true) };
}

const STATUS = ["billing", "status"];

describe("pgai billing status", () => {
  test("a payment method on file, at a terminal: one line, exit 0", async () => {
    const api = startFakeApi();
    try {
      const r = await runTty(STATUS, api.env);
      expect(r.screen).toContain("Payment method on file for acme.");
      expect(r.status).toBe(0);
    } finally {
      api.stop();
    }
  });

  test("no payment method, at a terminal: the billing page, exit 3", async () => {
    const api = startFakeApi({ options: { has_payment_method: false } });
    try {
      const r = await runTty(STATUS, api.env);
      expect(r.screen).toContain("No payment method on file for acme. Add one at https://console.example/acme/billing");
      expect(r.status).toBe(3);
    } finally {
      api.stop();
    }
  });

  test("--json: org, has_payment_method and billing_url only, on stdout", async () => {
    for (const [has, code] of [[true, 0], [false, 3]] as const) {
      const api = startFakeApi({ options: { has_payment_method: has } });
      try {
        const r = await runCli([...STATUS, "--json"], api.env);
        expect(JSON.parse(r.stdout)).toEqual({ org: "acme", has_payment_method: has, billing_url: "https://console.example/acme/billing" });
        expect(r.status).toBe(code);
      } finally {
        api.stop();
      }
    }
  });

  test("stdout not a terminal, no --json: JSON all the same", async () => {
    const api = startFakeApi({ options: { has_payment_method: false } });
    try {
      const r = await runCli(STATUS, api.env);
      expect(JSON.parse(r.stdout).has_payment_method).toBe(false);
      expect(r.status).toBe(3);
    } finally {
      api.stop();
    }
  });

  // The watcher waits for "true": anything else keeps it waiting.
  test("Stripe unreadable (null or no field): unknown, exit 1", async () => {
    for (const options of [{ has_payment_method: null }, { has_payment_method: undefined }]) {
      const api = startFakeApi({ options });
      try {
        const r = await runCli([...STATUS, "--json"], api.env);
        expect(JSON.parse(r.stdout).has_payment_method).toBeNull();
        expect(r.status).toBe(1);
        const t = await runTty(STATUS, api.env);
        expect(t.screen).toContain("Could not check whether acme has a payment method on file.");
        expect(t.status).toBe(1);
      } finally {
        api.stop();
      }
    }
  });

  test("reads the deploy options with no promo code: no quote, no coupon", async () => {
    const api = startFakeApi();
    try {
      await runCli([...STATUS, "--json", "--org", "acme"], api.env);
      expect(api.calls.map((c) => c.fn)).toEqual(["dblab_cloud_deploy_options"]);
      expect(api.calls[0].body).toEqual({});
      expect(api.calls[0].headers["x-pgai-org"]).toBe("acme");
    } finally {
      api.stop();
    }
  });

  test("no org alias in the reply: billing_url null, the Console named in words", async () => {
    const api = startFakeApi({ options: { has_payment_method: false, org_alias: null } });
    try {
      const r = await runCli([...STATUS, "--json"], api.env);
      expect(JSON.parse(r.stdout)).toEqual({ org: null, has_payment_method: false, billing_url: null });
      const t = await runTty(STATUS, api.env);
      expect(t.screen).toContain("Add one at https://console.example (your organization > Billing)");
    } finally {
      api.stop();
    }
  });

  test("--org-id selects the org by id", async () => {
    const api = startFakeApi();
    try {
      await runCli([...STATUS, "--json", "--org-id", "7957"], api.env);
      expect(api.calls[0].headers["x-pgai-org-id"]).toBe("7957");
    } finally {
      api.stop();
    }
  });

  // A 402 can mean a suspended organization: never read as "no card".
  test("every refusal is an error, 402 and 403 included: the platform's words on stderr, exit 1", async () => {
    for (const refusal of [
      { status: 402, body: { message: "Payment Required", details: "The token organization is suspended." } },
      { status: 403, body: { message: "Forbidden", details: "This token belongs to another organization." } },
    ]) {
      const api = startFakeApi({ refusal });
      try {
        const r = await runCli([...STATUS, "--json"], api.env);
        expect(r.stdout).toBe("");
        expect(r.stderr).toContain(refusal.body.details);
        expect(r.status).toBe(1);
      } finally {
        api.stop();
      }
    }
  });

  test("a global token without --org: names --org, sends nothing, exit 1", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli([...STATUS, "--json"], { ...api.env, PGAI_API_KEY: "pai_global_test" });
      expect(r.stderr).toContain("--org <alias>");
      expect(r.status).toBe(1);
      expect(api.calls).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("--org and --org-id together: refused before any request, exit 1", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli([...STATUS, "--org", "acme", "--org-id", "1"], api.env);
      expect(r.stderr).toContain("Pass either --org or --org-id, not both.");
      expect(r.status).toBe(1);
      expect(api.calls).toHaveLength(0);
    } finally {
      api.stop();
    }
  });

  test("no API key: says how to sign in, exit 1", async () => {
    const api = startFakeApi();
    try {
      const r = await runCli(STATUS, { ...api.env, PGAI_API_KEY: "" });
      expect(r.stderr).toContain("API key is required");
      expect(r.status).toBe(1);
    } finally {
      api.stop();
    }
  });

  test("the platform unreachable: an error, exit 1", async () => {
    const api = startFakeApi();
    api.stop();
    const r = await runCli([...STATUS, "--json"], api.env);
    expect(r.stdout).toBe("");
    expect(r.stderr).toContain("Payment method check: could not reach");
    expect(r.status).toBe(1);
  });
});
