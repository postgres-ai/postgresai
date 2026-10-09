import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";

// The commands as a user or an agent runs them: stdout not a TTY (so JSON),
// a fake platform API, a throwaway HOME. Frozen: the JSON contract
// (status, next, ...) and the exit codes 0 / 1 / 3.

const CH = "postgresql://postgres:adminpw@abc123.us-east-1.aws.pg.clickhouse.cloud:5432/postgres?sslmode=require";
const NAME = "abc123.us-east-1.aws.pg.clickhouse.cloud/postgres";
const ROW = { id: "i-1", name: NAME, provider: "clickhouse", mode: "cloud", status: "active", dashboard_url: "https://abc.pgai.watch", host_metrics: true, created_at: "2026-09-30T00:00:00Z" };

// What mon deploy, status and list show: the Health Matrix first, then Grafana through the PostgresAI sign-in.
const HEALTH = "https://console.example/acme/projects/abc-db/health";
const GRAFANA = "https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F";
const LOOKUPS = ["/rpc/orgs_list test-key {}", "/rpc/projects_list test-key {}"];

const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
const cliEnv = (env: Record<string, string>) => {
  const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-"));
  return { ...process.env, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "", CLICKHOUSE_KEY_ID: "", CLICKHOUSE_KEY_SECRET: "", ...env };
};

async function run(args: string[], env: Record<string, string>, stdin?: string) {
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    env: cliEnv(env), stdin: stdin === undefined ? "ignore" : new Blob([stdin]), stdout: "pipe", stderr: "pipe",
  });
  const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  return { status, stdout, stderr, json: () => JSON.parse(stdout) };
}

const BILLED = {
  plan: "scale", org_alias: "acme", billed: true, free_slots: { remaining: 0, total: 0 }, subscription: false, quantity: 0,
  price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: true, requires_payment_method: false,
};

async function withApi(fn: (env: Record<string, string>, calls: string[]) => Promise<void>, rows: unknown[] | (() => unknown[]) = [ROW]) {
  const calls: string[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1", port: 0,
    async fetch(req) {
      const path = new URL(req.url).pathname;
      calls.push(`${path} ${req.headers.get("access-token")} ${await req.text()}`);
      if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json(typeof rows === "function" ? rows() : rows);
      if (path.endsWith("/rpc/cloud_monitoring_quote")) return Response.json(BILLED);
      if (path.endsWith("/rpc/cloud_monitoring_disconnect")) return Response.json({ id: "i-1", status: "deleting_launched" });
      if (path.endsWith("/rpc/orgs_list")) return Response.json([{ org_id: 7, alias: "acme", name: "Acme", is_active: true }]);
      if (path.endsWith("/rpc/projects_list")) return Response.json([{ project_id: 3, alias: "abc-db", name: NAME, monitoring_instance_ids: ["i-1"] }]);
      return new Response("not found", { status: 404 });
    },
  });
  try {
    await fn({ PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}`, PGAI_UI_BASE_URL: "https://console.example" }, calls);
  } finally {
    server.stop(true);
  }
}

// Drop terminal control sequences and carriage returns, keep what a person reads.
const clean = (s: string) => s.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "");

/** Runs in a terminal, typing each answer (keys as sent, so end a line with \\r) when its prompt appears; returns the screen text. */
async function runTty(args: string[], env: Record<string, string>, answers: [prompt: string | RegExp, answer: string][]) {
  let out = "";
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    env: cliEnv({ PGAI_NO_FEEDBACK_TIP: "1", ...env }),
    terminal: {
      cols: 200, rows: 50,
      data(term, bytes) {
        out += new TextDecoder().decode(bytes);
        // Typed a moment after the prompt, once readline owns the terminal (input sent earlier can be flushed).
        const prompt = answers[0]?.[0];
        if (prompt !== undefined && (typeof prompt === "string" ? clean(out).endsWith(prompt) : prompt.test(clean(out)))) {
          const answer = answers.shift()![1];
          setTimeout(() => term.write(answer), 100);
        }
      },
    },
  });
  const status = await proc.exited;
  return { status, screen: clean(out) };
}

const KEY_PROMPT = "ClickHouse Cloud API key <key-id>:<key-secret> for CPU, memory and disk (not shown; Enter to skip): ";
const URL_PROMPT = "Database URL (postgresql://...; not shown): ";
const TO_CONNECT = { status: "action_required", provider: "self-managed", name: "", next: "pgai init is for a person at a terminal; agents and scripts: pgai mon deploy <database-url>" };

describe("pgai init", () => {
  test("piped stdin: points to pgai mon deploy, exit 3, nothing asked", async () => {
    const r = await run(["init"], {}, `${CH}\n`);
    expect(r.status).toBe(3);
    expect(r.json()).toEqual(TO_CONNECT);
    expect(r.stdout + r.stderr).not.toContain("Database URL");
  });

  test("--json in a terminal: points to pgai mon deploy, exit 3", async () => {
    const r = await runTty(["init", "--json"], {}, []);
    expect(r.status).toBe(3);
    expect(JSON.parse(r.screen)).toEqual(TO_CONNECT);
  });

  test("ClickHouse: asks for the URL and the key, then runs pgai mon deploy", async () => {
    await withApi(async (env, calls) => {
      const r = await runTty(["init"], env, [[URL_PROMPT, `${CH}\r`], [KEY_PROMPT, "\r"]]);
      expect(r.status).toBe(0);
      expect(r.screen).toBe([
        URL_PROMPT,
        KEY_PROMPT,
        `Monitoring for ${NAME} is ready.`,
        `  Health Matrix: ${HEALTH}`,
        `  Grafana:       ${GRAFANA}`,
        "",
      ].join("\n"));
      expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}", ...LOOKUPS]);
    });
  });

  test("a provider without a key: asks only for the URL", async () => {
    const url = "postgresql://postgres:pw@db.example.com:5432/app";
    await withApi(async (env) => {
      const r = await runTty(["init"], env, [[URL_PROMPT, `${url}\r`]]);
      expect(r.status).toBe(0);
      expect(r.screen).toStartWith(`${URL_PROMPT}\nMonitoring for db.example.com/app is ready.\n  Health Matrix: ${HEALTH}\n`);
      // The URL carries the admin password: it is not shown as typed.
      expect(r.screen).not.toContain(":pw@");
      expect(r.screen).not.toContain("ClickHouse");
    }, [{ ...ROW, name: "db.example.com/app", provider: "self-managed" }]);
  });

  test("Ctrl-C or Ctrl-D at a prompt: nothing connected, exit 130", async () => {
    await withApi(async (env, calls) => {
      for (const key of ["\x03", "\x04"]) expect((await runTty(["init"], env, [[URL_PROMPT, key]])).status).toBe(130);
      expect(calls).toEqual([]);
    });
  });

  test("Ctrl-C while waiting for the box ends the run at once (the prompt no longer holds the terminal)", async () => {
    await withApi(async (env) => {
      const started = Date.now();
      const r = await runTty(["init"], env, [[URL_PROMPT, "postgresql://postgres:pw@db.example.com:5432/app\r"], [/box: starting/, "\x03"]]);
      expect(Date.now() - started).toBeLessThan(4000);
      expect(r.status).not.toBe(0);
    }, [{ ...ROW, name: "db.example.com/app", provider: "self-managed", status: "launch_requested", dashboard_url: null }]);
  });

  test("the ClickHouse key is not shown as it is typed (and is checked: ClickHouse rejects this one)", async () => {
    const ch = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("", { status: 401 }) });
    try {
      await withApi(async (env) => {
        const r = await runTty(["init"], { ...env, CLICKHOUSE_API_URL: `http://127.0.0.1:${ch.port}` }, [[URL_PROMPT, `${CH}\r`], [KEY_PROMPT, "kid:Sec4b1dTestSecret\r"]]);
        expect(r.status).toBe(3);
        expect(r.screen).toStartWith(`${URL_PROMPT}\n${KEY_PROMPT}\nClickHouse Cloud rejected the API key (401). Check the key id and secret.`);
        expect(r.screen).not.toContain("Sec4b1d");
      });
    } finally {
      ch.stop(true);
    }
  });

  test("--json with a global token and no org: still points to pgai mon deploy", async () => {
    const r = await runTty(["init", "--json"], { PGAI_API_KEY: `pai_global_${"a".repeat(43)}` }, []);
    expect(r.status).toBe(3);
    expect(JSON.parse(r.screen)).toEqual(TO_CONNECT);
  });

  test("a bad key goes through pgai mon deploy's own check", async () => {
    await withApi(async (env) => {
      const r = await runTty(["init"], env, [[URL_PROMPT, `${CH}\r`], [KEY_PROMPT, "nocolon\r"]]);
      expect(r.status).toBe(1);
      expect(r.screen).toContain(`${KEY_PROMPT}\n--clickhouse-key must be <key-id>:<key-secret>\n`);
    });
  });
});

describe("pgai mon deploy / mon instances list / status / delete", () => {
  test("not signed in and not interactive: action required, exit 3", async () => {
    const r = await run(["mon", "deploy", CH], {});
    expect(r.status).toBe(3);
    expect(r.json()).toEqual({ status: "action_required", provider: "clickhouse", name: NAME, next: "Sign in: pgai auth login (agents: set PGAI_API_KEY), then re-run" });
    expect(r.stdout + r.stderr).not.toContain("adminpw");
  });

  test("not a URL: failed, exit 1", async () => {
    const r = await run(["mon", "deploy", "host=db dbname=app"], {});
    expect(r.status).toBe(1);
    expect(r.json().next).toBe("Pass a URL: pgai mon deploy postgresql://user:password@host:5432/dbname");
  });

  test("already monitored: the status, exit 0, no database touched", async () => {
    await withApi(async (env, calls) => {
      const r = await run(["mon", "deploy", CH], env);
      expect(r.status).toBe(0);
      expect(r.json()).toEqual({ status: "ready", provider: "clickhouse", name: NAME, id: "i-1", health_url: HEALTH, dashboard_url: GRAFANA, host_metrics: true, next: `Open ${HEALTH}` });
      expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}", ...LOOKUPS]);
      expect(r.stdout + r.stderr).not.toContain("adminpw");
    });
  });

  test("a retry whose existing box is removed for billing: action required, exit 3", async () => {
    let listed = 0;
    await withApi(async (env, calls) => {
      const r = await run(["mon", "deploy", CH, "--json"], env);
      expect(r.status).toBe(3);
      expect(r.json()).toEqual({
        status: "action_required", provider: "clickhouse", name: NAME, id: "i-1",
        next: "The first charge failed (Your card was declined.): the box is being removed and nothing is billed. Update the payment method, then re-run",
      });
      expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}", "/rpc/cloud_monitoring_list test-key {}"]);
    }, () => [{ ...ROW, status: listed++ ? "deleting_launched" : "registered", billing_error: "Your card was declined." }]);
  }, 20_000);

  test("mon instances list and status", async () => {
    await withApi(async (env) => {
      // The same words as mon instances status and mon deploy: the platform's "active" is "ready".
      const list = (await run(["mon", "instances", "list"], env)).json();
      expect(list).toEqual([{ ...ROW, status: "ready", health_url: HEALTH, dashboard_url: GRAFANA }]);
      expect(Object.keys(list[0]).indexOf("health_url")).toBe(Object.keys(list[0]).indexOf("dashboard_url") - 1);
      // watch (status is its other name) of a ready instance: shown at once, by name or by id.
      for (const ref of [NAME, "i-1"]) {
        const s = await run(["mon", "instances", "status", ref], env);
        expect(s.status).toBe(0);
        expect(s.json()).toEqual({ status: "ready", provider: "clickhouse", name: NAME, id: "i-1", health_url: HEALTH, dashboard_url: GRAFANA, host_metrics: true, next: `Open ${HEALTH}` });
      }
      const missing = await run(["mon", "instances", "watch", "nope"], env);
      expect(missing.status).toBe(1);
      expect(missing.json()).toEqual({ status: "failed", next: "No instance nope. See: pgai mon instances list" });
    });
  });

  test("an API error is a failed result in JSON, exit 1, for every command", async () => {
    const env = { PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: "http://127.0.0.1:9" };
    for (const args of [["mon", "instances", "list"], ["mon", "instances", "status", NAME], ["mon", "instances", "delete", NAME, "--yes"], ["mon", "deploy", CH]]) {
      const r = await run(args, env);
      expect(r.status).toBe(1);
      expect(r.json().status).toBe("failed");
      expect(r.json().next).not.toBe("");
    }
  });

  test("a re-run with a wrong --clickhouse-key: action required, exit 3, not connected", async () => {
    const ch = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("", { status: 401 }) });
    try {
      await withApi(async (env, calls) => {
        const r = await run(["mon", "deploy", CH, "--clickhouse-key", "kid:wrong"], { ...env, CLICKHOUSE_API_URL: `http://127.0.0.1:${ch.port}` });
        expect(r.status).toBe(3);
        expect(r.json()).toEqual({
          status: "action_required", provider: "clickhouse", name: NAME, id: "i-1",
          next: "ClickHouse Cloud rejected the API key (401). Check the key id and secret. Nothing was changed: re-run with the right key, or without one",
        });
        expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}"]);
      });
    } finally {
      ch.stop(true);
    }
  });

  test("mon instances delete needs --yes when not interactive", async () => {
    await withApi(async (env, calls) => {
      const asked = await run(["mon", "instances", "delete", NAME], env);
      expect(asked.status).toBe(3);
      expect(asked.json().next).toBe("pgai mon instances delete i-1 --yes");
      expect(calls.some((c) => c.includes("disconnect"))).toBe(false);

      const done = await run(["mon", "instances", "delete", NAME, "--yes"], env);
      expect(done.status).toBe(0);
      expect(done.json()).toEqual({ status: "deleting", provider: "clickhouse", name: NAME, id: "i-1", next: "Delete the ClickHouse Cloud API key you gave us" });
      expect(calls.at(-1)).toBe('/rpc/cloud_monitoring_disconnect test-key {"instance_id":"i-1"}');
    });
  });

  test("mon instances delete --json with TTY stdin does not prompt and prints valid JSON", async () => {
    await withApi(async (env, calls) => {
      const r = await runTty(["mon", "instances", "delete", NAME, "--json"], env, [[`Stop monitoring ${NAME} and delete its monitoring box? (y/N): `, "n\r"]]);
      expect(r.screen).not.toContain("(y/N)");
      expect(r.status).toBe(3);
      expect(JSON.parse(r.screen)).toEqual({ status: "action_required", provider: "clickhouse", name: NAME, id: "i-1", next: "pgai mon instances delete i-1 --yes" });
      expect(calls).toEqual(["/rpc/cloud_monitoring_list test-key {}"]);
    });
  });

  test("an unusable --wait is refused before anything is touched", async () => {
    const r = await run(["mon", "deploy", CH, "--wait", "soon"], { PGAI_API_KEY: "k", PGAI_API_BASE_URL: "http://127.0.0.1:9" });
    expect(r.status).toBe(1);
    expect(r.json().next).toBe("--wait must be a number of minutes (0 = do not wait)");
  });

  test("a failed result names the provider of --provider, not the host's", async () => {
    const url = "postgresql://postgres:pw@10.0.0.5:5432/app";
    const env = { PGAI_API_KEY: "k", PGAI_API_BASE_URL: "http://127.0.0.1:9" };
    for (const flags of [["--clickhouse-key", "nocolon"], ["--wait", "soon"]]) {
      const r = await run(["mon", "deploy", url, "--provider", "clickhouse", ...flags], env);
      expect(r.status).toBe(1);
      expect(r.json()).toMatchObject({ status: "failed", provider: "clickhouse", name: "10.0.0.5/app" });
    }
    expect((await run(["mon", "deploy", url, "--provider", "oracle"], env)).json()).toEqual({
      status: "failed", provider: "self-managed", name: "10.0.0.5/app", next: "--provider must be one of: clickhouse, rds, supabase, self-managed",
    });
  });

  test("mon instances delete picks the live instance, not an earlier one still being deleted", async () => {
    await withApi(async (env, calls) => {
      expect((await run(["mon", "instances", "delete", NAME, "--yes"], env)).status).toBe(0);
      expect(calls.at(-1)).toBe('/rpc/cloud_monitoring_disconnect test-key {"instance_id":"i-1"}');
    }, [{ ...ROW, id: "i-0", status: "deleting_launched" }, ROW]);
  });
});
