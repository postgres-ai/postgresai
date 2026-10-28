import { describe, expect, test } from "bun:test";
import { HttpRequestTimeoutError } from "../lib/util";
import { connect, connectStatus, listedDatabase, resultLines, stateOf, stepProgress, stepViewEnd, type ConnectDeps, type Database, type ProgressEvent, type StepViewLike } from "../lib/connect";

// platform-all#987: the monitoring box's IP before the box. When PostgresAI can't reach the
// database the way an IP allowlist refuses (or with --ip-first), the platform reserves the
// box's IP and deploy shows it (step allow_ip); --ip-allowed, or Enter at a terminal, creates
// the box with it. Every outside effect is faked and recorded.

const URL = "postgresql://postgres_ai_mon:pw@db.example.com:5432/app?sslmode=require";
const NAME = "db.example.com/app";
const QUOTE = {
  plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 1 }, subscription: false, quantity: 0,
  price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false,
};

const box = (status: string, extra: Partial<Database> = {}): Database => ({
  id: "i-1", name: NAME, provider: "self-managed", status, cluster: "db.example.com:5432", database: "app",
  dashboard_url: status === "active" ? "https://abc.pgai.watch" : null, host_metrics: false, ...extra,
});
const RESERVED = box("ip_reserved", { egress_ip: "198.51.100.7", reserved_until: "2026-10-29T12:00:00Z", cloud: "hetzner", location: "nbg1",
  precheck_error: 'connection to server at "db.example.com" (192.0.2.10), port 5432 failed: timeout expired' });

function fake(over: Partial<ConnectDeps> & { lists?: Database[][]; creates?: { id: string; name: string; status: string; error?: string; billing_warning?: string }[] } = {}) {
  const calls: string[] = [];
  const events: ProgressEvent[] = [];
  const lists = over.lists ?? [[]];
  const creates = over.creates ?? [{ id: "i-1", name: NAME, status: "launch_requested" }];
  let l = 0;
  let c = 0;
  const deps: ConnectDeps = {
    list: async () => { calls.push("list"); return lists[Math.min(l++, lists.length - 1)]; },
    create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); return creates[Math.min(c++, creates.length - 1)]; },
    prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL }),
    unprepare: async () => true,
    localStackRunning: () => false,
    clickhouseOrg: async () => ({ orgId: "x", state: "running" }),
    selfHosted: async () => {},
    supabaseStart: async () => { throw new Error("no"); },
    supabaseStatus: async () => { throw new Error("no"); },
    supabaseConnect: async () => { throw new Error("no"); },
    quote: async () => QUOTE,
    billingUrl: (a) => `https://console.postgres.ai/${a}/billing`,
    confirm: async () => false,
    resetLock: async () => ({ lock_id: "l" }),
    resetUnlock: async () => {},
    sleep: async () => { calls.push("sleep"); },
    now: () => Date.now(),
    progress: (e) => events.push(e),
    ...over,
  };
  return { deps, calls, events };
}

const creates = (calls: string[]) => calls.filter((x) => x.startsWith("create ")).map((x) => JSON.parse(x.slice(7)));

describe("allow_ip (platform-all#987)", () => {
  test("every deploy asks the platform to reserve the IP instead of refusing an allowlisted database", async () => {
    const { deps, calls } = fake({ lists: [[], [box("active", { egress_ip: "203.0.113.5" })]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(creates(calls)).toEqual([{ db_url: URL, allow_ip_fallback: true }]);
    expect(r.status).toBe("ready");
    expect(r.egress_ip).toBe("203.0.113.5");
    expect(resultLines(r)).toContain("  Connects from: 203.0.113.5");
  });

  test("an agent: the reserved IP comes back as step allow_ip (exit 3), nothing else is created", async () => {
    const { deps, calls, events } = fake({
      creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }],
      lists: [[], [box("ip_reserving")], [RESERVED]],
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, agent: true }, deps);
    expect(creates(calls)).toHaveLength(1);
    expect(r).toMatchObject({
      status: "action_required", step: "allow_ip", id: "i-1", ip: "198.51.100.7", cidr: "198.51.100.7/32",
      host: "db.example.com", port: 5432, reserved_until: "2026-10-29T12:00:00Z", cloud: "hetzner", location: "nbg1",
    });
    expect(r.next).toBe("Allow inbound TCP 5432 from 198.51.100.7/32 on db.example.com, then call deploy_monitoring again with ip_allowed: true");
    expect(events.find((e) => e.event === "allow_ip")).toMatchObject({ ip: "198.51.100.7", cidr: "198.51.100.7/32", port: 5432 });
  });

  test("a script without a terminal: the same stop, with the command to continue (the password masked)", async () => {
    const { deps } = fake({ creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }], lists: [[], [RESERVED]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(r.step).toBe("allow_ip");
    expect(r.next).toContain("re-run with --yes --ip-allowed: pgai mon deploy");
    expect(r.next).not.toContain(":pw@");
    expect(resultLines(r)[0]).toBe("Monitoring box IP: 198.51.100.7 (Hetzner, nbg1), held until 2026-10-29 12:00 UTC");
  });

  test("a person at a terminal continues: the reservation gets its box, then deploy follows it to ready", async () => {
    let shown: string[] = [];
    const { deps, calls } = fake({
      creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }, { id: "i-1", name: NAME, status: "launch_requested" }],
      lists: [[], [RESERVED], [box("waiting_for_access", { egress_ip: "198.51.100.7", last_error: "timeout expired" })], [box("active", { egress_ip: "198.51.100.7" })]],
      allowIp: async (lines) => { shown = lines; return "continue"; },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(shown[0]).toBe("Monitoring box IP: 198.51.100.7 (Hetzner, nbg1)");
    expect(shown[1]).toContain("PostgresAI could not connect to db.example.com:5432 (connection to server");
    expect(shown[2]).toContain("Allow inbound TCP 5432 from 198.51.100.7/32 on db.example.com, then continue.");
    expect(creates(calls)).toEqual([
      { db_url: URL, allow_ip_fallback: true },
      { db_url: URL, allow_ip_fallback: true, ip_allowed: true },
    ]);
    expect(r).toMatchObject({ status: "ready", egress_ip: "198.51.100.7" });
  });

  test("the box waits for database access: the step says from which IP and why", async () => {
    const { deps, events } = fake({
      creates: [{ id: "i-1", name: NAME, status: "launch_requested" }],
      lists: [[], [box("waiting_for_access", { egress_ip: "198.51.100.7", last_error: "timeout expired" })], [box("active")]],
    });
    await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(events.map((e) => e.message)).toContain("Monitoring box: waiting for database access from 198.51.100.7 (timeout expired)");
  });

  test("[s]top at the prompt: nothing more is created; the IP stays reserved", async () => {
    const { deps, calls } = fake({
      creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }],
      lists: [[], [RESERVED]],
      allowIp: async () => "stop",
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(creates(calls)).toHaveLength(1);
    expect(r.next).toMatch(/^Stopped\. 198\.51\.100\.7 stays reserved until 2026-10-29 12:00 UTC\. Allow inbound TCP 5432 from 198\.51\.100\.7\/32 on db\.example\.com, then: pgai mon deploy /);
  });

  test("a stop at allow_ip drops the role this run made with a generated password: the re-run makes it again", async () => {
    // Nobody but the platform has that password, and it sends it on over TLS only: kept, the
    // --ip-allowed re-run to a server without TLS would stop at "postgres_ai_mon already exists".
    for (const [generated, allowIp] of [[true, undefined], [true, async () => "stop" as const], [false, undefined]] as const) {
      const { deps, calls } = fake({
        creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }],
        lists: [[], [RESERVED]],
        prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, ...(generated ? { generated: true as const } : {}) }),
        unprepare: async () => { calls.push("unprepare"); return true; },
        ...(allowIp ? { allowIp } : {}),
      });
      const r = await connect(URL, { waitMs: 60_000, yes: true, agent: !allowIp }, deps);
      expect(r.step).toBe("allow_ip");
      expect(calls.filter((x) => x === "unprepare")).toHaveLength(generated ? 1 : 0);
    }
  });

  test("a stop at allow_ip is not a failure on the step list: the step is left waiting, not marked ✗", () => {
    expect(stepViewEnd({ status: "action_required", step: "allow_ip", provider: "self-managed", name: NAME, next: "Allow …" })).toBe("stop");
    expect(stepViewEnd({ status: "action_required", provider: "self-managed", name: NAME, next: "Add a payment method" })).toBe("fail");
    expect(stepViewEnd({ status: "ready", provider: "self-managed", name: NAME, next: "none" })).toBe("complete");
    expect(stepViewEnd({ status: "in_progress", provider: "self-managed", name: NAME, next: "pgai mon deploy …" })).toBe("stop");
  });

  test("a re-run of an IP asked for first (no pre-check error) does not say PostgresAI could not connect", async () => {
    let shown: string[] = [];
    const { deps } = fake({ lists: [[{ ...RESERVED, precheck_error: null }]], allowIp: async (lines) => { shown = lines; return "stop"; } });
    await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(shown[1]).toBe("The IP was reserved first, as asked.");
  });

  test("a re-run before the IP is allowed shows the same reservation: no second IP, no create", async () => {
    const { deps, calls } = fake({ lists: [[RESERVED]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(creates(calls)).toEqual([]);
    expect(r).toMatchObject({ step: "allow_ip", id: "i-1", ip: "198.51.100.7" });
  });

  test("a re-run with --ip-allowed continues the same reservation", async () => {
    const { deps, calls } = fake({ lists: [[RESERVED], [box("active", { egress_ip: "198.51.100.7" })]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(creates(calls)).toEqual([{ db_url: URL, allow_ip_fallback: true, ip_allowed: true }]);
    expect(r.status).toBe("ready");
  });

  test("--ip-first asks for the IP first even when PostgresAI can reach the database", async () => {
    const { deps, calls } = fake({ creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }], lists: [[], [RESERVED]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipFirst: true }, deps);
    expect(creates(calls)).toEqual([{ db_url: URL, allow_ip_fallback: true, ip_first: true }]);
    expect(r.step).toBe("allow_ip");
  });

  test("the box was refused until its wait ended: back to allow_ip, with the error, the same IP", async () => {
    const { deps } = fake({
      lists: [[RESERVED], [box("waiting_for_access", { egress_ip: "198.51.100.7" })],
        [box("ip_reserved", { egress_ip: "198.51.100.7", last_error: "timeout expired", reserved_until: "2026-10-30T00:00:00Z" })]],
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7", last_error: "timeout expired" });
    expect(resultLines(r)).toContain("The box could not connect: timeout expired");
  });

  test("two runs continue at once: the second follows the box the first created", async () => {
    const { HttpStatusError } = await import("../lib/util");
    const { deps } = fake({
      lists: [[RESERVED], [box("launch_requested")], [box("active")]],
      create: async () => { throw new HttpStatusError("cloud monitoring connect: HTTP 409 - Conflict\nThis database is already monitored as db.example.com/app.", 409); },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(r.status).toBe("ready");
  });

  test("two runs continue at once and the other made the role: this one follows that box, it asks for no password", async () => {
    // The stop at allow_ip dropped the role, so both --ip-allowed runs make it again; the
    // second sees it "created by someone else meanwhile" (rig story C15).
    const { deps, calls } = fake({
      lists: [[RESERVED], [RESERVED], [box("launch_requested")], [box("active", { egress_ip: "198.51.100.7" })]],
      prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { next: "postgres_ai_mon was created by someone else meanwhile. Set PGAI_MON_PASSWORD to the password of postgres_ai_mon" }),
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(creates(calls)).toHaveLength(0);
    expect(r).toMatchObject({ status: "ready", egress_ip: "198.51.100.7" });
    // Not continuing a reservation: the refusal stands.
    const other = fake({ lists: [[]], prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { next: "postgres_ai_mon was created by someone else meanwhile. Set PGAI_MON_PASSWORD" }) });
    expect((await connect(URL, { waitMs: 60_000, yes: true }, other.deps)).next).toMatch(/created by someone else meanwhile/);
  });

  test("--ip-first is for a database URL in PostgresAI Cloud: refused with --self-hosted", async () => {
    const { deps, calls } = fake();
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipFirst: true, selfHosted: true }, deps);
    expect(r.next).toBe("--ip-first and --ip-allowed apply to PostgresAI Cloud monitoring of a database URL only");
    expect(creates(calls)).toEqual([]);
  });
});

describe("allow_ip on the wire (platform-all#987)", () => {
  const { handleToolCall } = require("../lib/mcp-server");
  const { platformDeps } = require("../lib/connect");

  test("MCP deploy_monitoring: a reserved database comes back as step allow_ip, with the next call", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(req) {
        if (new globalThis.URL(req.url).pathname.endsWith("/rpc/cloud_monitoring_list")) return Response.json([RESERVED]);
        return new Response("not found", { status: 404 });
      },
    });
    try {
      const reply = await handleToolCall({ params: { name: "deploy_monitoring", arguments: { database_url: URL } } },
        { apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" });
      const r = JSON.parse(reply.content[0].text);
      expect(reply.isError).toBe(false);
      expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7", cidr: "198.51.100.7/32", port: 5432 });
      expect(r.next).toContain("call deploy_monitoring again with ip_allowed: true");
    } finally {
      server.stop(true);
    }
  });

  test("an older platform without allow_ip: re-sent without it; --ip-first/--ip-allowed are refused in words", async () => {
    const bodies: Record<string, unknown>[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const body = (await req.json()) as Record<string, unknown>;
        bodies.push(body);
        if ("allow_ip_fallback" in body || "ip_first" in body || "ip_allowed" in body) {
          return Response.json({ code: "PGRST202", message: "Could not find the v1.cloud_monitoring_connect(allow_ip_fallback, db_url) function in the schema cache" }, { status: 404 });
        }
        return Response.json({ id: "i-1", name: NAME, status: "launch_requested" });
      },
    });
    try {
      const deps = platformDeps({ apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" });
      expect(await deps.create({ db_url: URL, allow_ip_fallback: true })).toMatchObject({ id: "i-1", status: "launch_requested" });
      expect(bodies).toEqual([{ db_url: URL, allow_ip_fallback: true }, { db_url: URL }]);
      await expect(deps.create({ db_url: URL, allow_ip_fallback: true, ip_first: true }))
        .rejects.toThrow("This PostgresAI platform does not support --ip-first or --ip-allowed yet: deploy without them");
    } finally {
      server.stop(true);
    }
  });
});

describe("allow_ip, what a person and an agent read (platform-all#987 design review)", () => {
  test("a reservation is action_required with step allow_ip in list and watch, not in_progress", () => {
    expect(stateOf("ip_reserved")).toBe("action_required");
    expect(stateOf("ip_reserving")).toBe("in_progress");
    expect(stateOf("waiting_for_access")).toBe("in_progress");
    const r = connectStatus(RESERVED);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7", cidr: "198.51.100.7/32", host: "db.example.com", port: 5432 });
    expect(r.next).toBe("Allow inbound TCP 5432 from 198.51.100.7/32 on db.example.com, then re-run with --yes --ip-allowed: pgai mon deploy <database URL> --yes --ip-allowed");
  });

  test("instances list: a reservation's row carries step allow_ip next to action_required", () => {
    expect(listedDatabase(RESERVED)).toMatchObject({ status: "action_required", step: "allow_ip", egress_ip: "198.51.100.7" });
    expect(listedDatabase(box("active"))).toMatchObject({ status: "ready" });
    expect(listedDatabase(box("active")).step).toBeUndefined();
  });

  test("the allow_ip result says why, piped too, and carries precheck_error", async () => {
    const { deps } = fake({ creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }], lists: [[], [RESERVED]] });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(r.precheck_error).toBe(RESERVED.precheck_error ?? undefined);
    expect(resultLines(r)[1]).toBe(`PostgresAI could not connect to db.example.com:5432 (${RESERVED.precheck_error}).`);
  });

  test("after [s]top the IP and its hold are said once", async () => {
    const { deps } = fake({ creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }], lists: [[], [RESERVED]], allowIp: async () => "stop" });
    const lines = resultLines(await connect(URL, { waitMs: 60_000, yes: true }, deps));
    expect(lines.filter((l) => l.includes("198.51.100.7 ") || l.startsWith("Monitoring box IP"))).toHaveLength(1);
    expect(lines.join("\n").match(/2026-10-29 12:00 UTC/g)).toHaveLength(1);
  });

  test("the terminal prompt names the database host", async () => {
    let shown: string[] = [];
    const { deps } = fake({ creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }], lists: [[], [RESERVED]], allowIp: async (l) => { shown = l; return "stop"; } });
    await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(shown[2]).toContain("Allow inbound TCP 5432 from 198.51.100.7/32 on db.example.com, then continue.");
  });

  test("at a terminal, waiting for access: the note names the IP and the database's refusal is shown once", () => {
    const notes: string[] = [];
    const logs: string[] = [];
    const view: StepViewLike = { start() {}, advance() {}, log: (t) => logs.push(t), setNote: (n) => notes.push(n), complete() {}, fail() {}, stop() {} };
    const p = stepProgress({ json: false, tty: true, makeView: () => view, writeLine: () => {}, writeEvent: () => {} });
    p.progress({ event: "billing", elapsed_s: 0, message: "Estimated cost: free" });
    const e = { event: "box", elapsed_s: 1, message: "Monitoring box: waiting for database access from 198.51.100.7 (timeout expired)",
      state: "waiting_for_access", ip: "198.51.100.7", error: "timeout expired" } as ProgressEvent;
    p.progress(e);
    p.progress(e);
    expect(notes.at(-1)).toBe("box: waiting for database access from 198.51.100.7");
    expect(logs).toEqual(["The database refused 198.51.100.7: timeout expired. The box keeps trying: allow 198.51.100.7/32 on the database."]);
  });
});

describe("allow_ip, REV round 1 (platform-all#987)", () => {
  test("--ip-first / --ip-allowed are refused for Supabase before anything is created", async () => {
    for (const opts of [{ ipFirst: true }, { ipAllowed: true }]) {
      const { deps, calls } = fake({ lists: [[]] });
      const r = await connect("postgresql://postgres:pw@db.abcdefghijklmnopqrst.supabase.co:5432/postgres", { waitMs: 60_000, yes: true, ...opts }, deps);
      expect(r.status).toBe("action_required");
      expect(r.next).toContain("apply to PostgresAI Cloud monitoring of a database URL only");
      expect(creates(calls)).toEqual([]);
    }
  });

  test("the allow_ip event names no box: Ctrl-C afterwards cancels, it does not detach", () => {
    const ids: string[] = [];
    const p = stepProgress({ json: true, tty: false, makeView: () => { throw new Error("no view"); }, writeLine: () => {}, writeEvent: () => {}, onId: (id) => ids.push(id) });
    p.progress({ event: "allow_ip", elapsed_s: 0, message: "Reserved IP for the monitoring box: 198.51.100.7", id: "i-1" } as ProgressEvent);
    expect(ids).toEqual([]);
    p.progress({ event: "box", elapsed_s: 1, message: "Monitoring box: starting", state: "launch_requested", id: "i-1" });
    expect(ids).toEqual(["i-1"]);
  });

  test("[s]top drops the role only while the reservation is still waiting (another run may have continued it)", async () => {
    for (const [after, dropped] of [[RESERVED, 1], [box("launch_requested", { egress_ip: "198.51.100.7" }), 0]] as const) {
      const { deps, calls } = fake({
        creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }],
        lists: [[], [RESERVED], [after]],
        prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, generated: true as const }),
        unprepare: async () => { calls.push("unprepare"); return true; },
        allowIp: async () => "stop",
      });
      await connect(URL, { waitMs: 60_000, yes: true }, deps);
      expect(calls.filter((x) => x === "unprepare")).toHaveLength(dropped);
    }
  });

  test("a followed box the database refused: back to allow_ip, and the role this run made is dropped", async () => {
    const { deps, calls } = fake({
      creates: [{ id: "i-1", name: NAME, status: "launch_requested" }],
      lists: [[], [box("waiting_for_access", { egress_ip: "198.51.100.7", last_error: "timeout expired" })], [{ ...RESERVED, last_error: "timeout expired" }]],
      prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, generated: true as const }),
      unprepare: async () => { calls.push("unprepare"); return true; },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, agent: true }, deps);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", last_error: "timeout expired" });
    expect(calls.filter((x) => x === "unprepare")).toHaveLength(1);
  });

  test("continuing a reservation: its own row is not another database on the server (stored password, --reset-password)", async () => {
    let seen: { storedPassword?: boolean; others?: string[] } | undefined;
    const { deps } = fake({
      lists: [[{ ...RESERVED, monitoring_password_stored: true }]],
      prepare: async (_u, _p, o) => { if (!o?.check) seen = o; return o?.check ? { checked: true } : { monitoringUrl: URL }; },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true, resetPassword: true }, deps);
    expect(r.next ?? "").not.toContain("would cut off");
    // The platform keeps the password for the reservation's own row: still used (over TLS).
    expect(seen?.storedPassword).toBe(true);
    expect(seen?.others ?? []).toEqual([]);
  });

  test("the database's refusal is cleaned of escape sequences before the terminal", () => {
    const logs: string[] = [];
    const notes: string[] = [];
    const view: StepViewLike = { start() {}, advance() {}, log: (t) => logs.push(t), setNote: (n) => notes.push(n), complete() {}, fail() {}, stop() {} };
    const p = stepProgress({ json: false, tty: true, makeView: () => view, writeLine: () => {}, writeEvent: () => {} });
    p.progress({ event: "billing", elapsed_s: 0, message: "Estimated cost: free" });
    p.progress({ event: "box", elapsed_s: 1, message: "Monitoring box: waiting", state: "waiting_for_access", ip: "198.51.100.7\x1b]0;x\x07", error: "FATAL: \x1b[2Jboom" } as ProgressEvent);
    expect(logs.join("") + notes.join("")).not.toContain("\x1b");
  });

  // platform-all#987 round 2: the platform answers a Continue whose box SI refused with the
  // reservation again and why; nothing to follow, and the Continue is not sent twice.
  const REFUSED = { id: "i-1", name: NAME, status: "ip_reserved",
    error: "The monitoring box could not be launched (HTTP 400). The IP stays reserved and nothing is billed; try again later." };
  test("--ip-allowed refused by the platform: the same IP and the platform's reason, one create, the role dropped", async () => {
    const { deps, calls } = fake({
      creates: [REFUSED],
      lists: [[RESERVED], [RESERVED]],
      prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, generated: true as const }),
      unprepare: async () => { calls.push("unprepare"); return true; },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true, agent: true }, deps);
    expect(creates(calls)).toHaveLength(1);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7", cidr: "198.51.100.7/32", host: "db.example.com" });
    expect(r.next).toContain("could not be launched (HTTP 400)");
    expect(calls.filter((x) => x === "unprepare")).toHaveLength(1);
  });

  test("Enter at the prompt, then refused by the platform: the person reads why, with the same IP", async () => {
    const { deps, calls } = fake({
      creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }, { ...REFUSED, billing_warning: "Releasing its billing failed." }],
      lists: [[], [RESERVED], [RESERVED]],
      allowIp: async () => "continue",
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(creates(calls)).toHaveLength(2);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7" });
    expect(r.next).toContain("could not be launched (HTTP 400)");
    expect(r.next).toContain("Releasing its billing failed.");
    expect(resultLines(r).join("\n")).not.toContain("undefined");
  });

  // A Continue whose request gets no answer (postgresai#443's lookup): the platform puts the box on
  // the reservation's own row, which reads reserved until the platform commits.
  const noAnswer = async () => { throw new HttpRequestTimeoutError("cloud monitoring connect", 25000); };
  // A clock that the deps' sleep moves on, so the lookup's 5-minute window ends.
  const clock = () => { let t = 0; return { now: () => t, sleep: async (ms: number) => { t += ms; } }; };
  test("--ip-allowed with no answer: the box on the reservation's own row is found once it leaves the reservation", async () => {
    const { deps } = fake({
      lists: [[RESERVED], [RESERVED], [box("launch_requested", { egress_ip: "198.51.100.7" })], [box("active", { egress_ip: "198.51.100.7" })]],
      create: noAnswer, ...clock(),
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(r.status).toBe("ready");
  });

  test("a first request with no answer while the platform reserves: the reserved IP is shown, the role dropped", async () => {
    const { deps, calls } = fake({
      lists: [[], [box("ip_reserving")], [RESERVED]],
      create: noAnswer, ...clock(),
      prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, generated: true as const }),
      unprepare: async () => { calls.push("unprepare"); return true; },
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, agent: true }, deps);
    expect(r).toMatchObject({ status: "action_required", step: "allow_ip", ip: "198.51.100.7" });
    expect(calls).toContain("unprepare");
  });

  test("Enter with no answer: the still-reserved row is not taken for the box; the role is kept for it", async () => {
    let c = 0;
    const { deps, calls } = fake({
      lists: [[], [RESERVED], [RESERVED], [box("launch_requested", { egress_ip: "198.51.100.7" })], [box("active", { egress_ip: "198.51.100.7" })]],
      create: async (b) => { calls.push(`create ${JSON.stringify(b)}`); if (c++ === 0) return { id: "i-1", name: NAME, status: "ip_reserving" }; return noAnswer(); },
      prepare: async (_u, _p, o) => (o?.check ? { checked: true } : { monitoringUrl: URL, generated: true as const }),
      unprepare: async () => { calls.push("unprepare"); return true; },
      allowIp: async () => "continue", ...clock(),
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true }, deps);
    expect(calls).not.toContain("unprepare");
    expect(r.status).toBe("ready");
    expect(JSON.stringify(r)).not.toContain("undefined");
  });

  test("--ip-allowed on a reservation still being reserved: the box is asked for once the IP is known", async () => {
    const { deps, calls } = fake({
      creates: [{ id: "i-1", name: NAME, status: "ip_reserving" }, { id: "i-1", name: NAME, status: "launch_requested" }],
      lists: [[box("ip_reserving")], [RESERVED], [box("active", { egress_ip: "198.51.100.7" })]],
    });
    const r = await connect(URL, { waitMs: 60_000, yes: true, ipAllowed: true }, deps);
    expect(creates(calls)).toHaveLength(2);
    expect(r.status).toBe("ready");
  });
});
