import { afterAll, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { judgeTargetTestFailure, probePgaiMon } from "../lib/pgai-mon-health";
import { createCliSandbox } from "./cli-sandbox";

const TOKEN = "f00dfeedf00dfeedf00dfeedf00dfeed";

/** A stand-in for pgai-mon on 127.0.0.1: /livez open, /health behind the bearer token. */
function fakePgaiMon(opts: { livez?: number; collector?: unknown } = {}) {
  const seen: { path: string; auth: string | null }[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(req) {
      const url = new URL(req.url);
      seen.push({ path: url.pathname, auth: req.headers.get("authorization") });
      if (url.pathname === "/livez") return new Response("ok", { status: opts.livez ?? 200 });
      if (url.pathname === "/health") {
        if (req.headers.get("authorization") !== `Bearer ${TOKEN}`) return new Response("unauthorized", { status: 401 });
        return Response.json({ files: 12, series_known: 340, samples_in: 5600, ...(opts.collector ? { collector: opts.collector } : {}) });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return { url: `http://127.0.0.1:${server.port}`, seen, stop: () => server.stop(true) };
}

describe("probePgaiMon", () => {
  test("up and serving: both lines green, /livez without credentials, /health with the token", async () => {
    const mon = fakePgaiMon();
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(true);
      expect(r.lines).toEqual(["✓ pgai-mon /livez: up", "✓ pgai-mon /health: 12 Parquet files, 340 series, 5600 samples in"]);
      expect(mon.seen).toEqual([
        { path: "/livez", auth: null },
        { path: "/health", auth: `Bearer ${TOKEN}` },
      ]);
    } finally {
      mon.stop();
    }
  });

  // The shape of /health's `collector` part (mon/src/collector.rs, health_json).
  const db = (up: boolean, last_error: string | null = null) => ({
    datname: "app", up, last_success_ms: up ? 1 : null, connect_failures: up ? 0 : 3, last_error, last_error_ms: null, jobs: {},
  });
  const collector = (bySource: Record<string, Record<string, unknown>>) => ({
    sources: Object.keys(bySource).length,
    by_source: Object.fromEntries(Object.entries(bySource).map(([name, databases]) => [name, { state: "running", databases }])),
  });

  test("every target up: one line with the count, still ok", async () => {
    const mon = fakePgaiMon({ collector: collector({ prod: { app: db(true), postgres: db(true) }, staging: { app: db(true) } }) });
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(true);
      expect(r.lines.slice(2)).toEqual(["✓ targets: 3/3 up"]);
    } finally {
      mon.stop();
    }
  });

  test("a target that is down: not ok, one line for it with its last error, password and token left out", async () => {
    const mon = fakePgaiMon({
      collector: collector({
        prod: { app: db(true), other: db(false, "connect: password authentication failed\nfor postgresql://mon:s3cret@db:5432/other") },
        refused: { app: db(false, "connect: connection refused") },
      }),
    });
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(false);
      expect(r.lines.slice(2)).toEqual([
        "✗ targets: 1/3 up",
        "✗ target prod/other: down (connect: password authentication failed for postgresql://mon:***@db:5432/other)",
        "✗ target refused/app: down (connect: connection refused)",
      ]);
      expect(r.lines.join("\n")).not.toContain("s3cret");
      expect(r.lines.join("\n")).not.toContain(TOKEN);
    } finally {
      mon.stop();
    }
  });

  test("right after start: a target with no attempt finished yet is connecting, not down, and does not fail", async () => {
    // pgai-mon sets last_error on every failed attempt; not up and no error means its first attempt has not finished.
    const fresh = { ...db(false), connect_failures: 0 };
    const mon = fakePgaiMon({ collector: collector({ prod: { app: db(true), new: fresh } }) });
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(true);
      expect(r.lines.slice(2)).toEqual([
        "✓ targets: 1/2 up, 1 connecting",
        "- target prod/new: connecting (first connection not finished yet)",
      ]);
    } finally {
      mon.stop();
    }
  });

  test("a failed target still fails the check while another is connecting", async () => {
    const fresh = { ...db(false), connect_failures: 0 };
    const mon = fakePgaiMon({ collector: collector({ prod: { new: fresh, bad: db(false, "connect: timeout") } }) });
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(false);
      expect(r.lines.slice(2)).toEqual([
        "✗ targets: 0/2 up, 1 connecting",
        "✗ target prod/bad: down (connect: timeout)",
        "- target prod/new: connecting (first connection not finished yet)",
      ]);
    } finally {
      mon.stop();
    }
  });

  test("no target known yet, or a /health without the collector part: no target line", async () => {
    const empty = fakePgaiMon({ collector: collector({}) });
    try {
      const r = await probePgaiMon(empty.url, TOKEN);
      expect(r.ok).toBe(true);
      expect(r.lines.length).toBe(2);
    } finally {
      empty.stop();
    }
  });

  test("a wrong token is reported as rejected, and the token is never in the output", async () => {
    const mon = fakePgaiMon();
    try {
      const r = await probePgaiMon(mon.url, "wrong-token-value");
      expect(r.ok).toBe(false);
      expect(r.lines[1]).toContain("token rejected (HTTP 401)");
      expect(r.lines.join("\n")).not.toContain("wrong-token-value");
    } finally {
      mon.stop();
    }
  });

  test("no token: /health is not asked and the line says why", async () => {
    const mon = fakePgaiMon();
    try {
      const r = await probePgaiMon(mon.url, undefined);
      expect(r.ok).toBe(false);
      expect(r.lines[1]).toBe("✗ pgai-mon /health: no PGAI_MON_TOKEN in .env");
      expect(mon.seen.map((s) => s.path)).toEqual(["/livez"]);
    } finally {
      mon.stop();
    }
  });

  test("a failing /livez fails the probe", async () => {
    const mon = fakePgaiMon({ livez: 503 });
    try {
      const r = await probePgaiMon(mon.url, TOKEN);
      expect(r.ok).toBe(false);
      expect(r.lines[0]).toBe("✗ pgai-mon /livez: HTTP 503");
    } finally {
      mon.stop();
    }
  });

  test("nothing listening: unreachable, not an exception", async () => {
    const mon = fakePgaiMon();
    const url = mon.url;
    mon.stop();
    const r = await probePgaiMon(url, TOKEN, 1000);
    expect(r.ok).toBe(false);
    expect(r.lines).toEqual([`✗ pgai-mon /livez: unreachable at ${url}`, `✗ pgai-mon /health: unreachable at ${url}`]);
  });
});

describe("judgeTargetTestFailure", () => {
  // What pg throws when the host name does not resolve (Node's getaddrinfo error).
  const dnsError = Object.assign(new Error("getaddrinfo EAI_AGAIN target-standby"), { code: "EAI_AGAIN" });
  const seen = (db: unknown) => ({ by_source: { standby: { state: "running", databases: { standby: db } } } });
  const failed = "✗ Connection failed: getaddrinfo EAI_AGAIN target-standby";
  const noDns = "- The host name of 'standby' does not resolve from this machine (getaddrinfo EAI_AGAIN target-standby)";

  test("pgai-mon reports the target up: ok, and says why the local test could not connect", () => {
    expect(judgeTargetTestFailure("standby", dnsError, seen({ up: true, last_error: null }))).toEqual({
      ok: true,
      lines: [noDns, "✓ pgai-mon, which connects from inside the Docker network, reports 'standby' up"],
    });
  });

  test("pgai-mon reports the target down: not ok, with pgai-mon's error, password left out", () => {
    const err = Object.assign(new Error("getaddrinfo ENOTFOUND target-standby"), { code: "ENOTFOUND" });
    const r = judgeTargetTestFailure("standby", err, seen({ up: false, last_error: "connect postgresql://u:s3cret@h/db: refused" }));
    expect(r.ok).toBe(false);
    expect(r.lines[1]).toBe("✗ pgai-mon reports 'standby' down: connect postgresql://u:***@h/db: refused");
  });

  test("a target pgai-mon does not know yet, or pgai-mon not reachable: the old failure and one line more", () => {
    const unconfirmed = { ok: false, lines: [failed, "The host name does not resolve from this machine, and pgai-mon could not confirm 'standby'"] };
    expect(judgeTargetTestFailure("standby", dnsError, { by_source: { other: { databases: { other: { up: true } } } } })).toEqual(unconfirmed);
    expect(judgeTargetTestFailure("standby", dnsError, undefined)).toEqual(unconfirmed);
  });

  test("any other connection error is reported as before, whatever pgai-mon says", () => {
    const refused = Object.assign(new Error("connect ECONNREFUSED 10.0.0.5:5432"), { code: "ECONNREFUSED" });
    expect(judgeTargetTestFailure("standby", refused, seen({ up: true }))).toEqual({
      ok: false,
      lines: ["✗ Connection failed: connect ECONNREFUSED 10.0.0.5:5432"],
    });
  });
});

describe("mon health", () => {
  const sandbox = createCliSandbox();
  afterAll(() => sandbox.cleanup());

  test("probes pgai-mon with the token from .env and never prints it", async () => {
    fs.writeFileSync(path.join(sandbox.projectDir, ".env"), `PGAI_MON_TOKEN=${TOKEN}\n`);
    const mon = fakePgaiMon();
    try {
      // Async spawn: the fake server runs in this process and must keep answering.
      const proc = Bun.spawn([process.execPath, sandbox.cliPath, "mon", "health"], {
        cwd: sandbox.projectDir,
        env: {
          PATH: `${sandbox.binDir}${path.delimiter}${process.env.PATH || ""}`,
          HOME: sandbox.home,
          XDG_CONFIG_HOME: sandbox.configHome,
          PGAI_MON_URL: mon.url,
        },
        stdout: "pipe",
        stderr: "pipe",
      });
      const [code, stdout, stderr] = await Promise.all([
        proc.exited,
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
      expect(stdout).toContain("✓ pgai-mon /livez: up");
      expect(stdout).toContain("✓ pgai-mon /health: 12 Parquet files");
      expect(stdout + stderr).not.toContain(TOKEN);
      expect(mon.seen.find((s) => s.path === "/health")?.auth).toBe(`Bearer ${TOKEN}`);
      // The fake docker reports no containers, so the command still fails.
      expect(code).toBe(1);
    } finally {
      mon.stop();
    }
  }, 30000);
});
