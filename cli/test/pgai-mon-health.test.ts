import { afterAll, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { probePgaiMon } from "../lib/pgai-mon-health";
import { createCliSandbox } from "./cli-sandbox";

const TOKEN = "f00dfeedf00dfeedf00dfeedf00dfeed";

/** A stand-in for pgai-mon on 127.0.0.1: /livez open, /health behind the bearer token. */
function fakePgaiMon(opts: { livez?: number } = {}) {
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
        return Response.json({ files: 12, series_known: 340, samples_in: 5600 });
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
