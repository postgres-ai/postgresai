import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Client } from "pg";

// Two `pgai connect --reset-password` processes at once, for two databases on
// one server. postgres_ai_mon is one role for the whole server, so without a
// lock each sets a new password and the box requested first gets a URL that
// no longer works. The platform's per-server lock (here, a fake API that
// keeps one) lets one through; the other changes nothing.
// CI: the cli:clickhouse-like:tests job (a server that checks passwords).

const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
const ADMIN = process.env.PGAI_TEST_CLICKHOUSE_LIKE_URL;

describe.skipIf(!ADMIN)("two pgai connect --reset-password at once (real Postgres)", () => {
  let home: string;
  const admin = async () => { const c = new Client({ connectionString: ADMIN }); await c.connect(); return c; };
  const otherDb = () => { const u = new URL(ADMIN!); u.pathname = "/pgai_reset_race_db2"; return u.toString(); };

  beforeAll(async () => {
    home = mkdtempSync(resolve(tmpdir(), "pgai-reset-race-"));
    const c = await admin();
    await c.query("drop database if exists pgai_reset_race_db2");
    await c.query("create database pgai_reset_race_db2");
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    // The role exists and its password is lost: what --reset-password is for.
    await c.query("create role postgres_ai_mon login password 'lost-password-0'");
    await c.end();
  });
  afterAll(async () => {
    const c = await admin();
    await c.query("drop database if exists pgai_reset_race_db2");
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    rmSync(home, { recursive: true, force: true });
  });

  test("one run resets and requests its box; the other is refused, and the first box's URL still works", async () => {
    let held = false;
    const created: string[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        const body = await req.json().catch(() => ({})) as Record<string, string>;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
        if (path.endsWith("/rpc/cloud_monitoring_reset_lock")) {
          if (held) return Response.json({ message: "Conflict", details: `Another pgai connect --reset-password for ${body.server} is running` }, { status: 409 });
          held = true;
          return Response.json({ lock_id: "l-1", server: body.server });
        }
        if (path.endsWith("/rpc/cloud_monitoring_reset_unlock")) { held = false; return Response.json({ released: true }); }
        if (path.endsWith("/rpc/cloud_monitoring_connect")) {
          created.push(body.db_url);
          // A box request takes a while: the other run arrives meanwhile.
          await Bun.sleep(1500);
          return Response.json({ id: `i-${created.length}`, name: "x", status: "launch_requested" });
        }
        if (path.endsWith("/rpc/checkup_report_create")) return Response.json({ report_id: 1 });
        if (path.endsWith("/rpc/checkup_report_file_post")) return Response.json({ report_chunck_id: 1 });
        if (path.endsWith("/rpc/checkup_report_status_update")) return Response.json({ report_id: 1 });
        return new Response("not found", { status: 404 });
      },
    });
    try {
      const run = (url: string) => {
        const proc = Bun.spawn([process.execPath, CLI, "connect", url, "--reset-password", "--json", "--wait", "0"], {
          env: { ...process.env, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${server.port}`, PGAI_MON_PASSWORD: "" },
          stdin: "ignore", stdout: "pipe", stderr: "pipe",
        });
        return Promise.all([proc.exited, new Response(proc.stdout).text()]).then(([status, stdout]) => ({ status, result: JSON.parse(stdout) as { status: string; next: string } }));
      };
      const runs = await Promise.all([run(ADMIN!), run(otherDb())]);
      const through = runs.filter((r) => r.status === 0);
      const refused = runs.filter((r) => r.status === 3);
      expect(through.length).toBe(1);
      expect(refused.length).toBe(1);
      expect(refused[0].result.next).toMatch(/^Another pgai connect --reset-password for \S+ is running: wait for it to finish, then re-run$/);
      expect(created.length).toBe(1);

      // The box's URL works: nobody changed the password after it was sent.
      const mon = new Client({ connectionString: created[0] });
      await mon.connect();
      expect((await mon.query("select current_user as u")).rows[0].u).toBe("postgres_ai_mon");
      await mon.end();
      expect(held).toBe(false);
    } finally {
      server.stop(true);
    }
  }, 60000);
});
