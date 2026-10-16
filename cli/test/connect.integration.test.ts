import { afterAll, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Client } from "pg";
import { connect, platformDeps, expressCheckup, prepareDatabase, unprepareDatabase } from "../lib/connect";

// `pgai mon deploy`'s prepare step against a real Postgres, with a superuser URL
// like the one ClickHouse Managed Postgres hands out. CI: the
// cli:clickhouse-like:tests job (PG 15, 17 and 18), whose server checks passwords
// (scram-sha-256) on every connection, so the refusals below are asserted there.
const ADMIN = process.env.PGAI_TEST_CLICKHOUSE_LIKE_URL;

describe.skipIf(!ADMIN)("prepareDatabase (real Postgres)", () => {
  let monUrlFromEarlierTest = "";
  const admin = () => { const c = new Client({ connectionString: ADMIN }); return c.connect().then(() => c); };

  afterAll(async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.query("drop owned by pgai_connect_creator cascade").catch(() => {});
    await c.query("drop role if exists pgai_connect_creator");
    await c.query("drop role if exists pgai_connect_login");
    await c.query("drop database if exists pgai_connect_db2");
    await c.end();
  });

  test("channel_binding=require refuses the real self-signed server certificate before anything is quoted or changed", async () => {
    const url = new URL(ADMIN!);
    url.searchParams.set("sslmode", "require");
    url.searchParams.set("channel_binding", "require");
    const calls: string[] = [];
    const deps = {
      ...platformDeps({ apiKey: "unused", apiBaseUrl: "http://unused.invalid", uiBaseUrl: "http://unused.invalid" }),
      list: async () => { calls.push("list"); return []; },
      quote: async () => { calls.push("quote"); throw new Error("must not quote"); },
      create: async () => { calls.push("create"); throw new Error("must not create"); },
      prepare: async () => { calls.push("prepare"); throw new Error("must not prepare"); },
      progress: () => {}, confirm: async () => false, localStackRunning: () => false, selfHosted: async () => {},
    };
    const result = await connect(url.toString(), { waitMs: 0 }, deps);
    expect(result.status).toBe("action_required");
    expect(result.next).toContain("sslrootcert=<CA file>");
    expect(result.next).toContain("sslmode=verify-full");
    expect(result.next).toContain("removing channel_binding=require from the URL");
    expect(calls).toEqual([]);
  });

  test("an admin URL creates the monitoring role; its URL connects to the prepared database as postgres_ai_mon", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    // No database in the path and credentials in the query string: the monitoring
    // URL must still name the prepared database and carry only the new role.
    const tricky = new URL(ADMIN!);
    tricky.pathname = "";
    tricky.searchParams.set("user", tricky.username);
    tricky.searchParams.set("password", decodeURIComponent(tricky.password));
    const first = await prepareDatabase(tricky.toString(), "self-managed");
    if (!("monitoringUrl" in first)) throw new Error(`expected a URL, got: ${JSON.stringify(first)}`);
    // The password was generated here, for a role this run created.
    expect(first.generated).toBe(true);
    monUrlFromEarlierTest = first.monitoringUrl;
    const mon = new URL(first.monitoringUrl);
    expect(mon.username).toBe("postgres_ai_mon");
    expect(mon.searchParams.has("user") || mon.searchParams.has("password")).toBe(false);
    expect(first.monitoringUrl).not.toContain(new URL(ADMIN!).password);
    const m = new Client({ connectionString: first.monitoringUrl });
    await m.connect();
    expect((await m.query("select current_user as u, current_database() as d")).rows[0]).toEqual({ u: "postgres_ai_mon", d: "postgres" });
    await m.end();

    expect(await prepareDatabase(first.monitoringUrl, "self-managed")).toEqual({ monitoringUrl: first.monitoringUrl });
    // The monitoring role's own URL with the password in the query string: normalized the same way.
    const monQuery = new URL(first.monitoringUrl);
    monQuery.password = "";
    monQuery.searchParams.set("password", decodeURIComponent(mon.password));
    expect(await prepareDatabase(monQuery.toString(), "self-managed")).toEqual({ monitoringUrl: first.monitoringUrl });

    // The role already exists: another database on this server may use its
    // password, so it is never rotated silently.
    const again = await prepareDatabase(ADMIN!, "self-managed");
    expect("monitoringUrl" in again).toBe(false);
    expect((again as { next: string }).next).toContain("PGAI_MON_PASSWORD");
    process.env.PGAI_MON_PASSWORD = "   ";
    try {
      expect("monitoringUrl" in await prepareDatabase(ADMIN!, "self-managed")).toBe(false);
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
    const still = new Client({ connectionString: first.monitoringUrl });
    await still.connect();
    await still.end();

    process.env.PGAI_MON_PASSWORD = decodeURIComponent(mon.password);
    try {
      expect(await prepareDatabase(ADMIN!, "self-managed")).toHaveProperty("monitoringUrl");
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
  });

  // postgres_ai_mon is cluster-wide: preparing a second database must never
  // change the password the first database's box uses.
  test("a second database on the server: a wrong PGAI_MON_PASSWORD is refused and the first box keeps working", async () => {
    const c = await admin();
    await c.query("drop database if exists pgai_connect_db2");
    await c.query("create database pgai_connect_db2");
    // A hardened database: the existing role may not connect until prepared.
    await c.query("revoke connect on database pgai_connect_db2 from public");
    await c.end();
    // The role already exists from the test above; its password is in that URL.
    const db2 = new URL(ADMIN!);
    db2.pathname = "/pgai_connect_db2";
    const verifier = async () => { const a = await admin(); const r = (await a.query("select rolpassword from pg_authid where rolname = 'postgres_ai_mon'")).rows[0].rolpassword; await a.end(); return r; };
    const before = await verifier();
    // A server that does not check passwords (trust) lets any password in;
    // the stored one must still stay as it is.
    const wrong = new URL(monUrlFromEarlierTest);
    wrong.password = "not-the-password";
    const w = new Client({ connectionString: wrong.toString() });
    const checksPasswords = await w.connect().then(() => w.end().then(() => false), () => true);
    // CI's server must check them (POSTGRES_HOST_AUTH_METHOD in .gitlab-ci.yml), or the refusal below is never asserted.
    if (process.env.CI) expect(checksPasswords).toBe(true);
    process.env.PGAI_MON_PASSWORD = "not-the-password";
    try {
      const refused = await prepareDatabase(db2.toString(), "self-managed");
      if (!checksPasswords) expect(refused).toHaveProperty("monitoringUrl");
      else expect(refused).toEqual({ next: "postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. Set PGAI_MON_PASSWORD to the password of postgres_ai_mon, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as postgres_ai_mon then needs the new password)" });
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
    const probe = new Client({ connectionString: monUrlFromEarlierTest });
    await probe.connect();
    await probe.end();

    process.env.PGAI_MON_PASSWORD = decodeURIComponent(new URL(monUrlFromEarlierTest).password);
    try {
      const second = await prepareDatabase(db2.toString(), "self-managed");
      if (!("monitoringUrl" in second)) throw new Error(`expected a URL, got: ${JSON.stringify(second)}`);
      const m = new Client({ connectionString: second.monitoringUrl });
      await m.connect();
      expect((await m.query("select current_database() as d")).rows[0].d).toBe("pgai_connect_db2");
      await m.end();
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
    // The existing role's password is not even re-set to the same value.
    expect(await verifier()).toBe(before);
    const still = new Client({ connectionString: monUrlFromEarlierTest });
    await still.connect();
    await still.end();
  });

  test("a second database with the password PostgresAI keeps for the server: over TLS, no PGAI_MON_PASSWORD and the role's password stays; without TLS, refused before anything runs, with the way out", async () => {
    const c = await admin();
    await c.query("drop database if exists pgai_connect_db2");
    await c.query("create database pgai_connect_db2");
    await c.query("revoke connect on database pgai_connect_db2 from public");
    const verifier = async () => (await c.query("select rolpassword from pg_authid where rolname = 'postgres_ai_mon'")).rows[0].rolpassword;
    const before = await verifier();
    const tls = (await c.query("show ssl")).rows[0].ssl === "on";
    // CI's server has TLS (cli:clickhouse-like:tests), so both halves run there.
    if (process.env.CI) expect(tls).toBe(true);
    const db2 = (sslmode: string) => { const u = new URL(ADMIN!); u.pathname = "/pgai_connect_db2"; u.searchParams.set("sslmode", sslmode); return u.toString(); };
    // The first database's password, on the second database, as the box will log in.
    const m = new URL(monUrlFromEarlierTest);
    m.pathname = "/pgai_connect_db2";
    const monLogsIn = () => { const mon = new Client({ connectionString: m.toString() }); return mon.connect().then(() => mon.end().then(() => true), () => false); };
    try {
      // sslmode=disable: on a server with TLS, the URL needs sslmode=require; on one without (ssl off), there is no TLS to ask for.
      expect(await prepareDatabase(db2("disable"), "self-managed", { storedPassword: true, others: ["db.example.com/first"] })).toEqual({
        next: tls
          ? "postgres_ai_mon already exists on this server. The URL needs sslmode=require (or verify-full): the password PostgresAI keeps for this server is sent only over TLS, to a URL with each parameter once"
          : "postgres_ai_mon already exists on this server, but this server takes no TLS, and PostgresAI sends the password it keeps for postgres_ai_mon only over TLS. Set PGAI_MON_PASSWORD to its password, or turn on TLS on the server (ssl = on) and put sslmode=require in the URL. If nobody has the password: pgai mon instances delete db.example.com/first --yes, then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one (and deploy db.example.com/first again with it)",
      });
      if (!tls) {
        // No sslmode: the session falls back to plaintext, which says the same.
        const prefer = new URL(db2("disable"));
        prefer.searchParams.delete("sslmode");
        expect(await prepareDatabase(prefer.toString(), "self-managed", { storedPassword: true })).toHaveProperty("next", expect.stringContaining("but this server takes no TLS"));
      }
      expect(await monLogsIn()).toBe(false);
      if (tls) {
        const second = await prepareDatabase(db2("require"), "self-managed", { storedPassword: true });
        if (!("monitoringUrl" in second)) throw new Error(`expected a URL, got: ${JSON.stringify(second)}`);
        expect(second.storedPassword).toBe(true);
        expect(new URL(second.monitoringUrl).password).toBe("");
        expect(await monLogsIn()).toBe(true);
      }
      expect(await verifier()).toBe(before);
    } finally {
      await c.end();
    }
  });

  test("a new role's password with '%' and spaces survives into the monitoring URL", async () => {
    const c = await admin();
    await c.query("drop database if exists pgai_connect_db2");
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    process.env.PGAI_MON_PASSWORD = "p%40ss w%rd";
    try {
      const result = await prepareDatabase(ADMIN!, "self-managed");
      if (!("monitoringUrl" in result)) throw new Error(`expected a URL, got: ${JSON.stringify(result)}`);
      const m = new Client({ connectionString: result.monitoringUrl });
      await m.connect();
      await m.end();
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
    }
  });

  test("a new role's verifier uses the server's scram_iterations (PG 16+)", async () => {
    const c = await admin();
    await c.query("drop database if exists pgai_connect_db2");
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    if ((await c.query("select current_setting('scram_iterations', true) as i")).rows[0].i === null) return void (await c.end());
    await c.query("alter role current_user set scram_iterations = 10000");
    try {
      if (!("monitoringUrl" in await prepareDatabase(ADMIN!, "self-managed"))) throw new Error("expected a URL");
      const stored = await c.query("select rolpassword from pg_authid where rolname = 'postgres_ai_mon'");
      expect(stored.rows[0].rolpassword.startsWith("SCRAM-SHA-256$10000:")).toBe(true);
    } finally {
      await c.query("alter role current_user reset scram_iterations");
      await c.end();
    }
  });

  test("a URL that can neither create roles nor is the monitoring role: the SQL, passwords redacted", async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.query("create role pgai_connect_app login password 'app-pw-123'");
    await c.end();
    const url = new URL(ADMIN!);
    url.username = "pgai_connect_app";
    url.password = "app-pw-123";
    const result = await prepareDatabase(url.toString(), "clickhouse");
    if (!("sql" in result) || !result.sql) throw new Error("expected SQL");
    expect(result.sql).toContain("-- 01.role");
    // Running the SQL must not change the password of an existing postgres_ai_mon.
    expect(result.sql).not.toMatch(/^[ \t]*alter user[^;\n]*password/im);
    expect(result.sql).toContain("password '<redacted>'");
    expect(result.sql).not.toContain("app-pw-123");
  });

  test("a postgres_ai_mon URL without a password (PGPASSWORD): the monitoring URL carries the password used", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    const first = await prepareDatabase(ADMIN!, "self-managed");
    if (!("monitoringUrl" in first)) throw new Error(`expected a URL, got: ${JSON.stringify(first)}`);
    const bare = new URL(first.monitoringUrl);
    process.env.PGPASSWORD = decodeURIComponent(bare.password);
    bare.password = "";
    try {
      expect(await prepareDatabase(bare.toString(), "self-managed")).toEqual({ monitoringUrl: first.monitoringUrl });
    } finally {
      delete process.env.PGPASSWORD;
    }
  });

  // A CREATEROLE role counts as admin only when it can run the whole plan; each
  // case below lacks one thing the plan needs (PG 15 needs no ADMIN OPTION to grant a role).
  describe("a CREATEROLE role (not a superuser)", () => {
    const creator = async (grants: string[]) => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      await c.query("drop owned by pgai_connect_creator cascade").catch(() => {});
      await c.query("drop role if exists pgai_connect_creator");
      await c.query("create role pgai_connect_creator login createrole password 'creator-pw-123'");
      await c.query("create extension if not exists pg_stat_statements");
      for (const sql of grants) await c.query(sql);
      const url = new URL(ADMIN!);
      url.username = "pgai_connect_creator";
      url.password = "creator-pw-123";
      const roles = async () => (await c.query("select count(*)::int as n from pg_roles where rolname = 'postgres_ai_mon'")).rows[0].n;
      return { c, url: url.toString(), roles, pg16: !(await c.query("select current_setting('server_version_num')::int < 160000 as old")).rows[0].old };
    };
    const ROLE_GRANTS = ["grant pg_monitor to pgai_connect_creator with admin option", "grant pg_read_all_stats to pgai_connect_creator with admin option"];
    const CREATE = "grant create on database postgres to pgai_connect_creator";

    test("without what the plan needs (ADMIN OPTION on both roles on PG 16+, CREATE on the database, pg_stat_statements): the SQL, and no role left behind", async () => {
      const { c, pg16 } = await creator([]);
      await c.end();
      // On PG 15 CREATEROLE grants any role, so only CREATE on the database can be missing there.
      const lacking = pg16 ? [[], [CREATE], [ROLE_GRANTS[0]!, CREATE], ROLE_GRANTS] : [[]];
      for (const grants of lacking) {
        const s = await creator(grants);
        try {
          expect(await prepareDatabase(s.url, "self-managed")).toHaveProperty("sql");
          expect(await s.roles()).toBe(0);
        } finally {
          await s.c.end();
        }
      }
      const s = await creator([...ROLE_GRANTS, CREATE]);
      try {
        await s.c.query("drop extension pg_stat_statements");
        expect(await prepareDatabase(s.url, "self-managed")).toHaveProperty("sql");
        expect(await s.roles()).toBe(0);
      } finally {
        await s.c.query("create extension if not exists pg_stat_statements");
        await s.c.end();
      }
    });

    test("with all of it: the role is created and its URL logs in", async () => {
      const s = await creator([...ROLE_GRANTS, CREATE]);
      try {
        const result = await prepareDatabase(s.url, "self-managed");
        if (!("monitoringUrl" in result)) throw new Error(`expected a URL, got: ${JSON.stringify(result)}`);
        expect(result.generated).toBe(true);
        const m = new Client({ connectionString: result.monitoringUrl });
        await m.connect();
        expect((await m.query("select pg_has_role('pg_monitor', 'member') as monitor, has_schema_privilege('postgres_ai', 'usage') as schema")).rows[0]).toEqual({ monitor: true, schema: true });
        await m.end();
      } finally {
        await s.c.end();
      }
    });

    test("passes the check but a later step fails (postgres_ai belongs to someone else): the error, and no role left behind", async () => {
      const s = await creator([...ROLE_GRANTS, CREATE, "create schema postgres_ai"]);
      try {
        await expect(prepareDatabase(s.url, "self-managed")).rejects.toThrow('Failed at step "03.permissions": permission denied for schema postgres_ai');
        expect(await s.roles()).toBe(0);
      } finally {
        await s.c.end();
      }
    });
  });

  test("a login role whose session runs as an admin role (session_user is not current_user): prepared as the admin", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.query("drop role if exists pgai_connect_login");
    await c.query("create role pgai_connect_login login password 'login-pw-123'");
    const adminRole = `"${decodeURIComponent(new URL(ADMIN!).username).replace(/"/g, '""')}"`;
    await c.query(`grant ${adminRole} to pgai_connect_login`);
    await c.query(`alter role pgai_connect_login set role to ${adminRole}`);
    try {
      const url = new URL(ADMIN!);
      url.username = "pgai_connect_login";
      url.password = "login-pw-123";
      const probe = new Client({ connectionString: url.toString() });
      await probe.connect();
      expect((await probe.query("select session_user::text as s, current_user::text as c")).rows[0]).toEqual({ s: "pgai_connect_login", c: new URL(ADMIN!).username });
      await probe.end();
      const result = await prepareDatabase(url.toString(), "self-managed");
      if (!("monitoringUrl" in result)) throw new Error(`expected a URL, got: ${JSON.stringify(result)}`);
      expect(new URL(result.monitoringUrl).username).toBe("postgres_ai_mon");
      const m = new Client({ connectionString: result.monitoringUrl });
      await m.connect();
      await m.end();
    } finally {
      await c.end();
    }
  });

  test("at a terminal the SQL is printed as it is (to paste into psql), then what to do; exit 3", async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.query("create role pgai_connect_app login password 'app-pw-123'");
    await c.end();
    const url = new URL(ADMIN!);
    url.username = "pgai_connect_app";
    url.password = "app-pw-123";
    const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-tty-"));
    const api = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      // A free slot: no price to accept before the SQL.
      fetch: (req) => Response.json(new URL(req.url).pathname.endsWith("/rpc/cloud_monitoring_quote")
        ? { plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 1 }, subscription: false, quantity: 0, price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false }
        : []),
    });
    let out = "";
    const proc = Bun.spawn([process.execPath, resolve(import.meta.dir, "..", "bin", "postgres-ai.ts"), "mon", "deploy", url.toString()], {
      cwd: home,
      env: { PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_NO_FEEDBACK_TIP: "1", PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}` },
      terminal: { cols: 200, rows: 50, data(_term, bytes) { out += new TextDecoder().decode(bytes); } },
    });
    try {
      expect(await proc.exited).toBe(3);
      const screen = out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "");
      // The SQL as psql takes it: statements at the start of a line, not folded into a YAML string.
      expect(screen).toMatch(/^-- 01\.role$/m);
      expect(screen).toMatch(/^create extension if not exists pg_stat_statements with schema public;$/m);
      expect(screen).toContain("password '<redacted>'");
      expect(screen).not.toMatch(/^sql:/m);
      expect(screen).not.toMatch(/^status:/m);
      expect(screen).toMatch(/^Run the SQL as an admin/m);
      expect(screen.indexOf("-- 01.role")).toBeLessThan(screen.indexOf("Run the SQL as an admin"));
    } finally {
      api.stop(true);
      rmSync(home, { recursive: true, force: true });
    }
  }, 60_000);

  test("unprepareDatabase drops the role a run created, with its grants; the next run prepares again", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    try {
      expect(await prepareDatabase(ADMIN!, "self-managed")).toMatchObject({ generated: true });
      expect(await unprepareDatabase(ADMIN!)).toBe(true);
      expect((await c.query("select count(*)::int as n from pg_roles where rolname = 'postgres_ai_mon'")).rows[0].n).toBe(0);
      expect(await prepareDatabase(ADMIN!, "self-managed")).toMatchObject({ generated: true });
    } finally {
      await c.end();
    }
  });

  test("a deployment after deletion: --reset-password gives postgres_ai_mon a new password, no old one needed", async () => {
    const c = await admin();
    try {
      await prepareDatabase(ADMIN!, "self-managed");
      // The role exists and nobody has its password (the run that made it is gone).
      await c.query("alter role postgres_ai_mon password 'lost-and-gone'");
      expect("monitoringUrl" in await prepareDatabase(ADMIN!, "self-managed")).toBe(false);
      const reset = await prepareDatabase(ADMIN!, "self-managed", { resetPassword: true });
      if (!("monitoringUrl" in reset)) throw new Error(`expected a URL, got: ${JSON.stringify(reset)}`);
      expect(reset.generated).toBeUndefined();
      const m = new Client({ connectionString: reset.monitoringUrl });
      await m.connect();
      expect((await m.query("select pg_has_role('postgres_ai_mon', 'pg_read_all_stats', 'member') as ok")).rows[0].ok).toBe(true);
      await m.end();
      const old = new URL(reset.monitoringUrl);
      old.password = "lost-and-gone";
      await expect(new Client({ connectionString: old.toString() }).connect()).rejects.toThrow(/password authentication failed/);
    } finally {
      await c.end();
    }
  });

  // internal#432: a custom monitoring role, end to end on a real server.
  test("--monitoring-user: a name with a quote and a space is created, granted and logs in; a privileged role and a role without PGAI_MON_PASSWORD are refused; --reset-password rotates it", async () => {
    const role = 'pgai My"Mon';
    const qi = (s: string) => `"${s.replace(/"/g, '""')}"`;
    const c = await admin();
    const cleanup = async () => {
      for (const r of [role, "pgai_mon_priv"]) {
        await c.query(`drop owned by ${qi(r)} cascade`).catch(() => {});
        await c.query(`drop role if exists ${qi(r)}`);
      }
    };
    await cleanup();
    await c.query("create role pgai_mon_priv login createrole password 'priv-pw'");
    try {
      // No PGAI_MON_PASSWORD: a generated password would be lost, so nothing is created.
      expect(await prepareDatabase(ADMIN!, "self-managed", { monitoringUser: role })).toEqual({
        next: `Set PGAI_MON_PASSWORD to a password for ${role}: PostgresAI keeps no password for a custom role, so a generated one would be lost`,
      });
      expect((await c.query("select count(*)::int as n from pg_roles where rolname = $1", [role])).rows[0].n).toBe(0);

      // Through platformDeps, as the CLI and MCP call it: the role reaches prepare and unprepare.
      const deps = platformDeps({ apiKey: "unused", apiBaseUrl: "http://unused.invalid", uiBaseUrl: "http://unused.invalid" });
      process.env.PGAI_MON_PASSWORD = "first pw%1";
      const first = await deps.prepare(ADMIN!, "self-managed", { monitoringUser: role });
      if (!("monitoringUrl" in first)) throw new Error(`expected a URL, got: ${JSON.stringify(first)}`);
      expect(first.generated).toBeUndefined();
      expect(decodeURIComponent(new URL(first.monitoringUrl).username)).toBe(role);
      const m = new Client({ connectionString: first.monitoringUrl });
      await m.connect();
      expect((await m.query("select current_user as u, pg_has_role(current_user, 'pg_monitor', 'member') as mon, pg_has_role(current_user, 'pg_read_all_stats', 'member') as stats")).rows[0])
        .toEqual({ u: role, mon: true, stats: true });
      await m.end();

      // A role that can do more than monitor is refused, and left as it was.
      expect(await prepareDatabase(ADMIN!, "self-managed", { monitoringUser: "pgai_mon_priv" })).toEqual({
        next: "pgai_mon_priv is a superuser, or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS: a monitoring role must not. Use another --monitoring-user",
      });
      expect((await c.query("select rolconfig from pg_roles where rolname = 'pgai_mon_priv'")).rows[0].rolconfig).toBeNull();

      // --reset-password: the new PGAI_MON_PASSWORD, and the old one stops working.
      process.env.PGAI_MON_PASSWORD = "second pw";
      const reset = await prepareDatabase(ADMIN!, "self-managed", { monitoringUser: role, resetPassword: true });
      if (!("monitoringUrl" in reset)) throw new Error(`expected a URL, got: ${JSON.stringify(reset)}`);
      expect(decodeURIComponent(new URL(reset.monitoringUrl).password)).toBe("second pw");
      const n = new Client({ connectionString: reset.monitoringUrl });
      await n.connect();
      await n.end();
      await expect(new Client({ connectionString: first.monitoringUrl }).connect()).rejects.toThrow(/password authentication failed/);
      expect(await deps.unprepare(ADMIN!, role)).toBe(true);
      expect((await c.query("select count(*)::int as n from pg_roles where rolname = $1", [role])).rows[0].n).toBe(0);
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
      await cleanup();
      await c.end();
    }
  });

  test("--monitoring-user --reset-password: a CREATEDB role is refused", async () => {
    const c = await admin();
    await c.query("create role pgai_mon_createdb login createdb password 'old-pw'");
    process.env.PGAI_MON_PASSWORD = "new-pw";
    try {
      expect(await prepareDatabase(ADMIN!, "self-managed", { monitoringUser: "pgai_mon_createdb", resetPassword: true })).toEqual({
        next: "pgai_mon_createdb is a superuser, or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS: a monitoring role must not. Use another --monitoring-user",
      });
      expect((await c.query("select rolconfig from pg_roles where rolname = 'pgai_mon_createdb'")).rows[0].rolconfig).toBeNull();
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
      await c.query("drop owned by pgai_mon_createdb cascade");
      await c.query("drop role pgai_mon_createdb");
      await c.end();
    }
  });

  test.each(["grant", "membership"])("--monitoring-user --reset-password: an allowed %s with delegation rights is refused", async (kind) => {
    const c = await admin();
    const role = "pgai_mon_delegate";
    await c.query(`create role ${role} login password 'old-pw'`);
    if (kind === "grant") await c.query(`grant usage on schema public to ${role} with grant option`);
    else await c.query(`grant pg_monitor to ${role} with admin option`);
    const before = (await c.query("select rolpassword from pg_authid where rolname = $1", [role])).rows[0].rolpassword;
    process.env.PGAI_MON_PASSWORD = "new-pw";
    try {
      expect(await prepareDatabase(ADMIN!, "self-managed", { monitoringUser: role, resetPassword: true })).toHaveProperty("next", expect.stringContaining(kind === "grant" ? "has grants on schema public" : "WITH ADMIN OPTION"));
      expect((await c.query("select a.rolpassword, r.rolconfig from pg_authid a join pg_roles r using (oid) where a.rolname = $1", [role])).rows[0]).toEqual({ rolpassword: before, rolconfig: null });
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
      await c.query(`drop owned by ${role} cascade`);
      await c.query(`drop role ${role}`);
      await c.end();
    }
  });

  // A role this run did not create: a member of a superuser or of pg_execute_server_program, an
  // owner of a table, a role that may write a table, create in the database or receive future
  // grants, run a server-file function or read password verifiers, is refused on --reset-password,
  // and keeps its password and settings.
  test("--monitoring-user --reset-password: a role with other memberships, owned objects or grants beyond the plan's is refused and left as it was", async () => {
    const c = await admin();
    const roles = ["pgai_mon_sneaky", "pgai_mon_progx", "pgai_mon_owner", "pgai_mon_rw", "pgai_mon_dbcreate", "pgai_mon_defacl", "pgai_mon_loexport", "pgai_mon_readfile", "pgai_mon_authid", "pgai_mon_authpw", "pgai_mon_colwrite"];
    const cleanup = async () => {
      for (const r of roles) {
        await c.query(`drop owned by ${r} cascade`).catch(() => {});
        await c.query(`drop role if exists ${r}`);
      }
      await c.query("drop table if exists pgai_mon_rw_t");
    };
    await cleanup();
    for (const r of roles) await c.query(`create role ${r} login password 'old-pw'`);
    const adminUser = (await c.query("select current_user as u")).rows[0].u;
    const db = (await c.query("select current_database() as d")).rows[0].d;
    await c.query(`grant "${adminUser}" to pgai_mon_sneaky`);
    await c.query("grant pg_execute_server_program to pgai_mon_progx");
    await c.query("create table pgai_mon_owned_t (id int)");
    await c.query("alter table pgai_mon_owned_t owner to pgai_mon_owner");
    await c.query("create table pgai_mon_rw_t (a int)");
    await c.query("grant select, insert, update, delete, truncate on pgai_mon_rw_t to pgai_mon_rw");
    await c.query(`grant create on database "${db}" to pgai_mon_dbcreate`);
    await c.query("alter default privileges in schema public grant insert on tables to pgai_mon_defacl");
    await c.query("grant execute on function pg_catalog.lo_export(oid, text) to pgai_mon_loexport");
    await c.query("grant execute on function pg_catalog.pg_read_file(text) to pgai_mon_readfile");
    await c.query("grant select on pg_catalog.pg_authid to pgai_mon_authid");
    await c.query("grant select (rolpassword) on pg_catalog.pg_authid to pgai_mon_authpw");
    await c.query("grant insert (a) on pgai_mon_rw_t to pgai_mon_colwrite");
    const onlyRead = "a monitoring role may have only CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog and pg_toast, and the grants pgai makes. Use another --monitoring-user";
    process.env.PGAI_MON_PASSWORD = "new-pw";
    try {
      expect(await Promise.all(roles.map((r) => prepareDatabase(ADMIN!, "self-managed", { monitoringUser: r, resetPassword: true })))).toEqual([
        { next: `pgai_mon_sneaky is a member of pg_database_owner, ${adminUser}: a monitoring role may be a member of pg_monitor, pg_read_all_stats, pg_read_all_settings and pg_stat_scan_tables only. Use another --monitoring-user` },
        { next: "pgai_mon_progx is a member of pg_execute_server_program: a monitoring role may be a member of pg_monitor, pg_read_all_stats, pg_read_all_settings and pg_stat_scan_tables only. Use another --monitoring-user" },
        { next: "pgai_mon_owner owns objects on this server: a monitoring role must own nothing. Use another --monitoring-user" },
        { next: `pgai_mon_rw has grants on table public.pgai_mon_rw_t: ${onlyRead}` },
        { next: `pgai_mon_dbcreate has grants on database ${db}: ${onlyRead}` },
        { next: `pgai_mon_defacl has grants on default privileges on new relations belonging to role ${adminUser} in schema public: ${onlyRead}` },
        { next: `pgai_mon_loexport has grants on function lo_export(oid,text): ${onlyRead}` },
        { next: `pgai_mon_readfile has grants on function pg_read_file(text): ${onlyRead}` },
        { next: `pgai_mon_authid has grants on table pg_authid: ${onlyRead}` },
        { next: `pgai_mon_authpw has grants on column rolpassword of table pg_authid: ${onlyRead}` },
        { next: `pgai_mon_colwrite has grants on column a of table public.pgai_mon_rw_t: ${onlyRead}` },
      ]);
      expect((await c.query("select count(*)::int as n from pg_roles where rolname = any($1) and rolconfig is not null", [roles])).rows[0].n).toBe(0);
      for (const r of roles) {
        const url = new URL(ADMIN!);
        url.username = r;
        url.password = "old-pw";
        const o = new Client({ connectionString: url.toString() });
        await o.connect();
        await o.end();
      }
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
      await cleanup();
      await c.end();
    }
  });

  // Grants the in-database check cannot see: a grant made in another database of the server (the
  // role connects there, CONNECT is PUBLIC), a server-file group reached through another group or
  // through pg_monitor and pg_read_all_stats themselves (also for a role the plan has not yet made
  // their member), and reads of pg_toast (the TOAST of
  // pg_statistic and pg_authid). Each is refused on --reset-password, and the role keeps its password.
  test("--monitoring-user --reset-password: grants in another database, groups reached through groups, and pg_toast access are refused", async () => {
    const c = await admin();
    const other = "pgai_mon_other";
    const roles = ["pgai_mon_xdb_lo", "pgai_mon_xdb_authid", "pgai_mon_chain", "pgai_mon_toastns", "pgai_mon_toast", "pgai_mon_viamon", "pgai_mon_viastats", "pgai_mon_plain"];
    const otherUrl = new URL(ADMIN!);
    otherUrl.pathname = `/${other}`;
    const cleanup = async () => {
      await c.query("revoke pg_execute_server_program from pg_monitor").catch(() => {});
      await c.query("revoke pg_read_server_files from pg_read_all_stats").catch(() => {});
      await c.query(`drop database if exists ${other}`);
      for (const r of roles) {
        await c.query(`drop owned by ${r} cascade`).catch(() => {});
        await c.query(`drop role if exists ${r}`);
      }
      await c.query("drop role if exists pgai_mon_mid");
    };
    await cleanup();
    for (const r of roles) await c.query(`create role ${r} login password 'old-pw'`);
    await c.query(`create database ${other}`);
    const o = new Client({ connectionString: otherUrl.toString() });
    await o.connect();
    await o.query("grant execute on function pg_catalog.lo_export(oid, text) to pgai_mon_xdb_lo");
    await o.query("grant select on pg_catalog.pg_authid to pgai_mon_xdb_authid");
    await o.end();
    await c.query("create role pgai_mon_mid");
    await c.query("grant pg_execute_server_program to pgai_mon_mid");
    await c.query("grant pgai_mon_mid to pgai_mon_chain");
    await c.query("grant pg_monitor to pgai_mon_viamon");
    await c.query("grant pg_read_all_stats to pgai_mon_viastats");
    await c.query("grant usage on schema pg_toast to pgai_mon_toastns");
    await c.query("grant select on pg_toast.pg_toast_2619 to pgai_mon_toast");
    const groups = "a monitoring role may be a member of pg_monitor, pg_read_all_stats, pg_read_all_settings and pg_stat_scan_tables only. Use another --monitoring-user";
    const onlyRead = "a monitoring role may have only CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog and pg_toast, and the grants pgai makes. Use another --monitoring-user";
    const thisDb = "a monitoring role may have grants in this database only. Use another --monitoring-user";
    process.env.PGAI_MON_PASSWORD = "new-pw";
    try {
      const results = [];
      const reset = (r: string) => prepareDatabase(ADMIN!, "self-managed", { monitoringUser: r, resetPassword: true });
      for (const r of roles.slice(0, 5)) results.push(await reset(r));
      // These two reach every member of pg_monitor, and so every role the plan grants it to.
      await c.query("grant pg_execute_server_program to pg_monitor");
      await c.query("grant pg_read_server_files to pg_read_all_stats");
      for (const r of roles.slice(5)) results.push(await reset(r));
      expect(results).toEqual([
        { next: `pgai_mon_xdb_lo has grants in database ${other}: ${thisDb}` },
        { next: `pgai_mon_xdb_authid has grants in database ${other}: ${thisDb}` },
        { next: `pgai_mon_chain is a member of pg_execute_server_program, pgai_mon_mid: ${groups}` },
        { next: `pgai_mon_toastns has grants on schema pg_toast: ${onlyRead}` },
        { next: `pgai_mon_toast has grants on toast table pg_toast.pg_toast_2619: ${onlyRead}` },
        { next: `pgai_mon_viamon is a member of pg_execute_server_program, pg_read_server_files: ${groups}` },
        { next: `pgai_mon_viastats is a member of pg_read_server_files: ${groups}` },
        // Not a member of pg_monitor yet: the plan grants it, and with it pg_execute_server_program.
        { next: `pgai_mon_plain would become a member of pg_execute_server_program, pg_read_server_files through pg_monitor or pg_read_all_stats: ${groups}` },
      ]);
      expect((await c.query("select count(*)::int as n from pg_roles where rolname = any($1) and rolconfig is not null", [roles])).rows[0].n).toBe(0);
      for (const r of roles) {
        const url = new URL(ADMIN!);
        url.username = r;
        url.password = "old-pw";
        const m = new Client({ connectionString: url.toString() });
        await m.connect();
        await m.end();
      }
    } finally {
      delete process.env.PGAI_MON_PASSWORD;
      await cleanup();
      await c.end();
    }
  });

  test("the express checkup runs as postgres_ai_mon on the prepared database", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    await c.end();
    const prepared = await prepareDatabase(ADMIN!, "self-managed");
    if (!("monitoringUrl" in prepared)) throw new Error(`expected a URL, got: ${JSON.stringify(prepared)}`);
    const result = await expressCheckup(prepared.monitoringUrl);
    if ("error" in result) throw new Error(result.error);
    expect(result.checks).toBeGreaterThan(15);
    // A002 is ok on PG 17+, info (no verdict) on PG 15-16: either way it is counted once.
    if (!result.info.includes("A002")) expect(result.findings.find((f) => f.check_id === "A002")).toMatchObject({ status: "ok", message: expect.stringMatching(/^PostgreSQL \d+$/) });
    expect([...result.findings.map((f) => f.check_id), ...result.info].filter((id) => id === "A002")).toEqual(["A002"]);
    // Inventories ("382 settings collected") are not findings.
    expect(result.findings.every((f) => f.status !== "info")).toBe(true);
  });

  // A run that fails after (or before) the prepare step must not block the
  // re-run: the same command again, against a platform and a ClickHouse Cloud
  // API that answer differently each time.
  describe("pgai mon deploy re-runs (the real CLI, a fake platform)", () => {
    const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");

    test("a rejected ClickHouse key, then a refused launch, then a launch: each re-run goes through; a URL without a database matches its row", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      const roles = async () => (await c.query("select count(*)::int as n from pg_roles where rolname = 'postgres_ai_mon'")).rows[0].n;
      // No database in the URL: pg connects to the one named like the user.
      const url = new URL(ADMIN!);
      url.pathname = "";
      const name = `${url.hostname}${url.port && url.port !== "5432" ? `:${url.port}` : ""}/${url.username}`;
      const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-rerun-"));
      const SERVICE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
      let keyOk = false;
      let launch: "refuse" | "accept" = "refuse";
      const rows: unknown[] = [];
      const calls: string[] = [];
      const uploads: Record<string, string>[] = [];
      let stderr = "";
      const api = Bun.serve({
        hostname: "127.0.0.1", port: 0,
        async fetch(req) {
          const path = new URL(req.url).pathname;
          calls.push(path);
          if (path === "/v1/organizations") return keyOk ? Response.json({ result: [{ id: "11111111-2222-3333-4444-555555555555" }] }) : new Response("", { status: 401 });
          if (path.endsWith("/postgres")) return Response.json({ result: [{ id: SERVICE, name: "svc", state: "running" }] });
          if (path.endsWith(`/postgres/${SERVICE}`)) return Response.json({ result: { id: SERVICE, name: "svc", state: "running", hostname: url.hostname } });
          if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json(rows);
          // A free slot: no price to accept.
          if (path.endsWith("/rpc/cloud_monitoring_quote")) return Response.json({ plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 1 }, subscription: false, quantity: 0, price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false });
          if (path.endsWith("/rpc/cloud_monitoring_connect")) {
            const dbUrl = new URL(JSON.parse(await req.text()).db_url);
            const row = { id: "i-1", name: `${dbUrl.hostname}${dbUrl.port && dbUrl.port !== "5432" ? `:${dbUrl.port}` : ""}${dbUrl.pathname}`, provider: "clickhouse", status: "launch_requested", dashboard_url: null, host_metrics: true };
            if (launch === "refuse") return Response.json({ id: row.id, name: row.name, status: "failed", error: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later." });
            rows.push(row);
            return Response.json(row);
          }
          // The express checkup, saved as a report of the database's project.
          if (path.endsWith("/rpc/checkup_report_create")) { uploads.push({ create: JSON.parse(await req.text()).project }); return Response.json({ report_id: 41 }); }
          if (path.endsWith("/rpc/checkup_report_file_post")) { const b = JSON.parse(await req.text()); uploads.push({ file: `${b.checkup_report_id}/${b.filename}` }); return Response.json({ report_chunck_id: uploads.length }); }
          if (path.endsWith("/rpc/checkup_report_status_update")) { const b = JSON.parse(await req.text()); uploads.push({ status: `${b.report_id} ${b.status}` }); return Response.json({ report_id: b.report_id, status: b.status }); }
          return new Response("not found", { status: 404 });
        },
      });
      const run = async () => {
        const proc = Bun.spawn([process.execPath, CLI, "mon", "deploy", url.toString(), "--provider", "clickhouse", "--clickhouse-key", "kid:Sec4b1dTestSecret", "--wait", "0"], {
          cwd: home, stdout: "pipe", stderr: "pipe",
          env: {
            PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "test-key",
            PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}`, CLICKHOUSE_API_URL: `http://127.0.0.1:${api.port}`,
          },
        });
        let stdout: string;
        [stdout, stderr] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
        return { exit: proc.exitCode, ...JSON.parse(stdout) };
      };
      try {
        // The key is checked first: the database is not touched.
        expect(await run()).toEqual({ exit: 3, status: "action_required", provider: "clickhouse", name, next: "ClickHouse Cloud rejected the API key (401). Check the key id and secret. Then re-run" });
        expect(await roles()).toBe(0);

        // A refused launch: the role this run created is dropped again.
        keyOk = true;
        expect(await run()).toEqual({ exit: 1, status: "failed", provider: "clickhouse", name, id: "i-1", next: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later. Re-run pgai mon deploy later." });
        expect(await roles()).toBe(0);

        launch = "accept";
        // While the box starts, the express checkup ran as postgres_ai_mon against this server.
        const { checkup, ...launched } = await run();
        expect(launched).toEqual({ exit: 0, status: "in_progress", provider: "clickhouse", name, id: "i-1", dashboard_url: null, host_metrics: true, price: "free (1 of 1 free slots)", requires_payment_method: false, next: "pgai mon instances watch i-1" });
        expect(checkup.checks).toBe(19);
        expect([...checkup.findings.map((f: { check_id: string }) => f.check_id), ...checkup.info]).toContain("A002");
        // Every check is counted once: warning, ok, info, or could not run.
        expect(checkup.findings.length + checkup.info.length + (checkup.failed?.length ?? 0)).toBe(19);
        // Saved right away: pgai reports list shows it while the box starts.
        expect(checkup.report_id).toBe(41);
        expect(uploads[0]).toEqual({ create: name });
        expect(uploads.filter((u) => u.file).length).toBe(19 - (checkup.failed?.length ?? 0));
        expect(uploads.at(-1)).toEqual({ status: "41 completed" });
        // JSON on stdout, so the steps go to stderr as events, one JSON object a line
        // (and a check that cannot run, D004 without pg_stat_statements preloaded, a log event).
        const events = stderr.trim().split("\n").map((l) => JSON.parse(l).event);
        expect(events.filter((e) => e !== "log")).toEqual(["billing", "preparing", "provisioning", "checkup", "checkup", "box", "done"]);
        expect(await roles()).toBe(1);

        // The re-run finds its row by name: nothing is prepared or provisioned again; the key it is given is checked.
        const before = calls.length;
        expect((await run()).exit).toBe(0);
        expect(calls.slice(before)).toEqual([
          "/rpc/cloud_monitoring_list",
          "/v1/organizations",
          "/v1/organizations/11111111-2222-3333-4444-555555555555/postgres",
          "/v1/organizations/11111111-2222-3333-4444-555555555555/postgres/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
          // The result's Health Matrix link: the org alias and the database's project.
          "/rpc/orgs_list",
          "/rpc/projects_list",
        ]);
      } finally {
        api.stop(true);
        await c.end();
        rmSync(home, { recursive: true, force: true });
      }
    }, 120_000);
  });

  test("pgai mon deploy, a box active at once (the real CLI, a fake platform): first_checkup_eta is the run's end + 3 min", async () => {
    const c = await admin();
    await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
    await c.query("drop schema if exists postgres_ai cascade");
    await c.query("drop role if exists postgres_ai_mon");
    const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-eta-"));
    const api = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
        if (path.endsWith("/rpc/cloud_monitoring_quote")) return Response.json({ plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 1, total: 1 }, subscription: false, quantity: 0, price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false });
        if (path.endsWith("/rpc/cloud_monitoring_connect")) {
          const dbUrl = new URL(JSON.parse(await req.text()).db_url);
          return Response.json({ id: "i-eta", name: `${dbUrl.hostname}:${dbUrl.port}${dbUrl.pathname}`, provider: "self-managed", status: "active", dashboard_url: "https://grafana.test", host_metrics: false });
        }
        if (path.endsWith("/rpc/checkup_report_create")) return Response.json({ report_id: 1 });
        if (path.endsWith("/rpc/checkup_report_file_post")) return Response.json({ report_chunck_id: 1 });
        if (path.endsWith("/rpc/checkup_report_status_update")) return Response.json({});
        if (path.endsWith("/rpc/orgs_list") || path.endsWith("/rpc/projects_list")) return Response.json([]);
        return new Response("not found", { status: 404 });
      },
    });
    try {
      const proc = Bun.spawn([process.execPath, resolve(import.meta.dir, "..", "bin", "postgres-ai.ts"), "mon", "deploy", ADMIN!, "--vcpus", "2", "--wait", "0"], {
        cwd: home, stdout: "pipe", stderr: "pipe",
        env: { PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}` },
      });
      const [stdout, stderr] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
      const end = Date.now();
      if (proc.exitCode !== 0) throw new Error(`exit ${proc.exitCode}\n${stdout}\n${stderr}`);
      const { status, first_checkup_eta } = JSON.parse(stdout);
      // The CI log shows the real CLI's ETA next to the time the run ended.
      console.log(`mon deploy: status=${status} first_checkup_eta=${first_checkup_eta} run ended=${new Date(end).toISOString()}`);
      expect(status).toBe("ready");
      const ahead = Date.parse(first_checkup_eta) - end;
      expect(ahead).toBeGreaterThan(3 * 60_000 - 5_000);
      expect(ahead).toBeLessThanOrEqual(3 * 60_000);
    } finally {
      api.stop(true);
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      await c.end();
      rmSync(home, { recursive: true, force: true });
    }
  }, 120_000);

  // The price prompt needs a URL that can work: deploy checks it first.
  describe("pgai mon deploy at the price prompt (the real CLI in a terminal, a fake platform)", () => {
    const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    const PROMPT = /Provision [^\n]+\? \(y\/N\): $/;
    const SERVICE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

    async function atPrompt(answer: string, o: { args?: string[]; create?: (term: Bun.Terminal) => Response | Promise<Response>; timeoutMs?: number; vcpus?: string[] } = {}) {
      const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-prompt-"));
      const calls: string[] = [];
      const bodies: Record<string, unknown>[] = [];
      const api = Bun.serve({
        hostname: "127.0.0.1", port: 0,
        async fetch(req) {
          const path = new URL(req.url).pathname;
          calls.push(path);
          if (path.endsWith("/rpc/cloud_monitoring_connect")) bodies.push(JSON.parse(await req.text()));
          // ClickHouse Cloud, for --provider clickhouse: one org, one service on this server.
          if (path === "/v1/organizations") return Response.json({ result: [{ id: "11111111-2222-3333-4444-555555555555" }] });
          if (path.endsWith("/postgres")) return Response.json({ result: [{ id: SERVICE, name: "svc", state: "running" }] });
          if (path.endsWith(`/postgres/${SERVICE}`)) return Response.json({ result: { id: SERVICE, name: "svc", state: "running", hostname: new URL(ADMIN!).hostname } });
          if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json([]);
          if (path.endsWith("/rpc/cloud_monitoring_quote")) return Response.json({ plan: "scale", org_alias: "acme", billed: true, free_slots: { remaining: 0, total: 0 }, subscription: false, quantity: 0, price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: true, requires_payment_method: false });
          if (o.create && path.endsWith("/rpc/cloud_monitoring_connect")) return o.create(terminal!);
          return new Response("not found", { status: 404 });
        },
      });
      let out = "";
      let typed = false;
      let vcpusAsked = 0;
      let terminal: Bun.Terminal | undefined;
      try {
        const proc = Bun.spawn([process.execPath, CLI, "mon", "deploy", ADMIN!, ...(o.args ?? [])], {
          cwd: home,
          env: { PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}`, CLICKHOUSE_API_URL: `http://127.0.0.1:${api.port}`, PGAI_NO_FEEDBACK_TIP: "1" },
          terminal: {
            cols: 400, rows: 50,
            data(term, bytes) {
              out += new TextDecoder().decode(bytes);
              const screen = out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "");
              if (!typed && PROMPT.test(screen)) {
                typed = true;
                setTimeout(() => term.write(answer), 100);
              }
              // One answer each time the vCPU question is (re)asked.
              const asked = screen.split("vCPUs: ").length - 1;
              if (asked > vcpusAsked && screen.endsWith("vCPUs: ") && o.vcpus?.[vcpusAsked] !== undefined) {
                const a = o.vcpus[vcpusAsked];
                vcpusAsked = asked;
                setTimeout(() => term.write(a), 100);
              }
            },
          },
        });
        terminal = proc.terminal;
        // A run that never ends is killed: status null.
        const status = await Promise.race([proc.exited, Bun.sleep(o.timeoutMs ?? 60_000).then(() => { proc.kill(); return null; })]);
        return { status, raw: out, screen: out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, ""), calls, bodies };
      } finally {
        api.stop(true);
        rmSync(home, { recursive: true, force: true });
      }
    }

    test("no --vcpus at a terminal: asked after the price (no AAS without it); a bad answer is asked again; the answer goes to the create", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      try {
        const r = await atPrompt("y\r", {
          args: ["--wait", "0"], timeoutMs: 30_000, vcpus: ["0\r", "3\r"],
          create: () => Response.json({ id: "i-1", name: "x", provider: "self-managed", status: "launch_requested" }),
        });
        expect(r.screen).toContain("If you skip this (Enter), you get no AAS (database load) data");
        expect(r.screen).toContain("--vcpus must be a whole number from 1 to 1024");
        expect(r.bodies).toHaveLength(1);
        expect(r.bodies[0]).toMatchObject({ vcpus: 3 });
      } finally {
        await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
        await c.query("drop role if exists postgres_ai_mon").catch(() => {});
        await c.end();
      }
    }, 60_000);

    test("the platform refuses the create (an unknown --location): the steps stop, the reason stays on screen, exit 1", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      try {
        const r = await atPrompt("y\r", {
          args: ["--location", "xyz", "--vcpus", "2"], timeoutMs: 30_000,
          // As v1.cloud_monitoring_connect raises it: the reason in detail.
          create: () => Response.json({ code: "PT400", message: "Bad Request", details: 'Unknown location "xyz". Choose one of: fsn1, nbg1, hel1.', hint: "Nothing was created." }, { status: 400 }),
        });
        expect(r.status).toBe(1);
        const at = r.raw.lastIndexOf('Unknown location "xyz"');
        expect(at).toBeGreaterThan(0);
        // Nothing redraws over it: no cursor-up after the reason.
        expect(r.raw.slice(at)).not.toMatch(/\x1b\[\d+A/);
        expect(r.screen.trimEnd()).toMatch(/Unknown location "xyz"\. Choose one of: fsn1, nbg1, hel1\.\n.*Nothing was created\.$/s);
      } finally {
        await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
        await c.query("drop role if exists postgres_ai_mon").catch(() => {});
        await c.end();
      }
    }, 60_000);

    // REV r3 (MR !472): self-managed exited 0 with the box still starting, and gave no Resume line.
    test("Ctrl-C while the box request is in flight: it detaches with the Resume line, exit 130", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      try {
        const r = await atPrompt("y\r", {
          args: ["--vcpus", "2"], timeoutMs: 30_000,
          create: async (term) => {
            term.write("\x03");
            // The platform answers after the reply to Ctrl-C.
            await Bun.sleep(500);
            return Response.json({ id: "i-9", name: "x", provider: "self-managed", status: "launch_requested" });
          },
        });
        expect(r.status).toBe(130);
        expect(r.screen).toContain("Stopping: the monitoring box is already requested");
        expect(r.screen.trimEnd()).toEndWith("Stopped watching. The deploy continues on PostgresAI.\nResume: pgai mon instances watch i-9");
      } finally {
        await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
        await c.query("drop role if exists postgres_ai_mon").catch(() => {});
        await c.end();
      }
    }, 60_000);

    // REV r4 (MR !472): ClickHouse returns from the same in-flight path; it had no test of its own.
    test("ClickHouse: Ctrl-C while the box request is in flight detaches with the Resume line, exit 130", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      try {
        const r = await atPrompt("y\r", {
          args: ["--provider", "clickhouse", "--clickhouse-key", "kid:Sec4b1dTestSecret", "--vcpus", "2"], timeoutMs: 30_000,
          create: async (term) => {
            term.write("\x03");
            await Bun.sleep(500);
            return Response.json({ id: "i-8", name: "x", provider: "clickhouse", status: "launch_requested" });
          },
        });
        expect(r.calls).toContain("/v1/organizations");
        expect(r.status).toBe(130);
        expect(r.screen).toContain("Stopping: the monitoring box is already requested");
        expect(r.screen.trimEnd()).toEndWith("Stopped watching. The deploy continues on PostgresAI.\nResume: pgai mon instances watch i-8");
        // Detached at once: no checkup, no polling; the box keeps the role it was given.
        expect(r.calls.at(-1)).toBe("/rpc/cloud_monitoring_connect");
        expect((await c.query("select count(*)::int as n from pg_roles where rolname = 'postgres_ai_mon'")).rows[0].n).toBe(1);
      } finally {
        await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
        await c.query("drop role if exists postgres_ai_mon").catch(() => {});
        await c.end();
      }
    }, 60_000);

    test("Ctrl-C or Ctrl-D is exit 130, n (or anything but y/yes) is exit 3; the price is shown first; nothing created", async () => {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      try {
        for (const key of ["\x03", "\x04"]) expect((await atPrompt(key)).status).toBe(130);
        const n = await atPrompt("n\r");
        expect(n.status).toBe(3);
        expect(n.screen).toMatch(/^Estimated cost: \$512\.00\/month per database cluster \(scale plan\)\nProvision /m);
        expect(n.screen).toMatch(/^Re-run with --yes to accept \$512\.00\/month per database cluster \(scale plan\)$/m);
        expect((await atPrompt("yes, but not now\r")).status).toBe(3);
        expect(n.calls.filter((p) => p.includes("cloud_monitoring_connect"))).toEqual([]);
        expect((await c.query("select count(*)::int as n from pg_roles where rolname = 'postgres_ai_mon'")).rows[0].n).toBe(0);
      } finally {
        await c.end();
      }
    }, 120_000);
  });

  // The stub docker records who called it (argv and environment names, from
  // /proc), so the child `mon local-install` is checked as it was started.
  describe.skipIf(process.platform !== "linux")("pgai mon deploy --self-hosted (the real CLI, a stub docker)", () => {
    const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
    // A function: a skipped describe still runs its body, and ADMIN is unset there.
    const name = () => { const u = new URL(ADMIN!); return `${u.hostname}${u.port && u.port !== "5432" ? `:${u.port}` : ""}/postgres`; };

    async function selfHosted(dockerExit: number) {
      const c = await admin();
      await c.query("drop owned by postgres_ai_mon cascade").catch(() => {});
      await c.query("drop schema if exists postgres_ai cascade");
      await c.query("drop role if exists postgres_ai_mon");
      await c.end();
      const dir = mkdtempSync(resolve(tmpdir(), "pgai-self-hosted-"));
      for (const path of ["project", "project/.git", "bin", "home"]) mkdirSync(`${dir}/${path}`);
      writeFileSync(`${dir}/project/docker-compose.yml`, "services: {}\n");
      writeFileSync(`${dir}/bin/docker`, [
        "#!/bin/sh",
        `{ printf 'argv: '; tr '\\0' ' ' < /proc/$PPID/cmdline; echo; printf 'env: '; tr '\\0' '\\n' < /proc/$PPID/environ | grep -E '^(PGAI_|CLICKHOUSE_)' | cut -d= -f1 | sort | tr '\\n' ' '; echo; } >> ${dir}/docker.log`,
        `exit ${dockerExit}`,
        "",
      ].join("\n"));
      chmodSync(`${dir}/bin/docker`, 0o755);
      const registered: string[] = [];
      const api = Bun.serve({
        hostname: "127.0.0.1", port: 0,
        async fetch(req) {
          registered.push(`${new URL(req.url).pathname} ${await req.text()}`);
          return Response.json({ instance_id: "11111111-2222-3333-4444-555555555555", project_id: 7 });
        },
      });
      try {
        const proc = Bun.spawn([process.execPath, CLI, "mon", "deploy", ADMIN!, "--self-hosted", "--org", "acme"], {
          cwd: dir, stdout: "pipe", stderr: "pipe",
          env: {
            PATH: `${dir}/bin:${process.env.PATH}`, HOME: `${dir}/home`, XDG_CONFIG_HOME: `${dir}/home`, PGAI_PROJECT_DIR: `${dir}/project`,
            PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}`,
          },
        });
        const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
        const log = readFileSync(`${dir}/docker.log`, "utf8").split("\n");
        const child = log.findIndex((line) => line.startsWith("argv: ") && line.includes(" mon local-install "));
        const instances = await Bun.file(`${dir}/project/instances.yml`).text().catch(() => "");
        return { status, stdout, stderr, registered, instances, childArgv: log[child]?.split(`${CLI} `)[1]?.trim(), childEnv: log[child + 1] };
      } finally {
        api.stop(true);
        rmSync(dir, { recursive: true, force: true });
      }
    }

    test("the child gets the org and a project in argv, the URL and the key only in its environment; stdout is the result", async () => {
      const r = await selfHosted(0);
      expect(r.stderr).toContain("Local install completed!");
      expect(JSON.parse(r.stdout)).toEqual({ status: "ready", provider: "self-managed", name: name(), dashboard_url: "http://localhost:3000", host_metrics: false, next: "pgai mon health" });
      expect(r.status).toBe(0);
      const project = name().replace(/[^A-Za-z0-9._-]+/g, "-");
      expect(r.childArgv).toBe(`mon local-install -y --org acme --project ${project}`);
      expect(r.childEnv).toBe("env: PGAI_API_BASE_URL PGAI_API_KEY PGAI_DB_URL PGAI_PROJECT_DIR ");
      // The child took the monitoring URL from PGAI_DB_URL and registered with PGAI_API_KEY.
      expect(r.instances).toContain("postgresql://postgres_ai_mon:");
      expect(r.instances).not.toContain(new URL(ADMIN!).password);
      expect(r.registered).toEqual([`/rpc/monitoring_instance_register ${JSON.stringify({ api_token: "test-key", project_name: project })}`]);
    }, 120_000);

    test("a child that fails: failed, exit 1, stdout still only the result", async () => {
      const r = await selfHosted(1);
      expect(JSON.parse(r.stdout)).toEqual({ status: "failed", provider: "self-managed", name: name(), next: "mon local-install failed (see above)" });
      expect(r.status).toBe(1);
    }, 120_000);
  });
});
