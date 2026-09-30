import { afterAll, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Client } from "pg";
import { prepareDatabase, unprepareDatabase } from "../lib/connect";

// `pgai connect`'s prepare step against a real Postgres, with a superuser URL
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
    // A server that does not check passwords (trust, as for CI's localhost
    // service) lets any password in; the stored one must still stay as it is.
    const wrong = new URL(monUrlFromEarlierTest);
    wrong.password = "not-the-password";
    const w = new Client({ connectionString: wrong.toString() });
    const checksPasswords = await w.connect().then(() => w.end().then(() => false), () => true);
    process.env.PGAI_MON_PASSWORD = "not-the-password";
    try {
      const refused = await prepareDatabase(db2.toString(), "self-managed");
      if (!checksPasswords) expect(refused).toHaveProperty("monitoringUrl");
      else expect(refused).toEqual({ next: "postgres_ai_mon already exists on this server and PGAI_MON_PASSWORD is not its password. Set PGAI_MON_PASSWORD to the password of postgres_ai_mon, or change it explicitly with: pgai prepare-db <admin-url> --reset-password --password <new-password> (then update every monitoring box that uses it)" });
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

  test("at a terminal the SQL is printed as it is (to paste into psql), then the rest as YAML; exit 3", async () => {
    const c = await admin();
    await c.query("drop role if exists pgai_connect_app");
    await c.query("create role pgai_connect_app login password 'app-pw-123'");
    await c.end();
    const url = new URL(ADMIN!);
    url.username = "pgai_connect_app";
    url.password = "app-pw-123";
    const home = mkdtempSync(resolve(tmpdir(), "pgai-connect-tty-"));
    const api = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => Response.json([]) });
    let out = "";
    const proc = Bun.spawn([process.execPath, resolve(import.meta.dir, "..", "bin", "postgres-ai.ts"), "connect", url.toString()], {
      cwd: home,
      env: { PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_NO_FEEDBACK_TIP: "1", PGAI_API_KEY: "test-key", PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}` },
      terminal: { cols: 200, rows: 50, data(_term, bytes) { out += new TextDecoder().decode(bytes); } },
    });
    try {
      expect(await proc.exited).toBe(3);
      const screen = out.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r/g, "");
      // The SQL as psql takes it: statements at the start of a line, not folded into a YAML string.
      expect(screen).toMatch(/^-- 01\.role$/m);
      expect(screen).toMatch(/^create extension if not exists pg_stat_statements;$/m);
      expect(screen).toContain("password '<redacted>'");
      expect(screen).not.toMatch(/^sql:/m);
      expect(screen).toMatch(/^status: action_required$/m);
      expect(screen).toMatch(/^next: >?-?\s*Run the SQL as an admin/m);
      expect(screen.indexOf("-- 01.role")).toBeLessThan(screen.indexOf("status: action_required"));
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

  // A run that fails after (or before) the prepare step must not block the
  // re-run: the same command again, against a platform and a ClickHouse Cloud
  // API that answer differently each time.
  describe("pgai connect re-runs (the real CLI, a fake platform)", () => {
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
      const api = Bun.serve({
        hostname: "127.0.0.1", port: 0,
        async fetch(req) {
          const path = new URL(req.url).pathname;
          calls.push(path);
          if (path === "/v1/organizations") return keyOk ? Response.json({ result: [{ id: "11111111-2222-3333-4444-555555555555" }] }) : new Response("", { status: 401 });
          if (path.endsWith("/postgres")) return Response.json({ result: [{ id: SERVICE, name: "svc", state: "running" }] });
          if (path.endsWith(`/postgres/${SERVICE}`)) return Response.json({ result: { id: SERVICE, name: "svc", state: "running", hostname: url.hostname } });
          if (path.endsWith("/rpc/cloud_monitoring_list")) return Response.json(rows);
          if (path.endsWith("/rpc/cloud_monitoring_connect")) {
            const dbUrl = new URL(JSON.parse(await req.text()).db_url);
            const row = { id: "i-1", name: `${dbUrl.hostname}${dbUrl.port ? `:${dbUrl.port}` : ""}${dbUrl.pathname}`, provider: "clickhouse", status: "launch_requested", dashboard_url: null, host_metrics: true };
            if (launch === "refuse") return Response.json({ id: row.id, name: row.name, status: "failed", error: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later." });
            rows.push(row);
            return Response.json(row);
          }
          return new Response("not found", { status: 404 });
        },
      });
      const run = async () => {
        const proc = Bun.spawn([process.execPath, CLI, "connect", url.toString(), "--provider", "clickhouse", "--clickhouse-key", "kid:Sec4b1dTestSecret", "--wait", "0"], {
          cwd: home, stdout: "pipe", stderr: "pipe",
          env: {
            PATH: process.env.PATH!, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "test-key",
            PGAI_API_BASE_URL: `http://127.0.0.1:${api.port}`, CLICKHOUSE_API_URL: `http://127.0.0.1:${api.port}`,
          },
        });
        const [stdout, exit] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
        return { exit, ...JSON.parse(stdout) };
      };
      try {
        // The key is checked first: the database is not touched.
        expect(await run()).toEqual({ exit: 1, status: "failed", provider: "clickhouse", name, next: "ClickHouse Cloud rejected the API key (401). Check the key id and secret." });
        expect(await roles()).toBe(0);

        // A refused launch: the role this run created is dropped again.
        keyOk = true;
        expect(await run()).toEqual({ exit: 1, status: "failed", provider: "clickhouse", name, id: "i-1", next: "The monitoring box could not be launched (HTTP 422). Nothing is billed; try again later. Re-run pgai connect later." });
        expect(await roles()).toBe(0);

        launch = "accept";
        expect(await run()).toEqual({ exit: 0, status: "provisioning", provider: "clickhouse", name, id: "i-1", dashboard_url: null, host_metrics: true, next: `pgai status ${name}` });
        expect(await roles()).toBe(1);

        // The re-run finds its row by name: nothing is prepared or provisioned again.
        const before = calls.length;
        expect((await run()).exit).toBe(0);
        expect(calls.slice(before)).toEqual(["/rpc/cloud_monitoring_list"]);
      } finally {
        api.stop(true);
        await c.end();
        rmSync(home, { recursive: true, force: true });
      }
    }, 120_000);
  });

  // The stub docker records who called it (argv and environment names, from
  // /proc), so the child `mon local-install` is checked as it was started.
  describe.skipIf(process.platform !== "linux")("pgai connect --self-hosted (the real CLI, a stub docker)", () => {
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
        const proc = Bun.spawn([process.execPath, CLI, "connect", ADMIN!, "--self-hosted", "--org", "acme"], {
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
      expect(JSON.parse(r.stdout)).toEqual({ status: "connected", provider: "self-managed", name: name(), dashboard_url: "http://localhost:3000", host_metrics: false, next: "pgai mon health" });
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
