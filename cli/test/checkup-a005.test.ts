/**
 * A005 "Extensions": catalog facts of every database of the server, for the
 * A013 CVE prerequisites (platform-all#1108). Unit tests with the mock client;
 * the real-Postgres path is in checkup.integration.test.ts.
 */
import { describe, test, expect, afterEach } from "bun:test";
import { resolve } from "path";
import { readFileSync } from "fs";
import Ajv2020 from "ajv/dist/2020";

import * as checkup from "../lib/checkup";
import { generateCheckSummary } from "../lib/checkup-summary";
import { createMockClient } from "./test-utils";

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(
  JSON.parse(readFileSync(resolve(import.meta.dir, "../../reporter/schemas/A005.schema.json"), "utf8"))
);

function expectValid(report: any): void {
  if (!validate(report)) {
    throw new Error(validate.errors?.map((e) => `${e.instancePath}: ${e.message}`).join(", "));
  }
}

const original = checkup.setA005OpenDatabase(checkup.a005OpenDatabase);
afterEach(() => {
  checkup.setA005OpenDatabase(original);
});

describe("A005 Extensions", () => {
  test("is an express check with its dictionary title", () => {
    expect(typeof checkup.REPORT_GENERATORS.A005).toBe("function");
    expect(checkup.CHECK_INFO.A005).toBe("Extensions");
  });

  test("the connected database, not a superuser: hashes and hba rules unknown with the reason, never zero", async () => {
    const report = await checkup.generateA005(createMockClient() as any, "node-01");
    expectValid(report);
    const data = report.results["node-01"].data;
    expect(data.databases.testdb).toEqual({
      read: true,
      roles_with_create: 1,
      extensions: { plpgsql: "1.0" },
      index_access_methods: [{ am: "btree", extension: null, indexes: 3 }],
      row_security_policies: 0,
      extended_statistics: 0,
    });
    expect(data.roles.md5_passwords).toBeNull();
    expect(data.roles.nondefault_scram_iterations).toBeNull();
    expect(data.roles.password_hashes_error).toBe("permission denied for table pg_authid");
    expect(data.hba_rules).toBeNull();
    expect(data.hba_rules_error).toBe("permission denied for view pg_hba_file_rules");
    expect(data.conclusions.join("\n")).toContain("Stored password hashes are not known");
  });

  test("a superuser: MD5 and SCRAM iteration counts from pg_authid, hba rule counts", async () => {
    const client = createMockClient({
      a005PasswordHashes: { md5_passwords: 2, nondefault_scram_iterations: 0 },
      a005HbaRules: { cert: 0, trust: 1, hostgssenc: 0, errors: 0 },
    });
    const report = await checkup.generateA005(client as any, "node-01");
    expectValid(report);
    const data = report.results["node-01"].data;
    expect(data.roles.md5_passwords).toBe(2);
    expect(data.roles.nondefault_scram_iterations).toBe(0);
    expect(data.roles.password_hashes_error).toBeNull();
    expect(data.hba_rules).toEqual({ cert: 0, trust: 1, hostgssenc: 0, errors: 0 });
    expect(data.recommendations.join("\n")).toContain("MD5");
  });

  test("every other database is read over its own connection; one that fails is listed with its error", async () => {
    const opened: string[] = [];
    const closed: string[] = [];
    checkup.setA005OpenDatabase(async (_client, database) => {
      opened.push(database);
      if (database === "broken") throw new Error("FATAL:  pg_hba.conf rejects connection");
      const other = createMockClient({
        a005DatabaseRow: {
          extensions: { btree_gist: "1.7", plpgsql: "1.0" },
          index_access_methods: [{ am: "gist", extension: "btree_gist", indexes: 2 }],
          row_security_policies: 1,
          extended_statistics: 0,
        },
      }) as any;
      other.end = async () => { closed.push(database); };
      return other;
    });
    const client = createMockClient({
      a005DatabasesRows: [
        { datname: "app", is_current: false, can_connect: true, roles_with_create: 2 },
        { datname: "broken", is_current: false, can_connect: true, roles_with_create: 0 },
        { datname: "locked", is_current: false, can_connect: false, roles_with_create: 0 },
        { datname: "testdb", is_current: true, can_connect: true, roles_with_create: 1 },
      ],
      a005SlotsRows: [{ database: "app", plugin: "pgoutput", slots: 1, active: 0 }],
    });
    const report = await checkup.generateA005(client as any, "node-01");
    expectValid(report);
    const data = report.results["node-01"].data;
    // The connected database is read on the checkup's own connection; a database
    // without CONNECT is never tried.
    expect(opened).toEqual(["app", "broken"]);
    expect(closed).toEqual(["app"]);
    expect(data.databases.app.read).toBe(true);
    expect(data.databases.app.extensions).toEqual({ btree_gist: "1.7", plpgsql: "1.0" });
    expect(data.databases.app.roles_with_create).toBe(2);
    expect(data.databases.broken).toEqual({
      read: false, roles_with_create: 0, error: "FATAL: pg_hba.conf rejects connection",
    });
    expect(data.databases.locked.read).toBe(false);
    expect(data.databases.locked.error).toContain("no CONNECT privilege");
    // CREATE on a database is known even where the database was not read (pg_database is shared).
    expect(data.databases.locked.roles_with_create).toBe(0);
    expect(data.databases.testdb.read).toBe(true);
    expect(data.logical_slots).toEqual([{ database: "app", plugin: "pgoutput", slots: 1, active: 0 }]);
    expect(data.conclusions.join("\n")).toContain("Not read: broken, locked");

    const summary = generateCheckSummary("A005", report);
    expect(summary.status).toBe("warning");
    expect(summary.message).toBe("2 extensions in 2 of 4 databases; 2 not read");
  });

  test("a database named __proto__ is listed like any other", async () => {
    checkup.setA005OpenDatabase(async () => {
      const other = createMockClient({
        a005DatabaseRow: { extensions: { intarray: "1.5" }, index_access_methods: [], row_security_policies: 0, extended_statistics: 0 },
      }) as any;
      other.end = async () => {};
      return other;
    });
    const client = createMockClient({
      a005DatabasesRows: [
        { datname: "__proto__", is_current: false, can_connect: true, roles_with_create: 0 },
        { datname: "testdb", is_current: true, can_connect: true, roles_with_create: 0 },
      ],
    });
    const report = await checkup.generateA005(client as any, "node-01");
    const listed = JSON.parse(JSON.stringify(report)).results["node-01"].data.databases;
    expect(Object.keys(listed).sort()).toEqual(["__proto__", "testdb"]);
    expect(listed.__proto__.extensions).toEqual({ intarray: "1.5" });
  });

  test("at most A005_MAX_DATABASES other databases are opened; the rest are listed as not read", async () => {
    let opened = 0;
    checkup.setA005OpenDatabase(async () => {
      opened += 1;
      const other = createMockClient() as any;
      other.end = async () => {};
      return other;
    });
    const rows = Array.from({ length: checkup.A005_MAX_DATABASES + 3 }, (_, i) => ({
      datname: `db${String(i).padStart(3, "0")}`, is_current: false, can_connect: true, roles_with_create: 0,
    }));
    const report = await checkup.generateA005(createMockClient({ a005DatabasesRows: rows }) as any, "node-01");
    expectValid(report);
    const dbs = Object.values(report.results["node-01"].data.databases) as any[];
    expect(opened).toBe(checkup.A005_MAX_DATABASES);
    expect(dbs.filter((db) => !db.read)).toHaveLength(3);
    expect(dbs.find((db) => !db.read).error).toContain(`more than ${checkup.A005_MAX_DATABASES} databases`);
  });

  test("a mock client cannot reconnect: the default opener fails and the database is listed as not read", async () => {
    const client = createMockClient({
      a005DatabasesRows: [{ datname: "other", is_current: false, can_connect: true, roles_with_create: 0 }],
    });
    const report = await checkup.generateA005(client as any, "node-01");
    const db = report.results["node-01"].data.databases.other;
    expect(db.read).toBe(false);
    expect(db.error).toContain("cannot open another connection");
  });

  test("an error message is one line of at most 200 characters", async () => {
    checkup.setA005OpenDatabase(async () => {
      throw new Error("first line\n" + "x".repeat(500));
    });
    const client = createMockClient({
      a005DatabasesRows: [{ datname: "other", is_current: false, can_connect: true, roles_with_create: 0 }],
    });
    const report = await checkup.generateA005(client as any, "node-01");
    const error = report.results["node-01"].data.databases.other.error;
    expect(error.length).toBe(200);
    expect(error).not.toContain("\n");
  });
});
