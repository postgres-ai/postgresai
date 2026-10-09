import { describe, expect, test } from "bun:test";
import { Command } from "commander";
import {
  DEPLOY_EXIT,
  billingPage,
  databaseUrlArg,
  dblabStatus,
  jsonOutput,
  monitoringStatus,
  nameFromDatabase,
  noCardNext,
  notAdminNext,
  pickInstance,
  sharedDeployOptions,
  vcpusArg,
  waitMinutes,
} from "../lib/deploy-surface";

// The surface pgai dblab deploy and pgai mon deploy share (postgresai#412).

describe("deploy surface", () => {
  test("the database URL: the argument or --db-url, once", () => {
    expect(databaseUrlArg("postgresql://u@h/d", undefined, "pgai x")).toBe("postgresql://u@h/d");
    expect(databaseUrlArg(undefined, "postgresql://u@h/d", "pgai x")).toBe("postgresql://u@h/d");
    expect(databaseUrlArg("postgresql://u@h/d", "postgresql://u@h/d", "pgai x")).toBe("postgresql://u@h/d");
    expect(() => databaseUrlArg("postgresql://u@h/a", "postgresql://u@h/b", "pgai x")).toThrow("not both");
    expect(() => databaseUrlArg(undefined, undefined, "pgai dblab deploy")).toThrow("Pass the database URL: pgai dblab deploy postgresql://");
  });

  test("--wait in minutes; --no-wait is 0", () => {
    expect(waitMinutes(undefined, 20)).toBe(20);
    expect(waitMinutes("5", 20)).toBe(5);
    expect(waitMinutes("0", 20)).toBe(0);
    expect(waitMinutes(false, 120)).toBe(0);
    for (const bad of ["x", "-1", "5m", ""]) expect(() => waitMinutes(bad, 20)).toThrow("--wait must be a number of minutes");
  });

  test("one vocabulary for both products, and its exit codes", () => {
    expect([monitoringStatus("active"), monitoringStatus("launch_requested"), monitoringStatus("registered"), monitoringStatus("deleting_launched"),
      monitoringStatus("deleted"), monitoringStatus("deactivated"), monitoringStatus("failed"), monitoringStatus("deleting_failed")])
      .toEqual(["ready", "in_progress", "in_progress", "deleting", "deleted", "inactive", "failed", "failed"]);
    expect([dblabStatus("ready"), dblabStatus("launching"), dblabStatus("installing"), dblabStatus("retrieving"), dblabStatus("destroying"),
      dblabStatus("destroyed"), dblabStatus("failed"), dblabStatus("destroy_failed"), dblabStatus(null)])
      .toEqual(["ready", "in_progress", "in_progress", "in_progress", "deleting", "deleted", "failed", "failed", null]);
    expect(DEPLOY_EXIT).toEqual({ ready: 0, in_progress: 0, deleting: 0, deleted: 0, inactive: 1, failed: 1, action_required: 3 });
  });

  test("the not-an-admin step", () => {
    expect(notAdminNext).toBe("Only an organization admin can deploy: ask an admin to run it, or use an admin's API key.");
  });

  test("the no-card step and the billing page", () => {
    expect(noCardNext(billingPage("https://c.example/", "acme"))).toBe("Add a payment method at https://c.example/acme/billing, then re-run.");
    expect(billingPage(undefined, null)).toBe("https://console.postgres.ai (your organization > Billing)");
  });

  test("an instance by id or by name; an ambiguous name is refused with the ids", () => {
    const rows = [{ id: "1", name: "app" }, { id: "2", name: "shop" }, { id: "3", name: "shop" }];
    const pick = (ref: string) => pickInstance(rows, ref, (r) => r.id, (r) => r.name, "pgai x instances list");
    expect(pick("1").name).toBe("app");
    expect(pick("app").id).toBe("1");
    expect(pick("3").id).toBe("3");
    expect(() => pick("shop")).toThrow("More than one instance is named shop (ids 2, 3): use the id. See: pgai x instances list");
    expect(() => pick("nope")).toThrow("No instance nope. See: pgai x instances list");
    // A gone instance does not compete for its name (a name redeployed after a delete); its id still finds it.
    const live = (r: { id: string }) => r.id !== "2";
    expect(pickInstance(rows, "shop", (r) => r.id, (r) => r.name, "l", live).id).toBe("3");
    expect(pickInstance(rows, "2", (r) => r.id, (r) => r.name, "l", live).id).toBe("2");
    // Only gone instances have the name: say so, with their ids.
    expect(() => pickInstance(rows, "shop", (r) => r.id, (r) => r.name, "l", () => false))
      .toThrow("shop was deleted (ids 2, 3): use the id. See: l");
  });

  test("REV r1: JSON with --json, or when stdout is not a terminal", () => {
    const was = process.stdout.isTTY;
    try {
      Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
      expect([jsonOutput(true), jsonOutput(false), jsonOutput(undefined)]).toEqual([true, false, false]);
      Object.defineProperty(process.stdout, "isTTY", { value: false, configurable: true });
      expect([jsonOutput(true), jsonOutput(false), jsonOutput(undefined)]).toEqual([true, true, true]);
    } finally {
      Object.defineProperty(process.stdout, "isTTY", { value: was, configurable: true });
    }
  });

  test("REV r2: a URL whose password is not percent-encoded is refused before any name is taken from it", () => {
    for (const url of ["postgresql://u:12/SecretPart@db.example:5432/app", "postgresql://u:/Secret?Part@db.example.com/app",
      "postgresql://u:99/TopSecret#x@db.example.com/app"]) {
      expect(() => databaseUrlArg(url, undefined, "pgai x")).toThrow("percent-encode");
    }
  });

  test("REV r3: an unencoded '@' after the host is refused, so a password with '@' never reaches a name", () => {
    for (const url of ["postgresql://u:Pa@ss/Word@db.example.com/app", "postgresql://u:x@Top/Secret?y@db.example.com/app",
      "postgresql://u:a@b#c@db.example.com/app", "postgresql://u:p@h/my@db", "postgresql://u:p@h/app?application_name=a@b"]) {
      expect(() => databaseUrlArg(url, undefined, "pgai x")).toThrow("percent-encode");
    }
    // Encoded, or an '@' inside the password before the host: fine.
    for (const url of ["postgresql://u:p@h/my%40db", "postgresql://u:p%40ss@h/app", "postgresql://u:p@ss@h/app?application_name=a%40b"]) {
      expect(databaseUrlArg(url, undefined, "pgai x")).toBe(url);
    }
    expect(nameFromDatabase("postgresql://u:p@h/my%40db")).toBe("my-db");
  });

  test("--vcpus: a whole number from 1 to 1024", () => {
    expect(vcpusArg(undefined)).toBeUndefined();
    expect(vcpusArg("4")).toBe(4);
    expect(vcpusArg(" 1024 ")).toBe(1024);
    for (const bad of ["0", "-1", "1025", "2.5", "x", ""]) expect(() => vcpusArg(bad)).toThrow("--vcpus must be a whole number from 1 to 1024");
  });

  test("a DBLab name from the database's name", () => {
    expect(nameFromDatabase("postgresql://u:p@h:5432/My_Shop%20DB")).toBe("my-shop-db");
    expect(nameFromDatabase("postgresql://u:p@h:5432/d")).toBe("db-d");
    expect(nameFromDatabase("postgresql://u:p@h:5432/")).toBe("dblab");
    expect(nameFromDatabase(`postgresql://u@h/${"a".repeat(60)}`)).toHaveLength(48);
  });

  test("both commands get the same options, words and order", () => {
    const help = (wait: number) => {
      const cmd = sharedDeployOptions(new Command("deploy"), wait, "instance name (default: from the database)");
      return cmd.options.map((o) => `${o.flags} ${o.description}`);
    };
    expect(help(20)).toEqual([
      "--db-url <url> the database URL (the same as the argument)",
      "--name <name> instance name (default: from the database)",
      "--location <location> Hetzner location: fsn1, nbg1 or hel1 (default: any with capacity)",
      "--wait <minutes> how long to wait for it to be ready (0 = do not wait)",
      "--no-wait return once the deploy has started (the same as --wait 0)",
      "-y, --yes never prompt: accept the price and every confirmation",
      "--json JSON output: one result on stdout, progress on stderr (the default when stdout is not a terminal)",
      "--debug print HTTP requests (secrets masked)",
    ]);
  });
});
