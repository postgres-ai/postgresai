/**
 * The one deploy surface `pgai dblab deploy` and `pgai mon deploy` share
 * (postgres-ai/postgresai#412): the database URL, --name, --location,
 * --wait / --no-wait, --yes, --json, the status words and exit codes, the
 * no-card text, and instances named by id or by name.
 */

import type { Command } from "commander";

/** What a deploy, a watch or a delete ended in, as --json says it. */
export type DeployStatus = "ready" | "in_progress" | "deleting" | "deleted" | "inactive" | "failed" | "action_required";

/** 0 done or still going, 1 failed, 3 the user must act ("next" says how). */
export const DEPLOY_EXIT: Record<DeployStatus, number> = {
  ready: 0, in_progress: 0, deleting: 0, deleted: 0, inactive: 1, failed: 1, action_required: 3,
};

/** A monitoring instance's platform state in these words. */
export function monitoringStatus(raw: string | null): DeployStatus {
  if (raw === "active") return "ready";
  if (raw === "deleted") return "deleted";
  if (/delet/.test(raw ?? "") && !/fail/.test(raw ?? "")) return "deleting";
  if (raw === "deactivated") return "inactive";
  if (/fail|error/.test(raw ?? "")) return "failed";
  return "in_progress";
}

/** A DBLab cloud deploy's status in these words (null for a DBLab not deployed by PostgresAI). */
export function dblabStatus(deployStatus: string | null | undefined): DeployStatus | null {
  switch (deployStatus) {
    case "ready": return "ready";
    case "launching": case "installing": case "retrieving": return "in_progress";
    case "destroying": return "deleting";
    case "destroyed": return "deleted";
    case "failed": case "destroy_failed": return "failed";
    default: return null;
  }
}

/** The database URL: the argument, or --db-url (its older spelling); both only when they are the same. */
export function databaseUrlArg(positional: string | undefined, dbUrl: string | undefined, command: string): string {
  const a = positional?.trim();
  const b = dbUrl?.trim();
  if (a && b && a !== b) throw new Error(`Pass the database URL once: as the argument or as --db-url, not both.`);
  const url = a || b;
  if (!url) throw new Error(`Pass the database URL: ${command} postgresql://user:password@host:5432/dbname`);
  assertEncodedUserinfo(url);
  return url;
}

/** --wait <minutes> / --no-wait as minutes (0: return once the deploy has started). */
export function waitMinutes(wait: string | boolean | undefined, defaultMinutes: number): number {
  if (wait === false) return 0;
  if (wait === undefined || wait === true) return defaultMinutes;
  const n = Number(wait);
  if (!/^\s*\d+(\.\d+)?\s*$/.test(wait) || !Number.isFinite(n)) {
    throw new Error("--wait must be a number of minutes (0 = do not wait)");
  }
  return n;
}

/** --vcpus: the database server's vCPUs, a whole number from 1 to 1024 (undefined when not given). */
export function vcpusArg(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const n = Number(raw.trim());
  if (!/^\s*\d+\s*$/.test(raw) || !Number.isInteger(n) || n < 1 || n > 1024) {
    throw new Error("--vcpus must be a whole number from 1 to 1024 (the database server's vCPUs)");
  }
  return n;
}

/**
 * JSON output for a deploy command's result: asked for with --json, or stdout
 * is not a terminal (an agent or a script reads it). One rule for both commands.
 */
export function jsonOutput(json: boolean | undefined): boolean {
  return !!json || !process.stdout.isTTY;
}

/** The next step when the org has no payment method: the same words for both commands. */
export const noCardNext = (billingUrl: string): string => `Add a payment method at ${billingUrl}, then re-run.`;

/** The next step when the token's user is not an org admin (HTTP 403): the same words for both commands. */
export const notAdminNext = "Only an organization admin can deploy: ask an admin to run it, or use an admin's API key.";

/** The org's billing page in the Console. */
export function billingPage(uiBaseUrl: string | undefined, orgAlias: string | null | undefined): string {
  const ui = (uiBaseUrl ?? "https://console.postgres.ai").replace(/\/+$/, "");
  return orgAlias ? `${ui}/${orgAlias}/billing` : `${ui} (your organization > Billing)`;
}

/** The price line both commands print before asking. */
export const estimateLine = (price: string): string => `Estimated cost: ${price}`;

/**
 * One instance named by its id (digits, or the id as listed) or its name. A
 * name more than one instance has is refused, with their ids. An instance that
 * is gone (`live` false) is found by its id, never by its name: a name is
 * reused when an instance is deployed again after a delete.
 */
export function pickInstance<T>(rows: T[], ref: string, idOf: (r: T) => string, nameOf: (r: T) => string, listCommand: string,
  live: (r: T) => boolean = () => true): T {
  const want = ref.trim();
  const byId = rows.filter((r) => idOf(r) === want);
  if (byId.length === 1) return byId[0];
  const byName = rows.filter((r) => live(r) && nameOf(r) === want);
  if (byName.length === 1) return byName[0];
  if (byName.length > 1) {
    throw new Error(`More than one instance is named ${want} (ids ${byName.map(idOf).join(", ")}): use the id. See: ${listCommand}`);
  }
  const gone = rows.filter((r) => !live(r) && nameOf(r) === want);
  if (gone.length) throw new Error(`${want} was deleted (ids ${gone.map(idOf).join(", ")}): use the id. See: ${listCommand}`);
  throw new Error(`No instance ${want}. See: ${listCommand}`);
}

/**
 * Refuses an unencoded '@' anywhere but between the user and the host: a
 * password with a raw '/', '?', '#' or '@' would otherwise be read as the host
 * and the path, and a name taken from the URL would carry part of it.
 */
export function assertEncodedUserinfo(url: string): void {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return;
  }
  if (!u.username && /^[A-Za-z][A-Za-z0-9+.-]*:\/\/.*@/s.test(url.trim())) {
    throw new Error("The database URL has an '@' but no user before the host: percent-encode the password (/ as %2F, ? as %3F, # as %23, @ as %40).");
  }
  if ((u.pathname + u.search + u.hash).includes("@")) {
    throw new Error("The database URL has an '@' after the host: percent-encode it as %40 (in the password, the database name or the query; also / as %2F, ? as %3F, # as %23 in the password).");
  }
}

/** A DBLab name from the database's name: lowercase letters, digits and hyphens, 2 to 48 long. */
export function nameFromDatabase(url: string): string {
  let db = "";
  try {
    db = decodeURIComponent(new URL(url).pathname.replace(/^\//, ""));
  } catch {
    db = "";
  }
  const slug = (db || "dblab").toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48).replace(/-+$/, "");
  return slug.length >= 2 ? slug : `db-${slug || "x"}`.slice(0, 48);
}

/** The options both deploy commands take, with the same names and words. */
export function sharedDeployOptions(cmd: Command, waitDefaultMinutes: number, nameHelp: string): Command {
  return cmd
    .argument("[database-url]", "the database: postgresql://user:password@host:5432/dbname")
    .option("--db-url <url>", "the database URL (the same as the argument)")
    .option("--name <name>", nameHelp)
    .option("--location <location>", "Hetzner location: fsn1, nbg1 or hel1 (default: any with capacity)")
    .option("--wait <minutes>", "how long to wait for it to be ready (0 = do not wait)", String(waitDefaultMinutes))
    .option("--no-wait", "return once the deploy has started (the same as --wait 0)")
    .option("-y, --yes", "never prompt: accept the price and every confirmation")
    .option("--json", "JSON output: one result on stdout, progress on stderr (the default when stdout is not a terminal)")
    .option("--debug", "print HTTP requests (secrets masked)");
}
