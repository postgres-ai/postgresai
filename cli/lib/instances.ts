/**
 * Helpers for managing instances.yml (the pgwatch monitoring target list)
 * and for opening pg connections that honor libpq sslmode semantics.
 *
 * These helpers exist as a single source of truth so that `mon targets
 * add/remove/test` and `mon local-install` (which has its own inline copies
 * of the same logic) stay consistent — and so unit tests exercise the same
 * code path the CLI actually runs.
 */

import * as fs from "fs";
import * as yaml from "js-yaml";
import { parse as parseConnString } from "pg-connection-string";
import type { ClientConfig } from "pg";

export interface Instance {
  name: string;
  conn_str?: string;
  preset_metrics?: string;
  custom_metrics?: any;
  is_enabled?: boolean;
  group?: string;
  custom_tags?: Record<string, any>;
}

export class InstancesParseError extends Error {
  constructor(file: string, cause: unknown) {
    const causeMsg = cause instanceof Error ? cause.message : String(cause);
    super(`Failed to parse ${file}: ${causeMsg}`);
    this.name = "InstancesParseError";
  }
}

/**
 * Read instances.yml as an array of Instance.
 *
 * Returns `[]` for a missing/empty file (this is normal — fresh installs and
 * just-after-`remove` states). Throws InstancesParseError on a corrupted file
 * so callers can surface the corruption to the user instead of silently
 * overwriting it (the previous append-text behavior could erase several
 * targets — including their conn_strs with credentials — if the file had a
 * partial write or hand-edit problem).
 */
export function loadInstances(file: string): Instance[] {
  if (!fs.existsSync(file)) return [];
  if (fs.lstatSync(file).isDirectory()) return [];
  const text = fs.readFileSync(file, "utf8");
  if (text.trim() === "") return [];
  let parsed: unknown;
  try {
    parsed = yaml.load(text);
  } catch (err) {
    throw new InstancesParseError(file, err);
  }
  if (parsed === null || parsed === undefined) return [];
  if (!Array.isArray(parsed)) {
    throw new InstancesParseError(file, "expected a YAML list at the document root");
  }
  return parsed as Instance[];
}

export function buildInstance(name: string, connStr: string): Instance {
  return {
    name,
    conn_str: connStr,
    preset_metrics: "full",
    custom_metrics: null,
    is_enabled: true,
    group: "default",
    custom_tags: {
      env: "production",
      cluster: "default",
      node_name: name,
      // Sed-substituted placeholder by config/scripts/generate-pgwatch-sources.sh.
      // js-yaml emits this unquoted on dump (~ is only special at the start of a
      // scalar in the right context); sed s/~sink_type~/.../g still hits it as
      // raw text regardless.
      sink_type: "~sink_type~",
    },
  };
}

/**
 * Parse → mutate → serialize: load existing list, append, dump back.
 *
 * Replaces the previous text-append code path which corrupted instances.yml
 * after `remove` had left the empty marker `[]` in the file (the append
 * produced two YAML documents in one file → parse error on every subsequent
 * read).
 *
 * Replaces files where the previous code path treated the directory created
 * by Docker's bind-mount-into-missing-path as a target.
 */
export function addInstanceToFile(file: string, instance: Instance): void {
  if (fs.existsSync(file) && fs.lstatSync(file).isDirectory()) {
    fs.rmSync(file, { recursive: true, force: true });
  }
  const existing = loadInstances(file);
  if (existing.some((i) => i.name === instance.name)) {
    throw new Error(`Monitoring target '${instance.name}' already exists`);
  }
  existing.push(instance);
  fs.writeFileSync(file, yaml.dump(existing), "utf8");
}

/**
 * Remove a named instance from the file. Returns true if removed.
 */
export function removeInstanceFromFile(file: string, name: string): boolean {
  const instances = loadInstances(file);
  const filtered = instances.filter((i) => i.name !== name);
  if (filtered.length === instances.length) return false;
  fs.writeFileSync(file, yaml.dump(filtered), "utf8");
  return true;
}

/**
 * Extract `sslmode` (lowercased) from a postgresql:// URL. Returns `""` for
 * unset or unparseable.
 */
export function extractSslmode(connStr: string): string {
  try {
    return (new URL(connStr).searchParams.get("sslmode") || "").toLowerCase();
  } catch {
    return "";
  }
}

/**
 * Map libpq sslmode values to node-postgres' `ssl` option.
 *
 * libpq:                     node-postgres ssl:
 *   disable                    false
 *   allow / prefer / require   { rejectUnauthorized: false }   (encrypt, no chain check)
 *   verify-ca                  { rejectUnauthorized: true, checkServerIdentity: () => undefined }
 *   verify-full                { rejectUnauthorized: true }
 *   no-verify (pg extension)   { rejectUnauthorized: false }
 *
 * Default for unset: prefer-like → no chain verification, matches what
 * pgwatch (Go pgx) does and what users pass to psql every day.
 */
export type SslOption = false | { rejectUnauthorized: boolean; checkServerIdentity?: () => undefined };

export function sslOptionFromConnString(connStr: string): SslOption {
  return sslOptionFromSslmode(extractSslmode(connStr));
}

function sslOptionFromSslmode(sslmode: string): SslOption {
  switch (sslmode) {
    case "disable":
      return false;
    case "verify-ca":
      return { rejectUnauthorized: true, checkServerIdentity: () => undefined };
    case "verify-full":
      return { rejectUnauthorized: true };
    case "allow":
    case "prefer":
    case "require":
    case "no-verify":
    case "":
    default:
      return { rejectUnauthorized: false };
  }
}

/**
 * The set of sslmode values for which we DO NOT verify the certificate chain.
 * Exposed so callers can warn users about the lax security posture.
 */
export const LAX_SSLMODES = new Set(["", "allow", "prefer", "require"]);

export function isLaxSslmode(sslmode: string): boolean {
  return LAX_SSLMODES.has(sslmode);
}

/**
 * Print a stderr warning when the connection string uses an sslmode that
 * skips certificate-chain verification. Centralises the message so the
 * three Client-construction sites stay consistent.
 */
export function warnIfLaxSslmode(connStr: string): void {
  const sslmode = extractSslmode(connStr);
  if (!isLaxSslmode(sslmode)) return;
  const shown = sslmode || "(unset)";
  console.error(
    `⚠ sslmode=${shown}: TLS chain is NOT verified (matches libpq/psql semantics). ` +
      `Use sslmode=verify-full for full chain+hostname verification.`,
  );
}

/**
 * Build a `pg.Client` config from a connection string that ACTUALLY honors
 * libpq sslmode semantics.
 *
 * This is non-trivial because node-postgres' `Client` constructor, given a
 * `connectionString`, runs `Object.assign({}, config, parse(connectionString))`
 * — meaning the parsed sslmode-derived `ssl` value REPLACES any explicit
 * `ssl` you passed alongside `connectionString`. So setting
 * `{ connectionString, ssl: { rejectUnauthorized: false } }` does not work
 * when the URL contains `sslmode=require` (the parsed value `{}` wins, and
 * `{}` defaults to chain verification → "self-signed certificate in
 * certificate chain" against managed Postgres).
 *
 * The fix: parse the URL ourselves, pass discrete host/port/user/etc., and
 * include our explicit `ssl` — never pass `connectionString` so nothing
 * overrides us.
 *
 * We strip `sslmode` from the URL before handing it to `pg-connection-string`'s
 * `parse()` so that:
 *   1. its `process.emitWarning("SECURITY WARNING: SSL modes 'prefer'/'require'/
 *      'verify-ca' are treated as aliases for 'verify-full'…")` doesn't fire
 *      on every CLI invocation against a Supabase-shaped URL, and
 *   2. its `verify-ca` compatibility branch doesn't *throw* on us (requires
 *      sslrootcert which we don't have).
 *
 * We don't use the parser's `ssl` output anyway — we compute our own from
 * `sslOptionFromSslmode(extractSslmode(originalConnStr))` — so removing the
 * sslmode parameter before parsing is a no-op for the fields we actually use.
 */
export function buildClientConfig(
  connStr: string,
  extra: { connectionTimeoutMillis?: number } = {},
): ClientConfig {
  const sslmode = extractSslmode(connStr);
  const parsed = parseConnString(withoutSslmode(connStr));
  return {
    host: parsed.host || undefined,
    port: parsed.port ? Number(parsed.port) : undefined,
    user: parsed.user,
    password: parsed.password,
    database: parsed.database || undefined,
    ssl: sslOptionFromSslmode(sslmode),
    ...extra,
  };
}

function withoutSslmode(connStr: string): string {
  try {
    const u = new URL(connStr);
    u.searchParams.delete("sslmode");
    return u.toString();
  } catch {
    return connStr;
  }
}
