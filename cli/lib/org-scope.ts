/**
 * Org selection for global tokens (postgresai #327, platform-all #629).
 *
 * A global token is bound to a user, not an org, so the CLI cannot infer which
 * org a command meant -- and guessing is the silent cross-org write of #250.
 * Two flags, not one: an all-digit alias is legal server-side, so `--org 12345`
 * would have to guess id-vs-alias. Nothing is stored: a sticky default is
 * shared mutable state that parallel agents race (why #324 was rejected).
 */

import * as config from "./config";
import { formatHttpError, maskSecret, normalizeBaseUrl } from "./util";

/** Wire headers the platform's `request_selected_org()` reads. */
export const ORG_ALIAS_HEADER = "x-pgai-org";
export const ORG_ID_HEADER = "x-pgai-org-id";

export const ORG_ENV = "PGAI_ORG";
export const ORG_ID_ENV = "PGAI_ORG_ID";

/** The `--org` / `--org-id` pair, as Commander parses them. */
export interface OrgOptions {
  org?: string;
  orgId?: string | number;
}

export interface OrgScope {
  /** Org alias, if the caller selected by alias. */
  alias?: string;
  /** Numeric org id, if the caller selected by id. */
  id?: number;
  /** Where the selection came from, for error messages and --debug. */
  source: "--org" | "--org-id" | "PGAI_ORG" | "PGAI_ORG_ID" | "token" | "none";
}

export class OrgScopeError extends Error {}

/**
 * Resolve the org selector from flags, then environment. A conflict is an error
 * rather than a silent precedence win: picking one when the user named two is
 * how a command writes somewhere unintended. `source: "none"` means nothing
 * selected an org -- whether that is acceptable is `requireOrgScope`'s call.
 */
/** Trim a flag value, treating present-but-empty as an error rather than absent. */
function emptyFlagGuard(value: unknown, flag: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  const trimmed = `${value}`.trim();
  if (trimmed === "") {
    throw new OrgScopeError(`${flag} was given an empty value. Pass an organization, or omit ${flag}.`);
  }
  return trimmed;
}

/**
 * Header values must be ASCII-printable. Rejecting here names the flag; letting
 * it reach Headers() surfaces as "could not reach <url>", blaming the network
 * for a homoglyph in the alias.
 */
function assertHeaderSafeAlias(alias: string): void {
  if (!/^[\x20-\x7e]+$/.test(alias)) {
    throw new OrgScopeError(
      `--org contains characters that cannot be sent in a header (got '${alias}'). ` +
        "Check for a non-ASCII lookalike, or select the organization with --org-id."
    );
  }
}

export function resolveOrgScope(opts: OrgOptions = {}): OrgScope {
  // Present-but-empty is an error, not "absent". Falling through would let
  // `--org "$ORG"` with $ORG unset quietly inherit PGAI_ORG from the shell and
  // target a different organization than the one the user named.
  const flagAlias = emptyFlagGuard(opts.org, "--org");
  const flagId = emptyFlagGuard(opts.orgId, "--org-id");

  if (flagAlias !== undefined && flagId !== undefined) {
    throw new OrgScopeError("Pass either --org or --org-id, not both.");
  }

  if (flagAlias !== undefined) {
    assertHeaderSafeAlias(flagAlias);
    return { alias: flagAlias, source: "--org" };
  }

  if (flagId !== undefined) {
    return { id: parseOrgId(flagId, "--org-id"), source: "--org-id" };
  }

  const envAlias = trimmedEnv(ORG_ENV);
  const envId = trimmedEnv(ORG_ID_ENV);

  if (envAlias !== undefined && envId !== undefined) {
    throw new OrgScopeError(`Set either ${ORG_ENV} or ${ORG_ID_ENV}, not both.`);
  }

  if (envAlias !== undefined) {
    assertHeaderSafeAlias(envAlias);
    return { alias: envAlias, source: ORG_ENV };
  }

  if (envId !== undefined) {
    return { id: parseOrgId(envId, ORG_ID_ENV), source: ORG_ID_ENV };
  }

  return { source: "none" };
}

/**
 * Resolve the org selector and fail fast when a global token has none.
 *
 * The server enforces this too — `api_token_check()` refuses a global token
 * with no selector — but a local error costs no round trip and can name the
 * flags, the env vars, and the command that lists the available orgs.
 */
export function requireOrgScope(opts: OrgOptions = {}, apiKey?: string | null): OrgScope {
  const scope = resolveOrgScope(opts);

  if (scope.source !== "none") {
    return scope;
  }

  // The EFFECTIVE key, not the stored one. It may have come from --api-key or
  // PGAI_API_KEY, which is precisely the case automation uses most; reading the
  // config file here would misclassify an overridden credential.
  const effectiveKey = apiKey ?? config.readConfig().apiKey;

  if (config.isGlobalTokenValue(effectiveKey)) {
    throw new OrgScopeError(
      [
        "This command needs an organization, and you are signed in with a global token",
        "(which can reach every organization you belong to).",
        "",
        "  pgai <command> --org <alias>      # e.g. --org acme",
        "  pgai <command> --org-id <id>      # e.g. --org-id 5225",
        `  ${ORG_ENV}=acme pgai <command>  # for scripts and agents`,
        "",
        "Run 'pgai orgs' to list the organizations this token can reach.",
      ].join("\n")
    );
  }

  // A legacy per-org token carries its own org, so nothing to select.
  return { source: "token" };
}

/**
 * The org selected for the command currently running, set once by the CLI's
 * preAction hook. Threading a scope argument through the ~40 exported lib
 * functions instead would give 40 chances to forget one, and forgetting is
 * silent. Module state is safe because the CLI runs one command per process;
 * the MCP server serves many, so it passes its scope explicitly instead.
 */
let activeOrgScope: OrgScope | undefined;

export function setActiveOrgScope(scope: OrgScope | undefined): void {
  activeOrgScope = scope;
}

export function getActiveOrgScope(): OrgScope | undefined {
  return activeOrgScope;
}

/**
 * Headers carrying the selection to the platform, merged into the request
 * alongside `access-token`.
 */
/**
 * The org id a request BODY may inherit from the stored config.
 *
 * Under a global token the answer is always "none": the header carries the
 * real selection, so a leftover `orgId` from an earlier per-org login can only
 * contradict it — the server refuses the mismatch, or the command silently
 * returns nothing. Per-org tokens keep the stored value, unchanged.
 */
export function configOrgIdForBody(
  apiKey: string | null | undefined,
  cfgOrgId: number | null | undefined
): number | undefined {
  if (config.isGlobalTokenValue(apiKey)) return undefined;
  return cfgOrgId ?? undefined;
}

export function orgScopeHeaders(scope: OrgScope | undefined): Record<string, string> {
  if (!scope) return {};
  if (scope.alias !== undefined) return { [ORG_ALIAS_HEADER]: scope.alias };
  if (scope.id !== undefined) return { [ORG_ID_HEADER]: String(scope.id) };
  return {};
}

/**
 * A request's JSON auth headers: access token, org selection, per-call extras.
 * Centralised so the org cannot be forgotten at a call site -- forgetting it
 * sends a request with no org rather than failing loudly. storage and
 * checkup-api set their own content framing, so they merge
 * {@link orgScopeHeaders} directly instead of taking this Content-Type.
 */
export function buildAuthHeaders(
  apiKey: string,
  scope?: OrgScope,
  extra: Record<string, string> = {}
): Record<string, string> {
  return {
    "access-token": apiKey,
    "Content-Type": "application/json",
    Connection: "close",
    // Falls back to the invocation's selection, so a lib function that never
    // received an explicit scope still sends the right org.
    ...orgScopeHeaders(scope ?? activeOrgScope),
    ...extra,
  };
}

/** One row of `v1.orgs_list`. */
export interface OrgListItem {
  org_id: number;
  alias: string;
  name: string;
  is_active: boolean;
}

export interface ListOrgsParams {
  apiKey: string;
  apiBaseUrl: string;
  debug?: boolean;
}

/**
 * Organizations the current credential can reach. Deliberately sends NO
 * selector: this is the call you make in order to choose one, which is why the
 * backend authenticates it via api_token_principal rather than
 * api_token_check. A per-org token lists only its own org.
 */
export async function listOrgs(params: ListOrgsParams): Promise<OrgListItem[]> {
  const { apiKey, apiBaseUrl, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = `${base}/rpc/orgs_list`;
  // Headers built explicitly, NOT via buildAuthHeaders: this is the one
  // org-agnostic endpoint and must send no selector even when a process-wide
  // scope is active. resolveOrgIdForBody calls this to turn an alias into an id
  // during `issues create --org <alias>`, at which point the preAction hook has
  // already stashed that alias in activeOrgScope — buildAuthHeaders' fallback
  // would then leak x-pgai-org onto orgs_list, contradicting the contract above
  // and risking a filtered/rejected list exactly when resolving the alias.
  const headers: Record<string, string> = {
    "access-token": apiKey,
    "Content-Type": "application/json",
    Connection: "close",
  };

  if (debug) {
    console.error(`Debug: POST URL: ${url}`);
    console.error(
      `Debug: Request headers: ${JSON.stringify({ ...headers, "access-token": maskSecret(apiKey) })}`
    );
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({}),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(formatHttpError("list organizations", response.status, text));
  }

  const parsed = JSON.parse(text);
  return Array.isArray(parsed) ? (parsed as OrgListItem[]) : [];
}

/**
 * The NUMERIC org id for the few RPCs taking org_id as a body parameter (e.g.
 * v1.issue_create). Most commands never need it -- the org travels as a header
 * -- but for those an alias must become an id somewhere, which costs one extra
 * request, and only when a global token selected by alias.
 */
export async function resolveOrgIdForBody(params: {
  scope: OrgScope;
  apiKey: string;
  apiBaseUrl: string;
  fallbackOrgId?: number | null;
  debug?: boolean;
}): Promise<number | undefined> {
  const { scope, apiKey, apiBaseUrl, fallbackOrgId, debug } = params;

  if (scope.id !== undefined) return scope.id;

  if (scope.alias !== undefined) {
    const orgs = await listOrgs({ apiKey, apiBaseUrl, debug });
    const needle = scope.alias.toLowerCase();
    // `find` takes the first case-insensitive match. Safe to be lax here: the
    // request also carries x-pgai-org, and the server is authoritative -- it
    // refuses an ambiguous alias with PT400 rather than picking, so this id can
    // never decide the org on its own (verified against two orgs differing only
    // by case: both read and write refused, nothing was written).
    const match = orgs.find((o) => (o.alias ?? "").toLowerCase() === needle);
    if (!match) {
      throw new OrgScopeError(
        `Organization '${scope.alias}' is not one this token can reach. Run 'pgai orgs' to list them.`
      );
    }
    return match.org_id;
  }

  // source === "token": a per-org token carries its own org.
  return fallbackOrgId ?? undefined;
}

export function parseOrgId(value: string, sourceLabel: string, remedy = "Use --org for an alias."): number {
  // Bounded, matching the server's ^[0-9]{1,18}$: an unbounded digit run would
  // exceed the platform's bigint and come back as an opaque server error.
  if (!/^[0-9]{1,18}$/.test(value)) {
    throw new OrgScopeError(
      `${sourceLabel} must be a numeric organization id (got '${value}'). ${remedy}`
    );
  }
  const parsed = Number(value);
  // The regex admits up to 18 digits (~10^18), but a JS number is only exact to
  // Number.MAX_SAFE_INTEGER (~9.0×10^15). An id past that is silently rounded and
  // would target a DIFFERENT org than the one typed. Refuse it rather than send a
  // corrupted id; a real org id is small, and an oversized alias belongs on --org.
  if (!Number.isSafeInteger(parsed)) {
    throw new OrgScopeError(
      `${sourceLabel} '${value}' is too large to represent exactly. ${remedy}`
    );
  }
  return parsed;
}

function trimmedEnv(name: string): string | undefined {
  const raw = process.env[name];
  if (typeof raw !== "string") return undefined;
  const trimmed = raw.trim();
  return trimmed === "" ? undefined : trimmed;
}
