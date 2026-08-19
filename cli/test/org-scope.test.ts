import { describe, test, expect, beforeEach, afterEach } from "bun:test";

import {
  ORG_ALIAS_HEADER,
  ORG_ENV,
  ORG_ID_ENV,
  ORG_ID_HEADER,
  OrgScopeError,
  buildAuthHeaders,
  configOrgIdForBody,
  orgScopeHeaders,
  requireOrgScope,
  resolveOrgScope,
} from "../lib/org-scope";
import { GLOBAL_TOKEN_PREFIX, isGlobalTokenValue } from "../lib/config";

const GLOBAL_TOKEN = `${GLOBAL_TOKEN_PREFIX}${"a".repeat(43)}`;
// A legacy token is double-base64 of "<org_id>.<secret>".
const LEGACY_TOKEN = Buffer.from(
  Buffer.from("5225.abcdefghijklmnopqrstuvwx").toString("base64")
).toString("base64");

let savedOrg: string | undefined;
let savedOrgId: string | undefined;

beforeEach(() => {
  savedOrg = process.env[ORG_ENV];
  savedOrgId = process.env[ORG_ID_ENV];
  delete process.env[ORG_ENV];
  delete process.env[ORG_ID_ENV];
});

afterEach(() => {
  if (savedOrg === undefined) delete process.env[ORG_ENV];
  else process.env[ORG_ENV] = savedOrg;
  if (savedOrgId === undefined) delete process.env[ORG_ID_ENV];
  else process.env[ORG_ID_ENV] = savedOrgId;
});

describe("token kind detection", () => {
  test("recognises a global token by its wire prefix", () => {
    expect(isGlobalTokenValue(GLOBAL_TOKEN)).toBe(true);
  });

  test("a legacy per-org token is not global", () => {
    expect(isGlobalTokenValue(LEGACY_TOKEN)).toBe(false);
  });

  test("absent or empty keys are not global", () => {
    expect(isGlobalTokenValue(null)).toBe(false);
    expect(isGlobalTokenValue(undefined)).toBe(false);
    expect(isGlobalTokenValue("")).toBe(false);
  });
});

describe("resolveOrgScope", () => {
  test("--org selects by alias", () => {
    expect(resolveOrgScope({ org: "acme" })).toEqual({ alias: "acme", source: "--org" });
  });

  test("--org-id selects by numeric id", () => {
    expect(resolveOrgScope({ orgId: "5225" })).toEqual({ id: 5225, source: "--org-id" });
  });

  test("naming both is an error rather than a silent precedence win", () => {
    // Picking one when the user named two is how a command writes somewhere
    // unintended.
    expect(() => resolveOrgScope({ org: "acme", orgId: "5225" })).toThrow(OrgScopeError);
  });

  test("an all-digit value passed to --org stays an ALIAS", () => {
    // orgs.alias has no format constraint server-side, so "12345" is a legal
    // alias. The two-flag split is what makes it addressable at all.
    expect(resolveOrgScope({ org: "12345" })).toEqual({ alias: "12345", source: "--org" });
  });

  test("--org-id rejects a non-numeric value", () => {
    expect(() => resolveOrgScope({ orgId: "acme" })).toThrow(OrgScopeError);
  });

  test("--org-id rejects an over-long value instead of overflowing server-side", () => {
    // Mirrors the server's ^[0-9]{1,18}$; an unbounded run would exceed bigint
    // and come back as an opaque error.
    expect(() => resolveOrgScope({ orgId: "99999999999999999999" })).toThrow(OrgScopeError);
  });

  test("falls back to PGAI_ORG when no flag is given", () => {
    process.env[ORG_ENV] = "globex";
    expect(resolveOrgScope({})).toEqual({ alias: "globex", source: ORG_ENV });
  });

  test("falls back to PGAI_ORG_ID when no flag is given", () => {
    process.env[ORG_ID_ENV] = "6333";
    expect(resolveOrgScope({})).toEqual({ id: 6333, source: ORG_ID_ENV });
  });

  test("a flag beats the environment", () => {
    process.env[ORG_ENV] = "globex";
    expect(resolveOrgScope({ org: "acme" })).toEqual({ alias: "acme", source: "--org" });
  });

  test("setting both env vars is an error", () => {
    process.env[ORG_ENV] = "globex";
    process.env[ORG_ID_ENV] = "6333";
    expect(() => resolveOrgScope({})).toThrow(OrgScopeError);
  });

  test("a blank ENVIRONMENT value is treated as absent", () => {
    // Unset-but-exported is normal in shells and CI; it must not select an org.
    process.env[ORG_ENV] = "   ";
    expect(resolveOrgScope({})).toEqual({ source: "none" });
  });

  test("a blank FLAG is an error, not an absent selection", () => {
    // Deliberate change from the original behaviour, which treated a blank flag
    // as absent: that let `--org "$ORG"` with $ORG unset fall through to
    // PGAI_ORG and silently target a different org. Neither reading produces an
    // empty org header; the difference is falling through vs refusing.
    expect(() => resolveOrgScope({ org: "  " })).toThrow(OrgScopeError);
  });

  test("nothing selected reports source 'none'", () => {
    expect(resolveOrgScope({})).toEqual({ source: "none" });
  });
});

describe("requireOrgScope", () => {
  test("a global token with no selector fails, and the message says how to fix it", () => {
    let message = "";
    try {
      requireOrgScope({}, GLOBAL_TOKEN);
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toContain("--org");
    expect(message).toContain("--org-id");
    expect(message).toContain(ORG_ENV);
    expect(message).toContain("pgai orgs");
  });

  test("a legacy token with no selector is fine — it carries its own org", () => {
    // The backward-compatibility guarantee: nothing changes for existing users.
    expect(requireOrgScope({}, LEGACY_TOKEN)).toEqual({ source: "token" });
  });

  test("a global token with a selector resolves", () => {
    expect(requireOrgScope({ org: "acme" }, GLOBAL_TOKEN)).toEqual({
      alias: "acme",
      source: "--org",
    });
  });

  test("PGAI_ORG satisfies a global token, so agents need no flag threading", () => {
    process.env[ORG_ENV] = "acme";
    expect(requireOrgScope({}, GLOBAL_TOKEN)).toEqual({ alias: "acme", source: ORG_ENV });
  });

  test("classification uses the EFFECTIVE key, not whatever is in the config file", () => {
    // A key from --api-key/PGAI_API_KEY must be classified on its own merits;
    // this is the case automation uses most.
    expect(() => requireOrgScope({}, GLOBAL_TOKEN)).toThrow(OrgScopeError);
    expect(requireOrgScope({}, LEGACY_TOKEN)).toEqual({ source: "token" });
  });
});

describe("wire headers", () => {
  test("an alias selection travels as x-pgai-org", () => {
    expect(orgScopeHeaders({ alias: "acme", source: "--org" })).toEqual({
      [ORG_ALIAS_HEADER]: "acme",
    });
  });

  test("an id selection travels as x-pgai-org-id", () => {
    expect(orgScopeHeaders({ id: 5225, source: "--org-id" })).toEqual({
      [ORG_ID_HEADER]: "5225",
    });
  });

  test("the two headers are never sent together", () => {
    // The server rejects both-at-once, so the client must never produce it.
    const headers = orgScopeHeaders({ alias: "acme", id: 5225, source: "--org" });
    expect(Object.keys(headers)).toEqual([ORG_ALIAS_HEADER]);
  });

  test("a legacy token sends no org header at all", () => {
    expect(orgScopeHeaders({ source: "token" })).toEqual({});
    expect(orgScopeHeaders(undefined)).toEqual({});
  });

  test("buildAuthHeaders carries the token, the org, and any extras", () => {
    const headers = buildAuthHeaders(GLOBAL_TOKEN, { alias: "acme", source: "--org" }, {
      Prefer: "return=representation",
    });
    expect(headers["access-token"]).toBe(GLOBAL_TOKEN);
    expect(headers[ORG_ALIAS_HEADER]).toBe("acme");
    expect(headers["Prefer"]).toBe("return=representation");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  test("buildAuthHeaders omits the org header when nothing was selected", () => {
    const headers = buildAuthHeaders(LEGACY_TOKEN, { source: "token" });
    expect(headers[ORG_ALIAS_HEADER]).toBeUndefined();
    expect(headers[ORG_ID_HEADER]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// A stale config orgId must never reach a request body under a global token.
//
// `pgai projects --org test_org` was observed sending the right header
// (x-pgai-org: test_org) alongside a body of {"org_id": 90} left over from an
// earlier per-org login, and coming back with a silent empty list. The header
// carries the real selection; a body value from config can only contradict it.
// ---------------------------------------------------------------------------
describe("configOrgIdForBody", () => {
  test("a global token never inherits the stored orgId", () => {
    expect(configOrgIdForBody(GLOBAL_TOKEN, 90)).toBeUndefined();
  });

  test("a per-org token keeps the stored orgId (unchanged behaviour)", () => {
    expect(configOrgIdForBody(LEGACY_TOKEN, 90)).toBe(90);
  });

  test("null/undefined stored orgId is normalized to undefined", () => {
    expect(configOrgIdForBody(LEGACY_TOKEN, null)).toBeUndefined();
    expect(configOrgIdForBody(LEGACY_TOKEN, undefined)).toBeUndefined();
  });

  test("a missing api key is treated as non-global and keeps the value", () => {
    expect(configOrgIdForBody(undefined, 90)).toBe(90);
  });
});

describe("a flag the user typed never falls through to the environment", () => {
  test("--org '' is an error, not a silent hand-off to PGAI_ORG", () => {
    // `pgai issues list --org "$ORG"` with $ORG unset, in a shell exporting
    // PGAI_ORG for a different org, silently targeted the environment's org.
    process.env[ORG_ENV] = "envorg";
    expect(() => resolveOrgScope({ org: "" })).toThrow(OrgScopeError);
  });

  test("--org-id '' is an error too", () => {
    process.env[ORG_ID_ENV] = "6333";
    expect(() => resolveOrgScope({ orgId: "" })).toThrow(OrgScopeError);
  });

  test("an unset flag still falls through to the environment", () => {
    process.env[ORG_ENV] = "envorg";
    expect(resolveOrgScope({})).toEqual({ alias: "envorg", source: ORG_ENV });
  });

  test("a whitespace-only environment value is still treated as unset", () => {
    process.env[ORG_ENV] = "   ";
    expect(resolveOrgScope({})).toEqual({ source: "none" });
  });
});

describe("alias values that cannot become a header", () => {
  test("a non-ASCII alias is rejected by name, not as a connectivity failure", () => {
    // A Cyrillic homoglyph paste ("аcme") was rejected by Headers() deep inside
    // fetch, and the caller reframed it as "could not reach <url>".
    let message = "";
    try {
      resolveOrgScope({ org: "аcme" });
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toContain("--org");
  });

  test("ordinary aliases with dots and dashes still pass", () => {
    expect(resolveOrgScope({ org: "acme-corp.eu" })).toEqual({
      alias: "acme-corp.eu",
      source: "--org",
    });
  });
});
