import { describe, test, expect } from "bun:test";

import { resolveMcpOrgScope } from "../lib/org-scope-mcp";
import { GLOBAL_TOKEN_PREFIX } from "../lib/config";

const GLOBAL_TOKEN = `${GLOBAL_TOKEN_PREFIX}${"a".repeat(43)}`;
const LEGACY_TOKEN = Buffer.from(
  Buffer.from("5225.abcdefghijklmnopqrstuvwx").toString("base64")
).toString("base64");

describe("MCP org resolution — per-org token (unchanged behaviour)", () => {
  test("falls back to the stored org when the tool call omits org_id", () => {
    const result = resolveMcpOrgScope({}, LEGACY_TOKEN, { orgId: 5225 });
    expect(result.error).toBeUndefined();
    expect(result.orgId).toBe(5225);
  });

  test("an explicit org_id wins over the stored one", () => {
    const result = resolveMcpOrgScope({ org_id: 6333 }, LEGACY_TOKEN, { orgId: 5225 });
    expect(result.orgId).toBe(6333);
  });

  test("no org_id and no stored org leaves it undefined for the caller to reject", () => {
    const result = resolveMcpOrgScope({}, LEGACY_TOKEN, { orgId: null });
    expect(result.error).toBeUndefined();
    expect(result.orgId).toBeUndefined();
  });
});

describe("MCP org resolution — global token (closes #250)", () => {
  test("omitting org_id is an error, not a silent default", () => {
    // The whole point of #250: a silently defaulted org lets an agent write
    // into the wrong organization, caught on edit but not on create.
    const result = resolveMcpOrgScope({}, GLOBAL_TOKEN, { orgId: 5225 });
    expect(result.error).toBeDefined();
    expect(result.error?.isError).toBe(true);
    expect(result.orgId).toBeUndefined();
  });

  test("the stored org is NEVER used as a fallback", () => {
    // A stored orgId belongs to some previous per-org login; honouring it here
    // would silently re-scope the call.
    const result = resolveMcpOrgScope({}, GLOBAL_TOKEN, { orgId: 5225 });
    expect(result.orgId).not.toBe(5225);
    expect(result.error).toBeDefined();
  });

  test("the error tells the model how to recover", () => {
    const text = resolveMcpOrgScope({}, GLOBAL_TOKEN, { orgId: null }).error?.content[0]?.text ?? "";
    expect(text).toContain("org_id is required");
    expect(text).toContain("orgs_list");
  });

  test("an explicit org_id resolves and also travels as a header", () => {
    // Header as well as body: the server scopes the request itself rather than
    // trusting a body field the caller could set to any org.
    const result = resolveMcpOrgScope({ org_id: 6333 }, GLOBAL_TOKEN, { orgId: null });
    expect(result.error).toBeUndefined();
    expect(result.orgId).toBe(6333);
    expect(result.orgScope).toEqual({ id: 6333, source: "--org-id" });
  });

  test("a blank org_id counts as absent", () => {
    expect(resolveMcpOrgScope({ org_id: "  " }, GLOBAL_TOKEN, { orgId: null }).error).toBeDefined();
  });

  test("a non-numeric org_id is rejected rather than sent as NaN", () => {
    expect(resolveMcpOrgScope({ org_id: "acme" }, GLOBAL_TOKEN, { orgId: null }).error).toBeDefined();
  });
});

/**
 * The CLI's parseOrgId refuses ids it cannot represent exactly, because "an id
 * past MAX_SAFE_INTEGER is silently rounded and would target a DIFFERENT org
 * than the one typed". MCP reaches the same server with the same header, on the
 * surface an agent drives unattended, so it must refuse them too.
 */
describe("resolveMcpOrgScope id validation", () => {
  const reject = ["9007199254740993", "1e400", "0x10", "-5", "acme", "1.5"];
  for (const value of reject) {
    test(`rejects ${JSON.stringify(value)}`, () => {
      const r = resolveMcpOrgScope({ org_id: value }, GLOBAL_TOKEN, { orgId: null });
      expect({ value, isError: r.error?.isError === true }).toEqual({ value, isError: true });
    });
  }

  test("still accepts ordinary ids, including a zero-padded one", () => {
    expect(resolveMcpOrgScope({ org_id: "007" }, GLOBAL_TOKEN, { orgId: null }).orgScope).toEqual({
      id: 7,
      source: "--org-id",
    });
    expect(resolveMcpOrgScope({ org_id: 5225 }, GLOBAL_TOKEN, { orgId: null }).orgScope).toEqual({
      id: 5225,
      source: "--org-id",
    });
  });

  test("a per-org token rejects a malformed id too, rather than passing it through", () => {
    const r = resolveMcpOrgScope({ org_id: "9007199254740993" }, LEGACY_TOKEN, { orgId: 42 });
    expect(r.error?.isError).toBe(true);
  });
});

describe("MCP errors do not advise CLI flags", () => {
  // MCP has no --org and no alias argument at all, so "Use --org for an alias"
  // is not merely the wrong surface -- it is unactionable. Reusing the CLI's
  // parseOrgId verbatim leaked it.
  for (const value of ["9007199254740993", "0x10"]) {
    test(`the message for ${value} names an MCP-reachable remedy`, () => {
      const r = resolveMcpOrgScope({ org_id: value }, GLOBAL_TOKEN, { orgId: null });
      const text = r.error?.content[0].text ?? "";
      expect({ value, mentionsFlag: text.includes("--org") }).toEqual({ value, mentionsFlag: false });
      expect(text).toMatch(/orgs_list|pgai orgs/);
    });
  }
});
