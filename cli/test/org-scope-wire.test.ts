/**
 * The seam these tests guard: a command may accept `--org`, the preAction hook
 * may resolve it, and the request can STILL go out with no org header if the lib
 * module that builds the request never reads the scope. The 35 pure-function
 * tests in org-scope.test.ts pass regardless of whether any command calls
 * buildAuthHeaders — which is exactly how an incomplete wiring shipped green.
 *
 * So these assert the org selector reaches an ACTUAL request for every
 * org-scoped lib module (reports, joe, projects, dblab, checkup, storage),
 * covers resolveOrgIdForBody / the parseOrgId boundary, threads through the MCP
 * tool surface, and drift-guards the ORG_SCOPED_COMMANDS registration list.
 */
import { describe, test, expect, mock, afterEach, spyOn } from "bun:test";
import { EventEmitter } from "events";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as http from "http";

import {
  ORG_ALIAS_HEADER,
  ORG_ID_HEADER,
  OrgScopeError,
  listOrgs,
  resolveOrgIdForBody,
  resolveOrgScope,
  setActiveOrgScope,
  type OrgScope,
} from "../lib/org-scope";
import { GLOBAL_TOKEN_PREFIX } from "../lib/config";
import { fetchReports } from "../lib/reports";
import { listProjects, startCommand } from "../lib/joe";
import { listClones, createClone } from "../lib/dblab";
import { verifyApiKey, createCheckupReport } from "../lib/checkup-api";
import { uploadFile } from "../lib/storage";
import { handleToolCall } from "../lib/mcp-server";
import { ORG_SCOPED_COMMANDS, program } from "../bin/postgres-ai";
import type { Command } from "commander";

/**
 * Leaf commands not registered as org-scoped, so the drift guard below can treat
 * anything else as a wiring mistake. Adding a command here is a review decision.
 *
 * "Not registered" is not the same as "never touches an org": `mon local-install`
 * does register with the platform, but only when an API key is present and
 * --demo is off, so it takes the selector as plain options and enforces it at
 * the registration call rather than for every install.
 */
const ORG_AGNOSTIC_COMMANDS = [
  "auth login",
  "auth remove-key",
  "auth show-key",
  "feedback",
  "login",
  "mcp install",
  "mcp start",
  "mon check",
  "mon clean",
  "mon config",
  "mon generate-grafana-password",
  "mon health",
  "mon local-install",
  "mon logs",
  "mon reset",
  "mon restart",
  "mon shell",
  "mon show-grafana-credentials",
  "mon start",
  "mon status",
  "mon stop",
  "mon targets add",
  "mon targets list",
  "mon targets remove",
  "mon targets test",
  "mon update",
  "mon update-config",
  "orgs",
  "prepare-db",
  "set-default-project",
  "set-storage-url",
  "unprepare-db",
].sort();

const GLOBAL_TOKEN = `${GLOBAL_TOKEN_PREFIX}${"a".repeat(43)}`;
const API_BASE = "https://api.example.com";
const ALIAS_SCOPE: OrgScope = { alias: "acme", source: "--org" };

const originalFetch = globalThis.fetch;

/** Capture the URL and headers of the first request a stubbed fetch sees. */
function captureFetch(
  body: unknown,
  status = 200
): { headers: () => Record<string, string>; url: () => string } {
  let seen: Record<string, string> = {};
  let seenUrl = "";
  let first = true;
  globalThis.fetch = mock((url: string, options: RequestInit) => {
    if (first) {
      seen = (options.headers as Record<string, string>) ?? {};
      seenUrl = String(url);
      first = false;
    }
    return Promise.resolve(
      new Response(typeof body === "string" ? body : JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      })
    );
  }) as unknown as typeof fetch;
  return { headers: () => seen, url: () => seenUrl };
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  setActiveOrgScope(undefined);
});

// ---------------------------------------------------------------------------
// The org selector reaches a real request — one per org-scoped lib module.
//
// The CLI path relies on activeOrgScope (set once by the preAction hook); the
// modules also called from MCP (reports, storage) additionally accept an
// explicit scope. Both routes are exercised.
// ---------------------------------------------------------------------------
describe("the org header reaches the wire", () => {
  test("reports: fetchReports emits x-pgai-org via an explicit scope", async () => {
    const cap = captureFetch([]);
    await fetchReports({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE, orgScope: ALIAS_SCOPE });
    expect(cap.headers()[ORG_ALIAS_HEADER]).toBe("acme");
  });

  test("reports: fetchReports emits x-pgai-org via the activeOrgScope fallback", async () => {
    // This is the CLI path: no scope is threaded through the call, it rides the
    // process-wide selection the preAction hook stashed.
    setActiveOrgScope(ALIAS_SCOPE);
    const cap = captureFetch([]);
    await fetchReports({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    expect(cap.headers()[ORG_ALIAS_HEADER]).toBe("acme");
  });

  test("joe/projects: listProjects emits x-pgai-org-id from activeOrgScope", async () => {
    setActiveOrgScope({ id: 5225, source: "--org-id" });
    const cap = captureFetch([]);
    await listProjects({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    expect(cap.headers()[ORG_ID_HEADER]).toBe("5225");
  });

  test("joe: a command run (joe_command_run rpc) carries the org header", async () => {
    setActiveOrgScope(ALIAS_SCOPE);
    const cap = captureFetch(JSON.stringify("12345"));
    await startCommand({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE, instanceId: "7", command: "plan select 1" });
    expect(cap.headers()[ORG_ALIAS_HEADER]).toBe("acme");
  });

  test("dblab: a dblab_api_call carries the org header", async () => {
    setActiveOrgScope(ALIAS_SCOPE);
    const cap = captureFetch([]);
    await listClones({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE, instanceId: "7" });
    expect(cap.headers()[ORG_ALIAS_HEADER]).toBe("acme");
  });

  test("dblab: a role-gated DELETE (clone create/destroy path) also carries it", async () => {
    setActiveOrgScope({ id: 5225, source: "--org-id" });
    const cap = captureFetch({});
    await createClone({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE, instanceId: "7" });
    expect(cap.headers()[ORG_ID_HEADER]).toBe("5225");
  });

  test("checkup: the verify pre-flight (GET /checkup_reports) carries the org header", async () => {
    setActiveOrgScope(ALIAS_SCOPE);
    const cap = captureFetch([]);
    await verifyApiKey({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    expect(cap.headers()[ORG_ALIAS_HEADER]).toBe("acme");
  });

  test("checkup: the upload path (checkup_report_create over node:http) carries the org header", async () => {
    setActiveOrgScope({ id: 5225, source: "--org-id" });
    let seen: Record<string, string> = {};
    const spy = spyOn(http, "request").mockImplementation(((_url: unknown, options: { headers?: Record<string, string> }, cb: (res: unknown) => void) => {
      seen = options.headers ?? {};
      const res = new EventEmitter() as unknown as { statusCode: number; setEncoding: (e: string) => void } & EventEmitter;
      res.statusCode = 200;
      res.setEncoding = () => {};
      const req = new EventEmitter() as EventEmitter & { write: () => void; end: () => void; destroy: () => void };
      req.write = () => {};
      req.destroy = () => {};
      req.end = () => {
        cb(res);
        (res as EventEmitter).emit("data", JSON.stringify({ report_id: 42 }));
        (res as EventEmitter).emit("end");
      };
      return req as unknown as http.ClientRequest;
    }) as unknown as typeof http.request);
    try {
      const result = await createCheckupReport({ apiKey: GLOBAL_TOKEN, apiBaseUrl: "http://localhost:9999", project: "demo" });
      expect(result.reportId).toBe(42);
      expect(seen[ORG_ID_HEADER]).toBe("5225");
    } finally {
      spy.mockRestore();
    }
  });

  test("storage: uploadFile emits x-pgai-org and keeps its multipart Content-Type", async () => {
    const tmp = path.join(os.tmpdir(), `pgai-org-scope-${process.pid}.txt`);
    fs.writeFileSync(tmp, "hello");
    try {
      const cap = captureFetch({ success: true, url: "/files/1/x.txt", metadata: {}, requestId: "r" });
      await uploadFile({ apiKey: GLOBAL_TOKEN, orgScope: ALIAS_SCOPE, storageBaseUrl: API_BASE, filePath: tmp });
      const headers = cap.headers();
      expect(headers[ORG_ALIAS_HEADER]).toBe("acme");
      // Must NOT force application/json — FormData supplies its own boundary.
      expect(headers["Content-Type"]).toBeUndefined();
    } finally {
      fs.unlinkSync(tmp);
    }
  });
});

// ---------------------------------------------------------------------------
// resolveOrgIdForBody — the alias→org_id resolver `issues create` depends on.
// ---------------------------------------------------------------------------
describe("resolveOrgIdForBody", () => {
  test("an id selection is returned without a network round trip", async () => {
    globalThis.fetch = mock(() => {
      throw new Error("should not fetch when the id is already known");
    }) as unknown as typeof fetch;
    const id = await resolveOrgIdForBody({
      scope: { id: 5225, source: "--org-id" },
      apiKey: GLOBAL_TOKEN,
      apiBaseUrl: API_BASE,
    });
    expect(id).toBe(5225);
  });

  test("an alias resolves case-insensitively against the orgs listing", async () => {
    captureFetch([
      { org_id: 5225, alias: "acme", name: "Acme", is_active: true },
      { org_id: 6333, alias: "globex", name: "Globex", is_active: true },
    ]);
    const id = await resolveOrgIdForBody({
      scope: { alias: "ACME", source: "--org" },
      apiKey: GLOBAL_TOKEN,
      apiBaseUrl: API_BASE,
    });
    expect(id).toBe(5225);
  });

  test("an alias the token cannot reach is an OrgScopeError, not a silent miss", async () => {
    captureFetch([{ org_id: 5225, alias: "acme", name: "Acme", is_active: true }]);
    await expect(
      resolveOrgIdForBody({
        scope: { alias: "nope", source: "--org" },
        apiKey: GLOBAL_TOKEN,
        apiBaseUrl: API_BASE,
      })
    ).rejects.toBeInstanceOf(OrgScopeError);
  });

  test("a per-org token (source: token) uses the stored fallback org", async () => {
    globalThis.fetch = mock(() => {
      throw new Error("a per-org token must not need the orgs listing");
    }) as unknown as typeof fetch;
    const id = await resolveOrgIdForBody({
      scope: { source: "token" },
      apiKey: "legacy",
      apiBaseUrl: API_BASE,
      fallbackOrgId: 4,
    });
    expect(id).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// listOrgs — the discovery call. It is the ONE org-agnostic endpoint: it must
// send no org selector even when a process-wide scope is active, or orgs_list
// (authenticated via api_token_principal) could come back filtered/rejected
// exactly when resolveOrgIdForBody needs the full list to resolve an alias.
// ---------------------------------------------------------------------------
describe("listOrgs", () => {
  test("POSTs to orgs_list and returns the parsed org rows", async () => {
    const cap = captureFetch([
      { org_id: 5225, alias: "acme", name: "Acme", is_active: true },
      { org_id: 6333, alias: "globex", name: "Globex", is_active: true },
    ]);
    const orgs = await listOrgs({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    expect(cap.url()).toContain("/rpc/orgs_list");
    expect(orgs.map((o) => o.alias)).toEqual(["acme", "globex"]);
  });

  test("sends NO org header even when a process-wide scope is active", async () => {
    // Mirrors `issues create --org acme`: the preAction hook has stashed the
    // alias in activeOrgScope, yet the alias→id lookup must still see every org.
    setActiveOrgScope(ALIAS_SCOPE);
    const cap = captureFetch([]);
    await listOrgs({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    const headers = cap.headers();
    expect(headers[ORG_ALIAS_HEADER]).toBeUndefined();
    expect(headers[ORG_ID_HEADER]).toBeUndefined();
    expect(headers["access-token"]).toBe(GLOBAL_TOKEN);
  });

  test("a non-array response yields an empty list (drives the `orgs` empty exit)", async () => {
    captureFetch({ not: "an array" });
    const orgs = await listOrgs({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE });
    expect(orgs).toEqual([]);
  });

  test("a non-OK response throws rather than returning an empty list", async () => {
    captureFetch("forbidden", 403);
    await expect(listOrgs({ apiKey: GLOBAL_TOKEN, apiBaseUrl: API_BASE })).rejects.toThrow();
  });
});

// ---------------------------------------------------------------------------
// parseOrgId boundary — the regex admits 18 digits, but a JS number is only
// exact to MAX_SAFE_INTEGER; past that the id would silently target another org.
// ---------------------------------------------------------------------------
describe("--org-id numeric boundary", () => {
  test("the largest safe integer is accepted verbatim", () => {
    expect(resolveOrgScope({ orgId: String(Number.MAX_SAFE_INTEGER) })).toEqual({
      id: Number.MAX_SAFE_INTEGER,
      source: "--org-id",
    });
  });

  test("a value past MAX_SAFE_INTEGER is rejected rather than rounded", () => {
    // 9007199254740993 = MAX_SAFE_INTEGER + 2: passes ^[0-9]{1,18}$ but cannot be
    // represented exactly, so it must not be sent as a corrupted id.
    expect(() => resolveOrgScope({ orgId: "9007199254740993" })).toThrow(OrgScopeError);
  });
});

// ---------------------------------------------------------------------------
// MCP tool surface — the scope must thread into the actual request, and a global
// token with no org_id must stop before any request goes out.
// ---------------------------------------------------------------------------
describe("MCP tools thread the org scope onto the wire", () => {
  const ISSUE = "11111111-1111-1111-1111-111111111111";

  test("view_issue under a global token with no org_id errors before fetching", async () => {
    let fetched = false;
    globalThis.fetch = mock(() => {
      fetched = true;
      return Promise.resolve(new Response("[]", { status: 200 }));
    }) as unknown as typeof fetch;
    const res = await handleToolCall(
      { params: { name: "view_issue", arguments: { issue_id: ISSUE } } },
      { apiKey: GLOBAL_TOKEN }
    );
    expect(res.isError).toBe(true);
    expect(res.content[0].text).toContain("org_id is required");
    expect(fetched).toBe(false);
  });

  test("view_issue threads org_id onto the request headers", async () => {
    const cap = captureFetch([
      { id: ISSUE, title: "t", description: null, status: 0, created_at: "", author_display_name: "a", action_items: [] },
    ]);
    const res = await handleToolCall(
      { params: { name: "view_issue", arguments: { issue_id: ISSUE, org_id: 6333 } } },
      { apiKey: GLOBAL_TOKEN }
    );
    expect(res.isError).toBeUndefined();
    expect(cap.headers()[ORG_ID_HEADER]).toBe("6333");
  });

  test("list_reports threads org_id onto the request headers", async () => {
    const cap = captureFetch([]);
    const res = await handleToolCall(
      { params: { name: "list_reports", arguments: { org_id: 6333 } } },
      { apiKey: GLOBAL_TOKEN }
    );
    expect(res.isError).toBeUndefined();
    expect(cap.headers()[ORG_ID_HEADER]).toBe("6333");
  });
});

// ---------------------------------------------------------------------------
// Drift guard — every registered org-scoped command must actually expose the
// --org/--org-id selector. withOrgOptions() and the header wiring are two lists
// that must stay in sync; this catches a command that was added to one but not
// wired to emit the header via a lib module that routes through buildAuthHeaders.
// ---------------------------------------------------------------------------
describe("ORG_SCOPED_COMMANDS registration drift guard", () => {
  test("the registration set is populated (all 36 org-scoped subcommands)", () => {
    expect(ORG_SCOPED_COMMANDS.size).toBeGreaterThanOrEqual(36);
  });

  test("every org-scoped command exposes both --org and --org-id", () => {
    for (const command of ORG_SCOPED_COMMANDS) {
      const longs = command.options.map((o) => o.long);
      const name = command.name();
      expect({ name, hasOrg: longs.includes("--org") }).toEqual({ name, hasOrg: true });
      expect({ name, hasOrgId: longs.includes("--org-id") }).toEqual({ name, hasOrgId: true });
    }
  });

  // The two assertions above only range over commands ALREADY registered, so
  // they catch a removed flag but never a command nobody registered -- which is
  // exactly how `issues files download` shipped with no way to name an org
  // while its own downloadFile still read the active scope. Invert it: walk the
  // real tree and require every leaf to be either registered or explicitly
  // listed here as org-agnostic, so adding an org-scoped command without
  // wiring it fails loudly instead of silently sending no selector.
  test("every leaf command is either org-scoped or explicitly org-agnostic", () => {
    const leaves: { path: string; command: Command }[] = [];
    const walk = (command: Command, prefix: string) => {
      const subs = command.commands as Command[];
      const here = prefix ? `${prefix} ${command.name()}` : command.name();
      if (subs.length === 0) leaves.push({ path: here, command });
      else for (const sub of subs) walk(sub, here);
    };
    for (const sub of program.commands as Command[]) walk(sub, "");

    const unregistered = leaves
      .filter(({ command }) => !ORG_SCOPED_COMMANDS.has(command))
      .map(({ path }) => path)
      .filter((path) => !path.endsWith(" help") && path !== "help")
      .sort();

    expect(unregistered).toEqual(ORG_AGNOSTIC_COMMANDS);
  });
});
