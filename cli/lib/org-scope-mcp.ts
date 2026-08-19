/**
 * Org selection for MCP tools (postgresai #327, closing #250).
 *
 * Same rule as the CLI's flags, expressed as an `org_id` tool argument: a
 * credential reaching several orgs must be told which one, because the
 * alternative is the silent default of #250. Separate from lib/org-scope.ts
 * because the failure shape differs -- MCP returns a structured tool result
 * the model can act on rather than throwing.
 */

import * as config from "./config";
import { parseOrgId, type OrgScope } from "./org-scope";

export interface McpToolError {
  content: { type: string; text: string }[];
  isError: true;
}

export interface McpOrgScopeResult {
  /** Numeric org id to send in the request body, when the tool needs one. */
  orgId?: number;
  /** Header-borne selection, when a global token named an org. */
  orgScope?: OrgScope;
  /** Populated when the tool must stop and tell the model what to do. */
  error?: McpToolError;
}

/**
 * Resolve the org for one MCP tool call.
 *
 * Per-org token: `org_id` if given, else the stored one — unchanged behaviour.
 * Global token: `org_id` is REQUIRED and there is no config fallback, since the
 * stored org (if any) belongs to some previous per-org login and would silently
 * re-scope the call.
 */
/** MCP has no --org and no alias argument, so its errors must not suggest one. */
const MCP_ORG_REMEDY =
  "Call the orgs_list RPC (or run 'pgai orgs') to see the available organization ids.";

export function resolveMcpOrgScope(
  args: Record<string, unknown>,
  apiKey: string,
  cfg: Pick<config.Config, "orgId">
): McpOrgScopeResult {
  // Validate exactly as the CLI's --org-id does. A bare Number() accepts
  // 9007199254740993 (rounding it to a DIFFERENT org), 0x10, 1e400 and -5, and
  // this value goes on the wire as x-pgai-org-id.
  const raw =
    args.org_id !== undefined && args.org_id !== null && `${args.org_id}`.trim() !== ""
      ? `${args.org_id}`.trim()
      : undefined;
  let explicit: number | undefined;
  if (raw !== undefined) {
    try {
      explicit = parseOrgId(raw, "org_id", MCP_ORG_REMEDY);
    } catch (err) {
      return {
        error: {
          content: [{ type: "text", text: err instanceof Error ? err.message : String(err) }],
          isError: true,
        },
      };
    }
  }

  if (!config.isGlobalTokenValue(apiKey)) {
    return { orgId: explicit !== undefined ? explicit : cfg.orgId ?? undefined };
  }

  if (explicit === undefined || Number.isNaN(explicit)) {
    return {
      error: {
        content: [
          {
            type: "text",
            text:
              "org_id is required. This credential is a global token, so it can reach every " +
              "organization the user belongs to and will not assume one. " +
              "Call the orgs_list RPC (or run 'pgai orgs') to see the available organizations, " +
              "then pass org_id explicitly.",
          },
        ],
        isError: true,
      },
    };
  }

  // Send the selection as a header too, so the server scopes the request rather
  // than trusting a body field the caller could have set to any org.
  return { orgId: explicit, orgScope: { id: explicit, source: "--org-id" } };
}
