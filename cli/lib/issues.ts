import { formatHttpError, maskSecret, normalizeBaseUrl } from "./util";
import { buildAuthHeaders, type OrgScope } from "./org-scope";

/**
 * Issue status constants.
 * Used in updateIssue to change issue state.
 */
export const IssueStatus = {
  /** Issue is open and active */
  OPEN: 0,
  /** Issue is closed/resolved */
  CLOSED: 1,
} as const;

/**
 * Represents a PostgreSQL configuration parameter change recommendation.
 * Used in action items to suggest config tuning.
 */
export interface ConfigChange {
  /** PostgreSQL configuration parameter name (e.g., 'work_mem', 'shared_buffers') */
  parameter: string;
  /** Recommended value for the parameter (e.g., '256MB', '4GB') */
  value: string;
}

/**
 * Headers for every issue-scoped request.
 *
 * `x-pgai-include-hidden` is the server's opt-in for staff-only hidden issues
 * (platform-all #562). It is a client capability declaration, not a user
 * preference: it says "this client knows hidden issues exist and will mark
 * them". The platform default-excludes hidden rows from any token caller that
 * does not send it, precisely because old CLI versions, curl, scripts and MCP
 * clients cannot be upgraded on demand and would otherwise render
 * staff-internal commentary as an ordinary issue.
 *
 * Every issue-scoped request carries it, reads and writes alike. The same
 * staff predicate gates the v1.issue_comments and v1.issue_action_items views
 * and the issue_hidden_access_check() guard on issue_update /
 * issue_comment_create / issue_comment_update / issue_action_item_*, so a
 * request without it cannot reach a hidden issue's comments or action items,
 * or update it — and it fails silently, because that guard raises PT404 with
 * the same shape as "no such issue". Do not strip it from a path just because
 * that path does not print the flag.
 *
 * The display obligation is the narrower one: wherever the issue ITSELF is
 * rendered — `issues list`, `issues view`, and their MCP counterparts — route
 * it through withVisibleHiddenFlag(). Sub-resources inherit their marking from
 * the issue view.
 *
 * Non-staff callers are unaffected: the server never has hidden rows to return
 * to them, with or without the header.
 */
export function issueRequestHeaders(
  apiKey: string,
  orgScope?: OrgScope
): Record<string, string> {
  // Delegates to buildAuthHeaders so org scoping (global tokens, #327) keeps
  // working: this only layers the hidden-issue capability on top.
  return buildAuthHeaders(apiKey, orgScope, {
    Prefer: "return=representation",
    "x-pgai-include-hidden": "true",
  });
}

export interface IssueActionItem {
  id: string;
  issue_id: string;
  title: string;
  description: string | null;
  severity: number;
  is_done: boolean;
  done_by: number | null;
  done_at: string | null;
  status: string;
  status_reason: string | null;
  status_changed_by: number | null;
  status_changed_at: string | null;
  sql_action: string | null;
  configs: ConfigChange[];
  created_at: string;
  updated_at: string;
}

/**
 * Summary of an action item (minimal fields for list views).
 * Used in issue detail responses to provide quick overview of action items.
 */
export interface IssueActionItemSummary {
  /** Action item ID (UUID) */
  id: string;
  /** Action item title */
  title: string;
}

export interface Issue {
  id: string;
  title: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  status: number;
  url_main: string | null;
  urls_extra: string[] | null;
  data: unknown | null;
  author_id: number;
  org_id: number;
  project_id: number | null;
  is_ai_generated: boolean;
  assigned_to: number[] | null;
  labels: string[] | null;
  is_edited: boolean;
  author_display_name: string;
  comment_count: number;
  action_items: IssueActionItem[];
  // Staff-only hidden issues (platform-all #562). Never true for a non-staff
  // caller: the API filters hidden rows out server-side, so a non-staff
  // response can only ever carry false. Display it ONLY when true — printing
  // "is_hidden: false" would tell a customer the mechanism exists.
  is_hidden?: boolean;
}

export interface IssueComment {
  id: string;
  issue_id: string;
  author_id: number;
  parent_comment_id: string | null;
  content: string;
  created_at: string;
  updated_at: string;
  data: unknown | null;
}

export type IssueListItem = Pick<Issue, "id" | "title" | "status" | "created_at" | "is_hidden">;

export type IssueDetail = Pick<Issue, "id" | "title" | "description" | "status" | "created_at" | "author_display_name" | "is_hidden"> & {
  action_items: IssueActionItemSummary[];
};

/**
 * Drop `is_hidden` from an issue unless it is actually true.
 *
 * Hidden issues are staff-only (platform-all #562) and the API filters them out
 * server-side for everyone else, so a non-staff caller's rows can only ever
 * carry `is_hidden: false`. Rendering that would disclose to a customer that a
 * hidden-issue mechanism exists — the opposite of the point — so the flag is
 * shown only when set. Mirrors the console, which renders its HiddenIssueChip
 * on `issue.is_hidden &&` rather than on the caller's staff status.
 */
export function withVisibleHiddenFlag<T extends { is_hidden?: boolean }>(
  issue: T
): Omit<T, "is_hidden"> & { is_hidden?: true } {
  // The return type is deliberately narrower than T: after this call the key is
  // either absent or true, never false.
  type Visible = Omit<T, "is_hidden"> & { is_hidden?: true };
  if (!issue || issue.is_hidden === true) {
    return issue as Visible;
  }
  if (!("is_hidden" in issue)) {
    return issue as Visible;
  }
  const { is_hidden: _dropped, ...rest } = issue;
  return rest as Visible;
}

export interface FetchIssuesParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  orgId?: number;
  status?: "open" | "closed";
  limit?: number;
  offset?: number;
  hiddenOnly?: boolean;
  debug?: boolean;
}

export async function fetchIssues(params: FetchIssuesParams): Promise<IssueListItem[]> {
  const { apiKey, apiBaseUrl, orgScope, orgId, status, limit = 20, offset = 0, hiddenOnly, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/issues`);
  // is_hidden is assumed present on v1.issues (platform migration
  // 20260803_issue_hidden_flag, on main since 2026-08-06). There is no
  // capability probe: against an endpoint older than that, PostgREST answers
  // 42703 and `issues list` fails outright rather than just losing the flag.
  url.searchParams.set("select", "id,title,status,created_at,is_hidden");
  url.searchParams.set("order", "id.desc");
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("offset", String(offset));
  if (typeof orgId === "number") {
    url.searchParams.set("org_id", `eq.${orgId}`);
  }
  if (hiddenOnly) {
    // Staff-only in effect: a non-staff caller never has hidden rows to match,
    // so this returns an empty list rather than leaking anything.
    url.searchParams.set("is_hidden", "eq.true");
  }
  if (status === "open") {
    url.searchParams.set("status", "eq.0");
  } else if (status === "closed") {
    url.searchParams.set("status", "eq.1");
  }

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as IssueListItem[];
    } catch {
      throw new Error(`Failed to parse issues response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch issues", response.status, data));
  }
}


export interface FetchIssueCommentsParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  debug?: boolean;
}

export async function fetchIssueComments(params: FetchIssueCommentsParams): Promise<IssueComment[]> {
  const { apiKey, apiBaseUrl, orgScope, issueId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/issue_comments?issue_id=eq.${encodeURIComponent(issueId)}`);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as IssueComment[];
    } catch {
      throw new Error(`Failed to parse issue comments response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch issue comments", response.status, data));
  }
}

export interface FetchIssueParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  debug?: boolean;
}

export async function fetchIssue(params: FetchIssueParams): Promise<IssueDetail | null> {
  const { apiKey, apiBaseUrl, orgScope, issueId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/issues`);
  url.searchParams.set("select", "id,title,description,status,created_at,author_display_name,action_items,is_hidden");
  url.searchParams.set("id", `eq.${issueId}`);
  url.searchParams.set("limit", "1");

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      const parsed = JSON.parse(data);
      const rawIssue = Array.isArray(parsed) ? parsed[0] : parsed;
      if (!rawIssue) {
        return null;
      }
      // Map action_items to summary (id, title only)
      const actionItemsSummary: IssueActionItemSummary[] = Array.isArray(rawIssue.action_items)
        ? rawIssue.action_items.map((item: IssueActionItem) => ({ id: item.id, title: item.title }))
        : [];
      return {
        id: rawIssue.id,
        title: rawIssue.title,
        description: rawIssue.description,
        status: rawIssue.status,
        created_at: rawIssue.created_at,
        author_display_name: rawIssue.author_display_name,
        action_items: actionItemsSummary,
        // Carry the raw flag through; withVisibleHiddenFlag() normalizes it so
        // false/undefined disappear and only a genuine `true` is rendered.
        is_hidden: rawIssue.is_hidden,
      } as IssueDetail;
    } catch {
      throw new Error(`Failed to parse issue response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch issue", response.status, data));
  }
}

export interface CreateIssueParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  title: string;
  orgId: number;
  description?: string;
  projectId?: number;
  labels?: string[];
  debug?: boolean;
}

export interface CreatedIssue {
  id: string;
  title: string;
  description: string | null;
  created_at: string;
  status: number;
  project_id: number | null;
  labels: string[] | null;
}

/**
 * Create a new issue in the PostgresAI platform.
 *
 * @param params - The parameters for creating an issue
 * @param params.apiKey - API key for authentication
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.title - Issue title (required)
 * @param params.orgId - Organization ID (required)
 * @param params.description - Optional issue description
 * @param params.projectId - Optional project ID to associate with
 * @param params.labels - Optional array of label strings
 * @param params.debug - Enable debug logging
 * @returns The created issue object
 * @throws Error if API key, title, or orgId is missing, or if the API call fails
 */
export async function createIssue(params: CreateIssueParams): Promise<CreatedIssue> {
  const { apiKey, apiBaseUrl, orgScope, title, orgId, description, projectId, labels, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!title) {
    throw new Error("title is required");
  }
  if (typeof orgId !== "number") {
    throw new Error("orgId is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_create`);

  const bodyObj: Record<string, unknown> = {
    title: title,
    org_id: orgId,
  };
  if (description !== undefined) {
    bodyObj.description = description;
  }
  if (projectId !== undefined) {
    bodyObj.project_id = projectId;
  }
  if (labels && labels.length > 0) {
    bodyObj.labels = labels;
  }
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as CreatedIssue;
    } catch {
      throw new Error(`Failed to parse create issue response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to create issue", response.status, data));
  }
}

export interface CreateIssueCommentParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  content: string;
  parentCommentId?: string;
  debug?: boolean;
}

export async function createIssueComment(params: CreateIssueCommentParams): Promise<IssueComment> {
  const { apiKey, apiBaseUrl, orgScope, issueId, content, parentCommentId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }
  if (!content) {
    throw new Error("content is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_comment_create`);

  const bodyObj: Record<string, unknown> = {
    issue_id: issueId,
    content: content,
  };
  if (parentCommentId) {
    bodyObj.parent_comment_id = parentCommentId;
  }
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as IssueComment;
    } catch {
      throw new Error(`Failed to parse create comment response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to create issue comment", response.status, data));
  }
}

export interface UpdateIssueParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  title?: string;
  description?: string;
  status?: number;
  labels?: string[];
  debug?: boolean;
}

export interface UpdatedIssue {
  id: string;
  title: string;
  description: string | null;
  status: number;
  updated_at: string;
  labels: string[] | null;
}

/**
 * Update an existing issue in the PostgresAI platform.
 *
 * @param params - The parameters for updating an issue
 * @param params.apiKey - API key for authentication
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.issueId - ID of the issue to update (required)
 * @param params.title - New title (optional)
 * @param params.description - New description (optional)
 * @param params.status - New status: 0 = open, 1 = closed (optional)
 * @param params.labels - New labels array (optional, replaces existing)
 * @param params.debug - Enable debug logging
 * @returns The updated issue object
 * @throws Error if API key or issueId is missing, if no fields to update are provided, or if the API call fails
 */
export async function updateIssue(params: UpdateIssueParams): Promise<UpdatedIssue> {
  const { apiKey, apiBaseUrl, orgScope, issueId, title, description, status, labels, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }
  if (title === undefined && description === undefined && status === undefined && labels === undefined) {
    throw new Error("At least one field to update is required (title, description, status, or labels)");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_update`);

  // Prod RPC expects p_* argument names (see OpenAPI at /api/general/).
  const bodyObj: Record<string, unknown> = {
    p_id: issueId,
  };
  if (title !== undefined) {
    bodyObj.p_title = title;
  }
  if (description !== undefined) {
    bodyObj.p_description = description;
  }
  if (status !== undefined) {
    bodyObj.p_status = status;
  }
  if (labels !== undefined) {
    bodyObj.p_labels = labels;
  }
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as UpdatedIssue;
    } catch {
      throw new Error(`Failed to parse update issue response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to update issue", response.status, data));
  }
}

export interface UpdateIssueCommentParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  commentId: string;
  content: string;
  debug?: boolean;
}

export interface UpdatedIssueComment {
  id: string;
  issue_id: string;
  content: string;
  updated_at: string;
}

/**
 * Update an existing issue comment in the PostgresAI platform.
 *
 * @param params - The parameters for updating a comment
 * @param params.apiKey - API key for authentication
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.commentId - ID of the comment to update (required)
 * @param params.content - New comment content (required)
 * @param params.debug - Enable debug logging
 * @returns The updated comment object
 * @throws Error if API key, commentId, or content is missing, or if the API call fails
 */
export async function updateIssueComment(params: UpdateIssueCommentParams): Promise<UpdatedIssueComment> {
  const { apiKey, apiBaseUrl, orgScope, commentId, content, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!commentId) {
    throw new Error("commentId is required");
  }
  if (!content) {
    throw new Error("content is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_comment_update`);

  const bodyObj: Record<string, unknown> = {
    // Prod RPC expects p_* argument names (see OpenAPI at /api/general/).
    p_id: commentId,
    p_content: content,
  };
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as UpdatedIssueComment;
    } catch {
      throw new Error(`Failed to parse update comment response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to update issue comment", response.status, data));
  }
}

// ============================================================================
// Action Items API Functions
// ============================================================================

export interface FetchActionItemParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  actionItemIds: string | string[];
  debug?: boolean;
}

/**
 * Fetch action item(s) by ID(s).
 * Supports single ID or array of IDs.
 *
 * @param params - Fetch parameters
 * @param params.apiKey - API authentication key
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.actionItemIds - Single action item ID or array of IDs (UUIDs)
 * @param params.debug - Enable debug logging
 * @returns Array of action items matching the provided IDs
 * @throws Error if API key is missing or no valid IDs provided
 *
 * @example
 * // Fetch single action item
 * const items = await fetchActionItem({ apiKey, apiBaseUrl, actionItemIds: "uuid-123" });
 *
 * @example
 * // Fetch multiple action items
 * const items = await fetchActionItem({ apiKey, apiBaseUrl, actionItemIds: ["uuid-1", "uuid-2"] });
 */
export async function fetchActionItem(params: FetchActionItemParams): Promise<IssueActionItem[]> {
  const { apiKey, apiBaseUrl, orgScope, actionItemIds, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  // UUID format pattern for validation
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  // Normalize to array, filter out null/undefined, trim, and validate UUID format
  const rawIds = Array.isArray(actionItemIds) ? actionItemIds : [actionItemIds];
  const validIds = rawIds
    .filter((id): id is string => id != null && typeof id === "string")
    .map(id => id.trim())
    .filter(id => id.length > 0 && uuidPattern.test(id));
  if (validIds.length === 0) {
    throw new Error("actionItemId is required and must be a valid UUID");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/issue_action_items`);
  if (validIds.length === 1) {
    url.searchParams.set("id", `eq.${validIds[0]}`);
  } else {
    // PostgREST IN syntax: id=in.(val1,val2,val3)
    url.searchParams.set("id", `in.(${validIds.join(",")})`)
  }

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      const parsed = JSON.parse(data);
      if (Array.isArray(parsed)) {
        return parsed as IssueActionItem[];
      }
      return parsed ? [parsed as IssueActionItem] : [];
    } catch {
      throw new Error(`Failed to parse action item response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch action item", response.status, data));
  }
}

export interface FetchActionItemsParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  debug?: boolean;
}

/**
 * Fetch all action items for an issue.
 *
 * @param params - Fetch parameters
 * @param params.apiKey - API authentication key
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.issueId - Issue ID (UUID) to fetch action items for
 * @param params.debug - Enable debug logging
 * @returns Array of action items for the specified issue
 * @throws Error if API key or issue ID is missing
 */
export async function fetchActionItems(params: FetchActionItemsParams): Promise<IssueActionItem[]> {
  const { apiKey, apiBaseUrl, orgScope, issueId, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }
  // Validate UUID format to prevent PostgREST injection
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidPattern.test(issueId.trim())) {
    throw new Error("issueId must be a valid UUID");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/issue_action_items`);
  url.searchParams.set("issue_id", `eq.${issueId.trim()}`);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: GET URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as IssueActionItem[];
    } catch {
      throw new Error(`Failed to parse action items response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to fetch action items", response.status, data));
  }
}

export interface CreateActionItemParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  issueId: string;
  title: string;
  description?: string;
  sqlAction?: string;
  configs?: ConfigChange[];
  debug?: boolean;
}

/**
 * Create a new action item for an issue.
 *
 * @param params - Creation parameters
 * @param params.apiKey - API authentication key
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.issueId - Issue ID (UUID) to create action item for
 * @param params.title - Action item title
 * @param params.description - Optional detailed description
 * @param params.sqlAction - Optional SQL command to execute
 * @param params.configs - Optional configuration parameter changes
 * @param params.debug - Enable debug logging
 * @returns Created action item ID
 * @throws Error if required fields are missing or API call fails
 */
export async function createActionItem(params: CreateActionItemParams): Promise<string> {
  const { apiKey, apiBaseUrl, orgScope, issueId, title, description, sqlAction, configs, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!issueId) {
    throw new Error("issueId is required");
  }
  // Validate UUID format
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidPattern.test(issueId.trim())) {
    throw new Error("issueId must be a valid UUID");
  }
  if (!title) {
    throw new Error("title is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_action_item_create`);

  const bodyObj: Record<string, unknown> = {
    issue_id: issueId,
    title: title,
  };
  if (description !== undefined) {
    bodyObj.description = description;
  }
  if (sqlAction !== undefined) {
    bodyObj.sql_action = sqlAction;
  }
  if (configs !== undefined) {
    bodyObj.configs = configs;
  }
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  const data = await response.text();

  if (response.ok) {
    try {
      return JSON.parse(data) as string;
    } catch {
      throw new Error(`Failed to parse create action item response: ${data}`);
    }
  } else {
    throw new Error(formatHttpError("Failed to create action item", response.status, data));
  }
}

export interface UpdateActionItemParams {
  apiKey: string;
  /** Selected organization, required under a global token (postgresai #327). */
  orgScope?: OrgScope;
  apiBaseUrl: string;
  actionItemId: string;
  title?: string;
  description?: string;
  isDone?: boolean;
  status?: string;
  statusReason?: string;
  sqlAction?: string;
  configs?: ConfigChange[];
  debug?: boolean;
}

/**
 * Update an existing action item.
 *
 * @param params - Update parameters
 * @param params.apiKey - API authentication key
 * @param params.apiBaseUrl - Base URL for the API
 * @param params.actionItemId - Action item ID (UUID) to update
 * @param params.title - New title
 * @param params.description - New description
 * @param params.isDone - Mark as done/not done
 * @param params.status - Approval status: 'waiting_for_approval', 'approved', 'rejected'
 * @param params.statusReason - Reason for status change
 * @param params.sqlAction - SQL command to execute
 * @param params.configs - Configuration parameter changes
 * @param params.debug - Enable debug logging
 * @throws Error if required fields missing or no update fields provided
 */
export async function updateActionItem(params: UpdateActionItemParams): Promise<void> {
  const { apiKey, apiBaseUrl, orgScope, actionItemId, title, description, isDone, status, statusReason, sqlAction, configs, debug } = params;
  if (!apiKey) {
    throw new Error("API key is required");
  }
  if (!actionItemId) {
    throw new Error("actionItemId is required");
  }
  // Validate UUID format
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidPattern.test(actionItemId.trim())) {
    throw new Error("actionItemId must be a valid UUID");
  }

  // Check that at least one update field is provided
  const hasUpdateField = title !== undefined || description !== undefined ||
    isDone !== undefined || status !== undefined ||
    statusReason !== undefined || sqlAction !== undefined || configs !== undefined;
  if (!hasUpdateField) {
    throw new Error("At least one field to update is required");
  }

  const base = normalizeBaseUrl(apiBaseUrl);
  const url = new URL(`${base}/rpc/issue_action_item_update`);

  const bodyObj: Record<string, unknown> = {
    action_item_id: actionItemId,
  };
  if (title !== undefined) {
    bodyObj.title = title;
  }
  if (description !== undefined) {
    bodyObj.description = description;
  }
  if (isDone !== undefined) {
    bodyObj.is_done = isDone;
  }
  if (status !== undefined) {
    bodyObj.status = status;
  }
  if (statusReason !== undefined) {
    bodyObj.status_reason = statusReason;
  }
  if (sqlAction !== undefined) {
    bodyObj.sql_action = sqlAction;
  }
  if (configs !== undefined) {
    bodyObj.configs = configs;
  }
  const body = JSON.stringify(bodyObj);

  const headers: Record<string, string> = issueRequestHeaders(apiKey, orgScope);

  if (debug) {
    const debugHeaders: Record<string, string> = { ...headers, "access-token": maskSecret(apiKey) };
    console.error(`Debug: Resolved API base URL: ${base}`);
    console.error(`Debug: POST URL: ${url.toString()}`);
    console.error(`Debug: Auth scheme: access-token`);
    console.error(`Debug: Request headers: ${JSON.stringify(debugHeaders)}`);
    console.error(`Debug: Request body: ${body}`);
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    headers,
    body,
  });

  if (debug) {
    console.error(`Debug: Response status: ${response.status}`);
    console.error(`Debug: Response headers: ${JSON.stringify(Object.fromEntries(response.headers.entries()))}`);
  }

  if (!response.ok) {
    const data = await response.text();
    throw new Error(formatHttpError("Failed to update action item", response.status, data));
  }
}
