import pkg from "../package.json";
import * as config from "./config";
import {
  fetchIssues,
  fetchIssueComments,
  createIssueComment,
  fetchIssue,
  createIssue,
  updateIssue,
  updateIssueComment,
  fetchActionItem,
  fetchActionItems,
  createActionItem,
  updateActionItem,
  type ConfigChange,
} from "./issues";
import { fetchReports, fetchAllReports, fetchReportFiles, fetchReportFileData, parseFlexibleDate } from "./reports";
import { uploadFile, downloadFile, buildMarkdownLink, uploadAttachments, appendAttachmentsToContent } from "./storage";
import { resolveBaseUrls } from "./util";

// MCP SDK imports - Bun handles these directly
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

export interface RootOptsLike {
  apiKey?: string;
  apiBaseUrl?: string;
}

// Interpret escape sequences (e.g., \n -> newline). Input comes from JSON, but
// we still normalize common escapes for consistency.
export const interpretEscapes = (str: string): string =>
  (str || "")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\r/g, "\r")
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'");

export interface McpToolRequest {
  params: {
    name: string;
    arguments?: Record<string, unknown>;
  };
}

export interface McpToolResponse {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
}

/** Handle MCP tool calls - exported for testing */
export async function handleToolCall(
  req: McpToolRequest,
  rootOpts?: RootOptsLike,
  extra?: { debug?: boolean }
): Promise<McpToolResponse> {
  const toolName = req.params.name;
  const args = (req.params.arguments as Record<string, unknown>) || {};

  const cfg = config.readConfig();
  const apiKey = (rootOpts?.apiKey || process.env.PGAI_API_KEY || cfg.apiKey || "").toString();
  const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

  const debug = Boolean(args.debug ?? extra?.debug);

  if (!apiKey) {
    return {
      content: [
        {
          type: "text",
          text: "API key is required. Run 'pgai auth' or set PGAI_API_KEY.",
        },
      ],
      isError: true,
    };
  }

  try {
    if (toolName === "list_issues") {
      const orgId = args.org_id !== undefined ? Number(args.org_id) : cfg.orgId ?? undefined;
      const statusArg = args.status ? String(args.status) : undefined;
      let status: "open" | "closed" | undefined;
      if (statusArg === "open") status = "open";
      else if (statusArg === "closed") status = "closed";
      const limit = args.limit !== undefined ? Number(args.limit) : undefined;
      const offset = args.offset !== undefined ? Number(args.offset) : undefined;
      const issues = await fetchIssues({ apiKey, apiBaseUrl, orgId, status, limit, offset, debug });
      return { content: [{ type: "text", text: JSON.stringify(issues, null, 2) }] };
    }

    if (toolName === "view_issue") {
      const issueId = String(args.issue_id || "").trim();
      if (!issueId) {
        return { content: [{ type: "text", text: "issue_id is required" }], isError: true };
      }
      const issue = await fetchIssue({ apiKey, apiBaseUrl, issueId, debug });
      if (!issue) {
        return { content: [{ type: "text", text: "Issue not found" }], isError: true };
      }
      const comments = await fetchIssueComments({ apiKey, apiBaseUrl, issueId, debug });
      const combined = { issue, comments };
      return { content: [{ type: "text", text: JSON.stringify(combined, null, 2) }] };
    }

    if (toolName === "post_issue_comment") {
      const issueId = String(args.issue_id || "").trim();
      const rawContent = String(args.content || "");
      const parentCommentId = args.parent_comment_id ? String(args.parent_comment_id) : undefined;
      const attachments = Array.isArray(args.attachments) ? args.attachments.map(String).filter((p) => p.length > 0) : [];
      if (!issueId) {
        return { content: [{ type: "text", text: "issue_id is required" }], isError: true };
      }
      if (!rawContent && attachments.length === 0) {
        return { content: [{ type: "text", text: "content or attachments is required" }], isError: true };
      }
      let content = interpretEscapes(rawContent);
      if (attachments.length > 0) {
        const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
        const uploaded = await uploadAttachments({ apiKey, storageBaseUrl, attachmentPaths: attachments, debug });
        content = appendAttachmentsToContent(content, uploaded);
      }
      const result = await createIssueComment({ apiKey, apiBaseUrl, issueId, content, parentCommentId, debug });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }

    if (toolName === "create_issue") {
      const rawTitle = String(args.title || "").trim();
      if (!rawTitle) {
        return { content: [{ type: "text", text: "title is required" }], isError: true };
      }
      const title = interpretEscapes(rawTitle);
      const rawDescription = args.description ? String(args.description) : undefined;
      let description = rawDescription ? interpretEscapes(rawDescription) : undefined;
      const projectId = args.project_id !== undefined ? Number(args.project_id) : undefined;
      const labels = Array.isArray(args.labels) ? args.labels.map(String) : undefined;
      const attachments = Array.isArray(args.attachments) ? args.attachments.map(String).filter((p) => p.length > 0) : [];
      // Get orgId from args or fall back to config
      const orgId = args.org_id !== undefined ? Number(args.org_id) : cfg.orgId;
      // Note: orgId=0 is technically valid (though unlikely), so don't use falsy check
      if (orgId === undefined || orgId === null || Number.isNaN(orgId)) {
        return { content: [{ type: "text", text: "org_id is required. Either provide it as a parameter or run 'pgai auth' to set it in config." }], isError: true };
      }
      if (attachments.length > 0) {
        const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
        const uploaded = await uploadAttachments({ apiKey, storageBaseUrl, attachmentPaths: attachments, debug });
        description = appendAttachmentsToContent(description ?? "", uploaded);
      }
      const result = await createIssue({ apiKey, apiBaseUrl, title, orgId, description, projectId, labels, debug });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }

    if (toolName === "update_issue") {
      const issueId = String(args.issue_id || "").trim();
      if (!issueId) {
        return { content: [{ type: "text", text: "issue_id is required" }], isError: true };
      }
      const rawTitle = args.title !== undefined ? String(args.title) : undefined;
      const title = rawTitle !== undefined ? interpretEscapes(rawTitle) : undefined;
      const rawDescription = args.description !== undefined ? String(args.description) : undefined;
      let description = rawDescription !== undefined ? interpretEscapes(rawDescription) : undefined;
      const status = args.status !== undefined ? Number(args.status) : undefined;
      const labels = Array.isArray(args.labels) ? args.labels.map(String) : undefined;
      const attachments = Array.isArray(args.attachments) ? args.attachments.map(String).filter((p) => p.length > 0) : [];
      // Validate that at least one update field is provided (attachments alone counts)
      if (title === undefined && description === undefined && status === undefined && labels === undefined && attachments.length === 0) {
        return { content: [{ type: "text", text: "At least one field to update is required (title, description, status, labels, or attachments)" }], isError: true };
      }
      // Validate status value if provided (check for NaN and valid values)
      if (status !== undefined && (Number.isNaN(status) || (status !== 0 && status !== 1))) {
        return { content: [{ type: "text", text: "status must be 0 (open) or 1 (closed)" }], isError: true };
      }
      if (attachments.length > 0) {
        // If the caller did not supply a new description, fetch the existing one
        // and append to it so "add a screenshot to issue X" is one round-trip
        // for the agent. Same race-window tradeoff as the CLI flag.
        if (description === undefined) {
          const existing = await fetchIssue({ apiKey, apiBaseUrl, issueId, debug });
          if (!existing) {
            return { content: [{ type: "text", text: `Issue not found: ${issueId}` }], isError: true };
          }
          description = (existing as { description?: string | null }).description ?? "";
        }
        const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
        const uploaded = await uploadAttachments({ apiKey, storageBaseUrl, attachmentPaths: attachments, debug });
        description = appendAttachmentsToContent(description, uploaded);
      }
      const result = await updateIssue({ apiKey, apiBaseUrl, issueId, title, description, status, labels, debug });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }

    if (toolName === "update_issue_comment") {
      const commentId = String(args.comment_id || "").trim();
      const rawContent = String(args.content || "");
      const attachments = Array.isArray(args.attachments) ? args.attachments.map(String).filter((p) => p.length > 0) : [];
      if (!commentId) {
        return { content: [{ type: "text", text: "comment_id is required" }], isError: true };
      }
      if (!rawContent.trim() && attachments.length === 0) {
        return { content: [{ type: "text", text: "content or attachments is required" }], isError: true };
      }
      let content = interpretEscapes(rawContent);
      if (attachments.length > 0) {
        const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
        const uploaded = await uploadAttachments({ apiKey, storageBaseUrl, attachmentPaths: attachments, debug });
        content = appendAttachmentsToContent(content, uploaded);
      }
      const result = await updateIssueComment({ apiKey, apiBaseUrl, commentId, content, debug });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }

    if (toolName === "upload_file") {
      const filePath = String(args.path || "").trim();
      if (!filePath) {
        return { content: [{ type: "text", text: "path is required" }], isError: true };
      }
      const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
      const result = await uploadFile({ apiKey, storageBaseUrl, filePath, debug });
      const markdown = buildMarkdownLink(result.url, storageBaseUrl, result.metadata.originalName);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: result.success,
                url: result.url,
                markdown,
                metadata: result.metadata,
                requestId: result.requestId,
              },
              null,
              2
            ),
          },
        ],
      };
    }

    if (toolName === "download_file") {
      const fileUrl = String(args.url || "").trim();
      const outputPath = args.output_path !== undefined ? String(args.output_path) : undefined;
      if (!fileUrl) {
        return { content: [{ type: "text", text: "url is required" }], isError: true };
      }
      const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);
      const result = await downloadFile({ apiKey, storageBaseUrl, fileUrl, outputPath, debug });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }

    // Action Items Tools
    if (toolName === "view_action_item") {
      // Support both single ID and array of IDs
      let actionItemIds: string[];
      if (Array.isArray(args.action_item_ids)) {
        actionItemIds = args.action_item_ids.map((id: unknown) => String(id).trim()).filter((id: string) => id);
      } else if (args.action_item_id) {
        actionItemIds = [String(args.action_item_id).trim()];
      } else {
        actionItemIds = [];
      }
      if (actionItemIds.length === 0) {
        return { content: [{ type: "text", text: "action_item_id or action_item_ids is required" }], isError: true };
      }
      const actionItems = await fetchActionItem({ apiKey, apiBaseUrl, actionItemIds, debug });
      if (actionItems.length === 0) {
        return { content: [{ type: "text", text: "Action item(s) not found" }], isError: true };
      }
      return { content: [{ type: "text", text: JSON.stringify(actionItems, null, 2) }] };
    }

    if (toolName === "list_action_items") {
      const issueId = String(args.issue_id || "").trim();
      if (!issueId) {
        return { content: [{ type: "text", text: "issue_id is required" }], isError: true };
      }
      const actionItems = await fetchActionItems({ apiKey, apiBaseUrl, issueId, debug });
      return { content: [{ type: "text", text: JSON.stringify(actionItems, null, 2) }] };
    }

    if (toolName === "create_action_item") {
      const issueId = String(args.issue_id || "").trim();
      const rawTitle = String(args.title || "").trim();
      if (!issueId) {
        return { content: [{ type: "text", text: "issue_id is required" }], isError: true };
      }
      if (!rawTitle) {
        return { content: [{ type: "text", text: "title is required" }], isError: true };
      }
      const title = interpretEscapes(rawTitle);
      const rawDescription = args.description ? String(args.description) : undefined;
      const description = rawDescription ? interpretEscapes(rawDescription) : undefined;
      const sqlAction = args.sql_action !== undefined ? String(args.sql_action) : undefined;
      const configs = Array.isArray(args.configs) ? args.configs as ConfigChange[] : undefined;
      const result = await createActionItem({ apiKey, apiBaseUrl, issueId, title, description, sqlAction, configs, debug });
      return { content: [{ type: "text", text: JSON.stringify({ id: result }, null, 2) }] };
    }

    if (toolName === "update_action_item") {
      const actionItemId = String(args.action_item_id || "").trim();
      if (!actionItemId) {
        return { content: [{ type: "text", text: "action_item_id is required" }], isError: true };
      }
      const rawTitle = args.title !== undefined ? String(args.title) : undefined;
      const title = rawTitle !== undefined ? interpretEscapes(rawTitle) : undefined;
      const rawDescription = args.description !== undefined ? String(args.description) : undefined;
      const description = rawDescription !== undefined ? interpretEscapes(rawDescription) : undefined;
      const isDone = args.is_done !== undefined ? Boolean(args.is_done) : undefined;
      const status = args.status !== undefined ? String(args.status) : undefined;
      const statusReason = args.status_reason !== undefined ? String(args.status_reason) : undefined;

      // Validate that at least one update field is provided
      if (title === undefined && description === undefined &&
          isDone === undefined && status === undefined && statusReason === undefined) {
        return { content: [{ type: "text", text: "At least one field to update is required (title, description, is_done, status, or status_reason)" }], isError: true };
      }

      // Validate status value if provided
      if (status !== undefined && !["waiting_for_approval", "approved", "rejected"].includes(status)) {
        return { content: [{ type: "text", text: "status must be 'waiting_for_approval', 'approved', or 'rejected'" }], isError: true };
      }

      await updateActionItem({ apiKey, apiBaseUrl, actionItemId, title, description, isDone, status, statusReason, debug });
      return { content: [{ type: "text", text: JSON.stringify({ success: true }, null, 2) }] };
    }

    // Reports Tools
    if (toolName === "list_reports") {
      const projectId = args.project_id !== undefined ? Number(args.project_id) : undefined;
      const status = args.status ? String(args.status) : undefined;
      const limit = args.limit !== undefined ? Number(args.limit) : undefined;
      const beforeDate = args.before_date ? parseFlexibleDate(String(args.before_date)) : undefined;
      const all = args.all === true;
      let reports;
      if (all) {
        reports = await fetchAllReports({ apiKey, apiBaseUrl, projectId, status, limit, debug });
      } else {
        reports = await fetchReports({ apiKey, apiBaseUrl, projectId, status, limit, beforeDate, debug });
      }
      return { content: [{ type: "text", text: JSON.stringify(reports, null, 2) }] };
    }

    if (toolName === "list_report_files") {
      const reportId = args.report_id !== undefined ? Number(args.report_id) : undefined;
      if (reportId !== undefined && isNaN(reportId)) {
        return { content: [{ type: "text", text: "report_id must be a number" }], isError: true };
      }
      const type = args.type ? String(args.type) as "json" | "md" : undefined;
      const checkId = args.check_id ? String(args.check_id) : undefined;
      if (reportId === undefined && !checkId) {
        return { content: [{ type: "text", text: "Either report_id or check_id is required" }], isError: true };
      }
      const files = await fetchReportFiles({ apiKey, apiBaseUrl, reportId, type, checkId, debug });
      return { content: [{ type: "text", text: JSON.stringify(files, null, 2) }] };
    }

    if (toolName === "get_report_data") {
      const reportId = args.report_id !== undefined ? Number(args.report_id) : undefined;
      if (reportId !== undefined && isNaN(reportId)) {
        return { content: [{ type: "text", text: "report_id must be a number" }], isError: true };
      }
      const type = args.type ? String(args.type) as "json" | "md" : undefined;
      const checkId = args.check_id ? String(args.check_id) : undefined;
      if (reportId === undefined && !checkId) {
        return { content: [{ type: "text", text: "Either report_id or check_id is required" }], isError: true };
      }
      const files = await fetchReportFileData({ apiKey, apiBaseUrl, reportId, type, checkId, debug });
      return { content: [{ type: "text", text: JSON.stringify(files, null, 2) }] };
    }

    throw new Error(`Unknown tool: ${toolName}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { content: [{ type: "text", text: message }], isError: true };
  }
}

export async function startMcpServer(rootOpts?: RootOptsLike, extra?: { debug?: boolean }): Promise<void> {
  const server = new Server(
    {
      name: "postgresai-mcp",
      version: pkg.version,
      title: "PostgresAI MCP Server",
    },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: "list_issues",
          description: "List issues from PostgresAI API (same as CLI 'issues list')",
          inputSchema: {
            type: "object",
            properties: {
              org_id: { type: "number", description: "Organization ID (optional, falls back to config)" },
              status: { type: "string", description: "Filter by status: 'open', 'closed', or omit for all" },
              limit: { type: "number", description: "Max number of issues to return (default: 20)" },
              offset: { type: "number", description: "Number of issues to skip (default: 0)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            additionalProperties: false,
          },
        },
        {
          name: "view_issue",
          description: "View a specific issue with its comments",
          inputSchema: {
            type: "object",
            properties: {
              issue_id: { type: "string", description: "Issue ID (UUID)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["issue_id"],
            additionalProperties: false,
          },
        },
        {
          name: "post_issue_comment",
          description: "Post a new comment to an issue (optionally as a reply). Local files passed via 'attachments' are uploaded to PostgresAI storage and the resulting markdown links are appended to the comment body (image extensions render inline).",
          inputSchema: {
            type: "object",
            properties: {
              issue_id: { type: "string", description: "Issue ID (UUID)" },
              content: { type: "string", description: "Comment text (supports \\n as newline)" },
              parent_comment_id: { type: "string", description: "Parent comment ID (UUID) for replies" },
              attachments: {
                type: "array",
                items: { type: "string" },
                description: "Local file paths to upload and append as markdown links (images render inline). Either 'content' or 'attachments' must be non-empty.",
              },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["issue_id"],
            additionalProperties: false,
          },
        },
        {
          name: "create_issue",
          description: "Create a new issue in PostgresAI. Local files passed via 'attachments' are uploaded to PostgresAI storage and the resulting markdown links are appended to the issue description.",
          inputSchema: {
            type: "object",
            properties: {
              title: { type: "string", description: "Issue title (required)" },
              description: { type: "string", description: "Issue description (supports \\n as newline)" },
              org_id: { type: "number", description: "Organization ID (uses config value if not provided)" },
              project_id: { type: "number", description: "Project ID to associate the issue with" },
              labels: {
                type: "array",
                items: { type: "string" },
                description: "Labels to apply to the issue",
              },
              attachments: {
                type: "array",
                items: { type: "string" },
                description: "Local file paths to upload and append as markdown links to the description (images render inline)",
              },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["title"],
            additionalProperties: false,
          },
        },
        {
          name: "update_issue",
          description: "Update an existing issue (title, description, status, labels). Use status=1 to close, status=0 to reopen. Local files passed via 'attachments' are uploaded and appended to 'description'; if 'description' is omitted, the existing description is fetched first and appended to.",
          inputSchema: {
            type: "object",
            properties: {
              issue_id: { type: "string", description: "Issue ID (UUID)" },
              title: { type: "string", description: "New title (supports \\n as newline)" },
              description: { type: "string", description: "New description (supports \\n as newline)" },
              status: { type: "number", description: "Status: 0=open, 1=closed" },
              labels: {
                type: "array",
                items: { type: "string" },
                description: "Labels to set on the issue",
              },
              attachments: {
                type: "array",
                items: { type: "string" },
                description: "Local file paths to upload and append as markdown links (images render inline). When provided without 'description', the existing description is fetched and appended to.",
              },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["issue_id"],
            additionalProperties: false,
          },
        },
        {
          name: "update_issue_comment",
          description: "Update an existing issue comment. Local files passed via 'attachments' are uploaded and appended to 'content' as markdown links.",
          inputSchema: {
            type: "object",
            properties: {
              comment_id: { type: "string", description: "Comment ID (UUID)" },
              content: { type: "string", description: "New comment text (supports \\n as newline). Either 'content' or 'attachments' must be non-empty." },
              attachments: {
                type: "array",
                items: { type: "string" },
                description: "Local file paths to upload and append as markdown links (images render inline)",
              },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["comment_id"],
            additionalProperties: false,
          },
        },
        {
          name: "upload_file",
          description: "Upload a local file to PostgresAI storage. Returns the storage URL and a ready-to-paste markdown link (image extensions get the inline `![](url)` form, others get `[](url)`). For posting attachments alongside an issue or comment, prefer the 'attachments' parameter on the issue/comment tools — this tool is for ad-hoc uploads or when you need the URL out of band.",
          inputSchema: {
            type: "object",
            properties: {
              path: { type: "string", description: "Local file path to upload (absolute or relative to CWD)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["path"],
            additionalProperties: false,
          },
        },
        {
          name: "download_file",
          description: "Download a file from PostgresAI storage by URL (full URL under the storage base, or a relative path like /files/123/foo.png).",
          inputSchema: {
            type: "object",
            properties: {
              url: { type: "string", description: "Full URL (must be under the configured storage base) or relative storage path (e.g. /files/123/foo.png)" },
              output_path: { type: "string", description: "Local destination path (default: derive filename from URL, save in CWD). When omitted, the path-traversal guard restricts the destination to CWD." },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["url"],
            additionalProperties: false,
          },
        },
        // Action Items Tools
        {
          name: "view_action_item",
          description: "View action item(s) with all details. Supports single ID or multiple IDs.",
          inputSchema: {
            type: "object",
            properties: {
              action_item_id: { type: "string", description: "Single action item ID (UUID)" },
              action_item_ids: { type: "array", items: { type: "string" }, description: "Multiple action item IDs (UUIDs)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            additionalProperties: false,
          },
        },
        {
          name: "list_action_items",
          description: "List action items for an issue",
          inputSchema: {
            type: "object",
            properties: {
              issue_id: { type: "string", description: "Issue ID (UUID)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["issue_id"],
            additionalProperties: false,
          },
        },
        {
          name: "create_action_item",
          description: "Create a new action item for an issue",
          inputSchema: {
            type: "object",
            properties: {
              issue_id: { type: "string", description: "Issue ID (UUID)" },
              title: { type: "string", description: "Action item title" },
              description: { type: "string", description: "Detailed description" },
              sql_action: { type: "string", description: "SQL command to execute, e.g. 'DROP INDEX CONCURRENTLY idx_unused;'" },
              configs: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    parameter: { type: "string" },
                    value: { type: "string" },
                  },
                  required: ["parameter", "value"],
                },
                description: "Configuration parameter changes",
              },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["issue_id", "title"],
            additionalProperties: false,
          },
        },
        {
          name: "update_action_item",
          description: "Update an action item: mark as done/not done, approve/reject, or edit title/description",
          inputSchema: {
            type: "object",
            properties: {
              action_item_id: { type: "string", description: "Action item ID (UUID)" },
              title: { type: "string", description: "New title" },
              description: { type: "string", description: "New description" },
              is_done: { type: "boolean", description: "Mark as done (true) or not done (false)" },
              status: { type: "string", description: "Approval status: 'waiting_for_approval', 'approved', or 'rejected'" },
              status_reason: { type: "string", description: "Reason for approval/rejection" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            required: ["action_item_id"],
            additionalProperties: false,
          },
        },
        // Reports Tools
        {
          name: "list_reports",
          description: "List checkup reports. Returns report metadata: id, project, status, timestamps. Use get_report_data to fetch actual report content. Supports date-based filtering with before_date.",
          inputSchema: {
            type: "object",
            properties: {
              project_id: { type: "number", description: "Filter by project ID" },
              status: { type: "string", description: "Filter by status (e.g., 'completed')" },
              limit: { type: "number", description: "Max number of reports to return (default: 20)" },
              before_date: { type: "string", description: "Show reports created before this date (YYYY-MM-DD, DD.MM.YYYY, YYYY-MM-DD HH:mm, etc.)" },
              all: { type: "boolean", description: "Fetch all reports (paginated automatically)" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            additionalProperties: false,
          },
        },
        {
          name: "list_report_files",
          description: "List files in a checkup report (metadata only, no content). Each report contains json (raw data) and md (markdown analysis) files per check. Either report_id or check_id must be provided.",
          inputSchema: {
            type: "object",
            properties: {
              report_id: { type: "number", description: "Checkup report ID (optional if check_id is provided)" },
              type: { type: "string", description: "Filter by file type: 'json' or 'md'" },
              check_id: { type: "string", description: "Filter by check ID (e.g., 'H002', 'F004')" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            additionalProperties: false,
          },
        },
        {
          name: "get_report_data",
          description: "Get checkup report file content. Returns files with a 'data' field containing the actual content: markdown analysis or JSON raw data. Use type='md' for human-readable analysis with recommendations, type='json' for raw check data. Either report_id or check_id must be provided.",
          inputSchema: {
            type: "object",
            properties: {
              report_id: { type: "number", description: "Checkup report ID (optional if check_id is provided)" },
              type: { type: "string", description: "Filter by file type: 'json' for raw data, 'md' for markdown analysis" },
              check_id: { type: "string", description: "Filter by check ID (e.g., 'H002', 'F004')" },
              debug: { type: "boolean", description: "Enable verbose debug logs" },
            },
            additionalProperties: false,
          },
        },
      ],
    };
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  server.setRequestHandler(CallToolRequestSchema, async (req: any): Promise<any> => {
    return handleToolCall(req, rootOpts, extra);
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}
