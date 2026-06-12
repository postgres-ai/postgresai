import { describe, test, expect, mock, beforeEach, afterEach, spyOn } from "bun:test";
import { handleToolCall, interpretEscapes, type McpToolRequest } from "../lib/mcp-server";
import * as config from "../lib/config";
import * as issues from "../lib/issues";

// Save originals for restoration
const originalFetch = globalThis.fetch;
const originalEnv = { ...process.env };

// Helper to create MCP tool request
function createRequest(name: string, args?: Record<string, unknown>): McpToolRequest {
  return {
    params: {
      name,
      arguments: args,
    },
  };
}

// Helper to extract text from response
function getResponseText(response: { content: Array<{ text: string }> }): string {
  return response.content[0]?.text || "";
}

describe("MCP Server", () => {
  beforeEach(() => {
    // Clear env vars that might interfere
    delete process.env.PGAI_API_KEY;
  });

  afterEach(() => {
    // Restore originals
    globalThis.fetch = originalFetch;
    process.env = { ...originalEnv };
  });

  describe("interpretEscapes", () => {
    test("converts \\n to newline", () => {
      expect(interpretEscapes("line1\\nline2")).toBe("line1\nline2");
    });

    test("converts \\t to tab", () => {
      expect(interpretEscapes("col1\\tcol2")).toBe("col1\tcol2");
    });

    test("converts \\r to carriage return", () => {
      expect(interpretEscapes("text\\rmore")).toBe("text\rmore");
    });

    test('converts \\" to double quote', () => {
      expect(interpretEscapes('say \\"hello\\"')).toBe('say "hello"');
    });

    test("converts \\' to single quote", () => {
      expect(interpretEscapes("it\\'s")).toBe("it's");
    });

    test("handles multiple escape sequences", () => {
      expect(interpretEscapes("line1\\nline2\\ttab\\nline3")).toBe("line1\nline2\ttab\nline3");
    });

    test("handles empty string", () => {
      expect(interpretEscapes("")).toBe("");
    });

    test("handles null/undefined gracefully", () => {
      expect(interpretEscapes(null as unknown as string)).toBe("");
      expect(interpretEscapes(undefined as unknown as string)).toBe("");
    });
  });

  describe("API key validation", () => {
    test("returns error when no API key available", async () => {
      // Mock config to return no API key
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: null,
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("list_issues"));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("API key is required");

      readConfigSpy.mockRestore();
    });

    test("uses API key from rootOpts when provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: null,
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      // Mock fetch to verify API key is used
      let capturedHeaders: Record<string, string> | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedHeaders = options?.headers as Record<string, string> | undefined;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("list_issues"), { apiKey: "test-api-key" });

      expect(capturedHeaders).toBeDefined();
      expect(capturedHeaders!["access-token"]).toBe("test-api-key");

      readConfigSpy.mockRestore();
    });

    test("falls back to config API key when rootOpts not provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "config-api-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedHeaders: Record<string, string> | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedHeaders = options?.headers as Record<string, string> | undefined;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("list_issues"));

      expect(capturedHeaders).toBeDefined();
      expect(capturedHeaders!["access-token"]).toBe("config-api-key");

      readConfigSpy.mockRestore();
    });

    test("uses PGAI_API_KEY env var as fallback", async () => {
      process.env.PGAI_API_KEY = "env-api-key";

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: null,
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedHeaders: Record<string, string> | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedHeaders = options?.headers as Record<string, string> | undefined;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("list_issues"));

      expect(capturedHeaders).toBeDefined();
      expect(capturedHeaders!["access-token"]).toBe("env-api-key");

      readConfigSpy.mockRestore();
    });
  });

  describe("list_issues tool", () => {
    test("successfully returns issues list as JSON", async () => {
      const mockIssues = [
        { id: "issue-1", title: "First Issue" },
        { id: "issue-2", title: "Second Issue" },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify(mockIssues), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_issues"));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed).toHaveLength(2);
      expect(parsed[0].title).toBe("First Issue");

      readConfigSpy.mockRestore();
    });

    test("handles API errors gracefully", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response('{"message": "Unauthorized"}', {
            status: 401,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_issues"));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("401");
      // Agent-facing remediation: invalid key must point at re-auth, not dead-end
      expect(getResponseText(response)).toContain("Run 'postgresai auth'");

      readConfigSpy.mockRestore();
    });
  });

  describe("view_issue tool", () => {
    test("returns error when issue_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_issue", { issue_id: "" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when issue_id is whitespace only", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_issue", { issue_id: "   " }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when issue not found", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      // Return null for issue (not found)
      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response("null", {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("view_issue", { issue_id: "nonexistent-id" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("Issue not found");

      readConfigSpy.mockRestore();
    });

    test("successfully returns combined issue and comments", async () => {
      const mockIssue = { id: "issue-1", title: "Test Issue" };
      const mockComments = [{ id: "comment-1", content: "Test comment" }];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let callCount = 0;
      globalThis.fetch = mock((url: string) => {
        callCount++;
        // First call is for the issue, second is for comments
        if (url.includes("issue_get") || callCount === 1) {
          return Promise.resolve(
            new Response(JSON.stringify(mockIssue), {
              status: 200,
              headers: { "Content-Type": "application/json" },
            })
          );
        }
        return Promise.resolve(
          new Response(JSON.stringify(mockComments), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("view_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed.issue.title).toBe("Test Issue");
      expect(parsed.comments).toHaveLength(1);

      readConfigSpy.mockRestore();
    });
  });

  describe("post_issue_comment tool", () => {
    test("returns error when issue_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("post_issue_comment", { issue_id: "", content: "test" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when content is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("post_issue_comment", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", content: "" })
      );

      expect(response.isError).toBe(true);
      // Error message updated to reflect that attachments alone are also valid input.
      expect(getResponseText(response)).toBe("content or attachments is required");

      readConfigSpy.mockRestore();
    });

    test("interprets escape sequences in content", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "comment-1" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(
        createRequest("post_issue_comment", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          content: "line1\\nline2\\ttab",
        })
      );

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.content).toBe("line1\nline2\ttab");

      readConfigSpy.mockRestore();
    });

    test("successfully creates comment with parent_comment_id", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "comment-1", parent_comment_id: "parent-1" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("post_issue_comment", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          content: "Reply content",
          parent_comment_id: "parent-1",
        })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.parent_comment_id).toBe("parent-1");

      readConfigSpy.mockRestore();
    });
  });

  describe("create_issue tool", () => {
    test("returns error when title is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("create_issue", { title: "" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("title is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when title is whitespace only", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("create_issue", { title: "   " }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("title is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when org_id not provided and not in config", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("create_issue", { title: "Test Issue" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("org_id is required");

      readConfigSpy.mockRestore();
    });

    test("falls back to config orgId when not provided in args", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 42,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "new-issue" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("create_issue", { title: "Test Issue" }));

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.org_id).toBe(42);

      readConfigSpy.mockRestore();
    });

    test("interprets escape sequences in title and description", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "new-issue" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(
        createRequest("create_issue", {
          title: "Title\\nwith newline",
          description: "Desc\\twith tab",
        })
      );

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.title).toBe("Title\nwith newline");
      expect(parsed.description).toBe("Desc\twith tab");

      readConfigSpy.mockRestore();
    });

    test("successfully creates issue with all parameters", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "new-issue", title: "Test" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_issue", {
          title: "Test Issue",
          description: "Test description",
          org_id: 123,
          project_id: 456,
          labels: ["bug", "urgent"],
        })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.title).toBe("Test Issue");
      expect(parsed.description).toBe("Test description");
      expect(parsed.org_id).toBe(123);
      expect(parsed.project_id).toBe(456);
      expect(parsed.labels).toEqual(["bug", "urgent"]);

      readConfigSpy.mockRestore();
    });
  });

  describe("update_issue tool", () => {
    test("returns error when issue_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "", title: "New Title" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when no update fields provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("At least one field to update is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when status is not 0 or 1", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", status: 2 })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("status must be 0 (open) or 1 (closed)");

      readConfigSpy.mockRestore();
    });

    test("returns error when status is negative", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", status: -1 })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("status must be 0 (open) or 1 (closed)");

      readConfigSpy.mockRestore();
    });

    test("interprets escape sequences in title and description", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "issue-1" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(
        createRequest("update_issue", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          title: "Updated\\nTitle",
          description: "Updated\\tDescription",
        })
      );

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.p_title).toBe("Updated\nTitle");
      expect(parsed.p_description).toBe("Updated\tDescription");

      readConfigSpy.mockRestore();
    });

    test("successfully updates with only title", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify({ id: "issue-1", title: "New Title" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", title: "New Title" })
      );

      expect(response.isError).toBeUndefined();

      readConfigSpy.mockRestore();
    });

    test("successfully updates with only status", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "issue-1", status: 1 }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", status: 1 })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.p_status).toBe(1);

      readConfigSpy.mockRestore();
    });

    test("successfully updates with only labels", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "issue-1", labels: ["new-label"] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", labels: ["new-label"] })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.p_labels).toEqual(["new-label"]);

      readConfigSpy.mockRestore();
    });

    test("accepts status=0 to reopen issue", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "issue-1", status: 0 }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", status: 0 })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.p_status).toBe(0);

      readConfigSpy.mockRestore();
    });
  });

  describe("update_issue_comment tool", () => {
    test("returns error when comment_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_issue_comment", { comment_id: "", content: "new content" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("comment_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when content is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_issue_comment", { comment_id: "comment-1", content: "" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("content or attachments is required");

      readConfigSpy.mockRestore();
    });

    test("interprets escape sequences in content", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify({ id: "comment-1" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(
        createRequest("update_issue_comment", {
          comment_id: "comment-1",
          content: "updated\\ncontent\\twith escapes",
        })
      );

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.p_content).toBe("updated\ncontent\twith escapes");

      readConfigSpy.mockRestore();
    });

    test("successfully updates comment", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify({ id: "comment-1", content: "Updated content" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue_comment", {
          comment_id: "comment-1",
          content: "Updated content",
        })
      );

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed.content).toBe("Updated content");

      readConfigSpy.mockRestore();
    });
  });

  describe("view_action_item tool", () => {
    test("returns error when no IDs provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_action_item", {}));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("action_item_id or action_item_ids is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when action_item_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_action_item", { action_item_id: "" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("action_item_id or action_item_ids is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when action_item_ids is empty array", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_action_item", { action_item_ids: [] }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("action_item_id or action_item_ids is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when action_item_id is not a valid UUID", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("view_action_item", { action_item_id: "invalid-id-format" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("actionItemId is required and must be a valid UUID");

      readConfigSpy.mockRestore();
    });

    test("returns error when action item not found", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response("[]", {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("view_action_item", { action_item_id: "00000000-0000-0000-0000-000000000000" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("Action item(s) not found");

      readConfigSpy.mockRestore();
    });

    test("successfully returns single action item details", async () => {
      const mockActionItem = {
        id: "11111111-1111-1111-1111-111111111111",
        issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        title: "Fix index",
        description: "Drop unused index",
        severity: 3,
        is_done: false,
        status: "waiting_for_approval",
        sql_action: "DROP INDEX CONCURRENTLY idx_unused;",
        configs: [{ parameter: "work_mem", value: "256MB" }],
      };

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify([mockActionItem]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("view_action_item", { action_item_id: "11111111-1111-1111-1111-111111111111" }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(Array.isArray(parsed)).toBe(true);
      expect(parsed[0].title).toBe("Fix index");
      expect(parsed[0].sql_action).toBe("DROP INDEX CONCURRENTLY idx_unused;");
      expect(parsed[0].configs).toEqual([{ parameter: "work_mem", value: "256MB" }]);

      readConfigSpy.mockRestore();
    });

    test("successfully returns multiple action items", async () => {
      const mockActionItems = [
        { id: "11111111-1111-1111-1111-111111111111", title: "Fix index", severity: 3 },
        { id: "22222222-2222-2222-2222-222222222222", title: "Update config", severity: 2 },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify(mockActionItems), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("view_action_item", { action_item_ids: ["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"] }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed).toHaveLength(2);
      expect(parsed[0].title).toBe("Fix index");
      expect(parsed[1].title).toBe("Update config");
      // Verify the URL uses in.() syntax
      expect(capturedUrl).toContain("id=in.");

      readConfigSpy.mockRestore();
    });
  });

  describe("list_action_items tool", () => {
    test("returns error when issue_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("list_action_items", { issue_id: "" }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when issue_id is whitespace only", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("list_action_items", { issue_id: "   " }));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("successfully returns action items list as JSON", async () => {
      const mockActionItems = [
        { id: "action-1", title: "First Action", severity: 1 },
        { id: "action-2", title: "Second Action", severity: 2 },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify(mockActionItems), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_action_items", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed).toHaveLength(2);
      expect(parsed[0].title).toBe("First Action");

      readConfigSpy.mockRestore();
    });
  });

  describe("create_action_item tool", () => {
    test("returns error when issue_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("create_action_item", { issue_id: "", title: "Test" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("issue_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when title is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("create_action_item", { issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", title: "" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("title is required");

      readConfigSpy.mockRestore();
    });

    test("successfully creates action item with minimal params", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify("new-action-item-id"), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_action_item", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          title: "Fix the index",
        })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.issue_id).toBe("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
      expect(parsed.title).toBe("Fix the index");

      readConfigSpy.mockRestore();
    });

    test("successfully creates action item with all params", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify("new-action-item-id"), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_action_item", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          title: "Fix the index",
          description: "Drop the unused index to improve performance",
          sql_action: "DROP INDEX CONCURRENTLY idx_unused;",
          configs: [{ parameter: "work_mem", value: "256MB" }],
        })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.issue_id).toBe("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
      expect(parsed.title).toBe("Fix the index");
      expect(parsed.description).toBe("Drop the unused index to improve performance");
      expect(parsed.sql_action).toBe("DROP INDEX CONCURRENTLY idx_unused;");
      expect(parsed.configs).toEqual([{ parameter: "work_mem", value: "256MB" }]);

      readConfigSpy.mockRestore();
    });

    test("interprets escape sequences in title and description", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response(JSON.stringify("new-action-item-id"), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(
        createRequest("create_action_item", {
          issue_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          title: "Title\\nwith newline",
          description: "Desc\\twith tab",
        })
      );

      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.title).toBe("Title\nwith newline");
      expect(parsed.description).toBe("Desc\twith tab");

      readConfigSpy.mockRestore();
    });
  });

  describe("update_action_item tool", () => {
    test("returns error when action_item_id is empty", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_action_item", { action_item_id: "", title: "New Title" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("action_item_id is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when no update fields provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_action_item", { action_item_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("At least one field to update is required");

      readConfigSpy.mockRestore();
    });

    test("returns error when status is invalid", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(
        createRequest("update_action_item", { action_item_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", status: "invalid_status" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("status must be");

      readConfigSpy.mockRestore();
    });

    test("successfully updates with only title", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response("", {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_action_item", { action_item_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", title: "New Title" })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.action_item_id).toBe("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb");
      expect(parsed.title).toBe("New Title");

      readConfigSpy.mockRestore();
    });

    test("successfully updates is_done", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response("", {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_action_item", { action_item_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", is_done: true })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.is_done).toBe(true);

      readConfigSpy.mockRestore();
    });

    test("successfully updates status with status_reason", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedBody: string | undefined;
      globalThis.fetch = mock((url: string, options?: RequestInit) => {
        capturedBody = options?.body as string;
        return Promise.resolve(
          new Response("", {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_action_item", {
          action_item_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
          status: "approved",
          status_reason: "Looks good to me",
        })
      );

      expect(response.isError).toBeUndefined();
      expect(capturedBody).toBeDefined();
      const parsed = JSON.parse(capturedBody!);
      expect(parsed.status).toBe("approved");
      expect(parsed.status_reason).toBe("Looks good to me");

      readConfigSpy.mockRestore();
    });
  });

  describe("unknown tool handling", () => {
    test("returns error for unknown tool name", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("nonexistent_tool"));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("Unknown tool: nonexistent_tool");

      readConfigSpy.mockRestore();
    });
  });

  describe("list_reports tool", () => {
    test("successfully returns reports list as JSON", async () => {
      const mockReports = [
        { id: 1, org_id: 1, org_name: "TestOrg", project_id: 10, project_name: "prod-db", status: "completed" },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify(mockReports), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_reports"));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed).toHaveLength(1);
      expect(parsed[0].status).toBe("completed");

      readConfigSpy.mockRestore();
    });

    test("passes filters to API", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("list_reports", {
        project_id: 5,
        status: "completed",
        limit: 10,
      }));

      expect(capturedUrl).toContain("project_id=eq.5");
      expect(capturedUrl).toContain("status=eq.completed");
      expect(capturedUrl).toContain("limit=10");

      readConfigSpy.mockRestore();
    });

    test("handles API errors gracefully", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response('{"message": "Unauthorized"}', {
            status: 401,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_reports"));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("401");
      // Agent-facing remediation: invalid key must point at re-auth, not dead-end
      expect(getResponseText(response)).toContain("Run 'postgresai auth'");

      readConfigSpy.mockRestore();
    });

    test("passes before_date to API as created_at filter", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("list_reports", {
        before_date: "2025-01-15",
      }));

      expect(capturedUrl).toContain("created_at=lt.2025-01-15");

      readConfigSpy.mockRestore();
    });

    test("fetches all reports when all=true", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const mockReports = [
        { id: 10, org_id: 1, org_name: "O", project_id: 1, project_name: "P", status: "completed" },
      ];

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify(mockReports), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_reports", {
        all: true,
      }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed).toHaveLength(1);
      expect(parsed[0].id).toBe(10);

      readConfigSpy.mockRestore();
    });
  });

  describe("list_report_files tool", () => {
    test("returns error when neither report_id nor check_id is provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("list_report_files", {}));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("Either report_id or check_id is required");

      readConfigSpy.mockRestore();
    });

    test("works with only check_id (no report_id)", async () => {
      const mockFiles = [
        { id: 100, checkup_report_id: 1, filename: "H002.md", check_id: "H002", type: "md" },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify(mockFiles), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_report_files", {
        check_id: "H002",
      }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed[0].filename).toBe("H002.md");
      expect(capturedUrl).toContain("check_id=eq.H002");
      expect(capturedUrl).not.toContain("checkup_report_id");

      readConfigSpy.mockRestore();
    });

    test("successfully returns report files", async () => {
      const mockFiles = [
        { id: 100, checkup_report_id: 1, filename: "H002.md", check_id: "H002", type: "md" },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify(mockFiles), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("list_report_files", {
        report_id: 1,
        type: "md",
        check_id: "H002",
      }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed[0].filename).toBe("H002.md");
      expect(capturedUrl).toContain("checkup_report_id=eq.1");
      expect(capturedUrl).toContain("type=eq.md");
      expect(capturedUrl).toContain("check_id=eq.H002");

      readConfigSpy.mockRestore();
    });
  });

  describe("get_report_data tool", () => {
    test("returns error when neither report_id nor check_id is provided", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      const response = await handleToolCall(createRequest("get_report_data", {}));

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("Either report_id or check_id is required");

      readConfigSpy.mockRestore();
    });

    test("works with only check_id (no report_id)", async () => {
      const mockData = [
        {
          id: 100,
          checkup_report_id: 1,
          filename: "H002.md",
          check_id: "H002",
          type: "md",
          data: "# H002\n\nUnused indexes found.",
        },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify(mockData), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("get_report_data", {
        check_id: "H002",
      }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed[0].data).toContain("# H002");
      expect(capturedUrl).toContain("check_id=eq.H002");
      expect(capturedUrl).not.toContain("checkup_report_id");

      readConfigSpy.mockRestore();
    });

    test("successfully returns report data with content", async () => {
      const mockData = [
        {
          id: 100,
          checkup_report_id: 1,
          filename: "H002.md",
          check_id: "H002",
          type: "md",
          data: "# H002\n\nUnused indexes found.",
        },
      ];

      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response(JSON.stringify(mockData), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("get_report_data", {
        report_id: 1,
        type: "md",
        check_id: "H002",
      }));

      expect(response.isError).toBeUndefined();
      const parsed = JSON.parse(getResponseText(response));
      expect(parsed[0].data).toContain("# H002");

      readConfigSpy.mockRestore();
    });

    test("passes filters to API", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: null,
        defaultProject: null,
        projectName: null,
      });

      let capturedUrl: string | undefined;
      globalThis.fetch = mock((url: string) => {
        capturedUrl = url;
        return Promise.resolve(
          new Response(JSON.stringify([]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }) as unknown as typeof fetch;

      await handleToolCall(createRequest("get_report_data", {
        report_id: 42,
        type: "json",
        check_id: "F004",
      }));

      expect(capturedUrl).toContain("checkup_report_id=eq.42");
      expect(capturedUrl).toContain("type=eq.json");
      expect(capturedUrl).toContain("check_id=eq.F004");

      readConfigSpy.mockRestore();
    });
  });

  describe("error propagation", () => {
    test("propagates API errors through MCP layer", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() =>
        Promise.resolve(
          new Response('{"message": "Internal Server Error"}', {
            status: 500,
            headers: { "Content-Type": "application/json" },
          })
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_issue", { title: "Test Issue" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("500");

      readConfigSpy.mockRestore();
    });

    test("handles network errors gracefully", async () => {
      const readConfigSpy = spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });

      globalThis.fetch = mock(() => Promise.reject(new Error("Network error"))) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_issue", { title: "Test Issue" })
      );

      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toContain("Network error");

      readConfigSpy.mockRestore();
    });
  });

  describe("attachments parameter & file tools", () => {
    // Real-file approach (rather than fs mocking) — ESM module caching means
    // monkey-patching fs after `import * as fs from "fs"` does not affect the
    // already-resolved binding inside storage.ts. Real tmp files are simpler
    // and match how the existing storage tests work.
    const fs = require("fs") as typeof import("fs");
    const path = require("path") as typeof import("path");
    const os = require("os") as typeof import("os");

    const createdDirs: string[] = [];

    function mockTinyFile(name: string, body = "FAKE"): string {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-mcp-attach-"));
      createdDirs.push(dir);
      const p = path.join(dir, name);
      fs.writeFileSync(p, body);
      return p;
    }

    afterEach(() => {
      while (createdDirs.length > 0) {
        const d = createdDirs.pop();
        if (d) fs.rmSync(d, { recursive: true, force: true });
      }
    });

    function configWithKey() {
      return spyOn(config, "readConfig").mockReturnValue({
        apiKey: "test-key",
        baseUrl: null,
        storageBaseUrl: null,
        orgId: 1,
        defaultProject: null,
        projectName: null,
      });
    }

    test("upload_file tool returns url + ready-to-paste markdown link", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("shot.png");

      globalThis.fetch = mock((_url: string, _init?: RequestInit) =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              success: true,
              url: "/files/9/abc.png",
              metadata: { originalName: "shot.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r1",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          )
        )
      ) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("upload_file", { path: fakePath }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );

      expect(response.isError).toBeFalsy();
      const obj = JSON.parse(getResponseText(response));
      expect(obj.success).toBe(true);
      expect(obj.url).toBe("/files/9/abc.png");
      // Image extension renders inline.
      expect(obj.markdown).toBe("![shot.png](https://storage.example.com/files/9/abc.png)");

      cfgSpy.mockRestore();
    });

    test("upload_file requires path", async () => {
      const cfgSpy = configWithKey();
      const r = await handleToolCall(createRequest("upload_file", {}));
      expect(r.isError).toBe(true);
      expect(getResponseText(r)).toBe("path is required");
      cfgSpy.mockRestore();
    });

    test("download_file tool requires url", async () => {
      const cfgSpy = configWithKey();
      const r = await handleToolCall(createRequest("download_file", {}));
      expect(r.isError).toBe(true);
      expect(getResponseText(r)).toBe("url is required");
      cfgSpy.mockRestore();
    });

    test("post_issue_comment with attachments uploads and appends link", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("debug.png");

      const calls: Array<{ url: string; method?: string; bodyJson?: unknown }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        calls.push({
          url: u,
          method: init?.method,
          bodyJson: typeof init?.body === "string" ? JSON.parse(init.body as string) : undefined,
        });
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true,
              url: "/files/9/dbg.png",
              metadata: { originalName: "debug.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r1",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_comment_create")) {
          return new Response(
            JSON.stringify({
              id: "c1",
              issue_id: "i1",
              author_id: 1,
              parent_comment_id: null,
              content: "ignored",
              created_at: "",
              updated_at: "",
              data: null,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("post_issue_comment", {
          issue_id: "11111111-1111-1111-1111-111111111111",
          content: "see screenshot",
          attachments: [fakePath],
        }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );

      expect(response.isError).toBeFalsy();

      // Upload happened first, then comment-create with augmented content.
      expect(calls[0].url).toContain("/upload");
      const commentCall = calls.find((c) => c.url.endsWith("/rpc/issue_comment_create"));
      expect(commentCall).toBeTruthy();
      const body = commentCall!.bodyJson as { content: string };
      expect(body.content).toBe("see screenshot\n\n![debug.png](https://storage.example.com/files/9/dbg.png)");

      cfgSpy.mockRestore();
    });

    test("post_issue_comment with only attachments (no content) is allowed", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("only.png");

      const commentBodies: Array<{ content: string }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true,
              url: "/files/9/only.png",
              metadata: { originalName: "only.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_comment_create")) {
          commentBodies.push(JSON.parse(String(init?.body)));
          return new Response(
            JSON.stringify({ id: "c1", issue_id: "i1", author_id: 1, parent_comment_id: null, content: "", created_at: "", updated_at: "", data: null }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("post_issue_comment", {
          issue_id: "11111111-1111-1111-1111-111111111111",
          content: "",
          attachments: [fakePath],
        }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );

      expect(response.isError).toBeFalsy();
      expect(commentBodies[0].content).toBe("![only.png](https://storage.example.com/files/9/only.png)");

      cfgSpy.mockRestore();
    });

    test("post_issue_comment with no content and no attachments is rejected", async () => {
      const cfgSpy = configWithKey();
      const r = await handleToolCall(
        createRequest("post_issue_comment", {
          issue_id: "11111111-1111-1111-1111-111111111111",
          content: "",
        })
      );
      expect(r.isError).toBe(true);
      expect(getResponseText(r)).toBe("content or attachments is required");
      cfgSpy.mockRestore();
    });

    test("update_issue with attachments and no description fetches existing then appends", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("evidence.png");

      const calls: Array<{ url: string; method?: string; body?: unknown }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        calls.push({
          url: u,
          method: init?.method,
          body: typeof init?.body === "string" ? JSON.parse(init.body as string) : undefined,
        });
        if (u.includes("/issues?") && (init?.method ?? "GET") === "GET") {
          return new Response(
            JSON.stringify([
              { id: "i1", title: "T", description: "Existing description", status: 0, created_at: "", action_items: [] },
            ]),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true,
              url: "/files/9/ev.png",
              metadata: { originalName: "evidence.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_update")) {
          return new Response(
            JSON.stringify({ id: "i1", title: "T", description: "ignored", status: 0, updated_at: "" }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", {
          issue_id: "i1",
          attachments: [fakePath],
        }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );

      expect(response.isError).toBeFalsy();
      const updateCall = calls.find((c) => c.url.endsWith("/rpc/issue_update"));
      expect(updateCall).toBeTruthy();
      // Order: GET issues -> POST upload -> POST update.
      const fetchIdx = calls.findIndex((c) => c.url.includes("/issues?"));
      const uploadIdx = calls.findIndex((c) => c.url.endsWith("/upload"));
      const updateIdx = calls.findIndex((c) => c.url.endsWith("/rpc/issue_update"));
      expect(fetchIdx).toBeGreaterThanOrEqual(0);
      expect(uploadIdx).toBeGreaterThan(fetchIdx);
      expect(updateIdx).toBeGreaterThan(uploadIdx);

      const body = updateCall!.body as { p_description: string };
      expect(body.p_description).toBe(
        "Existing description\n\n![evidence.png](https://storage.example.com/files/9/ev.png)"
      );

      cfgSpy.mockRestore();
    });

    test("update_issue with only attachments is treated as a valid update", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("ok.png");

      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.includes("/issues?")) {
          return new Response(JSON.stringify([{ id: "i1", title: "T", description: "old", status: 0 }]), {
            status: 200, headers: { "Content-Type": "application/json" },
          });
        }
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true, url: "/files/9/ok.png",
              metadata: { originalName: "ok.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_update")) {
          return new Response(JSON.stringify({ id: "i1", title: "T", description: "ignored", status: 0, updated_at: "" }), {
            status: 200, headers: { "Content-Type": "application/json" },
          });
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue", { issue_id: "i1", attachments: [fakePath] }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );
      expect(response.isError).toBeFalsy();
      cfgSpy.mockRestore();
    });

    test("update_issue with no fields including no attachments is rejected", async () => {
      const cfgSpy = configWithKey();
      const r = await handleToolCall(createRequest("update_issue", { issue_id: "i1" }));
      expect(r.isError).toBe(true);
      expect(getResponseText(r)).toContain("At least one field to update is required");
      // The error message now mentions attachments as a valid update field.
      expect(getResponseText(r)).toContain("attachments");
      cfgSpy.mockRestore();
    });

    test("create_issue with attachments appends link to provided description", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("design.png");

      const createBodies: Array<{ description?: string }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true, url: "/files/9/dz.png",
              metadata: { originalName: "design.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_create")) {
          createBodies.push(JSON.parse(String(init?.body)));
          return new Response(
            JSON.stringify({ id: "i1", title: "T", description: "ignored", created_at: "", status: 0 }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_issue", {
          title: "Dx",
          description: "see design",
          attachments: [fakePath],
        }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );
      expect(response.isError).toBeFalsy();
      expect(createBodies[0].description).toBe(
        "see design\n\n![design.png](https://storage.example.com/files/9/dz.png)"
      );
      cfgSpy.mockRestore();
    });

    test("update_issue_comment with attachments appends to content", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("after.png");

      const bodies: Array<{ p_content?: string }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true, url: "/files/9/aft.png",
              metadata: { originalName: "after.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_comment_update")) {
          bodies.push(JSON.parse(String(init?.body)));
          return new Response(JSON.stringify({ id: "c1", issue_id: "i1", content: "ignored", updated_at: "" }), {
            status: 200, headers: { "Content-Type": "application/json" },
          });
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue_comment", { comment_id: "c1", content: "now updated", attachments: [fakePath] }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );
      expect(response.isError).toBeFalsy();
      expect(bodies[0].p_content).toBe("now updated\n\n![after.png](https://storage.example.com/files/9/aft.png)");
      cfgSpy.mockRestore();
    });

    test("update_issue_comment with attachments-only (no content) sends just the markdown link", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("only.png");

      const bodies: Array<{ p_content?: string }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true, url: "/files/9/only.png",
              metadata: { originalName: "only.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_comment_update")) {
          bodies.push(JSON.parse(String(init?.body)));
          return new Response(JSON.stringify({ id: "c1", issue_id: "i1", content: "ignored", updated_at: "" }), {
            status: 200, headers: { "Content-Type": "application/json" },
          });
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("update_issue_comment", { comment_id: "c1", attachments: [fakePath] }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );
      expect(response.isError).toBeFalsy();
      expect(bodies).toHaveLength(1);
      expect(bodies[0].p_content).toBe("![only.png](https://storage.example.com/files/9/only.png)");
      cfgSpy.mockRestore();
    });

    test("update_issue_comment without content and without attachments is rejected", async () => {
      const cfgSpy = configWithKey();
      const fetchSpy = mock(() => {
        throw new Error("should not be called");
      });
      globalThis.fetch = fetchSpy as unknown as typeof fetch;

      const response = await handleToolCall(createRequest("update_issue_comment", { comment_id: "c1" }));
      expect(response.isError).toBe(true);
      expect(getResponseText(response)).toBe("content or attachments is required");
      expect(fetchSpy).not.toHaveBeenCalled();
      cfgSpy.mockRestore();
    });

    test("create_issue with attachments and no description sets description to just the link", async () => {
      const cfgSpy = configWithKey();
      const fakePath = mockTinyFile("plan.png");

      const createBodies: Array<{ description?: string }> = [];
      globalThis.fetch = mock(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        if (u.endsWith("/upload")) {
          return new Response(
            JSON.stringify({
              success: true, url: "/files/9/plan.png",
              metadata: { originalName: "plan.png", size: 4, mimeType: "image/png", uploadedAt: "", duration: 0 },
              requestId: "r",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        if (u.endsWith("/rpc/issue_create")) {
          createBodies.push(JSON.parse(String(init?.body)));
          return new Response(
            JSON.stringify({ id: "i1", title: "T", description: "ignored", created_at: "", status: 0 }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("nope", { status: 404 });
      }) as unknown as typeof fetch;

      const response = await handleToolCall(
        createRequest("create_issue", { title: "Dx", attachments: [fakePath] }),
        { apiBaseUrl: "https://api.example.com", storageBaseUrl: "https://storage.example.com" }
      );
      expect(response.isError).toBeFalsy();
      expect(createBodies[0].description).toBe("![plan.png](https://storage.example.com/files/9/plan.png)");
      cfgSpy.mockRestore();
    });
  });
});
