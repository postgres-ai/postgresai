import { describe, test, expect, mock, beforeEach, afterEach } from "bun:test";
import { createIssue, updateIssue, updateIssueComment, fetchIssues, fetchIssue, withVisibleHiddenFlag, issueRequestHeaders } from "../lib/issues";

// Mock fetch globally
const originalFetch = globalThis.fetch;

describe("createIssue", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      createIssue({
        apiKey: "",
        apiBaseUrl: "https://api.example.com",
        title: "Test Issue",
        orgId: 1,
      })
    ).rejects.toThrow("API key is required");
  });

  test("throws when title is missing", async () => {
    await expect(
      createIssue({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        title: "",
        orgId: 1,
      })
    ).rejects.toThrow("title is required");
  });

  test("throws when orgId is not a number", async () => {
    await expect(
      createIssue({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        title: "Test Issue",
        orgId: undefined as unknown as number,
      })
    ).rejects.toThrow("orgId is required");
  });

  test("accepts orgId=0 as valid", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Test Issue",
      description: null,
      created_at: "2025-01-01T00:00:00Z",
      status: 0,
      project_id: null,
      labels: null,
    };

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await createIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      title: "Test Issue",
      orgId: 0,
    });

    expect(result.id).toBe("test-id");
  });

  test("makes correct API call with all parameters", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Test Issue",
      description: "Test description",
      created_at: "2025-01-01T00:00:00Z",
      status: 0,
      project_id: 123,
      labels: ["bug", "urgent"],
    };

    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await createIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      title: "Test Issue",
      orgId: 1,
      description: "Test description",
      projectId: 123,
      labels: ["bug", "urgent"],
    });

    expect(capturedRequest).not.toBeNull();
    expect(capturedRequest!.url).toBe("https://api.example.com/rpc/issue_create");
    expect(capturedRequest!.options.method).toBe("POST");

    const body = JSON.parse(capturedRequest!.options.body as string);
    expect(body.title).toBe("Test Issue");
    expect(body.org_id).toBe(1);
    expect(body.description).toBe("Test description");
    expect(body.project_id).toBe(123);
    expect(body.labels).toEqual(["bug", "urgent"]);

    expect(result).toEqual(mockResponse);
  });

  test("handles API error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Unauthorized"}', {
          status: 401,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      createIssue({
        apiKey: "invalid-key",
        apiBaseUrl: "https://api.example.com",
        title: "Test Issue",
        orgId: 1,
      })
    ).rejects.toThrow(/Failed to create issue/);
  });
});

describe("updateIssue", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      updateIssue({
        apiKey: "",
        apiBaseUrl: "https://api.example.com",
        issueId: "test-id",
        title: "Updated Title",
      })
    ).rejects.toThrow("API key is required");
  });

  test("throws when issueId is missing", async () => {
    await expect(
      updateIssue({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        issueId: "",
        title: "Updated Title",
      })
    ).rejects.toThrow("issueId is required");
  });

  test("throws when no update fields are provided", async () => {
    await expect(
      updateIssue({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        issueId: "test-id",
      })
    ).rejects.toThrow("At least one field to update is required");
  });

  test("accepts update with only title", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Updated Title",
      description: null,
      status: 0,
      updated_at: "2025-01-01T00:00:00Z",
      labels: null,
    };

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await updateIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "test-id",
      title: "Updated Title",
    });

    expect(result.title).toBe("Updated Title");
  });

  test("accepts update with only description", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Original Title",
      description: "New description",
      status: 0,
      updated_at: "2025-01-01T00:00:00Z",
      labels: null,
    };

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await updateIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "test-id",
      description: "New description",
    });

    expect(result.description).toBe("New description");
  });

  test("accepts update with only status", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Title",
      description: null,
      status: 1,
      updated_at: "2025-01-01T00:00:00Z",
      labels: null,
    };

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await updateIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "test-id",
      status: 1,
    });

    expect(result.status).toBe(1);
  });

  test("accepts update with only labels", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Title",
      description: null,
      status: 0,
      updated_at: "2025-01-01T00:00:00Z",
      labels: ["new-label"],
    };

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await updateIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "test-id",
      labels: ["new-label"],
    });

    expect(result.labels).toEqual(["new-label"]);
  });

  test("makes correct API call with all parameters", async () => {
    const mockResponse = {
      id: "test-id",
      title: "Updated Title",
      description: "Updated description",
      status: 1,
      updated_at: "2025-01-01T00:00:00Z",
      labels: ["bug"],
    };

    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await updateIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "test-id",
      title: "Updated Title",
      description: "Updated description",
      status: 1,
      labels: ["bug"],
    });

    expect(capturedRequest).not.toBeNull();
    expect(capturedRequest!.url).toBe("https://api.example.com/rpc/issue_update");
    expect(capturedRequest!.options.method).toBe("POST");

    const body = JSON.parse(capturedRequest!.options.body as string);
    expect(body.p_id).toBe("test-id");
    expect(body.p_title).toBe("Updated Title");
    expect(body.p_description).toBe("Updated description");
    expect(body.p_status).toBe(1);
    expect(body.p_labels).toEqual(["bug"]);
  });

  test("handles API error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Not found"}', {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      updateIssue({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        issueId: "nonexistent-id",
        title: "Updated Title",
      })
    ).rejects.toThrow(/Failed to update issue/);
  });
});

describe("updateIssueComment", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      updateIssueComment({
        apiKey: "",
        apiBaseUrl: "https://api.example.com",
        commentId: "test-id",
        content: "Updated content",
      })
    ).rejects.toThrow("API key is required");
  });

  test("throws when commentId is missing", async () => {
    await expect(
      updateIssueComment({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        commentId: "",
        content: "Updated content",
      })
    ).rejects.toThrow("commentId is required");
  });

  test("throws when content is missing", async () => {
    await expect(
      updateIssueComment({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        commentId: "test-id",
        content: "",
      })
    ).rejects.toThrow("content is required");
  });

  test("makes correct API call", async () => {
    const mockResponse = {
      id: "test-id",
      issue_id: "issue-id",
      content: "Updated content",
      updated_at: "2025-01-01T00:00:00Z",
    };

    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await updateIssueComment({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      commentId: "test-id",
      content: "Updated content",
    });

    expect(capturedRequest).not.toBeNull();
    expect(capturedRequest!.url).toBe("https://api.example.com/rpc/issue_comment_update");
    expect(capturedRequest!.options.method).toBe("POST");

    const body = JSON.parse(capturedRequest!.options.body as string);
    expect(body.p_id).toBe("test-id");
    expect(body.p_content).toBe("Updated content");

    expect(result).toEqual(mockResponse);
  });

  test("handles API error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Not found"}', {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      updateIssueComment({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        commentId: "nonexistent-id",
        content: "Updated content",
      })
    ).rejects.toThrow(/Failed to update issue comment/);
  });
});

describe("hidden issues (platform-all #562)", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("fetchIssues requests the is_hidden column", async () => {
    let capturedUrl = "";
    globalThis.fetch = mock((url: string) => {
      capturedUrl = url;
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchIssues({ apiKey: "test-key", apiBaseUrl: "https://api.example.com" });

    const select = new URL(capturedUrl).searchParams.get("select");
    expect(select?.split(",")).toContain("is_hidden");
  });

  test("fetchIssue requests the is_hidden column", async () => {
    let capturedUrl = "";
    globalThis.fetch = mock((url: string) => {
      capturedUrl = url;
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "00000000-0000-0000-0000-000000000000",
    });

    const select = new URL(capturedUrl).searchParams.get("select");
    expect(select?.split(",")).toContain("is_hidden");
  });

  test("fetchIssue propagates is_hidden through the response mapping", async () => {
    // Regression guard: fetchIssue re-builds the issue field-by-field, so a
    // requested column is silently dropped unless it is explicitly copied over.
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(
          JSON.stringify([
            {
              id: "00000000-0000-0000-0000-000000000000",
              title: "t",
              description: "d",
              status: 0,
              created_at: "2026-01-01",
              author_display_name: "staff",
              action_items: [],
              is_hidden: true,
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
    ) as unknown as typeof fetch;

    const issue = await fetchIssue({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      issueId: "00000000-0000-0000-0000-000000000000",
    });

    expect(issue?.is_hidden).toBe(true);
  });

  test("--hidden-only filters server-side, and is absent otherwise", async () => {
    const urls: string[] = [];
    globalThis.fetch = mock((url: string) => {
      urls.push(url);
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchIssues({ apiKey: "k", apiBaseUrl: "https://api.example.com", hiddenOnly: true });
    await fetchIssues({ apiKey: "k", apiBaseUrl: "https://api.example.com" });

    expect(new URL(urls[0]!).searchParams.get("is_hidden")).toBe("eq.true");
    expect(new URL(urls[1]!).searchParams.has("is_hidden")).toBe(false);
  });

  test("withVisibleHiddenFlag keeps the flag only when true", () => {
    // A non-staff caller can only ever receive false (hidden rows are filtered
    // out server-side); printing "is_hidden: false" would disclose that the
    // mechanism exists, so the key must disappear entirely.
    const hidden = { id: "a", is_hidden: true };
    const visible = { id: "a", is_hidden: false };
    // An older backend that does not return the column at all.
    const absent: { id: string; is_hidden?: boolean } = { id: "a" };

    expect(withVisibleHiddenFlag(hidden)).toEqual({ id: "a", is_hidden: true });
    expect(withVisibleHiddenFlag(visible)).toEqual({ id: "a" });
    expect(withVisibleHiddenFlag(absent)).toEqual({ id: "a" });
  });

  test("withVisibleHiddenFlag leaves other fields untouched", () => {
    const issue = { id: "a", title: "t", status: 0, is_hidden: false };
    expect(withVisibleHiddenFlag(issue)).toEqual({ id: "a", title: "t", status: 0 });
  });

  test("withVisibleHiddenFlag does not mutate its input", () => {
    const issue = { id: "a", is_hidden: false };
    withVisibleHiddenFlag(issue);
    expect(issue.is_hidden).toBe(false);
  });
});

describe("hidden-issue opt-in header (default-exclude)", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("issueRequestHeaders declares the capability alongside the token", () => {
    const h = issueRequestHeaders("k");
    expect(h["access-token"]).toBe("k");
    expect(h["x-pgai-include-hidden"]).toBe("true");
  });

  test("fetchIssues sends the opt-in header", async () => {
    let captured: Record<string, string> = {};
    globalThis.fetch = mock((_url: string, options: RequestInit) => {
      captured = options.headers as Record<string, string>;
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchIssues({ apiKey: "k", apiBaseUrl: "https://api.example.com" });
    expect(captured["x-pgai-include-hidden"]).toBe("true");
  });

  test("fetchIssue sends the opt-in header", async () => {
    let captured: Record<string, string> = {};
    globalThis.fetch = mock((_url: string, options: RequestInit) => {
      captured = options.headers as Record<string, string>;
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchIssue({
      apiKey: "k",
      apiBaseUrl: "https://api.example.com",
      issueId: "00000000-0000-0000-0000-000000000000",
    });
    expect(captured["x-pgai-include-hidden"]).toBe("true");
  });

  test("createIssue sends the opt-in header too", async () => {
    // Write paths need it as well. Not for flipping is_hidden — the CLI never
    // sends that key and has no flag for it — but for reaching an existing
    // hidden issue at all: issue_hidden_access_check() raises PT404 for a
    // non-staff caller, and without this header a token caller IS non-staff.
    let captured: Record<string, string> = {};
    globalThis.fetch = mock((_url: string, options: RequestInit) => {
      captured = options.headers as Record<string, string>;
      return Promise.resolve(
        new Response(JSON.stringify({ id: "x" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await createIssue({
      apiKey: "k",
      apiBaseUrl: "https://api.example.com",
      title: "t",
      orgId: 1,
    });
    expect(captured["x-pgai-include-hidden"]).toBe("true");
  });
});

describe("global tokens reach hidden issues (postgresai #327 + platform-all #562)", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("the org selector and the hidden opt-in travel together", () => {
    // A global token names no org of its own, so the platform rejects it with
    // PT400 unless the request selects one. user_is_staff() resolves global
    // tokens too, so BOTH headers must be present on the same request or staff
    // silently see no hidden issues (hidden => PT404 looks like "none exist").
    const h = issueRequestHeaders("pai_global_secret", { alias: "acme" });
    expect(h["x-pgai-org"]).toBe("acme");
    expect(h["x-pgai-include-hidden"]).toBe("true");
    expect(h["access-token"]).toBe("pai_global_secret");
  });

  test("numeric org selection works the same way", () => {
    const h = issueRequestHeaders("pai_global_secret", { id: 5225 });
    expect(h["x-pgai-org-id"]).toBe("5225");
    expect(h["x-pgai-include-hidden"]).toBe("true");
  });

  test("a per-org token still needs no selector, and keeps the opt-in", () => {
    const h = issueRequestHeaders("k");
    expect(h["x-pgai-org"]).toBeUndefined();
    expect(h["x-pgai-org-id"]).toBeUndefined();
    expect(h["x-pgai-include-hidden"]).toBe("true");
  });
});
