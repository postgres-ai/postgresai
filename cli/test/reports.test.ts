import { describe, test, expect, mock, afterEach, spyOn } from "bun:test";
import {
  fetchReports,
  fetchAllReports,
  fetchReportFiles,
  fetchReportFileData,
  renderMarkdownForTerminal,
  parseFlexibleDate,
} from "../lib/reports";

const originalFetch = globalThis.fetch;

describe("fetchReports", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      fetchReports({ apiKey: "", apiBaseUrl: "https://api.example.com" })
    ).rejects.toThrow("API key is required");
  });

  test("constructs correct URL with no filters", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.pathname).toBe("/checkup_reports");
    expect(url.searchParams.get("order")).toBe("id.desc");
    expect(url.searchParams.get("limit")).toBe("20");
    expect(url.searchParams.has("project_id")).toBe(false);
    expect(url.searchParams.has("status")).toBe(false);
  });

  test("constructs correct URL with all filters", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      projectId: 5,
      status: "completed",
      limit: 10,
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.get("project_id")).toBe("eq.5");
    expect(url.searchParams.get("status")).toBe("eq.completed");
    expect(url.searchParams.get("limit")).toBe("10");
  });

  test("sends correct headers", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
    });

    const headers = capturedRequest!.options.headers as Record<string, string>;
    expect(headers["access-token"]).toBe("test-key");
    expect(headers["Prefer"]).toBe("return=representation");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers["Connection"]).toBe("close");
  });

  test("returns parsed response array", async () => {
    const mockData = [
      {
        id: 1,
        org_id: 1,
        org_name: "TestOrg",
        project_id: 10,
        project_name: "prod-db",
        created_at: "2025-01-01T00:00:00Z",
        created_formatted: "2025-01-01 00:00:00",
        epoch: 1735689600,
        status: "completed",
      },
    ];

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockData), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
    });

    expect(result).toEqual(mockData);
    expect(result[0].id).toBe(1);
    expect(result[0].status).toBe("completed");
  });

  test("throws formatted error on non-200 response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Unauthorized"}', {
          status: 401,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReports({
        apiKey: "invalid-key",
        apiBaseUrl: "https://api.example.com",
      })
    ).rejects.toThrow(/Failed to fetch reports/);
  });

  test("sets created_at filter when beforeDate is provided", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      beforeDate: "2025-01-15T00:00:00.000Z",
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.get("created_at")).toBe("lt.2025-01-15T00:00:00.000Z");
    expect(url.searchParams.get("order")).toBe("id.desc");
  });

  test("sets id=lt.beforeId when beforeId is provided (internal pagination)", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      beforeId: 50,
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.get("id")).toBe("lt.50");
  });

  test("logs debug info when debug is true", async () => {
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      debug: true,
    });

    const calls = consoleSpy.mock.calls.map((c) => c[0]);
    expect(calls.some((c: string) => c.includes("Debug: Resolved API base URL"))).toBe(true);
    expect(calls.some((c: string) => c.includes("Debug: GET URL"))).toBe(true);
    expect(calls.some((c: string) => c.includes("Debug: Request headers"))).toBe(true);
    expect(calls.some((c: string) => c.includes("Debug: Response status"))).toBe(true);

    consoleSpy.mockRestore();
  });

  test("throws on invalid JSON response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("not valid json", {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReports({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
      })
    ).rejects.toThrow(/Failed to parse reports response/);
  });

  test("does not set id or created_at params when no before options provided", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
    });

    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.has("id")).toBe(false);
    expect(url.searchParams.has("created_at")).toBe(false);
  });
});

describe("fetchAllReports", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("fetches all pages until empty response", async () => {
    const page1 = [
      { id: 100, org_id: 1, org_name: "O", project_id: 1, project_name: "P", created_at: "", created_formatted: "", epoch: 0, status: "completed" },
      { id: 90, org_id: 1, org_name: "O", project_id: 1, project_name: "P", created_at: "", created_formatted: "", epoch: 0, status: "completed" },
    ];
    const page2 = [
      { id: 80, org_id: 1, org_name: "O", project_id: 1, project_name: "P", created_at: "", created_formatted: "", epoch: 0, status: "completed" },
    ];

    let callCount = 0;
    globalThis.fetch = mock((url: string) => {
      callCount++;
      const u = new URL(url);
      const idParam = u.searchParams.get("id");
      let data;
      if (!idParam) {
        data = page1;
      } else if (idParam === "lt.90") {
        data = page2;
      } else {
        data = [];
      }
      return Promise.resolve(
        new Response(JSON.stringify(data), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await fetchAllReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      limit: 2,
    });

    expect(result.length).toBe(3);
    expect(result[0].id).toBe(100);
    expect(result[1].id).toBe(90);
    expect(result[2].id).toBe(80);
  });

  test("returns empty array when no reports exist", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await fetchAllReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
    });

    expect(result).toEqual([]);
  });

  test("stops at MAX_ALL_REPORTS cap (10000)", async () => {
    const warnSpy = spyOn(console, "warn").mockImplementation(() => {});
    const pageSize = 500;
    let callCount = 0;

    globalThis.fetch = mock((url: string) => {
      callCount++;
      const u = new URL(url);
      const idParam = u.searchParams.get("id");
      // Generate a full page each time with descending IDs
      const startId = idParam ? parseInt(idParam.replace("lt.", "")) - 1 : 100000;
      const page = Array.from({ length: pageSize }, (_, i) => ({
        id: startId - i,
        org_id: 1,
        org_name: "O",
        project_id: 1,
        project_name: "P",
        created_at: "",
        created_formatted: "",
        epoch: 0,
        status: "completed",
      }));
      return Promise.resolve(
        new Response(JSON.stringify(page), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await fetchAllReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      limit: pageSize,
    });

    // Should stop at 10000 (MAX_ALL_REPORTS)
    expect(result.length).toBe(10000);
    // Should have logged a warning
    const warnCalls = warnSpy.mock.calls.map((c) => c[0]);
    expect(warnCalls.some((c: string) => c.includes("maximum of 10000"))).toBe(true);
    // Should have made exactly 20 calls (10000 / 500)
    expect(callCount).toBe(20);

    warnSpy.mockRestore();
  });

  test("stops when page has fewer items than limit", async () => {
    const page = [
      { id: 50, org_id: 1, org_name: "O", project_id: 1, project_name: "P", created_at: "", created_formatted: "", epoch: 0, status: "completed" },
    ];

    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      return Promise.resolve(
        new Response(JSON.stringify(page), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await fetchAllReports({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      limit: 10,
    });

    expect(result.length).toBe(1);
    expect(callCount).toBe(1);
  });
});

describe("fetchReportFiles", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      fetchReportFiles({
        apiKey: "",
        apiBaseUrl: "https://api.example.com",
        reportId: 1,
      })
    ).rejects.toThrow("API key is required");
  });

  test("throws when neither reportId nor checkId is provided", async () => {
    await expect(
      fetchReportFiles({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
      })
    ).rejects.toThrow("Either reportId or checkId is required");
  });

  test("works with only checkId (no reportId)", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFiles({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      checkId: "H002",
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.pathname).toBe("/checkup_report_files");
    expect(url.searchParams.has("checkup_report_id")).toBe(false);
    expect(url.searchParams.get("check_id")).toBe("eq.H002");
    expect(url.searchParams.get("order")).toBe("id.asc");
  });

  test("constructs URL with required checkup_report_id", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFiles({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 42,
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.pathname).toBe("/checkup_report_files");
    expect(url.searchParams.get("checkup_report_id")).toBe("eq.42");
    expect(url.searchParams.get("order")).toBe("id.asc");
    expect(url.searchParams.has("type")).toBe(false);
    expect(url.searchParams.has("check_id")).toBe(false);
  });

  test("applies optional type and check_id filters", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFiles({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 42,
      type: "md",
      checkId: "H002",
    });

    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.get("type")).toBe("eq.md");
    expect(url.searchParams.get("check_id")).toBe("eq.H002");
  });

  test("returns parsed response array", async () => {
    const mockData = [
      {
        id: 100,
        checkup_report_id: 42,
        filename: "H002.json",
        check_id: "H002",
        type: "json",
        created_at: "2025-01-01T00:00:00Z",
        created_formatted: "2025-01-01 00:00:00",
        project_id: 10,
        project_name: "prod-db",
      },
    ];

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockData), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await fetchReportFiles({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 42,
    });

    expect(result).toEqual(mockData);
    expect(result[0].filename).toBe("H002.json");
  });

  test("throws on error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Not found"}', {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReportFiles({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        reportId: 999,
      })
    ).rejects.toThrow(/Failed to fetch report files/);
  });

  test("logs debug info when debug is true", async () => {
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await fetchReportFiles({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 1,
      debug: true,
    });

    const calls = consoleSpy.mock.calls.map((c) => c[0]);
    expect(calls.some((c: string) => c.includes("Debug: Resolved API base URL"))).toBe(true);
    expect(calls.some((c: string) => c.includes("Debug: Response status"))).toBe(true);

    consoleSpy.mockRestore();
  });

  test("throws on invalid JSON response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("not valid json", {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReportFiles({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        reportId: 1,
      })
    ).rejects.toThrow(/Failed to parse report files response/);
  });
});

describe("fetchReportFileData", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      fetchReportFileData({
        apiKey: "",
        apiBaseUrl: "https://api.example.com",
        reportId: 1,
      })
    ).rejects.toThrow("API key is required");
  });

  test("throws when neither reportId nor checkId is provided", async () => {
    await expect(
      fetchReportFileData({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
      })
    ).rejects.toThrow("Either reportId or checkId is required");
  });

  test("works with only checkId (no reportId)", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFileData({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      checkId: "H002",
    });

    expect(capturedRequest).not.toBeNull();
    const url = new URL(capturedRequest!.url);
    expect(url.pathname).toBe("/checkup_report_file_data");
    expect(url.searchParams.has("checkup_report_id")).toBe(false);
    expect(url.searchParams.get("check_id")).toBe("eq.H002");
    expect(url.searchParams.get("order")).toBe("id.asc");
  });

  test("constructs correct URL with reportId", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFileData({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 123,
    });

    const url = new URL(capturedRequest!.url);
    expect(url.pathname).toBe("/checkup_report_file_data");
    expect(url.searchParams.get("checkup_report_id")).toBe("eq.123");
    expect(url.searchParams.get("order")).toBe("id.asc");
  });

  test("applies optional filters", async () => {
    let capturedRequest: { url: string; options: RequestInit } | null = null;

    globalThis.fetch = mock((url: string, options: RequestInit) => {
      capturedRequest = { url, options };
      return Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    await fetchReportFileData({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 123,
      type: "json",
      checkId: "H001",
    });

    const url = new URL(capturedRequest!.url);
    expect(url.searchParams.get("type")).toBe("eq.json");
    expect(url.searchParams.get("check_id")).toBe("eq.H001");
  });

  test("returns parsed response with data field", async () => {
    const mockData = [
      {
        id: 200,
        checkup_report_id: 123,
        filename: "H002.md",
        check_id: "H002",
        type: "md",
        created_at: "2025-01-01T00:00:00Z",
        created_formatted: "2025-01-01 00:00:00",
        project_id: 10,
        project_name: "prod-db",
        data: "# H002 Report\n\nUnused indexes found.\n",
      },
      {
        id: 201,
        checkup_report_id: 123,
        filename: "H002.json",
        check_id: "H002",
        type: "json",
        created_at: "2025-01-01T00:00:00Z",
        created_formatted: "2025-01-01 00:00:00",
        project_id: 10,
        project_name: "prod-db",
        data: '{"unused_indexes": []}',
      },
    ];

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify(mockData), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await fetchReportFileData({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 123,
    });

    expect(result.length).toBe(2);
    expect(result[0].data).toContain("# H002 Report");
    expect(result[1].data).toBe('{"unused_indexes": []}');
  });

  test("throws on error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response('{"message": "Unauthorized"}', {
          status: 401,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReportFileData({
        apiKey: "invalid-key",
        apiBaseUrl: "https://api.example.com",
        reportId: 123,
      })
    ).rejects.toThrow(/Failed to fetch report file data/);
  });

  test("logs debug info when debug is true", async () => {
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await fetchReportFileData({
      apiKey: "test-key",
      apiBaseUrl: "https://api.example.com",
      reportId: 1,
      debug: true,
    });

    const calls = consoleSpy.mock.calls.map((c) => c[0]);
    expect(calls.some((c: string) => c.includes("Debug: Resolved API base URL"))).toBe(true);
    expect(calls.some((c: string) => c.includes("Debug: Response status"))).toBe(true);

    consoleSpy.mockRestore();
  });

  test("throws on invalid JSON response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("not valid json", {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      fetchReportFileData({
        apiKey: "test-key",
        apiBaseUrl: "https://api.example.com",
        reportId: 1,
      })
    ).rejects.toThrow(/Failed to parse report file data response/);
  });
});

describe("renderMarkdownForTerminal", () => {
  test("returns empty string for empty input", () => {
    expect(renderMarkdownForTerminal("")).toBe("");
  });

  test("passes plain text through", () => {
    const result = renderMarkdownForTerminal("Just plain text");
    expect(result).toContain("Just plain text");
  });

  test("renders # heading with bold+underline", () => {
    const result = renderMarkdownForTerminal("# Hello World");
    expect(result).toContain("\x1b[");
    expect(result).toContain("Hello World");
  });

  test("renders ## heading with bold", () => {
    const result = renderMarkdownForTerminal("## Section Title");
    expect(result).toContain("\x1b[1m");
    expect(result).toContain("Section Title");
  });

  test("renders **bold** text", () => {
    const result = renderMarkdownForTerminal("This is **bold** text");
    expect(result).toContain("\x1b[1m");
    expect(result).toContain("bold");
  });

  test("renders `inline code`", () => {
    const result = renderMarkdownForTerminal("Run `SELECT 1`");
    expect(result).toContain("SELECT 1");
  });

  test("renders code blocks", () => {
    const input = "```\nSELECT 1;\nSELECT 2;\n```";
    const result = renderMarkdownForTerminal(input);
    expect(result).toContain("SELECT 1;");
    expect(result).toContain("SELECT 2;");
  });

  test("renders horizontal rules", () => {
    const result = renderMarkdownForTerminal("---");
    expect(result.replace(/\x1b\[[0-9;]*m/g, "").trim().length).toBeGreaterThan(3);
  });

  test("renders bullet lists", () => {
    const result = renderMarkdownForTerminal("- item one\n- item two");
    expect(result).toContain("item one");
    expect(result).toContain("item two");
  });

  test("does not italicize underscores inside identifiers", () => {
    const result = renderMarkdownForTerminal("goodvibes_local_monitoring_dev");
    // Strip ANSI codes and check the text is unchanged
    const stripped = result.replace(/\x1b\[[0-9;]*m/g, "");
    expect(stripped).toBe("goodvibes_local_monitoring_dev");
    // Must NOT contain italic ANSI code
    expect(result).not.toContain("\x1b[3m");
  });

  test("renders _word_ as italic at word boundaries", () => {
    const result = renderMarkdownForTerminal("This is _unknown_ value");
    expect(result).toContain("\x1b[3m");
    expect(result).toContain("unknown");
  });

  test("renders *italic* with single asterisks", () => {
    const result = renderMarkdownForTerminal("This is *italic* text");
    expect(result).toContain("\x1b[3m");
    expect(result).toContain("italic");
  });

  test("renders __bold__ with double underscores", () => {
    const result = renderMarkdownForTerminal("This is __bold__ text");
    expect(result).toContain("\x1b[1m");
    expect(result).toContain("bold");
  });
});

describe("parseFlexibleDate", () => {
  test("parses YYYY-MM-DD", () => {
    expect(parseFlexibleDate("2025-01-15")).toBe("2025-01-15T00:00:00.000Z");
  });

  test("parses YYYY-MM-DDTHH:mm:ss", () => {
    expect(parseFlexibleDate("2025-01-15T10:30:00")).toBe("2025-01-15T10:30:00.000Z");
  });

  test("parses YYYY-MM-DD HH:mm:ss", () => {
    expect(parseFlexibleDate("2025-01-15 10:30:00")).toBe("2025-01-15T10:30:00.000Z");
  });

  test("parses YYYY-MM-DD HH:mm", () => {
    expect(parseFlexibleDate("2025-01-15 10:30")).toBe("2025-01-15T10:30:00.000Z");
  });

  test("parses DD.MM.YYYY", () => {
    expect(parseFlexibleDate("15.01.2025")).toBe("2025-01-15T00:00:00.000Z");
  });

  test("parses DD.MM.YYYY HH:mm", () => {
    expect(parseFlexibleDate("15.01.2025 10:30")).toBe("2025-01-15T10:30:00.000Z");
  });

  test("parses DD.MM.YYYY HH:mm:ss", () => {
    expect(parseFlexibleDate("15.01.2025 10:30:45")).toBe("2025-01-15T10:30:45.000Z");
  });

  test("parses D.M.YYYY (single-digit day/month)", () => {
    expect(parseFlexibleDate("5.1.2025")).toBe("2025-01-05T00:00:00.000Z");
  });

  test("trims whitespace", () => {
    expect(parseFlexibleDate("  2025-01-15  ")).toBe("2025-01-15T00:00:00.000Z");
  });

  test("throws on unrecognized format", () => {
    expect(() => parseFlexibleDate("Jan 15, 2025")).toThrow(/Unrecognized date format/);
  });

  test("throws on invalid date values", () => {
    expect(() => parseFlexibleDate("2025-13-45")).toThrow(/Invalid date/);
  });

  test("throws on invalid DD.MM.YYYY values", () => {
    expect(() => parseFlexibleDate("45.13.2025")).toThrow(/Invalid date/);
  });
});
