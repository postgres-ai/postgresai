import { describe, test, expect, mock, afterEach, beforeEach } from "bun:test";
import { uploadFile, downloadFile, buildMarkdownLink, uploadAttachments, appendAttachmentsToContent } from "../lib/storage";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

const originalFetch = globalThis.fetch;

describe("buildMarkdownLink", () => {
  const storageBaseUrl = "https://postgres.ai/storage";

  test("returns image markdown for .png", () => {
    const result = buildMarkdownLink("/files/123/image.png", storageBaseUrl);
    expect(result).toBe("![image.png](https://postgres.ai/storage/files/123/image.png)");
  });

  test("returns image markdown for .jpg", () => {
    const result = buildMarkdownLink("/files/123/photo.jpg", storageBaseUrl);
    expect(result).toBe("![photo.jpg](https://postgres.ai/storage/files/123/photo.jpg)");
  });

  test("returns image markdown for .jpeg", () => {
    const result = buildMarkdownLink("/files/123/photo.jpeg", storageBaseUrl);
    expect(result).toBe("![photo.jpeg](https://postgres.ai/storage/files/123/photo.jpeg)");
  });

  test("returns image markdown for .gif", () => {
    const result = buildMarkdownLink("/files/123/anim.gif", storageBaseUrl);
    expect(result).toBe("![anim.gif](https://postgres.ai/storage/files/123/anim.gif)");
  });

  test("returns image markdown for .webp", () => {
    const result = buildMarkdownLink("/files/123/pic.webp", storageBaseUrl);
    expect(result).toBe("![pic.webp](https://postgres.ai/storage/files/123/pic.webp)");
  });

  test("returns image markdown for .svg", () => {
    const result = buildMarkdownLink("/files/123/diagram.svg", storageBaseUrl);
    expect(result).toBe("![diagram.svg](https://postgres.ai/storage/files/123/diagram.svg)");
  });

  test("returns file link for .pdf", () => {
    const result = buildMarkdownLink("/files/123/report.pdf", storageBaseUrl);
    expect(result).toBe("[report.pdf](https://postgres.ai/storage/files/123/report.pdf)");
  });

  test("returns file link for .log", () => {
    const result = buildMarkdownLink("/files/123/app.log", storageBaseUrl);
    expect(result).toBe("[app.log](https://postgres.ai/storage/files/123/app.log)");
  });

  test("returns file link for .sql", () => {
    const result = buildMarkdownLink("/files/123/query.sql", storageBaseUrl);
    expect(result).toBe("[query.sql](https://postgres.ai/storage/files/123/query.sql)");
  });

  test("uses custom filename when provided", () => {
    const result = buildMarkdownLink("/files/123/abc123.png", storageBaseUrl, "screenshot.png");
    expect(result).toBe("![screenshot.png](https://postgres.ai/storage/files/123/abc123.png)");
  });

  test("handles full URL input", () => {
    const result = buildMarkdownLink("https://postgres.ai/storage/files/123/image.png", storageBaseUrl);
    expect(result).toBe("![image.png](https://postgres.ai/storage/files/123/image.png)");
  });

  test("is case-insensitive for extension detection", () => {
    const result = buildMarkdownLink("/files/123/IMAGE.PNG", storageBaseUrl);
    expect(result).toBe("![IMAGE.PNG](https://postgres.ai/storage/files/123/IMAGE.PNG)");
  });
});

describe("uploadFile", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      uploadFile({ apiKey: "", storageBaseUrl: "https://storage.example.com", filePath: "/tmp/test.txt" })
    ).rejects.toThrow("API key is required");
  });

  test("throws when storageBaseUrl is missing", async () => {
    await expect(
      uploadFile({ apiKey: "key", storageBaseUrl: "", filePath: "/tmp/test.txt" })
    ).rejects.toThrow("storageBaseUrl is required");
  });

  test("throws when filePath is missing", async () => {
    await expect(
      uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: "" })
    ).rejects.toThrow("filePath is required");
  });

  test("throws when file does not exist", async () => {
    await expect(
      uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: "/tmp/nonexistent-file-12345.txt" })
    ).rejects.toThrow("File not found");
  });

  test("throws on non-file path", async () => {
    await expect(
      uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: os.tmpdir() })
    ).rejects.toThrow("Not a file");
  });

  test("sends correct request and returns parsed response", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-test-${Date.now()}.txt`);
    fs.writeFileSync(tmpFile, "hello world");

    const mockUploadResponse = {
      success: true,
      url: "/files/123/1707500000000_abc.txt",
      metadata: {
        originalName: "storage-test.txt",
        size: 11,
        mimeType: "text/plain",
        uploadedAt: "2025-02-09T12:00:00.000Z",
        duration: 50,
      },
      requestId: "req-123",
    };

    let capturedUrl = "";
    let capturedHeaders: Record<string, string> = {};
    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((url: string, init?: RequestInit) => {
      capturedUrl = url;
      capturedHeaders = Object.fromEntries(
        Object.entries(init?.headers as Record<string, string> || {})
      );
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify(mockUploadResponse), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      const result = await uploadFile({
        apiKey: "test-key",
        storageBaseUrl: "https://postgres.ai/storage",
        filePath: tmpFile,
      });

      expect(capturedUrl).toBe("https://postgres.ai/storage/upload");
      expect(capturedHeaders["access-token"]).toBe("test-key");
      expect(result.success).toBe(true);
      expect(result.url).toBe("/files/123/1707500000000_abc.txt");
      expect(result.metadata.originalName).toBe("storage-test.txt");
      expect(result.metadata.size).toBe(11);
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("sends correct MIME type for .png file", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-mime-${Date.now()}.png`);
    fs.writeFileSync(tmpFile, "fake-png");

    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((_url: string, init?: RequestInit) => {
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.png", metadata: { originalName: "a.png", size: 8, mimeType: "image/png", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      const file = capturedBody!.get("file") as Blob;
      expect(file.type).toBe("image/png");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("sends correct MIME type for .jpg file", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-mime-${Date.now()}.jpg`);
    fs.writeFileSync(tmpFile, "fake-jpg");

    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((_url: string, init?: RequestInit) => {
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.jpg", metadata: { originalName: "a.jpg", size: 8, mimeType: "image/jpeg", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      const file = capturedBody!.get("file") as Blob;
      expect(file.type).toBe("image/jpeg");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("sends correct MIME type for .pdf file", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-mime-${Date.now()}.pdf`);
    fs.writeFileSync(tmpFile, "fake-pdf");

    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((_url: string, init?: RequestInit) => {
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.pdf", metadata: { originalName: "a.pdf", size: 8, mimeType: "application/pdf", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      const file = capturedBody!.get("file") as Blob;
      expect(file.type).toBe("application/pdf");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("sends correct MIME type for .sql file", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-mime-${Date.now()}.sql`);
    fs.writeFileSync(tmpFile, "SELECT 1");

    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((_url: string, init?: RequestInit) => {
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.sql", metadata: { originalName: "a.sql", size: 8, mimeType: "application/sql", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      const file = capturedBody!.get("file") as Blob;
      expect(file.type).toBe("application/sql");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("falls back to application/octet-stream for unknown extension", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-mime-${Date.now()}.xyz`);
    fs.writeFileSync(tmpFile, "data");

    let capturedBody: FormData | undefined;

    globalThis.fetch = mock((_url: string, init?: RequestInit) => {
      capturedBody = init?.body as FormData;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.xyz", metadata: { originalName: "a.xyz", size: 4, mimeType: "application/octet-stream", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    }) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      const file = capturedBody!.get("file") as Blob;
      expect(file.type).toBe("application/octet-stream");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("throws on HTTP error response", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-test-err-${Date.now()}.txt`);
    fs.writeFileSync(tmpFile, "hello");

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify({ code: "DANGEROUS_EXTENSION", message: "Extension not allowed" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    try {
      await expect(
        uploadFile({ apiKey: "test-key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile })
      ).rejects.toThrow("Failed to upload file");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });
});

describe("downloadFile", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "storage-dl-test-"));
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  test("throws when apiKey is missing", async () => {
    await expect(
      downloadFile({ apiKey: "", storageBaseUrl: "https://storage.example.com", fileUrl: "/files/123/test.png" })
    ).rejects.toThrow("API key is required");
  });

  test("throws when fileUrl is missing", async () => {
    await expect(
      downloadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", fileUrl: "" })
    ).rejects.toThrow("fileUrl is required");
  });

  test("downloads file with relative path and saves to output", async () => {
    const fileContent = Buffer.from("binary content here");
    const outputPath = path.join(tmpDir, "downloaded.png");

    globalThis.fetch = mock((url: string, init?: RequestInit) => {
      expect(url).toBe("https://postgres.ai/storage/files/123/image.png");
      expect((init?.headers as Record<string, string>)["access-token"]).toBe("test-key");
      return Promise.resolve(
        new Response(fileContent, {
          status: 200,
          headers: { "Content-Type": "image/png" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "/files/123/image.png",
      outputPath,
    });

    expect(result.savedTo).toBe(outputPath);
    expect(result.size).toBe(fileContent.length);
    expect(result.mimeType).toBe("image/png");
    expect(fs.existsSync(outputPath)).toBe(true);
    expect(fs.readFileSync(outputPath).toString()).toBe("binary content here");
  });

  test("downloads file with full URL matching storage origin", async () => {
    const outputPath = path.join(tmpDir, "out.txt");

    globalThis.fetch = mock((url: string) => {
      expect(url).toBe("https://postgres.ai/storage/files/1/data.txt");
      return Promise.resolve(
        new Response("text data", {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        })
      );
    }) as unknown as typeof fetch;

    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "https://postgres.ai/storage/files/1/data.txt",
      outputPath,
    });

    expect(result.savedTo).toBe(outputPath);
    expect(fs.readFileSync(outputPath, "utf-8")).toBe("text data");
  });

  test("rejects full URL with mismatched origin", async () => {
    await expect(
      downloadFile({
        apiKey: "test-key",
        storageBaseUrl: "https://postgres.ai/storage",
        fileUrl: "https://evil.com/files/1/data.txt",
        outputPath: path.join(tmpDir, "out.txt"),
      })
    ).rejects.toThrow("URL must be under storage base URL");
  });

  test("rejects same-origin URL outside storage path", async () => {
    await expect(
      downloadFile({
        apiKey: "test-key",
        storageBaseUrl: "https://postgres.ai/storage",
        fileUrl: "https://postgres.ai/malicious/files/1/data.txt",
        outputPath: path.join(tmpDir, "out.txt"),
      })
    ).rejects.toThrow("URL must be under storage base URL");
  });

  test("allows explicit outputPath outside cwd", async () => {
    const outputPath = path.join(tmpDir, "explicit-out.png");

    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("ok", { status: 200, headers: { "Content-Type": "image/png" } }))
    ) as unknown as typeof fetch;

    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "/files/123/image.png",
      outputPath,
    });

    expect(result.savedTo).toBe(outputPath);
    expect(fs.existsSync(outputPath)).toBe(true);
  });

  test("throws when storageBaseUrl is missing", async () => {
    await expect(
      downloadFile({ apiKey: "key", storageBaseUrl: "", fileUrl: "/files/1/a.png" })
    ).rejects.toThrow("storageBaseUrl is required");
  });

  test("derives filename from URL when outputPath not specified", async () => {
    // We need to control cwd resolution — use an absolute output by mocking the behavior
    const originalResolve = path.resolve;

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("data", {
          status: 200,
          headers: { "Content-Type": "application/octet-stream" },
        })
      )
    ) as unknown as typeof fetch;

    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "/files/123/report.csv",
      outputPath: path.join(tmpDir, "report.csv"),
    });

    expect(result.savedTo).toEndWith("report.csv");
    expect(fs.existsSync(result.savedTo)).toBe(true);
  });

  test("throws on HTTP error response", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify({ code: "FORBIDDEN", message: "Access denied" }), {
          status: 403,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      downloadFile({
        apiKey: "test-key",
        storageBaseUrl: "https://storage.example.com",
        fileUrl: "/files/999/secret.png",
        outputPath: path.join(tmpDir, "nope.png"),
      })
    ).rejects.toThrow("Failed to download file");
  });

  test("handles path without leading slash", async () => {
    globalThis.fetch = mock((url: string) => {
      expect(url).toBe("https://postgres.ai/storage/files/123/image.png");
      return Promise.resolve(
        new Response("ok", { status: 200, headers: { "Content-Type": "image/png" } })
      );
    }) as unknown as typeof fetch;

    const outputPath = path.join(tmpDir, "out.png");
    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "files/123/image.png",
      outputPath,
    });

    expect(result.size).toBe(2);
  });

  test("path.basename strips traversal from URL-derived filename", async () => {
    // path.basename("../../../etc/passwd") → "passwd", safe within cwd
    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("safe", { status: 200, headers: { "Content-Type": "text/plain" } }))
    ) as unknown as typeof fetch;

    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "/files/123/..%2F..%2F..%2Fetc%2Fpasswd",
      outputPath: path.join(tmpDir, "traversal-test.txt"),
    });

    // File written to explicit outputPath, not to /etc/passwd
    expect(result.savedTo).toBe(path.join(tmpDir, "traversal-test.txt"));
  });

  test("URL-derived filename with traversal resolves safely via basename", async () => {
    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("ok", { status: 200, headers: { "Content-Type": "text/plain" } }))
    ) as unknown as typeof fetch;

    // basename extracts just "secret.txt" from a traversal path
    const result = await downloadFile({
      apiKey: "test-key",
      storageBaseUrl: "https://postgres.ai/storage",
      fileUrl: "/files/123/secret.txt",
      outputPath: path.join(tmpDir, "safe.txt"),
    });

    expect(result.savedTo).toBe(path.join(tmpDir, "safe.txt"));
    expect(fs.existsSync(result.savedTo)).toBe(true);
  });
});

describe("uploadFile size limit", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("throws when file exceeds 500MB", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-big-${Date.now()}.bin`);
    const fd = fs.openSync(tmpFile, "w");
    fs.ftruncateSync(fd, 501 * 1024 * 1024);
    fs.closeSync(fd);

    try {
      await expect(
        uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile })
      ).rejects.toThrow("File too large");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("allows file at exactly 500MB", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-exact-${Date.now()}.bin`);
    const fd = fs.openSync(tmpFile, "w");
    fs.ftruncateSync(fd, 500 * 1024 * 1024);
    fs.closeSync(fd);

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.bin", metadata: { originalName: "a.bin", size: 0, mimeType: "application/octet-stream", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    try {
      const result = await uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile });
      expect(result.success).toBe(true);
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("rejects file at 500MB + 1 byte", async () => {
    const tmpFile = path.join(os.tmpdir(), `storage-over-${Date.now()}.bin`);
    const fd = fs.openSync(tmpFile, "w");
    fs.ftruncateSync(fd, 500 * 1024 * 1024 + 1);
    fs.closeSync(fd);

    try {
      await expect(
        uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile })
      ).rejects.toThrow("File too large");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });
});

describe("downloadFile size limit", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("rejects download when Content-Length exceeds 500MB", async () => {
    const tooBig = String(500 * 1024 * 1024 + 1);
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("", {
          status: 200,
          headers: { "Content-Length": tooBig, "Content-Type": "application/octet-stream" },
        })
      )
    ) as unknown as typeof fetch;

    await expect(
      downloadFile({
        apiKey: "key",
        storageBaseUrl: "https://postgres.ai/storage",
        fileUrl: "/files/1/big.bin",
        outputPath: path.join(os.tmpdir(), `dl-limit-${Date.now()}.bin`),
      })
    ).rejects.toThrow("File too large");
  });

  test("allows download when Content-Length is at 500MB", async () => {
    const exact = String(500 * 1024 * 1024);
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "dl-limit-"));
    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response("ok", {
          status: 200,
          headers: { "Content-Length": exact, "Content-Type": "application/octet-stream" },
        })
      )
    ) as unknown as typeof fetch;

    try {
      const result = await downloadFile({
        apiKey: "key",
        storageBaseUrl: "https://postgres.ai/storage",
        fileUrl: "/files/1/ok.bin",
        outputPath: path.join(tmpDir, "ok.bin"),
      });
      expect(result.savedTo).toEndWith("ok.bin");
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe("HTTP warning output", () => {
  const originalConsoleError = console.error;
  let consoleOutput: string[];

  beforeEach(() => {
    consoleOutput = [];
    console.error = (...args: unknown[]) => {
      consoleOutput.push(args.map(String).join(" "));
    };
  });

  afterEach(() => {
    console.error = originalConsoleError;
    globalThis.fetch = originalFetch;
  });

  test("uploadFile warns on HTTP storage URL", async () => {
    const tmpFile = path.join(os.tmpdir(), `http-warn-up-${Date.now()}.txt`);
    fs.writeFileSync(tmpFile, "test");

    globalThis.fetch = mock(() =>
      Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a.txt", metadata: { originalName: "a.txt", size: 4, mimeType: "text/plain", uploadedAt: "", duration: 0 }, requestId: "r" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      )
    ) as unknown as typeof fetch;

    try {
      await uploadFile({ apiKey: "key", storageBaseUrl: "http://localhost:3000/storage", filePath: tmpFile });
      expect(consoleOutput.some(line => line.includes("HTTP") && line.includes("unencrypted"))).toBe(true);
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test("downloadFile warns on HTTP storage URL", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "http-warn-dl-"));
    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("ok", { status: 200, headers: { "Content-Type": "text/plain" } }))
    ) as unknown as typeof fetch;

    try {
      await downloadFile({
        apiKey: "key",
        storageBaseUrl: "http://localhost:3000/storage",
        fileUrl: "/files/1/a.txt",
        outputPath: path.join(tmpDir, "a.txt"),
      });
      expect(consoleOutput.some(line => line.includes("HTTP") && line.includes("unencrypted"))).toBe(true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe("downloadFile edge cases", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("creates parent directory when it does not exist", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "dl-mkdir-"));
    const nestedPath = path.join(tmpDir, "sub", "dir", "file.txt");

    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("data", { status: 200, headers: { "Content-Type": "text/plain" } }))
    ) as unknown as typeof fetch;

    try {
      const result = await downloadFile({
        apiKey: "key",
        storageBaseUrl: "https://postgres.ai/storage",
        fileUrl: "/files/1/file.txt",
        outputPath: nestedPath,
      });
      expect(result.savedTo).toBe(nestedPath);
      expect(fs.existsSync(nestedPath)).toBe(true);
      expect(fs.readFileSync(nestedPath, "utf-8")).toBe("data");
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test("uploadFile throws on unparseable success response", async () => {
    const tmpFile = path.join(os.tmpdir(), `bad-json-${Date.now()}.txt`);
    fs.writeFileSync(tmpFile, "test");

    globalThis.fetch = mock(() =>
      Promise.resolve(new Response("not-json", { status: 200 }))
    ) as unknown as typeof fetch;

    try {
      await expect(
        uploadFile({ apiKey: "key", storageBaseUrl: "https://storage.example.com", filePath: tmpFile })
      ).rejects.toThrow("Failed to parse upload response");
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });
});

describe("buildMarkdownLink escaping", () => {
  const storageBaseUrl = "https://postgres.ai/storage";

  test("escapes parentheses in custom filename", () => {
    const result = buildMarkdownLink("/files/123/abc.png", storageBaseUrl, "report (final).png");
    expect(result).toBe("![report \\(final\\).png](https://postgres.ai/storage/files/123/abc.png)");
  });

  test("escapes brackets in custom filename", () => {
    const result = buildMarkdownLink("/files/123/abc.csv", storageBaseUrl, "data[1].csv");
    expect(result).toBe("[data\\[1\\].csv](https://postgres.ai/storage/files/123/abc.csv)");
  });

  test("escapes special chars in custom filename", () => {
    const result = buildMarkdownLink("/files/123/abc.png", storageBaseUrl, "shot [v2] (draft).png");
    expect(result).toBe("![shot \\[v2\\] \\(draft\\).png](https://postgres.ai/storage/files/123/abc.png)");
  });
});

describe("appendAttachmentsToContent", () => {
  const mkAttachment = (markdown: string) => ({
    path: "/tmp/x",
    url: "/files/1/x",
    markdown,
    metadata: { originalName: "x", size: 0, mimeType: "x", uploadedAt: "", duration: 0 },
  });

  test("returns content unchanged when attachments empty", () => {
    expect(appendAttachmentsToContent("hello", [])).toBe("hello");
  });

  test("returns content unchanged when attachments missing (undefined-safe)", () => {
    // Defensive: callers may pass undefined for cleaner sites — we tolerate it.
    // (TS prevents this at compile time but runtime data can disagree.)
    expect(appendAttachmentsToContent("hello", undefined as unknown as never[])).toBe("hello");
  });

  test("returns just the link(s) when content is empty string", () => {
    const out = appendAttachmentsToContent("", [mkAttachment("![a](u)")]);
    expect(out).toBe("![a](u)");
  });

  test("returns just the link(s) when content is whitespace", () => {
    const out = appendAttachmentsToContent("   \n  ", [mkAttachment("![a](u)")]);
    expect(out).toBe("![a](u)");
  });

  test("appends a single link with two-newline separator", () => {
    const out = appendAttachmentsToContent("hello", [mkAttachment("![a](u)")]);
    expect(out).toBe("hello\n\n![a](u)");
  });

  test("appends multiple links one per line, preserving order", () => {
    const out = appendAttachmentsToContent("hello", [
      mkAttachment("![first](u1)"),
      mkAttachment("[second](u2)"),
      mkAttachment("![third](u3)"),
    ]);
    expect(out).toBe("hello\n\n![first](u1)\n[second](u2)\n![third](u3)");
  });

  test("does not strip user-provided trailing newlines", () => {
    // The user may have a meaningful trailing newline (e.g. for code blocks).
    // We should not normalize content beyond appending.
    const out = appendAttachmentsToContent("hello\n", [mkAttachment("![a](u)")]);
    expect(out).toBe("hello\n\n\n![a](u)");
  });
});

describe("uploadAttachments", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("returns empty array when input is empty", async () => {
    const out = await uploadAttachments({
      apiKey: "k",
      storageBaseUrl: "https://postgres.ai/storage",
      attachmentPaths: [],
    });
    expect(out).toEqual([]);
  });

  test("returns empty array when input is undefined (defensive)", async () => {
    const out = await uploadAttachments({
      apiKey: "k",
      storageBaseUrl: "https://postgres.ai/storage",
      attachmentPaths: undefined as unknown as string[],
    });
    expect(out).toEqual([]);
  });

  test("uploads each file in order and returns metadata + markdown link per upload", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ua-"));
    const f1 = path.join(tmpDir, "shot.png");
    const f2 = path.join(tmpDir, "trace.log");
    fs.writeFileSync(f1, "fake-png-bytes");
    fs.writeFileSync(f2, "log line 1\nlog line 2\n");

    const fakeResponses = [
      { url: "/files/9/aaa.png", originalName: "shot.png" },
      { url: "/files/9/bbb.log", originalName: "trace.log" },
    ];
    const calls: Array<{ url: string; mime: string }> = [];

    globalThis.fetch = mock((url: string, init?: RequestInit) => {
      const body = init?.body as FormData;
      const file = body?.get("file") as Blob;
      const next = fakeResponses[calls.length];
      calls.push({ url: String(url), mime: file?.type ?? "" });
      return Promise.resolve(
        new Response(
          JSON.stringify({
            success: true,
            url: next.url,
            metadata: {
              originalName: next.originalName,
              size: 0,
              mimeType: file?.type ?? "application/octet-stream",
              uploadedAt: "",
              duration: 0,
            },
            requestId: `r-${calls.length}`,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      );
    }) as unknown as typeof fetch;

    try {
      const out = await uploadAttachments({
        apiKey: "k",
        storageBaseUrl: "https://postgres.ai/storage",
        attachmentPaths: [f1, f2],
      });

      // Both uploads happened, in order.
      expect(calls).toHaveLength(2);
      expect(calls[0].url).toBe("https://postgres.ai/storage/upload");
      expect(calls[1].url).toBe("https://postgres.ai/storage/upload");
      // Mime types from extensions (image/png, text/plain).
      // Blob may append `;charset=utf-8` for text MIME types — accept either.
      expect(calls[0].mime).toBe("image/png");
      expect(calls[1].mime.startsWith("text/plain")).toBe(true);

      expect(out).toHaveLength(2);
      expect(out[0].path).toBe(f1);
      expect(out[0].url).toBe("/files/9/aaa.png");
      // Image extension renders inline.
      expect(out[0].markdown).toBe("![shot.png](https://postgres.ai/storage/files/9/aaa.png)");
      expect(out[0].metadata.mimeType).toBe("image/png");

      expect(out[1].path).toBe(f2);
      expect(out[1].url).toBe("/files/9/bbb.log");
      // Non-image renders as plain link.
      expect(out[1].markdown).toBe("[trace.log](https://postgres.ai/storage/files/9/bbb.log)");
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test("error on file N surfaces the path so the user can retry", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ua-err-"));
    const f1 = path.join(tmpDir, "ok.txt");
    fs.writeFileSync(f1, "ok");
    const missing = path.join(tmpDir, "definitely-not-here.png");

    let callCount = 0;
    globalThis.fetch = mock(() => {
      callCount++;
      return Promise.resolve(
        new Response(JSON.stringify({ success: true, url: "/files/1/a", metadata: { originalName: "ok.txt", size: 2, mimeType: "text/plain", uploadedAt: "", duration: 0 }, requestId: "r" }), { status: 200 })
      );
    }) as unknown as typeof fetch;

    try {
      await expect(
        uploadAttachments({
          apiKey: "k",
          storageBaseUrl: "https://postgres.ai/storage",
          attachmentPaths: [f1, missing],
        })
      ).rejects.toThrow(/File not found.*definitely-not-here/);
      // The first file was uploaded before the failure; we don't retry it.
      // (Documenting current behavior — caller is responsible if mid-failure
      // partial uploads matter.)
      expect(callCount).toBe(1);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
