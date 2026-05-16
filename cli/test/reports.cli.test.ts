import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { mkdtempSync, existsSync, readFileSync } from "fs";
import { tmpdir } from "os";

function runCli(args: string[], env: Record<string, string> = {}) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin =
    typeof process.execPath === "string" && process.execPath.length > 0
      ? process.execPath
      : "bun";
  const result = Bun.spawnSync([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
  });
  return {
    status: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

async function runCliAsync(
  args: string[],
  env: Record<string, string> = {}
) {
  const cliPath = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");
  const bunBin =
    typeof process.execPath === "string" && process.execPath.length > 0
      ? process.execPath
      : "bun";
  const proc = Bun.spawn([bunBin, cliPath, ...args], {
    env: { ...process.env, ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [status, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { status, stdout, stderr };
}

function isolatedEnv(extra: Record<string, string> = {}) {
  const cfgHome = mkdtempSync(resolve(tmpdir(), "postgresai-cli-test-"));
  return {
    XDG_CONFIG_HOME: cfgHome,
    HOME: cfgHome,
    ...extra,
  };
}

async function startFakeApi() {
  const requests: Array<{
    method: string;
    pathname: string;
    searchParams: Record<string, string>;
    headers: Record<string, string>;
  }> = [];

  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const headers: Record<string, string> = {};
      for (const [k, v] of req.headers.entries()) headers[k.toLowerCase()] = v;
      const searchParams: Record<string, string> = {};
      for (const [k, v] of url.searchParams.entries()) searchParams[k] = v;

      requests.push({
        method: req.method,
        pathname: url.pathname,
        searchParams,
        headers,
      });

      // GET /checkup_reports
      if (
        req.method === "GET" &&
        url.pathname.endsWith("/checkup_reports")
      ) {
        return new Response(
          JSON.stringify([
            {
              id: 1,
              org_id: 1,
              org_name: "TestOrg",
              project_id: 10,
              project_name: "TestProj",
              created_at: "2025-01-01T00:00:00Z",
              created_formatted: "2025-01-01 00:00:00",
              epoch: 1735689600,
              status: "completed",
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // GET /checkup_report_files
      if (
        req.method === "GET" &&
        url.pathname.endsWith("/checkup_report_files")
      ) {
        return new Response(
          JSON.stringify([
            {
              id: 100,
              checkup_report_id: 1,
              filename: "H002.json",
              check_id: "H002",
              type: "json",
              created_at: "2025-01-01T00:00:00Z",
              created_formatted: "2025-01-01 00:00:00",
              project_id: 10,
              project_name: "TestProj",
            },
            {
              id: 101,
              checkup_report_id: 1,
              filename: "H002.md",
              check_id: "H002",
              type: "md",
              created_at: "2025-01-01T00:00:00Z",
              created_formatted: "2025-01-01 00:00:00",
              project_id: 10,
              project_name: "TestProj",
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // GET /checkup_report_file_data
      if (
        req.method === "GET" &&
        url.pathname.endsWith("/checkup_report_file_data")
      ) {
        return new Response(
          JSON.stringify([
            {
              id: 100,
              checkup_report_id: 1,
              filename: "H002.md",
              check_id: "H002",
              type: "md",
              created_at: "2025-01-01T00:00:00Z",
              created_formatted: "2025-01-01 00:00:00",
              project_id: 10,
              project_name: "TestProj",
              data: "# H002 Report\n\nUnused indexes found.\n",
            },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      return new Response("not found", { status: 404 });
    },
  });

  return {
    baseUrl: `http://${server.hostname}:${server.port}/api/general`,
    requests,
    stop: () => server.stop(true),
  };
}

describe("CLI reports command group", () => {
  test("reports help exposes list, files, data subcommands", () => {
    const r = runCli(["reports", "--help"], isolatedEnv());
    expect(r.status).toBe(0);
    const out = `${r.stdout}\n${r.stderr}`;
    expect(out).toContain("list");
    expect(out).toContain("files");
    expect(out).toContain("data");
  });

  test("reports list fails fast when API key is missing", () => {
    const r = runCli(["reports", "list"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("reports files fails fast when API key is missing", () => {
    const r = runCli(["reports", "files", "1"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("reports files fails when reportId is not a number", () => {
    const r = runCli(
      ["reports", "files", "abc"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("reportId must be a number");
  });

  test("reports files fails when neither reportId nor --check-id is provided", () => {
    const r = runCli(
      ["reports", "files"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("Either reportId or --check-id is required");
  });

  test("reports data fails fast when API key is missing", () => {
    const r = runCli(["reports", "data", "1"], isolatedEnv());
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("API key is required");
  });

  test("reports data fails when reportId is not a number", () => {
    const r = runCli(
      ["reports", "data", "abc"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("reportId must be a number");
  });

  test("reports data fails when neither reportId nor --check-id is provided", () => {
    const r = runCli(
      ["reports", "data"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("Either reportId or --check-id is required");
  });

  test("reports list succeeds against a fake API", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "list"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(Array.isArray(out)).toBe(true);
      expect(out[0].id).toBe(1);
      expect(out[0].status).toBe("completed");

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.headers["access-token"]).toBe("test-key");
    } finally {
      api.stop();
    }
  });

  test("reports list passes correct filters to API", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        [
          "reports",
          "list",
          "--project-id",
          "10",
          "--status",
          "completed",
          "--limit",
          "5",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.project_id).toBe("eq.10");
      expect(req!.searchParams.status).toBe("eq.completed");
      expect(req!.searchParams.limit).toBe("5");
    } finally {
      api.stop();
    }
  });

  test("reports list --limit caps at 100", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--limit", "200"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.limit).toBe("100");
    } finally {
      api.stop();
    }
  });

  test("reports list --limit below cap passes through", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--limit", "50"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.limit).toBe("50");
    } finally {
      api.stop();
    }
  });

  test("reports list --limit with invalid value falls back to default", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--limit", "abc"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.limit).toBe("20");
    } finally {
      api.stop();
    }
  });

  test("reports list --limit with negative value clamps to 1", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--limit", "-5"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.limit).toBe("1");
    } finally {
      api.stop();
    }
  });

  test("reports files succeeds against a fake API", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "files", "1"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(Array.isArray(out)).toBe(true);
      expect(out[0].filename).toBe("H002.json");

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_files")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.checkup_report_id).toBe("eq.1");
    } finally {
      api.stop();
    }
  });

  test("reports files passes type and check-id filters", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "files", "1", "--type", "md", "--check-id", "H002"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_files")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.type).toBe("eq.md");
      expect(req!.searchParams.check_id).toBe("eq.H002");
    } finally {
      api.stop();
    }
  });

  test("reports data outputs raw markdown by default", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "data", "1"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);
      expect(r.stdout).toContain("# H002 Report");
      expect(r.stdout).toContain("Unused indexes found.");
    } finally {
      api.stop();
    }
  });

  test("reports data succeeds against a fake API with --json", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "data", "1", "--json"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(Array.isArray(out)).toBe(true);
      expect(out[0].data).toContain("# H002 Report");
    } finally {
      api.stop();
    }
  });

  test("reports files succeeds with only --check-id (no reportId)", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "files", "--check-id", "H002"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_files")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.check_id).toBe("eq.H002");
      expect(req!.searchParams.checkup_report_id).toBeUndefined();
    } finally {
      api.stop();
    }
  });

  test("reports data succeeds with only --check-id (no reportId)", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "data", "--check-id", "H002", "--json"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_file_data")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.check_id).toBe("eq.H002");
      expect(req!.searchParams.checkup_report_id).toBeUndefined();
    } finally {
      api.stop();
    }
  });

  test("reports data --output saves files to directory", async () => {
    const api = await startFakeApi();
    try {
      const outDir = mkdtempSync(resolve(tmpdir(), "pgai-output-test-"));
      const r = await runCliAsync(
        ["reports", "data", "1", "--output", outDir],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      // Should print the file path to stdout
      expect(r.stdout).toContain("H002.md");

      // File should exist with correct content
      const filePath = resolve(outDir, "H002.md");
      expect(existsSync(filePath)).toBe(true);
      const content = readFileSync(filePath, "utf-8");
      expect(content).toContain("# H002 Report");
      expect(content).toContain("Unused indexes found.");
    } finally {
      api.stop();
    }
  });

  test("reports data -o creates directory if it does not exist", async () => {
    const api = await startFakeApi();
    try {
      const base = mkdtempSync(resolve(tmpdir(), "pgai-output-test-"));
      const outDir = resolve(base, "nested", "dir");
      const r = await runCliAsync(
        ["reports", "data", "1", "-o", outDir],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);
      expect(existsSync(resolve(outDir, "H002.md"))).toBe(true);
    } finally {
      api.stop();
    }
  });

  test("reports data --formatted is accepted", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "data", "1", "--formatted"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);
      // The command should succeed — ANSI formatting only active in TTY
      // In non-TTY (our test pipe), it falls back to raw output
      expect(r.stdout).toContain("H002 Report");
    } finally {
      api.stop();
    }
  });

  test("reports data sends correct filters to API", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        [
          "reports",
          "data",
          "1",
          "--type",
          "md",
          "--check-id",
          "H001",
          "--json",
        ],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_file_data")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.checkup_report_id).toBe("eq.1");
      expect(req!.searchParams.type).toBe("eq.md");
      expect(req!.searchParams.check_id).toBe("eq.H001");
    } finally {
      api.stop();
    }
  });

  test("reports list --before filters by created_at with ISO date", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--before", "2025-01-15"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.created_at).toContain("lt.2025-01-15");
      expect(req!.searchParams.order).toBe("id.desc");
    } finally {
      api.stop();
    }
  });

  test("reports list --before accepts DD.MM.YYYY format", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "list", "--before", "15.01.2025"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_reports")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.created_at).toContain("lt.2025-01-15");
    } finally {
      api.stop();
    }
  });

  test("reports list --before rejects invalid date", async () => {
    const r = runCli(
      ["reports", "list", "--before", "not-a-date"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("Unrecognized date format");
  });

  test("reports list --all --before rejects conflicting flags", () => {
    const r = runCli(
      ["reports", "list", "--all", "--before", "2025-01-15"],
      isolatedEnv({ PGAI_API_KEY: "test-key" })
    );
    expect(r.status).toBe(1);
    expect(`${r.stdout}\n${r.stderr}`).toContain("--all and --before cannot be used together");
  });

  test("reports data without --type defaults to md for terminal output", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "data", "1"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_file_data")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.type).toBe("eq.md");
    } finally {
      api.stop();
    }
  });

  test("reports data --json without --type fetches all types", async () => {
    const api = await startFakeApi();
    try {
      await runCliAsync(
        ["reports", "data", "1", "--json"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );

      const req = api.requests.find((x) =>
        x.pathname.endsWith("/checkup_report_file_data")
      );
      expect(req).toBeTruthy();
      expect(req!.searchParams.type).toBeUndefined();
    } finally {
      api.stop();
    }
  });

  test("reports data --output strips path traversal from filenames", async () => {
    // Start a fake API that returns a filename with path traversal
    const traversalRequests: typeof Array.prototype = [];
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      async fetch(req) {
        const url = new URL(req.url);
        if (url.pathname.endsWith("/checkup_report_file_data")) {
          return new Response(
            JSON.stringify([
              {
                id: 100,
                checkup_report_id: 1,
                filename: "../../etc/malicious.md",
                check_id: "H002",
                type: "md",
                created_at: "2025-01-01T00:00:00Z",
                created_formatted: "2025-01-01 00:00:00",
                project_id: 10,
                project_name: "TestProj",
                data: "# Malicious content\n",
              },
            ]),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      },
    });

    try {
      const outDir = mkdtempSync(resolve(tmpdir(), "pgai-traversal-test-"));
      const r = await runCliAsync(
        ["reports", "data", "1", "--output", outDir],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: `http://${server.hostname}:${server.port}/api/general`,
        })
      );
      expect(r.status).toBe(0);

      // File should be saved as basename only, not the traversal path
      expect(existsSync(resolve(outDir, "malicious.md"))).toBe(true);
      // Traversal path should NOT exist
      expect(existsSync(resolve(outDir, "..", "..", "etc", "malicious.md"))).toBe(false);
      // Stdout should show the safe name
      expect(r.stdout).toContain("malicious.md");
      expect(r.stdout).not.toContain("../../");
    } finally {
      server.stop(true);
    }
  });

  test("reports list --all fetches all pages", async () => {
    const api = await startFakeApi();
    try {
      const r = await runCliAsync(
        ["reports", "list", "--all"],
        isolatedEnv({
          PGAI_API_KEY: "test-key",
          PGAI_API_BASE_URL: api.baseUrl,
        })
      );
      expect(r.status).toBe(0);

      const out = JSON.parse(r.stdout.trim());
      expect(Array.isArray(out)).toBe(true);
      // The fake API always returns the same 1-item array, so --all will get 1 item
      // (the page size > result count triggers stop)
      expect(out.length).toBeGreaterThanOrEqual(1);
    } finally {
      api.stop();
    }
  });
});
