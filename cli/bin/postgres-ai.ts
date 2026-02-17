#!/usr/bin/env bun

import { Command, Option } from "commander";
import pkg from "../package.json";
import * as config from "../lib/config";
import * as yaml from "js-yaml";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as crypto from "node:crypto";
import { Client } from "pg";
import { startMcpServer } from "../lib/mcp-server";
import { fetchIssues, fetchIssueComments, createIssueComment, fetchIssue, createIssue, updateIssue, updateIssueComment, fetchActionItem, fetchActionItems, createActionItem, updateActionItem, type ConfigChange } from "../lib/issues";
import { fetchReports, fetchAllReports, fetchReportFiles, fetchReportFileData, renderMarkdownForTerminal, parseFlexibleDate } from "../lib/reports";
import { resolveBaseUrls } from "../lib/util";
import { uploadFile, downloadFile, buildMarkdownLink } from "../lib/storage";
import { applyInitPlan, applyUninitPlan, buildInitPlan, buildUninitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER, KNOWN_PROVIDERS, redactPasswordsInSql, resolveAdminConnection, resolveMonitoringPassword, validateProvider, verifyInitSetup } from "../lib/init";
import { SupabaseClient, resolveSupabaseConfig, extractProjectRefFromUrl, applyInitPlanViaSupabase, verifyInitSetupViaSupabase, fetchPoolerDatabaseUrl, type PgCompatibleError } from "../lib/supabase";
import * as pkce from "../lib/pkce";
import * as authServer from "../lib/auth-server";
import { maskSecret } from "../lib/util";
import { createInterface } from "readline";
import * as childProcess from "child_process";
import { REPORT_GENERATORS, CHECK_INFO, generateAllReports } from "../lib/checkup";
import { getCheckupEntry } from "../lib/checkup-dictionary";
import { createCheckupReport, uploadCheckupReportJson, convertCheckupReportJsonToMarkdown, RpcError, formatRpcErrorForDisplay, withRetry } from "../lib/checkup-api";
import { generateCheckSummary } from "../lib/checkup-summary";

// Node.js version check - require Node 18+
// Node 14 reached EOL in April 2023, Node 16 in September 2023.
// Node 18+ is required for native ESM, modern crypto APIs, and security updates.
const nodeVersion = parseInt(process.versions.node.split('.')[0], 10);
if (nodeVersion < 18) {
  console.error(`\x1b[31mError: postgresai requires Node 18 or higher.\x1b[0m`);
  console.error(`You are running Node.js ${process.versions.node}.`);
  console.error(`Please upgrade to Node.js 20 LTS or Node.js 22 for security updates.`);
  console.error(`\nDownload: https://nodejs.org/`);
  process.exit(1);
}

// Singleton readline interface for stdin prompts
let rl: ReturnType<typeof createInterface> | null = null;
function getReadline() {
  if (!rl) {
    rl = createInterface({ input: process.stdin, output: process.stdout });
  }
  return rl;
}
function closeReadline() {
  if (rl) {
    rl.close();
    rl = null;
  }
}

// Helper functions for spawning processes - use Node.js child_process for compatibility
async function execPromise(command: string): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    childProcess.exec(command, (error, stdout, stderr) => {
      if (error) {
        const err = error as Error & { code: number };
        err.code = typeof error.code === "number" ? error.code : 1;
        reject(err);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

async function execFilePromise(file: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    childProcess.execFile(file, args, (error, stdout, stderr) => {
      if (error) {
        const err = error as Error & { code: number };
        err.code = typeof error.code === "number" ? error.code : 1;
        reject(err);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

function spawnSync(cmd: string, args: string[], options?: { stdio?: "pipe" | "ignore" | "inherit"; encoding?: string; env?: Record<string, string | undefined>; cwd?: string }): { status: number | null; stdout: string; stderr: string } {
  const result = childProcess.spawnSync(cmd, args, {
    stdio: options?.stdio === "inherit" ? "inherit" : "pipe",
    env: options?.env as NodeJS.ProcessEnv,
    cwd: options?.cwd,
    encoding: "utf8",
  });
  return {
    status: result.status,
    stdout: typeof result.stdout === "string" ? result.stdout : "",
    stderr: typeof result.stderr === "string" ? result.stderr : "",
  };
}

function spawn(cmd: string, args: string[], options?: { stdio?: "pipe" | "ignore" | "inherit"; env?: Record<string, string | undefined>; cwd?: string; detached?: boolean }): { on: (event: string, cb: (code: number | null, signal?: string) => void) => void; unref: () => void; pid?: number } {
  const proc = childProcess.spawn(cmd, args, {
    stdio: options?.stdio ?? "pipe",
    env: options?.env as NodeJS.ProcessEnv,
    cwd: options?.cwd,
    detached: options?.detached,
  });

  return {
    on(event: string, cb: (code: number | null, signal?: string) => void) {
      if (event === "close" || event === "exit") {
        proc.on(event, (code, signal) => cb(code, signal ?? undefined));
      } else if (event === "error") {
        proc.on("error", (err) => cb(null, String(err)));
      }
      return this;
    },
    unref() {
      proc.unref();
    },
    pid: proc.pid,
  };
}

// Simple readline-like interface for prompts using Bun
async function question(prompt: string): Promise<string> {
  return new Promise((resolve) => {
    getReadline().question(prompt, (answer) => {
      resolve(answer);
    });
  });
}

function expandHomePath(p: string): string {
  const s = (p || "").trim();
  if (!s) return s;
  if (s === "~") return os.homedir();
  if (s.startsWith("~/") || s.startsWith("~\\")) {
    return path.join(os.homedir(), s.slice(2));
  }
  return s;
}

function createTtySpinner(
  enabled: boolean,
  initialText: string
): { update: (text: string) => void; stop: (finalText?: string) => void } {
  if (!enabled) {
    return {
      update: () => {},
      stop: () => {},
    };
  }

  const frames = ["|", "/", "-", "\\"];
  const startTs = Date.now();
  let text = initialText;
  let frameIdx = 0;
  let stopped = false;

  const render = (): void => {
    if (stopped) return;
    const elapsedSec = ((Date.now() - startTs) / 1000).toFixed(1);
    const frame = frames[frameIdx % frames.length] ?? frames[0] ?? "⠿";
    frameIdx += 1;
    process.stdout.write(`\r\x1b[2K${frame} ${text} (${elapsedSec}s)`);
  };

  const timer = setInterval(render, 120);
  render(); // immediate feedback

  return {
    update: (t: string) => {
      text = t;
      render();
    },
    stop: (finalText?: string) => {
      if (stopped) return;
      // Set flag first so any queued render() calls exit early.
      // JavaScript is single-threaded, so this is safe: queued callbacks
      // run after stop() returns and will see stopped=true immediately.
      stopped = true;
      clearInterval(timer);
      process.stdout.write("\r\x1b[2K");
      if (finalText && finalText.trim()) {
        process.stdout.write(finalText);
      }
      process.stdout.write("\n");
    },
  };
}

// ============================================================================
// Checkup command helpers
// ============================================================================

interface CheckupOptions {
  checkId: string;
  nodeName: string;
  output?: string;
  upload?: boolean;
  project?: string;
  json?: boolean;
  markdown?: boolean;
}

interface UploadConfig {
  apiKey: string;
  apiBaseUrl: string;
  project: string;
}

interface UploadSummary {
  project: string;
  reportId: number;
  uploaded: Array<{ checkId: string; filename: string; chunkId: number }>;
}

/**
 * Prepare and validate output directory for checkup reports.
 * @returns Output path if valid, null if should exit with error
 */
function prepareOutputDirectory(outputOpt: string | undefined): string | null | undefined {
  if (!outputOpt) return undefined;

  const outputDir = expandHomePath(outputOpt);
  const outputPath = path.isAbsolute(outputDir) ? outputDir : path.resolve(process.cwd(), outputDir);

  if (!fs.existsSync(outputPath)) {
    try {
      fs.mkdirSync(outputPath, { recursive: true });
    } catch (e) {
      const errAny = e as any;
      const code = typeof errAny?.code === "string" ? errAny.code : "";
      const msg = errAny instanceof Error ? errAny.message : String(errAny);
      if (code === "EACCES" || code === "EPERM" || code === "ENOENT") {
        console.error(`Error: Failed to create output directory: ${outputPath}`);
        console.error(`Reason: ${msg}`);
        console.error("Tip: choose a writable path, e.g. --output ./reports or --output ~/reports");
        return null; // Signal to exit
      }
      throw e;
    }
  }
  return outputPath;
}

/**
 * Prepare upload configuration for checkup reports.
 * @returns Upload config if valid, null if should exit, undefined if upload not needed
 */
function prepareUploadConfig(
  opts: CheckupOptions,
  rootOpts: CliOptions,
  shouldUpload: boolean,
  uploadExplicitlyRequested: boolean
): { config: UploadConfig; projectWasGenerated: boolean } | null | undefined {
  if (!shouldUpload) return undefined;

  const { apiKey } = getConfig(rootOpts);
  if (!apiKey) {
    if (uploadExplicitlyRequested) {
      console.error("Error: API key is required for upload");
      console.error("Tip: run 'postgresai auth' or pass --api-key / set PGAI_API_KEY");
      return null; // Signal to exit
    }
    return undefined; // Skip upload silently
  }

  const cfg = config.readConfig();
  const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);
  let project = ((opts.project || cfg.defaultProject) || "").trim();
  let projectWasGenerated = false;

  if (!project) {
    project = `project_${crypto.randomBytes(6).toString("hex")}`;
    projectWasGenerated = true;
    try {
      config.writeConfig({ defaultProject: project });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error(`Warning: Failed to save generated default project: ${message}`);
    }
  }

  return {
    config: { apiKey, apiBaseUrl, project },
    projectWasGenerated,
  };
}

/**
 * Upload checkup reports to PostgresAI API.
 */
async function uploadCheckupReports(
  uploadCfg: UploadConfig,
  reports: Record<string, any>,
  spinner: ReturnType<typeof createTtySpinner>,
  logUpload: (msg: string) => void
): Promise<UploadSummary> {
  spinner.update("Creating remote checkup report");
  const created = await withRetry(
    () => createCheckupReport({
      apiKey: uploadCfg.apiKey,
      apiBaseUrl: uploadCfg.apiBaseUrl,
      project: uploadCfg.project,
    }),
    { maxAttempts: 3 },
    (attempt, err, delayMs) => {
      const errMsg = err instanceof Error ? err.message : String(err);
      logUpload(`[Retry ${attempt}/3] createCheckupReport failed: ${errMsg}, retrying in ${delayMs}ms...`);
    }
  );

  const reportId = created.reportId;

  const uploaded: Array<{ checkId: string; filename: string; chunkId: number }> = [];
  for (const [checkId, report] of Object.entries(reports)) {
    spinner.update(`Uploading ${checkId}.json`);
    const jsonText = JSON.stringify(report, null, 2);
    const r = await withRetry(
      () => uploadCheckupReportJson({
        apiKey: uploadCfg.apiKey,
        apiBaseUrl: uploadCfg.apiBaseUrl,
        reportId,
        filename: `${checkId}.json`,
        checkId,
        jsonText,
      }),
      { maxAttempts: 3 },
      (attempt, err, delayMs) => {
        const errMsg = err instanceof Error ? err.message : String(err);
        logUpload(`[Retry ${attempt}/3] Upload ${checkId}.json failed: ${errMsg}, retrying in ${delayMs}ms...`);
      }
    );
    uploaded.push({ checkId, filename: `${checkId}.json`, chunkId: r.reportChunkId });
  }

  return { project: uploadCfg.project, reportId, uploaded };
}

/**
 * Write checkup reports to files.
 */
function writeReportFiles(reports: Record<string, any>, outputPath: string): void {
  for (const [checkId, report] of Object.entries(reports)) {
    const filePath = path.join(outputPath, `${checkId}.json`);
    fs.writeFileSync(filePath, JSON.stringify(report, null, 2), "utf8");
    const title = report.checkTitle || checkId;
    console.log(`✓ ${checkId} ${title}: ${filePath}`);
  }
}

/**
 * Print upload summary to console.
 */
function printUploadSummary(
  summary: UploadSummary,
  projectWasGenerated: boolean,
  useStderr: boolean,
  reports: Record<string, any>
): void {
  const out = useStderr ? console.error : console.log;
  out("\nCheckup report uploaded");
  out("======================\n");
  if (projectWasGenerated) {
    out(`Project: ${summary.project} (generated and saved as default)`);
  } else {
    out(`Project: ${summary.project}`);
  }
  out(`Report ID: ${summary.reportId}`);
  out("View in Console: console.postgres.ai → Checkup → checkup reports");
  out("");

  // Show check summaries (filter out generic info messages)
  const summaries = [];
  let skippedCount = 0;

  for (const item of summary.uploaded) {
    const report = reports[item.checkId];
    if (report) {
      const { status, message } = generateCheckSummary(item.checkId, report);
      const title = report.checkTitle || item.checkId;

      // Show if: warning/ok status, or info with concrete data (contains numbers or version info)
      const isSignificant = status !== 'info' || /\d/.test(message) || message.includes('PostgreSQL') || message.includes('Version');

      if (isSignificant) {
        summaries.push({ checkId: item.checkId, title, status, message });
      } else {
        skippedCount++;
      }
    }
  }

  // Print significant checks
  for (const { checkId, title, message } of summaries) {
    out(`  ${checkId} (${title}): ${message}`);
  }

  // Show count of other checks
  if (skippedCount > 0) {
    out(`  ${skippedCount} other check${skippedCount > 1 ? 's' : ''} completed`);
  }
}

// ============================================================================
// CLI configuration
// ============================================================================

/**
 * CLI configuration options
 */
interface CliOptions {
  apiKey?: string;
  apiBaseUrl?: string;
  uiBaseUrl?: string;
  storageBaseUrl?: string;
}

/**
 * Configuration result
 */
interface ConfigResult {
  apiKey: string;
}

/**
 * Instance configuration
 */
interface Instance {
  name: string;
  conn_str?: string;
  preset_metrics?: string;
  custom_metrics?: any;
  is_enabled?: boolean;
  group?: string;
  custom_tags?: Record<string, any>;
}

/**
 * Path resolution result
 */
interface PathResolution {
  fs: typeof fs;
  path: typeof path;
  projectDir: string;
  composeFile: string;
  instancesFile: string;
}

function getDefaultMonitoringProjectDir(): string {
  const override = process.env.PGAI_PROJECT_DIR;
  if (override && override.trim()) return override.trim();
  // Keep monitoring project next to user-level config (~/.config/postgresai)
  return path.join(config.getConfigDir(), "monitoring");
}

async function downloadText(url: string): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} for ${url}`);
    }
    return await response.text();
  } finally {
    clearTimeout(timeout);
  }
}

async function ensureDefaultMonitoringProject(): Promise<PathResolution> {
  const projectDir = getDefaultMonitoringProjectDir();
  const composeFile = path.resolve(projectDir, "docker-compose.yml");
  const instancesFile = path.resolve(projectDir, "instances.yml");

  if (!fs.existsSync(projectDir)) {
    fs.mkdirSync(projectDir, { recursive: true, mode: 0o700 });
  }

  if (!fs.existsSync(composeFile)) {
    const refs = [
      process.env.PGAI_PROJECT_REF,
      pkg.version,
      `v${pkg.version}`,
      "main",
    ].filter((v): v is string => Boolean(v && v.trim()));

    let lastErr: unknown;
    for (const ref of refs) {
      const url = `https://gitlab.com/postgres-ai/postgres_ai/-/raw/${encodeURIComponent(ref)}/docker-compose.yml`;
      try {
        const text = await downloadText(url);
        fs.writeFileSync(composeFile, text, { encoding: "utf8", mode: 0o600 });
        break;
      } catch (err) {
        lastErr = err;
      }
    }

    if (!fs.existsSync(composeFile)) {
      const msg = lastErr instanceof Error ? lastErr.message : String(lastErr);
      throw new Error(`Failed to bootstrap docker-compose.yml: ${msg}`);
    }
  }

  // Ensure instances.yml exists as a FILE (avoid Docker creating a directory)
  if (!fs.existsSync(instancesFile)) {
    const header =
      "# PostgreSQL instances to monitor\n" +
      "# Add your instances using: pgai mon targets add <connection-string> <name>\n\n";
    fs.writeFileSync(instancesFile, header, { encoding: "utf8", mode: 0o600 });
  }

  // Ensure .pgwatch-config exists as a FILE for reporter (may remain empty)
  const pgwatchConfig = path.resolve(projectDir, ".pgwatch-config");
  if (!fs.existsSync(pgwatchConfig)) {
    fs.writeFileSync(pgwatchConfig, "", { encoding: "utf8", mode: 0o600 });
  }

  // Ensure .env exists and has PGAI_TAG (compose requires it)
  const envFile = path.resolve(projectDir, ".env");
  if (!fs.existsSync(envFile)) {
    const envText = `PGAI_TAG=${pkg.version}\n# PGAI_REGISTRY=registry.gitlab.com/postgres-ai/postgres_ai\n`;
    fs.writeFileSync(envFile, envText, { encoding: "utf8", mode: 0o600 });
  }

  return { fs, path, projectDir, composeFile, instancesFile };
}

/**
 * Get configuration from various sources
 * @param opts - Command line options
 * @returns Configuration object
 */
function getConfig(opts: CliOptions): ConfigResult {
  // Priority order:
  // 1. Command line option (--api-key)
  // 2. Environment variable (PGAI_API_KEY)
  // 3. User-level config file (~/.config/postgresai/config.json)
  // 4. Legacy project-local config (.pgwatch-config)

  let apiKey = opts.apiKey || process.env.PGAI_API_KEY || "";

  // Try config file if not provided via CLI or env
  if (!apiKey) {
    const fileConfig = config.readConfig();
    if (!apiKey) apiKey = fileConfig.apiKey || "";
  }

  return { apiKey };
}

// Human-friendly output helper: YAML for TTY by default, JSON when --json or non-TTY
function printResult(result: unknown, json?: boolean): void {
  if (typeof result === "string") {
    process.stdout.write(result);
    if (!/\n$/.test(result)) console.log();
    return;
  }
  if (json || !process.stdout.isTTY) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    let text = yaml.dump(result as any);
    if (Array.isArray(result)) {
      text = text.replace(/\n- /g, "\n\n- ");
    }
    console.log(text);
  }
}

const program = new Command();

program
  .name("postgres-ai")
  .description("PostgresAI CLI")
  .version(pkg.version)
  .option("--api-key <key>", "API key (overrides PGAI_API_KEY)")
  .option(
    "--api-base-url <url>",
    "API base URL for backend RPC (overrides PGAI_API_BASE_URL)"
  )
  .option(
    "--ui-base-url <url>",
    "UI base URL for browser routes (overrides PGAI_UI_BASE_URL)"
  )
  .option(
    "--storage-base-url <url>",
    "Storage base URL for file uploads (overrides PGAI_STORAGE_BASE_URL)"
  );

program
  .command("set-default-project <project>")
  .description("store default project for checkup uploads")
  .action(async (project: string) => {
    const value = (project || "").trim();
    if (!value) {
      console.error("Error: project is required");
      process.exitCode = 1;
      return;
    }
    config.writeConfig({ defaultProject: value });
    console.log(`Default project saved: ${value}`);
  });

program
  .command("set-storage-url <url>")
  .description("store storage base URL for file uploads")
  .action(async (url: string) => {
    const value = (url || "").trim();
    if (!value) {
      console.error("Error: url is required");
      process.exitCode = 1;
      return;
    }
    try {
      const { normalizeBaseUrl } = await import("../lib/util");
      const normalized = normalizeBaseUrl(value);
      config.writeConfig({ storageBaseUrl: normalized });
      console.log(`Storage URL saved: ${normalized}`);
    } catch {
      console.error(`Error: invalid URL: ${value}`);
      process.exitCode = 1;
    }
  });

program
  .command("prepare-db [conn]")
  .description("prepare database for monitoring: create monitoring user, required view(s), and grant permissions (idempotent)")
  .option("--db-url <url>", "PostgreSQL connection URL (admin) to run the setup against (deprecated; pass it as positional arg)")
  .option("-h, --host <host>", "PostgreSQL host (psql-like)")
  .option("-p, --port <port>", "PostgreSQL port (psql-like)")
  .option("-U, --username <username>", "PostgreSQL user (psql-like)")
  .option("-d, --dbname <dbname>", "PostgreSQL database name (psql-like)")
  .option("--admin-password <password>", "Admin connection password (otherwise uses PGPASSWORD if set)")
  .option("--monitoring-user <name>", "Monitoring role name to create/update", DEFAULT_MONITORING_USER)
  .option("--password <password>", "Monitoring role password (overrides PGAI_MON_PASSWORD)")
  .option("--skip-optional-permissions", "Skip optional permissions (RDS/self-managed extras)", false)
  .option("--provider <provider>", "Database provider (e.g., supabase). Affects which steps are executed.")
  .option("--verify", "Verify that monitoring role/permissions are in place (no changes)", false)
  .option("--reset-password", "Reset monitoring role password only (no other changes)", false)
  .option("--print-sql", "Print SQL plan and exit (no changes applied)", false)
  .option("--print-password", "Print generated monitoring password (DANGEROUS in CI logs)", false)
  .option("--supabase", "Use Supabase Management API instead of direct PostgreSQL connection", false)
  .option("--supabase-access-token <token>", "Supabase Management API access token (or SUPABASE_ACCESS_TOKEN env)")
  .option("--supabase-project-ref <ref>", "Supabase project reference (or SUPABASE_PROJECT_REF env)")
  .option("--json", "Output result as JSON (machine-readable)", false)
  .addHelpText(
    "after",
    [
      "",
      "Examples:",
      "  postgresai prepare-db postgresql://admin@host:5432/dbname",
      "  postgresai prepare-db \"dbname=dbname host=host user=admin\"",
      "  postgresai prepare-db -h host -p 5432 -U admin -d dbname",
      "",
      "Admin password:",
      "  --admin-password <password>   or  PGPASSWORD=... (libpq standard)",
      "",
      "Monitoring password:",
      "  --password <password>         or  PGAI_MON_PASSWORD=...  (otherwise auto-generated)",
      "  If auto-generated, it is printed only on TTY by default.",
      "  To print it in non-interactive mode: --print-password",
      "",
      "SSL connection (sslmode=prefer behavior):",
      "  Tries SSL first, falls back to non-SSL if server doesn't support it.",
      "  To force SSL: PGSSLMODE=require or ?sslmode=require in URL",
      "  To disable SSL: PGSSLMODE=disable or ?sslmode=disable in URL",
      "",
      "Environment variables (libpq standard):",
      "  PGHOST, PGPORT, PGUSER, PGDATABASE  — connection defaults",
      "  PGPASSWORD                          — admin password",
      "  PGSSLMODE                           — SSL mode (disable, require, verify-full)",
      "  PGAI_MON_PASSWORD                   — monitoring password",
      "",
      "Inspect SQL without applying changes:",
      "  postgresai prepare-db <conn> --print-sql",
      "",
      "Verify setup (no changes):",
      "  postgresai prepare-db <conn> --verify",
      "",
      "Reset monitoring password only:",
      "  postgresai prepare-db <conn> --reset-password --password '...'",
      "",
      "Offline SQL plan (no DB connection):",
      "  postgresai prepare-db --print-sql",
      "",
      "Supabase mode (use Management API instead of direct connection):",
      "  postgresai prepare-db --supabase --supabase-project-ref <ref>",
      "  SUPABASE_ACCESS_TOKEN=... postgresai prepare-db --supabase --supabase-project-ref <ref>",
      "",
      "  Generate a token at: https://supabase.com/dashboard/account/tokens",
      "  Find your project ref in: https://supabase.com/dashboard/project/<ref>",
      "",
      "Provider-specific behavior (for direct connections):",
      "  --provider supabase    Skip role creation (create user in Supabase dashboard)",
      "                         Skip ALTER USER (restricted by Supabase)",
    ].join("\n")
  )
  .action(async (conn: string | undefined, opts: {
    dbUrl?: string;
    host?: string;
    port?: string;
    username?: string;
    dbname?: string;
    adminPassword?: string;
    monitoringUser: string;
    password?: string;
    skipOptionalPermissions?: boolean;
    provider?: string;
    verify?: boolean;
    resetPassword?: boolean;
    printSql?: boolean;
    printPassword?: boolean;
    supabase?: boolean;
    supabaseAccessToken?: string;
    supabaseProjectRef?: string;
    json?: boolean;
  }, cmd: Command) => {
    // JSON output helper
    const jsonOutput = opts.json;
    const outputJson = (data: Record<string, unknown>) => {
      console.log(JSON.stringify(data, null, 2));
    };
    const outputError = (error: {
      message: string;
      step?: string;
      code?: string;
      detail?: string;
      hint?: string;
      httpStatus?: number;
    }) => {
      if (jsonOutput) {
        outputJson({
          success: false,
          mode: opts.supabase ? "supabase" : "direct",
          error,
        });
      } else {
        console.error(`Error: prepare-db${opts.supabase ? " (Supabase)" : ""}: ${error.message}`);
        if (error.step) console.error(`  Step: ${error.step}`);
        if (error.code) console.error(`  Code: ${error.code}`);
        if (error.detail) console.error(`  Detail: ${error.detail}`);
        if (error.hint) console.error(`  Hint: ${error.hint}`);
        if (error.httpStatus) console.error(`  HTTP Status: ${error.httpStatus}`);
      }
      process.exitCode = 1;
    };
    if (opts.verify && opts.resetPassword) {
      outputError({ message: "Provide only one of --verify or --reset-password" });
      return;
    }
    if (opts.verify && opts.printSql) {
      outputError({ message: "--verify cannot be combined with --print-sql" });
      return;
    }

    const shouldPrintSql = !!opts.printSql;
    const redactPasswords = (sql: string): string => redactPasswordsInSql(sql);

    // Validate provider and warn if unknown
    const providerWarning = validateProvider(opts.provider);
    if (providerWarning) {
      console.warn(`⚠ ${providerWarning}`);
    }

    // Offline mode: allow printing SQL without providing/using an admin connection.
    // Useful for audits/reviews; caller can provide -d/PGDATABASE.
    if (!conn && !opts.dbUrl && !opts.host && !opts.port && !opts.username && !opts.adminPassword) {
      if (shouldPrintSql) {
        const database = (opts.dbname ?? process.env.PGDATABASE ?? "postgres").trim();
        const includeOptionalPermissions = !opts.skipOptionalPermissions;

        // Use explicit password/env if provided; otherwise use a placeholder.
        // Printed SQL always redacts secrets.
        const monPassword =
          (opts.password ?? process.env.PGAI_MON_PASSWORD ?? "<redacted>").toString();

        const plan = await buildInitPlan({
          database,
          monitoringUser: opts.monitoringUser,
          monitoringPassword: monPassword,
          includeOptionalPermissions,
          provider: opts.provider,
        });

        console.log("\n--- SQL plan (offline; not connected) ---");
        console.log(`-- database: ${database}`);
        console.log(`-- monitoring user: ${opts.monitoringUser}`);
        console.log(`-- provider: ${opts.provider ?? "self-managed"}`);
        console.log(`-- optional permissions: ${includeOptionalPermissions ? "enabled" : "skipped"}`);
        for (const step of plan.steps) {
          console.log(`\n-- ${step.name}${step.optional ? " (optional)" : ""}`);
          console.log(redactPasswords(step.sql));
        }
        console.log("\n--- end SQL plan ---\n");
        console.log("Note: passwords are redacted in the printed SQL output.");
        return;
      }
    }

    // Supabase mode: use Supabase Management API instead of direct PG connection
    if (opts.supabase) {
      let supabaseConfig;
      try {
        // Try to extract project ref from connection URL if provided
        let projectRef = opts.supabaseProjectRef;
        if (!projectRef && conn) {
          projectRef = extractProjectRefFromUrl(conn);
        }
        supabaseConfig = resolveSupabaseConfig({
          accessToken: opts.supabaseAccessToken,
          projectRef,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        outputError({ message: msg });
        return;
      }

      const includeOptionalPermissions = !opts.skipOptionalPermissions;

      if (!jsonOutput) {
        console.log(`Supabase mode: project ref ${supabaseConfig.projectRef}`);
        console.log(`Monitoring user: ${opts.monitoringUser}`);
        console.log(`Optional permissions: ${includeOptionalPermissions ? "enabled" : "skipped"}`);
      }

      const supabaseClient = new SupabaseClient(supabaseConfig);

      // Fetch database URL for JSON output (best-effort, errors return null)
      let databaseUrl: string | null = null;
      if (jsonOutput) {
        databaseUrl = await fetchPoolerDatabaseUrl(supabaseConfig, opts.monitoringUser);
      }

      try {
        // Get current database name
        const database = await supabaseClient.getCurrentDatabase();
        if (!database) {
          throw new Error("Failed to resolve current database name");
        }
        if (!jsonOutput) {
          console.log(`Database: ${database}`);
        }

        if (opts.verify) {
          const v = await verifyInitSetupViaSupabase({
            client: supabaseClient,
            database,
            monitoringUser: opts.monitoringUser,
            includeOptionalPermissions,
          });
          if (v.ok) {
            if (jsonOutput) {
              const result: Record<string, unknown> = {
                success: true,
                mode: "supabase",
                action: "verify",
                database,
                monitoringUser: opts.monitoringUser,
                verified: true,
                missingOptional: v.missingOptional,
              };
              if (databaseUrl) {
                result.databaseUrl = databaseUrl;
              }
              outputJson(result);
            } else {
              console.log("✓ prepare-db verify: OK");
              if (v.missingOptional.length > 0) {
                console.error("⚠ Optional items missing:");
                for (const m of v.missingOptional) console.error(`- ${m}`);
              }
            }
            return;
          }
          if (jsonOutput) {
            const result: Record<string, unknown> = {
              success: false,
              mode: "supabase",
              action: "verify",
              database,
              monitoringUser: opts.monitoringUser,
              verified: false,
              missingRequired: v.missingRequired,
              missingOptional: v.missingOptional,
            };
            if (databaseUrl) {
              result.databaseUrl = databaseUrl;
            }
            outputJson(result);
          } else {
            console.error("✗ prepare-db verify failed: missing required items");
            for (const m of v.missingRequired) console.error(`- ${m}`);
            if (v.missingOptional.length > 0) {
              console.error("Optional items missing:");
              for (const m of v.missingOptional) console.error(`- ${m}`);
            }
          }
          process.exitCode = 1;
          return;
        }

        let monPassword: string;
        let passwordGenerated = false;
        try {
          const resolved = await resolveMonitoringPassword({
            passwordFlag: opts.password,
            passwordEnv: process.env.PGAI_MON_PASSWORD,
            monitoringUser: opts.monitoringUser,
          });
          monPassword = resolved.password;
          passwordGenerated = resolved.generated;
          if (resolved.generated) {
            const canPrint = process.stdout.isTTY || !!opts.printPassword || jsonOutput;
            if (canPrint) {
              if (!jsonOutput) {
                const shellSafe = monPassword.replace(/'/g, "'\\''");
                console.error("");
                console.error(`Generated monitoring password for ${opts.monitoringUser} (copy/paste):`);
                console.error(`PGAI_MON_PASSWORD='${shellSafe}'`);
                console.error("");
                console.log("Store it securely (or rerun with --password / PGAI_MON_PASSWORD to set your own).");
              }
              // For JSON mode, password will be included in the success output below
            } else {
              console.error(
                [
                  `✗ Monitoring password was auto-generated for ${opts.monitoringUser} but not printed in non-interactive mode.`,
                  "",
                  "Provide it explicitly:",
                  "  --password <password>   or   PGAI_MON_PASSWORD=...",
                  "",
                  "Or (NOT recommended) print the generated password:",
                  "  --print-password",
                ].join("\n")
              );
              process.exitCode = 1;
              return;
            }
          }
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          outputError({ message: msg });
          return;
        }

        const plan = await buildInitPlan({
          database,
          monitoringUser: opts.monitoringUser,
          monitoringPassword: monPassword,
          includeOptionalPermissions,
        });

        // For Supabase mode, skip RDS and self-managed steps (they don't apply)
        const supabaseApplicableSteps = plan.steps.filter(
          (s) => s.name !== "03.optional_rds" && s.name !== "04.optional_self_managed"
        );

        const effectivePlan = opts.resetPassword
          ? { ...plan, steps: supabaseApplicableSteps.filter((s) => s.name === "01.role") }
          : { ...plan, steps: supabaseApplicableSteps };

        if (shouldPrintSql) {
          console.log("\n--- SQL plan ---");
          for (const step of effectivePlan.steps) {
            console.log(`\n-- ${step.name}${step.optional ? " (optional)" : ""}`);
            console.log(redactPasswords(step.sql));
          }
          console.log("\n--- end SQL plan ---\n");
          console.log("Note: passwords are redacted in the printed SQL output.");
          return;
        }

        const { applied, skippedOptional } = await applyInitPlanViaSupabase({
          client: supabaseClient,
          plan: effectivePlan,
        });

        if (jsonOutput) {
          const result: Record<string, unknown> = {
            success: true,
            mode: "supabase",
            action: opts.resetPassword ? "reset-password" : "apply",
            database,
            monitoringUser: opts.monitoringUser,
            applied,
            skippedOptional,
            warnings: skippedOptional.length > 0
              ? ["Some optional steps were skipped (not supported or insufficient privileges)"]
              : [],
          };
          if (passwordGenerated) {
            result.generatedPassword = monPassword;
          }
          if (databaseUrl) {
            result.databaseUrl = databaseUrl;
          }
          outputJson(result);
        } else {
          console.log(opts.resetPassword ? "✓ prepare-db password reset completed" : "✓ prepare-db completed");
          if (skippedOptional.length > 0) {
            console.error("⚠ Some optional steps were skipped (not supported or insufficient privileges):");
            for (const s of skippedOptional) console.error(`- ${s}`);
          }
          if (process.stdout.isTTY) {
            console.log(`Applied ${applied.length} steps`);
          }
        }
      } catch (error) {
        const errAny = error as PgCompatibleError;
        let message = "";
        if (error instanceof Error && error.message) {
          message = error.message;
        } else if (errAny && typeof errAny === "object" && typeof errAny.message === "string" && errAny.message) {
          message = errAny.message;
        } else {
          message = String(error);
        }
        if (!message || message === "[object Object]") {
          message = "Unknown error";
        }

        // Surface step name if this was a plan step failure
        const stepMatch = typeof message === "string" ? message.match(/Failed at step "([^"]+)":/i) : null;
        const failedStep = stepMatch?.[1];

        // Build error object for JSON output
        const errorObj: {
          message: string;
          step?: string;
          code?: string;
          detail?: string;
          hint?: string;
          httpStatus?: number;
        } = { message };

        if (failedStep) errorObj.step = failedStep;
        if (errAny && typeof errAny === "object") {
          if (typeof errAny.code === "string" && errAny.code) errorObj.code = errAny.code;
          if (typeof errAny.detail === "string" && errAny.detail) errorObj.detail = errAny.detail;
          if (typeof errAny.hint === "string" && errAny.hint) errorObj.hint = errAny.hint;
          if (typeof errAny.httpStatus === "number") errorObj.httpStatus = errAny.httpStatus;
        }

        if (jsonOutput) {
          outputJson({
            success: false,
            mode: "supabase",
            error: errorObj,
          });
          process.exitCode = 1;
        } else {
          console.error(`Error: prepare-db (Supabase): ${message}`);

          if (failedStep) {
            console.error(`  Step: ${failedStep}`);
          }

          // Surface PostgreSQL-compatible error details
          if (errAny && typeof errAny === "object") {
            if (typeof errAny.code === "string" && errAny.code) {
              console.error(`  Code: ${errAny.code}`);
            }
            if (typeof errAny.detail === "string" && errAny.detail) {
              console.error(`  Detail: ${errAny.detail}`);
            }
            if (typeof errAny.hint === "string" && errAny.hint) {
              console.error(`  Hint: ${errAny.hint}`);
            }
            if (typeof errAny.httpStatus === "number") {
              console.error(`  HTTP Status: ${errAny.httpStatus}`);
            }
          }

          // Provide context hints for common errors
          if (errAny && typeof errAny === "object" && typeof errAny.code === "string") {
            if (errAny.code === "42501") {
              if (failedStep === "01.role") {
                console.error("  Context: role creation/update requires CREATEROLE or superuser");
              } else if (failedStep === "03.permissions") {
                console.error("  Context: grants/view/search_path require sufficient GRANT/DDL privileges");
              }
              console.error("  Fix: ensure your Supabase access token has sufficient permissions");
              console.error("  Tip: run with --print-sql to review the exact SQL plan");
            }
            // Schema already exists (42P06) or other duplicate object errors
            if (errAny.code === "42P06" || (message.includes("already exists") && failedStep === "03.permissions")) {
              console.error("  Hint: postgres_ai schema or objects already exist from a previous setup.");
              console.error("  Fix: run 'postgresai unprepare-db <connection>' first to clean up, then retry prepare-db.");
            }
          }
          if (errAny && typeof errAny === "object" && typeof errAny.httpStatus === "number") {
            if (errAny.httpStatus === 401) {
              console.error("  Hint: invalid or expired access token; generate a new one at https://supabase.com/dashboard/account/tokens");
            }
            if (errAny.httpStatus === 403) {
              console.error("  Hint: access denied; check your token permissions and project access");
            }
            if (errAny.httpStatus === 404) {
              console.error("  Hint: project not found; verify the project reference is correct");
            }
            if (errAny.httpStatus === 429) {
              console.error("  Hint: rate limited; wait a moment and try again");
            }
          }
          process.exitCode = 1;
        }
      }
      return;
    }

    let adminConn;
    try {
      adminConn = resolveAdminConnection({
        conn,
        dbUrlFlag: opts.dbUrl,
        // Allow libpq standard env vars as implicit defaults (common UX).
        host: opts.host ?? process.env.PGHOST,
        port: opts.port ?? process.env.PGPORT,
        username: opts.username ?? process.env.PGUSER,
        dbname: opts.dbname ?? process.env.PGDATABASE,
        adminPassword: opts.adminPassword,
        envPassword: process.env.PGPASSWORD,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (jsonOutput) {
        outputError({ message: msg });
      } else {
        console.error(`Error: prepare-db: ${msg}`);
        // When connection details are missing, show full init help (options + examples).
        if (typeof msg === "string" && msg.startsWith("Connection is required.")) {
          console.error("");
          cmd.outputHelp({ error: true });
        }
        process.exitCode = 1;
      }
      return;
    }

    const includeOptionalPermissions = !opts.skipOptionalPermissions;

    if (!jsonOutput) {
      console.log(`Connecting to: ${adminConn.display}`);
      console.log(`Monitoring user: ${opts.monitoringUser}`);
      console.log(`Optional permissions: ${includeOptionalPermissions ? "enabled" : "skipped"}`);
    }

    // Use native pg client instead of requiring psql to be installed
    let client: Client | undefined;
    try {
      const connResult = await connectWithSslFallback(Client, adminConn);
      client = connResult.client;

      const dbRes = await client.query("select current_database() as db");
      const database = dbRes.rows?.[0]?.db;
      if (typeof database !== "string" || !database) {
        throw new Error("Failed to resolve current database name");
      }

      if (opts.verify) {
        const v = await verifyInitSetup({
          client,
          database,
          monitoringUser: opts.monitoringUser,
          includeOptionalPermissions,
          provider: opts.provider,
        });
        if (v.ok) {
          if (jsonOutput) {
            outputJson({
              success: true,
              mode: "direct",
              action: "verify",
              database,
              monitoringUser: opts.monitoringUser,
              provider: opts.provider,
              verified: true,
              missingOptional: v.missingOptional,
            });
          } else {
            console.log(`✓ prepare-db verify: OK${opts.provider ? ` (provider: ${opts.provider})` : ""}`);
            if (v.missingOptional.length > 0) {
              console.error("⚠ Optional items missing:");
              for (const m of v.missingOptional) console.error(`- ${m}`);
            }
          }
          return;
        }
        if (jsonOutput) {
          outputJson({
            success: false,
            mode: "direct",
            action: "verify",
            database,
            monitoringUser: opts.monitoringUser,
            verified: false,
            missingRequired: v.missingRequired,
            missingOptional: v.missingOptional,
          });
        } else {
          console.error("✗ prepare-db verify failed: missing required items");
          for (const m of v.missingRequired) console.error(`- ${m}`);
          if (v.missingOptional.length > 0) {
            console.error("Optional items missing:");
            for (const m of v.missingOptional) console.error(`- ${m}`);
          }
        }
        process.exitCode = 1;
        return;
      }

      let monPassword: string;
      let passwordGenerated = false;
      try {
        const resolved = await resolveMonitoringPassword({
          passwordFlag: opts.password,
          passwordEnv: process.env.PGAI_MON_PASSWORD,
          monitoringUser: opts.monitoringUser,
        });
        monPassword = resolved.password;
        passwordGenerated = resolved.generated;
        if (resolved.generated) {
          const canPrint = process.stdout.isTTY || !!opts.printPassword || jsonOutput;
          if (canPrint) {
            if (!jsonOutput) {
              // Print secrets to stderr to reduce the chance they end up in piped stdout logs.
              const shellSafe = monPassword.replace(/'/g, "'\\''");
              console.error("");
              console.error(`Generated monitoring password for ${opts.monitoringUser} (copy/paste):`);
              // Quote for shell copy/paste safety.
              console.error(`PGAI_MON_PASSWORD='${shellSafe}'`);
              console.error("");
              console.log("Store it securely (or rerun with --password / PGAI_MON_PASSWORD to set your own).");
            }
            // For JSON mode, password will be included in the success output below
          } else {
            console.error(
              [
                `✗ Monitoring password was auto-generated for ${opts.monitoringUser} but not printed in non-interactive mode.`,
                "",
                "Provide it explicitly:",
                "  --password <password>   or   PGAI_MON_PASSWORD=...",
                "",
                "Or (NOT recommended) print the generated password:",
                "  --print-password",
              ].join("\n")
            );
            process.exitCode = 1;
            return;
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        outputError({ message: msg });
        return;
      }

      const plan = await buildInitPlan({
        database,
        monitoringUser: opts.monitoringUser,
        monitoringPassword: monPassword,
        includeOptionalPermissions,
        provider: opts.provider,
      });

      // For reset-password, we only want the role step. But if provider skips role creation,
      // reset-password doesn't make sense - warn the user.
      const effectivePlan = opts.resetPassword
        ? { ...plan, steps: plan.steps.filter((s) => s.name === "01.role") }
        : plan;

      if (opts.resetPassword && effectivePlan.steps.length === 0) {
        console.error(`✗ --reset-password not supported for provider "${opts.provider}" (role creation is skipped)`);
        process.exitCode = 1;
        return;
      }

      if (shouldPrintSql) {
        console.log("\n--- SQL plan ---");
        for (const step of effectivePlan.steps) {
          console.log(`\n-- ${step.name}${step.optional ? " (optional)" : ""}`);
          console.log(redactPasswords(step.sql));
        }
        console.log("\n--- end SQL plan ---\n");
              console.log("Note: passwords are redacted in the printed SQL output.");
        return;
      }

      const { applied, skippedOptional } = await applyInitPlan({ client, plan: effectivePlan });

      if (jsonOutput) {
        const result: Record<string, unknown> = {
          success: true,
          mode: "direct",
          action: opts.resetPassword ? "reset-password" : "apply",
          database,
          monitoringUser: opts.monitoringUser,
          applied,
          skippedOptional,
          warnings: skippedOptional.length > 0
            ? ["Some optional steps were skipped (not supported or insufficient privileges)"]
            : [],
        };
        if (passwordGenerated) {
          result.generatedPassword = monPassword;
        }
        outputJson(result);
      } else {
        console.log(opts.resetPassword ? "✓ prepare-db password reset completed" : "✓ prepare-db completed");
        if (skippedOptional.length > 0) {
          console.error("⚠ Some optional steps were skipped (not supported or insufficient privileges):");
          for (const s of skippedOptional) console.error(`- ${s}`);
        }
        // Keep output compact but still useful
        if (process.stdout.isTTY) {
          console.log(`Applied ${applied.length} steps`);
        }
      }
    } catch (error) {
      const errAny = error as any;
      let message = "";
      if (error instanceof Error && error.message) {
        message = error.message;
      } else if (errAny && typeof errAny === "object" && typeof errAny.message === "string" && errAny.message) {
        message = errAny.message;
      } else {
        message = String(error);
      }
      if (!message || message === "[object Object]") {
        message = "Unknown error";
      }

      // If this was a plan step failure, surface the step name explicitly to help users diagnose quickly.
      const stepMatch =
        typeof message === "string" ? message.match(/Failed at step "([^"]+)":/i) : null;
      const failedStep = stepMatch?.[1];

      // Build error object for JSON output
      const errorObj: {
        message: string;
        step?: string;
        code?: string;
        detail?: string;
        hint?: string;
      } = { message };

      if (failedStep) errorObj.step = failedStep;
      if (errAny && typeof errAny === "object") {
        if (typeof errAny.code === "string" && errAny.code) errorObj.code = errAny.code;
        if (typeof errAny.detail === "string" && errAny.detail) errorObj.detail = errAny.detail;
        if (typeof errAny.hint === "string" && errAny.hint) errorObj.hint = errAny.hint;
      }

      if (jsonOutput) {
        outputJson({
          success: false,
          mode: "direct",
          error: errorObj,
        });
        process.exitCode = 1;
      } else {
        console.error(`Error: prepare-db: ${message}`);
        if (failedStep) {
          console.error(`  Step: ${failedStep}`);
        }
        if (errAny && typeof errAny === "object") {
          if (typeof errAny.code === "string" && errAny.code) {
            console.error(`  Code: ${errAny.code}`);
          }
          if (typeof errAny.detail === "string" && errAny.detail) {
            console.error(`  Detail: ${errAny.detail}`);
          }
          if (typeof errAny.hint === "string" && errAny.hint) {
            console.error(`  Hint: ${errAny.hint}`);
          }
        }
        if (errAny && typeof errAny === "object" && typeof errAny.code === "string") {
          if (errAny.code === "42501") {
            if (failedStep === "01.role") {
              console.error("  Context: role creation/update requires CREATEROLE or superuser");
            } else if (failedStep === "03.permissions") {
              console.error("  Context: grants/view/search_path require sufficient GRANT/DDL privileges");
            }
            console.error("  Fix: connect as a superuser (or a role with CREATEROLE and sufficient GRANT privileges)");
            console.error("  Fix: on managed Postgres, use the provider's admin/master user");
            console.error("  Tip: run with --print-sql to review the exact SQL plan");
          }
          // Schema already exists (42P06) or other duplicate object errors
          if (errAny.code === "42P06" || (message.includes("already exists") && failedStep === "03.permissions")) {
            console.error("  Hint: postgres_ai schema or objects already exist from a previous setup.");
            console.error("  Fix: run 'postgresai unprepare-db <connection>' first to clean up, then retry prepare-db.");
          }
          if (errAny.code === "ECONNREFUSED") {
            console.error("  Hint: check host/port and ensure Postgres is reachable from this machine");
          }
          if (errAny.code === "ENOTFOUND") {
            console.error("  Hint: DNS resolution failed; double-check the host name");
          }
          if (errAny.code === "ETIMEDOUT") {
            console.error("  Hint: connection timed out; check network/firewall rules");
          }
        }
        process.exitCode = 1;
      }
    } finally {
      if (client) {
        try {
          await client.end();
        } catch {
          // ignore
        }
      }
    }
  });

program
  .command("unprepare-db [conn]")
  .description("remove monitoring setup: drop monitoring user, views, schema, and revoke permissions")
  .option("--db-url <url>", "PostgreSQL connection URL (admin) (deprecated; pass it as positional arg)")
  .option("-h, --host <host>", "PostgreSQL host (psql-like)")
  .option("-p, --port <port>", "PostgreSQL port (psql-like)")
  .option("-U, --username <username>", "PostgreSQL user (psql-like)")
  .option("-d, --dbname <dbname>", "PostgreSQL database name (psql-like)")
  .option("--admin-password <password>", "Admin connection password (otherwise uses PGPASSWORD if set)")
  .option("--monitoring-user <name>", "Monitoring role name to remove", DEFAULT_MONITORING_USER)
  .option("--keep-role", "Keep the monitoring role (only revoke permissions and drop objects)", false)
  .option("--provider <provider>", "Database provider (e.g., supabase). Affects which steps are executed.")
  .option("--print-sql", "Print SQL plan and exit (no changes applied)", false)
  .option("--force", "Skip confirmation prompt", false)
  .option("--json", "Output result as JSON (machine-readable)", false)
  .addHelpText(
    "after",
    [
      "",
      "Examples:",
      "  postgresai unprepare-db postgresql://admin@host:5432/dbname",
      "  postgresai unprepare-db \"dbname=dbname host=host user=admin\"",
      "  postgresai unprepare-db -h host -p 5432 -U admin -d dbname",
      "",
      "Admin password:",
      "  --admin-password <password>   or  PGPASSWORD=... (libpq standard)",
      "",
      "Keep role but remove objects/permissions:",
      "  postgresai unprepare-db <conn> --keep-role",
      "",
      "Inspect SQL without applying changes:",
      "  postgresai unprepare-db <conn> --print-sql",
      "",
      "Offline SQL plan (no DB connection):",
      "  postgresai unprepare-db --print-sql",
      "",
      "Skip confirmation prompt:",
      "  postgresai unprepare-db <conn> --force",
    ].join("\n")
  )
  .action(async (conn: string | undefined, opts: {
    dbUrl?: string;
    host?: string;
    port?: string;
    username?: string;
    dbname?: string;
    adminPassword?: string;
    monitoringUser: string;
    keepRole?: boolean;
    provider?: string;
    printSql?: boolean;
    force?: boolean;
    json?: boolean;
  }, cmd: Command) => {
    // JSON output helper
    const jsonOutput = opts.json;
    const outputJson = (data: Record<string, unknown>) => {
      console.log(JSON.stringify(data, null, 2));
    };
    const outputError = (error: {
      message: string;
      step?: string;
      code?: string;
      detail?: string;
      hint?: string;
    }) => {
      if (jsonOutput) {
        outputJson({
          success: false,
          error,
        });
      } else {
        console.error(`Error: unprepare-db: ${error.message}`);
        if (error.step) console.error(`  Step: ${error.step}`);
        if (error.code) console.error(`  Code: ${error.code}`);
        if (error.detail) console.error(`  Detail: ${error.detail}`);
        if (error.hint) console.error(`  Hint: ${error.hint}`);
      }
      process.exitCode = 1;
    };

    const shouldPrintSql = !!opts.printSql;
    const dropRole = !opts.keepRole;

    // Validate provider and warn if unknown
    const providerWarning = validateProvider(opts.provider);
    if (providerWarning) {
      console.warn(`⚠ ${providerWarning}`);
    }

    // Offline mode: allow printing SQL without providing/using an admin connection.
    if (!conn && !opts.dbUrl && !opts.host && !opts.port && !opts.username && !opts.adminPassword) {
      if (shouldPrintSql) {
        const database = (opts.dbname ?? process.env.PGDATABASE ?? "postgres").trim();

        const plan = await buildUninitPlan({
          database,
          monitoringUser: opts.monitoringUser,
          dropRole,
          provider: opts.provider,
        });

        console.log("\n--- SQL plan (offline; not connected) ---");
        console.log(`-- database: ${database}`);
        console.log(`-- monitoring user: ${opts.monitoringUser}`);
        console.log(`-- provider: ${opts.provider ?? "self-managed"}`);
        console.log(`-- drop role: ${dropRole}`);
        for (const step of plan.steps) {
          console.log(`\n-- ${step.name}`);
          console.log(step.sql);
        }
        console.log("\n--- end SQL plan ---\n");
        return;
      }
    }

    let adminConn;
    try {
      adminConn = resolveAdminConnection({
        conn,
        dbUrlFlag: opts.dbUrl,
        host: opts.host ?? process.env.PGHOST,
        port: opts.port ?? process.env.PGPORT,
        username: opts.username ?? process.env.PGUSER,
        dbname: opts.dbname ?? process.env.PGDATABASE,
        adminPassword: opts.adminPassword,
        envPassword: process.env.PGPASSWORD,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (jsonOutput) {
        outputError({ message: msg });
      } else {
        console.error(`Error: unprepare-db: ${msg}`);
        if (typeof msg === "string" && msg.startsWith("Connection is required.")) {
          console.error("");
          cmd.outputHelp({ error: true });
        }
        process.exitCode = 1;
      }
      return;
    }

    if (!jsonOutput) {
      console.log(`Connecting to: ${adminConn.display}`);
      console.log(`Monitoring user: ${opts.monitoringUser}`);
      console.log(`Drop role: ${dropRole}`);
    }

    // Confirmation prompt (unless --force or --json)
    if (!opts.force && !jsonOutput && !shouldPrintSql) {
      const answer = await new Promise<string>((resolve) => {
        const readline = getReadline();
        readline.question(
          `This will remove the monitoring setup for user "${opts.monitoringUser}"${dropRole ? " and drop the role" : ""}. Continue? [y/N] `,
          (ans) => resolve(ans.trim().toLowerCase())
        );
      });
      if (answer !== "y" && answer !== "yes") {
        console.log("Aborted.");
        return;
      }
    }

    let client: Client | undefined;
    try {
      const connResult = await connectWithSslFallback(Client, adminConn);
      client = connResult.client;

      const dbRes = await client.query("select current_database() as db");
      const database = dbRes.rows?.[0]?.db;
      if (typeof database !== "string" || !database) {
        throw new Error("Failed to resolve current database name");
      }

      const plan = await buildUninitPlan({
        database,
        monitoringUser: opts.monitoringUser,
        dropRole,
        provider: opts.provider,
      });

      if (shouldPrintSql) {
        console.log("\n--- SQL plan ---");
        for (const step of plan.steps) {
          console.log(`\n-- ${step.name}`);
          console.log(step.sql);
        }
        console.log("\n--- end SQL plan ---\n");
        return;
      }

      const { applied, errors } = await applyUninitPlan({ client, plan });

      if (jsonOutput) {
        outputJson({
          success: errors.length === 0,
          action: "unprepare",
          database,
          monitoringUser: opts.monitoringUser,
          dropRole,
          applied,
          errors,
        });
        if (errors.length > 0) {
          process.exitCode = 1;
        }
      } else {
        if (errors.length === 0) {
          console.log("✓ unprepare-db completed");
          console.log(`Applied ${applied.length} steps`);
        } else {
          console.error("⚠ unprepare-db completed with errors");
          console.log(`Applied ${applied.length} steps`);
          console.error("Errors:");
          for (const err of errors) {
            console.error(`  - ${err}`);
          }
          process.exitCode = 1;
        }
      }
    } catch (error) {
      const errAny = error as any;
      let message = "";
      if (error instanceof Error && error.message) {
        message = error.message;
      } else if (errAny && typeof errAny === "object" && typeof errAny.message === "string" && errAny.message) {
        message = errAny.message;
      } else {
        message = String(error);
      }
      if (!message || message === "[object Object]") {
        message = "Unknown error";
      }

      const errorObj: {
        message: string;
        code?: string;
        detail?: string;
        hint?: string;
      } = { message };

      if (errAny && typeof errAny === "object") {
        if (typeof errAny.code === "string" && errAny.code) errorObj.code = errAny.code;
        if (typeof errAny.detail === "string" && errAny.detail) errorObj.detail = errAny.detail;
        if (typeof errAny.hint === "string" && errAny.hint) errorObj.hint = errAny.hint;
      }

      if (jsonOutput) {
        outputJson({
          success: false,
          error: errorObj,
        });
        process.exitCode = 1;
      } else {
        console.error(`Error: unprepare-db: ${message}`);
        if (errAny && typeof errAny === "object") {
          if (typeof errAny.code === "string" && errAny.code) {
            console.error(`  Code: ${errAny.code}`);
          }
          if (typeof errAny.detail === "string" && errAny.detail) {
            console.error(`  Detail: ${errAny.detail}`);
          }
          if (typeof errAny.hint === "string" && errAny.hint) {
            console.error(`  Hint: ${errAny.hint}`);
          }
        }
        if (errAny && typeof errAny === "object" && typeof errAny.code === "string") {
          if (errAny.code === "42501") {
            console.error("  Context: dropping roles/objects requires sufficient privileges");
            console.error("  Fix: connect as a superuser (or a role with appropriate DROP privileges)");
          }
          if (errAny.code === "ECONNREFUSED") {
            console.error("  Hint: check host/port and ensure Postgres is reachable from this machine");
          }
          if (errAny.code === "ENOTFOUND") {
            console.error("  Hint: DNS resolution failed; double-check the host name");
          }
          if (errAny.code === "ETIMEDOUT") {
            console.error("  Hint: connection timed out; check network/firewall rules");
          }
        }
        process.exitCode = 1;
      }
    } finally {
      if (client) {
        try {
          await client.end();
        } catch {
          // ignore
        }
      }
      closeReadline();
    }
  });

program
  .command("checkup [checkIdOrConn] [conn]")
  .description("generate health check reports directly from PostgreSQL (express mode)")
  .option("--check-id <id>", `specific check to run (see list below), or ALL`)
  .option("--node-name <name>", "node name for reports", "node-01")
  .option("--output <path>", "output directory for JSON files")
  .option("--upload", "upload JSON results to PostgresAI (requires API key)")
  .option("--no-upload", "disable upload to PostgresAI")
  .option(
    "--project <project>",
    "project name or ID for remote upload (used with --upload; defaults to config defaultProject; auto-generated on first run)"
  )
  .option("--json", "output JSON to stdout")
  .option("--markdown", "output markdown to stdout")
  .addHelpText(
    "after",
    [
      "",
      "Available checks:",
      ...Object.entries(CHECK_INFO).map(([id, title]) => `  ${id}: ${title}`),
      "",
      "Examples:",
      "  postgresai checkup postgresql://user:pass@host:5432/db",
      "  postgresai checkup H002 postgresql://user:pass@host:5432/db",
      "  postgresai checkup postgresql://user:pass@host:5432/db --check-id H002",
      "  postgresai checkup postgresql://user:pass@host:5432/db --output ./reports",
      "  postgresai checkup postgresql://user:pass@host:5432/db --no-upload --json",
      "  postgresai checkup postgresql://user:pass@host:5432/db --no-upload --markdown",
    ].join("\n")
  )
  .action(async (checkIdOrConn: string | undefined, connArg: string | undefined, opts: CheckupOptions, cmd: Command) => {
    // Support both syntaxes:
    //   pgai checkup postgresql://...              -> run ALL checks
    //   pgai checkup H002 postgresql://...         -> run specific check (positional)
    //   pgai checkup --check-id H002 postgresql:// -> run specific check (option)
    const checkIdPattern = /^[A-Z]\d{3}$/i;
    let conn: string | undefined;
    let checkId: string;

    if (!checkIdOrConn) {
      cmd.outputHelp();
      process.exitCode = 1;
      return;
    }

    if (checkIdPattern.test(checkIdOrConn)) {
      // First arg is a check ID
      checkId = checkIdOrConn.toUpperCase();
      conn = connArg;
      if (!conn) {
        console.error(`Error: Connection string required when specifying check ID "${checkId}"`);
        console.error(`\nUsage: postgresai checkup ${checkId} postgresql://user@host:5432/dbname\n`);
        process.exitCode = 1;
        return;
      }
    } else {
      // First arg is the connection string
      conn = checkIdOrConn;
      checkId = opts.checkId?.toUpperCase() || "ALL";
    }

    if (!conn) {
      cmd.outputHelp();
      process.exitCode = 1;
      return;
    }

    const shouldPrintJson = !!opts.json;
    const shouldConvertMarkdown = !!opts.markdown;
    const uploadExplicitlyRequested = opts.upload === true;

    // Validate mutually exclusive flags
    if (shouldPrintJson && shouldConvertMarkdown) {
      console.error("Error: --json and --markdown are mutually exclusive");
      process.exitCode = 1;
      return;
    }
    // Note: --json, --markdown and --upload/--no-upload are independent flags.
    // Use --no-upload to explicitly disable upload when using --json or --markdown.
    const uploadExplicitlyDisabled = opts.upload === false;
    let shouldUpload = !uploadExplicitlyDisabled;

    // Preflight: validate/create output directory BEFORE connecting / running checks.
    const outputPath = prepareOutputDirectory(opts.output);
    if (outputPath === null) {
      process.exitCode = 1;
      return;
    }

    // Preflight: validate upload flags/credentials BEFORE connecting / running checks.
    const rootOpts = program.opts() as CliOptions;
    const uploadResult = prepareUploadConfig(opts, rootOpts, shouldUpload, uploadExplicitlyRequested);
    if (uploadResult === null) {
      process.exitCode = 1;
      return;
    }
    const uploadCfg = uploadResult?.config;
    const projectWasGenerated = uploadResult?.projectWasGenerated ?? false;
    shouldUpload = !!uploadCfg;

    // Connect and run checks
    const adminConn = resolveAdminConnection({
      conn,
      envPassword: process.env.PGPASSWORD,
    });
    let client: Client | undefined;
    // Show spinner when output is to TTY (not redirected) and not in JSON mode
    const spinnerEnabled = !!process.stdout.isTTY && !shouldPrintJson;
    const spinner = createTtySpinner(spinnerEnabled, "Connecting to Postgres");

    try {
      spinner.update("Connecting to Postgres");
      const connResult = await connectWithSslFallback(Client, adminConn);
      client = connResult.client as Client;

      // Generate reports
      let reports: Record<string, any>;
      if (checkId === "ALL") {
        reports = await generateAllReports(client, opts.nodeName, (p) => {
          spinner.update(`Running ${p.checkId}: ${p.checkTitle} (${p.index}/${p.total})`);
        });
      } else {
        const generator = REPORT_GENERATORS[checkId];
        if (!generator) {
          spinner.stop();
          // Check if it's a valid check ID from the dictionary (just not implemented in express mode)
          const dictEntry = getCheckupEntry(checkId);
          if (dictEntry) {
            console.error(`Check ${checkId} (${dictEntry.title}) is not yet available in express mode.`);
            console.error(`Express-mode checks: ${Object.keys(CHECK_INFO).join(", ")}`);
          } else {
            console.error(`Unknown check ID: ${checkId}`);
            console.error(`See 'postgresai checkup --help' for available checks.`);
          }
          process.exitCode = 1;
          return;
        }
        spinner.update(`Running ${checkId}: ${CHECK_INFO[checkId] || checkId}`);
        reports = { [checkId]: await generator(client, opts.nodeName) };
      }

      // Upload to PostgresAI API (if configured)
      let uploadSummary: UploadSummary | undefined;
      if (uploadCfg) {
        const logUpload = (msg: string): void => {
          (shouldPrintJson ? console.error : console.log)(msg);
        };
        uploadSummary = await uploadCheckupReports(uploadCfg, reports, spinner, logUpload);
      }

      spinner.stop();

      // Write to files (if output path specified)
      if (outputPath) {
        writeReportFiles(reports, outputPath);
      }

      // Print upload summary
      if (uploadSummary) {
        printUploadSummary(uploadSummary, projectWasGenerated, shouldPrintJson || shouldConvertMarkdown, reports);
      }

      // Convert to markdown if requested
      if (shouldConvertMarkdown) {
        let apiKey: string;
        let apiBaseUrl: string;

        try {
          const configResult = getConfig(rootOpts);
          apiKey = configResult.apiKey;
          const cfg = config.readConfig();
          apiBaseUrl = resolveBaseUrls(rootOpts, cfg).apiBaseUrl;
        } catch (error) {
          spinner.stop();
          console.error("Error: Failed to read configuration for markdown conversion");
          console.error(error instanceof Error ? error.message : String(error));
          process.exitCode = 1;
          return;
        }

        // NOTE: apiKey can be empty - this is intentional. The API will return:
        // - Without API key: Partial markdown with observations only (limited functionality)
        // - With API key: Full markdown reports with all details
        // This allows users to get basic insights without requiring authentication.

        const markdownResults: Array<{ checkId: string; markdown?: string; error?: Error }> = [];

        for (const [checkId, report] of Object.entries(reports)) {
          try {
            spinner.update(`Converting ${checkId} to markdown`);

            // For reports that share JSON files (e.g., A002/A013 share a002.json,
            // A003/D001/G003/F001 share a003.json), pass checkId as report_type
            // so the API can generate the correct markdown variant
            const markdownResult = await convertCheckupReportJsonToMarkdown({
              apiKey,
              apiBaseUrl,
              checkId,
              jsonPayload: report,
              reportType: checkId,
            });

            // Extract markdown from response structure
            // API returns: { reports: [{ markdown: "...", ... }], ... }
            const markdown = markdownResult?.reports?.[0]?.markdown || markdownResult?.markdown;

            markdownResults.push({
              checkId,
              markdown,
            });
          } catch (error) {
            markdownResults.push({
              checkId,
              error: error instanceof Error ? error : new Error(String(error)),
            });
          }
        }

        spinner.stop();

        // Output all markdown results
        for (const result of markdownResults) {
          if (result.error) {
            if (result.error instanceof RpcError) {
              console.error(`Error converting ${result.checkId} to markdown:`);
              for (const line of formatRpcErrorForDisplay(result.error)) {
                console.error(line);
              }
            } else {
              console.error(`Error converting ${result.checkId} to markdown: ${result.error.message}`);
            }
          } else if (result.markdown) {
            console.log(result.markdown);
            if (!result.markdown.endsWith('\n')) {
              console.log();
            }
          } else {
            console.error(`Warning: No markdown content returned for ${result.checkId}`);
          }
        }
      }

      // Output JSON to stdout (unless --output is specified, in which case files are written instead)
      if (shouldPrintJson && !outputPath) {
        console.log(JSON.stringify(reports, null, 2));
      }

      // If no output was produced, show summary
      const hadOutput = shouldPrintJson || shouldConvertMarkdown || outputPath || uploadSummary;
      if (!hadOutput) {
        const checkCount = Object.keys(reports).length;
        console.log(`Checkup completed: ${checkCount} check${checkCount > 1 ? 's' : ''}\n`);

        // Collect and filter summaries
        const summaries = [];
        let skippedCount = 0;

        for (const [checkId, report] of Object.entries(reports)) {
          const { status, message } = generateCheckSummary(checkId, report);
          const title = report.checkTitle || checkId;

          // Show if: warning/ok status, or info with concrete data (contains numbers or version info)
          const isSignificant = status !== 'info' || /\d/.test(message) || message.includes('PostgreSQL') || message.includes('Version');

          if (isSignificant) {
            summaries.push({ checkId, title, status, message });
          } else {
            skippedCount++;
          }
        }

        // Print significant checks
        for (const { checkId, title, message } of summaries) {
          console.log(`  ${checkId} (${title}): ${message}`);
        }

        // Show count of other checks
        if (skippedCount > 0) {
          console.log(`  ${skippedCount} other check${skippedCount > 1 ? 's' : ''} completed`);
        }

        console.log('\nFor details:');
        console.log('  --json          Output JSON');
        console.log('  --markdown      Output markdown');
        console.log('  --output <dir>  Save to directory');
      }
    } catch (error) {
      if (error instanceof RpcError) {
        for (const line of formatRpcErrorForDisplay(error)) {
          console.error(line);
        }
      } else {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`Error: ${message}`);
      }
      process.exitCode = 1;
    } finally {
      // Always stop spinner to prevent interval leak (idempotent - safe to call multiple times)
      spinner.stop();
      if (client) {
        await client.end();
      }
    }
  });

/**
 * Stub function for not implemented commands
 */
const stub = (name: string) => async (): Promise<void> => {
  // Temporary stubs until Node parity is implemented
  console.error(`${name}: not implemented in Node CLI yet; use bash CLI for now`);
  process.exitCode = 2;
};

/**
 * Resolve project paths
 */
function resolvePaths(): PathResolution {
  const startDir = process.cwd();
  let currentDir = startDir;

  while (true) {
    const composeFile = path.resolve(currentDir, "docker-compose.yml");
    if (fs.existsSync(composeFile)) {
      const instancesFile = path.resolve(currentDir, "instances.yml");
      return { fs, path, projectDir: currentDir, composeFile, instancesFile };
    }

    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir) break;
    currentDir = parentDir;
  }

  throw new Error(
    `docker-compose.yml not found. Run monitoring commands from the PostgresAI project directory or one of its subdirectories (starting search from ${startDir}).`
  );
}

async function resolveOrInitPaths(): Promise<PathResolution> {
  try {
    return resolvePaths();
  } catch {
    return ensureDefaultMonitoringProject();
  }
}

/**
 * Check if Docker daemon is running
 */
function isDockerRunning(): boolean {
  try {
    // Note: timeout is supported by Bun but not in @types/bun
    const result = spawnSync("docker", ["info"], { stdio: "pipe", timeout: 5000 } as Parameters<typeof spawnSync>[2]);
    return result.status === 0;
  } catch {
    return false;
  }
}

/**
 * Get docker compose command
 */
function getComposeCmd(): string[] | null {
  const tryCmd = (cmd: string, args: string[]): boolean =>
    spawnSync(cmd, args, { stdio: "ignore", timeout: 5000 } as Parameters<typeof spawnSync>[2]).status === 0;
  if (tryCmd("docker-compose", ["version"])) return ["docker-compose"];
  if (tryCmd("docker", ["compose", "version"])) return ["docker", "compose"];
  return null;
}

/**
 * Check if monitoring containers are already running
 */
function checkRunningContainers(): { running: boolean; containers: string[] } {
  try {
    const result = spawnSync(
      "docker",
      ["ps", "--filter", "name=grafana-with-datasources", "--filter", "name=pgwatch", "--format", "{{.Names}}"],
      { stdio: "pipe", encoding: "utf8", timeout: 5000 } as Parameters<typeof spawnSync>[2]
    );

    if (result.status === 0 && result.stdout) {
      const containers = result.stdout.trim().split("\n").filter(Boolean);
      return { running: containers.length > 0, containers };
    }
    return { running: false, containers: [] };
  } catch {
    return { running: false, containers: [] };
  }
}

/**
 * Register monitoring instance with the API (non-blocking).
 * Returns immediately, logs result in background.
 */
function registerMonitoringInstance(
  apiKey: string,
  projectName: string,
  opts?: { apiBaseUrl?: string; debug?: boolean }
): void {
  const { apiBaseUrl } = resolveBaseUrls(opts);
  const url = `${apiBaseUrl}/rpc/monitoring_instance_register`;
  const debug = opts?.debug;

  if (debug) {
    console.error(`\nDebug: Registering monitoring instance...`);
    console.error(`Debug: POST ${url}`);
    console.error(`Debug: project_name=${projectName}`);
  }

  // Fire and forget - don't block the main flow
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      api_token: apiKey,
      project_name: projectName,
    }),
  })
    .then(async (res) => {
      const body = await res.text().catch(() => "");
      if (!res.ok) {
        if (debug) {
          console.error(`Debug: Monitoring registration failed: HTTP ${res.status}`);
          console.error(`Debug: Response: ${body}`);
        }
        return;
      }
      if (debug) {
        console.error(`Debug: Monitoring registration response: ${body}`);
      }
    })
    .catch((err) => {
      if (debug) {
        console.error(`Debug: Monitoring registration error: ${err.message}`);
      }
    });
}

/**
 * Update .pgwatch-config file with key=value pairs.
 * Preserves existing values not being updated.
 */
function updatePgwatchConfig(configPath: string, updates: Record<string, string>): void {
  let lines: string[] = [];

  // Read existing config if it exists
  if (fs.existsSync(configPath)) {
    const stats = fs.statSync(configPath);
    if (!stats.isDirectory()) {
      const content = fs.readFileSync(configPath, "utf8");
      lines = content.split(/\r?\n/).filter(l => l.trim() !== "");
    }
  }

  // Update or add each key
  for (const [key, value] of Object.entries(updates)) {
    const existingIndex = lines.findIndex(l => l.startsWith(key + "="));
    if (existingIndex >= 0) {
      lines[existingIndex] = `${key}=${value}`;
    } else {
      lines.push(`${key}=${value}`);
    }
  }

  fs.writeFileSync(configPath, lines.join("\n") + "\n", { encoding: "utf8", mode: 0o600 });
}

/**
 * Run docker compose command
 */
async function runCompose(args: string[], grafanaPassword?: string): Promise<number> {
  let composeFile: string;
  let projectDir: string;
  try {
    ({ composeFile, projectDir } = await resolveOrInitPaths());
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exitCode = 1;
    return 1;
  }

  // Check if Docker daemon is running
  if (!isDockerRunning()) {
    console.error("Docker is not running. Please start Docker and try again");
    process.exitCode = 1;
    return 1;
  }

  const cmd = getComposeCmd();
  if (!cmd) {
    console.error("docker compose not found (need docker-compose or docker compose)");
    process.exitCode = 1;
    return 1;
  }

  // Set Grafana password from parameter or .pgwatch-config
  const env = { ...process.env };
  if (grafanaPassword) {
    env.GF_SECURITY_ADMIN_PASSWORD = grafanaPassword;
  } else {
    const cfgPath = path.resolve(projectDir, ".pgwatch-config");
    if (fs.existsSync(cfgPath)) {
      try {
        const stats = fs.statSync(cfgPath);
        if (!stats.isDirectory()) {
          const content = fs.readFileSync(cfgPath, "utf8");
          const match = content.match(/^grafana_password=([^\r\n]+)/m);
          if (match) {
            env.GF_SECURITY_ADMIN_PASSWORD = match[1].trim();
          }
        }
      } catch (err) {
        // If we can't read the config, log warning and continue without setting the password
        if (process.env.DEBUG) {
          console.warn(`Warning: Could not read Grafana password from config: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }
  }

  // Load VM auth credentials from .env if not already set
  const envFilePath = path.resolve(projectDir, ".env");
  if (fs.existsSync(envFilePath)) {
    try {
      const envContent = fs.readFileSync(envFilePath, "utf8");
      if (!env.VM_AUTH_USERNAME) {
        const m = envContent.match(/^VM_AUTH_USERNAME=([^\r\n]+)/m);
        if (m) env.VM_AUTH_USERNAME = m[1].trim().replace(/^["']|["']$/g, '');
      }
      if (!env.VM_AUTH_PASSWORD) {
        const m = envContent.match(/^VM_AUTH_PASSWORD=([^\r\n]+)/m);
        if (m) env.VM_AUTH_PASSWORD = m[1].trim().replace(/^["']|["']$/g, '');
      }
    } catch (err) {
      if (process.env.DEBUG) {
        console.warn(`Warning: Could not read VM auth from .env: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // On macOS, self-node-exporter can't mount host root filesystem - skip it
  const finalArgs = [...args];
  if (process.platform === "darwin" && args.includes("up")) {
    finalArgs.push("--scale", "self-node-exporter=0");
  }

  return new Promise<number>((resolve) => {
    const child = spawn(cmd[0], [...cmd.slice(1), "-f", composeFile, ...finalArgs], {
      stdio: "inherit",
      env: env,
      cwd: projectDir
    });
    child.on("close", (code) => resolve(code || 0));
  });
}

program.command("help", { isDefault: true }).description("show help").action(() => {
  program.outputHelp();
});

// Monitoring services management
const mon = program.command("mon").description("monitoring services management");

mon
  .command("local-install")
  .description("install local monitoring stack (generate config, start services)")
  .option("--demo", "demo mode with sample database", false)
  .option("--api-key <key>", "Postgres AI API key for automated report uploads")
  .option("--db-url <url>", "PostgreSQL connection URL to monitor")
  .option("--tag <tag>", "Docker image tag to use (e.g., 0.14.0, 0.14.0-dev.33)")
  .option("--project <name>", "Docker Compose project name (default: postgres_ai)")
  .option("-y, --yes", "accept all defaults and skip interactive prompts", false)
  .action(async (opts: { demo: boolean; apiKey?: string; dbUrl?: string; tag?: string; project?: string; yes: boolean }) => {
    // Get apiKey from global program options (--api-key is defined globally)
    // This is needed because Commander.js routes --api-key to the global option, not the subcommand's option
    const globalOpts = program.opts<CliOptions>();
    let apiKey = opts.apiKey || globalOpts.apiKey;

    console.log("\n=================================");
    console.log("  PostgresAI monitoring local install");
    console.log("=================================\n");
    console.log("This will install, configure, and start the monitoring system\n");

    // Ensure we have a project directory with docker-compose.yml even if running from elsewhere
    const { projectDir } = await resolveOrInitPaths();
    console.log(`Project directory: ${projectDir}\n`);

    // Save project name to .pgwatch-config if provided (used by reporter container)
    if (opts.project) {
      const cfgPath = path.resolve(projectDir, ".pgwatch-config");
      updatePgwatchConfig(cfgPath, { project_name: opts.project });
      console.log(`Using project name: ${opts.project}\n`);
    }

    // Update .env with custom tag if provided
    const envFile = path.resolve(projectDir, ".env");

    // Build .env content, preserving important existing values (registry, password)
    // Note: PGAI_TAG is intentionally NOT preserved - the CLI version should always match Docker images
    let existingRegistry: string | null = null;
    let existingPassword: string | null = null;

    if (fs.existsSync(envFile)) {
      const existingEnv = fs.readFileSync(envFile, "utf8");
      // Extract existing values (except tag - always use CLI version)
      const registryMatch = existingEnv.match(/^PGAI_REGISTRY=(.+)$/m);
      if (registryMatch) existingRegistry = registryMatch[1].trim();
      const pwdMatch = existingEnv.match(/^GF_SECURITY_ADMIN_PASSWORD=(.+)$/m);
      if (pwdMatch) existingPassword = pwdMatch[1].trim();
    }

    // Priority: CLI --tag flag > package version
    // Note: We intentionally do NOT use process.env.PGAI_TAG here because Bun auto-loads .env files,
    // which would cause stale .env values to override the CLI version. The CLI version should always
    // match the Docker images. Users can override with --tag if needed.
    const imageTag = opts.tag || pkg.version;

    const envLines: string[] = [`PGAI_TAG=${imageTag}`];
    if (existingRegistry) {
      envLines.push(`PGAI_REGISTRY=${existingRegistry}`);
    }
    if (existingPassword) {
      envLines.push(`GF_SECURITY_ADMIN_PASSWORD=${existingPassword}`);
    }
    fs.writeFileSync(envFile, envLines.join("\n") + "\n", { encoding: "utf8", mode: 0o600 });

    if (opts.tag) {
      console.log(`Using image tag: ${imageTag}\n`);
    }

    // Validate conflicting options
    if (opts.demo && opts.dbUrl) {
      console.error("⚠ Both --demo and --db-url provided. Demo mode includes its own database.");
      console.error("⚠ The --db-url will be ignored in demo mode.\n");
      opts.dbUrl = undefined;
    }

    if (opts.demo && apiKey) {
      console.error("✗ Cannot use --api-key with --demo mode");
      console.error("✗ Demo mode is for testing only and does not support API key integration");
      console.error("\nUse demo mode without API key: postgres-ai mon local-install --demo");
      console.error("Or use production mode with API key: postgres-ai mon local-install --api-key=your_key");
      process.exitCode = 1;
      return;
    }

    // Check if containers are already running
    const { running, containers } = checkRunningContainers();
    if (running) {
      console.error(`⚠ Monitoring services are already running: ${containers.join(", ")}`);
      console.log("Use 'postgres-ai mon restart' to restart them\n");
      return;
    }

    // Step 1: API key configuration (only in production mode)
    if (!opts.demo) {
      console.log("Step 1: Postgres AI API Configuration (Optional)");
      console.log("An API key enables automatic upload of PostgreSQL reports to Postgres AI\n");

      if (apiKey) {
        console.log("Using API key provided via --api-key parameter");
        config.writeConfig({ apiKey });
        // Keep reporter compatibility (docker-compose mounts .pgwatch-config)
        updatePgwatchConfig(path.resolve(projectDir, ".pgwatch-config"), { api_key: apiKey });
        console.log("✓ API key saved\n");
      } else if (opts.yes) {
        // Auto-yes mode without API key - skip API key setup
        console.log("Auto-yes mode: no API key provided, skipping API key setup");
        console.error("⚠ Reports will be generated locally only");
        console.log("You can add an API key later with: postgres-ai add-key <api_key>\n");
      } else {
        const answer = await question("Do you have a Postgres AI API key? (Y/n): ");
        const proceedWithApiKey = !answer || answer.toLowerCase() === "y";

        if (proceedWithApiKey) {
          while (true) {
            const inputApiKey = await question("Enter your Postgres AI API key: ");
            const trimmedKey = inputApiKey.trim();

            if (trimmedKey) {
              config.writeConfig({ apiKey: trimmedKey });
              // Keep reporter compatibility (docker-compose mounts .pgwatch-config)
              updatePgwatchConfig(path.resolve(projectDir, ".pgwatch-config"), { api_key: trimmedKey });
              apiKey = trimmedKey;  // Update for later use in registerMonitoringInstance
              console.log("✓ API key saved\n");
              break;
            }

            console.error("⚠ API key cannot be empty");
            const retry = await question("Try again or skip API key setup, retry? (Y/n): ");
            if (retry.toLowerCase() === "n") {
              console.error("⚠ Skipping API key setup - reports will be generated locally only");
              console.log("You can add an API key later with: postgres-ai add-key <api_key>\n");
              break;
            }
          }
        } else {
          console.error("⚠ Skipping API key setup - reports will be generated locally only");
          console.log("You can add an API key later with: postgres-ai add-key <api_key>\n");
        }
      }
    } else {
      console.log("Step 1: Demo mode - API key configuration skipped");
      console.log("Demo mode is for testing only and does not support API key integration\n");
    }

    // Step 2: Add PostgreSQL instance (if not demo mode)
    if (!opts.demo) {
      console.log("Step 2: Add PostgreSQL Instance to Monitor\n");

      // Clear instances.yml in production mode (start fresh)
      const { instancesFile: instancesPath, projectDir } = await resolveOrInitPaths();
      const emptyInstancesContent = "# PostgreSQL instances to monitor\n# Add your instances using: postgres-ai mon targets add\n\n";
      fs.writeFileSync(instancesPath, emptyInstancesContent, "utf8");
      console.log(`Instances file: ${instancesPath}`);
      console.log(`Project directory: ${projectDir}\n`);

      if (opts.dbUrl) {
        console.log("Using database URL provided via --db-url parameter");
        console.log(`Adding PostgreSQL instance from: ${opts.dbUrl}\n`);

        const match = opts.dbUrl.match(/^postgresql:\/\/[^@]+@([^:/]+)/);
        const autoInstanceName = match ? match[1] : "db-instance";

        const connStr = opts.dbUrl;
        const m = connStr.match(/^postgresql:\/\/([^:]+):([^@]+)@([^:\/]+)(?::(\d+))?\/(.+)$/);

        if (!m) {
          console.error("✗ Invalid connection string format");
          process.exitCode = 1;
          return;
        }

        const host = m[3];
        const db = m[5];
        const instanceName = `${host}-${db}`.replace(/[^a-zA-Z0-9-]/g, "-");

        const body = `- name: ${instanceName}\n  conn_str: ${connStr}\n  preset_metrics: full\n  custom_metrics:\n  is_enabled: true\n  group: default\n  custom_tags:\n    env: production\n    cluster: default\n    node_name: ${instanceName}\n    sink_type: ~sink_type~\n`;
        fs.appendFileSync(instancesPath, body, "utf8");
        console.log(`✓ Monitoring target '${instanceName}' added\n`);

        // Test connection
        console.log("Testing connection to the added instance...");
        try {
          const client = new Client({ connectionString: connStr });
          await client.connect();
          const result = await client.query("select version();");
          console.log("✓ Connection successful");
          console.log(`${result.rows[0].version}\n`);
          await client.end();
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.error(`✗ Connection failed: ${message}\n`);
        }
      } else if (opts.yes) {
        // Auto-yes mode without database URL - skip database setup
        console.log("Auto-yes mode: no database URL provided, skipping database setup");
        console.error("⚠ No PostgreSQL instance added");
        console.log("You can add one later with: postgres-ai mon targets add\n");
      } else {
        console.log("You need to add at least one PostgreSQL instance to monitor");
        const answer = await question("Do you want to add a PostgreSQL instance now? (Y/n): ");
        const proceedWithInstance = !answer || answer.toLowerCase() === "y";

        if (proceedWithInstance) {
          console.log("\nYou can provide either:");
          console.log("  1. A full connection string: postgresql://user:pass@host:port/database");
          console.log("  2. Press Enter to skip for now\n");

          const connStr = await question("Enter connection string (or press Enter to skip): ");

          if (connStr.trim()) {
            const m = connStr.match(/^postgresql:\/\/([^:]+):([^@]+)@([^:\/]+)(?::(\d+))?\/(.+)$/);
            if (!m) {
              console.error("✗ Invalid connection string format");
              console.error("⚠ Continuing without adding instance\n");
            } else {
              const host = m[3];
              const db = m[5];
              const instanceName = `${host}-${db}`.replace(/[^a-zA-Z0-9-]/g, "-");

              const body = `- name: ${instanceName}\n  conn_str: ${connStr}\n  preset_metrics: full\n  custom_metrics:\n  is_enabled: true\n  group: default\n  custom_tags:\n    env: production\n    cluster: default\n    node_name: ${instanceName}\n    sink_type: ~sink_type~\n`;
              fs.appendFileSync(instancesPath, body, "utf8");
              console.log(`✓ Monitoring target '${instanceName}' added\n`);

              // Test connection
              console.log("Testing connection to the added instance...");
              try {
                const client = new Client({ connectionString: connStr });
                await client.connect();
                const result = await client.query("select version();");
                console.log("✓ Connection successful");
                console.log(`${result.rows[0].version}\n`);
                await client.end();
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                console.error(`✗ Connection failed: ${message}\n`);
              }
            }
          } else {
            console.error("⚠ No PostgreSQL instance added - you can add one later with: postgres-ai mon targets add\n");
          }
        } else {
          console.error("⚠ No PostgreSQL instance added - you can add one later with: postgres-ai mon targets add\n");
        }
      }
    } else {
      console.log("Step 2: Demo mode enabled - using included demo PostgreSQL database\n");
    }

    // Step 3: Update configuration
    console.log(opts.demo ? "Step 3: Updating configuration..." : "Step 3: Updating configuration...");
    const code1 = await runCompose(["run", "--rm", "sources-generator"]);
    if (code1 !== 0) {
      process.exitCode = code1;
      return;
    }
    console.log("✓ Configuration updated\n");

    // Step 4: Ensure Grafana password is configured
    console.log(opts.demo ? "Step 4: Configuring Grafana security..." : "Step 4: Configuring Grafana security...");
    const cfgPath = path.resolve(projectDir, ".pgwatch-config");
    let grafanaPassword = "";
    let vmAuthUsername = "";
    let vmAuthPassword = "";

    try {
      if (fs.existsSync(cfgPath)) {
        const stats = fs.statSync(cfgPath);
        if (!stats.isDirectory()) {
          const content = fs.readFileSync(cfgPath, "utf8");
          const match = content.match(/^grafana_password=([^\r\n]+)/m);
          if (match) {
            grafanaPassword = match[1].trim();
          }
        }
      }

      if (!grafanaPassword) {
        console.log("Generating secure Grafana password...");
        const { stdout: password } = await execPromise("openssl rand -base64 12 | tr -d '\n'");
        grafanaPassword = password.trim();

        let configContent = "";
        if (fs.existsSync(cfgPath)) {
          const stats = fs.statSync(cfgPath);
          if (!stats.isDirectory()) {
            configContent = fs.readFileSync(cfgPath, "utf8");
          }
        }

        const lines = configContent.split(/\r?\n/).filter((l) => !/^grafana_password=/.test(l));
        lines.push(`grafana_password=${grafanaPassword}`);
        fs.writeFileSync(cfgPath, lines.filter(Boolean).join("\n") + "\n", "utf8");
      }

      console.log("✓ Grafana password configured\n");
    } catch (error) {
      console.error("⚠ Could not generate Grafana password automatically");
      console.log("Using default password: demo\n");
      grafanaPassword = "demo";
    }

    // Generate VictoriaMetrics auth credentials
    try {
      const envFile = path.resolve(projectDir, ".env");

      // Read existing VM auth from .env if present
      if (fs.existsSync(envFile)) {
        const envContent = fs.readFileSync(envFile, "utf8");
        const userMatch = envContent.match(/^VM_AUTH_USERNAME=([^\r\n]+)/m);
        const passMatch = envContent.match(/^VM_AUTH_PASSWORD=([^\r\n]+)/m);
        if (userMatch) vmAuthUsername = userMatch[1].trim().replace(/^["']|["']$/g, '');
        if (passMatch) vmAuthPassword = passMatch[1].trim().replace(/^["']|["']$/g, '');
      }

      if (!vmAuthUsername || !vmAuthPassword) {
        console.log("Generating VictoriaMetrics auth credentials...");
        vmAuthUsername = vmAuthUsername || "vmauth";
        if (!vmAuthPassword) {
          const { stdout: vmPass } = await execPromise("openssl rand -base64 12 | tr -d '\n'");
          vmAuthPassword = vmPass.trim();
        }

        // Update .env file with VM auth credentials
        let envContent = "";
        if (fs.existsSync(envFile)) {
          envContent = fs.readFileSync(envFile, "utf8");
        }
        const envLines = envContent.split(/\r?\n/)
          .filter((l) => !/^VM_AUTH_USERNAME=/.test(l) && !/^VM_AUTH_PASSWORD=/.test(l))
          .filter((l, i, arr) => !(i === arr.length - 1 && l === ''));
        envLines.push(`VM_AUTH_USERNAME=${vmAuthUsername}`);
        envLines.push(`VM_AUTH_PASSWORD=${vmAuthPassword}`);
        fs.writeFileSync(envFile, envLines.join("\n") + "\n", { encoding: "utf8", mode: 0o600 });
      }

      console.log("✓ VictoriaMetrics auth configured\n");
    } catch (error) {
      console.error("⚠ Could not generate VictoriaMetrics auth credentials automatically");
      if (process.env.DEBUG) {
        console.warn(`  ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    // Step 5: Start services
    console.log("Step 5: Starting monitoring services...");
    const code2 = await runCompose(["up", "-d", "--force-recreate"], grafanaPassword);
    if (code2 !== 0) {
      process.exitCode = code2;
      return;
    }
    console.log("✓ Services started\n");

    // Register monitoring instance with API (non-blocking, only if API key is configured)
    if (apiKey && !opts.demo) {
      const projectName = opts.project || "postgres-ai-monitoring";
      registerMonitoringInstance(apiKey, projectName, {
        apiBaseUrl: globalOpts.apiBaseUrl,
        debug: !!process.env.DEBUG,
      });
    }

    // Final summary
    console.log("=================================");
    console.log("  Local install completed!");
    console.log("=================================\n");

    console.log("What's running:");
    if (opts.demo) {
      console.log("  ✅ Demo PostgreSQL database (monitoring target)");
    }
    console.log("  ✅ PostgreSQL monitoring infrastructure");
    console.log("  ✅ Grafana dashboards (with secure password)");
    console.log("  ✅ Prometheus metrics storage");
    console.log("  ✅ Flask API backend");
    console.log("  ✅ Automated report generation (every 24h)");
    console.log("  ✅ Host stats monitoring (CPU, memory, disk, I/O)\n");

    if (!opts.demo) {
      console.log("Next steps:");
      console.log("  • Add more PostgreSQL instances: postgres-ai mon targets add");
      console.log("  • View configured instances: postgres-ai mon targets list");
      console.log("  • Check service health: postgres-ai mon health\n");
    } else {
      console.log("Demo mode next steps:");
      console.log("  • Explore Grafana dashboards at http://localhost:3000");
      console.log("  • Connect to demo database: postgresql://postgres:postgres@localhost:55432/target_database");
      console.log("  • Generate some load on the demo database to see metrics\n");
    }

    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    console.log("🚀 MAIN ACCESS POINT - Start here:");
    console.log("   Grafana Dashboard: http://localhost:3000");
    console.log(`   Login: monitor / ${grafanaPassword}`);
    if (vmAuthUsername && vmAuthPassword) {
      console.log(`   VictoriaMetrics Auth: ${vmAuthUsername} / ${vmAuthPassword}`);
    }
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  });

mon
  .command("start")
  .description("start monitoring services")
  .action(async () => {
    // Check if containers are already running
    const { running, containers } = checkRunningContainers();
    if (running) {
      console.log(`Monitoring services are already running: ${containers.join(", ")}`);
      console.log("Use 'postgres-ai mon restart' to restart them");
      return;
    }

    const code = await runCompose(["up", "-d"]);
    if (code !== 0) process.exitCode = code;
  });

// Known container names for cleanup
const MONITORING_CONTAINERS = [
  "postgres-ai-config-init",
  "self-node-exporter",
  "self-cadvisor",
  "grafana-with-datasources",
  "sink-postgres",
  "sink-prometheus",
  "target-db",
  "pgwatch-postgres",
  "pgwatch-prometheus",
  "self-postgres-exporter",
  "flask-pgss-api",
  "sources-generator",
  "postgres-reports",
];

/**
 * Network cleanup constants.
 * Docker Compose creates a default network named "{project}_default".
 * In CI environments, network cleanup can fail if containers are slow to disconnect.
 */
const COMPOSE_PROJECT_NAME = "postgres_ai";
const DOCKER_NETWORK_NAME = `${COMPOSE_PROJECT_NAME}_default`;
/** Delay before retrying network cleanup (allows container network disconnections to complete) */
const NETWORK_CLEANUP_DELAY_MS = 2000;

/** Remove orphaned containers that docker compose down might miss */
async function removeOrphanedContainers(): Promise<void> {
  for (const container of MONITORING_CONTAINERS) {
    try {
      await execFilePromise("docker", ["rm", "-f", container]);
    } catch {
      // Container doesn't exist, ignore
    }
  }
}

mon
  .command("stop")
  .description("stop monitoring services")
  .action(async () => {
    // Multi-stage cleanup strategy for reliable shutdown in CI environments:
    // Stage 1: Standard compose down with orphan removal
    // Stage 2: Force remove any orphaned containers, then retry compose down
    // Stage 3: Force remove the Docker network directly
    // This handles edge cases where containers are slow to disconnect from networks.
    let code = await runCompose(["down", "--remove-orphans"]);

    // Stage 2: If initial cleanup fails, try removing orphaned containers first
    if (code !== 0) {
      await removeOrphanedContainers();
      // Wait a moment for container network disconnections to complete
      await new Promise(resolve => setTimeout(resolve, NETWORK_CLEANUP_DELAY_MS));
      // Retry compose down
      code = await runCompose(["down", "--remove-orphans"]);
    }

    // Final cleanup: force remove the network if it still exists
    if (code !== 0) {
      try {
        await execFilePromise("docker", ["network", "rm", DOCKER_NETWORK_NAME]);
        // Network removal succeeded - cleanup is complete
        code = 0;
      } catch {
        // Network doesn't exist or couldn't be removed, ignore
      }
    }

    if (code !== 0) process.exitCode = code;
  });

mon
  .command("restart [service]")
  .description("restart all monitoring services or specific service")
  .action(async (service?: string) => {
    const args = ["restart"];
    if (service) args.push(service);
    const code = await runCompose(args);
    if (code !== 0) process.exitCode = code;
  });

mon
  .command("status")
  .description("show monitoring services status")
  .action(async () => {
    const code = await runCompose(["ps"]);
    if (code !== 0) process.exitCode = code;
  });

mon
  .command("logs [service]")
  .option("-f, --follow", "follow logs", false)
  .option("--tail <lines>", "number of lines to show from the end of logs", "all")
  .description("show logs for all or specific monitoring service")
  .action(async (service: string | undefined, opts: { follow: boolean; tail: string }) => {
    const args: string[] = ["logs"];
    if (opts.follow) args.push("-f");
    if (opts.tail) args.push("--tail", opts.tail);
    if (service) args.push(service);
    const code = await runCompose(args);
    if (code !== 0) process.exitCode = code;
  });
mon
  .command("health")
  .description("health check for monitoring services")
  .option("--wait <seconds>", "wait time in seconds for services to become healthy", parseInt, 0)
  .action(async (opts: { wait: number }) => {
    const services = [
      { name: "Grafana", container: "grafana-with-datasources" },
      { name: "Prometheus", container: "sink-prometheus" },
      { name: "PGWatch (Postgres)", container: "pgwatch-postgres" },
      { name: "PGWatch (Prometheus)", container: "pgwatch-prometheus" },
      { name: "Target DB", container: "target-db" },
      { name: "Sink Postgres", container: "sink-postgres" },
    ];

    const waitTime = opts.wait || 0;
    const maxAttempts = waitTime > 0 ? Math.ceil(waitTime / 5) : 1;

    console.log("Checking service health...\n");

    let allHealthy = false;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if (attempt > 1) {
        console.log(`Retrying (attempt ${attempt}/${maxAttempts})...\n`);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }

      allHealthy = true;
      for (const service of services) {
        try {
          const result = spawnSync("docker", ["inspect", "-f", "{{.State.Status}}", service.container], { stdio: "pipe" });
          const status = result.stdout.trim();

          if (result.status === 0 && status === 'running') {
            console.log(`✓ ${service.name}: healthy`);
          } else if (result.status === 0) {
            console.log(`✗ ${service.name}: unhealthy (status: ${status})`);
            allHealthy = false;
          } else {
            console.log(`✗ ${service.name}: unreachable`);
            allHealthy = false;
          }
        } catch (error) {
          console.log(`✗ ${service.name}: unreachable`);
          allHealthy = false;
        }
      }

      if (allHealthy) {
        break;
      }
    }

    console.log("");
    if (allHealthy) {
      console.log("All services are healthy");
    } else {
      console.log("Some services are unhealthy");
      process.exitCode = 1;
    }
  });
mon
  .command("config")
  .description("show monitoring services configuration")
  .action(async () => {
    let projectDir: string;
    let composeFile: string;
    let instancesFile: string;
    try {
      ({ projectDir, composeFile, instancesFile } = await resolveOrInitPaths());
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(message);
      process.exitCode = 1;
      return;
    }
    console.log(`Project Directory: ${projectDir}`);
    console.log(`Docker Compose File: ${composeFile}`);
    console.log(`Instances File: ${instancesFile}`);
    if (fs.existsSync(instancesFile)) {
      console.log("\nInstances configuration:\n");
      const text = fs.readFileSync(instancesFile, "utf8");
      process.stdout.write(text);
      if (!/\n$/.test(text)) console.log();
    }
  });
mon
  .command("update-config")
  .description("apply monitoring services configuration (generate sources)")
  .action(async () => {
    const code = await runCompose(["run", "--rm", "sources-generator"]);
    if (code !== 0) process.exitCode = code;
  });
mon
  .command("update")
  .description("update monitoring stack")
  .action(async () => {
    console.log("Updating PostgresAI monitoring stack...\n");

    try {
      // Check if we're in a git repo
      const gitDir = path.resolve(process.cwd(), ".git");
      if (!fs.existsSync(gitDir)) {
        console.error("Not a git repository. Cannot update.");
        process.exitCode = 1;
        return;
      }

      // Fetch latest changes
      console.log("Fetching latest changes...");
      await execPromise("git fetch origin");

      // Check current branch
      const { stdout: branch } = await execPromise("git rev-parse --abbrev-ref HEAD");
      const currentBranch = branch.trim();
      console.log(`Current branch: ${currentBranch}`);

      // Pull latest changes
      console.log("Pulling latest changes...");
      const { stdout: pullOut } = await execPromise("git pull origin " + currentBranch);
      console.log(pullOut);

      // Update Docker images
      console.log("\nUpdating Docker images...");
      const code = await runCompose(["pull"]);

      if (code === 0) {
        console.log("\n✓ Update completed successfully");
        console.log("\nTo apply updates, restart monitoring services:");
        console.log("  postgres-ai mon restart");
      } else {
        console.error("\n✗ Docker image update failed");
        process.exitCode = 1;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Update failed: ${message}`);
      process.exitCode = 1;
    }
  });
mon
  .command("reset [service]")
  .description("reset all or specific monitoring service")
  .action(async (service?: string) => {
    try {
      if (service) {
        // Reset specific service
        console.log(`\nThis will stop '${service}', remove its volume, and restart it.`);
        console.log("All data for this service will be lost!\n");

        const answer = await question("Continue? (y/N): ");
        if (answer.toLowerCase() !== "y") {
          console.log("Cancelled");
          return;
        }

        console.log(`\nStopping ${service}...`);
        await runCompose(["stop", service]);

        console.log(`Removing volume for ${service}...`);
        await runCompose(["rm", "-f", "-v", service]);

        console.log(`Restarting ${service}...`);
        const code = await runCompose(["up", "-d", service]);

        if (code === 0) {
          console.log(`\n✓ Service '${service}' has been reset`);
        } else {
          console.error(`\n✗ Failed to restart '${service}'`);
          process.exitCode = 1;
        }
      } else {
        // Reset all services
        console.log("\nThis will stop all services and remove all data!");
        console.log("Volumes, networks, and containers will be deleted.\n");

        const answer = await question("Continue? (y/N): ");
        if (answer.toLowerCase() !== "y") {
          console.log("Cancelled");
          return;
        }

        console.log("\nStopping services and removing data...");
        const downCode = await runCompose(["down", "-v"]);

        if (downCode === 0) {
          console.log("✓ Environment reset completed - all containers and data removed");
        } else {
          console.error("✗ Reset failed");
          process.exitCode = 1;
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Reset failed: ${message}`);
      process.exitCode = 1;
    }
  });
mon
  .command("clean")
  .description("cleanup monitoring services artifacts (stops services and removes volumes)")
  .option("--keep-volumes", "keep data volumes (only stop and remove containers)")
  .action(async (options: { keepVolumes?: boolean }) => {
    console.log("Cleaning up monitoring services...\n");

    try {
      // First, use docker-compose down to properly stop and remove containers/volumes
      const downArgs = options.keepVolumes ? ["down"] : ["down", "-v"];
      console.log(options.keepVolumes
        ? "Stopping and removing containers (keeping volumes)..."
        : "Stopping and removing containers and volumes...");

      const downCode = await runCompose(downArgs);
      if (downCode === 0) {
        console.log("✓ Monitoring services stopped and removed");
      } else {
        console.error("⚠ Could not stop services (may not be running)");
      }

      // Remove any orphaned containers that docker compose down missed
      await removeOrphanedContainers();
      console.log("✓ Removed orphaned containers");

      // Remove orphaned volumes from previous installs with different project names
      if (!options.keepVolumes) {
        const volumePatterns = [
          "monitoring_grafana_data",
          "monitoring_postgres_ai_configs",
          "monitoring_sink_postgres_data",
          "monitoring_target_db_data",
          "monitoring_victoria_metrics_data",
          "postgres_ai_configs_grafana_data",
          "postgres_ai_configs_sink_postgres_data",
          "postgres_ai_configs_target_db_data",
          "postgres_ai_configs_victoria_metrics_data",
          "postgres_ai_configs_postgres_ai_configs",
        ];

        const { stdout: existingVolumes } = await execFilePromise("docker", ["volume", "ls", "-q"]);
        const volumeList = existingVolumes.trim().split('\n').filter(Boolean);
        const orphanedVolumes = volumeList.filter(v => volumePatterns.includes(v));

        if (orphanedVolumes.length > 0) {
          let removedCount = 0;
          for (const vol of orphanedVolumes) {
            try {
              await execFilePromise("docker", ["volume", "rm", vol]);
              removedCount++;
            } catch {
              // Volume might be in use, skip silently
            }
          }
          if (removedCount > 0) {
            console.log(`✓ Removed ${removedCount} orphaned volume(s) from previous installs`);
          }
        }
      }

      // Remove any dangling resources
      await execFilePromise("docker", ["network", "prune", "-f"]);
      console.log("✓ Removed unused networks");

      await execFilePromise("docker", ["image", "prune", "-f"]);
      console.log("✓ Removed dangling images");

      console.log("\n✓ Cleanup completed - ready for fresh install");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Error during cleanup: ${message}`);
      process.exitCode = 1;
    }
  });
mon
  .command("shell <service>")
  .description("open shell to monitoring service")
  .action(async (service: string) => {
    const code = await runCompose(["exec", service, "/bin/sh"]);
    if (code !== 0) process.exitCode = code;
  });
mon
  .command("check")
  .description("monitoring services system readiness check")
  .action(async () => {
    const code = await runCompose(["ps"]);
    if (code !== 0) process.exitCode = code;
  });

// Monitoring targets (databases to monitor)
const targets = mon.command("targets").description("manage databases to monitor");

targets
  .command("list")
  .description("list monitoring target databases")
  .action(async () => {
    const { instancesFile: instancesPath, projectDir } = await resolveOrInitPaths();
    if (!fs.existsSync(instancesPath)) {
      console.error(`instances.yml not found in ${projectDir}`);
      process.exitCode = 1;
      return;
    }

    try {
      const content = fs.readFileSync(instancesPath, "utf8");
      const instances = yaml.load(content) as Instance[] | null;

      if (!instances || !Array.isArray(instances) || instances.length === 0) {
        console.log("No monitoring targets configured");
        console.log("");
        console.log("To add a monitoring target:");
        console.log("  postgres-ai mon targets add <connection-string> <name>");
        console.log("");
        console.log("Example:");
        console.log("  postgres-ai mon targets add 'postgresql://user:pass@host:5432/db' my-db");
        return;
      }

      // Filter out disabled instances (e.g., demo placeholders)
      const filtered = instances.filter((inst) => inst.name && inst.is_enabled !== false);

      if (filtered.length === 0) {
        console.log("No monitoring targets configured");
        console.log("");
        console.log("To add a monitoring target:");
        console.log("  postgres-ai mon targets add <connection-string> <name>");
        console.log("");
        console.log("Example:");
        console.log("  postgres-ai mon targets add 'postgresql://user:pass@host:5432/db' my-db");
        return;
      }

      for (const inst of filtered) {
        console.log(`Target: ${inst.name}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`Error parsing instances.yml: ${message}`);
      process.exitCode = 1;
    }
  });
targets
  .command("add [connStr] [name]")
  .description("add monitoring target database")
  .action(async (connStr?: string, name?: string) => {
    const { instancesFile: file } = await resolveOrInitPaths();
    if (!connStr) {
      console.error("Connection string required: postgresql://user:pass@host:port/db");
      process.exitCode = 1;
      return;
    }
    const m = connStr.match(/^postgresql:\/\/([^:]+):([^@]+)@([^:\/]+)(?::(\d+))?\/(.+)$/);
    if (!m) {
      console.error("Invalid connection string format");
      process.exitCode = 1;
      return;
    }
    const host = m[3];
    const db = m[5];
    const instanceName = name && name.trim() ? name.trim() : `${host}-${db}`.replace(/[^a-zA-Z0-9-]/g, "-");

    // Check if instance already exists
    try {
      if (fs.existsSync(file)) {
        const content = fs.readFileSync(file, "utf8");
        const instances = yaml.load(content) as Instance[] | null || [];
        if (Array.isArray(instances)) {
          const exists = instances.some((inst) => inst.name === instanceName);
          if (exists) {
            console.error(`Monitoring target '${instanceName}' already exists`);
            process.exitCode = 1;
            return;
          }
        }
      }
    } catch (err) {
      // If YAML parsing fails, fall back to simple check
      const content = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
      if (new RegExp(`^- name: ${instanceName}$`, "m").test(content)) {
        console.error(`Monitoring target '${instanceName}' already exists`);
        process.exitCode = 1;
        return;
      }
    }

    // Add new instance
    const body = `- name: ${instanceName}\n  conn_str: ${connStr}\n  preset_metrics: full\n  custom_metrics:\n  is_enabled: true\n  group: default\n  custom_tags:\n    env: production\n    cluster: default\n    node_name: ${instanceName}\n    sink_type: ~sink_type~\n`;
    const content = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
    fs.appendFileSync(file, (content && !/\n$/.test(content) ? "\n" : "") + body, "utf8");
    console.log(`Monitoring target '${instanceName}' added`);
  });
targets
  .command("remove <name>")
  .description("remove monitoring target database")
  .action(async (name: string) => {
    const { instancesFile: file } = await resolveOrInitPaths();
    if (!fs.existsSync(file)) {
      console.error("instances.yml not found");
      process.exitCode = 1;
      return;
    }

    try {
      const content = fs.readFileSync(file, "utf8");
      const instances = yaml.load(content) as Instance[] | null;

      if (!instances || !Array.isArray(instances)) {
        console.error("Invalid instances.yml format");
        process.exitCode = 1;
        return;
      }

      const filtered = instances.filter((inst) => inst.name !== name);

      if (filtered.length === instances.length) {
        console.error(`Monitoring target '${name}' not found`);
        process.exitCode = 1;
        return;
      }

      fs.writeFileSync(file, yaml.dump(filtered), "utf8");
      console.log(`Monitoring target '${name}' removed`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`Error processing instances.yml: ${message}`);
      process.exitCode = 1;
    }
  });
targets
  .command("test <name>")
  .description("test monitoring target database connectivity")
  .action(async (name: string) => {
    const { instancesFile: instancesPath } = await resolveOrInitPaths();
    if (!fs.existsSync(instancesPath)) {
      console.error("instances.yml not found");
      process.exitCode = 1;
      return;
    }

    try {
      const content = fs.readFileSync(instancesPath, "utf8");
      const instances = yaml.load(content) as Instance[] | null;

      if (!instances || !Array.isArray(instances)) {
        console.error("Invalid instances.yml format");
        process.exitCode = 1;
        return;
      }

      const instance = instances.find((inst) => inst.name === name);

      if (!instance) {
        console.error(`Monitoring target '${name}' not found`);
        process.exitCode = 1;
        return;
      }

      if (!instance.conn_str) {
        console.error(`Connection string not found for monitoring target '${name}'`);
        process.exitCode = 1;
        return;
      }

      console.log(`Testing connection to monitoring target '${name}'...`);

      // Use native pg client instead of requiring psql to be installed
      const client = new Client({ connectionString: instance.conn_str });

      try {
        await client.connect();
        const result = await client.query('select version();');
        console.log(`✓ Connection successful`);
        console.log(result.rows[0].version);
      } finally {
        await client.end();
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`✗ Connection failed: ${message}`);
      process.exitCode = 1;
    }
  });

// Authentication and API key management
const auth = program.command("auth").description("authentication and API key management");

auth
  .command("login", { isDefault: true })
  .description("authenticate via browser (OAuth) or store API key directly")
  .option("--set-key <key>", "store API key directly without OAuth flow")
  .option("--port <port>", "local callback server port (default: random)", parseInt)
  .option("--debug", "enable debug output")
  .action(async (opts: { setKey?: string; port?: number; debug?: boolean }) => {
    // If --set-key is provided, store it directly without OAuth
    if (opts.setKey) {
      const trimmedKey = opts.setKey.trim();
      if (!trimmedKey) {
        console.error("Error: API key cannot be empty");
        process.exitCode = 1;
        return;
      }
      
      // Read existing config to check for defaultProject before updating
      const existingConfig = config.readConfig();
      const existingProject = existingConfig.defaultProject;
      
      config.writeConfig({ apiKey: trimmedKey });
      // When API key is set directly, only clear orgId (org selection may differ).
      // Preserve defaultProject to avoid orphaning historical reports.
      // If the new key lacks access to the project, upload will fail with a clear error.
      config.deleteConfigKeys(["orgId"]);
      
      console.log(`API key saved to ${config.getConfigPath()}`);
      if (existingProject) {
        console.log(`Note: Your default project "${existingProject}" has been preserved.`);
        console.log(`      If this key belongs to a different account, use --project to specify a new one.`);
      }
      return;
    }

    // Otherwise, proceed with OAuth flow
    console.log("Starting authentication flow...\n");

    // Generate PKCE parameters
    const params = pkce.generatePKCEParams();

    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiBaseUrl, uiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

    if (opts.debug) {
      console.error(`Debug: Resolved API base URL: ${apiBaseUrl}`);
      console.error(`Debug: Resolved UI base URL: ${uiBaseUrl}`);
    }

    try {
      // Step 1: Start local callback server FIRST to get actual port
      console.log("Starting local callback server...");
      const requestedPort = opts.port || 0; // 0 = OS assigns available port
      const callbackServer = authServer.createCallbackServer(requestedPort, params.state, 120000); // 2 minute timeout

      // Wait for server to start and get the actual port
      const actualPort = await callbackServer.ready;
      // Use 127.0.0.1 to match the server bind address (avoids IPv6 issues on some hosts)
      const redirectUri = `http://127.0.0.1:${actualPort}/callback`;

      console.log(`Callback server listening on port ${actualPort}`);

      // Step 2: Initialize OAuth session on backend
      console.log("Initializing authentication session...");
      const initData = JSON.stringify({
        client_type: "cli",
        state: params.state,
        code_challenge: params.codeChallenge,
        code_challenge_method: params.codeChallengeMethod,
        redirect_uri: redirectUri,
      });

      // Build init URL by appending to the API base path (keep /api/general)
      const initUrl = new URL(`${apiBaseUrl}/rpc/oauth_init`);

      if (opts.debug) {
        console.error(`Debug: Trying to POST to: ${initUrl.toString()}`);
        console.error(`Debug: Request data: ${initData}`);
      }

      // Step 2: Initialize OAuth session on backend using fetch
      let initResponse: Response;
      try {
        initResponse = await fetch(initUrl.toString(), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: initData,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.error(`Failed to connect to API: ${message}`);
        callbackServer.server.stop();
        process.exit(1);
        return;
      }

      if (!initResponse.ok) {
        const data = await initResponse.text();
        console.error(`Failed to initialize auth session: ${initResponse.status}`);

        // Check if response is HTML (common for 404 pages)
        if (data.trim().startsWith("<!") || data.trim().startsWith("<html")) {
          console.error("Error: Received HTML response instead of JSON. This usually means:");
          console.error("  1. The API endpoint URL is incorrect");
          console.error("  2. The endpoint does not exist (404)");
          console.error(`\nAPI URL attempted: ${initUrl.toString()}`);
          console.error("\nPlease verify the --api-base-url parameter.");
        } else {
          console.error(data);
        }

        callbackServer.server.stop();
        process.exit(1);
        return;
      }

      // Step 3: Open browser
      // Pass api_url so UI calls oauth_approve on the same backend where oauth_init created the session
      const authUrl = `${uiBaseUrl}/cli/auth?state=${encodeURIComponent(params.state)}&code_challenge=${encodeURIComponent(params.codeChallenge)}&code_challenge_method=S256&redirect_uri=${encodeURIComponent(redirectUri)}&api_url=${encodeURIComponent(apiBaseUrl)}`;

      if (opts.debug) {
        console.error(`Debug: Auth URL: ${authUrl}`);
      }

      console.log(`\nOpening browser for authentication...`);
      console.log(`If browser does not open automatically, visit:\n${authUrl}\n`);

      // Open browser (cross-platform)
      const openCommand = process.platform === "darwin" ? "open" :
                         process.platform === "win32" ? "start" :
                         "xdg-open";
      spawn(openCommand, [authUrl], { detached: true, stdio: "ignore" }).unref();

      // Step 4: Wait for callback
      console.log("Waiting for authorization...");
      console.log("(Press Ctrl+C to cancel)\n");

      // Handle Ctrl+C gracefully
      const cancelHandler = () => {
        console.log("\n\nAuthentication cancelled by user.");
        callbackServer.server.stop();
        process.exit(130); // Standard exit code for SIGINT
      };
      process.on("SIGINT", cancelHandler);

      try {
        const { code } = await callbackServer.promise;

        // Remove the cancel handler after successful auth
        process.off("SIGINT", cancelHandler);

        // Step 5: Exchange code for token using fetch
        console.log("\nExchanging authorization code for API token...");
        const exchangeData = JSON.stringify({
          authorization_code: code,
          code_verifier: params.codeVerifier,
          state: params.state,
        });
        const exchangeUrl = new URL(`${apiBaseUrl}/rpc/oauth_token_exchange`);

        let exchangeResponse: Response;
        try {
          exchangeResponse = await fetch(exchangeUrl.toString(), {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: exchangeData,
          });
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`Exchange request failed: ${message}`);
          process.exit(1);
          return;
        }

        const exchangeBody = await exchangeResponse.text();

        if (!exchangeResponse.ok) {
          console.error(`Failed to exchange code for token: ${exchangeResponse.status}`);

          // Check if response is HTML (common for 404 pages)
          if (exchangeBody.trim().startsWith("<!") || exchangeBody.trim().startsWith("<html")) {
            console.error("Error: Received HTML response instead of JSON. This usually means:");
            console.error("  1. The API endpoint URL is incorrect");
            console.error("  2. The endpoint does not exist (404)");
            console.error(`\nAPI URL attempted: ${exchangeUrl.toString()}`);
            console.error("\nPlease verify the --api-base-url parameter.");
          } else {
            console.error(exchangeBody);
          }

          process.exit(1);
          return;
        }

        try {
          const result = JSON.parse(exchangeBody);
          const apiToken = result.api_token || result?.[0]?.result?.api_token; // There is a bug with PostgREST Caching that may return an array, not single object, it's a workaround to support both cases.
          const orgId = result.org_id || result?.[0]?.result?.org_id; // There is a bug with PostgREST Caching that may return an array, not single object, it's a workaround to support both cases.

          // Step 6: Save token to config
          // Check if org changed to decide whether to preserve defaultProject
          const existingConfig = config.readConfig();
          const existingOrgId = existingConfig.orgId;
          const existingProject = existingConfig.defaultProject;
          const orgChanged = existingOrgId && existingOrgId !== orgId;
          
          config.writeConfig({
            apiKey: apiToken,
            baseUrl: apiBaseUrl,
            orgId: orgId,
          });
          
          // Only clear defaultProject if org actually changed
          if (orgChanged && existingProject) {
            config.deleteConfigKeys(["defaultProject"]);
            console.log(`\nNote: Organization changed (${existingOrgId} → ${orgId}).`);
            console.log(`      Default project "${existingProject}" has been cleared.`);
          }

          console.log("\nAuthentication successful!");
          console.log(`API key saved to: ${config.getConfigPath()}`);
          console.log(`Organization ID: ${orgId}`);
          if (!orgChanged && existingProject) {
            console.log(`Default project: ${existingProject} (preserved)`);
          }
          console.log(`\nYou can now use the CLI without specifying an API key.`);
          process.exit(0);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`Failed to parse response: ${message}`);
          process.exit(1);
        }

      } catch (err) {
        // Remove the cancel handler in error case too
        process.off("SIGINT", cancelHandler);

        const message = err instanceof Error ? err.message : String(err);

        // Provide more helpful error messages
        if (message.includes("timeout")) {
          console.error(`\nAuthentication timed out.`);
          console.error(`This usually means you closed the browser window without completing authentication.`);
          console.error(`Please try again and complete the authentication flow.`);
        } else {
          console.error(`\nAuthentication failed: ${message}`);
        }

        process.exit(1);
      }

    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`Authentication error: ${message}`);
      process.exit(1);
    }
  });

auth
  .command("show-key")
  .description("show API key (masked)")
  .action(async () => {
    const cfg = config.readConfig();
    if (!cfg.apiKey) {
      console.log("No API key configured");
      console.log(`\nTo authenticate, run: pgai auth`);
      return;
    }
    console.log(`Current API key: ${maskSecret(cfg.apiKey)}`);
    if (cfg.orgId) {
      console.log(`Organization ID: ${cfg.orgId}`);
    }
    console.log(`Config location: ${config.getConfigPath()}`);
  });

auth
  .command("remove-key")
  .description("remove API key")
  .action(async () => {
    // Check both new config and legacy config
    const newConfigPath = config.getConfigPath();
    const hasNewConfig = fs.existsSync(newConfigPath);
    let legacyPath: string;
    try {
      const { projectDir } = await resolveOrInitPaths();
      legacyPath = path.resolve(projectDir, ".pgwatch-config");
    } catch {
      legacyPath = path.resolve(process.cwd(), ".pgwatch-config");
    }
    const hasLegacyConfig = fs.existsSync(legacyPath) && fs.statSync(legacyPath).isFile();

    if (!hasNewConfig && !hasLegacyConfig) {
      console.log("No API key configured");
      return;
    }

    // Remove from new config
    if (hasNewConfig) {
      config.deleteConfigKeys(["apiKey", "orgId"]);
    }

    // Remove from legacy config
    if (hasLegacyConfig) {
      try {
        const content = fs.readFileSync(legacyPath, "utf8");
        const filtered = content
          .split(/\r?\n/)
          .filter((l) => !/^api_key=/.test(l))
          .join("\n")
          .replace(/\n+$/g, "\n");
        fs.writeFileSync(legacyPath, filtered, "utf8");
      } catch (err) {
        // If we can't read/write the legacy config, just skip it
        console.warn(`Warning: Could not update legacy config: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    console.log("API key removed");
    console.log(`\nTo authenticate again, run: pgai auth`);
  });
mon
  .command("generate-grafana-password")
  .description("generate Grafana password for monitoring services")
  .action(async () => {
    const { projectDir } = await resolveOrInitPaths();
    const cfgPath = path.resolve(projectDir, ".pgwatch-config");

    try {
      // Generate secure password using openssl
      const { stdout: password } = await execPromise(
        "openssl rand -base64 12 | tr -d '\n'"
      );
      const newPassword = password.trim();

      if (!newPassword) {
        console.error("Failed to generate password");
        process.exitCode = 1;
        return;
      }

      // Read existing config
      let configContent = "";
      if (fs.existsSync(cfgPath)) {
        const stats = fs.statSync(cfgPath);
        if (stats.isDirectory()) {
          console.error(".pgwatch-config is a directory, expected a file. Skipping read.");
        } else {
          configContent = fs.readFileSync(cfgPath, "utf8");
        }
      }

      // Update or add grafana_password
      const lines = configContent.split(/\r?\n/).filter((l) => !/^grafana_password=/.test(l));
      lines.push(`grafana_password=${newPassword}`);

      // Write back
      fs.writeFileSync(cfgPath, lines.filter(Boolean).join("\n") + "\n", "utf8");

      console.log("✓ New Grafana password generated and saved");
      console.log("\nNew credentials:");
      console.log("  URL:      http://localhost:3000");
      console.log("  Username: monitor");
      console.log(`  Password: ${newPassword}`);
      console.log("\nReset Grafana to apply new password:");
      console.log("  postgres-ai mon reset grafana");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Failed to generate password: ${message}`);
      console.error("\nNote: This command requires 'openssl' to be installed");
      process.exitCode = 1;
    }
  });
mon
  .command("show-grafana-credentials")
  .description("show Grafana credentials for monitoring services")
  .action(async () => {
    const { projectDir } = await resolveOrInitPaths();
    const cfgPath = path.resolve(projectDir, ".pgwatch-config");
    if (!fs.existsSync(cfgPath)) {
      console.error("Configuration file not found. Run 'postgres-ai mon local-install' first.");
      process.exitCode = 1;
      return;
    }

    const stats = fs.statSync(cfgPath);
    if (stats.isDirectory()) {
      console.error(".pgwatch-config is a directory, expected a file. Cannot read credentials.");
      process.exitCode = 1;
      return;
    }

    const content = fs.readFileSync(cfgPath, "utf8");
    const lines = content.split(/\r?\n/);
    let password = "";
    for (const line of lines) {
      const m = line.match(/^grafana_password=(.+)$/);
      if (m) {
        password = m[1].trim();
        break;
      }
    }
    if (!password) {
      console.error("Grafana password not found in configuration");
      process.exitCode = 1;
      return;
    }
    console.log("\nGrafana credentials:");
    console.log("  URL:      http://localhost:3000");
    console.log("  Username: monitor");
    console.log(`  Password: ${password}`);

    // Show VM auth credentials from .env
    const envFile = path.resolve(projectDir, ".env");
    if (fs.existsSync(envFile)) {
      const envContent = fs.readFileSync(envFile, "utf8");
      const vmUser = envContent.match(/^VM_AUTH_USERNAME=([^\r\n]+)/m);
      const vmPass = envContent.match(/^VM_AUTH_PASSWORD=([^\r\n]+)/m);
      if (vmUser && vmPass) {
        console.log("\nVictoriaMetrics credentials:");
        console.log(`  Username: ${vmUser[1].trim().replace(/^["']|["']$/g, '')}`);
        console.log(`  Password: ${vmPass[1].trim().replace(/^["']|["']$/g, '')}`);
      }
    }
    console.log("");
  });

/**
 * Interpret escape sequences in a string (e.g., \n -> newline)
 * Note: In regex, to match literal backslash-n, we need \\n in the pattern
 * which requires \\\\n in the JavaScript string literal
 */
function interpretEscapes(str: string): string {
  // First handle double backslashes by temporarily replacing them
  // Then handle other escapes, then restore double backslashes as single
  return str
    .replace(/\\\\/g, '\x00') // Temporarily mark double backslashes
    .replace(/\\n/g, '\n') // Match literal backslash-n (\\\\n in JS string -> \\n in regex -> matches \n)
    .replace(/\\t/g, '\t')
    .replace(/\\r/g, '\r')
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'")
    .replace(/\x00/g, '\\'); // Restore double backslashes as single
}

// Issues management
const issues = program.command("issues").description("issues management");

issues
  .command("list")
  .description("list issues")
  .option("--status <status>", "filter by status: open, closed, or all (default: all)")
  .option("--limit <n>", "max number of issues to return (default: 20)", parseInt)
  .option("--offset <n>", "number of issues to skip (default: 0)", parseInt)
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (opts: { status?: string; limit?: number; offset?: number; debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching issues...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }
      const orgId = cfg.orgId ?? undefined;

      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      let statusFilter: "open" | "closed" | undefined;
      if (opts.status === "open") {
        statusFilter = "open";
      } else if (opts.status === "closed") {
        statusFilter = "closed";
      }

      const result = await fetchIssues({
        apiKey,
        apiBaseUrl,
        orgId,
        status: statusFilter,
        limit: opts.limit,
        offset: opts.offset,
        debug: !!opts.debug,
      });
      spinner.stop();
      const trimmed = Array.isArray(result)
        ? (result as any[]).map((r) => ({
            id: (r as any).id,
            title: (r as any).title,
            status: (r as any).status,
            created_at: (r as any).created_at,
          }))
        : result;
      printResult(trimmed, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("view <issueId>")
  .description("view issue details and comments")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (issueId: string, opts: { debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching issue...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const issue = await fetchIssue({ apiKey, apiBaseUrl, issueId, debug: !!opts.debug });
      if (!issue) {
        spinner.stop();
        console.error("Issue not found");
        process.exitCode = 1;
        return;
      }

      spinner.update("Fetching comments...");
      const comments = await fetchIssueComments({ apiKey, apiBaseUrl, issueId, debug: !!opts.debug });
      spinner.stop();
      const combined = { issue, comments };
      printResult(combined, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("post-comment <issueId> <content>")
  .description("post a new comment to an issue")
  .option("--parent <uuid>", "parent comment id")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (issueId: string, content: string, opts: { parent?: string; debug?: boolean; json?: boolean }) => {
    // Interpret escape sequences in content (e.g., \n -> newline)
    if (opts.debug) {
      // eslint-disable-next-line no-console
      console.error(`Debug: Original content: ${JSON.stringify(content)}`);
    }
    content = interpretEscapes(content);
    if (opts.debug) {
      // eslint-disable-next-line no-console
      console.error(`Debug: Interpreted content: ${JSON.stringify(content)}`);
    }

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Posting comment...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await createIssueComment({
        apiKey,
        apiBaseUrl,
        issueId,
        content,
        parentCommentId: opts.parent,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("create <title>")
  .description("create a new issue")
  .option("--org-id <id>", "organization id (defaults to config orgId)", (v) => parseInt(v, 10))
  .option("--project-id <id>", "project id", (v) => parseInt(v, 10))
  .option("--description <text>", "issue description (use \\n for newlines)")
  .option(
    "--label <label>",
    "issue label (repeatable)",
    (value: string, previous: string[]) => {
      previous.push(value);
      return previous;
    },
    [] as string[]
  )
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (rawTitle: string, opts: { orgId?: number; projectId?: number; description?: string; label?: string[]; debug?: boolean; json?: boolean }) => {
    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiKey } = getConfig(rootOpts);
    if (!apiKey) {
      console.error("API key is required. Run 'pgai auth' first or set --api-key.");
      process.exitCode = 1;
      return;
    }

    const title = interpretEscapes(String(rawTitle || "").trim());
    if (!title) {
      console.error("title is required");
      process.exitCode = 1;
      return;
    }

    const orgId = typeof opts.orgId === "number" && !Number.isNaN(opts.orgId) ? opts.orgId : cfg.orgId;
    if (typeof orgId !== "number") {
      console.error("org_id is required. Either pass --org-id or run 'pgai auth' to store it in config.");
      process.exitCode = 1;
      return;
    }

    const description = opts.description !== undefined ? interpretEscapes(String(opts.description)) : undefined;
    const labels = Array.isArray(opts.label) && opts.label.length > 0 ? opts.label.map(String) : undefined;
    const projectId = typeof opts.projectId === "number" && !Number.isNaN(opts.projectId) ? opts.projectId : undefined;

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Creating issue...");
    try {
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);
      const result = await createIssue({
        apiKey,
        apiBaseUrl,
        title,
        orgId,
        description,
        projectId,
        labels,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("update <issueId>")
  .description("update an existing issue (title/description/status/labels)")
  .option("--title <text>", "new title (use \\n for newlines)")
  .option("--description <text>", "new description (use \\n for newlines)")
  .option("--status <value>", "status: open|closed|0|1")
  .option(
    "--label <label>",
    "set labels (repeatable). If provided, replaces existing labels.",
    (value: string, previous: string[]) => {
      previous.push(value);
      return previous;
    },
    [] as string[]
  )
  .option("--clear-labels", "set labels to an empty list")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (issueId: string, opts: { title?: string; description?: string; status?: string; label?: string[]; clearLabels?: boolean; debug?: boolean; json?: boolean }) => {
    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiKey } = getConfig(rootOpts);
    if (!apiKey) {
      console.error("API key is required. Run 'pgai auth' first or set --api-key.");
      process.exitCode = 1;
      return;
    }

    const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

    const title = opts.title !== undefined ? interpretEscapes(String(opts.title)) : undefined;
    const description = opts.description !== undefined ? interpretEscapes(String(opts.description)) : undefined;

    let status: number | undefined = undefined;
    if (opts.status !== undefined) {
      const raw = String(opts.status).trim().toLowerCase();
      if (raw === "open") status = 0;
      else if (raw === "closed") status = 1;
      else {
        const n = Number(raw);
        if (!Number.isFinite(n)) {
          console.error("status must be open|closed|0|1");
          process.exitCode = 1;
          return;
        }
        status = n;
      }
      if (status !== 0 && status !== 1) {
        console.error("status must be 0 (open) or 1 (closed)");
        process.exitCode = 1;
        return;
      }
    }

    let labels: string[] | undefined = undefined;
    if (opts.clearLabels) {
      labels = [];
    } else if (Array.isArray(opts.label) && opts.label.length > 0) {
      labels = opts.label.map(String);
    }

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Updating issue...");
    try {
      const result = await updateIssue({
        apiKey,
        apiBaseUrl,
        issueId,
        title,
        description,
        status,
        labels,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("update-comment <commentId> <content>")
  .description("update an existing issue comment")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (commentId: string, content: string, opts: { debug?: boolean; json?: boolean }) => {
    if (opts.debug) {
      // eslint-disable-next-line no-console
      console.error(`Debug: Original content: ${JSON.stringify(content)}`);
    }
    content = interpretEscapes(content);
    if (opts.debug) {
      // eslint-disable-next-line no-console
      console.error(`Debug: Interpreted content: ${JSON.stringify(content)}`);
    }

    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiKey } = getConfig(rootOpts);
    if (!apiKey) {
      console.error("API key is required. Run 'pgai auth' first or set --api-key.");
      process.exitCode = 1;
      return;
    }

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Updating comment...");
    try {
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await updateIssueComment({
        apiKey,
        apiBaseUrl,
        commentId,
        content,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

// File upload/download (subcommands of issues)
const issueFiles = issues.command("files").description("upload and download files for issues");

issueFiles
  .command("upload <path>")
  .description("upload a file to storage and get a markdown link")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (filePath: string, opts: { debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Uploading file...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await uploadFile({
        apiKey,
        storageBaseUrl,
        filePath,
        debug: !!opts.debug,
      });
      spinner.stop();

      if (opts.json) {
        printResult(result, true);
      } else {
        const md = buildMarkdownLink(result.url, storageBaseUrl, result.metadata.originalName);
        const displayUrl = result.url.startsWith("/") ? `${storageBaseUrl}${result.url}` : `${storageBaseUrl}/${result.url}`;
        console.log(`URL: ${displayUrl}`);
        console.log(`File: ${result.metadata.originalName}`);
        console.log(`Size: ${result.metadata.size} bytes`);
        console.log(`Type: ${result.metadata.mimeType}`);
        console.log(`Markdown: ${md}`);
      }
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issueFiles
  .command("download <url>")
  .description("download a file from storage")
  .option("-o, --output <path>", "output file path (default: derive from URL)")
  .option("--debug", "enable debug output")
  .action(async (fileUrl: string, opts: { output?: string; debug?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Downloading file...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { storageBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await downloadFile({
        apiKey,
        storageBaseUrl,
        fileUrl,
        outputPath: opts.output,
        debug: !!opts.debug,
      });
      spinner.stop();
      console.log(`Saved: ${result.savedTo}`);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

// Action Items management (subcommands of issues)
issues
  .command("action-items <issueId>")
  .description("list action items for an issue")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (issueId: string, opts: { debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching action items...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await fetchActionItems({ apiKey, apiBaseUrl, issueId, debug: !!opts.debug });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("view-action-item <actionItemIds...>")
  .description("view action item(s) with all details (supports multiple IDs)")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (actionItemIds: string[], opts: { debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching action item(s)...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }

      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await fetchActionItem({ apiKey, apiBaseUrl, actionItemIds, debug: !!opts.debug });
      if (result.length === 0) {
        spinner.stop();
        console.error("Action item(s) not found");
        process.exitCode = 1;
        return;
      }
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("create-action-item <issueId> <title>")
  .description("create a new action item for an issue")
  .option("--description <text>", "detailed description (use \\n for newlines)")
  .option("--sql-action <sql>", "SQL command to execute")
  .option("--config <json>", "config change as JSON: {\"parameter\":\"...\",\"value\":\"...\"} (repeatable)", (value: string, previous: ConfigChange[]) => {
    try {
      previous.push(JSON.parse(value) as ConfigChange);
    } catch {
      console.error(`Invalid JSON for --config: ${value}`);
      process.exit(1);
    }
    return previous;
  }, [] as ConfigChange[])
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (issueId: string, rawTitle: string, opts: { description?: string; sqlAction?: string; config?: ConfigChange[]; debug?: boolean; json?: boolean }) => {
    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiKey } = getConfig(rootOpts);
    if (!apiKey) {
      console.error("API key is required. Run 'pgai auth' first or set --api-key.");
      process.exitCode = 1;
      return;
    }

    const title = interpretEscapes(String(rawTitle || "").trim());
    if (!title) {
      console.error("title is required");
      process.exitCode = 1;
      return;
    }

    const description = opts.description !== undefined ? interpretEscapes(String(opts.description)) : undefined;
    const sqlAction = opts.sqlAction;
    const configs = Array.isArray(opts.config) && opts.config.length > 0 ? opts.config : undefined;

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Creating action item...");
    try {
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);
      const result = await createActionItem({
        apiKey,
        apiBaseUrl,
        issueId,
        title,
        description,
        sqlAction,
        configs,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult({ id: result }, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

issues
  .command("update-action-item <actionItemId>")
  .description("update an action item (title, description, status, sql_action, configs)")
  .option("--title <text>", "new title (use \\n for newlines)")
  .option("--description <text>", "new description (use \\n for newlines)")
  .option("--done", "mark as done")
  .option("--not-done", "mark as not done")
  .option("--status <value>", "status: waiting_for_approval|approved|rejected")
  .option("--status-reason <text>", "reason for status change")
  .option("--sql-action <sql>", "SQL command (use empty string to clear)")
  .option("--config <json>", "config change as JSON (repeatable, replaces all configs)", (value: string, previous: ConfigChange[]) => {
    try {
      previous.push(JSON.parse(value) as ConfigChange);
    } catch {
      console.error(`Invalid JSON for --config: ${value}`);
      process.exit(1);
    }
    return previous;
  }, [] as ConfigChange[])
  .option("--clear-configs", "clear all config changes")
  .option("--debug", "enable debug output")
  .option("--json", "output raw JSON")
  .action(async (actionItemId: string, opts: { title?: string; description?: string; done?: boolean; notDone?: boolean; status?: string; statusReason?: string; sqlAction?: string; config?: ConfigChange[]; clearConfigs?: boolean; debug?: boolean; json?: boolean }) => {
    const rootOpts = program.opts<CliOptions>();
    const cfg = config.readConfig();
    const { apiKey } = getConfig(rootOpts);
    if (!apiKey) {
      console.error("API key is required. Run 'pgai auth' first or set --api-key.");
      process.exitCode = 1;
      return;
    }

    const title = opts.title !== undefined ? interpretEscapes(String(opts.title)) : undefined;
    const description = opts.description !== undefined ? interpretEscapes(String(opts.description)) : undefined;

    let isDone: boolean | undefined = undefined;
    if (opts.done) isDone = true;
    else if (opts.notDone) isDone = false;

    let status: string | undefined = undefined;
    if (opts.status !== undefined) {
      const validStatuses = ["waiting_for_approval", "approved", "rejected"];
      if (!validStatuses.includes(opts.status)) {
        console.error(`status must be one of: ${validStatuses.join(", ")}`);
        process.exitCode = 1;
        return;
      }
      status = opts.status;
    }

    const statusReason = opts.statusReason;
    const sqlAction = opts.sqlAction;

    let configs: ConfigChange[] | undefined = undefined;
    if (opts.clearConfigs) {
      configs = [];
    } else if (Array.isArray(opts.config) && opts.config.length > 0) {
      configs = opts.config;
    }

    // Check that at least one update field is provided
    if (title === undefined && description === undefined &&
        isDone === undefined && status === undefined && statusReason === undefined &&
        sqlAction === undefined && configs === undefined) {
      console.error("At least one update option is required");
      process.exitCode = 1;
      return;
    }

    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Updating action item...");
    try {
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);
      await updateActionItem({
        apiKey,
        apiBaseUrl,
        actionItemId,
        title,
        description,
        isDone,
        status,
        statusReason,
        sqlAction,
        configs,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult({ success: true }, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

// Reports management
const reports = program.command("reports").description("checkup reports management");

reports
  .command("list")
  .description("list checkup reports")
  .option("--project-id <id>", "filter by project id", (v: string) => parseInt(v, 10))
  .addOption(new Option("--status <status>", "filter by status (e.g., completed)").hideHelp())
  .option("--limit <n>", "max number of reports to return (default: 20, max: 100)", (v: string) => { const n = parseInt(v, 10); return Number.isNaN(n) ? 20 : Math.max(1, Math.min(n, 100)); })
  .option("--before <date>", "show reports created before this date (YYYY-MM-DD, DD.MM.YYYY, etc.)")
  .option("--all", "fetch all reports (paginated automatically)")
  .addOption(new Option("--debug", "enable debug output").hideHelp())
  .option("--json", "output raw JSON")
  .action(async (opts: { projectId?: number; status?: string; limit?: number; before?: string; all?: boolean; debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching reports...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }
      if (opts.all && opts.before) {
        spinner.stop();
        console.error("--all and --before cannot be used together");
        process.exitCode = 1;
        return;
      }
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      let result;
      if (opts.all) {
        result = await fetchAllReports({
          apiKey,
          apiBaseUrl,
          projectId: opts.projectId,
          status: opts.status,
          limit: opts.limit,
          debug: !!opts.debug,
        });
      } else {
        result = await fetchReports({
          apiKey,
          apiBaseUrl,
          projectId: opts.projectId,
          status: opts.status,
          limit: opts.limit,
          beforeDate: opts.before ? parseFlexibleDate(opts.before) : undefined,
          debug: !!opts.debug,
        });
      }
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

reports
  .command("files [reportId]")
  .description("list files of a checkup report (metadata only, no content)")
  .option("--type <type>", "filter by file type: json, md")
  .option("--check-id <id>", "filter by check ID (e.g., H002)")
  .addOption(new Option("--debug", "enable debug output").hideHelp())
  .option("--json", "output raw JSON")
  .action(async (reportId: string | undefined, opts: { type?: "json" | "md"; checkId?: string; debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching report files...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }
      let numericId: number | undefined;
      if (reportId !== undefined) {
        numericId = parseInt(reportId, 10);
        if (isNaN(numericId)) {
          spinner.stop();
          console.error("reportId must be a number");
          process.exitCode = 1;
          return;
        }
      }
      if (numericId === undefined && !opts.checkId) {
        spinner.stop();
        console.error("Either reportId or --check-id is required");
        process.exitCode = 1;
        return;
      }
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      const result = await fetchReportFiles({
        apiKey,
        apiBaseUrl,
        reportId: numericId,
        type: opts.type,
        checkId: opts.checkId,
        debug: !!opts.debug,
      });
      spinner.stop();
      printResult(result, opts.json);
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

reports
  .command("data [reportId]")
  .description("get checkup report file data (includes content)")
  .option("--type <type>", "filter by file type: json, md")
  .option("--check-id <id>", "filter by check ID (e.g., H002)")
  .option("--formatted", "render markdown with ANSI styling (experimental)")
  .option("-o, --output <dir>", "save files to directory (uses original filenames)")
  .addOption(new Option("--debug", "enable debug output").hideHelp())
  .option("--json", "output raw JSON")
  .action(async (reportId: string | undefined, opts: { type?: "json" | "md"; checkId?: string; formatted?: boolean; output?: string; debug?: boolean; json?: boolean }) => {
    const spinner = createTtySpinner(process.stdout.isTTY ?? false, "Fetching report data...");
    try {
      const rootOpts = program.opts<CliOptions>();
      const cfg = config.readConfig();
      const { apiKey } = getConfig(rootOpts);
      if (!apiKey) {
        spinner.stop();
        console.error("API key is required. Run 'pgai auth' first or set --api-key.");
        process.exitCode = 1;
        return;
      }
      let numericId: number | undefined;
      if (reportId !== undefined) {
        numericId = parseInt(reportId, 10);
        if (isNaN(numericId)) {
          spinner.stop();
          console.error("reportId must be a number");
          process.exitCode = 1;
          return;
        }
      }
      if (numericId === undefined && !opts.checkId) {
        spinner.stop();
        console.error("Either reportId or --check-id is required");
        process.exitCode = 1;
        return;
      }
      const { apiBaseUrl } = resolveBaseUrls(rootOpts, cfg);

      // Default to "md" for terminal output (human-readable); --json and --output get all types
      const effectiveType = opts.type ?? (!opts.json && !opts.output ? "md" as const : undefined);
      const result = await fetchReportFileData({
        apiKey,
        apiBaseUrl,
        reportId: numericId,
        type: effectiveType,
        checkId: opts.checkId,
        debug: !!opts.debug,
      });
      spinner.stop();

      if (opts.output) {
        const dir = path.resolve(opts.output);
        fs.mkdirSync(dir, { recursive: true });
        for (const f of result) {
          const safeName = path.basename(f.filename);
          const filePath = path.join(dir, safeName);
          const content = f.type === "json"
            ? JSON.stringify(tryParseJson(f.data), null, 2)
            : f.data;
          fs.writeFileSync(filePath, content, "utf-8");
          console.log(filePath);
        }
      } else if (opts.json) {
        const processed = result.map((f) => ({
          ...f,
          data: f.type === "json" ? tryParseJson(f.data) : f.data,
        }));
        printResult(processed, true);
      } else if (opts.formatted && process.stdout.isTTY) {
        for (const f of result) {
          if (result.length > 1) {
            console.log(`\x1b[1m--- ${f.filename} (${f.check_id}, ${f.type}) ---\x1b[0m`);
          }
          if (f.type === "md") {
            console.log(renderMarkdownForTerminal(f.data));
          } else if (f.type === "json") {
            const parsed = tryParseJson(f.data);
            console.log(typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2));
          } else {
            console.log(f.data);
          }
        }
      } else {
        for (const f of result) {
          if (result.length > 1) {
            console.log(`--- ${f.filename} (${f.check_id}, ${f.type}) ---`);
          }
          console.log(f.data);
        }
      }
    } catch (err) {
      spinner.stop();
      const message = err instanceof Error ? err.message : String(err);
      console.error(message);
      process.exitCode = 1;
    }
  });

function tryParseJson(s: string): unknown {
  try { return JSON.parse(s); } catch { return s; }
}

// MCP server
const mcp = program.command("mcp").description("MCP server integration");

mcp
  .command("start")
  .description("start MCP stdio server")
  .option("--debug", "enable debug output")
  .action(async (opts: { debug?: boolean }) => {
    const rootOpts = program.opts<CliOptions>();
    await startMcpServer(rootOpts, { debug: !!opts.debug });
  });

mcp
  .command("install [client]")
  .description("install MCP server configuration for AI coding tool")
  .action(async (client?: string) => {
    const supportedClients = ["cursor", "claude-code", "windsurf", "codex"];

    // If no client specified, prompt user to choose
    if (!client) {
      console.log("Available AI coding tools:");
      console.log("  1. Cursor");
      console.log("  2. Claude Code");
      console.log("  3. Windsurf");
      console.log("  4. Codex");
      console.log("");

      const answer = await question("Select your AI coding tool (1-4): ");

      const choices: Record<string, string> = {
        "1": "cursor",
        "2": "claude-code",
        "3": "windsurf",
        "4": "codex"
      };

      client = choices[answer.trim()];
      if (!client) {
        console.error("Invalid selection");
        process.exitCode = 1;
        return;
      }
    }

    client = client.toLowerCase();

    if (!supportedClients.includes(client)) {
      console.error(`Unsupported client: ${client}`);
      console.error(`Supported clients: ${supportedClients.join(", ")}`);
      process.exitCode = 1;
      return;
    }

    try {
      // Get the path to the current pgai executable
      let pgaiPath: string;
      try {
        const execPath = await execPromise("which pgai");
        pgaiPath = execPath.stdout.trim();
      } catch {
        // Fallback to just "pgai" if which fails
        pgaiPath = "pgai";
      }

      // Claude Code uses its own CLI to manage MCP servers
      if (client === "claude-code") {
        console.log("Installing PostgresAI MCP server for Claude Code...");

        try {
          const { stdout, stderr } = await execPromise(
            `claude mcp add -s user postgresai ${pgaiPath} mcp start`
          );

          if (stdout) console.log(stdout);
          if (stderr) console.error(stderr);

          console.log("");
          console.log("Successfully installed PostgresAI MCP server for Claude Code");
          console.log("");
          console.log("Next steps:");
          console.log("  1. Restart Claude Code to load the new configuration");
          console.log("  2. The PostgresAI MCP server will be available as 'postgresai'");
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.error("Failed to install MCP server using Claude CLI");
          console.error(message);
          console.error("");
          console.error("Make sure the 'claude' CLI tool is installed and in your PATH");
          console.error("See: https://docs.anthropic.com/en/docs/build-with-claude/mcp");
          process.exitCode = 1;
        }
        return;
      }

      // For other clients (Cursor, Windsurf, Codex), use JSON config editing
      const homeDir = os.homedir();
      let configPath: string;
      let configDir: string;

      // Determine config file location based on client
      switch (client) {
        case "cursor":
          configPath = path.join(homeDir, ".cursor", "mcp.json");
          configDir = path.dirname(configPath);
          break;

        case "windsurf":
          configPath = path.join(homeDir, ".windsurf", "mcp.json");
          configDir = path.dirname(configPath);
          break;

        case "codex":
          configPath = path.join(homeDir, ".codex", "mcp.json");
          configDir = path.dirname(configPath);
          break;

        default:
          console.error(`Configuration not implemented for: ${client}`);
          process.exitCode = 1;
          return;
      }

      // Ensure config directory exists
      if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
      }

      // Read existing config or create new one
      let config: any = { mcpServers: {} };
      if (fs.existsSync(configPath)) {
        try {
          const content = fs.readFileSync(configPath, "utf8");
          config = JSON.parse(content);
          if (!config.mcpServers) {
            config.mcpServers = {};
          }
        } catch (err) {
          console.error(`Warning: Could not parse existing config, creating new one`);
        }
      }

      // Add or update PostgresAI MCP server configuration
      config.mcpServers.postgresai = {
        command: pgaiPath,
        args: ["mcp", "start"]
      };

      // Write updated config
      fs.writeFileSync(configPath, JSON.stringify(config, null, 2), "utf8");

      console.log(`✓ PostgresAI MCP server configured for ${client}`);
      console.log(`  Config file: ${configPath}`);
      console.log("");
      console.log("Please restart your AI coding tool to activate the MCP server");

    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Failed to install MCP server: ${message}`);
      process.exitCode = 1;
    }
  });

program.parseAsync(process.argv).finally(() => {
  closeReadline();
});

