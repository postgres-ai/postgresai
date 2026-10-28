import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { Client } from "pg";
import { parse as parseConnString, parseIntoClientConfig } from "pg-connection-string";
import { findService } from "./clickhouse";
import {
  applyInitPlan, buildInitPlan, connectWithSslFallback, DEFAULT_MONITORING_USER,
  maskConnectionString, quoteIdent, quoteLiteral, renderInitPlanSql, resolveAdminConnection, resolveMonitoringPassword, verifyInitSetup,
} from "./init";
import { assertEncodedUserinfo, estimateLine, monitoringStatus, noCardNext, notAdminNext, type DeployStatus } from "./deploy-surface";
import { collectorConnection, splitChannelBinding, verifyCollectorTls } from "./instances";
import { callRpc, type ProjectListItem } from "./joe";
import { listOrgs, type OrgScope } from "./org-scope";
import {
  AwsError, PRINCIPAL_ARN, ROLE_ARN, TERRAFORM_URL, VPCE_SERVICE, awsRunner, callerIdentity, describeDbInstance, findStacks, isPrivateAddress, oneSubnetPerAz,
  publicAddresses, quickCreateUrl, rdsEndpoint, stackFailure, stackNameFor, stackState, terraformFiles, vcpusOfClass,
  type AwsRun, type DbInstance, type RdsEndpoint, type Stack, type StackInputs,
} from "./privatelink";
import { SUPABASE_API_BASE, extractProjectRefFromUrl } from "./supabase";
import { HttpRequestTimeoutError, HttpStatusError, requestTimeoutSignal, stripControlsToSpace } from "./util";

// `pgai mon deploy <database-url>`: put a database under PostgresAI's care
// (postgres-ai/internal#354). Each step is skipped when already done, so a
// re-run is safe; every outcome carries the exact next action.

export type Provider = "clickhouse" | "rds" | "supabase" | "self-managed";
/** One vocabulary with pgai dblab deploy (lib/deploy-surface). */
export type Status = DeployStatus;

/** A step of mon deploy, once each: a line for a person, a JSON event for an agent. */
export interface ProgressEvent {
  event: "billing" | "privatelink" | "preparing" | "provisioning" | "box" | "allow_ip";
  /** Where the box runs (the "Monitoring box: <cloud>, <location>" line, platform-all#972). */
  cloud?: string;
  location?: string;
  location_reason?: string;
  /** Seconds since mon deploy started. */
  elapsed_s: number;
  message: string;
  /** event box: the box's state (launch_requested, registered, active, ...). */
  state?: string;
  /** event box: the instance, for following it later (pgai mon instances watch <id>). */
  id?: string;
  /** Supabase: the page where the org approves PostgresAI's access (deploy waits for it). */
  auth_url?: string;
  /** Supabase without a URL: the project picked, as the step view names it. */
  project?: string;
  /** event allow_ip (platform-all#987): the box's IP, reserved before the box, to allow on the database. */
  ip?: string;
  cidr?: string;
  host?: string;
  port?: number;
  reserved_until?: string;
  /** event box, waiting_for_access: why the database refused the box's IP the last time. */
  error?: string;
  /** event privatelink (postgresai#454): the CloudFormation link, and the stack it creates. */
  cfn_url?: string;
  stack_name?: string;
}

export interface ConnectResult {
  status: Status;
  provider: Provider;
  name: string;
  id?: string;
  /** Where a new box runs: aws | hetzner, its region, and why (platform-all#972). */
  cloud?: string;
  location?: string;
  location_reason?: string;
  /** Why there, in words ("same region as your database"). */
  location_reason_text?: string;
  /** Not deployed for want of --yes: where else the box can run, each available cloud's locations. */
  choices?: Record<string, string[]>;
  /** The project's Health Matrix in the console. */
  health_url?: string;
  dashboard_url?: string | null;
  host_metrics?: boolean;
  first_checkup_eta?: string;
  /** A new box: what it costs ("$512.00/month per database cluster (scale plan)", "free (1 of 2 free slots)", "included: same database cluster as ..."). */
  price?: string;
  requires_payment_method?: boolean;
  coupon?: { code: string; valid: boolean; description?: string; error?: string };
  next: string;
  sql?: string;
  /** --provider supabase without a URL: the Supabase organization connected to the org. */
  supabase_org?: string;
  /** --provider supabase without a URL: the projects to pick from (postgresai#438). */
  projects?: SupabaseProject[];
  /**
   * step allow_ip (platform-all#987): the box's IP is reserved and the box waits for the
   * database to allow it: ip (cidr) on host:port, held until reserved_until; last_error is
   * why the database refused the box the last time, if it did.
   */
  step?: "allow_ip" | "privatelink_stack" | "prepare_db";
  ip?: string;
  cidr?: string;
  host?: string;
  port?: number;
  reserved_until?: string;
  last_error?: string;
  /** Why PostgresAI's own connection failed when the IP was reserved (none: it was asked for first). */
  precheck_error?: string;
  /** The IP the box connects to the database from (ready results and reservations). */
  egress_ip?: string;
  /**
   * AWS PrivateLink (postgresai#454). step privatelink_stack: the CloudFormation link that publishes
   * the RDS (cfn_url), the stack it creates, the RDS's region, the External ID it carries (host
   * metrics), and where --terraform wrote the module. step prepare_db: the SQL to run as the master
   * user (`sql`), and the nonce and password it sets, to resume with.
   */
  cfn_url?: string;
  stack_name?: string;
  region?: string;
  external_id?: string;
  terraform_dir?: string;
  nonce?: string;
  monitoring_password?: string;
  /** step privatelink_stack: the link's fields to choose in the form (no AWS here); several stacks for the RDS, to pick one. */
  choose_in_form?: string[];
  stacks?: { name: string; status: string; created: string | null; vpce_service_name: string }[];
}

/** A project of the Supabase organization connected to the org (v1.cloud_monitoring_supabase_projects). */
export interface SupabaseProject {
  ref: string;
  name: string;
  region: string | null;
  /** active, inactive (paused), unhealthy, coming_up, unknown, ... */
  status: string;
  postgres_version: string | null;
  /** Active and not monitored yet. */
  selectable: boolean;
  /** Why it can't be picked: monitored, paused or not_ready:<status> (null when selectable). */
  reason: string | null;
  /** The monitoring box it already has. */
  monitored: { id: string; name: string | null; status: string | null } | null;
}

/** v1.cloud_monitoring_supabase_projects: the org's Supabase projects, or the consent page when it is not connected. */
export type SupabaseProjects =
  | { connected: true; supabase_org_name: string; synced: boolean; projects: SupabaseProject[] }
  | { connected: false; auth_url: string; session_id: string; expires_at?: string };

/** v1.cloud_monitoring_quote: what the next box costs the org. Amounts in cents. */
export interface Quote {
  plan: string;
  org_alias: string;
  billed: boolean;
  free_slots: { remaining: number; total: number; until?: string };
  subscription: boolean;
  quantity: number;
  price: { amount: number; currency: string; interval: string };
  has_payment_method: boolean;
  requires_payment_method: boolean;
  promo?: { code: string; valid: boolean; error?: string; discount_description?: string; duration?: string; duration_in_months?: number };
  amount_after_promo?: number;
  /** The org's billing page on this platform's console (a preview's, on a preview). */
  billing_url?: string;
  /** A database the org already monitors in this cluster: this one is included (billed: false). */
  same_cluster?: string;
}

/** A row of v1.cloud_monitoring_list. */
export interface Database {
  id: string;
  name: string;
  /** lower(host):port of the server, when known. */
  cluster?: string | null;
  /** The percent-encoded database path, when known. */
  database?: string | null;
  provider: string;
  status: string | null;
  /** The project's Health Matrix in the console (added by platformDeps, not the platform). */
  health_url?: string;
  dashboard_url: string | null;
  host_metrics: boolean;
  /** Who feeds the box's host metrics (null: nobody). Absent from platforms before #992, where host_metrics meant a ClickHouse key. */
  host_metrics_source?: "clickhouse" | "supabase" | "rds" | null;
  /** When the box registered with the platform (it then sets up the monitoring). */
  registered_at?: string | null;
  /** The platform keeps postgres_ai_mon's password for this database's server (another database there connects without it). */
  monitoring_password_stored?: boolean;
  /** Why billing failed: a declined first charge (billing starts when the box is active) removes the box. */
  billing_error?: string | null;
  /** The database server's vCPUs; AAS is collected only with them (null: not set). */
  vcpus?: number | null;
  /** Where a box in PostgresAI cloud runs: aws | hetzner and its region (null: self-hosted). */
  cloud?: string | null;
  location?: string | null;
  /** The IP the box connects from: reserved before the box, or its own (platform-all#987). */
  egress_ip?: string | null;
  /** How long an IP reservation without a box is held. */
  reserved_until?: string | null;
  /** Why the database refused the box the last time. */
  last_error?: string | null;
  /** Why PostgresAI could not reach the database before the IP was reserved. */
  precheck_error?: string | null;
}

/** We hold a ClickHouse Cloud key for the box. A platform before #992 sends no host_metrics_source and sets host_metrics only for a key; null means nobody feeds host metrics. */
export const clickhouseKeyHeld = (d: Pick<Database, "host_metrics" | "host_metrics_source">): boolean =>
  d.host_metrics && (d.host_metrics_source === undefined || d.host_metrics_source === "clickhouse");

/**
 * The monitoring URL (with a note for the user, if any), or what to do first.
 * `generated`: this run created the role with a password generated here, which
 * nobody has once the run ends.
 */
export type Prepared = { monitoringUrl: string; note?: string; generated?: true; storedPassword?: true } | { next: string; sql?: string; resettable?: true } | { checked: true };

/** Gets a session's TLS warning (connectWithSslFallback). */
type Warn = (message: string) => void;

export interface ConnectDeps {
  verifyTls?(url: string): Promise<void>;
  list(): Promise<Database[]>;
  /** A row with its console links, for the result shown: the Health Matrix, and Grafana through the PostgresAI sign-in. */
  links?(row: Database): Promise<Database>;
  create(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; status: string; error?: string }>;
  /** `check`: only whether the URL can work, nothing changed ({ checked: true } or what to do first). */
  prepare(url: string, provider: Provider, opts?: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[]; otherRefs?: string[]; monitoringUser?: string; warn?: Warn }): Promise<Prepared>;
  /** Drops the role a `generated` prepare created; false when it could not. */
  unprepare(url: string, monitoringUser?: string, opts?: { warn?: Warn }): Promise<boolean>;
  localStackRunning(): boolean;
  clickhouseOrg(host: string, keyId: string, keySecret: string): Promise<{ orgId: string; state: string }>;
  selfHosted(monitoringUrl: string, env: Record<string, string>): Promise<void>;
  /**
   * Supabase (v1.cloud_monitoring_supabase_*): the org's connection, or the consent page that makes one;
   * whether that consent came back; the box for a project of the connected org.
   */
  supabaseStart(dbUrl: string): Promise<{ connected: true; project_ref: string } | { auth_url: string; session_id: string; expires_at?: string }>;
  supabaseStatus(sessionId: string): Promise<{ status: "pending" | "connected" | "error"; error?: string }>;
  supabaseConnect(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; status: string; error?: string }>;
  /** --provider supabase without a URL: the org's projects (synced first), or the consent page (postgresai#438). */
  supabaseProjects?(): Promise<SupabaseProjects>;
  /** Asks a person at a terminal to pick a project (its ref); undefined when skipped or nobody can be asked. */
  chooseSupabaseProject?(projects: SupabaseProject[], org: string): Promise<string | undefined>;
  /** Opens a page in this machine's browser (a person at a terminal only). */
  openUrl?(url: string): void;
  /**
   * v1.monitoring_deploy_options (platform-all#972): where the box would run for this database
   * and --cloud/--location, decided by the platform (the Console asks the same).
   */
  deployOptions?(args: { db_host?: string; supabase_project_ids?: string[]; cloud?: string; location?: string; check_stock?: boolean }): Promise<DeployOptions>;
  /** What the next box costs, with the coupon checked. */
  /** `cluster`: lower(host):port of the URL, which the price is per. */
  quote(coupon?: string, cluster?: string): Promise<Quote>;
  /** The console page where the org adds a payment method, when the quote names none. */
  billingUrl(orgAlias: string): string;
  /** Asks a person; false when nobody can be asked. */
  confirm(question: string): Promise<boolean>;
  /**
   * A person at a terminal, without --yes, --cloud or --location: deploy where the platform put the box
   * (true), not at all (false), or at the cloud and location they picked from `options`. Absent when
   * nobody can be asked: the result then carries the choices.
   */
  choosePlace?(options: DeployOptions, name: string): Promise<boolean | { cloud: string; location: string }>;
  /** Asks a person for the database server's vCPUs; undefined when skipped or nobody can be asked. */
  askVcpus?(question: string): Promise<number | undefined>;
  /** Sets an instance's vCPUs (v1.cloud_monitoring_set_vcpus, #904). */
  setVcpus?(instanceId: string, vcpus: number): Promise<unknown>;
  /** The platform's per-server lock for --reset-password: one run at a time may reset postgres_ai_mon. */
  resetLock(server: string): Promise<{ lock_id: string }>;
  resetUnlock(lockId: string): Promise<unknown>;
  sleep(ms: number): Promise<void>;
  now(): number;
  progress(event: ProgressEvent): void;
  /** Ctrl-C was pressed: stop before the box is requested; once it is, detach. */
  aborted?(): boolean;
  /**
   * A person at a terminal (platform-all#987): the box's IP is reserved; continue once the
   * database allows it, or stop (it stays reserved). Absent when nobody can be asked.
   */
  allowIp?(lines: string[]): Promise<"continue" | "stop">;
  /**
   * AWS PrivateLink (postgresai#454), v1.cloud_monitoring_rds_*: the org's External ID for the host
   * metrics role, the prepare SQL (a new nonce and password each call), the box, its verification.
   */
  rdsExternalId?(): Promise<string>;
  rdsPrepareSql?(dbName: string): Promise<RdsPrepareSql>;
  rdsProvision?(body: Record<string, string | boolean | number>): Promise<{ id: string; name: string; provider?: string; status: string; region?: string; verified_status?: string; launch_status?: string }>;
  rdsStatus?(id: string): Promise<RdsStatus>;
  /** The aws CLI under a profile (the standard chain without one); absent: AWS is never read. */
  aws?(profile?: string): AwsRun;
  /** The profiles the aws CLI knows, to pick one at a terminal. */
  awsProfiles?(): Promise<string[]>;
  /** The host's addresses as the internet resolves them; undefined when nobody answered. */
  publicAddresses?(host: string): Promise<string[] | undefined>;
  /** Runs the prepare SQL over the URL (one session); refusals are pg's errors. */
  runSql?(url: string, sql: string, warn?: Warn): Promise<void>;
  /** The newest ownership nonce in the database, read as the monitoring role; undefined: no marker. */
  readNonce?(url: string, warn?: Warn): Promise<string | undefined>;
  /** The Terraform module (TERRAFORM_URL), and where its files go (a directory here): the path written. */
  writeTerraform?(dirName: string, files: { mainTf: string; tfvars: string }): Promise<string>;
  terraformModule?(): Promise<string>;
  /** Asks a person at a terminal for a line (Enter: ""); absent when nobody can be asked. */
  ask?(question: string, lines?: string[]): Promise<string>;
}

/** v1.cloud_monitoring_rds_prepare_sql. */
export interface RdsPrepareSql {
  nonce: string;
  db_user: string;
  db_name: string;
  db_password: string;
  prepare_db_sql: string;
}

/** v1.cloud_monitoring_rds_status: the PrivateLink route of a box. */
export interface RdsStatus {
  id: string;
  /** pending until the collector's check passes: db_verified, or unreachable / auth_failed / nonce_mismatch. */
  verified_status: string;
  instance_status: string | null;
  vpce_service_name: string;
  rds_endpoint: string;
  region: string;
  dashboard_url?: string | null;
}

export interface ConnectOptions {
  provider?: string;
  clickhouseKey?: string;
  selfHosted?: boolean;
  /** An existing postgres_ai_mon gets a new password (its old one is not needed). */
  resetPassword?: boolean;
  /** --monitoring-user: the monitoring role (default postgres_ai_mon). */
  monitoringUser?: string;
  waitMs: number;
  /** The URL comes from an agent (the MCP tool): nothing of this process's environment or files goes to the host it names. */
  agent?: boolean;
  /** A billed box is accepted without asking. */
  yes?: boolean;
  /** A Stripe promotion code for the org's subscription. */
  coupon?: string;
  /** --name: the instance's name (default: host[:port]/database). */
  name?: string;
  /** --cloud: aws or hetzner (default: the platform's choice for the database, platform-all#972). */
  cloud?: string;
  /** --location: a region of the box's cloud (default: the platform's choice). */
  location?: string;
  /** --vcpus: the database server's vCPUs; AAS is collected only with them. */
  vcpus?: number;
  /** --supabase-project: the Supabase project (ref or name) to monitor, instead of its URL (postgresai#438). */
  supabaseProject?: string;
  /** --ip-first: reserve the box's IP and stop before the box, even when PostgresAI can reach the database (platform-all#987). */
  ipFirst?: boolean;
  /** --ip-allowed: the reserved IP is allowed on the database; create the box with it. */
  ipAllowed?: boolean;
  /** --privatelink / --no-privatelink: an RDS over AWS PrivateLink, or directly (default: private instances over PrivateLink). */
  privatelink?: boolean;
  /** --vpce-service: the stack's VpceServiceName, when it is not read from AWS here. */
  vpceService?: string;
  /** --host-stats: host CPU, memory, disk and IOPS through a read-only role in the customer's account. */
  hostStats?: boolean;
  /** --host-stats-role: that role's ARN (the stack's HostStatsRoleArn). */
  hostStatsRole?: string;
  /** --aws-profile: the aws CLI profile; false (--no-aws): AWS is not read. */
  awsProfile?: string;
  aws?: boolean;
  /** --terraform: the Terraform module instead of the CloudFormation link. */
  terraform?: boolean;
  /** The printed prepare SQL ran: its password and nonce (PGAI_MON_PASSWORD, PGAI_MON_NONCE), to resume without a new SQL. */
  monitoringPassword?: string;
  nonce?: string;
}

/**
 * An error's HTTP status, read by shape: the bundled CLI holds two copies of
 * HttpStatusError, so `instanceof` can miss the one a call threw.
 */
export function httpStatus(err: unknown): number | undefined {
  const status = err instanceof Error ? (err as { status?: unknown }).status : undefined;
  return typeof status === "number" ? status : undefined;
}

/** Asked at a terminal when --vcpus is missing: AAS (database load) is collected only with it. */
export const VCPUS_QUESTION = "How many vCPUs does the database server have? If you skip this (Enter), you get no AAS (database load) data for this database. vCPUs: ";
/** Said when a box is requested without vCPUs. */
export const NO_VCPUS_NOTE = "Note: no vCPUs given, so you get no AAS (database load) data for this database. Add them later: pgai mon instances set <id-or-name> --vcpus <n>.";

const money = (cents: number, currency: string) =>
  currency.toLowerCase() === "usd" ? `$${(cents / 100).toFixed(2)}` : `${(cents / 100).toFixed(2)} ${currency.toUpperCase()}`;

/** v1.monitoring_deploy_options: the clouds, their locations, and the box the deploy would get. */
export interface DeployOptions {
  clouds: { code: string; label: string; offered?: boolean; available: boolean }[];
  locations: Record<string, { code: string; label: string; world_part: string }[]>;
  choice: { cloud: string; cloud_label: string; location: string; location_label: string | null; reason: string; reason_text: string } | null;
  error: { code: string; message: string; detail: string | null; hint: string | null } | null;
}

/** Where a new box runs, as the result and the progress event carry it. */
type Place = { cloud: string; location: string; location_reason: string; location_reason_text: string };

/** placeBox's answer: the place, and the platform's options it came from (the choices). */
type Placed = { place: Place; options: DeployOptions };

/** The quote as one line: free, included (a cluster already billed), the price per cluster, or the price after the coupon. */
export function priceText(q: Quote): string {
  if (!q.billed && q.same_cluster) return `included: same database cluster as ${q.same_cluster}, no extra charge`;
  if (!q.billed) return q.free_slots.total ? `free (${q.free_slots.remaining} of ${q.free_slots.total} free slots)` : "free";
  const { amount, currency, interval } = q.price;
  const base = `${money(amount, currency)}/${interval} per database cluster (${q.plan} plan)${q.subscription ? `, cluster ${q.quantity + 1} on the subscription` : ""}`;
  if (q.promo?.valid && q.amount_after_promo !== undefined && q.amount_after_promo !== null) {
    const after = `${money(q.amount_after_promo, currency)}`;
    const how = `with ${q.promo.code} (${q.promo.discount_description})`;
    if (q.promo.duration === "forever") return `${after}/${interval} ${how}, instead of ${base}`;
    if (q.promo.duration === "repeating") return `${after}/${interval} for ${q.promo.duration_in_months} months ${how}, then ${base}`;
    return `${after} the first ${interval} ${how}, then ${base}`;
  }
  return base;
}

// The earliest first checkup: the reporter's first run starts 3 minutes after
// the stack starts (REPORTER_INITIAL_DELAY_SECONDS in
// config/scripts/postgres-reports.sh; a test keeps the two equal), then waits
// until every target has data, so a slow first collection makes it later.
export const FIRST_CHECKUP_DELAY_MS = 3 * 60_000;
const POLL_MS = 15_000;
export const PROVIDERS: Provider[] = ["clickhouse", "rds", "supabase", "self-managed"];

/**
 * --provider as given, trimmed and case-insensitive (as pgai dblab deploy reads it).
 * Generic names the direct PostgreSQL setup path, not who operates the database:
 * it keeps the established value, self-managed, in results and preparation.
 */
export const providerArg = (provider: string | undefined): string | undefined => {
  const p = provider?.trim().toLowerCase();
  return p === "generic" ? "self-managed" : p;
};

/** `new URL`, or undefined for text that is not a URL (URL.canParse needs Node 18.17). */
export function parseUrl(url: string): URL | undefined {
  try {
    return new URL(url);
  } catch {
    return undefined;
  }
}

export function detectCloudProvider(url: string): Provider {
  const host = new URL(url).hostname.toLowerCase().replace(/\.$/, "");
  if (host.endsWith(".clickhouse.cloud")) return "clickhouse";
  if (host.endsWith(".rds.amazonaws.com")) return "rds";
  if (host.endsWith(".supabase.co") || host.endsWith(".pooler.supabase.com")) return "supabase";
  return "self-managed";
}

/**
 * The instance name the platform derives from the monitoring URL
 * (monitoring_instance_create, #712). That URL always names the database, so
 * for a URL without one this is the database pg connects to (PGDATABASE, else the user).
 */
export function databaseName(url: string): string {
  const u = new URL(url);
  const port = u.port && u.port !== "5432" ? `:${u.port}` : "";
  let db = decodeURIComponent(u.pathname.replace(/^\//, ""));
  if (!db) {
    try {
      db = new Client(resolveAdminConnection({ conn: url }).clientConfig).database ?? "";
    } catch {
      // Not a URL pg can use: the connection reports that.
    }
  }
  return `${u.hostname}${port}${db ? `/${db}` : ""}`;
}

/** `<key-id>:<key-secret>`, from --clickhouse-key or CLICKHOUSE_KEY_ID + CLICKHOUSE_KEY_SECRET. */
export function parseClickhouseKey(value: string | undefined, env: Record<string, string | undefined>) {
  if (!value) {
    return env.CLICKHOUSE_KEY_ID && env.CLICKHOUSE_KEY_SECRET
      ? { keyId: env.CLICKHOUSE_KEY_ID, keySecret: env.CLICKHOUSE_KEY_SECRET }
      : undefined;
  }
  const i = value.indexOf(":");
  if (i <= 0 || i === value.length - 1) throw new Error("--clickhouse-key must be <key-id>:<key-secret>");
  return { keyId: value.slice(0, i), keySecret: value.slice(i + 1) };
}

/** A delete in flight (not one that failed to launch, which can be retried). */
export const disconnecting = (status: string | null) => /delet/.test(status ?? "") && !/fail/.test(status ?? "");

/** The platform's instance state in the words mon deploy, mon instances status and mon instances list all use. */
export function stateOf(raw: string | null): Status {
  return monitoringStatus(raw);
}

/** A rejected ClickHouse Cloud key, or one that cannot see the service: the user's to fix (exit 3). */
export class ClickhouseKeyError extends Error {}

/** Ctrl-C before the box was requested: the run stopped, its lock released and its generated role dropped. */
export class CancelledError extends Error {
  constructor() {
    super("Cancelled: nothing was created.");
  }
}

/** The server of a cluster or legacy database name, ignoring case and one trailing dot. */
const serverOf = (name: string) => name.split("/")[0].toLowerCase().replace(/\.(?=$|:)/, "");

/** lower(host):port of a URL (5432 when it names none): the cluster the price is per, as the platform keys it. */
export function clusterOf(url: string): string | undefined {
  const u = parseUrl(url);
  return u?.hostname ? `${serverOf(u.hostname)}:${u.port || "5432"}` : undefined;
}

/** An error's text; an AggregateError (node tried each address of a host name) has none of its own: its errors'. */
export const errorText = (err: unknown): string => {
  if (err instanceof AggregateError && !err.message) return err.errors.map(errorText).join("; ");
  // A platform refusal: its own words (details, hint), not the "<call>: HTTP 400 - Bad Request" line above them.
  if (err instanceof HttpStatusError) {
    const [first, ...rest] = err.message.split("\n");
    if (rest.length && /: HTTP \d{3}\b/.test(first)) return rest.join("\n");
  }
  return err instanceof Error ? err.message : String(err);
};

const urlPassword = (u: URL) => {
  for (const key of [...u.searchParams.keys()]) if (key !== "password") u.searchParams.delete(key);
  return parseConnString(u.toString()).password || "";
};

// What of the given URL's query string goes to the box: `options`, certificate
// paths and the rest describe this machine's session, not the box's.
const URL_PARAMS_KEPT = ["sslmode", "channel_binding", "application_name"];
// Kept for the logins made from this machine: the files are here.
const URL_PARAMS_TLS = ["sslrootcert", "sslcert", "sslkey", "uselibpqcompat"];

/**
 * Refuses query parameters that pg obeys over the URL itself. `host` and `port`
 * would prepare one server while the name and the box's URL say another. An
 * agent's URL may carry only what the box gets (and its password): `sslcert`
 * or `sslkey` would present this machine's files to a host the agent chose.
 */
export function checkUrlParams(url: string, agent?: boolean) {
  const u = new URL(url);
  const keys = [...new Set(u.searchParams.keys())];
  if (agent) {
    for (const key of ["password", "user", "host", "port", "dbname"]) {
      if (u.searchParams.getAll(key).length > 1) throw new Error(`database_url query parameter ${key} must appear only once`);
    }
    if (u.password && u.searchParams.has("password")) throw new Error("database_url must give the password only once, in the authority or in ?password=");
    const refused = keys.filter((k) => k !== "password" && !URL_PARAMS_KEPT.includes(k));
    if (refused.length) throw new Error(`database_url may carry only these query parameters: ${URL_PARAMS_KEPT.join(", ")} (got: ${refused.join(", ")})`);
    u.searchParams.delete("sslmode");
    u.searchParams.delete("channel_binding");
    const config = parseIntoClientConfig(u.toString());
    if (!/^postgres(ql)?:$/.test(u.protocol) || typeof config.password !== "string" || !config.password) throw new Error("database_url must be postgresql://user:password@host:5432/dbname, with the password in it");
    return { ...config, password: config.password };
  }
  const moved = keys.filter((k) => k === "host" || k === "port");
  if (moved.length) throw new Error(`The URL's query string sets ${moved.join(" and ")}: put the host and the port in the URL itself (postgresql://user:password@host:5432/dbname), so that the server prepared is the server monitored`);
}

/**
 * What a box URL without a password lacks before the platform fills in the one it keeps
 * (cloud_monitoring_connect): sslmode=require or verify-*, each kept parameter once, no empty
 * part. Undefined when it lacks nothing.
 */
function storedPasswordNeeds(monitoringUrl: string): string | undefined {
  const query = new URL(monitoringUrl).search.replace(/^\?/, "");
  const parts = query ? query.split("&") : [];
  const keys = parts.map((p) => p.split("=")[0]);
  const twice = URL_PARAMS_KEPT.filter((k) => keys.indexOf(k) !== keys.lastIndexOf(k));
  const tls = !twice.includes("sslmode") && parts.some((p) => /^sslmode=(require|verify-ca|verify-full)$/.test(p));
  const needs = [
    ...(tls ? [] : [`sslmode=require (or verify-full)${twice.includes("sslmode") ? ", once" : ""}`]),
    ...twice.filter((k) => k !== "sslmode").map((k) => `${k} once`),
    ...(parts.some((p) => !p) ? ["no stray &"] : []),
    // A kept key spelled with %-escapes: the platform reads the query string as written.
    ...(keys.some((k) => k && !URL_PARAMS_KEPT.includes(k)) ? ["the parameter names without %-escapes"] : []),
  ];
  return needs.length ? needs.join(" and ") : undefined;
}

/** A word for the user's shell: quoted unless plain (a name made in the console is free text). */
const shellWord = (s: string) => (/^[\w./:@%+=,-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);

/** What `pgai mon instances delete` takes for `d`: its name, or its id when another live box has that name. */
const deleteRef = (d: Database, list: Database[]) =>
  (list.some((o) => o.id !== d.id && !disconnecting(o.status) && o.name === d.name) ? d.id : d.name);

/** `pgai mon instances delete` for each database named. */
const disconnectEach = (names: string[]) => names.map((n) => `pgai mon instances delete ${shellWord(n)} --yes`).join(" and ");

/**
 * The monitoring role's login through `url`: a URL user `<session user>.<suffix>` (the
 * Supabase pooler's project ref, which the pooler routes on) keeps its suffix.
 */
function loginName(url: string, sessionUser: string, role: string): string {
  const user = decodeURIComponent(new URL(url).username);
  return user.startsWith(`${sessionUser}.`) ? role + user.slice(sessionUser.length) : role;
}

/** The monitoring role's URL (`user` logs in as it) for the prepared database, with the query parameters in `kept`. */
function roleUrlFor(url: string, user: string, db: string, password: string, kept: string[]): string {
  const u = new URL(url);
  u.username = encodeURIComponent(user);
  u.password = encodeURIComponent(password);
  u.pathname = `/${encodeURIComponent(db)}`;
  for (const name of [...u.searchParams.keys()]) if (!kept.includes(name)) u.searchParams.delete(name);
  // `mon targets add` refuses a raw '@' after the host (a delete above already encoded it).
  u.search = u.search.replace(/@/g, "%40");
  return u.toString();
}

/** The URL a box uses: only postgres_ai_mon's credentials, and the prepared database by name. */
const monitoringUrlFor = (url: string, user: string, db: string, password: string) => roleUrlFor(url, user, db, password, URL_PARAMS_KEPT.filter((p) => p !== "channel_binding"));

/** The same login from this machine: with the given URL's TLS files (a private CA, say). */
const loginUrlFor = (url: string, user: string, db: string, password: string) => roleUrlFor(url, user, db, password, [...URL_PARAMS_KEPT, ...URL_PARAMS_TLS]);

/** What to tell the user when the URL verifies the server with a CA file the box will not have. */
export function caNote(url: string): string | undefined {
  const q = new URL(url).searchParams;
  if (!q.get("sslrootcert") || !/^verify-/.test(q.get("sslmode") ?? "")) return undefined;
  return `the monitoring box has no copy of the CA in sslrootcert: with sslmode=${q.get("sslmode")} it connects only to a server certificate signed by a public CA (else use sslmode=require)`;
}

/** Why the box is removed for its first charge, and the fix: a promo code Stripe refused needs another code, not another card. */
function chargeFailed(error: string, rerun: string, billingUrl?: string, couponFlag = "--coupon"): string {
  const fix = /promotion code|coupon/i.test(error)
    ? `The promo code cannot be used by this organization: ${rerun} without ${couponFlag}, or with a code this organization can use`
    : `Update the payment method${billingUrl ? ` at ${billingUrl}` : ""}, then ${rerun}`;
  return `The first charge failed (${error}): the box is being removed and nothing is billed. ${fix}`;
}

/** A platform row as a result with its next action; `fresh` (provisioned just now) adds the first-checkup ETA. */
export function connectStatus(row: Database, provider = row.provider as Provider, fresh = false): ConnectResult {
  const base = { provider, name: row.name, id: row.id, ...(row.health_url ? { health_url: row.health_url } : {}), dashboard_url: row.dashboard_url, host_metrics: row.host_metrics };
  const status = stateOf(row.status);
  // Removed because its first charge failed: the user's to fix, whichever command reads it (deploy, watch, list).
  if (row.billing_error && (status === "deleting" || status === "deleted")) {
    return { ...base, status: "action_required",
      next: `${chargeFailed(row.billing_error, "re-run pgai mon deploy")}.` };
  }
  if (status === "ready") {
    return {
      ...base, status,
      ...(row.egress_ip ? { egress_ip: row.egress_ip } : {}),
      ...(fresh ? { first_checkup_eta: new Date(Date.now() + FIRST_CHECKUP_DELAY_MS).toISOString() } : {}),
      next: row.health_url || row.dashboard_url ? `Open ${row.health_url || row.dashboard_url}` : `pgai mon instances watch ${row.id}`,
    };
  }
  if (status === "deleting" || status === "deleted" || status === "inactive") return { ...base, status, next: "none" };
  if (status === "failed") return { ...base, status, next: `pgai mon instances delete ${row.id} --yes, then pgai mon deploy again` };
  // platform-all#987: list and watch show a reservation as the step it waits on (exit 3), not as in progress.
  if (row.status === "ip_reserved" && row.egress_ip) return { ...allowIpResult(row, provider, row.name, { waitMs: 0 }, ""), ...base, status: "action_required", step: "allow_ip" };
  return { ...base, status, next: `pgai mon instances watch ${row.id}` };
}

/** What renders progress: JSON events, plain lines, or a step view at a terminal. */
export interface StepViewLike {
  start(): void;
  advance(key: string): void;
  log(text: string): void;
  setNote(note: string): void;
  complete(): void;
  fail(key?: string): void;
  stop(): void;
}

/**
 * mon deploy's progress: JSON events (--json), a line a step (piped), or at a
 * terminal a step view that starts only after the price, since a prompt may
 * follow the price; with no price (a re-run, --self-hosted) no view at all.
 */
// ESC or C1 sequences: CSI (colors, cursor, clear), OSC (titles, links) to its end, and two-byte ones (a reset).
const ESCAPES = /(?:\x1b\[|\x9b)[0-?]*[ -\/]*[@-~]|(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?|\x1b[ -~]?/g;

/** Text from the platform, for a terminal: escape sequences dropped, other control characters a space; line breaks kept. */
export const terminalText = (text: string): string => text.replace(ESCAPES, "").split("\n").map(stripControlsToSpace).join("\n");

export function stepProgress(o: { json: boolean; tty: boolean; makeView: () => StepViewLike; writeLine: (s: string) => void;
  writeEvent: (e: ProgressEvent) => void; onId?: (id: string) => void }) {
  let view: StepViewLike | undefined;
  let priced = false;
  // platform-all#987: the database's last refusal of the box's IP, shown once each time it changes.
  let refused = "";
  const progress = (e: ProgressEvent): void => {
    // The reserved IP's event names no box yet: only a box request makes Ctrl-C detach.
    if (e.id && e.event !== "allow_ip") o.onId?.(e.id);
    if (o.json) return o.writeEvent(e);
    e = { ...e, message: terminalText(e.message), ...(e.state ? { state: terminalText(e.state) } : {}),
      ...(e.ip ? { ip: terminalText(e.ip) } : {}), ...(e.error ? { error: terminalText(e.error) } : {}) };
    if (!o.tty) return o.writeLine(progressText(e));
    if (e.event === "billing") {
      priced = true;
      return o.writeLine(e.message);
    }
    if (!view && !priced) return o.writeLine(e.message);
    if (!view) {
      view = o.makeView();
      view.start();
    }
    view.advance(e.event);
    // PrivateLink (postgresai#454): the link and the stack used stay on screen; the wait is a note under the step.
    const kept = e.event === "privatelink" && (!!e.cfn_url || e.message.startsWith("Using stack "));
    if (/^(Note|Warning):/.test(e.message) || kept) view.log(e.message);
    if (e.event === "privatelink" && !kept && !/^(Note|Warning):/.test(e.message)) view.setNote(e.message);
    if (e.event === "box" && e.state) {
      view.setNote(e.state === "waiting_for_access" && e.ip ? `box: waiting for database access from ${e.ip}` : `box: ${boxWords(e.state)}`);
      if (e.state === "waiting_for_access" && e.ip && e.error && e.error !== refused) {
        refused = e.error;
        view.log(`The database refused ${e.ip}: ${e.error}. The box keeps trying: allow ${e.ip}/32 on the database.`);
      }
    }
    if (e.message === STILL_CREATING) view.setNote(`still creating: PostgresAI is slow to answer (checking for up to ${LOOKUP_MS / 60_000} min)`);
  };
  // After a prompt drawn over a stopped view (the reserved IP, platform-all#987), the next step gets a new view.
  return { progress, view: () => view, reset: () => { view = undefined; } };
}

/** How the step list ends: a run stopped at the reserved IP (platform-all#987) or still in progress is left waiting, not failed. */
export function stepViewEnd(r: ConnectResult): "complete" | "fail" | "stop" {
  if (r.status === "ready") return "complete";
  if (r.status === "in_progress" || r.step || r.next.startsWith(STILL_IN_PROGRESS)) return "stop";
  return "fail";
}

/** A result for a person at a terminal: a few sentences, links whole (JSON keeps every field). */
export function resultLines(r: ConnectResult): string[] {
  return personLines(r).map(terminalText);
}

/** A box's cloud as users read it (the platform's labels). */
const cloudWord = (cloud: string): string => ({ aws: "AWS", hetzner: "Hetzner" } as Record<string, string>)[cloud] ?? cloud;

function personLines(r: ConnectResult): string[] {
  const who = r.name || "this database";
  // What the next action carries after its first part ("; --name ignored ...", a prepare note).
  const extra = (r.next ?? "").split("; ").slice(1).join("; ");
  const billing = (r as { billing?: string }).billing;
  switch (r.status) {
    case "ready":
      return [
        `Monitoring for ${who} is ready.`,
        ...(r.health_url ? [`  Health Matrix: ${r.health_url}`] : []),
        ...(r.dashboard_url ? [`  Grafana:       ${r.dashboard_url}`] : []),
        ...(r.price ? [`  Price:         ${r.price}`] : []),
        ...(r.cloud ? [`  Runs on:       ${cloudWord(r.cloud)}, ${r.location}`] : []),
        ...(r.egress_ip ? [`  Connects from: ${r.egress_ip}`] : []),
        ...(r.provider === "rds" && r.host_metrics ? ["  Host metrics:  on (CPU, memory, disk, IOPS)"] : []),
        ...(!r.health_url && !r.dashboard_url && r.next && r.next !== "none" ? [`Next: ${r.next}`] : []),
        ...(extra && (r.health_url || r.dashboard_url) ? [`Note: ${extra}`] : []),
      ];
    case "in_progress":
      return [`Monitoring for ${who} is still being set up. Follow it: ${r.next}`];
    case "deleting":
    case "deleted":
      return [`Deleting the monitoring of ${who}.${billing ? ` Billing: ${billing}.` : ""}${r.next && r.next !== "none" ? ` ${r.next}` : ""}`];
    default:
      // platform-all#987: the IP to allow, then how to go on.
      if (r.step === "allow_ip") {
        // After [s]top at the prompt (which showed the IP and why): only how to go on.
        if (r.next.startsWith("Stopped.")) return [r.next];
        return [
          `Monitoring box IP: ${r.ip}${r.cloud ? ` (${cloudWord(r.cloud)}, ${r.location})` : ""}${r.reserved_until ? `, held until ${heldUntil(r.reserved_until)}` : ""}`,
          r.last_error ? `The box could not connect: ${r.last_error}`
            : r.precheck_error ? `PostgresAI could not connect to ${r.host}:${r.port} (${r.precheck_error}).`
            : "The IP was reserved first, as asked.",
          r.next,
        ];
      }
      // Supabase without a URL: the projects one per line under the reason (JSON has them in `projects`).
      if (r.projects?.length) return [r.next, ...supabaseProjectRows(r.projects)];
      // PrivateLink (postgresai#454): the stack's link on its own line, whole.
      if (r.step === "privatelink_stack" && r.cfn_url && !r.terraform_dir) return [r.next, `  ${r.cfn_url}`];
      // Not deployed for want of --yes: where else it can run, a line per cloud (JSON has them in `choices`).
      return r.choices ? [r.next, ...Object.entries(r.choices).map(([cloud, codes]) => `  --cloud ${cloud} --location ${codes.join("|")}`)] : [r.next];
  }
}

/** A progress event as a person reads it: the time since the start ends its first line (before a colon). */
export function progressText(e: ProgressEvent): string {
  const s = Math.round(e.elapsed_s);
  const elapsed = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  const [first, ...rest] = e.message.split("\n");
  const colon = first.endsWith(":") ? ":" : "";
  return [`${first.slice(0, first.length - colon.length)} (+${elapsed})${colon}`, ...rest].join("\n");
}

/** The box's state as deploy shows it: registered between the launch and active. */
const boxState = (row: Database) => (row.status === "launch_requested" && row.registered_at ? "registered" : row.status ?? "starting");

/** The box's state in words (the raw state stays in the JSON event's `state`). */
export function boxWords(state: string | null | undefined): string {
  if (!state || state === "starting" || state === "launch_requested") return "starting";
  if (state === "registered") return "installing monitoring";
  if (state === "waiting_for_access") return "waiting for database access";
  if (state === "ip_reserving") return "reserving its IP";
  if (state === "ip_reserved") return "waiting for its IP to be allowed";
  return monitoringStatus(state).replace("_", " ");
}

type Billing = Pick<ConnectResult, "price" | "requires_payment_method" | "coupon">;

/**
 * A new box's billing: its fields for the result, the billing page for a
 * missing payment method, and `stop` when something comes first (a coupon
 * that does not apply, no payment method, a billed box not accepted).
 */
async function billingFor(q: Quote, name: string, opts: ConnectOptions, deps: ConnectDeps, show: (price: string) => void,
  ask?: (price: string) => Promise<Decision>): Promise<{ billing: Billing; billingPage: Billing & { next: string }; billingUrl: string; accepted: boolean; stop?: Partial<ConnectResult> & { next: string }; placed?: Placed }> {
  const billing: Billing = { price: priceText(q), requires_payment_method: q.requires_payment_method };
  if (opts.coupon) {
    billing.coupon = q.promo?.valid
      ? { code: q.promo.code, valid: true, description: q.promo.discount_description }
      : { code: opts.coupon, valid: false, error: q.promo?.error ?? "not valid" };
  }
  const billingUrl = q.billing_url ?? deps.billingUrl(q.org_alias);
  const billingPage = { ...billing, requires_payment_method: true, next: noCardNext(billingUrl) };
  const stop = (next: string) => ({ billing, billingPage, billingUrl, accepted: false, stop: { ...billing, next } });
  if (billing.coupon && !billing.coupon.valid) {
    return stop(`Promo code ${opts.coupon}: ${billing.coupon.error}. Nothing was changed: re-run with a valid code, or without ${opts.agent ? "coupon" : "--coupon"}`);
  }
  if (q.requires_payment_method) return { billing, billingPage, billingUrl, accepted: false, stop: billingPage };
  // The price on its own line first: a prompt may wrap or be cut where it is shown.
  show(billing.price!);
  // The platform placed the box: one question for any box, free or billed (A17).
  if (ask) {
    const d = await ask(billing.price!);
    if (!d.go) return { billing, billingPage, billingUrl, accepted: false, stop: { ...billing, ...d.stop } };
    return { billing, billingPage, billingUrl, accepted: q.billed, placed: d.place };
  }
  // A platform from before --cloud: a billed box only, as before.
  if (q.billed && !opts.yes && !(await deps.confirm(`Provision ${terminalText(name)}? (y/N): `))) {
    return stop(opts.agent ? `Call deploy_monitoring again with yes: true to accept ${billing.price}` : `Re-run with --yes to accept ${billing.price}`);
  }
  return { billing, billingPage, billingUrl, accepted: q.billed };
}

export async function connect(url: string, opts: ConnectOptions, deps: ConnectDeps): Promise<ConnectResult> {
  const started = deps.now();
  let step: ProgressEvent["event"] = "preparing";
  const progress = (event: ProgressEvent["event"], message: string, more: Partial<ProgressEvent> = {}) => {
    if (event !== "billing") step = event;
    deps.progress({ event, elapsed_s: Math.round((deps.now() - started) / 1000), message, ...more });
  };
  // Each session warns about its TLS: said once a run, as progress (a raw stderr line breaks the step view).
  const warned = new Set<string>();
  const warn = (message: string) => {
    if (warned.has(message)) return;
    warned.add(message);
    progress(step, message);
  };
  // --supabase-project names a Supabase project: no URL is needed (postgresai#438).
  const project = opts.supabaseProject?.trim();
  const provider = (providerArg(opts.provider) ?? (!url && project ? "supabase" : detectCloudProvider(url))) as Provider;
  if (!PROVIDERS.includes(provider)) throw new Error(`--provider must be one of: generic, ${PROVIDERS.join(", ")}`);
  if (project !== undefined && provider !== "supabase") {
    return { status: "action_required", provider, name: url ? databaseName(url) : "", next: "--supabase-project applies to Supabase only: leave it out, or use --provider supabase" };
  }
  // platform-all#987: before the Supabase branches, which would ignore the flags and create the box.
  if ((opts.ipFirst || opts.ipAllowed) && (opts.selfHosted || provider === "supabase")) {
    return { status: "action_required", provider, name: url ? databaseName(url) : "", next: `${opts.agent ? "ip_first / ip_allowed apply" : "--ip-first and --ip-allowed apply"} to PostgresAI Cloud monitoring of a database URL only` };
  }
  if (!url) {
    if (provider !== "supabase") return { status: "action_required", provider, name: "", next: "Pass the database URL: pgai mon deploy postgresql://user:password@host:5432/dbname (Supabase: --provider supabase)" };
    if (opts.selfHosted) return { status: "action_required", provider, name: "", next: "--self-hosted needs the database URL: pass the project's connection string" };
    return deploySupabaseProject(undefined, opts, deps, progress);
  }
  // Before any name is taken from the URL (the MCP path does not go through the CLI's argument check).
  assertEncodedUserinfo(url);
  // Supabase is never logged in to from here: its URL only names the project.
  if (!opts.selfHosted && provider === "supabase") return deploySupabase(url, opts, deps, progress);
  checkUrlParams(url, opts.agent);
  const name = databaseName(url);
  // The platform keeps postgres_ai_mon's password only; --reset-password locks the server for any role.
  const asked = opts.monitoringUser?.trim();
  if (opts.monitoringUser !== undefined && !asked) {
    const flag = opts.agent ? "monitoring_user" : "--monitoring-user";
    return { status: "action_required", provider, name, next: `${flag} is empty: pass a role name, or leave it out for ${DEFAULT_MONITORING_USER}` };
  }
  const monitoringUser = asked && asked !== DEFAULT_MONITORING_USER ? asked : undefined;
  const role = monitoringUser ?? DEFAULT_MONITORING_USER;
  // The setup SQL puts the role in quoted literals, dollar-quoted DO bodies and format() strings.
  if (monitoringUser && /['$%\\\u0000-\u001f\u007f]/.test(monitoringUser)) {
    return { status: "action_required", provider, name, next: `${opts.agent ? "monitoring_user" : "--monitoring-user"} must not contain ', $, %, backslash or control characters` };
  }
  if (monitoringUser && Buffer.byteLength(monitoringUser) > 63) {
    return { status: "action_required", provider, name, next: `${opts.agent ? "monitoring_user" : "--monitoring-user"} must be at most 63 bytes (Postgres truncates longer role names)` };
  }
  try {
    const collector = await collectorConnection(url, deps.verifyTls);
    const required = splitChannelBinding(url).value === "require";
    url = collector.url;
    if (collector.note) progress("preparing", `${required ? "Warning" : "Note"}: ${collector.note}`);
    if (required && caNote(url)) progress("preparing", `Note: ${caNote(url)}`);
  } catch (err) {
    return { status: "action_required", provider, name, next: err instanceof Error ? err.message : String(err) };
  }
  const cluster = clusterOf(url)!;
  let vcpus = opts.vcpus;
  // The exported key pair is read for ClickHouse only, and never for an agent's URL; elsewhere only the flag is an error.
  const key = parseClickhouseKey(opts.clickhouseKey, provider === "clickhouse" && !opts.agent ? process.env : {});
  if (key && provider !== "clickhouse") throw new Error("--clickhouse-key applies to ClickHouse Managed Postgres only");

  // An RDS that is not publicly accessible: over AWS PrivateLink (postgresai#454); a public one goes on below.
  const plFlag = privateLinkFlag(opts);
  if (!opts.selfHosted && provider === "rds") {
    const route = await rdsRoute(url, opts, deps, progress);
    if ("status" in route) return { ...route, provider, name };
    if (route.private) return deployPrivateLink(url, route, { ...opts, monitoringUser }, deps, progress);
    if (plFlag) return { status: "action_required", provider, name, next: `${plFlag} applies to AWS PrivateLink only, and ${name} is monitored directly${opts.privatelink === false ? ` (${opts.agent ? "privatelink: false" : "--no-privatelink"})` : " (it is publicly accessible)"}` };
  } else if (plFlag) {
    return { status: "action_required", provider, name, next: `${plFlag} applies to Amazon RDS monitored from PostgresAI Cloud only` };
  }

  if (opts.selfHosted && opts.resetPassword) {
    return { status: "action_required", provider, name, next: "--reset-password works with PostgresAI Cloud only (it checks what else monitors this server): deploy without --self-hosted, or set PGAI_MON_PASSWORD" };
  }

  if (opts.selfHosted && deps.localStackRunning()) {
    return { status: "action_required", provider, name, next: "A monitoring stack already runs on this machine: add the database with PGAI_DB_URL='<postgres_ai_mon URL>' pgai mon targets add" };
  }

  const byCluster = (d: Database) => !!d.cluster && serverOf(d.cluster) === cluster && d.database != null && decodeURIComponent(d.database) === name.split("/").slice(1).join("/");
  const byName = (d: Database) => (d.cluster == null || d.database == null) && d.name === name;
  const findDatabase = (list: Database[]) => {
    const live = list.filter((d) => !disconnecting(d.status));
    return live.find(byCluster) ?? live.find(byName);
  };
  const rows = opts.selfHosted ? [] : await deps.list();
  // The box a request without an answer made: a row of this database that was not there before, live or not.
  // platform-all#987: a Continue gives the box to the reservation's own row, which reads reserved
  // until the platform commits it. A row this run knew as a reservation (before the request, or
  // from its own first reply) counts once it has left the reservation; a new one, reserved or
  // not, is what the request made.
  const known = new Map(rows.map((d) => [d.id, d.status] as const));
  const findMade = (list: Database[]) => {
    const fresh = list.filter((d) => !known.has(d.id) || (reservedStatus(known.get(d.id)) && !reservedStatus(d.status)));
    return fresh.find(byCluster) ?? fresh.find(byName);
  };
  let row = findDatabase(rows);
  // platform-all#987: this database's IP is reserved. Without --ip-allowed it is shown again
  // (the same IP; no second one is ever reserved); with it, or once a person at the terminal
  // continues, the deploy below gives the reservation its box.
  let continuing = false;
  // The reservation being continued: not another database on this server (its role is this one's).
  let ownId: string | undefined;
  if (row && reservedStatus(row.status)) {
    if (!opts.ipAllowed) {
      const asked = await allowIpStep(row, { provider, name, url, opts, deps, progress });
      if (asked !== "continue") return asked;
    }
    continuing = true;
    ownId = row.id;
    row = undefined;
  }
  const fresh = !row;
  let note = "";
  let billing: Billing = {};
  let billingPage: Billing & { next: string } | undefined;
  let accepted = false;
  let billingUrl: string | undefined;
  let place: Place | undefined;
  // Drops the role this run created with a generated password, once its box is gone.
  let undoRole: (() => Promise<unknown>) | undefined;
  // The key is checked first, on a re-run too: a rejected key or a stopped service changes nothing.
  const findOrg = async () => {
    try {
      return await deps.clickhouseOrg(new URL(url).hostname, key!.keyId, key!.keySecret);
    } catch (err) {
      if (err instanceof ClickhouseKeyError) return { error: err.message };
      throw err;
    }
  };
  // A database already monitored keeps its name: --name names a new instance only.
  if (row && opts.name && row.name !== opts.name) note = `; --name ignored: this database is already monitored as ${row.name}`;
  // Its vCPUs (#904): missing ones are set (--vcpus, or asked at a terminal); a set
  // value is never changed here. ClickHouse's come from ClickHouse Cloud.
  if (row && !disconnecting(row.status) && provider !== "clickhouse") {
    const has = (row.vcpus ?? 0) > 0;
    let want = opts.vcpus;
    if (want === undefined && !has && !opts.yes && !opts.agent && deps.askVcpus) want = await deps.askVcpus(VCPUS_QUESTION);
    if (want !== undefined && !has && deps.setVcpus) {
      try {
        await deps.setVcpus(row.id, want);
        note += `; vCPUs set to ${want}: AAS (database load) is collected from now on`;
      } catch (err) {
        if (httpStatus(err) !== 403) throw err;
        note += `; vCPUs not set: only an organization admin can (pgai mon instances set ${row.id} --vcpus ${want})`;
      }
    } else if (want !== undefined && has && row.vcpus !== want) {
      note += `; --vcpus ignored: ${row.name} already has ${row.vcpus} vCPUs (to change them: pgai mon instances set ${row.id} --vcpus ${want})`;
    } else if (!has) {
      note += `; no vCPUs, so no AAS (database load) data for this database (add them: pgai mon instances set ${row.id} --vcpus <n>)`;
    }
  }
  if (row && opts.resetPassword) {
    return { status: "action_required", provider, name, id: row.id, next: `${row.name} is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: ${disconnectEach([deleteRef(row, rows)])}` };
  }
  if (row && key) {
    const found = await findOrg();
    if ("error" in found) return { status: "action_required", provider, name, id: row.id, next: `${found.error} Nothing was changed: re-run with the right key, or without one` };
  }
  if (!row) {
    if (!/^[A-Za-z0-9._-]*$/.test(new URL(url).pathname.replace(/^\//, "")) && rows.some((d) =>
      !disconnecting(d.status) && d.cluster && serverOf(d.cluster) === cluster && d.database == null && /^Monitoring [0-9a-f]{6}$/.test(d.name))) {
      return { status: "action_required", provider, name, next: "An earlier box on this server may already monitor this database (its name is not known): see pgai mon instances list, and delete that box first (pgai mon instances delete <id>) if it is this database" };
    }
    let ch: { orgId: string } | undefined;
    if (key) {
      const found = await findOrg();
      if ("error" in found) return { status: "action_required", provider, name, next: `${found.error} Then re-run` };
      if (found.state !== "running") {
        return { status: "action_required", provider, name, next: `Start the service in the ClickHouse Cloud console (it is ${found.state}), then re-run` };
      }
      ch = found;
    }
    // A refused launch starts no box, so a role with a generated password is
    // dropped again: nobody has that password, and the re-run would stop at it.
    const undo = async (p: Prepared) => { if ("generated" in p && p.generated) await deps.unprepare(url, monitoringUser, { warn }).catch(() => false); };
    let lockId: string | undefined;
    // The role is one for the whole server: a new password cuts off what uses the old one.
    // Without a cluster, the server is not known.
    const unread = (d: Database) => !d.cluster;
    const onServer = (d: Database) => !!d.cluster && serverOf(d.cluster) === cluster;
    // What may log in as the role: not a box being destroyed or gone, a deleted self-hosted one
    // (deactivated: inactive), or a launch that failed before its box registered (it never connected).
    // Only the raw status failed is a launch outcome: error is a running box whose health check failed.
    const usesRole = (d: Database) =>
      !["deleting", "deleted", "inactive"].includes(stateOf(d.status)) && !(d.status === "failed" && !d.registered_at);
    const othersOnServer = (list: Database[]) => list.filter((d) => d.id !== ownId && usesRole(d) && (unread(d) || onServer(d)));
    // The platform keeps postgres_ai_mon's password for a server it monitors a database on, for this org,
    // and fills it in while a box there is not being destroyed (a delete that failed counts as one).
    const storedPassword = !monitoringUser && rows.some((d) => d.monitoring_password_stored && !/delet/.test(d.status ?? "") && onServer(d)) || undefined;
    const cutOff = (others: Database[], list: Database[]): ConnectResult => {
      const label = (d: Database) => (deleteRef(d, list) === d.id ? `${d.name} (id ${d.id})` : d.name);
      const on = others.filter((d) => !unread(d)).map(label).join(", ");
      const maybe = others.filter(unread).map(label).join(", ");
      const what = [on && `would cut off the monitoring of ${on} on this server`, maybe && `may cut off the monitoring of ${maybe} (the platform could not read its URL, so its server is not known)`].filter(Boolean).join(", and ");
      const instead = [
        ...(storedPassword ? ["deploy without --reset-password (PostgresAI keeps its password for this server, and sends it only over TLS)"] : []),
        "set PGAI_MON_PASSWORD to its password instead",
        `${disconnectEach(others.map((d) => deleteRef(d, list)))} first`,
      ];
      return { status: "action_required", provider, name, next: `A new password for ${role} ${what}: ${instead.join(", or ")}` };
    };
    const onThisServer = othersOnServer(rows).filter((d) => !unread(d));
    const others = onThisServer.map((d) => d.name);
    const otherRefs = onThisServer.map((d) => deleteRef(d, rows));
    if (opts.resetPassword && othersOnServer(rows).length) return cutOff(othersOnServer(rows), rows);
    // What the box costs, before anything is touched: a billed box is accepted
    // (--yes, or at the prompt), and needs the org's payment method.
    if (opts.coupon !== undefined && !opts.coupon.trim()) {
      return { status: "action_required", provider, name, next: `${opts.agent ? "coupon" : "--coupon"} is empty: pass a promotion code, or leave ${opts.agent ? "coupon" : "--coupon"} out` };
    }
    const prepOpts = { resetPassword: opts.resetPassword, warn, ...(storedPassword ? { storedPassword, others, otherRefs } : {}), ...(monitoringUser ? { monitoringUser } : {}) };
    if (!opts.selfHosted) {
      // A URL that cannot work is told so before any price is asked (nothing is changed here).
      const probe = await deps.prepare(url, provider, { ...prepOpts, check: true }).catch((err) => {
        // A server that cannot be reached: say which ("timeout expired" alone does not).
        if (err instanceof HttpStatusError || /Could not connect to/.test(errorText(err))) throw err;
        throw new Error(`Could not connect to ${cluster}: ${errorText(err)}`);
      });
      if ("next" in probe) {
        const { resettable, ...refusal } = probe;
        if (resettable && !opts.resetPassword && !opts.agent && !opts.yes && await deps.confirm("A monitoring role from an earlier connection exists, and its password isn't stored. Reset it now? Anything else using this role will need the new password. [y/N] ")) {
          return connect(url, { ...opts, resetPassword: true }, deps);
        }
        return { status: "action_required", provider, name, ...refusal };
      }
      // Where the box runs, before the price and before the role is created: a
      // cloud or location the platform refuses changes nothing (platform-all#972).
      const where = { db_host: new URL(url).hostname };
      const placed = await placeBox(deps, opts, where, progress);
      place = placed?.place;
      let quote: Quote;
      try {
        quote = await deps.quote(opts.coupon, clusterOf(url));
      } catch (err) {
        // Not an org admin: the user's to act on, as with pgai dblab deploy (postgresai#412).
        if (err instanceof HttpStatusError && err.status === 403) return { status: "action_required", provider, name, next: notAdminNext };
        throw err;
      }
      const b = await billingFor(quote, name, opts, deps, (price) => progress("billing", estimateLine(price)),
        placed && ((price) => decide(placed, name, price, opts, deps, where, progress)));
      // Where the box would run goes with every result from here on.
      if (b.stop) return { status: "action_required", provider, name, ...place, ...b.stop };
      if (b.placed) place = b.placed.place;
      if (deps.aborted?.()) throw new CancelledError();
      ({ billing, billingPage, billingUrl, accepted } = b);
      // ClickHouse's vCPUs come from ClickHouse Cloud; elsewhere only the user knows them.
      if (vcpus === undefined && provider !== "clickhouse") {
        vcpus = !opts.yes && !opts.agent && deps.askVcpus ? await deps.askVcpus(VCPUS_QUESTION) : undefined;
        if (deps.aborted?.()) throw new CancelledError();
        if (vcpus === undefined) progress("preparing", NO_VCPUS_NOTE);
      }
    }
    if (opts.resetPassword) {
      // Another run at once for a database on this server would reset the
      // password too: the platform lets one run at a time hold the server.
      // Taken after the price is accepted: a stop above leaves the server free,
      // and the 15-minute lease does not run while a person decides.
      let lock: { lock_id?: unknown };
      try {
        lock = await deps.resetLock(cluster);
      } catch (err) {
        if (err instanceof HttpStatusError && err.status === 409) {
          return { status: "action_required", provider, name, ...billing, next: `Another pgai mon deploy --reset-password for ${cluster} is running: wait for it to finish, then re-run (a run that stopped frees the server 15 minutes after it started)` };
        }
        if (err instanceof HttpStatusError && err.status === 404) {
          return { status: "action_required", provider, name, ...billing, next: `This platform cannot lock the server for --reset-password yet: set PGAI_MON_PASSWORD to ${role}'s password instead` };
        }
        throw err;
      }
      // Without a lock id nothing holds the server: no reset.
      if (typeof lock.lock_id !== "string" || !lock.lock_id) throw new Error("cloud_monitoring_reset_lock returned no lock_id: nothing was changed");
      lockId = lock.lock_id;
    }
    let prepared: Prepared;
    // `result`: the allow_ip result when the run stopped at the reserved IP (platform-all#987).
    let created: (Awaited<ReturnType<ConnectDeps["create"]>> & { result?: ConnectResult }) | "payment" | "declined" | "price" | "stored" | "unanswered";
    // No answer to the box request: it may still be created with this URL.
    let unanswered = false;
    try {
      // What another run connected on this server before this one got the lock.
      if (lockId) {
        const now = await deps.list();
        const same = findDatabase(now.filter((d) => d.id !== ownId));
        if (same) return { status: "action_required", provider, name, id: same.id, ...billing, next: `${same.name} is already monitored, and its monitoring uses the current password. Delete it, then run this deploy again with --reset-password: ${disconnectEach([deleteRef(same, now)])}` };
        const others = othersOnServer(now);
        if (others.length) return { ...cutOff(others, now), ...billing };
      }
      if (deps.aborted?.()) throw new CancelledError();
      progress("preparing", `Preparing ${maskConnectionString(url)}`);
      prepared = await deps.prepare(url, provider, prepOpts);
      if (deps.aborted?.()) {
        await undo(prepared);
        throw new CancelledError();
      }
      if ("next" in prepared) {
        // platform-all#987: two runs continued this reservation at once, and the other made the
        // role: this one follows the box that run creates (it may take a moment to ask for it).
        if ((continuing || opts.ipAllowed) && /created by someone else meanwhile/.test(prepared.next)) {
          for (let tries = 0; tries < 10; tries++) {
            const box = findDatabase(await deps.list());
            if (!box) break;
            if (!reservedStatus(box.status)) return follow(box, { provider, name, fresh, billing, billingUrl, undoRole, note, place, url }, opts, deps, progress);
            await deps.sleep(2_000);
          }
        }
        const { resettable, ...refusal } = prepared;
        return { status: "action_required", provider, name, ...billing, ...refusal };
      }
      if (!("monitoringUrl" in prepared)) throw new Error("prepare returned no monitoring URL");
      if (prepared.note) note = `; ${prepared.note}`;
      if (opts.selfHosted) {
        await deps.selfHosted(prepared.monitoringUrl, key && ch
          ? { CLICKHOUSE_ORG_ID: ch.orgId, CLICKHOUSE_KEY_ID: key.keyId, CLICKHOUSE_KEY_SECRET: key.keySecret }
          : {});
        return { status: "ready", provider, name, dashboard_url: "http://localhost:3000", host_metrics: !!ch, next: `pgai mon health${note}` };
      }
      progress("provisioning", `Provisioning monitoring for ${name}`);
      const monitoringUrl = prepared.monitoringUrl;
      const request = (ipAllowed: boolean) => deps.create({
        db_url: monitoringUrl,
        // platform-all#987: a database PostgresAI can't reach the way an IP allowlist refuses gets
        // its box's IP reserved and shown (step allow_ip) instead of a refusal.
        allow_ip_fallback: true,
        ...(opts.ipFirst ? { ip_first: true } : {}),
        ...(ipAllowed ? { ip_allowed: true } : {}),
        ...(provider === "clickhouse" ? { provider } : {}),
        ...(key && ch ? { clickhouse_org_id: ch.orgId, clickhouse_key_id: key.keyId, clickhouse_key_secret: key.keySecret } : {}),
        ...(opts.coupon ? { promo_code: opts.coupon } : {}),
        // The price shown was accepted: without it the platform creates no billed box.
        ...(accepted ? { accept_price: true } : {}),
        // pgai mon deploy --name (postgresai#412); where the box runs, as the platform answered above.
        ...(opts.name ? { project_name: opts.name } : {}),
        ...(place ? { cloud: place.cloud, server_location: place.location }
          : opts.location ? { server_location: opts.location } : {}),
        // The database server's vCPUs: without them AAS is never collected.
        ...(vcpus ? { vcpus } : {}),
      });
      // platform-all#987: SI refused the box of a continued reservation. The platform keeps the
      // reservation (its IP is the one already allowed) and says why: shown with the IP, not followed.
      const refused = async (c: Awaited<ReturnType<typeof request>>) => {
        const row = (await waitForIp(c.id, deps)) ?? ({ id: c.id, name: c.name, status: c.status } as Database);
        const why = [c.error, (c as { billing_warning?: string }).billing_warning].filter(Boolean).join(" ");
        return { ...c, status: "allow_ip", result: { ...allowIpResult(row, provider, name, opts, url), next: `${why} Then: ${continueCommand(opts, url)}` } };
      };
      const continued = async () => {
        const c = await request(true);
        return reservedStatus(c.status) && c.error ? refused(c) : c;
      };
      created = await request(continuing || !!opts.ipAllowed).then(async (c) => {
        // Known now: a Continue of it without an answer is looked for on this row.
        if (reservedStatus(c.status)) known.set(c.id, c.status);
        if (reservedStatus(c.status) && c.error) return refused(c);
        if (!reservedStatus(c.status)) return c;
        // The IP is reserved and the box waits for it to be allowed: shown, then continued at
        // the terminal, or the run stops there (the reservation stays).
        const reservation = await waitForIp(c.id, deps);
        if (!reservation || !reservedStatus(reservation.status)) return { ...c, status: reservation?.status ?? "failed", error: reservation ? undefined : "The IP reservation did not report back in time; see pgai mon instances list." };
        // Continuing (--ip-allowed, or Enter on a reservation still being reserved): once its IP is
        // known the box is asked for, as the user already said the database allows it. Only after
        // the first answer was "still reserving": a reserved answer to a Continue is its refusal.
        if (continuing && c.status === "ip_reserving" && reservation.status === "ip_reserved") return continued();
        const asked = await allowIpStep(reservation, { provider, name, url, opts, deps, progress });
        if (asked !== "continue") return { ...c, status: "allow_ip", result: asked };
        return continued();
      }).catch(async (err) => {
        // platform-all#987: two runs continued the same reservation at once: this one follows the box.
        if ((continuing || opts.ipAllowed) && err instanceof HttpStatusError && err.status === 409 && /already monitored/.test(err.message)) {
          const box = findDatabase(await deps.list());
          if (box && !reservedStatus(box.status)) return { id: box.id, name: box.name, status: box.status ?? "launch_requested" };
        }
        // A 4xx, or a 5xx the function raised (its SQLSTATE: the transaction rolled back), is a
        // refusal; after anything else (no answer) a box may be starting with this URL.
        if (err instanceof HttpStatusError && (err.status < 500 || err.sqlstate)) await undo(prepared);
        else unanswered = true;
        // 402: the payment method went away since the quote; nothing was created.
        if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
        // 412: billed although quoted free (the last free slot went meanwhile); nothing was created.
        if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
        // 409 "none is stored" for a URL without a password: the platform no longer keeps the server's
        // (its last box went meanwhile). Its other 409s (a name taken, say) say what they are.
        if (err instanceof HttpStatusError && err.status === 409 && "storedPassword" in prepared && prepared.storedPassword && /none is stored/.test(err.message)) return "stored" as const;
        // No answer, but the platform may have made the box (postgresai#443): look for it below.
        if (mayHaveRun(err)) return "unanswered" as const;
        throw err;
      });
      if (created === "unanswered") {
        progress("provisioning", STILL_CREATING);
        const made = await lookForBox(findMade, opts, deps, started);
        // Ctrl-C: the person stopped watching, the box may still come (the step view is not failed then).
        if (!made && deps.aborted?.()) return { status: "action_required", provider, name, ...billing, next: STOPPED_LOOKING };
        if (!made) return { status: "action_required", provider, name, ...billing, next: opts.agent ? NOT_CONFIRMED_AGENT : NOT_CONFIRMED };
        // Refused and undone by the platform: as a "failed" reply.
        if (stateOf(made.status) === "failed" || disconnecting(made.status)) {
          await undo(prepared);
          // Nothing starts with this URL: the server lock can go (finally).
          unanswered = false;
          return { status: "failed", provider, name, id: made.id, next: disconnecting(made.status)
            ? "PostgresAI could not launch the monitoring box. Re-run pgai mon deploy later."
            // A failed row stays in the list: the next deploy would follow it.
            : `PostgresAI could not launch the monitoring box. Run pgai mon instances delete ${made.id} --yes, then pgai mon deploy again.` };
        }
        // The box has its URL: the server lock can go (finally).
        unanswered = false;
        created = { id: made.id, name: made.name ?? name, status: made.status ?? "launch_requested" };
      }
      if (created === "payment") return { status: "action_required", provider, name, ...billingPage! };
      // A card on file that Stripe declined is not a missing one.
      if (created === "declined") {
        return { status: "action_required", provider, name, ...billingPage!, next: billingPage!.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
      }
      if (created === "price") return { status: "action_required", provider, name, ...billing, next: "The price changed since it was shown: re-run pgai mon deploy to see it" };
      if (created === "stored") {
        // No box of this org on the server keeps it now; what else logs in as the role is not known here.
        const gone = `PostgresAI no longer keeps the password of ${DEFAULT_MONITORING_USER} for this server`;
        return { status: "action_required", provider, name, ...billing, next: opts.agent
          ? `${gone}: pass its URL as database_url, or run pgai mon deploy <admin-url> --reset-password in a terminal`
          : `${gone}: re-run with --reset-password (anything else that logs in as ${DEFAULT_MONITORING_USER} then needs the new one), or set PGAI_MON_PASSWORD to its password` };
      }
      // platform-all#987: stopped at the reserved IP (not continued, or nobody to ask). No box got
      // the URL, so a role with a generated password is dropped: the re-run makes it again.
      if (created.status === "allow_ip" && created.result) {
        // Only while it still waits: another run may have continued it into a box with this role.
        const now = findDatabase(await deps.list().catch(() => [] as Database[]));
        if (!now || reservedStatus(now.status)) await undo(prepared);
        return { ...created.result, ...billing };
      }
      // Before the lock goes: the next run may be creating the role this drops.
      if (created.status === "failed") {
        await undo(prepared);
        return { status: "failed", provider, name, id: created.id, next: `${created.error} Re-run pgai mon deploy later.` };
      }
    } finally {
      // The box has its URL (or none was requested): the next run may go. A box
      // that may still be starting keeps the server until the lease ends: a run
      // that did not see it would reset the password its URL carries.
      if (lockId && !unanswered) await deps.resetUnlock(lockId).catch(() => {});
    }
    row = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: !!ch };
    // Ctrl-C while the box was requested: it exists now, so the run detaches (it keeps going on PostgresAI).
    if (deps.aborted?.()) return { ...connectStatus(row, provider, true), ...billing };
    const made = prepared;
    undoRole = () => undo(made);
  }

  return follow(row, { provider, name, fresh, billing, billingUrl, undoRole, note, place, url }, opts, deps, progress);
}

/** A row of pgai mon instances list: its status in deploy's words, and a reservation's step (platform-all#987). */
export function listedDatabase(d: Database): Database & { status: Status; step?: "allow_ip" } {
  return { ...d, status: stateOf(d.status), ...(d.status === "ip_reserved" ? { step: "allow_ip" as const } : {}) };
}

/** platform-all#987: an IP reserved before its box, waiting to be allowed (or being reserved). */
export const reservedStatus = (status: string | null | undefined) => status === "ip_reserving" || status === "ip_reserved";

/** How often, and how long, deploy waits for a reserved IP to be reported (20-60 s is usual). */
const IP_POLL_MS = 3_000;
const IP_WAIT_MS = 120_000;

/** The reservation once its IP is known (or it failed); undefined when it did not report in time. */
async function waitForIp(id: string, deps: ConnectDeps): Promise<Database | undefined> {
  for (const deadline = deps.now() + IP_WAIT_MS; ;) {
    const row = (await deps.list().catch(() => [] as Database[])).find((d) => d.id === id);
    if (row && row.status !== "ip_reserving") return row;
    if (deps.now() >= deadline) return undefined;
    await deps.sleep(IP_POLL_MS);
  }
}

/** The command that continues a reservation, with the URL's password masked. */
const continueCommand = (opts: ConnectOptions, url: string) =>
  opts.agent ? "call deploy_monitoring again with ip_allowed: true"
  : `pgai mon deploy ${url ? quoteArg(maskConnectionString(url)) : "<database URL>"} --yes --ip-allowed`;

const quoteArg = (s: string) => (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);

const heldUntil = (iso?: string | null) => (iso ? iso.replace(/T(\d\d:\d\d).*$/, " $1 UTC") : "");

/** The allow_ip result: what to allow, where, and how to go on (exit 3). */
export function allowIpResult(row: Database, provider: Provider, name: string, opts: ConnectOptions, url: string, stopped = false): ConnectResult {
  const [host, port] = row.cluster ? [row.cluster.replace(/:\d+$/, ""), Number(row.cluster.match(/:(\d+)$/)?.[1] ?? 5432)] : [undefined, undefined];
  const ip = row.egress_ip ?? undefined;
  const cidr = ip ? `${ip}/32` : undefined;
  const allow = `Allow inbound TCP ${port ?? 5432} from ${cidr} on ${host ?? "the database"}`;
  return {
    status: "action_required",
    step: "allow_ip",
    provider, name, id: row.id,
    ...(row.cloud ? { cloud: row.cloud, location: row.location ?? undefined } : {}),
    ip, cidr, host, port,
    ...(row.reserved_until ? { reserved_until: row.reserved_until } : {}),
    ...(row.last_error ? { last_error: row.last_error } : {}),
    ...(row.precheck_error ? { precheck_error: row.precheck_error } : {}),
    egress_ip: ip,
    next: stopped
      ? `Stopped. ${ip} stays reserved until ${heldUntil(row.reserved_until)}. ${allow}, then: ${continueCommand(opts, url)}`
      : `${allow}, then ${opts.agent ? continueCommand(opts, url) : `re-run with --yes --ip-allowed: ${continueCommand(opts, url)}`}`,
  };
}

/**
 * platform-all#987: show the reserved IP. A person at a terminal continues (once the database
 * allows it) or stops; anyone else gets the allow_ip result. "continue" means: create the box.
 */
async function allowIpStep(row: Database, c: { provider: Provider; name: string; url: string; opts: ConnectOptions; deps: ConnectDeps; progress: Progress }): Promise<ConnectResult | "continue"> {
  if (row.status === "ip_reserving") row = (await waitForIp(row.id, c.deps)) ?? row;
  if (!reservedStatus(row.status) || !row.egress_ip) {
    return { status: "action_required", provider: c.provider, name: c.name, id: row.id,
      next: row.status === "ip_reserving" ? `The IP is still being reserved: re-run in a minute, or see pgai mon instances list` : `See pgai mon instances list (${row.status ?? "unknown"})` };
  }
  const r = allowIpResult(row, c.provider, c.name, c.opts, c.url);
  c.progress("allow_ip", `Reserved IP for the monitoring box: ${r.ip}`, { ip: r.ip, cidr: r.cidr, host: r.host, port: r.port, reserved_until: r.reserved_until, id: row.id });
  if (!c.deps.allowIp) return r;
  const why = row.last_error
    ? `The box could not connect from ${r.ip}: ${row.last_error}`
    : c.opts.ipFirst ? "You asked for the IP first (--ip-first)."
    // No pre-check error: PostgresAI reached the database; the IP was asked for first.
    : !row.precheck_error ? "The IP was reserved first, as asked."
    : `PostgresAI could not connect to ${r.host}:${r.port} (${row.precheck_error}). The database likely allows only listed IPs.`;
  const answer = await c.deps.allowIp([
    `Monitoring box IP: ${r.ip}${row.cloud ? ` (${cloudWord(row.cloud)}, ${row.location})` : ""}`,
    why,
    `Allow inbound TCP ${r.port} from ${r.cidr} on ${r.host}, then continue. The IP is held until ${heldUntil(row.reserved_until)} and stays this box's IP.`,
  ]);
  return answer === "continue" ? "continue" : allowIpResult(row, c.provider, c.name, c.opts, c.url, true);
}

/**
 * Where a new box runs: the platform's answer for this database and --cloud/--location
 * (v1.monitoring_deploy_options, platform-all#972), shown as "Monitoring box: AWS, us-east-2
 * (same region as your database)". A refusal is thrown (exit 1) before anything is created.
 * A platform from before --cloud has no answer: the box goes where it puts it.
 */
async function placeBox(deps: ConnectDeps, opts: ConnectOptions, where: { db_host?: string; supabase_project_ids?: string[] },
  progress: Progress): Promise<Placed | undefined> {
  if (!deps.deployOptions) return undefined;
  const cloud = opts.cloud?.trim().toLowerCase() || undefined;
  const location = opts.location?.trim() || undefined;
  const flag = (f: string) => (opts.agent ? f : `--${f}`);
  let options: DeployOptions;
  try {
    options = await deps.deployOptions({ ...where, ...(cloud ? { cloud } : {}), ...(location ? { location } : {}), check_stock: true });
  } catch (err) {
    if (httpStatus(err) !== 404) throw err;
    if (cloud) throw new Error(`This PostgresAI platform does not take ${flag("cloud")} yet: re-run without it. Nothing was created.`);
    // A database URL still sends --location as server_location; a Supabase box would drop it.
    if (location && where.supabase_project_ids) {
      throw new Error(`This PostgresAI platform does not take ${flag("location")} for Supabase yet: re-run without it. Nothing was created.`);
    }
    return undefined;
  }
  if (!options.choice) {
    const said = options.error?.detail ?? options.error?.message ?? "The platform did not say where the monitoring box can run.";
    // A location of another cloud: name the flag that picks that cloud.
    const elsewhere = location
      ? options.clouds.find((c) => c.available && c.code !== cloud && options.locations[c.code]?.some((l) => l.code === location))
      : undefined;
    const available = options.clouds.filter((c) => c.available).map((c) => c.code);
    const how = elsewhere ? ` ${cloud ? "Use" : "Add"} ${flag("cloud")} ${elsewhere.code}.`
      : /^(Monitoring on .* is not (available here|offered yet)|Unknown cloud)/.test(said) && available.length
        ? ` Re-run with ${flag("cloud")} ${available.length === 1 ? available[0] : `set to one of: ${available.join(", ")}`}.` : "";
    throw new Error(`${said}${how} Nothing was created.`);
  }
  const c = options.choice;
  progress("preparing", `Monitoring box: ${c.cloud_label}, ${c.location} (${c.reason_text})`, { cloud: c.cloud, location: c.location, location_reason: c.reason });
  return { place: { cloud: c.cloud, location: c.location, location_reason: c.reason, location_reason_text: c.reason_text }, options };
}

/**
 * The terminal's question once the place and price are shown (A17): deploy here (y), not at all
 * (n, Enter), or choose another cloud and location from the platform's options (c). `ask` reads an
 * answer, `say` prints a line; three wrong answers are a no.
 */
export async function pickPlace(options: DeployOptions, name: string, ask: (question: string) => Promise<string>,
  say: (line: string) => void): Promise<boolean | { cloud: string; location: string }> {
  const number = async (what: string, n: number): Promise<number | undefined> => {
    for (let tries = 0; tries < 3; tries++) {
      const answer = (await ask(`${what} (1-${n}, Enter to cancel): `)).trim();
      if (!answer) return undefined;
      const i = Number(answer);
      if (Number.isInteger(i) && i >= 1 && i <= n) return i - 1;
      say(`Type a number from 1 to ${n}.`);
    }
    return undefined;
  };
  for (let tries = 0; tries < 3; tries++) {
    const answer = (await ask(`Deploy ${terminalText(name)} here? [y]es / [c]hoose another location / [N]o: `)).trim().toLowerCase();
    if (answer === "y" || answer === "yes") return true;
    if (!answer || answer === "n" || answer === "no") return false;
    if (answer !== "c" && answer !== "choose") { say("Type y, c or n."); continue; }
    const clouds = options.clouds.filter((c) => c.available && options.locations[c.code]?.length);
    say("Clouds:");
    clouds.forEach((c, i) => say(`  ${i + 1}. ${c.label}`));
    const ci = await number("Cloud", clouds.length);
    if (ci === undefined) return false;
    const cloud = clouds[ci]!;
    const locations = options.locations[cloud.code]!;
    say(`Locations on ${cloud.label}:`);
    let part = "";
    locations.forEach((l, i) => {
      if (l.world_part !== part) say(`  ${(part = l.world_part)}`);
      say(`    ${i + 1}. ${l.code}${l.label ? `  ${terminalText(l.label)}` : ""}`);
    });
    const li = await number("Location", locations.length);
    if (li === undefined) return false;
    return { cloud: cloud.code, location: locations[li]!.code };
  }
  return false;
}

/** What deploy does once the box's place and price are shown: go (maybe elsewhere), or stop with what to do. */
type Decision = { go: true; place?: Placed } | { go: false; stop: Partial<ConnectResult> & { next: string } };

/**
 * Whether the box is made, and where (A17, platform-all#972), once its place and price are shown:
 * --yes deploys there; a person at a terminal is asked, and without --cloud/--location may pick
 * another place, which the platform places again; anyone else (piped, --json, an agent) gets the
 * place, the price and the choices back, and nothing is created.
 */
async function decide(placed: Placed, name: string, price: string, opts: ConnectOptions, deps: ConnectDeps,
  where: { db_host?: string; supabase_project_ids?: string[] }, progress: Progress): Promise<Decision> {
  if (opts.yes) return { go: true };
  const { place, options } = placed;
  const label = options.choice?.cloud_label ?? cloudWord(place.cloud);
  const explicit = !!(opts.cloud?.trim() || opts.location?.trim());
  const nothing = "Nothing was created";
  if (deps.choosePlace) {
    if (explicit) {
      return await deps.confirm(`Deploy ${terminalText(name)} on ${label}, ${place.location}? [y/N] `)
        ? { go: true } : { go: false, stop: { next: `${nothing}: re-run with --yes to deploy on ${label}, ${place.location}` } };
    }
    const answer = await deps.choosePlace(options, name);
    if (answer === true) return { go: true };
    if (answer === false) return { go: false, stop: { next: `${nothing}: re-run to deploy, with --cloud and --location to choose where` } };
    // Still the summary before the deploy: plain lines, so no step view starts before a later question (vCPUs).
    const summary: Progress = (_event, message, more) => progress("billing", message, more);
    const again = await placeBox(deps, { ...opts, cloud: answer.cloud, location: answer.location }, where, summary);
    if (!again) throw new Error(`The platform did not place the box on ${answer.cloud}, ${answer.location}. ${nothing}.`);
    summary("billing", `Next time, to skip this question: --cloud ${again.place.cloud} --location ${again.place.location} --yes`);
    return { go: true, place: again };
  }
  // The clouds the picker offers: available, with a location.
  const choices = Object.fromEntries(options.clouds.filter((c) => c.available && options.locations[c.code]?.length)
    .map((c) => [c.code, options.locations[c.code]!.map((l) => l.code)]));
  const here = `${label}, ${place.location} (${place.location_reason_text})`;
  const next = opts.agent
    ? `Show the user where the monitoring box would run (${here}) and the price (${price}), then call deploy_monitoring again with yes: true, or with cloud and location from choices and yes: true. ${nothing}.`
    : `The monitoring box would run on ${here} for ${price}: re-run with --yes to deploy there, or with --cloud and --location (choices below) and --yes. ${nothing}.`;
  return { go: false, stop: { location_reason_text: place.location_reason_text, choices, next } };
}

const isSupabaseUrl = (link: string): boolean => {
  try {
    const u = new URL(link);
    return u.origin === SUPABASE_API_BASE && !u.username && !u.password;
  } catch {
    return false;
  }
};

// How long deploy waits for the consent in the browser, and how often it asks.
const OAUTH_WAIT_MS = 10 * 60_000;
const OAUTH_POLL_MS = 2_000;

/**
 * A Supabase project: the platform creates postgres_ai_mon through the Supabase Management API,
 * so the database is never logged in to from here. The org connects Supabase once, in the browser.
 */
async function deploySupabase(url: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const provider = "supabase";
  const ref = extractProjectRefFromUrl(url);
  if (!ref) return { status: "action_required", provider, name: databaseName(url), next: "Not a Supabase project URL: use the project's connection string (db.<project-ref>.supabase.co, or the pooler with user postgres.<project-ref>), or pass --provider supabase without a URL to pick the project" };
  // One name for the direct and the pooler URL of a project.
  const db = decodeURIComponent(new URL(url).pathname.replace(/^\//, "")) || "postgres";
  const name = databaseName(supabaseProjectUrl(ref, db));
  const stop = (next: string, more: Partial<ConnectResult> = {}): ConnectResult => ({ status: "action_required", provider, name, ...more, next });
  // The box prepares and monitors the project's postgres database (prepare-db --supabase).
  if (db !== "postgres") return stop("PostgresAI monitors a Supabase project through its postgres database: use the URL of the postgres database");
  const refused = supabaseOptionsRefused(opts);
  if (refused) return stop(refused);
  // Both the URL and --supabase-project: they must name the same project, which the list decides.
  if (opts.supabaseProject?.trim()) return deploySupabaseProject(ref, opts, deps, progress);
  const row = await boxOnProject(ref, deps);
  if (row) {
    const note = opts.name && row.name !== opts.name ? `; --name ignored: this database is already monitored as ${row.name}` : "";
    return follow(row, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
  }
  let started: Awaited<ReturnType<ConnectDeps["supabaseStart"]>>;
  try {
    started = await deps.supabaseStart(supabaseProjectUrl(ref));
  } catch (err) {
    if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
    // Another Supabase organization is connected to this org: the platform says which.
    if (err instanceof HttpStatusError && err.status === 409) return stop(errorText(err));
    throw err;
  }
  if ("auth_url" in started) {
    const waited = await supabaseConsent(started, opts, deps, progress, stop);
    if (waited) return waited;
  }
  return supabaseBox(ref, name, opts, deps, progress, stop);
}

/**
 * The box that already monitors a Supabase project's postgres database, if any: also one the
 * platform does not link to the project (made through the generic path). A row of another
 * database there (deployed as self-managed) is not this box.
 */
async function boxOnProject(ref: string, deps: ConnectDeps): Promise<Database | undefined> {
  const cluster = clusterOf(supabaseProjectUrl(ref))!;
  return (await deps.list()).find((d) => !disconnecting(d.status) && d.cluster && serverOf(d.cluster) === cluster && (!d.database || d.database === "postgres"));
}

/** The URL that names a Supabase project's database: what the platform finds it by, and the instance name's source. */
function supabaseProjectUrl(ref: string, db = "postgres"): string {
  return `postgresql://db.${ref}.supabase.co:5432/${encodeURIComponent(db)}`;
}

/** Options that do not apply to a Supabase project, or undefined. */
function supabaseOptionsRefused(opts: ConnectOptions): string | undefined {
  if (opts.resetPassword) return "--reset-password does not apply to Supabase: PostgresAI creates postgres_ai_mon through the Supabase Management API";
  // The platform creates postgres_ai_mon there, and keeps its password: a custom role would not be used.
  if (opts.monitoringUser !== undefined && opts.monitoringUser.trim() !== DEFAULT_MONITORING_USER) {
    return `${opts.agent ? "monitoring_user" : "--monitoring-user"} does not apply to Supabase: PostgresAI creates ${DEFAULT_MONITORING_USER} through the Supabase Management API`;
  }
  return undefined;
}

/**
 * The org approves PostgresAI's access to Supabase in the browser: a person at a terminal waits for it;
 * an agent, or --wait 0, stops with the link. undefined once connected, else the result to return.
 */
async function supabaseConsent(started: { auth_url: string; session_id: string; expires_at?: string }, opts: ConnectOptions, deps: ConnectDeps,
  progress: Progress, stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult): Promise<ConnectResult | undefined> {
  // The link goes to the browser opener and the terminal: Supabase's own https page only.
  if (!isSupabaseUrl(started.auth_url)) return { ...stop(""), status: "failed", next: `The platform sent a Supabase consent link that is not ${SUPABASE_API_BASE}: not opened. Re-run pgai mon deploy later.` };
  const approve = `Open ${started.auth_url} in a browser and approve PostgresAI's access to Supabase`;
  if (opts.agent || opts.waitMs <= 0) return stop(`${approve}, then ${opts.agent ? "call deploy_monitoring again" : "re-run"}`);
  progress("preparing", `Connect Supabase to PostgresAI: approve access in the browser (open ${started.auth_url} if it did not open). Waiting...`, { auth_url: started.auth_url });
  deps.openUrl?.(started.auth_url);
  // A declined consent is never reported: the session stays pending until it expires.
  const expires = started.expires_at ? Date.parse(started.expires_at) : NaN;
  for (const deadline = deps.now() + Math.min(opts.waitMs, OAUTH_WAIT_MS); ;) {
    if (deps.aborted?.()) throw new CancelledError();
    const s = await deps.supabaseStatus(started.session_id);
    if (s.status === "connected") return undefined;
    if (s.status === "error") return stop(`Supabase did not connect: ${(s.error ?? "no reason given").replace(/\.$/, "")}. Re-run to try again`);
    const now = deps.now();
    if (now >= expires) return stop("The Supabase consent link expired before it was approved: re-run pgai mon deploy for a new one");
    if (now >= deadline) return stop(`${approve}, then re-run`);
    await deps.sleep(OAUTH_POLL_MS);
  }
}

/** A Supabase project status in words. */
export function supabaseStatusWords(status: string): string {
  if (status === "inactive" || status === "paused") return "paused";
  if (status === "coming_up") return "starting";
  if (status === "unknown") return "in an unknown state";
  return status.replace(/_/g, " ");
}

/** Why a project can't be picked, in words ("" when it can). */
function supabaseWhy(p: SupabaseProject): string {
  if (p.reason === "monitored") return `already monitored${p.monitored?.name ? ` as ${p.monitored.name}` : ""}`;
  if (p.reason === "paused") return "paused";
  if (p.reason?.startsWith("not_ready:")) return `not ready (${supabaseStatusWords(p.reason.slice("not_ready:".length))})`;
  return "";
}

/**
 * The projects as aligned rows for a person: name, region, project ID, and why one can't be
 * picked. `numbered`: the pickable ones get 1, 2, ... (the picker); the others "-".
 */
export function supabaseProjectRows(projects: SupabaseProject[], numbered = false): string[] {
  const nameW = Math.max(...projects.map((p) => p.name.length));
  const regionW = Math.max(...projects.map((p) => (p.region ?? "").length));
  const numW = String(projects.filter((p) => p.selectable).length).length;
  let n = 0;
  return projects.map((p) => {
    const why = supabaseWhy(p);
    const mark = numbered ? `${(p.selectable ? `${++n}.` : "-").padStart(numW + 1)} ` : "";
    return `  ${mark}${p.name.padEnd(nameW)}  ${(p.region ?? "").padEnd(regionW)}  ${p.ref}${why ? `  ${why}` : ""}`.trimEnd();
  });
}

/** The project --supabase-project names: by ref, else by its exact name (any case); a name several projects share is refused. */
export function matchSupabaseProject(projects: SupabaseProject[], wanted: string): { project: SupabaseProject } | { none: true } | { ambiguous: SupabaseProject[] } {
  const value = wanted.trim();
  const byRef = projects.find((p) => p.ref === value);
  if (byRef) return { project: byRef };
  const byName = projects.filter((p) => p.name.toLowerCase() === value.toLowerCase());
  if (byName.length === 1) return { project: byName[0]! };
  if (byName.length > 1) return { ambiguous: byName };
  return { none: true };
}

/**
 * --provider supabase without a URL (postgresai#438): connect Supabase if needed, list the org's
 * projects, pick one (--supabase-project, or a person at a terminal), then the box, as for a URL.
 * `urlRef`: a URL was given too, and the project picked must be that one.
 */
async function deploySupabaseProject(urlRef: string | undefined, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const provider = "supabase";
  let name = urlRef ? databaseName(supabaseProjectUrl(urlRef)) : "";
  const stop = (next: string, more: Partial<ConnectResult> = {}): ConnectResult => ({ status: "action_required", provider, name, ...more, next });
  const refused = supabaseOptionsRefused(opts);
  if (refused) return stop(refused);
  const urlHint = "pass the project's URL instead (pgai mon deploy 'postgresql://postgres@db.<project-ref>.supabase.co:5432/postgres')";
  if (!deps.supabaseProjects) return stop(`Picking a Supabase project needs a newer pgai: ${urlHint}`);
  const listed = async (): Promise<SupabaseProjects | ConnectResult> => {
    try {
      return await deps.supabaseProjects!();
    } catch (err) {
      if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
      // An API from before postgresai#438 has no such rpc.
      if (err instanceof HttpStatusError && err.status === 404) return stop(`This PostgresAI API cannot list Supabase projects yet: ${urlHint}`);
      throw err;
    }
  };
  let list = await listed();
  if ("status" in list) return list;
  if (!list.connected) {
    const waited = await supabaseConsent(list, opts, deps, progress, stop);
    if (waited) return waited;
    list = await listed();
    if ("status" in list) return list;
    // Approved, yet not connected: the grant went missing between the two calls.
    if (!list.connected) return stop("Supabase did not stay connected after the approval: re-run pgai mon deploy --provider supabase");
  }
  const org = list.supabase_org_name;
  const projects = list.projects;
  const shown = { supabase_org: org, projects };
  // Supabase did not answer the sync: the stored list, which may be out of date.
  if (!list.synced && !projects.length) return stop("Supabase did not answer the project list: re-run in a minute");
  const stale = list.synced ? "" : " (Supabase did not answer just now: the list may be out of date)";
  const inOrg = `organization "${org}"`;
  const useFlag = "--supabase-project <name or project ID>";
  if (!projects.length) return stop(`There are no projects in the Supabase ${inOrg}: create one there, then re-run`, shown);
  const wanted = opts.supabaseProject?.trim();
  let picked: SupabaseProject | undefined;
  if (wanted) {
    const m = matchSupabaseProject(projects, wanted);
    if ("none" in m) return stop(`No Supabase project "${wanted}" in ${inOrg}: re-run with ${useFlag} of one of its projects`, shown);
    if ("ambiguous" in m) return stop(`Several projects in ${inOrg} are called "${wanted}": re-run with the project ID instead`, { supabase_org: org, projects: m.ambiguous });
    picked = m.project;
  } else {
    // Nothing to pick: say why before any prompt.
    if (!projects.some((p) => p.selectable)) {
      return stop(projects.some((p) => p.monitored)
        ? `Every active project in ${inOrg} is already monitored: see pgai mon instances list (a paused project can be monitored once it is resumed in Supabase)`
        : `No active project in ${inOrg}: resume one in Supabase, then re-run`, shown);
    }
    const ask = !opts.agent && !opts.yes ? deps.chooseSupabaseProject : undefined;
    if (!ask) return stop(`Choose the Supabase project to monitor, then re-run with ${useFlag}${stale}`, shown);
    const ref = await ask(projects, org);
    if (deps.aborted?.()) throw new CancelledError();
    if (!ref) return stop(`No project chosen: re-run and pick one, or pass ${useFlag}`, shown);
    picked = projects.find((p) => p.ref === ref);
  }
  if (!picked) return stop(`No project chosen: re-run and pick one, or pass ${useFlag}`, shown);
  if (urlRef && picked.ref !== urlRef) {
    const urlProject = projects.find((p) => p.ref === urlRef);
    return stop(`--supabase-project is ${picked.name}, but the database URL is for ${urlProject ? urlProject.name : `project ${urlRef}`}: pass only one of them`);
  }
  name = databaseName(supabaseProjectUrl(picked.ref));
  // Already monitored: that box, as on a re-run with the URL; no second box.
  if (picked.monitored) {
    const box = (await deps.list()).find((d) => d.id === picked!.monitored!.id);
    if (box && !disconnecting(box.status)) {
      const note = opts.name && box.name !== opts.name ? `; --name ignored: this database is already monitored as ${box.name}` : "";
      return follow(box, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
    }
    return stop(`${picked.name} is already monitored${picked.monitored.name ? ` as ${picked.monitored.name}` : ""}: see pgai mon instances list`);
  }
  if (picked.reason === "paused") return stop(`${picked.name} is paused: resume it in Supabase, then re-run`);
  if (!picked.selectable) return stop(`${picked.name} is not ready in Supabase (${supabaseStatusWords(picked.status)}): re-run when it is active`);
  // A box on the project's database that the platform does not link to the project: that one, not a second.
  const existing = await boxOnProject(picked.ref, deps);
  if (existing) {
    const note = opts.name && existing.name !== opts.name ? `; --name ignored: this database is already monitored as ${existing.name}` : "";
    return follow(existing, { provider, name, fresh: false, billing: {}, note }, opts, deps, progress);
  }
  return supabaseBox(picked.ref, name, opts, deps, progress, stop, `${picked.name} (Supabase)`);
}

/** The price, then the box for a project of the connected Supabase organization, then following it. */
async function supabaseBox(ref: string, name: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress,
  stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult, label?: string): Promise<ConnectResult> {
  const provider = "supabase";
  // Where the box runs: by default the project's own region on AWS (platform-all#972).
  const where = { supabase_project_ids: [ref] };
  const placed = await placeBox(deps, opts, where, progress);
  let place = placed?.place;
  let quote: Quote;
  try {
    // The org's price, not a cluster's: the Supabase provision bills every box.
    quote = await deps.quote(opts.coupon);
  } catch (err) {
    if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
    throw err;
  }
  const b = await billingFor(quote, label ?? name, opts, deps, (price) => progress("billing", estimateLine(price), label ? { project: label } : {}),
    placed && ((price) => decide(placed, label ?? name, price, opts, deps, where, progress)));
  if (b.stop) return { status: "action_required", provider, name, ...place, ...b.stop };
  if (b.placed) place = b.placed.place;
  if (deps.aborted?.()) throw new CancelledError();
  progress("provisioning", `Provisioning monitoring for ${label ?? name}`, label ? { project: label } : {});
  // vCPUs: the platform reads the project's compute size; --vcpus overrides it.
  const created = await deps.supabaseConnect({
    project_ref: ref,
    ...(opts.name ? { name: opts.name } : {}),
    ...(b.accepted ? { accept_price: true } : {}),
    ...(opts.coupon ? { promo_code: opts.coupon } : {}),
    ...(opts.vcpus ? { vcpus: opts.vcpus } : {}),
    ...(place ? { cloud: place.cloud, location: place.location } : {}),
  }).catch((err) => {
    if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
    if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
    // An org billed by a partner (or a box that exists already): the platform says what to do.
    if (err instanceof HttpStatusError && err.status === 409) return { refused: errorText(err) };
    throw err;
  });
  if (typeof created === "object" && "refused" in created) return stop(created.refused);
  if (created === "payment") return { status: "action_required", provider, name, ...b.billingPage };
  if (created === "declined") return { status: "action_required", provider, name, ...b.billingPage, next: b.billingPage.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
  if (created === "price") return stop("The price changed since it was shown: re-run pgai mon deploy to see it", b.billing);
  if (created.status === "failed") return { status: "failed", provider, name, id: created.id, ...b.billing, next: `${created.error} Re-run pgai mon deploy later.` };
  const box: Database = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: false };
  // Ctrl-C while the box was requested: it exists now, so the run detaches (it keeps going on PostgresAI).
  if (deps.aborted?.()) return { ...connectStatus(box, provider, true), ...b.billing };
  return follow(box, { provider, name, fresh: true, billing: b.billing, billingUrl: b.billingUrl, note: "", place }, opts, deps, progress);
}

const PL = "AWS PrivateLink";
/** How long a terminal deploy waits for the customer's stack, and how often it reads it. */
export const STACK_WAIT_MS = 30 * 60_000;
const STACK_POLL_MS = 15_000;

/** The first option given that applies to AWS PrivateLink only, as given (an agent: its argument). */
function privateLinkFlag(opts: ConnectOptions): string | undefined {
  const named: [unknown, string, string][] = [
    [opts.privatelink === true ? true : undefined, "--privatelink", "privatelink"],
    [opts.vpceService, "--vpce-service", "vpce_service"],
    [opts.hostStats ? true : undefined, "--host-stats", "host_stats"],
    [opts.hostStatsRole, "--host-stats-role", "host_stats_role"],
    [opts.terraform ? true : undefined, "--terraform", "terraform"],
    // An agent's argument; the CLI's environment is read for an RDS only (a value left from another deploy).
    [opts.agent ? opts.nonce : undefined, "PGAI_MON_NONCE", "nonce"],
  ];
  const hit = named.find(([v]) => v !== undefined);
  return hit && (opts.agent ? hit[2] : hit[1]);
}

/** How mon deploy reaches an RDS: directly, or over PrivateLink with what AWS said of it here. */
type RdsRoute = { private: false } | { private: true; ep: RdsEndpoint; aws?: AwsRun; inst?: DbInstance };

/**
 * Whether an RDS is monitored over PrivateLink (postgresai#454): --privatelink / --no-privatelink,
 * else whether it is publicly accessible, from AWS when the aws CLI reads it here, else from
 * public DNS (inside the VPC, AWS resolves even a public RDS to a private address).
 */
async function rdsRoute(url: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<RdsRoute | { status: "action_required"; next: string }> {
  const host = new URL(url).hostname.toLowerCase().replace(/\.$/, "");
  const ep = rdsEndpoint(host);
  const flag = (cli: string, agent: string) => (opts.agent ? agent : cli);
  const stop = (next: string) => ({ status: "action_required" as const, next });
  if (opts.privatelink === false) return { private: false };
  const asked = opts.privatelink === true;
  let read: { aws?: AwsRun; inst?: DbInstance } = {};
  if (ep?.kind === "instance" && ep.partition === "aws" && opts.aws !== false && deps.aws) read = await readAws(ep, opts, deps, progress);
  let isPublic = read.inst?.publiclyAccessible;
  if (isPublic === undefined && ep && deps.publicAddresses) {
    progress("preparing", `Looking up ${host} in public DNS (cloudflare-dns.com, dns.google): is it publicly accessible?`);
    const addresses = await deps.publicAddresses(host);
    if (addresses?.length) {
      isPublic = !addresses.every(isPrivateAddress);
      progress("preparing", `${host} resolves to a ${isPublic ? "public address in public DNS, so it is monitored directly" : `private address in public DNS, so it is monitored over ${PL}`}`);
    }
  }
  if (isPublic === true) {
    if (asked) return stop(`${host} is publicly accessible, so it is monitored directly: deploy without ${flag("--privatelink", "privatelink")}`);
    return { private: false };
  }
  if (isPublic === undefined && !asked) {
    if (!deps.ask || opts.yes) return stop(`Could not tell whether ${host} is publicly accessible (AWS ${opts.aws === false ? "not read" : "not readable here"}, and public DNS did not answer): re-run with ${flag("--privatelink", "privatelink: true")} if it is not, or ${flag("--no-privatelink", "privatelink: false")} if it is`);
    const a = (await deps.ask(`Is ${host} publicly accessible? [y]es, monitor it directly / [n]o, over ${PL} / Enter to stop: `)).trim().toLowerCase();
    if (a === "y" || a === "yes") return { private: false };
    if (a !== "n" && a !== "no") return stop("Stopped: nothing was created");
  }
  if (!ep) return stop(`${PL} needs the RDS instance endpoint (<instance>.<id>.<region>.rds.amazonaws.com) in the URL, not ${host}; or monitor from this network with --self-hosted`);
  if (ep.partition !== "aws") return stop(`${ep.partition === "aws-cn" ? "AWS China" : "AWS GovCloud"} can't reach PostgresAI over ${PL} (it does not cross AWS partitions): monitor from this network with --self-hosted`);
  if (ep.kind !== "instance" || read.inst?.engine?.startsWith("aurora")) return stop(`${ep.kind === "proxy" ? "An RDS Proxy endpoint" : "Aurora"} is not supported over ${PL} yet: monitor from this network with --self-hosted, or ask support@postgres.ai`);
  return { private: true, ep, ...read };
}

/**
 * The RDS as the aws CLI reads it here (the standard profile chain, or --aws-profile), said with
 * the account it signs in as. When it can't (no aws CLI, an expired session, the RDS in another
 * account), a person at a terminal may pick another profile; otherwise deploy goes on without AWS.
 */
async function readAws(ep: RdsEndpoint, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<{ aws?: AwsRun; inst?: DbInstance }> {
  let profile = opts.awsProfile?.trim() || undefined;
  for (let tries = 0; tries < 3; tries++) {
    const run = deps.aws!(profile);
    const of = profile ? ` (profile ${profile})` : "";
    let why: string;
    let missing = false;
    try {
      const me = await callerIdentity(run);
      const inst = await describeDbInstance(run, ep.id, ep.region);
      progress("preparing", `AWS account ${me.account} (${me.arn.split(":").pop()}), read-only: ${ep.id} is ${inst.publiclyAccessible ? "publicly accessible, so it is monitored directly" : `not publicly accessible, so it is monitored over ${PL}`}`);
      return { aws: run, inst };
    } catch (err) {
      if (!(err instanceof AwsError)) throw err;
      missing = err.kind === "missing";
      why = missing ? "the aws CLI is not installed here"
        : err.kind === "not_found" ? `${ep.id} is not in ${ep.region} of this AWS account${of}`
        : err.kind === "credentials" ? `the AWS credentials${of} do not work: ${err.message}`
        : `AWS refused${of}: ${err.message}`;
    }
    if (missing || !deps.ask || opts.yes) {
      progress("preparing", `Note: ${why}; going on without AWS${missing ? "" : ` (${opts.agent ? "aws_profile" : "--aws-profile"} picks a profile)`}`);
      return {};
    }
    const profiles = deps.awsProfiles ? await deps.awsProfiles().catch(() => [] as string[]) : [];
    const a = (await deps.ask(`AWS profile to read ${ep.id} with${profiles.length ? ` (${profiles.join(", ")})` : ""}, or Enter to go on without AWS: `, [`${why}.`])).trim();
    if (!a) return {};
    profile = a;
  }
  return {};
}

/** The box's verification over PrivateLink in words (the console's), when it failed. */
const VERIFY_FAILED: Record<string, string> = {
  unreachable: "The monitoring box could not reach the database over PrivateLink: check that the stack finished, its load balancer's target is healthy, and the RDS is not publicly accessible.",
  auth_failed: `The monitoring box reached the database, but ${DEFAULT_MONITORING_USER} could not log in: its password changed after this deploy set it (another setup ran its SQL).`,
  nonce_mismatch: "A newer PostgresAI setup SQL ran in this database after this deploy's (another deploy, or the Console), so monitoring does not start.",
};

/** A box's PrivateLink route (postgresai#454), for list, watch and delete; undefined for any other box. */
export async function privateLinkOf(d: Database, rdsStatus?: ConnectDeps["rdsStatus"]): Promise<RdsStatus | undefined> {
  if (!rdsStatus || disconnecting(d.status)) return undefined;
  // Made by deploy: provider rds or aurora; in the console: no server in the list.
  if (d.cluster && d.provider !== "rds" && d.provider !== "aurora") return undefined;
  return rdsStatus(d.id).catch(() => undefined);
}

/** The box's check of the database over PrivateLink, in words. */
export function verifiedWords(verified: string): string {
  return verified === "db_verified" ? "verified" : verified === "pending" ? "not verified yet" : VERIFY_FAILED[verified] ?? verified;
}

/** Why a PrivateLink box can't monitor its database (it failed the check); undefined otherwise. */
export const privateLinkFailure = (s: RdsStatus | undefined): string | undefined => (s ? VERIFY_FAILED[s.verified_status] : undefined);

/** After a PrivateLink box is deleted: the customer's stack, which only they can delete. */
export function stackTeardown(s: RdsStatus, id: string): string {
  const ep = rdsEndpoint(s.rds_endpoint);
  const made = ep ? ` If deploy's link made it:\n  aws cloudformation delete-stack --stack-name ${stackNameFor(ep.id)} --region ${ep.region}` : "";
  return `Once it is deleted (pgai mon instances watch ${id}), and if nothing else is to monitor this RDS (a new deploy for it reuses the stack), delete the PrivateLink stack in your AWS account whose VpceServiceName is ${s.vpce_service_name} (terraform destroy, if it was made with Terraform).${made}`;
}

/** A time as people read it in UTC ("2026-10-28 12:02:03 UTC"); AWS writes "2026-10-28T12:02:03.456000+00:00". */
const utc = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : `${d.toISOString().slice(0, 19).replace("T", " ")} UTC`;
};

/**
 * The URL in database `db`: a URL without a path would connect pg to its user's default one. A path
 * given is kept as written (pg decodes it with decodeURI, so re-encoding it could name another database).
 */
function inDatabase(url: string, db: string): string {
  const u = new URL(url);
  if (u.pathname.length <= 1) u.pathname = `/${encodeURIComponent(db)}`;
  return u.toString();
}

/** The monitoring role's login from here, in database `db`: no user or password of the given URL's query string. */
function roleUrl(url: string, db: string, password: string): string {
  const u = new URL(loginUrlFor(url, DEFAULT_MONITORING_USER, db, password));
  u.pathname = new URL(inDatabase(url, db)).pathname;
  return u.toString();
}

/**
 * A private RDS over AWS PrivateLink (postgresai#454), the console's wizard from here: the price,
 * the customer's stack (the CloudFormation link, or Terraform), the monitoring role with the org's
 * ownership nonce, then the box on PostgresAI's AWS in the RDS's region. The role is prepared
 * last: every prepare SQL is a new nonce, and the box is made with the newest.
 */
async function deployPrivateLink(url: string, route: Extract<RdsRoute, { private: true }>, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const provider: Provider = "rds";
  const { ep, inst } = route;
  const name = databaseName(url);
  const db = name.split("/").slice(1).join("/") || "postgres";
  const port = Number(new URL(url).port || inst?.port || 5432);
  const cluster = `${ep.host}:${port}`;
  const flag = (cli: string, agent: string) => (opts.agent ? agent : cli);
  const stop = (next: string, more: Partial<ConnectResult> = {}): ConnectResult => ({ status: "action_required", provider, name, ...more, next });
  const notHere = ([[opts.cloud, "--cloud", "cloud"], [opts.location, "--location", "location"], [opts.name, "--name", "name"],
    [opts.ipFirst || undefined, "--ip-first", "ip_first"], [opts.ipAllowed || undefined, "--ip-allowed", "ip_allowed"],
    [opts.resetPassword || undefined, "--reset-password", "reset_password"], [opts.monitoringUser, "--monitoring-user", "monitoring_user"]] as const)
    .find(([v]) => v !== undefined && v !== "");
  if (notHere) return stop(`${flag(notHere[1], notHere[2])} does not apply over ${PL}: the box runs on AWS in ${ep.region} (the RDS's region), as ${DEFAULT_MONITORING_USER}, in a project named ${ep.id}`);
  if ((opts.monitoringPassword === undefined) !== (opts.nonce === undefined)) {
    return stop(opts.agent ? "Pass monitoring_password and nonce together (both from the prepare_db result)" : "Set PGAI_MON_PASSWORD and PGAI_MON_NONCE together (both from the printed SQL's result), or neither");
  }
  // One box per RDS: the prepare SQL sets the cluster-wide role's password anew (the platform keeps
  // none), so a box on this RDS (any database, made here or in the console) is found before anything is changed.
  const rows = await deps.list();
  const live = rows.filter((d) => !disconnecting(d.status));
  const onServer = live.filter((d) => d.cluster && serverOf(d.cluster) === cluster);
  const verify = async (row: Database) => VERIFY_FAILED[(await deps.rdsStatus?.(row.id).catch(() => undefined))?.verified_status ?? ""];
  const own = onServer.find((d) => (d.provider === "rds" || d.provider === "aurora") && d.database != null && decodeURIComponent(d.database) === db);
  if (own) {
    const note = opts.vpceService || opts.hostStatsRole ? "; the flags for a new box were not used: it exists" : "";
    return overPrivateLink(await follow(own, { provider, name, fresh: false, billing: {}, note, verify }, opts, deps, progress));
  }
  const already = (d: Database, how: string) => {
    const instead = d.database != null && decodeURIComponent(d.database) !== db ? `To monitor ${db} instead` : "To set it up from here instead";
    return stop(`${ep.id} is already monitored ${how} (instance ${d.name}). Setting up another instance would change ${DEFAULT_MONITORING_USER}'s password, which that instance logs in with. ${instead}, delete that instance first (pgai mon instances delete ${deleteRef(d, rows)})`);
  };
  if (onServer.length) return already(onServer[0], onServer[0].provider === "rds" ? `over ${PL}, database ${decodeURIComponent(onServer[0].database ?? "?")}` : "directly");
  // A console-made PrivateLink box names no server in the list: its route does.
  const candidates = live.filter((r) => !r.cluster || r.provider === "rds" || r.provider === "aurora");
  // 404: not a PrivateLink box. Any other failure stops here: the SQL below would change the role's password.
  let routes: (RdsStatus | undefined)[];
  try {
    routes = await Promise.all(candidates.map((d) => deps.rdsStatus?.(d.id).catch((err) => {
      if (httpStatus(err) === 404) return undefined;
      throw err;
    })));
  } catch (err) {
    return stop(`Could not check what already monitors ${ep.id} (${errorText(err)}): re-run pgai mon deploy. Nothing was changed`);
  }
  const viaConsole = candidates.find((_, i) => routes[i]?.rds_endpoint.toLowerCase().replace(/\.$/, "") === ep.host);
  if (viaConsole) return already(viaConsole, `over ${PL}, set up in the console`);
  const hostStats = !!opts.hostStats || !!opts.hostStatsRole?.trim();
  const place: Place = { cloud: "aws", location: ep.region, location_reason: "privatelink", location_reason_text: `same region as your database, over ${PL}` };
  progress("preparing", `Monitoring box: AWS, ${ep.region} (${place.location_reason_text})`, { cloud: "aws", location: ep.region, location_reason: "privatelink" });
  let quote: Quote;
  try {
    // The org's price: the PrivateLink box is billed as one more cluster.
    quote = await deps.quote(opts.coupon);
  } catch (err) {
    if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
    throw err;
  }
  // From AWS, said with the price.
  let vcpus = opts.vcpus;
  if (vcpus === undefined && route.aws && inst?.instanceClass) {
    vcpus = await vcpusOfClass(route.aws, inst.instanceClass, ep.region);
    if (vcpus) progress("billing", `vCPUs: ${vcpus} (${inst.instanceClass})`);
  }
  const where = `AWS, ${ep.region}`;
  const b = await billingFor(quote, name, opts, deps, (price) => progress("billing", estimateLine(price)), async (price) => {
    if (opts.yes) return { go: true };
    if (deps.ask) {
      return await deps.confirm(`Deploy ${terminalText(name)} over ${PL} (monitoring box on ${where})? [y/N] `)
        ? { go: true } : { go: false, stop: { next: `Nothing was created: re-run with --yes to deploy over ${PL}` } };
    }
    return { go: false, stop: { next: opts.agent
      ? `Show the user the price (${price}); the monitoring box runs on ${where}, reached over ${PL}. Then call deploy_monitoring again with yes: true. Nothing was created.`
      : `The monitoring box would run on ${where} for ${price}, reached over ${PL}: re-run with --yes to deploy. Nothing was created.` } };
  });
  if (b.stop) return { ...stop(b.stop.next), ...place, ...b.stop };
  if (deps.aborted?.()) throw new CancelledError();
  const more = { ...place, ...b.billing };
  // Asked after the price, before the steps (the first step is the customer's stack).
  if (vcpus === undefined) {
    vcpus = !opts.yes && !opts.agent && deps.askVcpus ? await deps.askVcpus(VCPUS_QUESTION) : undefined;
    if (deps.aborted?.()) throw new CancelledError();
    if (vcpus === undefined) progress("billing", NO_VCPUS_NOTE);
  }
  const stack = await privateLinkStack({ ep, aws: route.aws, inst, port, hostStats, opts, deps, progress, stop: (next, m = {}) => stop(next, { ...more, ...m }) });
  if ("status" in stack) return stack;
  if (deps.aborted?.()) throw new CancelledError();
  const role = await privateLinkRole(url, db, opts, deps, progress, (next, m = {}) => stop(next, { ...more, ...m }));
  if ("status" in role) return role;
  if (deps.aborted?.()) throw new CancelledError();
  progress("provisioning", `Provisioning monitoring for ${name} over ${PL}`);
  const created = await deps.rdsProvision!({
    vpce_service_name: stack.vpce,
    rds_endpoint: ep.host,
    db_password: role.password,
    rds_port: port,
    rds_database: db,
    region: ep.region,
    ...(b.accepted ? { accept_price: true } : {}),
    ...(opts.coupon ? { promo_code: opts.coupon } : {}),
    ...(stack.roleArn ? { host_stats_role_arn: stack.roleArn } : {}),
    ...(vcpus ? { vcpus } : {}),
  }).catch((err) => {
    if (err instanceof HttpStatusError && err.status === 402) return /declined/i.test(err.message) ? "declined" as const : "payment" as const;
    if (err instanceof HttpStatusError && err.status === 412) return "price" as const;
    // Refused before anything was made: the platform's words (a region not offered, the RDS monitored, a partner's org).
    if (err instanceof HttpStatusError && err.status < 500) return { refused: errorText(err) };
    if (mayHaveRun(err)) return "unanswered" as const;
    throw err;
  });
  if (typeof created === "object" && "refused" in created) return stop(`${created.refused.replace(/\.?$/, ".")} No monitoring box was created.`, more);
  if (created === "payment") return { status: "action_required", provider, name, ...place, ...b.billingPage };
  if (created === "declined") return { status: "action_required", provider, name, ...place, ...b.billingPage, next: b.billingPage.next.replace(/^Add a payment method at /, "The payment method on file was declined: update it at ") };
  if (created === "price") return stop("The price changed since it was shown: re-run pgai mon deploy to see it", more);
  let box: Database;
  if (created === "unanswered") {
    progress("provisioning", STILL_CREATING);
    const known = new Set(rows.map((d) => d.id));
    const made = await lookForBox((list) => list.find((d) => !known.has(d.id) && !!d.cluster && serverOf(d.cluster) === cluster), opts, deps, deps.now());
    if (!made) return stop(deps.aborted?.() ? STOPPED_LOOKING : opts.agent ? NOT_CONFIRMED_AGENT : NOT_CONFIRMED, more);
    box = made;
  } else {
    box = { id: created.id, name: created.name, provider, status: created.status, dashboard_url: null, host_metrics: !!stack.roleArn };
  }
  if (deps.aborted?.()) return { ...connectStatus(box, provider, true), ...more };
  return overPrivateLink(await follow(box, { provider, name, fresh: true, billing: b.billing, billingUrl: b.billingUrl, note: "", place, verify }, opts, deps, progress));
}

/** Why a PrivateLink box fails to launch, nearly always: the customer's endpoint service, which PostgresAI connects to. */
export const PRIVATELINK_LAUNCH_FAILED = `The monitoring box could not connect to the stack's endpoint service: check that the stack finished, and that the endpoint service (VPC console, Endpoint services, Allow principals) lists ${PRINCIPAL_ARN}.`;

/**
 * A PrivateLink box's result: it reaches the database over PrivateLink, so its public IP ("Connects from")
 * says nothing. One that is not ready says why first: its failed check of the database (`why`, or already
 * in the result), else for a failed launch, the endpoint service.
 */
export function overPrivateLink(r: ConnectResult, why?: string): ConnectResult {
  const { egress_ip: _ip, ...rest } = r;
  if (rest.status === "ready") return rest;
  const said = Object.values(VERIFY_FAILED).some((t) => rest.next.includes(t));
  const first = said ? undefined : why ?? (rest.status === "failed" ? PRIVATELINK_LAUNCH_FAILED : undefined);
  return first ? { ...rest, next: `${first} ${rest.status === "failed" ? "Then: " : ""}${rest.next}` } : rest;
}

/**
 * The customer's stack, which publishes the RDS to PostgresAI: its VpceServiceName (and the host
 * metrics role). Given (--vpce-service), read from AWS here, or pasted at a terminal; else the
 * link (or Terraform) to make it, and how to resume (step privatelink_stack). AWS is only read.
 */
async function privateLinkStack(c: {
  ep: RdsEndpoint; aws?: AwsRun; inst?: DbInstance; port: number; hostStats: boolean; opts: ConnectOptions; deps: ConnectDeps; progress: Progress;
  stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult;
}): Promise<ConnectResult | { vpce: string; roleArn?: string }> {
  const { ep, opts, deps, progress } = c;
  const flag = (cli: string, agent: string) => (opts.agent ? agent : cli);
  // At a terminal deploy waits for the stack; with --yes it asks nothing (no [o], no paste).
  const terminal = !!deps.ask;
  const ask = opts.yes ? undefined : deps.ask;
  // Terraform makes no CloudFormation stack to find: its output is pasted, or passed on the next run.
  // AWS still filled in terraform.tfvars (c.inst).
  let aws = opts.terraform ? undefined : c.aws;
  let vpce = opts.vpceService?.trim() || undefined;
  let roleArn = opts.hostStatsRole?.trim() || undefined;
  const badVpce = (v: string): string | undefined => {
    const m = VPCE_SERVICE.exec(v);
    if (!m) return `${v} is not a VpceServiceName (com.amazonaws.vpce.<region>.vpce-svc-<id>, the stack's output)`;
    if (m[1] !== ep.region) return `${v} is in ${m[1]}, and ${ep.id} in ${ep.region}: ${PL} connects within one region, so use the stack made in ${ep.region}`;
    return undefined;
  };
  if (roleArn && !ROLE_ARN.test(roleArn)) return c.stop(`${roleArn} is not an IAM role ARN (arn:aws:iam::<account>:role/<name>, the stack's HostStatsRoleArn output)`);
  if (vpce && badVpce(vpce)) return c.stop(badVpce(vpce)!);
  let externalId: string | undefined;
  const externalIdOf = async () => (externalId ??= await deps.rdsExternalId!());
  // What deploy said of the stack to make: once.
  let shown: Partial<ConnectResult> | undefined;
  const show = async (): Promise<Partial<ConnectResult>> => {
    const inputs: StackInputs = {
      region: ep.region, rdsId: ep.id, port: c.port,
      vpcId: c.inst?.vpcId, subnetIds: c.inst ? oneSubnetPerAz(c.inst.subnets) : undefined,
      // Several security groups: the customer picks the one that admits the RDS's traffic.
      securityGroupId: c.inst?.securityGroupIds.length === 1 ? c.inst.securityGroupIds[0] : undefined,
      ...(c.hostStats ? { externalId: await externalIdOf() } : {}),
    };
    const cfn = quickCreateUrl(inputs);
    // What the link leaves to choose in the form (no AWS here, or several security groups).
    // Named as the form (CloudFormation) or terraform.tfvars (--terraform) names them.
    const [vpcKey, subnetsKey, sgKey] = opts.terraform ? ["vpc_id", "subnet_ids", "rds_security_group_id"] : ["VpcId", "SubnetIds", "RdsSecurityGroupId"];
    const w = sgKey.length + 2;
    const blanks = [
      !inputs.vpcId && `  ${vpcKey.padEnd(w)}the RDS's VPC`,
      !inputs.subnetIds?.length && `  ${subnetsKey.padEnd(w)}one subnet per availability zone of the RDS's subnet group`,
      !inputs.securityGroupId && `  ${sgKey.padEnd(w)}the RDS's security group`,
    ].filter((l): l is string => !!l);
    const choose = blanks.length
      ? `\n${opts.terraform ? "In terraform.tfvars, fill in" : "In the form, choose (empty in the link)"}:\n${blanks.join("\n")}\n(the RDS console lists them: ${ep.id}, Connectivity & security)` : "";
    let dir: string | undefined;
    if (opts.terraform) dir = await deps.writeTerraform!(stackNameFor(ep.id), terraformFiles(await deps.terraformModule!(), inputs));
    // Where no browser opens here (SSH to a server, say), the person opens the link elsewhere.
    const where = deps.openUrl ? "" : "\nOpen it in a browser signed in to that AWS account.";
    const tf = opts.agent ? "" : `\nPrefer Terraform? ${deps.ask ? "Stop (Ctrl-C) and re-run" : "Re-run"} with --terraform: it writes the same stack as a module.`;
    progress("privatelink", dir
      ? `Terraform for the PrivateLink stack written to ${dir}: terraform init && terraform apply there${choose}`
      : `Create the PrivateLink stack in your AWS account from this link; nothing becomes public:${choose}\n${cfn}${where}${tf}`, { cfn_url: cfn, stack_name: stackNameFor(ep.id) });
    return { step: "privatelink_stack", cfn_url: cfn, stack_name: stackNameFor(ep.id), region: ep.region,
      ...(inputs.externalId ? { external_id: inputs.externalId } : {}), ...(dir ? { terraform_dir: dir } : {}),
      ...(blanks.length ? { choose_in_form: blanks.map((l) => l.trim().split(/\s+/)[0]) } : {}) };
  };
  const resume = () => {
    const output = opts.terraform ? "vpce_service_name output" : "VpceServiceName output";
    const role = c.hostStats && !roleArn ? ` and ${flag("--host-stats-role", "host_stats_role")} <its ${opts.terraform ? "host_stats_role_arn" : "HostStatsRoleArn"} output>` : "";
    return aws ? (opts.agent ? "call deploy_monitoring again" : "re-run pgai mon deploy") : `${opts.agent ? "call deploy_monitoring again with" : "re-run with"} ${flag("--vpce-service", "vpce_service")} <its ${output}>${role}`;
  };
  const deadline = deps.now() + STACK_WAIT_MS;
  // Why the last paste was refused (or that Enter does not stop): shown with the question again.
  let refusedPaste: string | undefined;
  let saidMisnamed = false;
  // The vpce came from a stack read here (not pasted or passed): its outputs were seen.
  let readStack = false;
  // Stacks for the RDS this run saw being made, or created since it started (a failure of theirs is
  // said), and older failed ones (said once while deploy waits for a new one).
  const seenMaking = new Set<string>();
  const saidFailed = new Set<string>();
  const startedAt = Date.now() - 60_000;
  const thisRuns = (s: Stack) => s.name === stackNameFor(ep.id) || seenMaking.has(s.name) || (!!s.created && Date.parse(s.created) >= startedAt);
  for (let pastes = 0, empties = 0; !vpce;) {
    let found: Stack[] = [];
    if (aws) {
      try {
        const r = await findStacks(aws, ep.id, ep.region);
        found = r.stacks;
        // A stack with deploy's name for another RDS is never used.
        if (r.misnamed && !saidMisnamed) {
          saidMisnamed = true;
          progress("privatelink", `Note: stack ${r.misnamed.name} is for RDS ${r.misnamed.rdsId ?? "(none)"}, not ${ep.id}: not used. Give the new stack another name in the form`);
        }
      } catch (err) {
        if (!(err instanceof AwsError)) throw err;
        progress("privatelink", `Note: could not read CloudFormation (${err.message}); going on without AWS`);
        aws = undefined;
      }
    }
    const ready = found.filter((s) => stackState(s.status) === "ready" && s.outputs.VpceServiceName);
    let stack: Stack | undefined;
    if (ready.length === 1) stack = ready[0];
    else if (ready.length > 1) {
      // Several stacks publish this RDS: the person picks one; elsewhere the result lists them.
      const rows = ready.map((s, i) => `  ${i + 1}. ${s.name}  ${s.status}${s.created ? `  created ${utc(s.created)}` : ""}  ${s.outputs.VpceServiceName}`);
      const listed = { step: "privatelink_stack" as const, region: ep.region,
        stacks: ready.map((s) => ({ name: s.name, status: s.status, created: s.created ?? null, vpce_service_name: s.outputs.VpceServiceName })) };
      if (!ask) {
        return c.stop(`Several stacks publish ${ep.id}: ${opts.agent ? "ask the user which one, then call deploy_monitoring again with vpce_service" : "re-run with --vpce-service"} <the VpceServiceName of the one to use>. Nothing was created on PostgresAI yet.\n${rows.join("\n")}`, listed);
      }
      for (let tries = 0; !stack && tries < 3; tries++) {
        const a = (await ask(`The stack to use (1-${ready.length}, Enter to stop): `, tries ? [`Type a number from 1 to ${ready.length}.`] : [`Several stacks publish ${ep.id}:`, ...rows])).trim();
        if (!a) return c.stop(`Stopped: re-run with --vpce-service <the VpceServiceName of the stack to use>. Nothing was created on PostgresAI yet.`, listed);
        const i = Number(a);
        if (Number.isInteger(i) && i >= 1 && i <= ready.length) stack = ready[i - 1];
      }
      if (!stack) return c.stop(`No stack picked: re-run with --vpce-service <the VpceServiceName of the stack to use>. Nothing was created on PostgresAI yet.`, listed);
    }
    if (stack) {
      vpce = stack.outputs.VpceServiceName;
      roleArn ??= stack.outputs.HostStatsRoleArn;
      readStack = true;
      progress("privatelink", `Using stack ${stack.name}${stack.created ? ` (created ${utc(stack.created)})` : ""}: ${vpce}`, { stack_name: stack.name });
      break;
    }
    // A failed stack, with nothing ready or being made: why, and how to make it again. Deploy's own, or one
    // made since this run started (any name); a run that does not wait (a script, an agent) reports the
    // newest failed one, as the next run would not know it as its own. Waiting at a terminal, an older
    // failed one is said once and deploy waits for a new stack.
    for (const s of found) if (stackState(s.status) === "in_progress") seenMaking.add(s.name);
    // A done stack without the template's output (made from another template).
    const foreign = found.find((s) => s.name === stackNameFor(ep.id) && stackState(s.status) === "ready" && !s.outputs.VpceServiceName);
    if (foreign && !found.some((s) => stackState(s.status) === "in_progress")) {
      return c.stop(`Stack ${foreign.name} has no VpceServiceName output: it was not made from the PostgresAI template. Delete it, create the stack from the link deploy prints, then ${resume()}`, { step: "privatelink_stack", stack_name: foreign.name, region: ep.region });
    }
    const failedNow = found.filter((s) => stackState(s.status) === "failed")
      .sort((a, b) => Date.parse(b.created ?? "") - Date.parse(a.created ?? "") || 0);
    const waits = terminal && !!aws;
    const failed = failedNow.find(thisRuns) ?? (waits ? undefined : failedNow[0]);
    for (const s of failedNow) {
      if (s === failed || saidFailed.has(s.name)) continue;
      saidFailed.add(s.name);
      progress("privatelink", `Note: stack ${s.name} for ${ep.id} failed earlier (${s.status}): not used`);
    }
    if (failed && !found.some((s) => stackState(s.status) === "in_progress")) {
      const why = ((aws && await stackFailure(aws, failed.name, ep.region)) || failed.reason || failed.status).replace(/\s*\(RequestId: [^)]*\)/g, "");
      // A stack that never got made is deleted and made again; one whose update failed is repaired where it is.
      const fix = /^(CREATE_FAILED|ROLLBACK_|DELETE_FAILED)/.test(failed.status)
        ? `delete the stack, then re-run pgai mon deploy:\n  aws cloudformation delete-stack --stack-name ${failed.name} --region ${ep.region}`
        : "repair the stack in the CloudFormation console (Stack actions, Continue update rollback), then re-run pgai mon deploy";
      return c.stop(`Stack ${failed.name} failed (${failed.status}): ${why}\nFix that, ${fix}`, { step: "privatelink_stack", stack_name: failed.name, region: ep.region });
    }
    const being = found.find((s) => stackState(s.status) === "in_progress");
    const first = !shown;
    shown ??= await show();
    const making = being ? `Stack ${being.name} is ${/ROLLBACK/.test(being.status) ? "rolling back" : "being created"} (${being.status})` : undefined;
    if (!terminal || (!aws && !ask)) {
      const make = shown.terraform_dir ? `Run terraform init && terraform apply in ${shown.terraform_dir}`
        : opts.agent ? `Give the user cfn_url to open in the AWS account of ${ep.id}, to create the stack there${shown.choose_in_form ? ` (choosing ${shown.choose_in_form.join(", ")} in the form)` : ""}, or call again with terraform: true for the same as Terraform`
        : `Open the link below in the AWS account of ${ep.id} and create the stack there${shown.choose_in_form ? ` (choose ${shown.choose_in_form.join(", ")} in the form)` : ""}, or re-run with --terraform for the same as Terraform`;
      return c.stop(`${making ? `${making}: once it is CREATE_COMPLETE` : `${make}; once it is done`}, ${resume()}. Nothing was created on PostgresAI yet.`, shown);
    }
    // The link in the browser, when the person asks for it ([o]) and a browser can open here.
    const openIt = !shown.terraform_dir && deps.openUrl ? ", [o] to open the link in your browser" : "";
    if (aws) {
      if (first && openIt && ask) {
        const a = (await ask(`[o] opens the link in your browser; Enter waits for the stack: `)).trim().toLowerCase();
        if (a === "o") deps.openUrl!(shown.cfn_url!);
      }
      if (deps.aborted?.()) throw new CancelledError();
      if (deps.now() >= deadline) return c.stop(`No stack for ${ep.id} was ready within ${STACK_WAIT_MS / 60_000} minutes: re-run pgai mon deploy once it is. Nothing was created on PostgresAI yet.`, shown);
      progress("privatelink", making ?? `Waiting for you to create the stack from the link above (any name; deploy finds it by its RDS; up to ${STACK_WAIT_MS / 60_000} min)`, { stack_name: being?.name ?? stackNameFor(ep.id) });
      await deps.sleep(STACK_POLL_MS);
      continue;
    }
    const output = opts.terraform ? "vpce_service_name output" : "VpceServiceName (CloudFormation, the stack, Outputs)";
    const a = (await ask!(`Paste the stack's ${output}${openIt}, or [s] to stop: `, refusedPaste ? [refusedPaste] : [])).trim();
    if (a.toLowerCase() === "s") return c.stop(`Stopped: once the stack is done, ${resume()}. Nothing was created on PostgresAI yet.`, shown);
    if (!a) {
      // Enter does not stop (it waits elsewhere); several in a row do (Ctrl-D ends the run, exit 130).
      if (++empties >= 3) return c.stop(`Stopped: once the stack is done, ${resume()}. Nothing was created on PostgresAI yet.`, shown);
      refusedPaste = `Paste the stack's ${opts.terraform ? "vpce_service_name output" : "VpceServiceName"} once it is created, or s to stop.`;
      continue;
    }
    empties = 0;
    if (openIt && a.toLowerCase() === "o") {
      deps.openUrl!(shown.cfn_url!);
      refusedPaste = undefined;
      continue;
    }
    refusedPaste = badVpce(a);
    if (!refusedPaste) vpce = a;
    else if (++pastes >= 3) return c.stop(refusedPaste, shown);
  }
  if (c.hostStats && !roleArn && !readStack) {
    // Pasted or passed: the stack or module made the role if it got the External ID; its output is not read here.
    const out = opts.terraform ? "host_stats_role_arn" : "HostStatsRoleArn";
    return c.stop(`Host metrics need the metrics role too: ${opts.agent ? "call deploy_monitoring again with" : "re-run with"} ${flag("--vpce-service", "vpce_service")} ${vpce} and ${flag("--host-stats-role", "host_stats_role")} <its ${out} output>. If it has none, it was made without this PostgresAiExternalId: ${opts.terraform ? "set postgres_ai_external_id in terraform.tfvars and apply again" : "update the stack with it"}; or deploy without ${flag("--host-stats", "host_stats")}.\n  PostgresAiExternalId = ${await externalIdOf()}`, { external_id: externalId });
  }
  if (c.hostStats && !roleArn) {
    return c.stop(`Host metrics need the stack's metrics role (its HostStatsRoleArn output), and it has none: update the stack with this PostgresAiExternalId (it then creates a read-only role), then ${resume()}; or deploy without ${flag("--host-stats", "host_stats")}.\n  PostgresAiExternalId = ${await externalIdOf()}`, { external_id: externalId });
  }
  return { vpce, ...(roleArn ? { roleArn } : {}) };
}

/**
 * postgres_ai_mon with the org's ownership nonce (v1.cloud_monitoring_rds_prepare_sql): run here
 * with the URL's user, or printed for the master user. Resumed with the password and nonce of SQL
 * that already ran. Either way the database's newest marker must be this nonce before the box is made.
 */
async function privateLinkRole(url: string, db: string, opts: ConnectOptions, deps: ConnectDeps, progress: Progress,
  stop: (next: string, more?: Partial<ConnectResult>) => ConnectResult): Promise<ConnectResult | { password: string; nonce: string }> {
  let password = opts.monitoringPassword;
  let nonce = opts.nonce;
  // --yes: nobody is asked (the step result says what to do).
  const ask = opts.yes ? undefined : deps.ask;
  // A session's TLS warning, once, through the step view (a raw line breaks it).
  const warned = new Set<string>();
  const warn = (m: string) => { if (!warned.has(m)) { warned.add(m); progress("preparing", m); } };
  const resumeWith = (n: string) => opts.agent
    ? `call deploy_monitoring again with monitoring_password and nonce from this result`
    : `re-run with PGAI_MON_NONCE=${n} and PGAI_MON_PASSWORD set to the v_password value near the top of the SQL`;
  if (password === undefined || nonce === undefined) {
    let p: RdsPrepareSql;
    try {
      p = await deps.rdsPrepareSql!(db);
    } catch (err) {
      if (err instanceof HttpStatusError && err.status === 403) return stop(notAdminNext);
      if (err instanceof HttpStatusError && err.status < 500) return stop(errorText(err));
      throw err;
    }
    if (!/^[A-Za-z0-9_-]{1,128}$/.test(p.nonce)) throw new Error("cloud_monitoring_rds_prepare_sql returned a malformed nonce: nothing was changed");
    password = p.db_password;
    nonce = p.nonce;
    progress("preparing", `Preparing ${maskConnectionString(url)}`);
    const failed = await deps.runSql!(inDatabase(url, db), p.prepare_db_sql, warn).then(() => undefined, (err: unknown) => err);
    if (failed) {
      // Not logged in (node's errors carry no severity; the server's FATAL ones refuse the login: a password,
      // pg_hba, a database that is not there): no SQL to print. A refusal of the SQL itself prints it.
      const e = failed as { severity?: unknown; code?: unknown };
      const refusedLogin = typeof e !== "object" || !e || !("severity" in e) || e.severity === "FATAL" || /^(28|3D|08)/.test(String(e.code ?? ""));
      if (refusedLogin) return stop(`Could not log in to ${maskConnectionString(inDatabase(url, db))}: ${errorText(failed)}. Check the URL's user and password, and run pgai mon deploy where the database is reachable (in its VPC)`);
      const why = `The user in this URL (${decodeURIComponent(new URL(url).username)}) can't create ${DEFAULT_MONITORING_USER} (${errorText(failed)})`;
      const printed = { step: "prepare_db" as const, sql: p.prepare_db_sql, nonce, monitoring_password: password };
      const run = `Run the SQL as the master user, connected to database ${db}`;
      // The platform makes the box with the newest SQL issued for this database (platform-all#1058).
      const only = `A deploy of this database without ${opts.agent ? "them" : "PGAI_MON_PASSWORD and PGAI_MON_NONCE"} issues a new SQL and replaces this one`;
      if (!ask) return stop(`${why}. ${run}, then ${resumeWith(nonce)}. ${only}.`, printed);
      const a = (await ask(`${run}, then press Enter ([s] to stop): `, [`${why}. The SQL:`, "", p.prepare_db_sql, ""])).trim().toLowerCase();
      if (a === "s" || a === "stop") return stop(`Stopped. ${run}, then ${resumeWith(nonce)}.`, { ...printed, sql: undefined });
    }
  }
  // The box logs in as the role and checks this nonce: both checked here, before it exists.
  progress("preparing", `Checking that ${DEFAULT_MONITORING_USER} can log in to ${db} with the password the SQL set`);
  let found: string | undefined;
  try {
    found = await deps.readNonce!(roleUrl(url, db, password), warn);
  } catch (err) {
    const fresh = opts.agent ? "call again without monitoring_password and nonce for a new SQL" : "re-run without PGAI_MON_PASSWORD and PGAI_MON_NONCE for a new SQL";
    return stop(opts.monitoringPassword === undefined
      ? `${DEFAULT_MONITORING_USER} could not log in to ${db} with the password the SQL set (${errorText(err)}): run the SQL again, connected to database ${db}`
      : opts.agent
        ? `${DEFAULT_MONITORING_USER} could not log in to ${db} with monitoring_password (${errorText(err)}): pass the monitoring_password of the prepare_db result whose SQL ran, or ${fresh}`
        : `${DEFAULT_MONITORING_USER} could not log in to ${db} with PGAI_MON_PASSWORD (${errorText(err)}): set it to the v_password value near the top of the SQL you ran, or ${fresh}`);
  }
  const again = opts.monitoringPassword !== undefined
    ? `${opts.agent ? "call again without monitoring_password and nonce" : "re-run without PGAI_MON_PASSWORD and PGAI_MON_NONCE"} for a new SQL, and run only that one`
    : "re-run pgai mon deploy";
  if (found === undefined) return stop(`The SQL has not run in database ${db} (perhaps in another database): run it connected to database ${db}, then ${resumeWith(nonce)}`);
  if (found !== nonce) return stop(`A newer PostgresAI setup SQL ran in database ${db} after this one (another deploy, or the Console): ${again}`);
  return { password, nonce };
}

type Progress = (event: ProgressEvent["event"], message: string, more?: Partial<ProgressEvent>) => void;

/** What deploy knows of a box when it starts following it. */
interface Followed {
  provider: Provider;
  name: string;
  /** Provisioned by this run: the result gets the first checkup's ETA. */
  fresh: boolean;
  billing: Billing;
  billingUrl?: string;
  /** Drops the role this run created with a generated password, once its box is gone. */
  undoRole?: () => Promise<unknown>;
  /** Appended to the next action ("; --name ignored ..."). */
  note: string;
  /** Where the box runs (placeBox), for the result. */
  place?: Place;
  /** The database URL as given, for the next command shown (its password is never printed). */
  url?: string;
  /** Why the box can't verify the database yet (PrivateLink, postgresai#454): said once, and with a result that is not ready. */
  verify?: (row: Database) => Promise<string | undefined>;
}

/** How the next action starts when deploy's wait ended before the box was ready (the step view is not failed then). */
export const STILL_IN_PROGRESS = "Still in progress";

/**
 * How long the box request (cloud_monitoring_connect) may take: just under the 60 s the
 * platform's nginx gives an API call, so a slow answer times out here, not as a 504. On
 * 2026-10-22 it took 31 s, waiting on a lock (platform-all#988).
 */
export const CONNECT_TIMEOUT_MS = 55_000;
/** An MCP tool call must answer within the 60 s many clients allow: the request keeps 25 s, and the look ends 50 s into the run. */
const AGENT_CONNECT_TIMEOUT_MS = 25_000;
/** Shown while deploy looks for the box of a request that got no answer (postgresai#443). */
export const STILL_CREATING = "Still creating the monitoring box";
export const NOT_CONFIRMED = "PostgresAI did not confirm the monitoring box within 5 minutes. Run pgai mon instances list: if this database is there, follow it with pgai mon instances watch <id>; if not, run pgai mon deploy again";
export const STOPPED_LOOKING = `${STILL_IN_PROGRESS}: deploy stopped watching before PostgresAI confirmed the monitoring box; it may still appear. Run pgai mon instances list before running deploy again`;
/** The MCP tool's: a second call finds the box (it is safe to call again). */
const NOT_CONFIRMED_AGENT = "PostgresAI did not confirm the monitoring box yet. Call deploy_monitoring again in a minute: it continues with the box if PostgresAI created it. If pgai mon instances list still has no box for this database, ask the user to run pgai mon deploy with --reset-password in a terminal";
/** How long deploy looks for that box: it appears once the platform's request commits. An agent's tool call waits less. */
export const LOOKUP_MS = 5 * 60_000;
const AGENT_RUN_MS = 50_000;

/**
 * No answer to a request the platform may still have run: a timeout, a 5xx without the
 * function's SQLSTATE (a proxy's 502/504/524, a lost database connection), or a connection
 * lost after it was sent. A 5xx with one (PT503 "deploys are unavailable") is the platform's
 * answer: its transaction rolled back. A refused connection or a name that does not
 * resolve: nothing reached the platform.
 */
export function mayHaveRun(err: unknown): boolean {
  if (err instanceof HttpRequestTimeoutError) return true;
  if (err instanceof HttpStatusError) return err.status >= 500 && !err.sqlstate;
  const message = err instanceof Error ? err.message : "";
  return / could not reach /.test(message)
    && !/\((ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ENETUNREACH|EHOSTUNREACH|ETIMEDOUT|UND_ERR_CONNECT_TIMEOUT|CERT_\w+|ERR_TLS_\w+|UNABLE_TO_\w+|DEPTH_ZERO_\w+|SELF_SIGNED_\w+)\)/.test(message);
}

/** The box of a request without an answer, polled until the lookup window ends or Ctrl-C. */
async function lookForBox(findMade: (list: Database[]) => Database | undefined, opts: ConnectOptions, deps: ConnectDeps, started: number): Promise<Database | undefined> {
  // An agent's whole run, steps before the request included, ends by AGENT_RUN_MS; it looks at least once.
  for (const deadline = opts.agent ? started + AGENT_RUN_MS : deps.now() + LOOKUP_MS; ;) {
    // A failed list is a miss: the next poll may answer.
    const made = findMade(await deps.list().catch(() => []));
    if (made) return made;
    const now = deps.now();
    if (deps.aborted?.() || now >= deadline) return undefined;
    // The last check lands at the deadline, not up to a poll past it.
    await deps.sleep(Math.min(POLL_MS, deadline - now));
  }
}

/** Waits for the box until it is no longer starting, or opts.waitMs passed: its state as the result. */
async function follow(row: Database, f: Followed, opts: ConnectOptions, deps: ConnectDeps, progress: Progress): Promise<ConnectResult> {
  const { provider, name, fresh, billingUrl, undoRole, note } = f;
  const url = (_: Database) => f.url ?? name;
  // The result carries where a new box runs, next to its billing.
  const billing = { ...f.billing, ...f.place };
  // The box's state, each time it changes while deploy waits for it (not on a re-run of a connected database).
  let shown: string | undefined;
  let unverified: string | undefined;
  for (const deadline = deps.now() + opts.waitMs; ;) {
    if (f.verify) {
      const why = await f.verify(row);
      if (why && why !== unverified) progress("box", `Note: ${why}`, { id: row.id });
      unverified = why;
    }
    if (row.billing_error && disconnecting(row.status)) {
      if (undoRole) await undoRole();
      return { status: "action_required", provider, name, id: row.id, ...billing,
        next: chargeFailed(row.billing_error, "re-run", billingUrl, opts.agent ? "coupon" : "--coupon") };
    }
    // platform-all#987: the database refused the box until the wait ended (or refused it for another
    // reason): the box is gone, its IP kept; allowing it and running again creates a new box with it.
    if (row.status === "ip_reserved") {
      // The box is gone: a role this run made with a generated password is dropped (the re-run makes it again).
      if (undoRole) await undoRole();
      return { ...allowIpResult(row, provider, name, opts, url(row)), ...billing };
    }
    const result = connectStatus(row, provider, fresh);
    const state = boxState(row);
    if (state !== shown && (shown !== undefined || result.status === "in_progress")) {
      progress("box", state === "waiting_for_access" && row.egress_ip
        ? `Monitoring box: waiting for database access from ${row.egress_ip}${row.last_error ? ` (${row.last_error})` : ""}`
        : `Monitoring box: ${boxWords(state)}`,
        { state, id: row.id, ...(state === "waiting_for_access" && row.egress_ip ? { ip: row.egress_ip, ...(row.last_error ? { error: row.last_error } : {}) } : {}) });
    }
    shown = state;
    if (result.status !== "in_progress" || deps.now() >= deadline) {
      const final = deps.links ? connectStatus(await deps.links(row), provider, fresh) : result;
      if (final.status === "ready" && provider === "clickhouse" && !row.host_metrics) {
        final.next += "; for CPU, memory and disk, delete the instance (pgai mon instances delete) and deploy again with --clickhouse-key <key-id>:<key-secret>";
      }
      const { next, ...rest } = final;
      const why = unverified && final.status !== "ready" ? `${unverified} ` : "";
      // The wait ran out with the box still starting: not done (exit 3); it goes on on the platform.
      if (final.status === "in_progress" && opts.waitMs > 0) {
        return { ...rest, status: "action_required", ...billing,
          next: `${STILL_IN_PROGRESS} (box: ${boxWords(boxState(row))}) when the wait ended: ${why}${why ? "Resume" : "resume"} with ${next}${note}` };
      }
      return { ...rest, ...billing, next: why + next + note };
    }
    await deps.sleep(POLL_MS);
    // The box is already requested: a failed poll keeps the last known state.
    const listed = await deps.list().catch(() => undefined);
    const now = listed?.find((d) => d.id === row.id);
    // A box this run created that is gone (its role, if this run made one, is dropped).
    if (listed && !now && fresh) {
      if (undoRole) await undoRole();
      return { status: "failed", provider, name, id: row.id, ...billing, next: "The monitoring box was removed before it became active: see pgai mon instances list, then re-run pgai mon deploy" };
    }
    row = now ?? row;
  }
}

type PgClientClass = Parameters<typeof connectWithSslFallback>[0];
type PgClient = Awaited<ReturnType<typeof connectWithSslFallback>>["client"];

export interface PrepareOptions {
  /** The URL comes from an agent (the MCP tool): PGAI_MON_PASSWORD is not read and a TLS failure is not retried in plaintext. */
  agent?: boolean;
  /** An existing postgres_ai_mon gets a new password (PGAI_MON_PASSWORD, else a generated one). */
  resetPassword?: boolean;
  /** Only whether the URL can work: every refusal, nothing created or changed. */
  check?: boolean;
  /** The platform keeps postgres_ai_mon's password for this server: an existing role needs no PGAI_MON_PASSWORD, and the box's URL carries none. */
  storedPassword?: boolean;
  /** The org's other databases on this server (named where they must be disconnected first). */
  others?: string[];
  /** What pgai mon instances delete takes for each of `others` (an id where a name is shared). */
  otherRefs?: string[];
  /** The monitoring role (default postgres_ai_mon). */
  monitoringUser?: string;
  /** The pg client class (tests). */
  Client?: PgClientClass;
  /** Gets the TLS warnings (default: stderr). */
  warn?: Warn;
}

/** A session over `url`; `refusedTls`: the server refused TLS, and the session fell back to plaintext. */
async function openConnection(url: string, opts: PrepareOptions) {
  const config = checkUrlParams(url, opts.agent);
  const conn = resolveAdminConnection({ conn: url });
  if (config) {
    const { connectionString, ...explicit } = conn.clientConfig;
    conn.clientConfig = { ...config, ...explicit };
  }
  const fallback = !opts.agent && !!conn.sslFallbackEnabled;
  const { client, usedSsl } = await connectWithSslFallback(opts.Client ?? Client, { ...conn, sslFallbackEnabled: fallback }, false, opts.warn);
  await client.query("set search_path = pg_catalog, pg_temp");
  return { client, usedSsl, refusedTls: fallback && !usedSsl };
}

/** Drops the monitoring role and what it was granted in this database; false when the server refuses. */
async function dropMonitoringRole(client: PgClient, role: string): Promise<boolean> {
  const q = quoteIdent(role);
  // Alone first: a role with no grants yet needs no DROP OWNED, which a CREATEROLE admin may not run.
  for (const sql of [`drop role ${q}`, `drop owned by ${q}; drop role ${q}`]) {
    if (await client.query(sql).then(() => true, () => false)) return true;
  }
  return false;
}

/** What the plan grants (pg_monitor and pg_read_all_stats), and what pg_monitor itself is a member of. */
const MONITORING_GROUPS = ["pg_monitor", "pg_read_all_stats", "pg_read_all_settings", "pg_stat_scan_tables"];

/** The functions the plan grants EXECUTE on (sql/04, sql/05, sql/06), as schema.name(argument types). */
const MONITORING_FUNCTIONS = ["pg_catalog.pg_stat_file(text)", "pg_catalog.pg_stat_file(text, boolean)", "pg_catalog.pg_ls_dir(text)",
  "pg_catalog.pg_ls_dir(text, boolean, boolean)", "rds_tools.pg_ls_multixactdir()", "postgres_ai.table_describe(text)"];

/** Undoes a `generated` prepareDatabase over the same admin URL: drops the role that run created. */
export async function unprepareDatabase(url: string, opts: PrepareOptions = {}): Promise<boolean> {
  const { client } = await openConnection(url, opts);
  try {
    return await dropMonitoringRole(client, opts.monitoringUser || DEFAULT_MONITORING_USER);
  } finally {
    await client.end();
  }
}

/** Creates (or checks) the monitoring role over the given URL. The admin URL is used for this run only. */
export async function prepareDatabase(url: string, provider: Provider, opts: PrepareOptions = {}): Promise<Prepared> {
  const pgProvider = provider === "rds" ? "self-managed" : provider;
  const role = opts.monitoringUser || DEFAULT_MONITORING_USER;
  const open = (u: string) => openConnection(u, opts);
  // Whether `u` logs in; false only when the server rejects the password.
  const logsIn = async (u: string): Promise<boolean> => {
    try {
      await (await open(u)).client.end();
      return true;
    } catch (err) {
      const { code, routine } = err as { code?: string; routine?: string };
      if (code === "28P01") return false;
      // No CONNECT on the database yet: CheckMyDatabase runs after the password was accepted.
      if (code === "42501" && routine === "CheckMyDatabase") return true;
      throw err;
    }
  };
  const { client, usedSsl, refusedTls } = await open(url).catch((err) => {
    // pg's words for "the server asked for a password and there is none".
    if (/client password must be a string/.test(String(err?.message))) throw new Error("The URL has no password (and PGPASSWORD is not set): pgai mon deploy postgresql://user:password@host:5432/dbname");
    throw err;
  });
  try {
    // Admin: a superuser, or CREATEROLE that can run the rest of the plan too: grant pg_monitor and
    // pg_read_all_stats (PG 16+ needs ADMIN OPTION for that), create the postgres_ai schema, and
    // find pg_stat_statements installed (creating it takes a superuser; on RDS, rds_superuser creates it).
    const me = (await client.query(
      `select session_user as name, current_database() as db,
         rolsuper or (rolcreaterole
           and (current_setting('server_version_num')::int < 160000
             or (pg_has_role(current_user, 'pg_monitor', 'USAGE WITH ADMIN OPTION')
               and pg_has_role(current_user, 'pg_read_all_stats', 'USAGE WITH ADMIN OPTION')))
           and has_database_privilege(current_user, current_database(), 'CREATE')
           and (exists (select 1 from pg_extension where extname = 'pg_stat_statements')
             or exists (select 1 from pg_roles where rolname = 'rds_superuser' and pg_has_role(current_user, oid, 'MEMBER')))) as admin,
         exists (select 1 from pg_roles where rolname = ${quoteLiteral(role)}) as mon_exists,
         exists (select 1 from pg_roles where rolname = ${quoteLiteral(role)} and (rolsuper or rolcreatedb or rolcreaterole or rolreplication or rolbypassrls)) as mon_privileged,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles m join pg_roles g
            on g.oid <> m.oid and pg_has_role(m.oid, g.oid, 'MEMBER')
          where m.rolname = ${quoteLiteral(role)} and g.rolname not in (${MONITORING_GROUPS.map(quoteLiteral).join(", ")})) as mon_groups,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles m
          join pg_auth_members a on a.member = m.oid join pg_roles g on g.oid = a.roleid
          where m.rolname = ${quoteLiteral(role)} and a.admin_option) as mon_admin_groups,
         (select string_agg(g.rolname::text, ', ' order by g.rolname::text collate "C") from pg_roles g
          where g.rolname not in (${[...MONITORING_GROUPS, role].map(quoteLiteral).join(", ")})
            and (pg_has_role('pg_monitor', g.oid, 'MEMBER') or pg_has_role('pg_read_all_stats', g.oid, 'MEMBER'))) as mon_plan_groups,
         exists (select 1 from pg_shdepend d join pg_roles m on m.oid = d.refobjid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'o' and m.rolname = ${quoteLiteral(role)}) as mon_owns,
         (select string_agg(distinct db.datname::text, ', ') from pg_shdepend d join pg_roles m on m.oid = d.refobjid join pg_database db on db.oid = d.dbid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'a' and m.rolname = ${quoteLiteral(role)}
            and db.datname <> current_database()) as mon_other_dbs,
         (select string_agg(distinct pg_describe_object(d.classid, d.objid, d.objsubid), ', ') from pg_shdepend d join pg_roles m on m.oid = d.refobjid
          where d.refclassid = 'pg_authid'::regclass and d.deptype = 'a' and m.rolname = ${quoteLiteral(role)}
            and d.dbid in (0, (select oid from pg_database where datname = current_database()))
            and not exists (select 1 from (
                select relacl as acl, 'SELECT' as ok from pg_class where d.classid = 'pg_class'::regclass and d.objsubid = 0 and oid = d.objid
                  and ((relnamespace not in ('pg_catalog'::regnamespace, 'pg_toast'::regnamespace) and relkind <> 't') or oid = 'pg_catalog.pg_index'::regclass)
                union all select attacl, 'SELECT' from pg_attribute where d.classid = 'pg_class'::regclass and attrelid = d.objid and attnum = d.objsubid
                  and (select relnamespace not in ('pg_catalog'::regnamespace, 'pg_toast'::regnamespace) and relkind <> 't' from pg_class where oid = d.objid)
                union all select nspacl, 'USAGE' from pg_namespace where d.classid = 'pg_namespace'::regclass and oid = d.objid and nspname !~ '^pg_toast'
                union all select proacl, 'EXECUTE' from pg_proc where d.classid = 'pg_proc'::regclass and oid = d.objid
                  and pronamespace::regnamespace::text || '.' || proname || '(' || array_to_string(proargtypes::oid[]::regtype[], ', ') || ')'
                    in (${MONITORING_FUNCTIONS.map(quoteLiteral).join(", ")})
                union all select datacl, 'CONNECT' from pg_database where d.classid = 'pg_database'::regclass and oid = d.objid) o
              where not exists (select 1 from aclexplode(o.acl) e where e.grantee = d.refobjid and (e.privilege_type <> o.ok or e.is_grantable)))) as mon_grants,
         current_setting('scram_iterations', true) as iterations, current_setting('ssl') as ssl
       from pg_roles where rolname = current_user`,
    )).rows[0];
    const user = loginName(url, me.name, role);
    // A custom role is the user's own: one that can do more than monitor is refused (its search_path
    // would be changed, and its credentials sent to the platform). Other memberships may carry
    // superuser or file access; owned objects may be the application's tables. Grants beyond the
    // plan's may write data, read password verifiers (pg_authid, or its TOAST) or run file functions.
    // Memberships count through any chain, also through pg_monitor and pg_read_all_stats, which the plan
    // grants: a role it has not made their member yet would inherit what they are members of. A grant in another database
    // cannot be judged from here (its catalog is not readable), so any is refused.
    const custom = role !== DEFAULT_MONITORING_USER;
    const unsafe = me.mon_privileged ? "is a superuser, or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS: a monitoring role must not"
      : me.mon_groups ? `is a member of ${me.mon_groups}: a monitoring role may be a member of ${MONITORING_GROUPS.slice(0, -1).join(", ")} and ${MONITORING_GROUPS.at(-1)} only`
      : me.mon_admin_groups ? `is a member of ${me.mon_admin_groups} WITH ADMIN OPTION: a monitoring role must not have ADMIN OPTION`
      : me.mon_plan_groups ? `would become a member of ${me.mon_plan_groups} through pg_monitor or pg_read_all_stats: a monitoring role may be a member of ${MONITORING_GROUPS.slice(0, -1).join(", ")} and ${MONITORING_GROUPS.at(-1)} only`
      : me.mon_owns ? "owns objects on this server: a monitoring role must own nothing"
      : me.mon_other_dbs ? `has grants in database ${me.mon_other_dbs}: a monitoring role may have grants in this database only`
      : me.mon_grants ? `has grants on ${me.mon_grants}: a monitoring role may have only CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog and pg_toast, and the grants pgai makes`
      : "";
    if (custom && unsafe) return { next: `${role} ${unsafe}. Use another ${opts.agent ? "monitoring_user" : "--monitoring-user"}` };
    const notes = [caNote(url)].filter((n): n is string => !!n);
    const noted = () => (notes.length ? { note: notes.join("; ") } : {});
    // A server that does not check passwords for this client (trust) accepts a random one too.
    const acceptsAnyPassword = async () =>
      logsIn(loginUrlFor(url, user, me.db, (await resolveMonitoringPassword({ monitoringUser: role })).password)).catch(() => false);
    const unchecked = (what: string) => `this server accepts any password from this host, so ${what} was not checked; if no data arrives, delete the instance (pgai mon instances delete) and deploy again with the right password`;
    if (me.name === role) {
      const v = await verifyInitSetup({ client, database: me.db, monitoringUser: me.name, includeOptionalPermissions: false, provider: pgProvider });
      if (v.ok) {
        // A URL without a password (PGPASSWORD, say): the box gets the one this client logged in with.
        const used = (client as { password?: unknown }).password;
        const inUrl = urlPassword(new URL(url));
        const password = inUrl || (typeof used === "string" ? used : "");
        const putInUrl = `Put the password of ${role} in the URL: the monitoring box logs in with it`;
        if (!password) return { next: putInUrl };
        if (await acceptsAnyPassword()) {
          // What pg read from PGPASSWORD may be for another role or server: unchecked, it is not sent on.
          if (!inUrl) return { next: `This server accepts any password from this host, so the one from the environment was not checked and is not used. ${putInUrl}` };
          notes.push(unchecked("the password in the URL"));
        }
        return { monitoringUrl: monitoringUrlFor(url, user, me.db, password), ...noted() };
      }
    } else if (me.admin && pgProvider !== "supabase") {
      // The role is cluster-wide: another database here may use its password,
      // so it is never changed, and only a password that logs in is used.
      const exists = `${role} already exists on this server`;
      const setPassword = custom
        ? `Set PGAI_MON_PASSWORD to the password of ${role}, or give it a new one: PGAI_MON_PASSWORD=<new password> pgai mon deploy <admin-url> --monitoring-user ${role} --reset-password (anything else that logs in as ${role} then needs the new password)`
        : `Set PGAI_MON_PASSWORD to the password of ${role}, or give it a new one: pgai mon deploy <admin-url> --reset-password (anything else that logs in as ${role} then needs the new password)`;
      // The platform fills in the password it keeps for the box (it never sends it here), over TLS only:
      // what the URL lacks for that, and whether the server takes TLS at all (it refused it, or has ssl off).
      const needs = storedPasswordNeeds(monitoringUrlFor(url, user, me.db, ""));
      const noTls = refusedTls || (!usedSsl && me.ssl === "off");
      const tlsOnly = `PostgresAI sends the password it keeps for ${role} only over TLS`;
      const others = opts.others?.length ? opts.others : undefined;
      // Nobody has the password (the first deploy generated it): only a new one, once nothing uses the old one.
      const disconnectOthers = others ? disconnectEach(opts.otherRefs ?? others) : "pgai mon instances delete the server's other databases (pgai mon instances list)";
      const reconnectOthers = `(and deploy ${others ? others.join(", ") : "them"} again with it)`;
      const noTlsWayOut = opts.agent
        ? `Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password. If nobody has it, in a terminal: ${disconnectOthers}, then pgai mon deploy <admin-url> --reset-password with PGAI_MON_PASSWORD set to a new one ${reconnectOthers}`
        : `Set PGAI_MON_PASSWORD to its password, or turn on TLS on the server (ssl = on) and put sslmode=require in the URL. If nobody has the password: ${disconnectOthers}, then re-run with --reset-password and PGAI_MON_PASSWORD set to a new one ${reconnectOthers}`;
      const reuse = me.mon_exists && !opts.resetPassword && !!opts.storedPassword && !(opts.agent ? "" : process.env.PGAI_MON_PASSWORD?.trim());
      if (reuse && needs) {
        if (noTls) return { next: `${exists}, but this server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}` };
        return { next: `${exists}. ${opts.agent ? "database_url" : "The URL"} needs ${needs}: the password PostgresAI keeps for this server is sent only over TLS, to a URL with each parameter once` };
      }
      if (reuse) {
        if (opts.check) return { checked: true };
        // The role is there: the plan grants this database and keeps its password (the one generated here is never set).
        const { password: unused } = await resolveMonitoringPassword({ monitoringUser: role });
        const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: unused, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
        await applyInitPlan({ client, plan });
        return { monitoringUrl: monitoringUrlFor(url, user, me.db, ""), storedPassword: true as const, ...noted() };
      }
      if (me.mon_exists && opts.agent) return { next: `${exists}. Pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set to its password` };
      // A PGAI_MON_PASSWORD that does not log in, where the platform keeps the password: without it, that one
      // is used (over TLS only).
      const unset = !opts.storedPassword ? setPassword
        : noTls ? `This server takes no TLS, and ${tlsOnly}. ${noTlsWayOut}`
        : !needs ? `Unset PGAI_MON_PASSWORD: PostgresAI keeps the password of ${role} for this server`
        : `Unset PGAI_MON_PASSWORD, and the URL needs ${needs}: PostgresAI keeps the password of ${role} for this server, and sends it only over TLS, to a URL with each parameter once`;
      const reset = !!opts.resetPassword && me.mon_exists;
      // PostgresAI keeps no password for a custom role: one it set is the user's (a generated one would be lost).
      if (custom && (!me.mon_exists || reset) && !(opts.agent ? "" : process.env.PGAI_MON_PASSWORD?.trim())) {
        return { next: opts.agent
          ? `${role} is a custom role, and PostgresAI keeps no password for it: pass its URL as database_url, or run pgai mon deploy in a terminal with PGAI_MON_PASSWORD set`
          : `Set PGAI_MON_PASSWORD to a password for ${role}: PostgresAI keeps no password for a custom role, so a generated one would be lost` };
      }
      const { password, generated } = await resolveMonitoringPassword({ passwordEnv: opts.agent ? undefined : process.env.PGAI_MON_PASSWORD, monitoringUser: role });
      if (me.mon_exists && !reset) {
        if (!process.env.PGAI_MON_PASSWORD?.trim()) return { next: `${exists}. ${setPassword}`, ...(custom ? {} : { resettable: true as const }) };
        let accepted: boolean;
        try {
          accepted = await logsIn(loginUrlFor(url, user, me.db, password));
        } catch (err) {
          // pg_hba for this client, say: neither accepted nor rejected. With the platform's password, no login is needed.
          const orUnset = opts.storedPassword && !noTls ? ", or unset PGAI_MON_PASSWORD: PostgresAI keeps its password for this server" : "";
          return { next: `${exists}, and PGAI_MON_PASSWORD could not be checked from this host (${err instanceof Error ? err.message : String(err)}). Run pgai mon deploy from a host that ${role} may connect from${orUnset}` };
        }
        if (!accepted) return { next: `${exists} and PGAI_MON_PASSWORD is not its password. ${unset}` };
        if (await acceptsAnyPassword()) notes.push(unchecked("PGAI_MON_PASSWORD"));
      }
      if (opts.check) return { checked: true };
      const monitoringUrl = monitoringUrlFor(url, user, me.db, password);
      const loginUrl = loginUrlFor(url, user, me.db, password);
      const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: password, iterations: Number(me.iterations), includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: !reset });
      try {
        await applyInitPlan({ client, plan });
      } catch (err) {
        // A later step failed after the role was created with a generated password (it logs in,
        // so the role is this run's): nobody has that password, so the role is dropped again.
        if (generated && !me.mon_exists && (await logsIn(loginUrl).catch(() => false))) await dropMonitoringRole(client, role);
        throw err;
      }
      // Another session may have created the role meanwhile, with its own password.
      // Other login errors (pg_hba for this client, say) do not tell, so they pass.
      if (!(await logsIn(loginUrl).catch(() => true))) return { next: `${role} was created by someone else meanwhile. ${setPassword}` };
      return { monitoringUrl, ...noted(), ...(generated && !me.mon_exists ? { generated: true as const } : {}) };
    }
    const plan = await buildInitPlan({ database: me.db, monitoringUser: role, monitoringPassword: "<password>", includeOptionalPermissions: true, provider: pgProvider, keepExistingPassword: true });
    return {
      sql: renderInitPlanSql(plan),
      next: `Run the SQL as an admin, with a password of your choice in place of <redacted> (or re-run with an admin URL), then pgai mon deploy again with the ${role} URL`,
    };
  } finally {
    await client.end();
  }
}

/** The ClickHouse Cloud organization that runs the service at `host`, found with the key itself. */
export async function clickhouseOrgFor(host: string, keyId: string, keySecret: string) {
  const apiUrl = process.env.CLICKHOUSE_API_URL || "https://api.clickhouse.cloud";
  const response = await fetch(`${new URL(apiUrl).origin}/v1/organizations`, {
    headers: { Authorization: `Basic ${Buffer.from(`${keyId}:${keySecret}`).toString("base64")}` },
    signal: requestTimeoutSignal().signal,
    redirect: "error",
  });
  if (response.status === 401) throw new ClickhouseKeyError("ClickHouse Cloud rejected the API key (401). Check the key id and secret.");
  if (!response.ok) throw new Error(`ClickHouse Cloud API request failed (${response.status}).`);
  const orgs = ((await response.json()) as { result?: { id: string }[] }).result;
  if (!Array.isArray(orgs)) throw new Error("ClickHouse Cloud API returned no organization list.");
  // An organization the key cannot read (403) says more than "no such service" in the next one.
  let notFound: unknown = new ClickhouseKeyError("The API key belongs to no ClickHouse Cloud organization.");
  let failed: unknown;
  for (const org of orgs) {
    try {
      const service = await findService({ apiUrl, orgId: org.id, keyId, keySecret, hostname: host });
      return { orgId: org.id, state: service.state };
    } catch (err) {
      if (/^No ClickHouse Managed Postgres service/.test(errorText(err))) notFound = new ClickhouseKeyError(errorText(err));
      // A key without the reader role (401/403 on the services) is the user's to fix, like a rejected one.
      else failed ??= /\((401|403)\)/.test(errorText(err)) ? new ClickhouseKeyError(errorText(err)) : err;
    }
  }
  throw failed ?? notFound;
}

/** What a delete did to the billing (monitoring_instance_delete's reply), for a person; undefined when nothing was released. */
export function disconnectBilling(reply: unknown): string | undefined {
  const r = (reply ?? {}) as { billing?: { subscription?: string; quantity?: number; refunded?: number; currency?: string; credited?: boolean }; billing_warning?: string };
  // The last cluster: the unused time is refunded to the card; when that refund fails it stays as account credit.
  if (r.billing?.subscription === "canceled" && r.billing.credited && r.billing_warning) return `subscription canceled: no further charges; the refund failed (${r.billing_warning})`;
  if (r.billing_warning) return `not released (${r.billing_warning}): contact support`;
  if (r.billing?.subscription === "canceled" && typeof r.billing.refunded === "number" && r.billing.refunded > 0) {
    return `subscription canceled: no further charges; refunded ${money(r.billing.refunded, r.billing.currency ?? "usd")} to your card`;
  }
  if (r.billing?.subscription === "canceled") return "subscription canceled: no further charges; the unused part of this period is credited (prorated)";
  if (r.billing?.subscription === "active" && typeof r.billing.quantity === "number") {
    return `${r.billing.quantity} ${r.billing.quantity === 1 ? "database cluster" : "database clusters"} left on the subscription`;
  }
  return undefined;
}

/** The project's Health Matrix: <console>/<org alias>/projects/<project alias>/health. */
export function healthMatrixUrl(uiBaseUrl: string, orgAlias: string, projectAlias: string): string {
  return `${uiBaseUrl.replace(/\/+$/, "")}/${encodeURIComponent(orgAlias)}/projects/${encodeURIComponent(projectAlias)}/health`;
}

/** The box's Grafana through its PostgresAI sign-in (generic OAuth), landing where the address points. */
export function grafanaOAuthUrl(dashboardUrl: string | null): string | null {
  let u: URL;
  try {
    u = new URL(dashboardUrl ?? "");
  } catch {
    return dashboardUrl;
  }
  if (u.pathname === "/login/generic_oauth") return dashboardUrl;
  return `${u.origin}/login/generic_oauth?redirectTo=${encodeURIComponent(`${u.pathname}${u.search}`)}`;
}

/** The platform side of mon deploy, for the CLI and the MCP server alike. */
export function platformDeps(p: { apiKey: string; apiBaseUrl: string; uiBaseUrl: string; orgScope?: OrgScope; debug?: boolean; agent?: boolean }) {
  const rpc = <T>(fn: string, body: Record<string, unknown> = {}, timeoutMs?: number) =>
    callRpc<T>({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn, body, operation: fn.replace(/_/g, " "), debug: p.debug, orgScope: p.orgScope,
      ...(timeoutMs ? { timeoutMs, timeoutCeilingMs: timeoutMs } : {}) });
  // The console links of a row: the org alias once, the projects again only while one is not found.
  let orgAlias: Promise<string | undefined> | undefined;
  let projects: Promise<ProjectListItem[]> | undefined;
  const healthUrl = async (row: Database): Promise<string | undefined> => {
    orgAlias ??= p.orgScope?.alias ? Promise.resolve(p.orgScope.alias) : listOrgs({ apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl })
      .then((orgs) => (orgs.length === 1 ? orgs[0] : orgs.find((o) => o.org_id === p.orgScope?.id))?.alias);
    projects ??= rpc<ProjectListItem[]>("projects_list");
    const [org, list] = await Promise.all([orgAlias, projects]);
    const project = list.find((x) => x.monitoring_instance_ids?.includes(row.id)) ?? list.find((x) => x.name === row.name);
    if (!project) projects = undefined;
    return org && project?.alias ? healthMatrixUrl(p.uiBaseUrl, org, project.alias) : undefined;
  };
  const withLinks = async (row: Database): Promise<Database> => {
    // A link that cannot be looked up is left out: it never fails the list.
    const health_url = await healthUrl(row).catch(() => {
      orgAlias = projects = undefined;
      return undefined;
    });
    // Ordered for a person reading it: the Health Matrix, then Grafana.
    const { dashboard_url, host_metrics, host_metrics_source, ...rest } = row;
    return { ...rest, ...(health_url ? { health_url } : {}), dashboard_url: grafanaOAuthUrl(dashboard_url), host_metrics,
      ...(host_metrics_source !== undefined ? { host_metrics_source } : {}) } as Database;
  };
  return {
    verifyTls: verifyCollectorTls,
    list: () => rpc<Database[]>("cloud_monitoring_list"),
    links: withLinks,
    create: async (body: Record<string, string | boolean | number>) => {
      // platform-all#987: a platform before the allow_ip step can't take its arguments. Without
      // --ip-first / --ip-allowed the deploy goes on as there (an unreachable database is refused);
      // with them it stops, since what was asked can't be done.
      const { allow_ip_fallback: _fallback, ip_first, ip_allowed, ...older } = body;
      const asked = !!(ip_first || ip_allowed);
      const connectMs = p.agent ? AGENT_CONNECT_TIMEOUT_MS : CONNECT_TIMEOUT_MS;
      const connectTimeout = { timeoutMs: connectMs, timeoutCeilingMs: connectMs };
      try {
        return await callRpc<{ id: string; name: string; status: string; error?: string }>({
          apiKey: p.apiKey, apiBaseUrl: p.apiBaseUrl, fn: "cloud_monitoring_connect", body,
          operation: "cloud monitoring connect", debug: p.debug, orgScope: p.orgScope,
          ...connectTimeout,
          ...(asked ? {} : { fallbackBody: older }),
        });
      } catch (err) {
        if (asked && err instanceof HttpStatusError && err.status === 404 && /cloud_monitoring_connect/.test(err.message)) {
          throw new Error("This PostgresAI platform does not support --ip-first or --ip-allowed yet: deploy without them");
        }
        throw err;
      }
    },
    disconnect: (id: string) => rpc("cloud_monitoring_disconnect", { instance_id: id }),
    setVcpus: (id: string, vcpus: number) => rpc<{ vcpus: number; previous_vcpus: number | null; changed: boolean; aas_status: string }>("cloud_monitoring_set_vcpus", { instance_id: id, vcpus }),
    resetLock: (server: string) => rpc<{ lock_id: string }>("cloud_monitoring_reset_lock", { server }),
    resetUnlock: (lockId: string) => rpc("cloud_monitoring_reset_unlock", { lock_id: lockId }),
    prepare: (url: string, provider: Provider, o: { resetPassword?: boolean; check?: boolean; storedPassword?: boolean; others?: string[]; otherRefs?: string[]; monitoringUser?: string; warn?: Warn } = {}) => prepareDatabase(url, provider, { ...o, agent: p.agent }),
    unprepare: (url: string, monitoringUser?: string, o: { warn?: Warn } = {}) => unprepareDatabase(url, { ...o, agent: p.agent, monitoringUser }),
    clickhouseOrg: clickhouseOrgFor,
    quote: (coupon?: string, cluster?: string) => rpc<Quote>("cloud_monitoring_quote", { ...(coupon ? { promo_code: coupon } : {}), ...(cluster ? { db_server: cluster } : {}) }),
    billingUrl: (orgAlias: string) => `${p.uiBaseUrl}/${orgAlias}/billing`,
    supabaseStart: (dbUrl: string) => rpc<{ connected: true; project_ref: string } | { auth_url: string; session_id: string; expires_at?: string }>("cloud_monitoring_supabase_start", { db_url: dbUrl }),
    supabaseStatus: (sessionId: string) => rpc<{ status: "pending" | "connected" | "error"; error?: string }>("cloud_monitoring_supabase_status", { session_id: sessionId }),
    supabaseConnect: (body: Record<string, string | boolean | number>) => rpc<{ id: string; name: string; status: string; error?: string }>("cloud_monitoring_supabase_connect", body),
    supabaseProjects: () => rpc<SupabaseProjects>("cloud_monitoring_supabase_projects", {}),
    deployOptions: (args: Parameters<NonNullable<ConnectDeps["deployOptions"]>>[0]) => rpc<DeployOptions>("monitoring_deploy_options", args),
    rdsExternalId: async () => (await rpc<{ external_id: string }>("cloud_monitoring_rds_external_id")).external_id,
    rdsPrepareSql: (dbName: string) => rpc<RdsPrepareSql>("cloud_monitoring_rds_prepare_sql", { db_name: dbName }),
    rdsProvision: (body: Record<string, string | boolean | number>) =>
      rpc<{ id: string; name: string; provider?: string; status: string; region?: string; verified_status?: string; launch_status?: string }>("cloud_monitoring_rds_provision", body, p.agent ? AGENT_CONNECT_TIMEOUT_MS : CONNECT_TIMEOUT_MS),
    rdsStatus: (id: string) => rpc<RdsStatus>("cloud_monitoring_rds_status", { instance_id: id }),
    aws: awsRunner,
    awsProfiles: () => new Promise<string[]>((resolve) => execFile("aws", ["configure", "list-profiles"], { timeout: 10_000 },
      (err, stdout) => resolve(err ? [] : stdout.split("\n").map((l) => l.trim()).filter(Boolean)))),
    publicAddresses: (host: string) => publicAddresses(host),
    runSql: async (url: string, sql: string, warn?: Warn) => {
      const { client } = await openConnection(url, { agent: p.agent, warn });
      try {
        await client.query(sql);
      } finally {
        await client.end().catch(() => {});
      }
    },
    readNonce: async (url: string, warn?: Warn) => {
      const { client } = await openConnection(url, { agent: p.agent, warn });
      try {
        const r = await client.query("select nonce from postgres_ai.monitoring_ownership order by issued_at desc limit 1");
        return r.rows[0]?.nonce as string | undefined;
      } catch (err) {
        // No schema or table yet: the SQL did not run in this database.
        if (["3F000", "42P01"].includes((err as { code?: string }).code ?? "")) return undefined;
        throw err;
      } finally {
        await client.end().catch(() => {});
      }
    },
    terraformModule: async () => {
      const r = await fetch(TERRAFORM_URL, { signal: AbortSignal.timeout(30_000) });
      if (!r.ok) throw new Error(`Could not download the Terraform module (${TERRAFORM_URL}: HTTP ${r.status})`);
      return r.text();
    },
    writeTerraform: async (dirName: string, files: { mainTf: string; tfvars: string }) => {
      const dir = path.resolve(dirName);
      await fs.promises.mkdir(dir, { recursive: true });
      // A terraform.tfvars the customer filled in already is kept.
      await fs.promises.writeFile(path.join(dir, "main.tf"), files.mainTf);
      await fs.promises.writeFile(path.join(dir, "terraform.tfvars"), files.tfvars, { flag: "wx" }).catch((err) => {
        if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
      });
      return dir;
    },
    sleep: (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
    now: () => Date.now(),
  };
}
