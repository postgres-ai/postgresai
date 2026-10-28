import { execFile } from "node:child_process";

// AWS PrivateLink for a private RDS (postgresai#454), the pieces mon deploy needs on this
// machine: what the RDS host names, whether it is public, the customer's stack (read-only,
// through the aws CLI), and the CloudFormation link or Terraform the console offers too.

/** The templates the console links (platform-all cloudformation/ and terraform/, published to S3). */
export const CFN_TEMPLATE_URL = "https://postgres-ai-cloudformation.s3.amazonaws.com/rds-privatelink-provider.yaml";
export const TERRAFORM_URL = "https://postgres-ai-cloudformation.s3.amazonaws.com/terraform/rds-privatelink-provider/main.tf";
/** PostgresAI's principal the endpoint service allows (the templates' default). */
export const PRINCIPAL_ARN = "arn:aws:iam::005923036815:user/pgai-rds-privatelink-monitoring";

/** What an RDS host name says: the instance (or cluster), its region, and which kind of endpoint it is. */
export interface RdsEndpoint {
  host: string;
  id: string;
  region: string;
  kind: "instance" | "aurora_writer" | "aurora_reader" | "aurora_custom" | "proxy";
  /** aws, or a partition PrivateLink to PostgresAI can't reach (China, GovCloud). */
  partition: "aws" | "aws-cn" | "aws-us-gov";
}

/** <id>.<hash>.<region>.rds.amazonaws.com[.cn], and Aurora's cluster-, cluster-ro-, cluster-custom- and proxy- forms. */
export function rdsEndpoint(hostname: string): RdsEndpoint | undefined {
  const host = hostname.toLowerCase().replace(/\.$/, "");
  const m = /^([a-z][a-z0-9-]*)\.([a-z0-9-]+)\.([a-z]{2}(?:-[a-z]+)+-\d)\.rds\.amazonaws\.com(\.cn)?$/.exec(host);
  if (!m) return undefined;
  const [, id, second, region, cn] = m;
  const kind = second.startsWith("cluster-ro-") ? "aurora_reader"
    : second.startsWith("cluster-custom-") ? "aurora_custom"
    : second.startsWith("cluster-") ? "aurora_writer"
    : second.startsWith("proxy-") ? "proxy" : "instance";
  const partition = cn ? "aws-cn" : region.startsWith("us-gov-") ? "aws-us-gov" : "aws";
  return { host, id, region, kind, partition };
}

/** The stack name mon deploy's link sets: one stack per RDS instance. */
export const stackNameFor = (rdsId: string): string => `postgresai-privatelink-${rdsId}`.slice(0, 128);

/** com.amazonaws.vpce.<region>.vpce-svc-<id>, as the stack's VpceServiceName output. */
export const VPCE_SERVICE = /^com\.amazonaws\.vpce\.([a-z]{2}(?:-[a-z]+)+-\d)\.vpce-svc-[0-9a-f]{8,}$/;

/** arn:aws:iam::<account>:role/<name>, as the stack's HostStatsRoleArn output. */
export const ROLE_ARN = /^arn:aws:iam::\d{12}:role\/[\w+=,.@\/-]{1,512}$/;

/** RFC 1918, carrier-grade NAT, link-local and IPv6 unique-local addresses: not reachable from the internet. */
export function isPrivateAddress(ip: string): boolean {
  const v4 = /^(\d+)\.(\d+)\.\d+\.\d+$/.exec(ip);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127) || (a === 169 && b === 254);
  }
  return /^f[cd][0-9a-f]{2}:/i.test(ip) || /^fe80:/i.test(ip);
}

type FetchLike = (url: string, init?: { headers?: Record<string, string>; signal?: AbortSignal }) => Promise<{ ok: boolean; json(): Promise<unknown>; text(): Promise<string> }>;

/** Public DNS resolvers over HTTPS: inside the VPC, AWS resolves even a public RDS to its private address. */
const DOH = ["https://cloudflare-dns.com/dns-query", "https://dns.google/resolve"];
// Short: where outbound traffic is blocked, both give up within seconds.
const DOH_TIMEOUT_MS = 3_000;

/** The host's IPv4 addresses as the internet sees them; undefined when no public resolver answered. */
export async function publicAddresses(host: string, fetchFn: FetchLike = fetch as unknown as FetchLike): Promise<string[] | undefined> {
  for (const base of DOH) {
    try {
      const r = await fetchFn(`${base}?name=${encodeURIComponent(host)}&type=A`, { headers: { accept: "application/dns-json" }, signal: AbortSignal.timeout(DOH_TIMEOUT_MS) });
      if (!r.ok) continue;
      const body = (await r.json()) as { Status?: number; Answer?: { type: number; data: string }[] };
      if (body.Status !== 0) continue;
      return (body.Answer ?? []).filter((a) => a.type === 1).map((a) => a.data);
    } catch {
      // The next resolver; none answering means "not known".
    }
  }
  return undefined;
}

/** Why an aws CLI call failed, as mon deploy acts on it. */
export class AwsError extends Error {
  constructor(message: string, readonly kind: "missing" | "credentials" | "not_found" | "denied" | "other") {
    super(message);
  }
}

/** Runs `aws <args>` with JSON output and returns the parsed reply. */
export type AwsRun = (args: string[]) => Promise<unknown>;

/** The aws CLI with the standard profile chain: --aws-profile, else AWS_PROFILE, default, the instance role. Read-only calls only. */
export function awsRunner(profile?: string): AwsRun {
  return (args) => new Promise((resolve, reject) => {
    execFile("aws", [...args, "--output", "json", ...(profile ? ["--profile", profile] : [])],
      { timeout: 30_000, maxBuffer: 16 * 1024 * 1024, env: { ...process.env, AWS_PAGER: "" } }, (err, stdout, stderr) => {
        if (!err) {
          try {
            return resolve(stdout.trim() ? JSON.parse(stdout) : null);
          } catch {
            return reject(new AwsError("the aws CLI returned no JSON", "other"));
          }
        }
        if ((err as NodeJS.ErrnoException).code === "ENOENT") return reject(new AwsError("the aws CLI is not installed", "missing"));
        const said = String(stderr || err.message).trim().split("\n").filter(Boolean).pop() ?? "";
        const kind = /DBInstanceNotFound|does not exist|NotFound/i.test(said) ? "not_found"
          : /expired|Unable to locate credentials|NoCredentialProviders|aws (sso )?login|profile .* could not be found|InvalidClientTokenId|SignatureDoesNotMatch/i.test(said) ? "credentials"
          : /AccessDenied|not authorized|UnauthorizedOperation/i.test(said) ? "denied" : "other";
        reject(new AwsError(said.replace(/^aws: \[ERROR\]: /, ""), kind));
      });
  });
}

/** What mon deploy reads of the RDS instance. */
export interface DbInstance {
  publiclyAccessible: boolean;
  port: number;
  vpcId?: string;
  /** The DB subnet group's subnets with their availability zones. */
  subnets: { id: string; az: string }[];
  securityGroupIds: string[];
  instanceClass?: string;
  /** postgres, or aurora-postgresql for an Aurora cluster's member. */
  engine?: string;
}

export async function describeDbInstance(run: AwsRun, id: string, region: string): Promise<DbInstance> {
  const d = (await run(["rds", "describe-db-instances", "--db-instance-identifier", id, "--region", region, "--query", "DBInstances[0]"])) as {
    PubliclyAccessible?: boolean; Endpoint?: { Port?: number }; DBInstanceClass?: string; Engine?: string;
    DBSubnetGroup?: { VpcId?: string; Subnets?: { SubnetIdentifier: string; SubnetAvailabilityZone?: { Name?: string } }[] };
    VpcSecurityGroups?: { VpcSecurityGroupId: string; Status?: string }[];
  } | null;
  if (!d) throw new AwsError(`DB instance ${id} not found`, "not_found");
  return {
    publiclyAccessible: !!d.PubliclyAccessible,
    port: d.Endpoint?.Port ?? 5432,
    vpcId: d.DBSubnetGroup?.VpcId,
    subnets: (d.DBSubnetGroup?.Subnets ?? []).map((s) => ({ id: s.SubnetIdentifier, az: s.SubnetAvailabilityZone?.Name ?? "" })),
    securityGroupIds: (d.VpcSecurityGroups ?? []).filter((g) => g.Status !== "inactive").map((g) => g.VpcSecurityGroupId),
    instanceClass: d.DBInstanceClass,
    engine: d.Engine,
  };
}

/** The account and identity the profile signs in as (announced before anything is read). */
export async function callerIdentity(run: AwsRun): Promise<{ account: string; arn: string }> {
  const r = (await run(["sts", "get-caller-identity"])) as { Account: string; Arn: string };
  return { account: r.Account, arn: r.Arn };
}

/** The vCPUs of a DB instance class (db.r6g.large → r6g.large); undefined for one EC2 does not list (serverless). */
export async function vcpusOfClass(run: AwsRun, instanceClass: string, region: string): Promise<number | undefined> {
  if (!instanceClass.startsWith("db.")) return undefined;
  try {
    const n = await run(["ec2", "describe-instance-types", "--instance-types", instanceClass.slice(3), "--region", region, "--query", "InstanceTypes[0].VCpuInfo.DefaultVCpus"]);
    return typeof n === "number" && n > 0 ? n : undefined;
  } catch {
    return undefined;
  }
}

/** The NLB takes one subnet per availability zone: the DB subnet group's first in each. */
export function oneSubnetPerAz(subnets: DbInstance["subnets"]): string[] {
  const seen = new Map<string, string>();
  for (const s of [...subnets].sort((a, b) => a.az.localeCompare(b.az) || a.id.localeCompare(b.id))) if (!seen.has(s.az)) seen.set(s.az, s.id);
  return [...seen.values()];
}

/** A CloudFormation stack as mon deploy follows it. */
export interface Stack {
  name: string;
  status: string;
  reason?: string;
  outputs: Record<string, string>;
  rdsId?: string;
  /** When it was created (ISO), to tell stacks apart. */
  created?: string;
}

type RawStack = { StackName: string; StackStatus: string; StackStatusReason?: string; CreationTime?: string; Outputs?: { OutputKey: string; OutputValue: string }[]; Parameters?: { ParameterKey: string; ParameterValue: string }[] };

const toStack = (s: RawStack): Stack => ({
  name: s.StackName, status: s.StackStatus, ...(s.StackStatusReason ? { reason: s.StackStatusReason } : {}),
  ...(s.CreationTime ? { created: s.CreationTime } : {}),
  outputs: Object.fromEntries((s.Outputs ?? []).map((o) => [o.OutputKey, o.OutputValue])),
  rdsId: s.Parameters?.find((p) => p.ParameterKey === "RdsInstanceIdentifier")?.ParameterValue,
});

/**
 * The stacks that publish this RDS, by their RdsInstanceIdentifier parameter, under any name (the
 * console's link names its own): done ones (with a VpceServiceName), ones being created, and failed
 * ones (said so). A stack being deleted is none.
 * `misnamed`: a stack with mon deploy's name for another RDS, which is never used.
 */
export async function findStacks(run: AwsRun, rdsId: string, region: string): Promise<{ stacks: Stack[]; misnamed?: Stack }> {
  const all = ((await run(["cloudformation", "describe-stacks", "--region", region])) as { Stacks?: RawStack[] }).Stacks ?? [];
  const live = all.map(toStack).filter((s) => stackState(s.status) !== "deleting");
  // RDS lowercases instance identifiers; one typed in the console form may not be.
  const forThis = (s: Stack) => s.rdsId?.toLowerCase() === rdsId;
  const named = live.find((s) => s.name === stackNameFor(rdsId));
  // Failed ones too: deploy says why when the stack being made fails, under any name.
  const stacks = live.filter((s) => forThis(s) && (s === named || "VpceServiceName" in s.outputs || stackState(s.status) !== "ready"));
  return { stacks, ...(named && !forThis(named) ? { misnamed: named } : {}) };
}

/** Why a stack failed: its first failed resource and the reason CloudFormation gives. */
export async function stackFailure(run: AwsRun, name: string, region: string): Promise<string | undefined> {
  try {
    const r = (await run(["cloudformation", "describe-stack-events", "--stack-name", name, "--region", region])) as
      { StackEvents?: { LogicalResourceId: string; ResourceStatus: string; ResourceStatusReason?: string }[] };
    const failed = (r.StackEvents ?? []).filter((e) => /_FAILED$/.test(e.ResourceStatus) && e.ResourceStatusReason && !/cancelled/i.test(e.ResourceStatusReason));
    // Events come newest first: the first failure is the last one listed.
    const first = failed[failed.length - 1];
    return first ? `${first.LogicalResourceId}: ${first.ResourceStatusReason}` : undefined;
  } catch {
    return undefined;
  }
}

/** A stack that is done (its outputs are there), being made, being deleted, or a failed one to repair or delete first. */
export function stackState(status: string): "ready" | "in_progress" | "deleting" | "failed" {
  if (status === "DELETE_IN_PROGRESS" || status === "DELETE_COMPLETE") return "deleting";
  // A failed create rolling back: wait for its end, then say why it failed.
  if (status === "ROLLBACK_IN_PROGRESS") return "in_progress";
  // An update rolled back, or resources imported: the stack and its outputs are there.
  if (/^(CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE|IMPORT_COMPLETE|IMPORT_ROLLBACK_COMPLETE)$/.test(status)) return "ready";
  // Being made, or an update (or its rollback) under way: wait.
  if (/^(CREATE|UPDATE|UPDATE_ROLLBACK|IMPORT|IMPORT_ROLLBACK|REVIEW)_IN_PROGRESS$|^UPDATE(_ROLLBACK)?_COMPLETE_CLEANUP_IN_PROGRESS$/.test(status)) return "in_progress";
  return "failed";
}

/** What the customer's stack takes, as far as this machine knows it. */
export interface StackInputs {
  region: string;
  rdsId: string;
  port: number;
  vpcId?: string;
  subnetIds?: string[];
  securityGroupId?: string;
  /** Host metrics (--host-stats): the org's External ID, which only PostgresAI's collector role can use. */
  externalId?: string;
}

/** The console's quick-create link, with every parameter known here filled in. */
export function quickCreateUrl(i: StackInputs): string {
  const params = new URLSearchParams({ templateURL: CFN_TEMPLATE_URL, stackName: stackNameFor(i.rdsId) });
  params.set("param_RdsInstanceIdentifier", i.rdsId);
  params.set("param_RdsPort", String(i.port));
  if (i.vpcId) params.set("param_VpcId", i.vpcId);
  if (i.subnetIds?.length) params.set("param_SubnetIds", i.subnetIds.join(","));
  if (i.securityGroupId) params.set("param_RdsSecurityGroupId", i.securityGroupId);
  params.set("param_PostgresAiAccountArn", PRINCIPAL_ARN);
  if (i.externalId) params.set("param_PostgresAiExternalId", i.externalId);
  return `https://${i.region}.console.aws.amazon.com/cloudformation/home?region=${encodeURIComponent(i.region)}#/stacks/quickcreate?${params.toString()}`;
}

/**
 * The Terraform module as published, and a terraform.tfvars with every value known here (the
 * region, the External ID, the RDS's own): the module's text is never edited. Absent ones stay
 * for the customer to fill.
 */
export function terraformFiles(mainTf: string, i: StackInputs): { mainTf: string; tfvars: string } {
  const q = JSON.stringify;
  const tfvars = [
    `aws_region              = ${q(i.region)}`,
    ...(i.externalId ? [`postgres_ai_external_id = ${q(i.externalId)}`] : []),
    `rds_instance_identifier = ${q(i.rdsId)}`,
    `rds_port                = ${i.port}`,
    i.vpcId ? `vpc_id                  = ${q(i.vpcId)}` : `# vpc_id                = "vpc-..."  (the RDS's VPC)`,
    i.subnetIds?.length ? `subnet_ids              = [${i.subnetIds.map((s) => q(s)).join(", ")}]` : `# subnet_ids            = ["subnet-...", "subnet-..."]  (one per availability zone)`,
    i.securityGroupId ? `rds_security_group_id   = ${q(i.securityGroupId)}` : `# rds_security_group_id = "sg-..."  (the RDS's security group)`,
  ].join("\n");
  return { mainTf, tfvars: `${tfvars}\n` };
}
