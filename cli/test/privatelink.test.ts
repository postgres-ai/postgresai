import { describe, expect, test } from "bun:test";
import { connect, unreachable, stepProgress, type StepViewLike, overPrivateLink, privateLinkFailure, privateLinkOf, stackTeardown, verifiedWords, type ConnectDeps, type Database, type ProgressEvent, type RdsPrepareSql } from "../lib/connect";
import { canOpenBrowser, HttpStatusError, redactSecretsForLog } from "../lib/util";
import { PRIVATELINK_STEPS } from "../lib/deploy";
import {
  AwsError, findStacks, isPrivateAddress, oneSubnetPerAz, publicAddresses, quickCreateUrl, rdsEndpoint, stackState, terraformFiles, type AwsRun,
} from "../lib/privatelink";

// pgai mon deploy for a private RDS over AWS PrivateLink (postgresai#454).

const HOST = "denis-cpl-rds.c5q8mmwa08s8.us-east-2.rds.amazonaws.com";
const URL_ = `postgresql://pgai_admin:pw@${HOST}:5432/app`;
const NAME = `${HOST}/app`;
const VPCE = "com.amazonaws.vpce.us-east-2.vpce-svc-0123456789abcdef0";
const ROLE = "arn:aws:iam::123456789012:role/postgresai-privatelink-HostStatsRole-1";
const FREE_QUOTE = {
  plan: "scale", org_alias: "acme", billed: false, free_slots: { remaining: 3, total: 5 }, subscription: false, quantity: 0,
  price: { amount: 51200, currency: "usd", interval: "month" }, has_payment_method: false, requires_payment_method: false,
};
const PREP: RdsPrepareSql = { nonce: "n1", db_user: "postgres_ai_mon", db_name: "app", db_password: "Secr3t", prepare_db_sql: "begin; select 1; commit;" };

type Stack = { StackName: string; StackStatus: string; Outputs?: { OutputKey: string; OutputValue: string }[]; Parameters?: { ParameterKey: string; ParameterValue: string }[] };

/** The aws CLI as AWS answers it: the instance, its stacks (one answer per read), and refusals. */
function fakeAws(o: { publicly?: boolean; stacks?: (Stack | undefined)[]; refuse?: AwsError; sgs?: string[]; calls: string[]; reads?: number; engine?: string }): AwsRun {
  return async (args) => {
    o.calls.push(`aws ${args.slice(0, 2).join(" ")}`);
    if (o.refuse) throw o.refuse;
    const [svc, op] = args;
    if (svc === "sts") return { Account: "123456789012", Arn: "arn:aws:iam::123456789012:user/me" };
    if (svc === "rds") return {
      PubliclyAccessible: !!o.publicly, Endpoint: { Port: 5432 }, DBInstanceClass: "db.t4g.micro", Engine: o.engine ?? "postgres",
      DBSubnetGroup: { VpcId: "vpc-1", Subnets: [
        { SubnetIdentifier: "subnet-b", SubnetAvailabilityZone: { Name: "us-east-2b" } },
        { SubnetIdentifier: "subnet-a", SubnetAvailabilityZone: { Name: "us-east-2a" } },
        { SubnetIdentifier: "subnet-a2", SubnetAvailabilityZone: { Name: "us-east-2a" } }] },
      VpcSecurityGroups: (o.sgs ?? ["sg-1"]).map((id) => ({ VpcSecurityGroupId: id, Status: "active" })),
    };
    if (svc === "ec2") return 2;
    if (svc === "cloudformation" && op === "describe-stack-events") {
      return { StackEvents: [{ LogicalResourceId: "Nlb", ResourceStatus: "CREATE_FAILED", ResourceStatusReason: "subnets in one AZ" }] };
    }
    if (svc === "cloudformation") {
      // The region's stacks, read after read (across runs): as AWS has them by then.
      o.reads = (o.reads ?? 0) + 1;
      const stack = o.stacks?.[Math.min(o.reads - 1, (o.stacks?.length ?? 1) - 1)];
      return { Stacks: stack ? [stack] : [] };
    }
    throw new Error(`unexpected aws ${args.join(" ")}`);
  };
}

const stack = (status: string, outputs: Record<string, string> = { VpceServiceName: VPCE }): Stack => ({
  StackName: "postgresai-privatelink-denis-cpl-rds", StackStatus: status,
  Outputs: Object.entries(outputs).map(([OutputKey, OutputValue]) => ({ OutputKey, OutputValue })),
  Parameters: [{ ParameterKey: "RdsInstanceIdentifier", ParameterValue: "denis-cpl-rds" }],
});

const box = (status: string, extra: Partial<Database> = {}): Database => ({
  id: "b-1", name: "denis-cpl-rds", provider: "rds", status, cluster: `${HOST}:5432`, database: "app",
  dashboard_url: status === "active" ? "https://x.pgai.watch" : null, host_metrics: false, ...extra,
});

function fake(o: Omit<Partial<ConnectDeps>, "aws"> & { rows?: Database[][]; aws?: Omit<Parameters<typeof fakeAws>[0], "calls"> & { calls?: string[] } | false; answers?: string[]; nonce?: string } = {}) {
  const { rows: _r, aws: _a, answers: _x, nonce: _n, ...over } = o;
  const calls: string[] = [];
  const events: ProgressEvent[] = [];
  const asked: string[] = [];
  const rows = o.rows ?? [[]];
  let listed = 0;
  const answers = [...(o.answers ?? [])];
  const awsOpts = o.aws === false ? undefined : { ...(o.aws ?? { stacks: [stack("CREATE_COMPLETE")] }), calls };
  let t = 0;
  const deps: ConnectDeps = {
    list: async () => { calls.push("list"); return rows[Math.min(listed++, rows.length - 1)]; },
    create: async (body) => { calls.push(`create ${JSON.stringify(body)}`); return { id: "d-1", name: NAME, status: "launch_requested" }; },
    prepare: async (_u, provider, o) => { calls.push(`${o?.check ? "check" : "prepare"} ${provider}`); return o?.check ? { checked: true } : { monitoringUrl: "postgresql://postgres_ai_mon@x/app" }; },
    unprepare: async () => true,
    localStackRunning: () => false,
    clickhouseOrg: async () => { throw new Error("no"); },
    selfHosted: async () => {},
    supabaseStart: async () => { throw new Error("no"); },
    supabaseStatus: async () => { throw new Error("no"); },
    supabaseConnect: async () => { throw new Error("no"); },
    quote: async (coupon, cluster) => { calls.push(`quote ${cluster ?? "-"}`); return FREE_QUOTE; },
    billingUrl: (a) => `https://console.postgres.ai/${a}/billing`,
    confirm: async (q) => { asked.push(q); return answers.length ? /^y/i.test(answers.shift()!) : false; },
    resetLock: async () => ({ lock_id: "l" }),
    resetUnlock: async () => {},
    sleep: async (ms) => { t += ms; },
    now: () => t,
    progress: (e) => events.push(e),
    rdsExternalId: async () => { calls.push("externalId"); return "ext-123"; },
    rdsPrepareSql: async (db) => { calls.push(`prepareSql ${db}`); return PREP; },
    runSql: async (url) => { calls.push(`runSql ${new URL(url).username}`); },
    readNonce: async (url) => { calls.push(`readNonce ${new URL(url).username}:${decodeURIComponent(new URL(url).password)}`); return o.nonce ?? "n1"; },
    rdsProvision: async (body) => { calls.push(`provision ${JSON.stringify(body)}`); return { id: "b-1", name: "denis-cpl-rds", status: "launch_requested", launch_status: "launched" }; },
    rdsStatus: async (id) => { calls.push(`status ${id}`); throw Object.assign(new Error("Not Found"), { status: 404 }); },
    publicAddresses: async () => undefined,
    ...(awsOpts ? { aws: () => fakeAws(awsOpts) } : {}),
    awsProfiles: async () => ["default", "personal"],
    ...over,
  };
  return { deps, calls, events, asked, answers };
}

/** A person at a terminal: confirm and ask answer in order. */
const terminal = (f: ReturnType<typeof fake>, lines: string[] = []) => {
  f.deps.ask = async (q, shown) => { f.asked.push(q); lines.push(...(shown ?? [])); return f.answers.shift() ?? ""; };
  return f;
};

const provisioned = (calls: string[]) => calls.filter((c) => c.startsWith("provision")).map((c) => JSON.parse(c.slice("provision ".length)));

describe("host names and helpers", () => {
  test("an RDS host: instance, region, Aurora and proxy forms, other partitions", () => {
    expect(rdsEndpoint(HOST)).toEqual({ host: HOST, id: "denis-cpl-rds", region: "us-east-2", kind: "instance", partition: "aws" });
    expect(rdsEndpoint("c1.cluster-abc.eu-west-1.rds.amazonaws.com")?.kind).toBe("aurora_writer");
    expect(rdsEndpoint("c1.cluster-ro-abc.eu-west-1.rds.amazonaws.com")?.kind).toBe("aurora_reader");
    expect(rdsEndpoint("c1.cluster-custom-abc.eu-west-1.rds.amazonaws.com")?.kind).toBe("aurora_custom");
    expect(rdsEndpoint("p1.proxy-abc.eu-west-1.rds.amazonaws.com")?.kind).toBe("proxy");
    expect(rdsEndpoint("db.abc.cn-north-1.rds.amazonaws.com.cn")?.partition).toBe("aws-cn");
    expect(rdsEndpoint("db.abc.us-gov-west-1.rds.amazonaws.com")?.partition).toBe("aws-us-gov");
    expect(rdsEndpoint("db.example.com")).toBeUndefined();
  });

  test("private addresses", () => {
    expect(["10.1.2.3", "172.16.0.1", "172.31.255.1", "192.168.1.1", "100.64.0.1", "fd00::1"].every(isPrivateAddress)).toBe(true);
    expect(["3.15.1.1", "172.32.0.1", "8.8.8.8", "2600::1"].some(isPrivateAddress)).toBe(false);
  });

  test("public DNS: A records, the next resolver on a failure, undefined when none answers", async () => {
    const seen: string[] = [];
    const answer = (body: unknown, ok = true) => ({ ok, json: async () => body, text: async () => "" });
    const f = async (url: string) => {
      seen.push(new URL(url).host);
      if (url.includes("cloudflare")) throw new Error("blocked");
      return answer({ Status: 0, Answer: [{ type: 5, data: "x." }, { type: 1, data: "172.31.1.2" }] });
    };
    expect(await publicAddresses(HOST, f)).toEqual(["172.31.1.2"]);
    expect(seen).toEqual(["cloudflare-dns.com", "dns.google"]);
    expect(await publicAddresses(HOST, async () => answer({ Status: 3 }))).toBeUndefined();
  });

  test("the quick-create link carries what is known; one subnet per AZ; stack states", () => {
    const u = new URL(quickCreateUrl({ region: "us-east-2", rdsId: "denis-cpl-rds", port: 5432, vpcId: "vpc-1", subnetIds: ["subnet-a", "subnet-b"], securityGroupId: "sg-1", externalId: "ext" }));
    expect(u.host).toBe("us-east-2.console.aws.amazon.com");
    const q = new URLSearchParams(u.hash.split("?")[1]);
    expect(Object.fromEntries(q)).toEqual({
      templateURL: "https://postgres-ai-cloudformation.s3.amazonaws.com/rds-privatelink-provider.yaml", stackName: "postgresai-privatelink-denis-cpl-rds",
      param_RdsInstanceIdentifier: "denis-cpl-rds", param_RdsPort: "5432", param_VpcId: "vpc-1", param_SubnetIds: "subnet-a,subnet-b",
      param_RdsSecurityGroupId: "sg-1", param_PostgresAiAccountArn: "arn:aws:iam::005923036815:user/pgai-rds-privatelink-monitoring", param_PostgresAiExternalId: "ext",
    });
    expect(oneSubnetPerAz([{ id: "s-b", az: "b" }, { id: "s-a2", az: "a" }, { id: "s-a1", az: "a" }])).toEqual(["s-a1", "s-b"]);
    expect(["CREATE_COMPLETE", "CREATE_IN_PROGRESS", "ROLLBACK_IN_PROGRESS", "ROLLBACK_COMPLETE", "DELETE_FAILED",
      "UPDATE_ROLLBACK_IN_PROGRESS", "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS", "UPDATE_ROLLBACK_COMPLETE", "IMPORT_COMPLETE", "UPDATE_ROLLBACK_FAILED"].map(stackState))
      .toEqual(["ready", "in_progress", "in_progress", "failed", "failed", "in_progress", "in_progress", "ready", "ready", "failed"]);
  });

  test("Terraform: the module as published; region, External ID and the RDS's values in terraform.tfvars", () => {
    const tf = 'variable "aws_region" {\n  default     = ""\n}\n';
    const out = terraformFiles(tf, { region: "us-east-2", rdsId: "db1", port: 5432, vpcId: "vpc-1", externalId: "ext" });
    expect(out.mainTf).toBe(tf);
    expect(out.tfvars).toBe('aws_region              = "us-east-2"\npostgres_ai_external_id = "ext"\nrds_instance_identifier = "db1"\nrds_port                = 5432\nvpc_id                  = "vpc-1"\n# subnet_ids            = ["subnet-...", "subnet-..."]  (one per availability zone)\n# rds_security_group_id = "sg-..."  (the RDS\'s security group)\n');
  });
});

describe("P: a private RDS over PrivateLink", () => {
  test("P1 terminal, AWS readable: price asked, link filled, waits for the stack, prepares, provisions, follows to ready", async () => {
    const lines: string[] = [];
    const f = terminal(fake({ answers: ["y"], aws: { stacks: [undefined, stack("CREATE_IN_PROGRESS"), stack("CREATE_COMPLETE")], calls: [] },
      rows: [[], [box("launch_requested")], [box("active", { egress_ip: "3.23.196.47" })]] }), lines);
    const opened: string[] = [];
    f.deps.openUrl = (u) => opened.push(u);
    const r = await connect(URL_, { waitMs: 60 * 60_000 }, f.deps);
    // The box's name is its project's: the RDS instance.
    expect(r).toMatchObject({ status: "ready", provider: "rds", name: "denis-cpl-rds", id: "b-1", cloud: "aws", location: "us-east-2", next: "Open https://x.pgai.watch" });
    expect(r.egress_ip).toBeUndefined();
    expect(f.asked[0]).toBe("Deploy denis-cpl-rds.c5q8mmwa08s8.us-east-2.rds.amazonaws.com/app over AWS PrivateLink (monitoring box on AWS, us-east-2)? [y/N] ");
    const link = f.events.find((e) => e.event === "privatelink" && e.cfn_url)!;
    expect(link.message).toStartWith("Create the PrivateLink stack in your AWS account from this link; nothing becomes public:\nhttps://us-east-2.console.aws.amazon.com/");
    // Every field filled from AWS: nothing to choose; Terraform offered; a browser can open here (openUrl), so no "open it" line.
    expect(link.message).not.toContain("In the form, choose");
    expect(link.message).toEndWith("\nPrefer Terraform? Stop (Ctrl-C) and re-run with --terraform: it writes the same stack as a module.");
    expect(link.message).not.toContain("Open it in a browser");
    expect(link.cfn_url).toContain("param_SubnetIds=subnet-a%2Csubnet-b");
    // Opened only when asked ([o]); Enter waits for the stack.
    expect(f.asked).toContain("[o] opens the link in your browser; Enter waits for the stack: ");
    expect(opened).toEqual([]);
    expect(f.events.map((e) => e.message)).toContain("Waiting for you to create the stack from the link above (any name; deploy finds it by its RDS; up to 30 min)");
    expect(f.calls.filter((c) => !c.startsWith("aws") && c !== "list")).toEqual([
      "quote -", "prepareSql app", "runSql pgai_admin", "readNonce postgres_ai_mon:Secr3t",
      `provision ${JSON.stringify({ vpce_service_name: VPCE, rds_endpoint: HOST, db_password: "Secr3t", rds_port: 5432, rds_database: "app", region: "us-east-2", vcpus: 2 })}`,
      "status b-1", "status b-1", "status b-1",
    ]);
    expect(f.events.some((e) => e.message === "AWS account 123456789012 (user/me), read-only: denis-cpl-rds is not publicly accessible, so it is monitored over AWS PrivateLink")).toBe(true);
    // The step view starts at the first event after the price: the stack, its first step.
    const afterPrice = f.events.slice(f.events.findLastIndex((e) => e.event === "billing") + 1);
    expect(afterPrice[0].event).toBe("privatelink");
  });

  test("P2 terminal, no AWS: the link without the VPC, the VpceServiceName pasted (a bad one asked again)", async () => {
    const lines: string[] = [];
    const f = terminal(fake({ aws: false, answers: ["y", "com.amazonaws.vpce.us-east-1.vpce-svc-0123456789abcdef0", VPCE], rows: [[], [box("active")]] }), lines);
    const r = await connect(URL_, { waitMs: 60_000, privatelink: true, vcpus: 2 }, f.deps);
    expect(r.status).toBe("ready");
    const link = f.events.find((e) => e.cfn_url)!;
    // The fields left to choose, on their own lines; no browser here (openUrl absent): open it elsewhere.
    expect(link.message).toContain("\nIn the form, choose (empty in the link):\n  VpcId               the RDS's VPC\n  SubnetIds           one subnet per availability zone of the RDS's subnet group\n  RdsSecurityGroupId  the RDS's security group\n(the RDS console lists them: denis-cpl-rds, Connectivity & security)\nhttps://");
    expect(link.message).toContain("\nOpen it in a browser signed in to that AWS account.");
    // Why it was refused: shown with the question again (the step view would not show it).
    expect(lines).toEqual([`com.amazonaws.vpce.us-east-1.vpce-svc-0123456789abcdef0 is in us-east-1, and denis-cpl-rds in us-east-2: AWS PrivateLink connects within one region, so use the stack made in us-east-2`]);
    expect(provisioned(f.calls)[0].vpce_service_name).toBe(VPCE);
  });

  test("P2 at the paste prompt, [o] opens the link in the browser and asks again", async () => {
    const f = terminal(fake({ aws: false, answers: ["y", "o", VPCE], rows: [[], [box("active")]] }));
    const opened: string[] = [];
    f.deps.openUrl = (u) => opened.push(u);
    expect((await connect(URL_, { waitMs: 60_000, privatelink: true, vcpus: 2 }, f.deps)).status).toBe("ready");
    expect(opened).toHaveLength(1);
    expect(f.asked.filter((q) => q.startsWith("Paste the stack's VpceServiceName (CloudFormation, the stack, Outputs), [o] to open the link in your browser, or [s] to stop: "))).toHaveLength(2);
  });

  test("P3 no terminal: stops at the price without --yes; with --yes at the stack (step privatelink_stack); nothing created", async () => {
    const f = fake({ aws: { stacks: [undefined], calls: [] } });
    const r1 = await connect(URL_, { waitMs: 0 }, f.deps);
    expect(r1).toMatchObject({ status: "action_required", cloud: "aws", location: "us-east-2",
      next: "The monitoring box would run on AWS, us-east-2 for free (3 of 5 free slots), reached over AWS PrivateLink: re-run with --yes to deploy. Nothing was created." });
    const r2 = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r2).toMatchObject({ status: "action_required", step: "privatelink_stack", stack_name: "postgresai-privatelink-denis-cpl-rds", region: "us-east-2",
      next: "Open the link below in the AWS account of denis-cpl-rds and create the stack there, or re-run with --terraform for the same as Terraform; once it is done, re-run pgai mon deploy. Nothing was created on PostgresAI yet." });
    expect(r2.cfn_url).toStartWith("https://us-east-2.console.aws.amazon.com/cloudformation/home");
    expect(f.calls.filter((c) => /prepareSql|runSql|provision/.test(c))).toEqual([]);
  });

  test("P4 agent: step privatelink_stack names vpce_service; the next call with it provisions (no AWS here)", async () => {
    const f = fake({ aws: false, rows: [[], [], [box("launch_requested")]] });
    const r1 = await connect(URL_, { waitMs: 0, agent: true, yes: true, privatelink: true }, f.deps);
    expect(r1.next).toBe("Give the user cfn_url to open in the AWS account of denis-cpl-rds, to create the stack there (choosing VpcId, SubnetIds, RdsSecurityGroupId in the form), or call again with terraform: true for the same as Terraform; once it is done, call deploy_monitoring again with vpce_service <its VpceServiceName output>. Nothing was created on PostgresAI yet.");
    expect(r1.choose_in_form).toEqual(["VpcId", "SubnetIds", "RdsSecurityGroupId"]);
    const r2 = await connect(URL_, { waitMs: 0, agent: true, yes: true, privatelink: true, vpceService: VPCE }, f.deps);
    expect(r2).toMatchObject({ status: "in_progress", id: "b-1" });
    expect(provisioned(f.calls)).toHaveLength(1);
  });

  test("P5 every choice as flags: no prompt", async () => {
    const f = terminal(fake({ aws: false, rows: [[], [box("active")]] }));
    const r = await connect(URL_, { waitMs: 60_000, privatelink: true, vpceService: VPCE, hostStatsRole: ROLE, vcpus: 4, yes: true }, f.deps);
    expect(r.status).toBe("ready");
    expect(f.asked).toEqual([]);
    expect(provisioned(f.calls)[0]).toMatchObject({ host_stats_role_arn: ROLE, vcpus: 4 });
  });

  test("P6 --host-stats: the External ID in the link, the role from the stack's outputs", async () => {
    const f = fake({ aws: { stacks: [undefined, stack("CREATE_COMPLETE", { VpceServiceName: VPCE, HostStatsRoleArn: ROLE })], calls: [] }, rows: [[], [], [box("active")]] });
    const r1 = await connect(URL_, { waitMs: 0, yes: true, hostStats: true }, f.deps);
    expect(r1.external_id).toBe("ext-123");
    expect(r1.cfn_url).toContain("param_PostgresAiExternalId=ext-123");
    await connect(URL_, { waitMs: 0, yes: true, hostStats: true }, f.deps);
    expect(provisioned(f.calls)[0].host_stats_role_arn).toBe(ROLE);
  });

  test("P6 --host-stats on a stack without the role: says how to add it, nothing created", async () => {
    const f = fake();
    const r = await connect(URL_, { waitMs: 0, yes: true, hostStats: true }, f.deps);
    expect(r.next).toBe("Host metrics need the stack's metrics role (its HostStatsRoleArn output), and it has none: update the stack with this PostgresAiExternalId (it then creates a read-only role), then re-run pgai mon deploy; or deploy without --host-stats.\n  PostgresAiExternalId = ext-123");
    expect(provisioned(f.calls)).toEqual([]);
  });

  test("P7 --terraform: the module written with the RDS's values; resume names the Terraform output", async () => {
    const written: unknown[] = [];
    const f = fake({ aws: false, terraformModule: async () => 'default     = ""\n', writeTerraform: async (dir, files) => { written.push([dir, files]); return `/work/${dir}`; } });
    const r = await connect(URL_, { waitMs: 0, yes: true, privatelink: true, terraform: true }, f.deps);
    expect(written[0]).toEqual(["postgresai-privatelink-denis-cpl-rds", { mainTf: 'default     = ""\n', tfvars: expect.stringContaining('aws_region              = "us-east-2"\nrds_instance_identifier = "denis-cpl-rds"') }]);
    expect(r).toMatchObject({ step: "privatelink_stack", terraform_dir: "/work/postgresai-privatelink-denis-cpl-rds",
      next: "Run terraform init && terraform apply in /work/postgresai-privatelink-denis-cpl-rds; once it is done, re-run with --vpce-service <its vpce_service_name output>. Nothing was created on PostgresAI yet." });
  });

  test("P8 a user that can't create the role: the SQL (step prepare_db); resumed with its password and nonce, no new SQL", async () => {
    const denied = Object.assign(new Error("permission denied to create role"), { code: "42501", severity: "ERROR" });
    const f = fake({ runSql: async () => { throw denied; }, rows: [[], [], [box("launch_requested")]] });
    const r1 = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r1).toMatchObject({ status: "action_required", step: "prepare_db", sql: PREP.prepare_db_sql, nonce: "n1", monitoring_password: "Secr3t",
      next: "The user in this URL (pgai_admin) can't create postgres_ai_mon (permission denied to create role). Run the SQL as the master user, connected to database app, then re-run with PGAI_MON_NONCE=n1 and PGAI_MON_PASSWORD set to the v_password value near the top of the SQL. A deploy of this database without PGAI_MON_PASSWORD and PGAI_MON_NONCE issues a new SQL and replaces this one." });
    f.calls.length = 0;
    const r2 = await connect(URL_, { waitMs: 0, yes: true, monitoringPassword: "Secr3t", nonce: "n1" }, f.deps);
    expect(r2.status).toBe("in_progress");
    expect(f.calls.filter((c) => /prepareSql|runSql/.test(c))).toEqual([]);
    expect(provisioned(f.calls)[0].db_password).toBe("Secr3t");
  });

  test("#459 a database this machine can't reach, a script: the SQL to run where it is reachable (step prepare_db); no box", async () => {
    const f = fake({ runSql: async () => { throw Object.assign(new Error("connect ETIMEDOUT 172.31.1.2:5432"), { code: "ETIMEDOUT" }); } });
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r).toMatchObject({ status: "action_required", step: "prepare_db", sql: PREP.prepare_db_sql, nonce: "n1", monitoring_password: "Secr3t",
      next: "This machine can't reach the database (connect ETIMEDOUT 172.31.1.2:5432), so the SQL does not run from here. Run the SQL as the master user, connected to database app, from a machine that reaches it (in its VPC), then re-run with PGAI_MON_NONCE=n1 and PGAI_MON_PASSWORD set to the v_password value near the top of the SQL. A deploy of this database without PGAI_MON_PASSWORD and PGAI_MON_NONCE issues a new SQL and replaces this one." });
    expect(provisioned(f.calls)).toEqual([]);
  });

  test("#459 resumed with the SQL's password and nonce from a machine that can't reach it: the box checks it, and is made", async () => {
    const f = fake({ readNonce: async () => { throw Object.assign(new Error("Connection terminated due to connection timeout"), {}); }, rows: [[], [], [box("launch_requested")]] });
    const r = await connect(URL_, { waitMs: 0, yes: true, monitoringPassword: "Secr3t", nonce: "n1" }, f.deps);
    expect(r.status).toBe("in_progress");
    expect(f.calls.filter((c) => /prepareSql|runSql/.test(c))).toEqual([]);
    expect(provisioned(f.calls)[0]).toMatchObject({ db_password: "Secr3t", vpce_service_name: VPCE });
    expect(f.events.map((e) => e.message)).toContain("Not checked from here (this machine can't reach the database: Connection terminated due to connection timeout): the monitoring box checks that postgres_ai_mon logs in to app and that this SQL ran, and deploy says if it fails");
  });

  test("#459 at a terminal, unreachable: the SQL shown, Enter once it ran elsewhere; no check from here, the box is made", async () => {
    const lines: string[] = [];
    const f = terminal(fake({ answers: ["y", ""], runSql: async () => { throw Object.assign(new AggregateError([Object.assign(new Error("x"), { code: "EHOSTUNREACH" })], "connect"), {}); }, rows: [[], [box("active")]] }), lines);
    expect((await connect(URL_, { waitMs: 60_000 }, f.deps)).status).toBe("ready");
    expect(lines).toContain(PREP.prepare_db_sql);
    expect(f.asked.at(-1)).toBe("Run the SQL as the master user, connected to database app, from a machine that reaches it (in its VPC), then press Enter ([s] to stop): ");
    expect(f.calls.filter((c) => c.startsWith("readNonce"))).toEqual([]);
    expect(provisioned(f.calls)).toHaveLength(1);
  });

  test("#459 a login the server refuses is not 'unreachable': no SQL, check the URL", async () => {
    const f = fake({ runSql: async () => { throw Object.assign(new Error('password authentication failed for user "pgai_admin"'), { code: "28P01", severity: "FATAL" }); } });
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r).toMatchObject({ status: "action_required", next: `Could not log in to postgresql://pgai_admin:*****@${HOST}:5432/app: password authentication failed for user "pgai_admin". Check the URL's user and password` });
    expect(r.sql).toBeUndefined();
    const g = fake({ readNonce: async () => { throw Object.assign(new Error('password authentication failed for user "postgres_ai_mon"'), { code: "28P01", severity: "FATAL" }); } });
    const r2 = await connect(URL_, { waitMs: 0, yes: true, monitoringPassword: "Secr3t", nonce: "n1" }, g.deps);
    expect(r2.next).toStartWith("postgres_ai_mon could not log in to app with PGAI_MON_PASSWORD");
    expect(provisioned(g.calls)).toEqual([]);
  });

  test("#459 unreachable(): network errors and timeouts, not the server's refusals", () => {
    for (const code of ["ECONNREFUSED", "ENOTFOUND", "EAI_AGAIN", "ENETUNREACH", "EHOSTUNREACH", "ETIMEDOUT", "ECONNRESET"]) expect(unreachable(Object.assign(new Error(code), { code }))).toBe(true);
    expect(unreachable(new AggregateError([Object.assign(new Error("a"), { code: "ECONNREFUSED" })], ""))).toBe(true);
    expect(unreachable(new Error("Connection terminated due to connection timeout"))).toBe(true);
    expect(unreachable(Object.assign(new Error("timeout"), { code: "ETIMEDOUT", severity: "FATAL" }))).toBe(false);
    expect(unreachable(Object.assign(new Error("permission denied"), { code: "42501", severity: "ERROR" }))).toBe(false);
    expect(unreachable(new Error("self-signed certificate"))).toBe(false);
    expect(unreachable("ETIMEDOUT")).toBe(false);
  });

  test("P8 at a terminal: the SQL shown, Enter once it ran, then the marker is checked", async () => {
    const lines: string[] = [];
    const f = terminal(fake({ answers: ["y", ""], runSql: async () => { throw Object.assign(new Error("permission denied"), { code: "42501", severity: "ERROR" }); }, rows: [[], [box("active")]] }), lines);
    expect((await connect(URL_, { waitMs: 60_000 }, f.deps)).status).toBe("ready");
    expect(lines).toContain(PREP.prepare_db_sql);
    expect(f.asked.at(-1)).toBe("Run the SQL as the master user, connected to database app, then press Enter ([s] to stop): ");
  });

  test("P9 a re-run: the box of this database is followed, nothing new", async () => {
    const f = fake({ rows: [[box("active")]] });
    const r = await connect(URL_, { waitMs: 0, vpceService: VPCE }, f.deps);
    expect(r).toMatchObject({ status: "ready", id: "b-1", next: "Open https://x.pgai.watch; the flags for a new box were not used: it exists" });
    expect(f.calls.filter((c) => /quote|prepareSql|provision/.test(c))).toEqual([]);
  });
});

describe("F: refusals and failures", () => {
  test("F1 a public RDS: the direct path; --privatelink on it refused", async () => {
    const f = fake({ aws: { publicly: true, calls: [] } });
    expect((await connect(URL_, { waitMs: 0, yes: true }, f.deps)).status).toBe("in_progress");
    expect(f.calls).toContain("check rds");
    expect(provisioned(f.calls)).toEqual([]);
    expect((await connect(URL_, { waitMs: 0, privatelink: true }, fake({ aws: { publicly: true, calls: [] } }).deps)).next)
      .toBe(`${HOST} is publicly accessible, so it is monitored directly: deploy without --privatelink`);
  });

  test("F1 public DNS decides without AWS: a public address goes direct, a private one over PrivateLink", async () => {
    const pub = fake({ aws: false, publicAddresses: async () => ["3.15.1.1"] });
    expect((await connect(URL_, { waitMs: 0, yes: true }, pub.deps)).status).toBe("in_progress");
    const priv = fake({ aws: false, publicAddresses: async () => ["172.31.1.2"] });
    expect((await connect(URL_, { waitMs: 0, yes: true }, priv.deps)).step).toBe("privatelink_stack");
  });

  test("F2 F3 Aurora, RDS Proxy, GovCloud over PrivateLink: refused before anything", async () => {
    const f = fake({ aws: false });
    const next = async (host: string) => (await connect(`postgresql://u:p@${host}:5432/app`, { waitMs: 0, privatelink: true }, f.deps)).next;
    expect(await next("c1.cluster-abc.us-east-2.rds.amazonaws.com")).toBe("Aurora is not supported over AWS PrivateLink yet: monitor from this network with --self-hosted, or ask support@postgres.ai");
    expect(await next("p1.proxy-abc.us-east-2.rds.amazonaws.com")).toStartWith("An RDS Proxy endpoint is not supported");
    expect(await next("db.abc.us-gov-west-1.rds.amazonaws.com")).toBe("AWS GovCloud can't reach PostgresAI over AWS PrivateLink (it does not cross AWS partitions): monitor from this network with --self-hosted");
    expect(f.calls).toEqual([]);
  });

  test("F4 F13 a failed stack: its first failure and the delete command; nothing created", async () => {
    const f = fake({ aws: { stacks: [stack("ROLLBACK_COMPLETE", {})], calls: [] } });
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r.next).toBe("Stack postgresai-privatelink-denis-cpl-rds failed (ROLLBACK_COMPLETE): Nlb: subnets in one AZ\nFix that, delete the stack, then re-run pgai mon deploy:\n  aws cloudformation delete-stack --stack-name postgresai-privatelink-denis-cpl-rds --region us-east-2");
    expect(provisioned(f.calls)).toEqual([]);
  });

  test("F5 a malformed or other-region --vpce-service: refused before the role", async () => {
    const f = fake({ aws: false });
    expect((await connect(URL_, { waitMs: 0, yes: true, privatelink: true, vpceService: "vpce-svc-1" }, f.deps)).next)
      .toBe("vpce-svc-1 is not a VpceServiceName (com.amazonaws.vpce.<region>.vpce-svc-<id>, the stack's output)");
    expect(f.calls.filter((c) => /prepareSql|runSql/.test(c))).toEqual([]);
  });

  test("F6 F14 the marker: a wrong password, no marker, another setup's nonce, all refused before the box", async () => {
    const wrong = fake({ readNonce: async () => { throw new Error("password authentication failed for user \"postgres_ai_mon\""); } });
    expect((await connect(URL_, { waitMs: 0, yes: true, monitoringPassword: "bad", nonce: "n1" }, wrong.deps)).next)
      .toBe("postgres_ai_mon could not log in to app with PGAI_MON_PASSWORD (password authentication failed for user \"postgres_ai_mon\"): set it to the v_password value near the top of the SQL you ran, or re-run without PGAI_MON_PASSWORD and PGAI_MON_NONCE for a new SQL");
    const none = fake({ readNonce: async () => undefined });
    expect((await connect(URL_, { waitMs: 0, yes: true }, none.deps)).next).toStartWith("The SQL has not run in database app (perhaps in another database): run it connected to database app");
    const other = fake({ nonce: "n0" });
    expect((await connect(URL_, { waitMs: 0, yes: true }, other.deps)).next)
      .toBe("A newer PostgresAI setup SQL ran in database app after this one (another deploy, or the Console): re-run pgai mon deploy");
    for (const x of [wrong, none, other]) expect(provisioned(x.calls)).toEqual([]);
    // A resume's check is the database step (the step view fails that step, not the stack's).
    expect(wrong.events.at(-1)).toMatchObject({ event: "preparing", message: "Checking that postgres_ai_mon can log in to app with the password the SQL set" });
    expect((await connect(URL_, { waitMs: 0, yes: true, nonce: "n1" }, fake().deps)).next).toBe("Set PGAI_MON_PASSWORD and PGAI_MON_NONCE together (both from the printed SQL's result), or neither");
  });

  test("F9 a box on this RDS already (another database, a direct one, one made in the console): refused before anything", async () => {
    const other = fake({ rows: [[box("active", { database: "other", name: "denis-cpl-rds/other" })]] });
    expect((await connect(URL_, { waitMs: 0, yes: true }, other.deps)).next)
      .toBe("denis-cpl-rds is already monitored over AWS PrivateLink, database other (instance denis-cpl-rds/other). Setting up another instance would change postgres_ai_mon's password, which that instance logs in with. To monitor app instead, delete that instance first (pgai mon instances delete denis-cpl-rds/other)");
    // A direct box on this database: its database is the one asked for, so not "To monitor app instead".
    const direct = fake({ rows: [[box("active", { provider: "self-managed" })]] });
    expect((await connect(URL_, { waitMs: 0, yes: true }, direct.deps)).next)
      .toBe("denis-cpl-rds is already monitored directly (instance denis-cpl-rds). Setting up another instance would change postgres_ai_mon's password, which that instance logs in with. To set it up from here instead, delete that instance first (pgai mon instances delete denis-cpl-rds)");
    const consoleBox = { ...box("active"), id: "c-1", name: "denis-cpl-rds", provider: "self-managed", cluster: null, database: null };
    const fromConsole = fake({ rows: [[consoleBox]], rdsStatus: async () => ({ id: "c-1", verified_status: "db_verified", instance_status: "active", vpce_service_name: VPCE, rds_endpoint: HOST.toUpperCase(), region: "us-east-2" }) });
    // A console-made box: which database it monitors is not known here.
    expect((await connect(URL_, { waitMs: 0, yes: true }, fromConsole.deps)).next)
      .toBe("denis-cpl-rds is already monitored over AWS PrivateLink, set up in the console (instance denis-cpl-rds). Setting up another instance would change postgres_ai_mon's password, which that instance logs in with. To set it up from here instead, delete that instance first (pgai mon instances delete denis-cpl-rds)");
    for (const x of [other, direct, fromConsole]) expect(x.calls.filter((c) => /quote|prepareSql|provision/.test(c))).toEqual([]);
  });

  test("F9 the platform refuses the box (PT409/PT400): its words, nothing created", async () => {
    const f = fake({ rdsProvision: async () => { throw new HttpStatusError("cloud monitoring rds provision: HTTP 400 - Bad Request\nRegion us-east-2 is not in the allowed RDS monitoring regions", 400, "PT400"); } });
    expect(await connect(URL_, { waitMs: 0, yes: true }, f.deps)).toMatchObject({ status: "action_required",
      next: "Region us-east-2 is not in the allowed RDS monitoring regions. No monitoring box was created." });
  });

  test("F11 unknown public or private: a terminal asks; [n] goes over PrivateLink, Enter stops", async () => {
    const no = terminal(fake({ aws: false, answers: ["n"] }));
    expect((await connect(URL_, { waitMs: 0 }, no.deps)).next).toStartWith("Nothing was created: re-run with --yes");
    expect(no.asked[0]).toBe(`Is ${HOST} publicly accessible? [y]es, monitor it directly / [n]o, over AWS PrivateLink / Enter to stop: `);
    const yes = terminal(fake({ aws: false, answers: ["y"] }));
    await connect(URL_, { waitMs: 0 }, yes.deps);
    expect(yes.calls).toContain("check rds");
    const stop = terminal(fake({ aws: false, answers: [""] }));
    expect((await connect(URL_, { waitMs: 0 }, stop.deps)).next).toBe("Stopped: nothing was created");
  });

  test("F12 AWS not usable: the CLI missing goes on without it; another account or an expired session: pick a profile at a terminal, else go on", async () => {
    const missing = fake({ aws: { refuse: new AwsError("the aws CLI is not installed", "missing"), calls: [] }, publicAddresses: async () => ["172.31.1.2"] });
    expect((await connect(URL_, { waitMs: 0, yes: true }, missing.deps)).step).toBe("privatelink_stack");
    expect(missing.events.map((e) => e.message)).toContain("Note: the aws CLI is not installed here; going on without AWS");
    const expired = fake({ aws: { refuse: new AwsError("Your session has expired. Please reauthenticate using 'aws login'.", "credentials"), calls: [] }, publicAddresses: async () => ["172.31.1.2"] });
    await connect(URL_, { waitMs: 0, yes: true }, expired.deps);
    expect(expired.events.map((e) => e.message)).toContain("Note: the AWS credentials do not work: Your session has expired. Please reauthenticate using 'aws login'.; going on without AWS (--aws-profile picks a profile)");
    const profiles: (string | undefined)[] = [];
    const calls: string[] = [];
    const pick = terminal(fake({ answers: ["personal", "y"], rows: [[], [box("active")]] }));
    pick.deps.aws = (p) => { profiles.push(p); return fakeAws(p === "personal" ? { stacks: [stack("CREATE_COMPLETE")], calls } : { refuse: new AwsError("DBInstanceNotFound", "not_found"), calls }); };
    expect((await connect(URL_, { waitMs: 60_000 }, pick.deps)).status).toBe("ready");
    expect(profiles).toEqual([undefined, "personal"]);
    expect(pick.asked[0]).toBe("AWS profile to read denis-cpl-rds with (default, personal), or Enter to go on without AWS: ");
  });

  test("F15 verification fails: said while waiting, and in the result when the wait ends", async () => {
    const f = fake({ rows: [[], [box("launch_requested")]], rdsStatus: async () => ({ id: "b-1", verified_status: "unreachable", instance_status: "launch_requested", vpce_service_name: VPCE, rds_endpoint: HOST, region: "us-east-2" }) });
    const r = await connect(URL_, { waitMs: 60_000, yes: true }, f.deps);
    expect(r.status).toBe("action_required");
    expect(r.next).toStartWith("Still in progress (box: starting) when the wait ended: The monitoring box could not reach the database over PrivateLink: check that the stack finished");
    expect(r.next).toEndWith("Resume with pgai mon instances watch b-1");
    expect(f.events.filter((e) => e.message.startsWith("Note: The monitoring box could not reach"))).toHaveLength(1);
  });

  test("F7 the box fails to launch (the endpoint service does not allow PostgresAI): why, then what to do; exit 1", async () => {
    const f = fake({ rows: [[], [box("failed")]] });
    const r = await connect(URL_, { waitMs: 60_000, yes: true }, f.deps);
    expect(r).toMatchObject({ status: "failed",
      next: "The monitoring box could not connect to the stack's endpoint service: check that the stack finished, and that the endpoint service (VPC console, Endpoint services, Allow principals) lists arn:aws:iam::005923036815:user/pgai-rds-privatelink-monitoring. Then: pgai mon instances delete b-1 --yes, then pgai mon deploy again" });
  });

  test("options that don't apply over PrivateLink are refused; PrivateLink options on a non-RDS host too", async () => {
    const f = fake({ aws: false });
    expect((await connect(URL_, { waitMs: 0, privatelink: true, cloud: "hetzner" }, f.deps)).next)
      .toBe("--cloud does not apply over AWS PrivateLink: the box runs on AWS in us-east-2 (the RDS's region), as postgres_ai_mon, in a project named denis-cpl-rds");
    expect((await connect("postgresql://u:p@10.0.0.5:5432/app", { waitMs: 0, vpceService: VPCE }, f.deps)).next)
      .toBe("--vpce-service applies to Amazon RDS monitored from PostgresAI Cloud only");
    expect((await connect(URL_, { waitMs: 0, privatelink: false, hostStats: true }, f.deps)).next)
      .toBe(`--host-stats applies to AWS PrivateLink only, and ${NAME} is monitored directly (--no-privatelink)`);
  });
});

describe("P10 P11: list, watch and delete of a PrivateLink box", () => {
  const route = { id: "b-1", verified_status: "db_verified", instance_status: "active", vpce_service_name: VPCE, rds_endpoint: HOST, region: "us-east-2" };
  test("the route is read for a deploy-made box (rds) and a console one (no server), not for others or a box being deleted", async () => {
    const asked: string[] = [];
    const status = async (id: string) => { asked.push(id); return route; };
    expect(await privateLinkOf(box("active"), status)).toEqual(route);
    expect(await privateLinkOf(box("active", { id: "c-1", provider: "self-managed", cluster: null }), status)).toEqual(route);
    expect(await privateLinkOf(box("active", { id: "d-1", provider: "self-managed" }), status)).toBeUndefined();
    expect(await privateLinkOf(box("deleting_launched", { id: "e-1" }), status)).toBeUndefined();
    expect(await privateLinkOf(box("active", { id: "f-1" }), async () => { throw new Error("404"); })).toBeUndefined();
    expect(asked).toEqual(["b-1", "c-1"]);
  });

  test("its check in words; a failed one says why", () => {
    expect(["db_verified", "pending", "auth_failed"].map(verifiedWords)).toEqual(["verified", "not verified yet",
      "The monitoring box reached the database, but postgres_ai_mon could not log in with the password this deploy's SQL set: the SQL did not run there, or another setup ran its SQL after it."]);
    expect(privateLinkFailure({ ...route, verified_status: "nonce_mismatch" })).toBe("The database's newest PostgresAI setup is not this deploy's: its SQL did not run in this database, or a newer one ran after it (another deploy, or the Console), so monitoring does not start.");
    expect(privateLinkFailure(route)).toBeUndefined();
  });

  test("watch's result: a failed check of the database says that, not the launch's endpoint service; no public IP", () => {
    const failed = { status: "failed" as const, provider: "rds" as const, name: "denis-cpl-rds", egress_ip: "3.1.1.1", next: "pgai mon instances delete b-1 --yes, then pgai mon deploy again" };
    const auth = privateLinkFailure({ ...route, verified_status: "auth_failed" })!;
    expect(overPrivateLink(failed, auth)).toEqual({ status: "failed", provider: "rds", name: "denis-cpl-rds", next: `${auth} Then: pgai mon instances delete b-1 --yes, then pgai mon deploy again` });
    expect(overPrivateLink(failed).next).toStartWith("The monitoring box could not connect to the stack's endpoint service");
    expect(overPrivateLink({ ...failed, status: "ready", next: "Open x" })).toEqual({ status: "ready", provider: "rds", name: "denis-cpl-rds", next: "Open x" });
  });

  test("delete names the customer's stack to delete once the box is gone", () => {
    expect(stackTeardown(route, "b-1")).toBe(`Once it is deleted (pgai mon instances watch b-1), and if nothing else is to monitor this RDS (a new deploy for it reuses the stack), delete the PrivateLink stack in your AWS account whose VpceServiceName is ${VPCE} (terraform destroy, if it was made with Terraform). If deploy's link made it:\n  aws cloudformation delete-stack --stack-name postgresai-privatelink-denis-cpl-rds --region us-east-2`);
  });
});

describe("edge cases: URLs, logins, --yes, Ctrl-C, stacks, the provision's answers", () => {
  test("a URL without a database path: the SQL and the role check both run in the database the SQL was made for", async () => {
    const urls: string[] = [];
    const f = fake({ rows: [[], [box("active", { database: "pgai_admin" })]],
      runSql: async (url) => { urls.push(url); },
      readNonce: async (url) => { urls.push(url); return "n1"; } });
    await connect(`postgresql://pgai_admin:pw@${HOST}:5432`, { waitMs: 0, yes: true }, f.deps);
    expect(f.calls).toContain("prepareSql pgai_admin");
    expect(urls.map((u) => [new URL(u).username, new URL(u).pathname])).toEqual([["pgai_admin", "/pgai_admin"], ["postgres_ai_mon", "/pgai_admin"]]);
  });

  test("the role check logs in as postgres_ai_mon with its password, whatever user or password the URL's query string names", async () => {
    const urls: string[] = [];
    const f = fake({ rows: [[], [box("active")]], readNonce: async (url) => { urls.push(url); return "n1"; } });
    await connect(`postgresql://pgai_admin@${HOST}:5432/app?user=admin&password=adminpw&sslmode=require`, { waitMs: 0, yes: true }, f.deps);
    const u = new URL(urls[0]);
    expect([u.username, decodeURIComponent(u.password), u.pathname, u.searchParams.get("user"), u.searchParams.get("password"), u.searchParams.get("sslmode")])
      .toEqual(["postgres_ai_mon", "Secr3t", "/app", null, null, "require"]);
  });

  test("a database path is used as written (pg decodes it with decodeURI)", async () => {
    const urls: string[] = [];
    const f = fake({ rows: [[], [box("active", { database: "a+b" })]], runSql: async (url) => { urls.push(url); }, readNonce: async (url) => { urls.push(url); return "n1"; } });
    await connect(`postgresql://pgai_admin:pw@${HOST}:5432/a+b`, { waitMs: 0, yes: true }, f.deps);
    expect(urls.map((x) => new URL(x).pathname)).toEqual(["/a+b", "/a+b"]);
  });

  test("findStacks: a stack being deleted is none; done and in-progress ones for this RDS are candidates", async () => {
    const named = (status: string, name: string): Stack => ({ ...stack(status), StackName: name });
    const run: AwsRun = async () => ({ Stacks: [named("DELETE_IN_PROGRESS", "postgresai-privatelink-denis-cpl-rds"), named("CREATE_IN_PROGRESS", "half"), named("UPDATE_ROLLBACK_COMPLETE", "console-made")] });
    expect((await findStacks(run, "denis-cpl-rds", "us-east-2")).stacks.map((s) => s.name)).toEqual(["half", "console-made"]);
  });

  test("the URL's own login refused (FATAL): says so, prints no SQL", async () => {
    const f = fake({ runSql: async () => { throw Object.assign(new Error("password authentication failed for user \"pgai_admin\""), { code: "28P01", severity: "FATAL" }); } });
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r.next).toStartWith(`Could not log in to postgresql://pgai_admin:*****@${HOST}:5432/app: password authentication failed`);
    expect(r.sql).toBeUndefined();
    expect(provisioned(f.calls)).toEqual([]);
    // Any FATAL at login (too many connections, say) is not a refused SQL either.
    const busy = fake({ runSql: async () => { throw Object.assign(new Error("sorry, too many clients already"), { code: "53300", severity: "FATAL" }); } });
    expect((await connect(URL_, { waitMs: 0, yes: true }, busy.deps)).next).toStartWith("Could not log in to");
  });

  test("a failure reading an existing box's route (not 404) stops before the SQL", async () => {
    const consoleBox = { ...box("active"), id: "c-1", provider: "self-managed", cluster: null, database: null };
    const f = fake({ rows: [[consoleBox]], rdsStatus: async () => { throw new HttpStatusError("HTTP 502", 502); } });
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(r.next).toStartWith("Could not check what already monitors denis-cpl-rds (HTTP 502): re-run pgai mon deploy. Nothing was changed");
    expect(f.calls.filter((c) => /quote|prepareSql|runSql/.test(c))).toEqual([]);
  });

  test("--yes at a terminal asks nothing: no AWS → the stack step; AWS → waits for the stack without [o]", async () => {
    const noAws = terminal(fake({ aws: false }));
    noAws.deps.openUrl = () => {};
    expect((await connect(URL_, { waitMs: 0, yes: true, privatelink: true }, noAws.deps)).step).toBe("privatelink_stack");
    expect(noAws.asked).toEqual([]);
    const withAws = terminal(fake({ aws: { stacks: [undefined, stack("CREATE_COMPLETE")], calls: [] }, rows: [[], [box("active")]] }));
    const opened: string[] = [];
    withAws.deps.openUrl = (u) => opened.push(u);
    expect((await connect(URL_, { waitMs: 60_000, yes: true }, withAws.deps)).status).toBe("ready");
    expect(withAws.asked).toEqual([]);
    expect(opened).toEqual([]);
  });

  test("Ctrl-C while waiting for the stack, answered by a ready stack: cancelled before the SQL", async () => {
    const f = terminal(fake({ answers: ["y", ""], aws: { stacks: [undefined, stack("CREATE_COMPLETE")], calls: [] } }));
    f.deps.aborted = () => f.calls.filter((c) => c.startsWith("aws cloudformation")).length >= 2;
    await expect(connect(URL_, { waitMs: 60_000 }, f.deps)).rejects.toThrow("Cancelled");
    expect(f.calls.filter((c) => /prepareSql|runSql/.test(c))).toEqual([]);
  });

  test("an Aurora cluster member's instance endpoint (Engine aurora-postgresql): refused", async () => {
    const f = fake({ aws: { engine: "aurora-postgresql", calls: [] } });
    expect((await connect(URL_, { waitMs: 0, yes: true }, f.deps)).next).toStartWith("Aurora is not supported over AWS PrivateLink yet");
  });

  test("stack states: being deleted is none (the link again); a create rolling back is waited for", () => {
    expect(["DELETE_IN_PROGRESS", "DELETE_COMPLETE", "ROLLBACK_IN_PROGRESS"].map(stackState)).toEqual(["deleting", "deleting", "in_progress"]);
  });

  test("findStacks: any name, matched by RdsInstanceIdentifier in any case; deploy's name for another RDS is never used", async () => {
    const named = (status: string, name: string, rds: string, outputs: Record<string, string> = { VpceServiceName: VPCE }): Stack =>
      ({ ...stack(status, outputs), StackName: name, Parameters: [{ ParameterKey: "RdsInstanceIdentifier", ParameterValue: rds }] });
    const run: AwsRun = async () => ({ Stacks: [named("CREATE_COMPLETE", "other", "other-db"), named("ROLLBACK_COMPLETE", "failed-other", "denis-cpl-rds", {}),
      named("CREATE_COMPLETE", "postgresai-rds-privatelink", "Denis-CPL-RDS"), named("CREATE_COMPLETE", "postgresai-privatelink-denis-cpl-rds", "another-db")] });
    const r = await findStacks(run, "denis-cpl-rds", "us-east-2");
    // The failed one is kept: deploy says why when it is the one being made.
    expect(r.stacks.map((s) => s.name)).toEqual(["failed-other", "postgresai-rds-privatelink"]);
    expect(r.misnamed?.rdsId).toBe("another-db");
  });

  test("the provision's answers: no card, a declined card, a changed price, no answer then found or not", async () => {
    const answer = async (err: Error, more: Parameters<typeof fake>[0] = {}) => {
      const f = fake({ ...more, rdsProvision: async () => { throw err; } });
      return connect(URL_, { waitMs: 0, yes: true }, f.deps);
    };
    expect((await answer(new HttpStatusError("HTTP 402\nno card", 402))).next).toStartWith("Add a payment method at https://console.postgres.ai/acme/billing");
    expect((await answer(new HttpStatusError("HTTP 402\nyour card was declined", 402))).next).toStartWith("The payment method on file was declined: update it at https://console.postgres.ai/acme/billing");
    expect((await answer(new HttpStatusError("HTTP 412", 412))).next).toBe("The price changed since it was shown: re-run pgai mon deploy to see it");
    let t = 0;
    const clockDeps = { now: () => t, sleep: async (ms: number) => { t += ms; } };
    const lost = await answer(new HttpStatusError("HTTP 504", 504), { ...clockDeps, rows: [[]] });
    expect(lost.next).toStartWith("PostgresAI did not confirm the monitoring box within 5 minutes");
    const found = await answer(new HttpStatusError("HTTP 504", 504), { ...clockDeps, rows: [[], [box("active")]] });
    expect(found).toMatchObject({ status: "ready", id: "b-1" });
  });

  test("the stack wait ends after 30 minutes: the stack step, nothing created", async () => {
    let t = 0;
    const f = terminal(fake({ answers: ["y", ""], aws: { stacks: [undefined, stack("CREATE_IN_PROGRESS")], calls: [] }, now: () => t, sleep: async (ms) => { t += ms; } }));
    const r = await connect(URL_, { waitMs: 60_000 }, f.deps);
    expect(r).toMatchObject({ status: "action_required", step: "privatelink_stack",
      next: "No stack for denis-cpl-rds was ready within 30 minutes: re-run pgai mon deploy once it is. Nothing was created on PostgresAI yet." });
    expect(f.calls.filter((c) => /prepareSql|provision/.test(c))).toEqual([]);
  });

  test("--debug never logs the prepare SQL (it sets the monitoring password)", () => {
    expect(redactSecretsForLog(JSON.stringify({ nonce: "n1", prepare_db_sql: "v_password text := 'Secr3t'" }))).not.toContain("Secr3t");
  });
});

describe("the user's final e2e (postgresai#454 Stage 6)", () => {
  const two = (name: string, vpce: string, created: string): Stack => ({ ...stack("CREATE_COMPLETE", { VpceServiceName: vpce }), StackName: name, CreationTime: created } as Stack & { CreationTime: string });
  const VPCE2 = "com.amazonaws.vpce.us-east-2.vpce-svc-0aaaaaaaaaaaaaaa2";

  test("the step labels: deploy waits for the person's stack; the box takes about 5 min", () => {
    expect(PRIVATELINK_STEPS.map((s) => s.label)).toEqual(["Wait for you to create the PrivateLink stack in AWS", "Prepare the database",
      "Create the monitoring box", "Connect over PrivateLink and install monitoring (about 5 min)", "Ready"]);
  });

  test("a browser can open on a desktop, not in an SSH login to a server", () => {
    expect(canOpenBrowser({ DISPLAY: ":0" }, "linux")).toBe(true);
    expect(canOpenBrowser({ WAYLAND_DISPLAY: "wayland-0" }, "linux")).toBe(true);
    expect(canOpenBrowser({ SSH_CONNECTION: "1.2.3.4 5 6.7.8.9 22" }, "linux")).toBe(false);
    expect(canOpenBrowser({}, "darwin")).toBe(true);
    expect(canOpenBrowser({ SSH_TTY: "/dev/pts/0" }, "darwin")).toBe(false);
  });

  test("no browser here: no [o] at the paste prompt", async () => {
    const f = terminal(fake({ aws: false, answers: ["y", VPCE], rows: [[], [box("active")]] }));
    await connect(URL_, { waitMs: 60_000, privatelink: true, vcpus: 2 }, f.deps);
    expect(f.asked).toContain("Paste the stack's VpceServiceName (CloudFormation, the stack, Outputs), or [s] to stop: ");
  });

  test("Enter at the paste prompt asks again with a hint; s stops; Enters with no input left stop", async () => {
    const lines: string[] = [];
    const f = terminal(fake({ aws: false, answers: ["y", "", VPCE], rows: [[], [box("active")]] }), lines);
    expect((await connect(URL_, { waitMs: 60_000, privatelink: true, vcpus: 2 }, f.deps)).status).toBe("ready");
    expect(lines).toContain("Paste the stack's VpceServiceName once it is created, or s to stop.");
    const s = terminal(fake({ aws: false, answers: ["y", "s"] }));
    expect((await connect(URL_, { waitMs: 0, privatelink: true, vcpus: 2 }, s.deps)).next).toStartWith("Stopped: once the stack is done, re-run with --vpce-service");
    const eof = terminal(fake({ aws: false, answers: ["y"] }));
    expect((await connect(URL_, { waitMs: 0, privatelink: true, vcpus: 2 }, eof.deps)).next).toStartWith("Stopped:");
    expect(eof.asked.filter((q) => q.startsWith("Paste the stack's"))).toHaveLength(3);
  });

  test("one stack for the RDS: used and named", async () => {
    const f = fake({ aws: { stacks: [two("my-own-name", VPCE, "2026-10-28T12:02:03.456000+00:00")], calls: [] }, rows: [[], [box("active")]] });
    await connect(URL_, { waitMs: 60_000, yes: true }, f.deps);
    expect(f.events.map((e) => e.message)).toContain(`Using stack my-own-name (created 2026-10-28 12:02:03 UTC): ${VPCE}`);
  });

  test("several stacks for the RDS: a terminal picks one; a script gets them listed (exit 3), nothing created", async () => {
    const both = { stacks: [undefined], calls: [] as string[] };
    const listed: Stack[] = [two("stack-a", VPCE, "2026-10-28T12:00:00Z"), two("stack-b", VPCE2, "2026-10-28T13:00:00Z")];
    const lines: string[] = [];
    const pick = terminal(fake({ answers: ["y", "x", "2"], rows: [[], [box("active")]] }), lines);
    pick.deps.aws = () => async (args) => (args[0] === "cloudformation" ? { Stacks: listed } : fakeAws({ ...both })(args));
    expect((await connect(URL_, { waitMs: 60_000 }, pick.deps)).status).toBe("ready");
    expect(lines.slice(0, 3)).toEqual(["Several stacks publish denis-cpl-rds:", `  1. stack-a  CREATE_COMPLETE  created 2026-10-28 12:00:00 UTC  ${VPCE}`, `  2. stack-b  CREATE_COMPLETE  created 2026-10-28 13:00:00 UTC  ${VPCE2}`]);
    expect(provisioned(pick.calls)[0].vpce_service_name).toBe(VPCE2);
    const script = fake({ rows: [[]] });
    script.deps.aws = () => async (args) => (args[0] === "cloudformation" ? { Stacks: listed } : fakeAws({ ...both })(args));
    const r = await connect(URL_, { waitMs: 0, yes: true }, script.deps);
    expect(r).toMatchObject({ status: "action_required", stacks: [{ name: "stack-a", vpce_service_name: VPCE }, { name: "stack-b", vpce_service_name: VPCE2 }] });
    expect(r.next).toStartWith("Several stacks publish denis-cpl-rds: re-run with --vpce-service <the VpceServiceName of the one to use>. Nothing was created on PostgresAI yet.");
    expect(provisioned(script.calls)).toEqual([]);
  });

  test("a stack with deploy's name for another RDS: said, and not used", async () => {
    const other = { ...stack("CREATE_COMPLETE"), Parameters: [{ ParameterKey: "RdsInstanceIdentifier", ParameterValue: "another-db" }] };
    const f = fake({ rows: [[]] });
    f.deps.aws = () => async (args) => (args[0] === "cloudformation" ? { Stacks: [other] } : fakeAws({ calls: [] })(args));
    const r = await connect(URL_, { waitMs: 0, yes: true }, f.deps);
    expect(f.events.map((e) => e.message)).toContain("Note: stack postgresai-privatelink-denis-cpl-rds is for RDS another-db, not denis-cpl-rds: not used. Give the new stack another name in the form");
    expect(r.step).toBe("privatelink_stack");
  });
});

test("the step view: the link and the stack used stay on screen; the wait is a note under the step", () => {
  const notes: string[] = [];
  const logs: string[] = [];
  const view: StepViewLike = { start() {}, advance() {}, log: (t) => logs.push(t), setNote: (n) => notes.push(n), complete() {}, fail() {}, stop() {} };
  const p = stepProgress({ json: false, tty: true, makeView: () => view, writeLine: () => {}, writeEvent: () => {} });
  p.progress({ event: "billing", elapsed_s: 0, message: "Estimated cost: free" });
  p.progress({ event: "privatelink", elapsed_s: 1, message: "Create the PrivateLink stack …\nhttps://x", cfn_url: "https://x" });
  p.progress({ event: "privatelink", elapsed_s: 2, message: "Waiting for you to create the stack from the link above (any name; deploy finds it by its RDS; up to 30 min)" });
  p.progress({ event: "privatelink", elapsed_s: 3, message: `Using stack s-2 (created 2026-10-28 12:02:03 UTC): ${VPCE}` });
  expect(logs).toEqual(["Create the PrivateLink stack …\nhttps://x", `Using stack s-2 (created 2026-10-28 12:02:03 UTC): ${VPCE}`]);
  expect(notes).toEqual(["Waiting for you to create the stack from the link above (any name; deploy finds it by its RDS; up to 30 min)"]);
});

describe("Stage 6 review: failed stacks under any name, Terraform keys, steps", () => {
  const named = (status: string, name: string, outputs: Record<string, string> = {}): Stack => ({ ...stack(status, outputs), StackName: name });
  test("a stack under another name that fails while deploy waits: why, and its delete command", async () => {
    let reads = 0;
    const seq = [[named("CREATE_IN_PROGRESS", "my-stack")], [named("ROLLBACK_IN_PROGRESS", "my-stack")], [named("ROLLBACK_COMPLETE", "my-stack")]];
    const f = terminal(fake({ answers: ["y", ""] }));
    f.deps.aws = () => async (args) => {
      if (args[0] === "cloudformation" && args[1] === "describe-stack-events") return { StackEvents: [{ LogicalResourceId: "PreflightCheck", ResourceStatus: "CREATE_FAILED", ResourceStatusReason: "DBInstance x not found (RequestId: 1)" }] };
      if (args[0] === "cloudformation") return { Stacks: seq[Math.min(reads++, seq.length - 1)] };
      return fakeAws({ calls: [] })(args);
    };
    const r = await connect(URL_, { waitMs: 60_000 }, f.deps);
    expect(r).toMatchObject({ status: "action_required", step: "privatelink_stack", stack_name: "my-stack",
      next: "Stack my-stack failed (ROLLBACK_COMPLETE): PreflightCheck: DBInstance x not found\nFix that, delete the stack, then re-run pgai mon deploy:\n  aws cloudformation delete-stack --stack-name my-stack --region us-east-2" });
    expect(f.events.map((e) => e.message)).toContain("Stack my-stack is rolling back (ROLLBACK_IN_PROGRESS)");
  });

  test("a run that does not wait (a script, an agent): the newest failed stack for the RDS is reported with its reason", async () => {
    const f = fake({ rows: [[]] });
    f.deps.aws = () => async (args) => {
      if (args[1] === "describe-stack-events") return { StackEvents: [{ LogicalResourceId: "Nlb", ResourceStatus: "CREATE_FAILED", ResourceStatusReason: "subnets in one AZ" }] };
      if (args[0] === "cloudformation") return { Stacks: [{ ...named("ROLLBACK_COMPLETE", "older"), CreationTime: "2026-10-01T10:00:00Z" }, { ...named("ROLLBACK_COMPLETE", "newer"), CreationTime: "2026-10-28T10:00:00Z" }] };
      return fakeAws({ calls: [] })(args);
    };
    for (const agent of [false, true]) {
      const r = await connect(URL_, { waitMs: 0, yes: true, agent }, f.deps);
      expect(r).toMatchObject({ step: "privatelink_stack", stack_name: "newer" });
      expect(r.next).toStartWith("Stack newer failed (ROLLBACK_COMPLETE): Nlb: subnets in one AZ");
    }
  });

  test("waiting at a terminal: an older failed stack is said once and deploy waits for a new one", async () => {
    let t = 0;
    const f = terminal(fake({ answers: ["y", ""], now: () => t, sleep: async (ms) => { t += ms; } }));
    f.deps.aws = () => async (args) => (args[0] === "cloudformation" ? { Stacks: [{ ...named("ROLLBACK_COMPLETE", "old-try"), CreationTime: "2020-01-01T00:00:00Z" }] } : fakeAws({ calls: [] })(args));
    const r = await connect(URL_, { waitMs: 60_000 }, f.deps);
    expect(f.events.filter((e) => e.message === "Note: stack old-try for denis-cpl-rds failed earlier (ROLLBACK_COMPLETE): not used")).toHaveLength(1);
    expect(r.next).toStartWith("No stack for denis-cpl-rds was ready within 30 minutes");
  });

  test("deploy's own stack done without the template's output: said, not waited on", async () => {
    const f = fake({ rows: [[]] });
    f.deps.aws = () => async (args) => (args[0] === "cloudformation" ? { Stacks: [named("CREATE_COMPLETE", "postgresai-privatelink-denis-cpl-rds")] } : fakeAws({ calls: [] })(args));
    expect((await connect(URL_, { waitMs: 0, yes: true }, f.deps)).next).toStartWith("Stack postgresai-privatelink-denis-cpl-rds has no VpceServiceName output: it was not made from the PostgresAI template");
  });

  test("--terraform names the fields as terraform.tfvars does", async () => {
    const f = fake({ aws: false, terraformModule: async () => "", writeTerraform: async (d) => `/w/${d}` });
    const r = await connect(URL_, { waitMs: 0, yes: true, privatelink: true, terraform: true }, f.deps);
    expect(r.choose_in_form).toEqual(["vpc_id", "subnet_ids", "rds_security_group_id"]);
    expect(f.events.find((e) => e.cfn_url)?.message).toContain("In terraform.tfvars, fill in:\n  vpc_id                 the RDS's VPC\n");
  });

  test("--terraform with AWS: terraform.tfvars from AWS; no CloudFormation stack is looked for, so resume names --vpce-service", async () => {
    const tfvars: string[] = [];
    const f = fake({ aws: { stacks: [undefined] }, terraformModule: async () => "", writeTerraform: async (d, files) => { tfvars.push(files.tfvars); return `/w/${d}`; } });
    const r = await connect(URL_, { waitMs: 0, yes: true, terraform: true }, f.deps);
    expect(r.next).toBe("Run terraform init && terraform apply in /w/postgresai-privatelink-denis-cpl-rds; once it is done, re-run with --vpce-service <its vpce_service_name output>. Nothing was created on PostgresAI yet.");
    expect(tfvars[0]).toContain('vpc_id                  = "vpc-1"');
    expect(r.choose_in_form).toBeUndefined();
    expect(f.calls.filter((c) => c.startsWith("aws cloudformation"))).toEqual([]);
  });

  test("--terraform with AWS at a terminal: the vpce_service_name output is pasted, not waited for", async () => {
    const f = terminal(fake({ answers: ["y", VPCE], aws: { stacks: [undefined] }, rows: [[], [], [box("launch_requested")]], terraformModule: async () => "", writeTerraform: async (d) => `/w/${d}` }));
    await connect(URL_, { waitMs: 0, terraform: true }, f.deps);
    expect(f.asked).toContain("Paste the stack's vpce_service_name output, or [s] to stop: ");
    expect(provisioned(f.calls)[0]).toMatchObject({ vpce_service_name: VPCE });
    expect(f.calls.filter((c) => c.startsWith("aws cloudformation"))).toEqual([]);
  });

  test("--host-stats with a pasted or passed vpce: the role's output is asked for, with that vpce; no box", async () => {
    const t = terminal(fake({ answers: ["y", VPCE], aws: { stacks: [undefined] }, terraformModule: async () => "", writeTerraform: async (d) => `/w/${d}` }));
    const r1 = await connect(URL_, { waitMs: 0, terraform: true, hostStats: true }, t.deps);
    expect(r1.next).toBe(`Host metrics need the metrics role too: re-run with --vpce-service ${VPCE} and --host-stats-role <its host_stats_role_arn output>. If it has none, it was made without this PostgresAiExternalId: set postgres_ai_external_id in terraform.tfvars and apply again; or deploy without --host-stats.\n  PostgresAiExternalId = ext-123`);
    expect(provisioned(t.calls)).toEqual([]);
    const f = fake({ aws: false });
    const r2 = await connect(URL_, { waitMs: 0, yes: true, privatelink: true, vpceService: VPCE, hostStats: true }, f.deps);
    expect(r2.next).toStartWith(`Host metrics need the metrics role too: re-run with --vpce-service ${VPCE} and --host-stats-role <its HostStatsRoleArn output>. If it has none, it was made without this PostgresAiExternalId: update the stack with it;`);
    expect(provisioned(f.calls)).toEqual([]);
  });
});
