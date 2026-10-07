import { afterEach, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { createCliSandbox } from "./cli-sandbox";
import { loadInstances, addInstanceToFile, buildInstance } from "../lib/instances";
import { targetChannelConfig, targetChannelHeaders } from "../lib/targets-sync-worker";

const url = "postgresql://worker:synthetic-password-878@database.invalid/db?sslmode=disable";
const token = "synthetic-box-token=878";
const cleanup: Array<() => Promise<void> | void> = [];
afterEach(async () => { for (const fn of cleanup.splice(0).reverse()) await fn(); });

async function until(check: () => boolean, ms = 4000) {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error("Timed out waiting for worker fixture");
    await Bun.sleep(20);
  }
}

function fixture(targets: Array<{ target_id: string; name: string; adopt: boolean }>, options: { loseSubmit?: boolean; hangPoll?: boolean; redirectPoll?: string } = {}) {
  const box = createCliSandbox();
  cleanup.push(() => box.cleanup());
  const file = path.join(box.projectDir, "instances.yml");
  const journal = path.join(box.projectDir, ".pgai-managed-targets.json");
  const calls: Array<{ rpc: string; body: any; headers: Headers }> = [];
  const submits: any[] = [];
  const acceptedSubmits: any[] = [];
  const stackLog = path.join(box.root, "stack.log");
  let polls = 0;
  let releasePoll: (() => void) | undefined;
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const rpc = new URL(req.url).pathname.split("/").pop()!;
    const body: any = await req.json();
    calls.push({ rpc, body, headers: req.headers });
    if (rpc === "monitoring_target_poll") {
      polls++;
      if (options.redirectPoll) return new Response(null, { status: 307, headers: { Location: options.redirectPoll } });
      if (options.hangPoll) return new Promise<Response>(resolve => {
        releasePoll = () => resolve(new Response(null, { status: 503 }));
      });
      return Response.json({ job: submits.length && !options.loseSubmit ? null : { id: 878, generation: polls, targets }, next_poll_ms: 1 });
    }
    if (rpc === "monitoring_target_secret") {
      if (targets.some(t => t.target_id === body.target_id && t.adopt)) return Response.json({ code: "PT404" }, { status: 404 });
      return Response.json({ db_url: url });
    }
    if (rpc === "monitoring_target_submit") {
      submits.push(body);
      if (options.loseSubmit && submits.length === 1) return Response.json({ code: "503", message: url }, { status: 503 });
      if (body.failed.some((t: any) => !targets.some(desired => desired.target_id === t.target_id))) {
        return Response.json({ code: "PT400", message: "outside the desired set" }, { status: 400 });
      }
      acceptedSubmits.push(body);
      return Response.json({ ok: true });
    }
    return new Response("Unexpected RPC", { status: 500 });
  } });
  cleanup.push(() => { releasePoll?.(); return server.stop(true); });
  fs.writeFileSync(file, "[]\n");
  fs.writeFileSync(path.join(box.projectDir, ".pgwatch-config"),
    `\ufeffapi_key= ${token} \r\napi_key=wrong\ninstance_id=never-send-this\napi_base_url=${server.url}api/general///\n`);
  fs.writeFileSync(path.join(box.binDir, "docker"), `#!/bin/sh
printf '%s\\n' "$*" >> '${stackLog}'
env >> '${stackLog}'
case "$*" in
  *sources-generator*)
    touch '${box.root}/applying'
    while test -f '${box.root}/hold'; do sleep 0.05; done
    if test -f '${box.root}/fail'; then cat '${file}'; exit 1; fi ;;
esac
exit 0
`, { mode: 0o700 });
  const env = { ...process.env, HOME: box.home, XDG_CONFIG_HOME: box.configHome,
    PATH: `${box.binDir}:${process.env.PATH}`, PGAI_API_BASE_URL: "http://127.0.0.1:1",
    CF_ACCESS_CLIENT_ID: "synthetic-cf-id", CF_ACCESS_CLIENT_SECRET: "synthetic-cf-secret" };
  for (const key of Object.keys(env)) if (key.startsWith("PGAI_") && key !== "PGAI_API_BASE_URL") delete (env as NodeJS.ProcessEnv)[key];
  function start(args = ["sync-worker"]) {
    const child = Bun.spawn([process.execPath, box.cliPath, "mon", "targets", ...args], { cwd: box.projectDir, env, stdout: "pipe", stderr: "pipe" });
    const output = Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text()]);
    cleanup.push(async () => {
      fs.rmSync(path.join(box.root, "hold"), { force: true });
      child.kill("SIGTERM"); await child.exited; await output;
    });
    return { child, output, async stop() { child.kill("SIGTERM"); await child.exited; return (await output).join("\n"); } };
  }
  return { box, file, journal, calls, submits, acceptedSubmits, stackLog, start, get polls() { return polls; } };
}

test("sync adds, adopts without a secret, and removes only journal targets", async () => {
  const f = fixture([{ target_id: "host", name: "host-db", adopt: true }, { target_id: "new", name: "new-db", adopt: false }]);
  addInstanceToFile(f.file, buildInstance("host-db", "postgresql://host/db"));
  addInstanceToFile(f.file, buildInstance("manual", "postgresql://manual/db"));
  addInstanceToFile(f.file, buildInstance("old-db", "postgresql://old/db"));
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "old-db" }]));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(loadInstances(f.file).map(i => i.name).sort()).toEqual(["host-db", "manual", "new-db"]);
  expect(f.submits[0].applied.sort()).toEqual(["host", "new"]);
  expect(f.submits[0].failed).toEqual([]);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").map(c => c.body)).toEqual([{ target_id: "new" }]);
  expect(fs.statSync(f.journal).mode & 0o777).toBe(0o600);
  expect(fs.readFileSync(f.journal, "utf8").includes(url)).toBe(false);
  expect(fs.statSync(f.file).mode & 0o777).toBe(0o600);
  const output = await worker.stop();
  expect(output.includes(url)).toBe(false);
  expect(output.includes("synthetic-password-878")).toBe(false);
  const childLog = fs.readFileSync(f.stackLog, "utf8");
  expect(childLog.includes(url)).toBe(false);
  expect(childLog.includes("synthetic-password-878")).toBe(false);
  for (const call of f.calls) {
    expect(call.headers.get("access-token") === token).toBe(true);
    expect(call.headers.get("CF-Access-Client-Secret")).toBeNull();
    expect(call.body.instance_id).toBeUndefined();
  }
  expect(f.calls[0].body).toEqual({ protocol_version: 1 });
});

test("journal-only removal reconciles an already absent name", async () => {
  const f = fixture([]);
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "absent" }]));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8"))).toEqual({ targets: [], desired: [] });
  expect(fs.readFileSync(f.stackLog, "utf8")).toContain("sources-generator");
  await worker.stop();
});

test("manual absent remove still applies and keeps its output and status", async () => {
  const f = fixture([]);
  const manual = f.start(["remove", "absent"]);
  expect(await manual.child.exited).toBe(1);
  const output = (await manual.output).join("\n");
  expect(output).toContain("Monitoring target 'absent' not found");
  expect(output).not.toContain("configuration applied");
  expect(fs.readFileSync(f.stackLog, "utf8")).toContain("sources-generator");
});

test("lost submit replay never redeems the secret or duplicates an add", async () => {
  const f = fixture([{ target_id: "new", name: "new-db", adopt: false }], { loseSubmit: true });
  const worker = f.start();
  await until(() => f.submits.length >= 2, 12000);
  expect(loadInstances(f.file).length).toBe(1);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(1);
  expect(f.submits[1].applied).toEqual(["new"]);
  expect((await worker.stop()).includes(url)).toBe(false);
}, 15000);

test("failed apply is journaled, redacted, and reconciled on replay", async () => {
  const f = fixture([{ target_id: "new", name: "new-db", adopt: false }], { loseSubmit: true });
  fs.writeFileSync(path.join(f.box.root, "fail"), "");
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual([]);
  expect(f.submits[0].failed[0].target_id).toBe("new");
  expect(JSON.stringify(f.submits).includes("synthetic-password-878")).toBe(false);
  fs.unlinkSync(path.join(f.box.root, "fail"));
  await until(() => f.submits.length >= 2, 12000);
  expect(f.submits[1].applied).toEqual(["new"]);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(1);
  expect((await worker.stop()).includes("synthetic-password-878")).toBe(false);
}, 15000);

test("worker refuses to adopt a manual target without adopt=true", async () => {
  const f = fixture([{ target_id: "new", name: "manual", adopt: false }]);
  addInstanceToFile(f.file, buildInstance("manual", "postgresql://manual/db"));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual([]);
  expect(f.submits[0].failed.length).toBe(1);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets).toEqual([]);
  await worker.stop();
});

test("manual mutations exclude the worker and SIGTERM releases the lock", async () => {
  const f = fixture([{ target_id: "new", name: "new-db", adopt: false }]);
  fs.writeFileSync(path.join(f.box.root, "hold"), "");
  const manual = f.start(["add", "postgresql://manual:fixture@database.invalid/db?sslmode=disable", "manual"]);
  await until(() => fs.existsSync(path.join(f.box.root, "applying")));
  const worker = f.start();
  await until(() => f.polls > 0);
  await Bun.sleep(250);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  await worker.stop();
  expect(await worker.child.exited).toBe(0);
  fs.unlinkSync(path.join(f.box.root, "hold"));
  expect(await manual.child.exited).toBe(0);
  const resumed = f.start();
  await until(() => f.submits.length === 1);
  await resumed.stop();
  expect(loadInstances(f.file).map(i => i.name).sort()).toEqual(["manual", "new-db"]);
});

test("SIGTERM aborts an in-flight poll and exits cleanly", async () => {
  const f = fixture([], { hangPoll: true });
  const worker = f.start();
  await until(() => f.polls > 0);
  await worker.stop();
  expect(await worker.child.exited).toBe(0);
});

test("instances writes replace the inode instead of truncating the reader's file", () => {
  const f = fixture([]);
  const fd = fs.openSync(f.file, "r");
  try {
    addInstanceToFile(f.file, buildInstance("new-db", url));
    expect(fs.readFileSync(fd, "utf8") === "[]\n").toBe(true);
    expect(loadInstances(f.file).length).toBe(1);
  } finally { fs.closeSync(fd); }
});

test("target transport mirrors config precedence and HTTPS-only CF Access headers", () => {
  const f = fixture([]);
  const env = { PGAI_API_BASE_URL: "https://inherited.invalid/api", CF_ACCESS_CLIENT_ID: " id ", CF_ACCESS_CLIENT_SECRET: " secret " };
  const config = targetChannelConfig(f.box.projectDir, env);
  expect(config.token === token).toBe(true);
  expect(config.baseURL.startsWith("http://127.0.0.1:")).toBe(true);
  const headers = targetChannelHeaders("https://box.invalid/api", token, env);
  expect(headers["CF-Access-Client-Id"] === "id").toBe(true);
  expect(headers["CF-Access-Client-Secret"] === "secret").toBe(true);
  expect(targetChannelHeaders(config.baseURL, token, env)["CF-Access-Client-Secret"]).toBeUndefined();
  expect(targetChannelHeaders("https://box.invalid/api", token, { CF_ACCESS_CLIENT_ID: "id" })["CF-Access-Client-Id"]).toBeUndefined();
  fs.writeFileSync(path.join(f.box.projectDir, ".pgwatch-config"), "api_key=fixture\n api_base_url=https://ignored.invalid\n");
  expect(targetChannelConfig(f.box.projectDir, env).baseURL).toBe(env.PGAI_API_BASE_URL);
  expect(targetChannelConfig(f.box.projectDir, {}).baseURL).toBe("https://postgres.ai/api/general");
  expect(() => targetChannelConfig(f.box.projectDir, { PGAI_API_BASE_URL: "http://public.invalid" })).toThrow();
});

test("poll redirects never replay the box credential", async () => {
  let requests = 0;
  const other = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch() { requests++; return Response.json({}); } });
  cleanup.push(() => other.stop(true));
  const f = fixture([], { redirectPoll: other.url.toString() });
  const worker = f.start();
  await until(() => f.polls === 1);
  await Bun.sleep(100);
  expect(requests).toBe(0);
  await worker.stop();
});

test("failed removal retains ownership until an absent-name replay reconciles", async () => {
  const f = fixture([], { loseSubmit: true });
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "old-db" }]));
  addInstanceToFile(f.file, buildInstance("old-db", url));
  fs.writeFileSync(path.join(f.box.root, "fail"), "");
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(loadInstances(f.file)).toEqual([]);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.length).toBe(1);
  fs.unlinkSync(path.join(f.box.root, "fail"));
  await until(() => f.submits.length >= 2, 12000);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets).toEqual([]);
  await worker.stop();
}, 15000);

test("sync publishes only applied desired projects, including adoption and detach", async () => {
  const f = fixture([{ target_id: "host", name: "app", adopt: true }, { target_id: "new", name: "app2", adopt: false }]);
  addInstanceToFile(f.file, buildInstance("app", "postgresql://host/db"));
  addInstanceToFile(f.file, buildInstance("manual", "postgresql://manual/db"));
  const projects = path.join(f.box.projectDir, ".pgai-report-projects.json");
  fs.writeFileSync(projects, JSON.stringify({ projects: [{ project: "detached", source: "old-db" }] }), { mode: 0o644 });
  const worker = f.start();
  await until(() => f.submits.length === 1);
  await worker.stop();
  expect(JSON.parse(fs.readFileSync(projects, "utf8"))).toEqual({ projects: [{ project: "app", source: "app" }, { project: "app2", source: "app2" }] });
  expect(fs.statSync(projects).mode & 0o777).toBe(0o600);
});

test("failed adds publish an empty projects file and never print the redeemed password", async () => {
  const f = fixture([{ target_id: "new", name: "app2", adopt: false }]);
  fs.writeFileSync(path.join(f.box.root, "fail"), "");
  const worker = f.start();
  await until(() => f.submits.length === 1);
  const output = await worker.stop();
  expect(output.includes(new URL(url).password)).toBe(false);
  expect(JSON.stringify(f.submits).includes(new URL(url).password)).toBe(false);
  const projects = path.join(f.box.projectDir, ".pgai-report-projects.json");
  expect(fs.existsSync(projects)).toBe(true);
  expect(JSON.parse(fs.readFileSync(projects, "utf8"))).toEqual({ projects: [] });
});

test("empty desired set retains a private empty projects file", async () => {
  const f = fixture([]);
  const projects = path.join(f.box.projectDir, ".pgai-report-projects.json");
  fs.writeFileSync(projects, JSON.stringify({ projects: [{ project: "detached", source: "old-db" }] }));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  await worker.stop();
  expect(JSON.parse(fs.readFileSync(projects, "utf8"))).toEqual({ projects: [] });
  expect(fs.statSync(projects).mode & 0o777).toBe(0o600);
});

test("inferred host name is adopted, replayed, and removed by its local name", async () => {
  const targets = [{ target_id: "host", name: "host-project", adopt: true }];
  const f = fixture(targets, { loseSubmit: true });
  addInstanceToFile(f.file, buildInstance("hostname-db", "postgresql://host/db"));
  addInstanceToFile(f.file, buildInstance("old-db", "postgresql://old/db"));
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "old-db" }]));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual(["host"]);
  expect(f.submits[0].failed).toEqual([]);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets).toEqual([
    { target_id: "host", name: "host-project", local_name: "hostname-db", pending: false, adopted: true },
  ]);
  const projects = path.join(f.box.projectDir, ".pgai-report-projects.json");
  expect(JSON.parse(fs.readFileSync(projects, "utf8"))).toEqual({ projects: [{ project: "host-project", source: "hostname-db" }] });
  await until(() => f.submits.length >= 2, 12000);
  expect(f.submits[1].applied).toEqual(["host"]);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  targets.splice(0);
  await until(() => f.submits.length >= 3, 12000);
  expect(loadInstances(f.file).map(i => i.name)).toEqual([]);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8"))).toEqual({ targets: [], desired: [] });
  expect(JSON.parse(fs.readFileSync(projects, "utf8"))).toEqual({ projects: [] });
  await worker.stop();
}, 30000);

for (const names of [[], ["first-db", "second-db"]]) {
  test(`host adoption refuses ${names.length} unjournaled candidates without a secret`, async () => {
    const f = fixture([{ target_id: "host", name: "host-project", adopt: true }]);
    for (const name of names) addInstanceToFile(f.file, buildInstance(name, "postgresql://host/db"));
    const worker = f.start();
    await until(() => f.submits.length === 1);
    expect(f.submits[0].applied).toEqual([]);
    expect(f.submits[0].failed.map((t: any) => t.target_id)).toEqual(["host"]);
    expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
    expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets).toEqual([]);
    expect(loadInstances(f.file).map(i => i.name)).toEqual(names);
    await worker.stop();
  });
}

test("only one new adoption is allowed per job, including exact-name matches", async () => {
  const f = fixture([{ target_id: "first", name: "first-db", adopt: true }, { target_id: "second", name: "second-db", adopt: true }], { loseSubmit: true });
  for (const name of ["first-db", "second-db"]) addInstanceToFile(f.file, buildInstance(name, "postgresql://host/db"));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual(["first"]);
  expect(f.submits[0].failed.map((t: any) => t.target_id)).toEqual(["second"]);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.map((t: any) => t.target_id)).toEqual(["first"]);
  await until(() => f.submits.length >= 2, 12000);
  expect(f.submits[1].applied).toEqual(["first"]);
  expect(f.submits[1].failed.map((t: any) => t.target_id)).toEqual(["second"]);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.map((t: any) => t.target_id)).toEqual(["first"]);
  await worker.stop();
}, 15000);

test("an adopted local name cannot be claimed by another desired target", async () => {
  const f = fixture([{ target_id: "host", name: "host-project", adopt: true }, { target_id: "new", name: "hostname-db", adopt: false }]);
  addInstanceToFile(f.file, buildInstance("hostname-db", "postgresql://host/db"));
  fs.writeFileSync(f.journal, JSON.stringify({
    targets: [{ target_id: "host", name: "host-project", local_name: "hostname-db", adopted: true }],
    desired: [{ target_id: "host", name: "host-project" }],
  }));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual(["host"]);
  expect(f.submits[0].failed.map((t: any) => t.target_id)).toEqual(["new"]);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  await worker.stop();
});

test("host adoption recovers a pending journal from the previous name-mismatch failure", async () => {
  const f = fixture([{ target_id: "host", name: "host-project", adopt: true }]);
  addInstanceToFile(f.file, buildInstance("hostname-db", "postgresql://host/db"));
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "host", name: "host-project", pending: true, adopted: false }]));
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].applied).toEqual(["host"]);
  expect(f.submits[0].failed).toEqual([]);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets).toEqual([
    { target_id: "host", name: "host-project", local_name: "hostname-db", pending: false, adopted: true },
  ]);
  await worker.stop();
});

test("failed removal submits successfully and retries on the next empty poll under the lock", async () => {
  const f = fixture([]);
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "old-db" }]));
  addInstanceToFile(f.file, buildInstance("old-db", url));
  fs.writeFileSync(path.join(f.box.root, "fail"), "");
  const worker = f.start();
  await until(() => f.submits.length === 1);
  expect(f.submits[0].failed).toEqual([]);
  expect(f.acceptedSubmits.length).toBe(1);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.map((t: any) => t.target_id)).toEqual(["old"]);
  fs.unlinkSync(path.join(f.box.root, "fail"));
  fs.rmSync(path.join(f.box.root, "applying"));
  fs.writeFileSync(path.join(f.box.root, "hold"), "");
  const manual = f.start(["remove", "absent"]);
  await until(() => fs.existsSync(path.join(f.box.root, "applying")));
  const stackCalls = fs.readFileSync(f.stackLog, "utf8");
  await until(() => f.polls >= 2, 12000);
  await Bun.sleep(250);
  expect(fs.readFileSync(f.stackLog, "utf8") === stackCalls).toBe(true);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.length).toBe(1);
  fs.unlinkSync(path.join(f.box.root, "hold"));
  await manual.child.exited;
  await until(() => JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.length === 0);
  expect(f.submits.length).toBe(1);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(0);
  await worker.stop();
}, 20000);

test("empty polls after restart use persisted desired ids and names and only retry removals", async () => {
  const f = fixture([{ target_id: "new", name: "new-db", adopt: false }]);
  fs.writeFileSync(f.journal, JSON.stringify([{ target_id: "old", name: "old-db" }]));
  addInstanceToFile(f.file, buildInstance("old-db", url));
  fs.writeFileSync(path.join(f.box.root, "fail"), "");
  const worker = f.start();
  await until(() => f.submits.length === 1);
  await worker.stop();
  expect(f.submits[0].failed.map((t: any) => t.target_id)).toEqual(["new"]);
  expect(f.acceptedSubmits.length).toBe(1);
  expect(JSON.parse(fs.readFileSync(f.journal, "utf8")).desired).toEqual([{ target_id: "new", name: "new-db" }]);
  const secretCalls = f.calls.filter(c => c.rpc === "monitoring_target_secret").length;
  fs.unlinkSync(path.join(f.box.root, "fail"));
  const resumed = f.start();
  await until(() => JSON.parse(fs.readFileSync(f.journal, "utf8")).targets.length === 1);
  const journal = JSON.parse(fs.readFileSync(f.journal, "utf8"));
  expect(journal.targets[0].target_id).toBe("new");
  expect(journal.targets[0].pending).toBe(true);
  expect(f.calls.filter(c => c.rpc === "monitoring_target_secret").length).toBe(secretCalls);
  expect(f.submits.length).toBe(1);
  expect(JSON.parse(fs.readFileSync(path.join(f.box.projectDir, ".pgai-report-projects.json"), "utf8"))).toEqual({ projects: [] });
  await resumed.stop();
});
