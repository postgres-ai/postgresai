import * as fs from "node:fs";
import * as path from "node:path";
import { loadInstances } from "./instances";
import { writePrivateFileAtomic } from "./atomic-file";
import { withTargetsLock } from "./targets-lock";

interface Target { target_id: string; name: string; adopt: boolean }
interface ManagedTarget { target_id: string; name: string; local_name?: string; pending?: boolean; adopted?: boolean }
interface Journal { targets: ManagedTarget[]; desired?: Array<{ target_id: string; name: string }> }
interface Job { id: string | number; generation: number; targets: Target[] }
interface WorkerHooks {
  add: (url: string, name: string) => Promise<boolean>;
  remove: (name: string) => Promise<boolean>;
}

class ChannelError extends Error {
  constructor(readonly longBackoff: boolean) { super("Target channel request failed"); }
}

export function targetChannelConfig(projectDir: string, env = process.env): { baseURL: string; token: string } {
  const file = path.join(projectDir, ".pgwatch-config");
  const values = new Map<string, string>();
  const text = fs.existsSync(file) ? fs.readFileSync(file, "utf8").replace(/^\ufeff/, "") : "";
  for (const line of text.split("\n")) {
    const equals = line.indexOf("=");
    if (line.startsWith("#") || equals < 0) continue;
    const key = line.slice(0, equals);
    if (!values.has(key)) values.set(key, line.slice(equals + 1).trim());
  }
  const token = values.get("api_key") || "";
  const baseURL = (values.get("api_base_url") || env.PGAI_API_BASE_URL?.trim() || "https://postgres.ai/api/general").replace(/\/+$/, "");
  let url: URL;
  try { url = new URL(baseURL); } catch { throw new ChannelError(true); }
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const loopback = host === "localhost" || host === "::1" || /^127\.\d+\.\d+\.\d+$/.test(host);
  if (!token || !(url.protocol === "https:" || (url.protocol === "http:" && loopback))) throw new ChannelError(true);
  return { baseURL, token };
}

export function targetChannelHeaders(baseURL: string, token: string, env = process.env): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json", Accept: "application/json", "access-token": token };
  const id = env.CF_ACCESS_CLIENT_ID?.trim();
  const secret = env.CF_ACCESS_CLIENT_SECRET?.trim();
  if (new URL(baseURL).protocol === "https:" && id && secret) {
    headers["CF-Access-Client-Id"] = id;
    headers["CF-Access-Client-Secret"] = secret;
  }
  return headers;
}

async function rpc(config: ReturnType<typeof targetChannelConfig>, name: string, body: unknown, signal: AbortSignal): Promise<any> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) controller.abort();
  const timeout = setTimeout(abort, 30000);
  try {
    const response = await fetch(`${config.baseURL}/rpc/${name}`, {
      method: "POST", headers: targetChannelHeaders(config.baseURL, config.token),
      body: JSON.stringify(body), redirect: "manual", signal: controller.signal,
    });
    const reader = response.body?.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    if (reader) {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          size += value.length;
          if (size > 4 * 1024 * 1024) { await reader.cancel(); throw new ChannelError(false); }
          chunks.push(value);
        }
      } finally { reader.releaseLock(); }
    }
    let result: any;
    try { result = JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { result = null; }
    if (!response.ok) {
      const code = typeof result?.code === "string" ? result.code : "";
      const transient = ["55P03", "40P01", "40001", "57014"].includes(code) ||
        (!(["PT404", "PT401", "PT402", "PT400"].includes(code)) && (response.status >= 500 || response.status === 429));
      throw new ChannelError(!transient);
    }
    if (!result || typeof result !== "object") throw new ChannelError(false);
    return result;
  } catch (err) {
    // Remote messages and transport errors may echo a one-use secret or token.
    throw err instanceof ChannelError ? err : new ChannelError(false);
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener("abort", abort);
  }
}

function readJournal(file: string): Journal {
  if (!fs.existsSync(file)) return { targets: [] };
  const value = JSON.parse(fs.readFileSync(file, "utf8"));
  const journal = Array.isArray(value) ? { targets: value } : value;
  const entries: unknown = journal?.targets;
  if (!Array.isArray(entries) || entries.some(e => !e || typeof e.target_id !== "string" ||
    typeof e.name !== "string" || !e.name || e.name !== e.name.trim() ||
    (e.local_name !== undefined && (typeof e.local_name !== "string" || !e.local_name || e.local_name !== e.local_name.trim())) ||
    (e.pending !== undefined && typeof e.pending !== "boolean") ||
    (e.adopted !== undefined && typeof e.adopted !== "boolean")) ||
    new Set(entries.map(e => e.target_id)).size !== entries.length ||
    new Set(entries.map(e => e.name)).size !== entries.length ||
    new Set(entries.map(e => e.local_name || e.name)).size !== entries.length) throw new ChannelError(true);
  const desired: unknown = journal.desired;
  if (desired !== undefined && (!Array.isArray(desired) || desired.some(t => !t || typeof t.target_id !== "string" || !t.target_id ||
    typeof t.name !== "string" || !t.name || t.name !== t.name.trim()) ||
    new Set(desired.map(t => t.target_id)).size !== desired.length ||
    new Set(desired.map(t => t.name)).size !== desired.length)) throw new ChannelError(true);
  return journal;
}

function saveJournal(file: string, journal: Journal) {
  writePrivateFileAtomic(file, JSON.stringify(journal) + "\n");
}

async function removeUndesired(journal: Journal, file: string, hooks: WorkerHooks, signal: AbortSignal) {
  if (!journal.desired) return;
  const desired = new Set(journal.desired.map(t => t.target_id));
  for (const entry of [...journal.targets]) {
    if (signal.aborted) break;
    if (desired.has(entry.target_id)) continue;
    try {
      if (!(await hooks.remove(entry.local_name || entry.name))) throw new Error();
      journal.targets = journal.targets.filter(e => e.target_id !== entry.target_id);
      saveJournal(file, journal);
    } catch {
      // Keep ownership for an empty-poll retry; removal failures are outside the desired set.
    }
  }
}

function validJob(job: any): job is Job {
  return job && (typeof job.id === "string" || Number.isSafeInteger(job.id)) && Number.isSafeInteger(job.generation) &&
    Array.isArray(job.targets) && job.targets.every((t: any) => t && typeof t.target_id === "string" && t.target_id &&
      typeof t.name === "string" && t.name && t.name === t.name.trim() && typeof t.adopt === "boolean") &&
    new Set(job.targets.map((t: Target) => t.target_id)).size === job.targets.length &&
    new Set(job.targets.map((t: Target) => t.name)).size === job.targets.length;
}

async function applyJob(projectDir: string, instancesFile: string, job: Job, hooks: WorkerHooks,
  config: ReturnType<typeof targetChannelConfig>, signal: AbortSignal) {
  return withTargetsLock(projectDir, async () => {
    const journalFile = path.join(path.dirname(instancesFile), ".pgai-managed-targets.json");
    const journal = readJournal(journalFile);
    journal.desired = job.targets.map(t => ({ target_id: t.target_id, name: t.name }));
    const save = () => saveJournal(journalFile, journal);
    save();
    const failed: Array<{ target_id: string; error: string }> = [];
    const applied: string[] = [];
    let adopted = false;
    for (const target of job.targets) {
      if (signal.aborted) break;
      try {
        let entry = journal.targets.find(e => e.target_id === target.target_id);
        const instances = loadInstances(instancesFile);
        let local = instances.find(i => i.name === (entry?.local_name || target.name));
        if ((entry && entry.name !== target.name) || (!entry && journal.targets.some(e =>
          e.name === target.name || (e.local_name || e.name) === target.name))) throw new Error();
        if (target.adopt) {
          if (adopted) throw new Error();
          adopted = true;
          if (!local) {
            const candidates = instances.filter(i => !journal.targets.some(e => (e.local_name || e.name) === i.name));
            if (candidates.length !== 1) throw new Error();
            local = candidates[0];
            if (entry) {
              entry.local_name = local.name;
              entry.pending = false;
              entry.adopted = true;
              save();
            }
          }
        }
        if (!entry) {
          if (local && !target.adopt) throw new Error();
          entry = { target_id: target.target_id, name: target.name, local_name: local?.name || target.name,
            pending: !local, adopted: !!local && target.adopt };
          // Persist ownership before redeeming the secret or mutating the stack.
          journal.targets.push(entry);
          save();
        }
        if (entry.pending || !local) {
          if (entry.adopted) throw new Error();
          const connStr = local?.conn_str || (await rpc(config, "monitoring_target_secret", { target_id: target.target_id }, signal)).db_url;
          if (typeof connStr !== "string" || !/^postgres(ql)?:\/\//i.test(connStr)) throw new Error();
          if (signal.aborted || !(await hooks.add(connStr, target.name))) throw new Error();
          if (!loadInstances(instancesFile).some(i => i.name === target.name)) throw new Error();
          entry.pending = false;
          save();
        }
        applied.push(target.target_id);
      } catch {
        failed.push({ target_id: target.target_id, error: "Could not add or reconcile the monitoring target" });
      }
    }
    await removeUndesired(journal, journalFile, hooks, signal);
    save();
    const projects = job.targets.filter(t => applied.includes(t.target_id)).map(t => {
      const entry = journal.targets.find(e => e.target_id === t.target_id)!;
      return { project: t.name, source: entry.local_name || entry.name };
    });
    writePrivateFileAtomic(path.join(path.dirname(instancesFile), ".pgai-report-projects.json"),
      JSON.stringify({ projects }) + "\n");
    return { job_id: job.id, generation: job.generation, applied, failed };
  }, signal);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    const done = () => { clearTimeout(timer); signal.removeEventListener("abort", done); resolve(); };
    const timer = setTimeout(done, ms);
    signal.addEventListener("abort", done, { once: true });
    if (signal.aborted) done();
  });
}

export async function runTargetsSyncWorker(projectDir: string, instancesFile: string, hooks: WorkerHooks,
  signal: AbortSignal, log: (message: string) => void = console.error): Promise<void> {
  let failures = 0;
  while (!signal.aborted) {
    let wait = 30000;
    try {
      const config = targetChannelConfig(projectDir);
      const poll = await rpc(config, "monitoring_target_poll", { protocol_version: 1 }, signal);
      if (poll.job !== null && !validJob(poll.job)) throw new ChannelError(true);
      if (typeof poll.next_poll_ms !== "number" || !Number.isFinite(poll.next_poll_ms)) throw new ChannelError(true);
      wait = Math.max(5000, Math.min(600000, poll.next_poll_ms));
      if (poll.job) {
        const submit = await applyJob(projectDir, instancesFile, poll.job, hooks, config, signal);
        if (!signal.aborted) {
          const reply = await rpc(config, "monitoring_target_submit", submit, signal);
          if (reply.ok !== true) throw new ChannelError(false);
        }
      } else {
        await withTargetsLock(projectDir, async () => {
          const journalFile = path.join(path.dirname(instancesFile), ".pgai-managed-targets.json");
          await removeUndesired(readJournal(journalFile), journalFile, hooks, signal);
        }, signal);
      }
      failures = 0;
    } catch (err) {
      if (signal.aborted) break;
      failures++;
      wait = err instanceof ChannelError && err.longBackoff ? 600000 : Math.min(60000, 2000 * 2 ** Math.min(failures - 1, 5));
      log("Target synchronization failed; retrying");
    }
    await sleep(Math.max(5000, Math.min(600000, wait * (0.8 + Math.random() * 0.4))), signal);
  }
}
