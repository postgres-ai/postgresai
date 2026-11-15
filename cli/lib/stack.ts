/**
 * What the CLI knows about the monitoring stack's compose file: which services
 * mark it as the current stack (pgai-mon) or the old one (pgwatch +
 * VictoriaMetrics), and what an upgrade from the old one leaves behind.
 */

import * as yaml from "js-yaml";

/**
 * The one service of the stack that collects, stores and serves metrics. Its
 * presence in a compose file is what marks the file as this stack's.
 */
export const PGAI_MON_SERVICE = "pgai-mon";

/** The service whose presence marks a compose file as the old stack's. */
export const OLD_STACK_KEYSTONE = "sink-prometheus";

/**
 * Volumes only the old stack wrote (metrics and query texts). An upgrade leaves
 * them in place; removing them is the operator's call.
 */
export const OLD_STACK_VOLUMES = ["victoria_metrics_data", "sink_postgres_data"];

/**
 * Every compose profile the old stack defined. Its `down` runs with all of them
 * active, because compose reaches a profile-gated container only while its
 * profile is.
 */
export const OLD_STACK_PROFILES = ["demo", "host-metrics", "instance-jobs", "rds"];

/**
 * Does this compose document define `service`? False for anything that is not
 * a parseable compose document with a `services` map.
 */
export function composeDefinesService(text: string | null | undefined, service: string): boolean {
  if (!text || !text.trim()) return false;
  let doc: unknown;
  try {
    doc = yaml.load(text);
  } catch {
    return false;
  }
  const services = (doc as { services?: unknown } | null)?.services;
  return !!services && typeof services === "object" && Object.prototype.hasOwnProperty.call(services, service);
}

/**
 * The compose project name, as compose derives it: COMPOSE_PROJECT_NAME when
 * set, else the project directory's name, lower-cased with every character but
 * [a-z0-9_-] dropped.
 */
export function composeProjectName(projectDirName: string, explicit?: string | null): string {
  const name = explicit?.trim() || projectDirName;
  return name.toLowerCase().replace(/[^a-z0-9_-]/g, "");
}

/** The old stack's volume names in a project. */
export function oldStackVolumeNames(project: string): string[] {
  return OLD_STACK_VOLUMES.map((v) => `${project}_${v}`);
}

/**
 * What `mon update` prints after switching an install to pgai-mon. `volumes`
 * are the old stack's volumes that still exist (empty when none do).
 */
export function stackSwitchNotice(volumes: string[]): string {
  const lines = [
    "Switched this install from pgwatch + VictoriaMetrics to pgai-mon.",
    "Metric history starts fresh: nothing was migrated from the old stack.",
  ];
  if (volumes.length > 0) {
    lines.push(
      `The old stack's data is still in ${volumes.length === 1 ? "its volume" : "its volumes"}, untouched: ${volumes.join(", ")}`,
      "Remove it once you no longer need it:",
      `  docker volume rm ${volumes.join(" ")}`,
    );
  }
  return lines.join("\n");
}

/**
 * The one line an install switched from the old stack needs about Grafana's
 * port, or null when nothing changed for it. The old compose published Grafana
 * on every interface when GRAFANA_BIND_HOST was unset; this one publishes it on
 * 127.0.0.1 then. A set value (empty included) means the same as before.
 */
export function grafanaBindNotice(bindHost: string | null | undefined): string | null {
  if (bindHost !== undefined && bindHost !== null) return null;
  return (
    "Grafana now listens on 127.0.0.1:3000 only (before, on all interfaces): reach it through an SSH tunnel " +
    "(ssh -L 3000:localhost:3000 <this host>), or, to expose it on purpose, set GRAFANA_BIND_HOST in .env " +
    "(e.g. GRAFANA_BIND_HOST=0.0.0.0:) and run 'postgresai mon restart grafana'."
  );
}

/**
 * The one line saying where metrics are kept. Unset (blank counts as unset)
 * means the local volume: fine for a laptop demo, while the product default is
 * a bucket. A set store is named without credentials or query string.
 */
export function storeNotice(store: string | null | undefined): string {
  if (store && store.trim()) return `Metrics history goes to ${redactStoreUrl(store.trim())}`;
  return "Metrics are stored on this machine (volume pgai_mon_data); set PGAI_MON_STORE=s3://<bucket>/<prefix> in .env to keep them in S3.";
}

/** A store URL without user info or query string. */
export function redactStoreUrl(store: string): string {
  const noQuery = store.split(/[?#]/)[0];
  return noQuery.replace(/^([a-z][a-z0-9+.-]*:\/\/)[^/@]*@/i, "$1");
}

/** One container as `docker compose ps -a --format json` reports it. */
export interface ComposePsEntry {
  ID?: string;
  Service: string;
  State: string;
  Health?: string;
  ExitCode?: number;
}

/** Services that run once and exit 0 when they did their job. */
export const ONE_SHOT_SERVICES = ["config-init"];

/**
 * Parse `docker compose ps --format json`: a JSON array (compose < 2.21) or
 * one JSON object per line. Lines that are not JSON objects are skipped.
 */
export function parseComposePs(out: string): ComposePsEntry[] {
  const text = out.trim();
  if (!text) return [];
  const keep = (v: unknown): v is ComposePsEntry =>
    !!v && typeof v === "object" && typeof (v as ComposePsEntry).Service === "string";
  if (text.startsWith("[")) {
    try {
      const arr = JSON.parse(text);
      return Array.isArray(arr) ? arr.filter(keep) : [];
    } catch {
      return [];
    }
  }
  const entries: ComposePsEntry[] = [];
  for (const line of text.split("\n")) {
    try {
      const v = JSON.parse(line);
      if (keep(v)) entries.push(v);
    } catch {
      // not JSON: skip
    }
  }
  return entries;
}

/**
 * The expected services that are not running and healthy, as
 * "<service>: <state>" (missing, created, exited (1), restarting, unhealthy,
 * starting, ...). Empty means the whole stack is up. A one-shot service is
 * fine once it exited 0, or when its container is gone.
 */
export function stackProblems(expected: string[], ps: ComposePsEntry[]): string[] {
  const problems: string[] = [];
  for (const service of expected) {
    const entries = ps.filter((e) => e.Service === service);
    if (entries.length === 0) {
      // A one-shot removed after its run (`rm -f -s config-init`) did its job.
      if (!ONE_SHOT_SERVICES.includes(service)) problems.push(`${service}: missing`);
      continue;
    }
    for (const e of entries) {
      const state = (e.State || "").toLowerCase();
      const health = (e.Health || "").toLowerCase();
      if (ONE_SHOT_SERVICES.includes(service)) {
        if (state === "exited" && (e.ExitCode ?? 0) === 0) continue;
        if (state === "running") continue;
      } else if (state === "running" && (health === "" || health === "healthy")) {
        continue;
      }
      const why = state === "running" ? health : state === "exited" ? `exited (${e.ExitCode ?? "?"})` : state || "unknown";
      problems.push(`${service}: ${why}`);
    }
  }
  return problems;
}

/** Services whose containers failed: unhealthy, restarting, or exited non-zero. */
export function failedServices(ps: ComposePsEntry[]): string[] {
  const failed = new Set<string>();
  for (const e of ps) {
    const state = (e.State || "").toLowerCase();
    const health = (e.Health || "").toLowerCase();
    if (health === "unhealthy" || state === "restarting" || state === "dead" || (state === "exited" && (e.ExitCode ?? 0) !== 0)) {
      failed.add(e.Service);
    }
  }
  return [...failed];
}

/** Restarts since an `up` began after which the install stops waiting. */
export const RESTART_LOOP_LIMIT = 3;

/** One container's restart count, as `docker inspect` reports it. */
export interface ContainerRestarts {
  id: string;
  service: string;
  restarts: number;
}

/**
 * Services with a container that restarted `limit` times or more since the
 * baseline (container id -> restart count at the first poll). A container
 * not in the baseline is new, so its baseline is 0. Counting from the
 * baseline keeps a restart from an old run from being blamed.
 */
export function restartLoops(
  baseline: Record<string, number>,
  now: ContainerRestarts[],
  limit = RESTART_LOOP_LIMIT,
): { service: string; restarts: number }[] {
  const worst = new Map<string, number>();
  for (const c of now) {
    const n = c.restarts - (baseline[c.id] ?? 0);
    if (n >= limit && n > (worst.get(c.service) ?? 0)) worst.set(c.service, n);
  }
  return [...worst].map(([service, restarts]) => ({ service, restarts }));
}
