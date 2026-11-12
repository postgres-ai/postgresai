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
 * The one line saying where metrics are kept when `PGAI_MON_STORE` is unset
 * (blank counts as unset), else null. Unset means the local volume: fine for a
 * laptop demo, while the product default is a bucket.
 */
export function storeNotice(store: string | null | undefined): string | null {
  if (store && store.trim()) return null;
  return "Metrics are stored on this machine (volume pgai_mon_data); set PGAI_MON_STORE=s3://<bucket>/<prefix> in .env to keep them in S3.";
}
