import { createHash } from "node:crypto";
import { chmodSync, lstatSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { dump, load } from "js-yaml";

type ScrapeConfig = { static_configs: { targets: string[]; labels: Record<string, string> }[] } & Record<string, unknown>;

/**
 * One scrape job as YAML, stamped with a `__pgai_rev` label. The "__" prefix keeps
 * it out of stored series, but the targets API lists it, so a reload can be
 * matched to this exact file.
 */
export function scrapeFileText(config: ScrapeConfig): string {
  const revision = `r${createHash("sha256").update(dump([config], { lineWidth: -1 })).digest("hex").slice(0, 16)}`;
  Object.assign(config.static_configs[0].labels, { __pgai_rev: revision });
  return dump([config], { lineWidth: -1 });
}

export function scrapeRevision(projectDir: string, file: string): string {
  const [config] = load(readFileSync(join(projectDir, "host-metrics", file), "utf8")) as any[];
  const revision = config?.static_configs?.[0]?.labels?.__pgai_rev;
  if (typeof revision !== "string" || !/^r[0-9a-f]{16}$/.test(revision)) throw new Error(`host-metrics/${file} has no __pgai_rev label.`);
  return revision;
}

/**
 * The host-metrics directory, 0700 also when an older CLI created it 0755: it
 * holds the ClickHouse API key. Never through a symlink, which would retarget
 * the chmod and the key.
 */
export function hostMetricsDir(projectDir: string): string {
  const dir = join(projectDir, "host-metrics");
  if (lstatSync(dir, { throwIfNoEntry: false })?.isSymbolicLink()) throw new Error(`${dir}: host-metrics must be a directory, not a symlink.`);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  chmodSync(dir, 0o700);
  return dir;
}

export function writeScrapeFile(dir: string, file: string, text: string): void {
  writeFileSync(join(dir, file), text, { mode: 0o644 });
  chmodSync(join(dir, file), 0o644);
}

// Runs inside sink-prometheus: sh -c SCRIPT sh NEEDLE present|absent. VictoriaMetrics
// applies a SIGHUP asynchronously and skips a scrape file it cannot parse while
// still counting the reload as successful, so only the targets list shows the
// outcome. Polls it until NEEDLE is present (or absent), for at most ~10s.
export const HOST_METRICS_VERIFY_SCRIPT = `auth="Authorization: Basic $(printf '%s:%s' "$VM_AUTH_USERNAME" "$VM_AUTH_PASSWORD" | base64 | tr -d '\\n')"; end=$(($(date +%s) + 10)); while [ "$(date +%s)" -lt "$end" ]; do if t=$(wget -qO- -T 2 --header "$auth" http://127.0.0.1:9090/api/v1/targets); then case "$t" in '{"status":"success"'*) case "$t" in *"$1"*) [ "$2" = present ] && exit 0 ;; *) [ "$2" = absent ] && exit 0 ;; esac ;; esac; fi; sleep 0.5; done; exit 1`;

export const SUPABASE_JOB = "supabase-host-metrics";

/** Scrape job for the instance-jobs relay. The relay serves one Supabase project, so a box has one. */
export function renderSupabaseScrapeConfig({ cluster, nodeName, target = "instance-jobs:9188" }: { cluster: string; nodeName: string; target?: string }): string {
  return scrapeFileText({
    job_name: SUPABASE_JOB,
    metrics_path: "/supabase/metrics",
    scrape_interval: "60s", scrape_timeout: "20s",
    static_configs: [{ targets: [target], labels: { cluster, node_name: nodeName } }],
    // node_time_seconds is the host clock the host_* rates are divided by.
    metric_relabel_configs: [{ source_labels: ["__name__"], regex: "node_cpu_seconds_total|node_time_seconds|node_memory_.*|node_disk_.*|node_network_.*_bytes_total|node_filesystem_.*|node_load.*", action: "keep" }],
  });
}

/**
 * The RDS instance an endpoint names: <instance>.<12 characters>.<region>.rds.amazonaws.com.
 * null for a cluster, reader, custom or proxy endpoint (its second label has a
 * hyphen), which names no instance; undefined when the host is not RDS.
 */
export function rdsInstance(host: string): { id: string; region: string } | null | undefined {
  if (!/\.rds\.amazonaws\.com\.?$/i.test(host)) return undefined;
  const m = /^([a-z][a-z0-9-]*)\.[a-z0-9]{12}\.([a-z]{2}(?:-[a-z]+)+-\d+)\.rds\.amazonaws\.com\.?$/i.exec(host);
  return m ? { id: m[1].toLowerCase(), region: m[2].toLowerCase() } : null;
}
