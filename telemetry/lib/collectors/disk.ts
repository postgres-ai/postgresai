/**
 * Free-disk collector.
 *
 * Uses fs.statfs (Node 18+) to get available bytes for the data volume.
 * The platform side compares this against a low-disk threshold so we just
 * need a single integer in bytes.
 */

import { statfs } from "node:fs/promises";

export interface StatfsResult {
  bavail: number | bigint;
  bsize: number | bigint;
}

export async function collectFreeDiskBytes(opts: {
  path?: string;
  statfs?: (path: string) => Promise<StatfsResult>;
} = {}): Promise<number> {
  const path = opts.path ?? "/";
  const fn = opts.statfs ?? ((p) => statfs(p) as unknown as Promise<StatfsResult>);
  try {
    const r = await fn(path);
    const bavail = typeof r.bavail === "bigint" ? Number(r.bavail) : r.bavail;
    const bsize = typeof r.bsize === "bigint" ? Number(r.bsize) : r.bsize;
    if (!Number.isFinite(bavail) || !Number.isFinite(bsize)) return 0;
    return bavail * bsize;
  } catch {
    return 0;
  }
}
