/**
 * Free-RAM collector.
 *
 * Reads /proc/meminfo and returns MemAvailable in bytes. MemAvailable is
 * the canonical "RAM that can be allocated without swapping out" estimate
 * the kernel exposes -- preferred over MemFree, which excludes reclaimable
 * cache and badly underestimates.
 *
 * If MemAvailable is missing (very old kernels), falls back to MemFree.
 */

import { readFile } from "node:fs/promises";

export function parseMemAvailableBytes(meminfo: string): number {
  if (!meminfo) return 0;
  const lookup = (key: string): number | null => {
    const re = new RegExp(`^${key}:\\s+(\\d+)\\s*kB`, "im");
    const m = meminfo.match(re);
    if (!m || m[1] === undefined) return null;
    const kb = Number.parseInt(m[1], 10);
    if (!Number.isFinite(kb)) return null;
    return kb * 1024;
  };
  return lookup("MemAvailable") ?? lookup("MemFree") ?? 0;
}

export async function collectFreeRamBytes(opts: {
  meminfoPath?: string;
  read?: (path: string) => Promise<string>;
} = {}): Promise<number> {
  const path = opts.meminfoPath ?? "/proc/meminfo";
  const read = opts.read ?? ((p) => readFile(p, "utf-8"));
  try {
    const text = await read(path);
    return parseMemAvailableBytes(text);
  } catch {
    return 0;
  }
}
