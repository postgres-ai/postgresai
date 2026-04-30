import { describe, test, expect } from "bun:test";
import { parseMemAvailableBytes, collectFreeRamBytes } from "../lib/collectors/memory";

const SAMPLE_MEMINFO = `MemTotal:       16336948 kB
MemFree:         1048576 kB
MemAvailable:    8388608 kB
Buffers:          262144 kB
Cached:          4194304 kB
SwapCached:            0 kB
`;

describe("parseMemAvailableBytes", () => {
  test("returns 0 on empty input", () => {
    expect(parseMemAvailableBytes("")).toBe(0);
  });

  test("returns MemAvailable in bytes", () => {
    // 8388608 kB * 1024 = 8589934592
    expect(parseMemAvailableBytes(SAMPLE_MEMINFO)).toBe(8589934592);
  });

  test("falls back to MemFree when MemAvailable is missing", () => {
    const meminfo = `MemTotal: 16336948 kB
MemFree:  524288 kB
`;
    // 524288 kB * 1024 = 536870912
    expect(parseMemAvailableBytes(meminfo)).toBe(536870912);
  });

  test("returns 0 when neither key is present", () => {
    const meminfo = `MemTotal: 16336948 kB
Buffers:    262144 kB
`;
    expect(parseMemAvailableBytes(meminfo)).toBe(0);
  });
});

describe("collectFreeRamBytes", () => {
  test("uses the injected reader and parses correctly", async () => {
    let readPath = "";
    const read = async (path: string) => {
      readPath = path;
      return SAMPLE_MEMINFO;
    };
    const bytes = await collectFreeRamBytes({ meminfoPath: "/tmp/meminfo", read });
    expect(bytes).toBe(8589934592);
    expect(readPath).toBe("/tmp/meminfo");
  });

  test("returns 0 when the reader rejects (e.g. /proc not present on macOS)", async () => {
    const read = async () => {
      throw new Error("ENOENT");
    };
    const bytes = await collectFreeRamBytes({ read });
    expect(bytes).toBe(0);
  });
});
