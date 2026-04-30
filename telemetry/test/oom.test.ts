import { describe, test, expect } from "bun:test";
import { parseOomCount, collectOomCount } from "../lib/collectors/oom";

describe("parseOomCount", () => {
  test("returns 0 on empty input", () => {
    expect(parseOomCount("")).toBe(0);
  });

  test("counts distinct PIDs from 'Out of memory: Killed process N' lines", () => {
    const dump = [
      "Apr 28 09:10:11 host kernel: Out of memory: Killed process 1234 (postgres) total-vm:...",
      "Apr 28 09:10:12 host kernel: Out of memory: Killed process 5678 (node) total-vm:...",
      "Apr 28 09:11:00 host kernel: just an unrelated kernel message",
    ].join("\n");
    expect(parseOomCount(dump)).toBe(2);
  });

  test("counts a multi-line OOM episode as ONE event (canonical + invoked + reaper)", () => {
    // A single real OOM kill emits multiple matching lines. We must count
    // events, not lines. Naive line counting would return 3 here.
    const dump = [
      "Apr 28 09:10:11 host kernel: postgres invoked oom-killer: gfp_mask=0xcc0...",
      "Apr 28 09:10:11 host kernel: CPU: 0 PID: 1234 Comm: postgres ...",
      "Apr 28 09:10:12 host kernel: Out of memory: Killed process 1234 (postgres) total-vm:8388608kB",
      "Apr 28 09:10:12 host kernel: oom_reaper: reaped process 1234 (postgres)",
    ].join("\n");
    expect(parseOomCount(dump)).toBe(1);
  });

  test("dedups repeated 'Killed process' lines for the same PID", () => {
    const dump = [
      "Apr 28 09:10:12 host kernel: Out of memory: Killed process 1234 (postgres) ...",
      "Apr 28 09:10:13 host kernel: Out of memory: Killed process 1234 (postgres) ...",
    ].join("\n");
    expect(parseOomCount(dump)).toBe(1);
  });

  test("falls back to counting 'oom-killer' invocations when the canonical line is rate-limited", () => {
    // No "Killed process N" line — only the surrounding context survives.
    // We treat each invoked-oom-killer line as one event.
    const dump = [
      "Apr 28 09:10:11 host kernel: postgres invoked oom-killer: gfp_mask=...",
      "Apr 28 09:11:11 host kernel: redis invoked oom-killer: gfp_mask=...",
    ].join("\n");
    expect(parseOomCount(dump)).toBe(2);
  });

  test("ignores unrelated kernel messages", () => {
    const dump = [
      "Apr 28 09:10:11 host kernel: TCP: out of memory -- consider tuning tcp_mem",
      "Apr 28 09:10:12 host kernel: docker0: port 1(veth1) entered disabled state",
    ].join("\n");
    expect(parseOomCount(dump)).toBe(0);
  });
});

describe("collectOomCount", () => {
  test("uses the provided lookback window via the exec injector", async () => {
    let receivedArgs: string[] = [];
    const exec = async (args: string[]) => {
      receivedArgs = args;
      return "Apr 28 09:10:11 host kernel: Out of memory: Killed process 1 (a)";
    };
    const count = await collectOomCount({ lookback: "24 hours ago", exec });
    expect(count).toBe(1);
    expect(receivedArgs).toEqual(["-k", "--no-pager", "--since", "24 hours ago"]);
  });

  test("returns 0 when journalctl exec rejects (e.g. binary missing)", async () => {
    const exec = async () => {
      throw new Error("journalctl: not found");
    };
    const count = await collectOomCount({ lookback: "24 hours ago", exec });
    expect(count).toBe(0);
  });
});
