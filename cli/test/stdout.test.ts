import { describe, expect, test } from "bun:test";
import * as path from "path";
import { isEpipe, writeStdout } from "../lib/stdout";

describe("writeStdout (#355)", () => {
  test("resolves only after the stream's write callback fires", async () => {
    const original = process.stdout.write;
    let flushed = false;
    // Simulate a slow pipe: the callback fires on a later tick.
    process.stdout.write = ((chunk: unknown, cb?: (err?: Error) => void) => {
      setTimeout(() => { flushed = true; cb?.(); }, 20);
      return false;
    }) as typeof process.stdout.write;
    try {
      const p = writeStdout("x".repeat(1024));
      expect(flushed).toBe(false);
      await p;
      expect(flushed).toBe(true);
    } finally {
      process.stdout.write = original;
    }
  });

  test("rejects on a write error and absorbs the follow-up 'error' event", async () => {
    const original = process.stdout.write;
    const listenersBefore = process.stdout.listenerCount("error");
    const epipe = Object.assign(new Error("write EPIPE"), { code: "EPIPE" });
    process.stdout.write = ((chunk: unknown, cb?: (err?: Error) => void) => {
      cb?.(epipe);
      return false;
    }) as typeof process.stdout.write;
    try {
      await expect(writeStdout("x")).rejects.toThrow("EPIPE");
      // The noop listener is what keeps Node from crashing on the emit that follows.
      expect(process.stdout.listenerCount("error")).toBe(listenersBefore + 1);
      process.stdout.emit("error", epipe); // must not throw
      expect(isEpipe(epipe)).toBe(true);
      expect(isEpipe(new Error("other"))).toBe(false);
    } finally {
      process.stdout.write = original;
      process.stdout.removeAllListeners("error");
    }
  });

  test("a real child process delivers the whole payload through a pipe", () => {
    const helper = path.resolve(import.meta.dir, "../lib/stdout.ts");
    const size = 300_000;
    const script = `import { writeStdout } from ${JSON.stringify(helper)}; await writeStdout("x".repeat(${size}) + "\\n");`;
    const r = Bun.spawnSync([process.execPath, "-e", script], { stdout: "pipe", stderr: "pipe" });
    expect(r.exitCode).toBe(0);
    expect(r.stdout.length).toBe(size + 1);
  });
});
