import { describe, test, expect } from "bun:test";
import { collectFreeDiskBytes } from "../lib/collectors/disk";

describe("collectFreeDiskBytes", () => {
  test("multiplies bavail * bsize", async () => {
    const fakeStatfs = async () => ({ bavail: 1024, bsize: 4096 });
    const bytes = await collectFreeDiskBytes({ path: "/", statfs: fakeStatfs });
    expect(bytes).toBe(1024 * 4096);
  });

  test("handles BigInt-typed bavail/bsize from statfs", async () => {
    const fakeStatfs = async () => ({ bavail: 2048n, bsize: 4096n });
    const bytes = await collectFreeDiskBytes({ path: "/data", statfs: fakeStatfs });
    expect(bytes).toBe(2048 * 4096);
  });

  test("returns 0 when statfs rejects", async () => {
    const fakeStatfs = async () => {
      throw new Error("ENOENT");
    };
    const bytes = await collectFreeDiskBytes({ path: "/nope", statfs: fakeStatfs });
    expect(bytes).toBe(0);
  });

  test("uses '/' as the default path", async () => {
    let receivedPath = "";
    const fakeStatfs = async (p: string) => {
      receivedPath = p;
      return { bavail: 1, bsize: 1 };
    };
    await collectFreeDiskBytes({ statfs: fakeStatfs });
    expect(receivedPath).toBe("/");
  });
});
