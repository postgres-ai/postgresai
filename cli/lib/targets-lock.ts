import * as fs from "node:fs";
import * as path from "node:path";
import { spawn } from "node:child_process";

// The helper owns the kernel lock until its stdin closes, including on a CLI crash.
export async function withTargetsLock<T>(projectDir: string, action: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) throw new Error("Target synchronization stopped");
  const file = path.join(projectDir, ".pgai-targets.lock");
  fs.closeSync(fs.openSync(file, "a", 0o600));
  fs.chmodSync(file, 0o600);
  const env = { ...process.env };
  delete env.PGAI_DB_URL;
  const command = process.platform === "darwin" ? "/usr/bin/python3" : "flock";
  const args = process.platform === "darwin" ? ["-c", [
    "import fcntl, sys",
    "with open(sys.argv[1], 'a') as lock:",
    "    fcntl.flock(lock, fcntl.LOCK_EX)",
    "    print('locked', flush=True)",
    "    sys.stdin.read()",
  ].join("\n"), file] : ["-x", file, "sh", "-c", 'printf "locked\\n"; cat >/dev/null'];
  const helper = spawn(command, args, {
    env, stdio: ["pipe", "pipe", "ignore"],
  });
  let ready = false;
  const closed = new Promise<void>((resolve) => {
    helper.on("close", () => resolve());
  });
  const stop = () => {
    if (!ready) { helper.stdin.destroy(); helper.kill("SIGTERM"); }
  };
  let cancelWait: (() => void) | undefined;
  try {
    await new Promise<void>((resolve, reject) => {
      helper.once("error", () => reject(new Error("Stack lock requires util-linux flock on Linux or python3 on macOS")));
      helper.once("exit", () => reject(new Error("Stack lock helper exited")));
      helper.stdout.once("data", () => {
        ready = true;
        if (signal?.aborted) reject(new Error("Target synchronization stopped"));
        else resolve();
      });
      cancelWait = () => {
        if (!ready) reject(new Error("Target synchronization stopped"));
      };
      signal?.addEventListener("abort", cancelWait, { once: true });
      signal?.addEventListener("abort", stop, { once: true });
      if (signal?.aborted) { stop(); reject(new Error("Target synchronization stopped")); }
    });
    return await action();
  } finally {
    signal?.removeEventListener("abort", stop);
    if (cancelWait) signal?.removeEventListener("abort", cancelWait);
    helper.stdin.end();
    if (!ready) helper.kill("SIGTERM");
    await closed;
  }
}
