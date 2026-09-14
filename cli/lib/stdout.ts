/**
 * Write a payload to stdout and resolve once the stream has accepted it.
 *
 * `console.log` returns before a large payload reaches a pipe; when the
 * process then ends naturally, the still-queued tail can be lost (observed on
 * bun 1.4.x: a 240 KB `checkup --json` cut at 219 KB in 2 of 5 runs, 0 of 10
 * with this helper; see #355). Awaiting the write callback keeps the process
 * alive until the kernel has accepted the whole payload.
 *
 * On a write error (typically EPIPE when the reader closed early, e.g. `| head`)
 * Node emits an 'error' event on the stream one tick after the callback; a
 * noop listener added inside the callback absorbs it, the same way Node's own
 * console does, so the caller sees a rejection instead of a crash.
 */
export function writeStdout(text: string): Promise<void> {
  return new Promise((resolve, reject) => {
    process.stdout.write(text, (err) => {
      if (err) {
        process.stdout.once("error", () => {});
        reject(err);
        return;
      }
      resolve();
    });
  });
}

/** True for the "reader went away" error that a CLI should treat as a clean exit. */
export function isEpipe(err: unknown): boolean {
  return (err as NodeJS.ErrnoException | undefined)?.code === "EPIPE";
}
