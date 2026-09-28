import { afterEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { HOST_METRICS_VERIFY_SCRIPT } from "../lib/clickhouse";

// Runs the exact script sink-prometheus executes, with a fake wget standing in
// for VictoriaMetrics' /api/v1/targets. Each call returns resp<N>, else resp.
const dirs: string[] = [];
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }); });

const targets = (...pools: [string, string][]) => JSON.stringify({ status: "success", data: {
  activeTargets: pools.map(([pool, rev]) => ({ discoveredLabels: { __pgai_rev: rev, job: pool }, scrapePool: pool })), droppedTargets: [],
} });
const newRev = `"__pgai_rev":"r1111111111111111"`;
const pool = `"scrapePool":"clickhouse-a"`;

async function verify(needle: string, mode: "present" | "absent", responses: string[], { hang = false } = {}) {
  const dir = mkdtempSync(`${tmpdir()}/clickhouse-verify-`);
  dirs.push(dir);
  mkdirSync(`${dir}/bin`);
  writeFileSync(`${dir}/bin/wget`, `#!/bin/sh
printf '%s\\n' "$@" > "$DIR/args"
n=$(cat "$DIR/count" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$DIR/count"
if [ -n "$HANG" ]; then sleep 2; exit 1; fi
f="$DIR/resp$n"; [ -f "$f" ] || f="$DIR/resp"; [ -f "$f" ] || exit 1
cat "$f"
`);
  chmodSync(`${dir}/bin/wget`, 0o755);
  responses.forEach((body, i) => writeFileSync(i === responses.length - 1 ? `${dir}/resp` : `${dir}/resp${i + 1}`, body));
  const started = Date.now();
  const proc = Bun.spawn(["/bin/sh", "-c", HOST_METRICS_VERIFY_SCRIPT, "sh", needle, mode], {
    env: { PATH: `${dir}/bin:/usr/bin:/bin`, DIR: dir, VM_AUTH_USERNAME: "u", VM_AUTH_PASSWORD: "p@ss:w/rd", ...(hang ? { HANG: "1" } : {}) },
  });
  const exitCode = await proc.exited;
  return { exitCode, seconds: (Date.now() - started) / 1000, calls: Number(readFileSync(`${dir}/count`, "utf8")), args: readFileSync(`${dir}/args`, "utf8").trim().split("\n") };
}

test("sends a bounded, authenticated request for the targets list", async () => {
  const result = await verify(newRev, "present", [targets(["clickhouse-a", "r1111111111111111"])]);
  expect(result.exitCode).toBe(0);
  expect(result.args).toEqual(["-qO-", "-T", "2", "--header", `Authorization: Basic ${btoa("u:p@ss:w/rd")}`, "http://127.0.0.1:9090/api/v1/targets"]);
});

test("add waits for the new revision instead of accepting the old job", async () => {
  const result = await verify(newRev, "present", [targets(["clickhouse-a", "r0000000000000000"]), targets(["clickhouse-a", "r1111111111111111"])]);
  expect(result.exitCode).toBe(0);
  expect(result.calls).toBe(2);
});

test("remove ignores a job whose name only starts with the removed one", async () => {
  const result = await verify(pool, "absent", [targets(["clickhouse-ab", "r2222222222222222"])]);
  expect(result.exitCode).toBe(0);
  expect(result.calls).toBe(1);
});

test("remove waits for the job to disappear", async () => {
  const result = await verify(pool, "absent", [targets(["clickhouse-a", "r0000000000000000"]), targets()]);
  expect(result.exitCode).toBe(0);
  expect(result.calls).toBe(2);
});

for (const [name, needle, mode, responses, hang] of [
  ["add fails when only the old revision stays loaded", newRev, "present", [targets(["clickhouse-a", "r0000000000000000"])], false],
  ["remove fails on an error response", pool, "absent", [JSON.stringify({ status: "error", error: "unavailable" })], false],
  ["remove fails when VictoriaMetrics does not answer", pool, "absent", [], false],
  ["remove fails within the deadline when every request times out", pool, "absent", [targets()], true],
] as const) {
  test.concurrent(name, async () => {
    const result = await verify(needle, mode, [...responses], { hang });
    expect(result.exitCode).toBe(1);
    expect(result.seconds).toBeGreaterThanOrEqual(9);
    expect(result.seconds).toBeLessThan(14);
  }, 20000);
}
