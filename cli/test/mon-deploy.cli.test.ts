import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";

// Cloud monitoring lives under `pgai mon` (as DBLab under `pgai dblab`):
// `mon deploy` and `mon instances list|status|delete`. No top-level `mon deploy`.

const CLI = resolve(import.meta.dir, "..", "bin", "postgres-ai.ts");

async function run(args: string[]) {
  const home = mkdtempSync(resolve(tmpdir(), "pgai-mon-deploy-"));
  const proc = Bun.spawn([process.execPath, CLI, ...args], {
    env: { ...process.env, HOME: home, XDG_CONFIG_HOME: home, PGAI_API_KEY: "" },
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
  });
  const [stdout, stderr, status] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  return { status, stdout, stderr };
}

describe("pgai mon deploy", () => {
  test("`pgai mon deploy --help` shows the deploy command", async () => {
    const r = await run(["mon", "deploy", "--help"]);
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("Usage: postgres-ai mon deploy [options] [database-url]");
    expect(r.stdout).toContain("--yes");
    expect(r.stdout).toContain("--coupon <code>");
  });

  // postgresai#412: one surface with pgai dblab deploy.
  test("the options both deploy commands share read the same in both helps", async () => {
    const shared = (help: string) => help.split("\n").filter((l) => /^ {2}(--db-url|--name|--location|--wait|--no-wait|-y, --yes|--json|--debug)\b/.test(l))
      .map((l) => l.replace(/\(default: "\d+"\)/, "").replace(/\(default: [^)]*\)$/, "").trim().split(/\s{2,}/)[0]);
    const mon = (await run(["mon", "deploy", "--help"])).stdout;
    const dblab = (await run(["dblab", "deploy", "--help"])).stdout;
    expect(shared(mon)).toEqual(["--db-url <url>", "--name <name>", "--location <location>", "--wait <minutes>", "--no-wait", "-y, --yes", "--json", "--debug"]);
    expect(shared(dblab)).toEqual(shared(mon));
    const words = (help: string, flag: string) => help.slice(help.indexOf(flag)).split("\n").slice(0, 2).join(" ").replace(/\s+/g, " ");
    for (const flag of ["--db-url <url>", "--location <location>", "--no-wait", "-y, --yes", "--json"]) {
      expect(words(dblab, flag).split(flag)[1].trim().slice(0, 40)).toBe(words(mon, flag).split(flag)[1].trim().slice(0, 40));
    }
  });

  test("`pgai mon instances --help` lists list, watch (status) and delete, by id or name", async () => {
    const r = await run(["mon", "instances", "--help"]);
    expect(r.status).toBe(0);
    for (const sub of ["list", "watch|status [options] <id-or-name>", "delete [options] <id-or-name>"]) expect(r.stdout).toContain(sub);
  });

  test.each([["connect"], ["disconnect"], ["databases"], ["status"]])("`pgai %s` no longer exists", async (cmd) => {
    const r = await run([cmd]);
    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain(`unknown command '${cmd}'`);
  });
});
