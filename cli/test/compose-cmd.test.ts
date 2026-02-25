import { describe, test, expect } from "bun:test";

/**
 * Test getComposeCmd() selection logic.
 * WARNING: This replicates the logic from postgres-ai.ts. If the production
 * function changes, this replica must be updated to match.
 * Since the function is internal to postgres-ai.ts, we replicate its logic
 * with an injectable command checker (same pattern as monitoring.test.ts).
 */
function getComposeCmd(
  tryCmd: (cmd: string, args: string[]) => boolean,
): string[] | null {
  if (tryCmd("docker", ["compose", "version"])) return ["docker", "compose"];
  if (tryCmd("docker-compose", ["version"])) return ["docker-compose"];
  return null;
}

describe("getComposeCmd", () => {
  test("prefers docker compose V2 when both are available", () => {
    const result = getComposeCmd(() => true);
    expect(result).toEqual(["docker", "compose"]);
  });

  test("falls back to docker-compose V1 when V2 is unavailable", () => {
    const result = getComposeCmd((cmd, args) => {
      // V2 plugin fails, V1 standalone succeeds
      if (cmd === "docker" && args[0] === "compose") return false;
      if (cmd === "docker-compose") return true;
      return false;
    });
    expect(result).toEqual(["docker-compose"]);
  });

  test("returns null when neither is available", () => {
    const result = getComposeCmd(() => false);
    expect(result).toBeNull();
  });

  test("does not check V1 when V2 succeeds", () => {
    const calls: Array<{ cmd: string; args: string[] }> = [];
    getComposeCmd((cmd, args) => {
      calls.push({ cmd, args });
      return cmd === "docker" && args[0] === "compose";
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual({ cmd: "docker", args: ["compose", "version"] });
  });

  test("checks V2 first, then V1", () => {
    const calls: Array<{ cmd: string; args: string[] }> = [];
    getComposeCmd((cmd, args) => {
      calls.push({ cmd, args });
      return false;
    });
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual({ cmd: "docker", args: ["compose", "version"] });
    expect(calls[1]).toEqual({ cmd: "docker-compose", args: ["version"] });
  });
});

/**
 * Test the monitoring startup sequence's container cleanup logic.
 * Before "up --force-recreate", stopped containers from "run --rm" dependencies
 * (e.g. config-init) must be removed to avoid docker-compose v1's
 * KeyError: 'ContainerConfig' bug.
 *
 * We replicate the relevant sequence from the monitoring start command
 * with an injectable runCompose to verify ordering and error tolerance.
 */
async function monitoringStartSequence(
  runCompose: (args: string[]) => Promise<number>,
): Promise<number> {
  // Best-effort: remove stopped containers left by "run --rm" dependencies
  await runCompose(["rm", "-f", "-s", "config-init"]);
  // Start services
  const code = await runCompose(["up", "-d", "--force-recreate"]);
  return code;
}

describe("monitoring start: config-init cleanup", () => {
  test("calls rm before up", async () => {
    const calls: string[][] = [];
    await monitoringStartSequence(async (args) => {
      calls.push(args);
      return 0;
    });
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(["rm", "-f", "-s", "config-init"]);
    expect(calls[1]).toEqual(["up", "-d", "--force-recreate"]);
  });

  test("continues to up even when rm fails", async () => {
    const calls: string[][] = [];
    await monitoringStartSequence(async (args) => {
      calls.push(args);
      // rm returns non-zero (container doesn't exist)
      if (args[0] === "rm") return 1;
      return 0;
    });
    expect(calls).toHaveLength(2);
    expect(calls[0][0]).toBe("rm");
    expect(calls[1][0]).toBe("up");
  });

  test("returns up exit code, not rm exit code", async () => {
    // rm fails but up succeeds → overall success
    const result1 = await monitoringStartSequence(async (args) => {
      if (args[0] === "rm") return 1;
      return 0;
    });
    expect(result1).toBe(0);

    // rm succeeds but up fails → overall failure
    const result2 = await monitoringStartSequence(async (args) => {
      if (args[0] === "up") return 2;
      return 0;
    });
    expect(result2).toBe(2);
  });
});
