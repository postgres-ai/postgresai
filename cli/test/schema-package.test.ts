import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const cliDir = resolve(import.meta.dir, "..");
const repoSchemasDir = resolve(cliDir, "../reporter/schemas");
const tempDir = mkdtempSync(join(tmpdir(), "postgresai-schema-package-"));
const extractedPackageDir = join(tempDir, "package");

function run(command: string[]): string {
  const result = Bun.spawnSync(command, {
    cwd: cliDir,
    env: process.env,
    stderr: "pipe",
    stdout: "pipe",
  });

  if (result.exitCode !== 0) {
    throw new Error(
      `${command.join(" ")} failed:\n${result.stderr.toString()}`,
    );
  }

  return result.stdout.toString();
}

beforeAll(() => {
  rmSync(resolve(cliDir, "schemas"), { recursive: true, force: true });
  const packOutput = run([
    "npm",
    "pack",
    "--json",
    "--pack-destination",
    tempDir,
  ]);
  // Lifecycle script output precedes npm's JSON payload on stdout.
  const jsonStart = packOutput.lastIndexOf("\n[");
  const packResult = JSON.parse(
    packOutput.slice(jsonStart < 0 ? 0 : jsonStart + 1),
  );
  const tarball = join(tempDir, packResult[0].filename);
  run(["tar", "-xzf", tarball, "-C", tempDir]);
});

afterAll(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

describe("published JSON Schemas", () => {
  test("tarball schemas exactly match reporter schemas", () => {
    const sourceFiles = readdirSync(repoSchemasDir)
      .filter((file) => file.endsWith(".schema.json"))
      .sort();
    const packagedSchemasDir = join(extractedPackageDir, "schemas");
    const packagedFiles = readdirSync(packagedSchemasDir).sort();

    expect(packagedFiles).toEqual(sourceFiles);
    for (const file of sourceFiles) {
      expect(readFileSync(join(packagedSchemasDir, file))).toEqual(
        readFileSync(join(repoSchemasDir, file)),
      );
    }
  });

  test("schema subpath export resolves for consumers", () => {
    const sourceFiles = readdirSync(repoSchemasDir)
      .filter((file) => file.endsWith(".schema.json"))
      .sort();
    const consumerDir = join(tempDir, "consumer");
    const nodeModulesDir = join(consumerDir, "node_modules");
    mkdirSync(nodeModulesDir, { recursive: true });
    symlinkSync(extractedPackageDir, join(nodeModulesDir, "postgresai"), "dir");

    const consumerRequire = createRequire(join(consumerDir, "consumer.cjs"));
    for (const file of sourceFiles) {
      const resolved = consumerRequire.resolve(`postgresai/schemas/${file}`);
      expect(readFileSync(resolved)).toEqual(
        readFileSync(join(repoSchemasDir, file)),
      );
    }
  });
});
