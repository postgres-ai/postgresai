/**
 * Cross-language guard for the checkup JSON report contract version.
 *
 * The report envelope carries a `contract_version` that BOTH report engines
 * must emit identically:
 *   - the TypeScript express engine (cli/lib/checkup.ts, CONTRACT_VERSION)
 *   - the Python reporter (reporter/postgres_reports.py, CONTRACT_VERSION)
 *
 * These are two independent source files, so they can drift. This test parses
 * the canonical constant out of each source and asserts they are equal, and
 * that the TS engine actually stamps it onto the report envelope. If someone
 * bumps one without the other, this fails.
 */
import { describe, test, expect } from "bun:test";
import { resolve } from "path";
import { readFileSync } from "fs";

import { CONTRACT_VERSION, createBaseReport } from "../lib/checkup";

const SEMVER = /^\d+\.\d+\.\d+$/;

function extractTsContractVersion(): string {
  const src = readFileSync(resolve(import.meta.dir, "../lib/checkup.ts"), "utf8");
  const m = src.match(/export const CONTRACT_VERSION\s*=\s*"([^"]+)"/);
  if (!m) throw new Error("CONTRACT_VERSION not found in cli/lib/checkup.ts");
  return m[1];
}

function extractPyContractVersion(): string {
  const src = readFileSync(
    resolve(import.meta.dir, "../../reporter/postgres_reports.py"),
    "utf8"
  );
  const m = src.match(/^CONTRACT_VERSION\s*=\s*"([^"]+)"/m);
  if (!m) throw new Error("CONTRACT_VERSION not found in reporter/postgres_reports.py");
  return m[1];
}

describe("checkup contract_version", () => {
  test("is a valid semver string", () => {
    expect(CONTRACT_VERSION).toMatch(SEMVER);
  });

  test("TS and Python report engines emit the same contract_version", () => {
    const ts = extractTsContractVersion();
    const py = extractPyContractVersion();
    // The runtime constant must match its own source (sanity)...
    expect(ts).toBe(CONTRACT_VERSION);
    // ...and both engines must agree, or JSON consumers see inconsistent versions.
    expect(py).toBe(ts);
  });

  test("createBaseReport stamps contract_version onto the envelope", () => {
    const report = createBaseReport("H002", "Unused indexes", "node-01");
    expect(report.contract_version).toBe(CONTRACT_VERSION);
  });
});
