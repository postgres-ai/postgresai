#!/usr/bin/env bun
/**
 * Build script to fetch checkup dictionary from API and embed it.
 *
 * This script fetches from https://postgres.ai/api/general/checkup_dictionary
 * and generates cli/lib/checkup-dictionary-embedded.ts with the data embedded.
 *
 * The generated file is NOT committed to git - it's regenerated at build time.
 *
 * Usage: bun run scripts/embed-checkup-dictionary.ts
 */

import * as fs from "fs";
import * as path from "path";

// API endpoint - always available without auth
const DICTIONARY_URL = "https://postgres.ai/api/general/checkup_dictionary";

// Output path relative to cli/ directory
const CLI_DIR = path.resolve(__dirname, "..");
const OUTPUT_PATH = path.resolve(CLI_DIR, "lib/checkup-dictionary-embedded.ts");

// Request timeout (10 seconds)
const FETCH_TIMEOUT_MS = 10_000;

interface CheckupDictionaryEntry {
  code: string;
  title: string;
  description: string;
  category: string;
  sort_order: number | null;
  is_system_report: boolean;
}

function generateTypeScript(data: CheckupDictionaryEntry[], sourceUrl: string): string {
  const lines: string[] = [
    "// AUTO-GENERATED FILE - DO NOT EDIT",
    `// Generated from: ${sourceUrl}`,
    `// Generated at: ${new Date().toISOString()}`,
    "// To regenerate: bun run embed-checkup-dictionary",
    "",
    'import { CheckupDictionaryEntry } from "./checkup-dictionary";',
    "",
    "/**",
    " * Embedded checkup dictionary data fetched from API at build time.",
    " * Contains all available checkup report codes, titles, and metadata.",
    " */",
    `export const CHECKUP_DICTIONARY_DATA: CheckupDictionaryEntry[] = ${JSON.stringify(data, null, 2)};`,
    "",
  ];
  return lines.join("\n");
}

// Allowed hosts for fetch requests to prevent SSRF
const ALLOWED_HOSTS = ["postgres.ai"];

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  // Validate URL against allowlist to prevent SSRF
  const parsed = new URL(url);
  if (!ALLOWED_HOSTS.includes(parsed.hostname)) {
    throw new Error(`Fetch blocked: host "${parsed.hostname}" is not in the allowlist`);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function main() {
  console.log(`Fetching checkup dictionary from: ${DICTIONARY_URL}`);

  try {
    const response = await fetchWithTimeout(DICTIONARY_URL, FETCH_TIMEOUT_MS);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data: CheckupDictionaryEntry[] = await response.json();

    if (!Array.isArray(data)) {
      throw new Error("Expected array response from API");
    }

    // Validate entries have required fields
    for (const entry of data) {
      if (!entry.code || !entry.title) {
        throw new Error(`Invalid entry missing code or title: ${JSON.stringify(entry)}`);
      }
    }

    const tsCode = generateTypeScript(data, DICTIONARY_URL);
    fs.writeFileSync(OUTPUT_PATH, tsCode, "utf8");

    console.log(`Generated: ${OUTPUT_PATH}`);
    console.log(`Dictionary contains ${data.length} entries`);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.warn(`Warning: Failed to fetch checkup dictionary: ${errorMsg}`);
    console.warn("Generating empty dictionary as fallback");

    // Generate empty dictionary to allow build to proceed
    const fallbackTs = generateTypeScript([], `N/A (fetch failed: ${errorMsg})`);
    fs.writeFileSync(OUTPUT_PATH, fallbackTs, "utf8");
    console.log(`Generated fallback dictionary at ${OUTPUT_PATH}`);
  }
}

main();
