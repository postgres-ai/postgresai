/**
 * Checkup Dictionary Module
 * =========================
 * Provides access to the checkup report dictionary data embedded at build time.
 *
 * The dictionary is fetched from https://postgres.ai/api/general/checkup_dictionary
 * during the build process and embedded into checkup-dictionary-embedded.ts.
 *
 * This ensures no API calls are made at runtime while keeping the data up-to-date.
 */

import { CHECKUP_DICTIONARY_DATA } from "./checkup-dictionary-embedded";

/**
 * A checkup dictionary entry describing a single check type.
 */
export interface CheckupDictionaryEntry {
  /** Unique check code (e.g., "A001", "H002") */
  code: string;
  /** Human-readable title for the check */
  title: string;
  /** Brief description of what the check covers */
  description: string;
  /** Category grouping (e.g., "system", "indexes", "vacuum") */
  category: string;
  /** Optional sort order within category */
  sort_order: number | null;
  /** Whether this is a system-level report */
  is_system_report: boolean;
}

/**
 * Module-level cache for O(1) lookups by code.
 * Initialized at module load time from embedded data.
 * Keys are normalized to uppercase for case-insensitive lookups.
 */
const dictionaryByCode: Map<string, CheckupDictionaryEntry> = new Map(
  CHECKUP_DICTIONARY_DATA.map((entry) => [entry.code.toUpperCase(), entry])
);

/**
 * Get all checkup dictionary entries.
 *
 * @returns Array of all checkup dictionary entries
 */
export function getAllCheckupEntries(): CheckupDictionaryEntry[] {
  return CHECKUP_DICTIONARY_DATA;
}

/**
 * Get a checkup dictionary entry by its code.
 *
 * @param code - The check code (e.g., "A001", "H002"). Lookup is case-insensitive.
 * @returns The dictionary entry or null if not found
 */
export function getCheckupEntry(code: string): CheckupDictionaryEntry | null {
  return dictionaryByCode.get(code.toUpperCase()) ?? null;
}

/**
 * Check if a code exists in the dictionary.
 *
 * @param code - The check code to validate
 * @returns True if the code exists in the dictionary
 */
export function isValidCheckupCode(code: string): boolean {
  return dictionaryByCode.has(code.toUpperCase());
}

/**
 * Get all check codes as an array.
 *
 * @returns Array of all check codes (e.g., ["A001", "A002", ...])
 */
export function getAllCheckupCodes(): string[] {
  return CHECKUP_DICTIONARY_DATA.map((entry) => entry.code);
}

/**
 * Get checkup entries filtered by category.
 *
 * @param category - The category to filter by (e.g., "indexes", "vacuum")
 * @returns Array of entries in the specified category
 */
export function getCheckupEntriesByCategory(category: string): CheckupDictionaryEntry[] {
  return CHECKUP_DICTIONARY_DATA.filter(
    (entry) => entry.category.toLowerCase() === category.toLowerCase()
  );
}

/**
 * Build a code-to-title mapping object.
 * Useful for backwards compatibility with CHECK_INFO style usage.
 *
 * @returns Object mapping check codes to titles (e.g., { "A001": "System information", ... })
 */
export function buildCheckInfoMap(): Record<string, string> {
  const result: Record<string, string> = {};
  for (const entry of CHECKUP_DICTIONARY_DATA) {
    result[entry.code] = entry.title;
  }
  return result;
}
