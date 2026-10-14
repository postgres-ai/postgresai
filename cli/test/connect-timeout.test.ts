import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { CONNECT_ATTEMPT_TIMEOUT_MS, nodeNet, raiseConnectAttemptTimeout } from "../lib/connect-timeout";

const getDefault = () => nodeNet.getDefaultAutoSelectFamilyAttemptTimeout!();
const setDefault = (ms: number) => nodeNet.setDefaultAutoSelectFamilyAttemptTimeout!(ms);

// The default is process-wide, so every case starts from Node 24's 250 ms and
// the original value is put back afterwards.
describe("raiseConnectAttemptTimeout (#421)", () => {
  let saved: number;
  beforeEach(() => {
    saved = getDefault();
    setDefault(250);
  });
  afterEach(() => setDefault(saved));

  test("raises Node's 250 ms default", () => {
    expect(raiseConnectAttemptTimeout([], undefined)).toBe(CONNECT_ATTEMPT_TIMEOUT_MS);
    expect(getDefault()).toBe(CONNECT_ATTEMPT_TIMEOUT_MS);
  });

  test("never lowers a higher default", () => {
    setDefault(5000);
    expect(raiseConnectAttemptTimeout([], undefined)).toBe(5000);
  });

  test("keeps the value the user set with the flag in NODE_OPTIONS", () => {
    setDefault(300);
    const nodeOptions = "--max-old-space-size=4096 --network-family-autoselection-attempt-timeout=300";
    expect(raiseConnectAttemptTimeout([], nodeOptions)).toBe(300);
  });

  test("keeps the value the user set with the flag on the node command line", () => {
    setDefault(300);
    expect(raiseConnectAttemptTimeout(["--network-family-autoselection-attempt-timeout=300"], undefined)).toBe(300);
  });

  test("keeps the value set with Node's underscore spelling of the flag", () => {
    setDefault(300);
    expect(raiseConnectAttemptTimeout([], "--network_family_autoselection_attempt_timeout=300")).toBe(300);
  });

  test("an unrelated NODE_OPTIONS does not stop the raise", () => {
    expect(raiseConnectAttemptTimeout([], "--max-old-space-size=4096")).toBe(CONNECT_ATTEMPT_TIMEOUT_MS);
  });

  test("the flag name inside another option's value is not the flag", () => {
    const nodeOptions = "--require=/opt/no--network-family-autoselection-attempt-timeout.js";
    expect(raiseConnectAttemptTimeout([], nodeOptions)).toBe(CONNECT_ATTEMPT_TIMEOUT_MS);
  });
});
