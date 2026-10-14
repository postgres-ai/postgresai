import * as net from "node:net";

/**
 * Per-address connect attempt timeout for Node's Happy Eyeballs.
 *
 * Node 20+ enables `autoSelectFamily` with a 250 ms attempt timeout (500 ms from
 * Node 25), and it aborts an attempt at that limit instead of racing it. On a
 * link where the TCP connect takes longer (VPN, mobile, satellite, a congested
 * Wi-Fi), every IPv4 attempt is cut off and, when the last address is an
 * unroutable IPv6, `fetch` fails with a bare "fetch failed" while curl works.
 * Measured with `tc netem delay 300ms` on Node 24 (#421): 3/3 failures at the
 * default, 3/3 successes at 2000 ms. See nodejs/node#54359.
 */
export const CONNECT_ATTEMPT_TIMEOUT_MS = 2000;

const ATTEMPT_TIMEOUT_FLAG = "--network-family-autoselection-attempt-timeout";

// The repo's @types/node (18.19) predates these two functions.
export const nodeNet = net as typeof net & {
  getDefaultAutoSelectFamilyAttemptTimeout?: () => number;
  setDefaultAutoSelectFamilyAttemptTimeout?: (ms: number) => void;
};

/**
 * Raise Node's default attempt timeout to CONNECT_ATTEMPT_TIMEOUT_MS. Never
 * lowers a higher default, and leaves it alone when the user set the flag.
 * A fast connect is unaffected. The cost: when the FIRST address silently
 * drops packets (a blackholed IPv6 sorted first), a new socket waits 2 s
 * instead of 250 ms before the next address; a failed request is worse.
 * Returns the timeout now in effect.
 */
export function raiseConnectAttemptTimeout(
  execArgv: readonly string[] = process.execArgv,
  nodeOptions: string | undefined = process.env.NODE_OPTIONS,
): number | undefined {
  // Missing before Node 18.18 / 19.8.
  if (!nodeNet.getDefaultAutoSelectFamilyAttemptTimeout || !nodeNet.setDefaultAutoSelectFamilyAttemptTimeout) {
    return undefined;
  }
  // Node accepts `_` for `-` in flag names.
  const userSet = [...execArgv, ...(nodeOptions ?? "").split(/\s+/)].some((arg) =>
    arg.replace(/_/g, "-").startsWith(ATTEMPT_TIMEOUT_FLAG),
  );
  if (!userSet && nodeNet.getDefaultAutoSelectFamilyAttemptTimeout() < CONNECT_ATTEMPT_TIMEOUT_MS) {
    nodeNet.setDefaultAutoSelectFamilyAttemptTimeout(CONNECT_ATTEMPT_TIMEOUT_MS);
  }
  return nodeNet.getDefaultAutoSelectFamilyAttemptTimeout();
}
