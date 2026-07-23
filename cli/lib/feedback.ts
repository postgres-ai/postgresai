import * as config from "./config";

/**
 * Single source of truth for where CLI users are pointed to share ideas and
 * feedback. Currently work item #300 ("RFC – Joe bot CLI").
 *
 * NOTE: its public reachability by external (non-member) users is unconfirmed.
 * If the project turns out to be private to them, swap this for a public
 * channel. It is deliberately the only place the URL is written so that swap
 * stays a one-line change.
 */
export const FEEDBACK_URL =
  "https://gitlab.com/postgres-ai/postgresai/-/work_items/300";

/**
 * Env var that silences the occasional runtime tip. Any non-empty value other
 * than "0"/"false"/"no" (case-insensitive) counts as "silence".
 */
export const FEEDBACK_SUPPRESS_ENV = "PGAI_NO_FEEDBACK_TIP";

/**
 * Minimum gap between two occasional runtime tips (~weekly). The tip shows on
 * the first qualifying interactive run and then at most once per this window,
 * so it stays a nudge rather than a nag even for heavy interactive users.
 */
export const FEEDBACK_TIP_MIN_INTERVAL_MS = 7 * 24 * 60 * 60 * 1000;

export interface FeedbackTipContext {
  /** process.stdout.isTTY */
  isTTY: boolean;
  /** whether the invoked command was given --json */
  json: boolean;
  /** raw value of process.env[FEEDBACK_SUPPRESS_ENV] */
  suppress: string | undefined;
}

/** Whether FEEDBACK_SUPPRESS_ENV's value means "silence the tip". */
export function feedbackTipSuppressed(suppress: string | undefined): boolean {
  if (suppress === undefined) return false;
  const value = suppress.trim().toLowerCase();
  return value !== "" && value !== "0" && value !== "false" && value !== "no";
}

/**
 * Whether the occasional feedback tip is permitted for this run. It must never
 * reach agents, CI, scripts, or machine-readable output, so it is allowed only
 * on an interactive TTY, never with --json, and never when suppressed.
 */
export function feedbackTipAllowed(ctx: FeedbackTipContext): boolean {
  if (ctx.json) return false;
  if (!ctx.isTTY) return false;
  if (feedbackTipSuppressed(ctx.suppress)) return false;
  return true;
}

/** Whether the tip is due, given when it was last shown (epoch ms, or null). */
export function isFeedbackTipDue(
  lastShownMs: number | null | undefined,
  nowMs: number,
): boolean {
  if (lastShownMs === null || lastShownMs === undefined || !Number.isFinite(lastShownMs)) {
    return true; // never shown
  }
  return nowMs - lastShownMs >= FEEDBACK_TIP_MIN_INTERVAL_MS;
}

/** The single dim stderr line. Exported so tests can assert its exact shape. */
export function feedbackTipLine(): string {
  // Dim (\x1b[2m … \x1b[0m) so it stays subtle next to real command output.
  return `\x1b[2m💡 Ideas / feedback: ${FEEDBACK_URL}  (silence: ${FEEDBACK_SUPPRESS_ENV}=1)\x1b[0m`;
}

function readLastShown(): number | null {
  const value = config.readConfig().feedbackTipLastShownAt;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function persistLastShown(nowMs: number): void {
  config.writeConfig({ feedbackTipLastShownAt: nowMs });
}

/** Injection points so the emit path is testable without touching real config. */
export interface FeedbackTipIO {
  write?: (line: string) => void;
  readLastShown?: () => number | null;
  persistLastShown?: (nowMs: number) => void;
  now?: () => number;
}

/**
 * Emit the occasional one-line feedback tip to stderr when allowed and due.
 * Returns true iff the tip was written. Best-effort: never throws into the
 * caller, and only reads/writes config on runs that are allowed (so agents, CI,
 * and --json output neither see the tip nor mutate config).
 */
export function maybeEmitFeedbackTip(ctx: FeedbackTipContext, io: FeedbackTipIO = {}): boolean {
  if (!feedbackTipAllowed(ctx)) return false;

  const now = (io.now ?? Date.now)();
  const read = io.readLastShown ?? readLastShown;
  const persist = io.persistLastShown ?? persistLastShown;
  const write = io.write ?? ((line: string) => { process.stderr.write(line); });

  let lastShown: number | null = null;
  try {
    lastShown = read();
  } catch {
    lastShown = null;
  }
  if (!isFeedbackTipDue(lastShown, now)) return false;

  try {
    persist(now);
  } catch {
    // Best-effort: a failed config write must not break a real command.
  }
  write(feedbackTipLine() + "\n");
  return true;
}

/** Machine-readable output for `pgai feedback --json`. */
export function feedbackJson(): string {
  return JSON.stringify({ url: FEEDBACK_URL }, null, 2);
}

/** Human-facing output for `pgai feedback`. */
export function feedbackMessage(): string {
  return [
    "💡 Help shape the PostgresAI CLI",
    "",
    "We read every idea and bug report. Share yours here:",
    "",
    `  ${FEEDBACK_URL}`,
    "",
    `Tip: set ${FEEDBACK_SUPPRESS_ENV}=1 to silence the occasional reminder.`,
  ].join("\n");
}
