/**
 * Leak detectors shared by every redaction suite.
 *
 * ONE definition on purpose. These helpers started life inside
 * aas-onboard.test.ts for #382; #383 needed the same measurements against the
 * shared `redactTextSecrets` in util.test.ts, and two leak detectors drifting
 * apart is the same failure mode as two redactors drifting apart — the weaker
 * copy reports clean and the bug ships.
 */

/** Insert `gap` into `secret` at `at`, i.e. what a wrapped error body does. */
export function splitAt(secret: string, at: number, gap: string): string {
  return secret.slice(0, at) + gap + secret.slice(at);
}

/**
 * The operator-facing criterion: not "is the secret present verbatim" but "is
 * it one whitespace-deletion away from usable". A scrub that runs BEFORE a
 * whitespace flatten passes the first check and fails this one.
 */
export function leaks(out: string, secret: string): boolean {
  return out.includes(secret) || out.replace(/\s+/g, "").includes(secret.replace(/\s+/g, ""));
}

/**
 * Longest run of `secret` that survives anywhere in `out`.
 *
 * `leaks` only catches a WHOLE secret, and that blind spot is how two real
 * leaks shipped: when a redactor replaces the HEAD of a wrapped credential the
 * tail prints on its own and `leaks` reports clean (#382 F1 found by
 * @akartasov on !423; then #383 for the same reason in the shared helper).
 * Measuring the longest shared run catches head-, tail- and middle-survival
 * with one assertion.
 */
export function longestRun(out: string, secret: string): number {
  const o = out.replace(/\s+/g, "");
  const t = secret.replace(/\s+/g, "");
  let best = 0;
  for (let i = 0; i < t.length; i++) {
    for (let j = i + 1; j <= t.length; j++) {
      const sub = t.slice(i, j);
      if (sub.length > best && o.includes(sub)) best = sub.length;
    }
  }
  return best;
}

/**
 * Longest surviving run we are willing to call "not a credential any more".
 *
 * Above any incidental overlap between these fixtures and the prose around
 * them ("token", "grafana"), and far below a usable credential. It is also the
 * floor `redactTextSecrets` uses to decide a run is credential material, so the
 * two numbers are deliberately the same: a run long enough to matter is a run
 * long enough to absorb.
 */
export const MAX_SURVIVING_RUN = 12;

/**
 * Credential fixtures, assembled from parts rather than written as one literal
 * so the repo's gitleaks hook does not flag a high-entropy string: only the
 * LENGTH and the CHARSET matter here, never the value.
 */
export const SECRET_SHAPES: ReadonlyArray<readonly [string, string]> = [
  ["snake_case token with a version segment", ["zztok", "v9", "neversentbyus", "eeeeffffgggghhhh"].join("_")],
  ["opaque base64url-style token", ["R7qZmN4v", "K1sT0xW8", "yB3cJhL6", "dPaG5nE2"].join("")],
  ["hex digest", ["9f2c4b7e", "1a83d05c", "6e418b9d", "20f73ca6"].join("")],
  // Deliberately word-like: ordinary vowel spacing, single case, no long
  // consonant run. EVERY tail of it is credential material only because of a
  // digit or an underscore, so this shape is what pins that signal on its own
  // rather than letting the consonant-run signal cover for it.
  ["word-like token carrying digits and separators", ["zzmix", "ea1", "retoken", "value2", "notasecret3"].join("_")],
  // Single case, ordinary vowel spacing, NO separator at all: every tail of it
  // is credential material only because of a DIGIT. Without this shape the
  // digit signal is covered for by the underscore in the shape above, and a
  // mutation dropping it survives (#383 review F4).
  ["token carrying digits and nothing else", "heza4bonu7cela9dimu6vota"],
  // Standard base64: `/` and `+`, which base64url does not use. Pins both in
  // the credential charset -- dropping `/` from it leaves the run unabsorbed.
  ["standard base64 with slash and plus", ["aB3/cD4+", "eF5/gH6x", "iJ7+kL8y", "mN9/oP0z"].join("")],
  // `+` and nothing else: no digit, single case, no letter consonant run long
  // enough. The only shape that pins `+` as a signal on its own.
  ["token carrying only a plus", ["aaaa+bbbb", "+cccc+dddd", "+eeee+ffff"].join("")],
  // `_` and nothing else. One segment with no vowel keeps it from reading as a
  // separated name, so the underscore is the only thing that qualifies it.
  ["token carrying only an underscore", ["nuvo", "mnp", "reka", "tesi", "luma", "zdr", "beto"].join("_")],
  // Base64 padding and nothing else: single case, no digit, no separator, no
  // long consonant run. The only shape that pins the trailing `=`.
  ["token carrying only base64 padding", ["nuvoreka", "tesiluma", "betokira"].join("") + "="],
  // Letters only, vowels evenly spread, no separator and no digit: the ONLY
  // thing that tells it from a word is the capital in the middle. Pins that,
  // and pins it in both directions -- it is also what keeps camelCase from
  // reading as a word and breaking the chain.
  ["mixed-case letters and nothing else", "aBeCiDoFuGaHeJiKoLuMaNeP"],
  // Dash-separated numbers with no word anywhere: pins the rule that a name
  // needs at least one word segment, without which every grouped number reads
  // as a diagnostic and a credential shaped like one survives.
  ["dash-separated digits with no word", ["1234", "5678", "9012", "3456", "7890"].join("-")],
  // Licence-key shape: all-caps groups and numbers. The caps groups are longer
  // than an acronym, which is the only thing keeping this from reading as a
  // name -- so it pins the acronym LENGTH bound, not just its existence.
  ["licence-key shape, all-caps groups", ["ZXCVBNM", "1234", "QWRTYPS", "5678", "DFGHJKL"].join("-")],
];

/**
 * Diagnostics that must survive a redaction next to them, byte for byte.
 *
 * The first version of #383 ate all of these. It guarded against ENGLISH and
 * holds up there -- but almost nothing in these messages is English. They are
 * libpq connection strings, `key=value` fields, paths and correlation ids, and
 * `password=` / `connstr=` / `dsn=` occur in the wild almost exclusively INSIDE
 * a connection string, i.e. surrounded by exactly this.
 */
export const DIAGNOSTIC_CORPUS: readonly string[] = [
  "sslmode=require host=db.example.com port=5432 dbname=app user=monitor",
  "sslmode=verify-full target_session_attrs=read-write connect_timeout=10",
  "request_id=7f3a9b2c-1d4e-4f8a-9c2b-1a2b3c4d5e6f status=429 retry-after:300",
  "7f3a9b2c-1d4e-4f8a-9c2b-1a2b3c4d5e6f is the correlation id",
  "/var/lib/postgresql/data is unreadable",
  "see docs/troubleshooting for instance_identifier",
  "instance_id=305 datasource_uid=pgai-vm armed_for=jobs",
  "pg_stat_statements is not loaded via shared_preload_libraries",
  "host db.example.com refused; monitoring-prod-01.internal is down",
  "the organization subscription requires reauthentication before configuration",
  "unauthorized, authentication failed for user monitor",
  "container platform-fix-383-redact-postgrest-1 exited with code 137",
  "<operator@example.com> was notified (instance-7-of-12) HTTP/1.1-429!",
  "rate-limited; self-service reactivation is unavailable",
  // Short alphanumeric runs. Nothing but the length floor keeps these: they
  // carry a digit and do not read as words, so a lower floor eats them.
  "pg16 replica lag 2s on node3 after wal1 replay",
];
