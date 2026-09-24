import { describe, expect, test } from "bun:test";

import { formatHttpError, describeFetchError, redactSecretsForLog, redactTextSecrets } from "../lib/util";
import { isSeparatedName, isWordShaped } from "../lib/util";
import { DIAGNOSTIC_CORPUS, MAX_SURVIVING_RUN, SECRET_SHAPES, leaks, longestRun, splitAt } from "./redaction-helpers";

describe("describeFetchError", () => {
  test("surfaces the connection cause + url instead of a bare 'fetch failed'", () => {
    // Node's fetch (undici) throws TypeError('fetch failed') and stashes the real
    // reason (ECONNREFUSED etc.) in err.cause — the CLI must surface it, not the
    // opaque top-level message.
    const err = Object.assign(new TypeError("fetch failed"), {
      cause: { code: "ECONNREFUSED", message: "connect ECONNREFUSED 127.0.0.1:1" },
    });
    const msg = describeFetchError("Failed to list projects", "http://127.0.0.1:1", err);
    expect(msg).toContain("Failed to list projects");
    expect(msg).toContain("could not reach http://127.0.0.1:1");
    expect(msg).toContain("ECONNREFUSED");
    expect(msg).not.toBe("fetch failed");
  });

  test("falls back to cause.message, then to the error message", () => {
    const withMsg = Object.assign(new TypeError("fetch failed"), {
      cause: { message: "bad port" },
    });
    expect(describeFetchError("op", "http://h:1", withMsg)).toContain("bad port");
    const bare = new Error("boom");
    expect(describeFetchError("op", "http://h", bare)).toContain("boom");
  });
});

describe("formatHttpError", () => {
  test("appends auth remediation hint on 401 with JSON body", () => {
    const msg = formatHttpError(
      "Failed to fetch issues",
      401,
      '{"hint": "Please check validity of token", "details": "Invalid token"}'
    );

    expect(msg).toContain("HTTP 401");
    expect(msg).toContain("Run 'postgresai auth'");
    expect(msg).toContain("PGAI_API_KEY");
  });

  test("appends auth remediation hint on 401 with HTML body (early-return path)", () => {
    const msg = formatHttpError("Failed to fetch issues", 401, "<html><body>Unauthorized</body></html>");

    expect(msg).toContain("HTTP 401");
    expect(msg).toContain("Run 'postgresai auth'");
  });

  test("appends auth remediation hint on 401 without body", () => {
    const msg = formatHttpError("Failed to fetch issues", 401);

    expect(msg).toContain("Run 'postgresai auth'");
  });

  test("does not append auth hint for non-401 statuses", () => {
    expect(formatHttpError("Failed to fetch issues", 403, '{"message": "denied"}')).not.toContain(
      "Run 'postgresai auth'"
    );
    expect(formatHttpError("Failed to fetch issues", 500)).not.toContain("Run 'postgresai auth'");
  });

  test("keeps structured JSON error details before the hint", () => {
    const msg = formatHttpError("Failed to fetch issues", 401, '{"message": "Invalid token"}');

    expect(msg.indexOf("Invalid token")).toBeGreaterThan(-1);
    expect(msg.indexOf("Invalid token")).toBeLessThan(msg.indexOf("Run 'postgresai auth'"));
  });

  // PostgREST v9's PTxyz custom-status convention carries the RPC's user-facing
  // `message` in the HTTP reason phrase (statusText); the JSON body holds only
  // `hint`/`details` (no `message`/`code`). The formatter must surface the
  // reason phrase as the headline instead of a hardcoded generic label, and
  // still show the `details` (plural) line.
  test("surfaces the PostgREST PTxyz reason-phrase message, not a generic label", () => {
    const msg = formatHttpError(
      "Failed to submit plan command",
      403,
      '{"hint":null,"details":"The LLM-facing Joe surface is gated on the organization\'s ai_enabled setting."}',
      "AI features disabled for this org — enable in AI Assistant Settings"
    );

    expect(msg).toContain("HTTP 403");
    expect(msg).toContain("AI features disabled for this org — enable in AI Assistant Settings");
    // the real reason must not be masked by the hardcoded generic label
    expect(msg).not.toContain("Forbidden - access denied");
    // the supplementary `details` (plural) line is still shown
    expect(msg).toContain("gated on the organization's ai_enabled setting");
    // and we never dump the raw JSON object
    expect(msg).not.toContain('"hint"');
  });

  test("prefers a JSON body `message` over the reason phrase when both exist", () => {
    const msg = formatHttpError("op", 400, '{"message":"real message","details":"more"}', "Bad Request");
    expect(msg).toContain("real message");
    expect(msg).toContain("more");
  });

  test("falls back to the generic label for a standard (non-custom) reason phrase", () => {
    // a bare standard reason phrase must not replace the friendlier generic label
    const msg = formatHttpError("op", 403, undefined, "Forbidden");
    expect(msg).toContain("Forbidden - access denied");
  });

  // Behind an h2/h3 proxy the reason phrase is dropped (statusText is empty),
  // so the PTxyz body fields are all we get — `code` and `hint` must surface.
  // CLI half of https://gitlab.com/postgres-ai/platform-all/-/issues/537.
  test("surfaces JSON body code and hint when the reason phrase is dropped (h2)", () => {
    const msg = formatHttpError(
      "Failed to submit plan command",
      403,
      '{"code":"PT403","details":"AI features disabled for this org","hint":"Enable AI in AI Assistant Settings"}',
      "" // h2: no reason phrase
    );
    expect(msg).toContain("HTTP 403");
    expect(msg).toContain("PT403");
    expect(msg).toContain("AI features disabled for this org");
    expect(msg).toContain("Hint: Enable AI in AI Assistant Settings");
  });

  test("surfaces code and hint alongside a body message", () => {
    const msg = formatHttpError(
      "op",
      429,
      '{"code":"PT429","message":"Rate limited","hint":"Retry in 60s"}'
    );
    expect(msg).toContain("Rate limited");
    expect(msg).toContain("PT429");
    expect(msg).toContain("Hint: Retry in 60s");
  });

  test("falls back to the raw (redacted) JSON body when no known fields are present", () => {
    // Pre-existing callers (issues.ts/reports.ts/storage.ts) rely on unknown
    // JSON error shapes still being surfaced — a body with none of
    // message/error/details/detail/code/hint must not be silently dropped.
    const msg = formatHttpError("op", 500, '{"weird_field":"boom","password":"pw-x"}');
    expect(msg).toContain("HTTP 500");
    expect(msg).toContain("weird_field");
    expect(msg).toContain("boom");
    expect(msg).not.toContain("pw-x");
    expect(msg).toContain("[REDACTED]");
  });

  test("redacts credentials from every structured JSON error field and the reason phrase", () => {
    const msg = formatHttpError(
      "op",
      400,
      JSON.stringify({
        message: "connStr=postgresql://joe:message-pw@db/app",
        details: "password=detail-pw",
        hint: "token=hint-token",
      }),
      "dsn=postgresql://joe:reason-pw@db/app"
    );
    expect(msg).toContain("[REDACTED]");
    for (const secret of ["message-pw", "detail-pw", "hint-token", "reason-pw"]) {
      expect(msg).not.toContain(secret);
    }
  });
});

describe("redactSecretsForLog", () => {
  test("redacts password-named keys at any depth (request body shape)", () => {
    const body = JSON.stringify({
      instance_id: "7",
      action: "/clone",
      method: "post",
      data: { protected: false, db: { username: "clone_user", password: "hunter2" } },
    });
    const out = redactSecretsForLog(body);
    expect(out).not.toContain("hunter2");
    expect(out).toContain('"password":"[REDACTED]"');
    // Non-secret fields survive untouched.
    expect(out).toContain('"username":"clone_user"');
    expect(out).toContain('"action":"/clone"');
  });

  test("redacts connStr and password in response bodies (clone create/status shape)", () => {
    const reply = JSON.stringify({
      id: "c1",
      db: { connStr: "host=h port=6002 user=joe password=pw-xyz", password: "pw-xyz", username: "joe" },
    });
    const out = redactSecretsForLog(reply);
    expect(out).not.toContain("pw-xyz");
    expect(out).toContain('"connStr":"[REDACTED]"');
    expect(out).toContain('"username":"joe"');
  });

  test("matches key variants case-insensitively (Password, DB_PASSWORD, dbPassword, connstr)", () => {
    const out = redactSecretsForLog(
      JSON.stringify({ Password: "a", DB_PASSWORD: "b", dbPassword: "c", connstr: "d", ok: "keep" })
    );
    expect(out).not.toContain('"a"');
    expect(out).not.toContain('"b"');
    expect(out).not.toContain('"c"');
    expect(out).not.toContain('"d"');
    expect(out).toContain('"ok":"keep"');
  });

  test("redacts alternate auth and credential key names", () => {
    const out = redactSecretsForLog(JSON.stringify({
      auth_key: "a",
      db_pass: "b",
      access_key: "c",
      credential: "d",
      ok: "keep",
    }));
    for (const secret of ["a", "b", "c", "d"]) expect(out).not.toContain(`\"${secret}\"`);
    expect(out).toContain('"ok":"keep"');
  });

  test("traverses arrays (e.g. result rows carrying a password column)", () => {
    const out = redactSecretsForLog(JSON.stringify({ rows: [{ usename: "app", password: "s3cr3t" }] }));
    expect(out).not.toContain("s3cr3t");
    expect(out).toContain('"usename":"app"');
  });

  test("keeps null secrets as null (shape stays readable) and passes non-JSON through", () => {
    expect(redactSecretsForLog(JSON.stringify({ password: null }))).toContain('"password":null');
    expect(redactSecretsForLog("plain text, not json")).toBe("plain text, not json");
  });

  test("does not redact benign keys that merely contain a sensitive substring", () => {
    // /auth/ must not eat author/authorized_at, /token/ must not eat
    // token_count, /cred/ must not eat credits — these are legitimate data
    // fields in exec_sql / DBLab / bot results.
    const out = redactSecretsForLog(JSON.stringify({
      author: "alice",
      authored_by: "bob",
      authorized_at: "2026-01-01",
      author_id: 12,
      author_name: "Ada",
      token_count: 42,
      token_type: "bearer",
      tokens: ["word"],
      total_tokens: 7,
      credits: 3,
      credited_at: "2026-01-02",
      password: "s3cr3t",
      auth_key: "k1",
    }));
    expect(out).toContain('"author":"alice"');
    expect(out).toContain('"authored_by":"bob"');
    expect(out).toContain('"authorized_at":"2026-01-01"');
    expect(out).toContain('"author_id":12');
    expect(out).toContain('"author_name":"Ada"');
    expect(out).toContain('"token_count":42');
    expect(out).toContain('"token_type":"bearer"');
    expect(out).toContain('"tokens":["word"]');
    expect(out).toContain('"total_tokens":7');
    expect(out).toContain('"credits":3');
    expect(out).toContain('"credited_at":"2026-01-02"');
    // real credentials still go
    expect(out).not.toContain("s3cr3t");
    expect(out).not.toContain('"k1"');
  });

  test("scrubs credential patterns from non-JSON text (error-path bodies)", () => {
    // A non-JSON body flows into thrown Errors (parse-failure paths), so the
    // text fallback must still scrub key=value credentials and URL userinfo.
    const out = redactSecretsForLog(
      'clone failed: connStr=postgresql://joe:pw-abc@host:6002/db password=hunter2 "secret": "s3cr3t" user=joe'
    );
    expect(out).not.toContain("pw-abc");
    expect(out).not.toContain("hunter2");
    expect(out).not.toContain("s3cr3t");
    expect(out).toContain("[REDACTED]");
    expect(out).toContain("user=joe");
  });
});

describe("redactTextSecrets: a credential-named key's WRAPPED value keeps no usable tail (#383)", () => {
  // The shared helper matched a bare value as `[^\s,;&]+`, i.e. up to the first
  // whitespace. A value that arrived already wrapped mid-token -- JSON pretty-
  // printing, a line break in an upstream error body -- therefore had its HEAD
  // replaced and its TAIL printed. All three callers (redactSecretsForLog,
  // formatHttpError, formatPlatformError) inherited it, so it is measured here
  // on the helper and again on each caller below.
  //
  // !423 cannot close this: it matches by VALUE, and this is a credential the
  // caller never sent. Only the key name identifies it.

  /** The gaps a wrapped body actually arrives with, including "already flattened". */
  const GAPS = ["\n", "\r\n", "\t", " "] as const;

  /** Credential-named keys, one per naming style the pattern supports. */
  const KEYS = ["api_token: ", "secret=", 'access_key: '] as const;

  for (const [shape, secret] of SECRET_SHAPES) {
    for (const key of KEYS) {
      test(`no usable run survives: ${shape}, behind \`${key.trim()}\`, intact`, () => {
        const out = redactTextSecrets(`${key}${secret} rejected`);
        expect(leaks(out, secret)).toBe(false);
        expect(longestRun(out, secret)).toBeLessThan(MAX_SURVIVING_RUN);
        expect(out).toContain("[REDACTED]");
        // ...and the diagnostic that followed it is still readable.
        expect(out.endsWith(" rejected")).toBe(true);
      });

      test(`no usable run survives: ${shape}, behind \`${key.trim()}\`, wrapped at EVERY offset`, () => {
        // Every split point, not a hand-picked one: the surviving run shrinks as
        // the split moves right, so a single offset can pass by luck while its
        // neighbours leak 27 characters.
        const survivors: Array<{ gap: string; at: number; run: number }> = [];
        for (const gap of GAPS) {
          for (let at = 1; at < secret.length; at++) {
            const out = redactTextSecrets(`${key}${splitAt(secret, at, gap)} rejected`);
            const run = longestRun(out, secret);
            if (run >= MAX_SURVIVING_RUN) survivors.push({ gap: JSON.stringify(gap), at, run } as never);
            if (!out.endsWith(" rejected")) survivors.push({ gap: JSON.stringify(gap), at, run: -1 } as never);
          }
        }
        expect(survivors).toEqual([]);
      });
    }

    test(`documented limit: ${shape}, UNKEYED, is left exactly as it arrived`, () => {
      // An unkeyed credential the caller never sent is indistinguishable from
      // ordinary text and no rule here can redact it (stated in #383). Pinned
      // as an equality so the wrapped-tail absorption can never start eating
      // bare prose instead: over-redaction is the safe direction for a
      // redactor, but only INSIDE a credential-named pair.
      for (const body of [secret, splitAt(secret, 12, "\n")]) {
        const input = `saw ${body} here`;
        expect(redactTextSecrets(input)).toBe(input);
      }
    });
  }

  // A value wrapped TWICE, swept over every pair of split offsets rather than
  // the two that were hand-picked here at first. Those two happened to fall in
  // the one region where every fragment clears the floor, so the suite reported
  // clean while 300 of 703 offset pairs kept >= 12 characters and the worst kept
  // 38 of 39 (#383 review F2). Choosing offsets is the exact trap the rest of
  // this file was written to avoid.
  //
  // This cannot assert zero, and the reason is the whole design: a chain of
  // fragments ends at a fragment that reads as an English WORD or at a set of
  // fragments that all read as separated NAMES, because at that point nothing
  // distinguishes them from the next word of the message or from
  // `instance_identifier`. What it CAN assert is that those two are the ONLY
  // ways a leak happens -- a characterisation, not a budget, so it keeps
  // meaning something as the code changes. Measured over every shape: 52 of
  // 3324 offset pairs leak, worst surviving run 16, all of them explained.
  for (const [shape, secret] of SECRET_SHAPES) {
    test(`a value wrapped TWICE leaks only behind a word-shaped fragment: ${shape}`, () => {
      const unexplained: Array<{ a: number; b: number; fragments: string[]; run: number }> = [];
      for (let a = 1; a < secret.length; a++) {
        for (let b = a + 1; b < secret.length; b++) {
          const body = `${secret.slice(0, a)}\n${secret.slice(a, b)}\n${secret.slice(b)}`;
          const out = redactTextSecrets(`api_token: ${body} rejected`);
          if (longestRun(out, secret) < MAX_SURVIVING_RUN) continue;
          const fragments = [secret.slice(a, b), secret.slice(b)];
          const explained = fragments.some(isWordShaped) || fragments.every(isSeparatedName);
          if (!explained) unexplained.push({ a, b, fragments, run: longestRun(out, secret) });
        }
      }
      expect(unexplained).toEqual([]);
    });
  }
});

describe("redactTextSecrets does not eat prose while absorbing a wrapped tail (#383)", () => {
  const [, SECRET] = SECRET_SHAPES[0]!;

  test("the English after a redacted credential survives verbatim", () => {
    expect(redactTextSecrets(`api_token: ${SECRET} rejected because the org is suspended`)).toBe(
      "api_token: [REDACTED] rejected because the org is suspended"
    );
  });

  test("a long English word is not mistaken for a wrapped credential tail", () => {
    // Every one of these is at least as long as the absorption floor, so LENGTH
    // alone cannot be the guard. What separates them from a token fragment is
    // that they carry no digit, no token separator, no internal capital, and no
    // consonant run English never produces.
    const words = [
      "unauthorized", "authentication", "configuration", "subscription",
      "organization", "unfortunately", "insufficient", "responsibility",
      "straightforward", "Unfortunately", "UNAUTHORIZED", "rate-limited",
      "self-service", "unauthorized.",
    ];
    for (const word of words) {
      expect(redactTextSecrets(`api_token: ${SECRET} ${word}`)).toBe(`api_token: [REDACTED] ${word}`);
    }
  });

  test("a whole English sentence after the credential is untouched", () => {
    const prose = "the organization subscription requires reauthentication before configuration changes apply";
    expect(redactTextSecrets(`password=${SECRET} ${prose}`)).toBe(`password=[REDACTED] ${prose}`);
  });

  test("a SECOND credential pair on the same line is still redacted on its own", () => {
    // The absorption must not swallow the next pair: that would both hide the
    // key name and stop it being redacted as a pair at all.
    const other = ["zzalt", "k3", "alsoneversent", "iiiijjjjkkkkllll"].join("_");
    const out = redactTextSecrets(`password=${SECRET} access_token=${other} user=joe`);
    expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
    expect(longestRun(out, other)).toBeLessThan(MAX_SURVIVING_RUN);
    expect(out).toContain("password=[REDACTED]");
    expect(out).toContain("access_token=[REDACTED]");
    expect(out).toContain("user=joe");
  });

  test("a run carrying prose or header punctuation is not credential material", () => {
    // `retry-after:300` is long enough and carries a digit, so LENGTH and the
    // digit signal alone would eat it. A token fragment contains no `:` -- nor
    // `@`, `(`, `!` -- so the charset check is what keeps a header value, an
    // email address and a parenthesised note readable next to a redacted one.
    for (const note of ["retry-after:300", "<operator@example.com>", "(instance-7-of-12)", "HTTP/1.1-429!"]) {
      expect(redactTextSecrets(`api_token: ${SECRET} ${note}`)).toBe(`api_token: [REDACTED] ${note}`);
    }
  });

  test("a word ends the chain, and the words themselves are never absorbed", () => {
    // Fragments are absorbed in chains, and a word breaks a chain. So the
    // English survives whatever follows it -- which is the property that
    // matters -- while a token-shaped run after the prose forms a chain of its
    // own and is absorbed on its own merits. That second half is a deliberate
    // change from the first version, which stopped dead at the first word: a
    // latch like that is what let a twice-wrapped value print (review F2).
    const later = ["zzref", "n8", "unrelatedvalue", "mmmmnnnnoooopppp"].join("_");
    expect(redactTextSecrets(`api_token: ${SECRET} rejected see ${later}`)).toBe(
      "api_token: [REDACTED] rejected see"
    );
    // The prose is intact in both halves; only the token-shaped run went.
    expect(redactTextSecrets(`api_token: ${SECRET} rejected see the org is suspended`)).toBe(
      "api_token: [REDACTED] rejected see the org is suspended"
    );
    // ...and a comma ends the tail outright, the same boundary a bare value has.
    expect(redactTextSecrets(`api_token: ${SECRET}, see ${later}`)).toBe(
      `api_token: [REDACTED], see ${later}`
    );
  });

  test("a quoted value still ends at its closing quote", () => {
    // A quoted value already spans whitespace, so there is no wrapped tail to
    // absorb and nothing past the quote belongs to the credential.
    const out = redactTextSecrets(`{"password": "hunter2", "instance": "monitoring-prod-01"}`);
    expect(out).toContain('"instance": "monitoring-prod-01"');
    expect(out).not.toContain("hunter2");
  });
});

describe("the guard does not eat the diagnostic it sits next to (#383 review F1)", () => {
  const CRED = "password=s3cretvalue1234";

  for (const diagnostic of DIAGNOSTIC_CORPUS) {
    test(`preserved byte-for-byte: ${diagnostic.slice(0, 48)}`, () => {
      expect(redactTextSecrets(`${CRED} ${diagnostic}`)).toBe(`password=[REDACTED] ${diagnostic}`);
    });
  }

  test("a libpq connection string keeps every field but the password", () => {
    const out = redactTextSecrets(
      "could not connect: host=db.example.com port=5432 dbname=app user=monitor password=s3cretvalue1234 sslmode=require"
    );
    expect(out).toContain("host=db.example.com");
    expect(out).toContain("sslmode=require");
    expect(out).toContain("user=monitor");
    expect(out).not.toContain("s3cretvalue1234");
  });

  test("base64 padding is not mistaken for an assignment", () => {
    // `=` means `key=value` and ends absorption -- except at the end of a run,
    // where it is base64 padding and a strong signal the run IS a credential.
    const padded = ["aB3cD4eF", "5gH6xiJ7", "kL8ymN9="].join("");
    const out = redactTextSecrets(`api_token: ${splitAt(padded, 12, "\n")} rejected`);
    expect(longestRun(out, padded)).toBeLessThan(MAX_SURVIVING_RUN);
    expect(out.endsWith(" rejected")).toBe(true);
  });
});

describe("punctuation around a wrapped tail does not save it (#383 review F3)", () => {
  const [, SECRET] = SECRET_SHAPES[0]!;

  test("a tail closed by a quote, bracket or sentence mark is still absorbed", () => {
    // The charset check rejected the whole run over one trailing character, so
    // 27 characters printed. The run's punctuation is now split off, the core
    // is judged, and the punctuation is put back so the message keeps its mark.
    for (const trail of ['"', "'", ")", "]", "}", ">", "!", "?", ".", ";", ":"]) {
      const out = redactTextSecrets(`api_token: ${splitAt(SECRET, 12, "\n")}${trail} rejected`);
      expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
      expect(out).toContain(trail);
      expect(out.endsWith(" rejected")).toBe(true);
    }
  });

  test("a tail opened by a quote or bracket is still absorbed", () => {
    for (const lead of ['"', "'", "(", "[", "{", "<"]) {
      const body = `${SECRET.slice(0, 12)}\n${lead}${SECRET.slice(12)}`;
      const out = redactTextSecrets(`api_token: ${body} rejected`);
      expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
    }
  });
});

describe("the wrapped-tail fix reaches every caller of redactTextSecrets (#383)", () => {
  const [, SECRET] = SECRET_SHAPES[0]!;
  const wrapped = splitAt(SECRET, 12, "\n");

  test("redactSecretsForLog, on the NON-JSON fallback that actually reaches it", () => {
    // redactSecretsForLog tries JSON.parse first; only a non-JSON body falls
    // through to redactTextSecrets, and that is the reachable path here.
    const out = redactSecretsForLog(`clone failed: api_token: ${wrapped} rejected`);
    expect(leaks(out, SECRET)).toBe(false);
    expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
    expect(out).toContain("clone failed:");
    expect(out.endsWith(" rejected")).toBe(true);
  });

  test("formatHttpError, plain-text body (the JSON.parse-failure path)", () => {
    const out = formatHttpError("op", 400, `api_token: ${wrapped} rejected`);
    expect(leaks(out, SECRET)).toBe(false);
    expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
    expect(out).toContain("HTTP 400");
  });

  test("formatHttpError, PostgREST message/details/hint fields", () => {
    const out = formatHttpError(
      "op",
      400,
      JSON.stringify({
        message: `api_token: ${wrapped} rejected`,
        details: `secret=${wrapped} stale`,
        hint: `access_key: ${wrapped} rotate it`,
      })
    );
    expect(leaks(out, SECRET)).toBe(false);
    expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
    // The human-readable remainder of each field survives.
    expect(out).toContain("rejected");
    expect(out).toContain("stale");
    expect(out).toContain("rotate it");
  });

  test("formatHttpError, the PTxyz reason phrase", () => {
    const out = formatHttpError("op", 400, undefined, `api_token: ${wrapped} rejected`);
    expect(leaks(out, SECRET)).toBe(false);
    expect(longestRun(out, SECRET)).toBeLessThan(MAX_SURVIVING_RUN);
  });
});
