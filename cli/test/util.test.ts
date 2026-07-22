import { describe, expect, test } from "bun:test";

import { formatHttpError, describeFetchError, redactSecretsForLog } from "../lib/util";

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
