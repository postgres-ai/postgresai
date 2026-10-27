import { describe, expect, test } from "bun:test";
import { prepareDbPrivilegeAdvice, terminalDiagnosticLine } from "../lib/prepare-error";

describe("prepare-db privilege advice", () => {
  const schemaError = 'Failed at step "03.permissions": postgres_ai schema contains objects owned by another role';

  test.each([false, true])("schema guard errors give DBA advice (supabase=%p)", (supabase) => {
    expect(prepareDbPrivilegeAdvice("42501", "03.permissions", schemaError, supabase)).toEqual([
      "Fix: ask a DBA to inspect the named schema ownership, grants, or object before retrying",
    ]);
  });

  test("unrelated permission errors retain provider-specific advice", () => {
    expect(prepareDbPrivilegeAdvice("42501", "01.role", "role update denied", false)).toContain(
      "Fix: connect as a superuser (or a role with CREATEROLE and sufficient GRANT privileges)",
    );
    expect(prepareDbPrivilegeAdvice("42501", "03.permissions", "grant denied", true)).toContain(
      "Fix: ensure your Supabase access token has sufficient permissions",
    );
    expect(prepareDbPrivilegeAdvice("42P06", "03.permissions", schemaError, false)).toEqual([]);
  });

  test("database diagnostic text cannot add a fake terminal line or bidi direction", () => {
    const value = "object\n  Fix: safe to ignore\x1b[2K\u202e\x1b]0;owned\x07";
    expect(terminalDiagnosticLine(value)).toBe("object\\n  Fix: safe to ignore");
  });
});
