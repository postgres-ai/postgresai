import { describe, expect, test } from "bun:test";

import { formatHttpError } from "../lib/util";

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
});
