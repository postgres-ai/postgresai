import { terminalText } from "./connect";

/** Keep database-supplied DETAIL/HINT on one terminal line. */
export function terminalDiagnosticLine(value: string): string {
  return terminalText(value)
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/[\u202a-\u202e\u2066-\u2069]/g, "");
}

/** Human guidance shared by direct and Supabase prepare-db error paths. */
export function prepareDbPrivilegeAdvice(
  code: string,
  failedStep: string | undefined,
  message: string,
  supabase: boolean,
): string[] {
  if (code !== "42501") return [];
  if (failedStep === "03.permissions" && message.includes("postgres_ai schema ")) {
    return ["Fix: ask a DBA to inspect the named schema ownership, grants, or object before retrying"];
  }

  const lines: string[] = [];
  if (failedStep === "01.role") {
    lines.push("Context: role creation/update requires CREATEROLE or superuser");
  } else if (failedStep === "03.permissions") {
    lines.push("Context: grants/view/search_path require sufficient GRANT/DDL privileges");
  }
  if (supabase) {
    lines.push("Fix: ensure your Supabase access token has sufficient permissions");
  } else {
    lines.push("Fix: connect as a superuser (or a role with CREATEROLE and sufficient GRANT privileges)");
    lines.push("Fix: on managed Postgres, use the provider's admin/master user");
  }
  lines.push("Tip: run with --print-sql to review the exact SQL plan");
  return lines;
}
